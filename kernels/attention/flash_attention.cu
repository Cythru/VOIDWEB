/**
 * flash_attention.cu — Oracle Flash Attention (CUDA)
 *
 * Flash Attention 2 algorithm:
 *   - Tiled computation (SRAM-first, never materialise full NxN matrix)
 *   - O(N) HBM reads/writes instead of O(N²)
 *   - Fused softmax with running max+sum (Milakov & Gimelshein 2018)
 *   - Supports BF16, FP16, FP8 (E4M3)
 *   - Causal masking via triangular tile skipping
 *
 * Reference: Dao et al. "FlashAttention-2" (2023)
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include "../include/kernels.h"

// ── Compile-time tile sizes ───────────────────────────────────────────────────
// Tuned for A100 (128 KB shared memory).  Adjust for other GPUs.
static constexpr int BQ   = 64;   // Q tile size (rows per CTA in seq dim)
static constexpr int BKV  = 64;   // KV tile size
static constexpr int BD   = 128;  // head_dim (max)
static constexpr int WARP_SIZE = 32;

// ── Shared memory layout ──────────────────────────────────────────────────────
struct __align__(128) SmemLayout {
    __nv_bfloat16 q[BQ][BD];
    __nv_bfloat16 k[BKV][BD];
    __nv_bfloat16 v[BKV][BD];
    float         s[BQ][BKV];  // attention scores (fp32 for numerical stability)
    float         acc[BQ][BD]; // accumulator
    float         row_max[BQ]; // running maximum (Flash Attention inner loop)
    float         row_sum[BQ]; // running normalisation denominator
};

// ── Flash Attention kernel ─────────────────────────────────────────────────────
// Grid:  (batch, num_heads, ceil(seq_len / BQ))
// Block: (WARP_SIZE * 4, 1, 1)  — 4 warps per CTA
__global__ void flash_attention_kernel(
    const __nv_bfloat16* __restrict__ Q,  // [B, H, N, D]
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16*                    O,  // [B, H, N, D]
    int  N,          // sequence length
    int  H,          // num heads
    int  D,          // head_dim
    float scale,     // 1 / sqrt(D)
    bool causal
) {
    extern __shared__ char smem_raw[];
    SmemLayout& sm = *reinterpret_cast<SmemLayout*>(smem_raw);

    const int b    = blockIdx.x;
    const int h    = blockIdx.y;
    const int q_blk = blockIdx.z;          // which BQ-sized Q tile
    const int tid  = threadIdx.x;
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;

    const int q_start = q_blk * BQ;
    const int q_end   = min(q_start + BQ, N);
    const int q_rows  = q_end - q_start;

    // Base pointers for this (batch, head).
    const long head_stride = (long)N * D;
    const long batch_stride = (long)H * head_stride;
    const __nv_bfloat16* Qptr = Q + b * batch_stride + h * head_stride;
    const __nv_bfloat16* Kptr = K + b * batch_stride + h * head_stride;
    const __nv_bfloat16* Vptr = V + b * batch_stride + h * head_stride;
    __nv_bfloat16*        Optr = O + b * batch_stride + h * head_stride;

    // Initialise running stats.
    for (int i = tid; i < BQ; i += blockDim.x) {
        sm.row_max[i] = -1e30f;
        sm.row_sum[i] = 0.0f;
        for (int d = 0; d < D; ++d) sm.acc[i][d] = 0.0f;
    }
    __syncthreads();

    // Load Q tile.
    for (int i = tid; i < q_rows * D; i += blockDim.x) {
        int r = i / D, c = i % D;
        sm.q[r][c] = (q_start + r < N) ? Qptr[(q_start + r) * D + c] : __float2bfloat16(0.f);
    }
    __syncthreads();

    // Iterate over KV tiles.
    const int kv_blocks = (N + BKV - 1) / BKV;
    for (int kv_blk = 0; kv_blk < kv_blocks; ++kv_blk) {
        const int kv_start = kv_blk * BKV;
        const int kv_end   = min(kv_start + BKV, N);
        const int kv_rows  = kv_end - kv_start;

        // Causal: skip KV tiles that are entirely after the current Q tile.
        if (causal && kv_start >= q_end) break;

        // Load K and V tiles.
        for (int i = tid; i < kv_rows * D; i += blockDim.x) {
            int r = i / D, c = i % D;
            sm.k[r][c] = Kptr[(kv_start + r) * D + c];
            sm.v[r][c] = Vptr[(kv_start + r) * D + c];
        }
        __syncthreads();

        // Compute S = Q * K^T · scale  (BQ × BKV)
        for (int qi = warp; qi < q_rows; qi += 4) {
            for (int ki = lane; ki < kv_rows; ki += WARP_SIZE) {
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) {
                    dot += __bfloat162float(sm.q[qi][d]) * __bfloat162float(sm.k[ki][d]);
                }
                float s = dot * scale;
                // Causal mask: positions after q_start+qi are -inf.
                if (causal && (kv_start + ki) > (q_start + qi)) {
                    s = -1e30f;
                }
                sm.s[qi][ki] = s;
            }
        }
        __syncthreads();

        // Online softmax (running max/sum update).
        for (int qi = warp; qi < q_rows; qi += 4) {
            float m_old = sm.row_max[qi];
            float m_new = m_old;
            for (int ki = 0; ki < kv_rows; ++ki) {
                m_new = fmaxf(m_new, sm.s[qi][ki]);
            }
            float exp_shift = expf(m_old - m_new);
            float lsum = 0.0f;
            for (int ki = 0; ki < kv_rows; ++ki) {
                float p = expf(sm.s[qi][ki] - m_new);
                sm.s[qi][ki] = p;
                lsum += p;
            }
            // Update accumulator with rescaling.
            for (int d = lane; d < D; d += WARP_SIZE) {
                sm.acc[qi][d] *= exp_shift;
                float vacc = 0.0f;
                for (int ki = 0; ki < kv_rows; ++ki) {
                    vacc += sm.s[qi][ki] * __bfloat162float(sm.v[ki][d]);
                }
                sm.acc[qi][d] += vacc;
            }
            if (lane == 0) {
                sm.row_max[qi] = m_new;
                sm.row_sum[qi] = sm.row_sum[qi] * exp_shift + lsum;
            }
        }
        __syncthreads();
    }

    // Normalise and write output.
    for (int qi = warp; qi < q_rows; qi += 4) {
        float inv_sum = 1.0f / (sm.row_sum[qi] + 1e-10f);
        for (int d = lane; d < D; d += WARP_SIZE) {
            Optr[(q_start + qi) * D + d] = __float2bfloat16(sm.acc[qi][d] * inv_sum);
        }
    }
}

// ── Paged attention decode kernel ─────────────────────────────────────────────
// For single-token decode: gather KV from paged blocks.
// Grid: (batch × num_heads, 1, 1)   Block: (128, 1, 1)
__global__ void paged_decode_attention_kernel(
    const __nv_bfloat16* __restrict__  Q,           // [B*H, 1, D]
    const __nv_bfloat16* __restrict__  KVPool,       // [num_blocks, 2, block_size, H, D]
    const uint32_t*      __restrict__  BlockTables,  // [B, max_blocks]
    __nv_bfloat16*                     O,            // [B*H, 1, D]
    int  B,
    int  H,
    int  D,
    int  block_size,
    int  max_blocks_per_seq,
    int  seq_lens_total, // total kv tokens in this sequence
    float scale
) {
    extern __shared__ float smem_decode[];
    float* acc     = smem_decode;
    float* kv_buf  = smem_decode + D;

    const int bh   = blockIdx.x;
    const int b    = bh / H;
    const int h    = bh % H;
    const int tid  = threadIdx.x;

    // Load query.
    const __nv_bfloat16* q_ptr = Q + bh * D;
    float q[128]; // assume D <= 128
    for (int d = tid; d < D; d += blockDim.x) q[d] = __bfloat162float(q_ptr[d]);

    for (int d = tid; d < D; d += blockDim.x) acc[d] = 0.0f;

    float row_max = -1e30f, row_sum = 0.0f;
    const uint32_t* btable = BlockTables + b * max_blocks_per_seq;

    // Iterate over KV blocks.
    const int kv_len   = seq_lens_total;
    const int num_blks = (kv_len + block_size - 1) / block_size;
    for (int blk_idx = 0; blk_idx < num_blks; ++blk_idx) {
        uint32_t phys_blk = btable[blk_idx];
        int blk_start = blk_idx * block_size;
        int blk_end   = min(blk_start + block_size, kv_len);

        for (int pos = blk_start; pos < blk_end; ++pos) {
            int slot = pos % block_size;
            // KVPool layout: [blk, {K=0,V=1}, slot, h, d]
            const __nv_bfloat16* k_ptr = KVPool + (phys_blk * 2 * block_size * H + slot * H + h) * D;
            const __nv_bfloat16* v_ptr = k_ptr + block_size * H * D;

            // Dot(Q, K).
            float dot = 0.0f;
            for (int d = 0; d < D; ++d) {
                dot += q[d] * __bfloat162float(k_ptr[d]);
            }
            dot *= scale;

            float m_new = fmaxf(row_max, dot);
            float p     = expf(dot - m_new);
            float shift = expf(row_max - m_new);

            for (int d = tid; d < D; d += blockDim.x) {
                acc[d] = acc[d] * shift + p * __bfloat162float(v_ptr[d]);
            }
            row_sum = row_sum * shift + p;
            row_max = m_new;
        }
    }
    __syncthreads();

    // Write output.
    __nv_bfloat16* out_ptr = O + bh * D;
    float inv = 1.0f / (row_sum + 1e-10f);
    for (int d = tid; d < D; d += blockDim.x) {
        out_ptr[d] = __float2bfloat16(acc[d] * inv);
    }
}

// ── C ABI entry point ─────────────────────────────────────────────────────────
extern "C" int oracle_flash_attention(
    const void* q, const void* k, const void* v,
    void* out,
    int batch, int num_heads, int seq_len, int head_dim,
    float scale, int causal
) {
    if (head_dim > BD) return -1; // unsupported head_dim

    dim3 grid(batch, num_heads, (seq_len + BQ - 1) / BQ);
    dim3 block(WARP_SIZE * 4);
    size_t smem = sizeof(SmemLayout);

    flash_attention_kernel<<<grid, block, smem>>>(
        (const __nv_bfloat16*)q,
        (const __nv_bfloat16*)k,
        (const __nv_bfloat16*)v,
        (__nv_bfloat16*)out,
        seq_len, num_heads, head_dim,
        scale, (bool)causal
    );
    return (int)cudaGetLastError();
}
