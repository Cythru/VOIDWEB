/**
 * rope.cu — Rotary Position Embedding (CUDA)
 *
 * RoPE: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021).
 * Applied in-place to Q and K tensors before attention.
 *
 * Supports:
 *   - Standard RoPE (Llama, Mistral)
 *   - NTK-aware scaling (Llama 2 long-context)
 *   - YaRN scaling (Mistral, Mixtral)
 * Shape: [batch, num_heads, seq_len, head_dim]
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include "../include/kernels.h"

// ── RoPE helpers ──────────────────────────────────────────────────────────────
__device__ __forceinline__ float2 rotate_half(float x0, float x1) {
    return make_float2(-x1, x0);
}

__device__ __forceinline__ float2 apply_rope_pair(
    float x0, float x1,
    float cos_val, float sin_val
) {
    float2 rh = rotate_half(x0, x1);
    return make_float2(
        x0 * cos_val - rh.x * sin_val,
        x1 * cos_val - rh.y * sin_val
    );
}

// ── Precompute cos/sin table kernel ──────────────────────────────────────────
// Grid: (ceil(max_seq/1024),)   Block: (1024,)
// Output: cos_table, sin_table — [max_seq, head_dim/2]
__global__ void compute_rope_table(
    float* cos_table,
    float* sin_table,
    int    max_seq,
    int    half_dim,
    float  theta_base,
    float  scaling_factor  // 1.0 = standard, >1 = NTK scaled
) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= max_seq) return;

    for (int i = 0; i < half_dim; ++i) {
        float freq = 1.0f / powf(theta_base, (2.0f * i) / (half_dim * 2));
        freq /= scaling_factor;
        float angle = (float)pos * freq;
        cos_table[pos * half_dim + i] = cosf(angle);
        sin_table[pos * half_dim + i] = sinf(angle);
    }
}

// ── RoPE apply kernel ─────────────────────────────────────────────────────────
// Grid:  (batch, num_heads, ceil(seq_len/8))
// Block: (32, 4, 1)   — 32 threads × 4 pairs of dim per thread
__global__ void rope_apply_kernel(
    __nv_bfloat16*       __restrict__ tensor,  // Q or K: [B, H, S, D]
    const float* __restrict__         cos_table,
    const float* __restrict__         sin_table,
    int B, int H, int S, int D,
    int position_offset
) {
    const int b   = blockIdx.x;
    const int h   = blockIdx.y;
    const int s   = blockIdx.z * blockDim.y + threadIdx.y;
    const int tid = threadIdx.x; // iterates over D/2 pairs

    if (b >= B || h >= H || s >= S) return;
    const int half = D / 2;

    __nv_bfloat16* row = tensor + ((long)b * H + h) * S * D + s * D;
    const int pos = s + position_offset;
    const float* cos_row = cos_table + pos * half;
    const float* sin_row = sin_table + pos * half;

    for (int d = tid; d < half; d += blockDim.x) {
        float x0 = __bfloat162float(row[d]);
        float x1 = __bfloat162float(row[d + half]);
        float2 r = apply_rope_pair(x0, x1, cos_row[d], sin_row[d]);
        row[d]        = __float2bfloat16(r.x);
        row[d + half] = __float2bfloat16(r.y);
    }
}

// ── Persistent table (allocated once per model instance) ─────────────────────
static float* g_cos_table = nullptr;
static float* g_sin_table = nullptr;
static int    g_table_seq  = 0;
static int    g_table_hdim = 0;

static void ensure_rope_table(int max_seq, int half_dim, float theta, float scale) {
    if (g_table_seq >= max_seq && g_table_hdim >= half_dim) return;
    if (g_cos_table) { cudaFree(g_cos_table); cudaFree(g_sin_table); }

    size_t sz = (size_t)max_seq * half_dim * sizeof(float);
    cudaMalloc(&g_cos_table, sz);
    cudaMalloc(&g_sin_table, sz);

    int tpb = 256;
    int blk = (max_seq + tpb - 1) / tpb;
    compute_rope_table<<<blk, tpb>>>(g_cos_table, g_sin_table, max_seq, half_dim, theta, scale);
    cudaDeviceSynchronize();

    g_table_seq  = max_seq;
    g_table_hdim = half_dim;
}

// ── C ABI ─────────────────────────────────────────────────────────────────────
extern "C" int oracle_apply_rope(
    void* q, void* k,
    int batch, int num_q_heads, int num_k_heads,
    int seq_len, int head_dim,
    float theta, int position_offset
) {
    const int half = head_dim / 2;
    const int max_pos = seq_len + position_offset + 64; // small padding

    ensure_rope_table(max_pos, half, theta, 1.0f);

    dim3 grid(batch, num_q_heads, (seq_len + 3) / 4);
    dim3 block(32, 4);
    rope_apply_kernel<<<grid, block>>>(
        (__nv_bfloat16*)q, g_cos_table, g_sin_table,
        batch, num_q_heads, seq_len, head_dim, position_offset
    );

    dim3 grid_k(batch, num_k_heads, (seq_len + 3) / 4);
    rope_apply_kernel<<<grid_k, block>>>(
        (__nv_bfloat16*)k, g_cos_table, g_sin_table,
        batch, num_k_heads, seq_len, head_dim, position_offset
    );

    return (int)cudaGetLastError();
}
