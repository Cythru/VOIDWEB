/**
 * rms_norm.cu — Root Mean Square Layer Norm (CUDA)
 *
 * RMSNorm: Zhang & Sennrich (2019).
 * Used by Llama, Mistral, Qwen, etc. (replaces LayerNorm).
 *
 * Forward:  y = x / rms(x) * weight
 * rms(x)  = sqrt(mean(x²) + eps)
 *
 * Fused kernel: reads x once, computes rms, applies scale — minimal HBM traffic.
 * Supports BF16 accumulation with FP32 reduction for numerical stability.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../include/kernels.h"

// ── Warp-level reduction ──────────────────────────────────────────────────────
__device__ __forceinline__ float warp_reduce_sum(float x) {
    x += __shfl_xor_sync(0xffffffff, x, 16);
    x += __shfl_xor_sync(0xffffffff, x,  8);
    x += __shfl_xor_sync(0xffffffff, x,  4);
    x += __shfl_xor_sync(0xffffffff, x,  2);
    x += __shfl_xor_sync(0xffffffff, x,  1);
    return x;
}

// ── RMSNorm kernel ────────────────────────────────────────────────────────────
// Grid:  (batch × seq_len,)
// Block: (min(hidden_size, 1024),)  — one CTA per token
__global__ void rms_norm_kernel(
    __nv_bfloat16*       __restrict__ x,       // [B*S, H] — modified in-place
    const __nv_bfloat16* __restrict__ weight,   // [H]
    int H,
    float eps
) {
    extern __shared__ float smem[];  // [H] or [warps] for partial sums

    const int token = blockIdx.x;
    __nv_bfloat16* row = x + (long)token * H;
    const int tid = threadIdx.x;

    // Compute sum of squares.
    float sq_sum = 0.0f;
    for (int d = tid; d < H; d += blockDim.x) {
        float v = __bfloat162float(row[d]);
        sq_sum += v * v;
    }
    // Warp reduce.
    sq_sum = warp_reduce_sum(sq_sum);
    // Block reduce across warps.
    if (H > 32) {
        const int warp_id = tid / 32;
        const int lane_id = tid % 32;
        if (lane_id == 0) smem[warp_id] = sq_sum;
        __syncthreads();
        if (tid < (blockDim.x / 32)) {
            sq_sum = smem[tid];
        } else {
            sq_sum = 0.0f;
        }
        if (tid < 32) sq_sum = warp_reduce_sum(sq_sum);
        if (tid == 0) smem[0] = sq_sum;
        __syncthreads();
        sq_sum = smem[0];
    }

    float rms_inv = rsqrtf(sq_sum / H + eps);

    // Apply normalisation and scale.
    for (int d = tid; d < H; d += blockDim.x) {
        float v = __bfloat162float(row[d]) * rms_inv;
        float w = __bfloat162float(weight[d]);
        row[d] = __float2bfloat16(v * w);
    }
}

// ── C ABI ─────────────────────────────────────────────────────────────────────
extern "C" int oracle_rms_norm(
    void* x, const void* weight,
    int batch, int seq_len, int hidden_size, float eps
) {
    int tokens = batch * seq_len;
    int tpb    = min(hidden_size, 1024);
    int smem   = (tpb / 32) * sizeof(float);

    rms_norm_kernel<<<tokens, tpb, smem>>>(
        (__nv_bfloat16*)x,
        (const __nv_bfloat16*)weight,
        hidden_size, eps
    );
    return (int)cudaGetLastError();
}
