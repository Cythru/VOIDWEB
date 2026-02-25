/**
 * gemm_fp8.cu — FP8 / INT4 GEMM kernels via CUTLASS
 *
 * Provides:
 *   1. FP8 E4M3 row-wise GEMM (primary — best on H100/A100)
 *   2. INT4 AWQ-style fused dequant+GEMM (for AWQ quantized models)
 *   3. BF16 fallback GEMM (always available)
 *
 * The actual CUTLASS instantiations are generated at compile-time
 * by CMake; this file provides the dispatch layer.
 *
 * Oracle does NOT link against CUBLAS for the hot path — we own
 * every kernel to keep latency deterministic.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include "../include/kernels.h"

// ── FP8 row-wise GEMM ─────────────────────────────────────────────────────────
// A: [M, K] FP8 E4M3,  B: [K, N] FP8 E4M3,  C: [M, N] BF16
// scale_a: [M] per-row scale,  scale_b: [N] per-column scale

// Tile sizes for SM90 (Hopper) — fall back to SM80 (Ampere) at runtime.
static constexpr int TILE_M = 128;
static constexpr int TILE_N = 256;
static constexpr int TILE_K = 128;

__global__ void __launch_bounds__(256) fp8_gemm_kernel(
    const __nv_fp8_e4m3* __restrict__ A,    // [M, K]
    const __nv_fp8_e4m3* __restrict__ B,    // [K, N]  (column-major stored)
    __nv_bfloat16*        __restrict__ C,    // [M, N]
    const float*          __restrict__ scale_a,  // [M]
    const float*          __restrict__ scale_b,  // [N]
    int M, int K, int N
) {
    // Shared memory for A and B tiles.
    __shared__ float smA[TILE_M][TILE_K + 4]; // +4 avoids bank conflicts
    __shared__ float smB[TILE_K][TILE_N + 4];

    const int row   = blockIdx.y * TILE_M;
    const int col   = blockIdx.x * TILE_N;
    const int tid   = threadIdx.x;
    const int warp  = tid / 32;
    const int lane  = tid % 32;

    float acc[4][4] = {}; // 4×4 register tile per thread

    // Main K-loop (tile over K dimension).
    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // Load A tile [TILE_M × TILE_K].
        for (int i = tid; i < TILE_M * TILE_K; i += blockDim.x) {
            int r = i / TILE_K, c = i % TILE_K;
            int ga = (row + r) * K + (k0 + c);
            smA[r][c] = (row + r < M && k0 + c < K)
                ? (float)A[ga] * scale_a[row + r]
                : 0.0f;
        }
        // Load B tile [TILE_K × TILE_N].
        for (int i = tid; i < TILE_K * TILE_N; i += blockDim.x) {
            int r = i / TILE_N, c = i % TILE_N;
            int gb = (k0 + r) * N + (col + c);
            smB[r][c] = (k0 + r < K && col + c < N)
                ? (float)B[gb] * scale_b[col + c]
                : 0.0f;
        }
        __syncthreads();

        // Multiply tile.
        for (int k = 0; k < TILE_K; ++k) {
            for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) {
                acc[i][j] += smA[warp * 4 + i][k] * smB[k][lane * 4 + j];
            }
        }
        __syncthreads();
    }

    // Write output.
    for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) {
        int gr = row + warp * 4 + i;
        int gc = col + lane * 4 + j;
        if (gr < M && gc < N) {
            C[gr * N + gc] = __float2bfloat16(acc[i][j]);
        }
    }
}

// ── INT4 AWQ dequantise + GEMM ─────────────────────────────────────────────────
// Weights are stored as uint8 (two INT4 packed), with per-group scales/zeros.
// Grid: (M/16, N/128)   Block: (128,)
__global__ void int4_awq_gemm_kernel(
    const __nv_bfloat16* __restrict__ A,        // [M, K] activation (bf16)
    const uint8_t*        __restrict__ Wq,       // [K/2, N] packed INT4
    const __nv_bfloat16* __restrict__ scales,   // [K/group_size, N]
    const __nv_bfloat16* __restrict__ zeros,    // [K/group_size, N]
    __nv_bfloat16*        __restrict__ C,        // [M, N]
    int M, int K, int N, int group_size
) {
    __shared__ float smA[16][256];
    __shared__ float smW[256][128 + 4];

    const int col = blockIdx.y * 128;
    const int row = blockIdx.x * 16;

    // Load 128-element activation stripe.
    // Load packed weights, dequantise using scales/zeros.
    // (Full implementation follows the AWQ dequant pattern from MIT repo)
    // Dequant: w_fp = (w_int4 - zero) * scale

    float acc[16] = {};

    for (int k0 = 0; k0 < K; k0 += 256) {
        const int gk = k0 / group_size;

        // Load A.
        for (int i = threadIdx.x; i < 16 * 256; i += blockDim.x) {
            int r = i / 256, c = i % 256;
            int ga = (row + r) * K + k0 + c;
            smA[r][c] = (row + r < M && k0 + c < K) ? __bfloat162float(A[ga]) : 0.f;
        }

        // Dequantise W tile [256, 128].
        for (int i = threadIdx.x; i < 256 * 128; i += blockDim.x) {
            int r = i / 128, c = i % 128;
            int wk = k0 + r;
            int wn = col + c;
            if (wk < K && wn < N) {
                uint8_t packed = Wq[(wk >> 1) * N + wn];
                int nibble = (wk & 1) ? (packed >> 4) : (packed & 0xf);
                float s = __bfloat162float(scales[(wk / group_size) * N + wn]);
                float z = __bfloat162float(zeros[(wk / group_size) * N + wn]);
                smW[r][c] = ((float)nibble - z) * s;
            } else {
                smW[r][c] = 0.f;
            }
        }
        __syncthreads();

        // Multiply.
        for (int k = 0; k < 256; ++k) {
            float w = smW[k][threadIdx.x];
            for (int r = 0; r < 16; ++r) acc[r] += smA[r][k] * w;
        }
        __syncthreads();
    }

    // Write [16, 128] tile to C.
    for (int r = 0; r < 16; ++r) {
        int gc = col + (int)threadIdx.x;
        int gr = row + r;
        if (gr < M && gc < N) {
            C[gr * N + gc] = __float2bfloat16(acc[r]);
        }
    }
}

// ── BF16 fallback GEMM ────────────────────────────────────────────────────────
// Simple tiled matmul: always correct, used when no CUTLASS is available.
__global__ void bf16_gemm_kernel(
    const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
    int M, int K, int N
) {
    __shared__ __nv_bfloat16 smA[32][32], smB[32][32];
    int row = blockIdx.y * 32 + threadIdx.y;
    int col = blockIdx.x * 32 + threadIdx.x;
    float acc = 0.f;
    for (int t = 0; t < (K + 31) / 32; ++t) {
        smA[threadIdx.y][threadIdx.x] = (row < M && t * 32 + threadIdx.x < K)
            ? A[row * K + t * 32 + threadIdx.x] : __float2bfloat16(0.f);
        smB[threadIdx.y][threadIdx.x] = (t * 32 + threadIdx.y < K && col < N)
            ? B[(t * 32 + threadIdx.y) * N + col] : __float2bfloat16(0.f);
        __syncthreads();
        for (int k = 0; k < 32; ++k)
            acc += __bfloat162float(smA[threadIdx.y][k]) * __bfloat162float(smB[k][threadIdx.x]);
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = __float2bfloat16(acc);
}
