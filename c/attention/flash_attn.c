/*
 * c/attention/flash_attn.c — Flash Attention 2 CPU reference + CUDA dispatch
 *
 * Language: C (C17)
 * Role:     Pure speed — no C++, no STL, no RAII.
 *           The Zig layer handles allocations; Rust drives the call.
 *
 * Flash Attention 2 algorithm (Dao et al., 2023):
 *   - Tiled SRAM computation: avoids O(N²) HBM reads/writes
 *   - Causal mask baked into the tile loop
 *   - Supports GQA (grouped-query attention): num_heads != num_kv_heads
 *   - bf16/fp16 accumulation with fp32 running stats
 *
 * Dispatch:
 *   oracle_flash_attn() → checks CUDA availability → falls back to CPU
 *   oracle_flash_attn_cuda() → CUDA implementation (separate TU)
 *   oracle_flash_attn_cpu()  → reference for testing / non-GPU paths
 */

#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include "flash_attn.h"

/* ── compile-time knobs ─────────────────────────────────────────────────────── */
#ifndef OA_BLOCK_Q
#  define OA_BLOCK_Q  64    /* query block size (rows)           */
#endif
#ifndef OA_BLOCK_KV
#  define OA_BLOCK_KV 64    /* key/value block size (rows)       */
#endif

/* ── helpers ─────────────────────────────────────────────────────────────────── */
static inline float _bf16_to_f32(uint16_t x) {
    uint32_t v = (uint32_t)x << 16;
    float f;
    memcpy(&f, &v, 4);
    return f;
}

static inline uint16_t _f32_to_bf16(float x) {
    uint32_t v;
    memcpy(&v, &x, 4);
    /* round-nearest-even */
    uint32_t rounding_bias = (v & 0x00010000u) >> 1;
    v += rounding_bias + 0x00007FFFu;
    return (uint16_t)(v >> 16);
}

/* ── CPU reference (bf16 I/O, fp32 accumulation) ─────────────────────────────── */
/*
 * q, k, v: [batch, seq_len, num_heads, head_dim]  (bf16, row-major)
 * out:     [batch, seq_len, num_heads, head_dim]  (bf16, row-major)
 * scale:   1 / sqrt(head_dim)
 * causal:  apply causal mask when true
 *
 * This is the pedagogical reference, not the optimised path.
 * The CUDA kernel in flash_attn.cu replaces this at runtime.
 */
void oracle_flash_attn_cpu(
    const uint16_t * restrict q,
    const uint16_t * restrict k,
    const uint16_t * restrict v,
    uint16_t       * restrict out,
    int32_t batch,
    int32_t seq_q,
    int32_t seq_kv,
    int32_t num_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    float   scale,
    int32_t causal
) {
    int32_t kv_group = num_heads / num_kv_heads;  /* GQA: heads per KV head */

    for (int32_t b = 0; b < batch; ++b) {
        for (int32_t h = 0; h < num_heads; ++h) {
            int32_t kv_h = h / kv_group;

            for (int32_t qi = 0; qi < seq_q; ++qi) {
                /* softmax running stats */
                float m = -1e30f;
                float l = 0.0f;
                float acc[head_dim];
                memset(acc, 0, sizeof(float) * (size_t)head_dim);

                /* iterate key blocks */
                int32_t kv_end = causal ? (qi + 1) : seq_kv;
                for (int32_t ki = 0; ki < kv_end; ++ki) {
                    /* dot(q[qi], k[ki]) */
                    float s = 0.0f;
                    const uint16_t *q_row = q + (b * seq_q  * num_heads    + qi * num_heads    + h)        * head_dim;
                    const uint16_t *k_row = k + (b * seq_kv * num_kv_heads + ki * num_kv_heads + kv_h) * head_dim;
                    for (int32_t d = 0; d < head_dim; ++d)
                        s += _bf16_to_f32(q_row[d]) * _bf16_to_f32(k_row[d]);
                    s *= scale;

                    /* online softmax update */
                    float m_new = m > s ? m : s;
                    float l_new = l * expf(m - m_new) + expf(s - m_new);
                    float rescale = expf(m - m_new);

                    const uint16_t *v_row = v + (b * seq_kv * num_kv_heads + ki * num_kv_heads + kv_h) * head_dim;
                    float p = expf(s - m_new);
                    for (int32_t d = 0; d < head_dim; ++d) {
                        acc[d] = acc[d] * rescale + p * _bf16_to_f32(v_row[d]);
                    }
                    m = m_new;
                    l = l_new;
                }

                /* normalise and write output */
                uint16_t *out_row = out + (b * seq_q * num_heads + qi * num_heads + h) * head_dim;
                float inv_l = (l > 1e-10f) ? 1.0f / l : 0.0f;
                for (int32_t d = 0; d < head_dim; ++d)
                    out_row[d] = _f32_to_bf16(acc[d] * inv_l);
            }
        }
    }
}

/* ── Dispatch ────────────────────────────────────────────────────────────────── */
/*
 * oracle_flash_attn: public entry point.
 * If CUDA is available (detected at build time via OA_HAS_CUDA), calls the
 * .cu kernel; otherwise falls through to the CPU reference.
 */
void oracle_flash_attn(
    const uint16_t * restrict q,
    const uint16_t * restrict k,
    const uint16_t * restrict v,
    uint16_t       * restrict out,
    int32_t batch,
    int32_t seq_q,
    int32_t seq_kv,
    int32_t num_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    float   scale,
    int32_t causal
) {
#ifdef OA_HAS_CUDA
    extern void oracle_flash_attn_cuda(
        const uint16_t *, const uint16_t *, const uint16_t *, uint16_t *,
        int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, float, int32_t);
    oracle_flash_attn_cuda(q, k, v, out,
                           batch, seq_q, seq_kv, num_heads, num_kv_heads,
                           head_dim, scale, causal);
#else
    oracle_flash_attn_cpu(q, k, v, out,
                          batch, seq_q, seq_kv, num_heads, num_kv_heads,
                          head_dim, scale, causal);
#endif
}
