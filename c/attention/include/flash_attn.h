/*
 * c/attention/include/flash_attn.h — Flash Attention 2 C API
 *
 * All functions take/return bf16 (uint16_t) tensors.
 * No heap allocation inside — callers provide output buffers.
 */
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Main dispatch: CUDA if available, CPU fallback otherwise */
void oracle_flash_attn(
    const uint16_t * restrict q,      /* [B, Sq,  H,  D] bf16 */
    const uint16_t * restrict k,      /* [B, Skv, Hkv,D] bf16 */
    const uint16_t * restrict v,      /* [B, Skv, Hkv,D] bf16 */
    uint16_t       * restrict out,    /* [B, Sq,  H,  D] bf16 */
    int32_t batch,
    int32_t seq_q,
    int32_t seq_kv,
    int32_t num_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    float   scale,      /* 1/sqrt(head_dim) */
    int32_t causal      /* 0 = bidirectional, 1 = causal */
);

/* CPU-only reference (always available; used in tests) */
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
);

#ifdef __cplusplus
}
#endif
