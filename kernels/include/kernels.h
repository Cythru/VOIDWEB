/**
 * kernels.h — Oracle CUDA Kernel Public API
 *
 * This header defines the C ABI exported from libkernels.so.
 * The Rust FFI layer links against these symbols.
 *
 * All CUDA work is serialised onto per-stream queues; callers must
 * NOT call forward_pass() concurrently from multiple threads.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Model handle ──────────────────────────────────────────────────────────────
typedef struct OracleModel OracleModel;

// ── Sequence descriptor (mirrors Rust SeqDescriptor) ─────────────────────────
typedef struct {
    uint64_t  seq_id;
    const uint32_t* token_ids;
    uint32_t  num_tokens;
    const uint32_t* block_table;   // physical KV block IDs
    uint32_t  num_blocks;
    uint8_t   is_prefill;          // 1 = prefill, 0 = decode
} SeqDescriptor;

// ── Lifecycle ─────────────────────────────────────────────────────────────────

/**
 * Allocate and initialise a model on the GPU.
 * @param config_json  Null-terminated JSON string (ModelConfig serialised by Rust).
 * @param config_len   Length of config_json in bytes.
 * @return             Opaque handle, or NULL on failure.
 */
OracleModel* oracle_model_alloc(const char* config_json, size_t config_len);

/**
 * Free all GPU memory and destroy the model.
 */
void oracle_model_free(OracleModel* model);

// ── Forward pass ──────────────────────────────────────────────────────────────

/**
 * Run one forward step for a mixed prefill+decode batch.
 *
 * @param model       Model handle.
 * @param seqs        Array of SeqDescriptor (length num_seqs).
 * @param num_seqs    Number of sequences in the batch.
 * @param kv_k        GPU pointer to K-cache buffer (may be NULL — model manages internally).
 * @param kv_v        GPU pointer to V-cache buffer (may be NULL).
 * @param logits_out  CPU pointer to output buffer [num_seqs × vocab_size] f32.
 *                    Caller must allocate; values are written synchronously.
 * @return            0 on success, non-zero error code on failure.
 */
int oracle_forward_pass(
    OracleModel*          model,
    const SeqDescriptor*  seqs,
    uint32_t              num_seqs,
    void*                 kv_k,
    void*                 kv_v,
    float*                logits_out
);

// ── Utility kernels (callable independently) ─────────────────────────────────

/**
 * Flash attention (in-place, fused softmax).
 * Shapes: Q/K/V — [batch, heads, seq_len, head_dim]
 * Output — [batch, heads, seq_len, head_dim]
 */
int oracle_flash_attention(
    const void* q,        // bf16 GPU ptr
    const void* k,
    const void* v,
    void*       out,
    int         batch,
    int         num_heads,
    int         seq_len,
    int         head_dim,
    float       scale,    // 1/sqrt(head_dim)
    int         causal    // 1 = causal mask
);

/**
 * Rotary position embedding (in-place).
 * Applies RoPE to Q and K tensors.
 */
int oracle_apply_rope(
    void*       q,          // bf16 GPU ptr [batch, heads, seq_len, head_dim]
    void*       k,
    int         batch,
    int         num_q_heads,
    int         num_k_heads,
    int         seq_len,
    int         head_dim,
    float       theta,
    int         position_offset
);

/**
 * RMS layer normalisation (in-place).
 */
int oracle_rms_norm(
    void*         x,        // bf16 GPU ptr [batch, seq_len, hidden_size]
    const void*   weight,   // bf16 GPU ptr [hidden_size]
    int           batch,
    int           seq_len,
    int           hidden_size,
    float         eps
);

#ifdef __cplusplus
}
#endif
