//! Oracle Sampler
//!
//! Converts raw logits → next token IDs.
//!
//! Supports:
//!   - Greedy (temperature = 0)
//!   - Temperature scaling
//!   - Top-k filtering
//!   - Top-p (nucleus) sampling
//!   - Repetition penalty (applied before top-k/top-p)
//!
//! Operates on a flat logit buffer shaped [num_seqs × vocab_size].
//! Uses Rayon for parallel sampling when batch_size > 1.

use anyhow::{Result, bail};
use rayon::prelude::*;
use serde::{Serialize, Deserialize};

pub use crate::SamplerConfig;

// ── Config ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplerConfig {
    pub temperature:        f32,
    pub top_p:              f32,
    pub top_k:              u32,
    pub repetition_penalty: f32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature:        0.7,
            top_p:              0.95,
            top_k:              50,
            repetition_penalty: 1.1,
        }
    }
}

// ── Sampler ───────────────────────────────────────────────────────────────────

pub struct Sampler {
    pub config: SamplerConfig,
}

impl Sampler {
    pub fn new(config: SamplerConfig) -> Self {
        Self { config }
    }

    /// Sample one token per sequence from a flat logit buffer.
    ///
    /// `logits`: [num_seqs × vocab_size] f32 slice
    /// `batch`:  only used to access per-sequence output tokens for rep-penalty
    /// Returns:  Vec<(RequestId, next_token)>
    pub fn sample_batch(
        &self,
        logits: &[f32],
        batch:  &scheduler::BatchView,
    ) -> Result<Vec<(scheduler::RequestId, u32)>> {
        let num_seqs = batch.num_seqs();
        if num_seqs == 0 { return Ok(vec![]); }

        let vocab_size = logits.len() / num_seqs;
        if logits.len() != num_seqs * vocab_size {
            bail!("logits len {} ≠ num_seqs {} × vocab_size {}", logits.len(), num_seqs, vocab_size);
        }

        // Collect (req_id, already_generated_tokens) for rep-penalty.
        let seq_meta: Vec<(scheduler::RequestId, &[u32])> = {
            let mut v = Vec::with_capacity(num_seqs);
            for pf in &batch.prefill_seqs {
                v.push((pf.req_id, pf.tokens.as_slice()));
            }
            for dc in &batch.decode_seqs {
                v.push((dc.req_id, std::slice::from_ref(&dc.last_token)));
            }
            v
        };

        let results: Vec<(scheduler::RequestId, u32)> = (0..num_seqs)
            .into_par_iter()
            .map(|i| {
                let (req_id, prev_tokens) = seq_meta[i];
                let slice = &logits[i * vocab_size .. (i+1) * vocab_size];
                let token = self.sample_one(slice, prev_tokens);
                (req_id, token)
            })
            .collect();

        Ok(results)
    }

    // ── Single-sequence sampling ──────────────────────────────────────────────

    pub fn sample_one(&self, logits: &[f32], prev_tokens: &[u32]) -> u32 {
        let mut scores: Vec<f32> = logits.to_vec();

        // 1. Repetition penalty.
        if (self.config.repetition_penalty - 1.0).abs() > 1e-6 {
            apply_rep_penalty(&mut scores, prev_tokens, self.config.repetition_penalty);
        }

        // 2. Greedy shortcut.
        if self.config.temperature <= 0.0 {
            return argmax(&scores);
        }

        // 3. Temperature scaling → softmax.
        let inv_temp = 1.0 / self.config.temperature;
        for s in scores.iter_mut() { *s *= inv_temp; }
        softmax_inplace(&mut scores);

        // 4. Top-k.
        if self.config.top_k > 0 && (self.config.top_k as usize) < scores.len() {
            top_k_filter(&mut scores, self.config.top_k as usize);
        }

        // 5. Top-p (nucleus).
        if self.config.top_p < 1.0 {
            top_p_filter(&mut scores, self.config.top_p);
        }

        // 6. Sample from remaining probability mass.
        multinomial_sample(&scores)
    }
}

// ── Sampling primitives ───────────────────────────────────────────────────────

fn argmax(logits: &[f32]) -> u32 {
    logits.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn softmax_inplace(v: &mut [f32]) {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() { *x = (*x - max).exp(); sum += *x; }
    if sum > 0.0 { for x in v.iter_mut() { *x /= sum; } }
}

fn apply_rep_penalty(scores: &mut [f32], prev_tokens: &[u32], penalty: f32) {
    for &t in prev_tokens {
        let idx = t as usize;
        if idx < scores.len() {
            scores[idx] = if scores[idx] < 0.0 {
                scores[idx] * penalty
            } else {
                scores[idx] / penalty
            };
        }
    }
}

fn top_k_filter(probs: &mut [f32], k: usize) {
    if k >= probs.len() { return; }
    // Partition: keep top-k, zero out the rest.
    let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (i, _) in indexed.iter().skip(k) {
        probs[*i] = 0.0;
    }
    // Renormalise.
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 { for x in probs.iter_mut() { *x /= sum; } }
}

fn top_p_filter(probs: &mut [f32], p: f32) {
    let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut cum = 0.0f32;
    let mut cut = indexed.len();
    for (i, (_, prob)) in indexed.iter().enumerate() {
        cum += prob;
        if cum >= p { cut = i + 1; break; }
    }
    for (i, _) in indexed.iter().skip(cut) {
        probs[*i] = 0.0;
    }
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 { for x in probs.iter_mut() { *x /= sum; } }
}

/// Simple O(n) multinomial sampler using a uniform random draw.
/// Uses xorshift64 seeded from stack address (good enough for inference).
fn multinomial_sample(probs: &[f32]) -> u32 {
    let rand = xorshift64_rand();
    let mut cum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if rand < cum { return i as u32; }
    }
    // Fallback to last non-zero token (floating-point rounding).
    probs.iter().rposition(|&p| p > 0.0).unwrap_or(0) as u32
}

#[inline]
fn xorshift64_rand() -> f32 {
    // Seed from stack pointer — unique enough per call, avoids dependency on rand crate.
    let mut x: u64 = (&x as *const _ as u64).wrapping_mul(0x9e3779b97f4a7c15);
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    let bits = (x.wrapping_mul(0x2545f4914f6cdd1d) >> 41) as u32 | 0x3f80_0000;
    f32::from_bits(bits) - 1.0
}
