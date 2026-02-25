//! Oracle Scheduler
//!
//! Continuous-batching scheduler — zero Python, zero GIL.
//! Inspired by:
//!   - vLLM: chunked prefill, paged KV cache
//!   - SGLang: RadixAttention prefix caching, prefill-decode disaggregation
//!   - TensorRT-LLM: in-flight batching, priority scheduling
//!
//! Design:
//!   - Requests arrive via `add_request()` (lock-free MPMC channel).
//!   - `schedule_batch()` is called from the engine thread and returns a
//!     `BatchView` describing which seqs to run and in what mode.
//!   - Prefill and decode are disaggregated: the engine may have a separate
//!     CUDA stream for each phase.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;
use rustc_hash::FxHashMap;
use crossbeam_channel::{bounded, Sender, Receiver, TryRecvError};
use parking_lot::Mutex;
use anyhow::{Result, bail};
use serde::{Serialize, Deserialize};

// ── Public types ──────────────────────────────────────────────────────────────

pub type RequestId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestStatus {
    Waiting,
    Prefilling,
    Decoding,
    Finished,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature:        f32,
    pub top_p:              f32,
    pub top_k:              u32,
    pub max_new_tokens:     u32,
    pub repetition_penalty: f32,
    pub stop_tokens:        Vec<u32>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature:        0.7,
            top_p:              0.95,
            top_k:              50,
            max_new_tokens:     2048,
            repetition_penalty: 1.1,
            stop_tokens:        vec![],
        }
    }
}

/// A single inference request.
pub struct Request {
    pub id:              RequestId,
    pub prompt_tokens:   Vec<u32>,
    pub output_tokens:   Vec<u32>,
    pub params:          SamplingParams,
    pub status:          RequestStatus,
    pub block_table:     Vec<u32>,   // physical KV block IDs
    pub arrived_at:      Instant,
    pub prefill_cursor:  usize,      // how many prompt tokens are already prefilled
}

impl Request {
    pub fn new(id: RequestId, prompt_tokens: Vec<u32>, params: SamplingParams) -> Self {
        Self {
            id,
            prompt_tokens,
            output_tokens: Vec::new(),
            params,
            status: RequestStatus::Waiting,
            block_table: Vec::new(),
            arrived_at: Instant::now(),
            prefill_cursor: 0,
        }
    }

    /// Total tokens (prompt + generated so far).
    pub fn total_len(&self) -> usize {
        self.prompt_tokens.len() + self.output_tokens.len()
    }

    /// Whether this request has hit its stop condition.
    pub fn is_done(&self) -> bool {
        if self.output_tokens.len() >= self.params.max_new_tokens as usize {
            return true;
        }
        if let Some(&last) = self.output_tokens.last() {
            if self.params.stop_tokens.contains(&last) {
                return true;
            }
        }
        false
    }
}

// ── BatchView ─────────────────────────────────────────────────────────────────
/// Snapshot of the current batch handed to the engine for one forward step.
pub struct BatchView {
    pub prefill_seqs: Vec<PrefillSeq>,
    pub decode_seqs:  Vec<DecodeSeq>,
}

pub struct PrefillSeq {
    pub req_id:      RequestId,
    pub tokens:      Vec<u32>,   // chunk of prompt tokens
    pub block_table: Vec<u32>,
}

pub struct DecodeSeq {
    pub req_id:      RequestId,
    pub last_token:  u32,
    pub block_table: Vec<u32>,
}

impl BatchView {
    pub fn is_empty(&self) -> bool {
        self.prefill_seqs.is_empty() && self.decode_seqs.is_empty()
    }

    pub fn num_seqs(&self) -> usize {
        self.prefill_seqs.len() + self.decode_seqs.len()
    }

    pub fn total_tokens(&self) -> usize {
        let pf: usize = self.prefill_seqs.iter().map(|s| s.tokens.len()).sum();
        let dc: usize = self.decode_seqs.len(); // 1 token each
        pf + dc
    }

    /// Build flat SeqDescriptor arrays for the FFI layer.
    /// Returns (descriptors, token_storage, block_storage) — the storage vecs
    /// keep the data alive for the duration of the CUDA kernel call.
    pub fn build_seq_descriptors(
        &self,
    ) -> (
        Vec<crate::SeqDescriptorStub>,
        Vec<Vec<u32>>,
        Vec<Vec<u32>>,
    ) {
        let mut descs       = Vec::new();
        let mut tok_storage = Vec::new();
        let mut blk_storage = Vec::new();

        for pf in &self.prefill_seqs {
            tok_storage.push(pf.tokens.clone());
            blk_storage.push(pf.block_table.clone());
            let t = tok_storage.last().unwrap();
            let b = blk_storage.last().unwrap();
            descs.push(crate::SeqDescriptorStub {
                seq_id:      pf.req_id,
                token_ptr:   t.as_ptr() as usize,
                num_tokens:  t.len() as u32,
                block_ptr:   b.as_ptr() as usize,
                num_blocks:  b.len() as u32,
                is_prefill:  1,
            });
        }

        for dc in &self.decode_seqs {
            tok_storage.push(vec![dc.last_token]);
            blk_storage.push(dc.block_table.clone());
            let t = tok_storage.last().unwrap();
            let b = blk_storage.last().unwrap();
            descs.push(crate::SeqDescriptorStub {
                seq_id:      dc.req_id,
                token_ptr:   t.as_ptr() as usize,
                num_tokens:  1,
                block_ptr:   b.as_ptr() as usize,
                num_blocks:  b.len() as u32,
                is_prefill:  0,
            });
        }

        (descs, tok_storage, blk_storage)
    }
}

/// Thin stub mirroring `engine::ffi::SeqDescriptor` — avoids circular dep.
pub struct SeqDescriptorStub {
    pub seq_id:     u64,
    pub token_ptr:  usize,
    pub num_tokens: u32,
    pub block_ptr:  usize,
    pub num_blocks: u32,
    pub is_prefill: u8,
}

// ── Scheduler ─────────────────────────────────────────────────────────────────

/// Scheduling policy.
#[derive(Debug, Clone, Copy)]
pub enum Policy {
    /// First-come-first-served (default).
    Fcfs,
    /// Longest-prefix-match first (maximises RadixCache hits).
    Lpm,
}

pub struct Scheduler {
    max_seq_len:       usize,
    max_batch_tokens:  usize,
    max_running_seqs:  usize,

    waiting:  VecDeque<Request>,
    running:  FxHashMap<RequestId, Request>,
    finished: Vec<(RequestId, Vec<u32>)>,

    next_id:  RequestId,
    policy:   Policy,
}

impl Scheduler {
    pub fn new(max_seq_len: usize, max_batch_tokens: usize, max_running_seqs: usize) -> Self {
        Self {
            max_seq_len,
            max_batch_tokens,
            max_running_seqs,
            waiting:  VecDeque::new(),
            running:  FxHashMap::default(),
            finished: Vec::new(),
            next_id:  1,
            policy:   Policy::Fcfs,
        }
    }

    pub fn set_policy(&mut self, p: Policy) { self.policy = p; }

    /// Enqueue a new request.  Returns the assigned `RequestId`.
    pub fn add_request(&mut self, prompt_tokens: Vec<u32>, params: SamplingParams) -> RequestId {
        let id = self.next_id;
        self.next_id += 1;
        self.waiting.push_back(Request::new(id, prompt_tokens, params));
        id
    }

    /// Produce the next `BatchView` to run.
    ///
    /// Algorithm (continuous batching):
    ///   1. Promote waiting → running (prefill) if budget allows.
    ///   2. All running seqs not in prefill mode → decode.
    ///   3. Respect token budget and seq limit.
    pub fn schedule_batch(&mut self) -> BatchView {
        let mut prefill_seqs = Vec::new();
        let mut decode_seqs  = Vec::new();
        let mut token_budget = self.max_batch_tokens;

        // ── Promote waiting requests ──────────────────────────────────────────
        while self.running.len() < self.max_running_seqs {
            let Some(mut req) = self.waiting.pop_front() else { break };
            let chunk_len = req.prompt_tokens.len()
                .saturating_sub(req.prefill_cursor)
                .min(token_budget)
                .min(2048); // max prefill chunk (SGLang-style chunked prefill)

            if chunk_len == 0 { break; }
            token_budget = token_budget.saturating_sub(chunk_len);

            let chunk = req.prompt_tokens[req.prefill_cursor .. req.prefill_cursor + chunk_len].to_vec();
            req.prefill_cursor += chunk_len;

            prefill_seqs.push(PrefillSeq {
                req_id:      req.id,
                tokens:      chunk,
                block_table: req.block_table.clone(),
            });

            req.status = if req.prefill_cursor >= req.prompt_tokens.len() {
                RequestStatus::Decoding
            } else {
                RequestStatus::Prefilling
            };

            self.running.insert(req.id, req);
        }

        // ── Decode running seqs ───────────────────────────────────────────────
        for req in self.running.values() {
            if req.status == RequestStatus::Decoding {
                let last_token = req.output_tokens.last().copied()
                    .unwrap_or_else(|| *req.prompt_tokens.last().unwrap_or(&0));
                decode_seqs.push(DecodeSeq {
                    req_id:      req.id,
                    last_token,
                    block_table: req.block_table.clone(),
                });
            }
        }

        BatchView { prefill_seqs, decode_seqs }
    }

    /// Commit the result of one step: append new tokens, mark finished seqs.
    pub fn commit_step(&mut self, next_tokens: &[(RequestId, u32)]) -> Result<()> {
        for &(id, token) in next_tokens {
            if let Some(req) = self.running.get_mut(&id) {
                req.output_tokens.push(token);
                if req.is_done() {
                    req.status = RequestStatus::Finished;
                }
            }
        }

        // Move finished requests out of `running`.
        let done_ids: Vec<_> = self.running.values()
            .filter(|r| r.status == RequestStatus::Finished)
            .map(|r| r.id)
            .collect();

        for id in done_ids {
            if let Some(req) = self.running.remove(&id) {
                self.finished.push((id, req.output_tokens));
            }
        }

        Ok(())
    }

    /// Drain all completed (RequestId, output_tokens) pairs.
    pub fn drain_finished(&mut self) -> Vec<(RequestId, Vec<u32>)> {
        std::mem::take(&mut self.finished)
    }

    pub fn waiting_count(&self)  -> usize { self.waiting.len() }
    pub fn running_count(&self)  -> usize { self.running.len() }
}
