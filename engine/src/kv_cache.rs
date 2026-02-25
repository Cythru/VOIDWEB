//! Paged KV-cache block manager (Rust side).
//!
//! Inspired by vLLM's PagedAttention + SGLang's RadixAttention.
//! Blocks are fixed-size (16 tokens by default).  Each sequence has a
//! *block table* mapping logical page indices to physical block IDs.
//!
//! The RadixCache layer sits on top: prefix-matching identical prompt prefixes
//! lets us reuse KV blocks without recomputing them.

use std::collections::VecDeque;
use rustc_hash::FxHashMap;
use anyhow::{Result, bail};

/// Physical block identifier.
pub type BlockId = u32;

/// One KV page descriptor (Rust-side bookkeeping; actual tensors live on GPU).
#[derive(Debug, Clone)]
pub struct KvPage {
    pub block_id:   BlockId,
    /// How many of the `block_size` token slots are occupied.
    pub used_slots: u16,
    /// Reference count (shared prefix caching).
    pub ref_count:  u32,
    /// Token-hash of the tokens stored in this block (for RadixCache lookup).
    pub token_hash: u64,
}

/// Manages allocation and deallocation of KV-cache blocks.
///
/// Thread-safety: wrap in `parking_lot::Mutex`.
pub struct BlockManager {
    pub num_layers:    usize,
    pub num_kv_heads:  usize,
    pub head_dim:      usize,
    pub block_size:    usize,   // tokens per block

    free_blocks:  VecDeque<BlockId>,
    used_blocks:  FxHashMap<BlockId, KvPage>,

    // ── RadixCache: token-sequence hash → list of (num_tokens, [BlockId]) ──
    // Keyed by the rolling hash of the token sequence up to that block.
    prefix_cache: FxHashMap<u64, Vec<BlockId>>,
}

impl BlockManager {
    pub fn new(
        num_layers:   usize,
        num_kv_heads: usize,
        head_dim:     usize,
        block_size:   usize,
        max_blocks:   usize,
    ) -> Self {
        let free_blocks = (0..max_blocks as u32).collect();
        Self {
            num_layers,
            num_kv_heads,
            head_dim,
            block_size,
            free_blocks,
            used_blocks: FxHashMap::default(),
            prefix_cache: FxHashMap::default(),
        }
    }

    /// Total physical blocks available.
    pub fn capacity(&self) -> usize {
        self.free_blocks.len() + self.used_blocks.len()
    }

    /// Free blocks remaining.
    pub fn free_count(&self) -> usize {
        self.free_blocks.len()
    }

    /// Allocate one fresh block.  Returns `None` when OOM.
    pub fn alloc(&mut self) -> Option<BlockId> {
        let id = self.free_blocks.pop_front()?;
        self.used_blocks.insert(id, KvPage {
            block_id:   id,
            used_slots: 0,
            ref_count:  1,
            token_hash: 0,
        });
        Some(id)
    }

    /// Allocate `n` contiguous logical blocks for a new sequence.
    pub fn alloc_seq(&mut self, n: usize) -> Result<Vec<BlockId>> {
        if self.free_blocks.len() < n {
            bail!("KV cache OOM: need {n} blocks, only {} free", self.free_blocks.len());
        }
        (0..n).map(|_| self.alloc().ok_or_else(|| anyhow::anyhow!("alloc race"))).collect()
    }

    /// Increment reference count (prefix sharing).
    pub fn incref(&mut self, id: BlockId) {
        if let Some(page) = self.used_blocks.get_mut(&id) {
            page.ref_count += 1;
        }
    }

    /// Decrement reference count; free when it reaches zero.
    pub fn decref(&mut self, id: BlockId) {
        let should_free = self.used_blocks.get_mut(&id)
            .map(|p| { p.ref_count = p.ref_count.saturating_sub(1); p.ref_count == 0 })
            .unwrap_or(false);
        if should_free {
            self.used_blocks.remove(&id);
            self.free_blocks.push_back(id);
        }
    }

    /// Free all blocks owned by a sequence (decref each).
    pub fn free_seq(&mut self, block_table: &[BlockId]) {
        for &id in block_table {
            self.decref(id);
        }
    }

    // ── RadixCache ─────────────────────────────────────────────────────────────

    /// Try to find a cached block chain matching the given token prefix.
    /// Returns the longest matching chain (may be empty).
    pub fn prefix_lookup(&self, token_ids: &[u32]) -> Vec<BlockId> {
        let mut result = Vec::new();
        let mut hash = 0u64;
        for chunk in token_ids.chunks(self.block_size) {
            hash = rolling_hash(hash, chunk);
            if let Some(blocks) = self.prefix_cache.get(&hash) {
                result.extend_from_slice(blocks);
            } else {
                break;
            }
        }
        result
    }

    /// Register a completed (fully-populated) block into the radix prefix cache.
    pub fn register_prefix_block(&mut self, block_id: BlockId, token_hash: u64) {
        if let Some(page) = self.used_blocks.get_mut(&block_id) {
            page.token_hash = token_hash;
        }
        self.prefix_cache
            .entry(token_hash)
            .or_default()
            .push(block_id);
        self.incref(block_id);
    }
}

// ── Hash utility ──────────────────────────────────────────────────────────────
/// Fast rolling polynomial hash over token IDs.
#[inline(always)]
fn rolling_hash(prev: u64, tokens: &[u32]) -> u64 {
    const P: u64 = 0x9e37_79b9_7f4a_7c15; // Knuth's constant
    let mut h = prev.wrapping_add(tokens.len() as u64 * P);
    for &t in tokens {
        h = h.wrapping_mul(P).wrapping_add(t as u64);
    }
    h
}
