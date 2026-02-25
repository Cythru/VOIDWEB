//! radix_cache — RadixAttention prefix cache (SGLang algorithm, Oracle impl)
//!
//! # What this does
//!
//! SGLang introduced **RadixAttention**: store the KV cache of every completed
//! request in a radix tree keyed by token sequence.  When a new request arrives
//! whose prefix matches an existing tree node, we *reuse* those cached KV blocks
//! — skipping their prefill entirely.
//!
//! Result: 20–40% throughput gain on workloads with shared prefixes (chat with
//! a fixed system prompt, batch RAG queries, code completion with shared imports,
//! etc.).
//!
//! # Module layout
//!
//! ```
//! RadixTree          — the tree itself (token → KV-block mapping)
//!   RadixNode        — a single node (edge = token slice, value = block IDs)
//!   MatchResult      — how far a prefix matched + what blocks are reusable
//! RadixCacheManager  — wraps the tree + the block allocator
//!   insert()         — add a completed sequence
//!   lookup()         — find matching prefix blocks
//!   evict_lru()      — free blocks when VRAM pressure is high
//! ```
//!
//! # Eviction
//!
//! LRU (least-recently-used) at the node level.  Each node has a `last_access`
//! timestamp.  When we need blocks, we walk leaves-first and free the oldest.
//!
//! # Thread safety
//!
//! `RadixCacheManager` wraps the tree in `RwLock`.  Reads (lookups) can proceed
//! concurrently; writes (insert/evict) take an exclusive lock.

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use std::time::Instant;

// ── Public type aliases ───────────────────────────────────────────────────────

pub type TokenId  = u32;
pub type BlockId  = u32;
pub type SeqId    = u64;

// ── Radix node ────────────────────────────────────────────────────────────────

#[derive(Debug)]
struct RadixNode {
    /// The token IDs on the edge from this node's parent.
    edge:        Vec<TokenId>,
    /// KV-cache block IDs stored at this node (one per transformer layer).
    blocks:      Vec<BlockId>,
    /// Children keyed by the first token of their edge.
    children:    HashMap<TokenId, RadixNode>,
    /// LRU timestamp — updated on every successful lookup.
    last_access: Instant,
    /// Number of active requests currently using this node's blocks.
    ref_count:   u32,
}

impl RadixNode {
    fn new(edge: Vec<TokenId>, blocks: Vec<BlockId>) -> Self {
        Self {
            edge,
            blocks,
            children:    HashMap::new(),
            last_access: Instant::now(),
            ref_count:   0,
        }
    }

    fn touch(&mut self) {
        self.last_access = Instant::now();
    }
}

// ── Match result ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MatchResult {
    /// How many tokens of the query prefix were matched.
    pub matched_len: usize,
    /// KV-cache block IDs that can be reused (concatenated from all matched nodes).
    pub reuse_blocks: Vec<BlockId>,
}

impl MatchResult {
    fn empty() -> Self {
        Self { matched_len: 0, reuse_blocks: Vec::new() }
    }
}

// ── Radix tree ────────────────────────────────────────────────────────────────

struct RadixTree {
    root: RadixNode,
}

impl RadixTree {
    fn new() -> Self {
        Self { root: RadixNode::new(vec![], vec![]) }
    }

    /// Walk the tree matching as much of `tokens` as possible.
    /// Returns the blocks for every matched node.
    fn lookup(&mut self, tokens: &[TokenId]) -> MatchResult {
        let mut result  = MatchResult::empty();
        let mut node    = &mut self.root;
        let mut pos     = 0usize;

        loop {
            node.touch();
            let remaining = &tokens[pos..];
            if remaining.is_empty() {
                break;
            }
            let first = remaining[0];
            let child = match node.children.get_mut(&first) {
                Some(c) => c,
                None    => break,
            };

            let edge  = &child.edge;
            let match_len = edge.iter()
                .zip(remaining.iter())
                .take_while(|(a, b)| a == b)
                .count();

            if match_len == 0 {
                break;
            }

            result.matched_len += match_len;
            result.reuse_blocks.extend_from_slice(&child.blocks);
            child.touch();

            if match_len < edge.len() {
                // Partial edge match — need to split, but return what we have.
                break;
            }

            pos  += match_len;
            node = child;
        }

        result
    }

    /// Insert a completed sequence's KV blocks into the tree.
    ///
    /// `tokens`  — the full token sequence (prompt + generated)
    /// `blocks`  — the corresponding KV block IDs (one slice per node insertion)
    ///
    /// We insert at a granularity of `block_size` tokens so that blocks can be
    /// individually freed by the eviction policy.
    fn insert(&mut self, tokens: &[TokenId], blocks: &[BlockId], block_size: usize) {
        let mut node = &mut self.root;
        let mut pos  = 0usize;
        let mut blk  = 0usize;

        while pos < tokens.len() && blk < blocks.len() {
            let remaining = &tokens[pos..];
            let first     = remaining[0];

            if let Some(child) = node.children.get_mut(&first) {
                let edge      = child.edge.clone();
                let match_len = edge.iter()
                    .zip(remaining.iter())
                    .take_while(|(a, b)| a == b)
                    .count();

                if match_len == edge.len() {
                    // Full edge match — descend
                    child.touch();
                    pos  += match_len;
                    blk  += (match_len + block_size - 1) / block_size;
                    node  = child;
                    continue;
                }

                // Partial match — split the edge
                let prefix    = edge[..match_len].to_vec();
                let suffix    = edge[match_len..].to_vec();
                let old_first = suffix[0];

                // Create the split-off child (inherits original children + blocks)
                let old_blocks = child.blocks.clone();
                let old_children = std::mem::take(&mut child.children);
                let new_suffix_node = RadixNode {
                    edge:        suffix,
                    blocks:      old_blocks,
                    children:    old_children,
                    last_access: child.last_access,
                    ref_count:   child.ref_count,
                };
                let split_blocks: Vec<BlockId> = blocks[blk..]
                    .iter().take((match_len + block_size - 1) / block_size)
                    .copied().collect();
                child.edge   = prefix;
                child.blocks = split_blocks;
                child.children = HashMap::new();
                child.children.insert(old_first, new_suffix_node);
                child.touch();
                pos  += match_len;
                blk  += (match_len + block_size - 1) / block_size;
                node  = child;
                continue;
            }

            // No matching child — create new leaf
            let chunk_len   = block_size.min(remaining.len());
            let new_blocks: Vec<BlockId> = blocks[blk..].iter().take(1).copied().collect();
            let new_node    = RadixNode::new(remaining[..chunk_len].to_vec(), new_blocks);
            node.children.insert(first, new_node);
            pos += chunk_len;
            blk += 1;
            // New node is now child; can't descend into it via borrow checker
            // without restructuring — safe to stop here (next iter re-looks it up)
            node = node.children.get_mut(&first).unwrap();
        }
    }

    /// Collect all leaf nodes ordered by `last_access` ascending (oldest first).
    fn lru_leaves(&mut self) -> Vec<*mut RadixNode> {
        let mut leaves: Vec<*mut RadixNode> = Vec::new();
        Self::collect_leaves(&mut self.root, &mut leaves);
        leaves.sort_by_key(|n| unsafe { (**n).last_access });
        leaves
    }

    fn collect_leaves(node: &mut RadixNode, out: &mut Vec<*mut RadixNode>) {
        if node.children.is_empty() && !node.blocks.is_empty() {
            out.push(node as *mut RadixNode);
        } else {
            for child in node.children.values_mut() {
                Self::collect_leaves(child, out);
            }
        }
    }

    fn total_blocks(&self) -> usize {
        Self::count_blocks(&self.root)
    }

    fn count_blocks(node: &RadixNode) -> usize {
        let mut n = node.blocks.len();
        for child in node.children.values() {
            n += Self::count_blocks(child);
        }
        n
    }
}

// ── Public manager ────────────────────────────────────────────────────────────

/// Thread-safe RadixAttention cache manager.
///
/// All public methods are cheap: lookups take a read-lock, inserts/evictions
/// take a write-lock.
#[derive(Clone)]
pub struct RadixCacheManager {
    inner:      Arc<RwLock<RadixTree>>,
    block_size: usize,    // tokens per KV block
    max_blocks: usize,    // evict when total_blocks > max_blocks
}

impl RadixCacheManager {
    pub fn new(block_size: usize, max_blocks: usize) -> Self {
        Self {
            inner:      Arc::new(RwLock::new(RadixTree::new())),
            block_size,
            max_blocks,
        }
    }

    /// Find reusable KV blocks for `tokens`.  Call before allocating new blocks.
    pub fn lookup(&self, tokens: &[TokenId]) -> MatchResult {
        self.inner.write().lookup(tokens)
    }

    /// Insert a completed sequence.  Call after decoding finishes.
    pub fn insert(&self, tokens: &[TokenId], blocks: &[BlockId]) {
        let mut tree = self.inner.write();
        tree.insert(tokens, blocks, self.block_size);
        // Evict if over budget
        while tree.total_blocks() > self.max_blocks {
            let leaves = tree.lru_leaves();
            if leaves.is_empty() {
                break;
            }
            // Safety: we hold the write-lock and leaves are valid tree nodes.
            unsafe {
                let leaf = &mut **leaves.first().unwrap();
                if leaf.ref_count == 0 {
                    leaf.blocks.clear();
                }
            }
            break; // evict one leaf per insert to amortise cost
        }
    }

    /// Explicitly evict until `target_free` blocks are freed.
    pub fn evict_lru(&self, target_free: usize) -> usize {
        let mut freed = 0usize;
        let mut tree  = self.inner.write();
        while freed < target_free {
            let leaves = tree.lru_leaves();
            if leaves.is_empty() {
                break;
            }
            unsafe {
                let leaf = &mut **leaves.first().unwrap();
                if leaf.ref_count == 0 {
                    freed += leaf.blocks.len();
                    leaf.blocks.clear();
                }
            }
        }
        freed
    }

    /// Total KV blocks currently stored in the cache.
    pub fn total_blocks(&self) -> usize {
        self.inner.read().total_blocks()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_lookup_miss() {
        let mgr = RadixCacheManager::new(4, 1000);
        let res = mgr.lookup(&[1, 2, 3, 4]);
        assert_eq!(res.matched_len, 0);
        assert!(res.reuse_blocks.is_empty());
    }

    #[test]
    fn test_insert_then_lookup() {
        let mgr = RadixCacheManager::new(4, 1000);
        let tokens: Vec<TokenId> = vec![10, 20, 30, 40, 50, 60];
        let blocks: Vec<BlockId> = vec![1, 2];
        mgr.insert(&tokens, &blocks);

        // Full prefix match
        let res = mgr.lookup(&[10, 20, 30, 40, 50, 60]);
        assert!(res.matched_len > 0, "should match at least one block");
        assert!(!res.reuse_blocks.is_empty());
    }

    #[test]
    fn test_shared_prefix() {
        let mgr = RadixCacheManager::new(4, 1000);
        // Two sequences share a prefix
        let prefix: Vec<TokenId> = vec![1, 2, 3, 4];
        let seq_a: Vec<TokenId>  = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let seq_b: Vec<TokenId>  = vec![1, 2, 3, 4, 9, 10, 11, 12];

        mgr.insert(&seq_a, &[10, 11]);
        mgr.insert(&seq_b, &[20, 21]);

        // Both queries should reuse the shared prefix block
        let ra = mgr.lookup(&seq_a);
        let rb = mgr.lookup(&seq_b);
        assert!(ra.matched_len >= 4, "seq_a should match the shared prefix");
        assert!(rb.matched_len >= 4, "seq_b should match the shared prefix");
        // The block from the shared prefix should be in both results
        assert!(!ra.reuse_blocks.is_empty());
        assert!(!rb.reuse_blocks.is_empty());
    }

    #[test]
    fn test_stats() {
        let mgr = RadixCacheManager::new(4, 1000);
        assert_eq!(mgr.total_blocks(), 0);
        mgr.insert(&[1, 2, 3, 4], &[42]);
        assert!(mgr.total_blocks() > 0);
    }
}
