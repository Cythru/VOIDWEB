//! Oracle Tokenizer
//!
//! Memory-mapped BPE tokenizer.  Supports HuggingFace `tokenizer.json`
//! (the standard format used by Llama, Qwen, Mistral, etc.).
//!
//! Hot paths:
//!   - Vocabulary lookup: `FxHashMap` for O(1) amortised encode
//!   - Batch encode: Rayon parallel over prompt list
//!   - Decode: simple reverse-lookup

use std::path::Path;
use std::fs::File;
use std::sync::Arc;
use rustc_hash::FxHashMap;
use memmap2::MmapOptions;
use anyhow::{Result, Context};
use rayon::prelude::*;
use serde::Deserialize;

// ── Serialisation helpers (tokenizer.json subset) ─────────────────────────────

#[derive(Deserialize)]
struct TokenizerJson {
    model: BpeModel,
    added_tokens: Vec<AddedToken>,
}

#[derive(Deserialize)]
struct BpeModel {
    vocab:  FxHashMap<String, u32>,
    merges: Vec<String>,
}

#[derive(Deserialize)]
struct AddedToken {
    id:      u32,
    content: String,
}

// ── Public API ────────────────────────────────────────────────────────────────

pub struct Tokenizer {
    vocab:          FxHashMap<Vec<u8>, u32>,  // bytes → token id
    vocab_rev:      Vec<Vec<u8>>,             // token id → bytes
    merges:         Vec<(u32, u32, u32)>,     // (left, right, merged)
    pub vocab_size: usize,
    unk_id:         u32,
    bos_id:         u32,
    eos_id:         u32,
}

impl Tokenizer {
    /// Load from a HuggingFace `tokenizer.json`.
    pub fn load(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("opening {}", path.display()))?;
        let mmap = unsafe { MmapOptions::new().map(&file) }
            .with_context(|| format!("mmap {}", path.display()))?;
        let tj: TokenizerJson = serde_json::from_slice(&mmap[..])?;

        let vocab_size = tj.model.vocab.len() + tj.added_tokens.len();

        // Build byte → id lookup.
        let mut vocab: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        for (tok, id) in &tj.model.vocab {
            vocab.insert(decode_hf_token(tok), *id);
        }
        for at in &tj.added_tokens {
            vocab.insert(at.content.as_bytes().to_vec(), at.id);
        }

        // Build id → bytes reverse lookup.
        let mut vocab_rev = vec![vec![0u8]; vocab_size];
        for (bytes, &id) in &vocab {
            if (id as usize) < vocab_size {
                vocab_rev[id as usize] = bytes.clone();
            }
        }

        // Parse merges.
        let merges: Vec<(u32, u32, u32)> = tj.model.merges.iter().filter_map(|m| {
            let mut parts = m.splitn(2, ' ');
            let l = parts.next()?;
            let r = parts.next()?;
            let lid = *vocab.get(l.as_bytes())?;
            let rid = *vocab.get(r.as_bytes())?;
            let merged_bytes = [l.as_bytes(), r.as_bytes()].concat();
            let mid = *vocab.get(&merged_bytes)?;
            Some((lid, rid, mid))
        }).collect();

        // Determine special token IDs (heuristic — override if needed).
        let unk_id = *vocab.get(b"<unk>" as &[u8]).unwrap_or(&0);
        let bos_id = *vocab.get(b"<s>" as &[u8])
            .or_else(|| vocab.get(b"<|begin_of_text|>" as &[u8]))
            .unwrap_or(&1);
        let eos_id = *vocab.get(b"</s>" as &[u8])
            .or_else(|| vocab.get(b"<|end_of_text|>" as &[u8]))
            .unwrap_or(&2);

        Ok(Self { vocab, vocab_rev, merges, vocab_size, unk_id, bos_id, eos_id })
    }

    /// Encode a single string → token IDs.
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let mut ids = if add_bos { vec![self.bos_id] } else { vec![] };
        ids.extend(self.bpe_encode(text.as_bytes()));
        ids
    }

    /// Batch-encode a slice of strings in parallel (Rayon).
    pub fn encode_batch(&self, texts: &[&str], add_bos: bool) -> Vec<Vec<u32>> {
        texts.par_iter()
            .map(|t| self.encode(t, add_bos))
            .collect()
    }

    /// Decode token IDs → UTF-8 string (best-effort).
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if (id as usize) < self.vocab_rev.len() {
                bytes.extend_from_slice(&self.vocab_rev[id as usize]);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn bos_id(&self) -> u32 { self.bos_id }
    pub fn eos_id(&self) -> u32 { self.eos_id }

    // ── BPE encode ─────────────────────────────────────────────────────────────
    fn bpe_encode(&self, bytes: &[u8]) -> Vec<u32> {
        if bytes.is_empty() { return vec![]; }

        // Initialise with byte-level tokens.
        let mut symbols: Vec<u32> = bytes.iter().map(|&b| {
            *self.vocab.get(&[b][..]).unwrap_or(&self.unk_id)
        }).collect();

        // Apply merges in priority order.
        loop {
            let mut best: Option<(usize, u32)> = None; // (position, priority)
            for i in 0..symbols.len().saturating_sub(1) {
                for (pri, &(l, r, _m)) in self.merges.iter().enumerate() {
                    if symbols[i] == l && symbols[i+1] == r {
                        if best.is_none() || pri < best.unwrap().1 as usize {
                            best = Some((i, pri as u32));
                        }
                        break;
                    }
                }
            }
            let Some((pos, pri)) = best else { break };
            let (_, _, merged) = self.merges[pri as usize];
            symbols[pos] = merged;
            symbols.remove(pos + 1);
        }

        symbols
    }
}

// ── HuggingFace token decoding ─────────────────────────────────────────────────
/// HF tokenizer.json stores tokens with special Ġ/Ċ escapes.
fn decode_hf_token(tok: &str) -> Vec<u8> {
    let mut bytes = Vec::new();
    for ch in tok.chars() {
        match ch {
            'Ġ' => bytes.push(b' '),
            'Ċ' => bytes.push(b'\n'),
            'ĉ' => bytes.push(b'\t'),
            c if (c as u32) < 256 => bytes.push(c as u8),
            _ => {
                let mut buf = [0u8; 4];
                let s = ch.encode_utf8(&mut buf);
                bytes.extend_from_slice(s.as_bytes());
            }
        }
    }
    bytes
}
