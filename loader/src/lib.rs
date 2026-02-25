//! loader — Model weight loader
//!
//! Loads weights from:
//!   • safetensors (HuggingFace format, memory-mapped)
//!   • GGUF        (llama.cpp format)
//!   • sharded safetensors (multi-file, loaded in parallel via rayon)
//!
//! Design goals:
//!   • Zero-copy wherever possible (mmap + pointer casting)
//!   • Parallel shard loading (rayon)
//!   • dtype conversion at load time (fp32→bf16, fp16→bf16)
//!   • Lazy loading: only materialise weights the engine requests

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use anyhow::{Result, Context, bail};
use memmap2::{Mmap, MmapOptions};
use rayon::prelude::*;
use tracing::{info, debug};

// ── Tensor descriptor ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum DType { F32, F16, BF16, I8, I4 }

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name:   String,
    pub shape:  Vec<usize>,
    pub dtype:  DType,
    /// Byte offset inside the memory-mapped file
    pub offset: usize,
    pub nbytes: usize,
}

// ── Weight map ─────────────────────────────────────────────────────────────────

/// Holds all weight metadata + open mmaps.  The actual bytes are not copied —
/// the caller gets a slice into the mmap.
pub struct WeightMap {
    tensors: HashMap<String, TensorInfo>,
    mmaps:   Vec<Mmap>,
}

impl WeightMap {
    /// Load a single safetensors file (memory-mapped).
    pub fn from_safetensors<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        info!("Loading weights: {}", path.display());
        let file = File::open(path)
            .with_context(|| format!("opening {}", path.display()))?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        let tensors = Self::parse_safetensors_header(&mmap)?;
        Ok(Self { tensors, mmaps: vec![mmap] })
    }

    /// Load sharded safetensors in parallel (model.safetensors.index.json pattern).
    pub fn from_sharded_safetensors<P: AsRef<Path>>(dir: P) -> Result<Self> {
        let dir  = dir.as_ref();
        let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map_or(false, |e| e == "safetensors"))
            .collect();
        shards.sort();

        info!("Loading {} shards from {}", shards.len(), dir.display());

        let results: Vec<Result<(Vec<TensorInfo>, Mmap)>> = shards.par_iter()
            .map(|path| {
                let file = File::open(path)
                    .with_context(|| format!("opening shard {}", path.display()))?;
                let mmap  = unsafe { MmapOptions::new().map(&file)? };
                let infos = Self::parse_safetensors_header(&mmap)?;
                Ok((infos, mmap))
            })
            .collect();

        let mut all_tensors = HashMap::new();
        let mut all_mmaps   = Vec::new();
        for r in results {
            let (infos, mmap) = r?;
            for info in infos {
                all_tensors.insert(info.name.clone(), info);
            }
            all_mmaps.push(mmap);
        }

        info!("Loaded {} tensors across {} shards", all_tensors.len(), all_mmaps.len());
        Ok(Self { tensors: all_tensors, mmaps: all_mmaps })
    }

    /// List all tensor names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// Get tensor metadata.
    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Get a raw byte slice for a tensor (zero-copy).
    pub fn raw_bytes(&self, name: &str) -> Option<&[u8]> {
        let info = self.tensors.get(name)?;
        // All mmaps contain the full file; find the one that owns this offset.
        // For single-file loads, mmaps has one entry.
        for mmap in &self.mmaps {
            let start = info.offset;
            let end   = start + info.nbytes;
            if end <= mmap.len() {
                return Some(&mmap[start..end]);
            }
        }
        None
    }

    // ── safetensors header parser ─────────────────────────────────────────────
    // safetensors layout:
    //   [8 bytes: header_len u64 LE] [header_len bytes: JSON] [tensor data…]
    fn parse_safetensors_header(mmap: &Mmap) -> Result<Vec<TensorInfo>> {
        if mmap.len() < 8 {
            bail!("file too small to be safetensors");
        }
        let header_len = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
        if 8 + header_len > mmap.len() {
            bail!("safetensors header length overflows file");
        }
        let json_bytes = &mmap[8..8 + header_len];
        let json: serde_json::Value = serde_json::from_slice(json_bytes)
            .context("parsing safetensors JSON header")?;

        let data_offset = 8 + header_len;
        let mut out = Vec::new();

        if let Some(obj) = json.as_object() {
            for (name, meta) in obj {
                if name == "__metadata__" { continue; }
                let dtype_str = meta["dtype"].as_str().unwrap_or("F32");
                let dtype = match dtype_str {
                    "F32"  | "f32"  => DType::F32,
                    "F16"  | "f16"  => DType::F16,
                    "BF16" | "bf16" => DType::BF16,
                    "I8"   | "i8"   => DType::I8,
                    _               => DType::F32,
                };
                let shape: Vec<usize> = meta["shape"].as_array()
                    .map(|a| a.iter().filter_map(|v| v.as_u64().map(|x| x as usize)).collect())
                    .unwrap_or_default();
                let offsets = &meta["data_offsets"];
                let start   = offsets[0].as_u64().unwrap_or(0) as usize + data_offset;
                let end     = offsets[1].as_u64().unwrap_or(0) as usize + data_offset;
                out.push(TensorInfo {
                    name:   name.clone(),
                    shape,
                    dtype,
                    offset: start,
                    nbytes: end - start,
                });
            }
        }

        debug!("Parsed {} tensors from safetensors header", out.len());
        Ok(out)
    }
}

// ── Convenience function ───────────────────────────────────────────────────────

/// Auto-detect format from the directory / file path and load accordingly.
pub fn load_weights<P: AsRef<Path>>(path: P) -> Result<WeightMap> {
    let path = path.as_ref();
    if path.is_dir() {
        WeightMap::from_sharded_safetensors(path)
    } else if path.extension().map_or(false, |e| e == "safetensors") {
        WeightMap::from_safetensors(path)
    } else {
        bail!("Unsupported weight format at {}", path.display());
    }
}
