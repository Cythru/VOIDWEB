//! Oracle Inference Engine
//!
//! Core orchestrator: loads weights, runs the forward pass, owns the KV cache.
//! C++ CUDA kernels are invoked via FFI (`libloading`).
//! No Python. No GIL. No mercy.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use parking_lot::{Mutex, RwLock};
use rustc_hash::FxHashMap;
use anyhow::{Result, bail, Context};
use tracing::{info, warn, debug, instrument};
use memmap2::MmapOptions;
use std::fs::File;
use serde::{Serialize, Deserialize};

// ── Re-exports ────────────────────────────────────────────────────────────────
pub use scheduler::{Scheduler, Request, RequestId, BatchView};
pub use tokenizer::Tokenizer;
pub use sampler::{Sampler, SamplerConfig};
pub use quantization::QuantScheme;

// ── FFI bridge to C++/CUDA kernels ───────────────────────────────────────────
mod ffi;
pub use ffi::KernelLib;

// ── KV-Cache block manager ────────────────────────────────────────────────────
pub mod kv_cache;
pub use kv_cache::{BlockManager, BlockId, KvPage};

// ── Model metadata ────────────────────────────────────────────────────────────
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_type:    String,
    pub hidden_size:   usize,
    pub num_heads:     usize,
    pub num_kv_heads:  usize,
    pub num_layers:    usize,
    pub head_dim:      usize,
    pub vocab_size:    usize,
    pub max_seq_len:   usize,
    pub rope_theta:    f32,
    pub rms_norm_eps:  f32,
    pub quant_scheme:  QuantScheme,
    /// Path to the weights directory (safetensors / GGUF)
    pub weights_path:  PathBuf,
    /// Optional tokenizer.json override
    pub tokenizer_path: Option<PathBuf>,
}

impl ModelConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }
}

// ── Engine runtime state ──────────────────────────────────────────────────────
pub struct Engine {
    pub config:       Arc<ModelConfig>,
    /// Memory-mapped weight slabs (one per shard file)
    weight_maps:      Vec<memmap2::Mmap>,
    /// Block manager for paged KV cache
    pub block_mgr:    Arc<Mutex<BlockManager>>,
    /// C++/CUDA kernel library handle
    kernels:          Arc<KernelLib>,
    /// Scheduler (owns request queues, RadixCache)
    pub scheduler:    Arc<Mutex<Scheduler>>,
    /// Tokenizer (memory-mapped BPE vocab)
    pub tokenizer:    Arc<Tokenizer>,
    /// Sampler
    pub sampler:      Arc<Sampler>,
}

impl Engine {
    /// Load model from disk and initialise all subsystems.
    #[instrument(skip_all, fields(model = %cfg.model_type))]
    pub fn load(cfg: ModelConfig, kernel_lib_path: &Path) -> Result<Self> {
        info!("Loading Oracle engine — model={} quant={:?}", cfg.model_type, cfg.quant_scheme);

        // ── Load CUDA kernel library ──────────────────────────────────────────
        let kernels = Arc::new(KernelLib::open(kernel_lib_path)?);
        info!("Kernel library loaded from {}", kernel_lib_path.display());

        // ── Memory-map weight files ───────────────────────────────────────────
        let weight_maps = Self::mmap_weights(&cfg.weights_path)?;
        info!("Mapped {} weight shard(s)", weight_maps.len());

        // ── Block manager (paged KV cache) ────────────────────────────────────
        let block_mgr = BlockManager::new(
            cfg.num_layers,
            cfg.num_kv_heads,
            cfg.head_dim(),
            /* block_size_tokens */ 16,
            /* max_gpu_blocks   */ Self::estimate_gpu_blocks(&cfg),
        );
        let block_mgr = Arc::new(Mutex::new(block_mgr));

        // ── Tokenizer ─────────────────────────────────────────────────────────
        let tok_path = cfg.tokenizer_path.clone()
            .unwrap_or_else(|| cfg.weights_path.join("tokenizer.json"));
        let tokenizer = Arc::new(Tokenizer::load(&tok_path)?);
        info!("Tokenizer loaded — vocab_size={}", tokenizer.vocab_size());

        // ── Scheduler ─────────────────────────────────────────────────────────
        let scheduler = Arc::new(Mutex::new(Scheduler::new(
            cfg.max_seq_len,
            /* max_batch_tokens */ 8192,
            /* max_running_seqs */ 256,
        )));

        // ── Sampler ───────────────────────────────────────────────────────────
        let sampler = Arc::new(Sampler::new(SamplerConfig::default()));

        Ok(Self {
            config: Arc::new(cfg),
            weight_maps,
            block_mgr,
            kernels,
            scheduler,
            tokenizer,
            sampler,
        })
    }

    // ── Forward pass (single decode step) ─────────────────────────────────────
    /// Run one decode step for the current running batch.
    /// Returns a `Vec<(RequestId, u32)>` of (request, next_token) pairs.
    pub fn step(&self) -> Result<Vec<(RequestId, u32)>> {
        let batch = {
            let mut sched = self.scheduler.lock();
            sched.schedule_batch()
        };

        if batch.is_empty() {
            return Ok(vec![]);
        }

        debug!("step: batch_size={} total_tokens={}", batch.num_seqs(), batch.total_tokens());

        // Build GPU input tensors from the batch view, invoke C++ forward pass.
        let logits = self.kernels.forward_pass(&self.config, &batch, &self.block_mgr)?;

        // Sample next token for each sequence.
        let next_tokens = self.sampler.sample_batch(&logits, &batch)?;

        // Update scheduler (append tokens, free finished requests).
        {
            let mut sched = self.scheduler.lock();
            sched.commit_step(&next_tokens)?;
        }

        Ok(next_tokens)
    }

    // ── Helpers ────────────────────────────────────────────────────────────────
    fn mmap_weights(dir: &Path) -> Result<Vec<memmap2::Mmap>> {
        let mut maps = Vec::new();
        let pattern = dir.to_str().context("non-UTF-8 path")?;

        // Accept safetensors shards and GGUF files.
        for ext in &["safetensors", "gguf", "bin"] {
            for entry in std::fs::read_dir(dir)
                .with_context(|| format!("opening weights dir: {}", dir.display()))?
            {
                let entry = entry?;
                let p = entry.path();
                if p.extension().and_then(|e| e.to_str()) == Some(ext) {
                    let f = File::open(&p)
                        .with_context(|| format!("opening {}", p.display()))?;
                    // SAFETY: we only read, never write.
                    let m = unsafe { MmapOptions::new().map(&f) }
                        .with_context(|| format!("mmap {}", p.display()))?;
                    maps.push(m);
                    debug!("Mapped {}", p.display());
                }
            }
            if !maps.is_empty() { break; }
        }

        if maps.is_empty() {
            bail!("No weight files found in {}", dir.display());
        }
        Ok(maps)
    }

    fn estimate_gpu_blocks(cfg: &ModelConfig) -> usize {
        // Heuristic: assume 80 % of an 80 GB A100 is free for KV cache.
        // Each block = block_size * num_layers * 2 (K+V) * num_kv_heads * head_dim * 2 bytes (bf16).
        let block_size_tokens: usize = 16;
        let bytes_per_block = block_size_tokens
            * cfg.num_layers
            * 2
            * cfg.num_kv_heads
            * cfg.head_dim()
            * 2; // bf16
        let available_bytes: usize = 80 * 1024 * 1024 * 1024 * 8 / 10; // 80 % of 80 GB
        (available_bytes / bytes_per_block).max(512)
    }
}
