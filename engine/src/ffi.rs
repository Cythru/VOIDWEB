//! FFI bridge to the C++/CUDA shared library (`libkernels.so`).
//!
//! The library is loaded once at startup with `libloading`.
//! Function pointers are resolved lazily (first call) so startup stays fast.

use std::path::Path;
use std::sync::Arc;
use parking_lot::Mutex;
use anyhow::{Result, Context};
use libloading::{Library, Symbol};

use crate::{ModelConfig, kv_cache::BlockManager, BatchView, RequestId};

// ── ABI types shared with the C++ side ───────────────────────────────────────

/// Opaque handle returned by the C++ model allocator.
#[repr(C)]
pub struct CModelHandle(std::ffi::c_void);

/// Flat descriptor of a single sequence for the CUDA forward pass.
#[repr(C)]
pub struct SeqDescriptor {
    pub seq_id:         u64,
    pub token_ids:      *const u32,
    pub num_tokens:     u32,
    pub block_table:    *const u32, // physical block IDs for this seq
    pub num_blocks:     u32,
    pub is_prefill:     u8,
}

// ── Function pointer types ────────────────────────────────────────────────────
type FnModelAlloc  = unsafe extern "C" fn(config_json: *const std::ffi::c_char, config_len: usize) -> *mut CModelHandle;
type FnModelFree   = unsafe extern "C" fn(handle: *mut CModelHandle);
type FnForwardPass = unsafe extern "C" fn(
    handle:     *mut CModelHandle,
    seqs:       *const SeqDescriptor,
    num_seqs:   u32,
    kv_k:       *mut f16_raw,
    kv_v:       *mut f16_raw,
    logits_out: *mut f32,  // [num_seqs, vocab_size]
) -> i32; // 0 = success

// BF16/F16 raw storage (u16)
type f16_raw = u16;

// ── KernelLib ─────────────────────────────────────────────────────────────────
pub struct KernelLib {
    _lib:         Library,
    model_alloc:  FnModelAlloc,
    model_free:   FnModelFree,
    forward_pass: FnForwardPass,
    model_handle: Mutex<*mut CModelHandle>,
}

// SAFETY: the C++ library is thread-safe — all mutable state is protected
// by CUDA stream serialization on the library side.
unsafe impl Send for KernelLib {}
unsafe impl Sync for KernelLib {}

impl KernelLib {
    pub fn open(path: &Path) -> Result<Self> {
        // SAFETY: valid path, library is trusted native code.
        let lib = unsafe {
            Library::new(path).with_context(|| format!("loading kernel lib: {}", path.display()))?
        };

        let model_alloc:  FnModelAlloc  = unsafe { *lib.get(b"oracle_model_alloc\0")? };
        let model_free:   FnModelFree   = unsafe { *lib.get(b"oracle_model_free\0")? };
        let forward_pass: FnForwardPass = unsafe { *lib.get(b"oracle_forward_pass\0")? };

        Ok(Self {
            _lib:         lib,
            model_alloc,
            model_free,
            forward_pass,
            model_handle: Mutex::new(std::ptr::null_mut()),
        })
    }

    /// Allocate a model instance on the GPU side.
    pub fn init_model(&self, cfg: &ModelConfig) -> Result<()> {
        let json = serde_json::to_string(cfg)?;
        let handle = unsafe {
            (self.model_alloc)(json.as_ptr() as *const _, json.len())
        };
        if handle.is_null() {
            anyhow::bail!("oracle_model_alloc returned null — check kernel log");
        }
        *self.model_handle.lock() = handle;
        Ok(())
    }

    /// Run one forward pass for the given batch.
    /// Returns logits as a flat Vec<f32> shaped [num_seqs × vocab_size].
    pub fn forward_pass(
        &self,
        cfg:       &ModelConfig,
        batch:     &BatchView,
        block_mgr: &Mutex<crate::kv_cache::BlockManager>,
    ) -> Result<Vec<f32>> {
        let handle = *self.model_handle.lock();
        if handle.is_null() {
            // Lazy init on first forward pass.
            drop(self.model_handle.lock()); // unlock before init
            self.init_model(cfg)?;
        }

        let (descs, _token_storage, _block_storage) = batch.build_seq_descriptors();
        let vocab = cfg.vocab_size;
        let num_seqs = descs.len();
        let mut logits = vec![0f32; num_seqs * vocab];

        // We pass null for kv pointers — the C++ side manages CUDA-side KV memory
        // directly through the block manager's physical addresses.
        let rc = unsafe {
            (self.forward_pass)(
                handle,
                descs.as_ptr(),
                num_seqs as u32,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                logits.as_mut_ptr(),
            )
        };

        if rc != 0 {
            anyhow::bail!("oracle_forward_pass returned error code {rc}");
        }
        Ok(logits)
    }
}

impl Drop for KernelLib {
    fn drop(&mut self) {
        let handle = *self.model_handle.lock();
        if !handle.is_null() {
            unsafe { (self.model_free)(handle) };
        }
    }
}
