//! Oracle Quantization
//!
//! Rust-side helpers for quantized weight formats.
//! The actual dequant math runs in C++/CUDA; this crate handles:
//!   - Format detection (AWQ, GPTQ, FP8, INT4, BF16)
//!   - Metadata parsing (scales, zeros, group sizes)
//!   - Weight tensor routing to the correct C++ dequant kernel

use std::path::Path;
use memmap2::MmapOptions;
use std::fs::File;
use anyhow::{Result, bail};
use serde::{Serialize, Deserialize};

// ── Quantization scheme ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum QuantScheme {
    #[default]
    BFloat16,   // Full precision BF16 (baseline)
    Float8,     // FP8 E4M3 — best throughput on H100/A100
    Int4Awq,    // Activation-aware weight quantization (MIT/LMDeploy)
    Int4Gptq,   // GPTQ 4-bit (per-group)
    Int8Sq,     // SmoothQuant INT8
}

impl QuantScheme {
    /// Bytes per element on disk.
    pub fn bytes_per_element(&self) -> f32 {
        match self {
            Self::BFloat16  => 2.0,
            Self::Float8    => 1.0,
            Self::Int4Awq | Self::Int4Gptq => 0.5,
            Self::Int8Sq    => 1.0,
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "bf16" | "bfloat16"  => Some(Self::BFloat16),
            "fp8" | "float8"     => Some(Self::Float8),
            "awq" | "int4_awq"   => Some(Self::Int4Awq),
            "gptq" | "int4_gptq" => Some(Self::Int4Gptq),
            "sq"  | "int8_sq"    => Some(Self::Int8Sq),
            _                    => None,
        }
    }
}

// ── AWQ metadata ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AwqConfig {
    pub group_size: usize,    // typically 128
    pub zero_point: bool,     // whether zero-point is stored
    pub version:    u8,       // AWQ version (1 or 2)
}

// ── GPTQ metadata ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GptqConfig {
    pub bits:        u32,    // quantization bits (typically 4)
    pub group_size:  i32,    // -1 = per-column, else per-group
    pub desc_act:    bool,   // activation reordering
    pub sym:         bool,   // symmetric quantization
}

// ── Weight tensor descriptor ───────────────────────────────────────────────────

/// All metadata the C++ kernel needs to dequantize one linear layer.
#[derive(Debug, Clone)]
pub struct WeightTensor {
    pub name:        String,
    pub scheme:      QuantScheme,
    /// Pointer (as usize) into the memory-mapped weight slab.
    /// SAFETY: caller must ensure the Mmap outlives this struct.
    pub data_ptr:    usize,
    pub data_len:    usize,
    pub rows:        usize,
    pub cols:        usize,
    pub scales_ptr:  Option<usize>,
    pub zeros_ptr:   Option<usize>,
    pub group_size:  Option<usize>,
}

impl WeightTensor {
    /// Size in bytes of the quantized weight data.
    pub fn size_bytes(&self) -> usize {
        (self.rows * self.cols) as f32 as usize
            * (self.scheme.bytes_per_element() as usize).max(1)
    }

    pub fn as_raw(&self) -> (*const u8, usize) {
        (self.data_ptr as *const u8, self.data_len)
    }
}

// ── Safe-tensors metadata parser (minimal) ────────────────────────────────────

/// Parse just enough of the safetensors header to extract tensor offsets.
pub fn parse_safetensors_header(mmap: &[u8]) -> Result<Vec<TensorMeta>> {
    if mmap.len() < 8 { bail!("safetensors too small"); }
    let header_size = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
    if mmap.len() < 8 + header_size { bail!("safetensors header truncated"); }
    let header_json = &mmap[8..8+header_size];
    let v: serde_json::Value = serde_json::from_slice(header_json)?;

    let mut tensors = Vec::new();
    if let Some(obj) = v.as_object() {
        for (name, meta) in obj {
            if name == "__metadata__" { continue; }
            if let (Some(dtype), Some(offs)) = (
                meta["dtype"].as_str(),
                meta["data_offsets"].as_array()
            ) {
                let start = offs.first().and_then(|x| x.as_u64()).unwrap_or(0) as usize;
                let end   = offs.last().and_then(|x| x.as_u64()).unwrap_or(0) as usize;
                let shape: Vec<usize> = meta["shape"].as_array()
                    .map(|a| a.iter().filter_map(|x| x.as_u64().map(|v| v as usize)).collect())
                    .unwrap_or_default();
                tensors.push(TensorMeta {
                    name:       name.clone(),
                    dtype:      dtype.to_string(),
                    data_start: 8 + header_size + start,
                    data_end:   8 + header_size + end,
                    shape,
                });
            }
        }
    }
    Ok(tensors)
}

#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub name:       String,
    pub dtype:      String,
    pub data_start: usize,
    pub data_end:   usize,
    pub shape:      Vec<usize>,
}

// ── Utility: detect quantisation scheme from config.json ─────────────────────

pub fn detect_quant_scheme(config_json: &serde_json::Value) -> QuantScheme {
    // Check quantization_config block.
    if let Some(qc) = config_json.get("quantization_config") {
        let quant_type = qc["quant_type"].as_str()
            .or_else(|| qc["quant_method"].as_str())
            .unwrap_or("");
        if let Some(s) = QuantScheme::from_str(quant_type) { return s; }
    }
    // Fall back to torch_dtype.
    if let Some(dtype) = config_json["torch_dtype"].as_str() {
        if let Some(s) = QuantScheme::from_str(dtype) { return s; }
    }
    QuantScheme::BFloat16
}
