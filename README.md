# VOIDWEB

High-performance LLM inference engine. No Python. No GIL. No mercy.

Written in [BlackMagic](https://github.com/Cythru/BlackMagic) — a systems language that compiles to native code and GPU kernels from a single source tree.

---

## What it is

VOIDWEB is a from-scratch inference engine for large language models, built with the same goals as vLLM and SGLang but without the Python runtime, GIL, or framework overhead.

- **Continuous batching** — requests enter and exit mid-batch without stalling running sequences
- **Paged KV cache** — PagedAttention block management, 16-token pages, ref-counted prefix sharing
- **RadixAttention** — SGLang-style prefix tree caches KV blocks across requests; TTFT drops 4× on shared-prefix workloads
- **FlashAttention-2** — tiled SRAM-resident attention, no O(N²) HBM blowup
- **FP8 GEMM** — E4M3 matrix multiplication via WGMMA, 2× throughput vs FP16 on H100/A100
- **RoPE + RMSNorm kernels** — warp-shuffle reductions, FP8 output path for quantised pipelines
- **OpenAI-compatible HTTP server** — `/v1/chat/completions` with SSE streaming, `/metrics` Prometheus endpoint
- **Zero external runtime deps** — dlopen kernel library at startup, everything else is native

---

## Performance

A100 80 GB · Qwen2.5-7B-Instruct-AWQ · concurrency=16

| Engine | Throughput | TTFT p50 | ITL p50 |
|---|---:|---:|---:|
| TensorRT-LLM | 2,641 tok/s | 44 ms | 5.1 ms |
| **Oracle + RadixCache** | **2,519 tok/s** | **51 ms** | **5.6 ms** |
| SGLang | 2,381 tok/s | 57 ms | 6.2 ms |
| OracleInference | 2,295 tok/s | 61 ms | 6.7 ms |
| vLLM | 2,109 tok/s | 69 ms | 7.4 ms |

With a shared 128-token system prompt, RadixCache drops TTFT from **51 ms → 12 ms** (4.2× reduction, 96.4% cache hit rate).

Full results: [BENCHMARKS.md](BENCHMARKS.md)

---

## Architecture

```
server/main.bm          HTTP server — OpenAI-compatible API, SSE streaming
scheduler/              Continuous-batching scheduler, FCFS + LPM policies
engine/engine.bm        Core orchestrator — loads weights, runs forward pass
engine/kv_cache.bm      Paged block manager, prefix cache hash table
engine/ffi.bm           C-ABI bridge to compiled GPU kernel library
radix_cache/            RadixAttention prefix tree — LRU eviction
tokenizer/              BPE tokenizer — HuggingFace tokenizer.json format
sampler/                Greedy / temperature / top-k / top-p / rep-penalty
quantization/           AWQ, GPTQ, FP8, INT8 scheme detection and routing
loader/                 safetensors loader — zero-copy mmap, parallel shards
metrics/                Prometheus counters — TTFT, ITL, cache hit rate
kernels/
  attention/            FlashAttention-2 (prefill) + PagedAttention (decode)
  gemm/                 FP8 E4M3 GEMM — WGMMA tiled, 128×128×64
  norm/                 RMSNorm — warp shuffle, FP8 output variant
  rope/                 Rotary position embeddings, comptime frequency table
utils/
  arena.bm              Bump allocator, 4 MB per-thread scratch
  ring.bm               Lock-free SPSC + MPMC ring buffers
  simd.bm               AVX-512 / AVX2 / NEON / scalar dispatch
bench/
  bench.bm              Native micro-benchmarks (KV cache, sampler, scheduler)
  bench.rs              HTTP benchmark runner vs vLLM / SGLang / TRT-LLM
  plot.cpp              Terminal chart renderer from bench_results.json
  launcher.zig          Engine process manager
```

---

## Build

```bash
bmc build
```

Requires the `bmc` compiler. GPU kernels compile to PTX via the `.gpu` target.
Links `cuda` and `cudart`.

```bash
# Run the server
bmc run server/main.bm -- /path/to/model ./libkernels.so 0.0.0.0 8000

# Run native benchmarks
bmc bench bench/bench.bm

# Run HTTP benchmarks (requires running engines)
rustc -O --edition 2021 bench/bench.rs -o bench_runner
./bench_runner --concurrency 1 4 16

# Render benchmark charts
g++ -std=c++20 -O2 bench/plot.cpp -o bench_plot
./bench_plot

# Launch all engines for comparison
zig build-exe bench/launcher.zig -O ReleaseFast
./launcher oracle vllm sglang
```

---

## Supported formats

| Format | Status |
|---|---|
| safetensors (single) | Supported |
| safetensors (sharded) | Supported — parallel mmap |
| GGUF | Supported |
| BF16 | Supported |
| FP8 E4M3 | Supported |
| AWQ (INT4) | Supported |
| GPTQ (INT4) | Supported |
| SmoothQuant (INT8) | Supported |

---

## API

OpenAI-compatible. Drop-in replacement for any OpenAI client.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

Endpoints:

| Method | Path | Description |
|---|---|---|
| POST | `/v1/chat/completions` | Chat inference, stream or batch |
| GET | `/v1/models` | List loaded model |
| GET | `/health` | Liveness probe |
| GET | `/metrics` | Prometheus metrics |

---

## License

GPL v3. See [LICENSE](LICENSE).

Part of the [Cythru](https://github.com/Cythru) open-source initiative — open systems, zero abstractions.
