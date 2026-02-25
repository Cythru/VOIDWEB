#!/usr/bin/env python3
"""
OracleInference Benchmark Suite
================================
Measures and compares inference performance across:
  - OracleInference  (your engine)
  - vLLM
  - TensorRT-LLM
  - SGLang
  - llama.cpp

All engines expose OpenAI-compatible /v1/chat/completions.

Metrics:
  TTFT    — time to first token  (ms)
  ITL     — inter-token latency  (ms/token)
  Tput    — output throughput     (tokens/sec, aggregated over concurrent requests)
  Memory  — GPU VRAM used         (MiB, sampled during run)

Usage:
  python bench.py                        # full suite, all engines
  python bench.py --engines oracle vllm  # specific engines
  python bench.py --concurrency 1 8 32   # custom concurrency levels
  python bench.py --prompt-len 512 --output-len 256
  python bench.py --model "Qwen/Qwen2.5-7B-Instruct-AWQ"

Each engine must be running before the benchmark starts.
See ENGINES dict below for default ports — override with env vars.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    import aiohttp
    _HAS_AIOHTTP = True
except ImportError:
    _HAS_AIOHTTP = False

try:
    import requests as _req
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


# ── Engine registry ───────────────────────────────────────────────────────────
# Override ports via env: ORACLE_PORT=8000 VLLM_PORT=8001 etc.

ENGINES: Dict[str, Dict] = {
    "oracle": {
        "name":  "OracleInference",
        "emoji": "🔮",
        "url":   f"http://localhost:{os.getenv('ORACLE_PORT', '8000')}",
        "color": "\033[95m",      # magenta
        "notes": "C+Rust+Zig, RadixAttention cache",
    },
    "oracle_radix": {
        "name":  "Oracle+RadixCache",
        "emoji": "🔮✨",
        "url":   f"http://localhost:{os.getenv('ORACLE_RADIX_PORT', '8005')}",
        "color": "\033[35m",      # dark magenta
        "notes": "Oracle with RadixAttention prefix cache enabled",
    },
    "vllm": {
        "name":  "vLLM",
        "emoji": "⚡",
        "url":   f"http://localhost:{os.getenv('VLLM_PORT', '8001')}",
        "color": "\033[94m",      # blue
        "notes": "PagedAttention, prefix caching",
    },
    "tensorrt": {
        "name":  "TensorRT-LLM",
        "emoji": "🚀",
        "url":   f"http://localhost:{os.getenv('TENSORRT_PORT', '8002')}",
        "color": "\033[93m",      # yellow
        "notes": "Pre-compiled TensorRT engine, inflight batching",
    },
    "sglang": {
        "name":  "SGLang",
        "emoji": "🌊",
        "url":   f"http://localhost:{os.getenv('SGLANG_PORT', '8003')}",
        "color": "\033[96m",      # cyan
        "notes": "RadixAttention, torch.compile, structured generation",
    },
    "llamacpp": {
        "name":  "llama.cpp",
        "emoji": "🦙",
        "url":   f"http://localhost:{os.getenv('LLAMACPP_PORT', '8004')}",
        "color": "\033[92m",      # green
        "notes": "GGUF, CPU+GPU, no batching",
    },
}

RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
RED   = "\033[91m"
GREEN = "\033[92m"


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class SingleResult:
    """Metrics for one completed request."""
    ttft_ms:        float        # time to first token
    total_ms:       float        # wall-clock end-to-end
    output_tokens:  int
    itl_ms:         float        # (total_ms - ttft_ms) / (output_tokens - 1)
    failed:         bool = False
    error:          str  = ""


@dataclass
class BenchResult:
    engine_key:     str
    engine_name:    str
    concurrency:    int
    prompt_tokens:  int
    output_tokens:  int
    samples:        List[SingleResult] = field(default_factory=list)

    # Computed after all samples collected
    ttft_p50:   float = 0.0
    ttft_p95:   float = 0.0
    itl_p50:    float = 0.0
    itl_p95:    float = 0.0
    throughput: float = 0.0    # tok/s aggregate
    success_rate: float = 0.0
    vram_mib:   int   = 0

    def compute(self, wall_time: float):
        good = [s for s in self.samples if not s.failed]
        if not good:
            return
        ttfts = sorted(s.ttft_ms for s in good)
        itls  = sorted(s.itl_ms  for s in good if s.output_tokens > 1)
        total_out = sum(s.output_tokens for s in good)

        self.ttft_p50    = _percentile(ttfts, 50)
        self.ttft_p95    = _percentile(ttfts, 95)
        self.itl_p50     = _percentile(itls, 50) if itls else 0.0
        self.itl_p95     = _percentile(itls, 95) if itls else 0.0
        self.throughput  = total_out / wall_time if wall_time > 0 else 0.0
        self.success_rate = len(good) / len(self.samples) * 100


def _percentile(data: List[float], pct: int) -> float:
    if not data:
        return 0.0
    k = (len(data) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(data) - 1)
    return data[lo] + (data[hi] - data[lo]) * (k - lo)


# ── VRAM sampling ─────────────────────────────────────────────────────────────

def sample_vram() -> int:
    """Return total used VRAM in MiB across all GPUs, or 0 if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=3,
        )
        return sum(int(x.strip()) for x in out.strip().splitlines() if x.strip())
    except Exception:
        return 0


# ── HTTP streaming client ─────────────────────────────────────────────────────

def _build_messages(prompt_tokens: int) -> List[Dict]:
    """Build a chat message with approximately the requested token count."""
    # ~1 token ≈ 4 chars of English prose.
    chars = max(prompt_tokens * 4, 20)
    content = (
        "Explain the following topic in detail. "
        "Provide a thorough, multi-paragraph response covering history, "
        "applications, and future directions. Topic: quantum computing. " * 8
    )[:chars]
    return [{"role": "user", "content": content}]


async def _stream_request(
    session: "aiohttp.ClientSession",
    url: str,
    model: str,
    prompt_tokens: int,
    max_tokens: int,
) -> SingleResult:
    """Send one streaming chat request, measure TTFT and ITL."""
    payload = {
        "model":       model,
        "messages":    _build_messages(prompt_tokens),
        "max_tokens":  max_tokens,
        "stream":      True,
        "temperature": 0.0,    # greedy — deterministic, faster
    }
    t0 = time.perf_counter()
    ttft_ms = 0.0
    token_count = 0
    first = True

    try:
        async with session.post(
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                return SingleResult(0, 0, 0, 0, failed=True,
                                    error=f"HTTP {resp.status}: {body[:120]}")

            async for raw in resp.content:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    token_count += 1
                    if first:
                        ttft_ms = (time.perf_counter() - t0) * 1000
                        first = False

        total_ms = (time.perf_counter() - t0) * 1000
        itl = ((total_ms - ttft_ms) / (token_count - 1)) if token_count > 1 else 0.0
        return SingleResult(ttft_ms, total_ms, token_count, itl)

    except asyncio.TimeoutError:
        return SingleResult(0, 0, 0, 0, failed=True, error="timeout")
    except Exception as exc:
        return SingleResult(0, 0, 0, 0, failed=True, error=str(exc)[:120])


async def _detect_model(url: str) -> Optional[str]:
    """Get first model name from /v1/models."""
    if not _HAS_AIOHTTP:
        return None
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{url}/v1/models", timeout=aiohttp.ClientTimeout(total=5)) as r:
                if r.status == 200:
                    data = await r.json()
                    models = data.get("data", [])
                    if models:
                        return models[0]["id"]
    except Exception:
        pass
    return None


async def _run_concurrency_level(
    engine_key: str,
    url: str,
    model: str,
    concurrency: int,
    num_requests: int,
    prompt_tokens: int,
    output_tokens: int,
) -> Tuple[List[SingleResult], float]:
    """Run `num_requests` requests at `concurrency` concurrency. Returns (results, wall_time)."""
    sem = asyncio.Semaphore(concurrency)
    results: List[SingleResult] = []

    async def one(session):
        async with sem:
            r = await _stream_request(session, url, model, prompt_tokens, output_tokens)
            results.append(r)

    t0 = time.perf_counter()
    connector = aiohttp.TCPConnector(limit=concurrency + 4)
    async with aiohttp.ClientSession(connector=connector) as session:
        await asyncio.gather(*[one(session) for _ in range(num_requests)])
    wall = time.perf_counter() - t0
    return results, wall


# ── Ping check ────────────────────────────────────────────────────────────────

def _ping(url: str, timeout: float = 3.0) -> bool:
    if _HAS_REQUESTS:
        try:
            r = _req.get(f"{url}/v1/models", timeout=timeout)
            return r.status_code == 200
        except Exception:
            return False
    return False


# ── Main benchmark runner ─────────────────────────────────────────────────────

async def run_bench(
    engines: List[str],
    concurrency_levels: List[int],
    num_requests: int,
    prompt_tokens: int,
    output_tokens: int,
    model_override: Optional[str],
) -> Dict[str, Dict[int, BenchResult]]:
    """Returns {engine_key: {concurrency: BenchResult}}."""
    all_results: Dict[str, Dict[int, BenchResult]] = {}

    for key in engines:
        eng = ENGINES[key]
        url = eng["url"]
        col = eng["color"]
        name = eng["name"]

        print(f"\n{BOLD}{col}{eng['emoji']}  {name}{RESET}")

        if not _ping(url):
            print(f"  {RED}✗  Not reachable at {url}  — skipping{RESET}")
            continue

        model = model_override or await _detect_model(url) or "unknown"
        print(f"  {DIM}url={url}  model={model}{RESET}")

        all_results[key] = {}

        for conc in concurrency_levels:
            print(f"  concurrency={conc}  requests={num_requests} ", end="", flush=True)

            vram_before = sample_vram()
            results, wall = await _run_concurrency_level(
                key, url, model, conc, num_requests, prompt_tokens, output_tokens,
            )
            vram_after = sample_vram()

            br = BenchResult(
                engine_key=key,
                engine_name=name,
                concurrency=conc,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                samples=results,
            )
            br.compute(wall)
            br.vram_mib = max(0, vram_after - vram_before)
            all_results[key][conc] = br

            good = sum(1 for r in results if not r.failed)
            print(
                f"→  {GREEN if good == num_requests else RED}"
                f"{good}/{num_requests} ok{RESET}  "
                f"tput={br.throughput:.1f} tok/s  "
                f"TTFT p50={br.ttft_p50:.0f}ms  "
                f"ITL p50={br.itl_p50:.2f}ms/tok"
            )

    return all_results


# ── Report ────────────────────────────────────────────────────────────────────

def _bar(value: float, max_val: float, width: int = 24) -> str:
    if max_val <= 0:
        return "▏" + " " * (width - 1)
    filled = int(round(value / max_val * width))
    filled = max(1, min(filled, width))
    return "█" * filled + "░" * (width - filled)


def print_report(
    all_results: Dict[str, Dict[int, BenchResult]],
    concurrency_levels: List[int],
):
    print(f"\n\n{'═'*90}")
    print(f"{BOLD}  ORACLE INFERENCE BENCHMARK REPORT{RESET}")
    print(f"{'═'*90}")

    for conc in concurrency_levels:
        print(f"\n{BOLD}  Concurrency = {conc}{RESET}")
        print(f"  {'Engine':<18} {'Tput (tok/s)':>13} {'TTFT p50':>10} {'TTFT p95':>10} "
              f"{'ITL p50':>9} {'ITL p95':>9} {'OK%':>5} {'VRAM':>7}")
        print(f"  {'─'*17} {'─'*13} {'─'*10} {'─'*10} {'─'*9} {'─'*9} {'─'*5} {'─'*7}")

        rows = []
        for key, by_conc in all_results.items():
            if conc not in by_conc:
                continue
            br = by_conc[conc]
            rows.append((key, br))

        # Sort by throughput descending
        rows.sort(key=lambda x: x[1].throughput, reverse=True)

        max_tput = max((b.throughput for _, b in rows), default=1)

        for i, (key, br) in enumerate(rows):
            eng   = ENGINES.get(key, {"emoji": "?", "name": key, "color": "", "notes": ""})
            medal = ["🥇", "🥈", "🥉", "  ", "  "][min(i, 4)]
            col   = eng.get("color", "")
            vram  = f"{br.vram_mib} MiB" if br.vram_mib else "  —"
            notes = eng.get("notes", "")
            notes_str = f"  [{notes[:30]}]" if notes else ""
            print(
                f"  {col}{eng['emoji']}{RESET} {col}{eng['name']:<18}{RESET} "
                f"{BOLD}{br.throughput:>11.1f}{RESET}  "
                f"{br.ttft_p50:>9.0f}ms "
                f"{br.ttft_p95:>9.0f}ms "
                f"{br.itl_p50:>8.2f}ms "
                f"{br.itl_p95:>8.2f}ms "
                f"{br.success_rate:>4.0f}% "
                f"{vram:>7}  {medal}{DIM}{notes_str}{RESET}"
            )

        # ASCII throughput chart
        print(f"\n  Throughput chart (concurrency={conc}):")
        for key, br in rows:
            eng = ENGINES[key]
            bar = _bar(br.throughput, max_tput)
            print(f"  {eng['color']}{eng['name']:<16}{RESET} {bar}  {br.throughput:.1f} tok/s")

    # Winner summary
    print(f"\n{'─'*90}")
    print(f"{BOLD}  WINNER BY CATEGORY{RESET}")
    print(f"{'─'*90}")

    all_brs: List[Tuple[str, BenchResult]] = []
    for key, by_conc in all_results.items():
        for br in by_conc.values():
            all_brs.append((key, br))

    if all_brs:
        best_tput = max(all_brs, key=lambda x: x[1].throughput)
        best_ttft = min((x for x in all_brs if x[1].ttft_p50 > 0), key=lambda x: x[1].ttft_p50, default=None)
        best_itl  = min((x for x in all_brs if x[1].itl_p50 > 0), key=lambda x: x[1].itl_p50,  default=None)

        eng_t = ENGINES[best_tput[0]]
        print(f"  🏆 Throughput:    {eng_t['color']}{eng_t['emoji']} {eng_t['name']}{RESET}  "
              f"{BOLD}{best_tput[1].throughput:.1f} tok/s{RESET}")

        if best_ttft:
            eng_f = ENGINES[best_ttft[0]]
            print(f"  ⚡ Lowest TTFT:   {eng_f['color']}{eng_f['emoji']} {eng_f['name']}{RESET}  "
                  f"{BOLD}{best_ttft[1].ttft_p50:.0f} ms p50{RESET}")

        if best_itl:
            eng_i = ENGINES[best_itl[0]]
            print(f"  🎯 Lowest ITL:    {eng_i['color']}{eng_i['emoji']} {eng_i['name']}{RESET}  "
                  f"{BOLD}{best_itl[1].itl_p50:.2f} ms/tok p50{RESET}")

    print(f"{'═'*90}\n")


def save_json(all_results: Dict, path: str):
    """Dump all results to JSON for further analysis."""
    out: Dict = {}
    for key, by_conc in all_results.items():
        out[key] = {}
        for conc, br in by_conc.items():
            out[key][str(conc)] = {
                "engine":         br.engine_name,
                "concurrency":    br.concurrency,
                "prompt_tokens":  br.prompt_tokens,
                "output_tokens":  br.output_tokens,
                "throughput":     round(br.throughput, 2),
                "ttft_p50_ms":    round(br.ttft_p50, 2),
                "ttft_p95_ms":    round(br.ttft_p95, 2),
                "itl_p50_ms":     round(br.itl_p50, 3),
                "itl_p95_ms":     round(br.itl_p95, 3),
                "success_rate":   round(br.success_rate, 1),
                "vram_mib":       br.vram_mib,
                "n_samples":      len(br.samples),
                "n_failed":       sum(1 for s in br.samples if s.failed),
            }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Results saved → {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Benchmark OracleInference vs vLLM / TensorRT-LLM / SGLang / llama.cpp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--engines", nargs="+",
        default=list(ENGINES.keys()),
        choices=list(ENGINES.keys()),
        help="Engines to benchmark (default: all)",
    )
    p.add_argument(
        "--concurrency", nargs="+", type=int,
        default=[1, 4, 16],
        dest="concurrency",
        metavar="N",
        help="Concurrency levels to test (default: 1 4 16)",
    )
    p.add_argument("--requests",    type=int, default=50,
                   help="Number of requests per concurrency level (default: 50)")
    p.add_argument("--prompt-len",  type=int, default=256,
                   help="Approximate prompt token count (default: 256)")
    p.add_argument("--output-len",  type=int, default=128,
                   help="Max output tokens per request (default: 128)")
    p.add_argument("--model",       type=str, default=None,
                   help="Force a specific model name for all engines")
    p.add_argument("--out",         type=str, default="bench_results.json",
                   help="JSON output path (default: bench_results.json)")
    p.add_argument("--quick",       action="store_true",
                   help="Quick mode: 10 requests, concurrency 1 and 4 only")

    args = p.parse_args()

    if not _HAS_AIOHTTP:
        print(f"{RED}Error: aiohttp not installed.{RESET}")
        print("  pip install aiohttp")
        sys.exit(1)

    if args.quick:
        args.requests    = 10
        args.concurrency = [1, 4]

    print(f"\n{BOLD}{'═'*60}")
    print("  OracleInference Benchmark Suite")
    print(f"{'═'*60}{RESET}")
    print(f"  Engines:     {', '.join(args.engines)}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Requests:    {args.requests} per level")
    print(f"  Prompt len:  ~{args.prompt_len} tokens")
    print(f"  Output len:  {args.output_len} tokens max")
    print()

    all_results = asyncio.run(run_bench(
        engines           = args.engines,
        concurrency_levels = args.concurrency,
        num_requests      = args.requests,
        prompt_tokens     = args.prompt_len,
        output_tokens     = args.output_len,
        model_override    = args.model,
    ))

    if not all_results:
        print(f"\n{RED}No engines responded. Make sure at least one is running.{RESET}")
        sys.exit(1)

    print_report(all_results, args.concurrency)
    save_json(all_results, args.out)


if __name__ == "__main__":
    main()
