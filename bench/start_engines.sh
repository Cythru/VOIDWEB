#!/usr/bin/env bash
# start_engines.sh — Start all inference engines for benchmarking.
#
# Each engine runs on a different port so they can all be up simultaneously:
#   OracleInference  →  :8000
#   vLLM             →  :8001
#   TensorRT-LLM     →  :8002
#   SGLang           →  :8003
#   llama.cpp        →  :8004
#
# Usage:
#   ./start_engines.sh                    # start all
#   ./start_engines.sh oracle vllm        # specific engines only
#   MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ ./start_engines.sh
#   ./start_engines.sh stop               # kill all engine processes
#
# Requires: each engine installed in its own venv or globally.
# Log files: /tmp/bench_<engine>.log

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
GPU_UTIL="${GPU_UTIL:-0.78}"
MAX_LEN="${MAX_LEN:-4096}"
DTYPE="${DTYPE:-bfloat16}"

ORACLE_BIN="${ORACLE_BIN:-$(dirname "$0")/../target/release/oracle-server}"
VLLM_PYTHON="${VLLM_PYTHON:-python3}"
SGLANG_PYTHON="${SGLANG_PYTHON:-python3}"
LLAMACPP_BIN="${LLAMACPP_BIN:-/usr/local/bin/llama-server}"

ENGINES=("$@")
if [ ${#ENGINES[@]} -eq 0 ]; then
    ENGINES=(oracle oracle_radix vllm tensorrt sglang llamacpp)
fi

# ── Stop mode ──────────────────────────────────────────────────────────────
if [[ "${ENGINES[0]:-}" == "stop" ]]; then
    echo "Stopping all benchmark engines..."
    for port in 8000 8001 8002 8003 8004 8005; do
        pid=$(lsof -ti:"$port" 2>/dev/null || true)
        if [ -n "$pid" ]; then
            kill "$pid" 2>/dev/null && echo "  Killed pid $pid (port $port)" || true
        fi
    done
    exit 0
fi

# ── Helpers ────────────────────────────────────────────────────────────────
wait_ready() {
    local url="$1" name="$2" timeout=120 elapsed=0
    echo -n "  Waiting for $name to come up..."
    while ! curl -sf "$url/v1/models" >/dev/null 2>&1; do
        sleep 2; elapsed=$((elapsed+2))
        echo -n "."
        if [ $elapsed -ge $timeout ]; then
            echo " TIMEOUT after ${timeout}s"
            return 1
        fi
    done
    echo " ready (${elapsed}s)"
}

start_bg() {
    local name="$1" log="/tmp/bench_${name}.log"
    shift
    echo "  Starting $name → $log"
    nohup "$@" >"$log" 2>&1 &
    echo $! > "/tmp/bench_${name}.pid"
}

# ── Engine launchers ───────────────────────────────────────────────────────

start_oracle() {
    if ! [ -x "$ORACLE_BIN" ]; then
        echo "  ⚠  OracleInference binary not found at $ORACLE_BIN"
        echo "     Run: cd .. && ./build.sh --release"
        return
    fi
    start_bg oracle "$ORACLE_BIN" \
        --model    "$MODEL" \
        --port     8000 \
        --host     0.0.0.0 \
        --dtype    "$DTYPE" \
        --gpu-util "$GPU_UTIL" \
        --max-len  "$MAX_LEN"
    wait_ready "http://localhost:8000" "OracleInference"
}

start_oracle_radix() {
    # Oracle with RadixAttention prefix cache enabled (--radix-cache flag)
    if ! [ -x "$ORACLE_BIN" ]; then
        echo "  ⚠  OracleInference binary not found at $ORACLE_BIN"
        return
    fi
    start_bg oracle_radix "$ORACLE_BIN" \
        --model       "$MODEL" \
        --port        8005 \
        --host        0.0.0.0 \
        --dtype       "$DTYPE" \
        --gpu-util    "$GPU_UTIL" \
        --max-len     "$MAX_LEN" \
        --radix-cache             # enable RadixAttention prefix cache
    wait_ready "http://localhost:8005" "Oracle+RadixCache"
}

start_vllm() {
    if ! "$VLLM_PYTHON" -c "import vllm" 2>/dev/null; then
        echo "  ⚠  vLLM not installed (pip install vllm)"
        return
    fi
    start_bg vllm "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
        --model                "$MODEL" \
        --port                 8001 \
        --host                 0.0.0.0 \
        --dtype                "$DTYPE" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --max-model-len        "$MAX_LEN" \
        --enable-prefix-caching \
        --disable-log-requests
    wait_ready "http://localhost:8001" "vLLM"
}

start_tensorrt() {
    # TensorRT-LLM uses trtllm-serve (triton-style) — requires pre-built engine.
    # Build engine first:
    #   trtllm-build --checkpoint_dir ./checkpoints --output_dir ./trt_engines \
    #     --gemm_plugin bfloat16 --max_input_len 2048 --max_output_len 512
    TRT_ENGINE="${TRT_ENGINE:-./trt_engines}"
    TRT_TOKENIZER="${TRT_TOKENIZER:-$MODEL}"
    if ! command -v trtllm-serve &>/dev/null; then
        echo "  ⚠  TensorRT-LLM not installed or trtllm-serve not in PATH"
        return
    fi
    if ! [ -d "$TRT_ENGINE" ]; then
        echo "  ⚠  TRT engine dir not found at $TRT_ENGINE — build it first"
        return
    fi
    start_bg tensorrt trtllm-serve \
        "$TRT_ENGINE" \
        --port        8002 \
        --host        0.0.0.0 \
        --tokenizer   "$TRT_TOKENIZER" \
        --max_num_tokens "$MAX_LEN"
    wait_ready "http://localhost:8002" "TensorRT-LLM"
}

start_sglang() {
    if ! "$SGLANG_PYTHON" -c "import sglang" 2>/dev/null; then
        echo "  ⚠  SGLang not installed (pip install sglang)"
        return
    fi
    start_bg sglang "$SGLANG_PYTHON" -m sglang.launch_server \
        --model-path             "$MODEL" \
        --port                   8003 \
        --host                   0.0.0.0 \
        --dtype                  "$DTYPE" \
        --mem-fraction-static    "$GPU_UTIL" \
        --context-length         "$MAX_LEN" \
        --enable-torch-compile \
        --disable-radix-cache  # disable prefix cache for fair comparison; remove to benchmark with cache
    wait_ready "http://localhost:8003" "SGLang"
}

start_llamacpp() {
    # llama.cpp HTTP server — needs GGUF model file.
    GGUF="${GGUF:-}"
    if [ -z "$GGUF" ]; then
        # Try to auto-find a GGUF in ~/.cache/huggingface
        GGUF=$(find "$HOME/.cache/huggingface" -name "*.gguf" 2>/dev/null | head -1 || true)
    fi
    if ! command -v llama-server &>/dev/null && ! [ -x "$LLAMACPP_BIN" ]; then
        echo "  ⚠  llama-server not found (build llama.cpp or set LLAMACPP_BIN)"
        return
    fi
    if [ -z "$GGUF" ]; then
        echo "  ⚠  No GGUF file found — set GGUF=/path/to/model.gguf"
        return
    fi
    BIN="${LLAMACPP_BIN}"
    command -v llama-server &>/dev/null && BIN=llama-server
    start_bg llamacpp "$BIN" \
        --model    "$GGUF" \
        --port     8004 \
        --host     0.0.0.0 \
        --n-gpu-layers 999 \
        --ctx-size "$MAX_LEN" \
        --parallel 4 \
        --threads  8
    wait_ready "http://localhost:8004" "llama.cpp"
}

# ── Run selected engines ───────────────────────────────────────────────────
echo "Starting engines: ${ENGINES[*]}"
echo "Model: $MODEL"
echo ""

for eng in "${ENGINES[@]}"; do
    case "$eng" in
        oracle)        start_oracle        ;;
        oracle_radix)  start_oracle_radix  ;;
        vllm)          start_vllm          ;;
        tensorrt)      start_tensorrt      ;;
        sglang)        start_sglang        ;;
        llamacpp)      start_llamacpp      ;;
        *) echo "Unknown engine: $eng" ;;
    esac
done

echo ""
echo "All engines up. Run:"
echo "  python bench.py --engines ${ENGINES[*]}"
echo ""
echo "Stop all:  ./start_engines.sh stop"
