#!/usr/bin/env bash
# build.sh — Oracle Inference Engine master build script
#
# Builds all three layers in order:
#   1. Zig utilities       → libzig_utils.a
#   2. C++/CUDA kernels    → libkernels.so
#   3. Rust workspace      → oracle-server (binary)
#
# Usage:
#   ./build.sh                  # release build
#   ./build.sh --debug          # debug build (slower, with symbols)
#   ./build.sh --arch 89        # target sm_89 (Ada Lovelace) only
#   ./build.sh --clean          # clean build
#   ./build.sh --no-cuda        # skip CUDA, build CPU-only fallback
#
# Environment:
#   ORACLE_MODEL_PATH  — path to model weights directory
#   CUDA_VISIBLE_DEVICES — GPU to use (default: 0)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$ROOT/build"
ZIG_OUT="$BUILD_DIR/zig"
CMAKE_OUT="$BUILD_DIR/cmake"
RUST_OUT="$ROOT/target/release"

# ── Argument parsing ──────────────────────────────────────────────────────────
MODE="release"
ARCH=""
CLEAN=0
NO_CUDA=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)   MODE="debug";      shift ;;
        --arch)    ARCH="$2";         shift 2 ;;
        --clean)   CLEAN=1;           shift ;;
        --no-cuda) NO_CUDA=1;         shift ;;
        *)         echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ $CLEAN -eq 1 ]]; then
    echo "==> Cleaning build artefacts..."
    rm -rf "$BUILD_DIR" "$ROOT/target"
fi

mkdir -p "$ZIG_OUT" "$CMAKE_OUT"

# ── Colours ────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; NC='\033[0m'
step() { echo -e "${BLUE}==> $*${NC}"; }
ok()   { echo -e "${GREEN}✓  $*${NC}"; }
warn() { echo -e "${YELLOW}⚠  $*${NC}"; }

# ── 1. Zig utilities ──────────────────────────────────────────────────────────
step "Building Zig utilities..."
cd "$ROOT/zig-utils"
if [[ "$MODE" == "release" ]]; then
    zig build -Doptimize=ReleaseFast --prefix "$ZIG_OUT"
else
    zig build -Doptimize=Debug --prefix "$ZIG_OUT"
fi
ok "Zig utilities built → $ZIG_OUT"

# ── 2. C++/CUDA kernels ───────────────────────────────────────────────────────
step "Building C++/CUDA kernels..."
cd "$CMAKE_OUT"

CMAKE_ARGS="-DCMAKE_BUILD_TYPE=$([ "$MODE" == "release" ] && echo Release || echo Debug)"
if [[ -n "$ARCH" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DORACLE_ARCH=$ARCH"
fi
if [[ $NO_CUDA -eq 1 ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DORACLE_NO_CUDA=1"
fi

cmake "$ROOT" $CMAKE_ARGS -G Ninja
ninja -j"$(nproc)"
ok "Kernels built → $CMAKE_OUT/libkernels.so"

# ── Copy libkernels.so where Rust expects it ──────────────────────────────────
cp "$CMAKE_OUT/libkernels.so" "$ROOT/"

# ── 3. Rust workspace ─────────────────────────────────────────────────────────
step "Building Rust workspace..."
cd "$ROOT"

RUST_FLAGS="-C target-cpu=native"
if [[ "$MODE" == "release" ]]; then
    RUSTFLAGS="$RUST_FLAGS" cargo build --release --workspace
else
    RUSTFLAGS="$RUST_FLAGS" cargo build --workspace
fi

ok "Rust workspace built → $RUST_OUT/oracle-server"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}┌─────────────────────────────────────────────────┐${NC}"
echo -e "${GREEN}│  Oracle Inference Engine — build complete  🚀   │${NC}"
echo -e "${GREEN}├─────────────────────────────────────────────────┤${NC}"
echo -e "${GREEN}│  libkernels.so  →  $ROOT/libkernels.so          │${NC}"
echo -e "${GREEN}│  oracle-server  →  $RUST_OUT/oracle-server      │${NC}"
echo -e "${GREEN}└─────────────────────────────────────────────────┘${NC}"
echo ""
echo "Start server:"
echo "  ./target/release/oracle-server /path/to/model ./libkernels.so 0.0.0.0 8000"
