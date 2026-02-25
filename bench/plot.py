#!/usr/bin/env python3
"""
plot.py — Generate comparison charts from bench_results.json.

Produces:
  - throughput_vs_concurrency.png
  - ttft_vs_concurrency.png
  - itl_vs_concurrency.png

Usage:
  python plot.py                          # reads bench_results.json
  python plot.py --input my_results.json
  python plot.py --out-dir ./charts/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")   # headless
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# Match the engine colours from bench.py (mpl hex equivalents)
ENGINE_STYLE = {
    "oracle":   {"label": "🔮 OracleInference", "color": "#c77dff", "marker": "o", "lw": 2.5, "zorder": 5},
    "vllm":     {"label": "⚡ vLLM",             "color": "#4895ef", "marker": "s", "lw": 2.0, "zorder": 4},
    "tensorrt": {"label": "🚀 TensorRT-LLM",     "color": "#ffd166", "marker": "^", "lw": 2.0, "zorder": 3},
    "sglang":   {"label": "🌊 SGLang",           "color": "#00d4ff", "marker": "D", "lw": 2.0, "zorder": 3},
    "llamacpp": {"label": "🦙 llama.cpp",        "color": "#06d6a0", "marker": "v", "lw": 2.0, "zorder": 2},
}

BG      = "#0a0a14"
PANEL   = "#0f1020"
GRID    = "#1a1a30"
TEXT    = "#ccd0e0"


def _style_ax(ax, title: str, xlabel: str, ylabel: str):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=TEXT, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=TEXT, fontsize=10)
    ax.set_ylabel(ylabel, color=TEXT, fontsize=10)
    ax.tick_params(colors=TEXT)
    ax.spines[:].set_color(GRID)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(which="major", color=GRID, linestyle="--", linewidth=0.6)
    ax.grid(which="minor", color=GRID, linestyle=":", linewidth=0.3)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(TEXT)


def plot(data: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all concurrency levels across all engines
    all_concs: set = set()
    for by_conc in data.values():
        all_concs.update(int(c) for c in by_conc)
    concs = sorted(all_concs)

    metrics = [
        ("throughput",  "Throughput (tokens / sec)",   "throughput_vs_concurrency.png",  True),
        ("ttft_p50_ms", "TTFT p50 (ms)",                "ttft_vs_concurrency.png",        False),
        ("itl_p50_ms",  "ITL p50 (ms / token)",         "itl_vs_concurrency.png",         False),
    ]

    for metric_key, ylabel, filename, higher_better in metrics:
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor(BG)

        plotted = 0
        for eng_key, by_conc in data.items():
            style = ENGINE_STYLE.get(eng_key, {
                "label": eng_key, "color": "#888", "marker": "x", "lw": 1.5, "zorder": 1
            })
            xs, ys = [], []
            for c in concs:
                row = by_conc.get(str(c))
                if row and row.get(metric_key, 0) > 0:
                    xs.append(c)
                    ys.append(row[metric_key])
            if not xs:
                continue
            ax.plot(xs, ys,
                    label    = style["label"],
                    color    = style["color"],
                    marker   = style["marker"],
                    linewidth = style["lw"],
                    markersize = 7,
                    zorder   = style["zorder"])
            # Annotate last point
            ax.annotate(
                f"{ys[-1]:.1f}",
                xy=(xs[-1], ys[-1]),
                xytext=(6, 0), textcoords="offset points",
                color=style["color"], fontsize=8, va="center",
            )
            plotted += 1

        if plotted == 0:
            plt.close(fig)
            continue

        direction = "↑ higher = better" if higher_better else "↓ lower = better"
        _style_ax(ax, f"{ylabel}  —  vs Concurrency  ({direction})",
                  "Concurrency (requests)", ylabel)
        ax.set_xticks(concs)

        legend = ax.legend(
            facecolor=PANEL, edgecolor=GRID,
            labelcolor=TEXT, fontsize=9,
            loc="best",
        )

        plt.tight_layout()
        out_path = out_dir / filename
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        print(f"  Saved: {out_path}")


def main():
    p = argparse.ArgumentParser(description="Plot bench_results.json")
    p.add_argument("--input",   default="bench_results.json")
    p.add_argument("--out-dir", default="charts/")
    args = p.parse_args()

    if not _HAS_MPL:
        print("matplotlib not installed — pip install matplotlib")
        sys.exit(1)

    inp = Path(args.input)
    if not inp.exists():
        print(f"Result file not found: {inp}")
        print("Run bench.py first.")
        sys.exit(1)

    with open(inp) as f:
        data = json.load(f)

    print(f"Plotting results from {inp}…")
    plot(data, Path(args.out_dir))
    print("Done.")


if __name__ == "__main__":
    main()
