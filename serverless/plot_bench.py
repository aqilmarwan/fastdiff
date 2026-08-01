#!/usr/bin/env python3
"""Plot the critical fastdiff benchmark metrics - a clean, minimal platform comparison.

Reads the JSON files written by `bench_compare.py --out` and renders a branded 2x2
figure: GPU latency percentiles, latency breakdown, throughput, and cost. Colors
follow each platform's brand (Modal green / RunPod purple - CVD-validated), logos
anchor identity, every bar is directly labeled, and deltas call out the winner.

    python serverless/plot_bench.py --in modal.json runpod.json --out serverless/bench.png
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

# Brand palette (CVD-validated on the light surface: ΔE 30.6 deutan, contrast pass).
BRAND = {"modal": "#16a34a", "runpod": "#7c3aed"}
FALLBACK = ["#0891b2", "#dc2626", "#ca8a04", "#4f46e5"]
INK = "#111827"
MUTED = "#6b7280"
FAINT = "#9ca3af"
GRID = "#edeff2"

# Logos (RunPod's is icon+wordmark; crop to the cube for a consistent chip).
LOGOS = {"modal": "public/modal-logo-icon.png", "runpod": "public/runpod-logo.png"}
LOGO_CROP = {"runpod": 1280}  # keep the first N px columns (the cube)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.dpi": 150,
})


def brand(label: str, i: int) -> str:
    return BRAND.get(label.lower(), FALLBACK[i % len(FALLBACK)])


def load_logo(label: str):
    path = LOGOS.get(label.lower())
    if not path:
        return None
    try:
        img = mpimg.imread(path)
    except FileNotFoundError:
        return None
    crop = LOGO_CROP.get(label.lower())
    return img[:, :crop] if crop else img


def place_logo(fig, img, x, y, target_in):
    """Place an image centered at figure-fraction (x, y), target_in inches tall."""
    if img is None:
        return
    zoom = (target_in * fig.dpi) / img.shape[0]
    fig.add_artist(AnnotationBbox(
        OffsetImage(img, zoom=zoom), (x, y), xycoords="figure fraction",
        frameon=False, box_alignment=(0.5, 0.5),
    ))


def label_bars(ax, bars, fmt="{:.2f}"):
    for b in bars:
        h = b.get_height()
        if h == h:
            ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                        fontsize=10, color=INK, fontweight="bold")


def style(ax, title, ylabel):
    ax.set_title(title, fontsize=12, color=INK, fontweight="bold", pad=14, loc="left")
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=1.2)
    ax.xaxis.grid(False)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def xtick_brand(ax, labels, colors, n):
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([lb.capitalize() for lb in labels], fontsize=11.5, fontweight="bold")
    for tick, c in zip(ax.get_xticklabels(), colors):
        tick.set_color(c)
    ax.set_xlim(-0.7, n - 0.3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inputs", nargs="+", required=True)
    ap.add_argument("--out", default="serverless/bench.png")
    args = ap.parse_args()

    runs = [json.load(open(p)) for p in args.inputs]
    labels = [r["label"] for r in runs]
    colors = [brand(r["label"], i) for i, r in enumerate(runs)]
    n = len(runs)

    variant, steps, samples = runs[0].get("variant", "?"), runs[0].get("steps", "?"), runs[0].get("n", "?")

    fig = plt.figure(figsize=(14, 9.6))
    fig.patch.set_facecolor("white")

    # ---- header -------------------------------------------------------------
    fig.text(0.062, 0.955, "fastdiff", fontsize=26, fontweight="bold", color=INK, va="center")
    fig.text(0.188, 0.958, "serverless GPU\nbenchmark", fontsize=10.5, color=MUTED, va="center", linespacing=1.15)
    fig.text(0.062, 0.905,
             f"NVIDIA L40S   ·   {variant}   ·   {steps} steps   ·   1024x1024   ·   N={samples}   ·   warm   ·   concurrency 1",
             fontsize=10.5, color=FAINT, va="center")

    # ---- logo legend (top-right) --------------------------------------------
    lx = 0.70 if n <= 2 else 0.55
    for i, r in enumerate(runs):
        place_logo(fig, load_logo(r["label"]), lx, 0.945, target_in=0.30)
        fig.text(lx + 0.028, 0.945, r["label"].capitalize(), fontsize=13.5, fontweight="bold",
                 color=colors[i], va="center", ha="left")
        lx += 0.145

    gs = fig.add_gridspec(2, 2, left=0.062, right=0.965, top=0.845, bottom=0.085, hspace=0.46, wspace=0.19)

    def delta(ax, vals, lower_better, verb):
        if n != 2 or any(v != v for v in vals):
            return
        win = min(range(2), key=lambda k: vals[k]) if lower_better else max(range(2), key=lambda k: vals[k])
        hi, lo = max(vals), min(vals)
        pct = (hi - lo) / hi * 100 if lower_better else (hi - lo) / lo * 100
        ax.text(1.0, 1.035, f"{labels[win].capitalize()} {pct:.0f}% {verb}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=10, color=colors[win], fontweight="bold")

    # -- Panel 1: GPU latency percentiles (server total) ----------------------
    ax = fig.add_subplot(gs[0, 0])
    pcts = ["p50", "p95", "p99"]
    w = 0.72 / n
    for i, r in enumerate(runs):
        s = r["summary"].get("total_ms", {})
        vals = [s.get(p, float("nan")) / 1000 for p in pcts]
        bars = ax.bar([j + i * w - 0.36 + w / 2 for j in range(len(pcts))], vals, w * 0.88,
                      color=colors[i], edgecolor="white", linewidth=1.5, zorder=3)
        label_bars(ax, bars, "{:.2f}")
    ax.set_xticks(range(len(pcts)))
    ax.set_xticklabels([p.upper() for p in pcts], fontsize=10.5, color=MUTED)
    style(ax, "GPU latency  ·  server-measured", "seconds / image")
    delta(ax, [r["summary"].get("total_ms", {}).get("p50", float("nan")) / 1000 for r in runs], True, "faster")

    # -- Panel 2: latency breakdown (denoise + vae, p50) ----------------------
    ax = fig.add_subplot(gs[0, 1])
    xs = list(range(n))
    den = [r["summary"].get("denoise_ms", {}).get("p50", float("nan")) / 1000 for r in runs]
    vae = [r["summary"].get("vae_ms", {}).get("p50", float("nan")) / 1000 for r in runs]
    ax.bar(xs, den, 0.46, color=colors, edgecolor="white", linewidth=1.5, zorder=3)
    ax.bar(xs, vae, 0.46, bottom=den, color=colors, alpha=0.4, edgecolor="white", linewidth=1.5, zorder=3)
    for i in xs:
        d, v = den[i], vae[i]
        if d == d:
            ax.text(i, d / 2, f"denoise\n{d:.2f}s", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        if d == d and v == v:
            ax.annotate(f"+ vae {v:.2f}s", (i, d + v), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, color=MUTED)
            ax.annotate(f"{d + v:.2f}s", (i, d + v), xytext=(0, 16), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, color=INK, fontweight="bold")
    xtick_brand(ax, labels, colors, n)
    style(ax, "Where the time goes  ·  p50", "seconds / image")
    ax.margins(y=0.28)

    # -- Panel 3: throughput --------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    tput = [r["summary"].get("throughput", {}).get("mean", float("nan")) for r in runs]
    label_bars(ax, ax.bar(xs, tput, 0.46, color=colors, edgecolor="white", linewidth=1.5, zorder=3), "{:.1f}")
    xtick_brand(ax, labels, colors, n)
    style(ax, "Throughput  ·  mean", "denoise steps / sec")
    ax.margins(y=0.20)
    delta(ax, tput, False, "higher")

    # -- Panel 4: cost --------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    cost = [r["cost"]["per_1000_p50"] for r in runs]
    label_bars(ax, ax.bar(xs, cost, 0.46, color=colors, edgecolor="white", linewidth=1.5, zorder=3), "${:.2f}")
    for i, r in enumerate(runs):
        ax.text(i, -max(cost) * 0.14, f"keep-warm  ${r['cost']['warm_per_day']:.0f}/day\n@ ${r['rate_hr']:.2f}/hr",
                ha="center", va="top", fontsize=8.5, color=MUTED)
    xtick_brand(ax, labels, colors, n)
    style(ax, "Cost per 1000 images  ·  p50", "USD / 1000 images")
    ax.margins(y=0.22)
    delta(ax, cost, True, "cheaper")

    fig.savefig(args.out, dpi=150, facecolor="white", bbox_inches="tight", pad_inches=0.35)
    print(f"wrote {args.out}  ({', '.join(labels)})")


if __name__ == "__main__":
    main()
