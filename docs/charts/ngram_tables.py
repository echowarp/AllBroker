#!/usr/bin/env python3
"""N-gram (PLE) tables served from the SSD by the broker: one graphic for docs/ngram-tables.md.
Numbers: Qwen3.8-Flash-Next runs, 2026-10-24 to 10-28 (see docs/ngram-tables.md for each measurement's setup).
  python3 ngram_tables.py [out_dir]   ->   ngram-ssd-tables.png / .svg"""
import os
import sys

import numpy as np

import style as S
from style import plt

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "img")

# disk reads per token, decode, cold cache: single-row reads -> grouped bins (20M bigram + 30M trigram bundles)
TEXTS = ["Code", "Article", "Fiction", "Summary"]
ROWS = [6.49, 10.87, 6.74, 10.55]
BINS = [2.59, 3.54, 2.48, 3.65]
# table formats on the broker (8 GiB RAM cache, new text after warm-up): reads per token, host time per token
FORMATS = [("bf16, single rows", 8.0, None, False), ("IQ4_NL + bins", 2.9, 35, False), ("Lossless bf16 + bins", 2.53, 25, True)]
# disk reads per round of ~1 M row lookups (A/B on the live engine)
ROUND = [("Engine's own reader", 555_000, False), ("Broker, cold", 268_000, True), ("Broker, warm", 19_000, True)]


def card(fig, rect, title, sub):
    x, y, w, h = rect
    bg = fig.add_axes(rect, zorder=1); bg.axis("off"); bg.set_xlim(0, 1); bg.set_ylim(0, 1)
    bg.add_patch(plt.matplotlib.patches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.03",
                                                       transform=bg.transAxes, facecolor=S.HERO_TILE,
                                                       alpha=S.HERO_TILE_ALPHA, edgecolor="none"))
    fig.text(x + 0.018, y + h - 0.045, title, fontfamily=S.DISPLAY, fontsize=19, fontweight="bold", color=S.INK,
             va="baseline", zorder=4)
    fig.text(x + 0.018, y + h - 0.075, sub, fontsize=10.5, color=S.MUTED, va="baseline", zorder=4)


def plain_ax(fig, rect):
    ax = fig.add_axes(rect, zorder=3); ax.set_facecolor("none"); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=S.MUTED, length=0, pad=6)
    return ax


def hbars(ax, labels, vals, ours, fmt, top=None):
    n = len(vals); y = np.arange(n)[::-1]; hi = top or max(vals)
    for yy, lab, v, o in zip(y, labels, vals, ours):
        ax.barh([yy], [v], height=0.6, color=S.OURS if o else S.HERO_OTHER)
        ax.text(v + hi * 0.03, yy, fmt(v), va="center", fontfamily=S.DISPLAY, fontsize=15, fontweight="bold",
                color=S.INK if o else S.MUTED)
    ax.set_yticks(y, labels, fontsize=12); ax.set_xticks([]); ax.set_xlim(0, hi * 1.38); ax.set_ylim(-0.6, n - 0.4)


fig = plt.figure(figsize=(13, 8.2))
S.background(fig)
fig.text(0.04, 0.925, "N-gram tables from the SSD, not VRAM", fontfamily=S.DISPLAY, fontsize=31, fontweight="bold",
         color=S.INK, va="baseline")
fig.text(0.04, 0.88, "Qwen3.8-Flash-Next's per-layer n-gram embeddings are read through AllBroker: grouped SSD reads, "
         "a shared RAM row cache, exact bf16 rows", fontsize=12.5, color=S.MUTED, va="baseline")

# 1: grouped bins vs single rows
card(fig, [0.04, 0.10, 0.45, 0.74], "A third of the disk reads", "Disk reads per token, decode, cold cache")
ax = plain_ax(fig, [0.085, 0.17, 0.39, 0.52])
x = np.arange(len(TEXTS)); bw = 0.38
b1 = ax.bar(x - bw / 2, ROWS, bw, color=S.HERO_OTHER, label="Single-row reads")
b2 = ax.bar(x + bw / 2, BINS, bw, color=S.OURS, label="Grouped bins")
for bars, col in ((b1, S.MUTED), (b2, S.INK)):
    for b in bars:
        ax.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontfamily=S.DISPLAY, fontsize=14, fontweight="bold",
                    color=col)
for xi, r, b in zip(x, ROWS, BINS):
    ax.annotate(f"−{1 - b / r:.0%}", (xi + bw / 2, b), xytext=(0, 24), textcoords="offset points", ha="center",
                fontsize=11, fontweight="bold", color=S.OURS_2)
ax.set_xticks(x, TEXTS, fontsize=12.5); ax.set_yticks([]); ax.set_ylim(0, 13.5)
ax.spines["bottom"].set_visible(True); ax.spines["bottom"].set_color(S.HERO_OTHER)
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), ncol=2, fontsize=11, handlelength=1.2)

# 2: formats
card(fig, [0.51, 0.47, 0.45, 0.37], "Exact rows cost only SSD space", "Disk reads per token on new text · host time per token")
ax = plain_ax(fig, [0.66, 0.505, 0.27, 0.22])
hbars(ax, [f[0] for f in FORMATS], [f[1] for f in FORMATS], [f[3] for f in FORMATS],
      lambda v: f"{v:.2f}" if v < 3 else f"~{v:.0f}")
for (name, v, us, o), yy in zip(FORMATS, np.arange(len(FORMATS))[::-1]):
    if us:
        ax.text(8.0 * 1.38, yy, f"{us} µs", va="center", ha="right", fontsize=12, color=S.INK if o else S.MUTED)
ax.text(8.0 * 1.38, 2.0, "host time", va="center", ha="right", fontsize=10.5, color=S.MUTED)

# 3: vs the engine's own reader
card(fig, [0.51, 0.10, 0.45, 0.34], "30× fewer reads once warm", "Disk reads per round of ~1 M row lookups")
ax = plain_ax(fig, [0.66, 0.13, 0.27, 0.2])
hbars(ax, [r[0] for r in ROUND], [r[1] for r in ROUND], [r[2] for r in ROUND], lambda v: f"{v / 1000:.0f}k")

fig.text(0.04, 0.055, "Same decode speed (±0.4%), prefill +1–4% · 0 GB of VRAM: the tables live on the SSD "
         "(lossless bf16 bins: 103 GB) with an 8 GiB RAM row cache that gives way to KV under memory pressure",
         fontsize=10.5, color=S.INK)
fig.text(0.04, 0.025, S.FOOT, fontsize=8.5, color=S.MUTED)
print(S.save(fig, OUT, "ngram-ssd-tables"))
