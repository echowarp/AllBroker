#!/usr/bin/env python3
"""N-gram (PLE) tables served from the SSD by the broker: one graphic for docs/ngram-tables.md.
Top: how a token's rows are found (hashing, bundles and 16 KiB bins, RAM row cache, SSD, rows to the engine).
Bottom: the measured results. Design: crates/ngram-store/DESIGN.md.
Numbers: Qwen3.8-Flash-Next runs, 2026-10-24 to 10-28 (see docs/ngram-tables.md for each measurement's setup).
  python3 ngram_tables.py [out_dir]   ->   ngram-ssd-tables.png / .svg"""
import os
import sys

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import style as S
from style import plt

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "img")

# disk reads per token, decode, cold cache: single-row reads -> grouped bins (20M bigram + 30M trigram bundles)
TEXTS = ["Code", "Article", "Fiction", "Summary"]
ROWS = [6.49, 10.87, 6.74, 10.55]
BINS = [2.59, 3.54, 2.48, 3.65]
# table formats on the broker (8 GiB RAM cache, new text after warm-up): reads per token, host time per token
FORMATS = [("bf16, single rows", 8.0, None, False), ("IQ4_NL + bins", 2.9, 35, False),
           ("Lossless bf16 + bins", 2.53, 25, True)]
# disk reads per round of ~1 M row lookups (A/B on the live engine)
ROUND = [("Engine's own reader", 555_000, False), ("Broker, cold", 268_000, True), ("Broker, warm", 19_000, True)]
# 990 PRO, random aligned reads, QD1 p50 (ngram-bench sizes)
READ_KIB = [4, 8, 16, 32, 64]
READ_US = [68, 71, 78, 90, 107]

W, H = 13.0, 24.55
fig = plt.figure(figsize=(W, H))
S.background(fig)
ax = fig.add_axes([0, 0, 1, 1], zorder=2); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
ax.set_facecolor("none")

TXT, MUT = S.INK, S.MUTED
BI, TRI = S.OURS_2, S.OURS           # bigram / trigram colours
EDGE = "#6E68A0"


def t(x, y, s, size=11, color=TXT, weight="normal", fam=S.BODY, ha="left", va="baseline", **kw):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, fontfamily=[fam, "DejaVu Sans"], ha=ha, va=va, **kw)


def box(x, y, w, h, fc, ec="none", lw=1.0, r=0.06, alpha=1.0, ls="-", z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                alpha=alpha, ls=ls, zorder=z))


def arrow(x0, y0, x1, y1, color=MUT, lw=1.8, rad=0.0, ms=14, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, ls=ls, zorder=3))


def tile(y0, y1):
    box(0.4, y0, W - 0.8, y1 - y0, S.HERO_TILE, r=0.14, alpha=S.HERO_TILE_ALPHA, z=1)


def stage(ytop, n, title, sub):
    t(0.62, ytop - 0.48, f"{n}", 30, weight="bold", fam=S.DISPLAY, color=S.OURS)
    t(1.08, ytop - 0.42, title, 17, weight="bold", fam=S.DISPLAY)
    t(1.08, ytop - 0.72, sub, 11, MUT, va="top", linespacing=1.2)


def stat(x, y, v, d):
    t(x, y, v, 27, weight="bold", fam=S.DISPLAY, color=S.OURS)
    t(x, y - 0.32, d, 11, MUT, va="baseline")


def bundle(x, y, w, h, label, tri, dashed=False, size=11.5):
    fc = TRI if tri else BI
    if dashed:
        box(x, y, w, h, "none", ec=fc, lw=1.6, r=0.04, ls=(0, (3, 2)))
        t(x + w / 2, y + h * 0.5, label, size, fc, weight="bold", fam=S.DISPLAY, ha="center", va="center")
        return
    box(x, y, w, h, fc, r=0.04)
    t(x + w / 2, y + h * 0.7, label, size, S.ON_LIGHT, weight="bold", fam=S.DISPLAY, ha="center", va="center")
    for k in range(8):   # the 8 rows (one per head)
        xx = x + 0.07 + k * (w - 0.14) / 8
        box(xx + 0.01, y + 0.07, (w - 0.14) / 8 - 0.025, h * 0.34, S.ON_LIGHT, r=0.01, alpha=0.28)


# ---------------------------------------------------------------- title
t(0.55, H - 0.72, "N-gram tables from the SSD, not VRAM", 34, weight="bold", fam=S.DISPLAY)
t(0.55, H - 1.12, "Qwen3.8-Flash-Next's per-layer n-gram embeddings are read through AllBroker: clustered SSD reads, "
                  "a shared RAM row cache, exact bf16 rows", 13, MUT)
t(0.55, H - 1.68, "How a token's rows are found", 21, weight="bold", fam=S.DISPLAY, color=S.OURS_2)

X0 = 3.3   # content column

# ---------------------------------------------------------------- 1: token -> keys
y_top = H - 1.95; y_bot = y_top - 1.55
tile(y_bot, y_top)
stage(y_top, 1, "Token", "a new token and its\n1–2 predecessors")
toks = ["…", "to", "be", "or", "not", "to", "be"]
tw, tg, ty = 0.66, 0.1, y_top - 0.78
for i, s in enumerate(toks):
    x = X0 + i * (tw + tg)
    last = i == len(toks) - 1
    box(x, ty, tw, 0.48, S.OURS if last else S.BADGE, r=0.06)
    t(x + tw / 2, ty + 0.24, s, 14, S.ON_LIGHT if last else TXT, weight="bold", fam=S.DISPLAY, ha="center",
      va="center")
xs_tok = [X0 + i * (tw + tg) for i in range(len(toks))]
t(xs_tok[-1] + tw / 2, ty + 0.62, "new token", 10.5, MUT, ha="center")
# brackets: trigram over the last 3, bigram over the last 2
for k, (n, col, lab, dy) in enumerate([(3, TRI, "trigram  (not, to, be)", -0.2), (2, BI, "bigram  (to, be)", -0.42)]):
    xa, xb = xs_tok[-n] + 0.04, xs_tok[-1] + tw - 0.04
    yy = ty + dy
    ax.plot([xa, xa, xb, xb], [yy + 0.07, yy, yy, yy + 0.07], color=col, lw=1.8, zorder=3)
    t(xa - 0.12, yy - 0.04, lab, 11.5, col, weight="bold", fam=S.DISPLAY, ha="right", va="center")
t(8.95, y_top - 0.5, "Per layer the engine needs 16 rows:", 12.5, TXT, weight="bold")
t(8.95, y_top - 0.8, "8 heads × (bigram, trigram).", 12.5, TXT)
t(8.95, y_top - 1.12, "The trigram is the longer, more", 11, MUT)
t(8.95, y_top - 1.38, "specific context.", 11, MUT)

# ---------------------------------------------------------------- 2: hash
y_top2 = y_bot - 0.6; y_bot2 = y_top2 - 1.6
arrow(X0 + 2.6, y_bot - 0.04, X0 + 2.6, y_top2 + 0.04, color=S.OURS)
tile(y_bot2, y_top2)
stage(y_top2, 2, "Hash", "in the broker, bit-identical\nto the engine's hash")
cw, cg = 0.24, 0.05
for k, (lab, col, yy) in enumerate([("hash(to, be)", BI, y_top2 - 0.62), ("hash(not, to, be)", TRI, y_top2 - 1.12)]):
    t(X0, yy + 0.13, lab, 12.5, col, weight="bold", fam=S.DISPLAY, va="center")
    arrow(X0 + 1.4, yy + 0.13, X0 + 1.8, yy + 0.13, color=col, lw=1.5, ms=11)
    for h in range(8):
        box(X0 + 1.9 + h * (cw + cg), yy, cw, 0.27, col, r=0.03)
    t(X0 + 1.9 + 8 * (cw + cg) + 0.08, yy + 0.13, "8 row ids", 11, MUT, va="center")
t(X0 + 1.9, y_top2 - 1.42, "one per head", 10.5, MUT)
t(8.95, y_top2 - 0.5, "Table: 320 M rows × 160, ~100 GB", 12.5, TXT, weight="bold")
t(8.95, y_top2 - 0.8, "Far too big for VRAM: it never", 11, MUT)
t(8.95, y_top2 - 1.06, "goes there. The broker hashes the", 11, MUT)
t(8.95, y_top2 - 1.32, "token runs itself, so it knows them.", 11, MUT)

# ---------------------------------------------------------------- 3: RAM row cache
y_top3 = y_bot2 - 0.6; y_bot3 = y_top3 - 1.75
arrow(X0 + 2.6, y_bot2 - 0.04, X0 + 2.6, y_top3 + 0.04, color=S.OURS)
tile(y_bot3, y_top3)
stage(y_top3, 3, "RAM row cache", "host memory, checked first")
stat(0.62, y_bot3 + 0.5, "~70%", "cache hits")
cx, cy, cwid, ch = X0, y_bot3 + 0.28, 5.3, 1.05
box(cx, cy, cwid, ch, "none", ec=EDGE, lw=1.3, r=0.08)
rng = np.random.default_rng(3)
ncol, nrow = 26, 3
for i in range(ncol):
    for j in range(nrow):
        r = rng.random()
        fc = BI if r < 0.35 else TRI if r < 0.72 else "#4A4475"
        box(cx + 0.14 + i * 0.193, cy + 0.14 + j * 0.27, 0.15, 0.2, fc, r=0.02, alpha=0.9 if r < 0.72 else 1)
t(8.95, y_top3 - 0.45, "Shared by every engine on the host", 12.5, TXT, weight="bold")
t(8.95, y_top3 - 0.74, "CLOCK eviction · 8 GiB (the knee: working", 11, MUT)
t(8.95, y_top3 - 1.0, "set ~7 GiB) · gives way to KV first", 11, MUT)
HIT_X = 12.35   # hit path: straight down the right edge to step 5

# ---------------------------------------------------------------- 4: SSD bins
y_top4 = y_bot3 - 0.6; y_bot4 = y_top4 - 5.5
arrow(X0 + 1.0, y_bot3 - 0.04, X0 + 1.0, y_top4 + 0.04, color=S.OURS)
t(X0 + 1.12, (y_bot3 + y_top4) / 2 - 0.06, "miss", 11.5, S.OURS_2, weight="bold")
tile(y_bot4, y_top4)
stage(y_top4, 4, "SSD bin", "on a miss: one O_DIRECT\nio_uring read of the\nwhole bin")
stat(0.62, y_bot4 + 2.05, "2.53", "disk reads per token")
stat(0.62, y_bot4 + 0.9, "103 GB", "of bins on the SSD")

t(X0, y_top4 - 0.42, "Directory in RAM: n-gram key → its best bin", 11, MUT)
# token sequence above the bin: each token's bigram + trigram bundle
seq = [("or", "be or", "to be or"), ("not", "or not", "be or not"), ("to", "not to", "or not to"),
       ("be", "to be", "not to be")]
bw, bh = 0.66, 0.78
bx0, by = X0, y_top4 - 2.0
for i, (tok, b2, b3) in enumerate(seq):
    x = bx0 + 2 * i * bw
    box(x + 0.06, by + bh + 0.32, 2 * bw - 0.12, 0.36, S.BADGE, r=0.05)
    t(x + bw, by + bh + 0.5, tok, 13, weight="bold", fam=S.DISPLAY, ha="center", va="center")
    bundle(x + 0.03, by, bw - 0.06, bh, b2, False, size=10.5)
    bundle(x + bw + 0.03, by, bw - 0.06, bh, b3, True, size=10.5)
bin_w = 8 * bw
box(bx0 - 0.08, by - 0.1, bin_w + 0.16, bh + 0.2, "none", ec=TXT, lw=1.6, r=0.06)
t(bx0 - 0.08, by - 0.38, "One 16 KiB bin = 8 bundles", 12.5, TXT, weight="bold", va="center")
for k, line in enumerate(["A bundle is one n-gram's 8 rows: one read, not eight.",
                          "Each bigram sits next to its frequent trigrams; the bin follows a",
                          "frequent token sequence from a ~1.5 B-token corpus profile, so one",
                          "read brings the next tokens' rows. All its bundles enter the cache."]):
    t(bx0 - 0.08, by - 0.68 - k * 0.26, line, 11, MUT, va="center")
# all bundles -> cache
ux = bx0 + bin_w - 0.3
arrow(ux, by + bh + 0.75, ux, y_bot3 + 0.3, color=S.OURS_2, lw=1.6, ms=12)
t(ux - 0.12, (y_bot3 + y_top4) / 2 - 0.06, "whole bin cached", 11, S.OURS_2, ha="right", weight="bold")

# rare n-grams, duplication
sw4 = 0.72
ry = y_bot4 + 0.95
box(X0 - 0.08, ry - 0.08, 2 * sw4 + 0.16, 0.66, "none", ec=MUT, lw=1.3, r=0.05)
bundle(X0, ry, sw4 - 0.06, 0.5, "or perish", False, size=10)
bundle(X0 + sw4, ry, sw4 - 0.06, 0.5, "be or perish", True, size=9)
t(X0 + 2 * sw4 + 0.25, ry + 0.38, "4 KiB bins: rare n-grams, 2 bundles,", 11, TXT, weight="bold")
t(X0 + 2 * sw4 + 0.25, ry + 0.12, "grouped by bigram · no bundle at all: single-row reads", 11, MUT)
ry2 = y_bot4 + 0.25
bundle(X0 + 0.37, ry2, sw4 - 0.06, 0.5, "to be", False, dashed=True, size=10.5)
t(X0 + 2 * sw4 + 0.25, ry2 + 0.38, "Hot bundles may be copied into several bins", 11, TXT, weight="bold")
t(X0 + 2 * sw4 + 0.25, ry2 + 0.12, "(disk is cheap)", 11, MUT)

# inset: read size vs latency
ix, iw = 9.2, 3.05
t(ix, y_top4 - 0.42, "Why 16 KiB: a bigger read", 12.5, TXT, weight="bold")
t(ix, y_top4 - 0.68, "is nearly free", 12.5, TXT, weight="bold")
t(ix, y_top4 - 0.95, "990 PRO, random aligned read, QD1 p50", 10, MUT)
t(ix, y_top4 - 1.33, "4× the data, +15% time", 14, S.OURS_2, weight="bold", fam=S.DISPLAY)
t(ix, y_top4 - 1.6, "the drive reads its 16 KiB NAND flash", 10.5, MUT)
t(ix, y_top4 - 1.84, "page anyway: bins match it, 16 KiB aligned", 10.5, MUT)
iy = y_bot4 + 0.8
scale = 2.15 / 107
bar_w = iw / len(READ_KIB)
for k, (kib, us) in enumerate(zip(READ_KIB, READ_US)):
    x = ix + k * bar_w + 0.08
    hi = kib == 16
    box(x, iy, bar_w - 0.16, us * scale, S.OURS if hi else S.HERO_OTHER, r=0.03)
    t(x + (bar_w - 0.16) / 2, iy + us * scale + 0.08, f"{us} µs", 12 if hi else 11, TXT if hi else MUT,
      weight="bold", fam=S.DISPLAY, ha="center")
    t(x + (bar_w - 0.16) / 2, iy - 0.24, f"{kib} KiB", 10.5, TXT if hi else MUT, weight="bold" if hi else "normal",
      ha="center")
ax.plot([ix, ix + iw], [iy, iy], color=EDGE, lw=1.0, zorder=3)
t(ix, iy - 0.52, "Measured; the drive reports 512 B sectors", 9.5, MUT)

# ---------------------------------------------------------------- 5: rows to the engine
y_top5 = y_bot4 - 0.6; y_bot5 = y_top5 - 2.05
arrow(X0 + 2.6, y_bot4 - 0.04, X0 + 2.6, y_top5 + 0.04, color=S.OURS)
tile(y_bot5, y_top5)
stage(y_top5, 5, "Rows to the engine", "exact rows, never in VRAM")
stat(0.62, y_bot5 + 0.42, "25 µs", "host time per token")

steps = [("Lossless bf16", "320 → 232 B per row,\nexact (Huffman exponents)"),
         ("fp16 on the host", "decoded in 7.7 µs\nper token"),
         ("Engine", "reads hide behind\nlayer 0's compute")]
sx, swd, sgap, sy = X0, 2.15, 0.45, y_top5 - 1.15
for k, (h, d) in enumerate(steps):
    x = sx + k * (swd + sgap)
    box(x, sy, swd, 0.95, S.BADGE, r=0.08)
    t(x + 0.15, sy + 0.66, h, 14.5, weight="bold", fam=S.DISPLAY)
    t(x + 0.15, sy + 0.42, d, 10.5, MUT, va="top", linespacing=1.15)
    if k < 2:
        arrow(x + swd + 0.05, sy + 0.48, x + swd + sgap - 0.05, sy + 0.48, color=S.OURS)
eng_r = sx + 3 * swd + 2 * sgap
# hit path from the cache, down the right edge into the engine
hy = cy + 0.12
ax.plot([cx + cwid + 0.02, HIT_X, HIT_X], [hy, hy, sy + 0.48], color=S.OURS, lw=1.8, zorder=3)
arrow(HIT_X, sy + 0.48, eng_r + 0.05, sy + 0.48, color=S.OURS, lw=1.8)
ax.text(HIT_X + 0.15, (hy + y_top5) / 2, "hit", fontsize=11.5, color=S.OURS, fontweight="bold", rotation=-90,
        ha="center", va="center", zorder=4)
# VRAM bar
vx, vy, vw = X0, y_bot5 + 0.22, eng_r - X0
box(vx, vy, vw, 0.42, S.OURS_MID, r=0.06)
t(vx + 0.15, vy + 0.21, "VRAM stays KV cache (and weights)", 12.5, TXT, weight="bold", fam=S.BODY, va="center")
t(vx + vw - 0.15, vy + 0.21, "n-gram table: 0 GB", 13, TXT, weight="bold", fam=S.DISPLAY, ha="right", va="center")
stat(eng_r + 0.3, y_bot5 + 0.42, "0 GB", "of VRAM")

# ---------------------------------------------------------------- results (bottom half)
RES_TOP = y_bot5 - 0.55
t(0.55, RES_TOP - 0.15, "Results", 21, weight="bold", fam=S.DISPLAY, color=S.OURS_2)


def R(x, y, w, h):
    """inches -> figure fraction"""
    return [x / W, y / H, w / W, h / H]


def card(rect_in, title, sub):
    x, y, w, h = rect_in
    box(x, y, w, h, S.HERO_TILE, r=0.14, alpha=S.HERO_TILE_ALPHA, z=1)
    t(x + 0.24, y + h - 0.45, title, 19, weight="bold", fam=S.DISPLAY)
    t(x + 0.24, y + h - 0.75, sub, 11, MUT)


def plain_ax(rect_in):
    a = fig.add_axes(R(*rect_in), zorder=3); a.set_facecolor("none"); a.grid(False)
    for sp in a.spines.values():
        sp.set_visible(False)
    a.tick_params(colors=S.MUTED, length=0, pad=6)
    return a


def hbars(a, labels, vals, ours, fmt, top=None, room=1.38):
    n = len(vals); y = np.arange(n)[::-1]; hi = top or max(vals)
    for yy, lab, v, o in zip(y, labels, vals, ours):
        a.barh([yy], [v], height=0.6, color=S.OURS if o else S.HERO_OTHER)
        a.text(v + hi * 0.03, yy, fmt(v), va="center", fontfamily=S.DISPLAY, fontsize=15, fontweight="bold",
               color=S.INK if o else S.MUTED)
    a.set_yticks(y, labels, fontsize=12); a.set_xticks([]); a.set_xlim(0, hi * room); a.set_ylim(-0.6, n - 0.4)


CB, CT = 1.25, RES_TOP - 0.45   # card bottom / top (inches)
LW = 5.85
# 1: grouped bins vs single rows
card((0.4, CB, LW, CT - CB), "A third of the disk reads", "Disk reads per token, decode, cold cache")
a = plain_ax((0.95, CB + 0.55, LW - 0.9, CT - CB - 1.9))
x = np.arange(len(TEXTS)); bw2 = 0.38
b1 = a.bar(x - bw2 / 2, ROWS, bw2, color=S.HERO_OTHER, label="Single-row reads")
b2 = a.bar(x + bw2 / 2, BINS, bw2, color=S.OURS, label="Grouped bins")
for bars, col in ((b1, S.MUTED), (b2, S.INK)):
    for b in bars:
        a.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 4),
                   textcoords="offset points", ha="center", fontfamily=S.DISPLAY, fontsize=14, fontweight="bold",
                   color=col)
for xi, r, b in zip(x, ROWS, BINS):
    a.annotate(f"−{1 - b / r:.0%}", (xi + bw2 / 2, b), xytext=(0, 24), textcoords="offset points", ha="center",
               fontsize=11, fontweight="bold", color=S.OURS_2)
a.set_xticks(x, TEXTS, fontsize=12.5); a.set_yticks([]); a.set_ylim(0, 13.5)
a.spines["bottom"].set_visible(True); a.spines["bottom"].set_color(S.HERO_OTHER)
a.legend(loc="upper left", bbox_to_anchor=(0.0, 1.06), ncol=2, fontsize=11, handlelength=1.2)

RX, RW = 0.4 + LW + 0.3, W - 0.8 - LW - 0.3
mid = (CB + CT) / 2
# 2: formats
card((RX, mid + 0.15, RW, CT - mid - 0.15), "Exact rows cost only SSD space",
     "Disk reads per token on new text · host time per token")
a = plain_ax((RX + 1.95, mid + 0.4, RW - 2.35, CT - mid - 1.65))
hbars(a, [f[0] for f in FORMATS], [f[1] for f in FORMATS], [f[3] for f in FORMATS],
      lambda v: f"{v:.2f}" if v < 3 else f"~{v:.0f}", room=1.5)
for (name, v, us, o), yy in zip(FORMATS, np.arange(len(FORMATS))[::-1]):
    a.text(8.0 * 1.5, yy, f"{us} µs" if us else "—", va="center", ha="right", fontsize=12,
           color=S.INK if o else S.MUTED)
a.text(8.0 * 1.5, 2.62, "host time", va="center", ha="right", fontsize=10.5, color=S.MUTED)

# 3: vs the engine's own reader
card((RX, CB, RW, mid - 0.15 - CB), "30× fewer reads once warm", "Disk reads per round of ~1 M row lookups")
a = plain_ax((RX + 1.95, CB + 0.3, RW - 2.35, mid - CB - 1.6))
hbars(a, [r[0] for r in ROUND], [r[1] for r in ROUND], [r[2] for r in ROUND], lambda v: f"{v / 1000:.0f}k")

t(0.55, 0.78, "Same decode speed (±0.4%), prefill +1–4% · 0 GB of VRAM: the tables live on the SSD "
              "(lossless bf16 bins: 103 GB) with an 8 GiB RAM row cache that gives way to KV under memory pressure",
  10.5, TXT)
t(0.55, 0.42, S.FOOT, 8.5, MUT)
print(S.save(fig, OUT, "ngram-ssd-tables"))
