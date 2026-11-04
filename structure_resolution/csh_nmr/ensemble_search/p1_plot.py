"""Direct against Diff1st on the 30 cells: recovered sites, and the per-cell distance with agreement."""
import glob, os, sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from d1_analyse import load, mic, evaluate, variant, geometry_only

OUT = sys.argv[1]
names = sorted(os.path.basename(f) for f in glob.glob("sweep_direct/cell_*.npz"))
A = [load(f"sweep_direct/{n}") for n in names]
B = [load(f"sweep_deriv1st/{n}") for n in names]
NCS, NCSO, NC = (11, 20, 14), (11, 20, 14, 8), (11, 20)
ROWS = [("Geometry only", geometry_only("soft")), ("Na+Ca", variant("soft", sp=NC)),
        ("Na+Ca+Si", variant("soft", sp=NCS)), ("Na+Ca+Si+O", variant("soft", sp=NCSO))]
DIRECT, DIFF, INK, MUTED, GRID, BG, DIS = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#6250d6"
plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": MUTED, "axes.edgecolor": GRID,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "figure.facecolor": BG, "axes.facecolor": BG})

fig, axs = plt.subplots(1, 3, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.15]}, constrained_layout=True)
y = np.arange(len(ROWS))[::-1]
for ax, th, ttl in ((axs[0], 0.5, "True site within 0.5 Å"), (axs[1], 1.0, "True site within 1 Å")):
    for yy, (_, fn) in zip(y, ROWS):
        na = int(np.sum(evaluate(A, fn)[0] < th)); nb = int(np.sum(evaluate(B, fn)[0] < th))
        ax.plot([na, nb], [yy, yy], color=GRID, lw=2, zorder=1, solid_capstyle="round")
        tie = 0.14 if na == nb else 0                 # keep both dots visible on a tie
        ax.scatter(na, yy + tie, s=70, color=DIRECT, edgecolor=BG, lw=1.5, zorder=3)
        ax.scatter(nb, yy - tie, s=70, color=DIFF, edgecolor=BG, lw=1.5, zorder=3)
        ax.text(31.5, yy, f"{na}, {nb}", va="center", ha="left", color=MUTED, fontsize=9)
    ax.set_xlim(0, 31); ax.set_ylim(-0.6, len(ROWS) - 0.4); ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in ROWS] if ax is axs[0] else [])
    ax.set_xlabel("Cells out of 30"); ax.set_title(ttl); ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
axs[0].scatter([], [], s=70, color=DIRECT, label="Direct"); axs[0].scatter([], [], s=70, color=DIFF, label="Diff1st")
axs[0].legend(loc="lower left", bbox_to_anchor=(0, -0.34), ncol=2, frameon=False, handletextpad=0.2)

ax = axs[2]; fn = variant("soft", sp=NCS)
def best(d):
    p, s = fn(d)
    return p[int(np.argmin(s))]
xa = [best(a) for a in A]; xb = [best(b) for b in B]
da = np.array([mic(p, a["true"], a["cell"]) for p, a in zip(xa, A)])
db = np.array([mic(p, b["true"], b["cell"]) for p, b in zip(xb, B)])
agree = np.array([mic(p, q, a["cell"]) for p, q, a in zip(xa, xb, A)]) < 1.0
lim = (0.04, 12)
ax.plot(lim, lim, color=GRID, lw=1.2, zorder=1)
ax.axvline(1, color=GRID, lw=1.2, ls=(0, (3, 3)), zorder=1); ax.axhline(1, color=GRID, lw=1.2, ls=(0, (3, 3)), zorder=1)
ax.scatter(da[agree], db[agree], s=46, color=MUTED, edgecolor=BG, lw=1, zorder=3,
           label=f"Models agree within 1 Å: {agree.sum()} cells, {np.sum(da[agree]<1)} right")
ax.scatter(da[~agree], db[~agree], s=52, color=DIS, marker="X", edgecolor=BG, lw=0.6, zorder=3,
           label=f"Models disagree: {(~agree).sum()} cells, {np.sum(da[~agree]<1)} right")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xticks([0.1, 1, 10]); ax.set_xticklabels(["0.1", "1", "10"]); ax.set_yticks([0.1, 1, 10]); ax.set_yticklabels(["0.1", "1", "10"])
ax.set_xlabel("Direct: distance from the true site (Å)"); ax.set_ylabel("Diff1st: distance from the true site (Å)")
ax.set_title("Per cell, Na+Ca+Si"); ax.tick_params(length=0, which="both"); ax.set_aspect("equal")
for s in ax.spines.values():
    s.set_color(GRID)
ax.legend(loc="lower right", bbox_to_anchor=(1, -0.34), frameon=False, handletextpad=0.2)
fig.savefig(OUT, dpi=200)
print(OUT)
