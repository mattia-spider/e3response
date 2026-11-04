"""Adam against the zoom: cells recovered as the distance threshold grows, with the truth-start control."""
import sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]
R = np.load("e8_results.npz")
ZOOM, ADAM, CTRL, INK, MUTED, GRID, BG = "#1baf7a", "#6250d6", "#52514e", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": MUTED, "axes.edgecolor": GRID,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "figure.facecolor": BG, "axes.facecolor": BG})
X = np.linspace(0, 1.5, 301)
fig, axs = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True, constrained_layout=True)
for ax, (spec, ttl) in zip(axs, (("NaCaSi", "Na+Ca+Si"), ("NaCaSiO", "Na+Ca+Si+O"))):
    for key, col, ls, lab in (("off", CTRL, (0, (4, 3)), "Started at the true site (control)"),
                              ("dz", ZOOM, "-", "Zoom"), ("da", ADAM, "-", "Adam")):
        ax.step(X, [np.sum(R[f"{spec}_{key}"] < x) for x in X], where="post", color=col, lw=2, ls=ls, label=lab, zorder=3)
    for t in (0.5,):
        nz, na = int(np.sum(R[f"{spec}_dz"] < t)), int(np.sum(R[f"{spec}_da"] < t))
        ax.axvline(t, color=GRID, lw=1.2, zorder=1)
        ax.text(t + 0.02, 1.2, f"at 0.5 Å: zoom {nz}, Adam {na}", color=MUTED, fontsize=9)
    ax.set_xlim(0, 1.5); ax.set_ylim(0, 31); ax.set_title(ttl); ax.set_xlabel("Distance from the true site (Å)")
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
axs[0].set_ylabel("Cells recovered, out of 30")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="outside lower center", ncol=3, frameon=False)
fig.savefig(OUT, dpi=200)
print(OUT)
