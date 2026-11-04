"""GIF of the Adam descents on one cell, Na+Ca+Si: the seeds in the cell, chi2 per step, peaks against the target.

    python p3_gif.py pick              rank the recovered cells by how typical they are
    python p3_gif.py <cell> <out.gif>
"""
import sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

SPEC, SP = "NaCaSi", {11: "Na", 20: "Ca", 14: "Si"}
R = np.load("e8_results.npz")
if sys.argv[1] == "pick":
    cid, da, gap = R[f"{SPEC}_cid"], R[f"{SPEC}_da"], R[f"{SPEC}_gapa"]
    ok = da < 0.5
    mda, mgap = np.median(da[ok]), np.median(gap[ok])
    print(f"recovered {ok.sum()}/30, median |dr| {mda:.2f}, median gap {mgap:.1f}")
    D = np.load("cavity/cells.npz")
    for i in sorted(np.where(ok)[0], key=lambda i: abs(da[i] - mda) / mda + abs(gap[i] - mgap) / mgap)[:6]:
        d = np.load(f"descent_{SPEC}/cell_{cid[i]}.npz")
        print(f"  cell {cid[i]:3d} n={len(D[f'z_{cid[i]}'])} seeds {int(d['n_seeds'])} |dr| {da[i]:.2f} gap {gap[i]:.1f} chi2 {R[f'{SPEC}_ca'][i]:.1f}")
    sys.exit()

cid, out = int(sys.argv[1]), sys.argv[2]
d = np.load(f"descent_{SPEC}/cell_{cid}.npz"); s = np.load(f"sweep_direct/cell_{cid}.npz"); D = np.load("cavity/cells.npz")
ns, cell, true, z, dft = int(d["n_seeds"]), d["cell"], d["true"], d["z"], d["dft"]
ic = np.linalg.inv(cell); k = int(d["k"])


def rel(x):
    """Coordinates around the true site, minimum image."""
    f = (x - true) @ ic
    return (f - np.round(f)) @ cell


pos = D[f"pos_{cid}"]; atoms = rel(pos); keep = np.arange(len(z)) != k
traj = np.empty((ns, 100, 3))
for i in range(ns):
    traj[i, 0] = rel(d["pos"][i, 0])
    for t in range(1, 100):
        st = (d["pos"][i, t] - d["pos"][i, t - 1]) @ ic
        traj[i, t] = traj[i, t - 1] + (st - np.round(st)) @ cell
guard = np.array([2.18, 2.18, 3.08, 2.78, 2.95]) * 0.9
grid = rel(s["pts"][np.all(s["dist"] >= guard, 1)])
chi = np.where(d["wall"] == 0, d["chi2"], np.nan)
best = int(np.argmin(np.where(np.isfinite(chi[:ns]), chi[:ns], np.inf).min(1))); tbest = int(np.nanargmin(chi[best]))
dist = np.linalg.norm(traj[best], axis=1)

BLUE, GREY, INK, MUTED, GRID, BG = "#2a78d6", "#a9a8a3", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = {20: "#eda100", 14: "#1baf7a", 8: "#e87ba4", 11: "#6250d6"}
NM = {20: "Ca", 14: "Si", 8: "O", 11: "other Na"}
plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": MUTED, "axes.edgecolor": GRID, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "figure.facecolor": BG, "axes.facecolor": BG})
fig = plt.figure(figsize=(11.5, 6.2), dpi=90)
gs = fig.add_gridspec(4, 2, width_ratios=[1.25, 1], height_ratios=[2.4, 1, 1, 1], left=0.01, right=0.975, top=0.84, bottom=0.09, hspace=0.75, wspace=0.08)
ax = fig.add_subplot(gs[:, 0], projection="3d"); ax.set_proj_type("persp", focal_length=0.6)
# the descents move a few tenths of an A, so frame the seeds, not the whole cell
lo, hi = traj.min((0, 1)) - 2.2, traj.max((0, 1)) + 2.2
box = lambda p: np.all((p >= lo) & (p <= hi), 1)
for sp, c in COL.items():
    m = keep & (z == sp) & box(atoms)
    if m.any():
        ax.scatter(*atoms[m].T, s=120 if sp in (20, 11) else 80 if sp == 14 else 34, color=c, alpha=0.85 if sp != 8 else 0.6, lw=0, label=NM[sp], depthshade=False)
ax.scatter(*grid[box(grid)].T, s=9, color=GREY, alpha=0.55, lw=0, label="Allowed by the guards", depthshade=False)
ax.scatter(0, 0, 0, s=190, marker="*", color=INK, edgecolor=BG, lw=0.8, label="True Na site", depthshade=False, zorder=20)
lines = [ax.plot([], [], [], color=BLUE if i == best else GREY, lw=2.4 if i == best else 1.1, zorder=10 if i == best else 5)[0] for i in range(ns)]
dots = [ax.plot([], [], [], "o", ms=8 if i == best else 4.5, color=BLUE if i == best else GREY, mec=BG, mew=0.8, zorder=11 if i == best else 6)[0] for i in range(ns)]
ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2]); ax.set_box_aspect(hi - lo, zoom=1.3); ax.set_axis_off()
h, l = ax.get_legend_handles_labels()
fig.legend(h, l, loc="lower left", bbox_to_anchor=(0.01, 0.005), frameon=False, ncol=len(l), fontsize=8.5, handletextpad=0.1, columnspacing=1.0)

a1 = fig.add_subplot(gs[0, 1])
steps = np.arange(100)
for i in range(ns):
    a1.plot(steps, chi[i], color=GRID, lw=1, zorder=1)
ct = a1.plot(steps, chi[-1], color=INK, lw=1.4, ls=(0, (4, 3)), label="Started at the true site (control)", alpha=0)[0]
cur = [a1.plot([], [], color=BLUE if i == best else GREY, lw=2.2 if i == best else 1.1, zorder=3 if i == best else 2)[0] for i in range(ns)]
top = min(np.nanmax(chi[:ns, 0]) if np.isfinite(chi[:ns, 0]).any() else 60, np.nanmin(chi[:ns]) * 3)
a1.set_ylim(np.nanmin(chi) * 0.92, top); a1.set_xlim(0, 99); a1.set_xlabel("Adam step", labelpad=1); a1.set_ylabel("χ² (Na+Ca+Si)")
a1.set_title("χ² of each seed, inside the guards"); a1.grid(axis="y", color=GRID, lw=0.7); a1.tick_params(length=0)
for sp_ in a1.spines.values():
    sp_.set_visible(False)

ax2, ticks = {}, {}
target = {sp: np.sort(dft[z == sp]) for sp in SP}
for row, sp in enumerate(SP):
    a = fig.add_subplot(gs[row + 1, 1]); ax2[sp] = a
    vals = np.concatenate([target[sp], d["iso"][best, :, z == sp].ravel()])
    pad = max(3.0, 0.08 * np.ptp(vals)); a.set_xlim(vals.min() - pad, vals.max() + pad); a.set_ylim(0, 2); a.set_yticks([])
    a.vlines(target[sp], 1.05, 1.95, color=INK, lw=1.5)
    ticks[sp] = a.vlines(target[sp], 0.05, 0.95, color=BLUE, lw=1.5)
    a.set_ylabel(SP[sp], rotation=0, labelpad=12, va="center", color=INK); a.tick_params(length=0); a.xaxis.set_major_locator(plt.MaxNLocator(5))
    for sp_ in a.spines.values():
        sp_.set_visible(False)
    a.axhline(1, color=GRID, lw=0.8)
ax2[11].set_title("Peaks: DFT target (top), best seed (bottom)")
ax2[14].set_xlabel("σ_iso (ppm)", labelpad=1)
title = fig.text(0.015, 0.94, "", fontsize=13, fontweight="bold")
sub = fig.text(0.015, 0.895, f"C-S-H cell {cid}, mean of 3 Direct folds, DFT target, framework fixed, H not drawn. Blue: the seed that ends with the lowest χ²; grey: the other seeds.", color=MUTED, fontsize=9.5)

frames_t = np.unique(np.round(99 * (np.linspace(0, 1, 56) ** 1.7)).astype(int))
imgs = []
for n, t in enumerate(frames_t):
    for i in range(ns):
        lines[i].set_data_3d(*traj[i, :t + 1].T); dots[i].set_data_3d(*traj[i, [t]].T)
        cur[i].set_data(steps[:t + 1], chi[i, :t + 1])
    for sp in SP:
        v = np.sort(d["iso"][best, t, z == sp])
        ticks[sp].set_segments([[(x, 0.05), (x, 0.95)] for x in v])
    ax.view_init(elev=18, azim=-62 + 42 * n / (len(frames_t) - 1))
    title.set_text(f"Adam step {t + 1:3d}/100   best seed: {dist[t]:.2f} Å from the true site")
    fig.canvas.draw()
    imgs.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]).quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
ct.set_alpha(0.9); a1.legend(loc="upper right", frameon=False, fontsize=8.5)
for i in range(ns):
    lines[i].set_data_3d(*traj[i].T); dots[i].set_data_3d(*traj[i, [-1]].T)
title.set_text(f"Done: best point {dist[tbest]:.2f} Å from the true site")
fig.canvas.draw()
imgs.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]).quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=[110] * (len(imgs) - 1) + [3500], loop=0, optimize=True)
print(out, len(imgs), "frames, best seed", best, "at step", tbest, f"|dr| {dist[tbest]:.2f}")
