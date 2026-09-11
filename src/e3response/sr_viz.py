"""Offline plotting for the structure-resolution demos.

The optimisation runs (structure_resolution/**/structure_resolution_*.py) are expensive and
need the model + a GPU; the figures and GIFs do not. So a run dumps everything a plot needs
into ``plot_data.npz`` + ``plot_meta.json`` (via `save_bundle`), and these functions rebuild
the figures from that bundle alone — pure numpy + matplotlib, no jax, no model, no GPU. Tweak
the aesthetics here (or pass the keyword overrides) and re-run ``replot_grid.py`` to regenerate
in seconds without touching the optimiser.

A bundle is a flat dict; `save_bundle` routes every ``np.ndarray`` value to the ``.npz`` and
everything else (scalars, strings, small lists) to the ``.json``, and `load_bundle` merges them
back. The grid-search bundle keys are documented on `grid_result_figure`.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# One fixed, clearly-visible colour + size per species (by atomic number), shared by every
# panel that draws the framework. Tuned for contrast on a white ground (H is a mid grey, not
# near-white; Si a darker orange; Ca a medium green that won't clash with the true-site star).
ATOM_COLOR = {1: "#9AA0A6", 6: "#505050", 7: "#3050F8", 8: "#E8483C",
              11: "#9B59D0", 14: "#D98E32", 20: "#2CA02C"}
ATOM_SIZE = {1: 20, 6: 60, 7: 60, 8: 52, 11: 95, 14: 82, 20: 95}
ATOM_SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 11: "Na", 14: "Si", 20: "Ca"}
MODEL_COLOR = "royalblue"

DATA_NAME = "plot_data.npz"
META_NAME = "plot_meta.json"


# ── bundle I/O ─────────────────────────────────────────────────────────────────

def save_bundle(results_dir, bundle: dict):
    """Write `bundle` to ``<results_dir>/plot_data.npz`` (arrays) + ``plot_meta.json`` (rest).

    Everything the offline plotters need, in one place. Array values go to the npz; scalars,
    strings and small lists go to the json so they stay human-readable and editable by hand.
    """
    os.makedirs(results_dir, exist_ok=True)
    arrays = {k: v for k, v in bundle.items() if isinstance(v, np.ndarray)}
    meta = {k: v for k, v in bundle.items() if not isinstance(v, np.ndarray)}
    np.savez_compressed(os.path.join(results_dir, DATA_NAME), **arrays)
    with open(os.path.join(results_dir, META_NAME), "w") as f:
        json.dump(meta, f, indent=2)


def load_bundle(results_dir) -> dict:
    """Inverse of `save_bundle`: merge the npz arrays and the json metadata into one dict."""
    out = {}
    with np.load(os.path.join(results_dir, DATA_NAME), allow_pickle=False) as npz:
        out.update({k: npz[k] for k in npz.files})
    with open(os.path.join(results_dir, META_NAME)) as f:
        out.update(json.load(f))
    return out


# ── grid-search figures ──────────────────────────────────────────────────────────

GRID_DOC = """Bundle keys used:
      framework_pos (n_atoms,3), framework_numbers (n_atoms,)  — the fixed framework
      pos_true (3,), atom_k (int)                              — validation-only truth marker
      cand_pos (N,3), cand_loss (N,)                           — every converged candidate
      cluster_centres (K,3)                                    — distinct minima (circled)
      spec_symbols (S,), spec_grid (S,G), spec_target (S,G),
      spec_best (S,G), spec_best_loss (S,)                     — target vs best-cluster spectra
      metric, mol_idx, atom_symbol, target_label              — labels (json)"""


def _draw_grid_3d(ax, b, *, ac, asz, cand_cmap, framework_alpha, top_candidates):
    """Draw the framework + the best `top_candidates` Na candidates into 3-D axes `ax`.

    Framework: one FIXED colour per species (the fixed structure). Candidates: only the
    `top_candidates` LOWEST-loss converged positions (plotting all of them is noise), in a
    single sequential colour (dark = low loss = better) — the only varying colour, so it
    can't be read as a second species scheme. True Na site = black star; distinct minima
    circled. Returns the candidate scatter handle (for a colorbar) or None.
    """
    fp = np.asarray(b["framework_pos"]); an = np.asarray(b["framework_numbers"]).astype(int)
    for z in dict.fromkeys(an.tolist()):
        m = an == z
        ax.scatter(fp[m, 0], fp[m, 1], fp[m, 2], s=asz.get(z, 40),
                   c=ac.get(z, "violet"), alpha=framework_alpha, depthshade=True,
                   edgecolors="none", label=str(ATOM_SYMBOL.get(z, z)))
    P = np.asarray(b["cand_pos"]); Lg = np.asarray(b["cand_loss"])
    fin = np.where(np.isfinite(Lg))[0]
    keep = fin[np.argsort(Lg[fin])[:top_candidates]] if top_candidates else fin  # best-N by loss
    sc = None
    if keep.size:
        sc = ax.scatter(P[keep, 0], P[keep, 1], P[keep, 2], c=Lg[keep], s=70, cmap=cand_cmap,
                        norm=mcolors.LogNorm(), edgecolors="k", linewidths=0.5, depthshade=False,
                        zorder=6)
    pos_true = np.asarray(b["pos_true"])
    ax.scatter(*pos_true, s=340, marker="*", c="black", edgecolors="white", linewidths=0.9,
               zorder=8, label="true Na site")
    for j, c in enumerate(np.asarray(b["cluster_centres"])[:top_candidates or 5]):
        ax.scatter(*c, s=170, marker="o", facecolors="none", edgecolors="red",
                   linewidths=1.6, zorder=7, label="distinct minimum" if j == 0 else None)
    ax.set_xlabel("x (Å)", fontsize=7); ax.set_ylabel("y (Å)", fontsize=7)
    ax.set_zlabel("z (Å)", fontsize=7)
    ax.set_title(f"framework (fixed colours) + best {int(keep.size)} Na candidates", fontsize=9)
    ax.legend(fontsize=6, loc="upper left", ncol=2, framealpha=0.9)
    ax.tick_params(labelsize=6)
    return sc


def _draw_spectrum(ax, b, col, sym):
    """Draw target vs best-cluster spectrum for fitted nucleus `col` (symbol `sym`) into `ax`."""
    grids = np.asarray(b["spec_grid"]); tgt = np.asarray(b["spec_target"])
    best = np.asarray(b["spec_best"]); best_loss = np.asarray(b["spec_best_loss"])
    ax.plot(grids[col], tgt[col], color="limegreen", linewidth=1.8, label="target")
    ax.plot(grids[col], best[col], color=MODEL_COLOR, linewidth=1.4, label="best cluster")
    ax.invert_xaxis()                   # NMR convention: δ decreases to the right
    ax.set_xlabel("δ (ppm)", fontsize=8); ax.set_ylabel("intensity", fontsize=8)
    ax.set_title(f"{sym} spectrum   loss {float(best_loss[col]):.3f}", fontsize=9)
    ax.legend(fontsize=7); ax.tick_params(labelsize=7)


def _suptitle(b):
    return (f"NMR grid-search structure resolution  –  structure {b.get('mol_idx', '?')}  "
            f"atom k={int(b['atom_k'])} ({b.get('atom_symbol', '?')})  "
            f"target={b.get('target_label', '?')}")


def grid_3d_figure(b: dict, out_path, *, atom_color=None, atom_size=None,
                   cand_cmap="Blues_r", framework_alpha=0.55, top_candidates=5, dpi=150):
    """Standalone 3-D candidate-site map (framework + best `top_candidates` Na candidates)."""
    metric = b.get("metric", "loss")
    fig = plt.figure(figsize=(7.0, 6.0), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    sc = _draw_grid_3d(ax, b, ac=atom_color or ATOM_COLOR, asz=atom_size or ATOM_SIZE,
                       cand_cmap=cand_cmap, framework_alpha=framework_alpha,
                       top_candidates=top_candidates)
    if sc is not None:
        fig.colorbar(sc, ax=ax, label=f"Na candidate — loss ({metric})", fraction=0.03, pad=0.08)
    fig.suptitle(_suptitle(b), fontsize=10)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def grid_spectra_figure(b: dict, out_path, *, dpi=150):
    """Standalone spectra figure: one target-vs-best panel per fitted nucleus, in a row."""
    symbols = [str(s) for s in b["spec_symbols"]]
    n_spec = max(1, len(symbols))
    fig, axes = plt.subplots(1, n_spec, figsize=(5.5 * n_spec, 5.0), squeeze=False,
                             facecolor="white")
    for col, sym in enumerate(symbols):
        _draw_spectrum(axes[0, col], b, col, sym)
    fig.suptitle(_suptitle(b), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def grid_result_figure(b: dict, out_path, *, atom_color=None, atom_size=None,
                       cand_cmap="Blues_r", framework_alpha=0.55, top_candidates=5, dpi=150):
    """Combined grid-search figure: 3-D candidate-site map + one spectrum panel per nucleus.

    """ + GRID_DOC
    metric = b.get("metric", "loss")
    symbols = [str(s) for s in b["spec_symbols"]]
    n_spec = len(symbols)

    fig = plt.figure(figsize=(5.5 * (1 + n_spec), 5.0), facecolor="white")
    gs = fig.add_gridspec(1, 1 + n_spec, wspace=0.28)

    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    sc = _draw_grid_3d(ax3d, b, ac=atom_color or ATOM_COLOR, asz=atom_size or ATOM_SIZE,
                       cand_cmap=cand_cmap, framework_alpha=framework_alpha,
                       top_candidates=top_candidates)
    if sc is not None:
        fig.colorbar(sc, ax=ax3d, label=f"Na candidate — loss ({metric})", fraction=0.03, pad=0.08)
    for col, sym in enumerate(symbols):
        _draw_spectrum(fig.add_subplot(gs[0, 1 + col]), b, col, sym)

    fig.suptitle(_suptitle(b), fontsize=10)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def grid_descent_gif(b: dict, out_path, *, top_n=8, max_frames=120, fps=15,
                     atom_color=None, atom_size=None, path_cmap="plasma",
                     framework_alpha=0.15):
    """Animate the GIF_TOP_N lowest-final-loss descent paths in 3-D + their loss curves.

    Bundle keys used: framework_pos, framework_numbers, pos_true, all_pos (N,steps,3),
    all_losses (N,steps), metric.
    """
    ac = atom_color or ATOM_COLOR
    asz = atom_size or ATOM_SIZE
    metric = b.get("metric", "loss")
    all_pos = np.asarray(b["all_pos"]); all_losses = np.asarray(b["all_losses"])
    fp = np.asarray(b["framework_pos"]); an = np.asarray(b["framework_numbers"]).astype(int)
    pos_true = np.asarray(b["pos_true"])

    finalL = np.where(np.isfinite(all_losses[:, -1]), all_losses[:, -1], np.inf)
    order = np.argsort(finalL)[:min(top_n, int(np.isfinite(finalL).sum()) or top_n)]
    trajs = all_pos[order]                       # (N, steps, 3)
    tl = all_losses[order]                        # (N, steps)
    steps = trajs.shape[1]
    idxs = (np.round(np.linspace(0, steps - 1, max_frames)).astype(int)
            if steps > max_frames else np.arange(steps))
    cmap = plt.get_cmap(path_cmap)
    lo, hi = float(np.nanmin(finalL[order])), float(np.nanmax(finalL[order]) + 1e-9)
    colors = [cmap((finalL[i] - lo) / (hi - lo + 1e-9)) for i in order]

    figg = plt.figure(figsize=(11, 5.5), facecolor="white")
    gsg = figg.add_gridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.25)
    axp = figg.add_subplot(gsg[0, 0], projection="3d")
    axl = figg.add_subplot(gsg[0, 1])

    for z in dict.fromkeys(an.tolist()):
        m = an == z
        axp.scatter(fp[m, 0], fp[m, 1], fp[m, 2], s=asz.get(z, 40) * 0.5,
                    c=ac.get(z, "violet"), alpha=framework_alpha, edgecolors="none")
    axp.scatter(*pos_true, s=240, marker="*", c="limegreen", edgecolors="k",
                linewidths=0.5, zorder=6, label="true site")
    trails = [axp.plot([], [], [], "-", color=colors[j], alpha=0.6, lw=1.2)[0]
              for j in range(len(order))]
    heads = [axp.plot([], [], [], "o", color=colors[j], ms=6, mec="k", mew=0.4,
                      zorder=7)[0] for j in range(len(order))]
    axp.set_xlabel("x (Å)", fontsize=7); axp.set_ylabel("y (Å)", fontsize=7)
    axp.set_zlabel("z (Å)", fontsize=7); axp.tick_params(labelsize=6)
    axp.set_title(f"top {len(order)} descent paths (colour = final loss)", fontsize=9)
    axp.legend(fontsize=7, loc="upper left")

    for j in range(len(order)):
        axl.plot(np.arange(steps), np.where(np.isfinite(tl[j]), tl[j], np.nan),
                 color=colors[j], lw=1.0, alpha=0.8)
    axl.set_yscale("log"); axl.set_xlabel("Adam step", fontsize=8)
    axl.set_ylabel(f"loss ({metric})", fontsize=8); axl.set_title("loss curves", fontsize=9)
    axl.tick_params(labelsize=7)
    marker = axl.axvline(0, color="gray", lw=0.8)

    def _update(fi):
        i = idxs[fi]
        for j in range(len(order)):
            pts = trajs[j, :i + 1]
            trails[j].set_data_3d(pts[:, 0], pts[:, 1], pts[:, 2])
            heads[j].set_data_3d([trajs[j, i, 0]], [trajs[j, i, 1]], [trajs[j, i, 2]])
        marker.set_xdata([i, i])
        return trails + heads + [marker]

    anim = FuncAnimation(figg, _update, frames=len(idxs), interval=1000 // fps, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(figg)
