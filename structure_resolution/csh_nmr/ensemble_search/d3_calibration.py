"""What a geometry-blind run could report: is chi2 calibrated, how big a candidate set covers the
truth, does the absolute chi2 flag the misses, and how far the model's own minimum sits from truth."""
import glob, sys, numpy as np
from d1_analyse import load, chi2, mic, ok, rank_of_true
SP = {"Na+Ca+Si": (11, 20, 14), "Na+Ca+Si+O": (11, 20, 14, 8)}
model = sys.argv[1]
D = [load(f) for f in sorted(glob.glob(f"sweep_{model}/cell_*.npz"))]


def minima(pts, s, cell, sep=1.0):
    """Well-separated minima, best first: (position, chi2)."""
    out = []
    for i in np.argsort(s):
        if not np.isfinite(s[i]):
            break
        if all(mic(pts[i], p, cell) > sep for p, _ in out):
            out.append((pts[i], s[i]))
    return out


for name, sp in SP.items():
    print(f"\n== {name}, soft guards, DFT target, ensemble of 3 folds ({len(D)} cells)")
    nsite = np.array([sum((d["z"] == z).sum() for z in sp) for d in D])
    ltrue = np.array([chi2(d["iso_true"][:3].mean(0), d["dft"], d["z"], sp)[0] for d in D])
    print(f"chi2 at truth / n_sites: median {np.median(ltrue/nsite):.2f}, 10-90% {np.percentile(ltrue/nsite,10):.2f}-{np.percentile(ltrue/nsite,90):.2f}  (1 = errors as assumed)")
    M, hit, best, dd = [], [], [], []
    for d in D:
        s = np.where(ok(d["cand_dist"], "soft"), chi2(d["cand_iso"].mean(0), d["dft"], d["z"], sp), np.inf)
        m = minima(d["cand_pts"], s, d["cell"])
        M.append(m); best.append(m[0][1]); dd.append(mic(m[0][0], d["true"], d["cell"]))
    best = np.array(best); dd = np.array(dd)
    print("candidate set = minima within delta-chi2 of the best: truth inside (<1 A) / median size / max")
    for tol in (0, 2, 4, 7, 10, 15, 20):
        cov, size = [], []
        for d, m in zip(D, M):
            c = [p for p, l in m if l <= m[0][1] + tol]
            size.append(len(c)); cov.append(any(mic(p, d["true"], d["cell"]) <= 1.0 for p in c))
        print(f"   tol {tol:3d}: {np.sum(cov):2d}/{len(D)}  size {np.median(size):.0f}  max {max(size)}")
    # does the gap to the runner-up say anything?
    gap = np.array([(m[1][1] - m[0][1]) if len(m) > 1 else np.inf for m in M])
    good = dd <= 1.0
    print(f"gap best -> runner-up: right answers median {np.median(gap[good]):.1f}, wrong {np.median(gap[~good]):.1f}")
    for g in (2, 4, 7):
        sel = gap >= g
        print(f"   only cells with gap >= {g}: {sel.sum():2d}/{len(D)} answered, {np.sum(good & sel)} right (precision {np.sum(good & sel)/max(sel.sum(),1):.2f})")
    print(f"absolute chi2/n_sites: right {np.median((best/nsite)[good]):.2f}, wrong {np.median((best/nsite)[~good]):.2f}")
    # the model's own minimum near the truth: zoom around truth, tags 11 and 12
    off = []
    for d in D:
        m = d["zoom_tag"] > 10
        s = chi2(d["zoom_iso"][:3][:, m].mean(0), d["dft"], d["z"], sp)
        off.append(mic(d["zoom_pts"][m][int(np.argmin(s))], d["true"], d["cell"]))
    print(f"offset of the local chi2 minimum from truth: median {np.median(off):.2f} A (stencil reach 0.28 A per axis)")
print("\nrecovery by distance threshold, Na+Ca+Si+O:")
for th in (0.2, 0.3, 0.5, 1.0):
    print(f"   <{th} A: {np.sum(dd < th)}/{len(D)}")
