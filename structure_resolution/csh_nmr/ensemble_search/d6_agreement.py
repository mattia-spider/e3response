"""Is Direct/Diff1st agreement a real confidence signal, or only the shared cavity leak?
Split the cells by whether geometry alone already finds the site, and compare with the gap."""
import glob, os, numpy as np
from d1_analyse import load, chi2, mic, ok, variant, geometry_only
from d3_minima import minima

names = sorted(os.path.basename(f) for f in glob.glob("sweep_direct/cell_*.npz"))
A = [load(f"sweep_direct/{n}") for n in names]
B = [load(f"sweep_deriv1st/{n}") for n in names]
geo = geometry_only("soft")


def best(fn, d):
    p, s = fn(d)
    return p[int(np.argmin(s))]


g = np.array([mic(best(geo, a), a["true"], a["cell"]) for a in A]) < 1.0
print("geometry alone right (<1 A):", g.sum(), "/ 30")
for sp, nm in (((11, 20, 14), "Na+Ca+Si"), ((11, 20, 14, 8), "Na+Ca+Si+O")):
    fn = variant("soft", sp=sp)
    xa = [best(fn, a) for a in A]; xb = [best(fn, b) for b in B]
    agree = np.array([mic(p, q, a["cell"]) for p, q, a in zip(xa, xb, A)]) < 1.0
    right = np.array([mic(p, a["true"], a["cell"]) for p, a in zip(xa, A)]) < 1.0
    gap = []
    for a in A:
        p, s = fn(a); m = minima(p, s, a["cell"])
        gap.append(m[1][1] - m[0][1] if len(m) > 1 else np.inf)
    gap = np.array(gap)
    print(f"\n{nm}")
    for lab, m in (("geometry right", g), ("geometry wrong", ~g)):
        print(f"  {lab:15s} n={m.sum():2d}: agree {np.sum(agree&m):2d}, right among agree {np.sum(agree&m&right):2d}; disagree {np.sum(~agree&m):2d}, right among disagree {np.sum(~agree&m&right):2d}")
    # same number of cells answered as agreement does, chosen by the gap instead
    k = int(agree.sum()); sel = np.argsort(-gap)[:k]
    print(f"  top {k} by gap: right {right[sel].sum()}/{k}   by agreement: right {right[agree].sum()}/{k}")
