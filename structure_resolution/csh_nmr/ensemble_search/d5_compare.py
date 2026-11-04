"""Direct against Diff1st on the same 30 cells: paired counts, the 12 easy / 18 other split,
a model-mixed ensemble on the coarse grid, and whether the two models agreeing flags right answers."""
import glob, os, numpy as np
from math import comb
from d1_analyse import load, chi2, w1, mic, ok, evaluate, variant, coarse_only, five_fold, geometry_only

names = sorted(os.path.basename(f) for f in glob.glob("sweep_direct/cell_*.npz"))
first12 = {os.path.basename(f) for f in glob.glob("sweep_direct12/cell_*.npz")}
A = [load(f"sweep_direct/{n}") for n in names]
B = [load(f"sweep_deriv1st/{n}") for n in names]
easy = np.array([n in first12 for n in names])
print("cells", len(names), "first 12:", easy.sum())
NCS, NCSO, NC = (11, 20, 14), (11, 20, 14, 8), (11, 20)


def sign_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def row(name, fn):
    da, ra = evaluate(A, fn); db, rb = evaluate(B, fn)
    t3 = lambda r, m: sum(1 for x, k in zip(r, m) if k and x and x <= 3)
    s = f"{name:34s}"
    for lab, m in (("12", easy), ("18", ~easy), ("30", np.ones(30, bool))):
        s += f" | {lab}: {np.sum(da[m]<.5):2d}/{np.sum(da[m]<1):2d}/{t3(ra,m):2d} vs {np.sum(db[m]<.5):2d}/{np.sum(db[m]<1):2d}/{t3(rb,m):2d}"
    for th in (0.5, 1.0):
        b = int(np.sum((da < th) & ~(db < th))); c = int(np.sum(~(da < th) & (db < th)))
        s += f" | {th}: D only {b} F only {c} p={sign_p(b,c):.2f}"
    print(s)


print("\neach block: Direct </1/top3 vs Diff1st, then discordant cells (exact sign test)")
row("geometry only", geometry_only("soft"))
row("chi2 NaCaSi", variant("soft"))
row("chi2 NaCaSi coarse only", coarse_only("soft", chi2))
row("chi2 NaCa coarse only", coarse_only("soft", chi2, NC))
row("W1 NaCaSi coarse only", coarse_only("soft", w1))
row("chi2 NaCaSi ens5 zoomed", five_fold("soft"))
row("chi2 NaCaSiO", variant("soft", sp=NCSO))
row("chi2 NaCa", variant("soft", sp=NC))
row("self target", variant("soft", tgt="self"))
row("held-out target", variant("soft", tgt="heldout"))

same = [np.allclose(a["pts"], b["pts"]) if a["pts"].shape == b["pts"].shape else False for a, b in zip(A, B)]
print("\nsame coarse grid in both sweeps:", sum(same), "/ 30")


def mixed(sp, a, b):
    n = a["cand_n_coarse"]
    e = 0.5 * (a["cand_iso"][:, :n].mean(0) + b["cand_iso"][:, :n].mean(0))
    return a["pts"], np.where(ok(a["dist"], "soft"), chi2(e, a["dft"], a["z"], sp), np.inf)


if all(same):
    for sp, nm in ((NCS, "Na+Ca+Si"), (NCSO, "Na+Ca+Si+O"), (NC, "Na+Ca")):
        for lab, S in (("Direct", A), ("Diff1st", B)):
            dd = np.array([mic(*(lambda p, s: (p[int(np.argmin(s))], d["true"]))(d["pts"], np.where(ok(d["dist"], "soft"), chi2(d["cand_iso"][:, :d["cand_n_coarse"]].mean(0), d["dft"], d["z"], sp), np.inf)), d["cell"]) for d in S])
            print(f"  coarse {nm:11s} {lab:8s} <0.5 {np.sum(dd<.5):2d}  <1 {np.sum(dd<1):2d}  med {np.median(dd):.2f}")
        dd = []
        for a, b in zip(A, B):
            p, s = mixed(sp, a, b); dd.append(mic(p[int(np.argmin(s))], a["true"], a["cell"]))
        dd = np.array(dd)
        print(f"  coarse {nm:11s} mixed 3+3 <0.5 {np.sum(dd<.5):2d}  <1 {np.sum(dd<1):2d}  med {np.median(dd):.2f}  (18 others: <0.5 {np.sum(dd[~easy]<.5)}, <1 {np.sum(dd[~easy]<1)})")

print("\ndo the two models agree on the best point? (needs no truth)")
for sp, nm in ((NCS, "Na+Ca+Si"), (NCSO, "Na+Ca+Si+O")):
    fn = variant("soft", sp=sp); agree, right_a, right_b = [], [], []
    for a, b in zip(A, B):
        pa, sa = fn(a); pb, sb = fn(b)
        xa, xb = pa[int(np.argmin(sa))], pb[int(np.argmin(sb))]
        agree.append(mic(xa, xb, a["cell"])); right_a.append(mic(xa, a["true"], a["cell"])); right_b.append(mic(xb, b["true"], b["cell"]))
    agree, right_a, right_b = map(np.array, (agree, right_a, right_b))
    for tol in (0.5, 1.0):
        m = agree < tol
        print(f"  {nm:11s} agree <{tol}: {m.sum():2d}/30, Direct's answer right(<1A) {np.sum(right_a[m]<1)}/{m.sum()}, (<0.5A) {np.sum(right_a[m]<.5)}/{m.sum()}; disagree: right {np.sum(right_a[~m]<1)}/{(~m).sum()}")
