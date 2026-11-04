"""Reference sigma -> delta: why a shift hurts on one side only, and what an uncertain reference costs. Offline.

Offset with a prior of width tau, per nucleus: the chi2 minimised over b with b ~ N(0, tau^2),
tau -> inf is a free offset and tau -> 0 a perfectly known reference.
"""
import glob, os, numpy as np
from math import comb
from d1_analyse import load, mic, ok

EPS = {11: 2.5, 20: 2.7, 14: 1.5, 8: 3.8, 1: 0.55}
NAME = {11: "Na", 20: "Ca", 14: "Si", 8: "O", 1: "H"}
NCS, NCSO, NS, NC = (11, 20, 14), (11, 20, 14, 8), (11, 14), (11, 20)


def sign_p(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def chi2(iso, tgt, z, sp, tau=0.0, shift=None):
    out = 0.0
    for s in sp:
        m = z == s
        if not m.any():
            continue
        t = np.sort(tgt[m]) + (shift or {}).get(s, 0.0)
        r = np.sort(iso[:, m], 1) - t
        q = np.sum(r ** 2, 1) / EPS[s] ** 2
        if tau > 0:
            n = m.sum()
            q = q - (r.sum(1) / EPS[s] ** 2) ** 2 / (n / EPS[s] ** 2 + 1 / tau ** 2)
        out = out + q
    return out


def best(D, **kw):
    out = []
    for d in D:
        s = np.where(ok(d["cand_dist"], "soft"), chi2(d["cand_iso"].mean(0), d["dft"], d["z"], **kw), np.inf)
        out.append(d["cand_pts"][int(np.argmin(s))])
    return np.array(out)


def dist(D, P):
    return np.array([mic(p, d["true"], d["cell"]) for p, d in zip(P, D)])


names = sorted(os.path.basename(f) for f in glob.glob("sweep_direct/cell_*.npz"))
A = [load(f"sweep_direct/{n}") for n in names]
B = [load(f"sweep_deriv1st/{n}") for n in names]

print("=== signed error at the true site, model - DFT, ensemble of 3: mean over cells of the per-cell mean, and its spread")
for lab, D in (("Direct", A), ("Diff1st", B)):
    print(" ", lab, {NAME[s]: (round(float(np.mean([(d["iso_true"][:3].mean(0)[0] - d["dft"])[d["z"] == s].mean() for d in D])), 2),
                               round(float(np.std([(d["iso_true"][:3].mean(0)[0] - d["dft"])[d["z"] == s].mean() for d in D])), 2)) for s in (11, 20, 14, 8)})

print("\n=== soft candidates more than 1 A from truth, minus the true site, median over cells of the per-cell mean (ppm)")
for lab, D in (("Direct", A), ("Diff1st", B)):
    out = {}
    for s in (11, 20, 14, 8):
        v = []
        for d in D:
            n = d["cand_n_coarse"]; iso = d["cand_iso"][:, :n].mean(0)
            far = ok(d["cand_dist"][:n], "soft") & (mic(d["cand_pts"][:n], d["true"], d["cell"]) > 1.0)
            if far.any():
                v.append((iso[far][:, d["z"] == s] - d["iso_true"][:3].mean(0)[0][d["z"] == s]).mean())
        out[NAME[s]] = round(float(np.median(v)), 2)
    print(" ", lab, out)

print("\n=== which nucleus makes a shift hurt? Na+Ca+Si+O, target of one nucleus shifted, cells <0.5 / <1 A")
print("                 ", "   ".join(f"{b:+d} ppm" for b in (-4, -2, 2, 4)))
for lab, D in (("Direct", A), ("Diff1st", B)):
    for s in (11, 20, 14, 8):
        row = []
        for b in (-4, -2, 2, 4):
            d = dist(D, best(D, sp=NCSO, shift={s: float(b)}))
            row.append(f"{np.sum(d<.5):2d}/{np.sum(d<1):2d}")
        print(f"  {lab:8s} {NAME[s]:2s}    ", "   ".join(row))

print("\n=== uncertain reference: offset per nucleus with a prior of width tau (ppm)")
print("                  tau:    0        1        2        4       free    (<0.5 / <1)  | Direct vs Diff1st at tau=2, <0.5 and <1")
for sp, nm in ((NS, "Na+Si"), (NC, "Na+Ca"), (NCS, "Na+Ca+Si"), (NCSO, "Na+Ca+Si+O")):
    res = {}
    for lab, D in (("Direct", A), ("Diff1st", B)):
        r = []
        for tau in (0, 1, 2, 4, 1e4):
            res[lab, tau] = dist(D, best(D, sp=sp, tau=tau))
            r.append(f"{np.sum(res[lab, tau]<.5):2d}/{np.sum(res[lab, tau]<1):2d}")
        print(f"  {nm:11s} {lab:8s}", "    ".join(r), end="")
        if lab == "Diff1st":
            s = ""
            for tau in (2, 1e4):
                for th in (.5, 1):
                    x = res["Direct", tau] < th; y = res["Diff1st", tau] < th
                    b, c = int(np.sum(x & ~y)), int(np.sum(~x & y))
                    s += f"  tau {tau:g} <{th}: D {b} F {c} p={sign_p(b, c):.2f}"
            print(s, end="")
        print()

print("\n=== does model agreement survive an uncertain reference? (<1 A, right = Direct's answer <1 A)")
for sp, nm in ((NCS, "Na+Ca+Si"), (NCSO, "Na+Ca+Si+O")):
    for tau in (0, 2, 1e4):
        pa, pb = best(A, sp=sp, tau=tau), best(B, sp=sp, tau=tau)
        ag = np.array([mic(p, q, d["cell"]) for p, q, d in zip(pa, pb, A)]) < 1
        ra, rb = dist(A, pa) < 1, dist(B, pb) < 1
        print(f"  {nm:11s} tau {tau:<5g} agree {ag.sum():2d}, right {np.sum(ag & ra):2d}; disagree {(~ag).sum():2d}, Direct right {np.sum(~ag & ra)}, Diff1st right {np.sum(~ag & rb)}")
