"""Why did the no-guard control work? Are unphysical decoys gone, or hidden by the ensemble / the grid?"""
import glob, numpy as np
from d1_analyse import load, chi2, w1, mic, DMIN, ok
SP = (11, 20, 14)
D = [load(f) for f in sorted(glob.glob("sweep_direct_noguard/cell_*.npz"))]


def report(name, loss, fold):
    n05, nphys = 0, 0
    deep = []
    for d in D:
        iso = d["cand_iso"].mean(0) if fold is None else d["cand_iso"][fold]
        tru = d["iso_true"][:3].mean(0) if fold is None else d["iso_true"][fold]
        s = loss(iso, d["dft"], d["z"], SP); lt = loss(tru, d["dft"], d["z"], SP)[0]
        i = int(np.argmin(s)); n05 += mic(d["cand_pts"][i], d["true"], d["cell"]) < 0.5
        nphys += ok(d["cand_dist"][[i]], "soft")[0]
        un = d["cand_dist"].min(1) < 1.5       # within 1.5 A of some atom: no Na ever was
        deep.append((s[un].min() / lt, d["cand_dist"][un][int(np.argmin(s[un]))].min()))
    deep = np.array(deep)
    print(f"{name:22s} <0.5A {n05}/6  best inside soft {nphys}/6 | best point with an atom <1.5 A: loss/loss@true median {np.median(deep[:,0]):.2f} (min {deep[:,0].min():.2f}), {np.sum(deep[:,0]<1)}/6 beat the truth")


for loss, nm in ((chi2, "chi2"), (w1, "W1")):
    report(f"{nm} ensemble 3", loss, None)
    for f in (0, 1, 2):
        report(f"{nm} fold {f}", loss, f)
d = D[0]
print("\ncoarse grid only, how many points have an atom < 1.5 A:", [int((x["dist"].min(1) < 1.5).sum()) for x in D], "of", [len(x["pts"]) for x in D])
print("spread between folds, median std of sigma_iso at Na (ppm): by nearest-atom distance")
for lo, hi in ((0.65, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 4)):
    v, vc = [], []
    for x in D:
        m = (x["dist"].min(1) >= lo) & (x["dist"].min(1) < hi)
        if m.any():
            v.append(np.std(x["coarse"][:3][:, m][:, :, x["z"] == 11], axis=0).mean(-1))
            vc.append(np.std(x["coarse"][:3][:, m][:, :, x["z"] == 20], axis=0).mean(-1))
    v = np.concatenate(v); vc = np.concatenate(vc)
    print(f"  {lo:.2f}-{hi:.2f} A: Na {np.median(v):7.2f}  Ca {np.median(vc):6.2f}  n={len(v)}")
print("  at the true site: Na", np.mean([x["iso_true"][:3, 0][:, x["z"] == 11].std(0).mean() for x in D]).round(2), " Ca", np.mean([x["iso_true"][:3, 0][:, x["z"] == 20].std(0).mean() for x in D]).round(2))
