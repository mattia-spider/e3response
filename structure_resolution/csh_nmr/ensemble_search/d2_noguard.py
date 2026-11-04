"""The control: the same model and loss, with only the branch's 0.65 A guard, over the whole cell."""
import glob, numpy as np
from d1_analyse import load, chi2, w1, mic, DMIN
Z=[1,8,11,14,20]; names=dict(zip(range(5),("H","O","Na","Si","Ca")))
for loss, nm in ((chi2,"chi2"),(w1,"W1")):
    print(nm, "Na+Ca+Si, ensemble of 3, DFT target, whole cell")
    res=[]
    for f in sorted(glob.glob("sweep_direct_noguard/cell_*.npz")):
        d=load(f); s=loss(d["cand_iso"].mean(0),d["dft"],d["z"],(11,20,14))
        i=int(np.argmin(s)); dd=d["cand_dist"][i]; rel=(dd/DMIN)
        t=loss(d["iso_true"][:3].mean(0),d["dft"],d["z"],(11,20,14))[0]
        res.append((mic(d["cand_pts"][i],d["true"],d["cell"]),rel.min()))
        print(f"  cell {int(d['cid']):3d}: n_cand {len(s)} |dr| {res[-1][0]:5.2f} loss best {s[i]:7.2f} true {t:7.2f}; nearest {names[int(np.argmin(dd))]} at {dd.min():.2f} A ({rel.min():.2f} of the data minimum)")
    r=np.array(res); print(f"  <0.5A {np.sum(r[:,0]<0.5)}/{len(r)}, median |dr| {np.median(r[:,0]):.2f}, best point inside soft guards {np.sum(r[:,1]>=0.9)}/{len(r)}")
