"""Geometry only, no model. Under pair guards, how many separate pockets can hold a Na, and how
much does the empty framework already give away? Also: are the 'shuffled_N' cells siblings, same
framework with the Na elsewhere?"""
import json, collections, numpy as np
from ase import Atoms
from scipy import ndimage
from ase.geometry import find_mic
E = json.load(open("/home/mattia/Desktop/ml_codes/CAMML/e3response/data/csh_nmr/Na_csh_filtered.json"))
A = [Atoms(numbers=e["numbers"], positions=e["positions"], cell=e["cell"], pbc=e["pbc"]) for e in E]
typ = [e["struct_type"] for e in E]
TIERS = {"tight (data min-0.1)": {8: 2.08, 1: 2.08, 14: 2.68, 20: 2.85, 11: 2.98},
         "soft (x0.9)": {8: 1.87, 1: 1.87, 14: 2.41, 20: 2.56, 11: 2.68},
         "loose (x0.8)": {8: 1.66, 1: 1.66, 14: 2.14, 20: 2.28, 11: 2.38}}

def pockets(a, k, guards, h=0.33):
    cell = np.array(a.cell); L = a.cell.lengths(); ns = np.maximum(4, np.round(L / h).astype(int))
    ax = [(np.arange(n) + 0.5) / n for n in ns]
    fr = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    pts = fr @ cell
    ok = np.ones(len(pts), bool); ic = np.linalg.inv(cell)
    idx = np.arange(len(a)) != k
    for z in set(a.numbers[idx].tolist()):
        pos = a.positions[idx & (a.numbers == z)]
        for c in np.array_split(np.arange(len(pts)), max(1, len(pts) // 4000)):
            c = c[ok[c]]
            if not len(c): continue
            df = (pts[c, None] - pos[None]) @ ic; df -= np.round(df)
            ok[c] &= (np.linalg.norm(df @ cell, axis=-1).min(1) >= guards[z])
    g = ok.reshape(ns)
    lab, n = ndimage.label(g)
    # periodic merge
    par = list(range(n + 1))
    def fd(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for ax_i in range(3):
        a0 = np.take(lab, 0, ax_i); a1 = np.take(lab, -1, ax_i)
        for u, v in zip(a0.ravel(), a1.ravel()):
            if u and v: par[fd(u)] = fd(v)
    lab = np.vectorize(fd)(lab)
    ids = [i for i in np.unique(lab) if i]
    vox = abs(np.linalg.det(cell)) / np.prod(ns)
    vols = sorted([(lab == i).sum() * vox for i in ids], reverse=True)
    # is the true site inside an allowed pocket, and how far is the nearest allowed point?
    fr_t = (a.positions[k] @ ic) % 1
    d = None
    if ok.any():
        df = (pts[ok] - a.positions[k]) @ ic; df -= np.round(df)
        d = np.linalg.norm(df @ cell, axis=-1).min()
    return len(ids), vols, ok.mean(), d

ids = np.random.default_rng(0).permutation([i for i, t in enumerate(typ) if t == "bulk"])[:30]
for name, gd in TIERS.items():
    ns, fr, ds, big = [], [], [], []
    for i in ids:
        a = A[i]; k = int(np.where(a.numbers == 11)[0][0])
        n, v, f, d = pockets(a, k, gd)
        ns.append(n); fr.append(f); ds.append(d if d is not None else np.nan); big.append(v[0] if v else 0)
    print(f"{name}: pockets median {np.median(ns):.0f} (min {min(ns)}, max {max(ns)}), allowed volume {100*np.mean(fr):.2f}% of cell, "
          f"largest pocket median {np.median(big):.1f} A3, true site reached in {np.mean(np.array(ds)<0.5)*100:.0f}% of cells (nearest allowed pt median {np.nanmedian(ds):.2f} A)")

# siblings
fam = collections.defaultdict(list)
for i, e in enumerate(E):
    fam[e["source"].split("/")[0]].append(i)
print("\nfamilies:", {k: len(v) for k, v in fam.items()})
def framework_rmsd(a, b):
    if len(a) != len(b) or (a.numbers != b.numbers).any():
        return None, None
    keep = a.numbers != 11
    fa = a.get_scaled_positions()[keep]; fb = b.get_scaled_positions()[keep]
    df = fa - fb; df -= np.round(df)
    df -= df.mean(0)                                # rigid shift
    df -= np.round(df)
    r = np.sqrt(np.mean(np.sum((df @ np.array(a.cell)) ** 2, 1)))
    na = (a.get_scaled_positions()[~keep] - b.get_scaled_positions()[~keep]); na -= np.round(na)
    return r, np.linalg.norm(na @ np.array(a.cell), axis=1).min()
for f, v in list(fam.items())[:8]:
    rs = []
    for x in v[:12]:
        for y in v[:12]:
            if x < y:
                r, dn = framework_rmsd(A[x], A[y])
                if r is not None: rs.append((r, dn))
    if rs:
        rs = np.array(rs); print(f"  {f}: pairs same atom order {len(rs)}, framework RMSD median {np.median(rs[:,0]):.2f} min {rs[:,0].min():.2f} A; Na separation median {np.median(rs[:,1]):.2f}")
    else:
        print(f"  {f}: no comparable pair")
