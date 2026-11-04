"""What the data says about Na: how many per cell, how close anything ever gets to it, and how
much empty room a Na has in these cells (the search space a physical guard would leave)."""
import json, numpy as np, collections
from ase import Atoms
from ase.neighborlist import neighbor_list
E = json.load(open("/home/mattia/Desktop/ml_codes/CAMML/e3response/data/csh_nmr/Na_csh_filtered.json"))
print(len(E), E[0].keys())
A = [Atoms(numbers=e["numbers"], positions=e["positions"], cell=e["cell"], pbc=e["pbc"]) for e in E]
typ = np.array([e.get("struct_type") for e in E])
nna = np.array([(a.numbers == 11).sum() for a in A])
nat = np.array([len(a) for a in A])
vol = np.array([a.get_volume() for a in A])
for g in ("bulk", "surf"):
    m = typ == g
    print(g, "cells", m.sum(), "Na/cell", collections.Counter(nna[m].tolist()).most_common(), "atoms", nat[m].min(), np.median(nat[m]), nat[m].max(), "vol", vol[m].min().round(), np.median(vol[m]).round(), vol[m].max().round())
    print("   Ca/cell", np.median([(a.numbers==20).sum() for a,t in zip(A,typ) if t==g]), "Si/cell", np.median([(a.numbers==14).sum() for a,t in zip(A,typ) if t==g]))
names = {1:"H",8:"O",11:"Na",14:"Si",20:"Ca"}
# nearest-neighbour distance Na-X over the dataset
dm = collections.defaultdict(list)
for a in A:
    i, j, d = neighbor_list("ijd", a, 6.0)
    na = a.numbers[i] == 11
    for k in np.where(a.numbers == 11)[0]:
        sel = i == k
        for z in set(a.numbers[j[sel]].tolist()):
            dm[z].append(d[sel & (a.numbers[j] == z)].min())
print("\nnearest Na-X distance over all Na sites (A): min / 1% / median / max")
for z, v in sorted(dm.items()):
    v = np.array(v); print(f"  Na-{names[z]:>2} n={len(v):4d} min {v.min():.2f} p1 {np.percentile(v,1):.2f} med {np.median(v):.2f} max {v.max():.2f}")
# Na coordination (O within 3.0 A)
cn = []
for a in A:
    i, j, d = neighbor_list("ijd", a, 3.0)
    for k in np.where(a.numbers == 11)[0]:
        cn.append(((i == k) & (a.numbers[j] == 8)).sum())
print("O within 3A of Na: ", sorted(collections.Counter(cn).items()))
np.save(__file__.replace("a2_geometry.py", "na_mindist.npy"), {z: np.array(v).min() for z, v in dm.items()}, allow_pickle=True)

# how much of the cell is reachable by a Na once other atoms exclude a sphere
# (remove every Na in turn is expensive; remove all Na for the framework view)
rng = np.random.default_rng(0)
guards_old = {z: 0.65 for z in dm}
guards_new = {z: float(np.percentile(dm[z], 0.0)) - 0.1 for z in dm}
print("\npair guards from data:", {names[z]: round(v, 2) for z, v in guards_new.items()})
fo, fn, nsites_per = [], [], []
for a in [x for x, t in zip(A, typ) if t == "bulk"][:60]:
    cell = np.array(a.cell); n = 4000
    fr = rng.random((n, 3)); pts = fr @ cell
    keep = a.numbers != 11
    pos = a.positions[keep]; zs = a.numbers[keep]
    df = (pts[:, None] - pos[None]) @ np.linalg.inv(cell); df -= np.round(df); d = np.linalg.norm(df @ cell, axis=-1)
    ok_old = np.ones(n, bool); ok_new = np.ones(n, bool)
    for z in set(zs.tolist()):
        dmin = d[:, zs == z].min(1)
        ok_old &= dmin >= guards_old[z]
        ok_new &= dmin >= guards_new[z]
    fo.append(ok_old.mean()); fn.append(ok_new.mean())
print(f"fraction of the cell allowed: guard 0.65 A {np.mean(fo):.3f}, data-derived pair guards {np.mean(fn):.3f} (min {np.min(fn):.3f}, max {np.max(fn):.3f})")
# are there duplicates among cells: same framework with Na elsewhere?
keyc = collections.Counter(tuple(np.round(np.sort(a.cell.lengths()), 2)) + (len(a),) for a in A)
print("distinct (cell lengths, natoms):", len(keyc), "of", len(A), "; top", keyc.most_common(5))
src = collections.Counter(str(e.get("source")).split("/")[0] for e in E)
print("sources:", src.most_common(8))
print("example source:", E[0].get("source"), E[100].get("source"))
