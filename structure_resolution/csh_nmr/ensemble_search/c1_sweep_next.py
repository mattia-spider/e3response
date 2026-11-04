"""Forward-only placement of one Na: no gradients, no Adam.

Coarse grid over the points a Na could physically occupy (loosest guards), then a two-level zoom
around the best seeds. sigma_iso of every atom at every candidate is saved per model, so every
loss, nucleus set, ensemble and guard tier can be compared offline without the GPU.

    python c1_sweep.py <model> <first_cell> <n_cells>
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE + "/sr")
from common import *

MODEL, FIRST, NCELLS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
NOGUARD = bool(os.environ.get("NOGUARD"))
OUT = f"{HERE}/sweep_{MODEL}{'_noguard' if NOGUARD else ''}"; os.makedirs(OUT, exist_ok=True)
H = 0.6 if NOGUARD else 0.5                 # coarse spacing, A
Z = [1, 8, 11, 14, 20]
# nearest Na-X over the 381 Na sites of the data set
DMIN = {8: 2.18, 1: 2.18, 14: 2.78, 20: 2.95, 11: 3.08}
TIER = {"tight": 1.0, "soft": 0.9, "loose": 0.8}
EPS = {11: 2.5, 20: 2.7, 14: 1.5}   # ppm, test sigma_iso RMSE of the 5-fold ensemble
COARSE_FOLDS, ALL_FOLDS = (0, 1, 2), (0, 1, 2, 3, 4)
STENCIL = np.array([[x, y, z] for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)
                    if abs(x) + abs(y) + abs(z) in (1, 3)], float)


def nearest(pts, a, k):
    """(m, 5) distance from each point to the nearest atom of each species, atom k excluded."""
    cell = np.array(a.cell); ic = np.linalg.inv(cell)
    keep = np.arange(len(a)) != k
    out = np.full((len(pts), len(Z)), np.inf)
    for iz, z in enumerate(Z):
        pos = a.positions[keep & (a.numbers == z)]
        if not len(pos):
            continue
        for c in np.array_split(np.arange(len(pts)), max(1, len(pts) // 3000)):
            df = (pts[c, None] - pos[None]) @ ic; df -= np.round(df)
            out[c, iz] = np.linalg.norm(df @ cell, axis=-1).min(1)
    return out


def allowed(dist, tier):
    return np.all(dist >= np.array([DMIN[z] * TIER[tier] for z in Z]) - (0.1 if tier == "tight" else 0.0), axis=1)


def grid_points(a):
    cell = np.array(a.cell); ns = np.maximum(4, np.round(a.cell.lengths() / H).astype(int))
    ax = [(np.arange(n) + 0.5) / n for n in ns]
    return np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3) @ cell


def chi2(iso, tgt, z, inflate=1.0):
    """Assignment-free chi2, per nucleus sorted against the target; iso (..., n_atoms)."""
    s = 0.0
    for zz, e in EPS.items():
        m = z == zz
        if m.any():
            s = s + np.sum((np.sort(iso[..., m], -1) - np.sort(tgt[m])) ** 2, -1) / (e * (inflate if zz == 11 else 1)) ** 2
    return s


def seeds_from(pts, score, ok, n, sep=0.8, taken=()):
    out = list(taken)
    added = 0
    for i in np.argsort(np.where(ok, score, np.inf)):
        if not ok[i] or added >= n:
            break
        if all(np.linalg.norm(pts[i] - p) > sep for p in out):
            out.append(pts[i]); added += 1
    return out


ds = dataset(); ids = test_ids(ds)
bulk = [i for i in ids if ds.structures[i].info["struct_type"] == "bulk"]
bulk = bulk[FIRST:FIRST + NCELLS]
print(MODEL, "cells", bulk, flush=True)
models = {f: load(MODEL, f) for f in ALL_FOLDS}
module = models[0][0]
iso_t = lambda t: jnp.trace(t, axis1=-2, axis2=-1) / 3

for cid in bulk:
    out_file = f"{OUT}/cell_{cid}.npz"
    if os.path.exists(out_file):
        continue
    t0 = time.time()
    a = ds.structures[cid]; k = int(np.where(a.numbers == 11)[0][0]); z = a.numbers.copy()
    dft = np.trace(a.arrays["nmr_tensors"], axis1=1, axis2=2) / 3
    pts = grid_points(a); dist = nearest(pts, a, k)
    # the control: only the branch's own guard, nearest atom at least 0.65 A away
    ok_loose = np.all(dist >= 0.65, 1) if NOGUARD else allowed(dist, "loose")
    pts, dist = pts[ok_loose], dist[ok_loose]
    probes = [np.where(np.arange(len(a))[..., None] == k, p, a.positions) for p in pts[np.random.default_rng(0).choice(len(pts), min(24, len(pts)), replace=False)]]
    one, cap = make_scan(module, models[0][1], a, k, probes=probes)
    one_iso = jax.jit(lambda x, q: iso_t(one(x, q)))

    def ev(P, folds):
        res = np.full((5, len(P), len(a)), np.nan, np.float32)
        for f in folds:
            q = models[f][1]
            for j, p in enumerate(np.asarray(P, np.float32)):
                res[f, j] = np.asarray(one_iso(jnp.asarray(p), q))
        return res

    true = a.positions[k].copy()
    iso_true = ev(true[None], ALL_FOLDS)
    ens_true = iso_true.mean(0)[0]
    coarse = ev(pts, COARSE_FOLDS)
    ec = np.nanmean(coarse, 0)
    tt = time.time() - t0
    sc_d = chi2(ec, dft, z, 3.0); sc_s = chi2(ec, ens_true, z, 3.0)
    if NOGUARD:
        seeds = seeds_from(pts, sc_d, np.ones(len(pts), bool), 6)
    else:
        seeds = seeds_from(pts, sc_d, allowed(dist, "tight"), 3)
        seeds = seeds_from(pts, sc_d, allowed(dist, "soft"), 3, taken=seeds)
        seeds = seeds_from(pts, sc_d, allowed(dist, "loose"), 2, taken=seeds)
        seeds = seeds_from(pts, sc_s, allowed(dist, "soft"), 2, taken=seeds)
    Zp, Zi, Zt = [], [], []       # zoom points, iso, level (1,2 = seed zoom; 11,12 = around truth)
    for tag, s0 in [(0, s) for s in seeds] + [(10, true)]:
        s = np.asarray(s0)
        for lev, step in ((1, 0.2), (2, 0.08)):
            P = s + STENCIL * step
            if lev == 1:
                P = np.vstack([s[None], P])
            r = ev(P, ALL_FOLDS)
            Zp.append(P); Zi.append(r); Zt.append(np.full(len(P), tag + lev))
            e = np.nanmean(r, 0)
            s = P[int(np.argmin(chi2(e, dft, z)))]
    zp = np.vstack(Zp); zi = np.concatenate(Zi, 1); zt = np.concatenate(Zt)
    np.savez_compressed(out_file, pts=pts, dist=dist, coarse=coarse, zoom_pts=zp, zoom_dist=nearest(zp, a, k), zoom_iso=zi, zoom_tag=zt,
                        true=true, iso_true=iso_true, dft=dft, z=z, k=k, cell=np.array(a.cell), cid=cid, cap=cap)
    print(f"cell {cid}: n={len(a)} cap={cap} coarse {len(pts)} pts ({tt:.0f}s) seeds {len(seeds)} total {time.time()-t0:.0f}s", flush=True)
    jax.clear_caches()
