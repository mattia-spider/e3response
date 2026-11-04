"""Adam on the mean of 3 Direct folds, from the sweep's own seeds, so it compares fairly with the zoom.

The last start is the true site, an oracle control that is never a candidate. Every step saves
position, chi2, guard penalty and the ensemble sigma_iso, so scoring stays offline. Sequential: a
vmap of 8 lanes runs out of memory on 8 GB.

    python e1_descent.py <NaCaSi|NaCaSiO> <first_cell> <n_cells>
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE + "/sr")
from common import *

SPEC, FIRST, NCELLS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
STEPS = int(os.environ.get("STEPS", 100))
OUT = os.environ.get("OUT", f"{HERE}/descent_{SPEC}"); os.makedirs(OUT, exist_ok=True)
LR0, LR1 = 0.04, 0.004  # A, cosine
WALL = 200.0            # chi2 per A^2 inside a pair guard
FOLDS = (0, 1, 2)
SPECIES = {"NaCaSi": (11, 20, 14), "NaCaSiO": (11, 20, 14, 8)}[SPEC]
EPS = {11: 2.5, 20: 2.7, 14: 1.5, 8: 3.8}
SOFT = {8: 2.18 * 0.9, 1: 2.18 * 0.9, 14: 2.78 * 0.9, 20: 2.95 * 0.9, 11: 3.08 * 0.9}

ds = dataset(); ids = test_ids(ds)
bulk = [i for i in ids if ds.structures[i].info["struct_type"] == "bulk"]
bulk = bulk[FIRST:FIRST + NCELLS]
print(SPEC, "cells", bulk, flush=True)
models = [load("direct", f) for f in FOLDS]
module, params = models[0][0], [m[1] for m in models]
lrs = LR1 + 0.5 * (LR0 - LR1) * (1 + np.cos(np.pi * np.arange(STEPS) / STEPS))

for cid in bulk:
    out_file = f"{OUT}/cell_{cid}.npz"
    if os.path.exists(out_file):
        continue
    t0 = time.time()
    a = ds.structures[cid]; k = int(np.where(a.numbers == 11)[0][0]); z = a.numbers.copy()
    dft = np.trace(a.arrays["nmr_tensors"], axis1=1, axis2=2) / 3
    sw = np.load(f"{HERE}/sweep_direct/cell_{cid}.npz")
    seeds = sw["zoom_pts"][sw["zoom_tag"] == 1][::15]
    true = sw["true"]
    starts = np.vstack([seeds, true[None]])
    # sized on the starts, not the sweep's cap: that one is short at a few points and 8 too big at others
    finder = ss.jax_neighbours.neighbour_finder(cutoff=float(R_MAX), cell=np.asarray(a.cell), pbc=tuple(bool(x) for x in a.pbc))
    pos0 = jnp.asarray(a.positions, dtype=jnp.float32)
    need = max(int((np.asarray(finder.get_neighbours(pos0.at[k].set(s), max_neighbours=128).neighbours) != ss.jax_neighbours.MASK_VALUE).sum(1).max())
               for s in np.asarray(starts, np.float32))
    one, cap = make_scan(module, params[0], a, k, capacity=need + 3)

    terms = [(jnp.asarray(np.where(z == s)[0]), jnp.asarray(np.sort(dft[z == s])), EPS[s]) for s in SPECIES]
    cell = jnp.asarray(np.array(a.cell), jnp.float32); icell = jnp.linalg.inv(cell)
    keep = np.arange(len(a)) != k
    others = [(jnp.asarray(a.positions[keep & (z == s)], jnp.float32), r) for s, r in SOFT.items() if (keep & (z == s)).any()]

    def chi2(iso):
        return sum(jnp.sum((jnp.sort(iso[i]) - t) ** 2) / e ** 2 for i, t, e in terms)

    def guard(x):
        w = 0.0
        for pos, r in others:
            f = (pos - x) @ icell; f = f - jnp.round(f)
            w = w + jnp.maximum(r - jnp.linalg.norm(f @ cell, axis=-1).min(), 0.0) ** 2
        return WALL * w

    def iso_of(x, p):
        return jnp.trace(one(x, p), axis1=-2, axis2=-1) / 3

    def total(x):
        iso = sum(iso_of(x, p) for p in params) / len(params)
        c, w = chi2(iso), guard(x)
        return c + w, (c, w, iso)

    joint = jax.jit(jax.value_and_grad(total, has_aux=True))
    fwd = jax.jit(iso_of)
    back = jax.jit(lambda x, p, w: jax.grad(lambda y: jnp.sum(iso_of(y, p) * w))(x))
    dchi, dwall = jax.jit(jax.value_and_grad(chi2)), jax.jit(jax.value_and_grad(guard))

    def split(x):
        """As joint, one fold at a time: slower, but no single 4 GB block on the biggest cells."""
        iso = sum(fwd(x, p) for p in params) / len(params)
        c, w = dchi(iso); wall, gw = dwall(x)
        return (c + wall, (c, wall, iso)), sum(back(x, p, w) for p in params) / len(params) + gw

    vg = split if len(a) * cap > 5000 else joint
    if "GRADCHECK" in os.environ:
        for s in np.asarray(starts[[0, -1]], np.float32):
            (lj, _), gj = joint(jnp.asarray(s)); (ls, _), gs = split(jnp.asarray(s))
            print(f"cell {cid}: loss {float(lj):.5f} {float(ls):.5f}  grad {np.asarray(gj)} {np.asarray(gs)}  rel diff {float(jnp.linalg.norm(gj - gs) / jnp.linalg.norm(gj)):.2e}", flush=True)
        sys.exit()
    X, C, W, I = [], [], [], []
    for s in np.asarray(starts, np.float32):
        x = s.copy(); m = np.zeros(3, np.float32); v = np.zeros(3, np.float32)
        xs, cs, ws, isos = [], [], [], []
        for t, lr in enumerate(lrs):
            (_, (c, w, iso)), g = vg(jnp.asarray(x))
            g = np.nan_to_num(np.asarray(g), nan=0.0, posinf=0.0, neginf=0.0)
            xs.append(x.copy()); cs.append(float(c)); ws.append(float(w)); isos.append(np.asarray(iso))
            m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
            x = x - lr * (m / (1 - 0.9 ** (t + 1))) / (np.sqrt(v / (1 - 0.999 ** (t + 1))) + 1e-8)
        X.append(xs); C.append(cs); W.append(ws); I.append(isos)
    C = np.array(C); W = np.array(W)
    # the first step sits on a sweep point, so the two code paths must give the same sigma_iso
    i0 = np.where(sw["zoom_tag"] == 1)[0][::15]
    ref = np.nanmean(sw["zoom_iso"][:3][:, i0], 0)
    check = float(np.abs(np.array(I)[:len(seeds), 0] - ref).max())
    np.savez_compressed(out_file, pos=np.array(X), chi2=C, wall=W, iso=np.array(I, np.float32), n_seeds=len(seeds),
                        true=true, dft=dft, z=z, k=k, cell=np.array(a.cell), cid=cid, cap=cap)
    print(f"cell {cid}: n={len(a)} cap={cap} seeds {len(seeds)} chi2 start->best {C[:len(seeds),0].min():.1f}->{np.where(W==0,C,np.inf)[:len(seeds)].min():.1f} "
          f"truth start {C[-1,0]:.1f}->{np.where(W==0,C,np.inf)[-1].min():.1f} check {check:.3f} ppm total {time.time()-t0:.0f}s", flush=True)
    jax.clear_caches()
