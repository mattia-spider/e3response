"""Recompute the sweep points whose neighbour list overflowed, with a sufficient capacity, in place.

Originals are kept in sweep_<model>_v0. One point that did not overflow is recomputed as a control.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE + "/sr")
from common import *
import structure_search as ss

CELLS = [26, 64, 77, 89, 101, 197, 239, 241, 287]
iso_t = lambda t: jnp.trace(t, axis1=-2, axis2=-1) / 3
ds = dataset()
flags = {}
for cid in CELLS:
    d = np.load(f"{HERE}/sweep_direct/cell_{cid}.npz"); a = ds.structures[cid]; k = int(d["k"]); cap = int(d["cap"])
    finder = ss.jax_neighbours.neighbour_finder(cutoff=float(R_MAX), cell=np.asarray(a.cell), pbc=tuple(bool(x) for x in a.pbc))
    pos0 = jnp.asarray(a.positions, dtype=jnp.float32)
    need = jax.jit(lambda x: finder.get_neighbours(pos0.at[k].set(x), max_neighbours=128).neighbours)
    count = lambda P: np.array([(np.asarray(need(jnp.asarray(p))) != ss.jax_neighbours.MASK_VALUE).sum(1).max() for p in np.asarray(P, np.float32)])
    nc, nz = count(d["pts"]), count(d["zoom_pts"])
    flags[cid] = (np.where(nc > cap)[0], np.where(nz > cap)[0], int(max(nc.max(), nz.max())) + 1)
    print(cid, "over cap:", len(flags[cid][0]), "coarse,", len(flags[cid][1]), "zoom, new cap", flags[cid][2], flush=True)
    jax.clear_caches()

for model in ("direct", "deriv1st"):
    models = {f: load(model, f) for f in range(5)}
    module = models[0][0]
    for cid in CELLS:
        f = f"{HERE}/sweep_{model}/cell_{cid}.npz"; d = dict(np.load(f)); a = ds.structures[cid]; k = int(d["k"])
        ic, iz, cap = flags[cid]
        one, _ = make_scan(module, models[0][1], a, k, capacity=cap)
        run = jax.jit(lambda x, q: iso_t(one(x, q)))
        ev = lambda p, fold: np.asarray(run(jnp.asarray(p, jnp.float32), models[fold][1]))
        ctrl = int(np.setdiff1d(np.arange(len(d["pts"])), ic)[0])
        c_err = max(np.abs(ev(d["pts"][ctrl], fold) - d["coarse"][fold, ctrl]).max() for fold in range(3))
        dm = 0.0
        for i in ic:
            for fold in range(3):
                new = ev(d["pts"][i], fold); dm = max(dm, np.abs(new - d["coarse"][fold, i]).max()); d["coarse"][fold, i] = new
        for i in iz:
            for fold in range(5):
                new = ev(d["zoom_pts"][i], fold); dm = max(dm, np.abs(new - d["zoom_iso"][fold, i]).max()); d["zoom_iso"][fold, i] = new
        np.savez_compressed(f, **d)
        print(f"{model} cell {cid}: {len(ic)}+{len(iz)} points, largest change {dm:.3f} ppm, control {c_err:.4f} ppm", flush=True)
        jax.clear_caches()
