"""Diff1st's zoom points are its own (its seeds differ from Direct's): audit and fix them on all cells."""
import glob, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE + "/sr")
from common import *
import structure_search as ss

iso_t = lambda t: jnp.trace(t, axis1=-2, axis2=-1) / 3
ds = dataset()
models = {f: load("deriv1st", f) for f in range(5)}
module = models[0][0]
total = 0
for f in sorted(glob.glob(f"{HERE}/sweep_deriv1st/cell_*.npz"), key=lambda x: int(x.split("_")[-1][:-4])):
    d = dict(np.load(f)); cid = int(d["cid"]); a = ds.structures[cid]; k = int(d["k"]); cap = int(d["cap"])
    finder = ss.jax_neighbours.neighbour_finder(cutoff=float(R_MAX), cell=np.asarray(a.cell), pbc=tuple(bool(x) for x in a.pbc))
    pos0 = jnp.asarray(a.positions, dtype=jnp.float32)
    need = jax.jit(lambda x: finder.get_neighbours(pos0.at[k].set(x), max_neighbours=128).neighbours)
    nz = np.array([(np.asarray(need(jnp.asarray(p))) != ss.jax_neighbours.MASK_VALUE).sum(1).max() for p in np.asarray(d["zoom_pts"], np.float32)])
    iz = np.where(nz > cap)[0]
    if not len(iz):
        continue
    one, _ = make_scan(module, models[0][1], a, k, capacity=int(nz.max()) + 1)
    run = jax.jit(lambda x, q: iso_t(one(x, q)))
    dm = 0.0
    for i in iz:
        for fold in range(5):
            new = np.asarray(run(jnp.asarray(d["zoom_pts"][i], jnp.float32), models[fold][1]))
            dm = max(dm, np.abs(new - d["zoom_iso"][fold, i]).max()); d["zoom_iso"][fold, i] = new
    np.savez_compressed(f, **d); total += len(iz)
    print(f"deriv1st cell {cid}: {len(iz)} zoom points, tags {sorted(set(d['zoom_tag'][iz].tolist()))}, largest change {dm:.3f} ppm", flush=True)
    jax.clear_caches()
print("points fixed:", total)
