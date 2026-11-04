"""Did any descent step overflow the neighbour capacity? Writes descent_<spec>/over_<cid>.npy, (starts, steps)."""
import glob, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE + "/sr")
from common import *
import structure_search as ss

ds = dataset()
for spec in ("NaCaSi", "NaCaSiO"):
    tot, cells = 0, []
    for f in sorted(glob.glob(f"{HERE}/descent_{spec}/cell_*.npz"), key=lambda x: int(x.split("_")[-1][:-4])):
        d = np.load(f); cid = int(d["cid"]); a = ds.structures[cid]; k = int(d["k"]); cap = int(d["cap"])
        finder = ss.jax_neighbours.neighbour_finder(cutoff=float(R_MAX), cell=np.asarray(a.cell), pbc=tuple(bool(x) for x in a.pbc))
        pos0 = jnp.asarray(a.positions, dtype=jnp.float32)
        count = jax.jit(jax.vmap(lambda x: (finder.get_neighbours(pos0.at[k].set(x), max_neighbours=128).neighbours != ss.jax_neighbours.MASK_VALUE).sum(1).max()))
        P = d["pos"].reshape(-1, 3).astype(np.float32)
        n = np.concatenate([np.asarray(count(jnp.asarray(c))) for c in np.array_split(P, max(1, len(P) // 32))]).reshape(d["pos"].shape[:2])
        over = n > cap
        np.save(f"{HERE}/descent_{spec}/over_{cid}.npy", over)
        tot += int(over.sum())
        if over.any():
            cells.append(cid)
            print(f"{spec} cell {cid}: cap {cap}, max need {n.max()}, {int(over.sum())}/{over.size} steps over, starts hit {int(over.any(1).sum())}, chi2 of the first over step's start {int(np.argmax(over.any(1)))}", flush=True)
        jax.clear_caches()
    print(f"{spec}: {tot} steps over the capacity, in cells {cells}", flush=True)
