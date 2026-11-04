"""Positive control for e7_audit: its batched counter must find the 4 known overflows on cell 26."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE + "/sr")
from common import *
import structure_search as ss

ds = dataset()
for cid, known in ((26, 4), (89, 4), (19, 0)):
    d = np.load(f"{HERE}/sweep_direct_v0/cell_{cid}.npz"); a = ds.structures[cid]; k = int(d["k"]); cap = int(d["cap"])
    finder = ss.jax_neighbours.neighbour_finder(cutoff=float(R_MAX), cell=np.asarray(a.cell), pbc=tuple(bool(x) for x in a.pbc))
    pos0 = jnp.asarray(a.positions, dtype=jnp.float32)
    count = jax.jit(jax.vmap(lambda x: (finder.get_neighbours(pos0.at[k].set(x), max_neighbours=128).neighbours != ss.jax_neighbours.MASK_VALUE).sum(1).max()))
    P = d["pts"].astype(np.float32)
    n = np.concatenate([np.asarray(count(jnp.asarray(c))) for c in np.array_split(P, max(1, len(P) // 32))])
    print(f"cell {cid}: cap {cap}, over {int(np.sum(n > cap))}, known {known}, max need {n.max()}")
    jax.clear_caches()
