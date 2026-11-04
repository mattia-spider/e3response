"""Did the sweeps' neighbour capacity overflow at any candidate? An overflow silently drops neighbours
and corrupts sigma_iso at that point. Geometry only, no model."""
import glob, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE + "/sr")
from common import *
import structure_search as ss

jn = ss.jax_neighbours
ds = dataset()
print("cell   n sweepcap direct/diff1st | need: true, max coarse, max zoom | points over cap: coarse, zoom(cand), zoom(truth)  worst atom need at seed")
rows = []
for f in sorted(glob.glob(f"{HERE}/sweep_direct/cell_*.npz"), key=lambda x: int(x.split("_")[-1][:-4])):
    d = np.load(f); cid = int(d["cid"]); a = ds.structures[cid]; k = int(d["k"])
    cap_b = int(np.load(f.replace("sweep_direct", "sweep_deriv1st"))["cap"]); cap = int(d["cap"])
    finder = jn.neighbour_finder(cutoff=float(R_MAX), cell=np.asarray(a.cell), pbc=tuple(bool(x) for x in a.pbc))
    pos0 = jnp.asarray(a.positions, dtype=jnp.float32)

    @jax.jit
    def need(x):
        return finder.get_neighbours(pos0.at[k].set(x), max_neighbours=128).neighbours

    def needs(P):
        out = []
        for p in np.asarray(P, np.float32):
            nb = np.asarray(need(jnp.asarray(p)))
            out.append((nb != jn.MASK_VALUE).sum(1).max())
        return np.array(out)

    nt = needs(d["true"][None])[0]; nc = needs(d["pts"])
    tag = d["zoom_tag"]; nz = needs(d["zoom_pts"])
    print(f"{cid:4d} {len(a):3d} {cap:3d}/{cap_b:3d} | {nt:3d} {nc.max():3d} {nz.max():3d} | {np.sum(nc>cap):4d}/{len(nc)} {np.sum(nz[tag<10]>cap):4d}/{np.sum(tag<10)} {np.sum(nz[tag>=10]>cap):3d}/{np.sum(tag>=10)}", flush=True)
    rows.append((cid, cap, int(nt), int(nc.max()), int(nz.max()), int(np.sum(nc > cap)), int(np.sum(nz[tag < 10] > cap))))
    jax.clear_caches()
np.save(f"{HERE}/overflow.npy", np.array(rows))
