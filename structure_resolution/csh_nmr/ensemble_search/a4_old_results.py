"""The old grid-search runs: where were their best candidates, compared with the distances Na
actually has in the data?"""
import glob, json, os, csv, numpy as np
R = "/home/mattia/Desktop/ml_codes/CAMML/e3response/structure_resolution/csh_nmr/grid_search/results/"
runs = sorted(glob.glob(R + "*"))
print(len(runs), "runs")
G = {8: 2.18, 1: 2.18, 14: 2.78, 20: 2.95, 11: 3.08}
N = {1: "H", 8: "O", 11: "Na", 14: "Si", 20: "Ca"}
tot, bad = 0, 0
rows = []
for r in runs:
    f = r + "/plot_data.npz"
    if not os.path.exists(f):
        continue
    d = np.load(f, allow_pickle=True); m = json.load(open(r + "/plot_meta.json"))
    cell = d["cell"]; ic = np.linalg.inv(cell); k = m["atom_k"]
    pos = np.delete(d["framework_pos"], k, 0); z = np.delete(d["framework_numbers"], k)
    cc, cl, cs = d["cluster_centres"], d["cluster_losses"], d["cluster_sizes"]
    def near(p):
        df = (pos - p) @ ic; df -= np.round(df); dd = np.linalg.norm(df @ cell, axis=1)
        return {N[zz]: dd[z == zz].min() for zz in set(z.tolist())}
    def ok(p):
        dn = near(p); return all(dn[N[zz]] >= G[zz] - 0.1 for zz in set(z.tolist()))
    top = np.argsort(cl)[:5]
    dtrue = near(d["pos_true"])
    nb = near(cc[top[0]])
    phys = [ok(cc[i]) for i in range(len(cc))]
    rows.append((os.path.basename(r), m["target_label"][:9], len(cc), cl[top[0]], m["loss_at_true"], m["rmsd"], min(nb, key=nb.get), min(nb.values()), np.mean(phys), ok(cc[top[0]]), near(d["pos_true"])))
    print(f"{os.path.basename(r)} {m['target_label'][:10]:10s} sp={list(d['spec_symbols'])} clusters {len(cc):3d} loss best {cl[top[0]]:.2f} true {m['loss_at_true']:.2f} |dr| {m['rmsd']:.2f} | best: nearest {min(nb,key=nb.get)} {min(nb.values()):.2f} A, all phys-ok? {ok(cc[top[0]])}; {100*np.mean(phys):.0f}% of clusters are phys-ok")
print()
ss = [r for r in rows]
print("best cluster physically admissible:", sum(r[9] for r in ss), "/", len(ss))
print("median nearest-atom distance of best cluster:", np.median([r[7] for r in ss]))
for f in glob.glob(R + "*/structure_resolution_statistics.csv"):
    print(f, sum(1 for _ in open(f)) - 1)
