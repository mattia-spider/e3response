"""Per-element sigma_iso error of the new C-S-H checkpoints, from the saved test predictions:
single model, ensemble over folds, with/without a per-cell per-element offset."""
import glob, re, sys, numpy as np
R = "/home/mattia/Desktop/ml_codes/nmr_project/jz_runs/parity/q1_csh/"
Z = {1: "H", 8: "O", 11: "Na", 14: "Si", 20: "Ca"}
def iso(t): return np.trace(t, axis1=1, axis2=2) / 3
runs = {}
for f in sorted(glob.glob(R + "*.npz")):
    n = re.match(r"(.*)_r4.*_f(\d)", f.split("/")[-1])
    d = np.load(f)
    runs.setdefault(n[1], {})[int(n[2])] = dict(
        t=d["test/nmr_tensors/target"], p=d["test/nmr_tensors/pred"], s=d["test/nmr_tensors/struct"], z=d["test/nmr_tensors/species"])

def stats(pi, ti, s, z):
    out = {}
    for zz, nm in Z.items():
        m = z == zz
        e = pi[m] - ti[m]
        cells = s[m]
        ec = e.copy()
        for c in np.unique(cells):
            ec[cells == c] -= e[cells == c].mean()
        out[nm] = (np.sqrt(np.mean(e**2)), np.sqrt(np.mean(ec**2)), np.mean(np.abs(e)), m.sum())
    return out

print("sigma_iso test RMSE ppm (36 cells). columns: raw / after per-cell per-element offset / MAE")
allp = []
for model, folds in runs.items():
    print("\n==", model)
    assert all((folds[0]["s"] == f["s"]).all() for f in folds.values()), "test sets differ"
    t = iso(folds[0]["t"]); s = folds[0]["s"]; z = folds[0]["z"]
    singles = [stats(iso(f["p"]), t, s, z) for f in folds.values()]
    ens = iso(np.mean([f["p"] for f in folds.values()], 0))
    allp.append(np.array([iso(f["p"]) for f in folds.values()]))
    es = stats(ens, t, s, z)
    for nm in Z.values():
        a = np.mean([x[nm] for x in singles], 0)
        print(f"  {nm:>2} n={es[nm][3]:4d} single {a[0]:6.2f}/{a[1]:6.2f}/{a[2]:6.2f}   ens5 {es[nm][0]:6.2f}/{es[nm][1]:6.2f}/{es[nm][2]:6.2f}")
ens15 = np.concatenate(allp).mean(0)
es = stats(ens15, t, s, z)
print("\n== ensemble of all 15")
for nm in Z.values():
    print(f"  {nm:>2} {es[nm][0]:6.2f}/{es[nm][1]:6.2f}/{es[nm][2]:6.2f}")

# split the Na error by bulk/surf
import json
sys.path.insert(0, "/home/mattia/Desktop/ml_codes/CAMML/e3response/src")
entries = json.load(open("/home/mattia/Desktop/ml_codes/CAMML/e3response/data/csh_nmr/Na_csh_filtered.json"))
typ = np.array([e.get("struct_type") for e in entries])
for g in ("bulk", "surf"):
    print("\n==", g, "ensemble of 5 deriv1st")
    folds = runs["deriv1st"]
    ens = iso(np.mean([f["p"] for f in folds.values()], 0))
    m = typ[s] == g
    r = stats(ens[m], t[m], s[m], z[m])
    for nm in Z.values():
        print(f"  {nm:>2} n={r[nm][3]:4d} {r[nm][0]:6.2f}/{r[nm][1]:6.2f}/{r[nm][2]:6.2f}")
print("\nstructs in test:", sorted(set(s.tolist()))[:40])
