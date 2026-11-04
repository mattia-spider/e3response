"""c1_sweep picked its last 2 seeds with the self target (the ensemble at the TRUE site): oracle information.
Rebuild the seed list, check it matches the saved zoom, and rescore without those seeds' zoom points and descents."""
import glob, os, numpy as np
from d1_analyse import load, mic, chi2 as chi2_eval, ok, rank_of_true

DMIN = {8: 2.18, 1: 2.18, 14: 2.78, 20: 2.95, 11: 3.08}
Z = [1, 8, 11, 14, 20]
TIER = {"tight": 1.0, "soft": 0.9, "loose": 0.8}
EPS3 = {11: 2.5, 20: 2.7, 14: 1.5}


def allowed(dist, tier):
    return np.all(dist >= np.array([DMIN[z] * TIER[tier] for z in Z]) - (0.1 if tier == "tight" else 0.0), axis=1)


def chi2_seed(iso, tgt, z, inflate=3.0):
    s = 0.0
    for zz, e in EPS3.items():
        m = z == zz
        if m.any():
            s = s + np.sum((np.sort(iso[..., m], -1) - np.sort(tgt[m])) ** 2, -1) / (e * (inflate if zz == 11 else 1)) ** 2
    return s


def seeds_from(pts, score, okm, n, sep=0.8, taken=()):
    out = list(taken); added = 0
    for i in np.argsort(np.where(okm, score, np.inf)):
        if not okm[i] or added >= n:
            break
        if all(np.linalg.norm(pts[i] - p) > sep for p in out):
            out.append(pts[i]); added += 1
    return out


def seed_info(raw):
    ec = np.nanmean(raw["coarse"], 0)
    sd = chi2_seed(ec, raw["dft"], raw["z"]); ss = chi2_seed(ec, raw["iso_true"].mean(0)[0], raw["z"])
    s = seeds_from(raw["pts"], sd, allowed(raw["dist"], "tight"), 3)
    s = seeds_from(raw["pts"], sd, allowed(raw["dist"], "soft"), 3, taken=s)
    s = seeds_from(raw["pts"], sd, allowed(raw["dist"], "loose"), 2, taken=s)
    n_dft = len(s)
    s = seeds_from(raw["pts"], ss, allowed(raw["dist"], "soft"), 2, taken=s)
    return np.array(s), n_dft


SP = {"Na+Ca+Si": (11, 20, 14), "Na+Ca+Si+O": (11, 20, 14, 8)}
names = sorted(os.path.basename(f) for f in glob.glob("sweep_direct/cell_*.npz"))
for model in ("direct", "deriv1st"):
    n_self, mism = [], 0
    rows = {k: ([], []) for k in SP}
    for n in names:
        raw = dict(np.load(f"sweep_{model}/{n}")); d = load(f"sweep_{model}/{n}")
        s, n_dft = seed_info(raw)
        saved = raw["zoom_pts"][raw["zoom_tag"] == 1][::15]
        mism += int(len(saved) != len(s) or not np.allclose(saved, s))
        n_self.append(len(s) - n_dft)
        # zoom points in seed order: 15 at level 1 and 14 at level 2 per seed, tags 1 and 2
        tag = raw["zoom_tag"]; seed_of = np.full(len(tag), -1)
        i1, i2 = np.where(tag == 1)[0], np.where(tag == 2)[0]
        seed_of[i1] = np.repeat(np.arange(len(s)), 15); seed_of[i2] = np.repeat(np.arange(len(s)), 14)
        keep_z = seed_of[tag < 10] < n_dft
        keep = np.concatenate([np.ones(d["cand_n_coarse"], bool), keep_z])
        for nm, sp in SP.items():
            sc = np.where(ok(d["cand_dist"], "soft"), chi2_eval(d["cand_iso"].mean(0), d["dft"], d["z"], sp), np.inf)
            for j, mask in enumerate((np.ones_like(keep), keep)):
                x = np.where(mask, sc, np.inf); i = int(np.argmin(x))
                rows[nm][j].append((mic(d["cand_pts"][i], d["true"], d["cell"]), rank_of_true(d["cand_pts"], x, d["true"], d["cell"])[0] or 99))
    print(f"\n{model}: seed lists rebuilt, mismatches with the saved zoom {mism}/30; oracle seeds per cell {np.bincount(n_self)}")
    for nm in SP:
        for j, lab in enumerate(("as reported", "without oracle seeds")):
            r = np.array(rows[nm][j])
            print(f"  zoom {nm:11s} {lab:21s} <0.5 {np.sum(r[:,0]<.5):2d}  <1 {np.sum(r[:,0]<1):2d}  top3 {np.sum(r[:,1]<=3):2d}  med {np.median(r[:,0]):.2f}")
        changed = [names[i][5:-4] for i in range(30) if abs(rows[nm][0][i][0] - rows[nm][1][i][0]) > 1e-6]
        print(f"     cells whose answer changes: {changed}")

print("\nAdam (Direct)")
for spec, nm in (("NaCaSi", "Na+Ca+Si"), ("NaCaSiO", "Na+Ca+Si+O")):
    for j, lab in enumerate(("as reported", "without oracle seeds")):
        res = []
        for n in names:
            raw = dict(np.load(f"sweep_direct/{n}")); s, n_dft = seed_info(raw)
            d = np.load(f"descent_{spec}/{n}"); ns = int(d["n_seeds"]); assert ns == len(s)
            over = np.load(f"descent_{spec}/over_{int(d['cid'])}.npy")
            c = np.where((d["wall"] == 0) & ~over, d["chi2"], np.inf)[: (ns if j == 0 else n_dft)]
            p = d["pos"][: c.shape[0]].reshape(-1, 3); cc = c.reshape(-1); i = int(np.argmin(cc))
            res.append((mic(p[i], d["true"], d["cell"]), rank_of_true(p, cc, d["true"], d["cell"])[0] or 99))
        r = np.array(res)
        print(f"  adam {nm:11s} {lab:21s} <0.5 {np.sum(r[:,0]<.5):2d}  <1 {np.sum(r[:,0]<1):2d}  top3 {np.sum(r[:,1]<=3):2d}  med {np.median(r[:,0]):.2f}")
