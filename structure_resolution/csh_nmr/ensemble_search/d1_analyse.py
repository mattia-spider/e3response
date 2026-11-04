"""Offline: every loss / nucleus set / ensemble / guard tier on the saved sweeps. No GPU.

Each cell: candidates = coarse grid + zoom around the seeds. Truth is used only to score the
answer; the zoom around the truth (tags 11, 12) is never a candidate.
"""
import glob, sys, numpy as np
Z = [1, 8, 11, 14, 20]
DMIN = np.array([2.18, 2.18, 3.08, 2.78, 2.95])  # order of Z
TIER = {"loose": 0.8, "soft": 0.9, "tight": 1.0}
EPS = {11: 2.5, 20: 2.7, 14: 1.5, 8: 3.8, 1: 0.55}
SITE = (0.5, 1.0)


def ok(dist, t):
    if TIER[t] is None:
        return np.ones(len(dist), bool)
    return np.all(dist >= DMIN * TIER[t] - (0.1 if t == "tight" else 0), 1)


def chi2(iso, tgt, z, sp):
    s = 0.0
    for zz in sp:
        m = z == zz
        if m.any():
            s = s + np.sum((np.sort(iso[:, m], 1) - np.sort(tgt[m])) ** 2, 1) / EPS[zz] ** 2
    return s


def w1(iso, tgt, z, sp):
    """The branch's Wasserstein, without broadening: mean over nuclei of mean |sorted diff|."""
    s = 0.0; n = 0
    for zz in sp:
        m = z == zz
        if m.any():
            s = s + np.mean(np.abs(np.sort(iso[:, m], 1) - np.sort(tgt[m])), 1); n += 1
    return s / n


def mic(p, q, cell):
    ic = np.linalg.inv(cell); d = (p - q) @ ic; d -= np.round(d)
    return np.linalg.norm(d @ cell, axis=-1)


def load(f):
    d = dict(np.load(f))
    m = d["zoom_tag"] < 10
    d["cand_pts"] = np.vstack([d["pts"], d["zoom_pts"][m]])
    d["cand_dist"] = np.vstack([d["dist"], d["zoom_dist"][m]])
    # coarse only has folds 0-2, so the ensemble over folds 0-2 is the one comparable everywhere
    d["cand_iso"] = np.concatenate([d["coarse"][:3], d["zoom_iso"][:3][:, m]], 1)
    d["z_iso5"] = d["zoom_iso"][:, m]; d["z_pts"] = d["zoom_pts"][m]; d["z_dist"] = d["zoom_dist"][m]
    d["cand_n_coarse"] = len(d["pts"])
    return d


def rank_of_true(pts, score, true, cell, sep=1.0):
    """Rank of the true site among well-separated minima."""
    order = np.argsort(score); centres = []
    for i in order:
        if not np.isfinite(score[i]):
            break
        if all(mic(pts[i], c, cell) > sep for c in centres):
            centres.append(pts[i])
            if mic(pts[i], true, cell) <= sep:
                return len(centres), len(centres)
    return None, len(centres)


def evaluate(D, fn):
    """fn(d) -> (pts, score) with inf for excluded points."""
    r = []
    for d in D:
        pts, s = fn(d)
        i = int(np.argmin(s))
        if not np.isfinite(s[i]):
            r.append((np.inf, None)); continue
        r.append((float(mic(pts[i], d["true"], d["cell"])), rank_of_true(pts, s, d["true"], d["cell"])[0]))
    dd = np.array([x[0] for x in r]); rk = [x[1] for x in r]
    return dd, rk


def line(name, D, fn):
    dd, rk = evaluate(D, fn)
    ranks = [x if x else 99 for x in rk]
    print(f"  {name:48s} <0.5A {np.sum(dd<0.5):2d}/{len(dd)}  <1A {np.sum(dd<1):2d}/{len(dd)}  top3 {np.sum(np.array(ranks)<=3):2d}/{len(dd)}  med |dr| {np.median(dd):5.2f}")


def variant(tier, loss=chi2, sp=(11, 20, 14), fold=None, tgt="dft"):
    def fn(d):
        iso = d["cand_iso"]
        e = iso.mean(0) if fold is None else iso[fold]
        t = d["dft"] if tgt == "dft" else (d["iso_true"].mean(0)[0] if tgt == "self" else d["iso_true"][4, 0])
        s = loss(e, t, d["z"], sp)
        return d["cand_pts"], np.where(ok(d["cand_dist"], tier), s, np.inf)
    return fn


def coarse_only(tier, loss, sp=(11, 20, 14), tgt="dft"):
    """Same candidates for every loss: the coarse grid, no zoom, so the seeds cannot favour one."""
    def fn(d):
        n = d["cand_n_coarse"]
        t = d["dft"] if tgt == "dft" else d["iso_true"].mean(0)[0]
        s = loss(d["cand_iso"][:, :n].mean(0), t, d["z"], sp)
        return d["cand_pts"][:n], np.where(ok(d["cand_dist"][:n], tier), s, np.inf)
    return fn


def five_fold(tier, sp=(11, 20, 14)):
    """Zoomed candidates only, ensemble of all five folds."""
    def fn(d):
        s = chi2(d["z_iso5"].mean(0), d["dft"], d["z"], sp)
        return d["z_pts"], np.where(ok(d["z_dist"], tier), s, np.inf)
    return fn


def geometry_only(tier):
    """No spectrum: the point with the most room around it (relaxed cavity = the leak)."""
    def fn(d):
        room = (d["cand_dist"] / DMIN).min(1)
        return d["cand_pts"], np.where(ok(d["cand_dist"], tier), -room, np.inf)
    return fn


for model in (sys.argv[1:] if __name__ == "__main__" else []):
    D = [load(f) for f in sorted(glob.glob(f"sweep_{model}/cell_*.npz"))]
    print(f"\n===== {model}: {len(D)} cells, coarse {int(np.median([d['cand_n_coarse'] for d in D]))} pts =====")
    print("baseline (no NMR): most roomy point")
    for t in ("loose", "soft", "tight"):
        line(f"geometry only, {t}", D, geometry_only(t))
    print("DFT target, ensemble of 3 folds, chi2 Na+Ca+Si, guard tiers")
    for t in TIER:
        line(t, D, variant(t))
    print("DFT target, soft guards: loss")
    line("chi2", D, variant("soft"))
    line("W1 (the branch's loss, Na+Ca+Si)", D, variant("soft", w1))
    print("Same candidates for both losses (coarse grid, no zoom), soft guards")
    for nm, sp in (("Na+Ca", (11, 20)), ("Na+Ca+Si", (11, 20, 14))):
        line(f"chi2 {nm}", D, coarse_only("soft", chi2, sp))
        line(f"W1   {nm}", D, coarse_only("soft", w1, sp))
    print("DFT target, soft guards: ensemble")
    line("ensemble of 5, zoomed points", D, five_fold("soft"))
    for f in (0, 1, 2):
        line(f"single fold {f}", D, variant("soft", fold=f))
    print("DFT target, soft guards: nuclei (chi2)")
    for nm, sp in (("Na", (11,)), ("Na+Si", (11, 14)), ("Na+Ca", (11, 20)), ("Na+Ca+Si", (11, 20, 14)), ("Na+Ca+Si+O", (11, 20, 14, 8)), ("all", (11, 20, 14, 8, 1)), ("no Na: Ca+Si", (20, 14))):
        line(nm, D, variant("soft", sp=sp))
    print("Other targets, soft guards (chi2 Na+Ca+Si)")
    line("self: target = ensemble at truth", D, variant("soft", tgt="self"))
    line("held-out: target = fold 4 at truth", D, variant("soft", tgt="fold4"))
    print("per-cell, soft, chi2 Na+Ca+Si, DFT target")
    for d in D:
        t = variant("soft")(d); s = t[1]; i = int(np.argmin(s))
        e = d["iso_true"][:3].mean(0)
        l_true = chi2(e, d["dft"], d["z"], (11, 20, 14))[0]
        print(f"   cell {int(d['cid']):3d} |dr| {mic(t[0][i], d['true'], d['cell']):5.2f}  chi2 best {s[i]:6.1f} true {l_true:6.1f}  "
              f"Na peak dft {d['dft'][d['z']==11][0]:7.1f} ens@true {e[0][d['z']==11][0]:7.1f}  rank-true {rank_of_true(t[0], s, d['true'], d['cell'])}")
