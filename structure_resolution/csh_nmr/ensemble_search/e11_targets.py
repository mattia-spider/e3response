"""How far do the results survive a more realistic target? Offline, on the saved sigma_iso. No GPU.

Reference offsets, noise on the peaks, other epsilons, broadened spectra. Broadened rows use the coarse
grid only, because the seeds were picked with the peak-list chi2.
"""
import glob, os, sys, numpy as np
from d1_analyse import load, mic, ok, rank_of_true

EPS = {11: 2.5, 20: 2.7, 14: 1.5, 8: 3.8, 1: 0.55}
EPS_DIRECT = {11: 4.35, 20: 3.25, 14: 1.61, 8: 4.06, 1: 0.65}
NCS, NCSO, NC, NS = (11, 20, 14), (11, 20, 14, 8), (11, 20), (11, 14)


def residuals(iso, tgt, z, s):
    m = z == s
    return np.sort(iso[:, m], 1) - np.sort(tgt[m])


def chi2(iso, tgt, z, sp, eps=EPS, free=()):
    out = 0.0
    for s in sp:
        if not (z == s).any():
            continue
        r = residuals(iso, tgt, z, s)
        if s in free:
            r = r - r.mean(1, keepdims=True)
        out = out + np.sum(r ** 2, 1) / eps[s] ** 2
    return out


def spectrum(iso, z, s, axis, fwhm):
    g = fwhm / 2.355
    v = iso[..., z == s]
    return np.exp(-0.5 * ((axis - v[..., None]) / g) ** 2).sum(-2) / (g * 2.5066)


def broad(iso, tgt, z, sp, width):
    """Normalised L2 between broadened spectra, each nucleus weighted by its number of sites."""
    out = 0.0
    for s in sp:
        m = z == s
        if not m.any():
            continue
        v = np.concatenate([iso[:, m].ravel(), tgt[m]])
        ax = np.arange(v.min() - 3 * width[s], v.max() + 3 * width[s], width[s] / 6)
        St = spectrum(tgt[None], z, s, ax, width[s])
        Sm = spectrum(iso, z, s, ax, width[s])
        out = out + m.sum() * np.sum((Sm - St) ** 2, 1) / np.sum(St ** 2)
    return out


def score(D, fn):
    dd, top = [], 0
    for d in D:
        p, s = fn(d)
        dd.append(mic(p[int(np.argmin(s))], d["true"], d["cell"]))
        r = rank_of_true(p, s, d["true"], d["cell"])[0]
        top += bool(r and r <= 3)
    dd = np.array(dd)
    return int(np.sum(dd < .5)), int(np.sum(dd < 1)), top, float(np.median(dd))


def full(loss):
    return lambda d: (d["cand_pts"], np.where(ok(d["cand_dist"], "soft"), loss(d["cand_iso"].mean(0), d), np.inf))


def coarse(loss):
    def fn(d):
        n = d["cand_n_coarse"]
        return d["cand_pts"][:n], np.where(ok(d["cand_dist"][:n], "soft"), loss(d["cand_iso"][:, :n].mean(0), d), np.inf)
    return fn


names = sorted(os.path.basename(f) for f in glob.glob("sweep_direct/cell_*.npz"))
for model in ("direct", "deriv1st"):
    D = [load(f"sweep_{model}/{n}") for n in names]
    def row(lab, fn):
        a, b, c, m = score(D, fn)
        print(f"  {lab:46s} <0.5 {a:2d}  <1 {b:2d}  top3 {c:2d}  med {m:.2f}")
    print(f"\n========== {model}")
    print("peak list, all candidates")
    for sp, nm in ((NCS, "Na+Ca+Si"), (NCSO, "Na+Ca+Si+O"), (NC, "Na+Ca"), (NS, "Na+Si")):
        row(nm, full(lambda i, d, sp=sp: chi2(i, d["dft"], d["z"], sp)))
    print("reference offset per nucleus, fitted freely (sigma to delta not calibrated)")
    for sp, nm in ((NCS, "Na+Ca+Si"), (NCSO, "Na+Ca+Si+O"), (NS, "Na+Si")):
        row(f"{nm}, free on all but Na", full(lambda i, d, sp=sp: chi2(i, d["dft"], d["z"], sp, free=tuple(x for x in sp if x != 11))))
        row(f"{nm}, free on all, Na peak lost", full(lambda i, d, sp=sp: chi2(i, d["dft"], d["z"], sp, free=sp)))
    print("wrong reference: every target peak of a nucleus shifted by b ppm, loss unaware")
    for b in (1, 2, 4):
        for sgn, lab in ((1, "+"), (-1, "-")):
            row(f"Na+Ca+Si+O, shift {lab}{b} ppm", full(lambda i, d: chi2(i, d["dft"] + sgn * b, d["z"], NCSO)))
    print("Gaussian noise on each target peak, mean of 20 draws")
    for sig in (0.5, 1, 2, 3):
        for sp, nm in ((NCS, "Na+Ca+Si"), (NCSO, "Na+Ca+Si+O")):
            rs = []
            for k in range(20):
                rng = np.random.default_rng(k)
                noise = [rng.normal(0, sig, len(d["dft"])) for d in D]
                nid = {id(d): n for d, n in zip(D, noise)}
                rs.append(score(D, full(lambda i, d: chi2(i, d["dft"] + nid[id(d)], d["z"], sp)))[:3])
            r = np.mean(rs, 0)
            print(f"  {nm:12s} noise {sig:3.1f} ppm    <0.5 {r[0]:4.1f}  <1 {r[1]:4.1f}  top3 {r[2]:4.1f}")
    print("epsilon: is the result tuned to the chosen weights?")
    row("Direct's own RMSE as epsilon, Na+Ca+Si", full(lambda i, d: chi2(i, d["dft"], d["z"], NCS, eps=EPS_DIRECT)))
    row("Direct's own RMSE as epsilon, Na+Ca+Si+O", full(lambda i, d: chi2(i, d["dft"], d["z"], NCSO, eps=EPS_DIRECT)))
    row("equal epsilon, Na+Ca+Si", full(lambda i, d: chi2(i, d["dft"], d["z"], NCS, eps={s: 1 for s in EPS})))
    row("equal epsilon, Na+Ca+Si+O", full(lambda i, d: chi2(i, d["dft"], d["z"], NCSO, eps={s: 1 for s in EPS})))
    for f in (0.5, 2):
        row(f"Na epsilon x{f}, Na+Ca+Si", full(lambda i, d: chi2(i, d["dft"], d["z"], NCS, eps={**EPS, 11: 2.5 * f})))
        row(f"O epsilon x{f}, Na+Ca+Si+O", full(lambda i, d: chi2(i, d["dft"], d["z"], NCSO, eps={**EPS, 8: 3.8 * f})))
    print("broadened spectra, coarse grid (peak list on the same candidates first)")
    for sp, nm in ((NS, "Na+Si"), (NC, "Na+Ca"), (NCS, "Na+Ca+Si"), (NCSO, "Na+Ca+Si+O")):
        row(f"{nm}, peak list", coarse(lambda i, d, sp=sp: chi2(i, d["dft"], d["z"], sp)))
        for lab, w in (("sharp", {11: 1, 20: 1, 14: 0.5, 8: 1}), ("medium", {11: 4, 20: 3, 14: 1.5, 8: 3}), ("broad", {11: 8, 20: 6, 14: 3, 8: 6})):
            row(f"{nm}, {lab} {w}", coarse(lambda i, d, sp=sp, w=w: broad(i, d["dft"], d["z"], sp, w)))
