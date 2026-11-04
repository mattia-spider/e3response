"""How sigma_iso responds to the Na position, from the 15 saved points around the true site. No DFT, no GPU.

J = d sigma / d x_Na, fitted on the symmetric stencil (+-0.2 A), so even terms do not enter.
1. Do folds and models agree on J?  The only available proxy for the off-equilibrium response.
2. Does the error at the true site explain where the local chi2 minimum sits?
   Linear prediction: dx = -(J^T W J)^-1 J^T W r, with r the sorted residual at the true site.
3. Cramer-Rao precision from J and epsilon, per nucleus, and the accuracy a target precision needs.
"""
import glob, os, numpy as np

EPS = {11: 2.5, 20: 2.7, 14: 1.5, 8: 3.8, 1: 0.55}
SETS = {"Na+Ca+Si": (11, 20, 14), "Na+Ca+Si+O": (11, 20, 14, 8)}
SPEC = {"Na+Ca+Si": "NaCaSi", "Na+Ca+Si+O": "NaCaSiO"}
NAME = {11: "Na", 20: "Ca", 14: "Si", 8: "O", 1: "H"}


def fit_J(d, fold=None):
    """(n_atoms, 3) per-atom gradient of sigma_iso at the true site, ppm/A; fold None = mean of folds 0-2."""
    m = d["zoom_tag"] == 11
    P = d["zoom_pts"][m] - d["true"]
    iso = d["zoom_iso"][:, m]
    y = iso[:3].mean(0) if fold is None else iso[fold]
    y = y - y[0]
    A = P[1:]
    return np.linalg.lstsq(A, y[1:], rcond=None)[0].T


def sorted_residual(iso, dft, z, sp):
    """Residual of each atom against the target peak of its rank, the matching the chi2 uses."""
    r = np.full(len(iso), np.nan)
    for s in sp:
        i = np.where(z == s)[0]
        o = i[np.argsort(iso[i])]
        r[o] = iso[o] - np.sort(dft[i])
    return r


names = sorted(os.path.basename(f) for f in glob.glob("sweep_direct/cell_*.npz"))
A = [dict(np.load(f"sweep_direct/{n}")) for n in names]
B = [dict(np.load(f"sweep_deriv1st/{n}")) for n in names]

print("=== 1. agreement on J, median over 30 cells of the correlation between flattened J, per nucleus")
print("nucleus  |J| ppm/A   fold-fold Direct  fold-fold Diff1st  Direct-Diff1st  (ens 5 v ens 5)")
pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
for s in (11, 20, 14, 8, 1):
    size, fa, fb, ab = [], [], [], []
    for a, b in zip(A, B):
        m = a["z"] == s
        Ja = [fit_J(a, f)[m].ravel() for f in range(5)]; Jb = [fit_J(b, f)[m].ravel() for f in range(5)]
        ea, eb = np.mean(Ja, 0), np.mean(Jb, 0)
        size.append(np.median(np.linalg.norm(ea.reshape(-1, 3), axis=1)))
        fa.append(np.mean([np.corrcoef(Ja[i], Ja[j])[0, 1] for i, j in pairs]))
        fb.append(np.mean([np.corrcoef(Jb[i], Jb[j])[0, 1] for i, j in pairs]))
        ab.append(np.corrcoef(ea, eb)[0, 1])
    print(f"{NAME[s]:7s} {np.median(size):8.2f} {np.median(fa):15.2f} {np.median(fb):17.2f} {np.median(ab):15.2f}")

print("\nsame, on the nearest atoms only (largest |J|, the ones that carry the signal): top 5 atoms of each cell, Ca+Si+O")
fa, fb, ab = [], [], []
for a, b in zip(A, B):
    Ja = np.array([fit_J(a, f) for f in range(5)]); Jb = np.array([fit_J(b, f) for f in range(5)])
    m = np.where(np.isin(a["z"], (20, 14, 8)))[0]
    top = m[np.argsort(-np.linalg.norm(Ja.mean(0)[m], axis=1))[:5]]
    va = Ja[:, top].reshape(5, -1); vb = Jb[:, top].reshape(5, -1)
    fa.append(np.mean([np.corrcoef(va[i], va[j])[0, 1] for i, j in pairs])); fb.append(np.mean([np.corrcoef(vb[i], vb[j])[0, 1] for i, j in pairs]))
    ab.append(np.corrcoef(va.mean(0), vb.mean(0))[0, 1])
print(f"  fold-fold Direct {np.median(fa):.2f}  fold-fold Diff1st {np.median(fb):.2f}  Direct-Diff1st {np.median(ab):.2f}  (worst cell {np.min(ab):.2f})")

for nm, sp in SETS.items():
    print(f"\n=== 2-3. {nm}, Direct, ensemble of 3 folds")
    obs, pred, cosv, cr, frac = [], [], [], [], {s: [] for s in sp}
    rms = {s: [] for s in sp}
    for n, a in zip(names, A):
        d = np.load(f"descent_{SPEC[nm]}/{n}")
        c = np.where(d["wall"] == 0, d["chi2"], np.inf)[-1]
        j = int(np.argmin(c))
        cell = a["cell"]; ic = np.linalg.inv(cell)
        dx = (d["pos"][-1, j] - a["true"]) @ ic; dx = (dx - np.round(dx)) @ cell
        J = fit_J(a); iso = a["iso_true"][:3].mean(0)[0]
        r = sorted_residual(iso, a["dft"], a["z"], sp)
        use = np.isfinite(r)
        w = np.array([1 / EPS[z] ** 2 for z in a["z"][use]])
        F = (J[use].T * w) @ J[use]
        px = -np.linalg.solve(F, (J[use].T * w) @ r[use])
        obs.append(dx); pred.append(px)
        cosv.append(dx @ px / (np.linalg.norm(dx) * np.linalg.norm(px)))
        cr.append(np.sqrt(np.trace(np.linalg.inv(F))))
        for s in sp:
            m = a["z"][use] == s
            frac[s].append(np.trace((J[use][m].T * w[m]) @ J[use][m]) / np.trace(F))
            rms[s].append(np.sqrt(np.mean(r[use][m] ** 2)))
    obs, pred = np.array(obs), np.array(pred)
    no, np_ = np.linalg.norm(obs, axis=1), np.linalg.norm(pred, axis=1)
    print(f"distance of the local minimum from truth: observed median {np.median(no):.2f}, linear prediction median {np.median(np_):.2f}")
    print(f"prediction vs observed: cosine median {np.median(cosv):.2f}, >0.7 in {np.sum(np.array(cosv) > 0.7)}/30; correlation of |dx| {np.corrcoef(no, np_)[0,1]:.2f}; vector correlation {np.corrcoef(obs.ravel(), pred.ravel())[0,1]:.2f}")
    print(f"mean residual |obs-pred|: {np.median(np.linalg.norm(obs - pred, axis=1)):.2f} A")
    cr = np.array(cr)
    print(f"Cramer-Rao rms from J and epsilon: median {np.median(cr):.2f} A, 10-90% {np.percentile(cr,10):.2f}-{np.percentile(cr,90):.2f}")
    print("share of the position information / sorted residual rms at truth (ppm):", {NAME[s]: (round(float(np.median(frac[s])), 2), round(float(np.median(rms[s])), 2)) for s in sp})
    for t in (0.2, 0.1, 0.05):
        print(f"  to reach {t} A rms, errors must scale by {t/np.median(cr):.2f}: " + ", ".join(f"{NAME[s]} {EPS[s]*t/np.median(cr):.1f}" for s in sp) + " ppm")
