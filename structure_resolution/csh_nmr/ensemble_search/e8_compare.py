"""Adam against the zoom, same seeds, same 3 Direct folds, same loss. Offline.

zoom = coarse grid + zoom points (the sweep's candidates); adam = points visited from the seeds, inside
the soft guards. The true-site start is a control and is never a candidate.
"""
import glob, os, sys, numpy as np
from math import comb
from d1_analyse import load, mic, chi2, ok, rank_of_true
from d3_minima import minima

SP = {"NaCaSi": (11, 20, 14), "NaCaSiO": (11, 20, 14, 8)}
TH = (0.2, 0.3, 0.5, 1.0)


def sign_p(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def counts(dd, top3=None):
    s = "  ".join(f"<{t}: {np.sum(dd < t):2d}" for t in TH) + f"  med {np.median(dd):.2f}"
    return s + (f"  top3 {top3}" if top3 is not None else "")


names = sorted(os.path.basename(f) for f in glob.glob("sweep_direct/cell_*.npz"))
S = {n: load(f"sweep_direct/{n}") for n in names}
diff = {n: load(f"sweep_deriv1st/{n}") for n in names}
out = {}
for spec, sp in SP.items():
    rows = []
    for n in names:
        s, d = S[n], np.load(f"descent_{spec}/{n}")
        ns = int(d["n_seeds"]); cid = int(d["cid"]); over = np.load(f"descent_{spec}/over_{cid}.npy")
        c = np.where((d["wall"] == 0) & ~over, d["chi2"], np.inf)
        sz = np.where(ok(s["cand_dist"], "soft"), chi2(s["cand_iso"].mean(0), s["dft"], s["z"], sp), np.inf)
        iz = int(np.argmin(sz))
        pa, ca = d["pos"][:ns].reshape(-1, 3), c[:ns].reshape(-1)
        ia = int(np.argmin(ca))
        pu, cu = np.vstack([s["cand_pts"], pa]), np.concatenate([sz, ca])
        iu = int(np.argmin(cu))
        M = minima(pa, ca, s["cell"]); Mz = minima(s["cand_pts"], sz, s["cell"])
        # per-seed final landing points: how many distinct basins do 10 descents reach?
        jt = int(np.argmin(c[-1]))
        sd = np.where(ok(diff[n]["cand_dist"], "soft"), chi2(diff[n]["cand_iso"].mean(0), diff[n]["dft"], diff[n]["z"], sp), np.inf)
        xd = diff[n]["cand_pts"][int(np.argmin(sd))]
        rows.append(dict(cid=cid, dz=mic(s["cand_pts"][iz], s["true"], s["cell"]), da=mic(pa[ia], s["true"], s["cell"]), du=mic(pu[iu], s["true"], s["cell"]),
                         cz=sz[iz], ca=ca[ia], cu=cu[iu], src="adam" if iu >= len(sz) else "zoom",
                         rz=rank_of_true(s["cand_pts"], sz, s["true"], s["cell"])[0] or 99, ra=rank_of_true(pa, ca, s["true"], s["cell"])[0] or 99,
                         ct0=d["chi2"][-1, 0], ct=c[-1, jt], off=mic(d["pos"][-1, jt], s["true"], s["cell"]),
                         gapa=M[1][1] - M[0][1] if len(M) > 1 else np.inf, gapz=Mz[1][1] - Mz[0][1] if len(Mz) > 1 else np.inf,
                         agree=mic(pa[ia], xd, s["cell"]), walled=np.mean(d["wall"][:ns, -1] > 0),
                         move=np.median(np.linalg.norm(d["pos"][:ns, -1] - d["pos"][:ns, 0], axis=-1)), tail=np.ptp(d["chi2"][:ns, -10:], axis=1).max(),
                         ibest=np.unravel_index(np.argmin(c[:ns]), c[:ns].shape)[1]))
    R = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    out[spec] = R
    print(f"\n===== {spec}, soft guards, DFT target, ensemble of 3 folds, {len(rows)} cells")
    print("zoom           ", counts(R["dz"], int(np.sum(R["rz"] <= 3))))
    print("adam           ", counts(R["da"], int(np.sum(R["ra"] <= 3))))
    print("best of both   ", counts(R["du"]), f"  answer comes from adam in {np.sum(R['src']=='adam')}/30")
    print("truth start    ", counts(R["off"]), "   <- the model's own minimum near the true site, oracle control")
    for t in TH:
        b, c = int(np.sum((R["dz"] < t) & ~(R["da"] < t))), int(np.sum(~(R["dz"] < t) & (R["da"] < t)))
        print(f"  discordant <{t}: zoom only {b}, adam only {c}, p={sign_p(b, c):.2f}")
    print(f"chi2 adam < zoom in {np.sum(R['ca'] < R['cz'] - 1e-3)}/30, median drop {np.median(R['cz'] - R['ca']):.2f}; median |dr| change {np.median(R['da'] - R['dz']):+.2f}")
    print(f"moved median {np.median(R['move']):.2f} A, last-10 chi2 range max {R['tail'].max():.2f}, final points on the wall {R['walled'].mean():.2f}, best step median {np.median(R['ibest']):.0f}")
    right = R["da"] < 1.0
    print(f"gap adam: right {np.median(R['gapa'][right]):.1f}, wrong {np.median(R['gapa'][~right]):.1f} (n wrong {np.sum(~right)})")
    for g in (2, 4, 7):
        m = R["gapa"] >= g
        print(f"   gap>={g}: answered {m.sum()}, right {np.sum(m & right)}")
    m = R["agree"] < 1.0
    print(f"agree with Diff1st's best (<1 A): {m.sum()} cells, right {np.sum(m & right)}; disagree {(~m).sum()}, right {np.sum(~m & right)}")
    # is a wrong answer a search failure or the model's? the truth-start minimum needs the oracle, so validation only
    wrong = ~right
    print(f"wrong answers: {wrong.sum()}; truth-basin chi2 lower than the answer (search failure) in {np.sum(wrong & (R['ct'] < R['ca'] - 1e-3))}, higher (model-limited) in {np.sum(wrong & (R['ct'] >= R['ca'] - 1e-3))}")
    print("  cell  dz    da    chi2zoom adam truth@polish  off   gap   ")
    for i in np.argsort(-R["da"]):
        if R["da"][i] >= 0.5:
            print(f"  {R['cid'][i]:4d} {R['dz'][i]:5.2f} {R['da'][i]:5.2f} {R['cz'][i]:6.1f} {R['ca'][i]:6.1f} {R['ct'][i]:6.1f} {R['off'][i]:6.2f} {R['gapa'][i]:5.1f}")
np.savez("e8_results.npz", **{f"{s}_{k}": v for s, R in out.items() for k, v in R.items()})
