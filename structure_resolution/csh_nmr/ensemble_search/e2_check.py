"""First look at the descents: how far they move, and how they score against the zoom on the same cells."""
import glob, os, sys, numpy as np
from d1_analyse import load, mic, chi2, ok
SPEC = sys.argv[1]; SP = {"NaCaSi": (11, 20, 14), "NaCaSiO": (11, 20, 14, 8)}[SPEC]
files = sorted(glob.glob(f"descent_{SPEC}/cell_*.npz"))
for f in files:
    d = np.load(f); n = int(d["n_seeds"]); c = np.where(d["wall"] == 0, d["chi2"], np.inf)
    s = load(f"sweep_direct/{os.path.basename(f)}")
    sc = np.where(ok(s["cand_dist"], "soft"), chi2(s["cand_iso"].mean(0), s["dft"], s["z"], SP), np.inf)
    iz = int(np.argmin(sc)); dz = mic(s["cand_pts"][iz], s["true"], s["cell"])
    i, j = np.unravel_index(np.argmin(c[:n]), c[:n].shape)
    dd = mic(d["pos"][i, j], d["true"], d["cell"])
    move = np.linalg.norm(d["pos"][:n, -1] - d["pos"][:n, 0], axis=-1)
    jt = int(np.argmin(c[-1])); dt = mic(d["pos"][-1, jt], d["true"], d["cell"])
    print(f"cell {int(d['cid']):3d}: zoom chi2 {sc[iz]:5.1f} |dr| {dz:.2f}  adam chi2 {c[:n].min():5.1f} |dr| {dd:.2f} step {j:3d}  move med {np.median(move):.2f}  last-10 chi2 range {np.ptp(d['chi2'][:n,-10:],axis=1).max():.2f}  truth start: chi2 {c[-1].min():.1f} offset {dt:.2f}")
