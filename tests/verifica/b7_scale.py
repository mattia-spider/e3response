"""B7: std of the predicted sigma at initialisation vs std of the QM9-NMR target, per species.

Usage: JAX_PLATFORMS=cpu python tests/verifica/b7_scale.py [--n-target 300] [--n-model 10]
Target statistics come from a random sample of the gas-phase archive (needs data/qm9_nmr/*.zip).
"""
import argparse, json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import *  # noqa
from e3response.data.qm9_nmr import Qm9NmrDataset

SYM = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}


def stats(arrs, zs):
    out = {}
    for z in ATOMIC_NUMBERS:
        s = np.concatenate([a[m == z] for a, m in zip(arrs, zs)]) if arrs else np.zeros((0, 3, 3))
        if len(s) == 0:
            continue
        iso = np.trace(s, axis1=1, axis2=2) / 3
        out[SYM[z]] = dict(n=int(len(s)), rms=float(np.sqrt((s**2).mean())), std_comp=float(s.std()),
                           std_iso=float(iso.std()), mean_iso=float(iso.mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-target", type=int, default=300)
    ap.add_argument("--n-model", type=int, default=10)
    ap.add_argument("--ell", type=int, default=2)
    ap.add_argument("--mul", type=int, default=8)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default=str(HERE / "results_B7.json"))
    a = ap.parse_args()
    ds = Qm9NmrDataset(r_max=4.0, data_dir=str(ROOT / "data/qm9_nmr"), limit=f"random:{a.n_target}:0")
    graphs = [ds[i] for i in range(len(ds))]
    tz = [np.asarray(g.nodes["atomic_numbers"])[:, 0] for g in graphs]
    tt = [np.asarray(g.nodes["nmr_tensors"]) for g in graphs]
    res = {"target": stats(tt, tz), "model": {}}
    print("target", json.dumps(res["target"], indent=1))
    sub = graphs[: a.n_model]
    # models were built with the dataset-wide n_elements; atomic_numbers go through SpeciesTransform
    for name in ("direct", "deriv1st", "deriv2nd"):
        res["model"][name] = {}
        for seed in a.seeds:
            m, _ = build_model(name, mul=a.mul, ell_max=a.ell, num_layers=a.layers)
            p = init(m, sub[0], seed)
            preds = [np.asarray(m.apply(p, g).nodes[NMR]) for g in sub]
            st = stats(preds, tz[: a.n_model])
            res["model"][name][seed] = st
            print(name, seed, {k: round(v["rms"], 5) for k, v in st.items()}, flush=True)
    json.dump(res, open(a.out, "w"), indent=1)

main()
