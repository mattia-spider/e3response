"""Per-atom error analysis of a trained NMR checkpoint (task C of REPORT_verifica.md).

Given a run directory (``config.yaml`` + ``checkpoints/*.ckpt``) it re-creates the held-out split
exactly as ``Qm9NmrDataModule.load_split`` does, predicts the shielding tensors atom by atom and
writes one CSV row per atom with

* the error split in the irreps of the 3x3 tensor: 0e (trace/3 * I), 1e (antisymmetric part),
  2e (symmetric traceless part), as Frobenius norms of the difference, plus the norm of the target
  and of the prediction of each part (a model that ignores the 1e part has |pred_1e| ~ 0);
* molecule descriptors: number of atoms / heavy atoms, ``planarity`` = smallest eigenvalue of the
  gyration tensor of the heavy atoms [A^2] (NaN below 3 heavy atoms), the plane normal, the
  number of neighbours within ``r_max``;
* for molecules flagged planar (``planarity < --planar-tol``): the error restricted to the plane
  (P D P with P = 1 - n n^T) and out of the plane (the rest).

A second file ``<out>_summary.csv`` aggregates the RMSE per element and irrep, per planarity bin,
per neighbour-count bin and, for planar molecules, in plane vs out of plane.

Works on CPU and on a GPU node (Jean Zay): only JAX, tensorial and e3response are needed. Example::

    python scripts/analyze_ckpt.py --run-dir logs/train/runs/2026-09-23_13-11-37 \
        --data-dir data/qm9_nmr --split test --limit 2000 --out analysis/run_X.csv
"""

import argparse
import glob
import logging
import os
import pathlib

import numpy as np

_LOGGER = logging.getLogger("analyze_ckpt")
SYM = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 14: "Si", 16: "S", 17: "Cl"}


def decompose(t: np.ndarray):
    """(n,3,3) -> trace part (0e), antisymmetric part (1e), symmetric traceless part (2e)."""
    iso = np.trace(t, axis1=1, axis2=2)[:, None, None] / 3 * np.eye(3)
    asym = 0.5 * (t - t.transpose(0, 2, 1))
    sym0 = 0.5 * (t + t.transpose(0, 2, 1)) - iso
    return iso, asym, sym0


def fro(x):
    return np.sqrt((x**2).sum(axis=(-2, -1)))


def gyration(pos: np.ndarray):
    c = pos - pos.mean(0)
    w, v = np.linalg.eigh(c.T @ c / len(pos))
    return w, v  # ascending; v[:, 0] = plane normal


def find_ckpt(run_dir: pathlib.Path, ckpt: str | None) -> str:
    if ckpt:
        return ckpt
    cands = sorted(glob.glob(str(run_dir / "checkpoints" / "*.ckpt")))
    best = [c for c in cands if "last" not in os.path.basename(c)]
    if not (best or cands):
        raise FileNotFoundError(f"no checkpoint in {run_dir}/checkpoints")
    return (best or cands)[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run-dir", required=True, type=pathlib.Path)
    ap.add_argument("--ckpt", default=None, help="checkpoint file (default: last non-'last' one)")
    ap.add_argument("--data-dir", default=None, help="override data.data_dir of the run config")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--limit", default=None, help="int / 'a:b' / 'random:N[:seed]' within the split")
    ap.add_argument("--planar-tol", type=float, default=0.01, help="A^2, gyration eigenvalue")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import hydra
    import jax
    import omegaconf
    import pandas as pd
    from tensorial.reaxkit import config as rconfig

    from e3response import keys

    limit = args.limit
    if limit is not None and limit.lstrip("-").isdigit():
        limit = int(limit)

    cfg_path = args.run_dir / "config.yaml"
    ckpt = find_ckpt(args.run_dir, args.ckpt)
    module, cfg = rconfig.load_module(str(cfg_path), ckpt, return_config=True)
    params, model = module.parameters(), module._model  # pylint: disable=protected-access
    apply = jax.jit(model.apply)

    data_cfg = omegaconf.OmegaConf.to_container(cfg.data, resolve=True)
    if args.data_dir:
        data_cfg["data_dir"] = args.data_dir
    dm = hydra.utils.instantiate(data_cfg, _convert_="object")
    if not hasattr(dm, "load_split"):
        raise SystemExit(f"{type(dm).__name__} has no load_split(); only Qm9NmrDataModule is supported")
    ds = dm.load_split(args.split, limit=limit)
    r_max = float(cfg.data.r_max)
    pred_key = keys.predicted(keys.NMR_TENSORS)
    _LOGGER.info("run=%s ckpt=%s split=%s n_mol=%d", args.run_dir.name, ckpt, args.split, len(ds))

    rows = []
    for m in range(len(ds)):
        g = ds[m]
        pred = np.asarray(apply(params, g).nodes[pred_key], dtype=np.float64)
        tgt = np.asarray(g.nodes[keys.NMR_TENSORS], dtype=np.float64)
        z = np.asarray(g.nodes[keys.ATOMIC_NUMBERS])[:, 0].astype(int)
        pos = np.asarray(g.nodes[keys.POSITIONS], dtype=np.float64)
        n = len(z)
        nneigh = np.bincount(np.asarray(g.receivers), minlength=n)
        heavy = z > 1
        if heavy.sum() >= 3:
            w, v = gyration(pos[heavy])
            planarity, normal = float(w[0]), v[:, 0]
        else:
            planarity, normal = float("nan"), np.array([0.0, 0.0, 1.0])
        planar = bool(planarity < args.planar_tol)

        parts_t, parts_p = decompose(tgt), decompose(pred)
        diff = pred - tgt
        P = np.eye(3) - np.outer(normal, normal)
        d_in = np.einsum("ij,njk,kl->nil", P, diff, P)
        d_out = diff - d_in
        for a in range(n):
            row = dict(
                mol=m, atom=a, Z=int(z[a]), symbol=SYM.get(int(z[a]), str(z[a])), natoms=n,
                nheavy=int(heavy.sum()), n_neigh=int(nneigh[a]), planarity=planarity,
                is_planar=planar, err_total=float(fro(diff[a])), tgt_total=float(fro(tgt[a])),
                nn_err=float(normal @ diff[a] @ normal),
                err_inplane=float(fro(d_in[a])), err_outplane=float(fro(d_out[a])),
            )
            for name, pt, pp in zip(("0e", "1e", "2e"), parts_t, parts_p):
                row[f"err_{name}"] = float(fro(pp[a] - pt[a]))
                row[f"tgt_{name}"] = float(fro(pt[a]))
                row[f"pred_{name}"] = float(fro(pp[a]))
            rows.append(row)
        if (m + 1) % 50 == 0:
            _LOGGER.info("%d / %d molecules", m + 1, len(ds))

    df = pd.DataFrame(rows)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    def rmse(x):
        return float(np.sqrt(np.mean(np.square(x))))

    cols = ["err_total", "err_0e", "err_1e", "err_2e", "tgt_1e", "pred_1e"]
    summ = []
    for by, label in (("symbol", "element"),):
        for k, sub in df.groupby(by):
            summ.append(dict(group=label, key=k, n=len(sub), **{c: rmse(sub[c]) for c in cols}))
    df["planarity_bin"] = pd.cut(df.planarity, [-1e-9, 0.01, 0.05, 0.2, 0.5, 1e9])
    df["neigh_bin"] = pd.cut(df.n_neigh, [-1, 2, 4, 8, 16, 1e9])
    for by in ("planarity_bin", "neigh_bin"):
        for k, sub in df.groupby(by, observed=True):
            summ.append(dict(group=by, key=str(k), n=len(sub), **{c: rmse(sub[c]) for c in cols}))
    pl = df[df.is_planar]
    for sym, sub in list(pl.groupby("symbol")) + [("all", pl)]:
        if len(sub):
            summ.append(dict(group="planar_in_vs_out", key=sym, n=len(sub),
                             err_inplane=rmse(sub.err_inplane), err_outplane=rmse(sub.err_outplane),
                             nn_err=rmse(sub.nn_err)))
    spath = out.with_name(out.stem + "_summary.csv")
    pd.DataFrame(summ).to_csv(spath, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(pd.DataFrame(summ).to_string(index=False))
    print(f"wrote {out} and {spath}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
