"""Does a trained deriv2nd checkpoint predict sigma == 0 on atoms / linear / planar molecules?

Loads ``<run-dir>/config.yaml`` + a checkpoint and evaluates sigma on hand-made geometries:
an isolated C atom, HCN and C2H2 (linear), formaldehyde and benzene (planar, rotated so that the
normal is z) and a generic non-planar molecule.  Prints max|sigma| (ppm) and, relative to the
generic molecule, the whole tensor and the in-plane (xy) block of the planar ones.  An exact zero
(~1e-6 relative in float32) on the first three groups confirms the missing mu x B path.

    python scripts/check_sigma_zero.py --run-dir logs/train/runs/<run> [--ckpt path]
"""
import argparse
import glob
import os
import pathlib

import ase
import numpy as np


def geometries():
    ang = np.arange(6) * np.pi / 3
    return {
        "C atom": ([6], [[0, 0, 0]]),
        "HCN (linear)": ([1, 6, 7], [[0, 0, -1.065], [0, 0, 0], [0, 0, 1.156]]),
        "C2H2 (linear)": ([6, 6, 1, 1], [[0, 0, .601], [0, 0, -.601], [0, 0, 1.663], [0, 0, -1.663]]),
        "H2CO (planar)": ([6, 8, 1, 1], [[0, 0, 0], [0, 0, 1.205], [0, .943, -.587], [0, -.943, -.587]]),
        "C6H6 (planar)": ([6] * 6 + [1] * 6, [[1.397 * np.cos(a), 1.397 * np.sin(a), 0] for a in ang]
                          + [[2.481 * np.cos(a), 2.481 * np.sin(a), 0] for a in ang]),
        "ethanol (generic)": ([6, 6, 8, 1, 1, 1, 1, 1, 1], [[0, 0, 0], [1.52, 0, 0], [2.0, 1.3, .4],
                             [-.4, .95, -.35], [-.38, -.9, -.5], [-.3, .1, 1.05], [1.9, -.85, -.55],
                             [1.85, -.2, 1.0], [2.95, 1.25, .5]]),
    }


def to_normal_z(pos):
    p = np.asarray(pos, float)
    p = p - p.mean(0)
    return p @ np.linalg.svd(p)[2].T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run-dir", required=True, type=pathlib.Path)
    ap.add_argument("--ckpt", default=None)
    a = ap.parse_args()
    from tensorial.gcnn import atomic
    from tensorial.reaxkit import config as rconfig

    from e3response import keys

    ckpt = a.ckpt or sorted(glob.glob(str(a.run_dir / "checkpoints" / "*.ckpt")))[-1]
    module, cfg = rconfig.load_module(str(a.run_dir / "config.yaml"), ckpt, return_config=True)
    params, model = module.parameters(), module._model  # pylint: disable=protected-access
    r_max = float(cfg.data.r_max)
    zs = {int(z) for z in cfg.from_data.atomic_numbers}
    key = keys.predicted(keys.NMR_TENSORS)

    def sigma(numbers, pos):
        at = ase.Atoms(numbers=numbers, positions=np.asarray(pos, float))
        at.info[keys.EXTERNAL_MAGNETIC_FIELD] = np.zeros(3)
        at.info[atomic.TOTAL_ENERGY] = np.array(0.0)
        at.arrays["nmr_tensors"] = np.zeros((len(at), 3, 3))
        g = atomic.graph_from_ase(at, r_max=r_max, atom_include_keys=("numbers", "nmr_tensors"),
                                  global_include_keys=[keys.EXTERNAL_MAGNETIC_FIELD, atomic.TOTAL_ENERGY])
        return np.asarray(model.apply(params, g).nodes[key], float)

    geos = geometries()
    ref = np.abs(sigma(*geos["ethanol (generic)"])).max()
    print(f"checkpoint: {ckpt}\nspecies in the run: {sorted(zs)}\nreference max|sigma| (generic) = {ref:.4g} ppm\n")
    print(f"{'molecule':20s} {'max|sigma|':>12s} {'rel':>10s} {'xy block rel':>14s}")
    for name, (nb, pos) in geos.items():
        if not set(nb) <= zs:
            print(f"{name:20s} skipped (species not in the run)")
            continue
        if "planar" in name:
            pos = to_normal_z(pos)
        s = sigma(nb, pos)
        m = np.abs(s).max()
        xy = np.abs(s[:, :2, :2]).max() / ref if "planar" in name else float("nan")
        print(f"{name:20s} {m:12.4g} {m / ref:10.2e} {xy:14.2e}")


if __name__ == "__main__":
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    main()
