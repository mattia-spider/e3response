"""Shared harness for the deriv2nd verification (REPORT_verifica.md).

Builds the three NMR models (direct / deriv1st / deriv2nd, optionally the `_muB` variant)
straight from the YAML in ``configs/model`` with random weights, and builds graphs from
hand-made geometries.  Nothing here touches the existing configs or code.

Run scripts with:  JAX_PLATFORMS=cpu python tests/verifica/<script>.py
"""

import os
import pathlib

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import ase
import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

# float64 by default (structural zeros show up at ~1e-16); E3R_X64=0 for the training test
jax.config.update("jax_enable_x64", os.environ.get("E3R_X64", "1") == "1")

from tensorial import gcnn  # noqa: E402
from tensorial.gcnn import atomic  # noqa: E402

from e3response import keys  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "configs" / "model"
HERE = pathlib.Path(__file__).resolve().parent

MODELS = {
    "direct": MODEL_DIR / "nequip_nmr.yaml",
    "deriv1st": MODEL_DIR / "nequip_nmr_deriv1st.yaml",
    "deriv2nd": MODEL_DIR / "nequip_nmr_deriv2nd.yaml",
    "deriv2nd_muB": MODEL_DIR / "nequip_nmr_deriv2nd_muB.yaml",
}

# Species used throughout the tests (QM9: H C N O F)
ATOMIC_NUMBERS = [1, 6, 7, 8, 9]
NMR = keys.predicted(keys.NMR_TENSORS)


def build_model(
    name, mul=4, ell_max=2, num_layers=2, r_max=4.0, avg_nn=4.0, energy_only=False, **extra
):
    """Instantiate ``model.model`` (the Sequential) of config ``name`` with the given sizes.

    ``energy_only`` drops the trailing InducedMagneticField / MagneticShieldingTensor layers of
    the deriv2nd models, leaving a model that maps (mu, B) -> ``predicted_energy`` (used for the
    finite-difference check).
    """
    cfg = OmegaConf.load(MODELS[name])
    root = OmegaConf.create(
        {
            "r_max": r_max,
            "mul": mul,
            "ell_max": ell_max,
            "num_layers": num_layers,
            "from_data": {
                "n_elements": len(ATOMIC_NUMBERS),
                "atomic_numbers": ATOMIC_NUMBERS,
                "avg_num_neighbours": avg_nn,
                "energy_shifts": [0.0] * len(ATOMIC_NUMBERS),
                "steps_per_epoch": 1,
            },
            **extra,
            "model": cfg,
        }
    )
    OmegaConf.resolve(root)
    if energy_only:
        layers = list(root.model.model.layers)
        root.model.model.layers = layers[:-2]
    return hydra.utils.instantiate(root.model.model, _convert_="object"), root


def make_graph(numbers, positions, r_max=4.0):
    atoms = ase.Atoms(numbers=list(numbers), positions=np.asarray(positions, dtype=float))
    atoms.info[keys.EXTERNAL_MAGNETIC_FIELD] = np.zeros(3)
    atoms.info[atomic.TOTAL_ENERGY] = np.array(0.0)
    atoms.arrays["nmr_tensors"] = np.zeros((len(atoms), 3, 3))
    return gcnn.atomic.graph_from_ase(
        atoms,
        r_max=r_max,
        atom_include_keys=("numbers", "nmr_tensors"),
        global_include_keys=[keys.EXTERNAL_MAGNETIC_FIELD, atomic.TOTAL_ENERGY],
    )


def sigma(model, params, graph):
    """Predicted per-atom shielding tensors, shape (n, 3, 3)."""
    out = model.apply(params, graph)
    return np.asarray(out.nodes[NMR])


def init(model, graph, seed):
    return model.init(jax.random.PRNGKey(seed), graph)


def rotation(seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


# ---------------------------------------------------------------------------- geometries (Å)
def geometries():
    """name -> (numbers, positions, kind)"""
    g = {}
    g["C_atom"] = ([6], [[0, 0, 0]], "atom")
    g["HCN"] = ([1, 6, 7], [[0, 0, -1.065], [0, 0, 0], [0, 0, 1.156]], "linear")
    g["C2H2"] = (
        [6, 6, 1, 1],
        [[0, 0, 0.601], [0, 0, -0.601], [0, 0, 1.663], [0, 0, -1.663]],
        "linear",
    )
    g["CO2"] = ([6, 8, 8], [[0, 0, 0], [0, 0, 1.16], [0, 0, -1.16]], "linear")
    g["H2CO"] = (
        [6, 8, 1, 1],
        [[0, 0, 0], [0, 0, 1.205], [0, 0.943, -0.587], [0, -0.943, -0.587]],
        "planar",
    )  # plane = yz, normal = x ; rotated to z by `to_normal_z`
    ang = np.arange(6) * np.pi / 3
    cc, hh = 1.397, 2.481
    g["C6H6"] = (
        [6] * 6 + [1] * 6,
        [[cc * np.cos(a), cc * np.sin(a), 0] for a in ang]
        + [[hh * np.cos(a), hh * np.sin(a), 0] for a in ang],
        "planar",
    )
    g["CH3F_ethanol_like"] = (  # generic, non planar
        [6, 6, 8, 1, 1, 1, 1, 1, 1],
        [
            [0.0, 0.0, 0.0],
            [1.52, 0.0, 0.0],
            [2.0, 1.3, 0.4],
            [-0.4, 0.95, -0.35],
            [-0.38, -0.9, -0.5],
            [-0.3, 0.1, 1.05],
            [1.9, -0.85, -0.55],
            [1.85, -0.2, 1.0],
            [2.95, 1.25, 0.5],
        ],
        "generic",
    )
    return g


def to_normal_z(pos):
    """Rotate a planar structure so that its plane normal is z."""
    p = np.asarray(pos, float)
    p = p - p.mean(0)
    _, _, vt = np.linalg.svd(p)
    return p @ vt.T  # rows of vt are principal axes, last = normal


# ---------------------------------------------------------------------------- dataset sample
def sample_graphs(n, max_atoms=12, pool=400, seed=0, r_max=4.0, require_all_species=True):
    """n small QM9-NMR graphs (<= max_atoms) drawn from a random pool; tries to include every
    species of ATOMIC_NUMBERS (F is rare)."""
    from e3response.data.qm9_nmr import Qm9NmrDataset

    ds = Qm9NmrDataset(r_max=r_max, data_dir=str(ROOT / "data/qm9_nmr"), limit=f"random:{pool}:{seed}")
    small = [ds[i] for i in range(len(ds)) if len(ds[i].nodes["positions"]) <= max_atoms]
    rng = np.random.default_rng(seed)
    rng.shuffle(small)
    chosen, seen = [], set()
    if require_all_species:
        for g in small:  # greedy: first molecules that bring a missing species
            zs = set(np.asarray(g.nodes["atomic_numbers"])[:, 0].tolist())
            if zs - seen and len(chosen) < n:
                chosen.append(g); seen |= zs
    for g in small:
        if len(chosen) >= n:
            break
        if all(g is not c for c in chosen):
            chosen.append(g)
    return chosen
