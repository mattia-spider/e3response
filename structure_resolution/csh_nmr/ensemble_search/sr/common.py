"""Load the k-fold C-S-H checkpoints and evaluate many Na positions in a batch."""
import glob, os, re, sys, pathlib
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.dirname(__file__))
import hydra, jax, jax.numpy as jnp, numpy as np, omegaconf, reax
from tensorial import gcnn
from tensorial.gcnn import atomic
from e3response import keys
from e3response.data import split_indices, GroupedKFoldSplit
from e3response.data.csh_nmr import CshNmrDataset
import structure_search as ss

ROOT = "/home/mattia/Desktop/ml_codes/nmr_project/jz_runs/q1_kfold_v2/csh"
DATA = "/home/mattia/Desktop/ml_codes/CAMML/e3response/data/csh_nmr/Na_csh_filtered.json"
TAG = {"direct": "direct_r4_l3_mul32_n4_bs2", "deriv1st": "deriv1st_r4_l3_mul32_n4_bs2",
       "deriv2nd": "deriv2nd_muB_r4_l3_mul32_n4_bs2"}
R_MAX = 4.0


def best_ckpt(run):
    return max(glob.glob(run + "/checkpoints/epoch_*.ckpt"),
               key=lambda c: int(re.search(r"epoch_(\d+)", c).group(1)))


def load(model="deriv1st", fold=0):
    run = f"{ROOT}/{TAG[model]}_f{fold}"
    cfg = omegaconf.OmegaConf.load(run + "/config.yaml")
    module = hydra.utils.instantiate(cfg.model, _convert_="object")
    ck = reax.training.get_default_checkpointing().load(best_ckpt(run))
    module.set_parameters(ck["parameters"])
    params = jax.tree_util.tree_map(jnp.asarray, module.parameters())
    return module, params


def dataset():
    return CshNmrDataset(r_max=R_MAX, data_file=DATA)


def test_ids(ds):
    return split_indices(ds, GroupedKFoldSplit(fold=0, n_folds=5, test_fraction=0.1, seed=42), "test")


def to_graph(a):
    return gcnn.atomic.graph_from_ase(a, r_max=R_MAX, atom_include_keys=("numbers", "nmr_tensors"),
                                      global_include_keys=[keys.EXTERNAL_MAGNETIC_FIELD, atomic.TOTAL_ENERGY])


def make_scan(module, params, atoms, k, probes=None, capacity=None):
    """f(positions_of_k (m,3)) -> tensors (m, n_atoms, 3, 3), vmapped with a live neighbour list."""
    rebuild, cap = ss.make_neighbour_rebuilder(atoms, R_MAX, capacity, probe_positions=probes)
    base = to_graph(atoms)
    pos0 = jnp.asarray(atoms.positions, dtype=jnp.float32)
    n = len(atoms)

    def one(pk, p):
        pos = pos0.at[k].set(pk)
        g = gcnn.experimental.update_graph(base).set(("nodes", "positions"), pos).get()
        g = rebuild(g, pos)
        return module._model.apply(p, g).nodes["predicted_nmr_tensors"][:n]

    return one, cap
