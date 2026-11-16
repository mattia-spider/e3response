"""Position-derivative models (configs/model/*posderiv*): the Born charges and Raman tensors they predict are the
position derivatives of their own unsupervised dipole moment and polarizability, in the index order of the labels.

Built from the real configs (small mul), on a periodic BaTiO3 cell with displaced atoms, in float64, against
central finite differences of d and χ:

    Z*_{i,αγ} = ∂d_α/∂u_{iγ}  (field index first, as the Sobolev's ∂F_{iγ}/∂ε_α and the transposed BTO labels)
    R_{i,γαβ} = ∂χ_αβ/∂u_{iγ}  (displacement index first, as the Sobolev's -∂³E/∂u∂ε∂ε)
"""

import pathlib

import ase
import hydra
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import pytest
from tensorial import gcnn

from e3response import keys

CONFIG_PATH = pathlib.Path(__file__).parent.parent / "configs"
CASES = {
    "nequip_bto_posderiv_bornonly": ("bto_sobolev_vs_direct", "predicted_dipole_moment", "predicted_born_charges"),
    "nequip_bto_pol_posderiv_bornonly": ("bto_pol_sobolev_vs_direct", "predicted_dipole_moment",
                                         "predicted_born_charges"),
    "nequip_bto_posderiv_ramanonly": ("bto_sobolev_vs_direct", "predicted_polarizability", "predicted_raman_tensors"),
}
NUMBERS = [56, 22, 8, 8, 8]  # BaTiO3, one perovskite cell
FROM_DATA = {"n_elements": 3, "atomic_numbers": [8, 22, 56], "avg_num_neighbours": 20.0, "steps_per_epoch": 1,
             "energy_shifts": [0.0, 0.0, 0.0]}


def _model(name, experiment):
    with hydra.initialize_config_dir(version_base=None, config_dir=str(CONFIG_PATH)):
        cfg = hydra.compose(config_name="train", overrides=[f"experiment={experiment}", f"model={name}", "mul=4",
                                                            "num_layers=2", "r_max=4.5"])
    cfg.from_data = FROM_DATA
    for weight in [k for k in cfg if str(k).startswith("w_")]:
        cfg[weight] = 1.0  # the experiments set them from the data
    return hydra.utils.instantiate(cfg.model, _convert_="object")._model


def _graph(seed):
    rng = np.random.default_rng(seed)
    frac = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    cell = 4.0 * np.eye(3) + 0.05 * rng.normal(size=(3, 3))
    atoms = ase.Atoms(numbers=NUMBERS, positions=frac @ cell + 0.1 * rng.normal(size=(5, 3)), cell=cell, pbc=True)
    graph = gcnn.atomic.graph_from_ase(atoms, r_max=4.5, atom_include_keys=("numbers",))
    return jax.tree_util.tree_map(
        lambda x: jnp.asarray(x, jnp.float64) if np.issubdtype(np.asarray(x).dtype, np.floating) else jnp.asarray(x),
        graph)


@pytest.mark.parametrize("name", CASES)
def test_position_derivative_is_the_derivative_of_its_own_readout(name):
    experiment, latent_key, out_key = CASES[name]
    with jax.experimental.enable_x64():
        model = _model(name, experiment)
        graph = _graph(0)
        params = model.init(jax.random.PRNGKey(0), graph)
        # random weights well away from zero, so that every component of the readout depends on the positions
        params = jax.tree_util.tree_map(lambda p: jnp.asarray(p, jnp.float64), params)
        out = model.apply(params, graph)
        derivative = np.asarray(out.nodes[out_key])
        positions = graph.nodes[keys.POSITIONS]

        def latent(u):
            g = graph._replace(nodes={**graph.nodes, keys.POSITIONS: u})
            return np.asarray(model.apply(params, g).globals[latent_key])[0]

        h = 1e-5
        fd = np.zeros((len(NUMBERS), 3) + latent(positions).shape)  # (i, γ, latent...)
        for i in range(len(NUMBERS)):
            for gamma in range(3):
                du = jnp.zeros_like(positions).at[i, gamma].set(h)
                fd[i, gamma] = (latent(positions + du) - latent(positions - du)) / (2 * h)
        if out_key == "predicted_born_charges":
            expected = np.swapaxes(fd, 1, 2)  # Z*_{i,αγ} = ∂d_α/∂u_{iγ}
        else:
            expected = fd  # R_{i,γαβ} = ∂χ_αβ/∂u_{iγ}
        assert np.abs(expected).max() > 1e-3, "readout independent of the positions: test says nothing"
        np.testing.assert_allclose(derivative, expected, rtol=1e-6, atol=1e-8 * np.abs(expected).max())
        # translation invariance gives the acoustic sum rule, Σ_i Z*_i = 0 (and Σ_i R_i = 0)
        np.testing.assert_allclose(derivative.sum(0), 0.0, atol=1e-10 * np.abs(derivative).max())
        if out_key == "predicted_raman_tensors":  # χ symmetric, so R symmetric in αβ
            np.testing.assert_allclose(derivative, np.swapaxes(derivative, 2, 3), atol=1e-12)
