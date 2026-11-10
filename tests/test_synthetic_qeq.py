"""The synthetic QEq molecules (nmr_project/synthetic) through the allegro-pol reader and the models.

The mock file holds three molecules of the finite-field set A_E3, as the generator writes them: no
stress, no cell, a different field on each structure.
"""

import pathlib

import ase.io
import hydra
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import pytest
import reax
import tensorial
from tensorial import gcnn
from tensorial import reaxkit as rkit

from e3response import keys
from e3response.data import BtoAllegroPolDataset

MOCK_DIR = pathlib.Path(__file__).parent / "mock_datasets" / "synthetic_qeq"
CONFIG_PATH = pathlib.Path(__file__).parent.parent / "configs"
LABELS = (keys.FORCES, keys.BORN_CHARGES, "energy", keys.DIPOLE_MOMENT, keys.POLARIZABILITY)


def _dataset():
    return BtoAllegroPolDataset(r_max=5.0, data_dir=MOCK_DIR, filename="synthetic_qeq.xyz",
                                labels=LABELS, include_electric_field=True)


def test_loader_keeps_the_stored_field():
    raw = ase.io.read(MOCK_DIR / "synthetic_qeq.xyz", index=":")
    dataset = _dataset()
    assert len(dataset) == len(raw) == 3
    for graph, atoms in zip(dataset, raw):
        field = graph.globals[keys.EXTERNAL_ELECTRIC_FIELD]
        np.testing.assert_allclose(field[0], atoms.info[keys.EXTERNAL_ELECTRIC_FIELD], rtol=1e-6)  # graphs are float32
        assert np.linalg.norm(field) > 0.0
        assert keys.STRESS not in graph.globals and "cell" not in graph.globals
        # Field index first, as in allegro-pol: row α of Z_i is the file columns 3α..3α+2
        born = graph.nodes[keys.BORN_CHARGES]
        np.testing.assert_allclose(born[:, 2, :], atoms.arrays["born_charge"][:, 6:9], rtol=1e-6)  # graphs are float32
        np.testing.assert_allclose(born.sum(axis=0), 0.0, atol=1e-5)


def test_another_file_is_not_downloaded(tmp_path):
    with pytest.raises(FileNotFoundError):
        BtoAllegroPolDataset(r_max=5.0, data_dir=tmp_path, filename="synthetic_qeq.xyz")


def _model_and_batch(model_name):
    """The model of a config, its from_data values computed on the mock set, and one batch of it."""
    trainer = reax.Trainer()
    with hydra.initialize_config_dir(version_base=None, config_dir=str(CONFIG_PATH)):
        cfg = hydra.compose(config_name="train", overrides=[
            f"model={model_name}", "experiment=synthetic_sobolev_vs_direct", "mul=4", "num_layers=2"])
    datamodule = gcnn.data.GraphDataModule.from_random_split(
        _dataset(), train_val_test_split=(1.0, 0.0, 0.0), batch_size=3)
    trainer.run(rkit.FromData(cfg["from_data"], engine=trainer.engine, rngs=trainer.rngs,
                              datamodule=datamodule))
    module = tensorial.config.instantiate(cfg["model"])
    datamodule.setup(trainer.engine)
    batch = jax.tree.map(jnp.asarray, next(iter(datamodule.train_dataloader()))[0])
    model = module._model  # pylint: disable=protected-access
    params = jax.jit(model.init)(jax.random.PRNGKey(0), batch)
    return model, params, batch


def _with_positions(graph, positions):
    return graph._replace(nodes={**graph.nodes, keys.POSITIONS: positions})


def test_sobolev_born_charges_are_the_derivative_of_its_dipole():
    """Through the real pipeline (loader, from_data, config), at the field of each structure: the
    Sobolev model's Born charge Z[i, α, γ] is ∂d_α/∂r_iγ of its own dipole, the index order of the
    stored label (field first, displacement second)."""
    model, params, batch = _model_and_batch("nequip_bto_pol_sobolev")
    apply = jax.jit(lambda graph: model.apply(params, graph))
    out = apply(batch)
    born = np.asarray(out.nodes["predicted_born_charges"])
    graph_of = np.repeat(np.arange(len(batch.n_node)), np.asarray(batch.n_node))
    positions = np.asarray(batch.nodes[keys.POSITIONS])

    # The atom of the first molecule with the least symmetric Born tensor, so a transpose would show
    first = np.flatnonzero(graph_of == 0)
    asym = np.linalg.norm(born[first] - born[first].transpose(0, 2, 1), axis=(1, 2))
    i = int(first[np.argmax(asym)])
    assert asym.max() / np.linalg.norm(born[i]) > 0.1

    h = 1e-3
    columns = []
    for gamma in range(3):
        step = np.zeros_like(positions)
        step[i, gamma] = h
        up = apply(_with_positions(batch, jnp.asarray(positions + step)))
        down = apply(_with_positions(batch, jnp.asarray(positions - step)))
        d_dipole = (np.asarray(up.globals["predicted_dipole_moment"][0])
                    - np.asarray(down.globals["predicted_dipole_moment"][0])) / (2 * h)
        np.testing.assert_allclose(born[i, :, gamma], d_dipole, rtol=2e-2, atol=1e-4)
        columns.append(d_dipole)
    # The transposed order would not pass: the tensor is far from symmetric
    columns = np.stack(columns, axis=1)  # [α, γ]
    print("Born vs FD of the dipole: column order", np.linalg.norm(born[i] - columns) / np.linalg.norm(columns),
          "transposed", np.linalg.norm(born[i].T - columns) / np.linalg.norm(columns))
    assert np.linalg.norm(born[i].T - columns) > 10 * np.linalg.norm(born[i] - columns)


@pytest.mark.parametrize("model_name", ["nequip_bto_pol_sobolev", "nequip_bto_pol_direct"])
def test_each_structure_sees_its_own_field(model_name):
    """In a batch of molecules at different fields, molecule g gets the prediction it gets when every
    molecule of the batch is at its field: the field is per structure (the Sobolev chain used to
    take the first structure's for the whole batch), and it is an input of both models."""
    model, params, batch = _model_and_batch(model_name)
    apply = jax.jit(lambda graph: model.apply(params, graph))
    fields = batch.globals[keys.EXTERNAL_ELECTRIC_FIELD]

    def at(new_fields):
        return apply(batch._replace(globals={**batch.globals, keys.EXTERNAL_ELECTRIC_FIELD: new_fields}))

    out = at(fields)
    zero = at(0 * fields)
    assert not np.allclose(out.globals["predicted_dipole_moment"][:3], zero.globals["predicted_dipole_moment"][:3],
                           rtol=1e-3), "the predictions do not depend on the field"
    for g in range(3):
        shared = at(jnp.broadcast_to(fields[g], fields.shape))
        for key in ("predicted_dipole_moment", "predicted_polarizability", "predicted_energy"):
            np.testing.assert_allclose(np.asarray(out.globals[key][g]), np.asarray(shared.globals[key][g]),
                                       rtol=1e-4, atol=1e-5, err_msg=f"{key}, graph {g}")
