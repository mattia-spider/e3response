from pathlib import Path

import ase.io
import numpy as np
import reax
from tensorial import gcnn

from e3response import keys
from e3response.data import BtoAllegroPolDataset

mock_dir = Path(__file__).parent / "mock_datasets" / "bto_allegro_pol"


def test_bto_allegro_pol_dataset():
    dataset = BtoAllegroPolDataset(r_max=5.0, data_dir=mock_dir, include_electric_field=True)
    raw = ase.io.read(mock_dir / "BaTiO3.xyz", index=":")
    assert len(dataset) == len(raw) == 3

    for graph, atoms in zip(dataset, raw):
        n_atoms = len(atoms)
        born = graph.nodes[keys.BORN_CHARGES]
        assert born.shape == (n_atoms, 3, 3)
        # Field index first: row α of Z*_i is ∂P_α/∂r_i, the 3 file columns 3α..3α+2
        np.testing.assert_allclose(born[:, 1, :], atoms.arrays["born_charge"][:, 3:6], rtol=1e-6)
        # Acoustic sum rule
        np.testing.assert_allclose(born.sum(axis=0), 0.0, atol=1e-6)

        assert graph.nodes[keys.FORCES].shape == (n_atoms, 3)
        assert graph.globals[keys.TOTAL_ENERGY].shape == (1, 1)
        assert graph.globals[keys.STRESS].shape == (1, 3, 3)
        assert graph.globals[keys.DIPOLE_MOMENT].shape == (1, 3)
        polarizability = graph.globals[keys.POLARIZABILITY]
        assert polarizability.shape == (1, 3, 3)
        np.testing.assert_allclose(
            polarizability[0, 0], np.asarray(atoms.info["polarizability"])[:3], rtol=1e-6
        )
        np.testing.assert_array_equal(graph.globals[keys.EXTERNAL_ELECTRIC_FIELD], 0.0)


def test_bto_allegro_pol_datamodule():
    datamodule = gcnn.data.GraphDataModule.from_random_split(
        BtoAllegroPolDataset(r_max=5.0, data_dir=mock_dir, labels=(keys.BORN_CHARGES,)),
        train_val_test_split=(0.34, 0.33, 0.33),
        batch_size=1,
    )
    datamodule.setup(reax.Engine())

    for loader in (datamodule.train_dataloader(), datamodule.val_dataloader()):
        batch = next(iter(loader))[0]
        assert batch.nodes[keys.BORN_CHARGES].shape[-2:] == (3, 3)


def test_folded_dipole_loss_ignores_lattice_jumps():
    """A prediction one lattice vector (times e) away from the stored dipole is not an error."""
    import jax.numpy as jnp
    import jraph

    from e3response import losses

    cell = jnp.diag(jnp.array([11.9, 11.9, 12.9]))
    target = jnp.array([[5.5, -1.0, 0.3]])
    graph = jraph.GraphsTuple(
        nodes={}, edges=None, senders=jnp.zeros(0, int), receivers=jnp.zeros(0, int),
        n_node=jnp.array([0]), n_edge=jnp.array([0]),
        globals={"cell": cell[None], keys.DIPOLE_MOMENT: target,
                 "predicted_dipole_moment": target + jnp.array([[-11.9 + 0.2, 12.9 * 0, 0.0]])},
    )
    loss = losses.FoldedDipoleLoss()(graph)
    # Only the 0.2 off the nearest image counts: squared error averaged over 3 components
    np.testing.assert_allclose(loss, 0.2**2 / 3, rtol=1e-5)
