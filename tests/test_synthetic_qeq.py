"""The synthetic QEq molecules (nmr_project/synthetic) through the allegro-pol reader and the models.

The mock file holds three molecules of the finite-field set A_E3, as the generator writes them: no
stress, no cell, a different field on each structure.
"""

import pathlib

import ase.io
import numpy as np
import pytest

from e3response import keys
from e3response.data import BtoAllegroPolDataset

MOCK_DIR = pathlib.Path(__file__).parent / "mock_datasets" / "synthetic_qeq"
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
