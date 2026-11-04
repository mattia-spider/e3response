"""Occupation-matrix inputs (e3response.occupation), the occupation-energy dataset and model."""

import json
import pathlib

from ase import Atoms
import e3nn_jax as e3j
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import omegaconf
import pytest
from scipy.spatial.transform import Rotation
import tensorial
import tensorial.config

from e3response import keys
from e3response.data.occupation_energy import OccupationEnergyDataset
from e3response.occupation import HubbardEnergy, OccupationMatrix, qe_to_e3nn_cob

CONFIGS = pathlib.Path(__file__).resolve().parents[1] / "configs"


def qe_d_harmonics(directions: np.ndarray) -> np.ndarray:
    """QE's real d harmonics at unit vectors, in its order z^2, xz, yz, x^2-y^2, xy and with its
    (-1)^m sign; unit norm at every direction."""
    x, y, z = directions.T
    s3 = np.sqrt(3.0)
    return np.stack(
        [0.5 * (3 * z**2 - 1), -s3 * x * z, -s3 * y * z, s3 / 2 * (x**2 - y**2), s3 * x * y], 1
    )


def unit_vectors(n: int, seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_qe_to_e3nn_cob_maps_qe_harmonics_to_e3nn():
    directions = unit_vectors(20, 0)
    cob = np.asarray(qe_to_e3nn_cob(2))
    e3nn = e3j.spherical_harmonics("2e", jnp.asarray(directions), True, "norm").array
    np.testing.assert_allclose(cob @ cob.T, np.eye(5), atol=1e-6)
    np.testing.assert_allclose(qe_d_harmonics(directions) @ cob, e3nn, atol=1e-3)


def test_occupation_matrix_rotates_with_the_structure():
    """An occupation matrix that is a function of the ligand directions (the ligand-field operator
    sum_k |Y(r_k)><Y(r_k)| in the QE basis) gives irreps that rotate with the directions."""
    attr = OccupationMatrix(2)
    assert attr.irreps == e3j.Irreps("0e + 2e + 4e")

    def ligand_field(directions):
        y = qe_d_harmonics(directions)
        return jnp.asarray(y.T @ y)

    directions = unit_vectors(6, 1)
    rotation = Rotation.random(random_state=2).as_matrix()
    before = attr.create_tensor(ligand_field(directions))
    after = attr.create_tensor(ligand_field(directions @ rotation.T))
    np.testing.assert_allclose(after.array, before.transform_by_matrix(rotation).array, atol=2e-3)
    # Not invariant: the test would pass trivially otherwise
    assert np.abs(after.array - before.array).max() > 0.1


def _octahedron(seed: int, distortion: float = 0.1) -> Atoms:
    """An Fe with six O around it, distorted, plus an H further out."""
    rng = np.random.default_rng(seed)
    ligands = 2.0 * np.vstack([np.eye(3), -np.eye(3)]) + distortion * rng.normal(size=(6, 3))
    positions = np.vstack([[0.0, 0.0, 0.0], ligands, [[2.5, 2.5, 0.3]]])
    return Atoms(numbers=[26] + [8] * 6 + [1], positions=positions)


def _random_occupations(rng, n_sites: int) -> np.ndarray:
    """Symmetric (n_sites, 2, 5, 5) matrices with eigenvalues in [0, 1]."""
    out = np.empty((n_sites, 2, 5, 5))
    for i in range(n_sites):
        for s in range(2):
            v = np.linalg.qr(rng.normal(size=(5, 5)))[0]
            out[i, s] = v @ np.diag(rng.random(5)) @ v.T
    return out


def _entry(group: str, compound: str, seed: int, n_states: int) -> dict:
    rng = np.random.default_rng(seed)
    atoms = _octahedron(seed)
    states = []
    for k in range(n_states):
        occupations = _random_occupations(rng, 1)
        u = 5.0
        e_u = 0.5 * u * sum(np.trace(n - n @ n) for n in occupations[0])
        states.append(
            {
                "pk": 100 * seed + k,
                "occupations": occupations.tolist(),
                "energy": -1000.0 * (seed + 1) + rng.normal(),
                "hubbard_energy": float(e_u),
                "total_magnetization": 4.0,
            }
        )
    return {
        "group": group,
        "compound": compound,
        "numbers": atoms.numbers.tolist(),
        "positions": atoms.positions.tolist(),
        "cell": np.zeros((3, 3)).tolist(),
        "pbc": [False] * 3,
        "hubbard_sites": [0],
        "hubbard_u": [5.0],
        "states": states,
    }


@pytest.fixture
def mock_scans(tmp_path: pathlib.Path) -> pathlib.Path:
    entries = [
        _entry("a_U5", "a", seed=0, n_states=6),
        _entry("a_U6", "a", seed=1, n_states=4),
        _entry("b_U5", "b", seed=2, n_states=5),
    ]
    path = tmp_path / "scans.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def test_dataset_energies_are_relative_to_each_group(mock_scans):
    dataset = OccupationEnergyDataset(r_max=3.0, data_file=mock_scans)
    assert len(dataset) == 15
    assert dataset.groups == ["a_U5"] * 6 + ["a_U6"] * 4 + ["b_U5"] * 5

    entries = json.loads(mock_scans.read_text())
    graphs = iter(dataset)
    for entry in entries:
        e_no_u = []
        for state in entry["states"]:
            graph = next(graphs)
            energy = float(np.ravel(graph.globals[keys.TOTAL_ENERGY])[0])
            hubbard = float(np.ravel(graph.globals[keys.HUBBARD_ENERGY])[0])
            assert hubbard == pytest.approx(state["hubbard_energy"], abs=1e-5)
            e_no_u.append(energy - hubbard)
            # The occupations sit on the Hubbard site only
            up = np.asarray(graph.nodes[keys.OCCUPATION_UP])
            np.testing.assert_allclose(up[0], state["occupations"][0][0], atol=1e-6)
            assert not up[1:].any()
            np.testing.assert_array_equal(
                np.ravel(graph.nodes[keys.HUBBARD_SITE]), [True] + [False] * 7
            )
        assert np.mean(e_no_u) == pytest.approx(0.0, abs=1e-4)


def test_dataset_selects_scans_by_group_or_compound(mock_scans):
    assert len(OccupationEnergyDataset(3.0, mock_scans, scans=["a"])) == 10
    assert len(OccupationEnergyDataset(3.0, mock_scans, scans=["a_U6", "b"])) == 9
    with pytest.raises(ValueError, match="Unknown scans"):
        OccupationEnergyDataset(3.0, mock_scans, scans=["c"])


def test_hubbard_energy_matches_the_closed_form(mock_scans):
    dataset = OccupationEnergyDataset(r_max=3.0, data_file=mock_scans)
    batch = jraph.batch([dataset[i] for i in (0, 7, 12)])
    batch = batch._replace(globals={**batch.globals, "predicted_energy": jnp.ones((3, 1))})
    module = HubbardEnergy()
    out = module.apply({}, batch)
    expected = np.ravel(batch.globals[keys.HUBBARD_ENERGY])
    np.testing.assert_allclose(
        np.ravel(out.globals["predicted_hubbard_energy"]), expected, rtol=1e-5
    )
    np.testing.assert_allclose(np.ravel(out.globals["predicted_energy"]), 1.0 + expected, rtol=1e-5)


@pytest.fixture(scope="module")
def model_and_species():
    """The model of configs/model/nequip_occupation.yaml, small."""
    species = [1, 8, 26]
    root = omegaconf.OmegaConf.create(
        {
            "r_max": 3.0,
            "mul": 2,
            "ell_max": 2,
            "num_layers": 1,
            "from_data": {
                "atomic_numbers": species,
                "n_elements": len(species),
                "avg_num_neighbours": 5.0,
                "steps_per_epoch": 1,
            },
            "model": omegaconf.OmegaConf.load(CONFIGS / "model" / "nequip_occupation.yaml"),
        }
    )
    return tensorial.config.instantiate(root.model.model), species


def _energy(model, params, graph) -> float:
    return float(np.ravel(model.apply(params, graph).globals["predicted_energy"])[0])


def _graph(atoms: Atoms, occupations: np.ndarray, r_max: float = 3.0) -> jraph.GraphsTuple:
    atoms = atoms.copy()
    n = len(atoms)
    up, down = np.zeros((n, 5, 5)), np.zeros((n, 5, 5))
    up[0], down[0] = occupations[0]
    atoms.arrays[keys.OCCUPATION_UP], atoms.arrays[keys.OCCUPATION_DOWN] = up, down
    atoms.arrays[keys.HUBBARD_U] = np.r_[5.0, np.zeros(n - 1)]
    atoms.arrays[keys.HUBBARD_SITE] = np.arange(n) == 0
    return tensorial.gcnn.atomic.graph_from_ase(
        atoms,
        r_max=r_max,
        atom_include_keys=(
            "numbers",
            keys.OCCUPATION_UP,
            keys.OCCUPATION_DOWN,
            keys.HUBBARD_U,
            keys.HUBBARD_SITE,
        ),
    )


def test_model_is_invariant_to_rotating_structure_and_occupations(model_and_species):
    model, _ = model_and_species
    atoms = _octahedron(3)
    occupations = _random_occupations(np.random.default_rng(4), 1)
    params = model.init(jax.random.PRNGKey(0), _graph(atoms, occupations))

    rotation = Rotation.random(random_state=5).as_matrix()
    rotated = atoms.copy()
    rotated.positions = atoms.positions @ rotation.T
    # The d-orbital rotation in the QE basis: Y(R r) = Y(r) @ d
    directions = unit_vectors(50, 6)
    d = np.linalg.lstsq(
        qe_d_harmonics(directions), qe_d_harmonics(directions @ rotation.T), rcond=None
    )[0]
    occupations_rotated = np.einsum("ba,...bc,cd->...ad", d, occupations, d)

    energy = _energy(model, params, _graph(atoms, occupations))
    assert _energy(model, params, _graph(rotated, occupations_rotated)) == pytest.approx(
        energy, abs=1e-4
    )


def test_model_is_invariant_to_the_spin_swap(model_and_species):
    model, _ = model_and_species
    atoms = _octahedron(7)
    occupations = _random_occupations(np.random.default_rng(8), 1)
    graph = _graph(atoms, occupations)
    params = model.init(jax.random.PRNGKey(1), graph)
    swapped = _graph(atoms, occupations[:, ::-1])
    assert _energy(model, params, swapped) == pytest.approx(_energy(model, params, graph), abs=1e-5)

    # The network alone is not invariant, so the average is what makes it so
    inner = model.layers[0].model
    inner_params = {"params": params["params"]["layers_0"]["model"]}
    assert abs(_energy(inner, inner_params, swapped) - _energy(inner, inner_params, graph)) > 1e-4
