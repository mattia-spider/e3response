import json
from pathlib import Path

from ase import Atoms
import numpy as np
import pytest
from tensorial import gcnn

from e3response.data import GroupedRandomSplit
from e3response.data.csh_nmr import CshNmrDataset

# Mock dataset constants: 5 bulk + 5 surf structures
_MOCK_N_BULK = 5
_MOCK_N_SURF = 5
_MOCK_N_STRUCTURES = _MOCK_N_BULK + _MOCK_N_SURF


def _make_entry(seed: int, struct_type: str, n_atoms: int = 3) -> dict:
    """Minimal JSON entry with the fields expected by CshNmrDataset."""
    rng = np.random.default_rng(seed)
    return {
        "numbers": [1] * n_atoms,
        "positions": (rng.random((n_atoms, 3)) * 2.0).tolist(),  # kept within r_max=3.0
        "cell": np.zeros((3, 3)).tolist(),
        "pbc": [False, False, False],
        "nmr_tensors": rng.random((n_atoms, 3, 3)).tolist(),
        "mask": [True] * n_atoms,
        "struct_type": struct_type,
        "ca_si_ratio": 1.0,
        "energy_Ry": -100.0,
    }


@pytest.fixture
def mock_csh_json(tmp_path: Path) -> Path:
    """JSON file with 5 bulk + 5 surf fake entries."""
    entries = [_make_entry(seed=i, struct_type="bulk") for i in range(_MOCK_N_BULK)]
    entries += [
        _make_entry(seed=_MOCK_N_BULK + i, struct_type="surf") for i in range(_MOCK_N_SURF)
    ]
    json_path = tmp_path / "csh_dataset.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(entries, f)
    return json_path


_STRATEGY = GroupedRandomSplit((0.6, 0.2, 0.2))


def _make_dm(data_file, test_engine, limit=None) -> gcnn.data.GraphDataModule:
    """A stratified data module over the mock file, set up as training sets it up."""
    dm = gcnn.data.GraphDataModule(
        CshNmrDataset(r_max=3.0, data_file=data_file, limit=limit), _STRATEGY, batch_size=1
    )
    dm.prepare_data()
    dm.setup(test_engine, stage="fit")
    return dm


def test_csh_nmr_datamodule_full_dataset_stratified(mock_csh_json, test_engine):
    """Full dataset: both bulk and surf structures appear in every partition."""
    dm = _make_dm(mock_csh_json, test_engine)

    n_train = len(dm.data_train)
    n_val = len(dm.data_val)
    n_test = len(dm.data_test)

    # All structures must be accounted for
    assert n_train + n_val + n_test == _MOCK_N_STRUCTURES

    # Every partition must be non-empty (guaranteed by stratification)
    assert n_train > 0, "train partition is empty"
    assert n_val > 0, "val partition is empty"
    assert n_test > 0, "test partition is empty"

    # With stratified 60/20/20 on 5 bulk + 5 surf:
    # each group gives (3, 1, 1) -> totals (6, 2, 2)
    assert n_train == 6
    assert n_val == 2
    assert n_test == 2


def test_csh_nmr_datamodule_only_bulk(mock_csh_json, test_engine):
    """limit='bulk' loads only the bulk structures."""
    dm = _make_dm(mock_csh_json, test_engine, limit="bulk")

    total = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert total == _MOCK_N_BULK, (
        f"Expected {_MOCK_N_BULK} structures with 'bulk', got {total}"
    )


def test_csh_nmr_datamodule_only_surface(mock_csh_json, test_engine):
    """limit='surface' loads only the surf structures."""
    dm = _make_dm(mock_csh_json, test_engine, limit="surface")

    total = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert total == _MOCK_N_SURF, (
        f"Expected {_MOCK_N_SURF} structures with 'surface', got {total}"
    )


@pytest.mark.parametrize(
    "limit, expected",
    [(None, _MOCK_N_STRUCTURES), (6, 6), ("0:6", 6), ("2:8:2", 3)],
)
def test_csh_nmr_constructor_slice_string_limit(mock_csh_json, limit, expected):
    """A slice-string / int limit is routed through the shared parse_limit inside
    _apply_limit in the constructor (distinct from the 'bulk'/'surface' path)."""
    assert len(CshNmrDataset(r_max=3.0, data_file=mock_csh_json, limit=limit)) == expected


def test_csh_nmr_load_split_structures(mock_csh_json, test_engine):
    """load_split_structures gives a strategy's partitions as ase.Atoms, the sizes training
    sees, and load_split the same structures as graphs."""
    dm = _make_dm(mock_csh_json, test_engine)

    dataset = CshNmrDataset(r_max=3.0, data_file=mock_csh_json)
    for split, data in (("train", dm.data_train), ("val", dm.data_val), ("test", dm.data_test)):
        structures = dataset.load_split_structures(split, _STRATEGY)
        assert all(isinstance(s, Atoms) for s in structures)
        assert len(structures) == len(data), f"'{split}' partition size differs from training"

        graphs = dataset.load_split(split, _STRATEGY)
        assert len(graphs) == len(structures)

    with pytest.raises(ValueError):
        dataset.load_split_structures("validation", _STRATEGY)


def test_csh_nmr_load_split_structures_limit(mock_csh_json):
    """The split's own `limit` restricts it with the same semantics as the
    constructor's, applied AFTER the split."""
    dataset = CshNmrDataset(r_max=3.0, data_file=mock_csh_json)
    train = dataset.load_split_structures("train", _STRATEGY)
    assert len(dataset.load_split_structures("train", _STRATEGY, limit=2)) == 2
    bulk = dataset.load_split_structures("train", _STRATEGY, limit="bulk")
    assert [a.info["struct_type"] for a in bulk] == ["bulk"] * sum(
        a.info["struct_type"] == "bulk" for a in train
    )


def test_csh_nmr_dataloaders(mock_csh_json, test_engine):
    """All three dataloaders yield valid graph batches with the expected fields."""
    dm = _make_dm(mock_csh_json, test_engine)

    for loader_fn in ["train_dataloader", "val_dataloader", "test_dataloader"]:
        loader = getattr(dm, loader_fn)()
        batch_tuple = next(iter(loader))

        batch = batch_tuple[0] if isinstance(batch_tuple, tuple) else batch_tuple

        assert hasattr(batch, "nodes"), f"{loader_fn} batch has no 'nodes'"
        assert "nmr_tensors" in batch.nodes, f"{loader_fn} batch missing 'nmr_tensors'"

        nmr = batch.nodes["nmr_tensors"]
        assert nmr.ndim == 3, f"'nmr_tensors' in {loader_fn} has wrong ndim {nmr.ndim}"
        assert nmr.shape[-2:] == (3, 3), (
            f"'nmr_tensors' last dims must be (3,3), got {nmr.shape[-2:]}"
        )


def _water_and_hydroxyl() -> Atoms:
    """A water molecule, a hydroxyl (Si-O-H) and a bare O, far enough apart not to mix."""
    return Atoms(
        "OHHOHSiO",
        positions=[
            [0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0],  # water
            [5.0, 0.0, 0.0], [5.96, 0.0, 0.0], [3.4, 0.0, 0.0],  # Si-O-H
            [10.0, 0.0, 0.0],  # bare O
        ],
    )


def test_water_atoms():
    from e3response.data.csh_nmr import water_atoms

    assert water_atoms(_water_and_hydroxyl()).tolist() == [
        True, True, True, False, False, False, False
    ]


def test_nmr_active_sites():
    from e3response.data.csh_nmr import nmr_active

    atoms = _water_and_hydroxyl()
    assert nmr_active(atoms, "all") is None
    assert nmr_active(atoms, None) is None
    assert nmr_active(atoms, "no_water").tolist() == [False] * 3 + [True] * 4
    assert nmr_active(atoms, [14]).tolist() == [False] * 5 + [True, False]
    with pytest.raises(ValueError):
        nmr_active(atoms, "water")


def test_csh_nmr_sites_in_graphs(mock_csh_json):
    """`nmr_sites` puts the label mask in the graphs; the default leaves them without it."""
    from e3response import keys

    plain = CshNmrDataset(r_max=3.0, data_file=mock_csh_json)
    assert keys.NMR_ACTIVE not in plain[0].nodes

    # The mock atoms are all H: keeping only Na leaves no active atom, keeping H all of them
    none_active = CshNmrDataset(r_max=3.0, data_file=mock_csh_json, nmr_sites=[11])
    all_active = CshNmrDataset(r_max=3.0, data_file=mock_csh_json, nmr_sites=[1])
    assert not np.asarray(none_active[0].nodes[keys.NMR_ACTIVE]).any()
    assert np.asarray(all_active[0].nodes[keys.NMR_ACTIVE]).all()


def _v2_entry(seed: int, struct_type: str, **quality) -> dict:
    """A Na_csh_v2.json entry: the v1 fields plus the susceptibility and the quality metrics."""
    rng = np.random.default_rng(seed)
    entry = _make_entry(seed, struct_type)
    entry.update(source=f"{struct_type}/shuffled_{seed}", chi_HH=(rng.random((3, 3)) - 500.0).tolist(),
                 sigma_macro=(7.5 * np.eye(3)).tolist(), fsum_err=0.001, anti_O=5.0, n_h2=0, n_si_not4=0,
                 listed_bad=False)
    entry.update(quality)
    return entry


@pytest.fixture
def mock_csh_v2_json(tmp_path: Path) -> Path:
    entries = [
        _v2_entry(0, "bulk"),
        _v2_entry(1, "bulk", fsum_err=0.002),
        _v2_entry(2, "surf", fsum_err=0.05),  # kept by v1, χ not labelled
        _v2_entry(3, "surf", anti_O=9.5, fsum_err=0.4),  # dropped by v1 (anti_O)
        _v2_entry(4, "surf", n_h2=1),  # dropped by v1 (H2), converged
        _v2_entry(5, "surf", listed_bad=True, fsum_err=3.0),  # dropped by v1 (first build)
        _v2_entry(6, "surf", fsum_err=0.005),
    ]
    path = tmp_path / "csh_v2.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "keep, max_fsum_err, expected",
    [("all", None, [0, 1, 2, 3, 4, 5, 6]), ("v1", None, [0, 1, 2, 6]), ("all", 0.01, [0, 1, 4, 6]),
     ("v1", 0.01, [0, 1, 6])],
)
def test_csh_nmr_v2_keep(mock_csh_v2_json, keep, max_fsum_err, expected):
    ds = CshNmrDataset(r_max=3.0, data_file=mock_csh_v2_json, keep=keep, max_fsum_err=max_fsum_err)
    assert [int(a.info["source"].split("_")[-1]) for a in ds.structures] == expected


def test_csh_nmr_v2_susceptibility(mock_csh_v2_json, mock_csh_json):
    """χ_HH is a graph global of every cell kept; a v1 file has none, so its graphs are as before."""
    ds = CshNmrDataset(r_max=3.0, data_file=mock_csh_v2_json, keep="v1", max_fsum_err=0.01)
    entries = {e["source"]: e for e in json.loads(Path(mock_csh_v2_json).read_text())}
    for atoms, graph in zip(ds.structures, ds):
        np.testing.assert_allclose(np.asarray(graph.globals["magnetic_susceptibility"]).reshape(3, 3),
                                   entries[atoms.info["source"]]["chi_HH"])

    v1 = CshNmrDataset(r_max=3.0, data_file=mock_csh_json)
    assert "magnetic_susceptibility" not in v1[0].globals
