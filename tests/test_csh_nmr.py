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
