from pathlib import Path

import numpy as np
import pytest
import reax
from tensorial import gcnn
from tensorial.gcnn import atomic

from e3response.data.qm9_nmr import DATASET_URLS, Qm9NmrDataset

mock_dir = Path(__file__).parent / "mock_datasets" / "qm9_nmr"


@pytest.mark.parametrize("dataset_name", list(DATASET_URLS.keys()))
def test_qm9_nmr_dataset(dataset_name):
    dataset = Qm9NmrDataset(
        dataset=dataset_name,
        atom_keys=["species", "anisotropy"],
        data_dir=mock_dir,
    )
    assert len(dataset) > 0

    for i, graph in enumerate(dataset):
        assert graph is not None, f"Graph {i} is None for dataset {dataset_name}"
        assert hasattr(
            graph, "nodes"
        ), f"Graph {i} contains no attribute 'nodes' for dataset {dataset_name}"
        assert (
            "nmr_tensors" in graph.nodes
        ), f"Graph {i} lacks 'nmr_tensors' for dataset {dataset_name}"
        assert isinstance(
            graph.nodes["nmr_tensors"], np.ndarray
        ), f"'nmr_tensors' in graph {i} is not a numpy array for dataset {dataset_name}"
        assert graph.nodes["nmr_tensors"].shape[-2:] == (
            3,
            3,
        ), f"Wrong nmr tensor shape in graph {i} for dataset {dataset_name}"
        assert (
            "nmr_tensors" in graph.nodes
        ), f"Graph {i} lacks 'nmr_tensors' for dataset {dataset_name}"
        assert isinstance(
            graph.nodes["mu"], np.ndarray
        ), f"'mu' in graph {i} is not a numpy array for dataset {dataset_name}"
        assert (
            atomic.TOTAL_ENERGY in graph.globals
        ), f"Graph {i} missing '{atomic.TOTAL_ENERGY}' in globals for dataset {dataset_name}"
        energy = graph.globals[atomic.TOTAL_ENERGY]
        assert isinstance(
            energy, (float, np.floating, np.ndarray)
        ), f"'{atomic.TOTAL_ENERGY}' in graph {i} is not a float for dataset {dataset_name}"
        assert not np.isnan(
            energy
        ), f"'{atomic.TOTAL_ENERGY}' in graph {i} is NaN for dataset {dataset_name}"


def _check_loaders(dm):
    for loader_fn in ["train_dataloader", "val_dataloader", "test_dataloader"]:
        loader = getattr(dm, loader_fn)()
        batch_tuple = next(iter(loader))

        assert isinstance(batch_tuple, tuple), f"{loader_fn} output is not a tuple"
        batch = batch_tuple[0]

        assert hasattr(batch, "nodes"), f"{loader_fn} batch has no 'nodes'"

        assert "nmr_tensors" in batch.nodes, f"{loader_fn} batch missing 'nmr_tensors'"

        nmr_tensors = batch.nodes["nmr_tensors"]

        # Shape
        assert isinstance(
            nmr_tensors, np.ndarray
        ), f"'nmr_tensors' in {loader_fn} is not a numpy array"
        assert (
            nmr_tensors.ndim == 3
        ), f"'nmr_tensors' in {loader_fn} has wrong shape {nmr_tensors.shape}"
        assert nmr_tensors.shape[-2:] == (
            3,
            3,
        ), f"Last dims of 'nmr_tensors' must be (3,3), got {nmr_tensors.shape[-2:]}"

        # Check mu
        assert "mu" in batch.nodes, f"{loader_fn} batch missing 'mu'"
        mu = batch.nodes["mu"]

        assert isinstance(mu, np.ndarray), f"'mu' in {loader_fn} is not a numpy array"
        # Per-node scalars are (n_node, 1) since tensorial normalised graph array shapes
        assert (
            mu.ndim == 2 and mu.shape[1] == 1
        ), f"'mu' in {loader_fn} has wrong shape {mu.shape}, expected (n_node, 1)"
        assert not np.any(np.isnan(mu)), f"'mu' in {loader_fn} contains NaNs"

        # Check energy
        assert hasattr(batch, "globals"), f"{loader_fn} batch has no 'globals'"
        assert (
            atomic.TOTAL_ENERGY in batch.globals
        ), f"{loader_fn} batch missing '{atomic.TOTAL_ENERGY}' in globals"
        energy = batch.globals[atomic.TOTAL_ENERGY]
        assert isinstance(energy, np.ndarray), f"'{atomic.TOTAL_ENERGY}' in {loader_fn} is not a numpy array"
        assert not np.any(np.isnan(energy)), f"'{atomic.TOTAL_ENERGY}' in {loader_fn} contains NaNs"


@pytest.mark.parametrize("dataset_name", list(DATASET_URLS.keys()))
def test_qm9_nmr_datamodule(dataset_name, test_engine):
    dm = gcnn.data.GraphDataModule.from_random_split(
        Qm9NmrDataset(dataset=dataset_name, data_dir=mock_dir),
        train_val_test_split=(0.6, 0.2, 0.2),
        batch_size=1,
    )
    dm.prepare_data()
    dm.setup(test_engine)
    _check_loaders(dm)


def test_qm9_nmr_datamodule_kfold(test_engine):
    dm = gcnn.data.GraphDataModule.from_kfold(
        Qm9NmrDataset(dataset="gasphase", data_dir=mock_dir),
        fold=1,
        n_folds=2,
        test_fraction=0.2,
        batch_size=1,
    )
    dm.prepare_data()
    dm.setup(test_engine)
    # 5 structures: 1 for test, the other 4 in 2 folds of 2
    assert (len(dm.data_train), len(dm.data_val), len(dm.data_test)) == (2, 2, 1)
    _check_loaders(dm)


def _positions(graphs) -> list[np.ndarray]:
    return [np.asarray(graph.nodes["positions"]) for graph in graphs]


@pytest.mark.parametrize(
    "strategy",
    [
        gcnn.data.RandomSplit((0.6, 0.2, 0.2)),
        gcnn.data.KFoldSplit(fold=1, n_folds=3, test_fraction=0.2),
    ],
    ids=["random", "kfold"],
)
def test_qm9_nmr_load_split(strategy):
    """`load_split` parses exactly the structures `GraphDataModule` puts in each split."""
    kwargs = dict(dataset=["gasphase", "THF"], data_dir=mock_dir)
    dm = gcnn.data.GraphDataModule(Qm9NmrDataset(**kwargs), strategy, batch_size=1)
    dm.prepare_data()
    dm.setup(reax.Engine())

    for split, graphs in zip(("train", "val", "test"), (dm.data_train, dm.data_val, dm.data_test)):
        loaded = Qm9NmrDataset.load_split(split, strategy, **kwargs)
        assert len(loaded) == len(graphs) > 0
        for expected, actual in zip(_positions(graphs), _positions(loaded)):
            np.testing.assert_array_equal(expected, actual)

    assert len(Qm9NmrDataset.load_split("train", strategy, split_limit=1, **kwargs)) == 1
    with pytest.raises(ValueError):
        Qm9NmrDataset.load_split("validation", strategy, **kwargs)
