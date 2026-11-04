import numpy as np
import pytest

from e3response.data import GroupedKFoldSplit, split_indices


class _Grouped(list):
    """A list of structure ids carrying one group label per structure."""

    def __init__(self, groups):
        super().__init__(range(len(groups)))
        self.groups = list(groups)


_GROUPS = ["bulk"] * 303 + ["surf"] * 62


def _parts(fold, n_folds=5, seed=42, groups=_GROUPS):
    dataset = _Grouped(groups)
    strategy = GroupedKFoldSplit(fold, n_folds, test_fraction=0.1, seed=seed)
    return {p: split_indices(dataset, strategy, p) for p in ("train", "val", "test")}


def test_grouped_kfold_is_stratified():
    groups = np.array(_GROUPS)
    for fold in range(5):
        parts = _parts(fold)
        assert sorted(sum(parts.values(), [])) == list(range(len(groups)))  # a partition
        n_surf = {p: int((groups[idx] == "surf").sum()) for p, idx in parts.items()}
        assert n_surf["test"] == 6
        assert n_surf["val"] in (11, 12)  # (62 - 6) surfaces in 5 near-equal folds


def test_grouped_kfold_folds_cover_the_rest_once_and_share_the_test():
    tests = {tuple(sorted(_parts(fold)["test"])) for fold in range(5)}
    assert len(tests) == 1
    vals = [i for fold in range(5) for i in _parts(fold)["val"]]
    assert len(vals) == len(set(vals)) == len(_GROUPS) - len(next(iter(tests)))


def test_grouped_kfold_depends_on_seed_only():
    assert _parts(2) == _parts(2)
    assert _parts(2, seed=1) != _parts(2)


def test_grouped_kfold_rejects_bad_fold():
    with pytest.raises(ValueError):
        GroupedKFoldSplit(5, 5)
