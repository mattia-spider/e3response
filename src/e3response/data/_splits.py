"""Split strategies for :class:`tensorial.gcnn.data.GraphDataModule`, and replaying a split
outside of training."""

import collections
from collections.abc import Hashable, Sequence
import math
import types
from typing import Final

from flax import nnx
import numpy as np
import reax
from tensorial import gcnn

__all__ = ("GroupedKFoldSplit", "GroupedRandomSplit", "split_indices")


class GroupedRandomSplit(gcnn.data.SplitStrategy):
    """A random train/val/test split made separately within each group of the dataset, so that
    every partition holds the groups in the same proportions (a stratified split).

    The dataset gives the groups as a ``groups`` attribute, one hashable label per structure, e.g.
    the Qn signature of a silicate or whether a C-S-H cell is bulk or surface.  Groups are split in
    the order they first appear, each with :func:`reax.data.random_split` on ``engine.rngs``.
    """

    def __init__(self, fractions: Sequence[float] = (0.8, 0.1, 0.1)) -> None:
        if len(fractions) != 3:
            raise ValueError(f"Expected 3 split fractions (train, val, test), got {len(fractions)}")
        if not math.isclose(sum(fractions), 1.0):
            raise ValueError(f"Split fractions must sum to 1.0, got {sum(fractions)}")
        self._fractions: Final[tuple[float, ...]] = tuple(fractions)

    def split(self, dataset, engine: "reax.Engine", /) -> gcnn.data.Split:
        groups: Sequence[Hashable] | None = getattr(dataset, "groups", None)
        if groups is None or len(groups) != len(dataset):
            raise TypeError(
                f"{type(self).__name__} needs a dataset with a `groups` attribute, one label per "
                f"structure; {type(dataset).__name__} has none"
            )

        by_group: dict[Hashable, list[int]] = collections.defaultdict(list)
        for index, group in enumerate(groups):
            by_group[group].append(index)

        parts: tuple[list[int], list[int], list[int]] = ([], [], [])
        for indices in by_group.values():
            for part, subset in zip(
                parts, reax.data.random_split(engine.rngs, dataset=indices, lengths=self._fractions)
            ):
                part.extend(subset)

        return gcnn.data.Split(*([dataset[i] for i in part] for part in parts))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(fractions={self._fractions})"


class GroupedKFoldSplit(gcnn.data.SplitStrategy):
    """K-fold cross-validation made separately within each group of the dataset (stratified),
    so that the test set and every fold hold the groups in the same proportions.

    Within each group, ``test_fraction`` of the structures is held out as test, the same for every
    fold, and the rest is cut into ``n_folds`` near-equal folds: fold ``fold`` is the validation
    set, the others the training set.  Everything depends on ``seed`` alone, not on the training
    engine, so the split can be rebuilt offline (see :func:`split_indices`).
    """

    def __init__(
        self, fold: int, n_folds: int = 5, *, test_fraction: float = 0.1, seed: int = 42
    ) -> None:
        if not 0 <= fold < n_folds:
            raise ValueError(f"fold must be in [0, {n_folds}), got {fold}")
        if not 0.0 <= test_fraction < 1.0:
            raise ValueError(f"test_fraction must be in [0, 1), got {test_fraction}")
        self._fold: Final[int] = fold
        self._n_folds: Final[int] = n_folds
        self._test_fraction: Final[float] = test_fraction
        self._seed: Final[int] = seed

    def split(self, dataset, engine: "reax.Engine", /) -> gcnn.data.Split:
        groups: Sequence[Hashable] | None = getattr(dataset, "groups", None)
        if groups is None or len(groups) != len(dataset):
            raise TypeError(
                f"{type(self).__name__} needs a dataset with a `groups` attribute, one label per "
                f"structure; {type(dataset).__name__} has none"
            )

        by_group: dict[Hashable, list[int]] = collections.defaultdict(list)
        for index, group in enumerate(groups):
            by_group[group].append(index)

        rng = np.random.default_rng(self._seed)
        train, val, test = [], [], []
        for indices in by_group.values():
            order = rng.permutation(indices).tolist()
            n_test = round(self._test_fraction * len(order))
            test.extend(order[:n_test])
            for k, fold in enumerate(np.array_split(order[n_test:], self._n_folds)):
                (val if k == self._fold else train).extend(int(i) for i in fold)

        return gcnn.data.Split(*([dataset[i] for i in part] for part in (train, val, test)))

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(fold={self._fold}, n_folds={self._n_folds}, "
            f"test_fraction={self._test_fraction}, seed={self._seed})"
        )


class _Indices(collections.abc.Sequence):
    """The positions 0..n-1 of a dataset, carrying its groups, so that a strategy splits them
    exactly as it splits the dataset."""

    def __init__(self, n: int, groups: Sequence[Hashable] | None) -> None:
        self._n = n
        if groups is not None:
            self.groups = groups

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index):
        if isinstance(index, slice):
            return list(range(self._n))[index]
        if not -self._n <= index < self._n:
            raise IndexError(index)
        return index % self._n


def split_indices(
    dataset: Sequence,
    strategy: "gcnn.data.SplitStrategy",
    split: str,
    rngs: "nnx.Rngs | None" = None,
) -> list[int]:
    """The positions in ``dataset`` of one partition ("train", "val" or "test") of ``strategy``.

    ``rngs`` must match the training run's to reproduce its split; it defaults to
    ``nnx.Rngs(0)``, REAX's own default when no `Trainer`/`Engine` override is given.
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train', 'val' or 'test', got {split!r}")
    engine = types.SimpleNamespace(rngs=nnx.Rngs(0) if rngs is None else rngs)
    result = strategy.split(_Indices(len(dataset), getattr(dataset, "groups", None)), engine)
    part = getattr(result, split)
    return [] if part is None else [int(i) for i in part]
