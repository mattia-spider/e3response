"""Shared helpers for parsing and applying dataset ``limit`` specifications."""

from collections.abc import Sequence
from typing import TypeVar

import numpy as np

T = TypeVar("T")

_RANDOM = "random"


def parse_limit(limit: int | str | None) -> slice:
    """Convert a limit spec to a slice over a sequence (structures, files, indices, ...).

    - None       → slice(None)       (everything)
    - int N      → slice(None, N)    (first N items)
    - "a:b"      → slice(a, b)       (items a through b-1)
    - "a:b:s"    → slice(a, b, s)    (with step)

    A random selection is not a slice, so ``"random:..."`` specs are handled by
    :func:`apply_limit` instead.
    """
    if limit is None:
        return slice(None)
    if isinstance(limit, int):
        return slice(None, limit)
    if _is_random(limit):
        raise ValueError(
            f"{limit!r} selects a random subset, which is not a slice: use apply_limit()"
        )
    parts = limit.split(":")
    indices = [int(p) if p else None for p in parts]
    if len(indices) == 2:
        return slice(indices[0], indices[1])
    if len(indices) == 3:
        return slice(indices[0], indices[1], indices[2])
    raise ValueError(
        f"Cannot parse limit {limit!r}: expected int, 'start:stop', 'start:stop:step' or "
        f"'random:N[:seed]'"
    )


def apply_limit(items: Sequence[T], limit: int | str | None) -> Sequence[T]:
    """Select from `items` according to a limit spec.

    Takes every spec :func:`parse_limit` does, with identical results, plus:

    - "random:N"       → N items drawn uniformly without replacement, seed 0
    - "random:N:seed"  → the same with an explicit seed

    A random selection is reproducible for a given seed and comes back in the original order.
    Unlike a prefix or a stride it does not depend on how the dataset happens to be ordered, so
    it samples the dataset's actual distribution.  Asking for more items than there are returns
    all of them.
    """
    if not _is_random(limit):
        return items[parse_limit(limit)]

    idx = _random_indices(len(items), limit)
    if hasattr(items, "__array__"):
        return items[idx]
    return [items[int(i)] for i in idx]


def _is_random(limit) -> bool:
    return isinstance(limit, str) and limit.split(":", 1)[0].strip().lower() == _RANDOM


def _random_indices(n_items: int, limit: str) -> np.ndarray:
    parts = limit.split(":")[1:]
    try:
        if len(parts) not in (1, 2):
            raise ValueError
        count = int(parts[0])
        seed = int(parts[1]) if len(parts) == 2 else 0
    except ValueError:
        raise ValueError(f"Cannot parse limit {limit!r}: expected 'random:N' or 'random:N:seed'")
    if count < 0 or seed < 0:
        raise ValueError(f"Cannot parse limit {limit!r}: N and seed must be non-negative")

    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_items, size=min(count, n_items), replace=False))
