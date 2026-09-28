"""Unit tests for the shared ``parse_limit`` / ``apply_limit`` helpers used by the NMR
datamodules."""

import numpy as np
import pytest

from e3response.data._limit import apply_limit, parse_limit


@pytest.mark.parametrize(
    "limit, expected",
    [
        (None, slice(None)),
        (5, slice(None, 5)),
        (0, slice(None, 0)),
        ("2:8", slice(2, 8)),
        ("2:8:2", slice(2, 8, 2)),
        (":8", slice(None, 8)),
        ("2:", slice(2, None)),
        ("::2", slice(None, None, 2)),
        (":", slice(None, None)),
    ],
)
def test_parse_limit_returns_expected_slice(limit, expected):
    assert parse_limit(limit) == expected


@pytest.mark.parametrize(
    "limit, expected",
    [
        (None, list(range(10))),
        (3, [0, 1, 2]),
        ("2:8", [2, 3, 4, 5, 6, 7]),
        ("2:8:2", [2, 4, 6]),
        (":4", [0, 1, 2, 3]),
        ("6:", [6, 7, 8, 9]),
        ("::3", [0, 3, 6, 9]),
    ],
)
def test_parse_limit_applied_to_sequence(limit, expected):
    """The parsed slice, applied to a list, selects the expected elements."""
    items = list(range(10))
    assert items[parse_limit(limit)] == expected


@pytest.mark.parametrize("limit", ["abc", "1:2:3:4", "1:2:3:4:5"])
def test_parse_limit_invalid_raises(limit):
    with pytest.raises(ValueError):
        parse_limit(limit)


@pytest.mark.parametrize("limit", [None, 3, "2:8", "2:8:2", ":4", "6:", "::3"])
def test_apply_limit_matches_parse_limit_for_slices(limit):
    """For every slice spec, apply_limit is exactly slicing with parse_limit."""
    items = list(range(10))
    assert apply_limit(items, limit) == items[parse_limit(limit)]


def test_apply_limit_random_selects_a_sorted_subset():
    items = list(range(100))
    picked = apply_limit(items, "random:10")
    assert len(picked) == 10
    assert len(set(picked)) == 10
    assert set(picked) <= set(items)
    assert picked == sorted(picked)  # original order is kept


def test_apply_limit_random_is_reproducible_and_seeded():
    items = list(range(100))
    assert apply_limit(items, "random:10") == apply_limit(items, "random:10")
    assert apply_limit(items, "random:10") == apply_limit(items, "random:10:0")
    assert apply_limit(items, "random:10:0") != apply_limit(items, "random:10:1")


def test_apply_limit_random_is_not_a_prefix():
    """The point of random over an int: it is not just the first N items."""
    items = list(range(1000))
    assert apply_limit(items, "random:50") != items[:50]


def test_apply_limit_random_clips_to_the_available_items():
    items = list(range(5))
    assert apply_limit(items, "random:50") == items
    assert apply_limit(items, "random:0") == []


def test_apply_limit_random_keeps_arrays_as_arrays():
    items = np.arange(20)
    picked = apply_limit(items, "random:5")
    assert isinstance(picked, np.ndarray)
    assert len(picked) == 5


@pytest.mark.parametrize("limit", ["random", "random:x", "random:5:x", "random:1:2:3", "random:-1"])
def test_apply_limit_random_invalid_raises(limit):
    with pytest.raises(ValueError):
        apply_limit(list(range(10)), limit)


def test_parse_limit_points_random_specs_at_apply_limit():
    with pytest.raises(ValueError, match="apply_limit"):
        parse_limit("random:5")
