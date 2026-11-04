"""Every interaction stack in configs/model gets one set of per-species weights per element.

Nequip's num_species and Mace's num_types default to 1.  With the default, the per-species weights have a single
row, and under jit an out-of-range species index is clamped silently: every element then shares the weights, with
no error and no warning (the BTO configs trained like this until 2026-11).
"""

import pathlib

import pytest
import yaml

CONFIGS = sorted((pathlib.Path(__file__).resolve().parents[1] / "configs" / "model").glob("*.yaml"))
SPECIES_ARGUMENT = {"Nequip": "num_species", "Mace": "num_types"}


def _stacks(node):
    """Every (class name, arguments) of an interaction stack in a config tree."""
    if isinstance(node, dict):
        target = str(node.get("_target_", "")).rsplit(".", 1)[-1]
        if target in SPECIES_ARGUMENT:
            yield target, node
        for value in node.values():
            yield from _stacks(value)
    elif isinstance(node, list):
        for value in node:
            yield from _stacks(value)


@pytest.mark.parametrize("path", CONFIGS, ids=[p.stem for p in CONFIGS])
def test_interaction_stack_has_per_species_weights(path):
    for target, args in _stacks(yaml.safe_load(path.read_text())):
        argument = SPECIES_ARGUMENT[target]
        assert args.get(argument) == "${from_data.n_elements}", (
            f"{path.name}: {target} needs {argument}: ${{from_data.n_elements}}"
        )
