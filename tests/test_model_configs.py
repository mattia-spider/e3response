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


def _edge_harmonics(node):
    """Every irreps of the spherical harmonics that encode the edge directions in a config tree."""
    if isinstance(node, dict):
        attrs = node.get("attrs")
        if isinstance(attrs, dict) and isinstance(attrs.get("edge_vectors"), dict):
            yield attrs["edge_vectors"].get("irreps")
        for value in node.values():
            yield from _edge_harmonics(value)
    elif isinstance(node, list):
        for value in node:
            yield from _edge_harmonics(value)


# the lab's own examples, kept as upstream has them
UPSTREAM = {"mace", "nequip_electric", "nequip_electric_deriv"}
OURS = [p for p in CONFIGS if p.stem not in UPSTREAM]


@pytest.mark.parametrize("path", OURS, ids=[p.stem for p in OURS])
def test_edge_harmonics_follow_ell_max(path):
    """The edge spherical harmonics go up to the ell_max of the hidden features, as in NequIP and MACE.

    A fixed 0e + 1o limits every message to the l = 1 direction of the neighbour whatever ell_max the features use (the
    BTO and allegro-pol configs trained like this until 2026-11).
    """
    for irreps in _edge_harmonics(yaml.safe_load(path.read_text())):
        assert isinstance(irreps, dict) and str(irreps.get("ell_max", "")).startswith("${oc.select:ell_max"), (
            f"{path.name}: edge spherical harmonics need make_irreps up to ell_max, got {irreps!r}"
        )
