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


def _after_stacks(node):
    """The layer that follows each interaction stack in every list of layers of a config tree."""
    if isinstance(node, dict):
        for value in node.values():
            yield from _after_stacks(value)
    elif isinstance(node, list):
        for i, value in enumerate(node[:-1]):
            if isinstance(value, dict) and str(value.get("_target_", "")).rsplit(".", 1)[-1] in SPECIES_ARGUMENT:
                yield node[i + 1]
        for value in node:
            yield from _after_stacks(value)


@pytest.mark.parametrize("path", OURS, ids=[p.stem for p in OURS])
def test_readouts_read_the_whole_stack(path):
    """No NodewiseLinear bottleneck between the interaction stack and the readouts, as in the NMR models (d3a856f).

    The BTO and allegro-pol configs had a fixed 4x0e + 4x0o + ... + 2x3e layer there until 2026-11.
    """
    for layer in _after_stacks(yaml.safe_load(path.read_text())):
        is_linear = str(layer.get("_target_", "")).endswith("NodewiseLinear")
        assert not is_linear or "out_field" in layer, f"{path.name}: bottleneck after the interaction stack: {layer!r}"


# Second derivatives of the energy in a field: the module that computes each one, and the input that gives the energy
# an explicit product of the field(s).  A NequIP stack is linear in its features (only the gates multiply them), so
# without that input the curvature in the field stays nearly flat: σ collapses without μ⊗B (tests/verifica), χ without
# B⊗B, and the dielectric-only Sobolev went from 0.0161 to 0.0108 with ε⊗ε (v2, test, 5 folds).
SECOND_ORDER = {
    "DielectricTensor": "NodewiseFieldSquare",  # χ = ∂²E/∂ε²
    "RamanTensors": "NodewiseFieldSquare",  # ∂³E/∂ε²∂u
    "FusedFieldChain": "NodewiseFieldSquare",  # χ and Raman from one evaluation
    "RamanTensorsFromPolarizability": "NodewiseFieldSquare",
    "MagneticShieldingTensor": "NodewiseFieldProduct",  # σ = ∂²E/∂μ∂B, when the energy is the root (InducedMagneticField)
    "MagneticSusceptibility": "NodewiseFieldProduct",  # χ = -∂²E/∂B², needs field_squared
}
# configs left without the product on purpose: the controls of the claim
WITHOUT_PRODUCT = {
    "nequip_nmr_deriv2nd": "Diff2nd without μ⊗B, the Q1 control",
    "mace_nmr_deriv2nd": "control for μ⊗B from MACE's product basis (mace_nmr_deriv2nd_muB has it explicit)",
}
# the electric field enters MACE only as a global attribute (Pack), where NodewiseFieldSquare cannot read it
MACE_GLOBAL_FIELD = {"mace_bto_sobolev", "mace_bto_pol_sobolev"}


def _targets(node):
    if isinstance(node, dict):
        yield str(node.get("_target_", "")), node
        for value in node.values():
            yield from _targets(value)
    elif isinstance(node, list):
        for value in node:
            yield from _targets(value)


def _layers_before_stack(node):
    """The layers before the interaction stack, in the list of layers that holds it."""
    if isinstance(node, dict):
        for value in node.values():
            yield from _layers_before_stack(value)
    elif isinstance(node, list):
        names = [str(v.get("_target_", "")).rsplit(".", 1)[-1] if isinstance(v, dict) else "" for v in node]
        for i, name in enumerate(names):
            if name in SPECIES_ARGUMENT:
                yield from node[:i]
        for value in node:
            yield from _layers_before_stack(value)


def _default(value):
    """The default of a ``${oc.select:key,default}`` interpolation, or the value itself."""
    text = str(value)
    return float(text.rsplit(",", 1)[1].rstrip("}")) if text.startswith("${oc.select:") else float(text)


SECOND_ORDER_CONFIGS = [
    p for p in OURS if p.stem not in WITHOUT_PRODUCT
    and any(t.startswith("e3response.") and t.rsplit(".", 1)[-1] in SECOND_ORDER
            for t, _ in _targets(yaml.safe_load(p.read_text())))
]


@pytest.mark.parametrize(
    "path",
    [pytest.param(p, marks=pytest.mark.xfail(strict=True, reason="field only a MACE global attribute: ε⊗ε input pending"))
     if p.stem in MACE_GLOBAL_FIELD else p for p in SECOND_ORDER_CONFIGS],
    ids=[p.stem for p in SECOND_ORDER_CONFIGS],
)
def test_second_field_derivatives_have_the_field_product(path):
    """Every model that differentiates the energy twice in a field has the product of that field among its inputs,
    on by default and before the interaction stack: ε⊗ε for χ and Raman, μ⊗B for σ, B⊗B for the magnetic χ.

    The BTO and allegro-pol Sobolev configs had no ε⊗ε until 2026-11-10 (only the dielectric-only one, off by default).
    """
    config = yaml.safe_load(path.read_text())
    names = {t.rsplit(".", 1)[-1] for t, _ in _targets(config)}
    if "MagneticShieldingTensor" in names and "InducedMagneticField" not in names:
        names.discard("MagneticShieldingTensor")  # deriv1st: σ is the first derivative of a B_ind readout
    needed = {SECOND_ORDER[n] for n in names if n in SECOND_ORDER}
    before = [layer for layer in _layers_before_stack(config) if isinstance(layer, dict)]
    for module in needed:
        found = [layer for layer in before if str(layer.get("_target_", "")).endswith("." + module)]
        assert found, f"{path.name}: second field derivative without {module} before the interaction stack"
        assert all(_default(layer.get("scale", 1.0)) != 0 for layer in found), f"{path.name}: {module} off by default"
    if "MagneticSusceptibility" in names:
        assert any(layer.get("field_squared") is True for layer in before
                   if str(layer.get("_target_", "")).endswith(".NodewiseFieldProduct")), (
            f"{path.name}: magnetic susceptibility without B⊗B (field_squared: True)"
        )


def _up_to_stack(path):
    """The model's layers up to and including its interaction stack: its inputs and features."""
    layers = yaml.safe_load(path.read_text())["model"]["layers"]
    for i, layer in enumerate(layers):
        if str(layer.get("_target_", "")).rsplit(".", 1)[-1] in SPECIES_ARGUMENT:
            return layers[: i + 1]
    raise AssertionError(f"{path.name}: no interaction stack")


PAIRS = [(p, p.with_name(p.name.replace("_direct", "_sobolev"))) for p in CONFIGS if p.stem.endswith("_direct")]
PAIRS = [(direct, sobolev) for direct, sobolev in PAIRS if sobolev.exists()]


@pytest.mark.parametrize("direct, sobolev", PAIRS, ids=[d.stem for d, _ in PAIRS])
def test_direct_and_sobolev_differ_only_in_readouts(direct, sobolev):
    """A Direct model and its Sobolev counterpart have the same inputs (the external field included) and the same
    interaction stack: they differ only in how the tensors are produced.

    The Direct configs had no field input until 2026-11: harmless at zero field, but at the finite fields of the
    synthetic data set a Direct model would not see the field its labels were computed at.
    """
    assert _up_to_stack(direct) == _up_to_stack(sobolev), f"{direct.name} and {sobolev.name} differ before the readouts"
