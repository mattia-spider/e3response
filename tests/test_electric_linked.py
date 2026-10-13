"""The linked electric response chain against analytic derivatives of a toy energy.

The toy energy is the one from ``test_electric``,

    E_g = Σ_{i∈g} [ -ε·Z0·u_i - ½ ε·χ0·ε - ½ (ε·R0·ε)(a·u_i) + c |u_i|² ]

for which, at a field ε shared by the batch,

    F_i          = -∂E/∂u_i    = Z0ᵀε + ½ (ε·R0·ε) a - 2c u_i
    d_g          = -∂E_g/∂ε    = Z0 Σu + n_g χ0 ε + (Σ a·u) R0 ε
    χ_g          = ∂d_g/∂ε     = n_g χ0 + (Σ a·u) R0,         ε_r = I + 4π χ_g
    Z*_{i,αγ}    = ∂F_iγ/∂ε_α  = Z0[α, γ] + (R0 ε)_α a_γ
    R_{i,γαβ}    = ∂Z*_{i,αγ}/∂ε_β = a_γ R0[α, β]

The chain is built as a model config builds it -- a `tensorial.nn.Sequential` of partial modules,
each wrapping the previous one -- and checked at zero and at a non-zero field, which the Born
charges depend on, so every link must evaluate at the field it is handed.
"""

import functools

from flax import linen
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tensorial import nn as tensorial_nn

from e3response import electric_linked, keys
from e3response.electric_linked import predicted
from test_electric import A, C, CHI0, FIELD, N_NODE, R0, Z0, ToyEnergy, _apply, _batch, _per_graph_sum

FULL_CHAIN = ("Polarization", "DielectricTensor", "Forces", "BornCharges", "RamanTensors")


def _chain(*names):
    layers = [ToyEnergy()] + [functools.partial(getattr(electric_linked, n)) for n in names]
    return tensorial_nn.Sequential(layers)


def _graph(seed, field):
    graph = _batch(seed)
    if field is not None:  # one field for the whole batch, as the chain inserts it
        graph = graph._replace(globals={**graph.globals, FIELD: jnp.asarray(field)})
    return graph


def _expected(graph, field):
    eps = jnp.zeros(3) if field is None else jnp.asarray(field)
    u = graph.nodes[keys.POSITIONS]
    n = jnp.asarray(N_NODE, dtype=float)
    sum_u, sum_au = _per_graph_sum(u), _per_graph_sum(u @ A)
    chi = n[:, None, None] * CHI0 + sum_au[:, None, None] * R0
    return {
        predicted(keys.FORCES): Z0.T @ eps + 0.5 * (eps @ R0 @ eps) * A - 2.0 * C * u,
        predicted(keys.DIPOLE_MOMENT): sum_u @ Z0.T + n[:, None] * (CHI0 @ eps) + sum_au[:, None] * (R0 @ eps),
        predicted(keys.DIELECTRIC_TENSOR): jnp.eye(3) + 4.0 * jnp.pi * chi,
        predicted(keys.BORN_CHARGES): jnp.broadcast_to(Z0 + jnp.outer(R0 @ eps, A), (len(u), 3, 3)),
        predicted(keys.RAMAN_TENSORS): jnp.broadcast_to(jnp.einsum("g,ab->gab", A, R0), (len(u), 3, 3, 3)),
    }


def _check(out, expected, keys_to_check):
    for key in keys_to_check:
        got = out.nodes[key] if key in out.nodes else out.globals[key]
        np.testing.assert_allclose(got, expected[key], rtol=1e-5, atol=1e-5, err_msg=key)


@pytest.mark.parametrize("field", [None, [0.3, -0.2, 0.5]])
@pytest.mark.parametrize("seed", [0, 1])
def test_full_chain_as_sequential(seed, field):
    graph = _graph(seed, field)
    out = _apply(_chain(*FULL_CHAIN), graph)
    expected = _expected(graph, field)
    _check(out, expected, expected)
    # No cell in the toy graphs, so the polarization is the dipole moment itself
    np.testing.assert_allclose(out.globals[predicted(keys.POLARIZATION)],
                               expected[predicted(keys.DIPOLE_MOMENT)], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("field", [None, [0.3, -0.2, 0.5]])
def test_born_and_raman_chain(field):
    """The chain a model with Born and Raman labels only would use."""
    graph = _graph(0, field)
    out = _apply(_chain("Forces", "BornCharges", "RamanTensors"), graph)
    expected = _expected(graph, field)
    _check(out, expected, [predicted(keys.FORCES), predicted(keys.BORN_CHARGES), predicted(keys.RAMAN_TENSORS)])


class _KnownBornCharges(linen.Module):
    """Stands in for the Born charges link: Z*_{i,αγ}(ε) = Σ_β W[α,γ,β] ε_β on every atom."""

    weights: jnp.ndarray

    def __call__(self, graph):
        field = graph.globals[FIELD]
        born = jnp.einsum("agb,b->ag", self.weights, field)
        n = len(graph.nodes[keys.POSITIONS])
        return graph._replace(nodes={**graph.nodes, predicted(keys.BORN_CHARGES): jnp.broadcast_to(born, (n, 3, 3))})


def test_raman_differentiates_the_born_charges_it_wraps():
    """Sobolev link: R is the field derivative of the Born charges handed to it, not of an energy."""
    weights = jax.random.normal(jax.random.PRNGKey(3), (3, 3, 3))
    graph = _graph(0, [0.1, 0.2, -0.3])
    out = _apply(electric_linked.RamanTensors(_KnownBornCharges(weights)), graph)
    n = len(graph.nodes[keys.POSITIONS])
    # R_{i,γαβ} = ∂Z*_{i,αγ}/∂ε_β = W[α, γ, β]
    expected = jnp.broadcast_to(jnp.transpose(weights, (1, 0, 2)), (n, 3, 3, 3))
    np.testing.assert_allclose(out.nodes[predicted(keys.RAMAN_TENSORS)], expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("field", [None, [0.3, -0.2, 0.5]])
@pytest.mark.parametrize("seed", [0, 1])
def test_dipole_route_chain(seed, field):
    """Born and Raman from the dipole side, in the order of the lab configs (dielectric, then Born,
    then Raman): Z* = ∂d/∂u and R = ∂χ/∂u, the same values as the force route."""
    graph = _graph(seed, field)
    out = _apply(_chain("Polarization", "DielectricTensor", "BornChargesFromDipole", "RamanTensorsFromPolarizability"), graph)
    expected = _expected(graph, field)
    _check(out, expected, [predicted(keys.DIPOLE_MOMENT), predicted(keys.DIELECTRIC_TENSOR),
                           predicted(keys.BORN_CHARGES), predicted(keys.RAMAN_TENSORS)])


@pytest.mark.parametrize("route", [("Forces", "BornCharges", "RamanTensors"),
                                   ("Polarization", "DielectricTensor", "BornChargesFromDipole", "RamanTensorsFromPolarizability")])
def test_raman_per_volume(route):
    """With per_volume, each atom's Raman tensor is divided by the volume of its own cell."""
    graph = _graph(0, None)
    lengths = jnp.array([2.0, 3.0, 1.5, 2.5])
    cells = jax.vmap(lambda length: length * jnp.eye(3))(lengths)
    graph = graph._replace(globals={**graph.globals, "cell": cells})
    layers = [ToyEnergy()] + [functools.partial(getattr(electric_linked, n)) for n in route[:-1]]
    layers.append(functools.partial(getattr(electric_linked, route[-1]), per_volume=True))
    out = _apply(tensorial_nn.Sequential(layers), graph)
    omega = jnp.repeat(lengths**3, N_NODE)
    expected = _expected(graph, None)[predicted(keys.RAMAN_TENSORS)] / omega[:, None, None, None]
    np.testing.assert_allclose(out.nodes[predicted(keys.RAMAN_TENSORS)], expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("field", [None, [0.3, -0.2, 0.5]])
def test_linked_branches(field):
    """The dielectric and Born/Raman chains side by side on one energy give every quantity."""
    graph = _graph(1, field)
    L = lambda name: functools.partial(getattr(electric_linked, name))
    branches = functools.partial(
        electric_linked.LinkedBranches,
        branches=((L("Polarization"), L("DielectricTensor")), (L("Forces"), L("BornCharges"), L("RamanTensors"))),
    )
    out = _apply(tensorial_nn.Sequential([ToyEnergy(), branches]), graph)
    expected = _expected(graph, field)
    _check(out, expected, expected)


def test_species_polarizability_adds_only_to_chi():
    """-½ Σ α_Z |ε|² adds Σ α_Z I to χ and leaves the Born charges and Raman tensors unchanged."""
    graph = _graph(0, [0.3, -0.2, 0.5])
    n = len(graph.nodes[keys.POSITIONS])
    species = jnp.arange(n) % 2
    graph = graph._replace(nodes={**graph.nodes, "species": species})
    alpha = 0.7
    polarizability = functools.partial(electric_linked.SpeciesPolarizability, num_species=2, init_value=alpha)
    L = lambda name: functools.partial(getattr(electric_linked, name))
    branches = functools.partial(
        electric_linked.LinkedBranches,
        branches=((L("Polarization"), L("DielectricTensor")), (L("Forces"), L("BornCharges"), L("RamanTensors"))),
    )

    class Energy(linen.Module):
        @linen.compact
        def __call__(self, g):
            return polarizability()(ToyEnergy()(g))

    out = _apply(tensorial_nn.Sequential([Energy(), branches]), graph)
    expected = _expected(graph, [0.3, -0.2, 0.5])
    extra = 4.0 * jnp.pi * alpha * jnp.asarray(N_NODE, dtype=float)[:, None, None] * jnp.eye(3)
    np.testing.assert_allclose(out.globals[predicted(keys.DIELECTRIC_TENSOR)],
                               expected[predicted(keys.DIELECTRIC_TENSOR)] + extra, rtol=1e-5, atol=1e-5)
    _check(out, expected, [predicted(keys.BORN_CHARGES), predicted(keys.RAMAN_TENSORS)])
