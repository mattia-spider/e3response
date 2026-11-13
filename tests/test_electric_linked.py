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
import jax.experimental
from test_electric import ENERGY, A, C, CHI0, FIELD, N_NODE, R0, Z0, ToyEnergy, _apply, _batch, _per_graph_sum

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


@pytest.mark.parametrize("chain", ["branches", "fused", "fused_to_born"])
@pytest.mark.parametrize("seed", [0, 1])
def test_per_graph_field(seed, chain):
    """A per-graph (n_graph, 3) field, as the data carries it: every graph is differentiated at its
    own field, with the values a batch-shared field equal to that graph's would give."""
    fields = jax.random.normal(jax.random.PRNGKey(10 + seed), (len(N_NODE), 3))
    graph = _batch(seed)
    graph = graph._replace(globals={**graph.globals, FIELD: fields})
    L = lambda name: functools.partial(getattr(electric_linked, name))
    module = {
        "branches": functools.partial(
            electric_linked.LinkedBranches,
            branches=((L("Polarization"), L("DielectricTensor")), (L("Forces"), L("BornCharges"), L("RamanTensors"))),
        ),
        "fused": functools.partial(electric_linked.FusedFieldChain),
        "fused_to_born": functools.partial(electric_linked.FusedFieldChain, raman=False),
    }[chain]
    out = _apply(tensorial_nn.Sequential([ToyEnergy(), module]), graph)
    starts = np.concatenate([[0], np.cumsum(N_NODE)])
    for g, field in enumerate(fields):
        expected = _expected(graph, field)  # every graph at this graph's field: keep row / atoms of g
        atoms = slice(starts[g], starts[g + 1])
        for key, value in expected.items():
            if key == predicted(keys.RAMAN_TENSORS) and chain == "fused_to_born":
                assert key not in out.nodes
                continue
            got = out.nodes[key][atoms] if key in out.nodes else out.globals[key][g]
            want = value[atoms] if key in out.nodes else value[g]
            np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5, err_msg=f"{key}, graph {g}")


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


@pytest.mark.parametrize("field", [None, [0.3, -0.2, 0.5]])
def test_fused_field_chain(field):
    """One evaluation gives every quantity of the two branches."""
    graph = _graph(1, field)
    out = _apply(tensorial_nn.Sequential([ToyEnergy(), functools.partial(electric_linked.FusedFieldChain)]), graph)
    expected = _expected(graph, field)
    _check(out, expected, expected)


class _NonlinearEnergy(linen.Module):
    """A trainable energy, non-polynomial in u and ε, so that the fused and branched chains are compared on
    parameter gradients that every link contributes to."""

    @linen.compact
    def __call__(self, graph):
        u = graph.nodes[keys.POSITIONS]
        field = graph.globals[FIELD]
        graph_of = jnp.repeat(jnp.arange(len(N_NODE)), N_NODE, total_repeat_length=len(u))
        eps = jnp.broadcast_to(field, u.shape) if field.ndim == 1 else field[graph_of]  # each atom its graph's field
        h = jnp.concatenate([u, eps, u * eps, jnp.sum(eps * eps, -1, keepdims=True)], axis=-1)
        h = jnp.tanh(linen.Dense(16, param_dtype=u.dtype)(h))
        h = jnp.tanh(linen.Dense(16, param_dtype=u.dtype)(h))
        e = linen.Dense(1, param_dtype=u.dtype)(h)[:, 0]
        energy = jax.ops.segment_sum(e, jnp.repeat(jnp.arange(len(N_NODE)), N_NODE, total_repeat_length=len(u)),
                                     len(N_NODE))[:, None]
        return graph._replace(globals={**graph.globals, ENERGY: energy})


@pytest.mark.parametrize("raman", [True, False])
@pytest.mark.parametrize("field", [None, [0.3, -0.2, 0.5], "per_graph"])
def test_fused_equals_linked_branches_float64(field, raman):
    """FusedFieldChain against LinkedBranches on a trainable energy, in float64: same parameter tree, same
    outputs, same loss and same parameter gradients to rounding.  raman=False against the branches without
    RamanTensors (allegro-pol, the synthetic set)."""
    L = lambda name: functools.partial(getattr(electric_linked, name))
    born_chain = (L("Forces"), L("BornCharges")) + ((L("RamanTensors"),) if raman else ())
    branched = functools.partial(
        electric_linked.LinkedBranches, branches=((L("Polarization"), L("DielectricTensor")), born_chain)
    )
    keys_out = [predicted(k) for k in (keys.DIPOLE_MOMENT, keys.POLARIZABILITY, keys.DIELECTRIC_TENSOR,
                                       keys.BORN_CHARGES)] + ["predicted_forces"]
    keys_out += [predicted(keys.RAMAN_TENSORS)] if raman else []
    with jax.experimental.enable_x64():
        if field == "per_graph":
            graph = _batch(2)
            graph = graph._replace(globals={**graph.globals, FIELD: jax.random.normal(jax.random.PRNGKey(3), (len(N_NODE), 3))})
        else:
            graph = _graph(2, field)
        graph = jax.tree.map(lambda x: x.astype(jnp.float64) if jnp.issubdtype(x.dtype, jnp.floating) else x, graph)
        models = [tensorial_nn.Sequential([_NonlinearEnergy(), m])
                  for m in (branched, functools.partial(electric_linked.FusedFieldChain, raman=raman))]
        params = [m.init(jax.random.PRNGKey(0), graph) for m in models]
        assert jax.tree.structure(params[0]) == jax.tree.structure(params[1])
        for a, b in zip(jax.tree.leaves(params[0]), jax.tree.leaves(params[1])):
            np.testing.assert_array_equal(a, b)
        targets = {k: jax.random.normal(jax.random.PRNGKey(i), v.shape, jnp.float64)
                   for i, (k, v) in enumerate(_outputs(models[0].apply(params[0], graph), keys_out).items())}

        def loss(p, m):
            out = _outputs(m.apply(p, graph), keys_out)
            return sum(jnp.sum((out[k] - targets[k]) ** 2) for k in targets)

        outs = [_outputs(m.apply(params[0], graph), keys_out) for m in models]
        for k in keys_out:
            np.testing.assert_allclose(outs[1][k], outs[0][k], rtol=1e-12, atol=1e-12, err_msg=k)
        (l0, g0), (l1, g1) = (jax.value_and_grad(loss)(params[0], m) for m in models)
        np.testing.assert_allclose(l1, l0, rtol=1e-12)
        for a, b in zip(jax.tree.leaves(g0), jax.tree.leaves(g1)):
            np.testing.assert_allclose(b, a, rtol=1e-10, atol=1e-12)


def _outputs(graph, names):
    return {k: jnp.asarray(graph.nodes[k] if k in graph.nodes else graph.globals[k]) for k in names}
