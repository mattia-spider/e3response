"""The susceptibility chain of :mod:`e3response.magnetic`, M = -∂E/∂B and χ = ∂M/∂B, against jax's
own derivatives of a toy energy that couples μ, B and the geometry, on a batch with a padding graph.

Checked on their own and side by side with the σ chain in a `LinkedBranches`, the configuration of
the NMR model with a susceptibility label: σ must be what the σ chain gives alone, and χ the second
field derivative of the same energy -- at zero field and at a field already in the graph.
"""

import functools

from flax import linen
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tensorial import nn as tensorial_nn

from e3response import electric_linked, magnetic
from test_magnetic import N_NODE, _batch, _field_on_nodes, _node_field_matrix

SCALE = 2.5


class ToyEnergy(linen.Module):
    """Per-graph energy with a mixed μ-B term (σ), and a B-B term that depends on the geometry (χ)."""

    @linen.compact
    def __call__(self, graph):
        A, gidx = _node_field_matrix(graph)
        K = linen.Dense(9)(graph.nodes["positions"]).reshape(-1, 3, 3)
        B = _field_on_nodes(graph, gidx)
        coupling = jnp.einsum("ni,nij,nj->n", graph.nodes["mu"], A, B)
        e = coupling + 0.3 * coupling**2 + jnp.einsum("ni,nij,nj->n", B, K, B) + 0.1 * jnp.sum(jnp.tanh(B) ** 3, -1)
        energy = jax.ops.segment_sum(e, gidx, len(N_NODE))
        return graph._replace(globals={**graph.globals, "E": energy[:, None]})


M = functools.partial(magnetic.Magnetization, energy_key="E", field_key="B_ext", mu_key="mu", out_key="M")
CHI = functools.partial(magnetic.MagneticSusceptibility, magnetization_key="M", field_key="B_ext", scale=SCALE,
                        out_key="chi")
B_IND = functools.partial(magnetic.InducedMagneticField, energy_key="E", mu_key="mu", out_key="B_ind")
SIGMA = functools.partial(magnetic.MagneticShieldingTensor, B_ind="B_ind", B_ext="B_ext", out_key="sigma",
                          B_ext_at_graph=True)


def _graph(field, per_graph=False):
    """The batch without μ (the chains insert it), with no field, a shared (3,) one, or -- as the
    datasets carry it -- one row per graph, all equal."""
    graph = _batch(0)
    nodes = {k: v for k, v in graph.nodes.items() if k != "mu"}
    if field is None:
        return graph._replace(nodes=nodes, globals={})
    field = jnp.asarray(field)
    if per_graph:
        field = jnp.broadcast_to(field, (len(N_NODE), 3))
    return graph._replace(nodes=nodes, globals={"B_ext": field})


def _energy_of_field(params, graph, mu=None):
    """The toy energy per graph as a function of one field shared by the batch."""
    n = graph.nodes["positions"].shape[0]
    mu = jnp.zeros((n, 3)) if mu is None else mu

    def energy(field):
        g = graph._replace(nodes={**graph.nodes, "mu": mu}, globals={**graph.globals, "B_ext": field})
        return ToyEnergy().apply(params, g).globals["E"][:, 0]

    return energy


def _apply(model, graph):
    with jax.default_matmul_precision("highest"):
        params = model.init(jax.random.PRNGKey(0), graph)
        return model.apply(params, graph), params


def _energy_params(params):
    """The toy energy's parameters inside a chain (it is the innermost module)."""
    found = []

    def walk(tree):
        if isinstance(tree, dict):
            if any(k.startswith("Dense") for k in tree):
                found.append(tree)
                return
            for v in tree.values():
                walk(v)

    walk(params)
    assert len(found) == 1, "one toy energy expected"
    return {"params": found[0]}


@pytest.mark.parametrize("field", [None, [0.3, -0.2, 0.5]])
def test_susceptibility_chain(field):
    graph = _graph(field)
    out, params = _apply(tensorial_nn.Sequential([ToyEnergy(), M, CHI]), graph)
    energy = _energy_of_field(_energy_params(params), graph)
    at = jnp.zeros(3) if field is None else jnp.asarray(field)
    with jax.default_matmul_precision("highest"):
        moment = -jax.jacfwd(energy)(at)
        chi = -SCALE * jax.jacfwd(jax.jacfwd(energy))(at)
    np.testing.assert_allclose(out.globals["M"], moment, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(out.globals["chi"], chi, rtol=1e-5, atol=1e-6)
    assert np.abs(chi).max() > 1e-2  # not trivially zero
    np.testing.assert_allclose(chi, np.swapaxes(chi, 1, 2), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("field", [[0.0, 0.0, 0.0], [0.3, -0.2, 0.5]])
def test_sigma_and_chi_branches(field):
    """σ and χ from one energy, side by side: σ as the σ chain alone gives it, χ = -∂²E/∂B².
    The field is in the graph one row per graph, as the C-S-H dataset carries it."""
    graph = _graph(field, per_graph=True)
    branches = functools.partial(electric_linked.LinkedBranches, branches=((B_IND, SIGMA), (M, CHI)))
    out, params = _apply(tensorial_nn.Sequential([ToyEnergy(), branches]), graph)
    alone, _ = _apply(tensorial_nn.Sequential([ToyEnergy(), B_IND, SIGMA]), graph)
    np.testing.assert_allclose(out.nodes["sigma"], alone.nodes["sigma"], rtol=1e-5, atol=1e-6)
    assert np.abs(np.asarray(alone.nodes["sigma"])).max() > 1e-3

    energy = _energy_of_field(_energy_params(params), _graph(None))
    at = jnp.asarray(field)
    with jax.default_matmul_precision("highest"):
        chi = -SCALE * jax.jacfwd(jax.jacfwd(energy))(at)
    np.testing.assert_allclose(out.globals["chi"], chi, rtol=1e-5, atol=1e-6)


def test_appending_chi_to_the_sigma_chain_loses_the_field():
    """Why the two chains must be branches: after σ, which evaluates at a fixed field, the energy
    no longer depends on B and a susceptibility differentiated through it is exactly zero."""
    graph = _graph(None)
    fixed_sigma = functools.partial(SIGMA, B_ext_at_graph=False)
    out, _ = _apply(tensorial_nn.Sequential([ToyEnergy(), B_IND, fixed_sigma, M, CHI]), graph)
    assert np.abs(np.asarray(out.globals["chi"])).max() == 0.0


def test_field_product_squared_block():
    """B (x) B appended as |B|² (0e) and its symmetric traceless part (2e); off, nothing changes."""
    import e3nn_jax as e3j
    import jraph

    from e3response.nodewise_product import NodewiseFieldProduct

    rng = np.random.default_rng(0)
    n = 4
    b = jnp.asarray(rng.normal(size=3))
    mu = jnp.asarray(rng.normal(size=(n, 3)))
    attrs = e3j.concatenate([e3j.IrrepsArray("2x0e", jnp.asarray(rng.normal(size=(n, 2)))),
                             e3j.IrrepsArray("1e", jnp.broadcast_to(b, (n, 3))), e3j.IrrepsArray("1e", mu)])
    graph = jraph.GraphsTuple(nodes={"attributes": attrs}, edges=None, senders=None, receivers=None,
                              n_node=jnp.array([n]), n_edge=jnp.array([0]), globals=None)
    plain = NodewiseFieldProduct(scale=2.0).apply({}, graph).nodes["attributes"]
    squared = NodewiseFieldProduct(scale=2.0, field_squared=True, squared_scale=3.0).apply({}, graph).nodes["attributes"]
    np.testing.assert_allclose(squared.array[:, : plain.irreps.dim], plain.array)
    extra = squared.array[:, plain.irreps.dim:]
    expected = 3.0 * e3j.tensor_product(e3j.IrrepsArray("1e", jnp.broadcast_to(b, (n, 3))),
                                        e3j.IrrepsArray("1e", jnp.broadcast_to(b, (n, 3))),
                                        filter_ir_out=["0e", "2e"])
    assert expected.irreps == e3j.Irreps("0e + 2e")
    np.testing.assert_allclose(extra, expected.array, rtol=1e-6)
    np.testing.assert_allclose(extra[:, 0], 3.0 * float(jnp.dot(b, b)) / np.sqrt(3.0), rtol=1e-5)  # e3nn normalisation


def test_species_term_in_label_units():
    """With unit = scale, the per-species term adds exactly Σ α_Z I to χ, whatever the scale."""
    graph = _graph(None)
    n = graph.nodes["positions"].shape[0]
    graph = graph._replace(nodes={**graph.nodes, "species": jnp.arange(n) % 2})
    alpha = (-3.0, 5.0)
    for scale in (1.0, 1000.0):

        class Energy(linen.Module):
            @linen.compact
            def __call__(self, g, scale=scale):
                g = ToyEnergy()(g)
                return electric_linked.SpeciesPolarizability(
                    num_species=2, electric_field_key="B_ext", energy_key="E", init_values=alpha, unit=scale)(g)

        chi = functools.partial(CHI, scale=scale)
        out, params = _apply(tensorial_nn.Sequential([Energy(), M, chi]), graph)
        energy = _energy_of_field(_energy_params(params), graph)
        with jax.default_matmul_precision("highest"):
            toy = -scale * jax.jacfwd(jax.jacfwd(energy))(jnp.zeros(3))
        counts = np.stack([np.bincount(np.repeat(np.arange(len(N_NODE)), N_NODE)[np.arange(n) % 2 == s],
                                       minlength=len(N_NODE)) for s in (0, 1)], 1)
        extra = (counts @ np.asarray(alpha))[:, None, None] * np.eye(3)
        np.testing.assert_allclose(out.globals["chi"], toy + extra, rtol=1e-4, atol=1e-3 * scale)
