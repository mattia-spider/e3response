"""The batch-wide external field of :class:`e3response.magnetic.MagneticShieldingTensor`.

By default B_ext is differentiated as one (3,) field shared by the whole batch, which is exact
only because graphs in a batch are independent.  These tests pin that down against the per-graph
derivative on a small hand-built batch -- several graphs plus a padding graph -- with toy models
that couple μ, B_ext and the geometry and reduce per graph, so any leak between graphs shows up.
"""

from flax import linen
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import pytest

from e3response.magnetic import InducedMagneticField, MagneticShieldingTensor

N_NODE = np.array([4, 2, 5, 3])  # the last graph plays the padding graph


def _batch(seed: int, common_field: bool = False) -> jraph.GraphsTuple:
    n = int(N_NODE.sum())
    key_pos, key_field = jax.random.split(jax.random.PRNGKey(seed))
    field = jax.random.normal(key_field, (len(N_NODE), 3))
    if common_field:
        field = jnp.broadcast_to(field[0], field.shape)
    return jraph.GraphsTuple(
        nodes={"positions": jax.random.normal(key_pos, (n, 3)), "mu": jnp.zeros((n, 3))},
        edges=None,
        senders=jnp.zeros((0,), dtype=jnp.int32),
        receivers=jnp.zeros((0,), dtype=jnp.int32),
        n_node=jnp.asarray(N_NODE),
        n_edge=jnp.zeros(len(N_NODE), dtype=jnp.int32),
        globals={"B_ext": field},
    )


def _graph_index(graph):
    n = len(graph.nodes["mu"])
    return jnp.repeat(jnp.arange(len(N_NODE)), graph.n_node, total_repeat_length=n)


def _field_on_nodes(graph, gidx):
    """B_ext on every node, reading a per-graph (n_graphs, 3) field or a batch-wide (3,) one the way
    tensorial's NodewiseEncoding does."""
    field = graph.globals["B_ext"]
    return jnp.broadcast_to(field, (len(gidx), 3)) if field.ndim == 1 else field[gidx]


def _node_field_matrix(graph):
    """A per-node 3x3 matrix from the node's position and its graph's mean position."""
    gidx = _graph_index(graph)
    pos = graph.nodes["positions"]
    centre = jax.ops.segment_sum(pos, gidx, len(N_NODE)) / graph.n_node[:, None]
    feats = jnp.concatenate([pos, pos - centre[gidx]], axis=-1)
    return linen.Dense(9)(jnp.tanh(linen.Dense(16)(feats))).reshape(-1, 3, 3), gidx


class ToyEnergy(linen.Module):
    """Per-graph energy, nonlinear in μ and B_ext and coupled to the geometry."""

    @linen.compact
    def __call__(self, graph):
        A, gidx = _node_field_matrix(graph)
        B = _field_on_nodes(graph, gidx)
        coupling = jnp.einsum("ni,nij,nj->n", graph.nodes["mu"], A, B)
        e = coupling + 0.3 * coupling**2 + 0.1 * jnp.sum(jnp.tanh(B) ** 2, axis=-1)
        energy = jax.ops.segment_sum(e, gidx, len(N_NODE))
        return graph._replace(globals={**graph.globals, "E": energy[:, None]})


class ToyInducedField(linen.Module):
    """A direct, nonlinear readout of B_ind, as in the first-order model."""

    @linen.compact
    def __call__(self, graph):
        A, gidx = _node_field_matrix(graph)
        b_ind = jnp.tanh(jnp.einsum("nij,nj->ni", A, _field_on_nodes(graph, gidx)))
        return graph._replace(nodes={**graph.nodes, "B_ind": b_ind})


def _induced():
    return InducedMagneticField(energy_fn=ToyEnergy(), energy_key="E", mu_key="mu", out_key="B_ind")


def _shielding(b_ind_fn, per_graph, b_ext_at_graph):
    return MagneticShieldingTensor(
        B_ind_fn=b_ind_fn,
        B_ind="B_ind",
        B_ext="B_ext",
        out_key="sigma",
        B_ext_at_graph=b_ext_at_graph,
        per_graph=per_graph,
    )


def _compare(make_b_ind_fn, seed, b_ext_at_graph):
    # With B_ext_at_graph the shared field takes one field for the whole batch, so the graphs must
    # agree on it for the per-graph derivative to be the same quantity
    graph = _batch(seed, common_field=b_ext_at_graph)
    shared = _shielding(make_b_ind_fn(), False, b_ext_at_graph)
    per_graph = _shielding(make_b_ind_fn(), True, b_ext_at_graph)
    with jax.default_matmul_precision("highest"):
        params = shared.init(jax.random.PRNGKey(seed), graph)
        new = shared.apply(params, graph).nodes["sigma"]
        old = per_graph.apply(params, graph).nodes["sigma"]
    np.testing.assert_allclose(new, old, rtol=1e-5, atol=1e-7)
    assert np.abs(np.asarray(new)).max() > 1e-3  # not trivially zero


@pytest.mark.parametrize("b_ext_at_graph", [False, True])
@pytest.mark.parametrize("seed", [0, 1])
def test_second_order_shielding_matches_per_graph(seed, b_ext_at_graph):
    _compare(_induced, seed, b_ext_at_graph)


@pytest.mark.parametrize("b_ext_at_graph", [False, True])
@pytest.mark.parametrize("seed", [0, 1])
def test_first_order_shielding_matches_per_graph(seed, b_ext_at_graph):
    _compare(ToyInducedField, seed, b_ext_at_graph)


def test_shielding_ignores_other_graphs():
    """Perturbing one graph's inputs must leave every other graph's tensors untouched."""
    graph = _batch(0)
    model = _shielding(_induced(), False, False)
    params = model.init(jax.random.PRNGKey(0), graph)
    base = model.apply(params, graph).nodes["sigma"]

    moved = graph.nodes["positions"].at[:4].add(0.5)  # graph 0 only
    shifted = model.apply(params, graph._replace(nodes={**graph.nodes, "positions": moved}))
    np.testing.assert_allclose(shifted.nodes["sigma"][4:], base[4:], rtol=1e-6, atol=1e-8)
    assert not np.allclose(shifted.nodes["sigma"][:4], base[:4])
