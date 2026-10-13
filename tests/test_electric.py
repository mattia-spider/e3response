"""The electric response modules against analytic derivatives of a toy energy.

The toy energy of each graph is

    E_g = Σ_{i∈g} [ -ε·Z0·u_i - ½ ε·χ0·ε - ½ (ε·R0·ε)(a·u_i) + c |u_i|² ]

so that, at zero field,

    P_g           = -∂E_g/∂ε           = Z0 Σ_{i∈g} u_i
    ε_r,g         = I - (∂²E_g/∂ε²)/ε0  = I + 4π (n_g χ0 + Σ_{i∈g} (a·u_i) R0)
    Z*_{i,αγ}     = -∂²E/∂u_iγ∂ε_α      = Z0[α, γ]
    R_{i,γαβ}     = -∂³E/∂u_iγ∂ε_α∂ε_β  = a_γ R0[α, β]

Z0 is not symmetric, so swapping the α and γ indices of the Born charges shows up.  The batch has
graphs of different sizes and ends with a padding graph, and the answers are independent of how
the derivatives are taken, so they also pin the choice of AD modes.
"""

import math

from flax import linen
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import pytest

from e3response import electric, keys

N_NODE = np.array([3, 5, 2, 4])  # the last graph plays the padding graph
N_GRAPH = len(N_NODE)
FIELD = keys.EXTERNAL_ELECTRIC_FIELD
ENERGY = "predicted_energy"

Z0 = jnp.array([[1.0, 0.2, -0.3], [0.1, 2.0, 0.4], [-0.5, 0.3, 1.5]])
CHI0 = jnp.array([[0.8, 0.1, 0.0], [0.1, 1.2, -0.2], [0.0, -0.2, 0.5]])
R0 = jnp.array([[0.3, -0.1, 0.2], [-0.1, 0.6, 0.05], [0.2, 0.05, -0.4]])
A = jnp.array([0.7, -1.1, 0.4])
C = 0.3


def _graph_index(n_nodes: int):
    return jnp.repeat(jnp.arange(N_GRAPH), N_NODE, total_repeat_length=n_nodes)


class ToyEnergy(linen.Module):
    """The toy energy above, reading the field the way tensorial's node encodings do: a (3,)
    field is shared by the batch, an (n_graph, 3) one is per graph."""

    @linen.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        u = graph.nodes[keys.POSITIONS]
        gidx = _graph_index(len(u))
        field = graph.globals[FIELD]
        eps = jnp.broadcast_to(field, u.shape) if field.ndim == 1 else field[gidx]
        e = (
            -jnp.einsum("na,ag,ng->n", eps, Z0, u)
            - 0.5 * jnp.einsum("na,ab,nb->n", eps, CHI0, eps)
            - 0.5 * jnp.einsum("na,ab,nb->n", eps, R0, eps) * (u @ A)
            + C * jnp.sum(u**2, axis=-1)
        )
        energy = jax.ops.segment_sum(e, gidx, N_GRAPH)[:, None]
        return graph._replace(globals={**graph.globals, ENERGY: energy})


def _batch(seed: int = 0) -> jraph.GraphsTuple:
    n = int(N_NODE.sum())
    return jraph.GraphsTuple(
        nodes={keys.POSITIONS: jax.random.normal(jax.random.PRNGKey(seed), (n, 3))},
        edges=None,
        senders=jnp.zeros((0,), dtype=jnp.int32),
        receivers=jnp.zeros((0,), dtype=jnp.int32),
        n_node=jnp.asarray(N_NODE),
        n_edge=jnp.zeros(N_GRAPH, dtype=jnp.int32),
        globals={FIELD: jnp.zeros((N_GRAPH, 3))},
    )


def _apply(module, graph):
    with jax.default_matmul_precision("highest"):
        return module.apply(module.init(jax.random.PRNGKey(0), graph), graph)


def _per_graph_sum(values):
    return jax.ops.segment_sum(values, _graph_index(len(values)), N_GRAPH)


@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("seed", [0, 1])
def test_polarization(seed, mode):
    graph = _batch(seed)
    out = _apply(electric.Polarization(ToyEnergy(), energy_key=ENERGY, mode=mode), graph)
    expected = _per_graph_sum(graph.nodes[keys.POSITIONS]) @ Z0.T
    np.testing.assert_allclose(out.globals[electric.predicted(keys.POLARIZATION)], expected,
                               rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("seed", [0, 1])
def test_dielectric_tensor(seed, mode):
    graph = _batch(seed)
    out = _apply(electric.DielectricTensor(ToyEnergy(), energy_key=ENERGY, mode=mode), graph)
    a_dot_u = _per_graph_sum(graph.nodes[keys.POSITIONS] @ A)
    susceptibility = jnp.asarray(N_NODE, dtype=float)[:, None, None] * CHI0 + a_dot_u[:, None, None] * R0
    expected = jnp.eye(3) + 4.0 * math.pi * susceptibility
    np.testing.assert_allclose(out.globals[electric.predicted(keys.DIELECTRIC_TENSOR)], expected,
                               rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("seed", [0, 1])
def test_born_charges(seed):
    graph = _batch(seed)
    out = _apply(electric.BornCharges(ToyEnergy(), energy_key=ENERGY), graph)
    n = len(graph.nodes[keys.POSITIONS])
    np.testing.assert_allclose(out.nodes[electric.predicted(keys.BORN_CHARGES)],
                               jnp.broadcast_to(Z0, (n, 3, 3)), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("seed", [0, 1])
def test_raman_tensors(seed):
    graph = _batch(seed)
    out = _apply(electric.RamanTensors(ToyEnergy(), energy_key=ENERGY), graph)
    n = len(graph.nodes[keys.POSITIONS])
    expected = jnp.broadcast_to(jnp.einsum("g,ab->gab", A, R0), (n, 3, 3, 3))
    np.testing.assert_allclose(out.nodes[electric.predicted(keys.RAMAN_TENSORS)], expected,
                               rtol=1e-5, atol=1e-6)
