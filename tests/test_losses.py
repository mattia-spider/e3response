import jax.numpy as jnp
import jraph
import numpy as np
import optax
import pytest
from tensorial.gcnn.keys import predicted

from e3response import keys
from e3response.losses import NmrTensorLoss


def _graphs(active: np.ndarray | None, padding: np.ndarray):
    rng = np.random.default_rng(0)
    n = len(padding)
    nodes = {
        keys.NMR_TENSORS: rng.random((n, 3, 3)),
        predicted(keys.NMR_TENSORS): rng.random((n, 3, 3)),
        "mask": padding,
    }
    if active is not None:
        nodes[keys.NMR_ACTIVE] = active
    return jraph.GraphsTuple(
        n_node=jnp.array([n]),
        n_edge=jnp.array([0]),
        nodes=nodes,
        edges={},
        globals={},
        senders=jnp.array([], dtype=int),
        receivers=jnp.array([], dtype=int),
    )


def _reference(graph, sel) -> float:
    err = optax.squared_error(
        graph.nodes[predicted(keys.NMR_TENSORS)][sel], graph.nodes[keys.NMR_TENSORS][sel]
    )
    return float(err.mean())


@pytest.mark.parametrize("with_active", [True, "column", False])
def test_nmr_tensor_loss_mask(with_active):
    """Only the labelled (and not padded) atoms count; without the field, every real atom."""
    padding = np.array([1, 1, 1, 1, 1, 0], dtype=bool)
    active = np.array([1, 0, 1, 0, 1, 1], dtype=bool)
    # graph_from_ase stores per-atom scalars as a (n_node, 1) column
    given = {True: active, "column": active[:, None], False: None}[with_active]
    graph = _graphs(given, padding)
    sel = padding & active if with_active else padding
    assert np.isclose(NmrTensorLoss()(graph), _reference(graph, sel))
