"""Hubbard occupation matrices as model inputs, and the energy terms built on them.

The occupation matrices come from Quantum ESPRESSO, in its basis of real spherical harmonics
(for d: z^2, xz, yz, x^2-y^2, xy, with the (-1)^m sign).  :class:`OccupationMatrix` moves them to
the e3nn basis and decomposes the symmetric matrix into irreps (0e + 2e + 4e for d).  The change of
basis follows Austin's ``occupation.py`` (Austin_models).

The energy of a state is split as E = E_noU + E_U, where E_U, the Hubbard energy, is known in closed
form from the occupation matrices (:class:`HubbardEnergy`), so the network only has to learn E_noU.
Swapping spin up and down on every site gives the same state; :class:`SpinFlipAverage` makes the
learned part invariant to it.
"""

from collections.abc import Sequence

import e3nn_jax as e3j
from flax import linen
import jax
import jax.numpy as jnp
import jraph
import tensorial
from tensorial.gcnn.keys import predicted

from . import keys

__all__ = (
    "HubbardEnergy",
    "NodewiseMask",
    "OccupationMatrix",
    "SpinFlipAverage",
    "qe_to_e3nn_cob",
)

ZXY_TO_XYZ = jnp.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _e3nn_to_standard_cob(l: int) -> jax.Array:
    """Change of basis from e3nn's real spherical harmonics to the standard (Wikipedia) ones."""
    return e3j.Irrep(l, (-1) ** l).D_from_matrix(ZXY_TO_XYZ)


def _qe_to_standard_cob(l: int) -> jax.Array:
    """Change of basis from QE's real spherical harmonics to the standard ones: QE orders m as
    0, 1, -1, 2, -2, ... and carries the Condon-Shortley phase (-1)^m, removed here."""
    ms = jnp.array([0] + [m for k in range(1, l + 1) for m in (k, -k)])
    return ((-1) ** ms * jax.nn.one_hot(jnp.argsort(ms), 2 * l + 1)).T


def qe_to_e3nn_cob(l: int) -> jax.Array:
    """Orthogonal change of basis from QE's real spherical harmonics of degree ``l`` to e3nn's:
    ``Y_e3nn = Y_qe @ cob``."""
    return _qe_to_standard_cob(l) @ _e3nn_to_standard_cob(l).T


class OccupationMatrix(tensorial.Attr):
    """A QE occupation matrix of one spin channel, (2l+1, 2l+1) per atom, as the irreps of a
    symmetric rank-2 tensor of degree ``l`` (0e + 2e + 4e for d).

    Atoms without Hubbard states carry a zero matrix, which stays zero.
    """

    change_of_basis: jax.Array
    qe_change_of_basis: jax.Array

    def __init__(self, l: int = 2) -> None:
        basis = e3j.reduced_tensor_product_basis("ij=ji", i=e3j.Irrep(l, (-1) ** l))
        self.change_of_basis = basis.array
        self.qe_change_of_basis = qe_to_e3nn_cob(l)
        super().__init__(basis.irreps)

    def create_tensor(self, value: jax.Array) -> e3j.IrrepsArray:
        qe_cob = jax.lax.stop_gradient(self.qe_change_of_basis)
        cob = jax.lax.stop_gradient(self.change_of_basis)
        value = qe_cob.T @ value @ qe_cob
        return e3j.IrrepsArray(self.irreps, jnp.einsum("ijz,...ij->...z", cob, value))


def _graph_ids(graph: jraph.GraphsTuple) -> jax.Array:
    n_nodes = graph.nodes[keys.POSITIONS].shape[0]
    return jnp.repeat(jnp.arange(graph.n_node.shape[0]), graph.n_node, total_repeat_length=n_nodes)


class HubbardEnergy(linen.Module):
    """The Hubbard energy of each structure, in the simplified rotationally invariant form of
    Dudarev used by Quantum ESPRESSO:

        E_U = sum_I U_I / 2 sum_sigma Tr[n_I^sigma - n_I^sigma n_I^sigma]

    with U in the units of the energy (eV).  It is written to ``out_field`` and, if ``add_to`` is
    given, added to that global, so that a network predicting E - E_U gives the total energy.
    """

    up_field: str = keys.OCCUPATION_UP
    down_field: str = keys.OCCUPATION_DOWN
    u_field: str = keys.HUBBARD_U
    out_field: str = predicted(keys.HUBBARD_ENERGY)
    add_to: str | None = predicted(keys.TOTAL_ENERGY)

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        nodes = graph.nodes
        per_site = 0.0
        for field in (self.up_field, self.down_field):
            n = nodes[field]
            per_site += jnp.trace(n - n @ n, axis1=-2, axis2=-1)
        per_site = 0.5 * jnp.reshape(nodes[self.u_field], per_site.shape) * per_site
        energy = jax.ops.segment_sum(per_site, _graph_ids(graph), graph.n_node.shape[0])

        globals_ = dict(graph.globals)
        globals_[self.out_field] = energy
        if self.add_to is not None:
            total = globals_[self.add_to]
            globals_[self.add_to] = total + jnp.reshape(energy, total.shape)
        return graph._replace(globals=globals_)


class NodewiseMask(linen.Module):
    """Multiplies a per-atom field by a per-atom mask, e.g. to keep the energy of the Hubbard
    sites only."""

    field: str
    mask: str = keys.HUBBARD_SITE

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        nodes = dict(graph.nodes)
        value = nodes[self.field]
        array = value.array if isinstance(value, e3j.IrrepsArray) else value
        mask = jnp.reshape(nodes[self.mask], (-1,) + (1,) * (array.ndim - 1)).astype(array.dtype)
        nodes[self.field] = value * mask if isinstance(value, e3j.IrrepsArray) else array * mask
        return graph._replace(nodes=nodes)


def _copy(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    """A graph with its own node, edge and global dicts: tensorial's layers write into them."""
    return graph._replace(
        nodes=dict(graph.nodes), edges=dict(graph.edges), globals=dict(graph.globals)
    )


class SpinFlipAverage(linen.Module):
    """Runs ``model`` on the graph and on the graph with spin up and down swapped on every atom,
    and averages ``fields`` over the two, which makes them invariant to the swap.

    The two calls share the parameters.  The other outputs are those of the first call.
    """

    model: linen.Module
    fields: Sequence[str] = (f"globals.{predicted(keys.TOTAL_ENERGY)}",)
    up_field: str = keys.OCCUPATION_UP
    down_field: str = keys.OCCUPATION_DOWN

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        flipped = _copy(graph)
        flipped.nodes[self.up_field] = graph.nodes[self.down_field]
        flipped.nodes[self.down_field] = graph.nodes[self.up_field]
        out = self.model(_copy(graph))
        out_flipped = self.model(flipped)

        mutator = tensorial.gcnn.experimental.update_graph(out)
        for field in self.fields:
            root, name = field.split(".", 1)
            value = getattr(out, root)[name]
            mutator = mutator.set((root, name), 0.5 * (value + getattr(out_flipped, root)[name]))
        return mutator.get()
