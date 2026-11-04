"""Explicit ``mu (x) B`` coupling in the node attributes.

In the plain NequIP stack the only place where two field dependent quantities meet is the gate, so
the energy has no direct bilinear ``mu x B`` path and ``sigma = d2E/dmu dB`` collapses
(see ``tests/verifica``).  :class:`NodewiseFieldProduct` appends the tensor product of the two
field blocks of ``nodes.attributes`` -- ``mu.B`` (0e), ``mu x B`` (1e) and the symmetric traceless
part (2e) -- to the attributes, right after ``NodewiseEncoding`` and before the first
``NodewiseLinear``.

The attributes are ``[species one-hot, B_ext (1e), mu (1e)]`` as built by ``NodewiseEncoding``, so
B is read from the first and mu from the second 1e block (already broadcast over the nodes).  At
``mu = B = 0`` the new block vanishes: the energy there, and so the ``mu``/``B`` independent part
of the model, is unchanged.

The same holds for the curvature in B alone: without a ``B (x) B`` path the energy at ``mu = 0``
is flat in B to round-off, and ``chi = -d2E/dB2`` stays whatever the per-species term gives,
the same for every structure.  ``field_squared=True`` also appends ``|B|^2`` (0e) and the symmetric
traceless part of ``B (x) B`` (2e), which the convolutions couple to the local environment (the
antisymmetric 1e part of ``B x B`` is zero).
"""

import e3nn_jax as e3j
from flax import linen
import jraph
from tensorial.gcnn import _base, keys

__all__ = ("NodewiseFieldProduct",)


class NodewiseFieldProduct(linen.Module):
    """``attributes <- [attributes, scale * (B (x) mu)]`` with 0e + 1e + 2e of the product, and
    with ``field_squared`` also ``squared_scale * (B (x) B)``, 0e + 2e."""

    scale: float = 1.0
    field: str = keys.ATTRIBUTES
    out_field: str = keys.ATTRIBUTES
    field_squared: bool = False
    squared_scale: float = 1.0

    @_base.shape_check
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        nodes = dict(graph.nodes)
        attrs: e3j.IrrepsArray = nodes[self.field]
        fields = attrs.filter(keep="1e")
        if fields.irreps.dim != 6:
            raise ValueError(
                f"expected exactly two 1e blocks (B_ext, mu) in the attributes, got {attrs.irreps}"
            )
        arr = fields.array
        field_b = e3j.IrrepsArray("1e", arr[..., :3])
        mu = e3j.IrrepsArray("1e", arr[..., 3:])
        product = e3j.tensor_product(field_b, mu)  # 0e + 1e + 2e
        blocks = [attrs, self.scale * product]
        if self.field_squared:
            blocks.append(self.squared_scale * e3j.tensor_product(field_b, field_b, filter_ir_out=["0e", "2e"]))
        nodes[self.out_field] = e3j.concatenate(blocks)
        return graph._replace(nodes=nodes)
