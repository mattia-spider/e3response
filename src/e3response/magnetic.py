from collections.abc import Sequence

from flax import linen
import jax.numpy as jnp
import jraph
from tensorial import gcnn
from tensorial.gcnn import atomic
from tensorial.gcnn.keys import predicted

from . import keys
from .electric_linked import _field_at

__all__ = (
    "InducedMagneticField",
    "MagneticShieldingTensor",
    "Magnetization",
    "MagneticSusceptibility",
    "SpeciesIsotropicShift",
)

class InducedMagneticField(linen.Module):
    """
    Flax.linen.Module for computing induced magnetic field B_ind(k) on each atom (node) k 
    via differentiation of total energy (global) with respect to the nuclear magnetic moment μ on each node.
    
    B_ind_{k, i} = ∂E_tot / ∂μ_{k, i}
    
    Returns:
    A graph where each node has a vector (3,) stored in `out_field`.
    """

    energy_fn: gcnn.GraphFunction
    energy_key: str = predicted(atomic.keys.ENERGY)  
    mu_key: str = keys.NUCLEAR_MAGNETIC_MOMENT
    out_key: str = predicted(keys.INDUCED_MAGNETIC_FIELD)
    # If True, evaluate the derivative at graph.nodes[mu]; else at zero.  The dataset carries
    # only the per-species moduli of μ, which are not the nodal vectors this derivative needs,
    # so in practice this stays False.
    mu_at_node: bool = False
    # Reverse mode: the energy is a scalar, so one VJP gives ∂E/∂μ for every node at once.
    # Forward mode would need 3 * n_nodes JVPs, which dominates the cost of the whole model.
    mode: str = "rev"
    
    def setup(self) -> None:
        # Diff E against μ (per-node). No at= — shape (N_atoms, 3) unknown at setup time.
        self._diff_E_wrt_mu = gcnn.diff(
            self.energy_fn,
            f"globals.{self.energy_key}:gk",
            wrt=[f"nodes.{self.mu_key}:Iα"],
            out=":Iα",
            return_graph=True,
            mode=self.mode,
        )
    
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        if self.mu_at_node:
            mu_val = graph.nodes[self.mu_key]
        else:
            # μ is not part of the dataset: pass a single (3,) vector and let `gcnn.diff`
            # broadcast it over the nodes, then write it into the graph before the energy runs.
            mu_val = jnp.zeros(3)

        B_ind, graph = self._diff_E_wrt_mu(
            graph,
            mu_val,
        )

        graph = (
            gcnn.experimental.update_graph(graph)
            .set(("nodes", self.out_key), B_ind)
            .get()
        )
        
        return graph

class MagneticShieldingTensor(linen.Module):
    """
    flax.linen.Module for computing magnetic shielding tensors σ_{k, ij}
    for each atom k, based on the linear response of the induced magnetic field
    B^{k}_ind to an applied external magnetic field B_ext.

    The Jacobian is computed as:
        σ_{k, ij} = ∂B_ind_{k, i} / ∂B_ext_j

    B_ext is differentiated as a single (3,) field shared by the whole batch, as
    `electric.Polarization` does with the electric field: the encoding broadcasts it to every node,
    so it costs 3 JVPs whatever the batch size.  Graphs in a batch never interact, so the
    derivative with respect to the shared field is each atom's derivative with respect to its own
    graph's field.  Differentiating each graph's field separately instead (`per_graph=True`) gives
    the same tensors from 3 JVPs *per graph*, so memory and time grow with the square of the batch.

    Returns:
        A graph where each node has a (3, 3) tensor stored in `out_field`.
    """

    B_ind_fn: gcnn.GraphFunction
    B_ind: str = predicted(keys.INDUCED_MAGNETIC_FIELD)
    B_ext: str = keys.EXTERNAL_MAGNETIC_FIELD
    out_key: str = predicted(keys.NMR_TENSORS)
    # If True, evaluate the Jacobian at the field stored in the graph; else at zero.  With the
    # shared field there is one field for the batch, taken from the first graph, so every graph in
    # the batch must carry the same one (e.g. a single structure).
    B_ext_at_graph: bool = False
    # Forward mode: B_ext has 3 components while the output has 3 per node, so this costs
    # 3 JVPs regardless of system size.  Keep this "fwd" even when `B_ind_fn` differentiates
    # in reverse mode — the two derivatives are free to use different modes precisely because
    # they are separate modules.
    mode: str = "fwd"
    # Differentiate each graph's own field separately.  Same result, quadratic cost in the batch
    # size; kept to check against and to allow a different field on every graph.
    per_graph: bool = False

    def setup(self) -> None:
        self._diff_fn = gcnn.diff(
            self.B_ind_fn,
            f"nodes.{self.B_ind}:Iγ",
            wrt=[f"globals.{self.B_ext}:{'gα' if self.per_graph else 'α'}"],
            out=":Iγα",
            return_graph=True,
            mode=self.mode,
        )

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        if self.per_graph:
            n_graphs = graph.n_node.shape[0]
            B_ext_val = (
                graph.globals[self.B_ext] if self.B_ext_at_graph else jnp.zeros((n_graphs, 3))
            )
        else:
            B_ext_val = graph.globals[self.B_ext][0] if self.B_ext_at_graph else jnp.zeros(3)

        shielding, graph = self._diff_fn(graph, B_ext_val)

        graph = (
            gcnn.experimental.update_graph(graph)
            .set(("nodes", self.out_key), shielding)
            .get()
        )
        return graph


class Magnetization(linen.Module):
    """Magnetic moment M_α = -∂E/∂B_α of each graph, the first link of the susceptibility chain.

    Built like :class:`e3response.electric_linked.Polarization`: it differentiates at the field
    already in the graph (zero if there is none), so that :class:`MagneticSusceptibility` wrapping
    it sees the field dependence.  The σ chain cannot be extended this way, since
    :class:`MagneticShieldingTensor` evaluates at a fixed field; put the two chains side by side
    with :class:`e3response.electric_linked.LinkedBranches` instead::

        - _target_: e3response.electric_linked.LinkedBranches
          _partial_: True
          branches:
            - - {_target_: e3response.magnetic.InducedMagneticField, _partial_: True}
              - {_target_: e3response.magnetic.MagneticShieldingTensor, _partial_: True}
            - - {_target_: e3response.magnetic.Magnetization, _partial_: True}
              - {_target_: e3response.magnetic.MagneticSusceptibility, _partial_: True}

    The energy of the NMR models also reads the nuclear moments, which only the σ chain inserts:
    here they are set to zero when absent.
    """

    energy_fn: gcnn.GraphFunction
    energy_key: str = predicted(atomic.keys.ENERGY)
    field_key: str = keys.EXTERNAL_MAGNETIC_FIELD
    mu_key: str = keys.NUCLEAR_MAGNETIC_MOMENT
    out_key: str = predicted(keys.MAGNETIZATION)

    def setup(self) -> None:
        # pylint: disable=attribute-defined-outside-init
        self._diff = gcnn.diff(
            self.energy_fn,
            f"globals.{self.energy_key}:gk",
            wrt=[f"globals.{self.field_key}:α"],
            out=":gα",
            return_graph=True,
            mode="fwd",
        )

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        if self.mu_key not in graph.nodes:
            n_nodes = graph.nodes[gcnn.keys.POSITIONS].shape[0]
            graph = gcnn.experimental.update_graph(graph).set(("nodes", self.mu_key), jnp.zeros((n_nodes, 3))).get()
        derivative, graph = self._diff(graph, _field_at(graph, self.field_key))
        return gcnn.experimental.update_graph(graph).set(("globals", self.out_key), -derivative).get()


class MagneticSusceptibility(linen.Module):
    """Magnetic susceptibility of each graph, the field derivative of the magnetic moment the
    chain predicts:

        χ_αβ = scale · ∂M_α/∂B_β = -scale · ∂²E/∂B_α∂B_β

    Extensive, one tensor per cell, like the GIPAW χ_bare in cm³ per mole of cells.  Wraps
    :class:`Magnetization`.  ``scale`` converts the model's units of E/B² into the label's: a
    constant, so χ stays the second field derivative of the energy whose mixed μ-B derivative
    is σ.
    """

    magnetization_fn: gcnn.GraphFunction
    magnetization_key: str = predicted(keys.MAGNETIZATION)
    field_key: str = keys.EXTERNAL_MAGNETIC_FIELD
    scale: float = 1.0
    out_key: str = predicted(keys.MAGNETIC_SUSCEPTIBILITY)

    def setup(self) -> None:
        # pylint: disable=attribute-defined-outside-init
        self._diff = gcnn.diff(
            self.magnetization_fn,
            f"globals.{self.magnetization_key}:gα",
            wrt=[f"globals.{self.field_key}:β"],
            out=":gαβ",
            return_graph=True,
            mode="fwd",
        )

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        chi, graph = self._diff(graph, _field_at(graph, self.field_key))
        return gcnn.experimental.update_graph(graph).set(("globals", self.out_key), self.scale * chi).get()


class SpeciesIsotropicShift(linen.Module):
    """Add a learnable isotropic tensor per species, c_{Z_i} I, to a per-atom (3, 3) field.

    For a direct readout of an extensive tensor that is nearly additive over the atoms (the
    susceptibility: 99% of it is a sum of per-element contributions), so that the network only
    learns what depends on the geometry.  The counterpart of
    :class:`e3response.electric_linked.SpeciesPolarizability` in the derivative models; start both
    from :class:`e3response.stats.SpeciesIsotropicContributions`.
    """

    num_species: int
    field: str
    init_values: Sequence[float] | None = None

    @linen.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        start = (
            jnp.zeros(self.num_species)
            if self.init_values is None
            else jnp.asarray(self.init_values, dtype=float).reshape(self.num_species)
        )
        shift = self.param("shift", lambda *_: start, (self.num_species,))
        species = graph.nodes[gcnn.keys.SPECIES].reshape(-1)
        values = graph.nodes[self.field]
        values = values + shift[species][:, None, None] * jnp.eye(3)
        return gcnn.experimental.update_graph(graph).set(("nodes", self.field), values).get()
