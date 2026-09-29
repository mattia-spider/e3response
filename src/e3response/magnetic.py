from flax import linen
import jax.numpy as jnp
import jraph
from tensorial import gcnn
from tensorial.gcnn import atomic
from tensorial.gcnn.keys import predicted

from . import keys

__all__ = "InducedMagneticField", "MagneticShieldingTensor"

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
