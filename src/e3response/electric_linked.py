"""Electric response quantities linked by differentiation, for Sobolev training.

Each module here differentiates the output of the module it wraps -- not the energy -- and takes
its field derivative at the field already in the graph rather than at a fixed one.  Listed one
after the other in a `tensorial.nn.Sequential`, where every partial module wraps the one before it,
they therefore link the predicted quantities through their derivatives, as the NMR models link
E → B_ind → σ:

    E ──(-∂/∂u)──► F ──(∂/∂ε)──► Z* ──(∂/∂ε)──► R
    E ──(-∂/∂ε)──► d ──(∂/∂ε)──► χ            with P = d/Ω and ε_r = I + χ/(ε0 Ω)

so that, e.g., the Raman tensors are the field derivative of the Born charges the model predicts.
The same Born charges and Raman tensors can also be reached from the dipole side, as position
derivatives of the dipole moment and of χ:

    E ──(-∂/∂ε)──► d ──(∂/∂ε)──► χ ──(∂/∂u)──► R          and   d ──(∂/∂u)──► Z*

with :class:`BornChargesFromDipole` and :class:`RamanTensorsFromPolarizability`.  Both routes give
the same values; the force route is cheaper, since its field derivatives are forward passes over a
single reverse one, while here every position derivative is a reverse pass per component.
The values are the same derivatives of the energy that :mod:`e3response.electric` computes.

The modules in :mod:`e3response.electric` cannot be chained like this: each differentiates the
*energy* at a field fixed to zero, and passes that energy on, so the next module's field derivative
sees no dependence on the field and comes out as exactly zero.  Use one of those on its own, or
these in a chain.

A chain may start from any module in the order above; e.g. for Born and Raman labels only::

    - _target_: e3response.electric_linked.Forces
      _partial_: True
    - _target_: e3response.electric_linked.BornCharges
      _partial_: True
    - _target_: e3response.electric_linked.RamanTensors
      _partial_: True

The field is one (3,) vector for the whole batch, which a node encoding broadcasts to every atom,
so the field derivatives cost 3 JVPs whatever the batch size.  A per-graph (n_graph, 3) field in the
data is accepted only if every graph carries the same one, and its first row is used.
"""

import math
from collections.abc import Callable, Sequence

from flax import linen
import jax
import jax.numpy as jnp
import jaxtyping as jt
import jraph
from tensorial import gcnn, nn_utils
from tensorial.gcnn import atomic
from tensorial.gcnn.keys import predicted
import tensorial.typing as tt

from . import keys

__all__ = (
    "Forces",
    "BornCharges",
    "RamanTensors",
    "Polarization",
    "DielectricTensor",
    "BornChargesFromDipole",
    "RamanTensorsFromPolarizability",
    "LinkedBranches",
    "Scale",
    "SpeciesPolarizability",
)


def _field_at(graph: jraph.GraphsTuple, field_key: str) -> jax.Array:
    """The field to evaluate at: the one in the graph (inserted by a module further out in the
    chain, or carried by the data), or zero if there is none."""
    if field_key not in graph.globals:
        return jnp.zeros(3)
    field = jnp.asarray(graph.globals[field_key])
    return field if field.ndim == 1 else field[0]


def _with_field(graph: jraph.GraphsTuple, field_key: str) -> jraph.GraphsTuple:
    """Make sure the energy finds a field to read, without replacing one that is already there."""
    if field_key in graph.globals:
        return graph
    return gcnn.experimental.update_graph(graph).set(("globals", field_key), jnp.zeros(3)).get()


def _per_volume(graph: jraph.GraphsTuple, values: jax.Array) -> jax.Array:
    """Divide per-graph values by the cell volume, if the graphs are periodic."""
    if gcnn.keys.CELL not in graph.globals:
        return values
    cells: jt.Float[tt.ArrayType, "n_graph 3 3"] = graph.globals[gcnn.keys.CELL]
    omega: jt.Float[tt.ArrayType, "n_graph"] = jax.vmap(gcnn.calc.cell_volume)(cells)
    # Avoid dividing by zero on padding graphs
    if keys.MASK in graph.globals:
        omega = jnp.where(nn_utils.prepare_mask(graph.globals[keys.MASK], omega), omega, 1.0)
    omega = jnp.where(omega == 0.0, 1.0, omega)
    return jax.vmap(jnp.divide)(values, omega)


def _per_volume_nodes(graph: jraph.GraphsTuple, values: jax.Array) -> jax.Array:
    """Divide per-atom values by the volume of the cell each atom belongs to, if periodic."""
    if gcnn.keys.CELL not in graph.globals:
        return values
    cells: jt.Float[tt.ArrayType, "n_graph 3 3"] = graph.globals[gcnn.keys.CELL]
    omega: jt.Float[tt.ArrayType, "n_graph"] = jax.vmap(gcnn.calc.cell_volume)(cells)
    # Padding graphs may have a zero cell; their atoms are masked out of losses anyway
    omega = jnp.where(omega == 0.0, 1.0, omega)
    omega_nodes = jnp.repeat(omega, graph.n_node, total_repeat_length=values.shape[0])
    return values / omega_nodes.reshape((-1,) + (1,) * (values.ndim - 1))


class Forces(linen.Module):
    """Forces F_{iγ} = -∂E/∂u_{iγ}, the first link of the Born and Raman chain.

    One reverse pass on the energy summed over the batch: each atom only enters its own graph's
    energy, so this is the per-graph gradient.
    """

    energy_fn: gcnn.GraphFunction
    energy_key: str = predicted(atomic.TOTAL_ENERGY)
    electric_field_key: str = keys.EXTERNAL_ELECTRIC_FIELD
    out_key: str = predicted(atomic.FORCES)

    def setup(self) -> None:
        # pylint: disable=attribute-defined-outside-init
        self._grad = gcnn.diff(
            self.energy_fn,
            f"globals.{self.energy_key}:gk",
            wrt=[f"nodes.{keys.POSITIONS}:Iγ"],
            out=":Iγ",
            return_graph=True,
            mode="rev",
        )

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        # Not differentiated here, but the energy reads a field and the data may not carry one
        graph = _with_field(graph, self.electric_field_key)
        gradient, graph = self._grad(
            graph, **{f"nodes.{keys.POSITIONS}": graph.nodes[gcnn.keys.POSITIONS]}
        )
        return gcnn.experimental.update_graph(graph).set(("nodes", self.out_key), -gradient).get()


class BornCharges(linen.Module):
    """Born effective charges as the field derivative of the forces the chain predicts.

        Z*_{i,αγ} = ∂F_{iγ}/∂ε_α = -∂²E/∂u_{iγ}∂ε_α

    Wraps :class:`Forces`, and is itself what :class:`RamanTensors` differentiates.
    """

    forces_fn: gcnn.GraphFunction
    forces_key: str = predicted(atomic.FORCES)
    electric_field_key: str = keys.EXTERNAL_ELECTRIC_FIELD
    out_key: str = predicted(keys.BORN_CHARGES)

    def setup(self) -> None:
        # pylint: disable=attribute-defined-outside-init
        self._diff = gcnn.diff(
            self.forces_fn,
            f"nodes.{self.forces_key}:Iγ",
            wrt=[f"globals.{self.electric_field_key}:α"],
            out=":Iαγ",
            return_graph=True,
            mode="fwd",
        )

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        born, graph = self._diff(graph, _field_at(graph, self.electric_field_key))
        return gcnn.experimental.update_graph(graph).set(("nodes", self.out_key), born).get()


class RamanTensors(linen.Module):
    """Raman tensors as the field derivative of the Born charges the chain predicts.

        R_{i,γαβ} = ∂Z*_{i,αγ}/∂ε_β = -∂³E/∂u_{iγ}∂ε_α∂ε_β

    Wraps :class:`BornCharges`.

    With ``per_volume=True`` the result is divided by the cell volume Ω, R/Ω = ∂[(ε_r - 1)/4π]/∂u,
    for labels normalised that way.  The BTO labels are not: across its snapshots the dielectric
    tensor changes as ε_r - I ≈ 4π/Ω ∫ R·du, i.e. R = ∂χ/∂u with ε_r = I + 4π χ/Ω, the default.
    The convention matters only when R and the dielectric tensor are trained together; otherwise
    the scale of the field absorbs it.
    """

    born_charges_fn: gcnn.GraphFunction
    born_charges_key: str = predicted(keys.BORN_CHARGES)
    electric_field_key: str = keys.EXTERNAL_ELECTRIC_FIELD
    per_volume: bool = False
    out_key: str = predicted(keys.RAMAN_TENSORS)

    def setup(self) -> None:
        # pylint: disable=attribute-defined-outside-init
        self._diff = gcnn.diff(
            self.born_charges_fn,
            f"nodes.{self.born_charges_key}:Iαγ",
            wrt=[f"globals.{self.electric_field_key}:β"],
            out=":Iγαβ",
            return_graph=True,
            mode="fwd",
        )

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        raman, graph = self._diff(graph, _field_at(graph, self.electric_field_key))
        if self.per_volume:
            raman = _per_volume_nodes(graph, raman)
        return gcnn.experimental.update_graph(graph).set(("nodes", self.out_key), raman).get()


class Polarization(linen.Module):
    """Dipole moment d_α = -∂E/∂ε_α of each graph, the first link of the dielectric chain, and the
    polarization P = d/Ω when the graphs are periodic.

    The dipole moment, not the polarization, is what :class:`DielectricTensor` differentiates, so
    that the cell volume is divided out only once.
    """

    energy_fn: gcnn.GraphFunction
    energy_key: str = predicted(atomic.TOTAL_ENERGY)
    electric_field_key: str = keys.EXTERNAL_ELECTRIC_FIELD
    dipole_key: str = predicted(keys.DIPOLE_MOMENT)
    out_key: str = predicted(keys.POLARIZATION)

    def setup(self) -> None:
        # pylint: disable=attribute-defined-outside-init
        self._diff = gcnn.diff(
            self.energy_fn,
            f"globals.{self.energy_key}:gk",
            wrt=[f"globals.{self.electric_field_key}:α"],
            out=":gα",
            return_graph=True,
            mode="fwd",
        )

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        derivative, graph = self._diff(graph, _field_at(graph, self.electric_field_key))
        dipole = -derivative
        return (
            gcnn.experimental.update_graph(graph)
            .set(("globals", self.dipole_key), dipole)
            .set(("globals", self.out_key), _per_volume(graph, dipole))
            .get()
        )


class DielectricTensor(linen.Module):
    """Relative dielectric tensor from the field derivative of the dipole moment the chain predicts.

        χ_αβ = ∂d_α/∂ε_β = -∂²E/∂ε_α∂ε_β,        ε_r = I + χ/(ε0 Ω)

    in atomic units (ε0 = 1/4π), with Ω the cell volume when the graphs are periodic.  Wraps
    :class:`Polarization`.  χ itself is also stored, for :class:`RamanTensorsFromPolarizability`.
    """

    dipole_fn: gcnn.GraphFunction
    dipole_key: str = predicted(keys.DIPOLE_MOMENT)
    electric_field_key: str = keys.EXTERNAL_ELECTRIC_FIELD
    include_identity: bool = True
    polarizability_key: str = predicted(keys.POLARIZABILITY)
    out_key: str = predicted(keys.DIELECTRIC_TENSOR)
    epsilon_0 = 1.0 / (4.0 * math.pi)

    def setup(self) -> None:
        # pylint: disable=attribute-defined-outside-init
        self._diff = gcnn.diff(
            self.dipole_fn,
            f"globals.{self.dipole_key}:gα",
            wrt=[f"globals.{self.electric_field_key}:β"],
            out=":gαβ",
            return_graph=True,
            mode="fwd",
        )

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        susceptibility, graph = self._diff(graph, _field_at(graph, self.electric_field_key))
        dielectric = _per_volume(graph, susceptibility / self.epsilon_0)
        if self.include_identity:
            dielectric = jnp.eye(3) + dielectric
        return (
            gcnn.experimental.update_graph(graph)
            .set(("globals", self.polarizability_key), susceptibility)
            .set(("globals", self.out_key), dielectric)
            .get()
        )


class BornChargesFromDipole(linen.Module):
    """Born effective charges as the position derivative of the dipole moment the chain predicts.

        Z*_{i,αγ} = ∂d_α/∂u_{iγ} = -∂²E/∂ε_α∂u_{iγ}

    the same values as :class:`BornCharges`, reached from the dipole side.  Wraps
    :class:`Polarization`, or a module that wraps it and passes its dipole moment on, such as
    :class:`DielectricTensor`.

    Reverse mode, one pass per dipole component on the dipole summed over the batch: each atom only
    enters its own graph's dipole, so this is the per-graph derivative.
    """

    dipole_fn: gcnn.GraphFunction
    dipole_key: str = predicted(keys.DIPOLE_MOMENT)
    out_key: str = predicted(keys.BORN_CHARGES)

    def setup(self) -> None:
        # pylint: disable=attribute-defined-outside-init
        self._diff = gcnn.diff(
            self.dipole_fn,
            f"globals.{self.dipole_key}:gα",
            wrt=[f"nodes.{keys.POSITIONS}:Iγ"],
            out=":Iαγ",
            return_graph=True,
            mode="rev",
        )

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        born, graph = self._diff(
            graph, **{f"nodes.{keys.POSITIONS}": graph.nodes[gcnn.keys.POSITIONS]}
        )
        return gcnn.experimental.update_graph(graph).set(("nodes", self.out_key), born).get()


class RamanTensorsFromPolarizability(linen.Module):
    """Raman tensors as the position derivative of the polarizability χ the chain predicts.

        R_{i,γαβ} = ∂χ_αβ/∂u_{iγ} = -∂³E/∂u_{iγ}∂ε_α∂ε_β

    the same values as :class:`RamanTensors`, reached from the dipole side.  Wraps
    :class:`DielectricTensor`, or a module that wraps it and passes χ on, such as
    :class:`BornChargesFromDipole`.

    Reverse mode, one pass per component of χ summed over the batch.  ``per_volume`` as in
    :class:`RamanTensors`.
    """

    polarizability_fn: gcnn.GraphFunction
    polarizability_key: str = predicted(keys.POLARIZABILITY)
    per_volume: bool = False
    out_key: str = predicted(keys.RAMAN_TENSORS)

    def setup(self) -> None:
        # pylint: disable=attribute-defined-outside-init
        self._diff = gcnn.diff(
            self.polarizability_fn,
            f"globals.{self.polarizability_key}:gαβ",
            wrt=[f"nodes.{keys.POSITIONS}:Iγ"],
            out=":Iγαβ",
            return_graph=True,
            mode="rev",
        )

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        raman, graph = self._diff(
            graph, **{f"nodes.{keys.POSITIONS}": graph.nodes[gcnn.keys.POSITIONS]}
        )
        if self.per_volume:
            raman = _per_volume_nodes(graph, raman)
        return gcnn.experimental.update_graph(graph).set(("nodes", self.out_key), raman).get()


def _copy_dicts(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    copy = lambda field: dict(field) if isinstance(field, dict) else field
    return graph._replace(nodes=copy(graph.nodes), edges=copy(graph.edges), globals=copy(graph.globals))


class LinkedBranches(linen.Module):
    """Several linked chains built on the same energy, side by side rather than nested.

    In a `tensorial.nn.Sequential` every partial module wraps everything before it, so listing the
    dielectric chain and the Born/Raman chain one after the other differentiates each chain through
    the other as well.  That gives the same values but far more work and memory than the two chains
    separately.  This module builds each chain on the energy on its own and merges their outputs::

        - _target_: e3response.electric_linked.LinkedBranches
          _partial_: True
          branches:
            - - {_target_: e3response.electric_linked.Polarization, _partial_: True}
              - {_target_: e3response.electric_linked.DielectricTensor, _partial_: True}
            - - {_target_: e3response.electric_linked.Forces, _partial_: True}
              - {_target_: e3response.electric_linked.BornCharges, _partial_: True}
              - {_target_: e3response.electric_linked.RamanTensors, _partial_: True}

    Each branch is a sequence of partial modules, each wrapping the one before it as in a
    `Sequential`.  The energy's parameters are shared by all branches.
    """

    energy_fn: gcnn.GraphFunction
    branches: Sequence[Sequence[Callable[[gcnn.GraphFunction], gcnn.GraphFunction]]]

    def setup(self) -> None:
        # pylint: disable=attribute-defined-outside-init
        chains = []
        for branch in self.branches:
            chain = self.energy_fn
            for make_link in branch:
                chain = make_link(chain)
            chains.append(chain)
        self._chains = chains

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        # Some tensorial layers write into the graph's dicts in place; give each branch its own
        # copies, so that values traced in one branch do not leak into the next
        outputs = [chain(_copy_dicts(graph)) for chain in self._chains]
        nodes, globals_ = dict(outputs[0].nodes), dict(outputs[0].globals)
        for out in outputs[1:]:
            nodes.update(out.nodes)
            globals_.update(out.globals)
        return outputs[0]._replace(nodes=nodes, globals=globals_)


class Scale(linen.Module):
    """Multiply one graph value by a constant, e.g. ``globals.external_electric_field``.

    In a Sobolev model only derivatives of the energy are labelled, so the units of the energy and
    of the field are free.  Scaling the field by c where it enters the network and the energy by 1/c
    where it leaves keeps the Born charges (∝ E/ε) unchanged, but makes the second field derivatives
    (χ and R, ∝ E/ε²) c times larger at initialisation.  For BTO, where ε_r = I + 4π χ/Ω with
    Ω ≈ 1700 Å³, χ must be of order 10² - 10³, which a freshly initialised network does not reach
    otherwise.
    """

    field: str
    factor: float
    invert: bool = False  # divide by the factor instead, so one config value sets both ends

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        path = tuple(self.field.split("."))
        value = graph
        for key in path:
            value = getattr(value, key) if isinstance(value, jraph.GraphsTuple) else value[key]
        return gcnn.experimental.update_graph(graph).set(path, value / self.factor if self.invert else value * self.factor).get()


class SpeciesPolarizability(linen.Module):
    """Add an isotropic polarizability per species to the energy, E += -½ Σ_i α_{Z_i} |ε|².

    The geometry-independent part of the dielectric response: it contributes Σ_i α_{Z_i} I to
    χ = -∂²E/∂ε² and nothing to the Born charges or Raman tensors.  Without it a network must build
    all of χ from terms that also depend on the positions, so χ can only grow as fast as R = ∂χ/∂u
    allows.  It plays the role that species-dependent constants play in a direct readout.  Place it
    after the total energy and before any rescaling of the energy.

    Any field works: with ``electric_field_key: external_magnetic_field`` it is the per-species
    diamagnetism of the NMR models, which adds Σ_i α_{Z_i} I to χ = -∂²E/∂B² and nothing to σ.
    ``init_values`` starts each species from its own value (e.g. from
    :class:`e3response.stats.SpeciesIsotropicContributions`), instead of ``init_value`` for all.

    ``unit`` sets the units of α: the energy gets -½ Σ α/unit |ε|².  With the constant that converts
    the model's -∂²E/∂ε² into the label's (MagneticSusceptibility's ``scale``), α is in the label's
    units, so that an optimiser step on α moves the predicted tensor by the same amount whatever
    that constant is.
    """

    num_species: int
    electric_field_key: str = keys.EXTERNAL_ELECTRIC_FIELD
    energy_key: str = predicted(atomic.TOTAL_ENERGY)
    init_value: float = 0.0
    init_values: Sequence[float] | None = None
    unit: float = 1.0

    @linen.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        if self.init_values is not None:
            start = jnp.asarray(self.init_values, dtype=float).reshape(self.num_species)
        else:
            start = jnp.full((self.num_species,), self.init_value)
        alpha = self.param("alpha", lambda *_: start, (self.num_species,))
        species = graph.nodes[gcnn.keys.SPECIES].reshape(-1)
        n_graph = graph.n_node.shape[0]
        node_graph = jnp.repeat(jnp.arange(n_graph), graph.n_node, total_repeat_length=species.shape[0])
        alpha_graph = jax.ops.segment_sum(alpha[species], node_graph, n_graph)
        field = jnp.asarray(graph.globals[self.electric_field_key])
        field_sq = jnp.sum(field**2, axis=-1)  # (3,) shared field -> scalar, (G, 3) -> (G,)
        energy = graph.globals[self.energy_key]
        delta = -0.5 * alpha_graph / self.unit * field_sq
        return (
            gcnn.experimental.update_graph(graph)
            .set(("globals", self.energy_key), energy + delta.reshape(energy.shape))
            .get()
        )
