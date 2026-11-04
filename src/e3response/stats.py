"""Dataset statistics for ``from_data``: the scale of each training target, to weight the loss.

A loss that adds squared errors of different quantities (energies in eV, Born charges in e,
tensors in ppm, ...) is dominated by whichever has the largest numbers unless each term is
divided by its own scale.  :class:`ResidualMeanSquare` measures that scale as the mean square of
the target left over once the part a constant model predicts is removed, the error a model
starts from, so that ``1 / residual`` puts every term at O(1) at the start of training:

.. code-block:: yaml

    from_data:
      w_born:
        _target_: e3response.stats.ResidualMeanSquare
        field: nodes.born_charges
        baseline: species_isotropic
        inverse: true

The baseline is what an equivariant model with no geometry information could output:

- ``zero``: vectors and tensors with no invariant part (forces, dipoles, Raman tensors).
- ``mean``: the mean of every component, the same for every entry.  For a global tensor in a
  dataset whose cells all share one orientation, where the anisotropic part of the mean is
  learnt at once from the lattice, so only the spread about it is left to learn.
- ``isotropic``: a rank-2 tensor's mean isotropic part, the same for every entry (a global
  tensor in cells of any orientation).
- ``species_isotropic``: a rank-2 per-atom tensor's mean isotropic part for each element (Born
  charges, NMR shielding).
- ``species_lstsq``: a global scalar fitted as a sum of per-element contributions (the total
  energy, as :class:`tensorial.gcnn.atomic.EnergyContributionLstsq` does).
- ``species_lstsq_isotropic``: a rank-2 global tensor whose isotropic part is fitted as a sum of
  per-element contributions (an extensive susceptibility, nearly additive over the atoms).

:class:`SpeciesIsotropicContributions` returns the per-element coefficients of the last baseline
themselves, to start a model from them.

Accumulated in float64 numpy: total energies are large and nearly cancel in the least-squares
residual.  Values are averaged over the tensor components, like the RMSE metrics.
"""

from typing import Literal

import equinox
import numpy as np
import reax
from tensorial import gcnn
from typing_extensions import override

__all__ = ("ResidualMeanSquare", "SpeciesContributions", "SpeciesIsotropicContributions")

Baseline = Literal["zero", "mean", "isotropic", "species_isotropic", "species_lstsq", "species_lstsq_isotropic"]
_BASELINES = ("zero", "mean", "isotropic", "species_isotropic", "species_lstsq", "species_lstsq_isotropic")
_N_Z = 119  # atomic numbers 0..118


class ResidualMeanSquare(reax.Metric):
    """Mean square of ``field`` after removing ``baseline``, per tensor component (see module)."""

    field: str = equinox.field(static=True)
    baseline: str = equinox.field(static=True)
    inverse: bool = equinox.field(static=True)
    species_field: str = equinox.field(static=True)
    # Accumulated sums, None while empty
    _state: dict | None = None

    def __init__(
        self,
        field: str,
        baseline: Baseline = "zero",
        inverse: bool = False,
        species_field: str = "nodes.atomic_numbers",
        _state: dict | None = None,
    ):
        if baseline not in _BASELINES:
            raise ValueError(f"Unknown baseline '{baseline}'")
        root = field.split(".")[0]
        if baseline == "species_isotropic" and root != "nodes":
            raise ValueError("species_isotropic needs a per-atom field (nodes.*)")
        if baseline in ("species_lstsq", "species_lstsq_isotropic") and root != "globals":
            raise ValueError(f"{baseline} needs a per-structure field (globals.*)")
        self.field = field
        self.baseline = baseline
        self.inverse = inverse
        self.species_field = species_field
        self._state = _state

    def _with(self, state: dict | None) -> "ResidualMeanSquare":
        return type(self)(self.field, self.baseline, self.inverse, self.species_field, state)

    @override
    def empty(self) -> "ResidualMeanSquare":
        return self._with(None)

    @override
    def create(self, graphs, /, *_, **__) -> "ResidualMeanSquare":
        return self._with(self._sums(graphs))

    @override
    def update(self, graphs, /, *_, **__) -> "ResidualMeanSquare":
        return self.merge(self.create(graphs))

    @override
    def merge(self, other: "ResidualMeanSquare") -> "ResidualMeanSquare":
        if other._state is None:  # pylint: disable=protected-access
            return self
        if self._state is None:
            return other
        other_state = other._state  # pylint: disable=protected-access
        # n_comp is the number of components per entry, the same in every batch
        merged = {k: v if k == "n_comp" else v + other_state[k] for k, v in self._state.items()}
        return self._with(merged)

    @override
    def compute(self) -> float:
        if self._state is None:
            raise RuntimeError("Nothing to compute, metric is empty!")
        s = self._state
        n_entries, n_comp = s["n"], s["n_comp"]
        residual = s["sum_sq"]
        if self.baseline == "mean":
            residual -= float((s["sum"] ** 2).sum()) / n_entries
        elif self.baseline == "isotropic":
            residual -= 3.0 * s["sum_iso"] ** 2 / n_entries
        elif self.baseline == "species_isotropic":
            seen = s["count_z"] > 0
            residual -= 3.0 * float((s["sum_iso_z"][seen] ** 2 / s["count_z"][seen]).sum())
        elif self.baseline in ("species_lstsq", "species_lstsq_isotropic"):
            seen, coeffs = self._lstsq(s)
            # min ||y - A c||^2 = y.y - (A^T y).c; for the isotropic part of a tensor,
            # ||T - a I||^2 = ||T||^2 - 3 iso^2 + 3 (iso - a)^2, so 3 times that
            factor = 3.0 if self.baseline == "species_lstsq_isotropic" else 1.0
            residual -= factor * float(s["atb"][seen] @ coeffs)
        mean_square = max(residual, 0.0) / (n_entries * n_comp)
        return 1.0 / mean_square if self.inverse else mean_square

    @staticmethod
    def _lstsq(s: dict) -> tuple[np.ndarray, np.ndarray]:
        """The elements present, and their least-squares contributions."""
        seen = np.diag(s["ata"]) > 0
        ata, atb = s["ata"][np.ix_(seen, seen)], s["atb"][seen]
        return seen, np.linalg.lstsq(ata, atb, rcond=None)[0]

    def _sums(self, graphs) -> dict:
        root, *path = self.field.split(".")
        container = getattr(graphs, root)
        try:
            values = container
            for key in path:
                values = values[key]
        except (KeyError, TypeError):
            raise reax.exceptions.DataNotFound(f"Missing key: {self.field}") from None

        values = np.asarray(values, dtype=np.float64)
        n_items = values.shape[0]
        mask = container.get(gcnn.keys.MASK) if isinstance(container, dict) else None
        mask = np.ones(n_items, bool) if mask is None else np.asarray(mask).reshape(n_items)
        mask = mask.astype(bool)
        values = values.reshape(n_items, -1)[mask]  # one row per entry (none if all are masked)
        n_comp = values.shape[1]
        state = {"n": float(len(values)), "sum_sq": float((values**2).sum())}

        if self.baseline == "mean":
            state["sum"] = values.sum(axis=0)
        elif self.baseline in ("isotropic", "species_isotropic"):
            if n_comp != 9:
                raise ValueError(f"{self.baseline} needs a 3x3 tensor, {self.field} has {n_comp}")
            iso = values[:, [0, 4, 8]].sum(axis=1) / 3.0
            if self.baseline == "isotropic":
                state["sum_iso"] = float(iso.sum())
            else:
                z = self._atomic_numbers(graphs)[mask]
                state["count_z"] = np.bincount(z, minlength=_N_Z).astype(np.float64)
                state["sum_iso_z"] = np.bincount(z, weights=iso, minlength=_N_Z)
        elif self.baseline == "species_lstsq":
            if n_comp != 1:
                raise ValueError(f"species_lstsq needs a scalar, {self.field} has {n_comp}")
            counts = self._counts_per_graph(graphs)[mask]
            state["ata"] = counts.T @ counts
            state["atb"] = counts.T @ values[:, 0]
        elif self.baseline == "species_lstsq_isotropic":
            if n_comp != 9:
                raise ValueError(f"{self.baseline} needs a 3x3 tensor, {self.field} has {n_comp}")
            counts = self._counts_per_graph(graphs)[mask]
            state["ata"] = counts.T @ counts
            state["atb"] = counts.T @ (values[:, [0, 4, 8]].sum(axis=1) / 3.0)

        state["n_comp"] = float(n_comp)
        return state

    def _atomic_numbers(self, graphs) -> np.ndarray:
        root, key = self.species_field.split(".")
        return np.asarray(getattr(graphs, root)[key]).reshape(-1).astype(np.int64)

    def _counts_per_graph(self, graphs) -> np.ndarray:
        z = self._atomic_numbers(graphs)
        n_node = np.asarray(graphs.n_node)
        graph_of = np.repeat(np.arange(len(n_node)), n_node)
        node_mask = graphs.nodes.get(gcnn.keys.MASK)
        if node_mask is not None:
            keep = np.asarray(node_mask).reshape(-1).astype(bool)
            z, graph_of = z[keep], graph_of[keep]
        counts = np.zeros((len(n_node), _N_Z))
        np.add.at(counts, (graph_of, z), 1.0)
        return counts


class SpeciesIsotropicContributions(ResidualMeanSquare):
    """Per-element contributions to the isotropic part of a global rank-2 tensor, fitted by least
    squares (the ``species_lstsq_isotropic`` baseline), in the order of ``type_map``; elements
    absent from the data get 0.  Divided by ``divide_by``, e.g. the constant that
    converts a model's output into the label's units, so that the result is a parameter value.
    """

    type_map: tuple = equinox.field(static=True)
    divide_by: float = equinox.field(static=True)

    def __init__(
        self,
        field: str,
        type_map,
        divide_by: float = 1.0,
        species_field: str = "nodes.atomic_numbers",
        _state: dict | None = None,
    ):
        super().__init__(field, "species_lstsq_isotropic", False, species_field, _state)
        self.type_map = tuple(int(z) for z in type_map)
        self.divide_by = float(divide_by)

    def _with(self, state: dict | None) -> "SpeciesIsotropicContributions":
        return type(self)(self.field, self.type_map, self.divide_by, self.species_field, state)

    @override
    def compute(self) -> list[float]:
        if self._state is None:
            raise RuntimeError("Nothing to compute, metric is empty!")
        seen, coeffs = self._lstsq(self._state)
        per_z = np.zeros(_N_Z)
        per_z[seen] = coeffs
        return [float(per_z[z] / self.divide_by) for z in self.type_map]


class SpeciesContributions(ResidualMeanSquare):
    """Per-element contributions to a global scalar (the total energy), fitted by least squares in
    float64 (the ``species_lstsq`` baseline), as ``[[c_Z] for Z in type_map]``: the ``shifts`` of
    :func:`tensorial.gcnn.atomic.per_species_rescale`.

    A drop-in for :class:`tensorial.gcnn.atomic.EnergyContributionLstsq`, which solves in float32
    on per-atom energies: with C-S-H's -4000 Ry cells, whose compositions span only 4 of the 5
    elements (charge neutrality), its shifts leave 2.2 Ry per cell where the fit leaves 0.063.
    Elements absent from the data get 0.
    """

    type_map: tuple = equinox.field(static=True)

    def __init__(self, field: str, type_map, species_field: str = "nodes.atomic_numbers", _state: dict | None = None):
        super().__init__(field, "species_lstsq", False, species_field, _state)
        self.type_map = tuple(int(z) for z in type_map)

    def _with(self, state: dict | None) -> "SpeciesContributions":
        return type(self)(self.field, self.type_map, self.species_field, state)

    @override
    def compute(self) -> list[list[float]]:
        if self._state is None:
            raise RuntimeError("Nothing to compute, metric is empty!")
        seen, coeffs = self._lstsq(self._state)
        per_z = np.zeros(_N_Z)
        per_z[seen] = coeffs
        return [[float(per_z[z])] for z in self.type_map]
