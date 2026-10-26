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
- ``isotropic``: a rank-2 tensor's mean isotropic part, the same for every entry (a global
  tensor such as the dielectric one).
- ``species_isotropic``: a rank-2 per-atom tensor's mean isotropic part for each element (Born
  charges, NMR shielding).
- ``species_lstsq``: a global scalar fitted as a sum of per-element contributions (the total
  energy, as :class:`tensorial.gcnn.atomic.EnergyContributionLstsq` does).

Accumulated in float64 numpy: total energies are large and nearly cancel in the least-squares
residual.  Values are averaged over the tensor components, like the RMSE metrics.
"""

from typing import Literal

import equinox
import numpy as np
import reax
from tensorial import gcnn
from typing_extensions import override

__all__ = ("ResidualMeanSquare",)

Baseline = Literal["zero", "isotropic", "species_isotropic", "species_lstsq"]
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
        if baseline not in ("zero", "isotropic", "species_isotropic", "species_lstsq"):
            raise ValueError(f"Unknown baseline '{baseline}'")
        root = field.split(".")[0]
        if baseline == "species_isotropic" and root != "nodes":
            raise ValueError("species_isotropic needs a per-atom field (nodes.*)")
        if baseline == "species_lstsq" and root != "globals":
            raise ValueError("species_lstsq needs a per-structure field (globals.*)")
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
        if self.baseline == "isotropic":
            residual -= 3.0 * s["sum_iso"] ** 2 / n_entries
        elif self.baseline == "species_isotropic":
            seen = s["count_z"] > 0
            residual -= 3.0 * float((s["sum_iso_z"][seen] ** 2 / s["count_z"][seen]).sum())
        elif self.baseline == "species_lstsq":
            seen = np.diag(s["ata"]) > 0
            ata, atb = s["ata"][np.ix_(seen, seen)], s["atb"][seen]
            coeffs = np.linalg.lstsq(ata, atb, rcond=None)[0]
            residual -= float(atb @ coeffs)  # min ||y - A c||^2 = y.y - (A^T y).c
        mean_square = max(residual, 0.0) / (n_entries * n_comp)
        return 1.0 / mean_square if self.inverse else mean_square

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
        values = values[mask].reshape(int(mask.sum()), -1)  # one row per entry
        n_comp = values.shape[1]
        state = {"n": float(len(values)), "sum_sq": float((values**2).sum())}

        if self.baseline in ("isotropic", "species_isotropic"):
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
