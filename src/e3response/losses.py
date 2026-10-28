from collections.abc import Callable

import jax
import jax.numpy as jnp
import jraph
import optax
import tensorial
from tensorial import gcnn
from tensorial.gcnn import atomic
from tensorial.gcnn.keys import predicted

from . import keys


def nearest_image_target(
    prediction: jax.Array, target: jax.Array, cell: jax.Array
) -> jax.Array:
    """Move each target dipole by whole lattice vectors (times e) to the image nearest the
    prediction.

    A periodic cell's dipole moment is only defined up to e·R for a lattice vector R, and datasets
    store one branch of it (allegro-pol folds into about ±L/2), so a prediction on another branch
    is not an error.  ``prediction`` and ``target`` are (n_graph, 3), ``cell`` (n_graph, 3, 3)
    with the lattice vectors as rows.  Graphs with a zero cell (padding) keep their target.
    """
    ok = jnp.abs(jnp.linalg.det(cell)) > 0.0
    safe_cell = jnp.where(ok[:, None, None], cell, jnp.eye(3))
    frac = jnp.einsum("ga,gab->gb", prediction - target, jnp.linalg.inv(safe_cell))
    shift = jax.lax.stop_gradient(jnp.round(frac))
    moved = target + jnp.einsum("gb,gba->ga", shift, safe_cell)
    return jnp.where(ok[:, None], moved, target)


def _with_nearest_image(
    predictions: jraph.GraphsTuple, targets: jraph.GraphsTuple, prediction_key: str, target_key: str
) -> jraph.GraphsTuple:
    prediction = tensorial.base.as_array(predictions.globals[prediction_key]).reshape(-1, 3)
    target = jnp.asarray(targets.globals[target_key])
    cell = jnp.asarray(targets.globals[gcnn.keys.CELL]).reshape(-1, 3, 3)
    moved = nearest_image_target(prediction, target.reshape(-1, 3), cell).reshape(target.shape)
    return targets._replace(globals={**targets.globals, target_key: moved})


class FoldedDipoleLoss(gcnn.Loss):
    """Squared error on a periodic cell's dipole moment, against the image of the target nearest
    the prediction (see :func:`nearest_image_target`)."""

    def __init__(
        self,
        targets: str = f"globals.{keys.DIPOLE_MOMENT}",
        predictions: str = f"globals.{predicted(keys.DIPOLE_MOMENT)}",
        *,
        reduction: str = "mean",
        label: str | None = None,
    ):
        super().__init__(optax.squared_error, targets, predictions, reduction=reduction, label=label)

    def _call(self, predictions: jraph.GraphsTuple, targets: jraph.GraphsTuple) -> jax.Array:
        targets = _with_nearest_image(
            predictions, targets, self._prediction_field[-1], self._target_field[-1]
        )
        return super()._call(predictions, targets)


def folded_dipole_metric(
    metric: str = "rmse",
    predictions: str = f"globals.{predicted(keys.DIPOLE_MOMENT)}",
    targets: str = f"globals.{keys.DIPOLE_MOMENT}",
):
    """A :func:`tensorial.gcnn.graph_metric` on the dipole moment, against the nearest image of the
    target, as :class:`FoldedDipoleLoss` uses."""
    base = type(gcnn.graph_metric(metric, predictions, targets))
    prediction_key, target_key = predictions.split(".")[-1], targets.split(".")[-1]

    class _FoldedDipoleMetric(base):  # pylint: disable=too-few-public-methods
        def create(self, predictions, targets=None):  # pylint: disable=arguments-differ
            targets = predictions if targets is None else targets
            targets = _with_nearest_image(predictions, targets, prediction_key, target_key)
            return super().create(predictions, targets)

    return _FoldedDipoleMetric()


class L2Regularization(gcnn.Loss):
    """Penalise the magnitude of a predicted field that has no ground truth.

    :class:`tensorial.gcnn.Loss` reads its target from the targets graph, but a quantity
    like the induced field at zero external field is only ever produced by the model, so
    there is nothing to read there.  Here the target is implicitly zero and the field is
    read from the predictions graph, which keeps the masking and per-graph reduction that
    :class:`~tensorial.gcnn.Loss` already implements.
    """

    def __init__(
        self,
        field: str,
        *,
        reduction: str = "mean",
        label: str = None,
    ):
        super().__init__(
            lambda predicted, _target: jnp.square(predicted),
            field,
            reduction=reduction,
            label=label or f"{field}_l2",
        )

    def _call(
        self, predictions: jraph.GraphsTuple, targets: jraph.GraphsTuple
    ) -> jax.Array:
        # The regularised quantity only exists in the model output
        return super()._call(predictions, predictions)


class NmrTensorLoss(gcnn.Loss):
    """Squared error on the NMR tensors, over the atoms that carry one.

    Datasets that label only some atoms (Si: the sites with a computed tensor; C-S-H with
    ``nmr_sites``) mark them in ``nodes.nmr_active`` and give the others a placeholder tensor,
    which must not be fitted.  Graphs without that field (QM9) count every atom.  Picking the
    mask up from the data, rather than from a config switch, means it cannot be forgotten.
    """

    def __init__(self, *, reduction: str = "mean", label: str | None = None):
        super().__init__(
            optax.squared_error,
            f"nodes.{keys.NMR_TENSORS}",
            f"nodes.{predicted(keys.NMR_TENSORS)}",
            reduction=reduction,
            label=label,
            mask_field=f"nodes.{keys.NMR_ACTIVE}",
        )

    def _call(self, predictions: jraph.GraphsTuple, targets: jraph.GraphsTuple) -> jax.Array:
        n_nodes = jnp.asarray(targets.nodes[keys.NMR_TENSORS]).shape[0]
        if keys.NMR_ACTIVE in targets.nodes:
            # Per-atom scalars come out of graph_from_ase as (n_node, 1); an entity mask is (n_node,)
            active = jnp.asarray(targets.nodes[keys.NMR_ACTIVE]).astype(bool).reshape(n_nodes)
        else:
            active = jnp.ones(n_nodes, dtype=bool)
        targets = targets._replace(nodes={**targets.nodes, keys.NMR_ACTIVE: active})
        return super()._call(predictions, targets)


def response_loss(
    energy: bool | float = False,
    forces: bool | float = False,
    polarization_tensors: bool | float = False,
    dielectric_tensor: bool | float = False,
    born_charges: bool | float = False,
    raman_tensors: bool | float = False,
    nmr_tensors: bool | float = False,
    induced_magnetic_field: bool | float = False,
    magnetic_susceptibility: bool | float = False,
) -> Callable[[jraph.GraphsTuple, jraph.GraphsTuple], jax.Array]:
    weights: list[float] = []
    loss_terms = []

    if energy:
        weights.append(1.0 if isinstance(energy, bool) else energy)
        loss_terms.append(
            gcnn.Loss(
                optax.squared_error,
                f"globals.{atomic.TOTAL_ENERGY}",
                f"globals.{predicted(atomic.TOTAL_ENERGY)}",
            )
        )

    if forces:
        weights.append(1.0 if isinstance(forces, bool) else forces)
        loss_terms.append(
            gcnn.Loss(
                optax.squared_error,
                f"nodes.{atomic.FORCES}",
                f"nodes.{predicted(atomic.FORCES)}",
            )
        )

    if born_charges:
        weights.append(1.0 if isinstance(born_charges, bool) else born_charges)
        loss_terms.append(
            gcnn.Loss(
                optax.squared_error,
                f"nodes.{keys.BORN_CHARGES}",
                f"nodes.{predicted(keys.BORN_CHARGES)}",
            )
        )

    if polarization_tensors:
        weights.append(1.0 if isinstance(polarization_tensors, bool) else polarization_tensors)
        loss_terms.append(
            gcnn.Loss(
                optax.squared_error,
                f"globals.{keys.POLARIZATION}",
                f"globals.{predicted(keys.POLARIZATION)}",
            )
        )

    if dielectric_tensor:
        weights.append(1.0 if isinstance(dielectric_tensor, bool) else dielectric_tensor)
        loss_terms.append(
            gcnn.Loss(
                optax.squared_error,
                f"globals.{keys.DIELECTRIC_TENSOR}",
                f"globals.{predicted(keys.DIELECTRIC_TENSOR)}",
            )
        )

    if raman_tensors:
        weights.append(1.0 if isinstance(raman_tensors, bool) else raman_tensors)
        loss_terms.append(
            gcnn.Loss(
                optax.squared_error,
                f"nodes.{keys.RAMAN_TENSORS}",
                f"nodes.{predicted(keys.RAMAN_TENSORS)}",
            )
        )

    if nmr_tensors:
        weights.append(1.0 if isinstance(nmr_tensors, bool) else nmr_tensors)
        loss_terms.append(NmrTensorLoss())

    if induced_magnetic_field:
        weights.append(
            1.0 if isinstance(induced_magnetic_field, bool) else induced_magnetic_field
        )
        loss_terms.append(
            L2Regularization(
                f"nodes.{predicted(keys.INDUCED_MAGNETIC_FIELD)}",
            )
        )

    if magnetic_susceptibility:
        weights.append(1.0 if isinstance(magnetic_susceptibility, bool) else magnetic_susceptibility)
        loss_terms.append(
            gcnn.Loss(
                optax.squared_error,
                f"globals.{keys.MAGNETIC_SUSCEPTIBILITY}",
                f"globals.{predicted(keys.MAGNETIC_SUSCEPTIBILITY)}",
            )
        )

    if not loss_terms:
        raise ValueError(
            "Could not create loss function because all terms (energy, forces, ...) are set to "
            "`False`"
        )

    if loss_terms:
        return gcnn.WeightedLoss(loss_terms, weights)

    return loss_terms[0]
