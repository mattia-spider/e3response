from collections.abc import Callable

import jax
import jax.numpy as jnp
import jraph
import optax
from tensorial import gcnn
from tensorial.gcnn import atomic
from tensorial.gcnn.keys import predicted

from . import keys


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


def response_loss(
    energy: bool | float = False,
    forces: bool | float = False,
    polarization_tensors: bool | float = False,
    dielectric_tensor: bool | float = False,
    born_charges: bool | float = False,
    raman_tensors: bool | float = False,
    nmr_tensors: bool | float = False,
    induced_magnetic_field: bool | float = False,

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
        loss_terms.append(
            gcnn.Loss(
                optax.squared_error, 
                f"nodes.{keys.NMR_TENSORS}",
                f"nodes.{predicted(keys.NMR_TENSORS)}",
            )
        )

    if induced_magnetic_field:
        weights.append(
            1.0 if isinstance(induced_magnetic_field, bool) else induced_magnetic_field
        )
        loss_terms.append(
            L2Regularization(
                f"nodes.{predicted(keys.INDUCED_MAGNETIC_FIELD)}",
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
