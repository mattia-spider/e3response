import collections.abc
from collections.abc import Callable
import functools
import logging
import os
import pathlib
from typing import Final

import ase
import ase.io
import ase.stress
import jraph
import numpy as np
from tensorial import gcnn

from e3response import keys
from e3response.data._limit import apply_limit

__all__ = ("BtoAllegroPolDataset",)

_LOGGER = logging.getLogger(__name__)

URL: Final[str] = "https://raw.githubusercontent.com/mir-group/allegro-pol/main/data/BaTiO3.xyz"
FILENAME: Final[str] = "BaTiO3.xyz"

# Every label in the file, under the key it has in the graphs
atomic = keys.FORCES, keys.BORN_CHARGES
global_labels = keys.TOTAL_ENERGY, keys.STRESS, keys.DIPOLE_MOMENT, keys.POLARIZABILITY


class BtoAllegroPolDataset(collections.abc.Sequence[jraph.GraphsTuple]):
    """The barium titanate dataset of allegro-pol (Falletta et al.): 59 snapshots of a 135-atom
    3x3x3 tetragonal supercell with energy, forces, stress, dipole moment, Born charges and
    polarizability, in eV, Å and V.

    The labels are stored as e3response defines them:

    - ``born_charges`` (n_atom, 3, 3): Z*_{i,αγ} = ∂d_α/∂u_{iγ} = ∂F_{iγ}/∂ε_α, field index first,
      the order of :class:`e3response.electric_linked.BornCharges`, in e
    - ``dipole_moment`` (3,): the cell dipole d = -∂E/∂ε in e·Å, which allegro-pol calls the
      polarization.  It is only defined up to a lattice vector (times e): the values are folded
      into about ±L/2, so a plain loss on it is wrong for snapshots near a fold.
    - ``polarizability`` (3, 3): χ_αβ = ∂d_α/∂ε_β in e·Å²/V, as
      :class:`e3response.electric_linked.DielectricTensor` stores it
    - ``energy`` (eV), ``forces`` (eV/Å) and ``stress`` (3, 3) (eV/Å³, ASE sign)

    A plain sequence of graphs, to be split and batched by
    :class:`tensorial.gcnn.data.GraphDataModule`, e.g.::

        _target_: tensorial.gcnn.data.GraphDataModule.from_random_split   # or .from_kfold
        dataset:
          _target_: e3response.data.BtoAllegroPolDataset
          r_max: ${r_max}
          data_dir: ${paths.data_dir}/bto_allegro_pol
    """

    def __init__(
        self,
        r_max: float,
        data_dir: str | pathlib.Path = "data/bto_allegro_pol",
        labels: collections.abc.Sequence[str] = atomic + global_labels,
        include_electric_field: bool = False,
        limit: int | str | None = None,
    ) -> None:
        """
        :param r_max: The cutoff radius used to build the graphs.
        :param data_dir: The directory holding ``BaTiO3.xyz``, downloaded there on first use.
        :param labels: The labels to load, from ``forces``, ``born_charges``, ``energy``,
            ``stress``, ``dipole_moment`` and ``polarizability``.
        :param include_electric_field: Give every structure a zero external electric field, for
            models that take the field as an input.
        :param limit: Which snapshots to load, as in :func:`e3response.data._limit.apply_limit`.
        """
        for label in labels:
            if label not in atomic + global_labels:
                raise ValueError(
                    f"Unknown label '{label}', choose from {atomic + global_labels}"
                )

        path = _ensure_file(pathlib.Path(data_dir))
        structures = [_relabel(atoms) for atoms in ase.io.read(path, index=":")]
        structures = list(apply_limit(structures, limit))

        global_include = [label for label in labels if label in global_labels]
        if include_electric_field:
            for structure in structures:
                structure.info[keys.EXTERNAL_ELECTRIC_FIELD] = np.zeros(3)
            global_include.append(keys.EXTERNAL_ELECTRIC_FIELD)
        to_graph: Callable[[ase.Atoms], jraph.GraphsTuple] = functools.partial(
            gcnn.atomic.graph_from_ase,
            r_max=r_max,
            atom_include_keys=("numbers", *[label for label in labels if label in atomic]),
            global_include_keys=global_include,
            use_calculator=False,
        )
        self._graphs: Final[list[jraph.GraphsTuple]] = list(map(to_graph, structures))

    def __len__(self) -> int:
        return len(self._graphs)

    def __getitem__(self, index):
        return self._graphs[index]


def _ensure_file(data_dir: pathlib.Path) -> pathlib.Path:
    path = data_dir / FILENAME
    if not path.exists():
        import urllib.request  # pylint: disable=import-outside-toplevel

        data_dir.mkdir(parents=True, exist_ok=True)
        _LOGGER.info("Downloading %s to %s", URL, path)
        # Download next to the target and rename, so an interrupted download leaves no file
        partial = path.with_suffix(".xyz.part")
        urllib.request.urlretrieve(URL, partial)  # nosec B310
        os.replace(partial, path)
    return path


def _relabel(atoms: ase.Atoms) -> ase.Atoms:
    """Move the labels to where graph_from_ase reads them -- per-atom ones in `arrays`, per-
    structure ones in `info` -- under e3response's keys and shapes.

    ASE's extxyz reader puts forces, stress and the polarization in a calculator, and keeps the
    Born charges and polarizability as flat 9-vectors.
    """
    results = atoms.calc.results if atoms.calc is not None else {}
    n_atoms = len(atoms)

    relabelled = ase.Atoms(
        numbers=atoms.numbers, positions=atoms.positions, cell=atoms.cell, pbc=atoms.pbc
    )
    # Row-major, so the first index is the polarization (field) one: allegro-pol stores
    # Z*[i, α, γ] = ∂P_α/∂r_{iγ}
    relabelled.arrays[keys.BORN_CHARGES] = atoms.arrays["born_charge"].reshape(n_atoms, 3, 3)
    relabelled.arrays[keys.FORCES] = np.asarray(results["forces"])
    relabelled.info[keys.TOTAL_ENERGY] = np.asarray(atoms.info["total_energy"])
    relabelled.info[keys.STRESS] = ase.stress.voigt_6_to_full_3x3_stress(results["stress"])
    relabelled.info[keys.DIPOLE_MOMENT] = np.asarray(results["polarization"])
    relabelled.info[keys.POLARIZABILITY] = np.asarray(atoms.info["polarizability"]).reshape(3, 3)
    return relabelled
