import collections.abc
from collections.abc import Callable, Iterable, Sequence
import functools
import logging
import pathlib
import re
import tarfile
from typing import Final

import ase
import ase.io
from ase.io import espresso
import jraph
import numpy as np
from tensorial import gcnn

from e3response import keys

__all__ = ("BtoDataset", "BtoDataModule")

_LOGGER = logging.getLogger(__name__)

atomic = "raman_tensors", "born_charges"
global_tensors = ("dielectric",)


class BtoDataset(collections.abc.Sequence[jraph.GraphsTuple]):
    """A barium titanate dataset containing various tensorial quantities created by Lorenzo
    Bastonero: Born charges and Raman tensors per atom, the dielectric tensor per structure.

    A plain sequence of graphs, to be split and batched by
    :class:`tensorial.gcnn.data.GraphDataModule`, e.g.::

        _target_: tensorial.gcnn.data.GraphDataModule.from_random_split   # or .from_kfold
        dataset:
          _target_: e3response.data.BtoDataset
          r_max: ${r_max}
          data_dir: ${paths.data_dir}/bto

    The structures of all archives are pooled; they are no longer split archive by archive.
    """

    def __init__(
        self,
        r_max: float,
        data_dir: str | pathlib.Path = "data/bto",
        archives: Sequence[str] = (
            "BTO_Pm-3m_5atoms_400K_3x3x3_ensemble.tar.gz",
            "BTO_Pm-3m_5atoms_800K_3x3x3.tar.gz",
        ),
        tensors: Sequence[str] = ("raman_tensors", "born_charges", "dielectric"),
        include_electric_field: bool = False,
        drop_unphysical_dielectric: bool = False,
    ) -> None:
        """
        :param r_max: The cutoff radius used to build the graphs.
        :param data_dir: The directory holding the archives.
        :param archives: The archives to read, each extracted next to itself on first use.
        :param tensors: The tensors to load, from ``raman_tensors``, ``born_charges`` and
            ``dielectric``.
        :param include_electric_field: Give every structure a zero external electric field, for
            models that take the field as an input.
        :param drop_unphysical_dielectric: Drop structures whose dielectric tensor has an
            eigenvalue below 1 (33 of the 228 in the default archives).
        """
        for tensor in tensors:
            if tensor not in atomic + global_tensors:
                raise ValueError(
                    f"Unknown tensor type name '{tensor}', choose from {atomic + global_tensors}"
                )

        structures: list[ase.Atoms] = []
        for archive in archives:
            output_dir = _extract_tarball(pathlib.Path(data_dir), archive)
            found = get_structures(output_dir, tensors, include_electric_field)
            if not found:
                _LOGGER.warning("No structure extracted from archive `%s`, skipping", archive)
                continue
            if drop_unphysical_dielectric and "dielectric" in tensors:
                found = _drop_unphysical_dielectric(found, archive)
            structures.extend(found)

        global_include = [tensor for tensor in tensors if tensor in global_tensors]
        if include_electric_field:
            global_include.append(keys.EXTERNAL_ELECTRIC_FIELD)
        to_graph: Callable[[ase.Atoms], jraph.GraphsTuple] = functools.partial(
            gcnn.atomic.graph_from_ase,
            r_max=r_max,
            atom_include_keys=("numbers", "forces", *[t for t in tensors if t in atomic]),
            global_include_keys=global_include,
            key_mapping={"dielectric": keys.DIELECTRIC_TENSOR},
        )
        self._graphs: Final[list[jraph.GraphsTuple]] = list(map(to_graph, structures))

    def __len__(self) -> int:
        return len(self._graphs)

    def __getitem__(self, index):
        return self._graphs[index]


def BtoDataModule(  # pylint: disable=invalid-name
    r_max: float,
    data_dir: str | pathlib.Path = "data/bto",
    archives: Sequence[str] = (
        "BTO_Pm-3m_5atoms_400K_3x3x3_ensemble.tar.gz",
        "BTO_Pm-3m_5atoms_800K_3x3x3.tar.gz",
    ),
    tensors: Sequence[str] = ("raman_tensors", "born_charges", "dielectric"),
    train_val_test_split: Sequence[float] = (0.8, 0.1, 0.1),
    batch_size: int = 64,
    include_electric_field: bool = False,
) -> gcnn.data.GraphDataModule:
    """The previous BTO data module, kept for the configs that build it by name (the BTO example):
    a :class:`tensorial.gcnn.data.GraphDataModule` with a random split of a :class:`BtoDataset`.

    New configs should build the ``GraphDataModule`` directly (see ``configs/data/bto.yaml``),
    which also offers k-fold cross-validation and the filter for corrupted dielectric tensors.
    """
    dataset = BtoDataset(
        r_max,
        data_dir=data_dir,
        archives=archives,
        tensors=tensors,
        include_electric_field=include_electric_field,
    )
    return gcnn.data.GraphDataModule.from_random_split(
        dataset, train_val_test_split, batch_size=batch_size
    )


def _extract_tarball(data_dir: pathlib.Path, tar_name: str) -> pathlib.Path:
    tar_path = data_dir / tar_name
    output_dir = data_dir / tar_name.split(".")[0]

    # Extract the full tarball
    if not output_dir.exists():
        with tarfile.open(tar_path, "r", encoding="utf-8") as tar:
            tar.extractall(data_dir)  # nosec

    return output_dir


def get_tensors(root_dir: pathlib.Path, tensor_type: str, index: str) -> np.ndarray:
    return np.load(f"{root_dir}/{tensor_type}/{index}.npy")


def _drop_unphysical_dielectric(structures: list[ase.Atoms], archive: str) -> list[ase.Atoms]:
    """Drop structures whose dielectric tensor has an eigenvalue below 1, which no physical ε∞ has.

    In the BTO archives 33 of 228 structures have one diagonal component near -64 while the others
    are near 5, the signature of a failed calculation rather than of the material.
    """
    kept = [s for s in structures if np.linalg.eigvalsh(s.info["dielectric"]).min() >= 1.0]
    if len(kept) < len(structures):
        _LOGGER.warning(
            "Dropped %d of %d structures with an unphysical dielectric tensor from `%s`",
            len(structures) - len(kept), len(structures), archive,
        )
    return kept


def get_structures(
    root_dir: pathlib.Path, tensors: Iterable[str], include_electric_field: bool = False
) -> list[ase.Atoms]:
    structures_dir = pathlib.Path(root_dir) / "structures" / "cif"
    _LOGGER.info("Loading structures from: %s", structures_dir.absolute())

    structures: list[ase.Atoms] = []
    for filename in structures_dir.iterdir():
        try:
            structure = ase.io.read(filename, format="cif")
            assert isinstance(structure, ase.Atoms), "ase.io.read did not return ase.Atoms"

            structure_number = re.findall(r"\d\d\d\d\d", filename.name)[0]

            for tensor in tensors:
                tens = get_tensors(root_dir, tensor, structure_number)
                # Per-atom tensors go in `arrays`, per-structure ones in `info`, which is where
                # graph_from_ase reads the global keys from
                if tensor in global_tensors:
                    structure.info[tensor] = tens
                else:
                    structure.arrays[tensor] = tens

            if include_electric_field:
                structure.info[keys.EXTERNAL_ELECTRIC_FIELD] = np.zeros(3)
        except FileNotFoundError:
            continue

        structures.append(structure)

    return structures


def read_scf(filename) -> ase.Atoms:
    with open(filename, encoding="utf-8") as fileobj:
        _data, card_lines = espresso.read_fortran_namelist(fileobj)

    cell, _ = espresso.get_cell_parameters(card_lines)

    nat = 0
    counting = False
    for line in card_lines:
        if counting:
            if not line.strip():
                break

            nat += 1
        elif line.strip().startswith("ATOMIC_POSITIONS"):
            counting = True

    positions_card = espresso.get_atomic_positions(card_lines, n_atoms=nat, cell=cell)

    symbols = [espresso.label_to_symbol(position[0]) for position in positions_card]
    positions = [position[1] for position in positions_card]

    # TODO: put more info into the atoms object
    # e.g magmom, forces.
    atoms = ase.Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)

    return atoms
