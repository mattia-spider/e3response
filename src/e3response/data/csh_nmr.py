import collections.abc
import json
import logging
import pathlib
from typing import Any, Final, Optional, Sequence, Union

from ase import Atoms
from ase.neighborlist import neighbor_list
from flax import nnx
import jraph
import numpy as np
from tensorial import gcnn

from e3response import keys
from e3response.data._limit import apply_limit
from e3response.data._splits import split_indices

_LOGGER = logging.getLogger(__name__)

__all__ = ("CshNmrDataset",)


# An O with exactly two H closer than this is a water molecule (O-H ~0.97 A, while the H-bond
# partner of a hydroxyl or water H sits at ~1.6 A or more)
WATER_OH_CUTOFF: Final[float] = 1.25


def water_atoms(atoms: Atoms) -> np.ndarray:
    """Boolean per-atom mask of the O and H atoms that form water molecules."""
    first, second = neighbor_list("ij", atoms, WATER_OH_CUTOFF)
    numbers = atoms.numbers
    oh = (numbers[first] == 8) & (numbers[second] == 1)
    n_h = np.bincount(first[oh], minlength=len(atoms))
    water_o = (numbers == 8) & (n_h == 2)
    water = water_o.copy()
    water[second[oh & water_o[first]]] = True
    return water


def nmr_active(atoms: Atoms, sites: Union[str, Sequence[int], None]) -> Optional[np.ndarray]:
    """Which atoms the NMR loss and metrics count, or None for all of them.

    :param sites: "all" (or None) for every atom, "no_water" to drop the water O and H, or a
        list of atomic numbers to keep only those elements (e.g. [11] for Na only).
    """
    if sites is None or sites == "all":
        return None
    if sites == "no_water":
        return ~water_atoms(atoms)
    if isinstance(sites, str):
        raise ValueError(f"Unknown nmr_sites '{sites}': use 'all', 'no_water' or atomic numbers")
    return np.isin(atoms.numbers, [int(z) for z in sites])


def _apply_limit(structures: list[Atoms], limit: int | str | None) -> list[Atoms]:
    """Restrict a structure list according to `limit`. Shared by the constructor
    `limit` (applied to the full dataset in `_load_structures`) and `load_split`'s
    own `limit` (applied to a single split) so both behave identically:

    - None                        → all structures
    - "bulk"/"surface"            → keep only that ``struct_type``
    - int N                       → first N structures
    - "a:b"/"a:b:s"               → Python-slice semantics
    - "random:N"/"random:N:seed"  → N structures drawn at random (see apply_limit)
    """
    if isinstance(limit, str):
        limit_lower = limit.lower()
        if limit_lower == "bulk":
            return [s for s in structures if s.info.get("struct_type") == "bulk"]
        if limit_lower == "surface":
            return [s for s in structures if s.info.get("struct_type") == "surf"]
    return apply_limit(structures, limit)


class CshNmrDataset(collections.abc.Sequence[jraph.GraphsTuple]):
    """Calcium Silicate Hydrate dataset from a pre-processed JSON file containing ASE atoms
    objects with NMR tensor data.

    A plain sequence of graphs, to be split and batched by
    :class:`tensorial.gcnn.data.GraphDataModule` (see ``configs/data/csh_nmr.yaml``).  Each
    structure's type ("bulk"/"surf") is in :attr:`groups`, for a stratified split with
    :class:`e3response.data.GroupedRandomSplit`.
    """

    def __init__(
        self,
        r_max: float,
        data_file: Union[str, pathlib.Path] = "data/csh_nmr/Na_csh.json",
        limit: Optional[Union[int, str]] = None,
        nmr_sites: Union[str, Sequence[int], None] = "all",
    ) -> None:
        """
        :param r_max: The cutoff radius used to build the graphs.
        :param data_file: The JSON file with the structures and their tensors.
        :param limit: Which structures to load, see `_apply_limit` ("bulk", "surface", N, ...).
        :param nmr_sites: which atoms the NMR loss and metrics count, see `nmr_active`; the
            others stay in the graph as environment.
        """
        self._rmax: Final[float] = r_max
        self._data_file: Final[str] = str(data_file)
        self._limit = limit
        if nmr_sites is not None and not isinstance(nmr_sites, str):
            nmr_sites = [int(z) for z in nmr_sites]  # a ListConfig when it comes from Hydra

        self.structures: Final[list[Atoms]] = self._load_structures()
        n_active = n_atoms = 0
        for atoms in self.structures:
            active = nmr_active(atoms, nmr_sites)
            if active is not None:
                atoms.arrays[keys.NMR_ACTIVE] = active
                n_active += int(active.sum())
            n_atoms += len(atoms)
        if nmr_sites not in (None, "all"):
            _LOGGER.info("NMR sites %s: %d of %d atoms", nmr_sites, n_active, n_atoms)
        # Bulk or surface, the label of a stratified split
        self.groups: Final[list] = [atoms.info.get("struct_type") for atoms in self.structures]
        self._graphs: Final[list[jraph.GraphsTuple]] = list(map(self._to_graph, self.structures))

    def __len__(self) -> int:
        return len(self._graphs)

    def __getitem__(self, index):
        return self._graphs[index]

    def _to_graph(self, atoms: Atoms) -> jraph.GraphsTuple:
        atom_keys = ("numbers", "nmr_tensors")
        if keys.NMR_ACTIVE in atoms.arrays:
            atom_keys += (keys.NMR_ACTIVE,)
        return gcnn.atomic.graph_from_ase(
            atoms,
            r_max=self._rmax,
            atom_include_keys=atom_keys,
            global_include_keys=[keys.EXTERNAL_MAGNETIC_FIELD, gcnn.atomic.TOTAL_ENERGY],
        )

    def load_split(
        self,
        split: str,
        strategy: "gcnn.data.SplitStrategy",
        limit: Optional[Union[int, str]] = None,
        rngs: "nnx.Rngs | None" = None,
    ) -> list[jraph.GraphsTuple]:
        """The graphs of one partition ("train"/"val"/"test") of `strategy`, e.g. to recover
        which structures an already-trained run held out.

        :param split: which partition to load: "train", "val" or "test".
        :param strategy: the split strategy used at training time, e.g.
            ``GroupedRandomSplit((0.8, 0.1, 0.1))``.
        :param limit: further restricts the returned split, with the SAME semantics as
            the constructor ``limit`` (see `_apply_limit`): "bulk"/"surface"
            filter by ``struct_type``, int N takes the first N, "start:stop"/"start:stop:step"
            apply Python-slice semantics. `None` (default) returns the whole split. It is
            applied on top of the constructor ``limit``, which already restricted the dataset.
        :param rngs: must match whatever was used at training time to reproduce the
            SAME split; defaults to `nnx.Rngs(0)`, REAX's own default when no
            `Trainer`/`Engine` override is given.
        """
        return list(map(self._to_graph, self.load_split_structures(split, strategy, limit, rngs)))

    def load_split_structures(
        self,
        split: str,
        strategy: "gcnn.data.SplitStrategy",
        limit: Optional[Union[int, str]] = None,
        rngs: "nnx.Rngs | None" = None,
    ) -> list[Atoms]:
        """Same as `load_split`, but returns the split's `ase.Atoms` rather than graphs.

        For analysis code that has to work on the geometry itself (e.g. move an atom and
        rebuild the neighbour list), for which a `jraph.GraphsTuple` — whose topology is
        frozen at construction — is not enough.

        See `load_split` for the parameters.
        """
        indices = split_indices(self, strategy, split, rngs)
        return _apply_limit([self.structures[i] for i in indices], limit)

    def _load_structures(self) -> list[Atoms]:
        path = pathlib.Path(self._data_file)
        _LOGGER.info("Loading dataset from %s", path.absolute())

        with open(path, encoding="utf-8") as file:
            entries = json.load(file)

        structures = [self._entry_to_atoms(entry) for entry in entries]
        structures = _apply_limit(structures, self._limit)

        _LOGGER.info("Number of loaded structures: %d", len(structures))

        return structures

    @staticmethod
    def _entry_to_atoms(entry: dict[str, Any]) -> Atoms:
        atoms = Atoms(
            numbers=entry["numbers"],
            positions=entry["positions"],
            cell=entry["cell"],
            pbc=entry["pbc"],
        )
        atoms.arrays["nmr_tensors"] = np.asarray(entry["nmr_tensors"])
        atoms.info["struct_type"] = entry.get("struct_type")
        atoms.info["ca_si_ratio"] = entry.get("ca_si_ratio")
        # Graph globals are read from `atoms.info`; `atoms.arrays` holds per-atom data only.
        atoms.info[keys.EXTERNAL_MAGNETIC_FIELD] = np.zeros(3)
        atoms.info[gcnn.atomic.TOTAL_ENERGY] = np.asarray(entry.get("energy_Ry"))
        return atoms
