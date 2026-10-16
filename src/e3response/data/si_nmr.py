import collections.abc
import json
import logging
import pathlib
from typing import Any, Final, Optional, Sequence, Union

from ase import Atoms
from flax import nnx
import jraph
from monty.json import MontyDecoder
import numpy as np
import pymatgen.io.ase
from tensorial import gcnn

from e3response import keys
from e3response.data._limit import apply_limit
from e3response.data._splits import split_indices

_LOGGER = logging.getLogger(__name__)

__all__ = ("SiNmrDataset",)


class SiNmrDataset(collections.abc.Sequence[jraph.GraphsTuple]):
    """Silicon dataset containing ASE atoms objects with nmr tensors per Silicon atom, nmr
    active atoms mask and Q_n classification.

    A plain sequence of graphs, to be split and batched by
    :class:`tensorial.gcnn.data.GraphDataModule` (see ``configs/data/si_nmr.yaml``).  Each
    structure's Qn signature is in :attr:`groups`, for a stratified split with
    :class:`e3response.data.GroupedRandomSplit`.
    """

    def __init__(
        self,
        r_max: float,
        data_file: Union[str, pathlib.Path] = "data/si_nmr/si_data.json",
        limit: Optional[Union[int, str]] = None,
    ) -> None:
        """
        :param r_max: The cutoff radius used to build the graphs.
        :param data_file: The JSON file with the structures and their tensors.
        :param limit: Which structures to load, as in :func:`e3response.data._limit.apply_limit`.
        """
        self._rmax: Final[float] = r_max
        self._data_file: Final[str] = str(data_file)
        self._limit = limit

        self.structures: Final[list[Atoms]] = self._load_structures()
        # The Qn classes present in each structure, the label of a stratified split
        self.groups: Final[list[tuple]] = [
            tuple(sorted(set(atoms.arrays.get("Qn", [])))) for atoms in self.structures
        ]
        self._graphs: Final[list[jraph.GraphsTuple]] = list(map(self._to_graph, self.structures))

    def __len__(self) -> int:
        return len(self._graphs)

    def __getitem__(self, index):
        return self._graphs[index]

    def _to_graph(self, atoms: Atoms) -> jraph.GraphsTuple:
        return gcnn.atomic.graph_from_ase(
            atoms,
            r_max=self._rmax,
            atom_include_keys=("numbers", "nmr_tensors", "mask"),
            global_include_keys=[],
            key_mapping={"mask": "nmr_active"},
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
        :param limit: further restricts the returned split - int N takes the first N
            structures of the split, "start:stop"/"start:stop:step" applies Python-slice
            semantics. `None` (default) returns the whole split.
        :param rngs: must match whatever was used at training time to reproduce the
            SAME split; defaults to `nnx.Rngs(0)`, REAX's own default when no
            `Trainer`/`Engine` override is given.
        """
        indices = apply_limit(split_indices(self, strategy, split, rngs), limit)
        return [self._graphs[i] for i in indices]

    def _load_structures(self) -> list[Atoms]:
        path = pathlib.Path(self._data_file)
        _LOGGER.info("Loading dataset from %s", path.absolute())

        with open(path, encoding="utf-8") as f:
            entries = json.load(f, cls=MontyDecoder)

        structures = []
        for entry in entries:
            atoms = pymatgen.io.ase.AseAtomsAdaptor.get_atoms(entry["structure"])

            ind = entry["ind"]
            n_atoms = entry["N"]

            tensors = np.zeros((n_atoms, 3, 3))
            tensors[ind] = entry["tensor"]

            mask = np.zeros(n_atoms, dtype=bool)
            mask[ind] = True

            atoms.arrays["nmr_tensors"] = tensors
            atoms.arrays["mask"] = mask
            atoms.arrays["Qn"] = entry["Qn"]

            structures.append(atoms)

        structures = apply_limit(structures, self._limit)

        _LOGGER.info("Number of loaded structures: %d", len(structures))
        return structures
