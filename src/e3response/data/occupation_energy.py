"""Self-consistent DFT+U states of Fe(II) MOFs found by constrained-occupation scans: structure and
Hubbard occupation matrices in, energy out.  Built by ``scripts/build_occupation_dataset.py``."""

import collections.abc
from collections.abc import Sequence
import json
import logging
import pathlib
from typing import Any, Final

from ase import Atoms
import jraph
import numpy as np
from tensorial import gcnn

from e3response import keys

_LOGGER = logging.getLogger(__name__)

__all__ = ("OccupationEnergyDataset",)


class OccupationEnergyDataset(collections.abc.Sequence[jraph.GraphsTuple]):
    """One graph per state: the atoms of the scanned structure, with the occupation matrices of
    each spin channel on the Hubbard atoms (zero elsewhere), U per atom, and the energies.

    Energies are in eV.  The total energies of different structures are thousands of eV apart and
    there are too few structures to fit per-element offsets, so each scan group (one geometry at
    one U) gets its own reference: the target ``energy`` is E - <E - E_U>_group, where E_U is the
    Hubbard energy of the state, also stored as ``hubbard_energy``.  A model then learns E - E_U
    about its group mean and adds E_U in closed form (:class:`e3response.occupation.HubbardEnergy`).
    The reference uses every state of the group, so it carries one number per group from the
    test states.

    The scan group of each state is in :attr:`groups`, for a split stratified by structure with
    :class:`e3response.data.GroupedKFoldSplit`.
    """

    def __init__(
        self,
        r_max: float,
        data_file: str | pathlib.Path = "data/occupation_energy/fe_mof_scans.json",
        scans: Sequence[str] | None = None,
    ) -> None:
        """
        :param r_max: The cutoff radius used to build the graphs.
        :param data_file: The JSON file written by ``scripts/build_occupation_dataset.py``.
        :param scans: The scan groups (e.g. ``mof-hc_Usc5.7716_pk10528``) or compounds (e.g.
            ``mof-hc``) to load, None for all.
        """
        self._rmax: Final[float] = r_max
        self._data_file: Final[str] = str(data_file)
        self._scans = None if scans is None else [str(scan) for scan in scans]

        self.structures: Final[list[Atoms]] = self._load_structures()
        self.groups: Final[list[str]] = [atoms.info["group"] for atoms in self.structures]
        self._graphs: Final[list[jraph.GraphsTuple]] = list(map(self._to_graph, self.structures))

    def __len__(self) -> int:
        return len(self._graphs)

    def __getitem__(self, index):
        return self._graphs[index]

    def _to_graph(self, atoms: Atoms) -> jraph.GraphsTuple:
        return gcnn.atomic.graph_from_ase(
            atoms,
            r_max=self._rmax,
            atom_include_keys=(
                "numbers",
                keys.OCCUPATION_UP,
                keys.OCCUPATION_DOWN,
                keys.HUBBARD_U,
                keys.HUBBARD_SITE,
            ),
            global_include_keys=(keys.TOTAL_ENERGY, keys.HUBBARD_ENERGY),
        )

    def _load_structures(self) -> list[Atoms]:
        path = pathlib.Path(self._data_file)
        _LOGGER.info("Loading dataset from %s", path.absolute())
        with open(path, encoding="utf-8") as file:
            entries = json.load(file)

        if self._scans is not None:
            unknown = (
                set(self._scans) - {e["group"] for e in entries} - {e["compound"] for e in entries}
            )
            if unknown:
                raise ValueError(f"Unknown scans: {sorted(unknown)}")
            entries = [e for e in entries if {e["group"], e["compound"]} & set(self._scans)]

        structures = []
        for entry in entries:
            states = entry["states"]
            reference = np.mean([s["energy"] - s["hubbard_energy"] for s in states])
            structures += [self._state_to_atoms(entry, state, reference) for state in states]
            _LOGGER.info("%s: %d states", entry["group"], len(states))
        _LOGGER.info("Number of loaded states: %d", len(structures))
        return structures

    @staticmethod
    def _state_to_atoms(entry: dict[str, Any], state: dict[str, Any], reference: float) -> Atoms:
        atoms = Atoms(
            numbers=entry["numbers"],
            positions=entry["positions"],
            cell=entry["cell"],
            pbc=entry["pbc"],
        )
        sites = entry["hubbard_sites"]
        occupations = np.asarray(state["occupations"])  # (n_sites, 2, m, m)
        m = occupations.shape[-1]
        up, down = np.zeros((len(atoms), m, m)), np.zeros((len(atoms), m, m))
        up[sites], down[sites] = occupations[:, 0], occupations[:, 1]
        u = np.zeros(len(atoms))
        u[sites] = entry["hubbard_u"]
        atoms.arrays[keys.OCCUPATION_UP] = up
        atoms.arrays[keys.OCCUPATION_DOWN] = down
        atoms.arrays[keys.HUBBARD_U] = u
        atoms.arrays[keys.HUBBARD_SITE] = np.isin(np.arange(len(atoms)), sites)

        atoms.info[keys.TOTAL_ENERGY] = np.asarray(state["energy"] - reference)
        atoms.info[keys.HUBBARD_ENERGY] = np.asarray(state["hubbard_energy"])
        atoms.info["group"] = entry["group"]
        atoms.info["compound"] = entry["compound"]
        atoms.info["pk"] = state["pk"]
        atoms.info["total_magnetization"] = state["total_magnetization"]
        return atoms
