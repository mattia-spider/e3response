"""Build the Fe(II) MOF occupation-energy dataset from the constrained-DFT scans.

Reads the LordCapulet scan files of ``constrainedDFT_project`` and writes one JSON file with, for
every scanned structure (one geometry at one U), its atoms and the unique self-consistent states
found by the scan: the Hubbard occupation matrices, the total energy and the Hubbard energy.  It is
the input of :class:`e3response.data.occupation_energy.OccupationEnergyDataset`.

What it relies on, checked when the data was audited (2026-10/11):

- ``oscdft_type=2`` switches the constraint off once converged, so every state is a self-consistent
  DFT+U solution and its occupation matrices are the SCF output, not the scan target.
- The CIFs hold no orientation: the occupation matrices are in the frame of the CIF as ASE reads
  it, as the submission built ``StructureData(ase=atoms)``.
- The occupation matrices are keyed by the index of the Hubbard atom; they are assigned to the Fe
  atoms of the CIF in index order.  In the structures with two Fe the two sites are related by
  inversion, so the model cannot tell them apart; only the Hubbard energy depends on it, and it is
  computed with the U of each atom's kind as Quantum ESPRESSO does.
- The Hubbard energy is not stored by the scans: it is recomputed from the occupation matrices
  and U with ``scan_tools.loaders.hubbard_energy``.
- ``qmof-65807e0_U5.79_scan_data_oneshot.json`` holds the cell of qmof-9024308, not 65807e0, so it
  is left out (``qmof-65807e0_U5.79_corr`` is the real scan).

States are deduplicated with the rule of ``scan_tools`` (energies within 5 meV and occupation
matrices within 10% in each spin channel), also matching a state against the spin-swapped copy of
another, as swapping up and down everywhere gives the same state.  The lowest energy is kept.

Usage::

    python scripts/build_occupation_dataset.py \
        --root /path/to/constrainedDFT_project --out data/occupation_energy/fe_mof_scans.json
"""

import argparse
import glob
import json
import os
import pathlib
import sys

import ase.io
import numpy as np

# Scan group -> (compound, CIF relative to the project root).  The group is the scan file name up
# to "_scan_data", without the "_gaussian" of the HS mof-hc scan run with gaussian smearing.
GROUPS = {
    "mof-hc_LS_Usc6.8386_pk10851": ("mof-hc", "sc_structures_LS/mof-hc_LS_Usc6.8386_pk10851.cif"),
    "mof-hc_Usc5.7716_pk10528": ("mof-hc", "sc_structures/mof-hc_Usc5.7716_pk10528_eff.cif"),
    "qmof-12d3e65_U7.2308": (
        "qmof-12d3e65",
        "oneshot_structures/qmof-12d3e65_U7.2308.cif.collapsed_spin_superseded",
    ),
    "qmof-12d3e65_Usc4.9005_pk769": (
        "qmof-12d3e65",
        "sc_structures/qmof-12d3e65_Usc4.9005_pk769.cif",
    ),
    "qmof-65807e0_U5.79_corr": ("qmof-65807e0", "oneshot_structures/qmof-65807e0_U5.79.cif"),
    "qmof-65807e0_Usc4.863_4.783_pk6878": (
        "qmof-65807e0",
        "sc_structures/qmof-65807e0_Usc4.863_4.783_pk6878.cif",
    ),
    "qmof-77f07ab_U6.5838": (
        "qmof-77f07ab",
        "oneshot_structures/qmof-77f07ab_U6.5838.cif.collapsed_spin_superseded",
    ),
    "qmof-77f07ab_Usc5.0294": (
        "qmof-77f07ab",
        "sc_structures/qmof-77f07ab_Usc5.0294_pk11242.cif",
    ),
    "qmof-9024308_U6.0708": ("qmof-9024308", "oneshot_structures/qmof-9024308_U6.0708.cif"),
    "qmof-9024308_Usc5.1136_pk3995": (
        "qmof-9024308",
        "sc_structures/qmof-9024308_Usc5.1136_pk3995.cif",
    ),
}

SCAN_DIRS = ("sc_structures_scans", "oneshot_structures_scans")
HUBBARD_Z = 26

# scan_tools duplicate rule
DUPLICATE_ENERGY = 0.005  # eV
DUPLICATE_OCCUPATION = 0.10  # relative Frobenius distance, per spin channel


def group_of(path: str) -> str:
    """The scan group of a scan file, from its ``<group>_results`` directory."""
    name = os.path.basename(os.path.dirname(path))
    if name in ("scan_500", "500"):
        name = os.path.basename(os.path.dirname(os.path.dirname(path)))
    return name.removesuffix("_results").replace("_gaussian", "")


def load_calculations(root: pathlib.Path) -> dict[str, dict[str, dict]]:
    """Every calculation of every scan group, by PK.  Merged scan files repeat PKs; the first copy
    is kept."""
    files = []
    for scan_dir in SCAN_DIRS:
        files += glob.glob(
            str(root / scan_dir / "*_results" / "**" / "*scan_data*.json"), recursive=True
        )
    groups: dict[str, dict[str, dict]] = {}
    for path in sorted(files):
        with open(path, encoding="utf-8") as file:
            calculations = json.load(file)["calculations"]
        groups.setdefault(group_of(path), {})
        for pk, calc in calculations.items():
            groups[group_of(path)].setdefault(pk, calc)
    return groups


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    """scan_tools' relative distance of two (n_sites, 2, m, m) states, the worst spin channel."""
    return max(
        np.linalg.norm(a[:, s] - b[:, s])
        / np.sqrt(np.linalg.norm(a[:, s]) * np.linalg.norm(b[:, s]))
        for s in (0, 1)
    )


def unique_states(states: list[dict]) -> list[dict]:
    """The states that are not duplicates of a lower-energy one, see the module docstring."""
    states = sorted(states, key=lambda state: state["energy"])
    kept: list[dict] = []
    for state in states:
        n = state["occupations"]
        for other in kept:
            m = other["occupations"]
            if abs(state["energy"] - other["energy"]) <= DUPLICATE_ENERGY and (
                _distance(n, m) <= DUPLICATE_OCCUPATION
                or _distance(n, m[:, ::-1]) <= DUPLICATE_OCCUPATION
            ):
                other["n_duplicates"] += 1
                break
        else:
            kept.append(state | {"n_duplicates": 0})
    return kept


def build_group(group: str, calculations: dict[str, dict], root: pathlib.Path, loaders) -> dict:
    compound, cif = GROUPS[group]
    atoms = ase.io.read(root / cif, format="cif")
    hubbard_sites = np.flatnonzero(atoms.numbers == HUBBARD_Z)
    u_spec = loaders.hubbard_u_from_path(f"{group}_scan_data.json")

    states = []
    for pk, calc in calculations.items():
        if not calc.get("converged") or calc.get("exit_status") not in (0, "0"):
            continue
        occupations = loaders.occupation_tensor(calc)
        if occupations.size == 0:
            continue
        out = calc["output_parameters"]
        if out["number_of_atoms"] != len(atoms) or not np.isclose(
            out["volume"], atoms.get_volume(), rtol=1e-3
        ):
            raise ValueError(f"{group}: PK {pk} does not match {cif} (number of atoms or volume)")
        if occupations.shape[0] != len(hubbard_sites):
            raise ValueError(
                f"{group}: PK {pk} has {occupations.shape[0]} Hubbard atoms, "
                f"{cif} has {len(hubbard_sites)} Fe"
            )
        u_values = loaders.resolve_u(u_spec, loaders.atom_items(calc))
        states.append(
            {
                "pk": int(pk),
                "source": calc.get("calculation_source"),
                "occupations": occupations,
                "hubbard_u": u_values,
                "energy": out["energy"],
                "hubbard_energy": loaders.hubbard_energy(calc, u_spec),
                "total_magnetization": out.get("total_magnetization"),
                "absolute_magnetization": out.get("absolute_magnetization"),
                "smearing": [out.get("smearing_type"), out.get("degauss")],
                "smearing_energy": out.get("energy_smearing"),
            }
        )

    u_values = {tuple(state.pop("hubbard_u")) for state in states}
    if len(u_values) != 1:
        raise ValueError(f"{group}: the U of the Fe atoms differs between calculations: {u_values}")
    n_converged = len(states)
    states = unique_states(states)
    print(
        f"{group:38s} {compound:14s} {len(atoms):3d} atoms, Fe {hubbard_sites.tolist()}, "
        f"U {u_values}: {n_converged} converged -> {len(states)} unique"
    )

    return {
        "group": group,
        "compound": compound,
        "cif": cif,
        "numbers": atoms.numbers.tolist(),
        "positions": atoms.positions.tolist(),
        "cell": atoms.cell.array.tolist(),
        "pbc": atoms.pbc.tolist(),
        "hubbard_sites": hubbard_sites.tolist(),
        "hubbard_u": list(next(iter(u_values))),
        "states": [state | {"occupations": state["occupations"].tolist()} for state in states],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument(
        "--root", type=pathlib.Path, required=True, help="the constrainedDFT_project directory"
    )
    parser.add_argument(
        "--out", type=pathlib.Path, default=pathlib.Path("data/occupation_energy/fe_mof_scans.json")
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.root / "scan_tools"))
    from scan_tools import loaders  # pylint: disable=import-outside-toplevel

    calculations = load_calculations(args.root)
    skipped = sorted(set(calculations) - set(GROUPS))
    if skipped:
        print("Left out:", ", ".join(skipped))
    entries = [build_group(group, calculations[group], args.root, loaders) for group in GROUPS]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as file:
        json.dump(entries, file)
    print(f"{sum(len(e['states']) for e in entries)} states in {len(entries)} groups -> {args.out}")


if __name__ == "__main__":
    main()
