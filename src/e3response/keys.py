from typing import Final

# pylint: disable=unused-wildcard-import
from tensorial.gcnn.atomic.keys import *  # We want all atomic keys pylint: disable=wildcard-import
from tensorial.gcnn.keys import *  # We want all graph keys pylint: disable=wildcard-import

# Electric
EXTERNAL_ELECTRIC_FIELD: Final[str] = "external_electric_field"
POLARIZATION: Final[str] = "polarization"
DIPOLE_MOMENT: Final[str] = "dipole_moment"
POLARIZABILITY: Final[str] = "polarizability"
DIELECTRIC_TENSOR: Final[str] = "dielectric_tensor"
BORN_CHARGES: Final[str] = "born_charges"
RAMAN_TENSORS: Final[str] = "raman_tensors"

# Magnetic
EXTERNAL_MAGNETIC_FIELD: Final[str] = "external_magnetic_field"
INDUCED_MAGNETIC_FIELD: Final[str] = "induced_magnetic_field"
NUCLEAR_MAGNETIC_MOMENT: Final[str] = "nuclear_magnetic_moment"
MAGNETIC_SHIELDING_TENSOR: Final[str] = "magnetic_shielding_tensor"
NMR_TENSORS: Final[str] = "nmr_tensors"
# Atoms that carry an NMR label; datasets that label only some atoms set it, see NmrTensorLoss
NMR_ACTIVE: Final[str] = "nmr_active"


MAGNETIZATION: Final[str] = "magnetization"
MAGNETIC_SUSCEPTIBILITY: Final[str] = "magnetic_susceptibility"

# Hubbard occupations (DFT+U): per-atom occupation matrices of each spin channel, zero on the atoms
# without Hubbard states, U per atom, which atoms carry Hubbard states, and the Hubbard energy
OCCUPATION_UP: Final[str] = "occupation_up"
OCCUPATION_DOWN: Final[str] = "occupation_down"
HUBBARD_U: Final[str] = "hubbard_u"
HUBBARD_SITE: Final[str] = "hubbard_site"
HUBBARD_ENERGY: Final[str] = "hubbard_energy"
