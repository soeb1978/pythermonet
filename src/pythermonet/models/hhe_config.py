# Standard library imports
from dataclasses import dataclass

# Third-party imports
from dataclasses_json import dataclass_json
import numpy as np


@dataclass_json
@dataclass
class HHEConfig:

    # Horizontal heat exchanger (HHE) topology and pipes
    source: str = "HHE"

    N_HHE: int = np.nan  # Number of HE loops (-)
    d: float = np.nan    # Outer diameter of HE pipe (m)
    SDR: float = np.nan  # SDR for HE pipes (-)
    D: float = np.nan    # Pipe segment spacing (m)
    V_brine: float = np.nan    # Volume of brine in the pipes (m^3) - calculated
    T_dimv: float = np.nan     # Vector of brine temperatures (Celcius) after each of the three pulses (year, month, peak) - calculated

    # Results
    FPH: float = np.nan
    FPC: float = np.nan
    L_HHE_H: float = np.nan
    L_HHE_C: float = np.nan
    Re_HHEmax_H: float = np.nan
    dpdL_HHEmax_H: float = np.nan
    Re_HHEmax_C: float = np.nan
    dpdL_HHEmax_C: float = np.nan
