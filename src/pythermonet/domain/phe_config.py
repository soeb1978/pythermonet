from dataclasses import dataclass

from dataclasses_json import dataclass_json
import numpy as np


@dataclass_json
@dataclass
class PHEConfig:

    # Pile heat exchangers (PHE)
    source: str = "PHE"

    S: float = np.nan           # Pile side length (m)
    L: float = np.nan           # Pile active length (m)
    AR: float = np.nan          # Pile aspect ratio (AR = L/S)
    n: int = np.nan             # Number of heat exchanger pipes in pile cross section (-)
    do: float = np.nan          # Pipe outer diameter (m)
    di: float = np.nan          # Pipe inner diameter (m)

    l_c: float = np.nan         # Concrete thermal conductivity
    l_ss: float = np.nan        # Soil thermal conductivity along PHEs (W/m/K)
    rhoc_ss: float = np.nan     # Volumetric heat capacity of soil along PHE (J/m^3/K)

    coord: float = np.nan       # Matrix of pile coordinates (m) - first and second columns contain x- and y-coordinates for the piles. The number of rows is equal to the number of piles
    Rc_coeff: float = np.nan    # Fitting parameters for concrete thermal resistance
    Gc_coeff: float = np.nan    # Fitting parameters for concrete G-functions
    Gg_coeff: float = np.nan    # Fitting parameters for soil G-function
