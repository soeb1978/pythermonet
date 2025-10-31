from dataclasses import dataclass

from dataclasses_json import dataclass_json
import numpy as np


@dataclass_json
@dataclass
class BHEConfig:

    # Borehole heat exchangers (BHE)
    source: str = "BHE"

    q_geo: float = np.nan    # Geothermal heat flux (W/m2)
    r_b: float = np.nan      # Borehole radius (m)
    r_p: float = np.nan      # Outer radius of U pipe (m)
    SDR: float = np.nan      # SDR for U-pipe (-)
    l_ss: float = np.nan     # Soil thermal conductivity along BHEs (W/m/K)
    rhoc_ss: float = np.nan  # Volumetric heat capacity of soil (along BHE). Assuming 70# quartz and 30# water (J/m3/K) #OK
    l_g: float = np.nan      # Grout thermal conductivity (W/m/K)
    rhoc_g: float = np.nan   # Grout volumetric heat capacity (J/m3/K)
    D_pipes: float = np.nan  # Wall to wall distance U-pipe legs (m)
    V_brine: float = np.nan  # Volume of brine in heat exchanger pipes (m^3) - calculated
    T_dimv: float = np.nan   # Vector of brine temperatures (Celcius) after each of the three pulses (year, month, peak) - calculated

    # BHE field
    NX: int = np.nan          # Number of boreholes in the x-direction (-)
    D_x: float = np.nan       # Spacing between boreholes in the x-direction (m)
    NY: int = np.nan          # Number of boreholes in the y-direction (-)
    D_y: float = np.nan       # Spacing between boreholes in the y-direction (m)
    gFuncMethod: str = "ICS"  # Method for calculating g-function: "ICS" for Infinite Cylindrical Source (default), "PYG" for pygfunction (finite source)

    # Results
    FPH: float = np.nan
    FPC: float = np.nan
    L_BHE_H: float = np.nan
    L_BHE_C: float = np.nan
    Re_BHEmax_H: float = np.nan
    dpdL_BHEmax_H: float = np.nan
    Re_BHEmax_C: float = np.nan
    dpdL_BHEmax_C: float = np.nan
