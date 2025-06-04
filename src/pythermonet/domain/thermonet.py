from dataclasses import dataclass

from dataclasses_json import dataclass_json
import numpy as np


@dataclass_json
@dataclass
class Thermonet:

    # Thermonet and HHE
    D_gridpipes: float = np.nan    # Distance between forward and return pipe centers (m)
    l_p: float = np.nan     # Pipe thermal conductivity (W/m/K). https://www.wavin.com/da-dk/catalog/Varme/Jordvarme/PE-80-lige-ror/40mm-jordvarme-PE-80PN6-100m
    l_s_H: float = np.nan   # Soil thermal conductivity thermonet and HHE (W/m/K) Guestimate (0.8-1.2 W/m/K)
    l_s_C: float = np.nan   # Soil thermal conductivity thermonet and HHE (W/m/K) Guestimate (0.8-1.2 W/m/K)
    rhoc_s: float = np.nan  # Soil volumetric heat capacity  thermonet and HHE (J/m3/K). Guestimate
    z_grid: float = np.nan  # Burial depth of thermonet and HHE (m)
    T0: float = np.nan      # Yearly average surface temperature (C)
    A: float = np.nan       # Amplitude of yearly sinusoidal temperature variation (C)

    # KART tilføjet topologi information fra TOPO_FILE
    SDR: float = np.nan
    L_traces: float = np.nan   # Trace length (m)
    N_traces: float = np.nan   # Number of traces in a pipe group (-)
    L_segments: float = np.nan # Total pipe length in a pipe group i.e. both forward and return pipes (m)
    I_PG: float = np.nan
    dp_PG: float = np.nan      # Max total pressure drop over the forward plus return pipes in a trace
    V_brine: float = np.nan    # Volume of brine in all pipes (m^3)
    T_dimv: float = np.nan     # Vector of brine temperatures (Celcius) after each of the three pulses (year, month, peak)

    d_selectedPipes_H: float = np.nan
    di_selected_H: float = np.nan
    Re_selected_H: float = np.nan
    d_selectedPipes_C: float = np.nan
    di_selected_C: float = np.nan
    Re_selected_C: float = np.nan

    # KART: tag stilling til om det er nødvendigt at beholde dem.
    # R_H: float = np.nan
    # R_C: float = np.nan
