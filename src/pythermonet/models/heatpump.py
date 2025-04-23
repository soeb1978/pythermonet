# Standard library imports
from dataclasses import dataclass

# Third-party imports
from dataclasses_json import dataclass_json
import numpy as np


@dataclass_json
@dataclass
class HeatPump:
    # Heat pump
    Ti_H: float = np.nan      # Design temperature for inlet (C) OK. Stress test conditions. Legislation stipulates Thi > -4C. Auxillary heater must be considered.
    Ti_C: float = np.nan      # Design temperature for inlet (C) OK. Stress test conditions. Legislation stipulates Thi > -4C. Auxillary heater must be considered.
    f_peak_H: float = np.nan  # Fraction of peak heating demand to be covered by the heat pump [0-1]. If f_peak = 0.8 then the heat pump delivers 80% of the peak heating load. The deficit is then supplied by an auxilliary heating device
    f_peak_C: float = np.nan
    HP_IDs: float = np.nan    # Unique IDs for each heatpump in the grid

    # Peak load duration in hours
    t_peak_H: float = np.nan
    t_peak_C: float = np.nan

    # Heating mode parameters
    dT_H: float = np.nan
    P_s_H: float = np.nan

    # Cooling mode parameters
    dT_C: float = np.nan
    P_s_C: float = np.nan
