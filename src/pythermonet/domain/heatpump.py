from dataclasses import dataclass, field
from typing import List

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

    # This is inherited from HeatPumpInput
    has_cooling: bool = False


@dataclass
class HeatPumpInput:
    """
    Unified dataclass aggregated heating and cooling input loaded from
    various sources.

    This class contains heating and cooling loads, COPs/EER values,
    temperature differences, and the number of consumers. The
    'has_cooling' attribute is automatically set based on whether a
    yearly cooling load is specified.
    """
    heat_pump_id: List[int]
    loads_yearly_heating: List[float]      # Average power over the whole year [W]
    loads_winter_heating: List[float]      # Average power during winter season [W]
    loads_daily_peak_heating: List[float]  # Peak daily heating loads [W]

    cops_yearly_heating: List[float]       # COPs over the whole year
    cops_winter_heating: List[float]       # COPs during winter season
    cops_hourly_peak_heating: List[float]  # COPs during the hourly peak condition
    delta_temps_heating: List[float]       # ΔT across the heat pump, heating mode [°C]

    loads_yearly_cooling: List[float]      # Average cooling power over the year [W]
    loads_summer_cooling: List[float]      # Average cooling power during summer [W]
    loads_daily_peak_cooling: List[float]  # Peak daily cooling load [W]

    eers_cooling: List[float]              # EER for the cooling mode
    delta_temps_cooling: List[float]       # ΔT across the heat pump, cooling mode [°C]

    # This will be set after initialization
    has_cooling: bool = field(init=False)

    def __post_init__(self):
        self.has_cooling = abs(self.load_yearly_cooling) > 1e-6
