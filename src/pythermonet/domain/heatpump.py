from dataclasses import dataclass, field
from typing import List, Optional

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
    heat_pump_ids: List[int]
    loads_yearly_heating: Optional[np.ndarray] = None      # Average power over the whole year [W]
    loads_winter_heating: Optional[np.ndarray] = None      # Average power during winter season [W]
    loads_daily_peak_heating: Optional[np.ndarray] = None  # Peak daily heating loads [W]

    cops_yearly_heating: Optional[np.ndarray] = None       # COPs over the whole year
    cops_winter_heating: Optional[np.ndarray] = None       # COPs during winter season
    cops_hourly_peak_heating: Optional[np.ndarray] = None  # COPs during the hourly peak condition
    delta_temps_heating: Optional[np.ndarray] = None       # ΔT across the heat pump, heating mode [°C]

    loads_yearly_cooling: Optional[np.ndarray] = None      # Average cooling power over the year [W]
    loads_summer_cooling: Optional[np.ndarray] = None      # Average cooling power during summer [W]
    loads_daily_peak_cooling: Optional[np.ndarray] = None  # Peak daily cooling load [W]

    eers_cooling: Optional[np.ndarray] = None              # EER for the cooling mode
    delta_temps_cooling: Optional[np.ndarray] = None       # ΔT across the heat pump, cooling mode [°C]

    # This will be set after initialization
    has_heating: bool = field(init=False)
    has_cooling: bool = field(init=False)

    def __post_init__(self):
        # Determine if heating or cooling is present
        self.has_heating = self._array_has_nonzero_values(
            self.loads_yearly_heating
        )
        self.has_cooling = self._array_has_nonzero_values(
            self.loads_yearly_cooling
        )

    @staticmethod
    def _array_has_nonzero_values(*arrays: Optional[np.ndarray]) -> bool:
        for arr in arrays:
            if arr is not None and np.any(np.abs(arr) > 1e-6):
                return True
        return False
