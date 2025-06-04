from dataclasses import dataclass, field

from dataclasses_json import dataclass_json
import numpy as np


# The old unrefactored version.
@dataclass_json
@dataclass
class AggregatedLoad:

    # Heat pump
    Ti_H: float = np.nan  # Design temperature for inlet (C) OK. Stress test conditions. Legislation stipulates Thi > -4C. Auxillary heater must be considered.
    Ti_C: float = np.nan  # Design temperature for inlet (C) OK. Stress test conditions. Legislation stipulates Thi > -4C. Auxillary heater must be considered.
    To_H: float = np.nan
    To_C: float = np.nan

    f_peak_H: float = np.nan  # Fraction of peak heating demand to be covered by the heat pump [0-1]. If f_peak = 0.8 then the heat pump delivers 80% of the peak heating load. The deficit is then supplied by an auxilliary heating device
    f_peak_C: float = np.nan

    # Peak load duration in hours
    t_peak_H: float = np.nan
    t_peak_C: float = np.nan

    # Heating mode parameters
    Qdim_H: float = np.nan
    P_s_H: float = np.nan

    # Cooling mode parameters
    Qdim_C: float = np.nan
    P_s_C: float = np.nan

    # This is inherited from AggregatedLoadInput
    has_cooling: bool = False


@dataclass
class AggregatedLoadInput:
    """
    Unified dataclass aggregated heating and cooling input loaded from
    various sources.

    This class contains heating and cooling loads, COP/EER values,
    temperature differences, and the number of consumers. The
    'has_cooling' attribute is automatically set based on whether a
    yearly cooling load is specified.
    """
    n_consumers_heating: int = 1
    load_yearly_heating: float = 0.0      # Average power over the whole year [W]
    load_winter_heating: float = 0.0      # Average power during winter season [W]
    load_daily_peak_heating: float = 0.0  # Peak daily heating load [W]

    cop_yearly_heating: float = 1.0       # COP over the whole year
    cop_winter_heating: float = 1.0       # COP during winter season
    cop_hourly_peak_heating: float = 1.0  # COP during the hourly peak condition
    delta_temp_heating: float = 3.0       # ΔT across the heat pump, heating mode [°C]

    n_consumers_cooling: int = 1
    load_yearly_cooling: float = 0.0      # Average cooling power over the year [W]
    load_summer_cooling: float = 0.0      # Average cooling power during summer [W]
    load_daily_peak_cooling: float = 0.0  # Peak daily cooling load [W]

    eer_cooling: float = 1.0              # EER for the cooling mode
    delta_temp_cooling: float = 3.0       # ΔT across the heat pump, cooling mode [°C]

    # This will be set after initialization
    has_cooling: bool = field(init=False)

    def __post_init__(self):
        self.has_cooling = abs(self.load_yearly_cooling) > 1e-6
