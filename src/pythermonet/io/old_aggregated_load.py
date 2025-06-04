import copy
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Literal

import pandas as pd
import numpy as np
from pythermonet.domain import (
    AggregatedLoad,
    Brine,
    RefactoredAggregatedLoad,
    RefactoredBrine
)

from pythermonet.core.physics import source_load_from_cop, mass_flow_from_load


@dataclass
class RefactoredAggregatedLoadInput:
    num_consumers: int = 1
    load_yearly_heating: float = 0.0      # Average power over the whole year [W]
    load_winter_heating: float = 0.0      # Average power during winter season [W]
    load_daily_peak_heating: float = 0.0  # Peak daily heating load [W]

    cop_yearly_heating: float = 1.0       # COP over the whole year
    cop_winter_heating: float = 1.0       # COP during winter season
    cop_hourly_peak_heating: float = 1.0  # COP during the hourly peak condition
    delta_temp_heating: float = 3.0       # ΔT across the heat pump, heating mode [°C]

    load_yearly_cooling: float = 0.0      # Average cooling power over the year [W]
    load_summer_cooling: float = 0.0      # Average cooling power during summer [W]
    load_daily_peak_cooling: float = 0.0  # Peak daily cooling load [W]

    eer_cooling: float = 1.0              # EER for the cooling mode
    delta_temp_cooling: float = 3.0       # ΔT across the heat pump, cooling mode [°C]

    # This will be set after initialization
    has_cooling: bool = field(init=False)

    def __post_init__(self):
        self.has_cooling = abs(self.load_yearly_cooling) > 1e-6


@dataclass
class AggregatedLoadInput:
    num_consumers: int = 1
    load_yearly_heating: float = 0.0      # Average power over the whole year [W]
    load_winter_heating: float = 0.0      # Average power during winter season [W]
    load_daily_peak_heating: float = 0.0  # Peak daily heating load [W]

    cop_yearly_heating: float = 1.0       # COP over the whole year
    cop_winter_heating: float = 1.0       # COP during winter season
    cop_hourly_peak_heating: float = 1.0  # COP during the hourly peak condition
    delta_temp_heating: float = 3.0       # ΔT across the heat pump, heating mode [°C]

    load_yearly_cooling: float = 0.0      # Average cooling power over the year [W]
    load_summer_cooling: float = 0.0      # Average cooling power during summer [W]
    load_daily_peak_cooling: float = 0.0  # Peak daily cooling load [W]

    eer_cooling: float = 1.0              # EER for the cooling mode
    delta_temp_cooling: float = 3.0       # ΔT across the heat pump, cooling mode [°C]

    # This will be set after initialization
    has_cooling: bool = field(init=False)

    def __post_init__(self):
        self.has_cooling = abs(self.load_yearly_cooling) > 1e-6

def aggregated_load_tsv(path: str) -> AggregatedLoadInput:
    """
    Load aggregated load input data from a TSV (tab separated) file into
    the AggregatedLoadInput dataclass.
    """
    # @@@### add longer docstring.
    df = pd.read_csv(path, sep='\t+', engine='python')
    df.columns = [col.strip() for col in df.columns]
    row = df.iloc[0]

    return RefactoredAggregatedLoadInput(
        num_consumers=int(row["No. consumers"]),

        load_yearly_heating=row["Yearly_heating_load_(W)"],
        load_winter_heating=row["Winter_heating_load_(W)"],
        load_daily_peak_heating=row["Daily_heating_load_(W)"],

        cop_yearly_heating=row["Year_COP"],
        cop_winter_heating=row["Winter_COP"],
        cop_hourly_peak_heating=row["Hour_COP"],
        delta_temp_heating=row["dT_HP_Heating"],

        load_yearly_cooling=row["Yearly_cooling_load_(W)"],
        load_summer_cooling=row["Summer_cooling_load_(W)"],
        load_daily_peak_cooling=row["Daily_cooling_load_(W)"],

        eer_cooling=row["EER"],
        delta_temp_cooling=row["dT_HP_Cooling"]
    )


def Refactored_aggload_combine_user_and_file_input(
        aggload_base: RefactoredAggregatedLoad,
        aggload_file_data: RefactoredAggregatedLoadInput,
        brine: RefactoredBrine
        ) -> RefactoredAggregatedLoad:
    """
    Combines the manually specified and file-based input data to produce
    a complete AggregatedLoad instance.

    This function creates a deep copy of the base load profile and
    updates it with the provided input data for heating and, if
    applicable, cooling loads. It also calculates peak mass flow rates
    and the annual load imbalance between heating and cooling.

    Parameters
    ----------
    aggload_base : RefactoredAggregatedLoad
        The base load configuration object to use as a starting point.

    aggload_file_data : RefactoredAggregatedLoadInput
        The input data specifying loads, COP/EER, and temperature
        differences for heating and cooling.

    brine : RefactoredBrine
        The fluid properties (density, heat capacity, etc.) used in
        mass flow calculations.

    Returns
    -------
    complete_agg_load : RefactoredAggregatedLoad
        A new 'RefactoredAggregatedLoad' object containing the updated
        heating and cooling loads, peak mass flows, and annual load
        imbalance.

    Notes
    -----
    - The cooling loads are only updated if the input data indicates
      that cooling is present ('has_cooling' flag is 'True').
    - The load imbalance is calculated as the difference between the
      annual heating and cooling loads, positive if heating load
      dominates.
    """
    complete_agg_load = copy.deepcopy(aggload_base)

    diversity_factor = (
        complete_agg_load.fraction_peak_load_covered_heating
        * (0.62 + 0.38/aggload_base.num_consumers)
    )
    # --- Heating load calculations ---
    complete_agg_load.loads_heating.yearly = source_load_from_cop(
        aggload_file_data.load_yearly_heating,
        aggload_file_data.cop_yearly_heating,
        heating=True
    )
    complete_agg_load.loads_heating.seasonal = source_load_from_cop(
        aggload_file_data.load_winter_heating,
        aggload_file_data.cop_winter_heating,
        heating=True
    )
    complete_agg_load.loads_heating.peak = source_load_from_cop(
        aggload_file_data.load_daily_peak_heating,
        aggload_file_data.cop_hourly_peak_heating,
        heating=True
    ) * diversity_factor

    complete_agg_load.peak_mass_flow_heating = mass_flow_from_load(
        complete_agg_load.loads_heating.peak,
        aggload_file_data.delta_temp_heating,
        brine.density,
        brine.heat_capacity
    )

    complete_agg_load.outlet_temp_heating = (
        complete_agg_load.inlet_temp_heating
        - aggload_file_data.delta_temp_heating
    )

    # --- Cooling load calculations ---
    if aggload_file_data.has_cooling:
        complete_agg_load.has_cooling = aggload_file_data.has_cooling
        complete_agg_load.loads_cooling.yearly = source_load_from_cop(
            aggload_file_data.load_yearly_cooling,
            aggload_file_data.eer_cooling,
            heating=False
        )
        complete_agg_load.loads_cooling.seasonal = source_load_from_cop(
            aggload_file_data.load_summer_cooling,
            aggload_file_data.eer_cooling,
            heating=False
        )
        complete_agg_load.loads_cooling.peak = source_load_from_cop(
            aggload_file_data.load_daily_peak_cooling,
            aggload_file_data.eer_cooling,
            heating=False
        ) * diversity_factor

        complete_agg_load.peak_mass_flow_cooling = mass_flow_from_load(
            complete_agg_load.loads_cooling.peak,
            aggload_file_data.delta_temp_cooling,
            brine.density,
            brine.heat_capacity
        )

        complete_agg_load.outlet_temp_cooling = (
            complete_agg_load.inlet_temp_cooling
            + aggload_file_data.delta_temp_cooling
        )

        complete_agg_load.load_imbalance_yearly = (
            complete_agg_load.loads_heating.yearly
            - complete_agg_load.loads_cooling.yearly
        )

    return complete_agg_load


def aggload_combine_user_and_file_input(
        aggload_base: AggregatedLoad,
        aggload_file_data: RefactoredAggregatedLoadInput,
        brine: Brine
        ) -> AggregatedLoad:
    """
    Combines the manually specified and file-based input data to produce
    a complete AggregatedLoad instance.

    This function creates a deep copy of the base load profile and
    updates it with the provided input data for heating and, if
    applicable, cooling loads. It also calculates peak mass flow rates
    and the annual load imbalance between heating and cooling.

    Parameters
    ----------
    aggload_base : AggregatedLoad
        The base load configuration object to use as a starting point.

    aggload_file_data : RefactoredAggregatedLoadInput
        The input data specifying loads, COP/EER, and temperature
        differences for heating and cooling.

    brine : Brine
        The fluid properties (density, heat capacity, etc.) used in
        mass flow calculations.

    Returns
    -------
    complete_agg_load : AggregatedLoad
        A new 'AggregatedLoad' object containing the updated
        heating and cooling loads, peak mass flows, and annual load
        imbalance.

    Notes
    -----
    - The cooling loads are only updated if the input data indicates
      that cooling is present ('has_cooling' flag is 'True').
    - The load imbalance is calculated as the difference between the
      annual heating and cooling loads, positive if heating load
      dominates.
    """
    complete_agg_load = copy.deepcopy(aggload_base)

    S_H = (
        complete_agg_load.f_peak_H
        * (0.62 + 0.38 / aggload_file_data.num_consumers)
        )

    # Calculate ground loads from COP (heating)
    P_s_H = np.zeros(3)
    P_s_H[0] = (
        (aggload_file_data.cop_yearly_heating-1)
        / aggload_file_data.cop_yearly_heating
        * aggload_file_data.load_yearly_heating
        )     # Annual load (W)
    P_s_H[1] = (
        (aggload_file_data.cop_winter_heating - 1)
        / aggload_file_data.cop_winter_heating
        * aggload_file_data.load_winter_heating
        ) # Monthly load (W)
    P_s_H[2] = (
        (aggload_file_data.cop_hourly_peak_heating - 1)
        / aggload_file_data.cop_hourly_peak_heating
        * aggload_file_data.load_daily_peak_heating
        ) * S_H  # Daily load with simultaneity factor (W)

    # KART COOLING
    if np.abs(aggload_file_data.load_yearly_cooling) > 1e-6:
        # Calculate ground loads from EER (cooling)
        S_C = (
            complete_agg_load.f_peak_C
            * (0.62 + 0.38 / aggload_file_data.num_consumers)
            )
        P_s_C = np.zeros(3)
        P_s_C[0] = (
            (aggload_file_data.eer_cooling + 1)
            / aggload_file_data.eer_cooling
            * aggload_file_data.load_yearly_cooling
            )       # Annual load (W)
        P_s_C[1] = (
            (aggload_file_data.eer_cooling + 1)
            / aggload_file_data.eer_cooling
            * aggload_file_data.load_summer_cooling
            )       # Monthly load (W)
        P_s_C[2] = (
            (aggload_file_data.eer_cooling + 1)
            / aggload_file_data.eer_cooling
            * aggload_file_data.load_daily_peak_cooling
            ) * S_C   # Daily load (W)

        # First columns in hp.P_s_H respectively hp.P_s_C are equal but with opposite signs 
        P_s_H[0] = P_s_H[0] - P_s_C[0]         # Annual imbalance between heating and cooling, positive for heating (W)
        P_s_C[0] = - P_s_H[0]                  # Negative for cooling

    complete_agg_load.Qdim_H = (
        P_s_H[2]
        / aggload_file_data.delta_temp_heating
        / brine.rho
        / brine.c
        )
    complete_agg_load.To_H = (
        complete_agg_load.Ti_H
        - aggload_file_data.delta_temp_heating
        )
    complete_agg_load.P_s_H = P_s_H;

    #KART COOLING
    if np.abs(aggload_file_data.load_yearly_cooling) > 1e-3:
        complete_agg_load.Qdim_C = (
            P_s_C[2]
            / aggload_file_data.delta_temp_cooling
            / brine.rho
            / brine.c
            )
        complete_agg_load.To_C = (
            complete_agg_load.Ti_C
            + aggload_file_data.delta_temp_cooling
            )
        complete_agg_load.P_s_C = P_s_C

    return complete_agg_load


def aggload_combine_user_and_file_input(
        aggload_user_input: AggregatedLoad,
        aggload_file_input: AggregatedLoadInput,
        brine: Brine
        ) -> RefactoredAggregatedLoad:
    """
    Combines the manually specified and file-based input data to produce
    a complete AggregatedLoad instance.

    This function creates a deep copy of the base load profile and
    updates it with the provided input data for heating and, if
    applicable, cooling loads. It also calculates peak mass flow rates
    and the annual load imbalance between heating and cooling.

    Parameters
    ----------
    aggload_base : RefactoredAggregatedLoad
        The base load configuration object to use as a starting point.

    aggload_file_data : RefactoredAggregatedLoadInput
        The input data specifying loads, COP/EER, and temperature
        differences for heating and cooling.

    brine : RefactoredBrine
        The fluid properties (density, heat capacity, etc.) used in
        mass flow calculations.

    Returns
    -------
    complete_agg_load : RefactoredAggregatedLoad
        A new 'RefactoredAggregatedLoad' object containing the updated
        heating and cooling loads, peak mass flows, and annual load
        imbalance.

    Notes
    -----
    - The cooling loads are only updated if the input data indicates
      that cooling is present ('has_cooling' flag is 'True').
    - The load imbalance is calculated as the difference between the
      annual heating and cooling loads, positive if heating load
      dominates.
    """
    agg_load = copy.deepcopy(aggload_user_input)

    diversity_factor = (
        agg_load.f_peak_H * (0.62 + 0.38/aggload_file_input.n_hp)
        )
    # --- Heating load calculations ---
    agg_load.loads_heating.yearly = source_load_from_cop(
        aggload_file_input.load_yearly_heating,
        aggload_file_input.cop_yearly_heating,
        heating=True
    )
    agg_load.loads_heating.seasonal = source_load_from_cop(
        aggload_file_input.load_winter_heating,
        aggload_file_input.cop_winter_heating,
        heating=True
    )
    agg_load.loads_heating.peak = source_load_from_cop(
        aggload_file_input.load_daily_peak_heating,
        aggload_file_input.cop_hourly_peak_heating,
        heating=True
    ) * diversity_factor

    agg_load.P_s_C[0] = source_load_from_cop(
        aggload_file_input, "yearly", heating=True
        )

    def source_load(
            model_file_input: AggregatedLoadInput | HeatPumpLoadInput,
            heating: bool = True,
            time_scale: Literal["yearly", "seasonal", "peak"] = "yearly", 
            ) -> float:
        """
        Calculates the thermal load on the source side for the specified
        time scale (yearly, seasonal, or peak) based on the input model
        data and Coefficient Of Preformance (COP) or Energy Efficiency
        Ratio (EER) value.

        Parameters
        ----------
        model_file_input : AggregatedLoadInput or HeatPumpLoadInput
            The input object containing the data loaded from file/server
            input.

        heating : bool, default=True
            If True, calculates the heating-side load.
            If False, calculates the cooling-side load.

        time_scale : {'yearly', 'seasonal', 'peak'}, default='yearly'
            The time scale for the load calculation:
            - 'yearly' : Use the yearly average load.
            - 'seasonal' : Use the seasonal (winter or summer) average
                load.
            - 'peak' : Use the peak load.

        Returns
        -------
        source_load : float or list of floats 
            The calculated thermal load on the source side for the
            selected time scale [W].

        Notes
        -----
        - This function calls `source_load_from_cop` internally, using the
        appropriate load and COP/EER values from the input model data.
        - If `heating` is True, it calculates the load as heating-side.
        If False, it calculates the load as cooling-side.
        """
        if time_scale =="yearly":
            return source_load_from_cop(
                model_file_input.P_y_H, model_file_input.COP_y_H, 
                heating=heating
                )
        elif time_scale == "seasonal":
            return source_load_from_cop(
                model_file_input.P_s_H, model_file_input.COP_y_H,
                heating=True
                )
        elif time_scale == "peak":
            return source_load_from_cop(
                model_file_input.P_d_H, model_file_input.COP_y_H,
                heating=True
                )

    # complete_agg_load.P_s_C[0] = source_load_from_cop(
    #         aggload_file_data.P_y_H, aggload_file_data.COP_y_H, heating=True
    #         )
    #     complete_agg_load.P_s_C[0] = source_load_from_cop(
    #         aggload_file_data.P_s_H, aggload_file_data.COP_y_H, heating=True
    #         )
    #     complete_agg_load.P_s_C[0] = source_load_from_cop(
    #         aggload_file_data.P_d_H, aggloa
    # d_file_data.COP_y_H, heating=True
    #         )

    complete_agg_load.peak_mass_flow_heating = mass_flow_from_load(
        complete_agg_load.loads_heating.peak,
        aggload_file_data.delta_temp_heating,
        brine.density,
        brine.heat_capacity
    )

    complete_agg_load.outlet_temp_heating = (
        complete_agg_load.inlet_temp_heating
        - aggload_file_data.delta_temp_heating
    )

    # --- Cooling load calculations ---
    if aggload_file_data.has_cooling:
        complete_agg_load.has_cooling = aggload_file_data.has_cooling
        complete_agg_load.loads_cooling.yearly = source_load_from_cop(
            aggload_file_data.load_yearly_cooling,
            aggload_file_data.eer_cooling,
            heating=False
        )
        complete_agg_load.loads_cooling.seasonal = source_load_from_cop(
            aggload_file_data.load_summer_cooling,
            aggload_file_data.eer_cooling,
            heating=False
        )
        complete_agg_load.loads_cooling.peak = source_load_from_cop(
            aggload_file_data.load_daily_peak_cooling,
            aggload_file_data.eer_cooling,
            heating=False
        ) * diversity_factor

        complete_agg_load.peak_mass_flow_cooling = mass_flow_from_load(
            complete_agg_load.loads_cooling.peak,
            aggload_file_data.delta_temp_cooling,
            brine.density,
            brine.heat_capacity
        )

        complete_agg_load.outlet_temp_cooling = (
            complete_agg_load.inlet_temp_cooling
            + aggload_file_data.delta_temp_cooling
        )

        complete_agg_load.load_imbalance_yearly = (
            complete_agg_load.loads_heating.yearly
            - complete_agg_load.loads_cooling.yearly
        )

    return complete_agg_load
    agg_load.P_s_C[0] = source_load_from_cop(aggload_file, "yearly", heating=True)

    def (model, time_scale: ["yearly","seasonal","peak"], heating=True):
        match time_scale:
            case "yearly":
                return source_load_from_cop(
                    aggload_file_data.P_y_H, aggload_file_data.COP_y_H, heating=True
                    )
            case "seasonal":
                return source_load_from_cop(
                    aggload_file_data.P_s_H, aggload_file_data.COP_y_H, heating=True
                    )
            case "peak":
                return source_load_from_cop(
                    aggload_file_data.P_d_H, aggload_file_data.COP_y_H, heating=True
                    )
        





@dataclass_json
@dataclass
class RefactoredLoadProfile:
    """
    Container for the thermal load profile with a yearly average, a
    seasonal average, and a peak load value.

    The seasonal value refers to the winter for heating case and to the
    summer for cooling case.
    """
    yearly: float = 0.0
    seasonal: float = 0.0
    peak: float = 0.0

    def to_array(self) -> np.ndarray:
        """
        Return the load profile as a NumPy array in the order:
        [yearly, seasonal, peak].
        """
        return np.array([self.yearly, self.seasonal, self.peak])


@dataclass_json
@dataclass
class RefactoredAggregatedLoad:
    """
    Contains the key load parameters for a scenario focused solely on
    source dimensioning, where heating and cooling demands can be
    treated in a combined and simplified manner.
    """
    # Inlet/Outlet temperatures
    inlet_temp_heating: float = np.nan      # [°C]
    outlet_temp_heating: float = np.nan     # [°C]
    inlet_temp_cooling: float = np.nan      # [°C]
    outlet_temp_cooling: float = np.nan     # [°C]

    # Load coverage configuration
    fraction_peak_load_covered_heating: float = np.nan
    fraction_peak_load_covered_cooling: float = np.nan

    # Duration of peak periods
    peak_duration_heating: float = np.nan   # [h]
    peak_duration_cooling: float = np.nan   # [h]

    # Design parameters
    peak_mass_flow_heating: float = np.nan  # [kg/s]
    peak_mass_flow_cooling: float = np.nan  # [kg/s]

    # Seasonal loads
    loads_heating: RefactoredLoadProfile = field(default_factory=RefactoredLoadProfile)
    loads_cooling: RefactoredLoadProfile = field(default_factory=RefactoredLoadProfile)

    load_imbalance_yearly: float = 0.0  # Annual imbalance between heating and cooling [W]

    has_cooling: bool = False
