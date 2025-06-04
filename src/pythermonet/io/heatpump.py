import copy

import pandas as pd

from pythermonet.domain import HeatPump, HeatPumpInput
from pythermonet.core.physics import source_loads_all_timescales


def read_heat_pump_tsv(path: str) -> HeatPumpInput:
    """
    Loads heat pump input data from a TSV (tab-separated values) file.

    This function reads a TSV file containing heating and optional
    cooling loads data, along with Coefficient Of Performances (COP) and
    Energy Efficiency Ratios (EER) and temperature differences.
    It returns the unified 'HeatPumpInput' dataclass populated
    with these values.

    Parameters
    ----------
    path : str
        Path to the TSV file containing the aggregated load input data.

    Returns
    -------
    HeatPumpInput
        A dataclass instance containing the loaded heating and cooling
        loads, performance metrics, and temperature differences.

    Notes
    -----
    - The TSV file must contain at least the following columns:
        'Heat_pump_ID',
        'Yearly_heating_load_(W)',
        'Winter_heating_load_(W)',
        'Daily_heating_load_(W)',
        'Year_COP',
        'Winter_COP',
        'Hour_COP',
        'dT_HP_Heating',
        'Yearly_cooling_load_(W)',
        'Summer_cooling_load_(W)',
        'Daily_cooling_load_(W)',
        'EER',
        'dT_HP_Cooling'

    - The cooling parameters should be set to 0 if they are not
        applicable

    ###@@@ We should check if all the cooling parameters are missing, it
    should be possible run the code without them instead of setting them
    to zero and the same with the heating parameters, we should just
    requiere one or the other and not both.
    """
    df = pd.read_csv(path, sep="\t+", engine="python")
    df.columns = [col.strip() for col in df.columns]

    return HeatPumpInput(
        heat_pump_id=df["Heat_pump_ID"].to_numpy(),

        load_yearly_heating=df["Yearly_heating_load_(W)"].to_numpy(),
        load_winter_heating=df["Winter_heating_load_(W)"].to_numpy(),
        load_daily_peak_heating=df["Daily_heating_load_(W)"].to_numpy(),

        cop_yearly_heating=df["Year_COP"].to_numpy(),
        cop_winter_heating=df["Winter_COP"].to_numpy(),
        cop_hourly_peak_heating=df["Hour_COP"].to_numpy(),
        delta_temp_heating=df["dT_HP_Heating"].to_numpy(),

        load_yearly_cooling=df["Yearly_cooling_load_(W)"].to_numpy(),
        load_summer_cooling=df["Summer_cooling_load_(W)"].to_numpy(),
        load_daily_peak_cooling=df["Daily_cooling_load_(W)"].to_numpy(),

        eer_cooling=df["EER"].to_numpy(),
        delta_temp_cooling=df["dT_HP_Cooling"].to_numpy(),
    )


def combine_heatpump_user_and_file_input(
        heatpump_user: HeatPump,
        heatpump_file: HeatPumpInput,
        ) -> HeatPump:
    """
    Combine user-defined and file-based input data into a complete
    HeatPump instance.

    This function creates a deep copy of the user-defined 'HeatPump'
    object and fills in computed thermal loads based on the provided
    'HeatPumpInput' data. It calculates thermal loads for all three
    timescales (yearly, seasonal, and peak) for both heating and,
    optionally, cooling. If cooling data is provided, it also computes
    the annual load imbalance between heating and cooling.

    Parameters
    ----------
    heatpump_user : HeatPump
        A base 'HeatPump' object containing user-defined parameters.

    heatpump_file : HeatPumpInput
        Input data containing heating and cooling loads, COP/EER values,
        and temperature differences, typically loaded from a TSV file.

    Returns
    -------
    HeatPump
        A new 'HeatPump' instance containing computed heating and (if
        present) cooling source loads for all timescales, with imbalance
        adjustment applied.

    Notes
    -----
    - If cooling is included ('has_cooling=True'), the function computes
      both 'P_s_H' and 'P_s_C' and ensures their yearly components
      reflect the net thermal imbalance.
    - The returned object is a deep copy of 'heatpump_user'; the input
      object is never mutated.
    """
    heatpump = copy.deepcopy(heatpump_user)

    # --- Heating load calculations ---
    P_s_H = source_loads_all_timescales(heatpump_file, heating=True)
    heatpump.P_s_H = P_s_H

    heatpump.has_cooling = heatpump_file.has_cooling
    if heatpump.has_cooling is True:
        P_s_C = source_loads_all_timescales(heatpump_file, heating=False)
        heatpump.P_s_C = P_s_C

        # First columns in hp.P_s_H and hp.P_s_C are equal but with
        # opposite signs
        # Annual imbalance between heating and cooling, positive for
        # heating [W]
        heatpump.P_s_H[:, 0] = heatpump.P_s_H[:, 0] - heatpump.P_s_C[:, 0]
        heatpump.P_s_C[:, 0] = - heatpump.P_s_H[:, 0]

    return heatpump
