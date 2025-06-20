import copy

import pandas as pd

from pythermonet.core.thermal_loads import source_loads_all_timescales
from pythermonet.core.pipe_hydraulics import mass_flow_from_load

from pythermonet.domain import (
    AggregatedLoad,
    AggregatedLoadInput,
    Brine
)


def read_aggregated_load_tsv(path: str) -> AggregatedLoadInput:
    """
    Load aggregated load input data from a TSV (tab-separated values)
    file.

    This function reads a TSV file containing heating and optional
    cooling load data, along with Coefficient Of Performance (COP) and
    Energy Efficiency Ratio (EER) and temperature differences.
    It returns the unified 'AggregatedLoadInput' dataclass populated
    with these values.

    If the number of consumers for heating or cooling is missing in the
    file, it defaults to the general number of consumers
    ('No. consumers').

    Parameters
    ----------
    path : str
        Path to the TSV file containing the aggregated load input data.

    Returns
    -------
    AggregatedLoadInput
        A dataclass instance containing the loaded heating and cooling
        loads, performance metrics, temperature differences, and the
        number of consumers.

    Notes
    -----
    - The TSV file must contain at least the following columns:
        'No. consumers', 'Yearly_heating_load_(W)',
        'Winter_heating_load_(W)', 'Daily_heating_load_(W)', 'Year_COP',
        'Winter_COP', 'Hour_COP', 'dT_HP_Heating',
        'Yearly_cooling_load_(W)', 'Summer_cooling_load_(W)',
        'Daily_cooling_load_(W)', 'EER', 'dT_HP_Cooling'.

    - Optional columns:
        - 'No. consumers heating' (defaults to 'No. consumers' if missing)
        - 'No. consumers cooling' (defaults to 'No. consumers' if missing)

    - The cooling parameters should be set to 0 if they are not
        applicable

    ###@@@ We should check if all the cooling parameters are missing, it
    should be possible run the code without them instead of setting them
    to zero and the same with the heating parameters, we should just
    requiere one or the other and not both.
    """
    df = pd.read_csv(path, sep='\t+', engine='python')
    df.columns = [col.strip() for col in df.columns]
    row = df.iloc[0]

    n_consumers_heating = int(
        row.get("No. consumers heating", row["No. consumers"])
    )

    n_consumers_cooling = int(
        row.get("No. consumers cooling", row["No. consumers"])
    )

    return AggregatedLoadInput(
        n_consumers_heating=n_consumers_heating,

        load_yearly_heating=row["Yearly_heating_load_(W)"],
        load_winter_heating=row["Winter_heating_load_(W)"],
        load_daily_peak_heating=row["Daily_heating_load_(W)"],

        cop_yearly_heating=row["Year_COP"],
        cop_winter_heating=row["Winter_COP"],
        cop_hourly_peak_heating=row["Hour_COP"],
        delta_temp_heating=row["dT_HP_Heating"],

        n_consumers_cooling=n_consumers_cooling,
        load_yearly_cooling=row["Yearly_cooling_load_(W)"],
        load_summer_cooling=row["Summer_cooling_load_(W)"],
        load_daily_peak_cooling=row["Daily_cooling_load_(W)"],

        eer_cooling=row["EER"],
        delta_temp_cooling=row["dT_HP_Cooling"]
    )


def combine_agg_load_user_and_file(
        aggload_user: AggregatedLoad,
        aggload_file: AggregatedLoadInput,
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
    agg_load = copy.deepcopy(aggload_user)

    S_H = (agg_load.f_peak_H * (0.62 + 0.38/aggload_file.n_consumers_heating))
    # --- Heating load calculations ---
    P_s_H = source_loads_all_timescales(aggload_file, heating=True)
    P_s_H[2] *= S_H

    agg_load.Qdim_H = mass_flow_from_load(
        P_s_H[2], aggload_file.delta_temp_heating, brine.rho, brine.c
    )
    agg_load.To_H = agg_load.Ti_H - aggload_file.delta_temp_heating
    agg_load.P_s_H = P_s_H

    agg_load.has_cooling = aggload_file.has_cooling

    if agg_load.has_cooling:
        S_C = (
            agg_load.f_peak_C
            * (0.62 + 0.38/aggload_file.n_consumers_cooling)
        )
        P_s_C = source_loads_all_timescales(aggload_file, heating=False)
        P_s_C[2] *= S_C

        agg_load.Qdim_C = mass_flow_from_load(
            P_s_C[2], aggload_file.delta_temp_cooling, brine.rho, brine.c
        )
        agg_load.To_C = agg_load.Ti_C + aggload_file.delta_temp_cooling
        agg_load.P_s_C = P_s_C

        # First columns in hp.P_s_H and hp.P_s_C are equal but with
        # opposite signs
        # Annual imbalance between heating and cooling, positive for
        # heating [W]
        agg_load.P_s_H[0] = agg_load.P_s_H[0] - agg_load.P_s_C[0]
        agg_load.P_s_C[0] = - agg_load.P_s_H[0]

    return agg_load
