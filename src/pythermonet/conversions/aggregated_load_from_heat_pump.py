import numpy as np

from pythermonet.domain import Brine, HeatPump, AggregatedLoad
from pythermonet.domain.utils import count_active_consumers
# TODO write function docstring and polish the function


def aggregated_load_from_heatpump(
        heat_pump: HeatPump, brine: Brine
        ) -> AggregatedLoad:
    """
    Converts a 'HeatPump' object into an 'AggregatedLoad' object using
    brine properties.

    This function aggregates heating and (optionally) cooling load
    parameters from a group of heat pumps. It calculates peak volumetric
    flow rates, flow-weighted outlet temperatures, and applies a
    diversity factor based on the number of active consumers.

    Parameters
    ----------
    heat_pump : HeatPump
        The heat pump data containing peak loads, design temperatures,
        and configuration for more than one unit.

    brine : Brine
        The brine fluid properties, including density and specific heat
        capacity, used to compute volumetric flow rates.

    Returns
    -------
    AggregatedLoad
        The aggregated heating and cooling loads, including flow rates,
        outlet temperatures, and peak volumetric flows.

    Notes
    -----
    - A diversity factor is applied to the total peak load and flow
      using the formula: S = f_peak * (0.62 + 0.38 / n_active), where
      'n_active' is the number of heat pumps with non-zero demand.
    - The cooling can handle zero consumers, heating cannot.
    """
    agg_load = AggregatedLoad(
        Ti_H=heat_pump.Ti_H,
        Ti_C=heat_pump.Ti_C,
        f_peak_H=heat_pump.f_peak_H,
        t_peak_H=heat_pump.t_peak_H,
        f_peak_C=heat_pump.f_peak_C,
        t_peak_C=heat_pump.t_peak_C,
        has_cooling=heat_pump.has_cooling
    )

    consumers_count_heating = count_active_consumers(
        heat_pump.P_s_H[:, 0]
        )
    S_H = heat_pump.f_peak_H * (0.62 + 0.38/consumers_count_heating)

    # calculate peak volumetric flow rates per heat pump
    peak_vol_flow_heating = (
        heat_pump.P_s_H[:, 2]
        / heat_pump.dT_H
        / brine.rho
        / brine.c
    )

    # flow-weighted average outlet temperature
    agg_load.To_H = (
        agg_load.Ti_H
        - np.sum(peak_vol_flow_heating*heat_pump.dT_H)
        / np.sum(peak_vol_flow_heating)
    )

    agg_load.P_s_H = np.sum(heat_pump.P_s_H, axis=0)
    # reduced the total peak load and peak flow by the diversity factor
    agg_load.P_s_H[2] *= S_H
    agg_load.Qdim_H = np.sum(peak_vol_flow_heating) * S_H

    if agg_load.has_cooling:
        consumers_count_cooling = count_active_consumers(
            heat_pump.P_s_C[:, 0]
        )
        S_C = heat_pump.f_peak_C * (0.62 + 0.38/consumers_count_cooling)

        # calculate peak volumetric flow rates per heat pump
        peak_vol_flow_cooling = (
            heat_pump.P_s_C[:, 2]
            / heat_pump.dT_C
            / brine.rho
            / brine.c
        )

        # flow-weighted average outlet temperature
        agg_load.To_C = (
            heat_pump.Ti_C
            + np.sum(peak_vol_flow_cooling*heat_pump.dT_C)
            / np.sum(peak_vol_flow_cooling)
        )

        agg_load.P_s_C = np.sum(heat_pump.P_s_C, axis=0)
        # reduced the total peak load and peak flow by the diversity factor
        agg_load.P_s_C[2] *= S_C
        agg_load.Qdim_C = np.sum(peak_vol_flow_cooling) * S_C

    return agg_load
