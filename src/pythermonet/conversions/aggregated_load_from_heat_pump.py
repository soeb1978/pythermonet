from pythermonet.domain import Brine, HeatPump, AggregatedLoad

# TODO write function docstring and polish the function


def aggregated_load_from_heatpump(
        heat_pump: HeatPump, brine: Brine
        ) -> AggregatedLoad:
    """
    write something
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

    consumers_count = len(heat_pump.P_s_H)
    S_H = heat_pump.f_peak_H*(0.62 + 0.38/consumers_count)

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
        - sum(peak_vol_flow_heating*heat_pump.dT_H)/sum(peak_vol_flow_heating)
    )

    agg_load.P_s_H = sum(heat_pump.P_s_H)
    # reduced the total peak load and peak flow by the diversity factor
    agg_load.P_s_H[2] = agg_load.P_s_H[2] * S_H
    agg_load.Qdim_H = sum(peak_vol_flow_heating) * S_H

    if agg_load.has_cooling:
        S_C = heat_pump.f_peak_C*(0.62 + 0.38/consumers_count)

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
            + sum(peak_vol_flow_cooling*heat_pump.dT_C)
            / sum(peak_vol_flow_cooling)
        )

        agg_load.P_s_C = sum(heat_pump.P_s_C)
        # reduced the total peak load and peak flow by the diversity factor
        agg_load.P_s_C[2] = agg_load.P_s_C[2] * S_C
        agg_load.Qdim_C = sum(peak_vol_flow_cooling) * S_C

    return agg_load
