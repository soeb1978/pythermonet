import copy

import pandas as pd
import numpy as np

from pythermonet.domain import Brine, DimensionedTopologyInput, Thermonet
from pythermonet.core.fThermonetDim import Re  # ###@@@ I should change this
from pythermonet.core.physics import (
    flow_velocity_from_volumetric_flow,
    pipe_brine_volume,
    pipe_inner_diameter
)


def read_dimensioned_topology_tsv(path: str) -> DimensionedTopologyInput:
    """
    Load dimensioned topology input data from a TSV (tab-separated values)
    file.

    This function reads a TSV file containing pipe dimensions, number
    of traces, and peak flows for heating and cooling. It returns the
    unified 'DimensionedTopologyInput' dataclass populated with these
    values.

    Parameters
    ----------
    path : str
        Path to the TSV file containing the dimensioned topology data.

    Returns
    -------
    DimensionedTopologyInput
        A dataclass instance containing the loaded topology data.
    """
    df = pd.read_csv(path, sep='\t+', engine='python')
    df.columns = [col.strip() for col in df.columns]

    return DimensionedTopologyInput(
        standard_dimension_ratio=df["SDR"].to_numpy(),
        outer_diameter=df["do_(mm)"].to_numpy() / 1000,
        trace_lengths=df["Trace_(m)"].to_numpy(),
        number_of_traces=df["Number_of_traces"].to_numpy(),
        peak_volumentric_flow_heating=df["Peak_flow_heating_m3/s"].to_numpy(),
        peak_volumentric_flow_cooling=df["Peak_flow_cooling_m3/s"].to_numpy(),
        pipe_group_names=df["Section"].to_list(),
    )


def combine_net_dimensioned_topology(
        net_user: Thermonet, topology: DimensionedTopologyInput,
        brine: Brine
        ) -> Thermonet:
    """
    Combines an existing Thermonet object with the loaded dimensioned
    topology input.

    This includes calculating Reynolds numbers for heating and cooling
    circuits and computing the total brine volume in the pipes.

    Note that the functions returns a new copy of the thermonet

    Parameters
    ----------
    net_user : Thermonet
        The base Thermonet object to which the input parameter are added

    topology : DimensionedTopologyInput
        The dimensioned topology as loaded from the data.

    brine : Brine
        Brine properties (density, viscosity, etc.).

    Returns
    -------
    Thermonet
        A copy of thermonet object now with the topology data.
    """
    net = copy.deepcopy(net_user)
    topology = compute_missing_reynolds_numbers(topology, brine)
    net.d_selectedPipes_H = topology.outer_diameter
    net.d_selectedPipes_C = net.d_selectedPipes_H
    net.di_selected_H = topology.inner_diameter
    net.di_selected_C = net.di_selected_H
    net.SDR = topology.standard_dimension_ratio
    net.L_traces = topology.trace_lengths
    net.N_traces = topology.number_of_traces

    net.Re_selected_H = topology.peak_reynold_heating
    net.Re_selected_C = topology.peak_reynold_cooling

    net.L_segments = 2 * net.L_traces * net.N_traces

    # Calculate total brine volume in the grid pipes
    net.V_brine = sum(pipe_brine_volume(net.L_segments, net.di_selected_H))

    return net, topology.pipe_group_names


def compute_missing_reynolds_numbers(
        topology: DimensionedTopologyInput, brine: Brine
        ) -> DimensionedTopologyInput:
    """
    Returns a copy of the DimensionedTopologyInput with missing Reynolds
    numbers calculated from volumetric flow and brine parameters.
    """
    topology = copy.deepcopy(topology)  # keep function pure

    if topology.has_heating and not topology._array_has_nonzero_values(
        topology.peak_reynold_heating
            ):
        v_H = flow_velocity_from_volumetric_flow(
            topology.peak_volumentric_flow_heating, topology.inner_diameter
        )
        topology.peak_reynold_heating = Re(
            brine.rho, brine.mu, v_H, topology.inner_diameter
        )

    if topology.has_cooling and not topology._array_has_nonzero_values(
        topology.peak_reynold_cooling
            ):
        v_C = flow_velocity_from_volumetric_flow(
            topology.peak_volumentric_flow_cooling, topology.inner_diameter
        )
        topology.peak_reynold_cooling = Re(
            brine.rho, brine.mu, v_C, topology.inner_diameter
        )

    return topology


def read_undimensioned_topology_tsv_to_net(
        path: str, net_user: Thermonet
        ) -> Thermonet:
    net = copy.deepcopy(net_user)

    df = pd.read_csv(path, sep='\t+', engine='python')
    df.columns = [col.strip() for col in df.columns]

    net.SDR = df["SDR"].astype(float).to_numpy()
    net.L_traces = df["Trace_(m)"].astype(float).to_numpy()
    net.N_traces = df["Number_of_traces"].to_numpy()
    net.dp_PG = df["Max_pressure_loss_(Pa)"].astype(float).to_numpy()      # Total allowed pressure drop over the forward + retun pipe in a trace

    # Calculate total length of segments
    net.L_segments = 2 * net.L_traces * net.N_traces

    # Create an array of the IDs of HPs connected to the different pipe groups
    net.I_PG = [
        np.array(ids.split(","), dtype=int,)for ids in df["HP_ID_vector"]
    ]

    # Extract pipe group IDs
    pipe_group_names = df["Section"]

    return net, pipe_group_names
