# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 08:53:07 2022

@author: KRI
"""
##########################
# To consider or implement
##########################
# 1: Samtidighedsfaktor anvendes også på køling. Er det en god ide?
# 2: Der designes to rørsystemer - et for varme og et for køling. Bør ændres?

# Conceptual model drawings are found below the code

import json

import numpy as np
import pandas as pd

from thermonet.dimensioning.dimensioning_functions import \
    print_project_id, print_source_dimensions, run_sourcedimensioning
from thermonet.dimensioning.pipe_flow_functions import pipe_inner_diameter
from thermonet.dimensioning.thermonet_classes import Brine, Thermonet, \
    aggregatedLoad, BHEconfig


# Read topology for thermonet with already dimesnioned pipe system
def read_dimensioned_topology(net, pipe_file_with_flow):
    """
    WRITE SOMETHING
    """
    # Load grid topology
    pipe_data = pd.read_csv(pipe_file_with_flow)
    # when loading pipe outer diameters convert from mm to m
    net.d_selectedPipes_H = pipe_data["outer_diameter(mm)"] / 1000
    net.d_selectedPipes_C = net.d_selectedPipes_H
    net.SDR = pipe_data['SDR']
    net.L_traces = pipe_data['length(m)']
    net.N_traces = np.ones_like(pipe_data['SDR'])  # $might be redundant check
    net.L_segments = 2 * net.L_traces

    # Calculate Reynolds number for selected pipes for heating
    net.di_selected_H = pipe_inner_diameter(net.d_selectedPipes_H, net.SDR)
    net.Re_selected_H = pipe_data["Reynolds number heating"]

    # Calculate Reynolds numbers for selected pipes for cooling
    net.di_selected_C = net.di_selected_H
    try:
        net.Re_selected_C = pipe_data["Reynolds number free_cooling"]
    except KeyError:
        print("No free cooling in pandapipes output")
        net.Re_selected_C = np.zeros_like(
            pipe_data["Reynolds number free_cooling"]
            )

    # Calculate total brine volume in the grid pipes
    net.V_brine = sum(net.L_segments*np.pi*net.di_selected_H**2/4)

    return net


def aggload_from_heat_pump_data_file(heat_pump_load_file,
                                     heat_pump_settings_file, brine):
    """
    Reads the heat pump loads and estimates the thermal load on the
    network

    Args:
    :param heat_pump_load_file: The file containing the heat pump data
    :type  heat_pump_load_file: str (path)
    :param heat_pump_settings_file: The file containing the temperature
        limits for the heat pumps, the fraction of the load covered
        by the ground source system, and the peak duration.
    :type  heat_pump_settings_file: str (path)
    :param brine: the brine used in the current network
    :type  brine: pythermonet brine object

    Returns:
    :param aggLoad: The aggregated load object containing the total load
        from the heat pumps on the ground source system
    :type  aggLoad: pythermonet aggregatedLoad object
    """
    # read the heat pump setting and initialize the aggload object
    with open(heat_pump_settings_file, 'r') as file:
        heat_pump_settings = json.load(file)

    # A translation dictionary from the file parameters to the aggload
    # object
    translation_dict_aggload = {
        'Ti_H': 'Minimum heat pump inlet temperature (C)',
        'Ti_C': 'Maximum heat pump inlet temperature (C)',
        'f_peak': 'Fraction of heating covered',
        't_peak': 'Peak load duration (h)'
        }

    pthn_agg_load = {}
    # do the translation
    for param in translation_dict_aggload:
        pthn_agg_load[param] = heat_pump_settings[
            translation_dict_aggload[param]]
    aggLoad = aggregatedLoad.from_dict(pthn_agg_load)

    heat_pump_data = pd.read_csv(heat_pump_load_file)

    # Extract the minimum heat temperature lift/drop at the heat pumps
    dT_H = np.min(heat_pump_data["dT_heat_pump_heating"])
    dT_C = np.min(heat_pump_data["dT_heat_pump_cooling"])

    # use the number of heat pumps to estimate the coincidence factor
    n_heat_pump = len(heat_pump_data)
    coincidence_factor = aggLoad.f_peak*(0.62 + 0.38/n_heat_pump)

    # Calculate ground loads from COP (heating)
    network_thermal_loads_heating = np.zeros(3)
    network_thermal_loads_heating[0] = np.sum(heat_pump_network_thermal_load(
        heat_pump_data["yearly_heating_load(W)"],
        heat_pump_data["year_COP"],
        heating=True
        ))
    network_thermal_loads_heating[1] = np.sum(heat_pump_network_thermal_load(
        heat_pump_data["winter_heating_load_(W)"],
        heat_pump_data["winter_COP"],
        heating=True
        ))
    network_thermal_loads_heating[2] = np.sum(heat_pump_network_thermal_load(
        heat_pump_data["peak_load(W)"],
        heat_pump_data["hour_COP"],
        heating=True
        )) * coincidence_factor

    # KART COOLING
    if np.abs(np.sum(heat_pump_data["yearly_cooling_load_(W)"])) > 1e-6:
        network_thermal_loads_cooling = np.zeros(3)
        # Calculate ground loads from EER (cooling)
        network_thermal_loads_cooling[0] = np.sum(
            heat_pump_network_thermal_load(
                heat_pump_data["yearly_cooling_load_(W)"],
                heat_pump_data["EER"],
                heating=False
            ))
        network_thermal_loads_cooling[1] = np.sum(
            heat_pump_network_thermal_load(
                heat_pump_data["summer_cooling_load_(W)"],
                heat_pump_data["EER"],
                heating=False
            ))
        network_thermal_loads_cooling[2] = np.sum(
            heat_pump_network_thermal_load(
                heat_pump_data["daily_cooling_load_(W)"],
                heat_pump_data["EER"],
                heating=False
            )) * coincidence_factor
        # Calculate the yearly imbalance respectively, the first column
        # in the two load vectors are equal but with oposite signs
        network_thermal_loads_heating[0] = network_thermal_loads_heating[0] \
            - network_thermal_loads_cooling[0]  # Annual imbalance
        network_thermal_loads_cooling[0] = - network_thermal_loads_heating[0]

    aggLoad.Qdim_H = network_thermal_loads_heating[2] / dT_H / brine.rho \
        / brine.c
    aggLoad.To_H = aggLoad.Ti_H - dT_H
    aggLoad.P_s_H = network_thermal_loads_heating

    # KART COOLING
    if np.abs(np.sum(heat_pump_data["yearly_cooling_load_(W)"])) > 1e-6:
        aggLoad.Qdim_C = network_thermal_loads_cooling[2] / dT_C / brine.rho \
            / brine.c
        aggLoad.To_C = aggLoad.Ti_C + dT_C
        aggLoad.P_s_C = network_thermal_loads_cooling

    return aggLoad


def heat_pump_network_thermal_load(heat_pump_load, cop=3, heating=True):
    """
    Calculates the themral load on the network given the heat pump load
    and Coefficient Of Performance(COP)

    Args
    :param heat_pump_load: The total load on the heat pump
    :type  heat_pump_load: float or list/1D array
    :param cop: The Coefficient Of Performance of the heat pump
    :type  cop: float or list/1D array
    :param heating: Toggle to indicate if the load is heating or cooling
    :type  heating: bool

    Returns
    :param thermal_load: The part of the load the thermal network has to
        provide
    :type  thermal_load: float or list/1D array
    """
    if heating is True:
        thermal_load = np.multiply(heat_pump_load, 1 - np.divide(1, cop))
    else:
        thermal_load = np.multiply(heat_pump_load, 1 + np.divide(1, cop))
    return thermal_load


def initialize_brine_thermonet_BHE_from_files(brine_settings_file,
                                              thermonet_file, BHE_file):
    """
    $$$
    This is written really repetitively, could be done in a more well
    order manner, though this way it is easy to read.
    $$$
    Loads the settings into the pythermonet classes, brine, thermonet,
    and BHEconfig

    Args:
    :param brine_settings_file:
    :type  brine_settings_file: str (path)
    :param thermonet_file:
    :type  thermonet_file: str (path)
    :param BHE_file:
    :type  BHE_file: str (path)

    Return:
    :param brine: Brine object initialized with parameters from the
        brine_settings_file
    :type  brine: pythermonet Brine object
    :param net: Thermonet object initialized with parameters from the
        thermonet_file
    :type  net: pythermonet Thermonet object
    :param source_config: Thermonet object initialized with parameters
        from the BHE_file
    :type  source_config: pythermonet BHEconfig object
    """

    # first setup the brine object
    with open(brine_settings_file, 'r') as file:
        brine_data = json.load(file)
    # translation from the file names to the pyhtermonet names
    translation_dict_brine = {
        'rho': 'fluid_density',
        'c': 'fluid_heat_capacity',
        'mu': 'fluid_viscosity',
        'l': 'fluid_thermal_conductivety'
    }
    pthn_brine = {}
    for param in translation_dict_brine:
        pthn_brine[param] = brine_data[translation_dict_brine[param]]

    brine = Brine.from_dict(pthn_brine)

    # Then setup the thermonet object
    with open(thermonet_file, 'r') as file:
        thermonet_data = json.load(file)
    # translation from the file names to the pyhtermonet names
    translation_dict_thermonet = {
        'D_gridpipes': 'pipe trace separation (m)',
        'l_p': 'pipe thermal conductivity (W/m/K)',
        'l_s_H': 'top soil thermal conductivity in heating mode (W/m/K)',
        'l_s_C': 'top soil thermal conductivity in cooling mode (W/m/K)',
        'rhoc_s': 'top soil volumetric heat capacity (J/m3/K)',
        'z_grid': 'pipe burial depth (m)',
        'T0': 'annual mean surface temperature (C)',
        'A': 'annual temperature amplitude (C)',

    }
    pthn_thermonet = {}
    for param in translation_dict_thermonet:
        pthn_thermonet[param] = thermonet_data[
            translation_dict_thermonet[param]]
    net = Thermonet.from_dict(pthn_thermonet)

    # and lastly the BHEconfig object
    with open(BHE_file, 'r') as file:
        BHE_data = json.load(file)
    # translation from the file names to the pyhtermonet names
    translation_dict_BHE = {
        "q_geo": "Geothermal heat flux (W/m2)",
        "r_b": "Borehole radius (m)",
        "r_p": "Outer U-pipe radius (m)",
        "SDR": "SDR of U-pipe",
        "l_ss": "Soil thermal conductivity along the BHE (W/m/K)",
        "rhoc_ss": "Soil volumetric hear capacity along the BHE ",
        "l_g": "Grout thermal conductivity (W/m/K)",
        "rhoc_g": "Grout volumentric heat capacity (J/m3/K)",
        "D_pipes": "Wall to wall distance between U-pipe legs (m)",
        "NX": "Number of boreholes in the x direction",
        "D_x": "Borehole spacing in x direction (m)",
        "NY": "Number of boreholes in the y direction",
        "D_y": "Borehole spacing in y direction (m)"
    }
    pthn_BHE = {}
    for param in translation_dict_BHE:
        pthn_BHE[param] = BHE_data[translation_dict_BHE[param]]
    source_config = BHEconfig.from_dict(pthn_BHE)

    return brine, net, source_config


def wrapper_pythermonet_source_dimensioning_from_csv(
        pipe_file, settings_file, thermonet_file, BHE_file,
        heat_pump_load_file, heat_pump_setting_file, project_ID):
    """
    $$$
    settings_file should probably be divided into several other files,
    I think a new overall structure is needed.
    $$$
    A wrapper for running pythermonet source dimensioning based on the
    flow calculations made by pandapipes

    , settings_file, thermonet_file, BHE_file,
        heat_pump_load_file, heat_pump_setting_file
    Args:
    :param pipe_file: The path to the csv file containing the topology
        and flow calculations from pandapipes
    :Type  pipe_file: str (path)
    :param settings_file: The path to the json file containing the brine
        information amoung other things
    :Type  settings_file: str (path)
    :param thermonet_file: The path to the json file containing the
        thermonet parameters
    :Type  thermonet_file: str (path)
    :param BHE_file: The path to the json file containing the borehole
        parameters
    :Type  BHE_file: str (path)
    :param heat_pump_load_file: The path to the csv file containing the
        heat pump heating and cooling load parameters
    :Type  heat_pump_load_file: str (path)
    :param heat_pump_setting_file: The path to the json file containing
        the overall heat pump parameters
    :Type  heat_pump_setting_file: str (path)
    :param project_ID: The name of the project
    :Type  project_ID: str

    Returns:
    ::None:: Prints the results to the standard output stream.
    """
    # initialize the required pythermonet classe from their respective
    # files
    brine, net, source_config = initialize_brine_thermonet_BHE_from_files(
        settings_file,
        thermonet_file,
        BHE_file
    )

    # Update the net with the topology, note that this have to include
    # the flow calculations from pandapipes
    net = read_dimensioned_topology(net, pipe_file)

    # load the aggregated heating loads from the heat pump files
    aggload = aggload_from_heat_pump_data_file(
        heat_pump_load_file,
        heat_pump_setting_file,
        brine
    )

    # run pythermonet
    source_config = run_sourcedimensioning(brine, net, aggload, source_config)
    print_project_id(project_ID)
    print_source_dimensions(source_config, net)

    return None
