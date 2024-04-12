import json

import numpy as np
import pandas as pd
import pandapipes as pp


def mass_flow_from_load(load, COP=3., delta_temperature=3.,
                        fluid_heat_capacity=4.184e3, heating=True, **kwargs):
    """
    Calculates the required mass flow of brine based on the specified
    thermal load in heating mode.

    :param load: the load of the heat pump [W]
    :type  load: float
    :param COP: the coefficient of performance in heating mode [~]
    :type  COP: float
    :param delta_temperature: the change in brine tempereature across
                              the heat pump [K]
    :type  delta_temperature: float
    :param fluid_heat_capacity: the heat capacity of the brine[J/(kg*K)]
    :type  fluid_heat_capacity: float
    """
    # Calculate the energy extracted from the brine
    heating_sign = -1 if heating is True else 1
    load_on_thermonet = np.multiply(load, (1 + heating_sign *
                                           np.divide(1, COP)))
    mass_flow = np.divide(load_on_thermonet,
                          np.multiply(fluid_heat_capacity, delta_temperature))
    return mass_flow


def fluid_update_constants(net, brine_dict, **kwargs):
    """
    Update the net fluid parameters to the constant values specified in
    brine_dict.
    The elements should be either "viscosity", "density", or
    "heat_capacity".

    input
    :param net: The pandapipes net initialized with a standard fluid,
        e.g. "water"
    :type  net: pandapipes Net object
    :param brine_dict: A dictionary containing the fluid parameters
        we want to change to constant values
    :type  brine_dict: dict

    output
    :None: The changes are stored in the pandapipesNet
    """
    # mute the warning from pandapipes
    flags = {"warn_on_duplicates": False, }
    # the current mutable properties
    fluid_parameters = ["viscosity", "density", "heat_capacity"]
    for f_param in fluid_parameters:
        if "fluid_" + f_param in brine_dict:
            pp.create_constant_property(net, f_param,
                                        brine_dict["fluid_" + f_param],
                                        **flags)
    return None


def pipe_inner_diameter(outer_diameter, SDR=17, **kwargs):
    """
    Calculates the inner pipe diameter given the outer diameter and the
    SDR

    Args
    :param outer_diameter: The outer diameter of the pipes
    :type  outer_diameter: float or list of floats
    :param SDR: The surface to diameter ratio of the pipe
    :type  SDR: float or list of floats

    Return
    :param -: The inner diameter of the pipes
    :type  -: float or list of floats

    """
    return np.multiply(outer_diameter, (1. - np.divide(2., SDR)))


def update_pp_net_before_save(net):
    """
    Updates the net object before the flow date is save in the geosjon

    The function adds three new columns to the net's sub classes.

    Args
    :param net: The net object containing the flow calculations from
        pandapipes
    :type  net: pp.net object

    Returns
    :None: The three new columns are added to the subclasses in the net
    """
    # Add the name(IDs) to "res_*" data frames
    net.res_pipe["name"] = net.pipe["name"]
    net.res_junction["name"] = net.junction["name"]
    # Calculate the pressure loss per meter and add to "res_pipe"
    net.res_pipe["p_loss_bar_per_m"] = (
        (net.res_pipe["p_from_bar"] - net.res_pipe["p_to_bar"])
        / (net.pipe["length_km"] * 1e3)  # km to m
    )
    return None


def load_and_prepare_data_pp(pipe_file, heat_pump_load_file, settings_file):
    """
    Unpacks the pipe data and settings and prepare it for the pandapipes

    Args
    :param pipe_file: The path to the csv file containing the pipe data
    :type  pipe_file: str (path)
    :param heat_pump_load_file: The path to the csv file containing the
        heat pump data
    :type  heat_pump_load_file: str (path)
    :param settings_file: The path to the file containing the settings
        for the fluid and friction model
    :type  settings_file: str (path)

    Returns
    :param pipe_data: A dataframe containing the properties of the
        pipes, the naming follows that of pandapipes
    :type  pipe_data: pandas Dataframe
    :param heat_pump_data: A dictionary containing the fluid settings
    :type  heat_pump_data: dict
    :param heat_pump_data: A dictionary containing the fluid settings
    :type  heat_pump_data: dict
    """
    pipe_data = pd.read_csv(pipe_file)
    heat_pump_data = pd.read_csv(heat_pump_load_file)
    with open(settings_file, 'r') as file:
        settings = json.load(file)

    # Check which mode is selected
    possible_modes = ["heating", "free_cooling", "pumped_cooling"]
    modes = settings["network_modes"]
    if isinstance(modes, str):
        modes = [modes]
        settings["network_modes"] = modes
    for mode in modes:
        if mode not in possible_modes:
            raise ValueError(f"The selected mode, {mode}, is not availible "
                             "please choose from the following "
                             f"{possible_modes}")

    # pandapipes needs the inner pipe diameters
    pipe_data["inner_diameter_m"] = pipe_inner_diameter(
            outer_diameter=np.divide(pipe_data["outer_diameter(mm)"], 1000.),
            SDR=pipe_data["SDR"]
        )

    # calculate the mass flow from the each heat pump
    for mode in modes:
        if mode == "heating":
            heat_pump_data['_'.join(['mass_flow_kg_per_s', mode])] = \
                mass_flow_from_load(
                    load=heat_pump_data['peak_load(W)'],
                    COP=heat_pump_data['hour_COP'],
                    delta_temperature=heat_pump_data['dT_heat_pump_heating'],
                    fluid_heat_capacity=settings['fluid_heat_capacity'],
                    heating=True
                )
        elif mode == "free_cooling":
            heat_pump_data['_'.join(['mass_flow_kg_per_s', mode])] = \
                mass_flow_from_load(
                    load=heat_pump_data["daily_cooling_load_(W)"],
                    COP=heat_pump_data["EER"],
                    delta_temperature=heat_pump_data['dT_heat_pump_cooling'],
                    fluid_heat_capacity=settings['fluid_heat_capacity'],
                    heating=False
                )
        elif mode == "pumped_cooling":
            raise ValueError("Network mode 'pumped_cooling' not implemented "
                             "yet")
            # load = heat_pump_data["daily_cooling_load_(W)"]
            # heating = False
            # COP = heat_pump_data["EER"]
            # delta_temperature = heat_pump_data['dT_heat_pump_cooling']
    return pipe_data, heat_pump_data, settings


def save_pp_flow_csv(net, pipe_file, mode="heating", overwrite_csv=True,
                     pipe_file_new=""):
    """
    Saves the results of the pythermonet calculation to the geojson

    Args
    :param net: The pandapipes net object containing the results from
        the flow calculation
    :type  net: pandapipes net object
    :param pipe_file: The path to the csv file containing the pipe data
    :type  pipe_file: str (path)
    :param overwrite_csv: Toggle to control whether the results are
        saved in the origional csv file (True) or if a new is created
        (False)
    :type  overwrite_csv: bool
    :param pipe_file_new: The path to the new csv file, is only used if
        overwrite_csv if False
    :type  pipe_file_new: str (path)

    Returns
    :None: The data is dumped in either the original csv file or in the
        new file specified in pipe_file_new
    """
    # reload the original pipe data
    pipe_data = pd.read_csv(pipe_file)

    # first add names and pressure loss to the res_pipes subclass
    update_pp_net_before_save(net)
    # the params which should be added to the pipes in the geojson
    # the names followes the names in pandapipes
    res_pipe_params = [
        "v_mean_m_per_s",
        "vdot_norm_m3_per_s",
        "reynolds",
        "lambda",
        "p_loss_bar_per_m"
        ]
    params_translate = {
        "v_mean_m_per_s": "Mean flow velocity (m/s)",
        "vdot_norm_m3_per_s": "Volumetric flow (m^3/s)",
        "reynolds": "Reynolds number",
        "lambda": "Friction factor",
        "p_loss_bar_per_m": "Pressure loss (bar/m)",
        "p_bar": "Pressure (bar)"
        }
    for pipe_ind in net.res_pipe["name"]:
        for param in res_pipe_params:
            param_with_mode = ' '.join([params_translate[param], mode])
            pipe_data.at[pipe_ind, param_with_mode] = \
                net.res_pipe[param][pipe_ind]

    if overwrite_csv is True:
        pipe_data.to_csv(pipe_file)
    else:
        if pipe_file_new == "":
            raise ValueError("Overwrite is False but no new path to where the"
                             " file should be save was given, specify as "
                             "pipe_file_new")
        else:
            pipe_data.to_csv(pipe_file_new, index=False)

    return True


def wrapper_pandapipes_flow_from_csv(pipe_file, heat_pump_load_file,
                                     settings_file, overwrite_pipe_csv=True,
                                     pipe_file_new="", modes=["heating"]):
    """
    A wrapper for running the flow calculations in pandapipes using the
    topology specified in csv

    Args
    :param pipe_file: The path to the csv file containing the topology
        data
    :type  pipe_file: str (path)
    :param heat_pump_load_file: The path to the csv file containing the
        heat pump data
    :type  heat_pump_load_file: str (path)
    :param setting_file: The path to the json file containing the
        fluid values and friction model
    :type  setting_file: str (path)
    :param overwrite_pipe_csv: Toggle to specify is the original pipe
        csv should be update (True) and a new should be made (False)
    :type  overwrite_pipe_csv: bool
    :param pipe_file_new: The path to the new csv file if
        overwrite_pipe_csv is false
    :type  pipe_file_new: str (path)

    Returns
    :None: The data is dumped in the original or the new pipe csv file.
    """

    # Load the layout of the topology
    pipe_data, heat_pump_data, settings = \
        load_and_prepare_data_pp(pipe_file, heat_pump_load_file, settings_file)

    # Extract how many junctions the grid contains
    n_junctions = np.max(pipe_data[["junction_from", "junction_to"]])+1

    # Set the location of the external gird, this is always 0 this setup
    ext_grid_loc = 0

    # initialize the pandapipes objects
    net = pp.create_empty_network(fluid='water')
    # The fluid parameters need to be updated accordingly to the settings
    # dictionary
    fluid_update_constants(net, settings)

    # add the different components to the net
    pp.create_junctions(net, n_junctions, pn_bar=1, tfluid_k=293.15)

    pp.create_pipes_from_parameters(
        net,
        from_junctions=pipe_data["junction_from"],
        to_junctions=pipe_data["junction_to"],
        length_km=np.divide(pipe_data['length(m)'], 1000.),
        k_mm=pipe_data["roughness(mm)"],
        diameter_m=pipe_data['inner_diameter_m'],
        name=list(pipe_data.index)
        )

    pp.create_ext_grid(net, ext_grid_loc, p_bar=1, type='p')

    # We need to keep track of whether the flow results have been saved
    first_run = True
    flow_saved_to_file = False
    for mode in settings["network_modes"]:
        # $$$ maybe I should use the time series functionallity of pandapower
        if first_run is False:
            pp.drop_elements_at_junctions(
                net,
                heat_pump_data["junction_connection"],
                branch_elements=False
                )
        pp.create_sources(
            net,
            junctions=heat_pump_data["junction_connection"],
            mdot_kg_per_s=heat_pump_data['_'.join(['mass_flow_kg_per_s',
                                                   mode])],
            name=list(heat_pump_data.index)
            )

        # run the flow simulation
        pp.pipeflow(
            net,
            mode="hydraulics",
            friction_model=settings["friction_model"]
            )

        # if the flow results of one mode have been saved to the output
        # file and we want to have a separate file for to save the
        # output in, then we need to add the second results to the
        # output file, this is done by loading the output of the first
        # flow mode as the original file for the saving routine for the
        # second flow mode
        if flow_saved_to_file is True and overwrite_pipe_csv is False:
            pipe_file = pipe_file_new

        flow_saved_to_file = save_pp_flow_csv(
            net,
            pipe_file,
            mode=mode,
            overwrite_csv=overwrite_pipe_csv,
            pipe_file_new=pipe_file_new,
            )
        
        # after the completion of the first run, change the toggle
        first_run = False

    return None
