# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 08:53:07 2022

@author: SOEB
"""
##########################
# To consider or implement
##########################
# 1: Samtidighedsfaktor anvendes også på køling. Er det en god ide?
# 2: Der designes to rørsystemer - et for varme og et for køling. Bør ændres?

# Conceptual model drawings are found below the code

import numpy as np
import pandas as pd
import pandapipes as pp
from .fThermonetDim import ils, Re, dp, Rp, CSM, RbMP, RbMPflc, Halley
from thermonet.dimensioning.thermonet_classes import aggregatedLoad
import pygfunction as gt



# Read heat pump data and calculate ground loads
def read_heatpumpdata(hp, HP_file):
    
    HPS = pd.read_csv(HP_file, sep = '\t+', engine='python');                                # Heat pump input file
    HPS = HPS.values  # Load numeric data from HP file
    
    # Sort heatpumps in ascending order by heat pump ID
    HPS = HPS[HPS[:,0].argsort()]
   
    P_y_H = HPS[:,1];
    P_m_H = HPS[:,2];
    P_d_H = HPS[:,3];
    COP_y_H = HPS[:,4];
    COP_m_H = HPS[:,5];
    COP_d_H = HPS[:,6];
    hp.dT_H = HPS[:,7];
    P_y_C = HPS[:,8];
    P_m_C = HPS[:,9];
    P_d_C = HPS[:,10];
    EER = HPS[:,11];
    hp.dT_C = HPS[:,12];
   
    
    # Calcualate ground loads from building loads
    N_HP = len(P_y_H);
        
    
    # Convert building loads to ground loads - heating
    hp.P_s_H = np.zeros([N_HP,3]);
    hp.P_s_H[:,0] = (COP_y_H-1)/COP_y_H * P_y_H; # Annual load (W)
    hp.P_s_H[:,1] = (COP_m_H-1)/COP_m_H * P_m_H; # Monthly load (W)
    hp.P_s_H[:,2] = (COP_d_H-1)/COP_d_H * P_d_H; # Daily load  (W)
    
    
    #KART COOLING - hvis ingen køl angivet forbliver alle værdier på default NaN
    if np.sum(np.abs(P_y_C)) > 1e-6:
    
        # Convert building loads to ground loads - cooling
        hp.P_s_C = np.zeros([N_HP,3]);
        hp.P_s_C[:,0] = (EER + 1)/EER * P_y_C; # Annual load (W)
        hp.P_s_C[:,1] = (EER + 1)/EER * P_m_C; # Monthly load (W)
        hp.P_s_C[:,2] = (EER + 1)/EER * P_d_C; # Daily load (W)
    
        # First column in hp.P_s_H respectively hp.P_s_C  are equal but with opposite signs 
        hp.P_s_H[:,0] = hp.P_s_H[:,0] - hp.P_s_C[:,0];                          # Annual imbalance between heating and cooling, positive for heating (W)
        hp.P_s_C[:,0] = - hp.P_s_H[:,0];                                        # Negative for cooling


    return hp


# Read topology data
def read_topology(net, TOPO_file):
    
    # Load grid topology
    TOPO = np.loadtxt(TOPO_file,skiprows = 1,usecols = (1,2,3,4));          # Load numeric data from topology file
    net.SDR = TOPO[:,0];
    net.L_traces = TOPO[:,1];
    net.N_traces = TOPO[:,2];
    net.dp_PG = TOPO[:,3];      # Total allowed pressure drop over the forward + retun pipe in a trace
    net.L_segments = 2 * net.L_traces * net.N_traces;
    
    
    I_PG = pd.read_csv(TOPO_file, sep = '\t+', engine='python');            # Load the entire file into Panda dataframe
    pipeGroupNames = I_PG.iloc[:,0];                                        # Extract pipe group IDs
    I_PG = I_PG.iloc[:,5];                                                  # Extract IDs of HPs connected to the different pipe groups
    # Create array containing arrays of integers with HP IDs for all pipe sections.
    # Convert 1-based indices from file to 0-based indices for code.
    IPGA = [np.asarray(I_PG.iloc[i].split(',')).astype(int) - 1 for i in range(len(I_PG))]
    I_PG = IPGA                                                             # Redefine I_PG
    net.I_PG = I_PG;
    del IPGA, I_PG;                                                         # Get rid of IPGA

    
    return net, pipeGroupNames

# Read topology for thermonet with already dimesnioned pipe system
def read_dimensioned_topology(net, brine, TOPO_file):
    
    # Load grid topology
    TOPO = np.loadtxt(TOPO_file,skiprows = 1,usecols = (1,2,3,4,5,6));      # Load numeric data from topology file
    net.d_selectedPipes_H = TOPO[:,0] / 1000;                               # Pipe outer diameters, convert from [mm] to [m]
    net.d_selectedPipes_C = net.d_selectedPipes_H                           # User specified pipe dimensions are both for heating and cooling
    
    net.SDR = TOPO[:,1];
    net.L_traces = TOPO[:,2];
    net.N_traces = TOPO[:,3];
    net.L_segments = 2 * net.L_traces * net.N_traces;
        
    I_PG = pd.read_csv(TOPO_file, sep = '\t+', engine='python');            # Load the entire file into Panda dataframe
    pipeGroupNames = I_PG.iloc[:,0];                                        # Extract pipe group IDs
    
    # User supplied peak flows
    Q_PG_H = TOPO[:,4]
    Q_PG_C = TOPO[:,5]
    
    
    # Calculate Reynolds number for selected pipes for heating
    net.di_selected_H = net.d_selectedPipes_H*(1-2/net.SDR);                # Compute inner diameter of selected pipes (m)
    v_H = Q_PG_H/np.pi/net.di_selected_H**2*4;                              # Compute flow velocity for selected pipes (m/s)
    net.Re_selected_H = Re(brine.rho,brine.mu,v_H,net.di_selected_H);       # Compute Reynolds numbers for the selected pipes (-)
    
    # Calculate Reynolds numbers for selected pipes for cooling
    net.di_selected_C = net.di_selected_H                                   # User specified pipe dimensions are both for heating and cooling
    v_C = Q_PG_C/np.pi/net.di_selected_C**2*4;                              # Compute flow velocity for selected pipes (m/s)
    net.Re_selected_C = Re(brine.rho,brine.mu,v_C,net.di_selected_C);       # Compute Reynolds numbers for the selected pipes (-)

    
    # Calculate total brine volume in the grid pipes
    net.V_brine = sum(net.L_segments*np.pi*net.di_selected_H**2/4);
    
    return net, pipeGroupNames


def update_net_with_pandapipes_flow(net_pthn, brine, TOPO_file, HP_file):
    """
    A wrapper from the pandapipes flow calculation which also updates 
    the pythermonet net object with both the topology used in pandapipes 
    (list of pipes) and with results of the flow calculation. 

    Args
    :param net_pthn: The pythermonet net to which we want to add the 
        topology and the results of the pandapipes flow calculation
    :type  net_pthn: pythermonet net object
    :param brine: The pythermonet brine containing the fluid parameters
    :type  brine: pythermonet brine object
    :param TOPO_file: The path to the csv file containing the pipe 
        topology need for pandapipes, one line for each pipe
    :type  TOPO_file: str (path)
    :param HP_file: The path to the csv file containing the heat pump 
        information, one line for each heat pump
    :type  HP_file: str (path)

    Returns
    :param net_pthn: The pythermonet net updated with the topology and
        the results of the pandapipes flow calculation
    :type  net_pthn: pythermonet net object
    """
    # initialise the pandapipes network
    net_pp = pp.create_empty_network(fluid='water')
    # Change the parameters of the pandapipe fluid to match the brine
    net_pp = update_pandapipes_fluid(net_pp, brine)
    
    # add the pipe and junctions to net_pp
    net_pp = read_pandapipes_topology(net_pp, TOPO_file)
    # and add it to net_pthn 
    net_pthn = read_pipe_list_topology(net_pthn, TOPO_file) 

    # Set the location of the external gird, this is always 0 this setup
    ext_grid_loc = 0
    pp.create_ext_grid(net_pp, ext_grid_loc, p_bar=1, type='p')

    # Load the heat pump data here, as we need to check if we also have
    # to run a flow calculation for the cooling case
    heat_pump_data = pd.read_csv(HP_file, sep='\t+', engine='python')
    
    # $$$ KRI, I'm doing it in this way because then it should be easier
    # to generalise to the case where we need many different
    # configurations
    net_modes = ['heating']
    # check if we need to calculate a flow cooling
    if (heat_pump_data['Yearly_cooling_load_(W)'] > 0).any():
        net_modes.append('cooling')

    # toggle to remove the sources between 1st and 2nd run
    first_run = True 
    for mode in net_modes:
        if mode == 'heating':
            mass_flow_kg_per_s = mass_flow_from_heat_pump_load(
                    load=heat_pump_data['Daily_heating_load_(W)'],
                    COP=heat_pump_data['Hour_COP'],
                    delta_temperature=heat_pump_data['dT_HP_Heating'],
                    fluid_heat_capacity=brine.c,
                    heating=True
                )
        elif mode == 'cooling':
            mass_flow_kg_per_s = mass_flow_from_heat_pump_load(
                load=heat_pump_data['Daily_cooling_load_(W)'],
                COP=heat_pump_data['EER'],
                delta_temperature=heat_pump_data['dT_HP_Cooling'],
                fluid_heat_capacity=brine.c,
                heating=False
            )
        # before the connecting the source in the second run we need to
        # remove the previous loads/sources
        if first_run is False:
            pp.drop_elements_at_junctions(
                net_pp,
                heat_pump_data['at_junction_no'],
                branch_elements=False
            )
        pp.create_sources(net_pp, junctions=heat_pump_data['at_junction_no'],
                          mdot_kg_per_s=mass_flow_kg_per_s)

        # after the completion of the first run, change the toggle
        first_run = False

        pp.pipeflow(net_pp, mode="hydraulics", 
                    friction_model=net_pthn.friction_model_pp)
        # update with flow (renolds numbers)
        net_pthn = update_pthn_net_with_pandapipes_flow(net_pthn, net_pp, mode)

    return net_pthn


def plot_pandapipes_topology(TOPO_file, HP_file, plot_pipe_ID=True, 
                             plot_source_ID=True, pipe_ID_color='k',
                             source_ID_color='g'):
    """
    A wrapper to combine the pipe list format with the pandapipes 
    plotting routine. Note the pandapipes can also compile features in
    collections, this is not done here.

    Args
    :param TOPO_file: The path to the csv file containing the pipe 
        topology need for pandapipes, one line for each pipe
    :type  TOPO_file: str (path)
    :param HP_file: The path to the csv file containing the heat pump 
        information, one line for each heat pump
    :type  HP_file: str (path)

    Returns
    :param net_pthn: The pythermonet net updated with the topology and
        the results of the pandapipes flow calculation
    :type  net_pthn: pythermonet net object
    """
    import matplotlib.pyplot as plt 
    # initialise the pandapipes network
    net_pp = pp.create_empty_network(fluid='water')
    # Change the parameters of the pandapipe fluid to match the brine
    
    # add the pipe and junctions to net_pp
    net_pp = read_pandapipes_topology(net_pp, TOPO_file)

    # Set the location of the external gird, this is always 0 this setup
    ext_grid_loc = 0
    pp.create_ext_grid(net_pp, ext_grid_loc, p_bar=1, type='p')

    # Load the heat pump data here, as we need to check if we also have
    # to run a flow calculation for the cooling case
    heat_pump_data = pd.read_csv(HP_file, sep='\t+', engine='python')
    
    mass_flow_kg_per_s = mass_flow_from_heat_pump_load(
        load=heat_pump_data['Daily_heating_load_(W)'],
        COP=heat_pump_data['Hour_COP'],
        delta_temperature=heat_pump_data['dT_HP_Heating'],
        heating=True
    )
    # before the connecting the source in the second run we need to
    # remove the previous loads/sources
    pp.create_sources(net_pp, junctions=heat_pump_data['at_junction_no'],
                      mdot_kg_per_s=mass_flow_kg_per_s)

    ax = pp.plotting.simple_plot(net_pp, plot_sources=True, show_plot=False)
    if plot_pipe_ID is True:
        ax = add_pipe_numbers_to_simple_plot(ax, net_pp, 
                                             text_color=pipe_ID_color)
    if plot_source_ID is True:
        ax = add_source_numbers_to_simple_plot(ax, net_pp, 
                                               text_color=source_ID_color)
        
    return ax


def add_pipe_numbers_to_simple_plot(ax, net_pp, text_color='k'):
    """
    Adds the pipe IDs  to the middle of the pipe in the supplied plot

    Args
    :param ax: A matplotlib.pyplot Axes object containing the layout of
        the network
    :type  ax: matplotlib.pyplot Axes object
    :param net_pp: The pandapipe net
    :type  net_pp: pandapipes net object
    :param text_color: The color that the IDs are plotted in. 
    :type  text_color: str

    Returns
    :param ax: A matplotlib.pyplot Axes object containing the layout of
        the network with the pipe ID added
    :type  ax: matplotlib.pyplot Axes object
    """
    for index, pipe in net_pp.pipe.iterrows():
        avg_pipe_coords = np.average(net_pp.junction_geodata.iloc[
            pipe[['from_junction', 'to_junction']].values], axis=0)
        ax.text(avg_pipe_coords[0], avg_pipe_coords[1], index, c=text_color)

    return ax    


def add_source_numbers_to_simple_plot(ax, net_pp, text_color='b'):
    """
    Adds the source IDs next to the junction to which the source is 
    connected following the layout of the supplied Axes

    Args
    :param ax: A matplotlib.pyplot Axes object containing the layout of
        the network
    :type  ax: matplotlib.pyplot Axes object
    :param net_pp: The pandapipe net
    :type  net_pp: pandapipes net object
    :param text_color: The color that the IDs are plotted in. 
    :type  text_color: str

    Returns
    :param ax: A matplotlib.pyplot Axes object containing the layout of
        the network with the source ID added
    :type  ax: matplotlib.pyplot Axes object
    """
    for index, source in net_pp.source.iterrows():
        source_coords = net_pp.junction_geodata.iloc[source['junction']].values
        ax.text(source_coords[0], source_coords[1], index, c=text_color)
    return ax    


def update_pthn_net_with_pandapipes_flow(net_pthn, net_pp, mode):
    """
    Adds the reynolds numbers to the pythermonet net object from the 
    pandapipes flow calculation

    Args
    :param net_pthn: The pythermonet net to which we want to add the
        results of the pandapipes flow calculation
    :type  net_pthn: pythermonet net object
    :param net_pp: The pandapipe net with flow calculations
    :type  net_pp: pandapipes net object
    :param mode: The current mode of the flow calculation, either
        'heating' or 'cooling'
    :type  mode: str
    
    Returns 
    :param net_pthn: The pythermonet net with reynolds numbers from the
        the pandapipes flow calculation
    :type  net_pthn: pythermonet net object
    """
    if mode == 'heating':
        net_pthn.Re_selected_H = net_pp.res_pipe['reynolds'].values
    elif mode == 'cooling':
        net_pthn.Re_selected_C = net_pp.res_pipe['reynolds'].values
    else:
        raise ValueError('Somehting is wrong the pandapipes mode, check the'
                         'loop in "update_net_with_pandapipes_flow"')

    return net_pthn


def read_pipe_list_topology(net_pthn, TOPO_file):
    """
    Loads the pipe list topology to the pythermonet net object

    Args
    :param net_pthn: The pythermonet net object 
    :type  net_pthn: pythermonet net object
    :param TOPO_file: The path to the csv file containg the pipe 
        topology in a list format, i.e., one line per pipe
    :type  TOPO_file: str (path)

    Returns    
    :param net_pthn: The pythermonet net object with the pipe topology
    :type  net_pthn: pythermonet net object
    """
    # Load grid topology
    pipe_data = pd.read_csv(TOPO_file, sep='\t+', engine='python')
    # when loading pipe outer diameters convert from mm to m
    net_pthn.d_selectedPipes_H = pipe_data["outer_diameter(mm)"].values / 1000
    net_pthn.d_selectedPipes_C = net_pthn.d_selectedPipes_H
    net_pthn.SDR = pipe_data['SDR'].values
    net_pthn.L_traces = pipe_data['length(m)'].values
    net_pthn.N_traces = np.ones_like(pipe_data['SDR'])  # is redundant,but keep
    net_pthn.L_segments = 2 * net_pthn.L_traces  # back and forth

    # Calculate Reynolds number for selected pipes for heating
    net_pthn.di_selected_H = pipe_inner_diameter(net_pthn.d_selectedPipes_H, 
                                                 net_pthn.SDR)
    net_pthn.di_selected_C = net_pthn.di_selected_H

    # Calculate total brine volume in the grid pipes
    net_pthn.V_brine = sum(net_pthn.L_segments*np.pi*net_pthn.di_selected_H**2
                           / 4)

    return net_pthn


def mass_flow_from_heat_pump_load(load, COP=3., delta_temperature=3.,
                                  fluid_heat_capacity=4.184e3, heating=True):
    """
    Calculates the required mass flow of brine based on the specified
    thermal load in heating mode.

    Args
    :param load: the load of the heat pump [W]
    :type  load: float or list
    :param COP: the coefficient of performance in heating mode [~]
    :type  COP: float or list
    :param delta_temperature: the change in brine tempereature across
                              the heat pump [K]
    :type  delta_temperature: float or list
    :param fluid_heat_capacity: the heat capacity of the brine[J/(kg*K)]
        The default value is for water
    :type  fluid_heat_capacity: float or list

    Returns
    :param mass_flow: The mass flow need to provide the thermal load of
        the heat pump(s)
    :type  mass_flow: float or numpy array
    """
    # Calculate the energy extracted from the brine
    heating_sign = -1 if heating is True else 1
    load_on_thermonet = np.multiply(load, (1 + heating_sign *
                                           np.divide(1, COP)))
    mass_flow = np.divide(load_on_thermonet,
                          np.multiply(fluid_heat_capacity, delta_temperature))
    return mass_flow


def update_pandapipes_fluid(net_pp, brine):
    """
    Update the pandapipes net fluid parameters to the constant values
    as specified in pythermonet brine object.

    Args
    :param net_pp: The pandapipes net initialized with a standard fluid,
        e.g. "water"
    :type  net_pp: pandapipes net object
    :param brine: The pythermonet brine object
    :type  brine: pythermonet brine object

    Return
    :param net_pp: The pandapipes net now with set fixed values for the 
        density, heat capacity, and viscosity
    :type  net_pp: pandapipes net object
    """
    # mute the warning from pandapipes
    flags = {"warn_on_duplicates": False, }
    # the current mutable properties
    pp.create_constant_property(net_pp, 'density', brine.rho, **flags)
    pp.create_constant_property(net_pp, 'heat_capacity', brine.c, **flags)
    pp.create_constant_property(net_pp, 'viscosity', brine.mu, **flags)
    return net_pp


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


def read_pandapipes_topology(net_pp, TOPO_file):
    """
    Reads the topology and passes it to the pandapipes structures

    Args
    :param net_pp: The pandapipe net
    :type  net_pp: pandapipes net object
    :param TOPO_file: The path to the csv file containing the pipe 
        topology need for pandapipes, one line for each pipe
    :type  TOPO_file: str (path)

    Returns 
    :param net_pp: The pandapipe net now with update pipe topology
    :type  net_pp: pandapipes net object
    """
    pipe_data = pd.read_csv(TOPO_file, sep='\t+', engine='python')
    # the junction count start at zero so we need +1 
    n_junctions = np.max(pipe_data[["from_junction", "to_junction"]]) + 1

    pipe_data["inner_diameter_m"] = pipe_inner_diameter(
            outer_diameter=np.divide(pipe_data["outer_diameter(mm)"], 1000.),
            SDR=pipe_data["SDR"]
        )
    
    # initialise the juncions and pipes
    pp.create_junctions(net_pp, n_junctions, pn_bar=1, tfluid_k=293.15)

    pp.create_pipes_from_parameters(
        net_pp,
        from_junctions=pipe_data["from_junction"],
        to_junctions=pipe_data["to_junction"],
        length_km=np.divide(pipe_data['length(m)'], 1000.),
        k_mm=pipe_data["roughness(mm)"],
        diameter_m=pipe_data['inner_diameter_m'],
        )
    
    return net_pp


def read_aggregated_load_pandapipes(aggLoad, brine, HP_file):
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
    heat_pump_load = pd.read_csv(HP_file, sep='\t+', engine='python')

    doCooling = (heat_pump_load['Yearly_cooling_load_(W)'] > 0).any()
    # Extract the minimum heat temperature lift/drop at the heat pumps
    dT_H = np.min(heat_pump_load['dT_HP_Heating'])
    dT_C = np.min(heat_pump_load['dT_HP_Cooling'])

    # use the number of heat pumps to estimate the coincidence factor
    n_heat_pump = len(heat_pump_load)
    coincidence_factor = aggLoad.f_peak*(0.62 + 0.38/n_heat_pump)

    # Calculate ground loads from COP (heating)
    network_thermal_loads_heating = np.zeros(3)
    network_thermal_loads_heating[0] = np.sum(heat_pump_network_thermal_load(
        heat_pump_load['Yearly_heating_load_(W)'],
        heat_pump_load['Year_COP'],
        heating=True
        ))
    network_thermal_loads_heating[1] = np.sum(heat_pump_network_thermal_load(
        heat_pump_load['Winter_heating_load_(W)'],
        heat_pump_load['Winter_COP'],
        heating=True
        ))
    network_thermal_loads_heating[2] = np.sum(heat_pump_network_thermal_load(
        heat_pump_load['Daily_heating_load_(W)'],
        heat_pump_load['Hour_COP'],
        heating=True
        )) * coincidence_factor

    # KART COOLING
    if doCooling:
        network_thermal_loads_cooling = np.zeros(3)
        # Calculate ground loads from EER (cooling)
        network_thermal_loads_cooling[0] = np.sum(
            heat_pump_network_thermal_load(
                heat_pump_load['Yearly_cooling_load_(W)'],
                heat_pump_load['EER'],
                heating=False
            ))
        network_thermal_loads_cooling[1] = np.sum(
            heat_pump_network_thermal_load(
                heat_pump_load['Summer_cooling_load_(W)'],
                heat_pump_load['EER'],
                heating=False
            ))
        network_thermal_loads_cooling[2] = np.sum(
            heat_pump_network_thermal_load(
                heat_pump_load['Daily_cooling_load_(W)'],
                heat_pump_load['EER'],
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

    # COOLING
    if doCooling:
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


def read_aggregated_load(aggLoad, brine, agg_load_file):
    
    # Load aggregated load from file
    load = pd.read_csv(agg_load_file, sep = '\t+', engine='python');        # Heat pump input file
    load = load.values 
    
    # Parse values from file
    N_HP = load[0,0];
    P_y_H = load[0,1];
    P_m_H = load[0,2];
    P_d_H = load[0,3];
    COP_y_H = load[0,4];
    COP_m_H = load[0,5];
    COP_d_H = load[0,6];
    dT_H = load[0,7];
    P_y_C = load[0,8];
    P_m_C = load[0,9];
    P_d_C = load[0,10];
    EER = load[0,11];
    dT_C = load[0,12];

  
    S = aggLoad.f_peak*(0.62 + 0.38/N_HP);

    # Calculate ground loads from COP (heating)
    P_s_H = np.zeros(3);
    P_s_H[0] = (COP_y_H-1)/COP_y_H * P_y_H;     # Annual load (W)
    P_s_H[1] = (COP_m_H-1)/COP_m_H * P_m_H;     # Monthly load (W)
    P_s_H[2] = (COP_d_H-1)/COP_d_H * P_d_H * S; # Daily load with simultaneity factor (W)

    #KART COOLING
    if np.abs(P_y_C) > 1e-6:
        # Calculate ground loads from EER (cooling)
        P_s_C = np.zeros(3);
        P_s_C[0] = (EER + 1)/EER * P_y_C;       # Annual load (W)
        P_s_C[1] = (EER + 1)/EER * P_m_C;       # Monthly load (W)
        P_s_C[2] = (EER + 1)/EER * P_d_C * S;   # Daily load (W)

        # First columns in hp.P_s_H respectively hp.P_s_C are equal but with opposite signs 
        P_s_H[0] = P_s_H[0] - P_s_C[0];         # Annual imbalance between heating and cooling, positive for heating (W)
        P_s_C[0] = - P_s_H[0];                  # Negative for cooling

    
    aggLoad.Qdim_H = P_s_H[2] / dT_H / brine.rho / brine.c;
    aggLoad.To_H = aggLoad.Ti_H - dT_H;
    aggLoad.P_s_H = P_s_H;
    
    #KART COOLING
    if np.abs(P_y_C) > 1e-3: 
        aggLoad.Qdim_C = P_s_C[2] / dT_C / brine.rho / brine.c;
        aggLoad.To_C = aggLoad.Ti_C + dT_C;
        aggLoad.P_s_C = P_s_C;
    
    
    return aggLoad


def print_project_id(PID):
    # Output to prompt
    print(' ');
    print('************************************************************************')
    print('************************** pythermonet v. 1 ************************')
    print('************************************************************************')
    print(' ');
    print(f'Project: {PID}');
    print(' ');

# Print results to console
def print_pipe_dimensions(net, pipeGroupNames):
    
    doCooling = not np.isnan(net.Re_selected_C).any();
    
    # Print pipe dimensioning results
    print('******************* Suggested pipe dimensions heating ******************'); 
    for i in range(len(net.I_PG)):
        print(f'{pipeGroupNames.iloc[i]}: Ø{int(1000*net.d_selectedPipes_H[i])} mm SDR {int(net.SDR[i])}, Re = {int(round(net.Re_selected_H[i]))}');
    print(' ');
    
    if doCooling:
        print('******************* Suggested pipe dimensions cooling ******************');
        for i in range(len(net.I_PG)):
            print(f'{pipeGroupNames.iloc[i]}: Ø{int(1000*net.d_selectedPipes_C[i])} mm SDR {int(net.SDR[i])}, Re = {int(round(net.Re_selected_C[i]))}');
            
    print(' ');


# Print source dimensioning results to console
def print_source_dimensions(source_config,net):
    
    FPH = source_config.FPH;
    FPC = source_config.FPC;
    doCooling = not np.isnan(FPC);
    
    # Print results to console
    print('***************** Thermonet energy production capacity *****************'); 
    print(f'The thermonet supplies {round(100*FPH)}% of the peak heating demand');  
        
    if doCooling:
        print(f'The thermonet supplies {round(100*FPC)}% of the peak cooling demand');  

    print(' ');

    # BHE specific results
    if source_config.source == 'BHE':
        N_BHE = source_config.NX * source_config.NY;
        
        # Display output in console
        print('********** Suggested length of borehole heat exchangers (BHE) **********'); 
        print(f'Required length of each of the {int(N_BHE)} BHEs = {int(np.ceil(source_config.L_BHE_H/N_BHE))} m for heating');
           
        if doCooling:
            print(f'Required length of each of the {int(N_BHE)} BHEs = {int(np.ceil(source_config.L_BHE_C/N_BHE))} m for cooling');
    
        print(f'Maximum pressure loss in BHEs in heating mode = {int(np.ceil(source_config.dpdL_BHEmax_H))} Pa/m, Re = {int(round(source_config.Re_BHEmax_H))}');
        
        if doCooling:
            print(f'Maximum pressure loss in BHEs in cooling mode = {int(np.ceil(source_config.dpdL_BHEmax_C))} Pa/m, Re = {int(round(source_config.Re_BHEmax_C))}');
               

    elif source_config.source =='HHE':

        N_HHE = source_config.N_HHE;
        
        # Output results to console
        print('********* Suggested length of horizontal heat exchangers (HHE) *********');
        print(f'Required length of each of the {int(N_HHE)} horizontal loops = {int(np.ceil(source_config.L_HHE_H/N_HHE))} m for heating');
            
        if doCooling:
            print(f'Required length of each of the {int(N_HHE)} horizontal loops = {int(np.ceil(source_config.L_HHE_C/N_HHE))} m for cooling');
        
        print(f'Maximum pressure loss in HHE pipes in heating mode = {int(np.ceil(source_config.dpdL_HHEmax_H))} Pa/m, Re = {int(round(source_config.Re_HHEmax_H))}');
            
        if doCooling:
            print(f'Maximum pressure loss in HHE pipes in cooling mode {int(np.ceil(source_config.dpdL_HHEmax_C))} Pa/m, Re = {int(round(source_config.Re_HHEmax_C))}');


    # Average brine temperature calculated as weighted average between source and grid
    T_dimv = (net.V_brine*net.T_dimv + source_config.V_brine*source_config.T_dimv)/(net.V_brine + source_config.V_brine)
 
    print('');
    print('********************** Average brine temperatures **********************');
    print(f'Long-term brine temperature: {T_dimv[0]:.2f}{chr(176)}C')
    print(f'Winter brine temperature: {T_dimv[1]:.2f}{chr(176)}C')
    print(f'Peak load brine temperature: {T_dimv[2]:.2f}{chr(176)}C')

    if source_config.T_dimv[0] < 0:
        print('WARNING: The long-term brine temperature is below zero degrees Celsius which can cause ground freezing. Consider increasing the minimum brine inlet temperature.')





# Wrapper for pygfunction
def pygfunction(t,BHE,L):
    
    BC = 'UHTR'
    D = 0
    a_ss = BHE.l_ss/BHE.rhoc_ss;
    t_pyg = np.flip(t) # t must be in ascending order - remember to flip output back for 3-pulse analysis
    
    BHEfield = gt.boreholes.rectangle_field(BHE.NX, BHE.NY, BHE.D_x, BHE.D_y, L, D, BHE.r_b)
    g_pyg = gt.gfunction.gFunction(BHEfield, a_ss, t_pyg, boundary_condition=BC) 
    
    g_values = np.flip(g_pyg.gFunc)
    
    return g_values


# Function for calculating g-function

# def gfunction(t,BHE,aggLoad,brine,net,Rb):
def gfunction(t,BHE,Rb):

# Note the gfunction gives the BHE wall temperature, but since the applied short 
# term model gives the fluid temperature we also require the borehole resistance
        
    a_ss = BHE.l_ss/BHE.rhoc_ss;                                    # BHE soil thermal diffusivity (m2/s)
    a_g = BHE.l_g/BHE.rhoc_g;                                       # Grout thermal diffusivity (W/m/K)
    
    
    # 1) Borehole field geometry
    x = np.linspace(0,BHE.NX-1,BHE.NX)*BHE.D_x;                     # x-coordinates of BHEs (m)                     
    y = np.linspace(0,BHE.NY-1,BHE.NY)*BHE.D_y;                     # y-coordinates of BHEs (m)
    [XX,YY] = np.meshgrid(x,y);                                     # Meshgrid arrays for distance calculations (m)    
    
    # Logistics for symmetry considerations and associated efficiency gains
    # KART: har ikke tjekket
    NXi = int(np.ceil(BHE.NX/2));                                   # Find half the number of boreholes in the x-direction. If not an equal number then round up to complete symmetry.
    NYi = int(np.ceil(BHE.NY/2));                                   # Find half the number of boreholes in the y-direction. If not an equal number then round up to complete symmetry.
    w = np.ones((NYi,NXi));                                         # Define weight matrix for temperature responses at a distance (-)
    if np.mod(BHE.NX/2,1) > 0:                                      # If NX is an unequal integer then the weight on the temperature responses from the boreholes on the center line is equal to 0.5 for symmetry reasons
        w[:,NXi-1] = 0.5*w[:,NXi-1];
    
    if np.mod(BHE.NY/2,1) > 0:                                      # If NY is an unequal integer then the weight on the temperature responses from the boreholes on the center line is equal to 0.5 for symmetry reasons
        w[NYi-1,:] = 0.5*w[NYi-1,:];
        
    wv = np.concatenate(w);                                         # Concatenate the weight matrix (-)
    swv = sum(wv);                                                  # Sum all weights (-)
    xi = np.linspace(0,NXi-1,NXi)*BHE.D_x;                          # x-coordinates of BHEs (m)                     
    yi = np.linspace(0,NYi-1,NYi)*BHE.D_y;                          # y-coordinates of BHEs (m)
    [XXi,YYi] = np.meshgrid(xi,yi);                                 # Meshgrid arrays for distance calculations (m)
    Yvi = np.concatenate(YYi);                                      # YY concatenated (m)
    Xvi = np.concatenate(XXi);                                      # XX concatenated (m)

    # 2) CSM for long term response (months + years)
    G_BHE = CSM(BHE.r_b,BHE.r_b,t[0:2],a_ss);                       # Compute G-functions for t[0] and t[1] with the cylindrical source model (-)
    s1 = 0;                                                         # Summation variable for t[0] G-function (-)
    s2 = 0;                                                         # Summation variable for t[1] G-function (-)
    for i in range(NXi*NYi):                                        # Line source superposition for all neighbour boreholes for 1/4 of the BHE field (symmetry)
        DIST = np.sqrt((XX-Xvi[i])**2 + (YY-Yvi[i])**2);            # Compute distance matrix (to neighbour boreholes) (m)
        DIST = DIST[DIST>0];                                        # Exclude the considered borehole to avoid r = 0 m
        s1 = s1 + wv[i]*sum(ils(a_ss,t[0],DIST));                   # Compute the sum of all thermal disturbances from neighbour boreholes (G-function contributions) for t[0] (-)
        s2 = s2 + wv[i]*sum(ils(a_ss,t[1],DIST));                   # Compute the sum of all thermal disturbances from neighbour boreholes (G-function contributions) for t[0] (-)
    G_BHE[0] = G_BHE[0] + s1/swv;                                   # Add the average neighbour contribution to the borehole field G-function for t[0] (-)
    G_BHE[1] = G_BHE[1] + s2/swv;                                   # Add the average neighbour contribution to the borehole field G-function for t[1] (-)

    # Composite cylindrical source model for short term response. Hu et al. 2014. Paper here: https://www.sciencedirect.com/science/article/abs/pii/S0378778814005866?via#3Dihub
    re = BHE.r_b/np.exp(2*np.pi*BHE.l_g*Rb);                    # Heating: Compute the equivalent pipe radius for cylindrical symmetry (m). This is how Hu et al. 2014 define it.
     
    G1 = CSM(BHE.r_b, BHE.r_b, t[2], a_ss)
    G2h = CSM(re, re, t[2], a_g)
    G3 = CSM(BHE.r_b, BHE.r_b, t[2], a_g)
    
    Rw = G1/BHE.l_ss + G2h/BHE.l_g - G3/BHE.l_g;                  # Step response for short term model on the form q*Rw = T (m*K/W). Rw indicates that it is in fact a thermal resistance
    

    # G-function for heating mode
    G_BHE = np.asarray([G_BHE[0], G_BHE[1], (Rw-Rb)*BHE.l_ss])

    # Convert to standard g-function definition (e.g. Eskilsson or Cimmino)
    g_BHE = 2*np.pi*G_BHE

    return g_BHE

# Function for dimensioning pipes
def run_pipedimensioning(d_pipes, brine, net, hp):
    
    # Determine if eth calculation includes cooling
    doCooling = not np.isnan(hp.P_s_C).any();
    
    N_PG = len(net.I_PG);                                               # Number of pipe groups
    
    # Allocate variables
    ind_H = np.zeros(N_PG);                                             # Index vector for pipe groups heating (-)
    d_selectedPipes_H = np.zeros(N_PG);                                 # Pipes selected from dimensioning for heating (length)
    Q_PG_H = np.zeros(N_PG);                                            # Design flow heating (m3/s)
    
    if doCooling:
        ind_C = np.zeros(N_PG);                                         # Index vector for pipe groups cooling (-)
        d_selectedPipes_C = np.zeros(N_PG);                             # Pipes selected from dimensioning for cooling (length)
        Q_PG_C = np.zeros(N_PG);                                        # Design flow cooling (m3/s)

        
    N_HP = len(hp.P_s_H);
   
    # KART Qdim fjernes fra hp -> flyttet til aggLoad
    Qdim_H =  hp.P_s_H[:,2]/hp.dT_H/brine.rho/brine.c;                  # Design flow heating (m3/s)
    
    if doCooling:
        Qdim_C =  hp.P_s_C[:,2]/hp.dT_C/brine.rho/brine.c;              # Design flow cooling (m3/s). 

    # Compute design flow for the pipes
    for i in range(N_PG):
        # KART: np.ndarray.tolist er overflødig?
        # KART: nye S'er for hver rørgruppe
       N_HP_per_trace = len(net.I_PG[i]) / net.N_traces[i]; 
       S = hp.f_peak*(0.62 + 0.38/N_HP_per_trace);
       
       Q_PG_H[i] =  S * sum(Qdim_H[np.ndarray.tolist(net.I_PG[i])])/net.N_traces[i];                        # Sum the heating brine flow for all consumers connected to a specific pipe group and normalize with the number of traces in that group to get flow in the individual pipes (m3/s)

       if doCooling:
           Q_PG_C[i] =  S * sum(Qdim_C[np.ndarray.tolist(net.I_PG[i])])/net.N_traces[i];                    # Sum the cooling brine flow for all consumers connected to a specific pipe group and normalize with the number of traces in that group to get flow in the individual pipes (m3/s)
    
    # Select the smallest diameter pipe that fulfills the pressure drop criterion
    for i in range(N_PG):                                 
        di_pipes = d_pipes*(1-2/net.SDR[i]);                                                                # Compute inner diameters (m). Variable TOPO_H or TOPO_C are identical here.
        ind_H[i] = np.argmax(2*net.L_traces[i]*dp(brine.rho,brine.mu,Q_PG_H[i],di_pipes)<net.dp_PG[i]);     # Find first pipe with a pressure loss less than the target for heating (-)
        d_selectedPipes_H[i] = d_pipes[int(ind_H[i])];                                                      # Store pipe selection for heating in new variable (m)

        if doCooling:
            ind_C[i] = np.argmax(2*net.L_traces[i]*dp(brine.rho,brine.mu,Q_PG_C[i],di_pipes)<net.dp_PG[i]);
            d_selectedPipes_C[i] = d_pipes[int(ind_C[i])]; 

    net.d_selectedPipes_H = d_selectedPipes_H;

    if doCooling:
        net.d_selectedPipes_C = d_selectedPipes_C;                      # Store pipe selection for cooling in new variable (m)
    
    # Compute Reynolds number for selected pipes for heating
    net.di_selected_H = d_selectedPipes_H*(1-2/net.SDR);                # Compute inner diameter of selected pipes (m)
    v_H = Q_PG_H/np.pi/net.di_selected_H**2*4;                          # Compute flow velocity for selected pipes (m/s)
    net.Re_selected_H = Re(brine.rho,brine.mu,v_H,net.di_selected_H);   # Compute Reynolds numbers for the selected pipes (-)
    
    # Calculate totale brine volume in grid pipes
    net.V_brine = sum(net.L_segments*np.pi*net.di_selected_H**2/4);

    if doCooling:
        # Compute Reynolds number for selected pipes for cooling
        net.di_selected_C = d_selectedPipes_C*(1-2/net.SDR);            # Compute inner diameter of selected pipes (m)
        v_C = Q_PG_C/np.pi/net.di_selected_C**2*4;                      # Compute flow velocity for selected pipes (m/s)
        net.Re_selected_C = Re(brine.rho,brine.mu,v_C,net.di_selected_C)# Compute Reynolds numbers for the selected pipes (-)
    
    
    # Calculate aggregated load for later calculations
    N_HP = len(hp.P_s_H);
    
    aggLoad = aggregatedLoad(Ti_H = hp.Ti_H, Ti_C = hp.Ti_C, f_peak=hp.f_peak, t_peak=hp.t_peak)
    S = hp.f_peak*(0.62 + 0.38/N_HP);
    
    aggLoad.Ti_H = hp.Ti_H;
    aggLoad.To_H = hp.Ti_H - sum(Qdim_H*hp.dT_H)/sum(Qdim_H);           # Volumetric flow rate weighted average brine delta-T (C)
    aggLoad.P_s_H = sum(hp.P_s_H);
    # KART korriger spidslast med samtidighedsfaktor
    aggLoad.P_s_H[2] = aggLoad.P_s_H[2] * S;
    # KART ditto dimensionerende flow 
    aggLoad.Qdim_H = sum(Qdim_H) * S;

    
    if doCooling:

        aggLoad.Ti_C = hp.Ti_C;    
        aggLoad.To_C = hp.Ti_C + sum(Qdim_C*hp.dT_C)/sum(Qdim_C);       # Volumetric flow rate weighted average brine delta-T (C)
        aggLoad.P_s_C = sum(hp.P_s_C);
        # KART korriger spidslast med samtidighedsfaktor
        aggLoad.P_s_C[2] = aggLoad.P_s_C[2] * S;
        # KART ditto dimensionerende flow 
        aggLoad.Qdim_C = sum(Qdim_C) * S;

    
    # Return the pipe sizing results
    return net, aggLoad

    
    
# Function for dimensioning sources
def run_sourcedimensioning(brine, net, aggLoad, source_config):
    
    
    # Determine if calculation includes cooling
    doCooling = not np.isnan(aggLoad.P_s_C).any();
    
    
    N_PG = len(net.L_segments);                                         # Number of pipe groups    
      
    # g-function evaluation times
    SECONDS_IN_HOUR = 3600;
    SECONDS_IN_MONTH = 24 * (365/12) * SECONDS_IN_HOUR
    SECONDS_IN_YEAR = 12 * SECONDS_IN_MONTH;

    # Evaluation times for three-pulse analysis are t = [20y 3m t_peak, 3m t_peak, t_peak]
    t_peak = aggLoad.t_peak; # Peak load duration [h]
    t = np.asarray([20 * SECONDS_IN_YEAR + 3 * SECONDS_IN_MONTH + t_peak * SECONDS_IN_HOUR, 3 * SECONDS_IN_MONTH + t_peak * SECONDS_IN_HOUR, t_peak * SECONDS_IN_HOUR], dtype=float);            # time = [10 years + 3 months + 4 hours; 3 months + 4 hours; 4 hours]. Time vector for the temporal superposition (s).       
    
    # Brine (fluid)
    nu_f = brine.mu/brine.rho;                                          # Brine kinematic viscosity (m2/s)  
    a_f = brine.l/(brine.rho*brine.c);                                  # Brine thermal diffusivity (m2/s)  
    Pr = nu_f/a_f;                                                      # Prandtl number (-)                

    # Shallow soil (not for BHEs! - see below)
    omega = 2*np.pi/86400/365.25;                                       # Angular frequency of surface temperature variation (rad/s) 
    a_s = net.l_s_H/net.rhoc_s; # KART potentielt et problem med to ledningsevner, her vælges bare den ene  # Shallow soil thermal diffusivity (m2/s) - ONLY for pipes!!! 
    # KART: følg op på brug af TP i forhold til bogen / gammel kode
    TP = net.A*np.exp(-net.z_grid*np.sqrt(omega/2/a_s));                # Temperature penalty at burial depth from surface temperature variation (K). Minimum undisturbed temperature is assumed . 

    # Compute thermal resistances for pipes in heating mode
    R_H = np.zeros(N_PG);                                               # Allocate pipe thermal resistance vector for heating (m*K/W)
    for i in range(N_PG):                                               # For all pipe groups
        R_H[i] = Rp(net.di_selected_H[i],net.d_selectedPipes_H[i],net.Re_selected_H[i],Pr,brine.l,net.l_p);             # Compute thermal resistances (m*K/W)
     
    
    if doCooling:
        # Compute thermal resistances for pipes in cooling mode
        R_C = np.zeros(N_PG);                                           # Allocate pipe thermal resistance vector for cooling (m*K/W)
        for i in range(N_PG):                                           # For all pipe groups
            R_C[i] = Rp(net.di_selected_C[i],net.d_selectedPipes_C[i],net.Re_selected_C[i],Pr,brine.l,net.l_p);         # Compute thermal resistances (m*K/W)

    
    # Compute delta-qs for superposition of heating load responsesS
    dP_s_H = np.zeros(3);                                               # Allocate power difference matrix for tempoeral superposition (W)
    dP_s_H[0] = aggLoad.P_s_H[0];                                       # First entry is just the annual average power (W)
    dP_s_H[1:] = np.diff(aggLoad.P_s_H);                                # Differences between year-month and month-hour are added (W)

    
    if doCooling:

        dP_s_C = np.zeros(3);                                           # Allocate power difference matrix for tempoeral superposition (W)
        dP_s_C[0] = aggLoad.P_s_C[0];                                   # First entry is just the annual average power (W)
        dP_s_C[1:] = np.diff(aggLoad.P_s_C);                            # Differences between year-month and month-hour are added (W)

    
    # Compute temperature responses in heating and cooling mode for all pipes
    # KART bliv enige om sigende navne der følger konvention og implementer x 4
    FPH = np.zeros(N_PG);                                               # Vector with total heating load fractions supplied by each pipe segment (-)
    G_grid_H = np.zeros([N_PG,3]); 

    if doCooling:    
       FPC = np.zeros(N_PG);                                            # Vector with total cooling load fractions supplied by each pipe segment (-)
       G_grid_C = np.zeros([N_PG,3]);
    
    
    T_tmp = np.zeros([N_PG,3])
    K1 = ils(a_s,t,net.D_gridpipes) - ils(a_s,t,2*net.z_grid) - ils(a_s,t,np.sqrt(net.D_gridpipes**2+4*net.z_grid**2));
    # KART: gennemgå nye varmeberegning - opsplittet på segmenter
    for i in range(N_PG):
        # G-function for grid pipes in i'th pipe group
        G_grid_H[i,:] = CSM(net.d_selectedPipes_H[i]/2,net.d_selectedPipes_H[i]/2,t,a_s) + K1;
        # Fraction of load that can be supplied by the pipe group
        FPH[i] = (net.T0 - (aggLoad.Ti_H + aggLoad.To_H)/2 - TP)*net.L_segments[i]/np.dot(dP_s_H, G_grid_H[i]/net.l_s_H + R_H[i]);    # Fraction of total heating that can be supplied by the i'th pipe segment (-)
        
        # Fluid temperature in i'th pipe group following three-pulse sequence (year,month,peak)
        T_tmp[i,:] = net.T0 - TP - FPH[i]*np.cumsum(dP_s_H*(G_grid_H[i]/net.l_s_H + R_H[i])/net.L_segments[i])
        # Multiply temperature by pipe group volume for calculation of weighted average after for-loop
        T_tmp[i,:] = T_tmp[i,:]*net.L_segments[i]*np.pi*net.di_selected_H[i]**2/4;
    
    # Weighted average fluid temperature for the grid pipes
    net.T_dimv = np.sum(T_tmp,0)/net.V_brine
    del T_tmp
        
    if doCooling:
        for i in range(N_PG):
            G_grid_C[i,:] = CSM(net.d_selectedPipes_C[i]/2,net.d_selectedPipes_C[i]/2,t,a_s) + K1;
            # KART opdateret aggregering
            FPC[i] = ((aggLoad.Ti_C + aggLoad.To_C)/2 - net.T0 - TP)*net.L_segments[i]/np.dot(dP_s_C, G_grid_C[i]/net.l_s_C + R_C[i]);    # Fraction of total heating that can be supplied by the i'th pipe segment (-)

    
    # KART - mangler at gennemgå ny beregning af energi fra grid/kilder
    
    # Heating supplied by thermonet 
    FPH = sum(FPH);                                                     # Total fraction of heating supplied by thermonet (-)
    
    PHEH = (1-FPH)*dP_s_H;                                              # Residual heat demand (W)

    if doCooling:    
        # Cooling supplied by thermonet
        FPC = sum(FPC);                                                 # Total fraction of cooling supplied by thermonet (-)
        
        PHEC = (1-FPC)*dP_s_C;                                          # Residual heat demand (W)
    

    
    ################################ Source sizing ################################
    
    # If BHEs are selected as source
    if source_config.source == 'BHE':
        ###########################################################################
        ############################ Borehole computation #########################
        ###########################################################################
        
        # For readability only
        BHE = source_config;
        

        ri = BHE.r_p*(1 - 2/BHE.SDR);                                   # Inner radius of U pipe (m)
        T0_BHE = net.T0;                                                # Measured undisturbed BHE temperature (C)
        s_BHE = 2*BHE.r_p + BHE.D_pipes;                                # Calculate shank spacing U-pipe (m)
        
        # BHE flow and pressure loss - heating mode
        N_BHE = BHE.NX*BHE.NY;                                          # Number of BHEs (-)
        Q_BHEmax_H = aggLoad.Qdim_H / N_BHE;                            # Peak flow in BHE pipes (m3/s)
        v_BHEmax_H = Q_BHEmax_H/np.pi/ri**2;                            # Flow velocity in BHEs (m/s)
        Re_BHEmax_H = Re(brine.rho,brine.mu,v_BHEmax_H,2*ri);           # Reynold number in BHEs (-)
        dpdL_BHEmax_H = dp(brine.rho,brine.mu,Q_BHEmax_H,2*ri);         # Pressure loss in BHE (Pa/m)
        
        BHE.Re_BHEmax_H = Re_BHEmax_H;                                  # Add Re to BHE instance
        BHE.dpdL_BHEmax_H = dpdL_BHEmax_H;                              # Add pressure loss to BHE instance


        if doCooling:
            # BHE cooling
            Q_BHEmax_C = aggLoad.Qdim_C/N_BHE;                          # Peak flow in BHE pipes (m3/s)
            v_BHEmax_C = Q_BHEmax_C/np.pi/ri**2;                        # Flow velocity in BHEs (m/s)
            Re_BHEmax_C = Re(brine.rho,brine.mu,v_BHEmax_C,2*ri);       # Reynold number in BHEs (-)
            dpdL_BHEmax_C = dp(brine.rho,brine.mu,Q_BHEmax_C,2*ri);     # Pressure loss in BHE (Pa/m)

            BHE.Re_BHEmax_C = Re_BHEmax_C;                              # Add Re to BHE instance
            BHE.dpdL_BHEmax_C = dpdL_BHEmax_C;                          # Add pressure loss to BHE instance

        ######################### Generate g-functions ############################)

        # Borehole resistance - ignoring length effects
        Rb_H = RbMP(brine.l,net.l_p,BHE.l_g,BHE.l_ss,BHE.r_b,BHE.r_p,ri,s_BHE,Re_BHEmax_H,Pr);  # Compute the borehole thermal resistance (m*K/W)    
        
        # Calculate g-function
        if BHE.gFuncMethod == 'ICS':
            g_BHE_H = gfunction(t,BHE,Rb_H)
        elif BHE.gFuncMethod == 'PYG':
            g_BHE_H = pygfunction(t,BHE,1000) # Large L for infinite source in initial estimate
        
        
        if doCooling:        
            Rb_C = RbMP(brine.l,net.l_p,BHE.l_g,BHE.l_ss,BHE.r_b,BHE.r_p,ri,s_BHE,Re_BHEmax_C,Pr);  # Compute the borehole thermal resistance (m*K/W)    
            
            # Calculate g-function
            if BHE.gFuncMethod == 'ICS':
                g_BHE_C = gfunction(t,BHE,Rb_C)
            elif BHE.gFuncMethod == 'PYG':
                g_BHE_C = pygfunction(t,BHE,1000)
        
        # Initial estimate of total BHE length - ignoring length effects in g-function
        dTdz = BHE.q_geo/BHE.l_ss                                       # Geothermal gradient (K/m)
        a = dTdz/(2*N_BHE)
        b = T0_BHE - (aggLoad.Ti_H + aggLoad.To_H)/2
        c = -np.dot(PHEH, g_BHE_H/(2*np.pi*BHE.l_ss) + Rb_H)
        L_BHE_H = (-b + np.sqrt(b**2-4*a*c))/(2*a)
        
        if doCooling:        
           
            b = - T0_BHE + (aggLoad.Ti_C + aggLoad.To_C)/2
            c = -np.dot(PHEC, g_BHE_C/(2*np.pi*BHE.l_ss) + Rb_C)
            L_BHE_C = (-b + np.sqrt(b**2-4*a*c))/(2*a)
        
            
        # Search neighbourhood of the approximate solution considering length effect - heating mode
        # Result is an updated estimate of L_BHE_H and Rb_H 
        eps = np.finfo(np.float64).eps # Machine precision
        tol = 1e-4; # Tolerance for search algorithm
        iter_max = 50;
        
        # Variables for single iteration of numerical search
        dL = L_BHE_H/N_BHE*np.sqrt(eps)                                 # Optimal stepsize depends on the square root of machine epsilon (https://en.wikipedia.org/wiki/Numerical_differentiation#Practical_considerations_using_floating_point_arithmetic)
        L_BHE_H_v = L_BHE_H/N_BHE + np.array([-dL, 0, dL])
        Rb_H_v = np.zeros(3) #KART NØDVENDIGT AT BEHOLDE ALLE TRE VÆRDIER? -> DITTO KØL

        Tbound_H = (aggLoad.Ti_H + aggLoad.To_H)/2
        error_Tf = np.ones(3)
         
        N_iter = 0
        while abs(error_Tf[1]) > tol and N_iter < iter_max + 1:
            for i in range(3):                                          # Compute Rb for the specified number of boreholes and lengths considering flow and length effects (m*K/W)
                
                # Recalculate lenghth dependent Rb
                Rb_H_v[i] = RbMPflc(brine.l,net.l_p,BHE.l_g,BHE.l_ss,brine.rho,brine.c,BHE.r_b,BHE.r_p,ri,L_BHE_H_v[i],s_BHE,Q_BHEmax_H,Re_BHEmax_H,Pr);    # Compute BHE length and flow corrected multipole estimates of Rb for all candidate solutions (m*K/W)
                
                # Update g-function
                if BHE.gFuncMethod == 'ICS':
                    g_BHE_H = gfunction(t,BHE,Rb_H_v[i])
                elif BHE.gFuncMethod == 'PYG':
                    g_BHE_H = pygfunction(t,BHE,L_BHE_H_v[i])
                
                # error is the difference between calculated fluid temperature and Tbound
                error_Tf[i] = T0_BHE + dTdz*L_BHE_H_v[i]/2 - (np.dot(PHEH,g_BHE_H / (2*np.pi*BHE.l_ss) + Rb_H_v[i])) / (L_BHE_H_v[i]*N_BHE) - Tbound_H;
                
            # Calculate updated length estimate    
            L_H_Halley = Halley(L_BHE_H_v[1],dL,error_Tf[0],error_Tf[1],error_Tf[2])
            L_BHE_H_v = L_H_Halley + np.array([-dL,0,dL]);
            
            N_iter += 1;
            
            # Test for convergence after exit of while loop
        if N_iter > iter_max - 1:   
            # If the maximum number of allowed iterations is exceeded fall back to initial estimate of Rb
            print('WARNING: Convergence failed for heating solution. Defaulting to solution without thermal short-circuiting between the U-pipe legs. Boreholes may be too short!')                   

        else:
            # Update combined length of all BHEs and borehole resistance
            L_BHE_H = L_H_Halley*N_BHE;
            Rb_H = Rb_H_v[1]

        # Save final estimate of borehole resistance
        source_config.Rb_H = Rb_H
        
        
        # Search neighbourhood of the approximate solution considering length effect - cooling mode
        # Result is an updated estimate of L_BHE_C and Rb_C 
        if doCooling:
            # Variables for single iteration of numerical search
            dL = L_BHE_C/N_BHE*np.sqrt(eps)                             # Optimal stepsize depends on the square root of machine epsilon (https://en.wikipedia.org/wiki/Numerical_differentiation#Practical_considerations_using_floating_point_arithmetic)
            L_BHE_C_v = L_BHE_C/N_BHE + np.array([-dL, 0, dL])
            Rb_C_v = np.zeros(3)

            Tbound_C = (aggLoad.Ti_C + aggLoad.To_C)/2
            error_Tf = np.ones(3)
            
            N_iter = 0
            while abs(error_Tf[1]) > tol and N_iter < iter_max + 1:
                for i in range(3):                                      # Compute Rb for the specified number of boreholes and lengths considering flow and length effects (m*K/W)
                    
                    # Recalculate length dependent Rb
                    Rb_C_v[i] = RbMPflc(brine.l,net.l_p,BHE.l_g,BHE.l_ss,brine.rho,brine.c,BHE.r_b,BHE.r_p,ri,L_BHE_C_v[i],s_BHE,Q_BHEmax_C,Re_BHEmax_C,Pr);    # Compute BHE length and flow corrected multipole estimates of Rb for all candidate solutions (m*K/W)
                    
                    # Update g-function 
                    if BHE.gFuncMethod == 'ICS':
                        g_BHE_C = gfunction(t,BHE,Rb_C_v[i])
                    elif BHE.gFuncMethod == 'PYG':
                        g_BHE_C = pygfunction(t,BHE,L_BHE_C_v[i])


                    # error is the difference between calculated fluid temperature and Tbound
                    error_Tf[i] = T0_BHE + dTdz*L_BHE_C_v[i]/2 + (np.dot(PHEC,g_BHE_C / (2*np.pi*BHE.l_ss) + Rb_C_v[i])) / (L_BHE_C_v[i]*N_BHE) - Tbound_C;
                    
                # Calculate updated length estimate    
                L_C_Halley = Halley(L_BHE_C_v[1],dL,error_Tf[0],error_Tf[1],error_Tf[2])
                L_BHE_C_v = L_C_Halley + np.array([-dL,0,dL]);
                
                N_iter += 1;
                
                # Test for convergence after exit of while loop
            if N_iter > iter_max - 1:
                
                # If the maximum number of allowed iterations is exceeded fall back to initial estimate of Rb
                print('WARNING: Convergence failed for heating solution. Defaulting to solution without thermal short-circuiting between the U-pipe legs. Boreholes may be too short!')                   
                source_config.Rb_C = Rb_C;

            else:
                
                # Update combined length of all BHEs and borehole resistance
                L_BHE_C = L_C_Halley*N_BHE;
                source_config.Rb_C = Rb_C_v[1];


        # For the final estimate of L_BHE_H and Rb_H calculate 
        # - Brine volume from L_BHE_H
        # - Final g-function (depends on L_BHE_H and Rb_H)
        # - Fluid temperature calculated from g-function
        
        # Total brine volume in BHE heat exchanger - 1U pipe        
        BHE.V_brine = 2*L_BHE_H*np.pi*ri**2;
        
        # Final g-function for heating mode
        if BHE.gFuncMethod == 'ICS':
            g_BHE_H = gfunction(t,BHE,Rb_H)
        elif BHE.gFuncMethod == 'PYG':    
            g_BHE_H = pygfunction(t,BHE,L_BHE_H/N_BHE)
        
        # Brine temperature after three pulses
        BHE.T_dimv =  T0_BHE + dTdz*L_BHE_H/(N_BHE*2) - np.cumsum((PHEH * (g_BHE_H/(2*np.pi*BHE.l_ss) + Rb_H)) / L_BHE_H)

        # Store results in BHE object
        BHE.L_BHE_H = L_BHE_H;
        BHE.FPH = FPH;
        
        if doCooling:
            BHE.L_BHE_C = L_BHE_C;
            BHE.FPC = FPC;
        
        source_config = BHE;
       
        
    # If HHEs are selected as source
    elif source_config.source == 'HHE':    
       
        # For readability only
        HHE = source_config;

        ###########################################################################
        ############################### HHE computation ###########################
        ###########################################################################
    
        ri_HHE = HHE.d*(1 - 2/HHE.SDR)/2;                               # Inner radius of HHE pipes (m)
        ro_HHE = HHE.d/2;                                               # Outer radius of HHE pipes (m)
    
        # Compute combined length of HHEs   
        ind = np.linspace(0,2*HHE.N_HHE-1,2*HHE.N_HHE);                 # Unit distance vector for HHE (-)
        s = np.zeros(2);                                                # s is a temperature summation variable, s[0]: annual, s[1] monthly, hourly effects are insignificant and ignored (C)
        DIST = HHE.D*ind;                                               # Distance vector for HHE (m)
        for i in range(HHE.N_HHE):                                      # For half the pipe segments (2 per loop). Advantage from symmetry.
            s[0] = s[0] + sum(ils(a_s,t[0],abs(DIST[ind!=i]-i*HHE.D))) - sum(ils(a_s,t[0],np.sqrt((DIST-i*HHE.D)**2 + 4*net.z_grid**2))); # Sum annual temperature responses from distant pipes (C)
            s[1] = s[1] + sum(ils(a_s,t[1],abs(DIST[ind!=i]-i*HHE.D))) - sum(ils(a_s,t[1],np.sqrt((DIST-i*HHE.D)**2 + 4*net.z_grid**2))); # Sum monthly temperature responses from distant pipes (C)
        G_HHE = CSM(ro_HHE,ro_HHE,t,a_s);                               # Pipe wall response (-)
        #KART: tjek - i tidligere version var en faktor 2 til forskel
        G_HHE[0:2] = G_HHE[0:2] + s/HHE.N_HHE;                          # Add thermal disturbance from neighbour pipes (-)
        
        # HHE heating
        # KART flyttet aggregering
        # Q_HHEmax_H = sum(hp.Qdim_H)/HHE.N_HHE;                        # Peak flow in HHE pipes (m3/s)
        Q_HHEmax_H = aggLoad.Qdim_H / HHE.N_HHE;                        # Peak flow in HHE pipes (m3/s)
        v_HHEmax_H = Q_HHEmax_H/np.pi/ri_HHE**2;                        # Peak flow velocity in HHE pipes (m/s)
        Re_HHEmax_H = Re(brine.rho,brine.mu,v_HHEmax_H,2*ri_HHE);       # Peak Reynolds numbers in HHE pipes (-)
        dpdL_HHEmax_H = dp(brine.rho,brine.mu,Q_HHEmax_H,2*ri_HHE);     # Peak pressure loss in HHE pipes (Pa/m)
    
        HHE.Re_HHEmax_H = Re_HHEmax_H;                                  # Add Re to HHE instance
        HHE.dpdL_HHEmax_H = dpdL_HHEmax_H;                              # Add pressure loss to HHE instance


        if doCooling:
            # HHE cooling
            # KART fyttet aggregering
            # Q_HHEmax_C = sum(hp.Qdim_C)/HHE.N_HHE;                    # Peak flow in HHE pipes (m3/s)
            Q_HHEmax_C = aggLoad.Qdim_C / HHE.N_HHE;                    # Peak flow in HHE pipes (m3/s)
            v_HHEmax_C = Q_HHEmax_C/np.pi/ri_HHE**2;                    # Peak flow velocity in HHE pipes (m/s)
            Re_HHEmax_C = Re(brine.rho,brine.mu,v_HHEmax_C,2*ri_HHE);   # Peak Reynolds numbers in HHE pipes (-)
            dpdL_HHEmax_C = dp(brine.rho,brine.mu,Q_HHEmax_C,2*ri_HHE); # Peak pressure loss in HHE pipes (Pa/m)

            HHE.Re_HHEmax_C = Re_HHEmax_C;                              # Add Re to HHE instance
            HHE.dpdL_HHEmax_C = dpdL_HHEmax_C;                          # Add pressure loss to HHE instance

        
        # Heating
        Rp_HHE_H = Rp(2*ri_HHE,2*ro_HHE,Re_HHEmax_H,Pr,brine.l,net.l_p);# Compute the pipe thermal resistance (m*K/W)
        G_HHE_H = G_HHE/net.l_s_H + Rp_HHE_H;                           # Add annual and monthly thermal resistances to G_HHE (m*K/W)
        L_HHE_H = np.dot(PHEH,G_HHE_H) / (net.T0 - (aggLoad.Ti_H + aggLoad.To_H)/2 - TP );
        
        # Total brine volume in HHE pipes
        HHE.V_brine = np.pi*ri_HHE**2*L_HHE_H
        # Brine temperature after three pulses (year, month, peak)
        HHE.T_dimv = net.T0 - TP - np.cumsum(PHEH*G_HHE_H)/L_HHE_H
               

        if doCooling:        
            # Cooling
            Rp_HHE_C = Rp(2*ri_HHE,2*ro_HHE,Re_HHEmax_C,Pr,brine.l,net.l_p);# Compute the pipe thermal resistance (m*K/W)
            G_HHE_C = G_HHE/net.l_s_C + Rp_HHE_C;                           # Add annual and monthly thermal resistances to G_HHE (m*K/W)
            #L_HHE_C = np.dot(PHEC,G_HHE_C/TCC1);                           # Sizing equation for computing the required borehole meters (m)
            L_HHE_C = np.dot(PHEC,G_HHE_C) / ((aggLoad.Ti_C + aggLoad.To_C)/2 - net.T0 - TP);
        
        
        # Add results to source configuration
        HHE.FPH = FPH;
        HHE.L_HHE_H = L_HHE_H;
        
        if doCooling:        
            HHE.FPC = FPC;
            HHE.L_HHE_C = L_HHE_C;
        
        source_config = HHE;
    
    return source_config
    

    

    ################## CONCEPTUAL MODEL DRAWINGS FOR REFERENCE ####################
    
    ################ Conceptual model for twin pipe in the ground #################
    #       x1      x2
    #
    #          Air
    #
    # ----------------------
    #
    #         Ground
    #
    #       o1      o2
    #
    #       Legend:
    #       x : mirror source, opposite sign of real source
    #       o : real source, actual pipe
    #     --- : the ground surface where T = 0. Actual ground temperatures are then superimposed.
    #       
    #       T(o1) = q*(R(o1) + R(o2) - R(x1) - R(x2)) + Tu(t)
    #       Tu(t) is the undisturbed seasonal temperature variation at depth
    #       Assumption: surface temperature equal to the undisturbed seasonal temperature (Dirichlet BC)
    #
    #
    ############## Conceptual model for twin pipe in the ground END ###############
    
    ################### Conceptual model for HHE in the ground ####################
    
    # Topology of horizontal heat exchangers (N_HHE = 3)
    # |  Loop  |	    |  Loop  |	      |  Loop  |
    # |	       |	    |	     |	      |	       |
    # |	       |  	    |	     |	      |	       |
    # |<--D--->|<--D--->|<--D--->|<--D--->|<--D--->|
    # |        |   	    |        |	      |        |
    # |        |   	    |        |	      |        |
    # |________|   	    |________|        |________|
    #
    # 
    # Mirror sources (above the ground surfaces) enforce Dirichlet BC on ground surface - similar to thermonet model
    
    ####################### Conceptual model for BHE field ########################
    #
    # Only compute the average temperature response for one of the four sub-rectangles below as there is symmetry between them
    #   
    #        weight = 1
    #            o           o     |     o           o
    #                              |
    #            o           o     |     o           o
    #                              |
    #            o           o     |     o           o
    #                              |
    # NY=7 ------o-----------o-----------o-----------o----- weight = 0.5 (if both NX and NY are unequal, then the center BHE has a weight of 0.25)    
    #                              |
    #            o           o     |     o           o         
    #                              |
    #            o           o     |     o           o
    #                              |
    #            o           o     |     o           o
    #                            
    #                            NX=4
    #
    #       Legend
    #       o : BHE
    #       -- or | : axes of symmetry
    ################# Conceptual model for HHE in the ground END ##################
    
