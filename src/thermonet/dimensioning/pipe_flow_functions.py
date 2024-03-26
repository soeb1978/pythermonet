import json
import os

import numpy as np
import pandas as pd
import pandapipes as pp


def load_topology_csv_to_pp_net(topo_file, settings_file):
    """
    """
    # Load the layout of the topology
    network_df = pd.read_csv(topo_file)

    # load the flow settings file.
    with open(settings_file, 'r') as file:
        data = json.load(file)

    # the number of junctions
    n_junctions = network_df.shape[0]
    # if we want to simulate the flow in both directions
    if setting_dict["full_network"]:
        n_junctions = n_junctions*2
        from_connection, to_connection, pipe_lengths = \
            pipe_connection_lists(network_df, **setting_dict)
    else:
        from_connection, to_connection, pipe_lengths = \
            pipe_connection_lists(network_df, **setting_dict)

    if setting_dict["add_BHE"]:
        n_junctions += 1

    # make a list of where the heat pumps are conncted.
    heat_pump_connections = np.where(network_df['heat_pump_loads'] > 0)[0]

    # calculate the mass flow from the each heat pump
    mass_flow_heat_pumps = flow_from_load_heating(network_df['heat_pump_loads']
                                                  [heat_pump_connections])

    # $$$ we do not use this. 
    total_mass_flow = np.sum(mass_flow_heat_pumps)

    if setting_dict["add_BHE"]:
        if setting_dict["full_network"]:
            ext_grid_loc = n_junctions-2
        else:
            ext_grid_loc = n_junctions - 1
    else:
        ext_grid_loc = 0

topo_file
{
    "fluid_density": 965.0,
    "fluid_heat_capacity": 4450.0,
    "fluid_viscosity": 0.005,
    "init_junction_pressure": 0,
    "init_junction_temp": 293.15,
    "init_source_pressure": 0,
    "init_source_temp": 293.15,
    "pipe_roughness_mm": 0.0001,
    "friction_model": "Swamee-Jain"
}
setting_dict = {
    "COP_heating": 3.0,
    "pipe_material": "PE 100",
    "temperature_drop_at_pump": 3.0,
    "add_BHE": True,
    "BHE_length": 150,
    "max_pressure_drop_per_m": 100,
    "full_network": False,  # toggle to save time, if false only half of
                            # the network is calculated due to symmetry
    "pipe_SDR": 17,
    }


# %%
# Test = optimal_pipe_diameter(0.20, 300, 0.2, 2, 1000, max_iter=100)
# print(Test)
# exit()
net = pp.create_empty_network(fluid='water')
pp.create_junctions(net, n_junctions, pn_bar=1, tfluid_k=293.15)
# should use pp.create_pipes_from_parameters(net, from_junctions, to_junctions, length_km, diameter_m, k_mm=0.2,
                                #  loss_coefficient=0, sections=1, alpha_w_per_m2k=0., text_k=293,
                                #  qext_w=0., name=None, index=None, geodata=None, in_service=True,
                                #  type="pipe", **kwargs)

pp.create_pipes(net, from_junctions=from_connection,
                to_junctions=to_connection, length_km=pipe_lengths,
                std_type=pipe_name)

pp.create_sources(
    net, junctions=heat_pump_connections,
    mdot_kg_per_s=mass_flow_heat_pumps,
    scaling=1.,
    name=None,
    index=None,
    in_service=True,
    type='source',
    )

pp.create_ext_grid(net, ext_grid_loc, p_bar=1, type='p')
# pp.create_sink(net, 0, Mass_flow_sink, scaling=1., name=None, index=None,
#                 in_service=True, type='sink')



# %% 
# exit()
pp.pipeflow(net, mode="hydraulics")
print('test')
print(net["_options"])
print(net.pipe)
net.res_pipe['p_loss_pa_per_m'] = np.abs(((net.res_pipe['p_from_bar']
                                           - net.res_pipe['p_to_bar'])
                                           / net.pipe['length_km']/1000)*10e5)
print("net.res_pipe")
print(net.res_pipe)
print("net.res_junction")
print(net.res_junction)

# %%



"""
A collection of all the functions I have made when working with 
pandapipes
"""




def load_pandapipes_pipe_data_to_dataframe(pipe_material="PE 100", pipe_SDR=False, **kwargs):
    # create a dictionary which only contains the specified pipe types
    pipe_file = os.path.join(pp.pp_dir, "std_types", "library", "Pipe.csv")
    pipe_data = pp.std_types.get_data(pipe_file, "pipe").T
    pipe_types = np.unique(pipe_data["material"])
    if pipe_material not in pipe_types:
        Errormsg = (f"Invalid pipe material, {pipe_material} given.\n"
                    f"Please select one of the following types"
                    f" {", ".join(pipe_types)}")
        raise ValueError(Errormsg)
    else:
        pipe_data = pipe_data[pipe_data["material"] == pipe_material]
    if pipe_SDR is not False:
        selected_pipes = pipe_data[pipe_data.index.str.contains("SDR_17")]
    return selected_pipes


def pipe_name_from_outer_diameter(df, requested_diameter):
    """
    #### Notes to self
    Should consider the SDR value if we want a certain presssure
    capacity
    ####

    Finds the pipe diameter closest to the input diameter, treating the
    input as the mininum acceptable diameter, i.e., the output will not
    be less than the input. The input is the outer diameter of the pipe

    input
    :param df: pandas DataFrame of the different pipe types, each pipe
               is its own row.
    :type dict: pandas DataFrame
    :param requested_diameter: the desired internal pipe diameter [mm]
    :type requested_diameter: float

    output
    :param output_diameter: the name of the pipe which have the closest
                            but larger diameter than the input diameter
                            [~]
    :type output_diameter: str
    """
    # requested_diameter *= 1000  # convert from m to mm
    temp_df = df.copy()

    temp_df = temp_df[temp_df["outer_diameter_mm"] >= requested_diameter]
    pipe_name = temp_df.index[
        np.argmin(temp_df["outer_diameter_mm"])
        ]
    # this might be faster less comparisons,
    # Current_smallest_name = ""
    # Current_smallest_diff = 1000
    # for pipe_diameter_diff in zip((df["outer_diameter_mm"]-requested_diameter),
    # df.index):
    #     if pipe_diameter_diff < 0:
    #         continue
    #     if Current_smallest_diff > pipe_diameter_diff:
    #         Current_smallest_diff = pipe_diameter_diff
    #         Current_smallest_name
    # if len(np.where)
    return pipe_name




def gen_pipe_connection_list_from_data_frame(df, add_BHE=False, BHE_length=100, **kwargs):
    from_connection = []
    to_connection = []
    pipe_lengths = []
    pipe_scale = 0.001  # from m to km
    if add_BHE:
        size_df = df.shape[0]
        BHE_from_connection = []
        BHE_to_connection = []
        BHE_pipe_lengths = []
    for n, pipe_connections in enumerate(df["pipe_connections"]):
        for m, pipe_connection in enumerate(pipe_connections):
            if pipe_connection == -1:
                continue
            else:
                from_connection.append(n)
                to_connection.append(pipe_connection)
                pipe_lengths.append(df["pipe_length"][n][m]*pipe_scale)
        if ((add_BHE is True) and (df["BHE_location(ext_net)"][n] == True)):
            BHE_from_connection.append(n)
            BHE_to_connection.append(size_df)
            BHE_pipe_lengths.append(BHE_length*pipe_scale)
            size_df += 1
    if add_BHE:
        # inverted flow so far everything is connected from the
        # sink to the sources, but to achieve this we need to say
        # from the bottom of the BHE to the surface
        from_connection += BHE_to_connection
        to_connection += BHE_from_connection
        pipe_lengths += BHE_pipe_lengths
    # pipe_lengths /=1000 # pandapipes works in km for some reason
    return from_connection, to_connection, pipe_lengths


def gen_pipe_connection_list_from_lists(connect_list, **kwargs):
    from_connection = []
    to_connection = []
    for n, pipe_connections in enumerate(connect_list):
        for m, pipe_connection in enumerate(pipe_connections):
            if pipe_connection == -1:
                continue
            else:
                from_connection.append(n)
                to_connection.append(pipe_connection)
    return from_connection, to_connection

def gen_pipe_connection_list_dublicate_from_data_frame(df, add_BHE=False, BHE_length=100,
                                       **kwargs):
    from_connection = []
    to_connection = []
    pipe_lengths = []
    pipe_scale = 0.001  # from m to km
    size_df = df.shape[0]
    if add_BHE:
        count_BHE_connections = 0
        BHE_from_connection = []
        BHE_to_connection = []
        BHE_pipe_lengths = []
    for n, pipe_connections in enumerate(df["pipe_connections"]):
        for m, pipe_connection in enumerate(pipe_connections):
            if pipe_connection == -1:
                continue
            else:
                from_connection.extend([n, pipe_connection + size_df])
                to_connection.extend([pipe_connection, n + size_df])
                pipe_lengths.extend([df["pipe_length"][n][m]*pipe_scale,]*2)
        if ((add_BHE is True) and (df["BHE_location(ext_net)"][n] == True)):
            BHE_from_connection.extend([2*size_df + count_BHE_connections,
                                        n + size_df])
            BHE_to_connection.extend([n, 2*size_df + count_BHE_connections])
            count_BHE_connections += 1
            BHE_pipe_lengths.extend([BHE_length*pipe_scale,]*2)
    if add_BHE:
        from_connection += BHE_from_connection
        to_connection += BHE_to_connection
        pipe_lengths += BHE_pipe_lengths
    # pipe_lengths /=1000 # pandapipes works in km for some reason
    return from_connection, to_connection, pipe_lengths

## Not done ##
def set_up_net(net, df, full_network=False,**kwargs):
    #### Currently working on this one
    n_junctions = df.shape[0]
    if full_network:
        n_junctions = n_junctions*2
    n_junctions += 1
    heat_pump_connections = np.where(df["heat_pump_loads"] > 0)[0]
    mass_flow_heat_pumps = flow_from_load_heating(
        df["heat_pump_loads"][heat_pump_connections])
    total_mass_flow = np.sum(mass_flow_heat_pumps)
    ## I need initial guesses for the pipe diameter

    pp.create_junctions(net, n_junctions, pn_bar=1, tfluid_k=293.15)
    pp.create_pipes(net, from_junctions=from_connection,
                to_junctions=to_connection, length_km=pipe_lengths,
                std_type=pipe_name)
    pp.create_sources(
        net, junctions=heat_pump_connections,
        mdot_kg_per_s=mass_flow_heat_pumps,
        scaling=1.,
        name=None,
        index=None,
        in_service=True,
        type="source",
    )

    return None


# Not done ##def Initial_optimal_pipe_diameter(net, pressure_loss_pa_per_m, k_mm):
def optimal_pipe_diameter_old_notworking(guess_pipe_diameter, pressure_loss_pa_per_m, k_mm, mass_flow_kg_per_s, density_fluid, max_iter=100):
    front_factor = (2 * mass_flow_kg_per_s**2 * np.log(10)**2) / (np.pi * density_fluid) 
    pipe_diameter = guess_pipe_diameter
    niter = 0
    converged = False
    k_mm /= 1000
    gamma = 0.01
    while not converged and niter < max_iter:
        logarithm_factor = np.log(k_mm / (3.17*pipe_diameter))
        f = front_factor / (np.power(pipe_diameter, 5) * logarithm_factor**2) \
            + pressure_loss_pa_per_m
        df_ddiameter = front_factor / np.power(pipe_diameter, 6) / \
                       np.power(logarithm_factor, 3) * (2 - 5*logarithm_factor)
        diameter_step = f/df_ddiameter
        # diameter_old = pipe_diameter
        pipe_diameter -= diameter_step * gamma
        print(pipe_diameter, diameter_step)
        # dx = np.abs(lambda_cb - lambda_cb_old) * dummy
        # error_lambda.append(linalg.norm(dx) / (len(dx)))

        if np.abs(diameter_step) <= 1e-4:
            converged = True

        niter += 1
    return pipe_diameter