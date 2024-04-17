# %% 
from thermonet.dimensioning.thermonet_classes import Brine, Thermonet, BHEconfig, aggregatedLoad
from thermonet.dimensioning.dimensioning_functions import print_project_id, read_dimensioned_topology, update_net_with_pandapipes_flow, read_aggregated_load_pandapipes, read_aggregated_load, run_sourcedimensioning, print_source_dimensions

# %%

if __name__ == '__main__':
    # Inputs

    # Project ID
    PID = 'Silkeborg'   # Project name

    # Input files
    HP_file = './data/sites/Silkeborg_HP_Heat_and_Cool_pandapipes.dat'
    agg_load_file = './data/sites/Silkeborg_aggregated_load_Heat_and_Cool.dat'  # Input file for specifying aggregated load for heating and cooling.
    TOPO_file = './data/sites/Silkeborg_TOPO_pandapipes.dat'                    # Input file containing topology information

# %%
    # User specified input

    # flow_calculation: str = 'tree_structure' # Toggle for whether the flow have been calculated in the tree structure model or if pandapipes should be used: 'tree_structure' the flow has been calculated according to the tree structure, 'pandapipes' is the net has been dimensioned and the flow need to be calculated.
    flow_calculation = "pandapipes"
    # Set brine properties
    brine = Brine(rho=965, c=4450, mu=5e-3, l=0.45)

    # Initialise thermonet object
    net = Thermonet(D_gridpipes=0.3, l_p=0.4, l_s_H=1.25, l_s_C=1.25, rhoc_s=2.5e6, z_grid=1.2, T0 = 9.03, A = 7.90, friction_model_pp='swamee-jain')
    # Read remaining data from user specified file
    if flow_calculation == "pandapipes":
        net = update_net_with_pandapipes_flow(net, brine, TOPO_file, HP_file)
    else:
        net, _ = read_dimensioned_topology(net, brine, TOPO_file)

    # Initialise aggregated load object
    aggLoad = aggregatedLoad(Ti_H=-3, Ti_C=20, f_peak=1, t_peak=4)
    # Read remaining data from user specified file
    # $$$ I need it here as well yes I do as the aggload is not aggregated. 
    if flow_calculation == "pandapipes":
        aggLoad = read_aggregated_load_pandapipes(aggLoad, brine, HP_file)
    else:
        aggLoad = read_aggregated_load(aggLoad, brine, agg_load_file)

    # Heat source (either BHE or HHE)
    source_config = BHEconfig(q_geo=0.0185, r_b=0.152/2, r_p=0.02, SDR=11, l_ss=2.36, rhoc_ss=2.65e6, l_g=1.75, rhoc_g=3e6, D_pipes=0.015, NX=1, D_x=15, NY=6, D_y=15, gFuncMethod='ICS')

    # Dimensioning of sources - results printed to console
    source_config = run_sourcedimensioning(brine, net, aggLoad, source_config)
    print_project_id(PID)
    print_source_dimensions(source_config, net)
