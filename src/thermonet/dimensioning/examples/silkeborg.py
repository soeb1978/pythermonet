import pandas as pd
from thermonet.dimensioning.thermonet_classes import Brine, Thermonet, Heatpump, HHEconfig, BHEconfig, aggregatedLoad
from thermonet.dimensioning.dimensioning_functions import read_heatpumpdata, read_topology, read_dimensioned_topology, read_aggregated_load, run_sourcedimensioning, print_source_dimensions
from thermonet.dimensioning.main import run_full_dimensioning

if __name__ == '__main__':
    # Inputs

    # Project ID
    PID = 'Silkeborg';                                     # Project name

    # Input files
    HP_file = './data/sites/Silkeborg_HPSC.dat';                 # Input file containing heat pump information
    # HP_file = './data/sites/Silkeborg_HPS_Heat.dat';             # Input file - only heating
    # HP_file = './data/sites/Silkeborg_HPS_Heat_COMPARE.dat';       # Input file - only heating with same yearly net load as Silkeborg_HPSC.dat for comparison


    TOPO_file = './data/sites/Silkeborg_TOPO.dat';               # Input file containing topology information
    pipe_file = '../data/equipment/PIPES.dat';                   # Input file with available pipe diameters

    d_pipes = pd.read_csv(pipe_file,sep='\t');  # Open file with available pipe outer diameters (mm). This file can be expanded with additional pipes and used directly.
    d_pipes = d_pipes.values;  # Get numerical values from pipes excluding the headers
    d_pipes = d_pipes / 1000;  # Convert d_pipes from mm to m

    # User specified input

    # Brine - with default parameters
    brine = Brine(rho=965, c=4450, mu=5e-3, l=0.45);

    # Initialise thermonet object
    net = Thermonet(D_gridpipes=0.3, l_p=0.4, l_s_H=1.25, l_s_C=1.25, rhoc_s=2.5e6, z_grid=1.2, T0 = 9.028258373009810, A = 7.900272987633280);
    net, pipeGroupNames = read_topology(net, TOPO_file); # Read remaining data from user specified file

    # Initialise heat pump object
    hp = Heatpump(Ti_H=-3, Ti_C=20, SF=1, t_peak=4);
    hp = read_heatpumpdata(hp, HP_file); # Read remaining data from user specified file

    # Heat source (either BHE or HHE)
    # source_config = HHEconfig(N_HHE=6, d=0.04, SDR=17, D=1.5)
    
    source_config = BHEconfig(q_geo = 0.0185, r_b=0.152/2, r_p=0.02, SDR=11, l_ss=2.36, rhoc_ss=2.65e6, l_g=1.75, rhoc_g=3e6, D_pipes=0.015, NX=1, D_x=15, NY=6, D_y=15);

    # Fuld beregning dimensionerer både rør og varmekilde, og udskriver begge til prompt
    run_full_dimensioning(PID, d_pipes, brine, net, hp, pipeGroupNames, source_config)
    
    
    # KART - EXPERIMENTAL. Slet alle variable og genkør kun kildedimensionering
    del net, hp, source_config, TOPO_file
    print('')
    print('Experimental - test aggergated load for source dimensioning')
    print('')
    
    TOPO_file = './data/sites/Silkeborg_TOPO_dimensioneret.dat';               # Input file containing topology information
    
    # Initialise thermonet object
    net = Thermonet(D_gridpipes=0.3, l_p=0.4, l_s_H=1.25, l_s_C=1.25, rhoc_s=2.5e6, z_grid=1.2, T0 = 9.028258373009810, A = 7.900272987633280);
    net, pipeGroupNames = read_dimensioned_topology(net, brine, TOPO_file); # Read remaining data from user specified file


    agg_load_file = './data/sites/Silkeborg_aggregated_load.dat';             # Input file for specifying only aggregated load for heating and cooling. Same totale load as in Silkeborg_HPSC.dat
    # agg_load_file = './data/sites/Silkeborg_aggregated_load_HEAT.dat';        # Input file - only heating
    # agg_load_file = './data/sites/Silkeborg_aggregated_load_HEAT_COMPARE.dat';  # Input file - only heating with same yearly net load as in Silkeborg_aggregated_load.dat for comparsion

    source_config = BHEconfig(q_geo = 0.0185, r_b=0.152/2, r_p=0.02, SDR=11, l_ss=2.36, rhoc_ss=2.65e6, l_g=1.75, rhoc_g=3e6, D_pipes=0.015, NX=1, D_x=15, NY=6, D_y=15);


    aggLoad = aggregatedLoad(Ti_H = -3, Ti_C = 20, SF=1, t_peak=4)
    aggLoad = read_aggregated_load(aggLoad, brine, agg_load_file)
    source_config = run_sourcedimensioning(brine, net, aggLoad, source_config);
    print_source_dimensions(source_config,net)

