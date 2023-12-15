import pandas as pd
from thermonet.dimensioning.thermonet_classes import Brine, Thermonet, Heatpump, BHEconfig, HHEconfig
from thermonet.dimensioning.dimensioning_functions import read_heatpumpdata, read_topology
from thermonet.dimensioning.main import run_full_dimensioning

if __name__ == '__main__':
    # Inputs

    # Project ID
    PID = 'Samsø'                                           # Project name

    # Input files
    HP_file = './data/sites/Samso_21HPS.dat'     # Input file containing heat pump information
    TOPO_file = './data/sites/Samso_21TOPO.dat'               # Input file containing topology information
    pipe_file = '../data/equipment/PIPES.dat'                   # Input file with available pipe diameters

    d_pipes = pd.read_csv(pipe_file,sep='\t')                   # Open file with available pipe outer diameters (mm). This file can be expanded with additional pipes and used directly.
    d_pipes = d_pipes.values                                    # Get numerical values from pipes excluding the headers
    d_pipes = d_pipes / 1000                                    # Convert d_pipes from mm to m

    # User specified input

    # Set brine properties
    brine = Brine(rho=960, c=4250, mu=5e-3, l=0.44)

    # Initialise thermonet object
    net = Thermonet(D_gridpipes=0.3, l_p=0.4, l_s_H=1.8, l_s_C=1.8, rhoc_s=2.6e6, z_grid=1.2, T0 = 8.39, A = 7.90)
    # Read remaining data from user specified file
    net, pipeGroupNames = read_topology(net, TOPO_file) 

    # Initialise heat pump object
    hp = Heatpump(Ti_H=-3, Ti_C=20, f_peak=1, t_peak=4)
    # Read remaining data from user specified file
    hp = read_heatpumpdata(hp, HP_file) 

    # Heat source (either BHE or HHE)    
    source_config = HHEconfig(N_HHE=7, d=0.04, SDR=17, D=1.5);

    # Full dimensioning of pipes and sources - results printed to console
    run_full_dimensioning(PID, d_pipes, brine, net, hp, pipeGroupNames, source_config)