from thermonet.dimensioning.thermonet_classes import Brine, Thermonet, Heatpump, HHEconfig
from thermonet.dimensioning.dimensioning_functions import read_heatpumpdata, read_topology
from thermonet.dimensioning.main import run_full_dimensioning
import pandas as pd

if __name__ == '__main__':
    # Inputs

    # Project ID
    PID='Energiakademiet, Samsø',                                     # Project name

    # Input files
    HP_file = './data/sites/Samso_HPSC.dat';                 # Input file containing heat pump information
    TOPO_file = './data/sites/Samso_TOPO.dat';               # Input file containing topology information
    pipe_file = '../data/equipment/PIPES.dat';                   # Input file with available pipe diameters

    d_pipes = pd.read_csv(pipe_file, sep = '\t');                       # Open file with available pipe outer diameters (mm). This file can be expanded with additional pipes and used directly.
    d_pipes = d_pipes.values;                                               # Get numerical values from pipes excluding the headers
    d_pipes = d_pipes/1000;                                                 # Convert d_pipes from mm to m

    # User specified input

    # Brine - with default parameters
    brine = Brine(rho=965, c=4450, mu=5e-3, l=0.45);

    # Initialise thermonet object - with default parameters
    net = Thermonet(D_gridpipes=0.3, dpdL_t=90, l_p=0.4, l_s_H=1.25, l_s_C=1.25, rhoc_s=2.5e6, z_grid=1.2);
    net, pipeGroupNames = read_topology(net, TOPO_file); # Read remaining data from user specified file

    # Initialise HP object - with default parameters
    hp = Heatpump(Ti_H=-3, Ti_C=20, SF=1);
    hp = read_heatpumpdata(hp, HP_file); # Read remaining data from user specified file

    # Heat source (either BHE or HHE) - with default parameters
    source_config = HHEconfig(N_HHE=6, d=0.04, SDR=17, D=1.5)
    # source_config = BHEconfig(r_b=0.152/2, r_p=0.02, SDR=11, l_ss=2.36, rhoc_ss=2.65e6, l_g=1.75, rhoc_g=3e6, D_pipes=0.015, NX=1, D_x=15, NY=6, D_y=15);

    run_full_dimensioning(PID, d_pipes, brine, net, hp, pipeGroupNames, source_config)