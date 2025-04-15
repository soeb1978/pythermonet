import sys
sys.path.insert(0, r"C:\Users\soeb\Documents\GitHub\pythermonet\src")
from pythermonet.dimensioning.thermonet_classes import Brine, Thermonet, HHEconfig, AggregatedLoad
from pythermonet.dimensioning.dimensioning_functions import print_project_id, read_dimensioned_topology, read_aggregated_load, run_sourcedimensioning, print_source_dimensions
import pythermonet.dimensioning.thermonet_classes
print(pythermonet.dimensioning.thermonet_classes.__file__)

if __name__ == '__main__':
    # Inputs

    # Project ID
    PID = 'Silkeborg'   # Project name

    # Input files
    agg_load_file = './data/sites/Silkeborg_aggregated_load_Heat_and_Cool.dat'       # Input file for specifying aggregated load for heating
    TOPO_file = './data/sites/Silkeborg_TOPO_dimensioned.dat'               # Input file containing topology information

    # User specified input

    # Set brine properties
    brine = Brine(rho=965, c=4450, mu=5e-3, l=0.45)

    # Initialise thermonet object
    net = Thermonet(D_gridpipes=0.3, l_p=0.4, l_s_H=1.25, l_s_C=1.25, rhoc_s=2.5e6, z_grid=1.2, T0 = 9.03, A = 7.90)
    # Read remaining data from user specified file
    net, pipeGroupNames = read_dimensioned_topology(net, brine, TOPO_file)  

    # Initialise aggregated load object
    import os
    print("Running from:", os.path.abspath(__file__))

    aggLoad = AggregatedLoad(Ti_H = -3, Ti_C = 20, f_peak_H=1, t_peak_H=10, f_peak_C=1, t_peak_C=10)
    # Read remaining data from user specified file
    aggLoad = read_aggregated_load(aggLoad, brine, agg_load_file)           

    # Heat source (either BHE or HHE)
    source_config = HHEconfig(N_HHE=10, d=0.04, SDR=17, D=1.5)
    # Dimensioning of sources - reuslts printed to console
    source_config = run_sourcedimensioning(brine, net, aggLoad, source_config)
    print_project_id(PID)
    print_source_dimensions(source_config,net)