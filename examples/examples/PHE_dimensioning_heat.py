from pythermonet.dimensioning.thermonet_classes import PHEconfig
from pythermonet.dimensioning.dimensioning_functions import read_PHEdata

if __name__ == '__main__':
    Rc_file = '../data/concrete_resistance_parameters.txt'
    Gc_file = '../data/concrete_Gfunction_parameters.txt'
    coord_file = './data/sites/PHE_coordinates.txt'
    
    source_config = PHEconfig(S=0.3, n=4, do=0.02, di=0.016, l_c=3.05, l_ss=2.21, rhoc_ss=2.47e6)
    
    source_config = read_PHEdata(source_config, Rc_file, Gc_file, coord_file)