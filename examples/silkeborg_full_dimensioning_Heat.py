from pathlib import Path
# import sys

from pythermonet.data.equipment.pipes import load_pipe_catalogue
from pythermonet.dimensioning.dimensioning_functions import (
    read_heatpumpdata,
    read_topology,
)
from pythermonet.dimensioning.main import run_full_dimensioning
from pythermonet.models import BHEConfig, Brine, HeatPump, Thermonet


# sys.path.insert(0, r"C:\Users\soeb\Documents\GitHub\pythermonet\src")
if __name__ == '__main__':
    # Project ID
    project_id = 'Silkeborg'

    # Define the parent directory (current script directory)
    script_directory = Path(__file__).parent
    site_data_dir = script_directory / "data" / "sites"
    # Input file containing heat pump information - only heating
    heat_pump_file = site_data_dir / "Silkeborg_HP_Heat.dat"
    # Input file containing topology information
    topology_file = site_data_dir / "Silkeborg_TOPO.dat"

    # Open file with available pipe outer diameters (mm). This file can be
    # expanded with additional pipes and used directly.
    # Convert d_pipes from mm to m
    d_pipes = load_pipe_catalogue().values / 1000

    # User specified input

    # Set brine properties
    brine = Brine(rho=965, c=4450, mu=5e-3, l=0.45)

    # Initialise thermonet object
    net = Thermonet(
        D_gridpipes=0.3,
        l_p=0.4,
        l_s_H=1.25,
        l_s_C=1.25,
        rhoc_s=2.5e6,
        z_grid=1.2,
        T0=9.03,
        A=7.90,
    )
    # Read remaining data from user specified file
    net, pipeGroupNames = read_topology(net, topology_file)

    # Initialise heat pump object
    hp = HeatPump(Ti_H=-3, Ti_C=20, f_peak_H=1, t_peak_H=4)
    # Read remaining data from user specified file
    hp = read_heatpumpdata(hp, heat_pump_file)

    # Heat source (either BHE or HHE)
    source_config = BHEConfig(
        q_geo=0.0185,
        r_b=0.152/2,
        r_p=0.02,
        SDR=11,
        l_ss=2.36,
        rhoc_ss=2.65e6,
        l_g=1.75,
        rhoc_g=3e6,
        D_pipes=0.015,
        NX=1,
        D_x=15,
        NY=6,
        D_y=15,
        gFuncMethod='PYG',
    )

    # Full dimensioning of pipes and sources - results printed to console
    run_full_dimensioning(
        project_id, d_pipes, brine, net, hp, pipeGroupNames, source_config
        )
