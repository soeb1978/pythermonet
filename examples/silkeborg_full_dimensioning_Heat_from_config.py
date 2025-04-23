import json
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
if __name__ == "__main__":
    # Setup path for the config file
    script_directory = Path(__file__).parent
    site_data_dir = script_directory / "data" / "sites"
    project_config_file = (
        site_data_dir / "silkeborg_full_dimensioning_heat_config.json"
        )
    # load the config file
    with open(project_config_file) as f:
        project_config = json.load(f)

    # Project ID
    project_id = project_config["project_id"]

    # Get the heat pump and topology files
    heat_pump_file = site_data_dir / project_config["heat_pump_file"]
    topology_file = site_data_dir / project_config["topology_file"]

    # Open file with available pipe outer diameters (mm). This file can be
    # expanded with additional pipes and used directly.
    # Convert d_pipes from mm to m
    d_pipes = load_pipe_catalogue().values / 1000

    # Set brine properties
    brine = Brine.from_dict(project_config["brine"])

    # Initialise thermonet object
    net = Thermonet.from_dict(project_config["thermonet"])
    # Read remaining data from user specified file
    net, pipeGroupNames = read_topology(net, topology_file)

    # Initialise heat pump object
    hp = HeatPump.from_dict(project_config["heat_pump"])
    # Read remaining data from user specified file
    hp = read_heatpumpdata(hp, heat_pump_file)

    # Heat source (either BHE or HHE)
    source_config = BHEConfig.from_dict(project_config["bhe_config"])

    # Full dimensioning of pipes and sources - results printed to console
    run_full_dimensioning(
        project_id, d_pipes, brine, net, hp, pipeGroupNames, source_config
        )
