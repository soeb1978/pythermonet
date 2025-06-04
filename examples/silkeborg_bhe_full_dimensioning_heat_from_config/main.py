from pathlib import Path

from pythermonet.data.equipment.pipes import load_pipe_catalogue
from pythermonet.core.dimensioning_functions import (
    read_heatpumpdata,
    read_topology,
)
from pythermonet.core.main import run_full_dimensioning
from pythermonet.io.config import load_project_config
from pythermonet.domain import BHEConfig, Brine, HeatPump, Thermonet


def main() -> None:
    """
    Runs the full dimensioning for Silkeborg example with bore hole heat
    exchangers (BHE) and heating loads (no cooling loads).
    The full dimensioning includes both sources sizes and pipe diameters
    of the distribution network.
    This example reads the model parameters from a config.json file

    Required input files:
        - project_config.json
        - data/silkeborg_heat_pump_heat.dat
        - data/silkeborg_topology.dat
    """
    # Define the parent directory (current script directory)
    project_dir = Path(__file__).parent.resolve()
    project_data_dir = project_dir / "data"
    project_config_file = project_data_dir / "project_config.json"

    # load the config file
    project_config = load_project_config(project_config_file)

    # Project ID
    project_id = project_config["project_id"]

    # Get the heat pump and topology files
    heat_pump_file = (
        project_data_dir / project_config["files"]["heat_pump_file"]
        )
    topology_file = (
        project_data_dir / project_config["files"]["topology_file"]
        )

    # Open file with available pipe outer diameters (mm). This file can be
    # expanded with additional pipes and used directly.
    # Convert d_pipes from mm to m
    d_pipes = load_pipe_catalogue().values / 1000

    # Set brine properties
    brine = Brine.from_dict(project_config["brine"])

    # Initialise thermonet object
    net = Thermonet.from_dict(project_config["thermonet"])
    # Read remaining data from user specified file
    net, pipe_group_names = read_topology(net, topology_file)

    # Initialise heat pump object
    hp = HeatPump.from_dict(project_config["heat_pump"])
    # Read remaining data from user specified file
    hp = read_heatpumpdata(hp, heat_pump_file)

    # Heat source (either BHE or HHE)
    source_config = BHEConfig.from_dict(project_config["bhe_config"])

    # Full dimensioning of pipes and sources - results printed to console
    run_full_dimensioning(
        project_id,
        d_pipes,
        brine,
        net,
        hp,
        pipe_group_names,
        source_config
        )


if __name__ == "__main__":
    main()
