from pathlib import Path

from pythermonet.core.dimensioning_functions import (
    print_project_id,
    run_sourcedimensioning,
    print_source_dimensions
)
from pythermonet.domain import Brine, Thermonet, HHEConfig, AggregatedLoad
from pythermonet.io import (
    combine_agg_load_user_and_file,
    combine_net_dimensioned_topology,
    read_aggregated_load_tsv,
    read_dimensioned_topology_tsv
)


def main() -> None:
    """
    Runs the source dimensioning for Silkeborg example with horizontal
    heat exchangers (HHE) and heating and cooling loads.
    The source dimensioning requires that the input distribution network
    is dimensioned.

    Note: Only the aggregated heating and cooling loads are required and
    not the loads of the individual heat pump.

    Required input files:
    - data/silkeborg_aggregated_load_heat_cool.dat
    - data/silkeborg_topology_dimensioned.dat
    """
    # Project ID
    project_id = 'Silkeborg'

    # Define the parent directory (current script directory)
    project_dir = Path(__file__).parent.resolve()
    project_data_dir = project_dir / "data"

    # Input files
    agg_load_file = (
        project_data_dir / "silkeborg_aggregated_load_heat_cool.dat"
    )
    topology_file = project_data_dir / "silkeborg_topology_dimensioned.dat"

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
        A=7.90
    )
    # Read remaining data from user specified file
    dimension_topology = read_dimensioned_topology_tsv(topology_file)
    net, _ = combine_net_dimensioned_topology(net, dimension_topology, brine)

    # Initialise aggregated load object
    agg_load = AggregatedLoad(
        Ti_H=-3,
        Ti_C=20,
        f_peak_H=1,
        t_peak_H=10,
        f_peak_C=1,
        t_peak_C=10
    )
    # Read remaining data from user specified file
    agg_load_input = read_aggregated_load_tsv(agg_load_file)
    agg_load = combine_agg_load_user_and_file(agg_load, agg_load_input, brine)

    # Heat source (either BHE or HHE)
    source_config = HHEConfig(
        N_HHE=10,
        d=0.04,
        SDR=17,
        D=1.5
    )
    # Dimensioning of sources - results printed to console
    source_config = run_sourcedimensioning(brine, net, agg_load, source_config)
    print_project_id(project_id)
    print_source_dimensions(source_config, net)


if __name__ == '__main__':
    main()
