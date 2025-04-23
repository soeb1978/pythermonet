# Standard library imports
import importlib.resources

# Third-party imports
import pandas as pd


def load_pipe_catalogue():
    """
    Loads and returns the pipe catalogue from the bundled "PIPES.dat" file.

    This function reads a line-delimited data file ("PIPES.dat") located in the
    "pythermonet.data.equipment" package and returns it as a pandas DataFrame.

    Returns:
    :param : A DataFrame containing pipe specification data, such as nominal
        widths, standard dimension ratios, and reference mass flows.
    :type  : pandas.DataFrame
    """
    with importlib.resources.files("pythermonet.data.equipment")\
            .joinpath("PIPES.dat").open("r") as f:

        return pd.read_csv(f, sep="\t")
