import json

from thermonet.dimensioning.main import run_full_dimensioning_single_combined_input
from thermonet.dimensioning.thermonet_classes import FullDimension
from pathlib import Path
import glob

# TODO fix test
def test_regression_dimensioning():

    regression_data_folder = Path(__file__).parent.joinpath("regression_test_data/*.json")

    regression_files = glob.glob(str(regression_data_folder))

    # Loop over all configurations in the regression database
    for regression_file in regression_files:

        with open(regression_file, "r") as f:
            regression_config_dict = json.load(f)
            config = FullDimension.from_dict(regression_config_dict)
            regression_result = FullDimension.from_dict(regression_config_dict)

        results = run_full_dimensioning_single_combined_input("test", config)

        # Generate new file
        #file = Path(__file__).parent.joinpath(f"regression_test_data/{key}.json")
        #with open(file, 'w') as f:
        #     f.write(results.to_json())

        # Check if output as expected
        assert results == regression_result, f"File {regression_file} created unexpected output."