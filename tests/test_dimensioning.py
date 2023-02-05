from .regression_test_data.regression_test_configurations import test_configurations
from thermonet.dimensioning.dimensioning import run_dimensioning
from pathlib import Path

def test_dimensioning():

    # Loop over all configurations in the regression database
    for key, c in test_configurations.items():

        results = run_dimensioning(c[0], print_computation_time=False)

        # Generate new file
        #file = Path(__file__).parent.joinpath("regression_test_data/result_basic_configuration.json")
        #with open(file, 'w') as f:
        #     f.write(results.to_json())

        # Check if output as expected
        assert results == c[1], f"Configuration {key} created unexpected output."