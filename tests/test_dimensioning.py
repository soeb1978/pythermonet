from .regression_test_data.regression_test_configurations import test_configurations
from thermonet.dimensioning.dimensioning import run_dimensioning

def test_dimensioning(capsys):

    # Loop over all configurations in the regression database
    for key, c in test_configurations.items():

        run_dimensioning(c[0], print_computation_time=False)

        captured = capsys.readouterr()

        # Generate new file
        #with open('result_output_dimensioning.txt', 'w') as f:
        #     f.write(captured.out)

        # Check if output as expected
        assert captured.out == c[1], f"Configuration {key} created unexpected output."