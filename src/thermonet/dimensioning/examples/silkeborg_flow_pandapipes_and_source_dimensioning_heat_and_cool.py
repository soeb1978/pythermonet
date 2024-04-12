# %%
from thermonet.dimensioning.dimensioning_functions_pandas import \
    wrapper_pythermonet_source_dimensioning_from_csv
from thermonet.dimensioning.pipe_flow_functions import \
    wrapper_pandapipes_flow_from_csv

##### inputs #####

# Project ID
project_ID = 'Silkeborg'
# The paths for the pipe topology
pipe_file = r'data\sites\silkeborg_pandapipes\pipe_list.csv'
# Toggle if the original topology file should be overwritten.
overwrite_pipe_csv = False
# and the path for the new file containing the topology plus flow data
pipe_file_new = r'data\sites\silkeborg_pandapipes\pipe_list_with_flow.csv'

# path for the files containing the heat pumps and the loads
heat_pump_load_file = r'data\sites\silkeborg_pandapipes\heat_pump_list.csv'
heat_pump_setting_file = \
    r'data\sites\silkeborg_pandapipes\heat_pump_settings.json'

# path for the settings file, for fluid properties and frictions model
settings_file = r'data\sites\silkeborg_pandapipes\settings.json'

# path to the file containing the thermonet parameters
thermonet_file = r'data\sites\silkeborg_pandapipes\thermonet_settings.json'

# path to the file containing the BHE settings
BHE_file = r'data\sites\silkeborg_pandapipes\BHE_settings.json'


# %%
##### Automatic from here #####

# I need to include cooling as well. Run the simulation again.
wrapper_pandapipes_flow_from_csv(pipe_file, heat_pump_load_file, settings_file,
                                 overwrite_pipe_csv, pipe_file_new)
# %%
if overwrite_pipe_csv is False:
    pipe_file = pipe_file_new
wrapper_pythermonet_source_dimensioning_from_csv(pipe_file, settings_file,
                                                 thermonet_file, BHE_file,
                                                 heat_pump_load_file,
                                                 heat_pump_setting_file,
                                                 project_ID)
