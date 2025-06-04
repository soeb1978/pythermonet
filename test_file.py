# %%
import pandas as pd

aggload_file_path = r"C:\Users\kri\Documents\TestPython\brugerflade\pythermonet\examples\silkeborg_bhe_source_dimensioning_heat_cool\data\silkeborg_aggregated_load_heat_cool.dat"

df = pd.read_csv(aggload_file_path, sep='\t+', engine='python')

# df.columns = [col.strip() for col in df.columns]

row = df.iloc[0]
# %%
import pandas as pd
undimensioned_topology_file = r"C:\Users\kri\Documents\TestPython\brugerflade\pythermonet\examples\silkeborg_bhe_full_dimensioning_heat\data\silkeborg_topology.dat"
df = pd.read_csv(undimensioned_topology_file, sep='\t+', engine='python')
df.columns = [col.strip() for col in df.columns]

# %%
