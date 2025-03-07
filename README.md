# Thermonet Dimensioning Tool

The tool can be used to dimension the diameters of gridpipes in a thermonet as well as the length of Borehole Heat Exchangers (BHES) or Horizontal Heat Exchangers (HHEs) supplying the grid.

## Input
Inputs are supplied by a combination of plain text data files and input parameters for class objects (e.g. brine, BHEs etc.). Please refer to the accompanying example files. 

Input parameters for class objects are defined below:

### Brine
| Var | Description              |Unit|
|-----|--------------------------|-----|
|rho	|density 		|[kg/m3]|
| c   | specific heat            |[J/kg K]|
|mu	|dynamic viscosity|	[Pa s]|
|l	|thermal conductivity| [W/m K]|


### Thermonet
| Var | Description              |Unit|
|-----|--------------------------|-----|
|D_gridpipes| 	distance between forward/return pipe centers                |		[m]|
|l_p	|pipe thermal conductivity| [W/m K]|
|l_s	|soil thermal conductivity| [W/m K]|
|rhoc_s|	soil volumetric heat capacity|	[J/m3 K]|
|z_grid|	burial depth of grid pipes	|[m]|
|T0|	yearly average surface temperature		|[°C]|
|A|	amplitude of sinusoidal yearly temperature variation		|[°C]|

### Heatpump and aggregated load
| Var | Description              |Unit|
|-----|--------------------------|-----|
|Ti|	inlet temperature		| [°C]|
|f_peak|		Fraction of peak load supplied by heatpumps	|[-]|
|t_peak|	duration of peak load 		|[h]|



### Borehole Heat Exchangers (BHE)
| Var | Description              |Unit|
|-----|--------------------------|-----|
|q_geo| geothermal heat flux|[W/m2]|
|r_b| borehole radius | [m] |
|r_p| U-pipe outer radius| [m]|
|SDR| U-pipe SDR value |[-]|
|l_ss| average soil thermal conductivity along BHE|[W/m K]|
|rhoc_ss| average soil volumetric heat capacity along BHE|[J/m3 K]|
|l_g| grout thermal conductivity | [W/m K] |
|rhoc_g| grout volumetric heat capacity | [J/m3 K] |
|D_pipes| U-pipe wall to wall distance| [m] |
|NX| Number of BHEs in x-direction|[-]|
|D_x| Spacing between BHEs in x-direction| [m] |
|NY|Number of BHEs in y-direction|[-]|
|D_y|Spacing between BHEs in x-direction| [m]|
|gFuncMethod| method for evaluating g-function| N/A |

Note the gFuncMethod is a string input, and may currently be set to 'ICS' for Infinite Cylinder Source or 'PYG' for the pygfunction implementation of the finite line source. If no input is given the default is 'ICS'.


### Horizontal Heat Exchangers (HHE)
| Var | Description              |Unit|
|-----|--------------------------|-----|
|N_HHE | Number of HE loops| [-]|
|d | outer diameter of HE pipes| [m] |
|SDR | SDR values for HE pipes| [-]|
|D| pipe segment spacing| [m] |

## Getting started
Download the latest version of the code from github. Navigate to the base folder (the folder that contains the file setup.py) and install
```
pip install -e .
```
You should now be able to run the examples provided in \src\thermonet\dimensioning\examples