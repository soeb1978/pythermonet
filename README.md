# Thermonet Dimensioning Tool

More text to be added!

## Code Nomenclature
Variables should consistently be named using a combination of variable names and suffixes below. For instance the density of the fluid (brine) is rho_f. An additional suffix is used to discriminate heating from cooling modes e.g. G_BHE_H and G_BHE_C are the G functions for the BHEs in heating and cooling mode respectively.
### Variables
| Var | Description              |Unit|
|-----|--------------------------|-----|
| c   | specific heat            |[J/kg K]|
| d   | 	outer diameter of x	    |[m]|
|D| 	distance                |		[m]|
|dpdL| 	pressure drop per meter	 |[Pa/m]|
|l	|thermal conductivity| [W/m K]|
|L	|length e.g. of pipes|	[m]|
|mu	|dynamic viscosity|	[Pa s]|
|N|	Number of…|		[-]|
|nu|	kinematic viscosity|	[m2/s]|
|P|	thermal ground load|	[W]|
|Q|	flow rate		|[m3/s]|
|r|	outer radius of x	|[m]|
|rho	|density 		|[kg/m3]|
|rhoc|	volumetric heat capacity|	[J/m3 K]|
|ri	|inner radius		|[m]|
|R|	thermal resistance	|[m K/W]|
|s_BHE|	Shank spacing	|	[m] |	Could use s_HHE for distance between pipes
|SFRP|	?? ||
|T|	temperature		|[K] or [°C]|
|v	|flow velocity		|[m/s]|
|z|	depth coordinate	|[m]|

### Suffixes
|b	|borehole|
|---|-------|
|BHE|	Borehole heat exchanger|
|f	|fluid (brine)|
|g|	grout|
|HHE|	Horisontal heat exchanger|
|p|	pipe|
|s|	soil|
|ss|	also soil  - but average over BHEs|
|t|	target|
