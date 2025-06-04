from dataclasses import dataclass

from dataclasses_json import dataclass_json
import numpy as np

@dataclass_json
@dataclass
class Brine:
    # Brine
    rho: float = np.nan  # Brine density (kg/m3), T = 0C. https://www.handymath.com/cgi-bin/isopropanolwghtvoltble5.cgi?submit=Entry
    c: float = np.nan  # Brine specific heat (J/kg/K). 4450 J/kg/K is loosly based on Ignatowicz, M., Mazzotti, W., Acuña, J., Melinder, A., & Palm, B. (2017). Different ethyl alcohol secondary fluids used for GSHP in Europe. Presented at the 12th IEA Heat Pump Conference, Rotterdam, 2017. Retrieved from http://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-215752
    mu: float = np.nan  # Brine dynamic viscosity (Pa*s). Source see above reference.
    l: float = np.nan  # Brine thermal conductivity (W/m/K). https://www.researchgate.net/publication/291350729_Investigation_of_ethanol_based_secondary_fluids_with_denaturing_agents_and_other_additives_used_for_borehole_heat_exchangers


@dataclass_json
@dataclass
class RefactoredBrine:
    """
    Data class containing the thermophysical properties of the brine
    fluid used in the thermal network system.
    """

    density: float = np.nan  # Density [kg/m³]
    heat_capacity: float = np.nan  # Specific heat capacity [J/kg/K]
    dynamic_viscosity: float = np.nan  # Dynamic viscosity [Pa·s]
    thermal_conductivity: float = np.nan  # Thermal conductivity [W/m/K]

    # References:
    # - Density: e.g., Isopropanol-water mixture at T=0°C
    # - Heat capacity: Ignatowicz et al. (2017)
    # - Dynamic viscosity: same as above
    # - Thermal conductivity: ResearchGate, 2017
