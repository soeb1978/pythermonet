from dataclasses import dataclass
from dataclasses_json import dataclass_json
import numpy as np
from typing import List


@dataclass_json
@dataclass
class Brine:
    # Brine
    rho: float = 965.0  # Brine density (kg/m3), T = 0C. https://www.handymath.com/cgi-bin/isopropanolwghtvoltble5.cgi?submit=Entry
    c: float = 4450.0  # Brine specific heat (J/kg/K). 4450 J/kg/K is loosly based on Ignatowicz, M., Mazzotti, W., Acuña, J., Melinder, A., & Palm, B. (2017). Different ethyl alcohol secondary fluids used for GSHP in Europe. Presented at the 12th IEA Heat Pump Conference, Rotterdam, 2017. Retrieved from http://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-215752
    mu: float = 5e-3  # Brine dynamic viscosity (Pa*s). Source see above reference.
    l: float = 0.45  # Brine thermal conductivity (W/m/K). https://www.researchgate.net/publication/291350729_Investigation_of_ethanol_based_secondary_fluids_with_denaturing_agents_and_other_additives_used_for_borehole_heat_exchangers

@dataclass_json
@dataclass
class Thermonet:
    # Thermonet and HHE
    D_gridpipes: float = 0.3    # Distance between forward and return pipe centers (m)
    dpdL_t: float = 90  # Target pressure loss in thermonet (Pa/m). 10# reduction to account for loss in fittings. Source: Oklahoma State University, Closed-loop/ground source heat pump systems. Installation guide., (1988). Interval: 98-298 Pa/m
    l_p: float = 0.4    # Pipe thermal conductivity (W/m/K). https://www.wavin.com/da-dk/catalog/Varme/Jordvarme/PE-80-lige-ror/40mm-jordvarme-PE-80PN6-100m
    l_s_H: float = 1.25  # Soil thermal conductivity thermonet and HHE (W/m/K) Guestimate (0.8-1.2 W/m/K)
    l_s_C: float = 1.25  # Soil thermal conductivity thermonet and HHE (W/m/K) Guestimate (0.8-1.2 W/m/K)
    rhoc_s: float = 2.5e6  # Soil volumetric heat capacity  thermonet and HHE (J/m3/K) OK. Guestimate
    z_grid: float = 1.2  # Burial depth of thermonet and HHE (m)

    # KART tilføjet topologi information fra TOPO_FILE
    SDR: float = np.nan;
    L_traces: float = np.nan;
    N_traces: float = np.nan;
    L_segments: float = np.nan;
    I_PG: float = np.nan; 
    
    d_selectedPipes_H: float = np.nan;
    di_selected_H: float = np.nan;
    Re_selected_H: float = np.nan;
    d_selectedPipes_C: float = np.nan;
    di_selected_C: float = np.nan;
    Re_selected_C: float = np.nan;
    
    
    
    # KART: tag stilling til om det er nødvendigt at beholde dem.
    # R_H: float = np.nan;
    # R_C: float = np.nan;


@dataclass_json
@dataclass
class Heatpump:
    # Heat pump
    Ti_H: float = -3  # Design temperature for inlet (C) OK. Stress test conditions. Legislation stipulates Thi > -4C. Auxillary heater must be considered.
    Ti_C: float = 20  # Design temperature for inlet (C) OK. Stress test conditions. Legislation stipulates Thi > -4C. Auxillary heater must be considered.
    SF: float = 1  # Ratio of peak heating demand to be covered by the heat pump [0-1]. If SF = 0.8 then the heat pump delivers 80% of the peak heating load. The deficit is then supplied by an auxilliary heating device

    # Heating mode parameters
    P_y_H: float = np.nan;
    P_m_H: float = np.nan;
    P_d_H: float = np.nan;
    COP_y_H: float = np.nan;
    COP_m_H: float = np.nan;
    COP_d_H: float = np.nan;
    dT_H: float = np.nan;
    Qdim_H: float = np.nan;
    P_s_H: float = np.nan;
    
    # Cooling mode parameters
    P_y_C: float = np.nan;
    P_m_C: float = np.nan;
    P_d_C: float = np.nan;
    EER: float = np.nan;
    dT_C: float = np.nan;
    Qdim_C: float = np.nan;
    P_s_C: float = np.nan;

@dataclass_json
@dataclass
class aggregatedLoad:
    
    # Heat pump
    Ti_H: float = np.nan  # Design temperature for inlet (C) OK. Stress test conditions. Legislation stipulates Thi > -4C. Auxillary heater must be considered.
    Ti_C: float = np.nan  # Design temperature for inlet (C) OK. Stress test conditions. Legislation stipulates Thi > -4C. Auxillary heater must be considered.
    SF: float = np.nan  # Ratio of peak heating demand to be covered by the heat pump [0-1]. If SF = 0.8 then the heat pump delivers 80% of the peak heating load. The deficit is then supplied by an auxilliary heating device

    # Heating mode parameters
    # P_y_H: float = np.nan;
    # P_m_H: float = np.nan;
    # P_d_H: float = np.nan;
    # COP_y_H: float = np.nan;
    # COP_m_H: float = np.nan;
    # COP_d_H: float = np.nan;
    # dT_H: float = np.nan;
    Qdim_H: float = np.nan;
    P_s_H: float = np.nan;
    
    # Cooling mode parameters
    # P_y_C: float = np.nan;
    # P_m_C: float = np.nan;
    # P_d_C: float = np.nan;
    # EER: float = np.nan;
    # dT_C: float = np.nan;
    Qdim_C: float = np.nan;
    P_s_C: float = np.nan;



@dataclass_json
@dataclass
class HHEconfig:
    # Horizontal heat exchanger (HHE) topology and pipes
    source:str = 'HHE';
    N_HHE:int = 6;  # Number of HE loops (-)
    d:float = 0.04;  # Outer diameter of HE pipe (m)
    SDR:float = 17;  # SDR for HE pipes (-)
    D:float = 1.5;  # Pipe segment spacing (m)
    
    # Results
    L_HHE_H:float = np.nan;
    L_HHE_C:float = np.nan;
    Re_HHEmax_H:float = np.nan;
    dpdL_HHEmax_H:float = np.nan;
    Re_HHEmax_C:float = np.nan;
    dpdL_HHEmax_C:float = np.nan;
    

@dataclass_json
@dataclass
class BHEconfig:
    # Borehole heat exchangers (BHE)
    source:str = 'BHE';
    r_b:float = 0.152 / 2  # Borehole radius (m)
    r_p:float = 0.02  # Outer radius of U pipe (m)
    SDR:float = 11  # SDR for U-pipe (-)
    l_ss:float = 2.36  # Soil thermal conductivity along BHEs (W/m/K)
    rhoc_ss:float = 2.65e6  # Volumetric heat capacity of soil (along BHE). Assuming 70# quartz and 30# water (J/m3/K) #OK
    l_g:float = 1.75  # Grout thermal conductivity (W/m/K)
    rhoc_g:float = 3e6  # Grout volumetric heat capacity (J/m3/K)
    D_pipes:float = 0.015  # Wall to wall distance U-pipe legs (m)

    # BHE field
    NX:int = 1  # Number of boreholes in the x-direction (-)
    D_x:float = 15  # Spacing between boreholes in the x-direction (m)
    NY:int = 6 # Number of boreholes in the y-direction (-)
    D_y:float = 15  # Spacing between boreholes in the y-direction (m)
    
    # Results
    L_BHE_H:float = np.nan;
    L_BHE_C:float = np.nan;
    Re_BHEmax_H:float = np.nan;
    dpdL_BHEmax_H:float = np.nan;
    Re_BHEmax_C:float = np.nan;
    dpdL_BHEmax_C:float = np.nan;
    

@dataclass_json
@dataclass
class FullDimension:
    brine: Brine
    thermonet: Thermonet
    heatpump: Heatpump
    source_config: HHEconfig | BHEconfig
    pipe_group_name: List[str]
    d_pipes: List[float]
    FPH: float=np.nan
    FPC: float=np.nan
