from dataclasses import dataclass
from dataclasses_json import dataclass_json
import numpy as np
from typing import List


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
class Thermonet:
    
    # Thermonet and HHE
    D_gridpipes: float = np.nan    # Distance between forward and return pipe centers (m)
    l_p: float = np.nan     # Pipe thermal conductivity (W/m/K). https://www.wavin.com/da-dk/catalog/Varme/Jordvarme/PE-80-lige-ror/40mm-jordvarme-PE-80PN6-100m
    l_s_H: float = np.nan   # Soil thermal conductivity thermonet and HHE (W/m/K) Guestimate (0.8-1.2 W/m/K)
    l_s_C: float = np.nan   # Soil thermal conductivity thermonet and HHE (W/m/K) Guestimate (0.8-1.2 W/m/K)
    rhoc_s: float = np.nan  # Soil volumetric heat capacity  thermonet and HHE (J/m3/K). Guestimate
    z_grid: float = np.nan  # Burial depth of thermonet and HHE (m)
    T0: float = np.nan      # Yearly average surface temperature (C)
    A: float = np.nan       # Amplitude of yearly sinusoidal temperature variation (C)
    friction_model_pp: str = 'swamee-jain'  # The friction models use in pandapipes' flow calulation, options 'colebrook', 'swamee-jain' and 'nikuradse'

    # KART tilføjet topologi information fra TOPO_FILE
    SDR: float = np.nan;
    L_traces: float = np.nan;   # Trace length (m)
    N_traces: float = np.nan;   # Number of traces in a pipe group (-)
    L_segments: float = np.nan; # Total pipe length in a pipe group i.e. both forward and return pipes (m)
    I_PG: float = np.nan;       
    dp_PG: float = np.nan;      # Max total pressure drop over the forward plus return pipes in a trace
    V_brine: float = np.nan;    # Volume of brine in all pipes (m^3)
    T_dimv: float = np.nan;     # Vector of brine temperatures (Celcius) after each of the three pulses (year, month, peak)
    
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
    Ti_H: float = np.nan    # Design temperature for inlet (C) OK. Stress test conditions. Legislation stipulates Thi > -4C. Auxillary heater must be considered.
    Ti_C: float = np.nan    # Design temperature for inlet (C) OK. Stress test conditions. Legislation stipulates Thi > -4C. Auxillary heater must be considered.
    f_peak: float = np.nan  # Fraction of peak heating demand to be covered by the heat pump [0-1]. If f_peak = 0.8 then the heat pump delivers 80% of the peak heating load. The deficit is then supplied by an auxilliary heating device

    # Peak load duration in hours
    t_peak: float = np.nan

    # Heating mode parameters
    dT_H: float = np.nan;
    P_s_H: float = np.nan;
    
    # Cooling mode parameters
    dT_C: float = np.nan;
    P_s_C: float = np.nan;

@dataclass_json
@dataclass
class aggregatedLoad:
    
    # Heat pump
    Ti_H: float = np.nan  # Design temperature for inlet (C) OK. Stress test conditions. Legislation stipulates Thi > -4C. Auxillary heater must be considered.
    Ti_C: float = np.nan  # Design temperature for inlet (C) OK. Stress test conditions. Legislation stipulates Thi > -4C. Auxillary heater must be considered.
    To_H: float = np.nan;
    To_C: float = np.nan;

    f_peak: float = np.nan  # Fraction of peak heating demand to be covered by the heat pump [0-1]. If f_peak = 0.8 then the heat pump delivers 80% of the peak heating load. The deficit is then supplied by an auxilliary heating device

    # Peak load duration in hours
    t_peak: float = np.nan

    # Heating mode parameters
    Qdim_H: float = np.nan;
    P_s_H: float = np.nan;
    
    # Cooling mode parameters
    Qdim_C: float = np.nan;
    P_s_C: float = np.nan;



@dataclass_json
@dataclass
class HHEconfig:
    
    # Horizontal heat exchanger (HHE) topology and pipes
    source:str = 'HHE';
    
    N_HHE:int = np.nan;  # Number of HE loops (-)
    d:float = np.nan;  # Outer diameter of HE pipe (m)
    SDR:float = np.nan;  # SDR for HE pipes (-)
    D:float = np.nan;  # Pipe segment spacing (m)
    V_brine: float = np.nan;    # Volume of brine in the pipes (m^3)
    T_dimv: float = np.nan;     # Vector of brine temperatures (Celcius) after each of the three pulses (year, month, peak)
    
    # Results
    FPH:float = np.nan;
    FPC:float = np.nan;
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
    
    T0: float = np.nan      # Surface temperature (C)
    q_geo: float = np.nan   # Geothermal heat flux (W/m2)
    r_b:float = np.nan      # Borehole radius (m)
    r_p:float = np.nan      # Outer radius of U pipe (m)
    SDR:float = np.nan      # SDR for U-pipe (-)
    l_ss:float = np.nan     # Soil thermal conductivity along BHEs (W/m/K)
    rhoc_ss:float = np.nan  # Volumetric heat capacity of soil (along BHE). Assuming 70# quartz and 30# water (J/m3/K) #OK
    l_g:float = np.nan      # Grout thermal conductivity (W/m/K)
    rhoc_g:float = np.nan   # Grout volumetric heat capacity (J/m3/K)
    D_pipes:float = np.nan  # Wall to wall distance U-pipe legs (m)
    V_brine: float = np.nan;# Volume of brine in heat exchanger pipes (m^3)
    T_dimv: float = np.nan; # Vector of brine temperatures (Celcius) after each of the three pulses (year, month, peak)

    # BHE field
    NX:int = np.nan         # Number of boreholes in the x-direction (-)
    D_x:float = np.nan      # Spacing between boreholes in the x-direction (m)
    NY:int = np.nan         # Number of boreholes in the y-direction (-)
    D_y:float = np.nan      # Spacing between boreholes in the y-direction (m)
    gFuncMethod: str = 'ICS'# Method for calculating g-function: 'ICS' for Infinite Cylindrical Source (default), 'PYG' for pygfunction (finite source)
    
    # Results
    FPH:float = np.nan;
    FPC:float = np.nan;    
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
