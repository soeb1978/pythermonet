# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 15:05:17 2023

@author: SOEB
"""

from ThermonetDim_v076 import ThermonetDim, BRINE, THERMONET, HEAT_PUMPS, HHE, BHE

############### User set flow and thermal parameters by medium ################

# Project ID
PID = 'Energiakademiet, Samsø';                                     # Project name

# Input files
HPFN = 'Samso_HPSC.dat';                                            # Input file containing heat pump information
TOPOFN = 'Samso_TOPO.dat';                                          # Input file containing topology information 

# Brine
rhob = 965;                                                         # Brine density (kg/m3), T = 0C. https://www.handymath.com/cgi-bin/isopropanolwghtvoltble5.cgi?submit=Entry
cb = 4450;                                                          # Brine specific heat (J/kg/K). 4450 J/kg/K is loosly based on Ignatowicz, M., Mazzotti, W., Acuña, J., Melinder, A., & Palm, B. (2017). Different ethyl alcohol secondary fluids used for GSHP in Europe. Presented at the 12th IEA Heat Pump Conference, Rotterdam, 2017. Retrieved from http://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-215752
mub = 5e-3;                                                         # Brine dynamic viscosity (Pa*s). Source see above reference.
lb = 0.45;                                                          # Brine thermal conductivity (W/m/K). https://www.researchgate.net/publication/291350729_Investigation_of_ethanol_based_secondary_fluids_with_denaturing_agents_and_other_additives_used_for_borehole_heat_exchangers

# PE Pipes
lp = 0.4;                                                           # Pipe thermal conductivity (W/m/K). https://www.wavin.com/da-dk/catalog/Varme/Jordvarme/PE-80-lige-ror/40mm-jordvarme-PE-80PN6-100m

# Thermonet and HHE
PWD = 0.3;                                                          # Distance between forward and return pipe centers (m)
dpt = 90;                                                           # Target pressure loss in thermonet (Pa/m). 10# reduction to account for loss in fittings. Source: Oklahoma State University, Closed-loop/ground source heat pump systems. Installation guide., (1988). Interval: 98-298 Pa/m
lsh = 2;                                                            # Soil thermal conductivity thermonet and HHE (W/m/K) Guestimate (0.8-1.2 W/m/K)
lsc = 2;                                                            # Soil thermal conductivity thermonet and HHE (W/m/K) Guestimate (0.8-1.2 W/m/K)
rhocs = 2.5e6;                                                      # Soil volumetric heat capacity  thermonet and HHE (J/m3/K) OK. Guestimate
zd = 1.2;                                                           # Burial depth of thermonet and HHE (m)

# Heat pump
Thi = -3;                                                           # Design temperature for inlet (C) OK. Stress test conditions. Legislation stipulates Thi > -4C. Auxillary heater must be considered.
Tci = 17;                                                           # Design temperature for inlet (C) OK. Stress test conditions. Legislation stipulates Thi > -4C. Auxillary heater must be considered.
SF = 1;                                                             # Ratio of peak heating demand to be covered by the heat pump [0-1]. If SF = 0.8 then the heat pump delivers 80% of the peak heating load. The deficit is then supplied by an auxilliary heating device

# Source selection
SS = 0;                                                             # SS = 1: Borehole heat exchangers; SS = 0: Horizontal heat exchangers  

if SS == 0:
    # Horizontal heat exchanger (HHE) topology and pipes
    NHHE = 7;                                                       # Number of HE loops (-)
    PDHE = 0.04;                                                    # Outer diameter of HE pipe (m)                   
    HHESDR = 17;                                                    # SDR for HE pipes (-)
    dd = 1.5;                                                       # Pipe segment spacing (m)                            
    SRC = HHE(NHHE,PDHE,HHESDR,dd)

if SS == 1:
    # Borehole heat exchangers (BHE)
    rb = 0.152/2;                                                   # Borehole radius (m)                              
    rp = 0.02;                                                      # Outer radius of U pipe (m)                        
    BHESDR = 11;                                                    # SDR for U-pipe (-)                               
    lss = 2.36;                                                     # Soil thermal conductivity along BHEs (W/m/K)     
    rhocss = 2.65e6;                                                # Volumetric heat capacity of soil (along BHE). Assuming 70# quartz and 30# water (J/m3/K) #OK
    lg = 1.75;                                                      # Grout thermal conductivity (W/m/K)               
    rhocg = 3e6;                                                    # Grout volumetric heat capacity (J/m3/K)          
    PD = 0.015;                                                     # Wall to wall distance U-pipe legs (m)                                

    # BHE field
    NX = 1;                                                         # Number of boreholes in the x-direction (-)
    dx = 15;                                                        # Spacing between boreholes in the x-direction (m)
    NY = 6;                                                         # Number of boreholes in the y-direction (-)
    dy = 15;                                                        # Spacing between boreholes in the y-direction (m)
    SRC = BHE(rb,rp,BHESDR,lss,rhocss,lg,rhocg,PD,NX,dx,NY,dy)

BR = BRINE(rhob,cb,mub,lb);
THM = THERMONET(PWD,dpt,lsh,lsc,rhocs,zd);
HPS = HEAT_PUMPS(Thi,Tci,SF)    
test = ThermonetDim(PID,HPFN,TOPOFN,BR,lp,THM,HPS,SS,SRC);
############### User set flow and thermal parameters by medium END ############