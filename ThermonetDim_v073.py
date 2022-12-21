# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 08:53:07 2022

@author: SOEB
"""

# Conceptual model drawings are found below the code

import numpy as np
import pandas as pd
import math as mt
from fThermonetDim import ils, ps, Re, dp, Rp, CSM, RbMP, GCLS, RbMPflc
import time

tic = time.time();                                                    # Track computation time (s)                                                           

############### User set flow and thermal parameters by medium ################

# Project ID
PID = 'Vejerslev';                                     # Project name

# Input files
HPFN = 'Vejerslev_HPS1.dat';                                                  # Input file containing heat pump information
TOPOFN = 'Vejerslev_TOPO1.dat';                                           # Input file containing topology information 

# Brine
rhob = 965;                                                         # Brine density (kg/m3), T = 0C. https://www.handymath.com/cgi-bin/isopropanolwghtvoltble5.cgi?submit=Entry
cb = 4450;                                                          # Brine specific heat (J/kg/K). 4450 J/kg/K is loosly based on Ignatowicz, M., Mazzotti, W., Acuña, J., Melinder, A., & Palm, B. (2017). Different ethyl alcohol secondary fluids used for GSHP in Europe. Presented at the 12th IEA Heat Pump Conference, Rotterdam, 2017. Retrieved from http://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-215752
mub = 5e-3;                                                         # Brine dynamic viscosity (Pa*s). Source see above reference.
lb = 0.45;                                                          # Brine thermal conductivity (W/m/K). https://www.researchgate.net/publication/291350729_Investigation_of_ethanol_based_secondary_fluids_with_denaturing_agents_and_other_additives_used_for_borehole_heat_exchangers

# PE Pipes
lp = 0.4;                                                           # Pipe thermal conductivity (W/m/K). https://www.wavin.com/da-dk/catalog/Varme/Jordvarme/PE-80-lige-ror/40mm-jordvarme-PE-80PN6-100m

# Thermonet
PWD = 0.5;                                                          # Distance between forward and return pipe centers (m)
dpt = 90;                                                           # Target pressure loss in thermonet (Pa/m). 10# reduction to account for loss in fittings. Source: Oklahoma State University, Closed-loop/ground source heat pump systems. Installation guide., (1988). Interval: 98-298 Pa/m
lsh = 1.15;                                                          # Soil thermal conductivity thermonet and HHE (W/m/K) OK. Guestimate (0.8-1.2 W/m/K)
lsc = 0.8;                                                          # Soil thermal conductivity thermonet and HHE (W/m/K) OK. Guestimate (0.8-1.2 W/m/K)
rhocs = 2.5e6;                                                        # Soil volumetric heat capacity  thermonet and HHE (J/m3/K) OK. Guestimate
zd = 1;                                                             # Burial depth of thermonet and HHE (m)

# Heat pump
Thi = -3;                                                            # Design temperature for inlet (C) OK. Stress test conditions. Legislation stipulates Thi > -4C. Auxillary heater must be considered.
Tci = 20;                                                            # Design temperature for inlet (C) OK. Stress test conditions. Legislation stipulates Thi > -4C. Auxillary heater must be considered.
SF = 1;                                                              # Ratio of peak heating demand to be covered by the heat pump [0-1]. If SF = 0.8 then the heat pump delivers 80% of the peak heating load. The deficit is then supplied by an auxilliary heating device

# Source selection
SS = 1;                                                              # SS = 1: Borehole heat exchangers; SS = 0: Horizontal heat exchangers  

if SS == 0:
    # Horizontal heat exchanger (HHE) topology and pipes
    NHHE = 5;                                                           # Number of HE loops (-)
    PDHE = 0.04;                                                          # Outer diameter of HE pipe (m)                   
    HHESDR = 17;                                                        # SDR for HE pipes (-)
    dd = 1.5;                                                           # Pipe segment spacing (m)                            

if SS == 1:
    # Borehole heat exchangers (BHE)
    rb = 0.152/2;                                                       # Borehole radius (m)                              
    rp = 0.02;                                                          # Outer radius of U pipe (m)                        
    BHESDR = 11;                                                        # SDR for U-pipe (-)                               
    lss = 2.36;                                                         # Soil thermal conductivity along BHEs (W/m/K)     
    rhocss = 2.65e6;                                                    # Volumetric heat capacity of soil (along BHE). Assuming 70# quartz and 30# water (J/m3/K) #OK
    lg = 1.75;                                                          # Grout thermal conductivity (W/m/K)               
    rhocg = 3e6;                                                        # Grout volumetric heat capacity (J/m3/K)          
    PD = 0.015;                                                         # Wall to wall distance U-pipe legs (m)                                

    # BHE field
    NX = 4;                                                             # Number of boreholes in the x-direction (-)
    dx = 15;                                                            # Spacing between boreholes in the x-direction (m)
    NY = 4;                                                             # Number of boreholes in the y-direction (-)
    dy = 15;                                                            # Spacing between boreholes in the y-direction (m)

############### User set flow and thermal parameters by medium END ############

############################# Load all input data #############################

# Load heat pump data
HPS = pd.read_csv(HPFN, sep = '\t');                                # Heat pump input file

# Load grid topology
TOPOH = np.loadtxt(TOPOFN,skiprows = 1,usecols = (1,2,3));            # Load numeric data from topology file
TOPOC = TOPOH;
IPG = pd.read_csv(TOPOFN, sep = '\t');                              # Load the entire file into Panda dataframe
PGROUP = IPG.iloc[:,0];                                             # Extract pipe group IDs
IPG = IPG.iloc[:,4];                                                # Extract IDs of HPs connected to the different pipe groups
NPG = len(IPG);                                                     # Number of pipe groups

# Load pipe database
PIPES = pd.read_csv('PIPES.dat', sep = '\t');                       # Open file with available pipe outer diameters (mm). This file can be expanded with additional pipes and used directly.
PIPES = PIPES.values;                                               # Get numerical values from pipes excluding the headers
NP = len(PIPES);                                                    # Number of available pipes

########################### Load all input data END ###########################

###############################################################################
################################## INPUT END ##################################
###############################################################################

# Output to prompt
print(' ');
print('******************************************************************')
print('*********************** ThermonetDim v0.72 ***********************')
print('******************************************************************')
print(' ');
print('Project:', PID);

########### Precomputations and variables that should not be changed ##########

# Heat pump information
HPS = HPS.values;                                                   # Load numeric data from HP file
CPS = HPS[:,8:];                                                    # Place cooling demand data in separate array
for i in range(3):                                                  # For monthly and hourly cooling demands do
    CPS[:,i] = CPS[:,3]/(CPS[:,3]-1)*CPS[:,i];                      # Add circulation pump power consumption to cooling load (W)
HPS = HPS[:,0:8];                                                   # Remove cooling demand data from HPS array
NHP = len(HPS);                                                     # Number of heat pumps

# G-function evaluation times (DO NOT MODIFY!!!!!!!)
t = np.asarray([323479800, 7903800, 14400],dtype=float);            # time = [10 years + 3 months + 4 hours; 3 months + 4 hours; 4 hours]. Time vector for the temporal superposition (s).   

# Create array containing arrays of integers with HP IDs for all pipe sections
IPGA = [];  
for i in range(NPG):                                                # For all pipe groups
    tmp = IPG.iloc[i].split(',');                                   # Split the heat pump IDs in to a list of strings
    tmp = np.asarray(tmp);                                          # Convert strings to ndarray
    IPGA.append(tmp.astype(int)-1);                                 # Add integer arrays containing IDs to IPGA ndarray
IPG=IPGA                                                            # Redefine IPG
del IPGA;                                                           # Get rid of IPGA

# Brine
kinb = mub/rhob;                                                    # Brine kinematic viscosity (m2/s)  
ab = lb/(rhob*cb);                                                  # Brine thermal diffusivity (m2/s)  
Pr = kinb/ab;                                                       # Prandtl number (-)                

# Shallow soil (not for BHEs! - see below)
A = 7.900272987633280;                                              # Surface temperature amplitude (K) 
T0 = 9.028258373009810;                                             # Undisturbed soil temperature (C) 
o = 2*np.pi/86400/365.25;                                           # Angular velocity of surface temperature variation (rad/s) 
ast = lsh/rhocs;                                                     # Shallow soil thermal diffusivity (m2/s) - ONLY for pipes!!! 
TP = A*mt.exp(-zd*mt.sqrt(o/2/ast));                                # Temperature penalty at burial depth from surface temperature variation (K). Minimum undisturbed temperature is assumed . 

# Convert pipe diameter database to meters
PIPES = PIPES/1000;                                                 # Convert PIPES from mm to m (m)

# Allocate variables
indh = np.zeros(NPG);                                                # Index vector for pipe groups
indc = np.zeros(NPG);                                                # Index vector for pipe groupss
PIPESELH = np.zeros(NPG);                                            # Pipes selected from dimensioning for heating
PIPESELC = np.zeros(NPG);                                            # Pipes selected from dimensioning for cooling
PSH = np.zeros((NHP,3));                                             # Thermal load from heating on the ground (W)
QPGH = np.zeros(NPG);                                                # Design flow heating (m3/s)
QPGC = np.zeros(NPG);                                                # Design flow cooling (m3/s)
Rh = np.zeros(NPG);                                                  # Allocate pipe thermal resistance vector for heating (m*K/W)
Rc = np.zeros(NPG);                                                  # Allocate pipe thermal resistance vector for cooling (m*K/W)
TPGH = np.zeros(NPG);                                                # UHF delta temperature reponse of thermonet in heating mode for each pipe (K)
TPGC = np.zeros(NPG);                                                # UHF delta temperature reponse of thermonet in cooling mode for each pipe (K)
TH = np.zeros(NHP);                                                  # Volume weighted average of TPGH to compute mean temperature response of thermonet for cumumlated thermal heating load from HPs
TC = np.zeros(NHP);                                                  # Volume weighted average of TPGH to compute mean temperature response of thermonet for cumumlated thermal cooling load from HPs

# Simultaneity factors to apply to annual, monthly and hourly heating and cooling demands
S = np.zeros(3);
S[2] = SF*(0.62 + 0.38/NHP);                                        # Hourly. Varme Ståbi. Ligning 3 i "Effekt- og samtidighedsforhold ved fjernvarmeforsyning af nye boligområder"
S[0]  = 1; #0.62 + 0.38/NHP;                                        # Annual. Varme Ståbi. Ligning 3 i "Effekt- og samtidighedsforhold ved fjernvarmeforsyning af nye boligområder"
S[1]  = 1; #S(1);                                                   # Monthly. Varme Ståbi. Ligning 3 i "Effekt- og samtidighedsforhold ved fjernvarmeforsyning af nye boligområder"

# If horizontal heat exchangers are selected
if SS == 0:
    # Horizontal heat exchangers
    rihhe = PDHE*(1 - 2/HHESDR)/2;                                  # Inner radius of HHE pipes (m)
    rohhe = PDHE/2;

# If borehole heat exchangers are selected
if SS == 1:
    # BHE
    ri = rp*(1 - 2/BHESDR);                                         # Inner radius of U pipe (m)
    ass = lss/rhocss;                                               # BHE soil thermal diffusivity (m2/s)
    ag = lg/rhocg;                                                  # Grout thermal diffusivity (W/m/K)
    T0BHE = T0;                                                     # Measured undisturbed BHE temperature (C)
    PD = 2*rp + PD;                                                 # Redefine PD to shank spacing U-pipe (m)

    # Borehole field
    x = np.linspace(0,NX-1,NX)*dx;                                  # x-coordinates of BHEs (m)                     
    y = np.linspace(0,NY-1,NY)*dy;                                  # y-coordinates of BHEs (m)
    NBHE = NX*NY;                                                   # Number of BHEs (-)
    [XX,YY] = np.meshgrid(x,y);                                     # Meshgrid arrays for distance calculations (m)    
    Yv = np.concatenate(YY);                                        # YY concatenated (m)
    Xv = np.concatenate(XX);                                        # XX concatenated (m)
    
    NXi = int(np.ceil(NX/2));
    NYi = int(np.ceil(NY/2));
    w = np.ones((NYi,NXi));
    
    if np.mod(NX/2,1) > 0:
        w[:,NXi-1] = 0.5*w[:,NXi-1];
    
    if np.mod(NY/2,1) > 0:

        w[NYi-1,:] = 0.5*w[NYi-1,:];
    wv = np.concatenate(w);
    swv = sum(wv);
    xi = np.linspace(0,NXi-1,NXi)*dx;                                 # x-coordinates of BHEs (m)                     
    yi = np.linspace(0,NYi-1,NYi)*dy;                                 # y-coordinates of BHEs (m)
    [XXi,YYi] = np.meshgrid(xi,yi);                                   # Meshgrid arrays for distance calculations (m)
    Yvi = np.concatenate(YYi);                                        # YY concatenated (m)
    Xvi = np.concatenate(XXi);                                        # XX concatenated (m)
    
    # Solver settings for computing the flow and length corrected length of BHEs
    dL = 0.1;                                                           # Step length for trial trial solutions (m)
    LL = 10;                                                            # Additional length segment for which trial solutions are generated (m)

######### Precomputations and variables that should not be changed END ########

################################# Pipe sizing #################################

# Convert thermal load profile on HPs to flow rates
for i in range(3):
    PSH[:,i] = ps(S[i]*HPS[:,i+1],HPS[:,i+4]);                      # Annual (0), monthly (1) and daily (2) thermal load on the ground (W)
PSH[:,0] = PSH[:,0] - CPS[:,0];
Qdimh = PSH[:,2]/HPS[:,7]/rhob/cb;                                  # Design flow (m3/s)
Qdimc = S[2]*CPS[:,2]/CPS[:,4]/rhob/cb;                             # Design flow (m3/s)
HPS = np.c_[HPS,Qdimh];                                             # Append to heat pump data structure for heating
CPS = np.c_[CPS,Qdimc];                                             # Append to heat pump data structure for cooling

# Heat pump
Tho = Thi - sum(Qdimh*HPS[:,7])/sum(Qdimh);                         # Volumetric flow rate weighted average brine delta-T (C)
TCH1 = T0 - (Thi + Tho)/2 - TP;                                     # Temperature condition for with heating termonet. Eq. 2.19 Advances in GSHP systems. Tp in the book refers to the influence from adjacent BHEs. This effect ignored in this tool.
Tco = Tci + sum(Qdimc*CPS[:,4])/sum(Qdimc);                         # Volumetric flow rate weighted average brine delta-T (C)
TCC1 = (Tci + Tco)/2 - T0 - TP;                                     # Temperature condition for with heating termonet. Eq. 2.19 Advances in GSHP systems. Tp in the book refers to the influence from adjacent BHEs. This effect ignored in this tool.

if SS == 1:                                                         
    TCH2 = T0BHE - (Thi + Tho)/2;                                   # Temperature condition for heating with BHE. Eq. 2.19 Advances in GSHP systems but surface temperature penalty is removed from the criterion as it doesn't apply to BHEs
    TCC2 = (Tci + Tco)/2 - T0BHE;                                   # Temperature condition for heating with BHE. Eq. 2.19 Advances in GSHP systems but surface temperature penalty is removed from the criterion as it doesn't apply to BHEs
    
# Compute flow and pressure loss in BHEs and HHEs under peak load conditions
if SS == 0:
    # HHE heating
    QHHEH = sum(Qdimh)/NHHE;                                        # Peak flow in HHE pipes (m3/s)
    vhheh = QHHEH/np.pi/rihhe**2;                                   # Peak flow velocity in HHE pipes (m/s)
    RENHHEH = Re(rhob,mub,vhheh,2*rihhe);                           # Peak Reynolds numbers in HHE pipes (-)
    dpHHEH = float(dp(rhob,mub,QHHEH,2*rihhe));                     # Peak pressure loss in HHE pipes (Pa/m)

    # HHE cooling
    QHHEC = sum(Qdimc)/NHHE;                                        # Peak flow in HHE pipes (m3/s)
    vhhec = QHHEC/np.pi/rihhe**2;                                   # Peak flow velocity in HHE pipes (m/s)
    RENHHEC = Re(rhob,mub,vhhec,2*rihhe);                           # Peak Reynolds numbers in HHE pipes (-)
    dpHHEC = float(dp(rhob,mub,QHHEC,2*rihhe));                     # Peak pressure loss in HHE pipes (Pa/m)

if SS == 1:
    # BHE heating
    QBHEH = sum(Qdimh)/NBHE;                                        # Peak flow in BHE pipes (m3/s)
    vbheh = QBHEH/np.pi/ri**2;                                      # Flow velocity in BHEs (m/s)
    RENBHEH = Re(rhob,mub,vbheh,2*ri);                              # Reynold number in BHEs (-)
    dpBHEH = dp(rhob,mub,QBHEH,2*ri);                               # Pressure loss in BHE (Pa/m)
    
    # BHE cooling
    QBHEC = sum(Qdimc)/NBHE;                                        # Peak flow in BHE pipes (m3/s)
    vbhec = QBHEC/np.pi/ri**2;                                      # Flow velocity in BHEs (m/s)
    RENBHEC = Re(rhob,mub,vbhec,2*ri);                              # Reynold number in BHEs (-)
    dpBHEC = dp(rhob,mub,QBHEC,2*ri);                               # Pressure loss in BHE (Pa/m)

# Compute design flow for the pipes
for i in range(NPG):
   QPGH[i]=sum(HPS[np.ndarray.tolist(IPG[i]),8])/TOPOH[i,2];        # Sum the heating brine flow for all consumers connected to a specific pipe group and normalize with the number of traces in that group to get flow in the individual pipes (m3/s)
   QPGC[i]=sum(CPS[np.ndarray.tolist(IPG[i]),5])/TOPOC[i,2];        # Sum the cooling brine flow for all consumers connected to a specific pipe group and normalize with the number of traces in that group to get flow in the individual pipes (m3/s)

# Select the smallest diameter pipe that fulfills the pressure drop criterion
for i in range(NPG):                                 
    PIPESI = PIPES*(1-2/TOPOH[i,0]);                                # Compute inner diameters (m). Variable TOPOH or TOPOC are identical here.
    indh[i] = np.argmax(dp(rhob,mub,QPGH[i],PIPESI)<dpt);           # Find first pipe with a pressure loss less than the target (-)
    indc[i] = np.argmax(dp(rhob,mub,QPGC[i],PIPESI)<dpt);           # Find first pipe with a pressure loss less than the target (-)
    PIPESELH[i] = PIPES[int(indh[i])];                              # Store pipe selection in new variable (m)
    PIPESELC[i] = PIPES[int(indc[i])];                              # Store pipe selection in new variable (m)
indh = indh.astype(int);                            
indc = indc.astype(int);

# Output the pipe sizing
print(' ');
print('*************** Suggested pipe dimensions heating ***************'); 
for i in range(NPG):
    print(f' {PGROUP.iloc[i]}: Ø{int(1000*PIPESELH[i])} mm SDR {int(TOPOH[i,0])}');
print(' ');

print('*************** Suggested pipe dimensions cooling ***************');
for i in range(NPG):
    print(f' {PGROUP.iloc[i]}: Ø{int(1000*PIPESELC[i])} mm SDR {int(TOPOC[i,0])}');
print(' ');

############################### Pipe sizing END ###############################

################## Compute temperature response of thermonet ##################

# Compute thermal resistances for pipes in heating mode
DiSELH = PIPESELH*(1-2/TOPOH[:,0]);                                 # Compute inner diameter of selected pipes (m)
vh = QPGH/np.pi/DiSELH**2*4;                                        # Compute flow velocity for selected pipes (m/s)
RENH = Re(rhob,mub,vh,DiSELH);                                      # Compute Reynolds numbers for the selected pipes (-)
LENGTHS = 2*TOPOH[:,1]*TOPOH[:,2];                                  # Total lengths of different pipe segments (m)
TLENGTH = sum(LENGTHS);                                             # Total length of termonet (m)
VOLH = LENGTHS*np.pi*DiSELH**2/4;                                   # Brine volume per pipe segment (m3)
TVOLH = sum(VOLH);                                                  # Total brine voume (m3)
TOPOH = np.c_[TOPOH,PIPESELH,RENH,LENGTHS];                         # Add pipe selection diameters (m), Reynolds numbers (-) and lengths as columns to the TOPO array
for i in range(NPG):                                                # For all pipe groups
    Rh[i] = Rp(DiSELH[i],PIPESELH[i],RENH[i],Pr,lb,lp);             # Compute thermal resistances (m*K/W)
TOPOH = np.c_[TOPOH, Rh];                                           # Append thermal resistances to pipe groups as a column in TOPO (m*K/W)

# Compute thermal resistances for pipes in cooling mode
DiSELC = PIPESELC*(1-2/TOPOC[:,0]);                                 # Compute inner diameter of selected pipes (m)
vc = QPGC/np.pi/DiSELC**2*4;                                        # Compute flow velocity for selected pipes (m/s)
RENC = Re(rhob,mub,vc,DiSELC);                                      # Compute Reynolds numbers for the selected pipes (-)
VOLC = LENGTHS*np.pi*DiSELC**2/4;                                   # Brine volume per pipe segment (m3)
TVOLC = sum(VOLC);                                                  # Total brine voume (m3)
TOPOC = np.c_[TOPOC,PIPESELC,RENC,LENGTHS];                         # Add pipe selection diameters (m), Reynolds numbers (-) and lengths as columns to the TOPO array
for i in range(NPG):                                                # For all pipe groups
    Rc[i] = Rp(DiSELC[i],PIPESELC[i],RENC[i],Pr,lb,lp);             # Compute thermal resistances (m*K/W)
TOPOC = np.c_[TOPOC, Rc];                                           # Append thermal resistances to pipe groups as a column in TOPO (m*K/W)

# Compute delta-qs for superposition of heating load responses
dPSH = np.zeros((NHP,3));                                           # Allocate power difference matrix for tempoeral superposition (W)
dPSH[:,0] = PSH[:,0];                                               # First entry is just the annual average power (W)
dPSH[:,1:] = np.diff(PSH);                                          # Differences between year-month and month-hour are added (W)
cdPSH = np.cumsum(dPSH,0);

# Compute delta-qs for superposition of cooling load responses.
dPSC = np.zeros((NHP,3));                                           # Allocate power difference matrix for tempoeral superposition (W)
dPSC = np.c_[-PSH[:,0],CPS[:,1:3]];
dPSC[:,1:] = np.diff(dPSC);                                         # Differences between year-month and month-hour are added (W)
cdPSC = np.cumsum(dPSC,0);

# Compute aggregated temperature responses in heating and cooling mode
GTHMH = np.zeros([NPG,3]);
GTHMC = np.zeros([NPG,3]);
K1 = ils(ast,t,PWD) - ils(ast,t,2*zd) - ils(ast,t,np.sqrt(PWD**2+zd**2));
for i in range(NPG):
  GTHMH[i,:] = CSM(PIPESELH[i]/2,PIPESELH[i]/2,t,ast) + K1;
  GTHMC[i,:] = CSM(PIPESELC[i]/2,PIPESELC[i]/2,t,ast) + K1;
 
# Compute the caliometric (volume) average brine temperature. This is a more consistent implementation of the UHF condition 
for j in range(NHP):
    for i in range(NPG):
        TPGH[i] = np.dot(cdPSH[j]/TLENGTH,GTHMH[i]/lsh + Rh[i])*VOLH[i];
        TPGC[i] = np.dot(cdPSC[j]/TLENGTH,GTHMC[i]/lsc + Rc[i])*VOLC[i];
    TH[j] = sum(TPGH)/TVOLH;
    TC[j] = sum(TPGC)/TVOLC;
        
NSHPH = np.argmax(TH > TCH1)-1;                                     # Find the first heat pump in the cumsum that exceeds the temperature condition and subtract one from this index (-)
dHPH = (TCH1 - TH[NSHPH])/(TH[NSHPH+1]-TH[NSHPH]);
THMqh = (sum(HPS[0:NSHPH+1,3]) + dHPH*HPS[NSHPH+1,3])/TLENGTH;      # Compute the heat pump power supplied supplied on the hot side of the HP per meter thermonet (W/m)
dPSH[NSHPH+1,:]=(1-dHPH)*dPSH[NSHPH+1,:];                           # Compute the fraction of that heat pumps ground thermal load that must be supplied by BHEs or HHEs and update dPS (W)
PHEH = sum(dPSH[(NSHPH+1):,:],0);                                   # Compute the ground thermal load to be supplied by BHE or HHE (W)

NSHPC = np.argmax(TC>TCC1)-1;                                       # Find the first heat pump in the cumsum that exceeds the temperature condition and subtract one from this index (-)
dHPC = (TCC1 - TC[NSHPH])/(TC[NSHPH+1]-TC[NSHPH]);
THMqc = (sum(CPS[0:NSHPH+1,2])+dHPC*CPS[NSHPH+1,2])/TLENGTH;        # Compute the heat pump power supplied supplied on the hot side of the HP per meter thermonet (W/m)
dPSC[NSHPH+1,:]=(1-dHPC)*dPSC[NSHPH+1,:];                           # Compute the fraction of that heat pumps ground thermal load that must be supplied by BHEs or HHEs and update dPS (W)
PHEC = sum(dPSC[(NSHPH+1):,:],0);                                   # Compute the ground thermal load to be supplied by BHE or HHE (W)

########################## Display results in console #########################
print('************** Thermonet energy production capacity **************'); 
#print('The thermonet supplies ' + str(round(THMq)) + ' W/m on the condenser sides of the HPs');
print(f'The thermonet supplies {round(100*TLENGTH*THMqh/sum(HPS[:,3],0))}% of the peak heating demand');  
print(f'The thermonet fully supplies the heat pumps with IDs 1 - {int(np.floor(NSHPH+1))} with heating' ) ;
print(f'The thermonet supplies {round(100*TLENGTH*THMqc/sum(CPS[:,2],0))}% of the peak cooling demand');  
print(f'The thermonet fully supplies the heat pumps with IDs 1 - {int(np.floor(NSHPC+1))} with cooling' );
print(' ');
######################## Display results in console END #######################

################################ Source sizing ################################

# If BHEs are selected as source
if SS == 1:                                     
    ###########################################################################
    ############################ Borehole computation #########################
    ###########################################################################

    ######################### Generate G-functions ############################
    GBHE = CSM(rb,rb,t[0:2],ass);                                   # Compute g-functions for t[0] and t[1] with the cylindrical source model (-)
    s1 = 0;                                                         # Summation variable for t[0] G-function (-)
    s2 = 0;                                                         # Summation variable for t[1] G-function (-)
    for i in range(NXi*NYi):                                           # Line source superposition for all neighbour boreholes
        DIST = np.sqrt((XX-Xvi[i])**2 + (YY-Yvi[i])**2);              # Compute distance matrix (to neighbour boreholes) (m)
        DIST = DIST[DIST>0];
        s1 = s1 + wv[i]*sum(ils(ass,t[0],DIST));                          # Compute the sum of all thermal disturbances from neighbour boreholes (G-function contributions) for t[0] (-)
        s2 = s2 + wv[i]*sum(ils(ass,t[1],DIST));                          # Compute the sum of all thermal disturbances from neighbour boreholes (G-function contributions) for t[0] (-)
    GBHE[0] = GBHE[0] + s1/swv;                                    # Add the average neighbour contribution to the borehole field G-function for t[0] (-)
    GBHE[1] = GBHE[1] + s2/swv;                                    # Add the average neighbour contribution to the borehole field G-function for t[1] (-)

    # Compute borehole resistance with the first order multipole method ignoring flow and length effects
    Rbh = RbMP(lb,lp,lg,lss,rb,rp,ri,PD,RENBHEH,Pr);                 # Compute the borehole thermal resistance (m*K/W)
    Rbc = RbMP(lb,lp,lg,lss,rb,rp,ri,PD,RENBHEC,Pr);                 # Compute the borehole thermal resistance (m*K/W)
    #Rb = 0.12;                                                     # TRT estimate can be supplied instread (m*K/W)

    # Composite cylindrical source model GCLS() for short term response. Hu et al. 2014. Paper here: https://www.sciencedirect.com/science/article/abs/pii/S0378778814005866?via#3Dihub
    reh = rb/np.exp(2*np.pi*lg*Rbh);                                  # Heating: Compute the equivalent pipe radius for cylindrical symmetry (m). This is how Hu et al. 2014 define it.
    rec = rb/np.exp(2*np.pi*lg*Rbc);                                  # Cooling: Compute the equivalent pipe radius for cylindrical symmetry (m). This is how Hu et al. 2014 define it.

    # The Fourier numbers Fo1-Fo3 are neccesary for computing the solution 
    Fo1 = ass*t[2]/rb**2;                                    
    G1 = GCLS(Fo1); 

    Fo2h = ag*t[2]/reh**2;
    G2h = GCLS(Fo2h);

    Fo2c = ag*t[2]/rec**2;
    G2c = GCLS(Fo2c);

    Fo3 = ag*t[2]/rb**2;
    G3 = GCLS(Fo3);

    Rwh = G1/lss + G2h/lg - G3/lg;                                    # Step response for short term model on the form q*Rw = T (m*K/W). Rw indicates that it is in fact a thermal resistance
    Rwc = G1/lss + G2c/lg - G3/lg;                                    # Step response for short term model on the form q*Rw = T (m*K/W). Rw indicates that it is in fact a thermal resistance

    # Compute approximate combined length of BHES (length effects not considered)
    GBHEF = GBHE;                                                   # Retain a copy of the G function for length correction later on (-)
    GBHEH = np.asarray([GBHE[0]/lss+Rbh,GBHE[1]/lss+Rbh, Rwh]);        # Heating G-function
    GBHEC = np.asarray([GBHE[0]/lss+Rbc,GBHE[1]/lss+Rbc, Rwc]);        # Heating G-function
    LBHEH = np.dot(PHEH,GBHEH/TCH2);                                    # Sizing equation for computing the required borehole meters (m)
    LBHEC = np.dot(PHEC,GBHEC/TCC2);                                    # Sizing equation for computing the required borehole meters (m)
    #BHEq = (sum(HPS[NSHPH+2:,3])+(1-dx)*HPS[NSHPH+1,3])/LBHE;         # Compute the peak heat pump power supplied on the hot side of the HP per meter BHE relative to the nominal effect of the heat pump (W/m)
    
    # Determine the exact solution by searching the neighbourhood of the approximate length solution
    # Heating model
    LBHEHv = LBHEH/NBHE + np.arange(0,LL,dL);
    NLBHEHv = len(LBHEHv);
    Rbhv = np.zeros(NLBHEHv);
    Tsolh = np.zeros(NLBHEHv);
    
    # Cooling mode
    LBHECv = LBHEC/NBHE + np.arange(0,LL,dL);
    NLBHECv = len(LBHECv);
    Rbcv = np.zeros(NLBHECv);
    Tsolc = np.zeros(NLBHECv);
    
    for i in range(NLBHEHv):                                         # Compute Rb for the specified number of boreholes and lengths considering flow and length effects (m*K/W)
        Rbhv[i] = RbMPflc(lb,lp,lg,lss,rhob,cb,rb,rp,ri,LBHEHv[i],PD,QBHEH,RENBHEH,Pr);    #K. Compute BHE length and flow corrected multipole estimates of Rb for all candidate solutions (m*K/W)
        Tsolh[i] = np.dot(PHEH,np.array([GBHEF[0]/lss + Rbhv[i], GBHEF[1]/lss + Rbhv[i], Rwh]))/LBHEHv[i]/NBHE;                             #OK. Use Spitlers sizing formula for computing the corresponding temperature response for all candidate solutions (C)
    indLBHEH = np.argmax(Tsolh<TCH2);                                    # OK. Get rid of candidates that undersize the system. 
    LBHEH = LBHEHv[indLBHEH]*NBHE;                                     # Exact solution BHE length (m)
    BHEqh = (sum(HPS[NSHPH+2:,3])+(1-dHPH)*HPS[NSHPH+1,3])/LBHEH;    
    
    if (Tsolh[indLBHEH]-TCH2) > 0.1:
        print('Warning - the length steps used for computing the exact length for heating are too big. Reduce the stepsize and recompute a solution.');
    
    for i in range(NLBHECv):                                         # Compute Rb for the specified number of boreholes and lengths considering flow and length effects (m*K/W)
        Rbcv[i] = RbMPflc(lb,lp,lg,lss,rhob,cb,rb,rp,ri,LBHECv[i],PD,QBHEC,RENBHEC,Pr);    #K. Compute BHE length and flow corrected multipole estimates of Rb for all candidate solutions (m*K/W)
        Tsolc[i] = np.dot(PHEC,np.array([GBHEF[0]/lss + Rbcv[i], GBHEF[1]/lss + Rbcv[i], Rwc]))/LBHECv[i]/NBHE;                             #OK. Use Spitlers sizing formula for computing the corresponding temperature response for all candidate solutions (C)
    indLBHEC = np.argmax(Tsolc<TCC2);                                    # OK. Get rid of candidates that undersize the system. 
    LBHEC = LBHECv[indLBHEC]*NBHE;                                     # Exact solution BHE length (m)
    BHEqc = (sum(CPS[NSHPH+2:,2])+(1-dHPC)*CPS[NSHPH+1,2])/LBHEC;
    if (Tsolc[indLBHEC]-TCC2) > 0.1:
        print('Warning - the length steps used for computing the exact length for cooling are too big. Reduce the stepsize and recompute a solution.');    
    # Display output in console
    print('******* Suggested length of borehole heat exchangers (BHE) *******'); 
    print(f'Required length of each of the {int(NBHE)} BHEs = {int(np.ceil(LBHEH/NBHE))} m for heating');
    print(f'Required length of each of the {int(NBHE)} BHEs = {int(np.ceil(LBHEC/NBHE))} m for cooling');
    print(f'Maximum pressure loss in BHEs in heating mode = {int(np.ceil(dpBHEH))} Pa/m');
    print(f'Maximum pressure loss in BHEs in cooling mode = {int(np.ceil(dpBHEC))} Pa/m');

# If HHEs are selected as source
if SS == 0:
    ###########################################################################
    ############################### HHE computation ###########################
    ###########################################################################

    # Compute combined length of HHEs   
    ind = np.linspace(0,2*NHHE-1,2*NHHE);                           # Unit distance vector for HHE (-)
    s = np.zeros(2);                                                # s is a temperature summation variable, s[0]: annual, s[1] monthly, hourly effects are insignificant and ignored (C)
    DIST = dd*ind;                                                  # Distance vector for HHE (m)
    for i in range(NHHE):                                           # For all pipe segments (2 per loop)
        s[0] = s[0] + sum(ils(ast,t[0],abs(DIST[ind!=i]-i*dd))) - sum(ils(ast,t[0],np.sqrt((DIST-i*dd)**2 + 4*zd**2))); # Sum annual temperature responses from distant pipes (C)
        s[1] = s[1] + sum(ils(ast,t[1],abs(DIST[ind!=i]-i*dd))) - sum(ils(ast,t[1],np.sqrt((DIST-i*dd)**2 + 4*zd**2))); # Sum monthly temperature responses from distant pipes (C)
    GHHE = CSM(rohhe,rohhe,t,ast);                                  # Add the average temperature disturbance at a distance s (C) at t[0] to the BHE wall temperature G[0] (C)
    GHHE[0:2] = GHHE[0:2] + s/NHHE;                                 # Add thermal disturbance from neighbour pipes (-)
    
    #Heating
    RHHEH = float(Rp(2*rihhe,2*rohhe,RENHHEH,Pr,lb,lp));            # Compute the pipe thermal resistance (m*K/W)
    GHHEH = np.asarray(GHHE/lsh+RHHEH);                             # Add annual and monthly thermal resistances to GHHE (m*K/W)
    LHHEH = np.dot(PHEH,GHHE/TCH1);                                 # Sizing equation for computing the required borehole meters (m)
    HHEqh = (sum(HPS[NSHPH+2:,3])+(1-dHPH)*HPS[NSHPH+1,3])/LHHEH;   # Compute the heat pump power supplied on the hot side of the HP per meter BHE (W/m)
    
    #Cooling
    RHHEC = float(Rp(2*rihhe,2*rohhe,RENHHEC,Pr,lb,lp));            # Compute the pipe thermal resistance (m*K/W)
    GHHEC = np.asarray(GHHE/lsc+RHHEC);                             # Add annual and monthly thermal resistances to GHHE (m*K/W)
    LHHEC = np.dot(PHEC,GHHE/TCC1);                                 # Sizing equation for computing the required borehole meters (m)
    HHEqc = (sum(CPS[NSHPC+2:,2])+(1-dHPC)*CPS[NSHPC+1,2])/LHHEC;   # Compute the heat pump power supplied on the hot side of the HP per meter BHE (W/m)
    
    # Output results to console
    print('****** Suggested length of horizontal heat exchangers (HHE) ******');
    print(f'Required length of each of the {int(NHHE)} horizontal loops = {int(np.ceil(LHHEH/NHHE))} m for heating');
    print(f'Required length of each of the {int(NHHE)} horizontal loops = {int(np.ceil(LHHEC/NHHE))} m for cooling');
    print(f'Maximum pressure loss in HHE pipes = {int(np.ceil(dpHHEH))} Pa/m during peak heating loads');
    print(f'Maximum pressure loss in HHE pipes = {int(np.ceil(dpHHEC))} Pa/m during peak cooling loads');
    
############################## Source sizing END ##############################

# Output computation time to console
print(' ');
print('************************ Computation time ************************');
toc = time.time();                                                  # Track computation time (S)
print(f'Elapsed time: {round(toc-tic,6)} seconds');

################## CONCEPTUAL MODEL DRAWINGS FOR REFERENCE ####################

################ Conceptual model for twin pipe in the ground #################
#       x1      x2
#
#          Air
#
# ----------------------
#
#         Ground
#
#       o1      o2
#
#       Legend:
#       x : mirror source
#       o : real source (actual pipe)
#     --- : the ground surface where T = 0. Actual ground temperatures are then superimposed
#       
#       T(o1) = q*(R(o1) + R(o2) - R(x1) - R(x2)) + Tu(t)
#       Tu(t) is the undisturbed seasonal temperature variation at depth
#       Assumption: surface temperature equal to the undisturbed seasonal temperature (Dirichlet BC)
#
#
############## Conceptual model for twin pipe in the ground END ###############

################### Conceptual model for HHE in the ground ####################

# Topology of horizontal heat exchangers (NHHE = 3)
# |  Loop  |	    |  Loop  |	      |  Loop  |
# |	       |	    |	     |	      |	       |
# |	       |  	    |	     |	      |	       |
# |<--dd-->|<--dd-->|<--dd-->|<--dd-->|<--dd-->|
# |        |   	    |        |	      |        |
# |        |   	    |        |	      |        |
# |________|   	    |________|        |________|
#
# 
# Mirror sources (above the ground surfaces) enforce Dirichlet BC on ground surface - similar to thermonet model

################# Conceptual model for HHE in the ground END ##################