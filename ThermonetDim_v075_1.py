# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 08:53:07 2022

@author: SOEB
"""
##########################
# To consider or implement
##########################
# 1: Samtidighedsfaktor anvendes også på køling. Er det en god ide?
# 2: Der designes to rørsystemer - et for varme og et for køling. Bør ændres?

# Conceptual model drawings are found below the code

import numpy as np
import pandas as pd
import math as mt
from fThermonetDim import ils, ps, Re, dp, Rp, CSM, RbMP, GCLS, RbMPflc
import time

tic = time.time();                                                    # Track computation time (s)                                                           

############### User set flow and thermal parameters by medium ################

# Project ID
PID = 'Energiakademiet, Samsø';                                     # Project name

# Input files
HP_file = 'Silkeborg_HPSC.dat';                                            # Input file containing heat pump information
TOPO_file = 'Silkeborg_TOPO.dat';                                          # Input file containing topology information 

# Brine (fluid)
rho_f = 965;                                                         # Brine density (kg/m3), T = 0C. https://www.handymath.com/cgi-bin/isopropanolwghtvoltble5.cgi?submit=Entry
c_f = 4450;                                                          # Brine specific heat (J/kg/K). 4450 J/kg/K is loosly based on Ignatowicz, M., Mazzotti, W., Acuña, J., Melinder, A., & Palm, B. (2017). Different ethyl alcohol secondary fluids used for GSHP in Europe. Presented at the 12th IEA Heat Pump Conference, Rotterdam, 2017. Retrieved from http://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-215752
mu_f = 5e-3;                                                         # Brine dynamic viscosity (Pa*s). Source see above reference.
l_f = 0.45;                                                          # Brine thermal conductivity (W/m/K). https://www.researchgate.net/publication/291350729_Investigation_of_ethanol_based_secondary_fluids_with_denaturing_agents_and_other_additives_used_for_borehole_heat_exchangers

# PE Pipes
l_p = 0.4;                                                           # Pipe thermal conductivity (W/m/K). https://www.wavin.com/da-dk/catalog/Varme/Jordvarme/PE-80-lige-ror/40mm-jordvarme-PE-80PN6-100m

# Thermonet and HHE
D_gridpipes = 0.3;                                                          # Distance between forward and return pipe centers (m)
dpdL_t = 90;                                                           # Target pressure loss in thermonet (Pa/m). 10# reduction to account for loss in fittings. Source: Oklahoma State University, Closed-loop/ground source heat pump systems. Installation guide., (1988). Interval: 98-298 Pa/m
l_s_H = 1.25; #KART H/L for Heating/Cooling? -> dokumenter                                                           # Soil thermal conductivity thermonet and HHE (W/m/K) Guestimate (0.8-1.2 W/m/K)
l_s_C = 1.25; #KART ditto                                                           # Soil thermal conductivity thermonet and HHE (W/m/K) Guestimate (0.8-1.2 W/m/K)
rhoc_s = 2.5e6;                                                      # Soil volumetric heat capacity  thermonet and HHE (J/m3/K) OK. Guestimate
z_grid = 1.2;                                                           # Burial depth of thermonet and HHE (m)

# Heat pump
Ti_H = -3;                                                           # Design temperature for inlet (C) OK. Stress test conditions. Legislation stipulates Ti_H > -4C. Auxillary heater must be considered.
Ti_C = 20;                                                           # Design temperature for inlet (C) OK. Stress test conditions. Legislation stipulates Ti_H > -4C. Auxillary heater must be considered.
SF = 1;                                                             # Ratio of peak heating demand to be covered by the heat pump [0-1]. If SF = 0.8 then the heat pump delivers 80% of the peak heating load. The deficit is then supplied by an auxilliary heating device

# Source selection
SS = 1;                                                             # SS = 1: Borehole heat exchangers; SS = 0: Horizontal heat exchangers  

if SS == 0:
    # Horizontal heat exchanger (HHE) topology and pipes
    N_HHE = 6;                                                       # Number of HE loops (-)
    d_HHE = 0.04;                                                    # Outer diameter of HE pipe (m)                   
    SDR_HHE = 17;                                                    # SDR for HE pipes (-)
    D_HHE = 1.5;                                                       # Pipe segment spacing (m)                            

if SS == 1:
    # Borehole heat exchangers (BHE)
    r_b = 0.152/2;                                                   # Borehole radius (m)                              
    r_p = 0.02;                                                      # Outer radius of U pipe (m)                        
    SDR_BHE = 11;                                                    # SDR for U-pipe (-)                               
    l_ss = 2.36;                                                     # Soil thermal conductivity along BHEs (W/m/K)     
    rhoc_ss = 2.65e6;                                                # Volumetric heat capacity of soil (along BHE). Assuming 70# quartz and 30# water (J/m3/K) #OK
    l_g = 1.75;                                                      # Grout thermal conductivity (W/m/K)               
    rhoc_g = 3e6;                                                    # Grout volumetric heat capacity (J/m3/K)          
    D_BHEpipes = 0.015;                                                     # Wall to wall distance U-pipe legs (m)                                

    # BHE field
    NX = 1;                                                         # Number of boreholes in the x-direction (-)
    dx = 15;                                                        # Spacing between boreholes in the x-direction (m)
    NY = 6;                                                         # Number of boreholes in the y-direction (-)
    dy = 15;                                                        # Spacing between boreholes in the y-direction (m)

############### User set flow and thermal parameters by medium END ############

############################# Load all input data #############################

# Load heat pump data
HPS = pd.read_csv(HP_file, sep = '\t+', engine='python');                                # Heat pump input file

# Load grid topology
TOPO_H = np.loadtxt(TOPO_file,skiprows = 1,usecols = (1,2,3));          # Load numeric data from topology file
TOPO_C = TOPO_H; 
I_PG = pd.read_csv(TOPO_file, sep = '\t+', engine='python');                              # Load the entire file into Panda dataframe
pipeGroupNames = I_PG.iloc[:,0];                                             # Extract pipe group IDs
I_PG = I_PG.iloc[:,4];                                                # Extract IDs of HPs connected to the different pipe groups
N_PG = len(I_PG);                                                     # Number of pipe groups

# Load pipe database
d_pipes = pd.read_csv('PIPES.dat', sep = '\t');                       # Open file with available pipe outer diameters (mm). This file can be expanded with additional pipes and used directly.
d_pipes = d_pipes.values;                                               # Get numerical values from pipes excluding the headers
#NP = len(d_pipes); #KART variablen bruges aldrig -> slet                                                   # Number of available pipes

########################### Load all input data END ###########################

###############################################################################
################################## INPUT END ##################################
###############################################################################

# Output to prompt
print(' ');
print('************************************************************************')
print('************************** ThermonetDim v0.75_1 ************************')
print('************************************************************************')
print(' ');
print(f'Project: {PID}');

########### Precomputations and variables that should not be changed ##########

# Heat pump information
HPS = HPS.values  # Load numeric data from HP file
CPS = HPS[:, 8:]  # Place cooling demand data in separate array
HPS = HPS[:, :8]  # Remove cooling demand data from HPS array
N_HP = len(HPS)  # Number of heat pumps

# Add circulation pump power consumption to cooling load (W)
# KART: til dokumentation Qcool -> (EER/(EER-1))*Qcool enig?
# KART hvorfor håndteres varme og køl forskelligt?
CPS[:, :3] = CPS[:, 3:4] / (CPS[:, 3:4] - 1) * CPS[:, :3]                                            

# G-function evaluation times (DO NOT MODIFY!!!!!!!)
SECONDS_IN_YEAR = 31536000; # KART: overvej at beregne disse -> mere klart hvor mange dage man regner for måned/år
SECONDS_IN_MONTH = 2628000; # KART ditto
SECONDS_IN_HOUR = 3600;
t = np.asarray([10 * SECONDS_IN_YEAR + 3 * SECONDS_IN_MONTH + 4 * SECONDS_IN_HOUR, 3 * SECONDS_IN_MONTH + 4 * SECONDS_IN_HOUR, 4 * SECONDS_IN_HOUR], dtype=float);            # time = [10 years + 3 months + 4 hours; 3 months + 4 hours; 4 hours]. Time vector for the temporal superposition (s).   

# Create array containing arrays of integers with HP IDs for all pipe sections
IPGA = [np.asarray(I_PG.iloc[i].split(',')).astype(int) - 1 for i in range(N_PG)]
I_PG=IPGA                                                            # Redefine I_PG
del IPGA;                                                           # Get rid of IPGA

# Brine (fluid)
nu_f = mu_f/rho_f;                                                    # Brine kinematic viscosity (m2/s)  
a_f = l_f/(rho_f*c_f);                                                  # Brine thermal diffusivity (m2/s)  
Pr = nu_f/a_f;                                                       # Prandtl number (-)                

# Shallow soil (not for BHEs! - see below)
A = 7.900272987633280;                                              # Surface temperature amplitude (K) 
T0 = 9.028258373009810;                                             # Undisturbed soil temperature (C) 
omega = 2*np.pi/86400/365.25;                                           # Angular velocity of surface temperature variation (rad/s) 
a_s = l_s_H/rhoc_s; # KART potentielt et problem med to ledningsevner, her vælges bare den ene                                                    # Shallow soil thermal diffusivity (m2/s) - ONLY for pipes!!! 
# KART: følg op på brug af TP i forhold til bogen / gammel kode
TP = A*mt.exp(-z_grid*mt.sqrt(omega/2/a_s));                               # Temperature penalty at burial depth from surface temperature variation (K). Minimum undisturbed temperature is assumed . 

# Convert pipe diameter database to meters
d_pipes = d_pipes/1000;                                                 # Convert d_pipes from mm to m (m)

# Allocate variables
ind_H = np.zeros(N_PG);                                               # Index vector for pipe groups heating (-)
ind_C = np.zeros(N_PG);                                               # Index vector for pipe groups cooling (-)
d_selectedPipes_H = np.zeros(N_PG);                                           # Pipes selected from dimensioning for heating (length)
d_selectedPipes_C = np.zeros(N_PG);                                           # Pipes selected from dimensioning for cooling (length)
#P_s_H = np.zeros((N_HP,3)); Unødigt - variablen oprettes i line 226                                           # Thermal load from heating on the ground (W)
Q_PG_H = np.zeros(N_PG);                                               # Design flow heating (m3/s)
Q_PG_C = np.zeros(N_PG);                                               # Design flow cooling (m3/s)
R_H = np.zeros(N_PG);                                                 # Allocate pipe thermal resistance vector for heating (m*K/W)
R_C = np.zeros(N_PG);                                                 # Allocate pipe thermal resistance vector for cooling (m*K/W)
# KART bliv enige om sigende navne der følger konvention og implementer x 4
FPH = np.zeros(N_PG);                                                # Vector with total heating load fractions supplied by each pipe segment (-)
FPC = np.zeros(N_PG);                                                # Vector with total cooling load fractions supplied by each pipe segment (-)
GTHMH = np.zeros([N_PG,3]);
GTHMC = np.zeros([N_PG,3]);

# Simultaneity factors to apply to annual, monthly and hourly heating and cooling demands
S = np.zeros(3);
# KART: er der gået ged i index oversættelse fra Matlab hvor S(3) = (51 - NHP)*NHP^-0.5/NHP ? Følg op alle tre.
S[2] = SF*(0.62 + 0.38/N_HP);                                        # Hourly. Varme Ståbi. Ligning 3 i "Effekt- og samtidighedsforhold ved fjernvarmeforsyning af nye boligområder"
S[0]  = 1; #0.62 + 0.38/N_HP;                                        # Annual. Varme Ståbi. Ligning 3 i "Effekt- og samtidighedsforhold ved fjernvarmeforsyning af nye boligområder"
S[1]  = 1; #S(1);                                                   # Monthly. Varme Ståbi. Ligning 3 i "Effekt- og samtidighedsforhold ved fjernvarmeforsyning af nye boligområder"

# If horizontal heat exchangers are selected
if SS == 0:
    # Horizontal heat exchangers
    ri_HHE = d_HHE*(1 - 2/SDR_HHE)/2;                                  # Inner radius of HHE pipes (m)
    ro_HHE = d_HHE/2;                                                 # Outer radius of HHE pipes (m)

# If borehole heat exchangers are selected
if SS == 1:
    # BHE
    # KART: overvej ri_BHE, ligesom ri_HHE?
    ri = r_p*(1 - 2/SDR_BHE);                                         # Inner radius of U pipe (m)
    a_ss = l_ss/rhoc_ss;                                               # BHE soil thermal diffusivity (m2/s)
    a_g = l_g/rhoc_g;                                                  # Grout thermal diffusivity (W/m/K)
    # KART: eksponer mod bruger eller slet hvis den altid er samme som T0?
    T0BHE = T0;                                                     # Measured undisturbed BHE temperature (C)
    s_BHE = 2*r_p + D_BHEpipes;                                     # Calculate shank spacing U-pipe (m)

    # Borehole field
    x = np.linspace(0,NX-1,NX)*dx;                                  # x-coordinates of BHEs (m)                     
    y = np.linspace(0,NY-1,NY)*dy;                                  # y-coordinates of BHEs (m)
    N_BHE = NX*NY;                                                   # Number of BHEs (-)
    [XX,YY] = np.meshgrid(x,y);                                     # Meshgrid arrays for distance calculations (m)    
    Yv = np.concatenate(YY);                                        # YY concatenated (m)
    Xv = np.concatenate(XX);                                        # XX concatenated (m)
    
    # Logistics for symmetry considerations and associated efficiency gains
    # KART: har ikke tjekket
    NXi = int(np.ceil(NX/2));                                       # Find half the number of boreholes in the x-direction. If not an equal number then round up to complete symmetry.
    NYi = int(np.ceil(NY/2));                                       # Find half the number of boreholes in the y-direction. If not an equal number then round up to complete symmetry.
    w = np.ones((NYi,NXi));                                         # Define weight matrix for temperature responses at a distance (-)
    if np.mod(NX/2,1) > 0:                                          # If NX is an unequal integer then the weight on the temperature responses from the boreholes on the center line is equal to 0.5 for symmetry reasons
        w[:,NXi-1] = 0.5*w[:,NXi-1];
    
    if np.mod(NY/2,1) > 0:                                          # If NY is an unequal integer then the weight on the temperature responses from the boreholes on the center line is equal to 0.5 for symmetry reasons
        w[NYi-1,:] = 0.5*w[NYi-1,:];
        
    wv = np.concatenate(w);                                         # Concatenate the weight matrix (-)
    swv = sum(wv);                                                  # Sum all weights (-)
    xi = np.linspace(0,NXi-1,NXi)*dx;                               # x-coordinates of BHEs (m)                     
    yi = np.linspace(0,NYi-1,NYi)*dy;                               # y-coordinates of BHEs (m)
    [XXi,YYi] = np.meshgrid(xi,yi);                                 # Meshgrid arrays for distance calculations (m)
    Yvi = np.concatenate(YYi);                                      # YY concatenated (m)
    Xvi = np.concatenate(XXi);                                      # XX concatenated (m)
    
    # Solver settings for computing the flow and length corrected length of BHEs
    dL = 0.1;                                                       # Step length for trial trial solutions (m)
    LL = 10;                                                        # Additional length segment for which trial solutions are generated (m)

######### Precomputations and variables that should not be changed END ########

################################# Pipe sizing #################################

# Convert thermal load profile on HPs to flow rates
P_s_H = ps(S*HPS[:,1:4],HPS[:,4:7]);                                  # Annual (0), monthly (1) and daily (2) thermal load on the ground (W)
P_s_H[:,0] = P_s_H[:,0] - CPS[:,0];                                     # Annual imbalance between heating and cooling, positive for heating (W)
Qdim_H = P_s_H[:,2]/HPS[:,7]/rho_f/c_f;                                  # Design flow heating (m3/s)
Qdim_C = CPS[:,2]/CPS[:,4]/rho_f/c_f;                             # Design flow cooling (m3/s). Using simultaneity factor!
HPS = np.c_[HPS,Qdim_H];                                             # Append to heat pump data structure for heating
CPS = np.c_[CPS,Qdim_C];                                             # Append to heat pump data structure for cooling

# Heat pump and temperature conditions in the sizing equation
To_H = Ti_H - sum(Qdim_H*HPS[:,7])/sum(Qdim_H);                         # Volumetric flow rate weighted average brine delta-T (C)
TCH1 = T0 - (Ti_H + To_H)/2 - TP;                                     # Temperature condition for with heating termonet. Eq. 2.19 Advances in GSHP systems. Tp in the book refers to the influence from adjacent BHEs. This effect ignored in this tool.
To_C = Ti_C + sum(Qdim_C*CPS[:,4])/sum(Qdim_C);                         # Volumetric flow rate weighted average brine delta-T (C)
TCC1 = (Ti_C + To_C)/2 - T0 - TP;                                     # Temperature condition for with heating termonet. Eq. 2.19 Advances in GSHP systems. Tp in the book refers to the influence from adjacent BHEs. This effect ignored in this tool.                                                    
    
# Compute flow and pressure loss in BHEs and HHEs under peak load conditions. Temperature conditions are computed as well.
if SS == 0:
    # HHE heating
    Q_HHEmax_H = sum(Qdim_H)/N_HHE;                                        # Peak flow in HHE pipes (m3/s)
    v_HHEmax_H = Q_HHEmax_H/np.pi/ri_HHE**2;                                   # Peak flow velocity in HHE pipes (m/s)
    Re_HHEmax_H = Re(rho_f,mu_f,v_HHEmax_H,2*ri_HHE);                           # Peak Reynolds numbers in HHE pipes (-)
    dpdL_HHEmax_H = dp(rho_f,mu_f,Q_HHEmax_H,2*ri_HHE);                            # Peak pressure loss in HHE pipes (Pa/m)

    # HHE cooling
    Q_HHEmax_C = sum(Qdim_C)/N_HHE;                                        # Peak flow in HHE pipes (m3/s)
    v_HHEmax_C = Q_HHEmax_C/np.pi/ri_HHE**2;                                   # Peak flow velocity in HHE pipes (m/s)
    Re_HHEmax_C = Re(rho_f,mu_f,v_HHEmax_C,2*ri_HHE);                           # Peak Reynolds numbers in HHE pipes (-)
    dpdL_HHEmax_C = dp(rho_f,mu_f,Q_HHEmax_C,2*ri_HHE);                            # Peak pressure loss in HHE pipes (Pa/m)

if SS == 1:
    TCH2 = T0BHE - (Ti_H + To_H)/2;                                   # Temperature condition for heating with BHE. Eq. 2.19 Advances in GSHP systems but surface temperature penalty is removed from the criterion as it doesn't apply to BHEs (C)
    TCC2 = (Ti_C + To_C)/2 - T0BHE;                                   # Temperature condition for cooling with BHE. Eq. 2.19 Advances in GSHP systems but surface temperature penalty is removed from the criterion as it doesn't apply to BHEs (C)
    
    # BHE heating
    Q_BHEmax_H = sum(Qdim_H)/N_BHE;                                        # Peak flow in BHE pipes (m3/s)
    v_BHEmax_H = Q_BHEmax_H/np.pi/ri**2;                                      # Flow velocity in BHEs (m/s)
    Re_BHEmax_H = Re(rho_f,mu_f,v_BHEmax_H,2*ri);                              # Reynold number in BHEs (-)
    dpdL_BHEmax_H = dp(rho_f,mu_f,Q_BHEmax_H,2*ri);                               # Pressure loss in BHE (Pa/m)
    
    # BHE cooling
    Q_BHEmax_C = sum(Qdim_C)/N_BHE;                                        # Peak flow in BHE pipes (m3/s)
    v_BHEmax_C = Q_BHEmax_C/np.pi/ri**2;                                      # Flow velocity in BHEs (m/s)
    Re_BHEmax_C = Re(rho_f,mu_f,v_BHEmax_C,2*ri);                              # Reynold number in BHEs (-)
    dpdL_BHEmax_C = dp(rho_f,mu_f,Q_BHEmax_C,2*ri);                               # Pressure loss in BHE (Pa/m)

# Compute design flow for the pipes
for i in range(N_PG):
   Q_PG_H[i]=sum(HPS[np.ndarray.tolist(I_PG[i]),8])/TOPO_H[i,2];        # Sum the heating brine flow for all consumers connected to a specific pipe group and normalize with the number of traces in that group to get flow in the individual pipes (m3/s)
   Q_PG_C[i]=sum(CPS[np.ndarray.tolist(I_PG[i]),5])/TOPO_C[i,2];        # Sum the cooling brine flow for all consumers connected to a specific pipe group and normalize with the number of traces in that group to get flow in the individual pipes (m3/s)

# Select the smallest diameter pipe that fulfills the pressure drop criterion
for i in range(N_PG):                                 
    di_pipes = d_pipes*(1-2/TOPO_H[i,0]);                                # Compute inner diameters (m). Variable TOPO_H or TOPO_C are identical here.
    ind_H[i] = np.argmax(dp(rho_f,mu_f,Q_PG_H[i],di_pipes)<dpdL_t);           # Find first pipe with a pressure loss less than the target for heating (-)
    ind_C[i] = np.argmax(dp(rho_f,mu_f,Q_PG_C[i],di_pipes)<dpdL_t);           # Find first pipe with a pressure loss less than the target for cooling (-)
    d_selectedPipes_H[i] = d_pipes[int(ind_H[i])];                              # Store pipe selection for heating in new variable (m)
    d_selectedPipes_C[i] = d_pipes[int(ind_C[i])];                              # Store pipe selection for cooling in new variable (m)
ind_H = ind_H.astype(int);                            
ind_C = ind_C.astype(int);

# Compute Reynolds number for selected pipes for heating
di_selected_H = d_selectedPipes_H*(1-2/TOPO_H[:,0]);                                 # Compute inner diameter of selected pipes (m)
v_H = Q_PG_H/np.pi/di_selected_H**2*4;                                        # Compute flow velocity for selected pipes (m/s)
Re_selected_H = Re(rho_f,mu_f,v_H,di_selected_H);                                      # Compute Reynolds numbers for the selected pipes (-)

# Compute Reynolds number for selected pipes for cooling
di_selected_C = d_selectedPipes_C*(1-2/TOPO_C[:,0]);                                 # Compute inner diameter of selected pipes (m)
v_C = Q_PG_C/np.pi/di_selected_C**2*4;                                        # Compute flow velocity for selected pipes (m/s)
Re_selected_C = Re(rho_f,mu_f,v_C,di_selected_C);                                      # Compute Reynolds numbers for the selected pipes (-)

# Output the pipe sizing
print(' ');
print('******************* Suggested pipe dimensions heating ******************'); 
for i in range(N_PG):
    print(f'{pipeGroupNames.iloc[i]}: Ø{int(1000*d_selectedPipes_H[i])} mm SDR {int(TOPO_H[i,0])}, Re = {int(round(Re_selected_H[i]))}');
print(' ');
print('******************* Suggested pipe dimensions cooling ******************');
for i in range(N_PG):
    print(f'{pipeGroupNames.iloc[i]}: Ø{int(1000*d_selectedPipes_C[i])} mm SDR {int(TOPO_C[i,0])}, Re = {int(round(Re_selected_C[i]))}');
print(' ');

############################### Pipe sizing END ###############################

################## Compute temperature response of thermonet ##################

# Compute thermal resistances for pipes in heating mode
LENGTHS = 2*TOPO_H[:,1]*TOPO_H[:,2];                                  # Total lengths of different pipe segments (m)
TLENGTH = sum(LENGTHS);                                             # Total length of termonet (m)
TOPO_H = np.c_[TOPO_H,d_selectedPipes_H,Re_selected_H,LENGTHS];                         # Add pipe selection diameters (m), Reynolds numbers (-) and lengths as columns to the TOPO array
for i in range(N_PG):                                                # For all pipe groups
    R_H[i] = Rp(di_selected_H[i],d_selectedPipes_H[i],Re_selected_H[i],Pr,l_f,l_p);             # Compute thermal resistances (m*K/W)
TOPO_H = np.c_[TOPO_H, R_H];                                           # Append thermal resistances to pipe groups as a column in TOPO (m*K/W)

# Compute thermal resistances for pipes in cooling mode
TOPO_C = np.c_[TOPO_C,d_selectedPipes_C,Re_selected_C,LENGTHS];                         # Add pipe selection diameters (m), Reynolds numbers (-) and lengths as columns to the TOPO array
for i in range(N_PG):                                                # For all pipe groups
    R_C[i] = Rp(di_selected_C[i],d_selectedPipes_C[i],Re_selected_C[i],Pr,l_f,l_p);             # Compute thermal resistances (m*K/W)
TOPO_C = np.c_[TOPO_C, R_C];                                           # Append thermal resistances to pipe groups as a column in TOPO (m*K/W)

# Compute delta-qs for superposition of heating load responses
dPSH = np.zeros((N_HP,3));                                           # Allocate power difference matrix for tempoeral superposition (W)
dPSH[:,0] = P_s_H[:,0];                                               # First entry is just the annual average power (W)
dPSH[:,1:] = np.diff(P_s_H);                                          # Differences between year-month and month-hour are added (W)
cdPSH = np.sum(dPSH,0);

# Compute delta-qs for superposition of cooling load responses
dPSC = np.zeros((N_HP,3));                                           # Allocate power difference matrix for tempoeral superposition (W)
dPSC = np.c_[-P_s_H[:,0],CPS[:,1:3]];
dPSC[:,1:] = np.diff(dPSC);                                         # Differences between year-month and month-hour are added (W)
cdPSC = np.sum(dPSC,0);

# Compute temperature responses in heating and cooling mode for all pipes
K1 = ils(a_s,t,D_gridpipes) - ils(a_s,t,2*z_grid) - ils(a_s,t,np.sqrt(D_gridpipes**2+4*z_grid**2));
for i in range(N_PG):
    GTHMH[i,:] = CSM(d_selectedPipes_H[i]/2,d_selectedPipes_H[i]/2,t,a_s) + K1;
    GTHMC[i,:] = CSM(d_selectedPipes_C[i]/2,d_selectedPipes_C[i]/2,t,a_s) + K1;
    FPH[i] = TCH1*LENGTHS[i]/np.dot(cdPSH,GTHMH[i]/l_s_H + R_H[i]);    # Fraction of total heating that can be supplied by the i'th pipe segment (-)
    FPC[i] = TCC1*LENGTHS[i]/np.dot(cdPSC,GTHMC[i]/l_s_C + R_C[i]);    # Fraction of total heating that can be supplied by the i'th pipe segment (-)

# Heating supplied by thermonet 
FPH = sum(FPH);                                                     # Total fraction of heating supplied by thermonet (-)
PHEH = (1-FPH)*cdPSH;                                               # Residual heat demand (W)

# Cooling supplied by thermonet
FPC = sum(FPC);                                                     # Total fraction of cooling supplied by thermonet (-)
PHEC = (1-FPC)*cdPSC;                                               # Residual heat demand (W)

########################## Display results in console #########################
print('***************** Thermonet energy production capacity *****************'); 
print(f'The thermonet supplies {round(100*FPH)}% of the peak heating demand');  #print(f'The thermonet fully supplies the heat pumps with IDs 1 - {int(np.floor(NSHPH+1))} with heating' ) ;
print(f'The thermonet supplies {round(100*FPC)}% of the peak cooling demand');  
print(' ');
######################## Display results in console END #######################

################################ Source sizing ################################

# If BHEs are selected as source
if SS == 1:                                     
    ###########################################################################
    ############################ Borehole computation #########################
    ###########################################################################

    ######################### Generate G-functions ############################
    GBHE = CSM(r_b,r_b,t[0:2],a_ss);                                   # Compute g-functions for t[0] and t[1] with the cylindrical source model (-)
    s1 = 0;                                                         # Summation variable for t[0] G-function (-)
    s2 = 0;                                                         # Summation variable for t[1] G-function (-)
    for i in range(NXi*NYi):                                        # Line source superposition for all neighbour boreholes for 1/4 of the BHE field (symmetry)
        DIST = np.sqrt((XX-Xvi[i])**2 + (YY-Yvi[i])**2);            # Compute distance matrix (to neighbour boreholes) (m)
        DIST = DIST[DIST>0];                                        # Exclude the considered borehole to avoid r = 0 m
        s1 = s1 + wv[i]*sum(ils(a_ss,t[0],DIST));                    # Compute the sum of all thermal disturbances from neighbour boreholes (G-function contributions) for t[0] (-)
        s2 = s2 + wv[i]*sum(ils(a_ss,t[1],DIST));                    # Compute the sum of all thermal disturbances from neighbour boreholes (G-function contributions) for t[0] (-)
    GBHE[0] = GBHE[0] + s1/swv;                                     # Add the average neighbour contribution to the borehole field G-function for t[0] (-)
    GBHE[1] = GBHE[1] + s2/swv;                                     # Add the average neighbour contribution to the borehole field G-function for t[1] (-)

    # Compute borehole resistance with the first order multipole method ignoring flow and length effects
    Rbh = RbMP(l_f,l_p,l_g,l_ss,r_b,r_p,ri,s_BHE,Re_BHEmax_H,Pr);                # Compute the borehole thermal resistance (m*K/W)
    Rbc = RbMP(l_f,l_p,l_g,l_ss,r_b,r_p,ri,s_BHE,Re_BHEmax_C,Pr);                # Compute the borehole thermal resistance (m*K/W)
    #Rb = 0.12;                                                     # TRT estimate can be supplied instread (m*K/W)

    # Composite cylindrical source model GCLS() for short term response. Hu et al. 2014. Paper here: https://www.sciencedirect.com/science/article/abs/pii/S0378778814005866?via#3Dihub
    reh = r_b/np.exp(2*np.pi*l_g*Rbh);                                # Heating: Compute the equivalent pipe radius for cylindrical symmetry (m). This is how Hu et al. 2014 define it.
    rec = r_b/np.exp(2*np.pi*l_g*Rbc);                                # Cooling: Compute the equivalent pipe radius for cylindrical symmetry (m). This is how Hu et al. 2014 define it.

    # The Fourier numbers Fo1-Fo3 are neccesary for computing the solution 
    Fo1 = a_ss*t[2]/r_b**2;                                    
    G1 = GCLS(Fo1); 

    Fo2h = a_g*t[2]/reh**2;
    G2h = GCLS(Fo2h);

    Fo2c = a_g*t[2]/rec**2;
    G2c = GCLS(Fo2c);

    Fo3 = a_g*t[2]/r_b**2;
    G3 = GCLS(Fo3);

    Rwh = G1/l_ss + G2h/l_g - G3/l_g;                                  # Step response for short term model on the form q*Rw = T (m*K/W). Rw indicates that it is in fact a thermal resistance
    Rwc = G1/l_ss + G2c/l_g - G3/l_g;                                  # Step response for short term model on the form q*Rw = T (m*K/W). Rw indicates that it is in fact a thermal resistance

    # Compute approximate combined length of BHES (length effects not considered)
    GBHEF = GBHE;                                                   # Retain a copy of the G function for length correction later on (-)
    GBHEH = np.asarray([GBHE[0]/l_ss+Rbh,GBHE[1]/l_ss+Rbh, Rwh]);     # Heating G-function
    GBHEC = np.asarray([GBHE[0]/l_ss+Rbc,GBHE[1]/l_ss+Rbc, Rwc]);     # Cooling G-function
    LBHEH = np.dot(PHEH,GBHEH/TCH2);                                # Sizing equation for computing the required borehole meters for heating (m)
    LBHEC = np.dot(PHEC,GBHEC/TCC2);                                # Sizing equation for computing the required borehole meters for cooling (m)
    
    # Determine the solution by searching the neighbourhood of the approximate length solution
    # Heating mode
    LBHEHv = LBHEH/N_BHE + np.arange(0,LL,dL);
    NLBHEHv = len(LBHEHv);
    Rbhv = np.zeros(NLBHEHv);
    Tsolh = np.zeros(NLBHEHv);
    
    # Cooling mode
    LBHECv = LBHEC/N_BHE + np.arange(0,LL,dL);
    NLBHECv = len(LBHECv);
    Rbcv = np.zeros(NLBHECv);
    Tsolc = np.zeros(NLBHECv);
    
    for i in range(NLBHEHv):                                         # Compute Rb for the specified number of boreholes and lengths considering flow and length effects (m*K/W)
        Rbhv[i] = RbMPflc(l_f,l_p,l_g,l_ss,rho_f,c_f,r_b,r_p,ri,LBHEHv[i],s_BHE,Q_BHEmax_H,Re_BHEmax_H,Pr);    # Compute BHE length and flow corrected multipole estimates of Rb for all candidate solutions (m*K/W)
        Tsolh[i] = np.dot(PHEH,np.array([GBHEF[0]/l_ss + Rbhv[i], GBHEF[1]/l_ss + Rbhv[i], Rwh]))/LBHEHv[i]/N_BHE;                             #OK. Use Spitlers sizing formula for computing the corresponding temperature response for all candidate solutions (C)
    indLBHEH = np.argmax(Tsolh<TCH2);                                # Get rid of candidates that undersize the system. 
    LBHEH = LBHEHv[indLBHEH]*N_BHE;                                   # Solution to BHE length for heating (m)
    
    if (Tsolh[indLBHEH]-TCH2) > 0.1:
        print('Warning - the length steps used for computing the BHE length for heating are too big. Reduce the stepsize and recompute a solution.');
    
    for i in range(NLBHECv):                                         # Compute Rb for the specified number of boreholes and lengths considering flow and length effects (m*K/W)
        Rbcv[i] = RbMPflc(l_f,l_p,l_g,l_ss,rho_f,c_f,r_b,r_p,ri,LBHECv[i],s_BHE,Q_BHEmax_C,Re_BHEmax_C,Pr);    #K. Compute BHE length and flow corrected multipole estimates of Rb for all candidate solutions (m*K/W)
        Tsolc[i] = np.dot(PHEC,np.array([GBHEF[0]/l_ss + Rbcv[i], GBHEF[1]/l_ss + Rbcv[i], Rwc]))/LBHECv[i]/N_BHE;                             #OK. Use Spitlers sizing formula for computing the corresponding temperature response for all candidate solutions (C)
    indLBHEC = np.argmax(Tsolc<TCC2);                                # Get rid of candidates that undersize the system. 
    LBHEC = LBHECv[indLBHEC]*N_BHE;                                   # Solution BHE length for cooling (m)
    
    if (Tsolc[indLBHEC]-TCC2) > 0.1:
        print('Warning - the length steps used for computing the BHE length for cooling are too big. Reduce the stepsize and recompute a solution.');    
    
    # Display output in console
    print('********** Suggested length of borehole heat exchangers (BHE) **********'); 
    print(f'Required length of each of the {int(N_BHE)} BHEs = {int(np.ceil(LBHEH/N_BHE))} m for heating');
    print(f'Required length of each of the {int(N_BHE)} BHEs = {int(np.ceil(LBHEC/N_BHE))} m for cooling');
    print(f'Maximum pressure loss in BHEs in heating mode = {int(np.ceil(dpdL_BHEmax_H))} Pa/m, Re = {int(round(Re_BHEmax_H))}');
    print(f'Maximum pressure loss in BHEs in cooling mode = {int(np.ceil(dpdL_BHEmax_C))} Pa/m, Re = {int(round(Re_BHEmax_C))}');

# If HHEs are selected as source
if SS == 0:
    ###########################################################################
    ############################### HHE computation ###########################
    ###########################################################################

    # Compute combined length of HHEs   
    ind = np.linspace(0,2*N_HHE-1,2*N_HHE);                           # Unit distance vector for HHE (-)
    s = np.zeros(2);                                                # s is a temperature summation variable, s[0]: annual, s[1] monthly, hourly effects are insignificant and ignored (C)
    DIST = D_HHE*ind;                                                  # Distance vector for HHE (m)
    for i in range(N_HHE):                                           # For half the pipe segments (2 per loop). Advantage from symmetry.
        s[0] = s[0] + sum(ils(a_s,t[0],abs(DIST[ind!=i]-i*D_HHE))) - sum(ils(a_s,t[0],np.sqrt((DIST-i*D_HHE)**2 + 4*z_grid**2))); # Sum annual temperature responses from distant pipes (C)
        s[1] = s[1] + sum(ils(a_s,t[1],abs(DIST[ind!=i]-i*D_HHE))) - sum(ils(a_s,t[1],np.sqrt((DIST-i*D_HHE)**2 + 4*z_grid**2))); # Sum monthly temperature responses from distant pipes (C)
    GHHE = CSM(ro_HHE,ro_HHE,t,a_s);                                  # Pipe wall response (-)
    GHHE[0:2] = GHHE[0:2] + s/N_HHE;                                 # Add thermal disturbance from neighbour pipes (-)
    
    # Heating
    RHHEH = Rp(2*ri_HHE,2*ro_HHE,Re_HHEmax_H,Pr,l_f,l_p);                   # Compute the pipe thermal resistance (m*K/W)
    GHHEH = GHHE/l_s_H+RHHEH;                                         # Add annual and monthly thermal resistances to GHHE (m*K/W)
    LHHEH = np.dot(PHEH,GHHEH/TCH1);                                 # Sizing equation for computing the required borehole meters (m)
    
    # Cooling
    RHHEC = Rp(2*ri_HHE,2*ro_HHE,Re_HHEmax_C,Pr,l_f,l_p);                   # Compute the pipe thermal resistance (m*K/W)
    GHHEC = GHHE/l_s_C+RHHEC;                                         # Add annual and monthly thermal resistances to GHHE (m*K/W)
    LHHEC = np.dot(PHEC,GHHEC/TCC1);                                 # Sizing equation for computing the required borehole meters (m)
    
    # Output results to console
    print('********* Suggested length of horizontal heat exchangers (HHE) *********');
    print(f'Required length of each of the {int(N_HHE)} horizontal loops = {int(np.ceil(LHHEH/N_HHE))} m for heating');
    print(f'Required length of each of the {int(N_HHE)} horizontal loops = {int(np.ceil(LHHEC/N_HHE))} m for cooling');
    print(f'Maximum pressure loss in HHE pipes in heating mode = {int(np.ceil(dpdL_HHEmax_H))} Pa/m, Re = {int(round(Re_HHEmax_H))}');
    print(f'Maximum pressure loss in HHE pipes in cooling mode {int(np.ceil(dpdL_HHEmax_C))} Pa/m, Re = {int(round(Re_HHEmax_C))}');
    
############################## Source sizing END ##############################

# Output computation time to console
print(' ');
print('*************************** Computation time ***************************');
toc = time.time();                                                  # Track computation time (s)
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
#       x : mirror source, opposite sign of real source
#       o : real source, actual pipe
#     --- : the ground surface where T = 0. Actual ground temperatures are then superimposed.
#       
#       T(o1) = q*(R(o1) + R(o2) - R(x1) - R(x2)) + Tu(t)
#       Tu(t) is the undisturbed seasonal temperature variation at depth
#       Assumption: surface temperature equal to the undisturbed seasonal temperature (Dirichlet BC)
#
#
############## Conceptual model for twin pipe in the ground END ###############

################### Conceptual model for HHE in the ground ####################

# Topology of horizontal heat exchangers (N_HHE = 3)
# |  Loop  |	    |  Loop  |	      |  Loop  |
# |	       |	    |	     |	      |	       |
# |	       |  	    |	     |	      |	       |
# |<D_HHE->|<D_HHE->|<D_HHE->|<D_HHE->|<D_HHE->|
# |        |   	    |        |	      |        |
# |        |   	    |        |	      |        |
# |________|   	    |________|        |________|
#
# 
# Mirror sources (above the ground surfaces) enforce Dirichlet BC on ground surface - similar to thermonet model

####################### Conceptual model for BHE field ########################
#
# Only compute the average temperature response for one of the four sub-rectangles below as there is symmetry between them
#   
#        weight = 1
#            o           o     |     o           o
#                              |
#            o           o     |     o           o
#                              |
#            o           o     |     o           o
#                              |
# NY=7 ------o-----------o-----------o-----------o----- weight = 0.5 (if both NX and NY are unequal, then the center BHE has a weight of 0.25)    
#                              |
#            o           o     |     o           o         
#                              |
#            o           o     |     o           o
#                              |
#            o           o     |     o           o
#                            
#                            NX=4
#
#       Legend
#       o : BHE
#       -- or | : axes of symmetry
################# Conceptual model for HHE in the ground END ##################