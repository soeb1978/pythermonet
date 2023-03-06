# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 09:36:52 2023

TODO

- bøvl med isinstance på egne classes -> forløbig fix med str (snak m Lasse til workshop -> lav kort script)

DISKUTER


- Qdim tilføjes i HPS hhv CPS bæres der videre i beregning - OK?

- Ryd op i håndtering af EER og COP - se ca Line 75.
    * Vi forstår køle-værdier i HPSC.dat som bygningslast ikke? -> jordlasten er større
    * Håndtering af EER: enheder + se ASHRAE p. 489 + egen note
    * hvor har vi EER værdier fra?

- Skal vi udvide Heatpump klassen og flytte værdier fra HPS og CPS ind i den for mere ensartet kode?

- Validering
    * Overvej om vi skal indføre en function der beregner Tfluid som en del af BHE-beregning -> denne del kan valideres mod kendte løsninger

@author: KART
"""

import numpy as np
import pandas as pd
from thermonet_classes import Brine, Thermonet, Heatpump, HHEconfig, BHEconfig
from dimensioning_functions import run_pipedimensioning, run_sourcedimensioning
import time

# Inputs
# Project ID
PID = 'Energiakademiet, Samsø';                                     # Project name


# Output to prompt
print(' ');
print('************************************************************************')
print('************************** ThermonetDim v. 1 ************************')
print('************************************************************************')
print(' ');
print(f'Project: {PID}');

# User specified input

# Brine - with default parameters
brine = Brine(rho=965, c=4450, mu=5e-3, l=0.45);
# Thermonet - with default parameters
net = Thermonet(D_gridpipes=0.3, dpdL_t=90, l_p=0.4, l_s_H=1.25, l_s_C=1.25, rhoc_s=2.5e6, z_grid=1.2);
# Heat pump - with default parameters
hp = Heatpump(Ti_H=-3, Ti_C=20, SF=1);
# Heat source (either BHE or HHE) - with default parameters

# source_config = HHEconfig(N_HHE=6, d=0.04, SDR=17, D=1.5)
source_config = BHEconfig(r_b=0.152/2, r_p=0.02, SDR=11, l_ss=2.36, rhoc_ss=2.65e6, l_g=1.75, rhoc_g=3e6, D_pipes=0.015, NX=1, D_x=15, NY=6, D_y=15);


# Input files
HP_file = 'Silkeborg_HPSC.dat';                                            # Input file containing heat pump information
TOPO_file = 'Silkeborg_TOPO.dat';                                          # Input file containing topology information 

# Load heat pump data
HPS = pd.read_csv(HP_file, sep = '\t+', engine='python');                                # Heat pump input file
HPS = HPS.values  # Load numeric data from HP file
CPS = HPS[:, 8:]  # Place cooling demand data in separate array
HPS = HPS[:, :8]  # Remove cooling demand data from HPS array
# Add circulation pump power consumption to cooling load (W)
# KART: til dokumentation Qcool -> (EER/(EER-1))*Qcool enig?
# KART den tilsvarende beregning for varme ligger ved kald til ps() inde i run_pipedimesnioning
# Det bør ensrettes så det håndteres på samme måde
CPS[:, :3] = CPS[:, 3:4] / (CPS[:, 3:4] - 1) * CPS[:, :3]                                            


# Load grid topology
TOPO = np.loadtxt(TOPO_file,skiprows = 1,usecols = (1,2,3));          # Load numeric data from topology file
net.SDR = TOPO[:,0];
net.L_traces = TOPO[:,1];
net.N_traces = TOPO[:,2];
net.L_segments = 2 * net.L_traces * net.N_traces;



I_PG = pd.read_csv(TOPO_file, sep = '\t+', engine='python');                              # Load the entire file into Panda dataframe
pipeGroupNames = I_PG.iloc[:,0];                                             # Extract pipe group IDs
I_PG = I_PG.iloc[:,4];                                                # Extract IDs of HPs connected to the different pipe groups

# Load pipe database
d_pipes = pd.read_csv('PIPES.dat', sep = '\t');                       # Open file with available pipe outer diameters (mm). This file can be expanded with additional pipes and used directly.
d_pipes = d_pipes.values;                                               # Get numerical values from pipes excluding the headers
# Convert pipe diameter database to meters
d_pipes = d_pipes/1000;                                                 # Convert d_pipes from mm to m (m)



# Record calculation time    
tic = time.time();

# Run pipe dimensioning                                                    # Track computation time (s)                                                           
# HPS, CPS, P_s_H, d_selectedPipes_H, di_selected_H, Re_selected_H, d_selectedPipes_C, di_selected_C, Re_selected_C = run_pipedimensioning(HPS, CPS, TOPO, I_PG, d_pipes, brine, net, hp)
net, HPS, CPS, P_s_H = run_pipedimensioning(HPS, CPS, I_PG, d_pipes, brine, net, hp)


# Print pipe dimensioning results
print(' ');
print('******************* Suggested pipe dimensions heating ******************'); 
for i in range(len(I_PG)):
    print(f'{pipeGroupNames.iloc[i]}: Ø{int(1000*net.d_selectedPipes_H[i])} mm SDR {int(net.SDR[i])}, Re = {int(round(net.Re_selected_H[i]))}');
print(' ');
print('******************* Suggested pipe dimensions cooling ******************');
for i in range(len(I_PG)):
    print(f'{pipeGroupNames.iloc[i]}: Ø{int(1000*net.d_selectedPipes_C[i])} mm SDR {int(net.SDR[i])}, Re = {int(round(net.Re_selected_C[i]))}');
print(' ');


# Run source dimensioning
# FPH, FPC, source_config = run_sourcedimensioning(P_s_H, HPS, CPS, TOPO_H, TOPO_C, I_PG, d_selectedPipes_H, di_selected_H, Re_selected_H, d_selectedPipes_C, di_selected_C, Re_selected_C, brine, net, hp, source_config)
FPH, FPC, source_config = run_sourcedimensioning(P_s_H, HPS, CPS, I_PG, brine, net, hp, source_config)

# Print results to console
print('***************** Thermonet energy production capacity *****************'); 
print(f'The thermonet supplies {round(100*FPH)}% of the peak heating demand');  #print(f'The thermonet fully supplies the heat pumps with IDs 1 - {int(np.floor(NSHPH+1))} with heating' ) ;
print(f'The thermonet supplies {round(100*FPC)}% of the peak cooling demand');  
print(' ');

# BHE specific results
if source_config.source == 'BHE':
    N_BHE = source_config.NX * source_config.NY;
    
    # Display output in console
    print('********** Suggested length of borehole heat exchangers (BHE) **********'); 
    print(f'Required length of each of the {int(N_BHE)} BHEs = {int(np.ceil(source_config.L_BHE_H/N_BHE))} m for heating');
    print(f'Required length of each of the {int(N_BHE)} BHEs = {int(np.ceil(source_config.L_BHE_C/N_BHE))} m for cooling');
    print(f'Maximum pressure loss in BHEs in heating mode = {int(np.ceil(source_config.dpdL_BHEmax_H))} Pa/m, Re = {int(round(source_config.Re_BHEmax_H))}');
    print(f'Maximum pressure loss in BHEs in cooling mode = {int(np.ceil(source_config.dpdL_BHEmax_C))} Pa/m, Re = {int(round(source_config.Re_BHEmax_C))}');

elif source_config.source =='HHE':

    N_HHE = source_config.N_HHE;
   
    
    
    # Output results to console
    print('********* Suggested length of horizontal heat exchangers (HHE) *********');
    print(f'Required length of each of the {int(N_HHE)} horizontal loops = {int(np.ceil(source_config.L_HHE_H/N_HHE))} m for heating');
    print(f'Required length of each of the {int(N_HHE)} horizontal loops = {int(np.ceil(source_config.L_HHE_C/N_HHE))} m for cooling');
    print(f'Maximum pressure loss in HHE pipes in heating mode = {int(np.ceil(source_config.dpdL_HHEmax_H))} Pa/m, Re = {int(round(source_config.Re_HHEmax_H))}');
    print(f'Maximum pressure loss in HHE pipes in cooling mode {int(np.ceil(source_config.dpdL_HHEmax_C))} Pa/m, Re = {int(round(source_config.Re_HHEmax_C))}');


# Output computation time to console
print(' ');
print('*************************** Computation time ***************************');
toc = time.time();                                                  # Track computation time (s)
print(f'Elapsed time: {round(toc-tic,6)} seconds');
