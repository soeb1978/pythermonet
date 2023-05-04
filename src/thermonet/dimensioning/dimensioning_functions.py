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
from .fThermonetDim import ils, Re, dp, Rp, CSM, RbMP, GCLS, RbMPflc
from thermonet.dimensioning.thermonet_classes import aggregatedLoad



# Read heat pump data and calculate ground loads
def read_heatpumpdata(hp, HP_file):
    
    HPS = pd.read_csv(HP_file, sep = '\t+', engine='python');                                # Heat pump input file
    HPS = HPS.values  # Load numeric data from HP file

    hp.P_y_H = HPS[:,1];
    hp.P_m_H = HPS[:,2];
    hp.P_d_H = HPS[:,3];
    hp.COP_y_H = HPS[:,4];
    hp.COP_m_H = HPS[:,5];
    hp.COP_d_H = HPS[:,6];
    hp.dT_H = HPS[:,7];
    hp.P_y_C = HPS[:,8];
    hp.P_m_C = HPS[:,9];
    hp.P_d_C = HPS[:,10];
    hp.EER = HPS[:,11];
    hp.dT_C = HPS[:,12];
    
    
    # Calcualate ground loads from building loads
    N_HP = len(hp.P_y_H);
        
    # Simultaneity factor to apply to hourly heating
    #KART Hvad med køling?
    S = hp.SF*(0.62 + 0.38/N_HP);                   # Varme Ståbi. Ligning 3 i "Effekt- og samtidighedsforhold ved fjernvarmeforsyning af nye boligområder"
    
    # Convert building loads to ground loads - heating
    hp.P_s_H = np.zeros([N_HP,3]);
    hp.P_s_H[:,0] = (hp.COP_y_H-1)/hp.COP_y_H * hp.P_y_H; # Annual load (W)
    hp.P_s_H[:,1] = (hp.COP_m_H-1)/hp.COP_m_H * hp.P_m_H; # Monthly load (W)
    hp.P_s_H[:,2] = (hp.COP_d_H-1)/hp.COP_d_H * hp.P_d_H * S; # Daily load with simultaneity factor (W)
    
    # Convert building loads to ground loads - cooling
    # KART Samtidighedsfaktor?
    hp.P_s_C = np.zeros([N_HP,3]);
    hp.P_s_C[:,0] = hp.EER/(hp.EER - 1) * hp.P_y_C; # Annual load (W)
    hp.P_s_C[:,1] = hp.EER/(hp.EER - 1) * hp.P_m_C; # Monthly load (W)
    hp.P_s_C[:,2] = hp.EER/(hp.EER - 1) * hp.P_d_C * S; # Daily load (W)
    
    # Første søjle i hp.P_s_H hhv hp.P_s_C er ens pånær fortegn. 
    hp.P_s_H[:,0] = hp.P_s_H[:,0] - hp.P_s_C[:,0];                                       # Annual imbalance between heating and cooling, positive for heating (W)
    hp.P_s_C[:,0] = - hp.P_s_H[:,0];                                                  # Negative for cooling


    return hp


# Read topology data
def read_topology(net, TOPO_file):
    
    # Load grid topology
    TOPO = np.loadtxt(TOPO_file,skiprows = 1,usecols = (1,2,3));          # Load numeric data from topology file
    net.SDR = TOPO[:,0];
    net.L_traces = TOPO[:,1];
    net.N_traces = TOPO[:,2];
    net.L_segments = 2 * net.L_traces * net.N_traces;
    
    I_PG = pd.read_csv(TOPO_file, sep = '\t+', engine='python');                              # Load the entire file into Panda dataframe
    pipeGroupNames = I_PG.iloc[:,0];                                             # Extract pipe group IDs
    I_PG = I_PG.iloc[:,4];                                                # Extract IDs of HPs connected to the different pipe groups
    # Create array containing arrays of integers with HP IDs for all pipe sections
    IPGA = [np.asarray(I_PG.iloc[i].split(',')).astype(int) - 1 for i in range(len(I_PG))]
    I_PG = IPGA                                                            # Redefine I_PG
    net.I_PG = I_PG;
    del IPGA, I_PG;                                                           # Get rid of IPGA

    
    return net, pipeGroupNames


# Print results to console
def print_pipe_dimensions(net, pipeGroupNames):
    
    # Print pipe dimensioning results
    print(' ');
    print('******************* Suggested pipe dimensions heating ******************'); 
    for i in range(len(net.I_PG)):
        print(f'{pipeGroupNames.iloc[i]}: Ø{int(1000*net.d_selectedPipes_H[i])} mm SDR {int(net.SDR[i])}, Re = {int(round(net.Re_selected_H[i]))}');
    print(' ');
    print('******************* Suggested pipe dimensions cooling ******************');
    for i in range(len(net.I_PG)):
        print(f'{pipeGroupNames.iloc[i]}: Ø{int(1000*net.d_selectedPipes_C[i])} mm SDR {int(net.SDR[i])}, Re = {int(round(net.Re_selected_C[i]))}');
    print(' ');


# Print source dimensioning results to console
def print_source_dimensions(FPH, FPC, source_config):
    
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


# Function for dimensioning pipes
def run_pipedimensioning(d_pipes, brine, net, hp):
    
    N_PG = len(net.I_PG);                                                     # Number of pipe groups
    
    # Allocate variables
    ind_H = np.zeros(N_PG);                                               # Index vector for pipe groups heating (-)
    ind_C = np.zeros(N_PG);                                               # Index vector for pipe groups cooling (-)
    d_selectedPipes_H = np.zeros(N_PG);                                           # Pipes selected from dimensioning for heating (length)
    d_selectedPipes_C = np.zeros(N_PG);                                           # Pipes selected from dimensioning for cooling (length)
    Q_PG_H = np.zeros(N_PG);                                               # Design flow heating (m3/s)
    Q_PG_C = np.zeros(N_PG);                                               # Design flow cooling (m3/s)

        
    hp.Qdim_H = hp.P_s_H[:,2]/hp.dT_H/brine.rho/brine.c;                                  # Design flow heating (m3/s)
    hp.Qdim_C = hp.P_s_C[:,2]/hp.dT_C/brine.rho/brine.c;                             # Design flow cooling (m3/s). 

    # Compute design flow for the pipes
    for i in range(N_PG):
       Q_PG_H[i] = sum(hp.Qdim_H[np.ndarray.tolist(net.I_PG[i])])/net.N_traces[i];        # Sum the heating brine flow for all consumers connected to a specific pipe group and normalize with the number of traces in that group to get flow in the individual pipes (m3/s)
       Q_PG_C[i] = sum(hp.Qdim_C[np.ndarray.tolist(net.I_PG[i])])/net.N_traces[i];        # Sum the cooling brine flow for all consumers connected to a specific pipe group and normalize with the number of traces in that group to get flow in the individual pipes (m3/s)
    
    # Select the smallest diameter pipe that fulfills the pressure drop criterion
    for i in range(N_PG):                                 
        di_pipes = d_pipes*(1-2/net.SDR[i]);                                # Compute inner diameters (m). Variable TOPO_H or TOPO_C are identical here.
        ind_H[i] = np.argmax(dp(brine.rho,brine.mu,Q_PG_H[i],di_pipes)<net.dpdL_t);           # Find first pipe with a pressure loss less than the target for heating (-)
        ind_C[i] = np.argmax(dp(brine.rho,brine.mu,Q_PG_C[i],di_pipes)<net.dpdL_t);           # Find first pipe with a pressure loss less than the target for cooling (-)
        d_selectedPipes_H[i] = d_pipes[int(ind_H[i])];                              # Store pipe selection for heating in new variable (m)
        d_selectedPipes_C[i] = d_pipes[int(ind_C[i])]; 

    net.d_selectedPipes_H = d_selectedPipes_H;
    net.d_selectedPipes_C = d_selectedPipes_C;
                             # Store pipe selection for cooling in new variable (m)
    
    # Compute Reynolds number for selected pipes for heating
    net.di_selected_H = d_selectedPipes_H*(1-2/net.SDR);                                 # Compute inner diameter of selected pipes (m)
    v_H = Q_PG_H/np.pi/net.di_selected_H**2*4;                                        # Compute flow velocity for selected pipes (m/s)
    net.Re_selected_H = Re(brine.rho,brine.mu,v_H,net.di_selected_H);                                      # Compute Reynolds numbers for the selected pipes (-)
    
    # Compute Reynolds number for selected pipes for cooling
    net.di_selected_C = d_selectedPipes_C*(1-2/net.SDR);                                 # Compute inner diameter of selected pipes (m)
    v_C = Q_PG_C/np.pi/net.di_selected_C**2*4;                                        # Compute flow velocity for selected pipes (m/s)
    net.Re_selected_C = Re(brine.rho,brine.mu,v_C,net.di_selected_C);                                      # Compute Reynolds numbers for the selected pipes (-)
    
    
    
    
    
    
    aggLoad = aggregatedLoad();
    aggLoad.Ti_H = hp.Ti_H;
    aggLoad.Ti_C = hp.Ti_C;
    
    aggLoad.To_H = hp.Ti_H - sum(hp.Qdim_H*hp.dT_H)/sum(hp.Qdim_H);                         # Volumetric flow rate weighted average brine delta-T (C)
    aggLoad.To_C = hp.Ti_C + sum(hp.Qdim_C*hp.dT_C)/sum(hp.Qdim_C);                         # Volumetric flow rate weighted average brine delta-T (C)

    aggLoad.P_s_H = sum(hp.P_s_H);
    aggLoad.P_s_C = sum(hp.P_s_C);
    
    aggLoad.Qdim_H = sum(hp.Qdim_H);
    aggLoad.Qdim_C = sum(hp.Qdim_C);



    
    
    
    # Return the pipe sizing results
    return net, aggLoad

    ############################### Pipe sizing END ###############################
    
    
# Function for dimensioning sources
#KART fjern hp input i endelig version
# def run_sourcedimensioning(brine, net, hp, source_config):
def run_sourcedimensioning(brine, net, aggLoad, source_config):
    
    
    N_PG = len(net.I_PG);                                                     # Number of pipe groups
    # N_HP = len(hp.P_y_H);    
      
    # G-function evaluation times (DO NOT MODIFY!!!!!!!)
    SECONDS_IN_HOUR = 3600;
    SECONDS_IN_MONTH = 24 * (365/12) * SECONDS_IN_HOUR
    SECONDS_IN_YEAR = 12 * SECONDS_IN_MONTH;

    # t = [10y 3m 4h, 3m 4h, 4h]
    t = np.asarray([10 * SECONDS_IN_YEAR + 3 * SECONDS_IN_MONTH + 4 * SECONDS_IN_HOUR, 3 * SECONDS_IN_MONTH + 4 * SECONDS_IN_HOUR, 4 * SECONDS_IN_HOUR], dtype=float);            # time = [10 years + 3 months + 4 hours; 3 months + 4 hours; 4 hours]. Time vector for the temporal superposition (s).       
    
    # Brine (fluid)
    nu_f = brine.mu/brine.rho;                                                    # Brine kinematic viscosity (m2/s)  
    a_f = brine.l/(brine.rho*brine.c);                                                  # Brine thermal diffusivity (m2/s)  
    Pr = nu_f/a_f;                                                       # Prandtl number (-)                

    # Shallow soil (not for BHEs! - see below)
    A = 7.900272987633280;                                              # Surface temperature amplitude (K) 
    T0 = 9.028258373009810;                                             # Undisturbed soil temperature (C) 
    omega = 2*np.pi/86400/365.25;                                           # Angular velocity of surface temperature variation (rad/s) 
    a_s = net.l_s_H/net.rhoc_s; # KART potentielt et problem med to ledningsevner, her vælges bare den ene                                                    # Shallow soil thermal diffusivity (m2/s) - ONLY for pipes!!! 
    # KART: følg op på brug af TP i forhold til bogen / gammel kode
    TP = A*np.exp(-net.z_grid*np.sqrt(omega/2/a_s));                               # Temperature penalty at burial depth from surface temperature variation (K). Minimum undisturbed temperature is assumed . 

    # Compute thermal resistances for pipes in heating mode
    R_H = np.zeros(N_PG);                                                 # Allocate pipe thermal resistance vector for heating (m*K/W)
    for i in range(N_PG):                                                # For all pipe groups
        R_H[i] = Rp(net.di_selected_H[i],net.d_selectedPipes_H[i],net.Re_selected_H[i],Pr,brine.l,net.l_p);             # Compute thermal resistances (m*K/W)
     
    # Compute thermal resistances for pipes in cooling mode
    R_C = np.zeros(N_PG);                                                 # Allocate pipe thermal resistance vector for cooling (m*K/W)
    for i in range(N_PG):                                                # For all pipe groups
        R_C[i] = Rp(net.di_selected_C[i],net.d_selectedPipes_C[i],net.Re_selected_C[i],Pr,brine.l,net.l_p);             # Compute thermal resistances (m*K/W)

    
    # Compute delta-qs for superposition of heating load responses
    # KART: ændret til aggregeret last
    # dP_s_H = np.zeros((N_HP,3));                                           # Allocate power difference matrix for tempoeral superposition (W)
    # dP_s_H[:,0] = hp.P_s_H[:,0];                                               # First entry is just the annual average power (W)
    # dP_s_H[:,1:] = np.diff(hp.P_s_H);                                          # Differences between year-month and month-hour are added (W)
    dP_s_H = np.zeros(3);                                           # Allocate power difference matrix for tempoeral superposition (W)
    dP_s_H[0] = aggLoad.P_s_H[0];                                               # First entry is just the annual average power (W)
    dP_s_H[1:] = np.diff(aggLoad.P_s_H);                                          # Differences between year-month and month-hour are added (W)


    
    # KART: tjek beregning
    # KART flyttet aggregering
    # cdPSH = np.sum(dP_s_H,0);
    
    # Compute delta-qs for superposition of cooling load responses
    # dP_s_C = np.zeros((N_HP,3));                                           # Allocate power difference matrix for tempoeral superposition (W)
    # dP_s_C[:,0] = hp.P_s_C[:,0];                                               # First entry is just the annual average power (W)
    # dP_s_C[:,1:] = np.diff(hp.P_s_C);                                          # Differences between year-month and month-hour are added (W)
    dP_s_C = np.zeros(3);                                           # Allocate power difference matrix for tempoeral superposition (W)
    dP_s_C[0] = aggLoad.P_s_C[0];                                               # First entry is just the annual average power (W)
    dP_s_C[1:] = np.diff(aggLoad.P_s_C);                                          # Differences between year-month and month-hour are added (W)

    # KART: ditto køl
    # KART: flyttet aggregering
    # cdPSC = np.sum(dP_s_C,0);
    
    # Compute temperature responses in heating and cooling mode for all pipes
    # KART bliv enige om sigende navne der følger konvention og implementer x 4
    FPH = np.zeros(N_PG);                                                # Vector with total heating load fractions supplied by each pipe segment (-)
    FPC = np.zeros(N_PG);                                                # Vector with total cooling load fractions supplied by each pipe segment (-)
    GTHMH = np.zeros([N_PG,3]); #KART: G_grid_H
    GTHMC = np.zeros([N_PG,3]); #KART: G_grid_C
    
    
    # Heat pump and temperature conditions in the sizing equation
    # KART flyttet ind i pipedim
    # To_H = hp.Ti_H - sum(hp.Qdim_H*hp.dT_H)/sum(hp.Qdim_H);                         # Volumetric flow rate weighted average brine delta-T (C)
    # To_C = hp.Ti_C + sum(hp.Qdim_C*hp.dT_C)/sum(hp.Qdim_C);                         # Volumetric flow rate weighted average brine delta-T (C)
    
    K1 = ils(a_s,t,net.D_gridpipes) - ils(a_s,t,2*net.z_grid) - ils(a_s,t,np.sqrt(net.D_gridpipes**2+4*net.z_grid**2));
    # KART: gennemgå nye varmeberegning - opsplittet på segmenter
    for i in range(N_PG):
        GTHMH[i,:] = CSM(net.d_selectedPipes_H[i]/2,net.d_selectedPipes_H[i]/2,t,a_s) + K1;
        GTHMC[i,:] = CSM(net.d_selectedPipes_C[i]/2,net.d_selectedPipes_C[i]/2,t,a_s) + K1;
        # FPH[i] = (T0 - (aggLoad.Ti_H + aggLoad.To_H)/2 - TP)*net.L_segments[i]/np.dot(cdPSH,GTHMH[i]/net.l_s_H + R_H[i]);    # Fraction of total heating that can be supplied by the i'th pipe segment (-)
        # KART opdateret aggregering
        FPH[i] = (T0 - (aggLoad.Ti_H + aggLoad.To_H)/2 - TP)*net.L_segments[i]/np.dot(dP_s_H, GTHMH[i]/net.l_s_H + R_H[i]);    # Fraction of total heating that can be supplied by the i'th pipe segment (-)
        # KART opdateret aggregering
        FPC[i] = ((aggLoad.Ti_C + aggLoad.To_C)/2 - T0 - TP)*net.L_segments[i]/np.dot(dP_s_C, GTHMC[i]/net.l_s_C + R_C[i]);    # Fraction of total heating that can be supplied by the i'th pipe segment (-)
    
    # KART - mangler at gennemgå ny beregning af energi fra grid/kilder
    
    # Heating supplied by thermonet 
    FPH = sum(FPH);                                                     # Total fraction of heating supplied by thermonet (-)
# KART flyttet aggregering
    # PHEH = (1-FPH)*cdPSH;                                               # Residual heat demand (W)
    PHEH = (1-FPH)*dP_s_H;                                               # Residual heat demand (W)

    
    # Cooling supplied by thermonet
    FPC = sum(FPC);                                                     # Total fraction of cooling supplied by thermonet (-)
   # KART flyttet aggregering
    # PHEC = (1-FPC)*cdPSC;                                               # Residual heat demand (W)
    PHEC = (1-FPC)*dP_s_C;                                               # Residual heat demand (W)
    

    
    ################################ Source sizing ################################
    
    # If BHEs are selected as source
    if source_config.source == 'BHE':
        ###########################################################################
        ############################ Borehole computation #########################
        ###########################################################################
        
        # For readability only
        BHE = source_config;
        

        ri = BHE.r_p*(1 - 2/BHE.SDR);                                         # Inner radius of U pipe (m)
        a_ss = BHE.l_ss/BHE.rhoc_ss;                                               # BHE soil thermal diffusivity (m2/s)
        a_g = BHE.l_g/BHE.rhoc_g;                                                  # Grout thermal diffusivity (W/m/K)
        # KART: eksponer mod bruger eller slet hvis den altid er samme som T0?
        T0_BHE = T0;                                                     # Measured undisturbed BHE temperature (C)
        s_BHE = 2*BHE.r_p + BHE.D_pipes;                                     # Calculate shank spacing U-pipe (m)
    
        # Borehole field
        x = np.linspace(0,BHE.NX-1,BHE.NX)*BHE.D_x;                                  # x-coordinates of BHEs (m)                     
        y = np.linspace(0,BHE.NY-1,BHE.NY)*BHE.D_y;                                  # y-coordinates of BHEs (m)
        N_BHE = BHE.NX*BHE.NY;                                                   # Number of BHEs (-)
        [XX,YY] = np.meshgrid(x,y);                                     # Meshgrid arrays for distance calculations (m)    
        # Yv = np.concatenate(YY);                                        # YY concatenated (m)
        # Xv = np.concatenate(XX);                                        # XX concatenated (m)
        
        # Logistics for symmetry considerations and associated efficiency gains
        # KART: har ikke tjekket
        NXi = int(np.ceil(BHE.NX/2));                                       # Find half the number of boreholes in the x-direction. If not an equal number then round up to complete symmetry.
        NYi = int(np.ceil(BHE.NY/2));                                       # Find half the number of boreholes in the y-direction. If not an equal number then round up to complete symmetry.
        w = np.ones((NYi,NXi));                                         # Define weight matrix for temperature responses at a distance (-)
        if np.mod(BHE.NX/2,1) > 0:                                          # If NX is an unequal integer then the weight on the temperature responses from the boreholes on the center line is equal to 0.5 for symmetry reasons
            w[:,NXi-1] = 0.5*w[:,NXi-1];
        
        if np.mod(BHE.NY/2,1) > 0:                                          # If NY is an unequal integer then the weight on the temperature responses from the boreholes on the center line is equal to 0.5 for symmetry reasons
            w[NYi-1,:] = 0.5*w[NYi-1,:];
            
        wv = np.concatenate(w);                                         # Concatenate the weight matrix (-)
        swv = sum(wv);                                                  # Sum all weights (-)
        xi = np.linspace(0,NXi-1,NXi)*BHE.D_x;                               # x-coordinates of BHEs (m)                     
        yi = np.linspace(0,NYi-1,NYi)*BHE.D_y;                               # y-coordinates of BHEs (m)
        [XXi,YYi] = np.meshgrid(xi,yi);                                 # Meshgrid arrays for distance calculations (m)
        Yvi = np.concatenate(YYi);                                      # YY concatenated (m)
        Xvi = np.concatenate(XXi);                                      # XX concatenated (m)
        
        # Solver settings for computing the flow and length corrected length of BHEs
        dL = 0.1;                                                       # Step length for trial trial solutions (m)
        LL = 10;                                                        # Additional length segment for which trial solutions are generated (m)
    
    
    
        ######################### Generate G-functions ############################
        G_BHE = CSM(BHE.r_b,BHE.r_b,t[0:2],a_ss);                                   # Compute g-functions for t[0] and t[1] with the cylindrical source model (-)
        s1 = 0;                                                         # Summation variable for t[0] G-function (-)
        s2 = 0;                                                         # Summation variable for t[1] G-function (-)
        for i in range(NXi*NYi):                                        # Line source superposition for all neighbour boreholes for 1/4 of the BHE field (symmetry)
            DIST = np.sqrt((XX-Xvi[i])**2 + (YY-Yvi[i])**2);            # Compute distance matrix (to neighbour boreholes) (m)
            DIST = DIST[DIST>0];                                        # Exclude the considered borehole to avoid r = 0 m
            s1 = s1 + wv[i]*sum(ils(a_ss,t[0],DIST));                    # Compute the sum of all thermal disturbances from neighbour boreholes (G-function contributions) for t[0] (-)
            s2 = s2 + wv[i]*sum(ils(a_ss,t[1],DIST));                    # Compute the sum of all thermal disturbances from neighbour boreholes (G-function contributions) for t[0] (-)
        G_BHE[0] = G_BHE[0] + s1/swv;                                     # Add the average neighbour contribution to the borehole field G-function for t[0] (-)
        G_BHE[1] = G_BHE[1] + s2/swv;                                     # Add the average neighbour contribution to the borehole field G-function for t[1] (-)
    
    
        # BHE heating
        # KART flyttet aggregering
        # Q_BHEmax_H = sum(hp.Qdim_H)/N_BHE;                                        # Peak flow in BHE pipes (m3/s)
        Q_BHEmax_H = aggLoad.Qdim_H / N_BHE;                                        # Peak flow in BHE pipes (m3/s)
        v_BHEmax_H = Q_BHEmax_H/np.pi/ri**2;                                      # Flow velocity in BHEs (m/s)
        Re_BHEmax_H = Re(brine.rho,brine.mu,v_BHEmax_H,2*ri);                              # Reynold number in BHEs (-)
        dpdL_BHEmax_H = dp(brine.rho,brine.mu,Q_BHEmax_H,2*ri);                               # Pressure loss in BHE (Pa/m)
        
        BHE.Re_BHEmax_H = Re_BHEmax_H;  # Add Re to BHE instance
        BHE.dpdL_BHEmax_H = dpdL_BHEmax_H;  # Add pressure loss to BHE instance
        
        # BHE cooling
        # KART flyttet aggregering
        # Q_BHEmax_C = sum(hp.Qdim_C)/N_BHE;                                        # Peak flow in BHE pipes (m3/s)
        Q_BHEmax_C = aggLoad.Qdim_C/N_BHE;                                        # Peak flow in BHE pipes (m3/s)
        v_BHEmax_C = Q_BHEmax_C/np.pi/ri**2;                                      # Flow velocity in BHEs (m/s)
        Re_BHEmax_C = Re(brine.rho,brine.mu,v_BHEmax_C,2*ri);                              # Reynold number in BHEs (-)
        dpdL_BHEmax_C = dp(brine.rho,brine.mu,Q_BHEmax_C,2*ri);                               # Pressure loss in BHE (Pa/m)

        BHE.Re_BHEmax_C = Re_BHEmax_C;  # Add Re to BHE instance
        BHE.dpdL_BHEmax_C = dpdL_BHEmax_C;  # Add pressure loss to BHE instance

    
        # Compute borehole resistance with the first order multipole method ignoring flow and length effects
        Rb_H = RbMP(brine.l,net.l_p,BHE.l_g,BHE.l_ss,BHE.r_b,BHE.r_p,ri,s_BHE,Re_BHEmax_H,Pr);                # Compute the borehole thermal resistance (m*K/W)
        Rb_C = RbMP(brine.l,net.l_p,BHE.l_g,BHE.l_ss,BHE.r_b,BHE.r_p,ri,s_BHE,Re_BHEmax_C,Pr);                # Compute the borehole thermal resistance (m*K/W)
        
        
        # Composite cylindrical source model GCLS() for short term response. Hu et al. 2014. Paper here: https://www.sciencedirect.com/science/article/abs/pii/S0378778814005866?via#3Dihub
        re_H = BHE.r_b/np.exp(2*np.pi*BHE.l_g*Rb_H);                                # Heating: Compute the equivalent pipe radius for cylindrical symmetry (m). This is how Hu et al. 2014 define it.
        re_C = BHE.r_b/np.exp(2*np.pi*BHE.l_g*Rb_C);                                # Cooling: Compute the equivalent pipe radius for cylindrical symmetry (m). This is how Hu et al. 2014 define it.
    
        # The Fourier numbers Fo1-Fo3 are neccesary for computing the solution 
        Fo1 = a_ss*t[2]/BHE.r_b**2;                                    
        G1 = GCLS(Fo1); 
    
        Fo2h = a_g*t[2]/re_H**2;
        G2h = GCLS(Fo2h);
    
        Fo2c = a_g*t[2]/re_C**2;
        G2c = GCLS(Fo2c);
    
        Fo3 = a_g*t[2]/BHE.r_b**2;
        G3 = GCLS(Fo3);
    
        Rw_H = G1/BHE.l_ss + G2h/BHE.l_g - G3/BHE.l_g;                                  # Step response for short term model on the form q*Rw = T (m*K/W). Rw indicates that it is in fact a thermal resistance
        Rw_C = G1/BHE.l_ss + G2c/BHE.l_g - G3/BHE.l_g;                                  # Step response for short term model on the form q*Rw = T (m*K/W). Rw indicates that it is in fact a thermal resistance
    
        # Compute approximate combined length of BHES (length effects not considered)
        # KART: hvorfor kun een GBHEF (ingen H/C)
        GBHEF = G_BHE;                                                  # Retain a copy of the G function for length correction later on (-)
        G_BHE_H = np.asarray([G_BHE[0]/BHE.l_ss+Rb_H,G_BHE[1]/BHE.l_ss+Rb_H, Rw_H]);     # Heating G-function
        G_BHE_C = np.asarray([G_BHE[0]/BHE.l_ss+Rb_C,G_BHE[1]/BHE.l_ss+Rb_C, Rw_C]);     # Cooling G-function
        L_BHE_H = np.dot(PHEH,G_BHE_H) / (T0_BHE - (aggLoad.Ti_H + aggLoad.To_H)/2);    # Sizing equation for computing the required borehole meters for heating (m)
        L_BHE_C = np.dot(PHEC,G_BHE_C) / (-T0_BHE + (aggLoad.Ti_C + aggLoad.To_C)/2);      
            
        # Determine the solution by searching the neighbourhood of the approximate length solution
        # Heating mode
        L_BHE_H_v = L_BHE_H/N_BHE + np.arange(0,LL,dL);
        NLBHEHv = len(L_BHE_H_v);
        Rb_H_v = np.zeros(NLBHEHv);
        
        # Cooling mode
        L_BHE_C_v = L_BHE_C/N_BHE + np.arange(0,LL,dL);
        NLBHECv = len(L_BHE_C_v);
        Rb_C_v = np.zeros(NLBHECv);
        # Tsolc = np.zeros(NLBHECv);
        
        Tf_BHE_H = np.zeros(NLBHEHv);
        for i in range(NLBHEHv):                                         # Compute Rb for the specified number of boreholes and lengths considering flow and length effects (m*K/W)
            Rb_H_v[i] = RbMPflc(brine.l,net.l_p,BHE.l_g,BHE.l_ss,brine.rho,brine.c,BHE.r_b,BHE.r_p,ri,L_BHE_H_v[i],s_BHE,Q_BHEmax_H,Re_BHEmax_H,Pr);    # Compute BHE length and flow corrected multipole estimates of Rb for all candidate solutions (m*K/W)
            # KART: beregn væske temperatur
            Tf_BHE_H[i] = T0_BHE - np.dot(PHEH,np.array([GBHEF[0]/BHE.l_ss + Rb_H_v[i], GBHEF[1]/BHE.l_ss + Rb_H_v[i], Rw_H]))/L_BHE_H_v[i]/N_BHE;
    
    
        Tbound_H = (aggLoad.Ti_H + aggLoad.To_H)/2; # Tjek om den skal bruges tidligere og flyt op
        Tf_BHE_H[Tf_BHE_H < Tbound_H] = np.nan;                                # Remove solutions that violate the bound Tf < Tbound_H    
        indLBHEH = np.argmin(np.isnan(Tf_BHE_H));                          # Find index of the first viable solution
    
        L_BHE_H = L_BHE_H_v[indLBHEH]*N_BHE;                                   # Solution to BHE length for heating (m)
        
        if Tf_BHE_H[indLBHEH]-Tbound_H > 0.1:        
            print('Warning - the length steps used for computing the BHE length for heating are too big. Reduce the stepsize and recompute a solution.');
        
        Tf_BHE_C = np.zeros(NLBHECv)
        for i in range(NLBHECv):                                         # Compute Rb for the specified number of boreholes and lengths considering flow and length effects (m*K/W)
            Rb_C_v[i] = RbMPflc(brine.l,net.l_p,BHE.l_g,BHE.l_ss,brine.rho,brine.c,BHE.r_b,BHE.r_p,ri,L_BHE_C_v[i],s_BHE,Q_BHEmax_C,Re_BHEmax_C,Pr);    #K. Compute BHE length and flow corrected multipole estimates of Rb for all candidate solutions (m*K/W)
            # KART: beregn væske temperatur
            Tf_BHE_C[i] = T0_BHE + np.dot(PHEC,np.array([GBHEF[0]/BHE.l_ss + Rb_C_v[i], GBHEF[1]/BHE.l_ss + Rb_C_v[i], Rw_C]))/L_BHE_C_v[i]/N_BHE;
    
        Tbound_C = (aggLoad.Ti_C + aggLoad.To_C)/2; # Tjek om den skal bruges tidligere og flyt op
        Tf_BHE_C[Tf_BHE_C > Tbound_C] = np.nan;                                # Remove solutions that violate the bound Tf > Tbound_C    
        indLBHEC = np.argmin(np.isnan(Tf_BHE_C));                          # Find index of the first viable solution
    
        
        #indLBHEC = np.argmax(Tsolc<TCC2);                                # Get rid of candidates that undersize the system. 
        L_BHE_C = L_BHE_C_v[indLBHEC]*N_BHE;                                   # Solution BHE length for cooling (m)
        
        if Tf_BHE_C[indLBHEC]-Tbound_C > 0.1:
            print('Warning - the length steps used for computing the BHE length for cooling are too big. Reduce the stepsize and recompute a solution.');    
        
        BHE.L_BHE_H = L_BHE_H;
        BHE.L_BHE_C = L_BHE_C;
        source_config = BHE;
       
        
       
    # If HHEs are selected as source
    elif source_config.source == 'HHE':    
       
        # For readability only
        HHE = source_config;

        ###########################################################################
        ############################### HHE computation ###########################
        ###########################################################################
    
        ri_HHE = HHE.d*(1 - 2/HHE.SDR)/2;                                  # Inner radius of HHE pipes (m)
        ro_HHE = HHE.d/2;                                                 # Outer radius of HHE pipes (m)
    
        # Compute combined length of HHEs   
        ind = np.linspace(0,2*HHE.N_HHE-1,2*HHE.N_HHE);                           # Unit distance vector for HHE (-)
        s = np.zeros(2);                                                # s is a temperature summation variable, s[0]: annual, s[1] monthly, hourly effects are insignificant and ignored (C)
        DIST = HHE.D*ind;                                                  # Distance vector for HHE (m)
        for i in range(HHE.N_HHE):                                           # For half the pipe segments (2 per loop). Advantage from symmetry.
            s[0] = s[0] + sum(ils(a_s,t[0],abs(DIST[ind!=i]-i*HHE.D))) - sum(ils(a_s,t[0],np.sqrt((DIST-i*HHE.D)**2 + 4*net.z_grid**2))); # Sum annual temperature responses from distant pipes (C)
            s[1] = s[1] + sum(ils(a_s,t[1],abs(DIST[ind!=i]-i*HHE.D))) - sum(ils(a_s,t[1],np.sqrt((DIST-i*HHE.D)**2 + 4*net.z_grid**2))); # Sum monthly temperature responses from distant pipes (C)
        G_HHE = CSM(ro_HHE,ro_HHE,t,a_s);                                  # Pipe wall response (-)
        #KART: tjek - i tidligere version var en faktor 2 til forskel
        G_HHE[0:2] = G_HHE[0:2] + s/HHE.N_HHE;                                 # Add thermal disturbance from neighbour pipes (-)
        
        # HHE heating
        # KART flyttet aggregering
        # Q_HHEmax_H = sum(hp.Qdim_H)/HHE.N_HHE;                                        # Peak flow in HHE pipes (m3/s)
        Q_HHEmax_H = aggLoad.Qdim_H / HHE.N_HHE;                                        # Peak flow in HHE pipes (m3/s)
        v_HHEmax_H = Q_HHEmax_H/np.pi/ri_HHE**2;                                   # Peak flow velocity in HHE pipes (m/s)
        Re_HHEmax_H = Re(brine.rho,brine.mu,v_HHEmax_H,2*ri_HHE);                           # Peak Reynolds numbers in HHE pipes (-)
        dpdL_HHEmax_H = dp(brine.rho,brine.mu,Q_HHEmax_H,2*ri_HHE);                            # Peak pressure loss in HHE pipes (Pa/m)
    
        HHE.Re_HHEmax_H = Re_HHEmax_H;  # Add Re to HHE instance
        HHE.dpdL_HHEmax_H = dpdL_HHEmax_H;  # Add pressure loss to HHE instance

    
        # HHE cooling
        # KART fyttet aggregering
        # Q_HHEmax_C = sum(hp.Qdim_C)/HHE.N_HHE;                                        # Peak flow in HHE pipes (m3/s)
        Q_HHEmax_C = aggLoad.Qdim_C / HHE.N_HHE;                                        # Peak flow in HHE pipes (m3/s)
        v_HHEmax_C = Q_HHEmax_C/np.pi/ri_HHE**2;                                   # Peak flow velocity in HHE pipes (m/s)
        Re_HHEmax_C = Re(brine.rho,brine.mu,v_HHEmax_C,2*ri_HHE);                           # Peak Reynolds numbers in HHE pipes (-)
        dpdL_HHEmax_C = dp(brine.rho,brine.mu,Q_HHEmax_C,2*ri_HHE);                            # Peak pressure loss in HHE pipes (Pa/m)

        HHE.Re_HHEmax_C = Re_HHEmax_C;  # Add Re to HHE instance
        HHE.dpdL_HHEmax_C = dpdL_HHEmax_C;  # Add pressure loss to HHE instance

        
        # Heating
        Rp_HHE_H = Rp(2*ri_HHE,2*ro_HHE,Re_HHEmax_H,Pr,brine.l,net.l_p);                   # Compute the pipe thermal resistance (m*K/W)
        G_HHE_H = G_HHE/net.l_s_H + Rp_HHE_H;                                         # Add annual and monthly thermal resistances to G_HHE (m*K/W)
        L_HHE_H = np.dot(PHEH,G_HHE_H) / (T0 - (aggLoad.Ti_H + aggLoad.To_H)/2 - TP );
        
        # Cooling
        Rp_HHE_C = Rp(2*ri_HHE,2*ro_HHE,Re_HHEmax_C,Pr,brine.l,net.l_p);                   # Compute the pipe thermal resistance (m*K/W)
        G_HHE_C = G_HHE/net.l_s_C + Rp_HHE_C;                                         # Add annual and monthly thermal resistances to G_HHE (m*K/W)
        #L_HHE_C = np.dot(PHEC,G_HHE_C/TCC1);                                 # Sizing equation for computing the required borehole meters (m)
        L_HHE_C = np.dot(PHEC,G_HHE_C) / ((aggLoad.Ti_C + aggLoad.To_C)/2 - T0 - TP);
        
        
        # Add results to source configuration
        HHE.L_HHE_H = L_HHE_H;
        HHE.L_HHE_C = L_HHE_C;
        source_config = HHE;
    ############################## Source sizing END ##############################
    
    return FPH, FPC, source_config
    

    

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
    # |<--D--->|<--D--->|<--D--->|<--D--->|<--D--->|
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
    
