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

from .dimensioning_classes import BTESConfiguation, HorizontalConfiguration, DimensioningConfiguration, \
    DimensioningResults, PipeResult, EnergyProductionResult, HEResult, HEType
from .fThermonetDim import ils, ps, Re, dp, Rp, CSM, RbMP, GCLS, RbMPflc
import time

from pathlib import Path


def run_dimensioning(c: DimensioningConfiguration, print_computation_time=True):
    tic = time.time();  # Track computation time (s)

    folder = Path(__file__).parent.joinpath("examples")


    ############################# Load all input data #############################

    # Load heat pump data
    HPS = pd.read_csv(folder.joinpath(c.HPFN), sep='\t+', engine='python');  # Heat pump input file

    # Load grid topology
    TOPOH = np.loadtxt(folder.joinpath(c.TOPOFN), skiprows=1, usecols=(1, 2, 3));  # Load numeric data from topology file
    TOPOC = TOPOH;
    IPG = pd.read_csv(folder.joinpath(c.TOPOFN), sep='\t+', engine='python');  # Load the entire file into Panda dataframe
    PGROUP = IPG.iloc[:, 0];  # Extract pipe group IDs
    IPG = IPG.iloc[:, 4];  # Extract IDs of HPs connected to the different pipe groups
    NPG = len(IPG);  # Number of pipe groups

    # Load pipe database

    file = Path(__file__).parent.joinpath('data/equipment/PIPES.dat')
    PIPES = pd.read_csv(file, sep='\t');  # Open file with available pipe outer diameters (mm). This file can be expanded with additional pipes and used directly.
    PIPES = PIPES.values;  # Get numerical values from pipes excluding the headers
    NP = len(PIPES);  # Number of available pipes

    ########################### Load all input data END ###########################

    ###############################################################################
    ################################## INPUT END ##################################
    ###############################################################################

    # Output to prompt
    print(' ');
    print('************************************************************************')
    print('************************** ThermonetDim v0.75 **************************')
    print('************************************************************************')
    print(' ');
    print(f'Project: {c.PID}');

    ########### Precomputations and variables that should not be changed ##########

    # Heat pump information
    HPS = HPS.values  # Load numeric data from HP file
    CPS = HPS[:, 8:]  # Place cooling demand data in separate array
    HPS = HPS[:, :8]  # Remove cooling demand data from HPS array
    NHP = len(HPS)  # Number of heat pumps

    # Add circulation pump power consumption to cooling load (W)
    CPS[:, :3] = CPS[:, 3:4] / (CPS[:, 3:4] - 1) * CPS[:, :3]  # Number of heat pumps

    # G-function evaluation times (DO NOT MODIFY!!!!!!!)
    SECONDS_IN_YEAR = 31536000;
    SECONDS_IN_MONTH = 2628000;
    SECONDS_IN_HOUR = 3600;
    t = np.asarray(
        [10 * SECONDS_IN_YEAR + 3 * SECONDS_IN_MONTH + 4 * SECONDS_IN_HOUR, 3 * SECONDS_IN_MONTH + 4 * SECONDS_IN_HOUR,
         4 * SECONDS_IN_HOUR],
        dtype=float);  # time = [10 years + 3 months + 4 hours; 3 months + 4 hours; 4 hours]. Time vector for the temporal superposition (s).

    # Create array containing arrays of integers with HP IDs for all pipe sections
    IPGA = [np.asarray(IPG.iloc[i].split(',')).astype(int) - 1 for i in range(NPG)]
    IPG = IPGA  # Redefine IPG
    del IPGA;  # Get rid of IPGA

    # Brine
    kinb = c.brine.mub / c.brine.rhob;  # Brine kinematic viscosity (m2/s)
    ab = c.brine.lb / (c.brine.rhob * c.brine.cb);  # Brine thermal diffusivity (m2/s)
    Pr = kinb / ab;  # Prandtl number (-)

    # Shallow soil (not for BHEs! - see below)
    A = 7.900272987633280;  # Surface temperature amplitude (K)
    T0 = 9.028258373009810;  # Undisturbed soil temperature (C)
    o = 2 * np.pi / 86400 / 365.25;  # Angular velocity of surface temperature variation (rad/s)
    ast = c.thermonet.lsh / c.thermonet.rhocs;  # Shallow soil thermal diffusivity (m2/s) - ONLY for pipes!!!
    TP = A * mt.exp(-c.thermonet.zd * mt.sqrt(
        o / 2 / ast));  # Temperature penalty at burial depth from surface temperature variation (K). Minimum undisturbed temperature is assumed .

    # Convert pipe diameter database to meters
    PIPES = PIPES / 1000;  # Convert PIPES from mm to m (m)

    # Allocate variables
    indh = np.zeros(NPG);  # Index vector for pipe groups heating (-)
    indc = np.zeros(NPG);  # Index vector for pipe groups cooling (-)
    PIPESELH = np.zeros(NPG);  # Pipes selected from dimensioning for heating (length)
    PIPESELC = np.zeros(NPG);  # Pipes selected from dimensioning for cooling (length)
    PSH = np.zeros((NHP, 3));  # Thermal load from heating on the ground (W)
    QPGH = np.zeros(NPG);  # Design flow heating (m3/s)
    QPGC = np.zeros(NPG);  # Design flow cooling (m3/s)
    Rh = np.zeros(NPG);  # Allocate pipe thermal resistance vector for heating (m*K/W)
    Rc = np.zeros(NPG);  # Allocate pipe thermal resistance vector for cooling (m*K/W)
    FPH = np.zeros(NPG);  # Vector with total heating load fractions supplied by each pipe segment (-)
    FPC = np.zeros(NPG);  # Vector with total cooling load fractions supplied by each pipe segment (-)
    GTHMH = np.zeros([NPG, 3]);
    GTHMC = np.zeros([NPG, 3]);

    # Simultaneity factors to apply to annual, monthly and hourly heating and cooling demands
    S = np.zeros(3);
    S[2] = c.heatpump.SF * (
                0.62 + 0.38 / NHP);  # Hourly. Varme Ståbi. Ligning 3 i "Effekt- og samtidighedsforhold ved fjernvarmeforsyning af nye boligområder"
    S[
        0] = 1;  # 0.62 + 0.38/NHP;                                        # Annual. Varme Ståbi. Ligning 3 i "Effekt- og samtidighedsforhold ved fjernvarmeforsyning af nye boligområder"
    S[
        1] = 1;  # S(1);                                                   # Monthly. Varme Ståbi. Ligning 3 i "Effekt- og samtidighedsforhold ved fjernvarmeforsyning af nye boligområder"

    # If horizontal heat exchangers are selected
    if isinstance(c.ground_heatexchanger_configuration, HorizontalConfiguration):
        # Horizontal heat exchangers
        rihhe = c.ground_heatexchanger_configuration.PDHE * (1 - 2 / c.ground_heatexchanger_configuration.HHESDR) / 2;  # Inner radius of HHE pipes (m)
        rohhe = c.ground_heatexchanger_configuration.PDHE / 2;  # Outer radius of HHE pipes (m)

    # If borehole heat exchangers are selected
    elif isinstance(c.ground_heatexchanger_configuration, BTESConfiguation):
        c_btes = c.ground_heatexchanger_configuration
        # BHE
        ri = c_btes.rp * (1 - 2 / c_btes.BHESDR)  # Inner radius of U pipe (m)
        ass = c_btes.lss / c_btes.rhocss;  # BHE soil thermal diffusivity (m2/s)
        ag = c_btes.lg / c_btes.rhocg;  # Grout thermal diffusivity (W/m/K)
        T0BHE = T0;  # Measured undisturbed BHE temperature (C)
        PD = 2 * c_btes.rp + c_btes.PD;  # Redefine PD to shank spacing U-pipe (m)

        NX = c.ground_heatexchanger_configuration.NX
        NY = c.ground_heatexchanger_configuration.NY
        dx = c.ground_heatexchanger_configuration.dx
        dy = c.ground_heatexchanger_configuration.dy


        # Borehole field
        x = np.linspace(0, NX - 1, NX) * dx;  # x-coordinates of BHEs (m)
        y = np.linspace(0, NY - 1, NY) * dy;  # y-coordinates of BHEs (m)
        NBHE = NX * NY;  # Number of BHEs (-)
        [XX, YY] = np.meshgrid(x, y);  # Meshgrid arrays for distance calculations (m)
        Yv = np.concatenate(YY);  # YY concatenated (m)
        Xv = np.concatenate(XX);  # XX concatenated (m)

        # Logistics for symmetry considerations and associated efficiency gains
        NXi = int(np.ceil(
            NX / 2));  # Find half the number of boreholes in the x-direction. If not an equal number then round up to complete symmetry.
        NYi = int(np.ceil(
            NY / 2));  # Find half the number of boreholes in the y-direction. If not an equal number then round up to complete symmetry.
        w = np.ones((NYi, NXi));  # Define weight matrix for temperature responses at a distance (-)
        if np.mod(NX / 2,
                  1) > 0:  # If NX is an unequal integer then the weight on the temperature responses from the boreholes on the center line is equal to 0.5 for symmetry reasons
            w[:, NXi - 1] = 0.5 * w[:, NXi - 1];

        if np.mod(NY / 2,
                  1) > 0:  # If NY is an unequal integer then the weight on the temperature responses from the boreholes on the center line is equal to 0.5 for symmetry reasons
            w[NYi - 1, :] = 0.5 * w[NYi - 1, :];

        wv = np.concatenate(w);  # Concatenate the weight matrix (-)
        swv = sum(wv);  # Sum all weights (-)
        xi = np.linspace(0, NXi - 1, NXi) * dx;  # x-coordinates of BHEs (m)
        yi = np.linspace(0, NYi - 1, NYi) * dy;  # y-coordinates of BHEs (m)
        [XXi, YYi] = np.meshgrid(xi, yi);  # Meshgrid arrays for distance calculations (m)
        Yvi = np.concatenate(YYi);  # YY concatenated (m)
        Xvi = np.concatenate(XXi);  # XX concatenated (m)

        # Solver settings for computing the flow and length corrected length of BHEs
        dL = 0.1;  # Step length for trial trial solutions (m)
        LL = 10;  # Additional length segment for which trial solutions are generated (m)

    else:
        raise TypeError(f"Unknown configuration type: {c.ground_heatexchanger_configuration}")
    ######### Precomputations and variables that should not be changed END ########

    ################################# Pipe sizing #################################

    # Convert thermal load profile on HPs to flow rates
    PSH = ps(S * HPS[:, 1:4], HPS[:, 4:7]);  # Annual (0), monthly (1) and daily (2) thermal load on the ground (W)
    PSH[:, 0] = PSH[:, 0] - CPS[:, 0];  # Annual imbalance between heating and cooling, positive for heating (W)
    Qdimh = PSH[:, 2] / HPS[:, 7] / c.brine.rhob / c.brine.cb;  # Design flow heating (m3/s)
    Qdimc = CPS[:, 2] / CPS[:, 4] / c.brine.rhob / c.brine.cb;  # Design flow cooling (m3/s). Using simultaneity factor!
    HPS = np.c_[HPS, Qdimh];  # Append to heat pump data structure for heating
    CPS = np.c_[CPS, Qdimc];  # Append to heat pump data structure for cooling

    # Heat pump and temperature conditions in the sizing equation
    Tho = c.heatpump.Thi - sum(Qdimh * HPS[:, 7]) / sum(Qdimh);  # Volumetric flow rate weighted average brine delta-T (C)
    TCH1 = T0 - (
                c.heatpump.Thi + Tho) / 2 - TP;  # Temperature condition for with heating termonet. Eq. 2.19 Advances in GSHP systems. Tp in the book refers to the influence from adjacent BHEs. This effect ignored in this tool.
    Tco = c.heatpump.Tci + sum(Qdimc * CPS[:, 4]) / sum(Qdimc);  # Volumetric flow rate weighted average brine delta-T (C)
    TCC1 = (
                       c.heatpump.Tci + Tco) / 2 - T0 - TP;  # Temperature condition for with heating termonet. Eq. 2.19 Advances in GSHP systems. Tp in the book refers to the influence from adjacent BHEs. This effect ignored in this tool.

    # Compute flow and pressure loss in BHEs and HHEs under peak load conditions. Temperature conditions are computed as well.
    if isinstance(c.ground_heatexchanger_configuration, HorizontalConfiguration):
        c_hor = c.ground_heatexchanger_configuration
        # HHE heating
        QHHEH = sum(Qdimh) / c_hor.NHHE;  # Peak flow in HHE pipes (m3/s)
        vhheh = QHHEH / np.pi / rihhe ** 2;  # Peak flow velocity in HHE pipes (m/s)
        RENHHEH = Re(c.brine.rhob, c.brine.mub, vhheh, 2 * rihhe);  # Peak Reynolds numbers in HHE pipes (-)
        dpHHEH = dp(c.brine.rhob, c.brine.mub, QHHEH, 2 * rihhe);  # Peak pressure loss in HHE pipes (Pa/m)

        # HHE cooling
        QHHEC = sum(Qdimc) /c_hor.NHHE;  # Peak flow in HHE pipes (m3/s)
        vhhec = QHHEC / np.pi / rihhe ** 2;  # Peak flow velocity in HHE pipes (m/s)
        RENHHEC = Re(c.brine.rhob, c.brine.mub, vhhec, 2 * rihhe);  # Peak Reynolds numbers in HHE pipes (-)
        dpHHEC = dp(c.brine.rhob, c.brine.mub, QHHEC, 2 * rihhe);  # Peak pressure loss in HHE pipes (Pa/m)

    elif isinstance(c.ground_heatexchanger_configuration, BTESConfiguation):
        c_btes = c.ground_heatexchanger_configuration

        TCH2 = T0BHE - (
                    c.Thi + Tho) / 2;  # Temperature condition for heating with BHE. Eq. 2.19 Advances in GSHP systems but surface temperature penalty is removed from the criterion as it doesn't apply to BHEs (C)
        TCC2 = (
                           c.Tci + Tco) / 2 - T0BHE;  # Temperature condition for cooling with BHE. Eq. 2.19 Advances in GSHP systems but surface temperature penalty is removed from the criterion as it doesn't apply to BHEs (C)

        # BHE heating
        QBHEH = sum(Qdimh) / NBHE;  # Peak flow in BHE pipes (m3/s)
        vbheh = QBHEH / np.pi / ri ** 2;  # Flow velocity in BHEs (m/s)
        RENBHEH = Re(c.rhob, c.mub, vbheh, 2 * ri);  # Reynold number in BHEs (-)
        dpBHEH = dp(c.rhob, c.mub, QBHEH, 2 * ri);  # Pressure loss in BHE (Pa/m)

        # BHE cooling
        QBHEC = sum(Qdimc) / NBHE;  # Peak flow in BHE pipes (m3/s)
        vbhec = QBHEC / np.pi / ri ** 2;  # Flow velocity in BHEs (m/s)
        RENBHEC = Re(c.rhob, c.mub, vbhec, 2 * ri);  # Reynold number in BHEs (-)
        dpBHEC = dp(c.rhob, c.mub, QBHEC, 2 * ri);  # Pressure loss in BHE (Pa/m)

    # Compute design flow for the pipes
    for i in range(NPG):
        QPGH[i] = sum(HPS[np.ndarray.tolist(IPG[i]), 8]) / TOPOH[
            i, 2];  # Sum the heating brine flow for all consumers connected to a specific pipe group and normalize with the number of traces in that group to get flow in the individual pipes (m3/s)
        QPGC[i] = sum(CPS[np.ndarray.tolist(IPG[i]), 5]) / TOPOC[
            i, 2];  # Sum the cooling brine flow for all consumers connected to a specific pipe group and normalize with the number of traces in that group to get flow in the individual pipes (m3/s)

    # Select the smallest diameter pipe that fulfills the pressure drop criterion
    for i in range(NPG):
        PIPESI = PIPES * (1 - 2 / TOPOH[i, 0]);  # Compute inner diameters (m). Variable TOPOH or TOPOC are identical here.
        indh[i] = np.argmax(dp(c.brine.rhob, c.brine.mub, QPGH[i],
                               PIPESI) < c.thermonet.dpt);  # Find first pipe with a pressure loss less than the target for heating (-)
        indc[i] = np.argmax(dp(c.brine.rhob, c.brine.mub, QPGC[i],
                               PIPESI) < c.thermonet.dpt);  # Find first pipe with a pressure loss less than the target for cooling (-)
        PIPESELH[i] = PIPES[int(indh[i])];  # Store pipe selection for heating in new variable (m)
        PIPESELC[i] = PIPES[int(indc[i])];  # Store pipe selection for cooling in new variable (m)
    indh = indh.astype(int);
    indc = indc.astype(int);

    # Compute Reynolds number for selected pipes for heating
    DiSELH = PIPESELH * (1 - 2 / TOPOH[:, 0]);  # Compute inner diameter of selected pipes (m)
    vh = QPGH / np.pi / DiSELH ** 2 * 4;  # Compute flow velocity for selected pipes (m/s)
    RENH = Re(c.brine.rhob, c.brine.mub, vh, DiSELH);  # Compute Reynolds numbers for the selected pipes (-)

    # Compute Reynolds number for selected pipes for cooling
    DiSELC = PIPESELC * (1 - 2 / TOPOC[:, 0]);  # Compute inner diameter of selected pipes (m)
    vc = QPGC / np.pi / DiSELC ** 2 * 4;  # Compute flow velocity for selected pipes (m/s)
    RENC = Re(c.brine.rhob, c.brine.mub, vc, DiSELC);  # Compute Reynolds numbers for the selected pipes (-)



    # Output the pipe sizing
    print(' ');
    print('******************* Suggested pipe dimensions heating ******************');
    suggested_pipe_dimensions_heating = []
    for i in range(NPG):
        print(f'{PGROUP.iloc[i]}: Ø{int(1000 * PIPESELH[i])} mm SDR {int(TOPOH[i, 0])}, Re = {int(round(RENH[i]))}');
        suggested_pipe_dimensions_heating.append(PipeResult(diameter=1000 * PIPESELH[i], sdr=TOPOH[i, 0], Re=RENH[i]))
    print(' ');
    print('******************* Suggested pipe dimensions cooling ******************');
    suggested_pipe_dimensions_cooling = []
    for i in range(NPG):
        print(f'{PGROUP.iloc[i]}: Ø{int(1000 * PIPESELC[i])} mm SDR {int(TOPOC[i, 0])}, Re = {int(round(RENC[i]))}');
        suggested_pipe_dimensions_cooling.append(PipeResult(diameter=1000 * PIPESELC[i], sdr=TOPOC[i, 0], Re=RENC[i]))
    print(' ');

    ############################### Pipe sizing END ###############################

    ################## Compute temperature response of thermonet ##################

    # Compute thermal resistances for pipes in heating mode
    LENGTHS = 2 * TOPOH[:, 1] * TOPOH[:, 2];  # Total lengths of different pipe segments (m)
    TLENGTH = sum(LENGTHS);  # Total length of termonet (m)
    TOPOH = np.c_[
        TOPOH, PIPESELH, RENH, LENGTHS];  # Add pipe selection diameters (m), Reynolds numbers (-) and lengths as columns to the TOPO array
    for i in range(NPG):  # For all pipe groups
        Rh[i] = Rp(DiSELH[i], PIPESELH[i], RENH[i], Pr, c.brine.lb, c.lp);  # Compute thermal resistances (m*K/W)
    TOPOH = np.c_[TOPOH, Rh];  # Append thermal resistances to pipe groups as a column in TOPO (m*K/W)

    # Compute thermal resistances for pipes in cooling mode
    TOPOC = np.c_[
        TOPOC, PIPESELC, RENC, LENGTHS];  # Add pipe selection diameters (m), Reynolds numbers (-) and lengths as columns to the TOPO array
    for i in range(NPG):  # For all pipe groups
        Rc[i] = Rp(DiSELC[i], PIPESELC[i], RENC[i], Pr, c.brine.lb, c.lp);  # Compute thermal resistances (m*K/W)
    TOPOC = np.c_[TOPOC, Rc];  # Append thermal resistances to pipe groups as a column in TOPO (m*K/W)

    # Compute delta-qs for superposition of heating load responses
    dPSH = np.zeros((NHP, 3));  # Allocate power difference matrix for tempoeral superposition (W)
    dPSH[:, 0] = PSH[:, 0];  # First entry is just the annual average power (W)
    dPSH[:, 1:] = np.diff(PSH);  # Differences between year-month and month-hour are added (W)
    cdPSH = np.sum(dPSH, 0);

    # Compute delta-qs for superposition of cooling load responses
    dPSC = np.zeros((NHP, 3));  # Allocate power difference matrix for tempoeral superposition (W)
    dPSC = np.c_[-PSH[:, 0], CPS[:, 1:3]];
    dPSC[:, 1:] = np.diff(dPSC);  # Differences between year-month and month-hour are added (W)
    cdPSC = np.sum(dPSC, 0);

    # Compute temperature responses in heating and cooling mode for all pipes
    K1 = ils(ast, t, c.thermonet.PWD) - ils(ast, t, 2 * c.thermonet.zd) - ils(ast, t, np.sqrt(c.thermonet.PWD ** 2 + 4 * c.thermonet.zd ** 2));
    for i in range(NPG):
        GTHMH[i, :] = CSM(PIPESELH[i] / 2, PIPESELH[i] / 2, t, ast) + K1;
        GTHMC[i, :] = CSM(PIPESELC[i] / 2, PIPESELC[i] / 2, t, ast) + K1;
        FPH[i] = TCH1 * LENGTHS[i] / np.dot(cdPSH, GTHMH[i] / c.thermonet.lsh + Rh[
            i]);  # Fraction of total heating that can be supplied by the i'th pipe segment (-)
        FPC[i] = TCC1 * LENGTHS[i] / np.dot(cdPSC, GTHMC[i] / c.thermonet.lsc + Rc[
            i]);  # Fraction of total heating that can be supplied by the i'th pipe segment (-)

    # Heating supplied by thermonet
    FPH = sum(FPH);  # Total fraction of heating supplied by thermonet (-)
    PHEH = (1 - FPH) * cdPSH;  # Residual heat demand (W)

    # Cooling supplied by thermonet
    FPC = sum(FPC);  # Total fraction of cooling supplied by thermonet (-)
    PHEC = (1 - FPC) * cdPSC;  # Residual heat demand (W)

    ########################## Display results in console #########################
    print('***************** Thermonet energy production capacity *****************');
    print(
        f'The thermonet supplies {round(100 * FPH)}% of the peak heating demand');  # print(f'The thermonet fully supplies the heat pumps with IDs 1 - {int(np.floor(NSHPH+1))} with heating' ) ;
    print(f'The thermonet supplies {round(100 * FPC)}% of the peak cooling demand');
    print(' ');
    energy_production_result = EnergyProductionResult(FPH=FPH*100, FPC=FPC*100)
    ######################## Display results in console END #######################

    ################################ Source sizing ################################

    # If BHEs are selected as source
    if isinstance(c.ground_heatexchanger_configuration, BTESConfiguation):
        ###########################################################################
        ############################ Borehole computation #########################
        ###########################################################################

        ######################### Generate G-functions ############################
        GBHE = CSM(c.rb, c.rb, t[0:2], ass);  # Compute g-functions for t[0] and t[1] with the cylindrical source model (-)
        s1 = 0;  # Summation variable for t[0] G-function (-)
        s2 = 0;  # Summation variable for t[1] G-function (-)
        for i in range(
                NXi * NYi):  # Line source superposition for all neighbour boreholes for 1/4 of the BHE field (symmetry)
            DIST = np.sqrt((XX - Xvi[i]) ** 2 + (YY - Yvi[i]) ** 2);  # Compute distance matrix (to neighbour boreholes) (m)
            DIST = DIST[DIST > 0];  # Exclude the considered borehole to avoid r = 0 m
            s1 = s1 + wv[i] * sum(ils(ass, t[0],
                                      DIST));  # Compute the sum of all thermal disturbances from neighbour boreholes (G-function contributions) for t[0] (-)
            s2 = s2 + wv[i] * sum(ils(ass, t[1],
                                      DIST));  # Compute the sum of all thermal disturbances from neighbour boreholes (G-function contributions) for t[0] (-)
        GBHE[0] = GBHE[
                      0] + s1 / swv;  # Add the average neighbour contribution to the borehole field G-function for t[0] (-)
        GBHE[1] = GBHE[
                      1] + s2 / swv;  # Add the average neighbour contribution to the borehole field G-function for t[1] (-)

        # Compute borehole resistance with the first order multipole method ignoring flow and length effects
        Rbh = RbMP(lb, lp, lg, lss, rb, rp, ri, PD, RENBHEH, Pr);  # Compute the borehole thermal resistance (m*K/W)
        Rbc = RbMP(lb, lp, lg, lss, rb, rp, ri, PD, RENBHEC, Pr);  # Compute the borehole thermal resistance (m*K/W)
        # Rb = 0.12;                                                     # TRT estimate can be supplied instread (m*K/W)

        # Composite cylindrical source model GCLS() for short term response. Hu et al. 2014. Paper here: https://www.sciencedirect.com/science/article/abs/pii/S0378778814005866?via#3Dihub
        reh = rb / np.exp(
            2 * np.pi * lg * Rbh);  # Heating: Compute the equivalent pipe radius for cylindrical symmetry (m). This is how Hu et al. 2014 define it.
        rec = rb / np.exp(
            2 * np.pi * lg * Rbc);  # Cooling: Compute the equivalent pipe radius for cylindrical symmetry (m). This is how Hu et al. 2014 define it.

        # The Fourier numbers Fo1-Fo3 are neccesary for computing the solution
        Fo1 = ass * t[2] / rb ** 2;
        G1 = GCLS(Fo1);

        Fo2h = ag * t[2] / reh ** 2;
        G2h = GCLS(Fo2h);

        Fo2c = ag * t[2] / rec ** 2;
        G2c = GCLS(Fo2c);

        Fo3 = ag * t[2] / rb ** 2;
        G3 = GCLS(Fo3);

        Rwh = G1 / lss + G2h / lg - G3 / lg;  # Step response for short term model on the form q*Rw = T (m*K/W). Rw indicates that it is in fact a thermal resistance
        Rwc = G1 / lss + G2c / lg - G3 / lg;  # Step response for short term model on the form q*Rw = T (m*K/W). Rw indicates that it is in fact a thermal resistance

        # Compute approximate combined length of BHES (length effects not considered)
        GBHEF = GBHE;  # Retain a copy of the G function for length correction later on (-)
        GBHEH = np.asarray([GBHE[0] / lss + Rbh, GBHE[1] / lss + Rbh, Rwh]);  # Heating G-function
        GBHEC = np.asarray([GBHE[0] / lss + Rbc, GBHE[1] / lss + Rbc, Rwc]);  # Cooling G-function
        LBHEH = np.dot(PHEH, GBHEH / TCH2);  # Sizing equation for computing the required borehole meters for heating (m)
        LBHEC = np.dot(PHEC, GBHEC / TCC2);  # Sizing equation for computing the required borehole meters for cooling (m)

        # Determine the solution by searching the neighbourhood of the approximate length solution
        # Heating mode
        LBHEHv = LBHEH / NBHE + np.arange(0, LL, dL);
        NLBHEHv = len(LBHEHv);
        Rbhv = np.zeros(NLBHEHv);
        Tsolh = np.zeros(NLBHEHv);

        # Cooling mode
        LBHECv = LBHEC / NBHE + np.arange(0, LL, dL);
        NLBHECv = len(LBHECv);
        Rbcv = np.zeros(NLBHECv);
        Tsolc = np.zeros(NLBHECv);

        for i in range(
                NLBHEHv):  # Compute Rb for the specified number of boreholes and lengths considering flow and length effects (m*K/W)
            Rbhv[i] = RbMPflc(lb, lp, lg, lss, rhob, cb, rb, rp, ri, LBHEHv[i], PD, QBHEH, RENBHEH,
                              Pr);  # Compute BHE length and flow corrected multipole estimates of Rb for all candidate solutions (m*K/W)
            Tsolh[i] = np.dot(PHEH, np.array([GBHEF[0] / lss + Rbhv[i], GBHEF[1] / lss + Rbhv[i], Rwh])) / LBHEHv[
                i] / NBHE;  # OK. Use Spitlers sizing formula for computing the corresponding temperature response for all candidate solutions (C)
        indLBHEH = np.argmax(Tsolh < TCH2);  # Get rid of candidates that undersize the system.
        LBHEH = LBHEHv[indLBHEH] * NBHE;  # Solution to BHE length for heating (m)

        if (Tsolh[indLBHEH] - TCH2) > 0.1:
            print(
                'Warning - the length steps used for computing the BHE length for heating are too big. Reduce the stepsize and recompute a solution.');

        for i in range(
                NLBHECv):  # Compute Rb for the specified number of boreholes and lengths considering flow and length effects (m*K/W)
            Rbcv[i] = RbMPflc(lb, lp, lg, lss, rhob, cb, rb, rp, ri, LBHECv[i], PD, QBHEC, RENBHEC,
                              Pr);  # K. Compute BHE length and flow corrected multipole estimates of Rb for all candidate solutions (m*K/W)
            Tsolc[i] = np.dot(PHEC, np.array([GBHEF[0] / lss + Rbcv[i], GBHEF[1] / lss + Rbcv[i], Rwc])) / LBHECv[
                i] / NBHE;  # OK. Use Spitlers sizing formula for computing the corresponding temperature response for all candidate solutions (C)
        indLBHEC = np.argmax(Tsolc < TCC2);  # Get rid of candidates that undersize the system.
        LBHEC = LBHECv[indLBHEC] * NBHE;  # Solution BHE length for cooling (m)

        if (Tsolc[indLBHEC] - TCC2) > 0.1:
            print(
                'Warning - the length steps used for computing the BHE length for cooling are too big. Reduce the stepsize and recompute a solution.');

            # Display output in console
        print('********** Suggested length of borehole heat exchangers (BHE) **********');
        print(f'Required length of each of the {int(NBHE)} BHEs = {int(np.ceil(LBHEH / NBHE))} m for heating');
        print(f'Required length of each of the {int(NBHE)} BHEs = {int(np.ceil(LBHEC / NBHE))} m for cooling');
        print(f'Maximum pressure loss in BHEs in heating mode = {int(np.ceil(dpBHEH))} Pa/m, Re = {int(round(RENBHEH))}');
        print(f'Maximum pressure loss in BHEs in cooling mode = {int(np.ceil(dpBHEC))} Pa/m, Re = {int(round(RENBHEC))}');

        heat_exchanger_result = HEResult(HE_type=HEType.BHE, NHE=int(NBHE), lengthHE_heating=np.ceil(LBHEH / NBHE), lengthHE_cooling=int(np.ceil(LBHEC / NBHE)), max_pressure_drop_heating=dpBHEH, max_pressure_drop_cooling=dpBHEC,
                                         Re_cooling=RENBHEC, Re_heating=RENBHEC)

    # If HHEs are selected as source
    elif isinstance(c.ground_heatexchanger_configuration, HorizontalConfiguration):
        ###########################################################################
        ############################### HHE computation ###########################
        ###########################################################################
        c_hor = c.ground_heatexchanger_configuration
        # Compute combined length of HHEs
        ind = np.linspace(0, 2 * c_hor.NHHE - 1, 2 * c_hor.NHHE);  # Unit distance vector for HHE (-)
        s = np.zeros(
            2);  # s is a temperature summation variable, s[0]: annual, s[1] monthly, hourly effects are insignificant and ignored (C)
        DIST = c_hor.dd * ind;  # Distance vector for HHE (m)
        for i in range(c_hor.NHHE):  # For half the pipe segments (2 per loop). Advantage from symmetry.
            s[0] = s[0] + sum(ils(ast, t[0], abs(DIST[ind != i] - i * c_hor.dd))) - sum(ils(ast, t[0], np.sqrt(
                (DIST - i * c_hor.dd) ** 2 + 4 * c.thermonet.zd ** 2)));  # Sum annual temperature responses from distant pipes (C)
            s[1] = s[1] + sum(ils(ast, t[1], abs(DIST[ind != i] - i * c_hor.dd))) - sum(ils(ast, t[1], np.sqrt(
                (DIST - i * c_hor.dd) ** 2 + 4 * c.thermonet.zd ** 2)));  # Sum monthly temperature responses from distant pipes (C)
        GHHE = CSM(rohhe, rohhe, t, ast);  # Pipe wall response (-)
        GHHE[0:2] = GHHE[0:2] + s / c_hor.NHHE;  # Add thermal disturbance from neighbour pipes (-)

        # Heating
        RHHEH = Rp(2 * rihhe, 2 * rohhe, RENHHEH, Pr, c.brine.lb, c.lp);  # Compute the pipe thermal resistance (m*K/W)
        GHHEH = GHHE / c.thermonet.lsh + RHHEH;  # Add annual and monthly thermal resistances to GHHE (m*K/W)
        LHHEH = np.dot(PHEH, GHHEH / TCH1);  # Sizing equation for computing the required borehole meters (m)

        # Cooling
        RHHEC = Rp(2 * rihhe, 2 * rohhe, RENHHEC, Pr, c.brine.lb, c.lp);  # Compute the pipe thermal resistance (m*K/W)
        GHHEC = GHHE / c.thermonet.lsc + RHHEC;  # Add annual and monthly thermal resistances to GHHE (m*K/W)
        LHHEC = np.dot(PHEC, GHHEC / TCC1);  # Sizing equation for computing the required borehole meters (m)

        # Output results to console
        print('********* Suggested length of horizontal heat exchangers (HHE) *********');
        print(f'Required length of each of the {int(c_hor.NHHE)} horizontal loops = {int(np.ceil(LHHEH / c_hor.NHHE))} m for heating');
        print(f'Required length of each of the {int(c_hor.NHHE)} horizontal loops = {int(np.ceil(LHHEC / c_hor.NHHE))} m for cooling');
        print(
            f'Maximum pressure loss in HHE pipes in heating mode = {int(np.ceil(dpHHEH))} Pa/m, Re = {int(round(RENHHEH))}');
        print(
            f'Maximum pressure loss in HHE pipes in cooling mode {int(np.ceil(dpHHEC))} Pa/m, Re = {int(round(RENHHEC))}');

        heat_exchanger_result = HEResult(HE_type=HEType.HHE, NHE=c_hor.NHHE, lengthHE_heating=np.ceil(LHHEH / c_hor.NHHE), lengthHE_cooling=np.ceil(LHHEC / c_hor.NHHE),
                                         max_pressure_drop_heating=dpHHEH, max_pressure_drop_cooling=dpHHEC, Re_heating=RENHHEH, Re_cooling=RENHHEC)

    results = DimensioningResults(pipe_dimensions_heating=suggested_pipe_dimensions_heating,
                                  pipe_dimensions_cooling=suggested_pipe_dimensions_cooling,
                                  energy_production=energy_production_result,
                                  heat_exchanger=heat_exchanger_result
                                  )

    ############################## Source sizing END ##############################
    if print_computation_time:
        # Output computation time to console
        print(' ');
        print('*************************** Computation time ***************************');
        toc = time.time();  # Track computation time (s)
        print(f'Elapsed time: {round(toc - tic, 6)} seconds');

    return results

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