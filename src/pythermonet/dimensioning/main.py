from pythermonet.dimensioning.dimensioning_functions import print_project_id, run_pipedimensioning, print_pipe_dimensions, print_source_dimensions, run_sourcedimensioning
from pythermonet.models import Brine, Thermonet, HeatPump, HHEConfig, FullDimension, BHEConfig

import time

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 09:36:52 2023

TODO

- bøvl med isinstance på egne classes -> forløbig fix med str (snak m Lasse til workshop -> lav kort script)

DISKUTER

- Ryd op i EER/COP beregning

- Validering
    * Overvej om vi skal indføre en function der beregner Tfluid som en del af BHE-beregning -> denne del kan valideres mod kendte løsninger

@author: KART
"""

def run_full_dimensioning(PID:str, d_pipes, brine:Brine, net:Thermonet, hp:HeatPump, pipeGroupNames, source_config:HHEConfig|BHEConfig):
    # # Output to prompt
    # print(' ');
    # print('************************************************************************')
    # print('************************** ThermonetDim v. 1 ************************')
    # print('************************************************************************')
    # print(' ');
    # print(f'Project: {PID}');
    print_project_id(PID)




    # Record calculation time
    tic = time.time();

    # Run pipe dimensioning
    # KART indført aggregeret last
    # net = run_pipedimensioning(d_pipes, brine, net, hp)
    net, aggLoad = run_pipedimensioning(d_pipes, brine, net, hp)


    # Print results to console
    print_pipe_dimensions(net, pipeGroupNames)

    # Run source dimensioning
    # KART indført aggregeret last
    # FPH, FPC, source_config = run_sourcedimensioning(brine, net, hp, source_config)
    # FPH, FPC, source_config = run_sourcedimensioning(brine, net, aggLoad, source_config)
    source_config = run_sourcedimensioning(brine, net, aggLoad, source_config)

    # Print results to console
    print_source_dimensions(source_config,net)

    # Output computation time to console
    print(' ');
    print('*************************** Computation time ***************************');
    toc = time.time();                                                  # Track computation time (s)
    print(f'Elapsed time: {round(toc-tic,6)} seconds');


    #KART + LASSE: PAK FPH OG FPC IND I SOURCE_CONFIG?
    FPH = source_config.FPH;
    FPC = source_config.FPC;
    
    return net, FPH, FPC, source_config


def run_full_dimensioning_single_combined_input(pid:str, config: FullDimension):
    config.thermonet, config.FPH, config.FPC, config.source_config = run_full_dimensioning(pid, d_pipes=config.d_pipes,
                          brine=config.brine,
                          net=config.thermonet,
                          hp=config.heatpump,
                          pipeGroupNames=config.pipe_group_name,
                          source_config=config.source_config
                          )
    return config