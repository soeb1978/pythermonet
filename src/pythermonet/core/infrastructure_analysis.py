from typing import Literal, Iterable
import pandas as pd

from pythermonet.domain.thermonet import Thermonet
from pythermonet.domain.infrastructure import Pricing
from pythermonet.data.equipment.pipes import load_pipe_catalogue
from pythermonet.data.misc.entrepeneur_prices import EntrepeneurPrices
from pythermonet.domain import BHEConfig, HHEConfig

entrep_prices = EntrepeneurPrices()
pipe_catalogue = load_pipe_catalogue()

class CalcPriceInput:
    net_pipe_dims = None
    net_pipe_lengths = None
    source_pipe_dims = None
    source_pipe_lengths = None

def _as_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, (str, bytes)):
        return [x]
    if isinstance(x, Iterable):
        return list(x)
    return [x]

def _calc_pipe_prices(pipe_dims: list[float], pipe_lengths: list[int], calc_price: Pricing, pipe_type: Literal["Thermal Exchange Pipe","Borehole Heat Exchanger"]) -> None:
    if calc_price.component_cost is None:
        pipe_prices = {}
    else:
        pipe_prices = calc_price.component_cost

    pipe_dims = _as_list(pipe_dims)
    pipe_lengths = _as_list(pipe_lengths)


    for index, dim in enumerate(pipe_dims):
        # price_pr_meter = pipe_catalogue[(pipe_catalogue["Pipe diameters (mm)"] == dim*1000) & (pipe_catalogue["Type"] == pipe_type)]
        sorted_diameter = pipe_catalogue[pipe_catalogue["Pipe diameters (mm)"] == dim*1000]
        price_pr_meter = sorted_diameter[sorted_diameter["Type"] == pipe_type]
        if price_pr_meter.empty:
            print(f"Pipe dimension {dim} does not exist in the pipe catalogue")
            return
        
        piping_price = price_pr_meter["Price pr meter"].values * pipe_lengths[index]
        pipe_prices[dim] = piping_price
        calc_price.total_cost += piping_price

    calc_price.component_cost=pipe_prices


def _calc_worker_prices(chosen_worker: dict) -> int:
    price = 0
    for key in chosen_worker:
        entr_position = chosen_worker.get(key)
        hourly_pay = entr_position.get("hourly_pay")
        hours_required = entr_position.get("number_of_hours_required")
        
        price += hourly_pay * hours_required

    return price


def _calc_labor_prices(calc_price: Pricing) -> None:
    bore_specialist_price = _calc_worker_prices(entrep_prices.borehole_specialist)
    entrep_price = _calc_worker_prices(entrep_prices.entrepeneurs)

    calc_price.borehole_labor_cost = bore_specialist_price
    calc_price.entrep_labor_cost = entrep_price

    calc_price.total_cost += (bore_specialist_price + entrep_price)


def _calc_total_cost(input: CalcPriceInput, calc_price: Pricing) -> None:
    grid_pipe_dim = input.net_pipe_dims
    grid_pipe_length = input.net_pipe_lengths

    source_pipe_dim = input.source_pipe_dims
    source_pipe_length = input.source_pipe_lengths

    # pipe_dims: list[float], pipe_lengths: list[int]
    _calc_pipe_prices(grid_pipe_dim, grid_pipe_length, calc_price, "Thermal Exchange Pipe")
    _calc_labor_prices(calc_price)
    
    _calc_pipe_prices(source_pipe_dim, source_pipe_length, calc_price, "Borehole Heat Exchanger")    

    return None

def _set_calc_model(net: Thermonet, source_config: HHEConfig|BHEConfig):
    """
    Returns a model for calculating heating and cooling in that order
    """

    cooling_model = CalcPriceInput()
    heating_model = CalcPriceInput()

    heating_model.net_pipe_dims = net.d_selectedPipes_H
    cooling_model.net_pipe_dims = net.d_selectedPipes_C
    
    heating_model.net_pipe_lengths = net.L_segments #Defined in meters
    cooling_model.net_pipe_lengths = net.L_segments #Defined in meters

    if isinstance(source_config, HHEConfig):
        heating_model.source_pipe_dims =  source_config.d
        cooling_model.source_pipe_dims =  source_config.d
        heating_model.source_pipe_lengths = source_config.L_HHE_H
        cooling_model.source_pipe_lengths = source_config.L_HHE_C
    elif isinstance(source_config, BHEConfig):
        heating_model.source_pipe_dims = source_config.D_pipes
        cooling_model.source_pipe_dims = source_config.D_pipes
        heating_model.source_pipe_lengths = source_config.L_BHE_H
        cooling_model.source_pipe_lengths = source_config.L_BHE_C


    return heating_model, cooling_model

def calc_pipe_cost(net: Thermonet, source_config: HHEConfig|BHEConfig) -> dict[str,Pricing]:

    # Iterate over models, and create a _calc_total_cost on these
    # consider if two models should be present (heating and cooling)
    # or if one is enough
    
    models = _set_calc_model(net, source_config) # Heating model, cooling model
    
    completed_models = []

    for model in models:
        calc_price = Pricing()
        _calc_total_cost(model, calc_price)

        completed_models.append(calc_price)


    
    return {"heating": completed_models[0], "cooling": completed_models[1]}

