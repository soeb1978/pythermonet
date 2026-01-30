from typing import Literal, Iterable
import pandas as pd

from pythermonet.domain.thermonet import Thermonet
from pythermonet.domain.infrastructure import Pricing
from pythermonet.data.equipment.pipes import load_pipe_catalogue
from pythermonet.data.misc.entrepeneur_prices import EntrepeneurPrices
from pythermonet.data.equipment.component_cost import ComponentCost
from pythermonet.domain import BHEConfig, HHEConfig

entrep_prices = EntrepeneurPrices()
pipe_catalogue = load_pipe_catalogue()

class CalcPriceInput:
    net_pipe_dims = None
    net_pipe_lengths = None
    net_price_pr_m = None
    source_pipe_dims = None
    source_pipe_lengths = None
    source_price_pr_m = None


def _as_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, (str, bytes)):
        return [x]
    if isinstance(x, Iterable):
        return list(x)
    return [x]


def _calc_pipe_prices(pipe_dims: list[float], pipe_lengths: list[int], calc_price: Pricing, pipe_type: Literal["Thermal Exchange Pipe","Borehole Heat Exchanger"]) -> None:
    if calc_price.pipe_cost is None:
        pipe_prices = {}
    else:
        pipe_prices = calc_price.pipe_cost

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

    calc_price.pipe_cost=pipe_prices


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


def _calc_price_by_m(net_length: float, net_price_pr_m: int, source_length: float, source_price_pr_m: int, calc_price: Pricing):
    assert net_price_pr_m > 0, "Price per meter must be above 0"
    assert source_price_pr_m > 0, "Price per meter must be above 0"

    net_price = sum(net_length)*net_price_pr_m
    source_price = sum(source_length)*source_price_pr_m if isinstance(source_length, list) else source_length*source_price_pr_m

    calc_price.entrep_labor_cost = net_price
    calc_price.borehole_labor_cost = source_price

    calc_price.total_cost += (net_price+source_price)
    

def _calc_component_price(num_hps: int, calc_price: Pricing) -> None:
    component_cost = ComponentCost()

    heatpump_cost = component_cost.heatpump_price * num_hps
    energy_meter_cost = component_cost.energy_meter_price * num_hps

    calc_price.component_cost["heatpump_cost"] = heatpump_cost
    calc_price.component_cost["energy_meter_cost"] = energy_meter_cost

    calc_price.total_cost += (heatpump_cost+energy_meter_cost)


def _calc_total_cost(input: CalcPriceInput, calc_price: Pricing, num_hps: int, advanced: bool) -> None:
    grid_pipe_dim = input.net_pipe_dims
    grid_pipe_length = input.net_pipe_lengths

    source_pipe_dim = input.source_pipe_dims
    source_pipe_length = input.source_pipe_lengths

    _calc_pipe_prices(grid_pipe_dim, grid_pipe_length, calc_price, "Thermal Exchange Pipe")
    _calc_pipe_prices(source_pipe_dim, source_pipe_length, calc_price, "Borehole Heat Exchanger")    
    
    if advanced:
        _calc_labor_prices(calc_price)    
    else:
        _calc_price_by_m(grid_pipe_length, input.net_price_pr_m, source_pipe_length, input.source_price_pr_m, calc_price)
    
    _calc_component_price(num_hps, calc_price)

    return None

def _set_calc_model(net: Thermonet, source_config: HHEConfig|BHEConfig, advanced: bool, net_price_pr_m: int, source_price_pr_m: int):
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

    if not advanced:
        heating_model.net_price_pr_m = net_price_pr_m
        cooling_model.net_price_pr_m = net_price_pr_m

        heating_model.source_price_pr_m = source_price_pr_m
        cooling_model.source_price_pr_m = source_price_pr_m


    return heating_model, cooling_model

def calc_pipe_cost(net: Thermonet, source_config: HHEConfig|BHEConfig, num_hps: int, advanced: bool = False, net_price_pr_m: int = 300, source_price_pr_m: int = 650) -> dict[str,Pricing]:
    """
    This model calculates the total cost of buying and installing the piping for a thermonet. It is based on the
    pipes in /src/pythermonet/data/equipment/PIPES.dat and the prices in /src/pythermonet/data/misc/entrepeneur_prices.py

    The function can be toggled to advanced or not. This defaults to using either a simple price estimation on a price pr 
    meter for the net and for the boreholes or a more detailed version. If the simple version is toggled by setting advanced = False,
    the user has to supply values for net_price_pr_m and source_price_pr_m.
    The net is the cost for establishing the horizontal grid (the thermonet) while the source is the designated heat generator (e.g. horizontal heat exchanger or
    borehole heat exchanger).
    
    :param net: Has to be the Thermonet dataclass. 
    :type net: Thermonet
    :param source_config: Has to be either a HHEConfig or BHEConfig dataclass. It contains how many meters of which diameters
    there are needed to create the thermonet.
    :type source_config: HHEConfig | BHEConfig
    :param advanced: If true, uses the in depth prices of the various types of entrepeneurs so that the user can define the detail level as they wish. If false, the user can define a price pr running meter in the grid and the borehole for a simplified estimation.
    :type advanced: bool
    :param net_price_pr_m: The price pr meter for establishing a thermonet
    :type net_price_pr_m: int
    :param source_price_pr_m: The price pr meter for establishing the heat source
    :type source_price_pr_m: int
    :return: The pricing estimation for heating and cooling.
    :rtype: dict[str, Pricing]
    """
    # Can set pricing pr meter by input parameters here
    models = _set_calc_model(net, source_config, advanced, net_price_pr_m=net_price_pr_m, source_price_pr_m=source_price_pr_m) # Heating model, cooling model
    
    completed_models = []
    for model in models:
        calc_price = Pricing()
        _calc_total_cost(model, calc_price, num_hps, advanced)
        completed_models.append(calc_price)

    return {"heating": completed_models[0], "cooling": completed_models[1]}

