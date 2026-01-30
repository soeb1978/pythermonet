from typing import Literal
from dataclasses import fields

from pythermonet.domain.infrastructure import Pricing

def _print_pipe_dim_dict(object: dict):
    for item in object:
        print(f"Ø {item*1000}mm: {object[item][0]}")


def _print_single_lvl_dict(object: dict):
    for item in object:
        print(f"{item}: {object[item]}")

def print_pricing(calculated_price: Pricing, modes: Literal["heating", "cooling"] = ["heating", "cooling"]) -> None:
    print('***************** Pipe cost estimation *****************')
    print('Pipe prices defined in data/equipment/PIPES.dat')

    for mode in modes:
        print(" ")
        print(f"Price for {mode}:")

        for pricing in fields(calculated_price[mode]):
            
            if pricing.type == int | float or pricing.type == int or pricing.type == float: 
                print(f"{pricing.name}: {int(getattr(calculated_price[mode], pricing.name))}")
            elif pricing.type == dict:
                if pricing.name == "pipe_cost":
                    _print_pipe_dim_dict(getattr(calculated_price[mode], pricing.name))
                elif pricing.name == "component_cost":
                    _print_single_lvl_dict(getattr(calculated_price[mode], pricing.name))

    print(" ")


