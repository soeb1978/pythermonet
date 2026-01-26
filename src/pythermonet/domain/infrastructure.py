from dataclasses import dataclass
import numpy as np

@dataclass
class Pricing:
    total_cost: int | float = 0
    component_cost: dict = None
    entrep_labor_cost: float = np.nan
    borehole_labor_cost: float = np.nan
