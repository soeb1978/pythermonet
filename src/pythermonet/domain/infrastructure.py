from dataclasses import dataclass, field
import numpy as np

@dataclass
class Pricing:
    total_cost: int | float = 0
    pipe_cost: dict = None
    component_cost: dict = field(default_factory= lambda: {})
    entrep_labor_cost: float = np.nan
    borehole_labor_cost: float = np.nan
