# Standard library imports
from dataclasses import dataclass
from typing import List

# Third-party imports
from dataclasses_json import dataclass_json
import numpy as np

# Local application imports
from .bhe_config import BHEConfig
from .brine import Brine
from .heatpump import HeatPump
from .hhe_config import HHEConfig
from .thermonet import Thermonet


@dataclass_json
@dataclass
class FullDimension:
    brine: Brine
    thermonet: Thermonet
    heatpump: HeatPump
    source_config: HHEConfig | BHEConfig
    pipe_group_name: List[str]
    d_pipes: List[float]
    FPH: float = np.nan
    FPC: float = np.nan
