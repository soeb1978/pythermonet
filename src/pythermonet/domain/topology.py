from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class DimensionedTopologyInput:
    standard_dimension_ratio: np.ndarray
    outer_diameter: np.ndarray
    trace_lengths: np.ndarray
    number_of_traces: np.ndarray
    peak_flow_heating: np.ndarray
    peak_flow_cooling: np.ndarray
    pipe_group_names: List[str] = field(default_factory=list)
