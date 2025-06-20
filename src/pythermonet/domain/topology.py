from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from pythermonet.core.pipe_hydraulics import (
    pipe_inner_diameter,
    pipe_outer_diameter
)

@dataclass
class DimensionedTopologyInput:
    standard_dimension_ratio: np.ndarray # [-]
    trace_lengths: np.ndarray   # [m]
    number_of_traces: np.ndarray  # [-]

    outer_diameter: Optional[np.ndarray] = None  # [m]
    inner_diameter: Optional[np.ndarray] = None  # [m]
    peak_volumentric_flow_heating: Optional[np.ndarray] = None  # [m³/s]
    peak_volumentric_flow_cooling: Optional[np.ndarray] = None  # [m³/s]
    peak_reynold_heating: Optional[np.ndarray] = None  # [-]
    peak_reynold_cooling: Optional[np.ndarray] = None  # [-]

    pipe_group_names: List[str] = field(default_factory=list)

    has_heating: bool = field(init=False)
    has_cooling: bool = field(init=False)

    def __post_init__(self):

        # calculate the unspecified pipe diameters
        if self.inner_diameter is None:
            self.inner_diameter = pipe_inner_diameter(
                self.outer_diameter, self.standard_dimension_ratio
            )
        elif self.outer_diameter is None:
            self.outer_diameter = pipe_outer_diameter(
                self.inner_diameter, self.standard_dimension_ratio
            )

        # Determine if heating or cooling is present
        self.has_heating = self._array_has_nonzero_values(
            self.peak_volumentric_flow_heating, self.peak_reynold_heating
        )
        self.has_cooling = self._array_has_nonzero_values(
            self.peak_volumentric_flow_cooling, self.peak_reynold_cooling
        )

    @staticmethod
    def _array_has_nonzero_values(*arrays: Optional[np.ndarray]) -> bool:
        for arr in arrays:
            if arr is not None and np.any(np.abs(arr) > 1e-6):
                return True
        return False
