import numpy as np


def mass_flow_from_load(
        load: float, delta_temp: float, density: float, heat_capacity: float
        ) -> float:
    """
    Calculates the mass flow rate required to handle the specified
    thermal load, given the fluid properties and the temperature
    difference across the heat pump.

    Parameters
    ----------
    load : float
        The thermal load [W].

    delta_temp : float
        The temperature difference across the heat pump [K or °C].

    density : float
        Density of the fluid [kg/m³].

    heat_capacity : float
        Specific heat capacity of the fluid [J/(kg·K)].

    Returns
    -------
    mass_flow : float
        Required mass flow rate [kg/s].
    """
    mass_flow = load / (delta_temp * density * heat_capacity)
    return mass_flow


def flow_velocity_from_volumetric_flow(
        volumetric_flow: float | np.ndarray,
        pipe_inner_diameter:  float | np.ndarray,
        ) -> float | np.ndarray:
    """
    Calculates fluid velocity in a circular pipe from volumetric flow.

    This function computes the average flow velocity based on the
    volumetric flow rate and the pipe's inner diameter, assuming a
    circular cross-section.

    Parameters
    ----------
    volumetric_flow : float or ndarray
        Volumetric flow rate [m³/s].

    pipe_inner_diameter : float or ndarray
        Inner diameter of the pipe [m].

    Returns
    -------
    velocity : float or ndarray
        Average velocity of the fluid in the pipe [m/s].
    """
    return np.divide(4 * volumetric_flow, np.pi * pipe_inner_diameter**2)


def pipe_inner_diameter(
        outer_diameter: float | np.ndarray, SDR: float | np.ndarray = 17.0
        ) -> float | np.ndarray:
    """
    Calculates the inner pipe diameter given the outer diameter and the
    SDR

    Args
    :param outer_diameter: The outer diameter of the pipes
    :type  outer_diameter: float or list of floats
    :param SDR: The surface to diameter ratio of the pipe
    :type  SDR: float or list of floats

    Return
    :param -: The inner diameter of the pipes
    :type  -: float or list of floats

    """
    return np.multiply(outer_diameter, (1. - np.divide(2., SDR)))


def pipe_outer_diameter(
        inner_diameter: float | np.ndarray, SDR: float | np.ndarray = 17.0
        ) -> float | np.ndarray:
    """
    Calculate the outer pipe diameter given the inner diameter and SDR.

    Parameters
    ----------
    inner_diameter : float or list of floats
        The inner diameter(s) of the pipe(s).
    SDR : float or list of floats
        The standard dimension ratio of the pipe(s).

    Returns
    -------
    float or list of floats
        The outer diameter(s) of the pipe(s).
    """
    return np.divide(inner_diameter, (1. - np.divide(2., SDR)))


def pipe_brine_volume(
        pipe_length: float | np.ndarray,
        pipe_inner_diameter: float | np.ndarray
        ) -> float | np.ndarray:

    return np.multiply(pipe_length * np.pi / 4, pipe_inner_diameter**2)
