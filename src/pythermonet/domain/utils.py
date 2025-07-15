import numpy as np

def count_active_consumers(
        load_vector: np.ndarray, threshold: float = 1e-6
        ) -> int:
    """
    Counts the number of active consumers in a load array.

    Parameters
    ----------
    P_s : np.ndarray
        Array of heat pump loads (rows = heat pumps, columns = time scales).

    threshold : float, optional
        Values below this threshold are considered zero 
        (default is 1e-6).

    Returns
    -------
    int
        Number of consumers with non-zero load in the specified column.
    """
    if load_vector is None or load_vector.size == 0:
        return 0
    return int(np.count_nonzero(np.abs(load_vector) > threshold))
