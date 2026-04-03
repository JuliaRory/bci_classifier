from numpy import finfo, log10

def unit_to_db(value, eps=None):
    """
    Convert value (e.g. PSD) to decibel (dB) scale.

    Parameters
    ----------
    value : array-like
    eps : float or None, optional
        Small value added to avoid log(0). If None, uses machine epsilon.

    Returns
    -------
    value_dB : ndarray
        value values in decibels (dB).
    """
    if eps is None:
        eps = finfo(float).eps
    return 10 * log10(value + eps)