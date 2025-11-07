"""Signal normalization methods for nanopore data"""

import numpy as np

from .constants import NormalizationMethod


def normalize_signal(signal: np.ndarray, method: NormalizationMethod) -> np.ndarray:
    """Normalize signal data using specified method

    Args:
        signal: Raw signal array (numpy array)
        method: Normalization method to apply

    Returns:
        Normalized signal array
    """
    if method == NormalizationMethod.NONE:
        return signal
    elif method == NormalizationMethod.ZNORM:
        return z_normalize(signal)
    elif method == NormalizationMethod.MEDIAN:
        return median_normalize(signal)
    elif method == NormalizationMethod.MAD:
        return mad_normalize(signal)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def z_normalize(signal: np.ndarray) -> np.ndarray:
    """Z-score normalization (mean=0, std=1)

    This is the most common normalization for comparing signals across reads.

    Args:
        signal: Raw signal array

    Returns:
        Z-normalized signal
    """
    mean = np.mean(signal)
    std = np.std(signal)
    if std == 0:
        return signal - mean
    return (signal - mean) / std


def median_normalize(signal: np.ndarray) -> np.ndarray:
    """Median-centered normalization

    Centers signal around 0 by subtracting the median.
    More robust to outliers than mean-based normalization.

    Args:
        signal: Raw signal array

    Returns:
        Median-normalized signal
    """
    median = np.median(signal)
    return signal - median


def mad_normalize(signal: np.ndarray) -> np.ndarray:
    """Median absolute deviation (MAD) normalization

    Robust normalization method that's less sensitive to outliers
    than z-score normalization. Centers by median and scales by MAD.

    Args:
        signal: Raw signal array

    Returns:
        MAD-normalized signal
    """
    median = np.median(signal)
    mad = np.median(np.abs(signal - median))
    if mad == 0:
        return signal - median
    return (signal - median) / (1.4826 * mad)  # 1.4826 makes MAD consistent with std
