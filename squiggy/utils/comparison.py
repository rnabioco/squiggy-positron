"""Comparison and delta utilities for Squiggy"""

import numpy as np


def compare_read_sets(read_ids_a: list[str], read_ids_b: list[str]) -> dict:
    """
    Compare two sets of read IDs and find common/unique reads

    Analyzes overlap between two read ID lists, useful for comparing
    which reads were sequenced in each sample.

    Args:
        read_ids_a: List of read IDs from sample A
        read_ids_b: List of read IDs from sample B

    Returns:
        Dict with comparison results:
            - common_reads (set): Read IDs present in both
            - unique_to_a (set): Read IDs only in A
            - unique_to_b (set): Read IDs only in B
            - common_count (int): Number of common reads
            - unique_a_count (int): Number unique to A
            - unique_b_count (int): Number unique to B
            - overlap_percent_a (float): Percentage of A's reads in common
            - overlap_percent_b (float): Percentage of B's reads in common

    Examples:
        >>> from squiggy.utils import compare_read_sets
        >>> result = compare_read_sets(reads_a, reads_b)
        >>> print(f"Common reads: {result['common_count']}")
        >>> print(f"A unique: {result['unique_a_count']}")
        >>> print(f"B unique: {result['unique_b_count']}")
    """
    set_a = set(read_ids_a)
    set_b = set(read_ids_b)

    common = set_a & set_b
    unique_a = set_a - set_b
    unique_b = set_b - set_a

    # Calculate overlap percentages
    overlap_percent_a = (len(common) / len(set_a) * 100) if set_a else 0
    overlap_percent_b = (len(common) / len(set_b) * 100) if set_b else 0

    return {
        "common_reads": common,
        "unique_to_a": unique_a,
        "unique_to_b": unique_b,
        "common_count": len(common),
        "unique_a_count": len(unique_a),
        "unique_b_count": len(unique_b),
        "overlap_percent_a": overlap_percent_a,
        "overlap_percent_b": overlap_percent_b,
    }


def calculate_delta_stats(
    stats_a: dict, stats_b: dict, stat_names: list[str] | None = None
) -> dict:
    """
    Calculate differences (deltas) between corresponding statistics

    Computes delta values for statistics arrays, useful for visualizing
    differences between two basecalling models or runs.

    Args:
        stats_a: Dictionary of statistics from sample A
                (e.g., from calculate_aggregate_signal())
        stats_b: Dictionary of statistics from sample B (same structure)
        stat_names: List of stat keys to compute deltas for
                   (default: all matching keys with array values)

    Returns:
        Dict with delta arrays:
            - delta_{stat_name}: Array of differences (B - A)
            - positions: Position array (if available)

    Examples:
        >>> from squiggy.utils import calculate_delta_stats
        >>> delta = calculate_delta_stats(stats_a, stats_b, ['mean_signal'])
        >>> print(f"Max delta: {np.max(np.abs(delta['delta_mean_signal']))}")
    """
    deltas = {}

    # If no stat names provided, infer from matching keys
    if stat_names is None:
        stat_names = []
        for key in stats_a.keys():
            if key in stats_b and isinstance(stats_a[key], np.ndarray):
                stat_names.append(key)

    # First pass: determine minimum length across all arrays
    min_len = None
    for stat_name in stat_names:
        if stat_name not in stats_a or stat_name not in stats_b:
            continue

        val_a = stats_a[stat_name]
        val_b = stats_b[stat_name]

        if isinstance(val_a, np.ndarray) and isinstance(val_b, np.ndarray):
            curr_min = min(len(val_a), len(val_b))
            if min_len is None:
                min_len = curr_min
            else:
                min_len = min(min_len, curr_min)

    # If no arrays found, return empty
    if min_len is None:
        return deltas

    # Calculate deltas (all will be same length now)
    for stat_name in stat_names:
        if stat_name not in stats_a or stat_name not in stats_b:
            continue

        val_a = stats_a[stat_name]
        val_b = stats_b[stat_name]

        if isinstance(val_a, np.ndarray) and isinstance(val_b, np.ndarray):
            delta = val_b[:min_len] - val_a[:min_len]
            deltas[f"delta_{stat_name}"] = delta

    # Include positions if available - TRUNCATE to match delta length
    if "positions" in stats_a:
        deltas["positions"] = stats_a["positions"][:min_len]
    elif "positions" in stats_b:
        deltas["positions"] = stats_b["positions"][:min_len]

    return deltas


def compare_signal_distributions(signal_a: np.ndarray, signal_b: np.ndarray) -> dict:
    """
    Compare signal distributions from two samples

    Computes statistical measures to characterize differences in signal
    distributions between two samples.

    Args:
        signal_a: Signal array from sample A
        signal_b: Signal array from sample B

    Returns:
        Dict with distribution comparison:
            - mean_a, mean_b: Mean signal
            - median_a, median_b: Median signal
            - std_a, std_b: Standard deviation
            - min_a, min_b: Minimum signal
            - max_a, max_b: Maximum signal
            - mean_diff: Difference in means
            - std_diff: Difference in standard deviations

    Examples:
        >>> from squiggy.utils import compare_signal_distributions
        >>> result = compare_signal_distributions(signal_a, signal_b)
        >>> print(f"Mean difference: {result['mean_diff']:.2f}")
    """
    return {
        "mean_a": float(np.mean(signal_a)),
        "mean_b": float(np.mean(signal_b)),
        "median_a": float(np.median(signal_a)),
        "median_b": float(np.median(signal_b)),
        "std_a": float(np.std(signal_a)),
        "std_b": float(np.std(signal_b)),
        "min_a": float(np.min(signal_a)),
        "min_b": float(np.min(signal_b)),
        "max_a": float(np.max(signal_a)),
        "max_b": float(np.max(signal_b)),
        "mean_diff": float(np.mean(signal_b) - np.mean(signal_a)),
        "std_diff": float(np.std(signal_b) - np.std(signal_a)),
    }
