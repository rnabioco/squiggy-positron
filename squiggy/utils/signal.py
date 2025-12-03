"""Signal processing utilities for Squiggy"""

import numpy as np


def downsample_signal(signal: np.ndarray, downsample_factor: int = None) -> np.ndarray:
    """Downsample signal array by taking every Nth point

    Reduces the number of data points for faster plotting while preserving
    the overall shape of the signal.

    Args:
        signal: Raw signal array (numpy array)
        downsample_factor: Factor by which to downsample (None = use default, 1 = no downsampling)

    Returns:
        Downsampled signal array
    """
    from ..constants import DEFAULT_DOWNSAMPLE

    if downsample_factor is None:
        downsample_factor = DEFAULT_DOWNSAMPLE

    if downsample_factor <= 1:
        return signal

    # Take every Nth point
    return signal[::downsample_factor]


def calculate_aligned_move_indices(
    moves: np.ndarray, query_start_offset: int, query_end_offset: int
) -> tuple[np.ndarray, set[int]]:
    """
    Calculate move table indices corresponding to aligned (non-soft-clipped) bases

    This utility function filters a move table to identify which move indices correspond
    to bases that are actually aligned to the reference (i.e., not soft-clipped).

    BAM files contain soft-clipped bases at the start/end of alignments that don't match
    the reference. The move table includes moves for ALL bases in the query sequence,
    including soft-clipped ones. When mapping signal to reference positions, we must
    skip the soft-clipped bases to avoid incorrect position assignments.

    Args:
        moves: Move table array (stride already removed, just 0s and 1s)
               where 1 indicates a new base starts at this signal position
        query_start_offset: Number of soft-clipped bases at start of alignment
                           (extracted from CIGAR, e.g., 4S7M3S -> offset=4)
        query_end_offset: Number of soft-clipped bases at end of alignment
                         (extracted from CIGAR, e.g., 4S7M3S -> offset=3)

    Returns:
        Tuple of (aligned_indices_array, aligned_indices_set):
        - aligned_indices_array: np.ndarray of move table indices for aligned bases only
        - aligned_indices_set: Set version for O(1) membership testing

    Examples:
        >>> moves = np.array([0, 1, 0, 0, 1, 0, 1, 0, 0, 1])
        >>> # Bases are at indices: [1, 4, 6, 9] (4 bases total)
        >>> # CIGAR shows 1S2M1S (1 soft-clip start, 2 matched, 1 soft-clip end)
        >>> aligned, aligned_set = calculate_aligned_move_indices(
        ...     moves, query_start_offset=1, query_end_offset=1
        ... )
        >>> print(aligned)  # [4, 6] (only the 2 middle aligned bases)
        >>> print(4 in aligned_set)  # True (base at index 4 is aligned)
        >>> print(1 in aligned_set)  # False (base at index 1 is soft-clipped)

    Technical Details:
        Move table format: [stride, move1, move2, ...]
        - stride: Neural network downsampling factor (5 for DNA, 10-12 for RNA)
        - moves: Array of 0s and 1s
          - 1 = New base starts at this signal position
          - 0 = Signal continues from previous base

        CIGAR codes: 4 = Soft clip, 0 = Match, 1 = Insertion, 2 = Deletion

    References:
        - Issue #88: Extend soft-clip boundary handling
        - PR #85: Original fix for signal_overlay_comparison
        - SAM/BAM spec: https://samtools.github.io/hts-specs/SAMv1.pdf
    """
    # Find all indices where move=1 (these are base positions in query sequence)
    query_base_move_indices = np.where(moves == 1)[0]

    if len(query_base_move_indices) == 0:
        # No bases found in move table
        return np.array([], dtype=int), set()

    # Calculate slice indices to exclude soft-clipped bases
    # Example: If we have 4 bases [1, 4, 6, 9] with offsets (1, 1)
    #          start_idx=1, end_idx=4-1=3
    #          Slice [1:3] gives indices [4, 6] (middle 2 bases)
    start_idx = query_start_offset
    end_idx = len(query_base_move_indices) - query_end_offset

    # Handle edge case: all bases are soft-clipped
    if start_idx >= end_idx:
        return np.array([], dtype=int), set()

    # Slice to get only aligned base indices
    aligned_indices = query_base_move_indices[start_idx:end_idx]

    # Create set for O(1) membership testing in loops
    aligned_set = set(aligned_indices)

    return aligned_indices, aligned_set


def get_aligned_move_indices_from_read(read: dict) -> tuple[np.ndarray, set[int]]:
    """
    Extract aligned move indices from a read dict (convenience wrapper)

    This is a convenience function that extracts the move table and soft-clip offsets
    from a read dict and calls calculate_aligned_move_indices(). Reduces code
    duplication in functions that process multiple reads.

    Args:
        read: Read dict from extract_reads_for_reference() containing:
              - move_table: np.ndarray of move values
              - query_start_offset: int (soft-clipped bases at start, default 0)
              - query_end_offset: int (soft-clipped bases at end, default 0)

    Returns:
        Tuple of (aligned_indices_array, aligned_indices_set)
        Same as calculate_aligned_move_indices()

    Examples:
        >>> for read in reads_data:
        >>>     aligned_indices, aligned_set = get_aligned_move_indices_from_read(read)
        >>>     if len(aligned_indices) == 0:
        >>>         continue  # Skip reads with no aligned bases
        >>>     # Process aligned bases...
    """
    moves = read["move_table"]
    query_start_offset = read.get("query_start_offset", 0)
    query_end_offset = read.get("query_end_offset", 0)

    return calculate_aligned_move_indices(moves, query_start_offset, query_end_offset)


def iter_aligned_bases(read: dict):
    """
    Iterate over aligned bases in a read, yielding position information

    This generator yields information about each aligned (non-soft-clipped) base,
    handling the common pattern of:
    - Skipping soft-clipped bases
    - Tracking aligned base index
    - Tracking sequence index
    - Mapping to reference positions (correctly handling insertions/deletions)

    Args:
        read: Read dict from extract_reads_for_reference()

    Yields:
        Tuples of (move_idx, base_idx, seq_idx, ref_pos) for each aligned base:
        - move_idx: Index in move table where this base starts
        - base_idx: Index among aligned bases (0-based, excludes soft-clipped)
        - seq_idx: Index in query sequence (includes soft-clipped bases)
        - ref_pos: Reference genome position (correctly accounts for indels)

    Examples:
        >>> for move_idx, base_idx, seq_idx, ref_pos in iter_aligned_bases(read):
        >>>     # Process this aligned base
        >>>     base = read["sequence"][seq_idx]
        >>>     qual = read["quality_scores"][seq_idx]
        >>>     # Map to reference position ref_pos
    """
    moves = read["move_table"]
    query_to_ref = read.get("query_to_ref", None)
    ref_start = read.get("reference_start")

    # Get aligned move indices (skips soft-clipped bases)
    aligned_indices, aligned_set = get_aligned_move_indices_from_read(read)

    if len(aligned_indices) == 0:
        # No aligned bases in this read
        return

    base_idx = 0  # Index for aligned bases only
    seq_idx = 0  # Index in query sequence (includes soft-clipped bases)

    for move_idx, move in enumerate(moves):
        if move == 1:
            # This is a base position
            if move_idx in aligned_set:
                # This base is aligned (not soft-clipped)
                if query_to_ref is not None:
                    # Use query_to_ref mapping which handles insertions/deletions correctly
                    ref_pos = query_to_ref.get(seq_idx)
                else:
                    # Fallback for test/legacy data: assume consecutive positions
                    # This doesn't handle deletions/insertions correctly!
                    ref_pos = ref_start + base_idx if ref_start is not None else None

                if ref_pos is not None:
                    yield move_idx, base_idx, seq_idx, ref_pos
                base_idx += 1

            seq_idx += 1
