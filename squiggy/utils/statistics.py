"""Aggregate statistics utilities for Squiggy"""

import numpy as np

from .signal import get_aligned_move_indices_from_read, iter_aligned_bases


def calculate_modification_statistics(
    reads_data, mod_filter=None, min_frequency=0.0, min_modified_reads=1
):
    """Calculate aggregate modification statistics across multiple reads

    Args:
        reads_data: List of read dicts from extract_reads_for_reference()
        mod_filter: Optional dict mapping mod_code -> minimum probability threshold
                   (e.g., {'m': 0.8, 'a': 0.7}). If None, all modifications included.

                   Filter behavior:
                   - Disabled modification types (not in filter dict) are excluded entirely
                   - Only modifications with probability >= threshold are counted
                   - mean/median/std are calculated from filtered modifications only
                   - Positions are only output if they have at least one mod >= threshold
        min_frequency: Minimum fraction of reads that must be modified (0.0-1.0, default 0.0).
                      Positions with lower modification frequency are excluded.
        min_modified_reads: Minimum number of reads that must have the modification (default 1).
                           Positions with fewer modified reads are excluded.

    Returns:
        Dict with keys:
            - mod_stats: Dict mapping mod_code -> position -> statistics
                {
                    'mod_code': {
                        position: {
                            'probabilities': [float],  # Probabilities >= threshold
                            'mean': float,              # Mean of filtered probabilities
                            'median': float,            # Median of filtered probabilities
                            'std': float,               # Std dev of filtered probabilities
                            'count': int,               # Count of reads with prob >= threshold
                            'total_coverage': int,      # Total reads covering this position
                            'frequency': float          # count / total_coverage
                        }
                    }
                }
            - positions: Sorted list of all positions with modifications >= threshold

    Example interpretation:
        frequency=0.03, mean=0.92, count=3, total_coverage=100
        → "3% of reads are modified (3 out of 100)"
        → "Among those 3 modified reads, average probability is 0.92"
    """
    # Map genomic_pos -> mod_code -> list of probabilities
    # We collect TWO sets of data:
    # 1. position_mods_all: ALL probabilities (kept for reference, not used in stats)
    # 2. position_mods_filtered: Only probabilities >= threshold (used for ALL statistics)
    # This ensures frequency and mean probability use the same denominator
    position_mods_all = {}
    position_mods_filtered = {}

    for read in reads_data:
        modifications = read.get("modifications", [])
        for mod in modifications:
            if mod.genomic_pos is None:
                continue

            pos = mod.genomic_pos
            mod_code = mod.mod_code

            # Convert mod_code to string for consistent comparison with filter keys
            mod_code_str = str(mod_code)

            # Skip if mod_code not in filter (user disabled this modification type)
            if mod_filter is not None and mod_code_str not in mod_filter:
                continue

            # Always add to ALL probabilities (for unbiased statistics)
            if pos not in position_mods_all:
                position_mods_all[pos] = {}
            if mod_code not in position_mods_all[pos]:
                position_mods_all[pos][mod_code] = []
            position_mods_all[pos][mod_code].append(mod.probability)

            # Add to filtered probabilities only if meets threshold
            # This is used for count and frequency calculations
            if mod_filter is None or mod.probability >= mod_filter[mod_code_str]:
                if pos not in position_mods_filtered:
                    position_mods_filtered[pos] = {}
                if mod_code not in position_mods_filtered[pos]:
                    position_mods_filtered[pos][mod_code] = []
                position_mods_filtered[pos][mod_code].append(mod.probability)

    # Calculate total coverage per position (all reads, not just modified ones)
    position_coverage = {}
    for read in reads_data:
        # Generate reference positions from the alignment range
        # reference_start and reference_end are from pysam and already account for soft-clipping
        ref_start = read.get("reference_start")
        ref_end = read.get("reference_end")

        if ref_start is not None and ref_end is not None:
            # Iterate over all reference positions covered by this read
            for pos in range(ref_start, ref_end):
                if pos not in position_coverage:
                    position_coverage[pos] = 0
                position_coverage[pos] += 1

    # Calculate statistics per position/mod_type
    # Only process positions that have at least one modification >= threshold
    # Calculate mean/median/std from FILTERED probabilities (>= threshold only)
    # This ensures frequency and mean probability use the same denominator
    mod_stats = {}
    all_positions = set()

    for pos, mod_dict_filtered in position_mods_filtered.items():
        all_positions.add(pos)
        total_coverage = position_coverage.get(pos, 0)

        for mod_code, probabilities_filtered in mod_dict_filtered.items():
            if mod_code not in mod_stats:
                mod_stats[mod_code] = {}

            # Calculate statistics from filtered probabilities (>= threshold only)
            # This makes mean/median/std consistent with frequency
            # Interpretation: "Among high-confidence modified reads, what's the average probability?"
            probs_array_filtered = np.array(probabilities_filtered)
            mod_count = len(probabilities_filtered)

            # Calculate modification frequency (fraction of reads with high-confidence mod)
            frequency = mod_count / total_coverage if total_coverage > 0 else 0.0

            # Apply frequency and count filters
            if frequency < min_frequency or mod_count < min_modified_reads:
                continue  # Skip this position - doesn't meet filtering criteria

            mod_stats[mod_code][pos] = {
                "probabilities": probabilities_filtered,  # Store filtered for reference
                "mean": float(np.mean(probs_array_filtered)),  # Mean of filtered only
                "median": float(np.median(probs_array_filtered)),  # Median of filtered
                "std": float(np.std(probs_array_filtered)),  # Std of filtered
                "count": mod_count,  # Count of filtered (>= threshold)
                "total_coverage": total_coverage,
                "frequency": float(frequency),  # Frequency based on filtered count
            }

    result = {
        "mod_stats": mod_stats,
        "positions": sorted(all_positions),
    }

    return result


def calculate_dwell_time_statistics(reads_data):
    """Calculate aggregate dwell time statistics across multiple reads

    Dwell time is calculated from move tables as the number of signal samples
    per base divided by the sample rate, giving time in milliseconds.

    Args:
        reads_data: List of read dicts from extract_reads_for_reference()

    Returns:
        Dict with keys:
            - positions: Array of reference positions
            - mean_dwell: Mean dwell time (ms) at each position
            - std_dwell: Standard deviation (ms) at each position
            - median_dwell: Median dwell time (ms) at each position
            - coverage: Number of reads covering each position
    """
    # Map reference position -> list of dwell times (in milliseconds)
    position_dwells = {}

    for read in reads_data:
        stride = read["stride"]
        moves = read["move_table"]
        ref_start = read["reference_start"]
        sample_rate = read.get("sample_rate", 4000)  # Default 4kHz

        # Get aligned move indices (skips soft-clipped bases)
        aligned_indices, aligned_set = get_aligned_move_indices_from_read(read)

        if len(aligned_indices) == 0:
            # No aligned bases in this read (all soft-clipped)
            continue

        # Calculate dwell time for each aligned base
        signal_pos = 0
        base_idx = 0  # Index for aligned bases only

        for move_idx, move in enumerate(moves):
            if move == 1:
                # Only process if this base is aligned (not soft-clipped)
                if move_idx in aligned_set:
                    # This is an aligned base boundary
                    # Find end of this base's signal
                    signal_end = signal_pos + stride  # Default

                    for j in range(move_idx + 1, len(moves)):
                        if moves[j] == 1:
                            signal_end = signal_pos + ((j - move_idx) * stride)
                            break
                    else:
                        signal_end = signal_pos + ((len(moves) - move_idx) * stride)

                    # Calculate dwell time in milliseconds
                    num_samples = signal_end - signal_pos
                    dwell_ms = (num_samples / sample_rate) * 1000.0

                    # Map to reference position (only for aligned bases)
                    ref_pos = ref_start + base_idx

                    if ref_pos not in position_dwells:
                        position_dwells[ref_pos] = []
                    position_dwells[ref_pos].append(dwell_ms)

                    base_idx += 1  # Only increment for aligned bases

            signal_pos += stride

    # Calculate statistics
    positions = sorted(position_dwells.keys())
    mean_dwell = []
    std_dwell = []
    median_dwell = []
    coverage = []

    for pos in positions:
        dwells = np.array(position_dwells[pos])
        mean_dwell.append(np.mean(dwells))
        std_dwell.append(np.std(dwells))
        median_dwell.append(np.median(dwells))
        coverage.append(len(dwells))

    return {
        "positions": np.array(positions),
        "mean_dwell": np.array(mean_dwell),
        "std_dwell": np.array(std_dwell),
        "median_dwell": np.array(median_dwell),
        "coverage": np.array(coverage),
    }


def calculate_aggregate_signal(reads_data, normalization_method):
    """Calculate aggregate signal statistics aligned to reference positions

    Args:
        reads_data: List of read dicts from extract_reads_for_reference()
        normalization_method: Normalization method to apply to signals

    Returns:
        Dict with keys:
            - positions: Array of reference positions
            - mean_signal: Mean signal at each position
            - std_signal: Standard deviation at each position
            - median_signal: Median signal at each position
            - coverage: Number of unique reads covering each position
    """
    from ..normalization import normalize_signal

    # Build a dict mapping reference positions to signal values and read IDs
    position_signals = {}
    position_reads = {}  # Track which reads cover each position

    for read in reads_data:
        read_id = read.get(
            "read_id", str(id(read))
        )  # Use read_id if available, else use object id
        # Normalize the signal
        signal = normalize_signal(read["signal"], normalization_method)
        stride = read["stride"]
        moves = np.array(read["move_table"], dtype=np.uint8)
        ref_start = read["reference_start"]

        # Soft-clipped bases: the move table includes moves for ALL signal samples
        # We need to skip signal samples corresponding to soft-clipped query bases
        query_start_offset = read.get("query_start_offset", 0)
        query_end_offset = read.get("query_end_offset", 0)

        # Find indices in move table where move=1 (these correspond to query bases)
        query_base_move_indices = np.where(moves == 1)[0]

        if len(query_base_move_indices) == 0:
            # No aligned bases, skip this read
            continue

        # The first query_start_offset indices are soft-clipped at the start
        # The last query_end_offset indices are soft-clipped at the end
        aligned_query_base_indices = query_base_move_indices[
            query_start_offset : len(query_base_move_indices) - query_end_offset
        ]
        aligned_set = set(aligned_query_base_indices)  # For O(1) lookup

        # Map signal to reference positions using move table
        # Only process aligned (non-soft-clipped) query bases
        ref_pos = ref_start

        for move_idx in range(len(moves)):
            move = moves[move_idx]
            sig_idx = move_idx * stride

            # Only process if this move index corresponds to an aligned query base
            if move_idx in aligned_set and sig_idx < len(signal):
                # Add signal value at this reference position
                if ref_pos not in position_signals:
                    position_signals[ref_pos] = []
                    position_reads[ref_pos] = set()
                position_signals[ref_pos].append(signal[sig_idx])
                position_reads[ref_pos].add(read_id)

            # Advance reference position only on move=1
            if move == 1:
                ref_pos += 1

    # Calculate statistics for each position
    positions = sorted(position_signals.keys())
    mean_signals = []
    std_signals = []
    median_signals = []
    coverages = []

    for pos in positions:
        values = np.array(position_signals[pos])
        mean_signals.append(np.mean(values))
        std_signals.append(np.std(values))
        median_signals.append(np.median(values))
        coverages.append(
            len(position_reads[pos])
        )  # Count unique reads, not signal samples

    return {
        "positions": np.array(positions),
        "mean_signal": np.array(mean_signals),
        "std_signal": np.array(std_signals),
        "median_signal": np.array(median_signals),
        "coverage": np.array(coverages),
    }


def calculate_base_pileup(
    reads_data, bam_file=None, reference_name=None, fasta_file=None
):
    """Calculate IGV-style base pileup at each reference position

    Args:
        reads_data: List of read dicts from extract_reads_for_reference()
        bam_file: Optional path to BAM file (for extracting reference sequence from MD tag)
        reference_name: Optional reference name (for extracting reference sequence)
        fasta_file: Optional path to FASTA file (preferred source for reference sequence)

    Returns:
        Dict with keys:
            - positions: Array of reference positions
            - counts: Dict mapping each position to dict of base counts
                     e.g., {pos: {'A': 10, 'C': 2, 'G': 5, 'T': 8}}
            - reference_bases: Dict mapping position to reference base (from FASTA or BAM)
    """
    import pysam

    from .bam import get_reference_sequence_for_read

    position_bases = {}

    for read in reads_data:
        sequence = read["sequence"]

        # Iterate over aligned bases only (skips soft-clipped bases)
        for _move_idx, _base_idx, seq_idx, ref_pos in iter_aligned_bases(read):
            if seq_idx < len(sequence):
                base = sequence[seq_idx].upper()

                if ref_pos not in position_bases:
                    position_bases[ref_pos] = {}
                if base not in position_bases[ref_pos]:
                    position_bases[ref_pos][base] = 0
                position_bases[ref_pos][base] += 1

    # Only include positions that have actual coverage
    # The x-axis will automatically span the correct range, and bars will be
    # at their correct reference positions
    positions = sorted(position_bases.keys())

    result = {
        "positions": np.array(positions),
        "counts": {pos: position_bases[pos] for pos in positions},
    }

    # Extract reference bases from FASTA (preferred) or BAM
    if reference_name and reads_data and positions:
        reference_bases = {}

        # Try FASTA first (most accurate)
        if fasta_file:
            try:
                fasta = pysam.FastaFile(str(fasta_file))
                # Get the range of positions we need
                min_pos = int(min(positions))
                max_pos = int(max(positions))

                # Fetch reference sequence for the region
                ref_seq = fasta.fetch(reference_name, min_pos, max_pos + 1)

                # Map each position to its reference base
                for pos in positions:
                    idx = int(pos) - min_pos
                    if 0 <= idx < len(ref_seq):
                        reference_bases[pos] = ref_seq[idx].upper()

                fasta.close()

                if reference_bases:
                    result["reference_bases"] = reference_bases
                    return result
            except Exception:
                # FASTA failed, fall back to BAM
                pass

        # Fall back to BAM file (uses MD tag or query sequence)
        # Iterate through all reads to build complete reference coverage
        if bam_file and not reference_bases:
            try:
                for read in reads_data:
                    ref_seq, ref_start, aligned_read = get_reference_sequence_for_read(
                        bam_file, read["read_id"]
                    )

                    if ref_seq and ref_start is not None:
                        # Add reference bases from this read's alignment
                        for pos in positions:
                            if pos in reference_bases:
                                continue  # Already have this position
                            # Calculate index in reference sequence
                            idx = pos - ref_start
                            if 0 <= idx < len(ref_seq):
                                reference_bases[pos] = ref_seq[idx].upper()

                    # Stop early if we've covered all positions
                    if len(reference_bases) >= len(positions):
                        break

                if reference_bases:
                    result["reference_bases"] = reference_bases
            except Exception:
                # If we can't get reference sequence, just skip it
                pass

    return result


def calculate_quality_by_position(reads_data):
    """Calculate average quality scores at each reference position

    Args:
        reads_data: List of read dicts from extract_reads_for_reference()

    Returns:
        Dict with keys:
            - positions: Array of reference positions
            - mean_quality: Mean quality score at each position
            - std_quality: Standard deviation of quality at each position
    """
    position_qualities = {}

    for read in reads_data:
        if read["quality_scores"] is None:
            continue

        quality_scores = read["quality_scores"]

        # Iterate over aligned bases only (skips soft-clipped bases)
        for _move_idx, _base_idx, seq_idx, ref_pos in iter_aligned_bases(read):
            if seq_idx < len(quality_scores):
                qual = quality_scores[seq_idx]

                if ref_pos not in position_qualities:
                    position_qualities[ref_pos] = []
                position_qualities[ref_pos].append(qual)

    positions = sorted(position_qualities.keys())
    mean_qualities = []
    std_qualities = []

    for pos in positions:
        values = np.array(position_qualities[pos])
        mean_qualities.append(np.mean(values))
        std_qualities.append(np.std(values))

    return {
        "positions": np.array(positions),
        "mean_quality": np.array(mean_qualities),
        "std_quality": np.array(std_qualities),
    }


# =============================================================================
# Pileup-only functions (work without move tables)
# =============================================================================


def iter_aligned_positions(read: dict):
    """
    Iterate over aligned positions using aligned_pairs mapping (no move table required)

    This generator yields information about each aligned base using the aligned_pairs
    dict from extract_alignments_for_reference(). Unlike iter_aligned_bases(), this
    does NOT require move tables.

    Args:
        read: Read dict from extract_alignments_for_reference() containing:
            - aligned_pairs: Dict mapping query_pos -> ref_pos

    Yields:
        Tuples of (seq_idx, ref_pos) for each aligned base:
        - seq_idx: Index in query sequence
        - ref_pos: Reference genome position
    """
    aligned_pairs = read.get("aligned_pairs", {})

    # Iterate over aligned pairs in order
    for seq_idx in sorted(aligned_pairs.keys()):
        ref_pos = aligned_pairs[seq_idx]
        if ref_pos is not None:
            yield seq_idx, ref_pos


def calculate_base_pileup_from_alignments(
    reads_data, bam_file=None, reference_name=None, fasta_file=None
):
    """Calculate IGV-style base pileup using alignment data (no move table required)

    This function works with data from extract_alignments_for_reference() which
    doesn't require move tables.

    Args:
        reads_data: List of read dicts from extract_alignments_for_reference()
        bam_file: Optional path to BAM file (for extracting reference sequence from MD tag)
        reference_name: Optional reference name (for extracting reference sequence)
        fasta_file: Optional path to FASTA file (preferred source for reference sequence)

    Returns:
        Dict with keys:
            - positions: Array of reference positions
            - counts: Dict mapping each position to dict of base counts
                     e.g., {pos: {'A': 10, 'C': 2, 'G': 5, 'T': 8}}
            - reference_bases: Dict mapping position to reference base (from FASTA or BAM)
    """
    import pysam

    from .bam import get_reference_sequence_for_read

    position_bases = {}

    for read in reads_data:
        sequence = read.get("sequence", "")
        if not sequence:
            continue

        # Use aligned_pairs mapping (works without move tables)
        for seq_idx, ref_pos in iter_aligned_positions(read):
            if seq_idx < len(sequence):
                base = sequence[seq_idx].upper()

                if ref_pos not in position_bases:
                    position_bases[ref_pos] = {}
                if base not in position_bases[ref_pos]:
                    position_bases[ref_pos][base] = 0
                position_bases[ref_pos][base] += 1

    # Only include positions that have actual coverage
    positions = sorted(position_bases.keys())

    result = {
        "positions": np.array(positions),
        "counts": {pos: position_bases[pos] for pos in positions},
    }

    # Extract reference bases from FASTA (preferred) or BAM
    if reference_name and reads_data and positions:
        reference_bases = {}

        # Try FASTA first (most accurate)
        if fasta_file:
            try:
                fasta = pysam.FastaFile(str(fasta_file))
                # Get the range of positions we need
                min_pos = int(min(positions))
                max_pos = int(max(positions))

                # Fetch reference sequence for the region
                ref_seq = fasta.fetch(reference_name, min_pos, max_pos + 1)

                # Map each position to its reference base
                for pos in positions:
                    idx = int(pos) - min_pos
                    if 0 <= idx < len(ref_seq):
                        reference_bases[pos] = ref_seq[idx].upper()

                fasta.close()

                if reference_bases:
                    result["reference_bases"] = reference_bases
                    return result
            except Exception:
                # FASTA failed, fall back to BAM
                pass

        # Fall back to BAM file (uses MD tag or query sequence)
        # Iterate through all reads to build complete reference coverage
        if bam_file and not reference_bases:
            try:
                for read in reads_data:
                    ref_seq, ref_start, aligned_read = get_reference_sequence_for_read(
                        bam_file, read["read_id"]
                    )

                    if ref_seq and ref_start is not None:
                        # Add reference bases from this read's alignment
                        for pos in positions:
                            if pos in reference_bases:
                                continue  # Already have this position
                            # Calculate index in reference sequence
                            idx = pos - ref_start
                            if 0 <= idx < len(ref_seq):
                                reference_bases[pos] = ref_seq[idx].upper()

                    # Stop early if we've covered all positions
                    if len(reference_bases) >= len(positions):
                        break

                if reference_bases:
                    result["reference_bases"] = reference_bases
            except Exception:
                # If we can't get reference sequence, just skip it
                pass

    return result


def calculate_quality_by_position_from_alignments(reads_data):
    """Calculate average quality scores using alignment data (no move table required)

    This function works with data from extract_alignments_for_reference() which
    doesn't require move tables.

    Args:
        reads_data: List of read dicts from extract_alignments_for_reference()

    Returns:
        Dict with keys:
            - positions: Array of reference positions
            - mean_quality: Mean quality score at each position
            - std_quality: Standard deviation of quality at each position
    """
    position_qualities = {}

    for read in reads_data:
        quality_scores = read.get("quality_scores")
        if quality_scores is None:
            continue

        # Use aligned_pairs mapping (works without move tables)
        for seq_idx, ref_pos in iter_aligned_positions(read):
            if seq_idx < len(quality_scores):
                qual = quality_scores[seq_idx]

                if ref_pos not in position_qualities:
                    position_qualities[ref_pos] = []
                position_qualities[ref_pos].append(qual)

    positions = sorted(position_qualities.keys())
    mean_qualities = []
    std_qualities = []

    for pos in positions:
        values = np.array(position_qualities[pos])
        mean_qualities.append(np.mean(values))
        std_qualities.append(np.std(values))

    return {
        "positions": np.array(positions),
        "mean_quality": np.array(mean_qualities),
        "std_quality": np.array(std_qualities),
    }


def calculate_modification_statistics_from_alignments(
    reads_data, mod_filter=None, min_frequency=0.0, min_modified_reads=1
):
    """Calculate modification statistics using alignment data (no move table required)

    This function works with data from extract_alignments_for_reference() which
    doesn't require move tables. The modification data comes from MM/ML tags in BAM.

    Args:
        reads_data: List of read dicts from extract_alignments_for_reference()
        mod_filter: Optional dict mapping mod_code -> minimum probability threshold
        min_frequency: Minimum fraction of reads that must be modified (0.0-1.0)
        min_modified_reads: Minimum number of reads that must have the modification

    Returns:
        Dict with keys (same format as calculate_modification_statistics):
            - mod_stats: Dict mapping mod_code -> position -> statistics
            - positions: Sorted list of all positions with modifications >= threshold
    """
    # Reuse the existing modification statistics logic
    # It doesn't actually depend on move tables - only on the modifications list
    # which is extracted from MM/ML tags
    return calculate_modification_statistics(
        reads_data,
        mod_filter=mod_filter,
        min_frequency=min_frequency,
        min_modified_reads=min_modified_reads,
    )


def calculate_coverage_from_alignments(reads_data):
    """Calculate coverage depth at each reference position (no move table required)

    This function calculates how many reads cover each reference position,
    using alignment data from extract_alignments_for_reference().

    Args:
        reads_data: List of read dicts from extract_alignments_for_reference()

    Returns:
        Dict with keys:
            - positions: Array of reference positions
            - coverage: Number of reads covering each position
    """
    position_coverage = {}

    for read in reads_data:
        ref_start = read.get("reference_start")
        ref_end = read.get("reference_end")

        if ref_start is not None and ref_end is not None:
            for pos in range(ref_start, ref_end):
                if pos not in position_coverage:
                    position_coverage[pos] = 0
                position_coverage[pos] += 1

    positions = sorted(position_coverage.keys())
    coverage = [position_coverage[pos] for pos in positions]

    return {
        "positions": np.array(positions),
        "coverage": np.array(coverage),
    }
