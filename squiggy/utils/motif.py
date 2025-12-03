"""Motif extraction utilities for Squiggy"""

import numpy as np
import pod5
import pysam


def extract_reads_for_motif(
    pod5_file,
    bam_file,
    fasta_file,
    motif,
    match_index=0,
    window=None,
    upstream=None,
    downstream=None,
    max_reads=100,
):
    """Extract signal and alignment data for reads overlapping a motif match

    This function finds reads that overlap a specific motif match position,
    similar to extract_reads_for_reference() but centered on a motif position.

    Args:
        pod5_file: Path to POD5 file
        bam_file: Path to BAM file with alignments and move tables
        fasta_file: Path to indexed FASTA file
        motif: IUPAC motif pattern (e.g., "DRACH")
        match_index: Which motif match to use (0-based index)
        window: Number of bases around motif center to include (±window, symmetric).
                Deprecated: use upstream/downstream for asymmetric windows.
        upstream: Number of bases upstream (5') of motif center
        downstream: Number of bases downstream (3') of motif center
        max_reads: Maximum number of reads to return

    Returns:
        Tuple of (reads_data, motif_match) where:
            - reads_data: List of dicts with adjusted reference coordinates
              (same structure as extract_reads_for_reference())
            - motif_match: MotifMatch object for the selected match

    Raises:
        ValueError: If no motif matches found or match_index out of range
    """
    import random

    from ..motif import search_motif

    # Handle window parameter for backward compatibility
    if window is not None and upstream is None and downstream is None:
        upstream = window
        downstream = window
    elif upstream is None or downstream is None:
        raise ValueError(
            "Must provide either 'window' or both 'upstream' and 'downstream'"
        )

    # Search for motif matches
    matches = list(search_motif(fasta_file, motif))

    if not matches:
        raise ValueError(f"No matches found for motif '{motif}' in FASTA file")

    if match_index >= len(matches):
        raise ValueError(
            f"Match index {match_index} out of range (found {len(matches)} matches)"
        )

    # Get the selected match
    motif_match = matches[match_index]

    # Define asymmetric window around motif center
    motif_center = motif_match.position + (motif_match.length // 2)
    region_start = max(0, motif_center - upstream)
    region_end = motif_center + downstream

    # Extract reads overlapping this region, clipping to window using BAM alignment
    reads_info = []

    try:
        with pysam.AlignmentFile(str(bam_file), "rb", check_sq=False) as bam:
            # BAM fetch requires non-negative coordinates - clamp region_start to 0
            fetch_start = max(0, region_start)
            for read in bam.fetch(motif_match.chrom, fetch_start, region_end):
                if read.is_unmapped:
                    continue

                # Extract move table
                if not read.has_tag("mv"):
                    continue

                move_table = np.array(read.get_tag("mv"), dtype=np.uint8)
                stride = int(move_table[0])
                moves = move_table[1:]

                # Get BAM alignment: reference_position → query_position mapping
                # aligned_pairs = [(query_pos, ref_pos), ...]
                # query_pos = None for deletions, ref_pos = None for insertions
                aligned_pairs = read.get_aligned_pairs()

                # Find read positions that map to our window [region_start, region_end]
                # We need to clip the read to only include bases mapping to this window
                read_positions_in_window = []
                ref_positions_in_window = []

                for query_pos, ref_pos in aligned_pairs:
                    if ref_pos is not None and region_start <= ref_pos < region_end:
                        if query_pos is not None:  # Skip deletions
                            read_positions_in_window.append(query_pos)
                            ref_positions_in_window.append(ref_pos)

                # Skip reads with no alignment to our window
                if not read_positions_in_window:
                    continue

                # Extract only the portion of sequence/quality that maps to window
                min_read_pos = min(read_positions_in_window)
                max_read_pos = max(read_positions_in_window)

                # Clip sequence and quality scores to window
                clipped_sequence = read.query_sequence[min_read_pos : max_read_pos + 1]
                quality_scores = None
                if read.query_qualities:
                    quality_scores = np.array(
                        read.query_qualities[min_read_pos : max_read_pos + 1]
                    )

                # Now clip signal using move table
                # Walk move table to find signal indices corresponding to read positions
                signal_start_idx = None
                signal_end_idx = None
                sig_idx = 0
                for query_pos_walk in range(len(read.query_sequence)):
                    if query_pos_walk == min_read_pos:
                        signal_start_idx = sig_idx
                    if query_pos_walk == max_read_pos:
                        signal_end_idx = (
                            sig_idx + stride
                        )  # Include signal for last base
                        break
                    # Count signal samples for this base (number of 0s until next 1)
                    if query_pos_walk < len(moves):
                        sig_idx += stride

                # Store read info with motif-relative coordinates
                # Reference coordinates are now in [region_start, region_end] range
                reads_info.append(
                    {
                        "read_id": read.query_name,
                        "reference_start": min(ref_positions_in_window)
                        - motif_center,  # Motif-relative
                        "reference_end": max(ref_positions_in_window)
                        + 1
                        - motif_center,  # Motif-relative
                        "chrom": motif_match.chrom,
                        "sequence": clipped_sequence,
                        "move_table": moves[
                            min_read_pos : max_read_pos + 1
                        ],  # Clipped to window
                        "stride": stride,
                        "quality_scores": quality_scores,
                        "signal_start_idx": signal_start_idx,  # For POD5 signal extraction
                        "signal_end_idx": signal_end_idx,
                    }
                )

        # Subsample if needed
        if len(reads_info) > max_reads:
            reads_info = random.sample(reads_info, max_reads)

        # Extract signal data from POD5, clipping to the window
        read_id_set = {r["read_id"] for r in reads_info}
        signal_data = {}

        with pod5.Reader(pod5_file) as reader:
            for pod5_read in reader.reads():
                read_id_str = str(pod5_read.read_id)
                if read_id_str in read_id_set:
                    # Find the corresponding read_info to get signal indices
                    read_info = next(
                        r for r in reads_info if r["read_id"] == read_id_str
                    )
                    start_idx = read_info.get("signal_start_idx", 0)
                    end_idx = read_info.get("signal_end_idx", len(pod5_read.signal))

                    # Extract only the signal for the window
                    signal_data[read_id_str] = {
                        "signal": pod5_read.signal[start_idx:end_idx]
                        if start_idx is not None and end_idx is not None
                        else pod5_read.signal,
                        "sample_rate": pod5_read.run_info.sample_rate,
                    }
                    if len(signal_data) == len(read_id_set):
                        break

        # Combine BAM and POD5 data
        result = []
        for read_info in reads_info:
            read_id = read_info["read_id"]
            if read_id in signal_data:
                # Remove temporary signal index fields
                clean_read_info = {
                    k: v
                    for k, v in read_info.items()
                    if k not in ["signal_start_idx", "signal_end_idx"]
                }
                result.append(
                    {
                        **clean_read_info,
                        **signal_data[read_id],
                    }
                )

        return result, motif_match

    except Exception as e:
        raise ValueError(f"Error extracting reads for motif '{motif}': {str(e)}") from e


def align_reads_to_motif_center(reads_data, motif_center):
    """Adjust read coordinates to be relative to motif center position

    Converts absolute reference coordinates to motif-centered coordinates
    where the motif center is position 0.

    Args:
        reads_data: List of read dicts from extract_reads_for_motif()
        motif_center: Genomic position of motif center

    Returns:
        List of read dicts with adjusted reference_start relative to motif_center

    Examples:
        If motif_center=1000 and read starts at 950:
        Original: reference_start=950
        Adjusted: reference_start=-50 (50bp before motif center)
    """
    adjusted_reads = []

    for read in reads_data:
        adjusted_read = read.copy()
        # Adjust reference coordinates to be motif-centered
        adjusted_read["reference_start"] = read["reference_start"] - motif_center
        adjusted_read["reference_end"] = read["reference_end"] - motif_center
        adjusted_reads.append(adjusted_read)

    return adjusted_reads


def clip_reads_to_window(reads_data, window_start, window_end):
    """Clip reads and their signals to a specific coordinate window

    Trims the move table, sequence, and quality scores so that only signal
    within [window_start, window_end] is retained. This is essential for
    motif-centered plots to prevent reads extending beyond the ROI.

    Args:
        reads_data: List of read dicts with reference_start/end in motif-relative coords
        window_start: Start of window (e.g., -10 for 10bp upstream)
        window_end: End of window (e.g., +10 for 10bp downstream)

    Returns:
        List of clipped read dicts with trimmed move tables and sequences
    """
    clipped_reads = []

    for read in reads_data:
        ref_start = read["reference_start"]
        ref_end = read["reference_end"]
        moves = read["move_table"]
        sequence = read["sequence"]
        signal = read["signal"]
        stride = read["stride"]
        quality_scores = read.get("quality_scores")

        # Skip reads entirely outside the window
        if ref_end <= window_start or ref_start >= window_end:
            continue

        # Walk through move table to find signal/sequence indices
        ref_pos = ref_start
        seq_idx = 0
        sig_idx = 0
        start_seq_idx = None
        start_sig_idx = None
        start_move_idx = None
        end_seq_idx = len(sequence)
        end_sig_idx = len(signal)
        end_move_idx = len(moves)
        found_start = False

        for i, move in enumerate(moves):
            # Mark start position when we first reach the window
            if not found_start and ref_pos >= window_start:
                start_seq_idx = seq_idx
                start_sig_idx = sig_idx
                start_move_idx = i
                found_start = True

            # Mark end position when we exit the window
            if ref_pos >= window_end:
                end_seq_idx = seq_idx
                end_sig_idx = sig_idx
                end_move_idx = i
                break

            if move == 1:
                seq_idx += 1
                ref_pos += 1

            sig_idx += stride

        # If we never entered the window, use defaults
        if start_seq_idx is None:
            start_seq_idx = 0
            start_sig_idx = 0
            start_move_idx = 0

        # Clip the data - use the exact same indices we found while walking the move table
        clipped_read = read.copy()
        clipped_read["reference_start"] = max(ref_start, window_start)
        clipped_read["reference_end"] = min(ref_end, window_end)
        clipped_read["sequence"] = sequence[start_seq_idx:end_seq_idx]
        clipped_read["signal"] = signal[start_sig_idx:end_sig_idx]

        # Clip move table using the same indices
        clipped_read["move_table"] = moves[start_move_idx:end_move_idx]

        if quality_scores is not None:
            clipped_read["quality_scores"] = quality_scores[start_seq_idx:end_seq_idx]

        clipped_reads.append(clipped_read)

    return clipped_reads
