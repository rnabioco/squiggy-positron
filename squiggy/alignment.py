"""Alignment handling for event-aligned squiggle visualization"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pysam


@dataclass
class BaseAnnotation:
    """Single base annotation with signal alignment information"""

    base: str  # Base character (A, C, G, T)
    position: int  # Position in sequence (0-indexed)
    signal_start: int  # Signal sample start index
    signal_end: int  # Signal sample end index
    genomic_pos: int | None = None  # Genomic position (if aligned)
    quality: int | None = None  # Base quality score


@dataclass
class AlignedRead:
    """POD5 read with base call annotations"""

    read_id: str
    sequence: str
    bases: list[BaseAnnotation]
    chromosome: str | None = None
    genomic_start: int | None = None
    genomic_end: int | None = None
    strand: str | None = None  # '+' or '-'
    is_reverse: bool = False
    modifications: list = field(
        default_factory=list
    )  # List of ModificationAnnotation objects


def extract_alignment_from_bam(bam_path: Path, read_id: str) -> AlignedRead | None:
    """Extract alignment information for a read from BAM file

    Args:
        bam_path: Path to BAM file
        read_id: Read identifier to search for

    Returns:
        AlignedRead object or None if not found
    """
    try:
        with pysam.AlignmentFile(str(bam_path), "rb", check_sq=False) as bam:
            for alignment in bam.fetch(until_eof=True):
                if alignment.query_name == read_id:
                    return _parse_alignment(alignment)
    except Exception:
        # Error reading BAM file - return None
        pass

    return None


def _parse_alignment(alignment) -> AlignedRead | None:
    """Parse a pysam AlignmentSegment into AlignedRead

    Args:
        alignment: pysam AlignmentSegment object

    Returns:
        AlignedRead object or None if move table not available
    """
    # Get sequence
    sequence = alignment.query_sequence
    if not sequence:
        return None

    # Get move table from BAM tags (required for signal-to-base mapping)
    if not alignment.has_tag("mv"):
        return None

    move_table = np.array(alignment.get_tag("mv"), dtype=np.uint8)

    # Extract stride (first element) and moves (remaining elements)
    # Stride represents the neural network downsampling factor
    # Typical values: 5 for DNA models, 10-12 for RNA models
    stride = int(move_table[0])
    moves = move_table[1:]

    # Build query_pos -> ref_pos mapping using get_aligned_pairs()
    # This properly handles insertions, deletions, and matches
    query_to_ref = {}
    if not alignment.is_unmapped:
        for query_pos, ref_pos in alignment.get_aligned_pairs():
            if query_pos is not None and ref_pos is not None:
                query_to_ref[query_pos] = ref_pos

    # Convert move table to base annotations
    bases = []
    signal_pos = 0
    base_idx = 0

    for move_idx, move in enumerate(moves):
        if move == 1:  # New base starts here
            if base_idx < len(sequence):
                # Find end of this base's signal (next base or end of signal)
                # Look ahead to find where next base starts
                signal_end = signal_pos + stride  # Default to next position

                for j in range(move_idx + 1, len(moves)):
                    if moves[j] == 1:
                        # Next base found at position j
                        signal_end = signal_pos + ((j - move_idx) * stride)
                        break
                else:
                    # No next base found, extend to end of signal
                    signal_end = signal_pos + ((len(moves) - move_idx) * stride)

                # Get genomic position using aligned_pairs (handles indels correctly)
                genomic_pos = query_to_ref.get(base_idx)

                # Get quality score
                quality = None
                if alignment.query_qualities is not None:
                    quality = alignment.query_qualities[base_idx]

                base = BaseAnnotation(
                    base=sequence[base_idx],
                    position=base_idx,
                    signal_start=signal_pos,
                    signal_end=signal_end,
                    genomic_pos=genomic_pos,
                    quality=quality,
                )
                bases.append(base)
                base_idx += 1

        signal_pos += stride

    # Extract alignment info
    chromosome = alignment.reference_name if not alignment.is_unmapped else None
    genomic_start = alignment.reference_start if not alignment.is_unmapped else None
    genomic_end = alignment.reference_end if not alignment.is_unmapped else None
    strand = "-" if alignment.is_reverse else "+"

    # Extract base modifications if present
    modifications = []
    try:
        from .modifications import extract_modifications_from_alignment

        modifications = extract_modifications_from_alignment(alignment, bases)
    except Exception:
        # Modifications are optional, don't fail if extraction fails
        pass

    return AlignedRead(
        read_id=alignment.query_name,
        sequence=sequence,
        bases=bases,
        chromosome=chromosome,
        genomic_start=genomic_start,
        genomic_end=genomic_end,
        strand=strand,
        is_reverse=alignment.is_reverse,
        modifications=modifications,
    )


def get_base_to_signal_mapping(aligned_read: AlignedRead) -> tuple[str, np.ndarray]:
    """Extract sequence and signal mapping from AlignedRead

    Args:
        aligned_read: AlignedRead object with base annotations

    Returns:
        tuple: (sequence, seq_to_sig_map) compatible with existing plotter
    """
    sequence = aligned_read.sequence
    seq_to_sig_map = np.array([base.signal_start for base in aligned_read.bases])

    return sequence, seq_to_sig_map
