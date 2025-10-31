"""Base modification (modBAM) analysis for Squiggy

This module handles parsing, aggregation, and analysis of base modifications
from modBAM files (MM/ML tags). Supports both single-letter modification codes
(e.g., 'm', 'a') and ChEBI numeric codes (e.g., 17596 for inosine, 17802 for
pseudouridine).

Key components:
- ModificationAnnotation: Single modification with probability and signal alignment
- ModPositionStats: Per-position modification statistics across multiple reads
- extract_modifications_from_alignment(): Parse MM/ML tags from BAM alignments
- calculate_modification_pileup(): Aggregate modifications across reads
- detect_modification_provenance(): Extract basecaller info from BAM header

References:
- modBAM spec: https://github.com/samtools/hts-specs
- modkit: https://github.com/nanoporetech/modkit
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pysam
from rich.console import Console

# Create Rich console for styled output
console = Console()


# ==============================================================================
# Data Structures
# ==============================================================================


@dataclass
class ModificationAnnotation:
    """Single base modification annotation with probability and signal alignment"""

    position: int  # Position in read sequence (0-indexed)
    genomic_pos: int | None  # Genomic position (if aligned)
    mod_code: str | int  # Modification code: str (e.g., 'm') or int (ChEBI code)
    canonical_base: str  # The unmodified base (C, A, T, G, U)
    probability: float  # Modification probability from ML tag (0-1)
    signal_start: int  # Signal sample start index
    signal_end: int  # Signal sample end index


@dataclass
class ModPositionStats:
    """Statistics for a modification at a specific genomic position

    Stores both continuous probabilities (always available) and optional
    threshold-based classification statistics (when tau is provided).
    """

    ref_pos: int  # Reference position
    mod_type: str | int  # Modification code: str or ChEBI int
    canonical_base: str  # The unmodified base (C, A, T, G, U)
    coverage: int  # Total reads overlapping this position
    probs: list[float]  # All probability values from reads
    mean_prob: float  # Mean probability across all reads
    # Threshold-based stats (only populated if tau is provided)
    n_mod_tau: int | None = None  # Number of reads >= threshold
    n_unmod_tau: int | None = None  # Number of reads < threshold
    frequency: float | None = None  # n_mod_tau / coverage
    mean_conf_modified: float | None = None  # Mean prob of modified reads only
    read_ids_modified: set[str] | None = None  # Read IDs classified as modified


# ==============================================================================
# Parsing Functions
# ==============================================================================


def extract_modifications_from_alignment(
    alignment, bases: list
) -> list[ModificationAnnotation]:
    """Extract modification annotations from a BAM alignment

    Parses MM/ML tags via pysam's modified_bases property and maps modifications
    to signal positions using base annotations.

    Args:
        alignment: pysam AlignmentSegment object
        bases: List of BaseAnnotation objects with signal mappings

    Returns:
        List of ModificationAnnotation objects (empty if no modifications)

    Note:
        The modified_bases property returns a dict with format:
        {(canonical_base, strand, mod_code): [(position, quality), ...]}
        where:
        - canonical_base: str (e.g., 'C', 'A')
        - strand: int (0=forward, 1=reverse)
        - mod_code: str or int (e.g., 'm' for 5mC, 17596 for inosine)
        - position: int (base position in read, 0-indexed)
        - quality: int (encoded as 256*probability, or -1 if unknown)

    Example:
        >>> from squiggy.alignment import extract_alignment_from_bam
        >>> aligned_read = extract_alignment_from_bam(bam_path, read_id)
        >>> if aligned_read.modifications:
        ...     for mod in aligned_read.modifications:
        ...         print(f"{mod.mod_code} at pos {mod.position}: p={mod.probability}")
    """
    modifications = []

    # Check if modifications are present
    if not hasattr(alignment, "modified_bases") or not alignment.modified_bases:
        return modifications

    # Get modified bases dict from pysam
    # Format: {(canonical, strand, mod_type): [(pos, qual), ...]}
    modified_bases_dict = alignment.modified_bases

    # Create a quick lookup for bases by position
    base_lookup = {base.position: base for base in bases}

    # Iterate over all modification types in the alignment
    for (canonical_base, strand, mod_code), mod_list in modified_bases_dict.items():
        # Process each modified position
        for position, quality in mod_list:
            # Convert quality to probability
            # Quality is encoded as (256 * probability), or -1 if unknown
            if quality < 0:
                # Unknown probability, skip
                continue

            probability = quality / 256.0

            # Cap probability at 1.0 (allow small floating point overflow)
            probability = min(probability, 1.0)

            # Get signal positions from the base annotation
            if position not in base_lookup:
                # Position not in base annotations (shouldn't happen, but be defensive)
                console.print(
                    f"[yellow]Warning:[/yellow] Modification at position {position} "
                    f"not found in base annotations"
                )
                continue

            base = base_lookup[position]

            # Create modification annotation
            mod_annotation = ModificationAnnotation(
                position=position,
                genomic_pos=base.genomic_pos,
                mod_code=mod_code,
                canonical_base=canonical_base,
                probability=probability,
                signal_start=base.signal_start,
                signal_end=base.signal_end,
            )
            modifications.append(mod_annotation)

    return modifications


# ==============================================================================
# Analysis Functions
# ==============================================================================


def calculate_modification_pileup(
    reads: list, tau: float | None = None, scope: str = "position"
) -> dict[tuple[int, str | int], ModPositionStats]:
    """Calculate per-position modification statistics from a list of AlignedRead objects

    Aggregates modifications across multiple reads to produce per-position statistics.
    Supports both continuous probability mode (default) and exploratory thresholding.

    Args:
        reads: List of AlignedRead objects with modifications
        tau: Optional probability threshold for classification (0-1)
            If provided, computes n_mod_tau, n_unmod_tau, frequency
            If None (default), only continuous probabilities are computed
        scope: Classification scope when tau is provided
            - "position": per-position classification (each position independent)
            - "any": read-level classification (if ANY mod >= tau, read is modified)

    Returns:
        Dict mapping (ref_pos, mod_type) -> ModPositionStats

    Example:
        >>> # Continuous mode (no threshold)
        >>> pileup = calculate_modification_pileup(reads)
        >>> for (pos, mod_type), stats in pileup.items():
        ...     print(f"{mod_type} at {pos}: mean_prob={stats.mean_prob:.2f}")
        >>>
        >>> # Threshold mode
        >>> pileup_thresh = calculate_modification_pileup(reads, tau=0.5)
        >>> for (pos, mod_type), stats in pileup_thresh.items():
        ...     print(f"{mod_type} at {pos}: {stats.n_mod_tau}/{stats.coverage} modified")

    Note:
        Requires AlignedRead objects with:
        - modifications: list[ModificationAnnotation]
        - genomic_pos in each ModificationAnnotation (unmapped mods are skipped)
    """
    # Import here to avoid circular dependency
    from squiggy.alignment import AlignedRead

    # Group modifications by (genomic_pos, mod_type)
    pileup = defaultdict(lambda: {"probs": [], "canonical": None, "read_ids": set()})

    # If scope is "any", track which reads have ANY modification >= tau
    reads_modified_any = set()
    if tau is not None and scope == "any":
        for read in reads:
            if not isinstance(read, AlignedRead) or not read.modifications:
                continue
            for mod in read.modifications:
                if mod.genomic_pos is not None and mod.probability >= tau:
                    reads_modified_any.add(read.read_id)
                    break  # One is enough

    # Build pileup
    for read in reads:
        if not isinstance(read, AlignedRead) or not read.modifications:
            continue

        for mod in read.modifications:
            if mod.genomic_pos is None:
                continue  # Skip unmapped modifications

            key = (mod.genomic_pos, mod.mod_code)
            pileup[key]["probs"].append(mod.probability)
            pileup[key]["canonical"] = mod.canonical_base
            pileup[key]["read_ids"].add(read.read_id)

    # Convert to ModPositionStats
    result = {}
    for (ref_pos, mod_type), data in pileup.items():
        probs = data["probs"]
        coverage = len(probs)
        mean_prob = float(np.mean(probs))
        canonical = data["canonical"]

        # Initialize threshold-based stats
        n_mod_tau = None
        n_unmod_tau = None
        frequency = None
        mean_conf_modified = None
        read_ids_modified = None

        # Compute threshold-based stats if tau provided
        if tau is not None:
            if scope == "position":
                # Per-position classification
                probs_above_tau = [p for p in probs if p >= tau]
                n_mod_tau = len(probs_above_tau)
                n_unmod_tau = coverage - n_mod_tau
                frequency = n_mod_tau / coverage if coverage > 0 else 0.0
                mean_conf_modified = (
                    float(np.mean(probs_above_tau)) if probs_above_tau else 0.0
                )
            elif scope == "any":
                # Read-level classification: use reads_modified_any set
                n_mod_tau = len(data["read_ids"] & reads_modified_any)
                n_unmod_tau = coverage - n_mod_tau
                frequency = n_mod_tau / coverage if coverage > 0 else 0.0
                # Mean conf of modified reads at this position
                probs_modified_reads = [
                    probs[i]
                    for i, read_id in enumerate(data["read_ids"])
                    if read_id in reads_modified_any
                ]
                mean_conf_modified = (
                    float(np.mean(probs_modified_reads))
                    if probs_modified_reads
                    else 0.0
                )
                read_ids_modified = data["read_ids"] & reads_modified_any

        stats = ModPositionStats(
            ref_pos=ref_pos,
            mod_type=mod_type,
            canonical_base=canonical,
            coverage=coverage,
            probs=probs,
            mean_prob=mean_prob,
            n_mod_tau=n_mod_tau,
            n_unmod_tau=n_unmod_tau,
            frequency=frequency,
            mean_conf_modified=mean_conf_modified,
            read_ids_modified=read_ids_modified,
        )
        result[(ref_pos, mod_type)] = stats

    return result


def detect_modification_provenance(bam_file: Path) -> dict:
    """Detect modification calling provenance from BAM header

    Parses @PG (program) header lines to extract basecaller information, version,
    and model details. Useful for displaying metadata and understanding modification
    calling parameters.

    Args:
        bam_file: Path to BAM file

    Returns:
        Dict with keys:
            - basecaller: str (e.g., "dorado", "remora", "guppy", or "Unknown")
            - version: str (e.g., "0.8.0" or "Unknown")
            - model: str (model name or "Unknown")
            - full_info: str (complete @PG command line for reference)
            - unknown: bool (True if provenance could not be determined)

    Example:
        >>> provenance = detect_modification_provenance(bam_path)
        >>> if not provenance["unknown"]:
        ...     print(f"Basecaller: {provenance['basecaller']} v{provenance['version']}")
        ...     print(f"Model: {provenance['model']}")
    """
    result = {
        "basecaller": "Unknown",
        "version": "Unknown",
        "model": "Unknown",
        "full_info": "",
        "unknown": True,
    }

    try:
        with pysam.AlignmentFile(str(bam_file), "rb", check_sq=False) as bam:
            header = bam.header.to_dict()

            # Look for @PG lines
            if "PG" not in header:
                return result

            for pg_entry in header["PG"]:
                # Check for dorado
                if "PN" in pg_entry and "dorado" in pg_entry["PN"].lower():
                    result["basecaller"] = "dorado"
                    result["version"] = pg_entry.get("VN", "Unknown")

                    # Try to extract model from command line (CL field)
                    if "CL" in pg_entry:
                        cl = pg_entry["CL"]
                        result["full_info"] = cl

                        # Look for model argument
                        if "--model" in cl or "-m" in cl:
                            parts = cl.split()
                            for i, part in enumerate(parts):
                                if part in ["--model", "-m"] and i + 1 < len(parts):
                                    result["model"] = parts[i + 1]
                                    break

                    result["unknown"] = False
                    break

                # Check for remora
                elif "PN" in pg_entry and "remora" in pg_entry["PN"].lower():
                    result["basecaller"] = "remora"
                    result["version"] = pg_entry.get("VN", "Unknown")

                    if "CL" in pg_entry:
                        result["full_info"] = pg_entry["CL"]

                    result["unknown"] = False
                    break

                # Check for guppy
                elif "PN" in pg_entry and "guppy" in pg_entry["PN"].lower():
                    result["basecaller"] = "guppy"
                    result["version"] = pg_entry.get("VN", "Unknown")

                    if "CL" in pg_entry:
                        result["full_info"] = pg_entry["CL"]

                    result["unknown"] = False
                    break

    except Exception as e:
        console.print(f"[yellow]Warning:[/yellow] Error reading BAM header: {e}")

    return result
