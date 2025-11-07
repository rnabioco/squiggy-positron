"""Base modification (modBAM) parsing for Squiggy Positron extension

This module handles parsing of base modifications from modBAM files (MM/ML tags)
for visualization in single-read plots. Supports both single-letter modification
codes (e.g., 'm', 'a') and ChEBI numeric codes (e.g., 17596, 17802).

Key components:
- ModificationAnnotation: Single modification with probability and signal alignment
- extract_modifications_from_alignment(): Parse MM/ML tags from BAM alignments
- detect_modification_provenance(): Extract basecaller info from BAM header

References:
- modBAM spec: https://github.com/samtools/hts-specs
- modkit: https://github.com/nanoporetech/modkit
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pysam


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


def extract_modifications_from_alignment(
    alignment: pysam.AlignedSegment, bases: list
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

    Examples:
        >>> from squiggy.alignment import extract_alignment_from_bam
        >>> aligned_read = extract_alignment_from_bam(bam_path, read_id)
        >>> mods = extract_modifications_from_alignment(alignment, aligned_read.bases)
        >>> for mod in mods:
        ...     print(f"{mod.mod_code} at pos {mod.position}: p={mod.probability}")
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
    for (canonical_base, _strand, mod_code), mod_list in modified_bases_dict.items():
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


def detect_modification_provenance(bam_file: Path) -> dict[str, Any]:
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

    Examples:
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

    except Exception:
        pass

    return result
