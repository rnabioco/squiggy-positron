"""
Motif search functionality with IUPAC nucleotide code support

This module provides utilities for searching genomic sequences for motif patterns
using IUPAC nucleotide codes (e.g., DRACH for m6A motif).

Examples:
    >>> from squiggy.motif import iupac_to_regex, search_motif
    >>> pattern = iupac_to_regex("DRACH")
    >>> # pattern = "[AGT][AG]AC[ACT]"
    >>>
    >>> matches = list(search_motif(fasta_file, "DRACH", region="chrI:1000-2000"))
    >>> for match in matches:
    ...     print(f"{match.chrom}:{match.position} {match.sequence} ({match.strand})")
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pysam

# IUPAC nucleotide codes mapping
IUPAC_CODES = {
    "A": "A",
    "C": "C",
    "G": "G",
    "T": "T",
    "U": "T",  # Treat U as T
    "R": "[AG]",  # Purine
    "Y": "[CT]",  # Pyrimidine
    "S": "[GC]",  # Strong (3 H-bonds)
    "W": "[AT]",  # Weak (2 H-bonds)
    "K": "[GT]",  # Keto
    "M": "[AC]",  # Amino
    "B": "[CGT]",  # Not A
    "D": "[AGT]",  # Not C
    "H": "[ACT]",  # Not G
    "V": "[ACG]",  # Not T
    "N": "[ACGT]",  # Any base
}


@dataclass
class MotifMatch:
    """
    Represents a single motif match in a genomic sequence

    Attributes:
        chrom: Chromosome/reference name
        position: 0-based genomic position of match start
        sequence: Matched sequence
        strand: Strand ('+' or '-')
        length: Length of matched sequence
    """

    chrom: str
    position: int
    sequence: str
    strand: Literal["+", "-"]

    @property
    def length(self) -> int:
        """Length of matched sequence"""
        return len(self.sequence)

    @property
    def end(self) -> int:
        """End position (exclusive, 0-based)"""
        return self.position + self.length

    def __repr__(self) -> str:
        return (
            f"MotifMatch(chrom='{self.chrom}', position={self.position}, "
            f"sequence='{self.sequence}', strand='{self.strand}')"
        )


def iupac_to_regex(pattern: str) -> str:
    """
    Convert IUPAC nucleotide pattern to regular expression

    Args:
        pattern: IUPAC nucleotide pattern (e.g., "DRACH", "YGCY")

    Returns:
        Regular expression pattern string

    Examples:
        >>> iupac_to_regex("DRACH")
        '[AGT][AG]AC[ACT]'
        >>> iupac_to_regex("YGCY")
        '[CT]GC[CT]'

    Raises:
        ValueError: If pattern contains invalid IUPAC codes
    """
    regex_parts = []

    for char in pattern.upper():
        if char not in IUPAC_CODES:
            valid_codes = ", ".join(sorted(IUPAC_CODES.keys()))
            raise ValueError(
                f"Invalid IUPAC code '{char}' in pattern '{pattern}'. "
                f"Valid codes: {valid_codes}"
            )
        regex_parts.append(IUPAC_CODES[char])

    return "".join(regex_parts)


def reverse_complement(seq: str) -> str:
    """
    Get reverse complement of DNA sequence

    Args:
        seq: DNA sequence

    Returns:
        Reverse complement sequence

    Examples:
        >>> reverse_complement("ATCG")
        'CGAT'
        >>> reverse_complement("DRACH")
        'DYTCH'
    """
    # Base complements
    complement_map = {
        "A": "T",
        "T": "A",
        "C": "G",
        "G": "C",
        "N": "N",
        # IUPAC codes - complement of ambiguity codes
        "R": "Y",  # [AG] -> [CT]
        "Y": "R",  # [CT] -> [AG]
        "S": "S",  # [GC] -> [GC]
        "W": "W",  # [AT] -> [AT]
        "K": "M",  # [GT] -> [AC]
        "M": "K",  # [AC] -> [GT]
        "B": "V",  # [CGT] -> [GCA]
        "D": "H",  # [AGT] -> [ACT]
        "H": "D",  # [ACT] -> [AGT]
        "V": "B",  # [ACG] -> [TGC]
    }

    result = []
    for base in reversed(seq.upper()):
        result.append(complement_map.get(base, base))

    return "".join(result)


def parse_region(region: str) -> tuple[str, int | None, int | None]:
    """
    Parse genomic region string

    Args:
        region: Region string in format "chrom", "chrom:start", or "chrom:start-end"
                Positions are 1-based (converted to 0-based internally)

    Returns:
        Tuple of (chrom, start, end) where start/end are 0-based
        start and end can be None if not specified

    Examples:
        >>> parse_region("chrI")
        ('chrI', None, None)
        >>> parse_region("chrI:1000-2000")
        ('chrI', 999, 2000)
        >>> parse_region("chrI:1000")
        ('chrI', 999, None)

    Raises:
        ValueError: If region format is invalid
    """
    if ":" not in region:
        # Just chromosome
        return region, None, None

    parts = region.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid region format: {region}")

    chrom = parts[0]
    coord_str = parts[1]

    if "-" in coord_str:
        # chrom:start-end
        start_str, end_str = coord_str.split("-")
        try:
            start = int(start_str) - 1  # Convert to 0-based
            end = int(end_str)  # End is exclusive
        except ValueError as e:
            raise ValueError(f"Invalid coordinates in region: {region}") from e

        if start < 0:
            raise ValueError(f"Start position must be >= 1: {region}")
        if end <= start:
            raise ValueError(f"End must be greater than start: {region}")

        return chrom, start, end
    else:
        # chrom:start
        try:
            start = int(coord_str) - 1  # Convert to 0-based
        except ValueError as e:
            raise ValueError(f"Invalid start position in region: {region}") from e

        if start < 0:
            raise ValueError(f"Start position must be >= 1: {region}")

        return chrom, start, None


def search_motif(
    fasta_file: str | Path,
    motif: str,
    region: str | None = None,
    strand: Literal["+", "-", "both"] = "both",
) -> Iterator[MotifMatch]:
    """
    Search for motif matches in FASTA file

    Lazy iteration over matches for memory efficiency.

    Args:
        fasta_file: Path to indexed FASTA file (.fai required)
        motif: IUPAC nucleotide pattern (e.g., "DRACH", "YGCY")
        region: Optional region filter ("chrom", "chrom:start", "chrom:start-end")
                Positions are 1-based in input, converted to 0-based internally
        strand: Search strand ('+', '-', or 'both')

    Yields:
        MotifMatch objects for each match found

    Examples:
        >>> matches = list(search_motif("genome.fa", "DRACH", region="chr1:1000-2000"))
        >>> for match in matches:
        ...     print(f"{match.chrom}:{match.position+1} {match.sequence} ({match.strand})")

    Raises:
        FileNotFoundError: If FASTA file or index not found
        ValueError: If motif contains invalid IUPAC codes or region format is invalid
    """
    fasta_path = Path(fasta_file).resolve()

    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    # Check for index
    fai_path = Path(str(fasta_path) + ".fai")
    if not fai_path.exists():
        raise FileNotFoundError(
            f"FASTA index not found: {fai_path}. "
            f"Create index with: samtools faidx {fasta_path}"
        )

    # Convert motif to regex
    regex_pattern = iupac_to_regex(motif)
    regex = re.compile(regex_pattern, re.IGNORECASE)

    # Parse region if provided
    target_chrom = None
    target_start = None
    target_end = None

    if region:
        target_chrom, target_start, target_end = parse_region(region)

    # Open FASTA file
    with pysam.FastaFile(str(fasta_path)) as fasta:
        # Determine which chromosomes to search
        if target_chrom:
            if target_chrom not in fasta.references:
                raise ValueError(
                    f"Chromosome '{target_chrom}' not found in FASTA file. "
                    f"Available: {', '.join(fasta.references)}"
                )
            chroms = [target_chrom]
        else:
            chroms = list(fasta.references)

        # Search each chromosome
        for chrom in chroms:
            # Get sequence for region
            if target_chrom == chrom:
                seq = fasta.fetch(chrom, target_start, target_end)
                offset = target_start if target_start is not None else 0
            else:
                seq = fasta.fetch(chrom)
                offset = 0

            # Search forward strand
            if strand in ("+", "both"):
                for match in regex.finditer(seq):
                    yield MotifMatch(
                        chrom=chrom,
                        position=match.start() + offset,
                        sequence=match.group().upper(),
                        strand="+",
                    )

            # Search reverse strand
            if strand in ("-", "both"):
                rc_seq = reverse_complement(seq)
                rc_motif = reverse_complement(motif)
                rc_regex_pattern = iupac_to_regex(rc_motif)
                rc_regex = re.compile(rc_regex_pattern, re.IGNORECASE)

                for match in rc_regex.finditer(rc_seq):
                    # Convert position back to forward strand coordinates
                    rc_position = match.start()
                    match_len = len(match.group())
                    # Position on forward strand
                    fw_position = len(seq) - rc_position - match_len

                    # Get the actual sequence from the forward strand
                    fw_sequence = seq[fw_position : fw_position + match_len].upper()

                    yield MotifMatch(
                        chrom=chrom,
                        position=fw_position + offset,
                        sequence=fw_sequence,
                        strand="-",
                    )


def count_motifs(
    fasta_file: str | Path,
    motif: str,
    region: str | None = None,
    strand: Literal["+", "-", "both"] = "both",
) -> int:
    """
    Count total motif matches in FASTA file

    Args:
        fasta_file: Path to indexed FASTA file (.fai required)
        motif: IUPAC nucleotide pattern (e.g., "DRACH", "YGCY")
        region: Optional region filter ("chrom:start-end")
        strand: Search strand ('+', '-', or 'both')

    Returns:
        Total number of matches

    Examples:
        >>> count = count_motifs("genome.fa", "DRACH", region="chr1")
        >>> print(f"Found {count} DRACH motifs on chr1")
    """
    return sum(1 for _ in search_motif(fasta_file, motif, region, strand))


__all__ = [
    "MotifMatch",
    "iupac_to_regex",
    "search_motif",
    "count_motifs",
    "IUPAC_CODES",
]
