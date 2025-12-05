"""BAM file utilities for Squiggy"""

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pod5
import pysam

from .paths import writable_working_directory


@dataclass
class ModelProvenance:
    """
    Metadata about the basecalling model used to generate a dataset

    Extracted from BAM file @PG headers, which contain information about
    the basecalling process and model version.

    Attributes:
        model_name: Name of the basecalling model (e.g., "guppy", "dorado")
        model_version: Version of the basecalling model
        flow_cell_kit: Sequencing kit used
        basecalling_model: Specific basecalling model identifier
        command_line: Full command line used for basecalling
    """

    model_name: str | None = None
    model_version: str | None = None
    flow_cell_kit: str | None = None
    basecalling_model: str | None = None
    command_line: str | None = None

    def __repr__(self) -> str:
        """Return informative summary of model provenance"""
        parts = []

        if self.model_name:
            parts.append(f"Model: {self.model_name}")

        if self.model_version:
            parts.append(f"v{self.model_version}")

        if self.basecalling_model:
            parts.append(f"({self.basecalling_model})")

        if not parts:
            return "<ModelProvenance: Unknown>"

        return f"<ModelProvenance: {' '.join(parts)}>"

    def matches(self, other: "ModelProvenance") -> bool:
        """Check if two ModelProvenance instances describe the same model"""
        if other is None:
            return False

        # Consider a match if both have same model_name and basecalling_model
        # (version differences might be acceptable)
        return (
            self.model_name == other.model_name
            and self.basecalling_model == other.basecalling_model
        )


def validate_bam_reads_in_pod5(bam_file, pod5_file):
    """Validate that all reads in BAM file exist in POD5 file

    This is a sanity check to ensure the BAM file corresponds to the POD5 file.
    If any BAM reads are missing from the POD5, something is horribly wrong.

    Args:
        bam_file: Path to BAM file
        pod5_file: Path to POD5 file

    Returns:
        dict: Validation results with keys:
            - is_valid (bool): True if all BAM reads are in POD5
            - bam_read_count (int): Number of reads in BAM file
            - pod5_read_count (int): Number of reads in POD5 file
            - missing_count (int): Number of BAM reads not in POD5
            - missing_reads (set): Set of read IDs in BAM but not in POD5

    Raises:
        Exception: If files cannot be opened or read
    """
    # Read all read IDs from POD5 file
    pod5_read_ids = set()
    with writable_working_directory():
        with pod5.Reader(pod5_file) as reader:
            for read in reader.reads():
                pod5_read_ids.add(str(read.read_id))

    # Read all read IDs from BAM file
    bam_read_ids = set()
    with pysam.AlignmentFile(str(bam_file), "rb", check_sq=False) as bam:
        for read in bam.fetch(until_eof=True):
            bam_read_ids.add(read.query_name)

    # Find reads in BAM but not in POD5
    missing_reads = bam_read_ids - pod5_read_ids

    return {
        "is_valid": len(missing_reads) == 0,
        "bam_read_count": len(bam_read_ids),
        "pod5_read_count": len(pod5_read_ids),
        "missing_count": len(missing_reads),
        "missing_reads": missing_reads,
    }


def get_basecall_data(
    bam_file: Path, read_id: str
) -> tuple[str | None, np.ndarray | None]:
    """Extract basecall sequence and signal mapping from BAM file

    Args:
        bam_file: Path to BAM file
        read_id: Read identifier to search for

    Returns:
        (sequence, seq_to_sig_map) or (None, None) if not available
    """
    if not bam_file:
        return None, None

    try:
        bam = pysam.AlignmentFile(str(bam_file), "rb", check_sq=False)

        # Find the read in BAM
        for read in bam.fetch(until_eof=True):
            if read.query_name == read_id:
                # Get sequence
                sequence = read.query_sequence

                # Get move table from BAM tags
                if read.has_tag("mv"):
                    move_table = np.array(read.get_tag("mv"), dtype=np.uint8)

                    # Extract stride (first element) and moves (remaining elements)
                    # Stride represents the neural network downsampling factor
                    # Typical values: 5 for DNA models, 10-12 for RNA models
                    stride = int(move_table[0])
                    moves = move_table[1:]

                    # Convert move table to signal-to-sequence mapping
                    seq_to_sig_map = []
                    sig_pos = 0
                    for move in moves:
                        if move == 1:
                            seq_to_sig_map.append(sig_pos)
                        sig_pos += stride

                    bam.close()
                    return sequence, np.array(seq_to_sig_map)

        bam.close()

    except Exception:
        pass

    return None, None


def parse_region(region_str: str) -> tuple[str | None, int | None, int | None]:
    """Parse a genomic region string into components

    Supports formats:
    - "chr1" (entire chromosome)
    - "chr1:1000" (single position)
    - "chr1:1000-2000" (range)
    - "chr1:1,000-2,000" (with commas)

    Args:
        region_str: Region string to parse

    Returns:
        (chromosome, start, end) where start/end are None if not specified.
        Returns (None, None, None) if parsing fails

    Examples:
        >>> parse_region("chr1")
        ("chr1", None, None)
        >>> parse_region("chr1:1000-2000")
        ("chr1", 1000, 2000)
    """
    if not region_str or not region_str.strip():
        return None, None, None

    region_str = region_str.strip().replace(",", "")  # Remove commas

    # Check for colon (indicates coordinates)
    if ":" in region_str:
        parts = region_str.split(":")
        if len(parts) != 2:
            return None, None, None

        chrom = parts[0].strip()
        coords = parts[1].strip()

        # Check for range (start-end)
        if "-" in coords:
            coord_parts = coords.split("-")
            if len(coord_parts) != 2:
                return None, None, None
            try:
                start = int(coord_parts[0].strip())
                end = int(coord_parts[1].strip())
                return chrom, start, end
            except ValueError:
                return None, None, None
        else:
            # Single position
            try:
                pos = int(coords)
                return chrom, pos, pos
            except ValueError:
                return None, None, None
    else:
        # Just chromosome name
        return region_str.strip(), None, None


def index_bam_file(bam_file):
    """Generate BAM index file (.bai)

    Creates a .bai index file for the given BAM file using pysam.

    Args:
        bam_file: Path to BAM file

    Raises:
        FileNotFoundError: If BAM file doesn't exist
        Exception: If indexing fails
    """
    if not bam_file or not Path(bam_file).exists():
        raise FileNotFoundError(f"BAM file not found: {bam_file}")

    try:
        pysam.index(str(bam_file))
    except Exception as e:
        raise Exception(f"Failed to index BAM file: {str(e)}") from e


def get_bam_references(bam_file: Path) -> list[dict]:
    """Get list of reference sequences from BAM file with read counts

    Args:
        bam_file: Path to BAM file

    Returns:
        List of dicts with keys:
            - name: Reference name
            - length: Reference sequence length
            - read_count: Number of aligned reads (requires index)

    Raises:
        FileNotFoundError: If BAM file doesn't exist
    """
    if not bam_file or not Path(bam_file).exists():
        raise FileNotFoundError(f"BAM file not found: {bam_file}")

    references = []

    try:
        with pysam.AlignmentFile(str(bam_file), "rb", check_sq=False) as bam:
            # Get reference names and lengths from header
            for ref_name, ref_length in zip(bam.references, bam.lengths, strict=False):
                ref_info = {
                    "name": ref_name,
                    "length": ref_length,
                    "read_count": None,
                }

                # Try to count reads if BAM is indexed
                bai_path = Path(str(bam_file) + ".bai")
                if bai_path.exists():
                    try:
                        # Count mapped reads for this reference
                        count = bam.count(ref_name)
                        ref_info["read_count"] = count
                    except Exception:
                        # If counting fails, leave as None
                        pass

                # Only include references that have reads
                # If read_count is None (no index), include it
                # If read_count is 0, skip it
                if ref_info["read_count"] is None or ref_info["read_count"] > 0:
                    references.append(ref_info)

    except Exception as e:
        raise ValueError(f"Error reading BAM file: {str(e)}") from e

    return references


def get_read_to_reference_mapping(bam_file, pod5_read_ids):
    """Get mapping of read IDs to their reference sequences from BAM file

    Args:
        bam_file: Path to BAM file
        pod5_read_ids: Set or list of read IDs from POD5 file (to filter)

    Returns:
        dict: Mapping of read_id -> reference_name for aligned reads
              Only includes reads that are in pod5_read_ids
              Example: {"read1": "chr1", "read2": "chr1", "read3": "chr2"}

    Note:
        Unmapped reads and reads not in pod5_read_ids are excluded
    """
    if not bam_file or not Path(bam_file).exists():
        return {}

    read_to_ref = {}
    pod5_read_set = set(pod5_read_ids)

    try:
        with pysam.AlignmentFile(str(bam_file), "rb", check_sq=False) as bam:
            for read in bam.fetch(until_eof=True):
                # Skip unmapped reads
                if read.is_unmapped:
                    continue

                read_id = read.query_name

                # Only include reads that are in our POD5 file
                if read_id not in pod5_read_set:
                    continue

                # Get reference name
                ref_name = bam.get_reference_name(read.reference_id)
                read_to_ref[read_id] = ref_name

    except Exception:
        # If there's an error reading BAM, return empty mapping
        return {}

    return read_to_ref


def get_reads_in_region(
    bam_file: Path, chromosome: str, start: int | None = None, end: int | None = None
) -> dict:
    """Query BAM file for reads aligning to a specific region

    Requires BAM file to be indexed (.bai file must exist).

    Args:
        bam_file: Path to BAM file
        chromosome: Chromosome/reference name
        start: Start position (0-based, inclusive) or None for entire chromosome
        end: End position (0-based, exclusive) or None for entire chromosome

    Returns:
        Dictionary mapping read_id -> alignment_info with keys:
            - read_id: Read identifier
            - chromosome: Reference name
            - start: Alignment start position
            - end: Alignment end position
            - strand: '+' or '-'
            - is_reverse: Boolean

    Raises:
        ValueError: If BAM file is not indexed or region is invalid
        FileNotFoundError: If BAM file doesn't exist
    """
    if not bam_file or not Path(bam_file).exists():
        raise FileNotFoundError(f"BAM file not found: {bam_file}")

    # Check for BAM index - will be created by caller if needed
    bai_path = Path(str(bam_file) + ".bai")
    if not bai_path.exists():
        raise ValueError(
            f"BAM index file not found: {bai_path}\n"
            "The BAM file must be indexed before querying regions."
        )

    reads_dict = {}

    try:
        with pysam.AlignmentFile(str(bam_file), "rb") as bam:
            # Verify chromosome exists in BAM header
            if chromosome not in bam.references:
                available = ", ".join(bam.references[:5])
                raise ValueError(
                    f"Chromosome '{chromosome}' not found in BAM file.\n"
                    f"Available references: {available}..."
                )

            # Query region
            if start is not None and end is not None:
                # Specific region
                alignments = bam.fetch(chromosome, start, end)
            else:
                # Entire chromosome
                alignments = bam.fetch(chromosome)

            # Collect alignment info
            for aln in alignments:
                if aln.is_unmapped:
                    continue

                reads_dict[aln.query_name] = {
                    "read_id": aln.query_name,
                    "chromosome": aln.reference_name,
                    "start": aln.reference_start,
                    "end": aln.reference_end,
                    "strand": "-" if aln.is_reverse else "+",
                    "is_reverse": aln.is_reverse,
                }

    except Exception as e:
        raise ValueError(f"Error querying BAM file: {str(e)}") from e

    return reads_dict


def reverse_complement(seq: str) -> str:
    """
    Return the reverse complement of a DNA sequence.

    Args:
        seq: DNA sequence string (A, C, G, T, N)

    Returns:
        Reverse complement sequence
    """
    complement = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    return "".join(complement.get(base, base) for base in reversed(seq))


def get_reference_sequence_for_read(bam_file, read_id):
    """
    Extract the reference sequence for a given aligned read.

    Args:
        bam_file: Path to BAM file
        read_id: Read identifier

    Returns:
        Tuple of (reference_sequence, reference_start, aligned_read)
        Returns (None, None, None) if read not found or not aligned
    """
    try:
        with pysam.AlignmentFile(str(bam_file), "rb", check_sq=False) as bam:
            # Find the alignment for this read
            aligned_read = None
            for aln in bam.fetch(until_eof=True):
                if aln.query_name == read_id:
                    aligned_read = aln
                    break

            if not aligned_read or aligned_read.is_unmapped:
                return None, None, None

            # Get reference sequence from the alignment
            ref_name = aligned_read.reference_name
            ref_start = aligned_read.reference_start

            # Check if reference sequence is available in BAM header
            if bam.header.get("SQ"):
                # Try to get reference sequence from header (if embedded)
                for sq in bam.header["SQ"]:
                    if sq["SN"] == ref_name:
                        # Some BAMs have embedded reference sequences
                        if "M5" in sq or "UR" in sq:
                            # Reference not embedded, need to reconstruct from alignment
                            break

            # Reconstruct reference sequence from aligned read
            # Use the aligned pairs to build the reference sequence
            ref_seq_list = []
            ref_positions = []

            for query_pos, ref_pos in aligned_read.get_aligned_pairs():
                if ref_pos is not None:  # Skip insertions in read
                    ref_positions.append(ref_pos)
                    if query_pos is not None:
                        # Match or mismatch
                        base = aligned_read.query_sequence[query_pos]
                        ref_seq_list.append(base)
                    else:
                        # Deletion in read (gap in query)
                        ref_seq_list.append("N")  # Use N for deletions

            ref_seq = "".join(ref_seq_list)

            return ref_seq, ref_start, aligned_read

    except Exception as e:
        raise ValueError(f"Error extracting reference sequence: {str(e)}") from e


def get_reference_sequence_from_fasta(
    fasta_file, reference_name, start, end, bam_file=None, read_id=None
):
    """
    Get reference sequence using FASTA-first pattern

    Tries to fetch reference from FASTA file first (most accurate), falls back
    to BAM reconstruction if FASTA unavailable or fetch fails.

    Args:
        fasta_file: Path to FASTA reference file (can be None)
        reference_name: Name of reference sequence (chromosome/contig)
        start: Start position (0-based, inclusive)
        end: End position (0-based, exclusive)
        bam_file: Optional BAM file path for fallback reconstruction
        read_id: Optional read ID for BAM fallback (if getting seq for single read)

    Returns:
        str: Reference sequence, or empty string if unavailable

    Examples:
        >>> # Fetch from FASTA
        >>> seq = get_reference_sequence_from_fasta(
        ...     fasta_file="ref.fa",
        ...     reference_name="chr1",
        ...     start=1000,
        ...     end=1100
        ... )
        >>>
        >>> # With BAM fallback
        >>> seq = get_reference_sequence_from_fasta(
        ...     fasta_file=None,  # No FASTA available
        ...     reference_name="chr1",
        ...     start=1000,
        ...     end=1100,
        ...     bam_file="alignments.bam",
        ...     read_id="read_001"
        ... )
    """
    reference_sequence = ""

    # Try FASTA first (most accurate and complete)
    if fasta_file:
        try:
            fasta = pysam.FastaFile(str(fasta_file))
            reference_sequence = fasta.fetch(reference_name, start, end)
            fasta.close()
            return reference_sequence
        except Exception:
            # FASTA fetch failed, will try BAM fallback
            pass

    # Fallback to BAM reconstruction if FASTA unavailable
    if not reference_sequence and bam_file and read_id:
        try:
            ref_seq, ref_start, aligned_read = get_reference_sequence_for_read(
                bam_file, read_id
            )
            if ref_seq and ref_start is not None:
                # Trim to requested region
                region_start = max(0, start - ref_start)
                region_end = min(len(ref_seq), end - ref_start)
                reference_sequence = ref_seq[region_start:region_end]
        except Exception:
            # BAM fallback failed, return empty string
            pass

    return reference_sequence


def get_available_reads_for_reference(bam_file, reference_name):
    """Count total reads available for a reference in a BAM file

    Args:
        bam_file: Path to BAM file with alignments
        reference_name: Name of reference sequence to count reads for

    Returns:
        Integer count of reads mapping to the reference

    Raises:
        ValueError: If BAM file cannot be read or reference not found
    """
    try:
        count = 0
        with pysam.AlignmentFile(str(bam_file), "rb", check_sq=False) as bam:
            for read in bam.fetch(until_eof=True):
                if read.is_unmapped:
                    continue

                # Check if read maps to the specified reference
                ref_name = bam.get_reference_name(read.reference_id)
                if ref_name == reference_name:
                    count += 1

        return count

    except Exception as e:
        raise ValueError(
            f"Error counting reads for reference '{reference_name}': {str(e)}"
        ) from e


def extract_reads_for_reference(
    pod5_file, bam_file, reference_name, max_reads=100, random_sample=True
):
    """Extract signal and alignment data for reads mapping to a reference

    Args:
        pod5_file: Path to POD5 file
        bam_file: Path to BAM file with alignments and move tables
        reference_name: Name of reference sequence to extract reads from
        max_reads: Maximum number of reads to return (will subsample if more available)
        random_sample: If True, randomly sample reads; if False, take first N reads

    Returns:
        List of dicts with keys:
            - read_id: Read identifier
            - signal: Raw signal array
            - sample_rate: Sampling rate
            - reference_start: Start position on reference
            - reference_end: End position on reference
            - sequence: Basecalled sequence
            - move_table: Move table array
            - stride: Stride value from move table
            - quality_scores: Per-base quality scores
            - modifications: List of ModificationAnnotation objects
    """
    import random

    # First, get all reads that map to this reference from BAM
    reads_info = []

    try:
        with pysam.AlignmentFile(str(bam_file), "rb", check_sq=False) as bam:
            for read in bam.fetch(until_eof=True):
                if read.is_unmapped:
                    continue

                # Check if read maps to the specified reference
                ref_name = bam.get_reference_name(read.reference_id)
                if ref_name != reference_name:
                    continue

                # Extract move table and quality scores
                if not read.has_tag("mv"):
                    continue

                move_table = np.array(read.get_tag("mv"), dtype=np.uint8)
                stride = int(move_table[0])
                moves = move_table[1:]

                # Get quality scores
                quality_scores = (
                    np.array(read.query_qualities) if read.query_qualities else None
                )

                # Extract modifications using _parse_alignment
                modifications = []
                try:
                    from ..alignment import _parse_alignment

                    aligned_read = _parse_alignment(read)
                    if aligned_read:
                        modifications = aligned_read.modifications
                except Exception:
                    # Modifications are optional, don't fail if extraction fails
                    pass

                # Calculate soft-clipped bases at the start and end of the read
                # This tells us how many bases in the raw signal to skip
                cigar = read.cigartuples
                query_start_offset = 0  # Bases to skip at start due to soft clipping
                query_end_offset = 0  # Bases to skip at end due to soft clipping

                if cigar:
                    # Check for soft clip at start (operation code 4)
                    if cigar[0][0] == 4:
                        query_start_offset = cigar[0][1]
                    # Check for soft clip at end (operation code 4)
                    if cigar[-1][0] == 4:
                        query_end_offset = cigar[-1][1]

                # Get aligned pairs for proper position mapping (handles indels)
                # Build query_pos -> ref_pos mapping
                query_to_ref = {}
                for query_pos, ref_pos in read.get_aligned_pairs():
                    if query_pos is not None and ref_pos is not None:
                        query_to_ref[query_pos] = ref_pos

                reads_info.append(
                    {
                        "read_id": read.query_name,
                        "reference_start": read.reference_start,
                        "reference_end": read.reference_end,
                        "sequence": read.query_sequence,
                        "move_table": moves,
                        "stride": stride,
                        "quality_scores": quality_scores,
                        "modifications": modifications,
                        "query_start_offset": query_start_offset,  # Soft-clipped bases at start
                        "query_end_offset": query_end_offset,  # Soft-clipped bases at end
                        "query_to_ref": query_to_ref,  # Query pos -> Ref pos mapping (handles indels)
                    }
                )

        # Subsample if we have too many reads
        if len(reads_info) > max_reads:
            if random_sample:
                reads_info = random.sample(reads_info, max_reads)
            else:
                reads_info = reads_info[:max_reads]

        # Now extract signal data from POD5 for these reads
        read_id_set = {r["read_id"] for r in reads_info}
        signal_data = {}

        with pod5.Reader(pod5_file) as reader:
            for pod5_read in reader.reads():
                read_id_str = str(pod5_read.read_id)
                if read_id_str in read_id_set:
                    signal_data[read_id_str] = {
                        "signal": pod5_read.signal,
                        "sample_rate": pod5_read.run_info.sample_rate,
                    }
                    if len(signal_data) == len(read_id_set):
                        break

        # Combine BAM and POD5 data
        result = []
        for read_info in reads_info:
            read_id = read_info["read_id"]
            if read_id in signal_data:
                result.append(
                    {
                        **read_info,
                        **signal_data[read_id],
                    }
                )

        return result

    except Exception as e:
        raise ValueError(
            f"Error extracting reads for reference {reference_name}: {str(e)}"
        ) from e


def extract_model_provenance(bam_file: str) -> ModelProvenance:
    """
    Extract basecalling model information from BAM file @PG headers

    The @PG header record contains information about the program used to
    generate the alignments. For ONT sequencing, this typically includes
    basecalling information from guppy or dorado.

    Args:
        bam_file: Path to BAM file

    Returns:
        ModelProvenance object with extracted metadata

    Examples:
        >>> from squiggy.utils import extract_model_provenance
        >>> provenance = extract_model_provenance('alignments.bam')
        >>> print(provenance.model_name)
        >>> print(provenance.basecalling_model)
    """
    if not os.path.exists(bam_file):
        raise FileNotFoundError(f"BAM file not found: {bam_file}")

    provenance = ModelProvenance()

    try:
        with pysam.AlignmentFile(str(bam_file), "rb", check_sq=False) as bam:
            # Get @PG header records
            if bam.header and "PG" in bam.header:
                pg_records = bam.header["PG"]

                # PG records are a list of dicts
                if not isinstance(pg_records, list):
                    pg_records = [pg_records]

                # Process each @PG record (usually the last one is the most relevant)
                for pg in pg_records:
                    # Extract program name
                    if "PN" in pg:
                        provenance.model_name = pg["PN"]

                    # Extract version
                    if "VN" in pg:
                        provenance.model_version = pg["VN"]

                    # Extract command line
                    if "CL" in pg:
                        provenance.command_line = pg["CL"]
                        # Try to extract model info from command line
                        # Common patterns: --model, -m, or model name in path
                        cl = pg["CL"]
                        if "--model" in cl or "-m " in cl:
                            # Parse model identifier from command line
                            parts = cl.split()
                            for i, part in enumerate(parts):
                                if part in ("--model", "-m") and i + 1 < len(parts):
                                    provenance.basecalling_model = parts[i + 1]
                                    break

                    # Try to extract kit info from description
                    if "DS" in pg:
                        provenance.flow_cell_kit = pg["DS"]

    except Exception:
        # Return partial provenance if there's an error
        pass

    return provenance


def validate_sq_headers(bam_file_a: str, bam_file_b: str) -> dict:
    """
    Validate that two BAM files have matching reference sequences

    Compares the SQ (sequence) headers from two BAM files to ensure they
    have the same references. This is important for comparison analysis
    to ensure reads can be meaningfully compared.

    Args:
        bam_file_a: Path to first BAM file
        bam_file_b: Path to second BAM file

    Returns:
        Dict with validation results:
            - is_valid (bool): True if references match
            - references_a (list): References in file A
            - references_b (list): References in file B
            - missing_in_b (list): References in A but not B
            - missing_in_a (list): References in B but not A
            - matching_count (int): Number of matching references

    Examples:
        >>> from squiggy.utils import validate_sq_headers
        >>> result = validate_sq_headers('align_a.bam', 'align_b.bam')
        >>> if result['is_valid']:
        ...     print("References match!")
        ... else:
        ...     print(f"Missing in B: {result['missing_in_b']}")
    """
    if not os.path.exists(bam_file_a):
        raise FileNotFoundError(f"BAM file A not found: {bam_file_a}")

    if not os.path.exists(bam_file_b):
        raise FileNotFoundError(f"BAM file B not found: {bam_file_b}")

    refs_a = []
    refs_b = []

    try:
        # Get references from file A
        with pysam.AlignmentFile(str(bam_file_a), "rb", check_sq=False) as bam:
            refs_a = list(bam.references)

        # Get references from file B
        with pysam.AlignmentFile(str(bam_file_b), "rb", check_sq=False) as bam:
            refs_b = list(bam.references)

    except Exception as e:
        raise ValueError(f"Error reading BAM files: {str(e)}") from e

    # Find differences
    refs_set_a = set(refs_a)
    refs_set_b = set(refs_b)

    missing_in_b = list(refs_set_a - refs_set_b)
    missing_in_a = list(refs_set_b - refs_set_a)
    matching = list(refs_set_a & refs_set_b)

    is_valid = len(missing_in_a) == 0 and len(missing_in_b) == 0

    return {
        "is_valid": is_valid,
        "references_a": refs_a,
        "references_b": refs_b,
        "missing_in_b": missing_in_b,
        "missing_in_a": missing_in_a,
        "matching_count": len(matching),
    }


def extract_alignments_for_reference(
    bam_file,
    reference_name,
    max_reads=100,
    random_sample=True,
):
    """Extract alignment data for reads mapping to a reference (no mv tag required)

    This function extracts BAM alignment data without requiring move tables (mv tag).
    It enables pileup-only visualizations from BAM files that lack signal-to-base mapping.

    Args:
        bam_file: Path to BAM file with alignments
        reference_name: Name of reference sequence to extract reads from
        max_reads: Maximum number of reads to return (will subsample if more available)
        random_sample: If True, randomly sample reads; if False, take first N reads

    Returns:
        List of dicts with keys:
            - read_id: Read identifier
            - reference_start: Start position on reference
            - reference_end: End position on reference
            - sequence: Basecalled sequence
            - quality_scores: Per-base quality scores
            - modifications: List of ModificationAnnotation objects
            - aligned_pairs: Dict mapping query_pos -> ref_pos (handles indels)

    Note:
        Unlike extract_reads_for_reference(), this function does NOT require POD5 files
        or move tables. It only extracts alignment information from BAM files.
    """
    import random

    reads_info = []

    try:
        with pysam.AlignmentFile(str(bam_file), "rb", check_sq=False) as bam:
            for read in bam.fetch(until_eof=True):
                if read.is_unmapped:
                    continue

                # Check if read maps to the specified reference
                ref_name = bam.get_reference_name(read.reference_id)
                if ref_name != reference_name:
                    continue

                # Get quality scores
                quality_scores = (
                    np.array(read.query_qualities) if read.query_qualities else None
                )

                # Extract modifications using _parse_alignment
                modifications = []
                try:
                    from ..alignment import _parse_alignment

                    aligned_read = _parse_alignment(read)
                    if aligned_read:
                        modifications = aligned_read.modifications
                except Exception:
                    # Modifications are optional, don't fail if extraction fails
                    pass

                # Build query_pos -> ref_pos mapping from aligned pairs
                aligned_pairs = {}
                for query_pos, ref_pos in read.get_aligned_pairs():
                    if query_pos is not None and ref_pos is not None:
                        aligned_pairs[query_pos] = ref_pos

                reads_info.append(
                    {
                        "read_id": read.query_name,
                        "reference_start": read.reference_start,
                        "reference_end": read.reference_end,
                        "sequence": read.query_sequence,
                        "quality_scores": quality_scores,
                        "modifications": modifications,
                        "aligned_pairs": aligned_pairs,
                    }
                )

        # Subsample if we have too many reads
        if len(reads_info) > max_reads:
            if random_sample:
                reads_info = random.sample(reads_info, max_reads)
            else:
                reads_info = reads_info[:max_reads]

        return reads_info

    except Exception as e:
        raise ValueError(
            f"Error extracting alignments for reference {reference_name}: {str(e)}"
        ) from e


@contextmanager
def open_bam_safe(bam_path: str | Path):
    """
    Context manager for safely opening and closing BAM files

    This utility eliminates duplicate BAM file handling code by providing
    a consistent pattern for opening BAM files with proper resource cleanup.

    Args:
        bam_path: Path to BAM file (string or Path object)

    Yields:
        pysam.AlignmentFile: Opened BAM file handle

    Raises:
        FileNotFoundError: If BAM file doesn't exist
        IOError: If BAM file cannot be opened

    Examples:
        >>> from squiggy.utils import open_bam_safe
        >>> with open_bam_safe("alignments.bam") as bam:
        ...     for read in bam:
        ...         print(read.query_name)

        >>> # Automatically closes even on error
        >>> with open_bam_safe("alignments.bam") as bam:
        ...     references = list(bam.references)
    """
    bam_path = Path(bam_path)

    if not bam_path.exists():
        raise FileNotFoundError(f"BAM file not found: {bam_path}")

    bam = None
    try:
        # Open with check_sq=False to avoid issues with missing SQ headers
        bam = pysam.AlignmentFile(str(bam_path), "rb", check_sq=False)
        yield bam
    finally:
        if bam is not None:
            bam.close()
