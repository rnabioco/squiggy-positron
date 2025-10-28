"""Utility functions for Squiggy application"""

import platform
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pod5
import pysam


def get_icon_path():
    """Get the path to the application icon file

    Returns the appropriate icon file for the current platform.
    Works both when running from source and when bundled with PyInstaller.

    Returns:
        Path or None: Path to icon file, or None if not found
    """
    # Determine icon file based on platform
    if platform.system() == "Windows":
        icon_name = "squiggy.ico"
    elif platform.system() == "Darwin":  # macOS
        icon_name = "squiggy.icns"
    else:  # Linux and others
        icon_name = "squiggy.png"

    # Try multiple locations
    # 1. PyInstaller bundle location (when bundled)
    if getattr(sys, "_MEIPASS", None):
        icon_path = Path(sys._MEIPASS) / icon_name
        if icon_path.exists():
            return icon_path

    # 2. Package data directory (when installed)
    try:
        if sys.version_info >= (3, 9):
            import importlib.resources as resources

            files = resources.files("squiggy")
            icon_path = files / "data" / icon_name
            if hasattr(icon_path, "as_posix"):
                path = Path(str(icon_path))
                if path.exists():
                    return path
        else:
            import pkg_resources

            icon_path = Path(
                pkg_resources.resource_filename("squiggy", f"data/{icon_name}")
            )
            if icon_path.exists():
                return icon_path
    except Exception:
        pass

    # 3. Development location (relative to package)
    package_dir = Path(__file__).parent
    icon_path = package_dir / "data" / icon_name
    if icon_path.exists():
        return icon_path

    # 4. Build directory (during development)
    build_dir = Path(__file__).parent.parent.parent / "build" / icon_name
    if build_dir.exists():
        return build_dir

    return None


def get_logo_path():
    """Get the path to the PNG logo for display in dialogs

    Returns:
        Path or None: Path to logo file, or None if not found
    """
    # Try multiple locations for the PNG logo
    # 1. PyInstaller bundle
    if getattr(sys, "_MEIPASS", None):
        logo_path = Path(sys._MEIPASS) / "squiggy.png"
        if logo_path.exists():
            return logo_path

    # 2. Package data directory
    try:
        if sys.version_info >= (3, 9):
            import importlib.resources as resources

            files = resources.files("squiggy")
            logo_path = files / "data" / "squiggy.png"
            if hasattr(logo_path, "as_posix"):
                path = Path(str(logo_path))
                if path.exists():
                    return path
        else:
            import pkg_resources

            logo_path = Path(
                pkg_resources.resource_filename("squiggy", "data/squiggy.png")
            )
            if logo_path.exists():
                return logo_path
    except Exception:
        pass

    # 3. Development location
    package_dir = Path(__file__).parent
    logo_path = package_dir / "data" / "squiggy.png"
    if logo_path.exists():
        return logo_path

    # 4. Build directory
    build_dir = Path(__file__).parent.parent.parent / "build" / "squiggy.png"
    if build_dir.exists():
        return build_dir

    return None


def get_sample_data_path():
    """Get the path to the bundled sample data file

    Returns:
        Path: Path to mod_reads.pod5 file

    Raises:
        FileNotFoundError: If sample data cannot be found
    """
    try:
        # For Python 3.9+
        if sys.version_info >= (3, 9):
            import importlib.resources as resources

            files = resources.files("squiggy")
            sample_path = files / "data" / "mod_reads.pod5"
            if hasattr(sample_path, "as_posix"):
                return Path(sample_path)
            # For traversable objects, we need to extract to temp
            temp_dir = Path(tempfile.gettempdir()) / "squiggy_data"
            temp_dir.mkdir(exist_ok=True)
            temp_file = temp_dir / "mod_reads.pod5"
            if not temp_file.exists():
                with resources.as_file(sample_path) as f:
                    shutil.copy(f, temp_file)
            return temp_file
        else:
            # Fallback for older Python
            import pkg_resources

            sample_path = pkg_resources.resource_filename("squiggy", "data/mod_reads.pod5")
            return Path(sample_path)
    except Exception as e:
        # Fallback: look in installed package directory
        package_dir = Path(__file__).parent
        sample_path = package_dir / "data" / "mod_reads.pod5"
        if sample_path.exists():
            return sample_path
        raise FileNotFoundError(f"Sample data not found. Error: {e}") from None


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


def downsample_signal(signal, downsample_factor=1):
    """Downsample signal array by taking every Nth point

    Reduces the number of data points for faster plotting while preserving
    the overall shape of the signal.

    Args:
        signal: Raw signal array (numpy array)
        downsample_factor: Factor by which to downsample (1 = no downsampling)

    Returns:
        numpy array: Downsampled signal array
    """
    if downsample_factor <= 1:
        return signal

    # Take every Nth point
    return signal[::downsample_factor]


def get_basecall_data(bam_file, read_id):
    """Extract basecall sequence and signal mapping from BAM file

    Args:
        bam_file: Path to BAM file
        read_id: Read identifier to search for

    Returns:
        tuple: (sequence, seq_to_sig_map) or (None, None) if not available
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

    except Exception as e:
        print(f"Error reading BAM file for {read_id}: {e}")

    return None, None


def parse_region(region_str):
    """Parse a genomic region string into components

    Supports formats:
    - "chr1" (entire chromosome)
    - "chr1:1000" (single position)
    - "chr1:1000-2000" (range)
    - "chr1:1,000-2,000" (with commas)

    Args:
        region_str: Region string to parse

    Returns:
        tuple: (chromosome, start, end) where start/end are None if not specified
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


def get_bam_references(bam_file):
    """Get list of reference sequences from BAM file with read counts

    Args:
        bam_file: Path to BAM file

    Returns:
        list: List of dicts with keys:
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
            for ref_name, ref_length in zip(bam.references, bam.lengths):
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

                references.append(ref_info)

    except Exception as e:
        raise ValueError(f"Error reading BAM file: {str(e)}") from e

    return references


def get_reads_in_region(bam_file, chromosome, start=None, end=None):
    """Query BAM file for reads aligning to a specific region

    Requires BAM file to be indexed (.bai file must exist).

    Args:
        bam_file: Path to BAM file
        chromosome: Chromosome/reference name
        start: Start position (0-based, inclusive) or None for entire chromosome
        end: End position (0-based, exclusive) or None for entire chromosome

    Returns:
        dict: Dictionary mapping read_id -> alignment_info with keys:
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


def reverse_complement(seq):
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
