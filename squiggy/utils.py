"""Utility functions for Squiggy application"""

import os
import platform
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pod5
import pysam


@contextmanager
def writable_working_directory():
    """Context manager to temporarily change to a writable working directory

    The pod5 library creates temporary directories in the current working directory
    during format migration. When running from a PyInstaller bundle or other read-only
    location, the CWD may not be writable. This context manager temporarily changes
    to a writable temp directory, then restores the original CWD.

    Usage:
        with writable_working_directory():
            with pod5.Reader(pod5_file) as reader:
                # Process reads...
    """
    original_cwd = os.getcwd()
    temp_dir = Path(tempfile.gettempdir()) / "squiggy_workdir"
    temp_dir.mkdir(exist_ok=True)

    try:
        os.chdir(temp_dir)
        yield temp_dir
    finally:
        os.chdir(original_cwd)


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
        import importlib.resources as resources

        files = resources.files("squiggy")
        icon_path = files / "data" / icon_name
        if hasattr(icon_path, "as_posix"):
            path = Path(str(icon_path))
            if path.exists():
                return path
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
        import importlib.resources as resources

        files = resources.files("squiggy")
        logo_path = files / "data" / "squiggy.png"
        if hasattr(logo_path, "as_posix"):
            path = Path(str(logo_path))
            if path.exists():
                return path
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

    When running from a PyInstaller bundle, the sample data is in a read-only
    location. This function copies the sample POD5 file to a writable temporary
    directory to allow the pod5 library to perform format migration if needed.

    Returns:
        Path: Path to yeast_trna_reads.pod5 file (may be in temp directory)

    Raises:
        FileNotFoundError: If sample data cannot be found
    """
    # Create temp directory for sample data
    temp_dir = Path(tempfile.gettempdir()) / "squiggy_data"
    temp_dir.mkdir(exist_ok=True)
    temp_file = temp_dir / "yeast_trna_reads.pod5"

    try:
        # Find the bundled sample data
        source_path = None

        import importlib.resources as resources

        files = resources.files("squiggy")
        sample_path = files / "data" / "yeast_trna_reads.pod5"

        # If it's a regular file path, use it directly as source
        if hasattr(sample_path, "as_posix"):
            source_path = Path(str(sample_path))
        else:
            # For traversable objects, extract to temp
            if not temp_file.exists():
                with resources.as_file(sample_path) as f:
                    shutil.copy(f, temp_file)
            return temp_file

        # If we got a source path, check if it's in a read-only location
        # (PyInstaller bundle) and copy to temp
        if source_path and source_path.exists():
            # Always copy when running from PyInstaller bundle
            # Check sys._MEIPASS (PyInstaller) or if parent dir is not writable
            is_pyinstaller = getattr(sys, "_MEIPASS", None) is not None
            is_readonly = not _is_writable_dir(source_path.parent)

            if is_pyinstaller or is_readonly:
                # Copy to temp directory to ensure pod5 library can create temp files
                # Always copy if source is newer or temp doesn't exist
                should_copy = (
                    not temp_file.exists()
                    or temp_file.stat().st_size == 0
                    or temp_file.stat().st_size != source_path.stat().st_size
                )
                if should_copy:
                    shutil.copy(source_path, temp_file)
                return temp_file
            else:
                # Source is in writable location, return as-is
                return source_path

        # Last resort: look in package directory or development location
        package_dir = Path(__file__).parent
        sample_path = package_dir / "data" / "yeast_trna_reads.pod5"

        # Try development location (tests/data relative to project root)
        if not sample_path.exists():
            project_root = package_dir.parent.parent
            sample_path = project_root / "tests" / "data" / "yeast_trna_reads.pod5"

        if sample_path.exists():
            # Always copy when running from PyInstaller bundle
            is_pyinstaller = getattr(sys, "_MEIPASS", None) is not None
            is_readonly = not _is_writable_dir(sample_path.parent)

            if is_pyinstaller or is_readonly:
                should_copy = (
                    not temp_file.exists()
                    or temp_file.stat().st_size == 0
                    or temp_file.stat().st_size != sample_path.stat().st_size
                )
                if should_copy:
                    shutil.copy(sample_path, temp_file)
                return temp_file
            return sample_path

        raise FileNotFoundError("Sample data file not found in any location")

    except Exception as e:
        raise FileNotFoundError(f"Sample data not found. Error: {e}") from None


def _is_writable_dir(dir_path):
    """Check if a directory is writable

    Args:
        dir_path: Path to directory

    Returns:
        bool: True if directory is writable, False otherwise
    """
    try:
        test_file = dir_path / ".write_test"
        test_file.touch()
        test_file.unlink()
        return True
    except (OSError, PermissionError):
        return False


def get_sample_bam_path():
    """Get the path to the bundled sample BAM file

    When running from a PyInstaller bundle, the sample data is in a read-only
    location. This function copies the sample BAM file (and its index) to a
    writable temporary directory.

    Returns:
        Path: Path to yeast_trna_mappings.bam file (may be in temp directory)
        Returns None if BAM file not found

    """
    # Create temp directory for sample data
    temp_dir = Path(tempfile.gettempdir()) / "squiggy_data"
    temp_dir.mkdir(exist_ok=True)
    temp_bam = temp_dir / "yeast_trna_mappings.bam"
    temp_bai = temp_dir / "yeast_trna_mappings.bam.bai"

    try:
        # Find the bundled sample BAM file
        source_bam = None
        source_bai = None

        import importlib.resources as resources

        files = resources.files("squiggy")
        sample_bam_path = files / "data" / "yeast_trna_mappings.bam"
        sample_bai_path = files / "data" / "yeast_trna_mappings.bam.bai"

        # If it's a regular file path, use it directly as source
        if hasattr(sample_bam_path, "as_posix"):
            source_bam = Path(str(sample_bam_path))
            source_bai = Path(str(sample_bai_path))
        else:
            # For traversable objects, extract to temp
            if not temp_bam.exists():
                with resources.as_file(sample_bam_path) as f:
                    shutil.copy(f, temp_bam)
            if not temp_bai.exists() and sample_bai_path:
                try:
                    with resources.as_file(sample_bai_path) as f:
                        shutil.copy(f, temp_bai)
                except Exception:
                    pass  # BAI file optional
            return temp_bam

        # If we got a source path, check if it's in a read-only location
        if source_bam and source_bam.exists():
            # Always copy when running from PyInstaller bundle
            is_pyinstaller = getattr(sys, "_MEIPASS", None) is not None
            is_readonly = not _is_writable_dir(source_bam.parent)

            if is_pyinstaller or is_readonly:
                # Copy to temp directory
                should_copy_bam = (
                    not temp_bam.exists()
                    or temp_bam.stat().st_size == 0
                    or temp_bam.stat().st_size != source_bam.stat().st_size
                )
                if should_copy_bam:
                    shutil.copy(source_bam, temp_bam)

                # Copy index file if it exists
                if source_bai and source_bai.exists():
                    should_copy_bai = (
                        not temp_bai.exists()
                        or temp_bai.stat().st_size == 0
                        or temp_bai.stat().st_size != source_bai.stat().st_size
                    )
                    if should_copy_bai:
                        shutil.copy(source_bai, temp_bai)
                return temp_bam
            else:
                # Source is in writable location, return as-is
                return source_bam

        # Last resort: look in package directory or development location
        package_dir = Path(__file__).parent
        sample_bam = package_dir / "data" / "yeast_trna_mappings.bam"
        sample_bai = package_dir / "data" / "yeast_trna_mappings.bam.bai"

        # Try development location (tests/data relative to project root)
        if not sample_bam.exists():
            project_root = package_dir.parent.parent
            sample_bam = project_root / "tests" / "data" / "yeast_trna_mappings.bam"
            sample_bai = project_root / "tests" / "data" / "yeast_trna_mappings.bam.bai"

        if sample_bam.exists():
            # Always copy when running from PyInstaller bundle
            is_pyinstaller = getattr(sys, "_MEIPASS", None) is not None
            is_readonly = not _is_writable_dir(sample_bam.parent)

            if is_pyinstaller or is_readonly:
                should_copy_bam = (
                    not temp_bam.exists()
                    or temp_bam.stat().st_size == 0
                    or temp_bam.stat().st_size != sample_bam.stat().st_size
                )
                if should_copy_bam:
                    shutil.copy(sample_bam, temp_bam)

                if sample_bai.exists():
                    should_copy_bai = (
                        not temp_bai.exists()
                        or temp_bai.stat().st_size == 0
                        or temp_bai.stat().st_size != sample_bai.stat().st_size
                    )
                    if should_copy_bai:
                        shutil.copy(sample_bai, temp_bai)
                return temp_bam
            return sample_bam

        return None  # BAM file is optional

    except Exception:
        return None  # BAM file is optional


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


def downsample_signal(signal: np.ndarray, downsample_factor: int = 1) -> np.ndarray:
    """Downsample signal array by taking every Nth point

    Reduces the number of data points for faster plotting while preserving
    the overall shape of the signal.

    Args:
        signal: Raw signal array (numpy array)
        downsample_factor: Factor by which to downsample (1 = no downsampling)

    Returns:
        Downsampled signal array
    """
    if downsample_factor <= 1:
        return signal

    # Take every Nth point
    return signal[::downsample_factor]


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

    except Exception as e:
        print(f"Warning: Error reading BAM file for {read_id}: {e}")

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

                reads_info.append(
                    {
                        "read_id": read.query_name,
                        "reference_start": read.reference_start,
                        "reference_end": read.reference_end,
                        "sequence": read.query_sequence,
                        "move_table": moves,
                        "stride": stride,
                        "quality_scores": quality_scores,
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
            - coverage: Number of reads covering each position
    """
    from .normalization import normalize_signal

    # Build a dict mapping reference positions to signal values
    position_signals = {}

    for read in reads_data:
        # Normalize the signal
        signal = normalize_signal(read["signal"], normalization_method)
        stride = read["stride"]
        moves = read["move_table"]
        ref_start = read["reference_start"]

        # Map signal to reference positions using move table
        ref_pos = ref_start
        sig_idx = 0

        for move in moves:
            if sig_idx < len(signal):
                # Add signal value at this reference position
                if ref_pos not in position_signals:
                    position_signals[ref_pos] = []
                position_signals[ref_pos].append(signal[sig_idx])

            sig_idx += stride
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
        coverages.append(len(values))

    return {
        "positions": np.array(positions),
        "mean_signal": np.array(mean_signals),
        "std_signal": np.array(std_signals),
        "median_signal": np.array(median_signals),
        "coverage": np.array(coverages),
    }


def calculate_base_pileup(reads_data, bam_file=None, reference_name=None):
    """Calculate IGV-style base pileup at each reference position

    Args:
        reads_data: List of read dicts from extract_reads_for_reference()
        bam_file: Optional path to BAM file (for extracting reference sequence)
        reference_name: Optional reference name (for extracting reference sequence)

    Returns:
        Dict with keys:
            - positions: Array of reference positions
            - counts: Dict mapping each position to dict of base counts
                     e.g., {pos: {'A': 10, 'C': 2, 'G': 5, 'T': 8}}
            - reference_bases: Dict mapping position to reference base (if BAM provided)
    """
    position_bases = {}

    for read in reads_data:
        sequence = read["sequence"]
        moves = read["move_table"]
        ref_start = read["reference_start"]

        # Map bases to reference positions using move table
        ref_pos = ref_start
        seq_idx = 0

        for move in moves:
            if move == 1:
                if seq_idx < len(sequence):
                    base = sequence[seq_idx].upper()
                    if ref_pos not in position_bases:
                        position_bases[ref_pos] = {}
                    if base not in position_bases[ref_pos]:
                        position_bases[ref_pos][base] = 0
                    position_bases[ref_pos][base] += 1
                    seq_idx += 1
                ref_pos += 1

    positions = sorted(position_bases.keys())

    result = {
        "positions": np.array(positions),
        "counts": {pos: position_bases[pos] for pos in positions},
    }

    # Extract reference bases if BAM file is provided
    if bam_file and reference_name and reads_data:
        try:
            # Get reference sequence from any read (they all map to same reference)
            first_read = reads_data[0]
            ref_seq, ref_start, aligned_read = get_reference_sequence_for_read(
                bam_file, first_read["read_id"]
            )

            if ref_seq and ref_start is not None:
                # Create dict mapping position to reference base
                reference_bases = {}
                for pos in positions:
                    # Calculate index in reference sequence
                    idx = pos - ref_start
                    if 0 <= idx < len(ref_seq):
                        reference_bases[pos] = ref_seq[idx].upper()

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
        moves = read["move_table"]
        ref_start = read["reference_start"]

        # Map quality scores to reference positions using move table
        ref_pos = ref_start
        seq_idx = 0

        for move in moves:
            if move == 1:
                if seq_idx < len(quality_scores):
                    qual = quality_scores[seq_idx]
                    if ref_pos not in position_qualities:
                        position_qualities[ref_pos] = []
                    position_qualities[ref_pos].append(qual)
                    seq_idx += 1
                ref_pos += 1

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


def extract_reads_for_motif(
    pod5_file, bam_file, fasta_file, motif, match_index=0, window=50, max_reads=100
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
        window: Number of bases around motif center to include (±window)
        max_reads: Maximum number of reads to return

    Returns:
        Tuple of (reads_data, motif_match) where:
            - reads_data: List of dicts with adjusted reference coordinates
              (same structure as extract_reads_for_reference())
            - motif_match: MotifMatch object for the selected match

    Raises:
        ValueError: If no motif matches found or match_index out of range
    """
    from .motif import search_motif

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

    # Define window around motif center
    motif_center = motif_match.position + (motif_match.length // 2)
    region_start = max(0, motif_center - window)
    region_end = motif_center + window

    # Extract reads overlapping this region using existing function
    # We'll use extract_reads_for_reference pattern but with region-based fetch
    import random

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

                # Get quality scores
                quality_scores = (
                    np.array(read.query_qualities) if read.query_qualities else None
                )

                # Store read info with ORIGINAL reference coordinates
                # (we'll adjust them for motif-centered plotting later)
                reads_info.append(
                    {
                        "read_id": read.query_name,
                        "reference_start": read.reference_start,
                        "reference_end": read.reference_end,
                        "chrom": motif_match.chrom,  # Add chromosome name
                        "sequence": read.query_sequence,
                        "move_table": moves,
                        "stride": stride,
                        "quality_scores": quality_scores,
                        "motif_center": motif_center,  # Add motif center for alignment
                    }
                )

        # Subsample if needed
        if len(reads_info) > max_reads:
            reads_info = random.sample(reads_info, max_reads)

        # Extract signal data from POD5
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

    Example:
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


def _route_to_plots_pane(fig) -> None:
    """
    Route Bokeh figure to Positron Plots pane via bokeh.io.show()

    Positron intercepts bokeh.io.show() calls and routes them to Plots pane
    by inspecting the call stack for bokeh.io.showing.show function.

    This ensures plots appear in the Plots pane (with history and navigation)
    rather than the Viewer pane.

    Args:
        fig: Bokeh figure object
    """
    import os
    import sys

    # Skip if running in test environment (pytest)
    if "pytest" in sys.modules or os.getenv("PYTEST_CURRENT_TEST"):
        return

    try:
        from bokeh.io import show

        show(fig)  # Positron intercepts this and routes to Plots pane
    except Exception:
        # Silently fail if bokeh.io not available or not in Positron
        pass
