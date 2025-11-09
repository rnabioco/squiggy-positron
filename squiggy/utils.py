"""Utility functions for Squiggy application"""

import os
import platform
import shutil
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pod5
import pysam


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

        # Last resort: look in package directory
        package_dir = Path(__file__).parent
        sample_path = package_dir / "data" / "yeast_trna_reads.pod5"

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


def get_test_data_path():
    """Get the path to the bundled test data directory

    Returns the path to the squiggy/data directory which contains test/demo files:
    - yeast_trna_reads.pod5
    - yeast_trna_mappings.bam
    - yeast_trna_mappings.bam.bai
    - yeast_trna.fa
    - yeast_trna.fa.fai

    Returns:
        str: Path to the squiggy/data directory

    Raises:
        FileNotFoundError: If data directory cannot be found

    Examples:
        >>> from pathlib import Path
        >>> data_dir = Path(get_test_data_path())
        >>> pod5_file = data_dir / 'yeast_trna_reads.pod5'
    """
    import importlib.util

    try:
        # Find the squiggy package location
        spec = importlib.util.find_spec("squiggy")
        if spec is None or spec.origin is None:
            raise FileNotFoundError("Could not locate squiggy package")

        package_dir = os.path.dirname(spec.origin)
        data_dir = os.path.join(package_dir, "data")

        # Verify the directory exists
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory not found at {data_dir}")

        return data_dir

    except Exception as e:
        raise FileNotFoundError(f"Test data directory not found. Error: {e}") from None


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

        # Last resort: look in package directory
        package_dir = Path(__file__).parent
        sample_bam = package_dir / "data" / "yeast_trna_mappings.bam"
        sample_bai = package_dir / "data" / "yeast_trna_mappings.bam.bai"

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
    from .constants import DEFAULT_DOWNSAMPLE

    if downsample_factor is None:
        downsample_factor = DEFAULT_DOWNSAMPLE

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
                    from .alignment import _parse_alignment

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


def calculate_modification_statistics(reads_data, mod_filter=None):
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
    from .normalization import normalize_signal

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
                import pysam

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
        if bam_file and not reference_bases:
            try:
                # Get reference sequence from any read (they all map to same reference)
                first_read = reads_data[0]
                ref_seq, ref_start, aligned_read = get_reference_sequence_for_read(
                    bam_file, first_read["read_id"]
                )

                if ref_seq and ref_start is not None:
                    # Create dict mapping position to reference base
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
    from .motif import search_motif

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


# Phase 2: Comparison and Aggregation Functions


def compare_read_sets(read_ids_a: list[str], read_ids_b: list[str]) -> dict:
    """
    Compare two sets of read IDs and find common/unique reads

    Analyzes overlap between two read ID lists, useful for comparing
    which reads were sequenced in each sample.

    Args:
        read_ids_a: List of read IDs from sample A
        read_ids_b: List of read IDs from sample B

    Returns:
        Dict with comparison results:
            - common_reads (set): Read IDs present in both
            - unique_to_a (set): Read IDs only in A
            - unique_to_b (set): Read IDs only in B
            - common_count (int): Number of common reads
            - unique_a_count (int): Number unique to A
            - unique_b_count (int): Number unique to B
            - overlap_percent_a (float): Percentage of A's reads in common
            - overlap_percent_b (float): Percentage of B's reads in common

    Examples:
        >>> from squiggy.utils import compare_read_sets
        >>> result = compare_read_sets(reads_a, reads_b)
        >>> print(f"Common reads: {result['common_count']}")
        >>> print(f"A unique: {result['unique_a_count']}")
        >>> print(f"B unique: {result['unique_b_count']}")
    """
    set_a = set(read_ids_a)
    set_b = set(read_ids_b)

    common = set_a & set_b
    unique_a = set_a - set_b
    unique_b = set_b - set_a

    # Calculate overlap percentages
    overlap_percent_a = (len(common) / len(set_a) * 100) if set_a else 0
    overlap_percent_b = (len(common) / len(set_b) * 100) if set_b else 0

    return {
        "common_reads": common,
        "unique_to_a": unique_a,
        "unique_to_b": unique_b,
        "common_count": len(common),
        "unique_a_count": len(unique_a),
        "unique_b_count": len(unique_b),
        "overlap_percent_a": overlap_percent_a,
        "overlap_percent_b": overlap_percent_b,
    }


def calculate_delta_stats(
    stats_a: dict, stats_b: dict, stat_names: list[str] | None = None
) -> dict:
    """
    Calculate differences (deltas) between corresponding statistics

    Computes delta values for statistics arrays, useful for visualizing
    differences between two basecalling models or runs.

    Args:
        stats_a: Dictionary of statistics from sample A
                (e.g., from calculate_aggregate_signal())
        stats_b: Dictionary of statistics from sample B (same structure)
        stat_names: List of stat keys to compute deltas for
                   (default: all matching keys with array values)

    Returns:
        Dict with delta arrays:
            - delta_{stat_name}: Array of differences (B - A)
            - positions: Position array (if available)

    Examples:
        >>> from squiggy.utils import calculate_delta_stats
        >>> delta = calculate_delta_stats(stats_a, stats_b, ['mean_signal'])
        >>> print(f"Max delta: {np.max(np.abs(delta['delta_mean_signal']))}")
    """
    deltas = {}

    # If no stat names provided, infer from matching keys
    if stat_names is None:
        stat_names = []
        for key in stats_a.keys():
            if key in stats_b and isinstance(stats_a[key], np.ndarray):
                stat_names.append(key)

    # First pass: determine minimum length across all arrays
    min_len = None
    for stat_name in stat_names:
        if stat_name not in stats_a or stat_name not in stats_b:
            continue

        val_a = stats_a[stat_name]
        val_b = stats_b[stat_name]

        if isinstance(val_a, np.ndarray) and isinstance(val_b, np.ndarray):
            curr_min = min(len(val_a), len(val_b))
            if min_len is None:
                min_len = curr_min
            else:
                min_len = min(min_len, curr_min)

    # If no arrays found, return empty
    if min_len is None:
        return deltas

    # Calculate deltas (all will be same length now)
    for stat_name in stat_names:
        if stat_name not in stats_a or stat_name not in stats_b:
            continue

        val_a = stats_a[stat_name]
        val_b = stats_b[stat_name]

        if isinstance(val_a, np.ndarray) and isinstance(val_b, np.ndarray):
            delta = val_b[:min_len] - val_a[:min_len]
            deltas[f"delta_{stat_name}"] = delta

    # Include positions if available - TRUNCATE to match delta length
    if "positions" in stats_a:
        deltas["positions"] = stats_a["positions"][:min_len]
    elif "positions" in stats_b:
        deltas["positions"] = stats_b["positions"][:min_len]

    return deltas


def compare_signal_distributions(signal_a: np.ndarray, signal_b: np.ndarray) -> dict:
    """
    Compare signal distributions from two samples

    Computes statistical measures to characterize differences in signal
    distributions between two samples.

    Args:
        signal_a: Signal array from sample A
        signal_b: Signal array from sample B

    Returns:
        Dict with distribution comparison:
            - mean_a, mean_b: Mean signal
            - median_a, median_b: Median signal
            - std_a, std_b: Standard deviation
            - min_a, min_b: Minimum signal
            - max_a, max_b: Maximum signal
            - mean_diff: Difference in means
            - std_diff: Difference in standard deviations

    Examples:
        >>> from squiggy.utils import compare_signal_distributions
        >>> result = compare_signal_distributions(signal_a, signal_b)
        >>> print(f"Mean difference: {result['mean_diff']:.2f}")
    """
    return {
        "mean_a": float(np.mean(signal_a)),
        "mean_b": float(np.mean(signal_b)),
        "median_a": float(np.median(signal_a)),
        "median_b": float(np.median(signal_b)),
        "std_a": float(np.std(signal_a)),
        "std_b": float(np.std(signal_b)),
        "min_a": float(np.min(signal_a)),
        "min_b": float(np.min(signal_b)),
        "max_a": float(np.max(signal_a)),
        "max_b": float(np.max(signal_b)),
        "mean_diff": float(np.mean(signal_b) - np.mean(signal_a)),
        "std_diff": float(np.std(signal_b) - np.std(signal_a)),
    }


def parse_plot_parameters(
    mode: str | None = None,
    normalization: str = "ZNORM",
    theme: str = "LIGHT",
):
    """
    Parse and validate plot parameters (mode, normalization, theme)

    This utility function eliminates duplicate parameter parsing code across
    plotting functions by centralizing the conversion from string parameters
    to enum values.

    Args:
        mode: Plot mode string (e.g., "SINGLE", "OVERLAY", "EVENTALIGN", "AGGREGATE").
              If None, only normalization and theme are parsed.
        normalization: Normalization method string (default: "ZNORM").
                       Valid values: "NONE", "ZNORM", "MEDIAN", "MAD"
        theme: Color theme string (default: "LIGHT").
               Valid values: "LIGHT", "DARK"

    Returns:
        Dictionary with parsed enum values:
        - "mode": PlotMode enum (if mode parameter provided, else None)
        - "normalization": NormalizationMethod enum
        - "theme": Theme enum

    Raises:
        KeyError: If invalid mode, normalization, or theme string provided

    Examples:
        >>> params = parse_plot_parameters(mode="SINGLE", normalization="ZNORM", theme="LIGHT")
        >>> params["mode"]
        <PlotMode.SINGLE: 'SINGLE'>

        >>> params = parse_plot_parameters(normalization="MEDIAN", theme="DARK")
        >>> params["normalization"]
        <NormalizationMethod.MEDIAN: 'MEDIAN'>
    """
    from .constants import NormalizationMethod, PlotMode, Theme

    result = {
        "normalization": NormalizationMethod[normalization.upper()],
        "theme": Theme[theme.upper()],
    }

    if mode is not None:
        result["mode"] = PlotMode[mode.upper()]

    return result


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
