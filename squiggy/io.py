"""
I/O functions for loading POD5 and BAM files

These functions are called from the Positron extension via the Jupyter kernel.
"""

import os

import pod5
import pysam

from .utils import get_bam_references


class SquiggySession:
    """
    Manages state for loaded POD5 and BAM files

    This class consolidates all squiggy kernel state into a single object,
    providing cleaner variable pane UX and better resource management.

    Attributes:
        reader: POD5 file reader
        pod5_path: Path to loaded POD5 file
        read_ids: List of read IDs from POD5 file
        bam_path: Path to loaded BAM file
        bam_info: Metadata about loaded BAM file
        ref_mapping: Mapping of reference names to read IDs

    Examples:
        >>> from squiggy import load_pod5, load_bam
        >>> session = load_pod5('data.pod5')
        >>> print(session)
        <SquiggySession: POD5: data.pod5 (1,234 reads)>
        >>> session = load_bam('alignments.bam')
        >>> print(session)
        <SquiggySession: POD5: data.pod5 (1,234 reads) | BAM: alignments.bam (1,234 reads)>
    """

    def __init__(self):
        self.reader: pod5.Reader | None = None
        self.pod5_path: str | None = None
        self.read_ids: list[str] = []
        self.bam_path: str | None = None
        self.bam_info: dict | None = None
        self.ref_mapping: dict[str, list[str]] | None = None

    def __repr__(self) -> str:
        """Return informative summary of loaded files"""
        parts = []

        if self.pod5_path:
            filename = os.path.basename(self.pod5_path)
            parts.append(f"POD5: {filename} ({len(self.read_ids):,} reads)")

        if self.bam_path:
            filename = os.path.basename(self.bam_path)
            num_reads = self.bam_info.get("num_reads", 0) if self.bam_info else 0
            parts.append(f"BAM: {filename} ({num_reads:,} reads)")

            if self.bam_info:
                if self.bam_info.get("has_modifications"):
                    mod_types = ", ".join(str(m) for m in self.bam_info["modification_types"])
                    parts.append(f"Modifications: {mod_types}")
                if self.bam_info.get("has_event_alignment"):
                    parts.append("Event alignment: yes")

        if not parts:
            return "<SquiggySession: No files loaded>"

        return f"<SquiggySession: {' | '.join(parts)}>"

    def close_pod5(self):
        """Close POD5 reader and clear POD5 state"""
        if self.reader is not None:
            self.reader.close()
            self.reader = None
        self.pod5_path = None
        self.read_ids = []

    def close_bam(self):
        """Clear BAM state"""
        self.bam_path = None
        self.bam_info = None
        self.ref_mapping = None

    def close_all(self):
        """Close all resources and clear all state"""
        self.close_pod5()
        self.close_bam()


# Global session instance (single source of truth for kernel state)
_squiggy_session = SquiggySession()


def load_pod5(file_path: str) -> SquiggySession:
    """
    Load a POD5 file into the kernel session

    This function is called from the Positron extension and makes
    the POD5 data available in the kernel via the _squiggy_session object.

    Args:
        file_path: Path to POD5 file

    Returns:
        SquiggySession object with loaded POD5 data

    Examples:
        >>> from squiggy import load_pod5
        >>> session = load_pod5('data.pod5')
        >>> print(f"Loaded {len(session.read_ids)} reads")
        >>> # Session is available as _squiggy_session in kernel
        >>> first_read = next(session.reader.reads())
    """
    # Convert to absolute path
    abs_path = os.path.abspath(file_path)

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Failed to open pod5 file at: {abs_path}")

    # Close previous reader if exists
    _squiggy_session.close_pod5()

    # Open new reader (no need for writable_working_directory in extension context)
    reader = pod5.Reader(abs_path)

    # Extract read IDs
    read_ids = [str(read.read_id) for read in reader.reads()]

    # Store state in session (no global keyword needed - just mutating object!)
    _squiggy_session.reader = reader
    _squiggy_session.pod5_path = abs_path
    _squiggy_session.read_ids = read_ids

    return _squiggy_session


def get_bam_event_alignment_status(file_path: str) -> bool:
    """
    Check if BAM file contains event alignment data (mv tag)

    The mv tag contains the move table from basecalling, which maps
    nanopore signal events to basecalled nucleotides. This is required
    for event-aligned plotting mode.

    Args:
        file_path: Path to BAM file

    Returns:
        True if mv tag is found in sampled reads

    Examples:
        >>> from squiggy import get_bam_event_alignment_status
        >>> has_events = get_bam_event_alignment_status('alignments.bam')
        >>> if has_events:
        ...     print("BAM contains event alignment data")
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"BAM file not found: {file_path}")

    max_reads_to_check = 100  # Sample first 100 reads

    try:
        bam = pysam.AlignmentFile(file_path, "rb", check_sq=False)

        for i, read in enumerate(bam.fetch(until_eof=True)):
            if i >= max_reads_to_check:
                break

            # Check for mv tag (move table)
            if read.has_tag("mv"):
                bam.close()
                return True

        bam.close()

    except Exception as e:
        print(f"Warning: Error checking BAM event alignment: {e}")
        return False

    return False


def get_bam_modification_info(file_path: str) -> dict:
    """
    Check if BAM file contains base modification tags (MM/ML)

    Args:
        file_path: Path to BAM file

    Returns:
        Dict with:
            - has_modifications: bool
            - modification_types: list of modification codes (e.g., ['m', 'h'])
            - sample_count: number of reads checked

    Examples:
        >>> from squiggy import get_bam_modification_info
        >>> mod_info = get_bam_modification_info('alignments.bam')
        >>> if mod_info['has_modifications']:
        ...     print(f"Found modifications: {mod_info['modification_types']}")
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"BAM file not found: {file_path}")

    modification_types = set()
    has_modifications = False
    has_ml = False
    reads_checked = 0
    max_reads_to_check = 100  # Sample first 100 reads

    try:
        bam = pysam.AlignmentFile(file_path, "rb", check_sq=False)

        for read in bam.fetch(until_eof=True):
            if reads_checked >= max_reads_to_check:
                break

            reads_checked += 1

            # Check for modifications using pysam's modified_bases property
            # This is cleaner and more reliable than parsing MM tags manually
            if hasattr(read, "modified_bases") and read.modified_bases:
                has_modifications = True

                # modified_bases returns dict with format:
                # {(canonical_base, strand, mod_code): [(position, quality), ...]}
                for (
                    _canonical_base,
                    _strand,
                    mod_code,
                ), _mod_list in read.modified_bases.items():
                    # mod_code can be str (e.g., 'm') or int (e.g., 17596)
                    # Store as-is to preserve type
                    modification_types.add(mod_code)

            # Check for ML tag (modification probabilities)
            if read.has_tag("ML"):
                has_ml = True

        bam.close()

    except Exception as e:
        print(f"Warning: Error checking BAM modifications: {e}")
        return {
            "has_modifications": False,
            "modification_types": [],
            "sample_count": 0,
            "has_probabilities": False,
        }

    # Convert modification_types to list, handling mixed str/int types
    # Sort with custom key that converts to string for comparison
    mod_types_list = sorted(modification_types, key=str)

    return {
        "has_modifications": has_modifications,
        "modification_types": mod_types_list,
        "sample_count": reads_checked,
        "has_probabilities": has_ml,
    }


def load_bam(file_path: str) -> SquiggySession:
    """
    Load a BAM file into the kernel session

    Args:
        file_path: Path to BAM file

    Returns:
        SquiggySession object with loaded BAM data

    Examples:
        >>> from squiggy import load_bam
        >>> session = load_bam('alignments.bam')
        >>> print(session.bam_info['references'])
        >>> if session.bam_info['has_modifications']:
        ...     print(f"Modifications: {session.bam_info['modification_types']}")
        >>> if session.bam_info['has_event_alignment']:
        ...     print("Event alignment data available")
    """
    # Convert to absolute path
    abs_path = os.path.abspath(file_path)

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"BAM file not found: {abs_path}")

    # Get references
    references = get_bam_references(abs_path)

    # Check for base modifications
    mod_info = get_bam_modification_info(abs_path)

    # Check for event alignment data
    has_event_alignment = get_bam_event_alignment_status(abs_path)

    # Build metadata dict
    bam_info = {
        "file_path": abs_path,
        "num_reads": sum(ref["read_count"] for ref in references),
        "references": references,
        "has_modifications": mod_info["has_modifications"],
        "modification_types": mod_info["modification_types"],
        "has_probabilities": mod_info["has_probabilities"],
        "has_event_alignment": has_event_alignment,
    }

    # Store state in session (no global keyword needed - just mutating object!)
    _squiggy_session.bam_path = abs_path
    _squiggy_session.bam_info = bam_info

    return _squiggy_session


def get_read_to_reference_mapping() -> dict[str, list[str]]:
    """
    Get mapping of reference names to read IDs from currently loaded BAM

    Returns:
        Dict mapping reference name to list of read IDs

    Raises:
        RuntimeError: If no BAM file is loaded

    Examples:
        >>> from squiggy import load_bam, get_read_to_reference_mapping
        >>> load_bam('alignments.bam')
        >>> mapping = get_read_to_reference_mapping()
        >>> print(f"References: {list(mapping.keys())}")
    """
    if _squiggy_session.bam_path is None:
        raise RuntimeError("No BAM file is currently loaded")

    if not os.path.exists(_squiggy_session.bam_path):
        raise FileNotFoundError(f"BAM file not found: {_squiggy_session.bam_path}")

    # Open BAM file
    bam = pysam.AlignmentFile(_squiggy_session.bam_path, "rb", check_sq=False)

    # Map reference to read IDs
    ref_to_reads: dict[str, list[str]] = {}

    try:
        for read in bam.fetch(until_eof=True):
            if read.is_unmapped:
                continue

            ref_name = bam.get_reference_name(read.reference_id)
            read_id = read.query_name

            if ref_name not in ref_to_reads:
                ref_to_reads[ref_name] = []

            ref_to_reads[ref_name].append(read_id)

    finally:
        bam.close()

    # Store in session (no global keyword needed!)
    _squiggy_session.ref_mapping = ref_to_reads

    return ref_to_reads


def get_current_files() -> dict[str, str | None]:
    """
    Get paths of currently loaded files

    Returns:
        Dict with pod5_path and bam_path (may be None)
    """
    return {"pod5_path": _squiggy_session.pod5_path, "bam_path": _squiggy_session.bam_path}


def get_read_ids() -> list[str]:
    """
    Get list of read IDs from currently loaded POD5 file

    Returns:
        List of read ID strings
    """
    if not _squiggy_session.read_ids:
        raise ValueError("No POD5 file is currently loaded")

    return _squiggy_session.read_ids


def close_pod5():
    """
    Close the currently open POD5 reader

    Call this to free resources when done.

    Examples:
        >>> from squiggy import load_pod5, close_pod5
        >>> load_pod5('data.pod5')
        >>> # ... work with data ...
        >>> close_pod5()
    """
    # Clear session (no global keyword needed!)
    _squiggy_session.close_pod5()


def close_bam():
    """
    Clear the currently loaded BAM file state

    Call this to free BAM-related resources when done. Unlike close_pod5(),
    this doesn't need to close a file handle since BAM files are opened
    and closed per-operation.

    Examples:
        >>> from squiggy import load_bam, close_bam
        >>> load_bam('alignments.bam')
        >>> # ... work with alignments ...
        >>> close_bam()
    """
    # Clear session (no global keyword needed!)
    _squiggy_session.close_bam()
