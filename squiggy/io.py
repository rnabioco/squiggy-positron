"""
I/O functions for loading POD5 and BAM files

These functions are called from the Positron extension via the Jupyter kernel.
"""

import os
from typing import Dict, List, Tuple, Optional
import pod5
import pysam
from .utils import get_bam_references


# Global state for currently loaded files
_current_pod5_reader: Optional[pod5.Reader] = None
_current_pod5_path: Optional[str] = None
_current_bam_path: Optional[str] = None
_current_read_ids: List[str] = []


def load_pod5(file_path: str) -> Tuple[pod5.Reader, List[str]]:
    """
    Load a POD5 file and return reader and list of read IDs

    This function is called from the Positron extension and makes
    the reader available in the kernel for user inspection.

    Args:
        file_path: Path to POD5 file

    Returns:
        Tuple of (reader, read_ids)

    Example:
        >>> from squiggy import load_pod5
        >>> reader, read_ids = load_pod5('data.pod5')
        >>> print(f"Loaded {len(read_ids)} reads")
        >>> # Reader is now available for inspection
        >>> first_read = next(reader.reads())
    """
    global _current_pod5_reader, _current_pod5_path, _current_read_ids

    # Convert to absolute path
    abs_path = os.path.abspath(file_path)

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Failed to open pod5 file at: {abs_path}")

    # Close previous reader if exists
    if _current_pod5_reader is not None:
        _current_pod5_reader.close()

    # Open new reader (no need for writable_working_directory in extension context)
    reader = pod5.Reader(abs_path)

    # Extract read IDs
    read_ids = [str(read.read_id) for read in reader.reads()]

    # Store state
    _current_pod5_reader = reader
    _current_pod5_path = abs_path
    _current_read_ids = read_ids

    return reader, read_ids


def load_bam(file_path: str) -> Dict:
    """
    Load a BAM file and return metadata

    Args:
        file_path: Path to BAM file

    Returns:
        Dict with file metadata including references

    Example:
        >>> from squiggy import load_bam
        >>> bam_info = load_bam('alignments.bam')
        >>> print(bam_info['references'])
    """
    global _current_bam_path

    # Convert to absolute path
    abs_path = os.path.abspath(file_path)

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"BAM file not found: {abs_path}")

    # Get references
    references = get_bam_references(abs_path)

    # Store path
    _current_bam_path = abs_path

    return {
        'file_path': abs_path,
        'num_reads': sum(ref['read_count'] for ref in references),
        'references': references
    }


def get_current_files() -> Dict[str, Optional[str]]:
    """
    Get paths of currently loaded files

    Returns:
        Dict with pod5_path and bam_path (may be None)
    """
    return {
        'pod5_path': _current_pod5_path,
        'bam_path': _current_bam_path
    }


def get_read_ids() -> List[str]:
    """
    Get list of read IDs from currently loaded POD5 file

    Returns:
        List of read ID strings
    """
    if not _current_read_ids:
        raise ValueError("No POD5 file is currently loaded")

    return _current_read_ids


def close_pod5():
    """
    Close the currently open POD5 reader

    Call this to free resources when done.
    """
    global _current_pod5_reader, _current_pod5_path, _current_read_ids

    if _current_pod5_reader is not None:
        _current_pod5_reader.close()
        _current_pod5_reader = None
        _current_pod5_path = None
        _current_read_ids = []
