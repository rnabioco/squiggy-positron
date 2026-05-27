"""
Squiggy I/O module - pure utility functions for BAM/POD5 metadata extraction

The global state management (SquiggyKernel, load_pod5, etc.) has been removed.
Use the OO API (Pod5File, BamFile, FastaFile, Sample) instead.
"""

# Pure functions (no global state)
from .core import (
    _collect_bam_metadata_single_pass,
    get_bam_event_alignment_status,
    get_bam_modification_info,
)
from .performance import LazyReadList

__all__ = [
    "LazyReadList",
    "_collect_bam_metadata_single_pass",
    "get_bam_event_alignment_status",
    "get_bam_modification_info",
]
