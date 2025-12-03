"""
Squiggy I/O module - POD5/BAM/FASTA file handling and kernel state management

This module provides:
- File loading functions (load_pod5, load_bam, load_fasta)
- Kernel state management (SquiggyKernel, squiggy_kernel)
- Multi-sample workflow support (Sample, load_sample, get_sample)
- Performance optimization classes (LazyReadList, Pod5Index)
- Comparison functions (get_common_reads, get_unique_reads, compare_samples)
"""

# Performance optimization classes
# Comparison functions
from .comparison import (
    compare_samples,
    get_common_reads,
    get_unique_reads,
)

# Core I/O functions
from .core import (
    close_all_samples,
    close_bam,
    close_fasta,
    close_pod5,
    get_bam_event_alignment_status,
    get_bam_modification_info,
    # General functions
    get_current_files,
    get_read_by_id,
    get_read_ids,
    get_read_to_reference_mapping,
    get_reads_batch,
    get_reads_batch_multi_sample,
    get_reads_for_reference_paginated,
    get_sample,
    list_samples,
    # BAM functions
    load_bam,
    # FASTA functions
    load_fasta,
    # POD5 functions
    load_pod5,
    # Multi-sample convenience functions
    load_sample,
    remove_sample,
)

# Kernel state management
from .kernel import SquiggyKernel, squiggy_kernel
from .performance import LazyReadList, Pod5Index

# Sample management
from .samples import Sample

__all__ = [
    # Performance classes
    "LazyReadList",
    "Pod5Index",
    # Sample management
    "Sample",
    # Kernel state
    "SquiggyKernel",
    "squiggy_kernel",
    # POD5 functions
    "load_pod5",
    "close_pod5",
    "get_read_ids",
    "get_reads_batch",
    "get_reads_batch_multi_sample",
    "get_read_by_id",
    # BAM functions
    "load_bam",
    "close_bam",
    "get_bam_modification_info",
    "get_bam_event_alignment_status",
    "get_read_to_reference_mapping",
    "get_reads_for_reference_paginated",
    # FASTA functions
    "load_fasta",
    "close_fasta",
    # General functions
    "get_current_files",
    # Multi-sample functions
    "load_sample",
    "get_sample",
    "list_samples",
    "remove_sample",
    "close_all_samples",
    # Comparison functions
    "get_common_reads",
    "get_unique_reads",
    "compare_samples",
]
