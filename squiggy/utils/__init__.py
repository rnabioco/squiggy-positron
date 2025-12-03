"""Utility functions for Squiggy application

This module provides a unified interface to all utility functions,
organized into submodules by functionality:
- paths: Path and file location utilities
- bam: BAM file operations
- signal: Signal processing utilities
- statistics: Aggregate statistics calculations
- comparison: Comparison and delta utilities
- motif: Motif extraction utilities
- plotting: Plotting utilities
"""

# Path utilities
# BAM utilities
from .bam import (
    ModelProvenance,
    extract_model_provenance,
    extract_reads_for_reference,
    get_available_reads_for_reference,
    get_bam_references,
    get_basecall_data,
    get_read_to_reference_mapping,
    get_reads_in_region,
    get_reference_sequence_for_read,
    get_reference_sequence_from_fasta,
    index_bam_file,
    open_bam_safe,
    parse_region,
    reverse_complement,
    validate_bam_reads_in_pod5,
    validate_sq_headers,
)

# Comparison utilities
from .comparison import (
    calculate_delta_stats,
    compare_read_sets,
    compare_signal_distributions,
)

# Motif utilities
from .motif import (
    align_reads_to_motif_center,
    clip_reads_to_window,
    extract_reads_for_motif,
)
from .paths import (
    _is_writable_dir,
    get_icon_path,
    get_logo_path,
    get_sample_bam_path,
    get_sample_data_path,
    get_test_data_path,
    writable_working_directory,
)

# Plotting utilities
from .plotting import (
    _route_to_plots_pane,
    parse_plot_parameters,
)

# Signal processing utilities
from .signal import (
    calculate_aligned_move_indices,
    downsample_signal,
    get_aligned_move_indices_from_read,
    iter_aligned_bases,
)

# Statistics utilities
from .statistics import (
    calculate_aggregate_signal,
    calculate_base_pileup,
    calculate_dwell_time_statistics,
    calculate_modification_statistics,
    calculate_quality_by_position,
)

__all__ = [
    # Path utilities
    "writable_working_directory",
    "get_icon_path",
    "get_logo_path",
    "get_sample_data_path",
    "get_test_data_path",
    "_is_writable_dir",
    "get_sample_bam_path",
    # BAM utilities
    "ModelProvenance",
    "validate_bam_reads_in_pod5",
    "get_basecall_data",
    "parse_region",
    "index_bam_file",
    "get_bam_references",
    "get_read_to_reference_mapping",
    "get_reads_in_region",
    "reverse_complement",
    "get_reference_sequence_for_read",
    "get_reference_sequence_from_fasta",
    "get_available_reads_for_reference",
    "extract_reads_for_reference",
    "extract_model_provenance",
    "validate_sq_headers",
    "open_bam_safe",
    # Signal processing utilities
    "downsample_signal",
    "calculate_aligned_move_indices",
    "get_aligned_move_indices_from_read",
    "iter_aligned_bases",
    # Statistics utilities
    "calculate_modification_statistics",
    "calculate_dwell_time_statistics",
    "calculate_aggregate_signal",
    "calculate_base_pileup",
    "calculate_quality_by_position",
    # Comparison utilities
    "compare_read_sets",
    "calculate_delta_stats",
    "compare_signal_distributions",
    # Motif utilities
    "extract_reads_for_motif",
    "align_reads_to_motif_center",
    "clip_reads_to_window",
    # Plotting utilities
    "_route_to_plots_pane",
    "parse_plot_parameters",
]
