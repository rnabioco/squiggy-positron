"""
Squiggy: Visualize Oxford Nanopore sequencing data

This package provides functions for visualizing POD5 signal data with
optional base annotations from BAM files. Can be used standalone or
integrated with the Positron extension.

Example usage in Positron console:
    >>> import squiggy
    >>> reader, read_ids = squiggy.load_pod5('data.pod5')
    >>> html = squiggy.plot_read('read_001')
    >>> # HTML is automatically displayed in extension webview

Example usage in Jupyter notebook:
    >>> from squiggy import load_pod5, plot_read
    >>> from bokeh.plotting import output_notebook, show
    >>> from bokeh.models import Div
    >>>
    >>> reader, read_ids = load_pod5('data.pod5')
    >>> html = plot_read(read_ids[0], mode='EVENTALIGN')
    >>>
    >>> output_notebook()
    >>> show(Div(text=html))
"""

__version__ = "0.1.16"

# Object-oriented API (NEW - notebook-friendly)
# Core data structures and constants
from .alignment import AlignedRead, BaseAnnotation, extract_alignment_from_bam
from .api import BamFile, FastaFile, Pod5File, Read, figure_to_html
from .constants import (
    BASE_COLORS,
    BASE_COLORS_DARK,
    NormalizationMethod,
    PlotMode,
    Theme,
)

# I/O functions
from .io import (
    LazyReadList,
    Pod5Index,
    Sample,
    SquiggyKernel,
    close_all_samples,
    close_bam,
    close_fasta,
    close_pod5,
    compare_samples,
    get_bam_event_alignment_status,
    get_bam_modification_info,
    get_common_reads,
    get_current_files,
    get_read_by_id,
    get_read_ids,
    get_read_to_reference_mapping,
    get_reads_batch,
    get_reads_for_reference_paginated,
    get_sample,
    get_unique_reads,
    list_samples,
    load_bam,
    load_fasta,
    load_pod5,
    load_sample,
    remove_sample,
    squiggy_kernel,
)
from .motif import (
    IUPAC_CODES,
    MotifMatch,
    count_motifs,
    iupac_to_regex,
    search_motif,
)
from .normalization import normalize_signal
from .plot_factory import create_plot_strategy

# Plotting functions
from .plotting import (
    plot_aggregate,
    plot_aggregate_comparison,
    plot_delta_comparison,
    plot_motif_aggregate_all,
    plot_read,
    plot_reads,
    plot_signal_overlay_comparison,
)

# Utility functions and data classes
from .utils import (
    ModelProvenance,
    downsample_signal,
    extract_model_provenance,
    get_bam_references,
    get_reads_in_region,
    get_reference_sequence_for_read,
    get_test_data_path,
    open_bam_safe,
    parse_plot_parameters,
    parse_region,
    reverse_complement,
    validate_sq_headers,
)

# Import pod5 for user convenience
try:
    import pod5
except ImportError:
    pod5 = None

# Import pysam for user convenience
try:
    import pysam
except ImportError:
    pysam = None
__all__ = [
    # Version
    "__version__",
    # Object-oriented API (NEW)
    "Pod5File",
    "Read",
    "BamFile",
    "FastaFile",
    "figure_to_html",
    # Main functions (legacy API - for Positron extension)
    "load_pod5",
    "load_bam",
    "load_fasta",
    "plot_read",
    "plot_reads",
    "plot_aggregate",
    "plot_motif_aggregate_all",
    "plot_delta_comparison",  # Phase 3 - NEW
    "plot_signal_overlay_comparison",  # Phase 1 - NEW multi-sample comparison
    "plot_aggregate_comparison",  # Multi-sample aggregate statistics comparison
    "get_current_files",
    "get_read_ids",
    "get_bam_modification_info",
    "get_bam_event_alignment_status",
    "get_read_to_reference_mapping",
    "close_pod5",
    "close_bam",
    "close_fasta",
    # Multi-sample API (Phase 1 - NEW)
    "load_sample",
    "get_sample",
    "list_samples",
    "remove_sample",
    "close_all_samples",
    # Comparison API (Phase 2 - NEW)
    "get_common_reads",
    "get_unique_reads",
    "compare_samples",
    # Kernel state management
    "SquiggyKernel",
    "squiggy_kernel",
    "Sample",
    # Performance optimization classes
    "LazyReadList",
    "Pod5Index",
    "get_reads_batch",
    "get_read_by_id",
    "get_reads_for_reference_paginated",
    # Data structures
    "AlignedRead",
    "BaseAnnotation",
    "MotifMatch",
    "ModelProvenance",
    # Constants
    "NormalizationMethod",
    "PlotMode",
    "Theme",
    "BASE_COLORS",
    "BASE_COLORS_DARK",
    "IUPAC_CODES",
    # Functions
    "extract_alignment_from_bam",
    "normalize_signal",
    "get_bam_references",
    "get_reads_in_region",
    "get_reference_sequence_for_read",
    "get_test_data_path",
    "parse_region",
    "reverse_complement",
    "downsample_signal",
    "create_plot_strategy",
    "iupac_to_regex",
    "search_motif",
    "count_motifs",
    "extract_model_provenance",
    "validate_sq_headers",
    # Comparison utilities (Phase 2 - NEW)
    "compare_read_sets",
    "calculate_delta_stats",
    "compare_signal_distributions",
    # Refactoring utilities (Phase 3 - NEW)
    "parse_plot_parameters",
    "open_bam_safe",
    # Classes
]
