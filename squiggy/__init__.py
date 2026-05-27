"""
Squiggy: Visualize Oxford Nanopore sequencing data

This package provides functions for visualizing POD5 signal data with
optional base annotations from BAM files.

Example usage:
    >>> import squiggy
    >>> from bokeh.plotting import show, output_notebook
    >>> output_notebook()
    >>>
    >>> pod5 = squiggy.Pod5File('data.pod5')
    >>> bam = squiggy.BamFile('alignments.bam')
    >>> read = pod5.get_read(pod5.read_ids[0])
    >>> fig = read.plot(mode='EVENTALIGN', bam_file=bam)
    >>> show(fig)
"""

__version__ = "0.1.32"

# Core data access classes
# Data structures and constants
from .alignment import AlignedRead, BaseAnnotation, extract_alignment_from_bam
from .api import BamFile, FastaFile, Pod5File, Read, Sample, figure_to_html
from .constants import (
    BASE_COLORS,
    BASE_COLORS_DARK,
    NormalizationMethod,
    PlotMode,
    Theme,
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

# Plotting functions (accept OO objects, return Bokeh figures)
from .plotting import (
    plot_aggregate,
    plot_aggregate_comparison,
    plot_delta_comparison,
    plot_motif_aggregate_all,
    plot_pileup,
    plot_read,
    plot_reads,
    plot_signal_overlay_comparison,
)

# Utility functions
from .utils import (
    ModelProvenance,
    calculate_delta_stats,
    compare_read_sets,
    compare_signal_distributions,
    downsample_signal,
    extract_alignments_for_reference,
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
    # Core data access classes
    "Pod5File",
    "Read",
    "BamFile",
    "FastaFile",
    "Sample",
    "figure_to_html",
    # Plotting functions
    "plot_read",
    "plot_reads",
    "plot_aggregate",
    "plot_pileup",
    "plot_motif_aggregate_all",
    "plot_delta_comparison",
    "plot_signal_overlay_comparison",
    "plot_aggregate_comparison",
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
    "extract_alignments_for_reference",
    "validate_sq_headers",
    "compare_read_sets",
    "calculate_delta_stats",
    "compare_signal_distributions",
    "parse_plot_parameters",
    "open_bam_safe",
]
