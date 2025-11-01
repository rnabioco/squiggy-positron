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

__version__ = "0.1.4"

# Object-oriented API (NEW - notebook-friendly)
# Core data structures and constants
from .alignment import AlignedRead, BaseAnnotation, extract_alignment_from_bam
from .api import BamFile, Pod5File, Read, figure_to_html
from .constants import (
    BASE_COLORS,
    BASE_COLORS_DARK,
    NormalizationMethod,
    PlotMode,
    Theme,
)

# I/O functions
from .io import (
    close_pod5,
    get_bam_event_alignment_status,
    get_bam_modification_info,
    get_current_files,
    get_read_ids,
    get_read_to_reference_mapping,
    load_bam,
    load_pod5,
)
from .normalization import normalize_signal
from .plotter import SquigglePlotter

# Utility functions
from .utils import (
    downsample_signal,
    get_bam_references,
    get_reads_in_region,
    get_reference_sequence_for_read,
    parse_region,
    reverse_complement,
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


def plot_read(
    read_id: str,
    mode: str = "SINGLE",
    normalization: str = "ZNORM",
    theme: str = "LIGHT",
    downsample: int = 1,
    show_dwell_time: bool = False,
    show_labels: bool = True,
    position_label_interval: int = 100,
    scale_dwell_time: bool = False,
    min_mod_probability: float = 0.5,
    enabled_mod_types: list = None,
    show_signal_points: bool = False,
) -> str:
    """
    Generate a Bokeh HTML plot for a single read

    Args:
        read_id: Read ID to plot
        mode: Plot mode (SINGLE, EVENTALIGN)
        normalization: Normalization method (NONE, ZNORM, MEDIAN, MAD)
        theme: Color theme (LIGHT, DARK)
        downsample: Downsampling factor (1 = no downsampling, 10 = every 10th point)
        show_dwell_time: Color bases by dwell time (requires event-aligned mode)
        show_labels: Show base labels on plot (event-aligned mode)
        position_label_interval: Interval for position labels
        scale_dwell_time: Scale x-axis by cumulative dwell time instead of regular time
        min_mod_probability: Minimum probability threshold for displaying modifications (0-1)
        enabled_mod_types: List of modification type codes to display (None = all)
        show_signal_points: Show individual signal points as circles

    Returns:
        Bokeh HTML string

    Example:
        >>> html = plot_read('read_001', mode='EVENTALIGN')
        >>> # Extension displays this automatically
        >>> # Or save to file:
        >>> with open('plot.html', 'w') as f:
        >>>     f.write(html)
    """
    from .io import _current_bam_path, _current_pod5_reader

    if _current_pod5_reader is None:
        raise ValueError("No POD5 file loaded. Call load_pod5() first.")

    # Get read data
    read_obj = None
    for read in _current_pod5_reader.reads():
        if str(read.read_id) == read_id:
            read_obj = read
            break

    if read_obj is None:
        raise ValueError(f"Read not found: {read_id}")

    # Extract signal (no automatic downsampling - use user-specified factor)
    signal = read_obj.signal

    # Get alignment if available
    aligned_read = None
    if _current_bam_path and mode.upper() == "EVENTALIGN":
        from .alignment import extract_alignment_from_bam

        aligned_read = extract_alignment_from_bam(_current_bam_path, read_id)

    # Parse parameters
    plot_mode = PlotMode[mode.upper()]
    norm_method = NormalizationMethod[normalization.upper()]

    # Extract sequence, seq_to_sig_map, and modifications if available
    sequence = None
    seq_to_sig_map = None
    modifications = None
    if aligned_read:
        sequence = aligned_read.sequence
        # Build seq_to_sig_map from base annotations
        if aligned_read.bases:
            seq_to_sig_map = [ann.signal_start for ann in aligned_read.bases]
        # Extract modifications
        if hasattr(aligned_read, "modifications") and aligned_read.modifications:
            modifications = aligned_read.modifications

    # Generate plot (returns HTML and figure)
    if plot_mode in (PlotMode.SINGLE, PlotMode.EVENTALIGN):
        if plot_mode == PlotMode.EVENTALIGN and aligned_read is None:
            raise ValueError(
                "EVENTALIGN mode requires a BAM file. Call load_bam() first."
            )

        # Parse theme parameter
        theme_enum = Theme[theme.upper()]

        html, figure = SquigglePlotter.plot_single_read(
            signal=signal,
            read_id=read_id,
            sample_rate=read_obj.run_info.sample_rate,
            sequence=sequence,
            seq_to_sig_map=seq_to_sig_map,
            normalization=norm_method,
            downsample=downsample,
            show_dwell_time=show_dwell_time,
            show_labels=show_labels,
            show_signal_points=show_signal_points,
            modifications=modifications,
            scale_dwell_time=scale_dwell_time,
            min_mod_probability=min_mod_probability,
            enabled_mod_types=enabled_mod_types,
            theme=theme_enum,
        )
        return html
    else:
        raise ValueError(
            f"Plot mode {plot_mode} not yet supported in extension. Use SINGLE or EVENTALIGN."
        )


def plot_reads(
    read_ids: list,
    mode: str = "OVERLAY",
    normalization: str = "ZNORM",
    theme: str = "LIGHT",
    downsample: int = 1,
    show_dwell_time: bool = False,
    show_labels: bool = True,
    scale_dwell_time: bool = False,
    min_mod_probability: float = 0.5,
    enabled_mod_types: list = None,
    show_signal_points: bool = False,
) -> str:
    """
    Generate a Bokeh HTML plot for multiple reads

    Args:
        read_ids: List of read IDs to plot
        mode: Plot mode (OVERLAY, STACKED)
        normalization: Normalization method (NONE, ZNORM, MEDIAN, MAD)
        theme: Color theme (LIGHT, DARK)
        downsample: Downsampling factor (1 = no downsampling, 10 = every 10th point)
        show_dwell_time: Color bases by dwell time
        show_labels: Show base labels on plot
        scale_dwell_time: Scale x-axis by cumulative dwell time
        min_mod_probability: Minimum probability threshold for displaying modifications
        enabled_mod_types: List of modification type codes to display
        show_signal_points: Show individual signal points as circles

    Returns:
        Bokeh HTML string

    Example:
        >>> html = plot_reads(['read_001', 'read_002'], mode='OVERLAY')
    """
    from .io import _current_pod5_reader

    if _current_pod5_reader is None:
        raise ValueError("No POD5 file loaded. Call load_pod5() first.")

    # Parse parameters
    plot_mode = PlotMode[mode.upper()]

    # For now, only OVERLAY mode is supported by plotting multiple in sequence
    if plot_mode == PlotMode.OVERLAY:
        # Generate HTML for each read separately and combine
        # This is a simplified implementation
        htmls = []
        for read_id in read_ids:
            html = plot_read(
                read_id,
                mode="SINGLE",
                normalization=normalization,
                theme=theme,
                downsample=downsample,
                show_dwell_time=show_dwell_time,
                show_labels=show_labels,
                scale_dwell_time=scale_dwell_time,
                min_mod_probability=min_mod_probability,
                enabled_mod_types=enabled_mod_types,
                show_signal_points=show_signal_points,
            )
            htmls.append(html)
        # Return first plot for now - full OVERLAY implementation TODO
        return htmls[0] if htmls else ""
    else:
        raise ValueError(
            f"Plot mode {mode} not yet fully implemented. Use plot_read() for single reads."
        )


def plot_aggregate(
    reference_name: str,
    max_reads: int = 100,
    normalization: str = "ZNORM",
    theme: str = "LIGHT",
) -> str:
    """
    Generate aggregate multi-read visualization for a reference sequence

    Creates a three-track plot showing:
    1. Aggregate signal (mean ± std dev across reads)
    2. Base pileup (IGV-style stacked bar chart)
    3. Quality scores by position

    Args:
        reference_name: Name of reference sequence from BAM file
        max_reads: Maximum number of reads to sample for aggregation (default 100)
        normalization: Normalization method (NONE, ZNORM, MEDIAN, MAD)
        theme: Color theme (LIGHT, DARK)

    Returns:
        Bokeh HTML string with three synchronized tracks

    Example:
        >>> import squiggy
        >>> squiggy.load_pod5('data.pod5')
        >>> squiggy.load_bam('alignments.bam')
        >>> html = squiggy.plot_aggregate('chr1', max_reads=50)
        >>> # Extension displays this automatically

    Raises:
        ValueError: If POD5 or BAM files not loaded
    """
    from .io import _current_bam_path, _current_pod5_path, _current_pod5_reader
    from .utils import (
        calculate_aggregate_signal,
        calculate_base_pileup,
        calculate_quality_by_position,
        extract_reads_for_reference,
    )

    # Validate state
    if _current_pod5_reader is None:
        raise ValueError("No POD5 file loaded. Call load_pod5() first.")
    if _current_bam_path is None:
        raise ValueError(
            "No BAM file loaded. Aggregate plots require alignments. Call load_bam() first."
        )

    # Parse parameters
    norm_method = NormalizationMethod[normalization.upper()]
    theme_enum = Theme[theme.upper()]

    # Extract reads for this reference (expects file paths, not reader objects)
    reads_data = extract_reads_for_reference(
        pod5_file=_current_pod5_path,
        bam_file=_current_bam_path,
        reference_name=reference_name,
        max_reads=max_reads,
    )

    if not reads_data:
        raise ValueError(
            f"No reads found for reference '{reference_name}'. Check BAM file and reference name."
        )

    num_reads = len(reads_data)

    # Calculate aggregate statistics
    aggregate_stats = calculate_aggregate_signal(reads_data, norm_method)
    pileup_stats = calculate_base_pileup(
        reads_data, bam_file=_current_bam_path, reference_name=reference_name
    )
    quality_stats = calculate_quality_by_position(reads_data)

    # Generate plot
    html, _ = SquigglePlotter.plot_aggregate(
        aggregate_stats=aggregate_stats,
        pileup_stats=pileup_stats,
        quality_stats=quality_stats,
        reference_name=reference_name,
        num_reads=num_reads,
        normalization=norm_method,
        theme=theme_enum,
    )

    return html


__all__ = [
    # Version
    "__version__",
    # Object-oriented API (NEW)
    "Pod5File",
    "Read",
    "BamFile",
    "figure_to_html",
    # Main functions (legacy API - for Positron extension)
    "load_pod5",
    "load_bam",
    "plot_read",
    "plot_reads",
    "plot_aggregate",
    "get_current_files",
    "get_read_ids",
    "get_bam_modification_info",
    "get_bam_event_alignment_status",
    "get_read_to_reference_mapping",
    "close_pod5",
    # Data structures
    "AlignedRead",
    "BaseAnnotation",
    # Constants
    "NormalizationMethod",
    "PlotMode",
    "Theme",
    "BASE_COLORS",
    "BASE_COLORS_DARK",
    # Functions
    "extract_alignment_from_bam",
    "normalize_signal",
    "get_bam_references",
    "get_reads_in_region",
    "get_reference_sequence_for_read",
    "parse_region",
    "reverse_complement",
    "downsample_signal",
    # Classes
    "SquigglePlotter",
]
