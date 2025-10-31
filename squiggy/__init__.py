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

__version__ = "0.1.0"

# Core data structures and constants
from .alignment import AlignedRead, BaseAnnotation, extract_alignment_from_bam
from .constants import (
    NormalizationMethod,
    PlotMode,
    Theme,
    BASE_COLORS,
    BASE_COLORS_DARK,
)
from .normalization import normalize_signal
from .plotter import SquigglePlotter

# I/O functions
from .io import (
    load_pod5,
    load_bam,
    get_current_files,
    get_read_ids,
    close_pod5,
)

# Utility functions
from .utils import (
    get_bam_references,
    get_reads_in_region,
    get_reference_sequence_for_read,
    parse_region,
    reverse_complement,
    downsample_signal,
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
    downsample: bool = True,
    show_dwell_time: bool = False,
    position_label_interval: int = 100,
) -> str:
    """
    Generate a Bokeh HTML plot for a single read

    Args:
        read_id: Read ID to plot
        mode: Plot mode (SINGLE, EVENTALIGN)
        normalization: Normalization method (NONE, ZNORM, MEDIAN, MAD)
        theme: Color theme (LIGHT, DARK)
        downsample: Whether to downsample long signals
        show_dwell_time: Show dwell time on base annotations
        position_label_interval: Interval for position labels

    Returns:
        Bokeh HTML string

    Example:
        >>> html = plot_read('read_001', mode='EVENTALIGN')
        >>> # Extension displays this automatically
        >>> # Or save to file:
        >>> with open('plot.html', 'w') as f:
        >>>     f.write(html)
    """
    from .io import _current_pod5_reader, _current_bam_path

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

    # Extract signal and downsample if needed
    signal = read_obj.signal
    if downsample and len(signal) > 100000:
        signal = downsample_signal(signal, downsample_factor=len(signal) // 100000)

    # Get alignment if available
    aligned_read = None
    if _current_bam_path and mode.upper() == "EVENTALIGN":
        from .alignment import extract_alignment_from_bam
        aligned_read = extract_alignment_from_bam(_current_bam_path, read_id)

    # Create plotter
    plot_mode = PlotMode[mode.upper()]
    norm_method = NormalizationMethod[normalization.upper()]
    theme_enum = Theme[theme.upper()]

    plotter = SquigglePlotter(theme=theme_enum)

    # Extract sequence and seq_to_sig_map if available
    sequence = None
    seq_to_sig_map = None
    if aligned_read:
        sequence = aligned_read.sequence
        # Build seq_to_sig_map from base annotations
        if aligned_read.base_annotations:
            seq_to_sig_map = [ann.signal_start for ann in aligned_read.base_annotations]

    # Generate figure
    if plot_mode in (PlotMode.SINGLE, PlotMode.EVENTALIGN):
        if plot_mode == PlotMode.EVENTALIGN and aligned_read is None:
            raise ValueError("EVENTALIGN mode requires a BAM file. Call load_bam() first.")

        figure = plotter.plot_single_read(
            signal=signal,
            read_id=read_id,
            sample_rate=read_obj.run_info.sample_rate,
            sequence=sequence,
            seq_to_sig_map=seq_to_sig_map,
            normalization=norm_method,
            show_dwell_time=show_dwell_time,
            show_labels=True,
        )
    else:
        raise ValueError(f"Plot mode {plot_mode} not yet supported in extension. Use SINGLE or EVENTALIGN.")

    # Convert to HTML
    return plotter.figure_to_html(figure)


def plot_reads(
    read_ids: list,
    mode: str = "OVERLAY",
    normalization: str = "ZNORM",
    theme: str = "LIGHT",
    downsample: bool = True,
) -> str:
    """
    Generate a Bokeh HTML plot for multiple reads

    Args:
        read_ids: List of read IDs to plot
        mode: Plot mode (OVERLAY, STACKED)
        normalization: Normalization method
        theme: Color theme
        downsample: Whether to downsample

    Returns:
        Bokeh HTML string

    Example:
        >>> html = plot_reads(['read_001', 'read_002'], mode='OVERLAY')
    """
    from .io import _current_pod5_reader, _current_bam_path

    if _current_pod5_reader is None:
        raise ValueError("No POD5 file loaded. Call load_pod5() first.")

    # Extract read data
    read_data_list = []
    for read in _current_pod5_reader.reads():
        if str(read.read_id) in read_ids:
            read_data = extract_read_data(read, downsample, downsample_threshold=100000)
            read_data_list.append(read_data)

    if not read_data_list:
        raise ValueError(f"No reads found for IDs: {read_ids}")

    # Get alignments if available
    aligned_reads = {}
    if _current_bam_path:
        aligned_reads = get_aligned_reads_for_ids(_current_bam_path, read_ids)

    # Create plotter
    plot_mode = PlotMode[mode.upper()]
    norm_method = NormalizationMethod[normalization.upper()]
    theme_enum = Theme[theme.upper()]

    plotter = SquigglePlotter(theme=theme_enum)

    # Generate figure
    if plot_mode == PlotMode.OVERLAY:
        figures = [
            plotter.plot_single(rd, norm_method, aligned_read=aligned_reads.get(rd["read_id"]))
            for rd in read_data_list
        ]
        figure = plotter.plot_overlay(figures)
    elif plot_mode == PlotMode.STACKED:
        figure = plotter.plot_stacked(read_data_list, norm_method, aligned_reads_dict=aligned_reads)
    else:
        raise ValueError(f"Mode {mode} not supported for multiple reads")

    # Convert to HTML
    return plotter.figure_to_html(figure)


__all__ = [
    # Version
    "__version__",
    # Main functions
    "load_pod5",
    "load_bam",
    "plot_read",
    "plot_reads",
    "get_current_files",
    "get_read_ids",
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
    # Classes
    "SquigglePlotter",
]
