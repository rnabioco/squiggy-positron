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
    BASE_COLORS,
    BASE_COLORS_DARK,
    NormalizationMethod,
    PlotMode,
    Theme,
)

# I/O functions
from .io import (
    close_pod5,
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
    downsample: bool = True,
    show_dwell_time: bool = False,
    show_labels: bool = True,
    position_label_interval: int = 100,
    scale_dwell_time: bool = False,
    min_mod_probability: float = 0.5,
    enabled_mod_types: list = None,
) -> str:
    """
    Generate a Bokeh HTML plot for a single read

    Args:
        read_id: Read ID to plot
        mode: Plot mode (SINGLE, EVENTALIGN)
        normalization: Normalization method (NONE, ZNORM, MEDIAN, MAD)
        theme: Color theme (LIGHT, DARK)
        downsample: Whether to downsample long signals
        show_dwell_time: Color bases by dwell time (requires event-aligned mode)
        show_labels: Show base labels on plot (event-aligned mode)
        position_label_interval: Interval for position labels
        scale_dwell_time: Scale x-axis by cumulative dwell time instead of regular time
        min_mod_probability: Minimum probability threshold for displaying modifications (0-1)
        enabled_mod_types: List of modification type codes to display (None = all)

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

    # Extract signal and downsample if needed
    signal = read_obj.signal
    if downsample and len(signal) > 100000:
        signal = downsample_signal(signal, downsample_factor=len(signal) // 100000)

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

        html, figure = SquigglePlotter.plot_single_read(
            signal=signal,
            read_id=read_id,
            sample_rate=read_obj.run_info.sample_rate,
            sequence=sequence,
            seq_to_sig_map=seq_to_sig_map,
            normalization=norm_method,
            show_dwell_time=show_dwell_time,
            show_labels=show_labels,
            modifications=modifications,
            scale_dwell_time=scale_dwell_time,
            min_mod_probability=min_mod_probability,
            enabled_mod_types=enabled_mod_types,
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
    downsample: bool = True,
    show_dwell_time: bool = False,
    show_labels: bool = True,
    scale_dwell_time: bool = False,
    min_mod_probability: float = 0.5,
    enabled_mod_types: list = None,
) -> str:
    """
    Generate a Bokeh HTML plot for multiple reads

    Args:
        read_ids: List of read IDs to plot
        mode: Plot mode (OVERLAY, STACKED)
        normalization: Normalization method (NONE, ZNORM, MEDIAN, MAD)
        theme: Color theme (LIGHT, DARK)
        downsample: Whether to downsample long signals
        show_dwell_time: Color bases by dwell time
        show_labels: Show base labels on plot
        scale_dwell_time: Scale x-axis by cumulative dwell time
        min_mod_probability: Minimum probability threshold for displaying modifications
        enabled_mod_types: List of modification type codes to display

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
            )
            htmls.append(html)
        # Return first plot for now - full OVERLAY implementation TODO
        return htmls[0] if htmls else ""
    else:
        raise ValueError(
            f"Plot mode {mode} not yet fully implemented. Use plot_read() for single reads."
        )


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
    "get_bam_modification_info",
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
