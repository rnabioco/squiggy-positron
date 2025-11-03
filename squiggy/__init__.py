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

__version__ = "0.1.7"

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
    Sample,
    SquiggySession,
    close_all_samples,
    close_bam,
    close_fasta,
    close_pod5,
    compare_samples,
    get_bam_event_alignment_status,
    get_bam_modification_info,
    get_common_reads,
    get_current_files,
    get_read_ids,
    get_read_to_reference_mapping,
    get_sample,
    get_unique_reads,
    list_samples,
    load_bam,
    load_fasta,
    load_pod5,
    load_sample,
    remove_sample,
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

# Legacy SquigglePlotter removed - use plot_read() or Read.plot() instead
# Utility functions and data classes
from .utils import (
    ModelProvenance,
    downsample_signal,
    extract_model_provenance,
    get_bam_references,
    get_reads_in_region,
    get_reference_sequence_for_read,
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

    Examples:
        >>> html = plot_read('read_001', mode='EVENTALIGN')
        >>> # Extension displays this automatically
        >>> # Or save to file:
        >>> with open('plot.html', 'w') as f:
        >>>     f.write(html)
    """
    from .io import _squiggy_session
    from .plot_factory import create_plot_strategy

    if _squiggy_session.reader is None:
        raise ValueError("No POD5 file loaded. Call load_pod5() first.")

    # Get read data
    read_obj = None
    for read in _squiggy_session.reader.reads():
        if str(read.read_id) == read_id:
            read_obj = read
            break

    if read_obj is None:
        raise ValueError(f"Read not found: {read_id}")

    # Parse parameters
    plot_mode = PlotMode[mode.upper()]
    norm_method = NormalizationMethod[normalization.upper()]
    theme_enum = Theme[theme.upper()]

    # Prepare data based on plot mode
    if plot_mode == PlotMode.SINGLE:
        # Single read mode: no alignment needed
        data = {
            "signal": read_obj.signal,
            "read_id": read_id,
            "sample_rate": read_obj.run_info.sample_rate,
        }

        options = {
            "normalization": norm_method,
            "downsample": downsample,
            "show_signal_points": show_signal_points,
            "x_axis_mode": "dwell_time" if scale_dwell_time else "regular_time",
        }

    elif plot_mode == PlotMode.EVENTALIGN:
        # Event-aligned mode: requires alignment
        if _squiggy_session.bam_path is None:
            raise ValueError(
                "EVENTALIGN mode requires a BAM file. Call load_bam() first."
            )

        from .alignment import extract_alignment_from_bam

        aligned_read = extract_alignment_from_bam(_squiggy_session.bam_path, read_id)
        if aligned_read is None:
            raise ValueError(f"No alignment found for read {read_id} in BAM file.")

        data = {
            "reads": [(read_id, read_obj.signal, read_obj.run_info.sample_rate)],
            "aligned_reads": [aligned_read],
        }

        options = {
            "normalization": norm_method,
            "downsample": downsample,
            "show_dwell_time": scale_dwell_time,
            "show_labels": show_labels,
            "show_signal_points": show_signal_points,
            "position_label_interval": position_label_interval,
        }

    else:
        raise ValueError(
            f"Plot mode {plot_mode} not supported for single read. Use SINGLE or EVENTALIGN."
        )

    # Create strategy and generate plot
    strategy = create_plot_strategy(plot_mode, theme_enum)
    html, fig = strategy.create_plot(data, options)

    # Route to Positron Plots pane if running in Positron
    from .utils import _route_to_plots_pane

    _route_to_plots_pane(fig)

    return html


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
        mode: Plot mode (OVERLAY, STACKED, EVENTALIGN)
        normalization: Normalization method (NONE, ZNORM, MEDIAN, MAD)
        theme: Color theme (LIGHT, DARK)
        downsample: Downsampling factor (1 = no downsampling, 10 = every 10th point)
        show_dwell_time: Color bases by dwell time (EVENTALIGN mode only)
        show_labels: Show base labels on plot (EVENTALIGN mode only)
        scale_dwell_time: Scale x-axis by cumulative dwell time (EVENTALIGN mode only)
        min_mod_probability: Minimum probability threshold for displaying modifications
        enabled_mod_types: List of modification type codes to display
        show_signal_points: Show individual signal points as circles

    Returns:
        Bokeh HTML string

    Examples:
        >>> html = plot_reads(['read_001', 'read_002'], mode='OVERLAY')
        >>> html = plot_reads(['read_001', 'read_002'], mode='STACKED')
        >>> html = plot_reads(['read_001', 'read_002'], mode='EVENTALIGN')
    """
    from .io import _squiggy_session
    from .plot_factory import create_plot_strategy

    if _squiggy_session.reader is None:
        raise ValueError("No POD5 file loaded. Call load_pod5() first.")

    if not read_ids:
        raise ValueError("No read IDs provided.")

    # Parse parameters
    plot_mode = PlotMode[mode.upper()]
    norm_method = NormalizationMethod[normalization.upper()]
    theme_enum = Theme[theme.upper()]

    # Collect read data
    reads_data = []
    for read_id in read_ids:
        read_obj = None
        for read in _squiggy_session.reader.reads():
            if str(read.read_id) == read_id:
                read_obj = read
                break

        if read_obj is None:
            raise ValueError(f"Read not found: {read_id}")

        reads_data.append((read_id, read_obj.signal, read_obj.run_info.sample_rate))

    # Prepare data and options based on mode
    if plot_mode in (PlotMode.OVERLAY, PlotMode.STACKED):
        data = {"reads": reads_data}
        options = {
            "normalization": norm_method,
            "downsample": downsample,
            "show_signal_points": show_signal_points,
        }

    elif plot_mode == PlotMode.EVENTALIGN:
        # Event-aligned mode for multiple reads
        if _squiggy_session.bam_path is None:
            raise ValueError(
                "EVENTALIGN mode requires a BAM file. Call load_bam() first."
            )

        from .alignment import extract_alignment_from_bam

        aligned_reads = []
        for read_id in read_ids:
            aligned_read = extract_alignment_from_bam(
                _squiggy_session.bam_path, read_id
            )
            if aligned_read is None:
                raise ValueError(f"No alignment found for read {read_id} in BAM file.")
            aligned_reads.append(aligned_read)

        data = {
            "reads": reads_data,
            "aligned_reads": aligned_reads,
        }

        options = {
            "normalization": norm_method,
            "downsample": downsample,
            "show_dwell_time": scale_dwell_time,
            "show_labels": show_labels,
            "show_signal_points": show_signal_points,
        }

    else:
        raise ValueError(
            f"Plot mode {plot_mode} not supported for multiple reads. "
            f"Use OVERLAY, STACKED, or EVENTALIGN."
        )

    # Create strategy and generate plot
    strategy = create_plot_strategy(plot_mode, theme_enum)
    html, fig = strategy.create_plot(data, options)

    # Route to Positron Plots pane if running in Positron
    from .utils import _route_to_plots_pane

    _route_to_plots_pane(fig)

    return html


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

    Examples:
        >>> import squiggy
        >>> squiggy.load_pod5('data.pod5')
        >>> squiggy.load_bam('alignments.bam')
        >>> html = squiggy.plot_aggregate('chr1', max_reads=50)
        >>> # Extension displays this automatically

    Raises:
        ValueError: If POD5 or BAM files not loaded
    """
    from .io import _squiggy_session
    from .plot_factory import create_plot_strategy
    from .utils import (
        calculate_aggregate_signal,
        calculate_base_pileup,
        calculate_quality_by_position,
        extract_reads_for_reference,
    )

    # Validate state
    if _squiggy_session.reader is None:
        raise ValueError("No POD5 file loaded. Call load_pod5() first.")
    if _squiggy_session.bam_path is None:
        raise ValueError(
            "No BAM file loaded. Aggregate plots require alignments. Call load_bam() first."
        )

    # Parse parameters
    norm_method = NormalizationMethod[normalization.upper()]
    theme_enum = Theme[theme.upper()]

    # Extract reads for this reference (expects file paths, not reader objects)
    reads_data = extract_reads_for_reference(
        pod5_file=_squiggy_session.pod5_path,
        bam_file=_squiggy_session.bam_path,
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
        reads_data, bam_file=_squiggy_session.bam_path, reference_name=reference_name
    )
    quality_stats = calculate_quality_by_position(reads_data)

    # Prepare data for AggregatePlotStrategy
    data = {
        "aggregate_stats": aggregate_stats,
        "pileup_stats": pileup_stats,
        "quality_stats": quality_stats,
        "reference_name": reference_name,
        "num_reads": num_reads,
    }

    options = {"normalization": norm_method}

    # Create strategy and generate plot
    strategy = create_plot_strategy(PlotMode.AGGREGATE, theme_enum)
    html, grid = strategy.create_plot(data, options)

    # Route to Positron Plots pane if running in Positron
    from .utils import _route_to_plots_pane

    _route_to_plots_pane(grid)

    return html


def plot_motif_aggregate(
    fasta_file: str,
    motif: str,
    match_index: int = 0,
    window: int = 50,
    max_reads: int = 100,
    normalization: str = "ZNORM",
    theme: str = "LIGHT",
) -> str:
    """
    Generate aggregate multi-read visualization centered on a motif match

    Creates a three-track plot showing:
    1. Aggregate signal (mean ± std dev across reads, centered on motif)
    2. Base pileup (IGV-style stacked bar chart)
    3. Quality scores by position

    The plot is centered on the motif position, with x-axis showing positions
    relative to the motif center (e.g., -50, 0, +50).

    Args:
        fasta_file: Path to indexed FASTA file (.fai required)
        motif: IUPAC nucleotide pattern (e.g., "DRACH", "YGCY")
        match_index: Which motif match to plot (0-based index, default=0)
        window: Number of bases around motif center to include (±window, default=50)
        max_reads: Maximum number of reads to sample for aggregation (default 100)
        normalization: Normalization method (NONE, ZNORM, MEDIAN, MAD)
        theme: Color theme (LIGHT, DARK)

    Returns:
        Bokeh HTML string with three synchronized tracks

    Example:
        >>> import squiggy
        >>> squiggy.load_pod5('data.pod5')
        >>> squiggy.load_bam('alignments.bam')
        >>> html = squiggy.plot_motif_aggregate(
        ...     fasta_file='genome.fa',
        ...     motif='DRACH',
        ...     match_index=0,
        ...     window=50
        ... )
        >>> # Extension displays this automatically

    Raises:
        ValueError: If POD5/BAM not loaded, no motif matches found,
                    or invalid match_index
    """
    from .io import _squiggy_session
    from .plot_factory import create_plot_strategy
    from .utils import (
        align_reads_to_motif_center,
        calculate_aggregate_signal,
        calculate_base_pileup,
        calculate_quality_by_position,
        extract_reads_for_motif,
    )

    # Validate state
    if _squiggy_session.reader is None:
        raise ValueError("No POD5 file loaded. Call load_pod5() first.")
    if _squiggy_session.bam_path is None:
        raise ValueError(
            "No BAM file loaded. Motif aggregate plots require alignments. "
            "Call load_bam() first."
        )

    # Parse parameters
    norm_method = NormalizationMethod[normalization.upper()]
    theme_enum = Theme[theme.upper()]

    # Extract reads overlapping this motif match
    reads_data, motif_match = extract_reads_for_motif(
        pod5_file=_squiggy_session.pod5_path,
        bam_file=_squiggy_session.bam_path,
        fasta_file=fasta_file,
        motif=motif,
        match_index=match_index,
        window=window,
        max_reads=max_reads,
    )

    if not reads_data:
        raise ValueError(
            f"No reads found overlapping motif match {match_index}. "
            f"Try a different match or increase window size."
        )

    num_reads = len(reads_data)

    # Calculate motif center position
    motif_center = motif_match.position + (motif_match.length // 2)

    # Align reads to motif center (adjust coordinates to be motif-relative)
    aligned_reads = align_reads_to_motif_center(reads_data, motif_center)

    # Calculate aggregate statistics
    aggregate_stats = calculate_aggregate_signal(aligned_reads, norm_method)

    # For pileup, we need the reference sequence around the motif
    # Get chromosome sequence from BAM or FASTA
    pileup_stats = calculate_base_pileup(
        aligned_reads,
        bam_file=_squiggy_session.bam_path,
        reference_name=motif_match.chrom,
    )

    quality_stats = calculate_quality_by_position(aligned_reads)

    # Generate plot with motif-specific title
    plot_title = (
        f"{motif} motif at {motif_match.chrom}:{motif_match.position + 1} "
        f"({motif_match.strand} strand, {num_reads} reads)"
    )

    # Prepare data for AggregatePlotStrategy
    data = {
        "aggregate_stats": aggregate_stats,
        "pileup_stats": pileup_stats,
        "quality_stats": quality_stats,
        "reference_name": plot_title,
        "num_reads": num_reads,
    }

    options = {"normalization": norm_method}

    # Create strategy and generate plot
    strategy = create_plot_strategy(PlotMode.AGGREGATE, theme_enum)
    html, grid = strategy.create_plot(data, options)

    # Route to Positron Plots pane if running in Positron
    from .utils import _route_to_plots_pane

    _route_to_plots_pane(grid)

    return html


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
    "plot_motif_aggregate",
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
    # Session management
    "SquiggySession",
    "Sample",
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
    # Classes
]
