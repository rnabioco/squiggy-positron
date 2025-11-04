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

__version__ = "0.1.8"

# Standard library
import numpy as np

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


def plot_motif_aggregate_all(
    fasta_file: str,
    motif: str,
    upstream: int = 10,
    downstream: int = 10,
    max_reads_per_motif: int = 100,
    normalization: str = "ZNORM",
    theme: str = "LIGHT",
    strand: str = "both",
) -> str:
    """
    Generate aggregate multi-read visualization across ALL motif matches

    Creates a three-track plot showing aggregate statistics from reads aligned
    to ALL instances of the motif in the genome. The x-axis is centered on the
    motif position with configurable upstream/downstream windows.

    This function combines reads from all motif matches into one aggregate view,
    providing a genome-wide perspective on signal patterns around the motif.

    Args:
        fasta_file: Path to indexed FASTA file (.fai required)
        motif: IUPAC nucleotide pattern (e.g., "DRACH", "YGCY")
        upstream: Number of bases upstream (5') of motif center (default=10)
        downstream: Number of bases downstream (3') of motif center (default=10)
        max_reads_per_motif: Maximum reads per motif match (default=100)
        normalization: Normalization method (NONE, ZNORM, MEDIAN, MAD)
        theme: Color theme (LIGHT, DARK)
        strand: Which strand to search ('+', '-', or 'both')

    Returns:
        Bokeh HTML string with three synchronized tracks showing aggregate
        statistics across all motif instances

    Example:
        >>> import squiggy
        >>> squiggy.load_pod5('data.pod5')
        >>> squiggy.load_bam('alignments.bam')
        >>> html = squiggy.plot_motif_aggregate_all(
        ...     fasta_file='genome.fa',
        ...     motif='DRACH',
        ...     upstream=20,
        ...     downstream=50
        ... )
        >>> # Extension displays this automatically

    Raises:
        ValueError: If POD5/BAM not loaded or no motif matches found
    """
    from .io import _squiggy_session
    from .motif import search_motif
    from .plot_factory import create_plot_strategy
    from .utils import (
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

    # Search for all motif matches
    matches = list(search_motif(fasta_file, motif, strand=strand))

    if not matches:
        raise ValueError(f"No matches found for motif '{motif}' in FASTA file")

    # Extract and align reads from all motif matches
    all_aligned_reads = []
    num_matches_with_reads = 0

    for match_index, _motif_match in enumerate(matches):
        try:
            # Extract reads for this motif match
            reads_data, _ = extract_reads_for_motif(
                pod5_file=_squiggy_session.pod5_path,
                bam_file=_squiggy_session.bam_path,
                fasta_file=fasta_file,
                motif=motif,
                match_index=match_index,
                upstream=upstream,
                downstream=downstream,
                max_reads=max_reads_per_motif,
            )

            if reads_data:
                # Reads are already clipped and in motif-relative coordinates from extract_reads_for_motif()
                all_aligned_reads.extend(reads_data)
                num_matches_with_reads += 1

        except Exception:
            # Skip motif matches that fail (e.g., no reads, edge of chromosome)
            continue

    if not all_aligned_reads:
        raise ValueError(
            f"No reads found overlapping any of {len(matches)} motif matches. "
            "Try a different motif or increase window size."
        )

    num_reads = len(all_aligned_reads)

    # Reads are already clipped to the window in extract_reads_for_motif()
    # Calculate aggregate statistics across all reads
    aggregate_stats = calculate_aggregate_signal(all_aligned_reads, norm_method)

    # Calculate base pileup across all reads
    # Don't pass reference_name because reads are in motif-relative coordinates,
    # not genomic coordinates - we can't extract reference sequence from BAM
    pileup_stats = calculate_base_pileup(
        all_aligned_reads,
        bam_file=None,  # Don't try to extract reference sequence
        reference_name=None,
    )

    quality_stats = calculate_quality_by_position(all_aligned_reads)

    # Generate plot with informative title
    plot_title = (
        f"{motif} motif aggregate ({num_matches_with_reads}/{len(matches)} matches, "
        f"{num_reads} reads, -{upstream}bp to +{downstream}bp)"
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

    # Update x-axis title to reflect motif-relative coordinates
    # The aggregate plot strategy already handles the coordinate system,
    # but we can add a note about the window in the title

    # Route to Positron Plots pane if running in Positron
    from .utils import _route_to_plots_pane

    _route_to_plots_pane(grid)

    return html


def plot_delta_comparison(
    sample_names: list[str],
    reference_name: str = "Default",
    normalization: str = "NONE",
    theme: str = "LIGHT",
) -> str:
    """
    Generate delta comparison plot between two or more samples

    Creates a visualization showing differences in aggregate statistics
    between samples. Shows:
    1. Delta Signal Track: Mean signal differences (B - A)
    2. Delta Stats Track: Coverage comparisons

    Args:
        sample_names: List of sample names to compare (minimum 2 required)
        reference_name: Optional reference name for plot title (default: "Default")
        normalization: Normalization method (NONE, ZNORM, MEDIAN, MAD)
        theme: Color theme (LIGHT, DARK)

    Returns:
        Bokeh HTML string with delta comparison visualization

    Example:
        >>> import squiggy
        >>> squiggy.load_sample('v4.2', 'data_v4.2.pod5', 'align_v4.2.bam')
        >>> squiggy.load_sample('v5.0', 'data_v5.0.pod5', 'align_v5.0.bam')
        >>> html = squiggy.plot_delta_comparison(['v4.2', 'v5.0'])
        >>> # Extension displays this automatically

    Raises:
        ValueError: If fewer than 2 samples provided or samples not found
    """
    from .io import _squiggy_session
    from .plot_factory import create_plot_strategy
    from .utils import calculate_aggregate_signal, calculate_delta_stats

    # Validate input
    if len(sample_names) < 2:
        raise ValueError("Delta comparison requires at least 2 samples")

    # Get samples
    samples = []
    for name in sample_names:
        sample = _squiggy_session.get_sample(name)
        if sample is None:
            raise ValueError(f"Sample '{name}' not found")
        samples.append(sample)

    # For now, use first two samples for comparison (A vs B)
    sample_a = samples[0]
    sample_b = samples[1]

    # Validate both samples have POD5 and BAM loaded
    if sample_a.pod5_reader is None or sample_b.pod5_reader is None:
        raise ValueError("Both samples must have POD5 files loaded")

    if sample_a.bam_path is None or sample_b.bam_path is None:
        raise ValueError(
            "Both samples must have BAM files loaded for delta comparison. "
            "BAM files are required to align signals to reference positions."
        )

    # Parse parameters
    norm_method = NormalizationMethod[normalization.upper()]
    theme_enum = Theme[theme.upper()]

    # Get first reference from sample A's BAM
    # (assumes both samples have the same reference genome)
    if not sample_a.bam_info or "references" not in sample_a.bam_info:
        raise ValueError("BAM file must be loaded with reference information")

    references = sample_a.bam_info["references"]
    if not references:
        raise ValueError("No references found in BAM file")

    reference_name = references[0]["name"]

    # Extract aligned reads from both samples using the proper utility function
    from .utils import extract_reads_for_reference

    reads_a = extract_reads_for_reference(
        pod5_file=sample_a.pod5_path,
        bam_file=sample_a.bam_path,
        reference_name=reference_name,
        max_reads=100,
        random_sample=True,
    )

    reads_b = extract_reads_for_reference(
        pod5_file=sample_b.pod5_path,
        bam_file=sample_b.bam_path,
        reference_name=reference_name,
        max_reads=100,
        random_sample=True,
    )

    if not reads_a:
        raise ValueError(f"No reads found for sample A on reference '{reference_name}'")
    if not reads_b:
        raise ValueError(f"No reads found for sample B on reference '{reference_name}'")

    # Calculate aggregate statistics for both samples
    stats_a = calculate_aggregate_signal(reads_a, norm_method)
    stats_b = calculate_aggregate_signal(reads_b, norm_method)

    # Calculate deltas
    delta_stats = calculate_delta_stats(stats_a, stats_b)

    # Prepare data for DeltaPlotStrategy
    positions = stats_a.get("positions", delta_stats.get("positions"))
    if positions is None:
        positions = np.arange(len(delta_stats.get("delta_mean_signal", [])))

    data = {
        "positions": positions,
        "delta_mean_signal": delta_stats.get("delta_mean_signal", np.array([])),
        "delta_std_signal": delta_stats.get("delta_std_signal", np.array([])),
        "sample_a_name": sample_a.name,
        "sample_b_name": sample_b.name,
        "sample_a_coverage": stats_a.get("coverage", [1] * len(positions)),
        "sample_b_coverage": stats_b.get("coverage", [1] * len(positions)),
    }

    options = {"normalization": norm_method}

    # Create strategy and generate plot
    strategy = create_plot_strategy(PlotMode.DELTA, theme_enum)
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
    "plot_motif_aggregate_all",
    "plot_delta_comparison",  # Phase 3 - NEW
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
