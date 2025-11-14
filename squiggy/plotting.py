"""
Plotting functions for Squiggy

This module contains all the high-level plotting functions for generating
Bokeh visualizations of nanopore signal data.
"""

import numpy as np

from .constants import (
    DEFAULT_DOWNSAMPLE,
    DEFAULT_MOTIF_WINDOW_DOWNSTREAM,
    DEFAULT_MOTIF_WINDOW_UPSTREAM,
    DEFAULT_POSITION_LABEL_INTERVAL,
    PlotMode,
)
from .io import get_read_by_id, squiggy_kernel
from .motif import search_motif
from .plot_factory import create_plot_strategy
from .utils import (
    _route_to_plots_pane,
    calculate_aggregate_signal,
    calculate_base_pileup,
    calculate_delta_stats,
    calculate_dwell_time_statistics,
    calculate_modification_statistics,
    calculate_quality_by_position,
    extract_reads_for_motif,
    extract_reads_for_reference,
    get_available_reads_for_reference,
    parse_plot_parameters,
)


def plot_read(
    read_id: str,
    mode: str = "SINGLE",
    normalization: str = "ZNORM",
    theme: str = "LIGHT",
    downsample: int = None,
    show_dwell_time: bool = False,
    show_labels: bool = True,
    position_label_interval: int = None,
    scale_dwell_time: bool = False,
    min_mod_probability: float = 0.5,
    enabled_mod_types: list = None,
    show_signal_points: bool = False,
    clip_x_to_alignment: bool = True,
    sample_name: str | None = None,
    coordinate_space: str = "signal",
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
        clip_x_to_alignment: If True, x-axis shows only aligned region (default True).
                             If False, x-axis extends to include soft-clipped regions.
        sample_name: (Multi-sample mode) Name of the sample to plot from. If provided,
                     plots from that specific sample instead of the global session.
        coordinate_space: X-axis coordinate system ('signal' or 'sequence').
                         'signal' uses sample indices, 'sequence' uses genomic positions (requires BAM).

    Returns:
        Bokeh HTML string

    Examples:
        >>> html = plot_read('read_001', mode='EVENTALIGN')
        >>> # Extension displays this automatically
        >>> # Or save to file:
        >>> with open('plot.html', 'w') as f:
        >>>     f.write(html)
    """

    # Determine which POD5 reader to use
    if sample_name:
        # Multi-sample mode: get reader from specific sample
        sample = squiggy_kernel.get_sample(sample_name)
        if not sample or sample._pod5_reader is None:
            raise ValueError(f"Sample '{sample_name}' not loaded or has no POD5 file.")
        reader = sample._pod5_reader
    else:
        # Single-file mode: use global reader
        reader = squiggy_kernel._reader
        if reader is None:
            raise ValueError("No POD5 file loaded. Call load_pod5() first.")

    # Apply defaults if not specified
    if downsample is None:
        downsample = DEFAULT_DOWNSAMPLE
    if position_label_interval is None:
        position_label_interval = DEFAULT_POSITION_LABEL_INTERVAL

    # Get read data (optimized with index if available)
    read_obj = get_read_by_id(read_id, sample_name=sample_name)

    if read_obj is None:
        raise ValueError(f"Read not found: {read_id}")

    # Parse parameters
    params = parse_plot_parameters(mode=mode, normalization=normalization, theme=theme)
    plot_mode = params["mode"]
    norm_method = params["normalization"]
    theme_enum = params["theme"]

    # Prepare data based on plot mode
    if plot_mode == PlotMode.SINGLE:
        # Single read mode
        data = {
            "signal": read_obj.signal,
            "read_id": read_id,
            "sample_rate": read_obj.run_info.sample_rate,
        }

        # If sequence coordinate space requested, get alignment data
        if coordinate_space == "sequence":
            if sample_name:
                sample = squiggy_kernel.get_sample(sample_name)
                bam_path = sample._bam_path if sample else None
            else:
                bam_path = squiggy_kernel._bam_path

            if bam_path is None:
                raise ValueError(
                    "Sequence coordinate space requires a BAM file. Call load_bam() first or use coordinate_space='signal'."
                )

            from .alignment import extract_alignment_from_bam

            aligned_read = extract_alignment_from_bam(bam_path, read_id)
            if aligned_read is None:
                raise ValueError(f"No alignment found for read {read_id} in BAM file.")

            data["aligned_read"] = aligned_read

        options = {
            "normalization": norm_method,
            "downsample": downsample,
            "show_signal_points": show_signal_points,
            "x_axis_mode": "dwell_time" if scale_dwell_time else "regular_time",
            "coordinate_space": coordinate_space,
        }

    elif plot_mode == PlotMode.EVENTALIGN:
        # Event-aligned mode: requires alignment
        if sample_name:
            sample = squiggy_kernel.get_sample(sample_name)
            bam_path = sample._bam_path if sample else None
            fasta_path = sample._fasta_path if sample else None
        else:
            bam_path = squiggy_kernel._bam_path
            fasta_path = squiggy_kernel._fasta_path

        if bam_path is None:
            raise ValueError(
                "EVENTALIGN mode requires a BAM file. Call load_bam() first."
            )

        from .alignment import extract_alignment_from_bam
        from .utils import get_reference_sequence_from_fasta

        aligned_read = extract_alignment_from_bam(bam_path, read_id)
        if aligned_read is None:
            raise ValueError(f"No alignment found for read {read_id} in BAM file.")

        # Fetch reference sequence (FASTA-first pattern)
        reference_sequence = ""
        if (
            hasattr(aligned_read, "reference_name")
            and hasattr(aligned_read, "reference_start")
            and hasattr(aligned_read, "reference_end")
        ):
            reference_sequence = get_reference_sequence_from_fasta(
                fasta_file=fasta_path,
                reference_name=aligned_read.reference_name,
                start=aligned_read.reference_start,
                end=aligned_read.reference_end,
                bam_file=bam_path,
                read_id=read_id,
            )

        data = {
            "reads": [(read_id, read_obj.signal, read_obj.run_info.sample_rate)],
            "aligned_reads": [aligned_read],
            "reference_sequence": reference_sequence,  # Add reference sequence
        }

        options = {
            "normalization": norm_method,
            "downsample": downsample,
            "show_dwell_time": scale_dwell_time,
            "show_labels": show_labels,
            "show_signal_points": show_signal_points,
            "position_label_interval": position_label_interval,
            "clip_x_to_alignment": clip_x_to_alignment,
        }

    else:
        raise ValueError(
            f"Plot mode {plot_mode} not supported for single read. Use SINGLE or EVENTALIGN."
        )

    # Create strategy and generate plot
    strategy = create_plot_strategy(plot_mode, theme_enum)
    html, fig = strategy.create_plot(data, options)

    # Route to Positron Plots pane if running in Positron
    _route_to_plots_pane(fig)

    return html


def plot_reads(
    read_ids: list,
    mode: str = "OVERLAY",
    normalization: str = "ZNORM",
    theme: str = "LIGHT",
    downsample: int = None,
    show_dwell_time: bool = False,
    show_labels: bool = True,
    scale_dwell_time: bool = False,
    min_mod_probability: float = 0.5,
    enabled_mod_types: list = None,
    show_signal_points: bool = False,
    sample_name: str | None = None,
    read_sample_map: dict[str, str] | None = None,
    read_colors: dict[str, str] | None = None,
    coordinate_space: str = "signal",
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
        sample_name: (Single-sample mode) Name of the sample to plot from. If provided,
                     plots from that specific sample instead of the global session.
        read_sample_map: (Multi-sample mode) Dict mapping read_id → sample_name.
                         If provided, reads are loaded from their respective samples.
                         Takes precedence over sample_name parameter.
        read_colors: (Multi-sample mode) Dict mapping read_id → color hex string.
                     If provided, each read uses its specified color instead of
                     the default color cycling. Useful for sample-based coloring.
        coordinate_space: Coordinate system for x-axis ('signal' or 'sequence').
                          'signal' uses raw sample points, 'sequence' uses BAM alignment positions.

    Returns:
        Bokeh HTML string

    Examples:
        >>> # Single sample
        >>> html = plot_reads(['read_001', 'read_002'], mode='OVERLAY')
        >>>
        >>> # Multi-sample with custom colors
        >>> read_map = {'read_001': 'sample_A', 'read_002': 'sample_B'}
        >>> colors = {'read_001': '#E69F00', 'read_002': '#56B4E9'}
        >>> html = plot_reads(['read_001', 'read_002'], mode='OVERLAY',
        ...                   read_sample_map=read_map, read_colors=colors)
    """

    if not read_ids:
        raise ValueError("No read IDs provided.")

    # Apply defaults if not specified
    if downsample is None:
        downsample = DEFAULT_DOWNSAMPLE

    # Parse parameters
    params = parse_plot_parameters(mode=mode, normalization=normalization, theme=theme)
    plot_mode = params["mode"]
    norm_method = params["normalization"]
    theme_enum = params["theme"]

    # Collect read data - use multi-sample fetching if read_sample_map provided
    if read_sample_map:
        # Multi-sample mode: fetch reads from different samples
        from .io import get_reads_batch_multi_sample

        read_objs = get_reads_batch_multi_sample(read_sample_map)
    elif sample_name:
        # Single-sample mode: fetch from specified sample
        from .io import get_reads_batch

        read_objs = get_reads_batch(read_ids, sample_name=sample_name)
    else:
        # Legacy mode: fetch from global session
        from .io import get_reads_batch

        if squiggy_kernel._reader is None:
            raise ValueError("No POD5 file loaded. Call load_pod5() first.")
        read_objs = get_reads_batch(read_ids, sample_name=None)

    # Verify all reads were found
    missing = set(read_ids) - set(read_objs.keys())
    if missing:
        if len(missing) == 1:
            raise ValueError(f"Read not found: {list(missing)[0]}")
        else:
            raise ValueError(f"Reads not found: {list(missing)}")

    # Build reads_data list in original order
    reads_data = [
        (read_id, read_objs[read_id].signal, read_objs[read_id].run_info.sample_rate)
        for read_id in read_ids
    ]

    # Prepare data and options based on mode
    if plot_mode in (PlotMode.OVERLAY, PlotMode.STACKED):
        data = {"reads": reads_data}

        # If using sequence space, we need BAM alignments
        if coordinate_space == "sequence":
            # Determine which BAM file(s) to use
            if read_sample_map:
                # Multi-sample mode: each read may come from a different BAM
                from .alignment import extract_alignment_from_bam

                aligned_reads = []
                for read_id in read_ids:
                    sample_name_for_read = read_sample_map[read_id]
                    sample = squiggy_kernel.get_sample(sample_name_for_read)
                    if not sample or not sample._bam_path:
                        raise ValueError(
                            f"Sequence space requires BAM files. Sample '{sample_name_for_read}' has no BAM file loaded."
                        )
                    aligned_read = extract_alignment_from_bam(sample._bam_path, read_id)
                    if aligned_read is None:
                        raise ValueError(
                            f"No alignment found for read {read_id} in sample '{sample_name_for_read}'."
                        )
                    aligned_reads.append(aligned_read)
            elif sample_name:
                # Single-sample mode: use sample's BAM file
                from .alignment import extract_alignment_from_bam

                sample = squiggy_kernel.get_sample(sample_name)
                if not sample or not sample._bam_path:
                    raise ValueError(
                        f"Sequence space requires a BAM file. Sample '{sample_name}' has no BAM file loaded."
                    )
                aligned_reads = []
                for read_id in read_ids:
                    aligned_read = extract_alignment_from_bam(sample._bam_path, read_id)
                    if aligned_read is None:
                        raise ValueError(
                            f"No alignment found for read {read_id} in BAM file."
                        )
                    aligned_reads.append(aligned_read)
            else:
                # Legacy mode: use global BAM file
                from .alignment import extract_alignment_from_bam

                if squiggy_kernel._bam_path is None:
                    raise ValueError(
                        "Sequence space requires a BAM file. Call load_bam() first."
                    )
                aligned_reads = []
                for read_id in read_ids:
                    aligned_read = extract_alignment_from_bam(
                        squiggy_kernel._bam_path, read_id
                    )
                    if aligned_read is None:
                        raise ValueError(
                            f"No alignment found for read {read_id} in BAM file."
                        )
                    aligned_reads.append(aligned_read)

            # Add aligned reads to data
            data["aligned_reads"] = aligned_reads

        options = {
            "normalization": norm_method,
            "downsample": downsample,
            "show_signal_points": show_signal_points,
            "coordinate_space": coordinate_space,
        }
        # Add read colors if provided (for multi-sample coloring)
        if read_colors:
            options["read_colors"] = read_colors

    elif plot_mode == PlotMode.EVENTALIGN:
        # Event-aligned mode for multiple reads
        # Determine which BAM file to use
        if sample_name:
            # Multi-sample mode: use sample's BAM file
            sample = squiggy_kernel.get_sample(sample_name)
            if not sample or not sample._bam_path:
                raise ValueError(
                    f"EVENTALIGN mode requires a BAM file. Sample '{sample_name}' has no BAM file loaded."
                )
            bam_path = sample._bam_path
        else:
            # Single-file mode: use global BAM file
            if squiggy_kernel._bam_path is None:
                raise ValueError(
                    "EVENTALIGN mode requires a BAM file. Call load_bam() first."
                )
            bam_path = squiggy_kernel._bam_path

        from .alignment import extract_alignment_from_bam

        aligned_reads = []
        for read_id in read_ids:
            aligned_read = extract_alignment_from_bam(bam_path, read_id)
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
    _route_to_plots_pane(fig)

    return html


def plot_aggregate(
    reference_name: str,
    max_reads: int = 100,
    normalization: str = "ZNORM",
    theme: str = "LIGHT",
    show_modifications: bool = True,
    mod_filter: dict | None = None,
    show_pileup: bool = True,
    show_dwell_time: bool = True,
    show_signal: bool = True,
    show_quality: bool = True,
    clip_x_to_alignment: bool = True,
    transform_coordinates: bool = True,
    sample_name: str | None = None,
) -> str:
    """
    Generate aggregate multi-read visualization for a reference sequence

    Creates up to five synchronized tracks:
    1. Modifications heatmap (optional, if BAM has MM/ML tags)
    2. Base pileup (IGV-style stacked bar chart)
    3. Dwell time per base (mean ± std dev)
    4. Aggregate signal (mean ± std dev across reads)
    5. Quality scores by position

    Args:
        reference_name: Name of reference sequence from BAM file
        max_reads: Maximum number of reads to sample for aggregation (default 100)
        normalization: Normalization method (NONE, ZNORM, MEDIAN, MAD)
        theme: Color theme (LIGHT, DARK)
        show_modifications: Show modifications heatmap panel (default True)
        mod_filter: Dictionary mapping modification codes to minimum probability thresholds
                   (e.g., {'m': 0.8, 'a': 0.7}). If None, all modifications shown.
        show_pileup: Show base pileup panel (default True)
        show_dwell_time: Show dwell time panel (default True)
        show_signal: Show signal panel (default True)
        show_quality: Show quality panel (default True)
        clip_x_to_alignment: If True, x-axis shows only aligned region (default True).
                             If False, x-axis extends to include soft-clipped regions.
        transform_coordinates: If True, transform to 1-based coordinates anchored to first
                               reference base (default True). If False, use raw genomic coordinates.
        sample_name: (Multi-sample mode) Name of the sample to plot from. If provided,
                     plots from that specific sample instead of the global session.

    Returns:
        Bokeh HTML string with synchronized tracks

    Examples:
        >>> import squiggy
        >>> squiggy.load_pod5('data.pod5')
        >>> squiggy.load_bam('alignments.bam')
        >>> html = squiggy.plot_aggregate('chr1', max_reads=50)
        >>> # Filter modifications by type and probability
        >>> html = squiggy.plot_aggregate('chr1', mod_filter={'m': 0.8, 'a': 0.7})
        >>> # Extension displays this automatically

    Raises:
        ValueError: If POD5 or BAM files not loaded
    """

    # Determine which sample to use
    if sample_name:
        # Multi-sample mode: use sample-specific paths
        sample = squiggy_kernel.get_sample(sample_name)
        if not sample or sample._pod5_path is None:
            raise ValueError(f"Sample '{sample_name}' not loaded or has no POD5 file.")
        if not sample._bam_path:
            raise ValueError(
                f"Sample '{sample_name}' has no BAM file loaded. Aggregate plots require alignments."
            )
        pod5_path = sample._pod5_path
        bam_path = sample._bam_path
        fasta_path = sample._fasta_path
    else:
        # Single-file mode: use global session paths
        if squiggy_kernel._reader is None:
            raise ValueError("No POD5 file loaded. Call load_pod5() first.")
        if squiggy_kernel._bam_path is None:
            raise ValueError(
                "No BAM file loaded. Aggregate plots require alignments. Call load_bam() first."
            )
        pod5_path = squiggy_kernel._pod5_path
        bam_path = squiggy_kernel._bam_path
        fasta_path = squiggy_kernel._fasta_path

    # Parse parameters
    params = parse_plot_parameters(normalization=normalization, theme=theme)
    norm_method = params["normalization"]
    theme_enum = params["theme"]

    # Extract reads for this reference (expects file paths, not reader objects)
    reads_data = extract_reads_for_reference(
        pod5_file=pod5_path,
        bam_file=bam_path,
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
        reads_data,
        bam_file=bam_path,
        reference_name=reference_name,
        fasta_file=fasta_path,
    )
    quality_stats = calculate_quality_by_position(reads_data)
    modification_stats = calculate_modification_statistics(
        reads_data, mod_filter=mod_filter
    )
    dwell_stats = calculate_dwell_time_statistics(reads_data)

    # Convert to relative coordinates (1-based) for intuitive visualization

    # Store diagnostic info for plot title
    transformation_info = ""

    if transform_coordinates:
        # Anchor to first base of reference sequence (from pileup reference_bases)
        # This ensures x-axis position 1 = first base of the reference sequence
        reference_bases = pileup_stats.get("reference_bases", {})

        if reference_bases:
            # Use first position from reference_bases as anchor
            min_pos = min(reference_bases.keys())
        else:
            # Fallback: find minimum position across all tracks
            all_positions = []
            if "positions" in aggregate_stats and len(aggregate_stats["positions"]) > 0:
                all_positions.extend(list(aggregate_stats["positions"]))
            if "positions" in pileup_stats and len(pileup_stats["positions"]) > 0:
                all_positions.extend(list(pileup_stats["positions"]))
            if "positions" in quality_stats and len(quality_stats["positions"]) > 0:
                all_positions.extend(list(quality_stats["positions"]))

            if not all_positions:
                min_pos = None
            else:
                min_pos = int(np.min(all_positions))

        if min_pos is not None:
            offset = min_pos - 1  # Offset to make positions 1-based

            transformation_info = f"Ref-anchored (genomic pos {min_pos}→1)"

            # Transform aggregate_stats
            if "positions" in aggregate_stats and len(aggregate_stats["positions"]) > 0:
                old = list(aggregate_stats["positions"])
                aggregate_stats["positions"] = np.array([int(p) - offset for p in old])

            # Transform pileup_stats
            if "positions" in pileup_stats and len(pileup_stats["positions"]) > 0:
                old_positions = list(pileup_stats["positions"])
                new_positions = np.array([int(p) - offset for p in old_positions])
                pileup_stats["positions"] = new_positions

                # Remap counts dict - ensure consistent int types
                old_counts = pileup_stats["counts"]
                new_counts = {}
                for p in old_positions:
                    new_p = int(p) - offset
                    # Use int() to ensure Python int type for dictionary keys
                    new_counts[int(new_p)] = old_counts[int(p)]
                pileup_stats["counts"] = new_counts

                # Remap reference_bases dict
                if "reference_bases" in pileup_stats:
                    old_ref = pileup_stats["reference_bases"]
                    new_ref = {}
                    for p in old_positions:
                        old_p = int(p)
                        if old_p in old_ref:
                            new_p = int(old_p - offset)
                            new_ref[new_p] = old_ref[old_p]
                    pileup_stats["reference_bases"] = new_ref

            # Transform quality_stats
            if "positions" in quality_stats and len(quality_stats["positions"]) > 0:
                old = list(quality_stats["positions"])
                quality_stats["positions"] = np.array([int(p) - offset for p in old])

            # Transform modification_stats
            if modification_stats and modification_stats.get("mod_stats"):
                mod_stats = modification_stats["mod_stats"]
                new_mod_stats = {}
                for mod_code, pos_dict in mod_stats.items():
                    new_mod_stats[mod_code] = {}
                    for p, stats in pos_dict.items():
                        new_p = int(int(p) - offset)
                        new_mod_stats[mod_code][new_p] = stats
                modification_stats["mod_stats"] = new_mod_stats

                if "positions" in modification_stats:
                    old = modification_stats["positions"]
                    modification_stats["positions"] = [int(p) - offset for p in old]

            # Transform dwell_stats
            if (
                dwell_stats
                and "positions" in dwell_stats
                and len(dwell_stats["positions"]) > 0
            ):
                old = list(dwell_stats["positions"])
                dwell_stats["positions"] = np.array([int(p) - offset for p in old])

    # Prepare data for AggregatePlotStrategy
    data = {
        "aggregate_stats": aggregate_stats,
        "pileup_stats": pileup_stats,
        "quality_stats": quality_stats,
        "modification_stats": modification_stats,
        "dwell_stats": dwell_stats,
        "reference_name": reference_name,
        "num_reads": num_reads,
        "transformation_info": transformation_info,  # Diagnostic info
    }

    options = {
        "normalization": norm_method,
        "show_modifications": show_modifications,
        "show_pileup": show_pileup,
        "show_dwell_time": show_dwell_time,
        "show_signal": show_signal,
        "show_quality": show_quality,
        "clip_x_to_alignment": clip_x_to_alignment,
    }

    # Create strategy and generate plot
    strategy = create_plot_strategy(PlotMode.AGGREGATE, theme_enum)
    html, grid = strategy.create_plot(data, options)

    # Route to Positron Plots pane if running in Positron
    _route_to_plots_pane(grid)

    return html


def plot_motif_aggregate_all(
    fasta_file: str,
    motif: str,
    upstream: int = None,
    downstream: int = None,
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

    Examples:
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

    # Apply defaults if not specified
    if upstream is None:
        upstream = DEFAULT_MOTIF_WINDOW_UPSTREAM
    if downstream is None:
        downstream = DEFAULT_MOTIF_WINDOW_DOWNSTREAM

    # Validate state
    if squiggy_kernel._reader is None:
        raise ValueError("No POD5 file loaded. Call load_pod5() first.")
    if squiggy_kernel._bam_path is None:
        raise ValueError(
            "No BAM file loaded. Motif aggregate plots require alignments. "
            "Call load_bam() first."
        )

    # Parse parameters
    params = parse_plot_parameters(normalization=normalization, theme=theme)
    norm_method = params["normalization"]
    theme_enum = params["theme"]

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
                pod5_file=squiggy_kernel._pod5_path,
                bam_file=squiggy_kernel._bam_path,
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
    pileup_stats = calculate_base_pileup(
        all_aligned_reads,
        bam_file=None,  # Reads are in motif-relative coordinates
        reference_name=None,
        fasta_file=fasta_file,  # Use FASTA for accurate reference sequence
    )

    # Add motif sequence as reference bases for display
    # Center the motif in the coordinate system
    motif_length = len(motif)
    motif_center = 0  # Motif is centered at position 0 in motif-relative coordinates
    motif_start = motif_center - motif_length // 2

    # Create reference_bases dict mapping positions to motif letters
    reference_bases = {}
    for i, base in enumerate(motif.upper()):
        pos = motif_start + i
        reference_bases[pos] = base

    # Add to pileup_stats
    pileup_stats["reference_bases"] = reference_bases

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

    # Highlight motif positions (make them bold)
    motif_positions_set = set(range(motif_start, motif_start + motif_length))

    options = {"normalization": norm_method, "motif_positions": motif_positions_set}

    # Create strategy and generate plot
    strategy = create_plot_strategy(PlotMode.AGGREGATE, theme_enum)
    html, grid = strategy.create_plot(data, options)

    # Update x-axis title to reflect motif-relative coordinates
    # The aggregate plot strategy already handles the coordinate system,
    # but we can add a note about the window in the title

    # Route to Positron Plots pane if running in Positron
    _route_to_plots_pane(grid)

    return html


def plot_delta_comparison(
    sample_names: list[str],
    reference_name: str = "Default",
    normalization: str = "NONE",
    theme: str = "LIGHT",
    max_reads: int | None = None,
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
        max_reads: Maximum reads per sample to load (default: min of available reads, capped at 100)

    Returns:
        Bokeh HTML string with delta comparison visualization

    Examples:
        >>> import squiggy
        >>> squiggy.load_sample('v4.2', 'data_v4.2.pod5', 'align_v4.2.bam')
        >>> squiggy.load_sample('v5.0', 'data_v5.0.pod5', 'align_v5.0.bam')
        >>> html = squiggy.plot_delta_comparison(['v4.2', 'v5.0'])
        >>> # Extension displays this automatically

    Raises:
        ValueError: If fewer than 2 samples provided or samples not found
    """

    # Validate input
    if len(sample_names) < 2:
        raise ValueError("Delta comparison requires at least 2 samples")

    # Get samples
    samples = []
    for name in sample_names:
        sample = squiggy_kernel.get_sample(name)
        if sample is None:
            raise ValueError(f"Sample '{name}' not found")
        samples.append(sample)

    # For now, use first two samples for comparison (A vs B)
    sample_a = samples[0]
    sample_b = samples[1]

    # Validate both samples have POD5 and BAM loaded
    if sample_a._pod5_reader is None or sample_b._pod5_reader is None:
        raise ValueError("Both samples must have POD5 files loaded")

    if sample_a._bam_path is None or sample_b._bam_path is None:
        raise ValueError(
            "Both samples must have BAM files loaded for delta comparison. "
            "BAM files are required to align signals to reference positions."
        )

    # Parse parameters
    params = parse_plot_parameters(normalization=normalization, theme=theme)
    norm_method = params["normalization"]
    theme_enum = params["theme"]

    # Get first reference from sample A's BAM
    # (assumes both samples have the same reference genome)
    if not sample_a._bam_info or "references" not in sample_a._bam_info:
        raise ValueError("BAM file must be loaded with reference information")

    references = sample_a._bam_info["references"]
    if not references:
        raise ValueError("No references found in BAM file")

    reference_name = references[0]["name"]

    # Determine max_reads if not provided
    if max_reads is None:
        # Calculate available reads per sample and use minimum

        available_reads_per_sample = []
        for sample in [sample_a, sample_b]:
            try:
                available = get_available_reads_for_reference(
                    bam_file=sample._bam_path,
                    reference_name=reference_name,
                )
                available_reads_per_sample.append(available)
            except Exception:
                available_reads_per_sample.append(100)  # Fallback

        # Use minimum available, capped at 100
        max_reads = (
            min(available_reads_per_sample) if available_reads_per_sample else 100
        )
        max_reads = min(max_reads, 100)  # Cap at 100 for performance

    # Extract aligned reads from both samples using the proper utility function

    reads_a = extract_reads_for_reference(
        pod5_file=sample_a._pod5_path,
        bam_file=sample_a._bam_path,
        reference_name=reference_name,
        max_reads=max_reads,
        random_sample=True,
    )

    reads_b = extract_reads_for_reference(
        pod5_file=sample_b._pod5_path,
        bam_file=sample_b._bam_path,
        reference_name=reference_name,
        max_reads=max_reads,
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
    # Use positions from delta_stats (already truncated to match delta arrays)
    positions = delta_stats.get("positions")
    if positions is None:
        # Fallback: create position array matching delta length
        delta_signal = delta_stats.get("delta_mean_signal", [])
        positions = np.arange(len(delta_signal))

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
    _route_to_plots_pane(grid)

    return html


def plot_signal_overlay_comparison(
    sample_names: list[str],
    reference_name: str | None = None,
    normalization: str = "ZNORM",
    theme: str = "LIGHT",
    max_reads: int | None = None,
) -> str:
    """
    Generate signal overlay comparison plot for multiple samples

    Creates a visualization overlaying raw signals from 2+ samples, each with
    distinct color from Okabe-Ito palette. Includes:
    1. Signal Overlay Track: All sample signals overlaid with color per sample
    2. Coverage Track: Read count per position for each sample
    3. Reference Display: Nucleotide sequence annotations below signal track

    Args:
        sample_names: List of sample names to compare (minimum 2 required)
        reference_name: Optional reference name (auto-detected from first sample's BAM)
        normalization: Normalization method (NONE, ZNORM, MEDIAN, MAD) - default ZNORM
        theme: Color theme (LIGHT, DARK)
        max_reads: Maximum reads per sample to load (default: min of available reads, capped at 100)

    Returns:
        Bokeh HTML string with signal overlay comparison visualization

    Examples:
        >>> import squiggy
        >>> squiggy.load_sample('alanine', 'ala_subset.pod5', 'ala_subset.aln.bam')
        >>> squiggy.load_sample('arginine', 'arg_subset.pod5', 'arg_subset.aln.bam')
        >>> html = squiggy.plot_signal_overlay_comparison(
        ...     ['alanine', 'arginine'],
        ...     normalization='ZNORM'
        ... )
        >>> # Extension displays this automatically

    Raises:
        ValueError: If fewer than 2 samples provided, samples not found, or missing BAM files
    """

    # Validate input
    if len(sample_names) < 2:
        raise ValueError(
            f"Signal overlay comparison requires at least 2 samples, got {len(sample_names)}"
        )

    # Get samples
    samples = []
    for name in sample_names:
        sample = squiggy_kernel.get_sample(name)
        if sample is None:
            raise ValueError(f"Sample '{name}' not found")
        samples.append(sample)

    # Validate all samples have POD5 and BAM loaded
    for sample in samples:
        if sample._pod5_reader is None:
            raise ValueError(f"Sample '{sample.name}' must have POD5 file loaded")
        if sample._bam_path is None:
            raise ValueError(
                f"Sample '{sample.name}' must have BAM file loaded. "
                "BAM files are required to align signals to reference positions."
            )

    # Parse parameters
    params = parse_plot_parameters(normalization=normalization, theme=theme)
    norm_method = params["normalization"]
    theme_enum = params["theme"]

    # Determine reference name
    if reference_name is None:
        # Get first reference from first sample's BAM
        if not samples[0]._bam_info or "references" not in samples[0]._bam_info:
            raise ValueError("BAM file must contain reference information")

        references = samples[0]._bam_info["references"]
        if not references:
            raise ValueError("No references found in BAM file")

        reference_name = references[0]["name"]

    # Validate all samples have same reference (Issue #121)
    samples_missing_reference = []
    for sample in samples:
        if not sample._bam_info or "references" not in sample._bam_info:
            samples_missing_reference.append((sample.name, "no BAM metadata"))
            continue

        refs = sample._bam_info["references"]
        ref_names = [r["name"] for r in refs]

        if reference_name not in ref_names:
            available_refs = ", ".join(ref_names[:3])
            if len(ref_names) > 3:
                available_refs += f", ... ({len(ref_names)} total)"
            samples_missing_reference.append(
                (sample.name, available_refs or "no references")
            )

    # If any samples are missing the reference, raise a clear error
    if samples_missing_reference:
        error_lines = [
            f"Cannot plot signal overlay comparison: Some samples do not have reads aligned to reference '{reference_name}':",
            "",
        ]
        for sample_name, available in samples_missing_reference:
            error_lines.append(f"  • Sample '{sample_name}': has {available}")

        error_lines.extend(
            [
                "",
                "Suggestion: Check which references each sample has in the Samples panel,",
                "or ensure all samples are aligned to the same reference genome before loading.",
            ]
        )
        raise ValueError("\n".join(error_lines))

    # Determine max_reads if not provided
    if max_reads is None:
        # Calculate available reads per sample and use minimum

        available_reads_per_sample = []
        for sample in samples:
            try:
                available = get_available_reads_for_reference(
                    bam_file=sample._bam_path,
                    reference_name=reference_name,
                )
                available_reads_per_sample.append(available)
            except Exception:
                available_reads_per_sample.append(100)  # Fallback

        # Use minimum available, capped at 100
        max_reads = (
            min(available_reads_per_sample) if available_reads_per_sample else 100
        )
        max_reads = min(max_reads, 100)  # Cap at 100 for performance

    # Extract aligned reads for each sample
    plot_data = []
    coverage_data = {}

    for sample in samples:
        reads = extract_reads_for_reference(
            pod5_file=sample._pod5_path,
            bam_file=sample._bam_path,
            reference_name=reference_name,
            max_reads=max_reads,
            random_sample=True,
        )

        if not reads:
            raise ValueError(
                f"No reads found for sample '{sample.name}' on reference '{reference_name}'"
            )

        # Calculate aggregate signal and coverage for this sample

        agg_stats = calculate_aggregate_signal(reads, norm_method)

        plot_data.append(
            {
                "name": sample.name,
                "positions": agg_stats.get(
                    "positions", np.arange(len(agg_stats.get("mean_signal", [])))
                ),
                "signal": agg_stats.get("mean_signal", np.array([])),
            }
        )

        coverage_data[sample.name] = agg_stats.get(
            "coverage", [1] * len(agg_stats.get("mean_signal", []))
        )

    # Get reference sequence (FASTA-first pattern)
    reference_sequence = ""
    if plot_data:
        # Determine genomic region from plot data
        all_positions = []
        for data in plot_data:
            positions = data.get("positions", [])
            if len(positions) > 0:
                all_positions.extend(positions)

        if all_positions:
            min_pos = int(min(all_positions))
            max_pos = int(max(all_positions))

            # Try FASTA first (most accurate and complete)
            first_sample = samples[0]
            if first_sample._fasta_path:
                try:
                    import pysam

                    fasta = pysam.FastaFile(first_sample._fasta_path)
                    reference_sequence = fasta.fetch(
                        reference_name, min_pos, max_pos + 1
                    )
                    fasta.close()
                except Exception:
                    # FASTA fetch failed, will fall back to BAM
                    pass

            # Fallback to BAM reconstruction if FASTA unavailable
            if not reference_sequence:
                try:
                    reads = extract_reads_for_reference(
                        pod5_file=first_sample._pod5_path,
                        bam_file=first_sample._bam_path,
                        reference_name=reference_name,
                        max_reads=1,
                    )
                    if reads:
                        reference_sequence = (
                            reads[0].get("reference_sequence", "") or ""
                        )
                except Exception:
                    # If we can't get reference sequence, continue without it
                    pass

    # Prepare data for plot strategy
    data = {
        "samples": plot_data,
        "reference_sequence": reference_sequence,
        "coverage": coverage_data,
    }

    options = {"normalization": norm_method}

    # Create strategy and generate plot
    strategy = create_plot_strategy(PlotMode.SIGNAL_OVERLAY_COMPARISON, theme_enum)
    html, grid = strategy.create_plot(data, options)

    # Route to Positron Plots pane if running in Positron
    _route_to_plots_pane(grid)

    return html


def plot_aggregate_comparison(
    sample_names: list[str],
    reference_name: str,
    metrics: list[str] | None = None,
    max_reads: int | None = None,
    normalization: str = "ZNORM",
    theme: str = "LIGHT",
    sample_colors: dict[str, str] | None = None,
) -> str:
    """
    Generate aggregate comparison plot for multiple samples

    Creates a visualization comparing aggregate statistics (signal, dwell time,
    quality) from 2+ samples overlaid on the same axes. Each sample is color-coded
    for easy comparison. Includes:
    1. Signal Statistics Track: Mean signal ± std for each sample
    2. Dwell Time Statistics Track: Mean dwell time ± std for each sample (optional)
    3. Quality Statistics Track: Mean quality ± std for each sample (optional)
    4. Coverage Track: Read count per position for each sample

    Args:
        sample_names: List of sample names to compare (minimum 2 required)
        reference_name: Name of reference sequence from BAM files
        metrics: List of metrics to display: 'signal', 'dwell_time', 'quality'.
                 If None, displays all available metrics (default)
        max_reads: Maximum reads per sample to load (default: min of available reads, capped at 100)
        normalization: Normalization method (NONE, ZNORM, MEDIAN, MAD) - default ZNORM
        theme: Color theme (LIGHT, DARK)
        sample_colors: Optional dict mapping sample names to hex colors.
                       If None, uses Okabe-Ito palette (default)

    Returns:
        Bokeh HTML string with aggregate comparison visualization

    Example:
        >>> import squiggy
        >>> squiggy.load_sample('control', 'control.pod5', 'control.bam')
        >>> squiggy.load_sample('treatment', 'treatment.pod5', 'treatment.bam')
        >>> html = squiggy.plot_aggregate_comparison(
        ...     ['control', 'treatment'],
        ...     reference_name='chr1',
        ...     metrics=['signal', 'dwell_time', 'quality']
        ... )
        >>> # Extension displays this automatically

    Raises:
        ValueError: If fewer than 2 samples provided or samples not found
    """
    from .constants import MULTI_READ_COLORS, NormalizationMethod, Theme

    # Validate input
    if len(sample_names) < 2:
        raise ValueError("Aggregate comparison requires at least 2 samples")

    # Get samples
    samples = []
    for name in sample_names:
        sample = squiggy_kernel.get_sample(name)
        if sample is None:
            raise ValueError(f"Sample '{name}' not found")
        samples.append(sample)

    # Validate all samples have POD5 and BAM loaded
    for sample in samples:
        if sample._pod5_reader is None:
            raise ValueError(f"Sample '{sample.name}' must have POD5 file loaded")
        if sample._bam_path is None:
            raise ValueError(
                f"Sample '{sample.name}' must have BAM file loaded for aggregate comparison. "
                "BAM files are required to align signals to reference positions."
            )

    # Parse parameters
    norm_method = NormalizationMethod[normalization.upper()]
    theme_enum = Theme[theme.upper()]

    # Determine which metrics to display
    if metrics is None:
        metrics = ["signal", "dwell_time", "quality"]

    # Validate that all samples have the requested reference (Issue #121)
    samples_missing_reference = []
    for sample in samples:
        if sample._bam_info is None or "references" not in sample._bam_info:
            samples_missing_reference.append((sample.name, "no BAM metadata"))
            continue

        # Check if reference exists in this sample's BAM
        ref_names = [ref["name"] for ref in sample._bam_info.get("references", [])]
        if reference_name not in ref_names:
            available_refs = ", ".join(ref_names[:3])
            if len(ref_names) > 3:
                available_refs += f", ... ({len(ref_names)} total)"
            samples_missing_reference.append(
                (sample.name, available_refs or "no references")
            )

    # If any samples are missing the reference, raise a clear error
    if samples_missing_reference:
        error_lines = [
            f"Cannot plot aggregate comparison: Some samples do not have reads aligned to reference '{reference_name}':",
            "",
        ]
        for sample_name, available in samples_missing_reference:
            error_lines.append(f"  • Sample '{sample_name}': has {available}")

        error_lines.extend(
            [
                "",
                "Suggestion: Check which references each sample has in the Samples panel,",
                "or ensure all samples are aligned to the same reference genome before loading.",
            ]
        )
        raise ValueError("\n".join(error_lines))

    # Determine max_reads if not provided
    if max_reads is None:
        available_reads_per_sample = []
        for sample in samples:
            try:
                available = get_available_reads_for_reference(
                    bam_file=sample._bam_path,
                    reference_name=reference_name,
                )
                available_reads_per_sample.append(available)
            except Exception:
                available_reads_per_sample.append(100)  # Fallback

        # Use minimum available, capped at 100
        max_reads = (
            min(available_reads_per_sample) if available_reads_per_sample else 100
        )
        max_reads = min(max_reads, 100)  # Cap at 100 for performance

    # Extract and calculate statistics for each sample
    sample_data = []

    for i, sample in enumerate(samples):
        # Extract aligned reads for this sample
        reads = extract_reads_for_reference(
            pod5_file=sample._pod5_path,
            bam_file=sample._bam_path,
            reference_name=reference_name,
            max_reads=max_reads,
            random_sample=True,
        )

        if not reads:
            raise ValueError(
                f"No reads found for sample '{sample.name}' on reference '{reference_name}'"
            )

        # Calculate statistics based on requested metrics
        sample_stats = {"name": sample.name}

        # Assign color
        if sample_colors and sample.name in sample_colors:
            sample_stats["color"] = sample_colors[sample.name]
        else:
            sample_stats["color"] = MULTI_READ_COLORS[i % len(MULTI_READ_COLORS)]

        # Calculate signal statistics (always included if 'signal' in metrics)
        if "signal" in metrics:
            signal_stats = calculate_aggregate_signal(reads, norm_method)
            sample_stats["signal_stats"] = signal_stats
            sample_stats["coverage"] = {
                "positions": signal_stats.get(
                    "positions", np.arange(len(signal_stats.get("mean_signal", [])))
                ),
                "coverage": signal_stats.get(
                    "coverage", [1] * len(signal_stats.get("mean_signal", []))
                ),
            }

        # Calculate dwell time statistics (if requested)
        if "dwell_time" in metrics:
            dwell_stats = calculate_dwell_time_statistics(reads)
            sample_stats["dwell_stats"] = dwell_stats

        # Calculate quality statistics (if requested)
        if "quality" in metrics:
            quality_stats = calculate_quality_by_position(reads)
            sample_stats["quality_stats"] = quality_stats

        sample_data.append(sample_stats)

    # Prepare data for AggregateComparisonStrategy
    data = {
        "samples": sample_data,
        "reference_name": reference_name,
        "enabled_metrics": metrics,
    }

    options = {"normalization": norm_method}

    # Create strategy and generate plot
    strategy = create_plot_strategy(PlotMode.AGGREGATE_COMPARISON, theme_enum)
    html, grid = strategy.create_plot(data, options)

    # Route to Positron Plots pane if running in Positron
    _route_to_plots_pane(grid)

    return html
