"""
Plotting functions for Squiggy

All plotting functions accept OO API objects (Pod5File, BamFile, FastaFile, Sample)
as explicit parameters and return Bokeh figure objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .constants import (
    DEFAULT_DOWNSAMPLE,
    DEFAULT_MOTIF_WINDOW_DOWNSTREAM,
    DEFAULT_MOTIF_WINDOW_UPSTREAM,
    DEFAULT_POSITION_LABEL_INTERVAL,
    PlotMode,
)
from .motif import search_motif
from .plot_factory import create_plot_strategy
from .utils import (
    calculate_aggregate_signal,
    calculate_base_pileup,
    calculate_base_pileup_from_alignments,
    calculate_coverage_from_alignments,
    calculate_delta_stats,
    calculate_dwell_time_statistics,
    calculate_modification_statistics,
    calculate_modification_statistics_from_alignments,
    calculate_quality_by_position,
    calculate_quality_by_position_from_alignments,
    extract_alignments_for_reference,
    extract_reads_for_motif,
    extract_reads_for_reference,
    get_available_reads_for_reference,
    parse_plot_parameters,
)

if TYPE_CHECKING:
    from .api import BamFile, FastaFile, Pod5File, Sample


def plot_read(
    read_id: str,
    pod5_file: Pod5File,
    bam_file: BamFile | None = None,
    fasta_file: FastaFile | None = None,
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
    coordinate_space: str = "signal",
    trim_primers: bool = True,
):
    """
    Generate a Bokeh plot for a single read

    Args:
        read_id: Read ID to plot
        pod5_file: Pod5File object containing the read
        bam_file: BamFile object (required for EVENTALIGN and sequence coordinate space)
        fasta_file: FastaFile object (optional, for reference sequence display)
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
        coordinate_space: X-axis coordinate system ('signal' or 'sequence').
        trim_primers: If True (default), trim primer/adapter regions identified by PT tag.

    Returns:
        Bokeh figure object

    Examples:
        >>> pod5 = squiggy.Pod5File('data.pod5')
        >>> bam = squiggy.BamFile('alignments.bam')
        >>> fig = squiggy.plot_read(pod5.read_ids[0], pod5_file=pod5, bam_file=bam)
        >>> show(fig)
    """

    # Apply defaults if not specified
    if downsample is None:
        downsample = DEFAULT_DOWNSAMPLE
    if position_label_interval is None:
        position_label_interval = DEFAULT_POSITION_LABEL_INTERVAL

    # Get read data from POD5
    read_obj = None
    try:
        for r in pod5_file._reader.reads(selection=[read_id]):
            read_obj = r
            break
    except RuntimeError:
        pass

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
            if bam_file is None:
                raise ValueError(
                    "Sequence coordinate space requires a bam_file parameter."
                )

            from .alignment import extract_alignment_from_bam

            aligned_read = extract_alignment_from_bam(bam_file.path, read_id)
            if aligned_read is None:
                raise ValueError(f"No alignment found for read {read_id} in BAM file.")

            # Apply primer trimming if requested
            if trim_primers and aligned_read.primer_regions:
                from .alignment import trim_primers as do_trim_primers

                aligned_read, trimmed_sig = do_trim_primers(
                    aligned_read, data["signal"]
                )
                data["signal"] = trimmed_sig

            data["aligned_read"] = aligned_read

        options = {
            "normalization": norm_method,
            "downsample": downsample,
            "show_signal_points": show_signal_points,
            "x_axis_mode": "dwell_time" if scale_dwell_time else "regular_time",
            "coordinate_space": coordinate_space,
            "base_offset": 1,
        }

    elif plot_mode == PlotMode.EVENTALIGN:
        # Event-aligned mode: requires alignment
        if bam_file is None:
            raise ValueError("EVENTALIGN mode requires a bam_file parameter.")

        bam_path = bam_file.path
        fasta_path = fasta_file.path if fasta_file else None

        from .alignment import extract_alignment_from_bam
        from .utils import get_reference_sequence_from_fasta

        aligned_read = extract_alignment_from_bam(bam_path, read_id)
        if aligned_read is None:
            raise ValueError(f"No alignment found for read {read_id} in BAM file.")

        # Apply primer trimming if requested
        signal = read_obj.signal
        if trim_primers and aligned_read.primer_regions:
            from .alignment import trim_primers as do_trim_primers

            aligned_read, signal = do_trim_primers(aligned_read, signal)

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
            "reads": [(read_id, signal, read_obj.run_info.sample_rate)],
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
            "base_offset": 1,
        }

    else:
        raise ValueError(
            f"Plot mode {plot_mode} not supported for single read. Use SINGLE or EVENTALIGN."
        )

    # Create strategy and generate plot
    strategy = create_plot_strategy(plot_mode, theme_enum)
    fig = strategy.create_figure(data, options)

    return fig


def plot_reads(
    read_ids: list,
    pod5_file: Pod5File,
    bam_file: BamFile | None = None,
    fasta_file: FastaFile | None = None,
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
    read_colors: dict[str, str] | None = None,
    coordinate_space: str = "signal",
    trim_primers: bool = True,
):
    """
    Generate a Bokeh plot for multiple reads

    Args:
        read_ids: List of read IDs to plot
        pod5_file: Pod5File object containing the reads
        bam_file: BamFile object (required for EVENTALIGN, REFERENCE_OVERLAY, and sequence space)
        fasta_file: FastaFile object (optional, for reference sequence display)
        mode: Plot mode (OVERLAY, STACKED, EVENTALIGN, REFERENCE_OVERLAY)
        normalization: Normalization method (NONE, ZNORM, MEDIAN, MAD)
        theme: Color theme (LIGHT, DARK)
        downsample: Downsampling factor (1 = no downsampling, 10 = every 10th point)
        show_dwell_time: Color bases by dwell time (EVENTALIGN mode only)
        show_labels: Show base labels on plot (EVENTALIGN mode only)
        scale_dwell_time: Scale x-axis by cumulative dwell time (EVENTALIGN mode only)
        min_mod_probability: Minimum probability threshold for displaying modifications
        enabled_mod_types: List of modification type codes to display
        show_signal_points: Show individual signal points as circles
        read_colors: Dict mapping read_id → color hex string (optional)
        coordinate_space: Coordinate system for x-axis ('signal' or 'sequence')
        trim_primers: If True (default), trim primer/adapter regions identified by PT tag.

    Returns:
        Bokeh figure object

    Examples:
        >>> pod5 = squiggy.Pod5File('data.pod5')
        >>> bam = squiggy.BamFile('alignments.bam')
        >>> fig = squiggy.plot_reads(pod5.read_ids[:3], pod5_file=pod5, bam_file=bam)
        >>> show(fig)
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

    # Fetch reads from POD5
    read_objs = {}
    for read in pod5_file._reader.reads(selection=read_ids, missing_ok=True):
        read_objs[str(read.read_id)] = read

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
            if bam_file is None:
                raise ValueError(
                    "Sequence coordinate space requires a bam_file parameter."
                )

            from .alignment import extract_alignment_from_bam

            aligned_reads = []
            for read_id in read_ids:
                aligned_read = extract_alignment_from_bam(bam_file.path, read_id)
                if aligned_read is None:
                    raise ValueError(
                        f"No alignment found for read {read_id} in BAM file."
                    )
                aligned_reads.append(aligned_read)

            # Apply primer trimming if requested
            if trim_primers:
                from .alignment import trim_primers as do_trim_primers

                trimmed_data = []
                trimmed_aligned = []
                for (rid, sig, sr), ar in zip(reads_data, aligned_reads, strict=False):
                    if ar.primer_regions:
                        ar, sig = do_trim_primers(ar, sig)
                    trimmed_data.append((rid, sig, sr))
                    trimmed_aligned.append(ar)
                reads_data = trimmed_data
                aligned_reads = trimmed_aligned
                data["reads"] = reads_data

            # Add aligned reads to data
            data["aligned_reads"] = aligned_reads

        options = {
            "normalization": norm_method,
            "downsample": downsample,
            "show_signal_points": show_signal_points,
            "coordinate_space": coordinate_space,
        }
        if read_colors:
            options["read_colors"] = read_colors

    elif plot_mode == PlotMode.EVENTALIGN:
        if bam_file is None:
            raise ValueError("EVENTALIGN mode requires a bam_file parameter.")

        from .alignment import extract_alignment_from_bam

        aligned_reads = []
        for read_id in read_ids:
            aligned_read = extract_alignment_from_bam(bam_file.path, read_id)
            if aligned_read is None:
                raise ValueError(f"No alignment found for read {read_id} in BAM file.")
            aligned_reads.append(aligned_read)

        # Apply primer trimming if requested
        if trim_primers:
            from .alignment import trim_primers as do_trim_primers

            trimmed_data = []
            trimmed_aligned = []
            for (rid, sig, sr), ar in zip(reads_data, aligned_reads, strict=False):
                if ar.primer_regions:
                    ar, sig = do_trim_primers(ar, sig)
                trimmed_data.append((rid, sig, sr))
                trimmed_aligned.append(ar)
            reads_data = trimmed_data
            aligned_reads = trimmed_aligned

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
            "base_offset": 1,
        }

    elif plot_mode == PlotMode.REFERENCE_OVERLAY:
        if bam_file is None:
            raise ValueError("REFERENCE_OVERLAY mode requires a bam_file parameter.")

        from .alignment import extract_alignment_from_bam

        aligned_reads = []
        for read_id in read_ids:
            aligned_read = extract_alignment_from_bam(bam_file.path, read_id)
            if aligned_read is None:
                raise ValueError(f"No alignment found for read {read_id} in BAM file.")
            aligned_reads.append(aligned_read)

        # Apply primer trimming if requested
        if trim_primers:
            from .alignment import trim_primers as do_trim_primers

            trimmed_data = []
            trimmed_aligned = []
            for (rid, sig, sr), ar in zip(reads_data, aligned_reads, strict=False):
                if ar.primer_regions:
                    ar, sig = do_trim_primers(ar, sig)
                trimmed_data.append((rid, sig, sr))
                trimmed_aligned.append(ar)
            reads_data = trimmed_data
            aligned_reads = trimmed_aligned

        data = {
            "reads": reads_data,
            "aligned_reads": aligned_reads,
        }

        options = {
            "normalization": norm_method,
            "downsample": downsample,
            "show_labels": show_labels,
            "show_signal_points": show_signal_points,
            "scale_x_by_dwell": scale_dwell_time,
        }
        if read_colors:
            options["read_colors"] = read_colors

    else:
        raise ValueError(
            f"Plot mode {plot_mode} not supported for multiple reads. "
            f"Use OVERLAY, STACKED, EVENTALIGN, or REFERENCE_OVERLAY."
        )

    # Create strategy and generate plot
    strategy = create_plot_strategy(plot_mode, theme_enum)
    fig = strategy.create_figure(data, options)

    return fig


def plot_aggregate(
    reference_name: str,
    pod5_file: Pod5File,
    bam_file: BamFile,
    fasta_file: FastaFile | None = None,
    max_reads: int = 100,
    normalization: str = "ZNORM",
    theme: str = "LIGHT",
    show_modifications: bool = True,
    mod_filter: dict | None = None,
    min_mod_frequency: float = 0.0,
    min_modified_reads: int = 1,
    show_pileup: bool = True,
    show_dwell_time: bool = True,
    show_signal: bool = True,
    show_quality: bool = True,
    show_coverage: bool = True,
    clip_x_to_alignment: bool = True,
    transform_coordinates: bool = True,
    rna_mode: bool = False,
    trim_primers: bool = True,
    primer_5p: str | None = None,
    adapter_3p: str | None = None,
):
    """
    Generate aggregate multi-read visualization for a reference sequence

    Creates up to six synchronized tracks:
    1. Modifications heatmap (optional, if BAM has MM/ML tags)
    2. Base pileup (IGV-style stacked bar chart)
    3. Aggregate signal (mean ± std dev across reads)
    4. Quality scores by position
    5. Coverage depth (reads per position)
    6. Dwell time per base (mean ± std dev)

    Args:
        reference_name: Name of reference sequence from BAM file
        pod5_file: Pod5File object containing signal data
        bam_file: BamFile object containing alignments
        fasta_file: FastaFile object (optional, for reference sequence display)
        max_reads: Maximum number of reads to sample for aggregation (default 100)
        normalization: Normalization method (NONE, ZNORM, MEDIAN, MAD)
        theme: Color theme (LIGHT, DARK)
        show_modifications: Show modifications heatmap panel (default True)
        mod_filter: Dict mapping modification codes to min probability thresholds
        min_mod_frequency: Min fraction of reads modified at a position (0.0-1.0)
        min_modified_reads: Min number of reads modified at a position
        show_pileup: Show base pileup panel (default True)
        show_dwell_time: Show dwell time panel (default True)
        show_signal: Show signal panel (default True)
        show_quality: Show quality panel (default True)
        show_coverage: Show coverage depth panel (default True)
        clip_x_to_alignment: If True, x-axis shows only aligned region
        transform_coordinates: If True, transform to 1-based coordinates
        rna_mode: If True, use RNA-specific display
        trim_primers: If True, trim primer/adapter regions identified by PT tag
        primer_5p: Optional 5' primer sequence override
        adapter_3p: Optional 3' adapter sequence override

    Returns:
        Bokeh figure object with synchronized tracks

    Examples:
        >>> pod5 = squiggy.Pod5File('data.pod5')
        >>> bam = squiggy.BamFile('alignments.bam')
        >>> fig = squiggy.plot_aggregate('chr1', pod5_file=pod5, bam_file=bam)
        >>> show(fig)
    """

    pod5_path = pod5_file.path
    bam_path = bam_file.path
    fasta_path = fasta_file.path if fasta_file else None

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
    total_reads = None  # Set when primer filtering reduces the count
    primer_trim_bounds = None  # (start_ref_pos, end_ref_pos) for x-axis clipping

    # Primer trimming: derive body bounds from FASTA reference and optionally
    # filter to reads with both adapters detected.
    if trim_primers:
        # Derive body bounds from FASTA primer/adapter sequences
        if fasta_path:
            from .alignment import find_body_bounds

            bounds_kwargs = {}
            if primer_5p is not None:
                bounds_kwargs["primer_5p"] = primer_5p
            if adapter_3p is not None:
                bounds_kwargs["adapter_3p"] = adapter_3p
            primer_trim_bounds = find_body_bounds(
                fasta_path, reference_name, **bounds_kwargs
            )

        # Filter to reads with both adapters if PT tags are available
        from .alignment import has_both_adapters

        reads_with_adapters = [
            rd for rd in reads_data if has_both_adapters(rd.get("primer_regions", []))
        ]
        if reads_with_adapters:
            total_reads = num_reads
            reads_data = reads_with_adapters
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
        reads_data,
        mod_filter=mod_filter,
        min_frequency=min_mod_frequency,
        min_modified_reads=min_modified_reads,
    )
    dwell_stats = calculate_dwell_time_statistics(reads_data)

    # Convert to relative coordinates (1-based) for intuitive visualization

    # Store diagnostic info for plot title
    transformation_info = ""

    if transform_coordinates:
        if primer_trim_bounds is not None:
            # When primer-trimmed, anchor to body start so position 1 = first body base
            min_pos = primer_trim_bounds[0]
        elif reference_bases := pileup_stats.get("reference_bases", {}):
            # Anchor to first base of reference sequence
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

            if primer_trim_bounds is not None:
                transformation_info = f"Body-anchored (genomic pos {min_pos}→1)"
            else:
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

            # Transform primer_trim_bounds
            if primer_trim_bounds is not None:
                primer_trim_bounds = (
                    primer_trim_bounds[0] - offset,
                    primer_trim_bounds[1] - offset,
                )

    # Prepare data for AggregatePlotStrategy
    data = {
        "aggregate_stats": aggregate_stats,
        "pileup_stats": pileup_stats,
        "quality_stats": quality_stats,
        "modification_stats": modification_stats,
        "dwell_stats": dwell_stats,
        "reference_name": reference_name,
        "num_reads": num_reads,
        "total_reads": total_reads,  # Non-None when primer filtering reduced count
        "primer_trim_bounds": primer_trim_bounds,  # (start, end) for x-axis clipping
        "transformation_info": transformation_info,  # Diagnostic info
    }

    options = {
        "normalization": norm_method,
        "show_modifications": show_modifications,
        "show_pileup": show_pileup,
        "show_dwell_time": show_dwell_time,
        "show_signal": show_signal,
        "show_quality": show_quality,
        "show_coverage": show_coverage,
        "clip_x_to_alignment": clip_x_to_alignment,
        "rna_mode": rna_mode,
    }

    # Create strategy and generate plot
    strategy = create_plot_strategy(PlotMode.AGGREGATE, theme_enum)
    grid = strategy.create_figure(data, options)

    return grid


def plot_pileup(
    reference_name: str,
    bam_file: BamFile,
    fasta_file: FastaFile | None = None,
    max_reads: int = 100,
    theme: str = "LIGHT",
    show_modifications: bool = True,
    mod_filter: dict | None = None,
    min_mod_frequency: float = 0.0,
    min_modified_reads: int = 1,
    show_pileup: bool = True,
    show_quality: bool = True,
    show_coverage: bool = True,
    clip_x_to_alignment: bool = True,
    transform_coordinates: bool = True,
    rna_mode: bool = False,
):
    """
    Generate pileup-only visualization (BAM only, no POD5 needed)

    Creates up to four synchronized tracks:
    1. Modifications heatmap (optional, if BAM has MM/ML tags)
    2. Base pileup (IGV-style stacked bar chart)
    3. Quality scores by position
    4. Coverage depth track

    Args:
        reference_name: Name of reference sequence from BAM file
        bam_file: BamFile object containing alignments
        fasta_file: FastaFile object (optional, for reference sequence display)
        max_reads: Maximum number of reads to sample (default 100)
        theme: Color theme (LIGHT, DARK)
        show_modifications: Show modifications heatmap panel
        mod_filter: Dict mapping modification codes to min probability thresholds
        min_mod_frequency: Min fraction of reads modified at a position
        min_modified_reads: Min number of reads modified at a position
        show_pileup: Show base pileup panel
        show_quality: Show quality panel
        show_coverage: Show coverage depth panel
        clip_x_to_alignment: If True, x-axis shows only aligned region
        transform_coordinates: If True, transform to 1-based coordinates
        rna_mode: If True, use RNA-specific display

    Returns:
        Bokeh figure object with synchronized tracks

    Examples:
        >>> bam = squiggy.BamFile('alignments.bam')
        >>> fig = squiggy.plot_pileup('chr1', bam_file=bam)
        >>> show(fig)
    """

    bam_path = bam_file.path
    fasta_path = fasta_file.path if fasta_file else None

    # Parse parameters
    params = parse_plot_parameters(theme=theme)
    theme_enum = params["theme"]

    # Extract alignment data (no mv tag required!)
    reads_data = extract_alignments_for_reference(
        bam_file=bam_path,
        reference_name=reference_name,
        max_reads=max_reads,
    )

    if not reads_data:
        raise ValueError(
            f"No reads found for reference '{reference_name}'. Check BAM file and reference name."
        )

    num_reads = len(reads_data)

    # Calculate statistics using pileup-only functions (no move tables required)
    pileup_stats = calculate_base_pileup_from_alignments(
        reads_data,
        bam_file=bam_path,
        reference_name=reference_name,
        fasta_file=fasta_path,
    )
    quality_stats = calculate_quality_by_position_from_alignments(reads_data)
    coverage_stats = calculate_coverage_from_alignments(reads_data)
    modification_stats = calculate_modification_statistics_from_alignments(
        reads_data,
        mod_filter=mod_filter,
        min_frequency=min_mod_frequency,
        min_modified_reads=min_modified_reads,
    )

    # Store diagnostic info for plot title
    transformation_info = ""

    if transform_coordinates:
        # Anchor to first base of reference sequence (from pileup reference_bases)
        reference_bases = pileup_stats.get("reference_bases", {})

        if reference_bases:
            min_pos = min(reference_bases.keys())
        else:
            # Fallback: find minimum position across all tracks
            all_positions = []
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

            # Transform pileup_stats
            if "positions" in pileup_stats and len(pileup_stats["positions"]) > 0:
                old_positions = list(pileup_stats["positions"])
                new_positions = np.array([int(p) - offset for p in old_positions])
                pileup_stats["positions"] = new_positions

                # Remap counts dict
                old_counts = pileup_stats["counts"]
                new_counts = {}
                for p in old_positions:
                    new_p = int(p) - offset
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

            # Transform coverage_stats
            if "positions" in coverage_stats and len(coverage_stats["positions"]) > 0:
                old = list(coverage_stats["positions"])
                coverage_stats["positions"] = np.array([int(p) - offset for p in old])

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

    # Build aggregate_stats from coverage (for x-range calculation in strategy)
    # We don't have signal data, but we need the structure for the strategy
    aggregate_stats = {
        "positions": coverage_stats.get("positions", np.array([])),
        "mean_signal": np.zeros(
            len(coverage_stats.get("positions", []))
        ),  # Placeholder
        "std_signal": np.zeros(len(coverage_stats.get("positions", []))),  # Placeholder
        "coverage": coverage_stats.get("coverage", np.array([])),
    }

    # Prepare data for AggregatePlotStrategy (reuse existing strategy)
    data = {
        "aggregate_stats": aggregate_stats,
        "pileup_stats": pileup_stats,
        "quality_stats": quality_stats,
        "modification_stats": modification_stats,
        "dwell_stats": None,  # No dwell time without move tables
        "reference_name": reference_name,
        "num_reads": num_reads,
        "transformation_info": transformation_info,
    }

    options = {
        "normalization": None,  # No signal normalization
        "show_modifications": show_modifications,
        "show_pileup": show_pileup,
        "show_dwell_time": False,  # No dwell time without move tables
        "show_signal": False,  # No signal without POD5
        "show_quality": show_quality,
        "show_coverage": show_coverage,
        "clip_x_to_alignment": clip_x_to_alignment,
        "rna_mode": rna_mode,
    }

    # Create strategy and generate plot (reuse AggregatePlotStrategy)
    strategy = create_plot_strategy(PlotMode.AGGREGATE, theme_enum)
    grid = strategy.create_figure(data, options)

    return grid


def plot_motif_aggregate_all(
    motif: str,
    pod5_file: Pod5File,
    bam_file: BamFile,
    fasta_file: FastaFile,
    upstream: int = None,
    downstream: int = None,
    max_reads_per_motif: int = 100,
    normalization: str = "ZNORM",
    theme: str = "LIGHT",
    strand: str = "both",
):
    """
    Generate aggregate multi-read visualization across ALL motif matches

    Creates a three-track plot showing aggregate statistics from reads aligned
    to ALL instances of the motif. The x-axis is centered on the motif position
    with configurable upstream/downstream windows.

    Args:
        motif: IUPAC nucleotide pattern (e.g., "DRACH", "YGCY")
        pod5_file: Pod5File object containing signal data
        bam_file: BamFile object containing alignments
        fasta_file: FastaFile object containing reference sequences
        upstream: Bases upstream of motif center (default=10)
        downstream: Bases downstream of motif center (default=10)
        max_reads_per_motif: Maximum reads per motif match (default=100)
        normalization: Normalization method (NONE, ZNORM, MEDIAN, MAD)
        theme: Color theme (LIGHT, DARK)
        strand: Which strand to search ('+', '-', or 'both')

    Returns:
        Bokeh figure object with aggregate tracks

    Examples:
        >>> pod5 = squiggy.Pod5File('data.pod5')
        >>> bam = squiggy.BamFile('alignments.bam')
        >>> fasta = squiggy.FastaFile('genome.fa')
        >>> fig = squiggy.plot_motif_aggregate_all(
        ...     'DRACH', pod5_file=pod5, bam_file=bam, fasta_file=fasta
        ... )
        >>> show(fig)
    """

    # Apply defaults if not specified
    if upstream is None:
        upstream = DEFAULT_MOTIF_WINDOW_UPSTREAM
    if downstream is None:
        downstream = DEFAULT_MOTIF_WINDOW_DOWNSTREAM

    # Parse parameters
    params = parse_plot_parameters(normalization=normalization, theme=theme)
    norm_method = params["normalization"]
    theme_enum = params["theme"]

    # Search for all motif matches
    matches = list(search_motif(fasta_file.path, motif, strand=strand))

    if not matches:
        raise ValueError(f"No matches found for motif '{motif}' in FASTA file")

    # Extract and align reads from all motif matches
    all_aligned_reads = []
    num_matches_with_reads = 0

    for match_index, _motif_match in enumerate(matches):
        try:
            # Extract reads for this motif match
            reads_data, _ = extract_reads_for_motif(
                pod5_file=pod5_file.path,
                bam_file=bam_file.path,
                fasta_file=fasta_file.path,
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
        fasta_file=fasta_file.path,  # Use FASTA for accurate reference sequence
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
    grid = strategy.create_figure(data, options)

    return grid


def plot_delta_comparison(
    samples: list[Sample],
    reference_name: str | None = None,
    normalization: str = "NONE",
    theme: str = "LIGHT",
    max_reads: int | None = None,
):
    """
    Generate delta comparison plot between two samples

    Shows:
    1. Delta Signal Track: Mean signal differences (B - A)
    2. Delta Stats Track: Coverage comparisons

    Args:
        samples: List of Sample objects (minimum 2 required)
        reference_name: Reference name (auto-detected from first sample's BAM if None)
        normalization: Normalization method (NONE, ZNORM, MEDIAN, MAD)
        theme: Color theme (LIGHT, DARK)
        max_reads: Maximum reads per sample (default: min available, capped at 100)

    Returns:
        Bokeh figure object

    Examples:
        >>> wt = squiggy.Sample('wt', 'wt.pod5', 'wt.bam')
        >>> trub = squiggy.Sample('trub', 'trub.pod5', 'trub.bam')
        >>> fig = squiggy.plot_delta_comparison([wt, trub])
        >>> show(fig)
    """

    if len(samples) < 2:
        raise ValueError("Delta comparison requires at least 2 samples")

    sample_a = samples[0]
    sample_b = samples[1]

    # Validate both samples have POD5 and BAM
    if sample_a.pod5._reader is None or sample_b.pod5._reader is None:
        raise ValueError("Both samples must have POD5 files loaded")

    if sample_a.bam is None or sample_b.bam is None:
        raise ValueError(
            "Both samples must have BAM files loaded for delta comparison."
        )

    # Parse parameters
    params = parse_plot_parameters(normalization=normalization, theme=theme)
    norm_method = params["normalization"]
    theme_enum = params["theme"]

    # Auto-detect reference name from first sample's BAM
    if reference_name is None:
        bam_info = sample_a.bam.info
        references = bam_info.get("references", [])
        if not references:
            raise ValueError("No references found in BAM file")
        reference_name = references[0]["name"]

    # Determine max_reads if not provided
    if max_reads is None:
        available_reads_per_sample = []
        for sample in [sample_a, sample_b]:
            try:
                available = get_available_reads_for_reference(
                    bam_file=sample.bam.path,
                    reference_name=reference_name,
                )
                available_reads_per_sample.append(available)
            except Exception:
                available_reads_per_sample.append(100)

        max_reads = (
            min(available_reads_per_sample) if available_reads_per_sample else 100
        )
        max_reads = min(max_reads, 100)

    reads_a = extract_reads_for_reference(
        pod5_file=sample_a.pod5.path,
        bam_file=sample_a.bam.path,
        reference_name=reference_name,
        max_reads=max_reads,
        random_sample=True,
    )

    reads_b = extract_reads_for_reference(
        pod5_file=sample_b.pod5.path,
        bam_file=sample_b.bam.path,
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
    grid = strategy.create_figure(data, options)

    return grid


def plot_signal_overlay_comparison(
    samples: list[Sample],
    reference_name: str | None = None,
    normalization: str = "ZNORM",
    theme: str = "LIGHT",
    max_reads: int | None = None,
):
    """
    Generate signal overlay comparison plot for multiple samples

    Creates a visualization overlaying raw signals from 2+ samples, each with
    distinct color from Okabe-Ito palette.

    Args:
        samples: List of Sample objects (minimum 2 required)
        reference_name: Reference name (auto-detected from first sample's BAM if None)
        normalization: Normalization method (NONE, ZNORM, MEDIAN, MAD)
        theme: Color theme (LIGHT, DARK)
        max_reads: Maximum reads per sample (default: min available, capped at 100)

    Returns:
        Bokeh figure object

    Examples:
        >>> wt = squiggy.Sample('wt', 'wt.pod5', 'wt.bam')
        >>> trub = squiggy.Sample('trub', 'trub.pod5', 'trub.bam')
        >>> fig = squiggy.plot_signal_overlay_comparison([wt, trub])
        >>> show(fig)
    """

    if len(samples) < 2:
        raise ValueError(
            f"Signal overlay comparison requires at least 2 samples, got {len(samples)}"
        )

    # Validate all samples have POD5 and BAM
    for sample in samples:
        if sample.pod5._reader is None:
            raise ValueError(f"Sample '{sample.name}' must have POD5 file loaded")
        if sample.bam is None:
            raise ValueError(f"Sample '{sample.name}' must have BAM file loaded.")

    # Parse parameters
    params = parse_plot_parameters(normalization=normalization, theme=theme)
    norm_method = params["normalization"]
    theme_enum = params["theme"]

    # Determine reference name
    if reference_name is None:
        bam_info = samples[0].bam.info
        references = bam_info.get("references", [])
        if not references:
            raise ValueError("No references found in BAM file")
        reference_name = references[0]["name"]

    # Validate all samples have same reference
    samples_missing_reference = []
    for sample in samples:
        bam_info = sample.bam.info
        refs = bam_info.get("references", [])
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
        available_reads_per_sample = []
        for sample in samples:
            try:
                available = get_available_reads_for_reference(
                    bam_file=sample.bam.path,
                    reference_name=reference_name,
                )
                available_reads_per_sample.append(available)
            except Exception:
                available_reads_per_sample.append(100)

        max_reads = (
            min(available_reads_per_sample) if available_reads_per_sample else 100
        )
        max_reads = min(max_reads, 100)

    # Extract aligned reads for each sample
    plot_data = []
    coverage_data = {}

    for sample in samples:
        reads = extract_reads_for_reference(
            pod5_file=sample.pod5.path,
            bam_file=sample.bam.path,
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
            if first_sample.fasta:
                try:
                    reference_sequence = first_sample.fasta.fetch(
                        reference_name, min_pos, max_pos + 1
                    )
                except Exception:
                    pass

            # Fallback to BAM reconstruction if FASTA unavailable
            if not reference_sequence:
                try:
                    reads = extract_reads_for_reference(
                        pod5_file=first_sample.pod5.path,
                        bam_file=first_sample.bam.path,
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
    grid = strategy.create_figure(data, options)

    return grid


def plot_aggregate_comparison(
    samples: list[Sample],
    reference_name: str,
    metrics: list[str] | None = None,
    max_reads: int | None = None,
    normalization: str = "ZNORM",
    theme: str = "LIGHT",
    sample_colors: dict[str, str] | None = None,
    trim_primers: bool = True,
):
    """
    Generate aggregate comparison plot for multiple samples

    Compares aggregate statistics (signal, dwell time, quality) from 2+ samples
    overlaid on the same axes.

    Args:
        samples: List of Sample objects (minimum 2 required)
        reference_name: Name of reference sequence from BAM files
        metrics: List of metrics: 'signal', 'dwell_time', 'quality' (default: all)
        max_reads: Maximum reads per sample (default: min available, capped at 100)
        normalization: Normalization method (NONE, ZNORM, MEDIAN, MAD)
        theme: Color theme (LIGHT, DARK)
        sample_colors: Dict mapping sample names to hex colors (optional)
        trim_primers: If True, trim primer/adapter regions

    Returns:
        Bokeh figure object

    Examples:
        >>> wt = squiggy.Sample('wt', 'wt.pod5', 'wt.bam', 'ref.fa')
        >>> trub = squiggy.Sample('trub', 'trub.pod5', 'trub.bam', 'ref.fa')
        >>> fig = squiggy.plot_aggregate_comparison(
        ...     [wt, trub], reference_name='chr1'
        ... )
        >>> show(fig)
    """
    from .constants import MULTI_READ_COLORS, NormalizationMethod, Theme

    if len(samples) < 2:
        raise ValueError("Aggregate comparison requires at least 2 samples")

    # Validate all samples have POD5 and BAM
    for sample in samples:
        if sample.pod5._reader is None:
            raise ValueError(f"Sample '{sample.name}' must have POD5 file loaded")
        if sample.bam is None:
            raise ValueError(f"Sample '{sample.name}' must have BAM file loaded.")

    # Parse parameters
    norm_method = NormalizationMethod[normalization.upper()]
    theme_enum = Theme[theme.upper()]

    if metrics is None:
        metrics = ["signal", "dwell_time", "quality"]

    # Validate all samples have the requested reference
    samples_missing_reference = []
    for sample in samples:
        bam_info = sample.bam.info
        ref_names = [ref["name"] for ref in bam_info.get("references", [])]
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
                    bam_file=sample.bam.path,
                    reference_name=reference_name,
                )
                available_reads_per_sample.append(available)
            except Exception:
                available_reads_per_sample.append(100)

        max_reads = (
            min(available_reads_per_sample) if available_reads_per_sample else 100
        )
        max_reads = min(max_reads, 100)

    # Compute body bounds from FASTA if primer trimming requested
    primer_trim_bounds = None
    if trim_primers:
        fasta = samples[0].fasta if samples else None
        if fasta:
            from .alignment import find_body_bounds

            primer_trim_bounds = find_body_bounds(fasta.path, reference_name)

    # Extract and calculate statistics for each sample
    sample_data = []

    for i, sample in enumerate(samples):
        reads = extract_reads_for_reference(
            pod5_file=sample.pod5.path,
            bam_file=sample.bam.path,
            reference_name=reference_name,
            max_reads=max_reads,
            random_sample=True,
        )

        if not reads:
            raise ValueError(
                f"No reads found for sample '{sample.name}' on reference '{reference_name}'"
            )

        # Apply primer trimming: filter reads and derive body bounds from FASTA
        if trim_primers:
            from .alignment import has_both_adapters

            reads_with_adapters = [
                rd for rd in reads if has_both_adapters(rd.get("primer_regions", []))
            ]
            if reads_with_adapters:
                reads = reads_with_adapters

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
        "primer_trim_bounds": primer_trim_bounds,
    }

    options = {"normalization": norm_method}

    # Create strategy and generate plot
    strategy = create_plot_strategy(PlotMode.AGGREGATE_COMPARISON, theme_enum)
    grid = strategy.create_figure(data, options)

    return grid
