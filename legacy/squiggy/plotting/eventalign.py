"""Event-aligned plot mode with base annotations"""

import numpy as np
from bokeh.embed import file_html
from bokeh.layouts import column
from bokeh.models import BoxAnnotation, ColumnDataSource, HoverTool
from bokeh.plotting import figure
from bokeh.resources import CDN

from ..constants import (
    DEFAULT_MOD_OVERLAY_OPACITY,
    DEFAULT_POSITION_LABEL_INTERVAL,
    MODIFICATION_CODES,
    MODIFICATION_COLORS,
    NormalizationMethod,
    Theme,
)
from .base import (
    MULTI_READ_COLORS,
    add_hover_tool,
    add_signal_renderers,
    configure_legend,
    create_figure,
    create_signal_data_source,
    format_html_title,
    format_plot_title,
    get_base_colors,
    get_signal_line_color,
    normalize_signal,
    process_signal,
)
from .base_annotations import (
    add_base_labels_position_mode,
    add_base_type_patches,
    calculate_base_regions_position_mode,
)


def add_base_annotations_eventalign(
    p,
    reads_data: list[tuple[str, np.ndarray, int]],
    normalization: NormalizationMethod,
    aligned_reads: list,
    show_dwell_time: bool,
    show_labels: bool,
    position_label_interval: int = DEFAULT_POSITION_LABEL_INTERVAL,
    use_reference_positions: bool = False,
    theme: Theme = Theme.LIGHT,
):
    """Add base annotations for event-aligned plots"""
    if not reads_data or not aligned_reads:
        return None

    base_colors = get_base_colors(theme)
    first_aligned = aligned_reads[0]
    base_annotations = first_aligned.bases

    # Calculate signal range across all reads
    all_signals = [
        normalize_signal(signal, normalization) for _, signal, _ in reads_data
    ]

    if not all_signals:
        return None

    signal_min = min(np.min(s) for s in all_signals)
    signal_max = max(np.max(s) for s in all_signals)
    sample_rate = reads_data[0][2]
    signal_length = len(all_signals[0])

    # Calculate and add base patches (time-scaled or position-based)
    (base_regions,) = calculate_base_regions_position_mode(
        base_annotations,
        signal_min,
        signal_max,
        sample_rate,
        signal_length,
        show_dwell_time,
        base_colors,
    )
    add_base_type_patches(p, base_regions, base_colors)

    # Add labels if requested
    if show_labels:
        add_base_labels_position_mode(
            p,
            base_annotations,
            signal_max,
            show_dwell_time,
            base_colors,
            sample_rate,
            signal_length,
            position_label_interval,
            use_reference_positions,
        )

    return None


def plot_eventalign_signals(
    p,
    reads_data: list[tuple[str, np.ndarray, int]],
    normalization: NormalizationMethod,
    aligned_reads: list,
    show_dwell_time: bool = False,
    downsample: int = 1,
    show_signal_points: bool = False,
    theme: Theme = Theme.LIGHT,
):
    """Plot signal lines for event-aligned reads

    Args:
        show_dwell_time: If True, use cumulative time for x-axis instead of base position
        downsample: Downsampling factor (1 = no downsampling, 10 = every 10th point)
        theme: Color theme (LIGHT or DARK)
    """
    line_renderers = []

    for idx, (read_id, signal, sample_rate) in enumerate(reads_data):
        aligned_read = aligned_reads[idx]
        signal, _ = process_signal(signal, normalization)
        base_annotations = aligned_read.bases

        # Create signal coordinates - plot ALL signal samples
        signal_x = []
        signal_y = []
        signal_base_labels = []

        if show_dwell_time:
            # Use cumulative time for x-axis - spread samples evenly across each base's time duration
            cumulative_time = 0.0

            for i, base_annotation in enumerate(base_annotations):
                base = base_annotation.base
                start_idx = base_annotation.signal_start

                # Determine end index and dwell time for this base
                if i + 1 < len(base_annotations):
                    end_idx = base_annotations[i + 1].signal_start
                else:
                    end_idx = len(signal)

                dwell_samples = end_idx - start_idx
                dwell_time = (dwell_samples / sample_rate) * 1000  # ms

                # Plot signal samples within this base's region (with downsampling)
                for sample_offset in range(0, dwell_samples, downsample):
                    sample_idx = start_idx + sample_offset
                    if sample_idx < len(signal):
                        # Map sample to time: evenly distribute across base's time duration
                        time_offset = (
                            (sample_offset / dwell_samples) * dwell_time
                            if dwell_samples > 0
                            else 0
                        )
                        signal_x.append(cumulative_time + time_offset)
                        signal_y.append(signal[sample_idx])
                        signal_base_labels.append(base)

                cumulative_time += dwell_time
        else:
            # Use base position for x-axis - spread samples evenly across each base's position region
            for i, base_annotation in enumerate(base_annotations):
                base = base_annotation.base
                start_idx = base_annotation.signal_start

                # Determine end index for this base
                if i + 1 < len(base_annotations):
                    end_idx = base_annotations[i + 1].signal_start
                else:
                    end_idx = len(signal)

                num_samples = end_idx - start_idx

                # Plot signal samples within this base's region (with downsampling)
                for sample_offset in range(0, num_samples, downsample):
                    sample_idx = start_idx + sample_offset
                    if sample_idx < len(signal):
                        # Map sample to position: evenly distribute from i-0.5 to i+0.5
                        if num_samples > 1:
                            # Linear interpolation from -0.5 to +0.5
                            position_offset = -0.5 + (sample_offset / (num_samples - 1))
                        else:
                            position_offset = 0.0
                        signal_x.append(i + position_offset)
                        signal_y.append(signal[sample_idx])
                        signal_base_labels.append(base)

        # Create data source
        source = create_signal_data_source(
            np.array(signal_x),
            np.array(signal_y),
            read_id,
            signal_base_labels,
        )

        # Use theme-aware signal color for first read, then use multi-read colors
        if idx == 0:
            color = get_signal_line_color(theme)
        else:
            color = MULTI_READ_COLORS[idx % len(MULTI_READ_COLORS)]

        # Add signal renderers
        renderers = add_signal_renderers(
            p,
            source,
            color,
            show_signal_points,
            read_id[:12],
            line_width=2,
        )
        line_renderers.extend(renderers)

    return line_renderers


def create_modification_track_eventalign(
    aligned_reads: list,
    show_dwell_time: bool,
    sample_rate: int,
    theme: Theme,
    show_overlay: bool = True,
    overlay_opacity: float = DEFAULT_MOD_OVERLAY_OPACITY,
    mod_type_filter: str = "all",
    threshold_enabled: bool = False,
    threshold: float = 0.5,
):
    """Create a separate track for base modifications in eventalign mode

    Args:
        aligned_reads: List of AlignedRead objects with modifications
        show_dwell_time: If True, use time-based x-axis; otherwise use position-based
        sample_rate: Sampling rate in Hz
        theme: Color theme (LIGHT or DARK)
        show_overlay: Whether to show modification track
        overlay_opacity: Opacity for modification rectangles (0-1)
        mod_type_filter: Filter to show specific modification type ("all" or "base+code" like "C+m")
        threshold_enabled: Whether to apply probability threshold
        threshold: Probability threshold (only show mods with prob >= threshold)

    Returns:
        Bokeh figure with modification track, or None if no modifications
    """
    if not show_overlay or not aligned_reads:
        return None

    # Use first aligned read for modifications (assuming single-read)
    first_read = aligned_reads[0]
    if not first_read.modifications:
        return None

    base_annotations = first_read.bases

    # Prepare data for modification rectangles
    mod_data = {
        "x": [],  # Center position
        "y": [],  # Y position (always 0.5 for single row)
        "left": [],  # Left edge
        "right": [],  # Right edge
        "width": [],  # Width
        "mod_type": [],  # Modification type code
        "mod_name": [],  # Modification name
        "probability": [],  # Modification probability
        "base": [],  # Canonical base
        "position": [],  # Base position
        "color": [],  # Color for each modification
    }

    for mod in first_read.modifications:
        # Find the base annotation for this modification position
        matching_base = None
        for base_ann in base_annotations:
            if base_ann.position == mod.position:
                matching_base = base_ann
                break

        if not matching_base:
            continue

        # Apply modification type filter
        if mod_type_filter != "all":
            # Filter format is "base+code" like "C+m"
            mod_key = f"{matching_base.base}+{mod.mod_code}"
            if mod_key != mod_type_filter:
                continue

        # Apply probability threshold filter
        if threshold_enabled and mod.probability < threshold:
            continue

        # Calculate x-axis range based on mode
        if show_dwell_time:
            # Time-based: calculate cumulative time up to this base
            cumulative_time = 0.0
            for i, base_ann in enumerate(base_annotations):
                if base_ann.position == mod.position:
                    # Calculate dwell time for this base
                    if i + 1 < len(base_annotations):
                        end_idx = base_annotations[i + 1].signal_start
                    else:
                        # Last base - approximate
                        end_idx = base_ann.signal_end

                    dwell_samples = end_idx - base_ann.signal_start
                    dwell_time = (dwell_samples / sample_rate) * 1000  # ms

                    left_x = cumulative_time
                    right_x = cumulative_time + dwell_time
                    break

                # Accumulate time for previous bases
                if i + 1 < len(base_annotations):
                    next_start = base_annotations[i + 1].signal_start
                else:
                    next_start = base_ann.signal_end
                dwell_samples = next_start - base_ann.signal_start
                cumulative_time += (dwell_samples / sample_rate) * 1000
        else:
            # Position-based: use base index ± 0.5
            left_x = mod.position - 0.5
            right_x = mod.position + 0.5

        # Get modification color
        mod_color = MODIFICATION_COLORS.get(
            mod.mod_code, MODIFICATION_COLORS["default"]
        )

        # Get modification name
        mod_name = MODIFICATION_CODES.get(mod.mod_code, str(mod.mod_code))

        # Add data point
        mod_data["x"].append((left_x + right_x) / 2)
        mod_data["y"].append(0.5)  # Single row
        mod_data["left"].append(left_x)
        mod_data["right"].append(right_x)
        mod_data["width"].append(right_x - left_x)
        mod_data["mod_type"].append(str(mod.mod_code))
        mod_data["mod_name"].append(mod_name)
        mod_data["probability"].append(mod.probability)
        mod_data["base"].append(matching_base.base)
        mod_data["position"].append(mod.position)
        mod_data["color"].append(mod_color)

    # If no modifications, return None
    if not mod_data["x"]:
        return None

    # Create modification track figure
    p_mod = create_figure(
        title="Base Modifications (modBAM)",
        x_label="",  # No label needed - axes are shared with main plot
        y_label="",
        theme=theme,
    )

    # Hide toolbar (main plot below will have the toolbar)
    p_mod.toolbar_location = None

    # Hide y-axis (only one row)
    p_mod.yaxis.visible = False

    # Hide x-axis labels but keep ticks (axes are shared with main plot below)
    p_mod.xaxis.major_label_text_font_size = "0pt"

    # Minimize borders to reduce gap with main plot
    p_mod.min_border_bottom = 0
    p_mod.min_border_left = 5
    p_mod.min_border_right = 5
    p_mod.min_border_top = 5

    # Create data source
    mod_source = ColumnDataSource(data=mod_data)

    # Create rectangles for modifications
    # Use alpha from overlay_opacity and probability
    alphas = [overlay_opacity * p for p in mod_data["probability"]]
    mod_source.data["alpha"] = alphas

    rects = p_mod.rect(
        x="x",
        y="y",
        width="width",
        height=0.8,  # Height of rectangle (0-1 range)
        source=mod_source,
        fill_color="color",
        fill_alpha="alpha",
        line_color=None,
    )

    # Add hover tool with modification details
    hover_mod = HoverTool(
        renderers=[rects],
        tooltips=[
            ("Position", "@position"),
            ("Base", "@base"),
            ("Modification", "@mod_name (@mod_type)"),
            ("Probability", "@probability{0.3f}"),
        ],
        mode="mouse",
    )
    p_mod.add_tools(hover_mod)

    # Set y-axis range
    p_mod.y_range.start = 0
    p_mod.y_range.end = 1

    # Set height for modification track
    p_mod.sizing_mode = "stretch_width"
    p_mod.height = 80

    return p_mod


def add_modification_overlays_eventalign(
    p,
    aligned_reads: list,
    show_dwell_time: bool,
    sample_rate: int,
    show_overlay: bool = True,
    overlay_opacity: float = DEFAULT_MOD_OVERLAY_OPACITY,
):
    """Add vertical shaded regions for base modifications in eventalign mode

    DEPRECATED: This function creates full-height vertical overlays.
    Use create_modification_track_eventalign() instead for a separate track.

    Args:
        p: Bokeh figure
        aligned_reads: List of AlignedRead objects with modifications
        show_dwell_time: If True, use time-based x-axis; otherwise use position-based
        sample_rate: Sampling rate in Hz
        show_overlay: Whether to show modification overlays
        overlay_opacity: Opacity for modification shading (0-1)
    """
    if not show_overlay or not aligned_reads:
        return

    # Use first aligned read for modifications (assuming single-read or merged)
    first_read = aligned_reads[0]
    if not first_read.modifications:
        return

    base_annotations = first_read.bases

    for mod in first_read.modifications:
        # Find the base annotation for this modification position
        matching_base = None
        for base_ann in base_annotations:
            if base_ann.position == mod.position:
                matching_base = base_ann
                break

        if not matching_base:
            continue

        # Calculate x-axis range based on mode
        if show_dwell_time:
            # Time-based: calculate cumulative time up to this base
            cumulative_time = 0.0
            for i, base_ann in enumerate(base_annotations):
                if base_ann.position == mod.position:
                    # Calculate dwell time for this base
                    if i + 1 < len(base_annotations):
                        end_idx = base_annotations[i + 1].signal_start
                    else:
                        # Last base - approximate
                        end_idx = base_ann.signal_end

                    dwell_samples = end_idx - base_ann.signal_start
                    dwell_time = (dwell_samples / sample_rate) * 1000  # ms

                    left_x = cumulative_time
                    right_x = cumulative_time + dwell_time
                    break

                # Accumulate time for previous bases
                if i + 1 < len(base_annotations):
                    next_start = base_annotations[i + 1].signal_start
                else:
                    next_start = base_ann.signal_end
                dwell_samples = next_start - base_ann.signal_start
                cumulative_time += (dwell_samples / sample_rate) * 1000
        else:
            # Position-based: use base index ± 0.5
            left_x = mod.position - 0.5
            right_x = mod.position + 0.5

        # Get modification color
        mod_color = MODIFICATION_COLORS.get(
            mod.mod_code, MODIFICATION_COLORS["default"]
        )

        # Scale opacity by probability
        alpha = overlay_opacity * mod.probability

        # Add vertical box annotation
        box = BoxAnnotation(
            left=left_x,
            right=right_x,
            fill_color=mod_color,
            fill_alpha=alpha,
            line_width=0,
            level="underlay",  # Draw behind signal
        )
        p.add_layout(box)


def plot_eventalign(
    reads_data: list[tuple[str, np.ndarray, int]],
    normalization: NormalizationMethod,
    aligned_reads: list | None,
    downsample: int = 1,
    show_dwell_time: bool = False,
    show_labels: bool = True,
    show_signal_points: bool = False,
    position_label_interval: int = DEFAULT_POSITION_LABEL_INTERVAL,
    use_reference_positions: bool = False,
    theme: Theme = Theme.LIGHT,
    show_modification_overlay: bool = True,
    modification_overlay_opacity: float = DEFAULT_MOD_OVERLAY_OPACITY,
    modification_type_filter: str = "all",
    modification_threshold_enabled: bool = False,
    modification_threshold: float = 0.5,
) -> tuple[str, figure]:
    """Plot event-aligned reads with base annotations"""
    if not aligned_reads:
        raise ValueError("Event-aligned mode requires aligned_reads data")

    # Create figure with conditional x-axis label and status information
    title = format_plot_title("Event-Aligned", reads_data, normalization, downsample)
    x_label = "Time (ms)" if show_dwell_time else "Base Position"
    p = create_figure(
        title=title,
        x_label=x_label,
        y_label=f"Signal ({normalization.value})",
        theme=theme,
    )

    # Add base annotations
    add_base_annotations_eventalign(
        p,
        reads_data,
        normalization,
        aligned_reads,
        show_dwell_time,
        show_labels,
        position_label_interval,
        use_reference_positions,
        theme,
    )

    # Plot signal lines
    line_renderers = plot_eventalign_signals(
        p,
        reads_data,
        normalization,
        aligned_reads,
        show_dwell_time,
        downsample,
        show_signal_points,
        theme,
    )

    # Add hover tool with conditional tooltip and configure legend
    x_tooltip = (
        ("Time (ms)", "@x{0.2f}") if show_dwell_time else ("Base Position", "@x")
    )
    add_hover_tool(
        p,
        line_renderers,
        [
            ("Read", "@read_id"),
            x_tooltip,
            ("Base", "@base"),
            ("Signal", "@y{0.2f}"),
        ],
    )
    configure_legend(p)

    # Create modification track if available
    sample_rate = reads_data[0][2]
    p_mod = create_modification_track_eventalign(
        aligned_reads,
        show_dwell_time,
        sample_rate,
        theme,
        show_modification_overlay,
        modification_overlay_opacity,
        modification_type_filter,
        modification_threshold_enabled,
        modification_threshold,
    )

    # Generate HTML - either single plot or gridplot with modification track
    html_title = format_html_title("Event-Aligned", reads_data)

    if p_mod is not None:
        # Link x-axes for synchronized zoom/pan
        p_mod.x_range = p.x_range

        # Minimize top border of main plot to remove gap with modification track
        p.min_border_top = 0
        p.min_border_left = 5
        p.min_border_right = 5

        # Modification track has fixed height (80px), main plot stretches to fill remaining space
        # Keep default stretch_both on main plot for proper filling
        # The column layout will manage overall sizing with minimal spacing

        # Create column layout with tracks stacked vertically
        layout = column(
            p_mod,
            p,
            sizing_mode="stretch_both",  # Stretch both width and height
            spacing=0,  # No spacing between plots
        )
        html = file_html(layout, CDN, title=html_title)
        return html, layout
    else:
        # No modifications - return single plot
        html = file_html(p, CDN, title=html_title)
        return html, p
