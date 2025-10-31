"""Single read plotting with base annotations"""

import numpy as np
from bokeh.embed import file_html
from bokeh.layouts import column
from bokeh.models import BoxAnnotation, ColorBar, ColumnDataSource, HoverTool
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
    add_hover_tool,
    add_signal_renderers,
    create_figure,
    format_plot_title,
    get_base_colors,
    get_signal_line_color,
    process_signal,
)
from .base_annotations import (
    add_base_labels_time_mode,
    add_base_type_patches,
    add_dwell_time_patches,
    add_simple_labels,
    calculate_base_regions_time_mode,
)


def add_base_annotations_single_read(
    p,
    signal: np.ndarray,
    time_ms: np.ndarray,
    sequence: str | None,
    seq_to_sig_map: list[int] | None,
    sample_rate: int,
    show_dwell_time: bool,
    show_labels: bool,
    theme: Theme = Theme.LIGHT,
):
    """Add base annotations for single read plots

    Returns:
        tuple: (color_mapper, toggle_widget) - both may be None
    """
    if not sequence or seq_to_sig_map is None or len(seq_to_sig_map) == 0:
        return None, None

    base_colors = get_base_colors(theme)
    signal_min = np.min(signal)
    signal_max = np.max(signal)
    color_mapper = None

    if show_dwell_time:
        # Calculate and add dwell time patches
        all_regions, all_dwell_times, all_labels_data = (
            calculate_base_regions_time_mode(
                sequence,
                seq_to_sig_map,
                time_ms,
                signal,
                signal_min,
                signal_max,
                sample_rate,
                show_dwell_time,
                base_colors,
            )
        )
        color_mapper = add_dwell_time_patches(p, all_regions, all_dwell_times)

        # Add labels if requested (always visible, no toggle)
        if show_labels:
            base_sources = add_base_labels_time_mode(
                p, all_labels_data, show_dwell_time, base_colors
            )
            add_simple_labels(p, base_sources, base_colors, theme)
    else:
        # Calculate and add base type patches
        base_regions, base_labels_data = calculate_base_regions_time_mode(
            sequence,
            seq_to_sig_map,
            time_ms,
            signal,
            signal_min,
            signal_max,
            sample_rate,
            show_dwell_time,
            base_colors,
        )
        add_base_type_patches(p, base_regions, base_colors)

        # Add labels if requested (always visible, no toggle)
        if show_labels:
            base_sources = add_base_labels_time_mode(
                p, base_labels_data, show_dwell_time, base_colors
            )
            add_simple_labels(p, base_sources, base_colors, theme)

        p.legend.click_policy = "hide"
        p.legend.location = "top_right"

    return color_mapper, None


def create_modification_track_single(
    modifications: list,
    time_ms: np.ndarray,
    theme: Theme,
    show_overlay: bool = True,
    overlay_opacity: float = DEFAULT_MOD_OVERLAY_OPACITY,
    mod_type_filter: str = "all",
    sequence: str | None = None,
    threshold_enabled: bool = False,
    threshold: float = 0.5,
):
    """Create a separate track for base modifications in single read mode

    Args:
        modifications: List of ModificationAnnotation objects
        time_ms: Time array in milliseconds
        theme: Color theme (LIGHT or DARK)
        show_overlay: Whether to show modification track
        overlay_opacity: Opacity for modification rectangles (0-1)
        mod_type_filter: Filter to show specific modification type ("all" or "base+code" like "C+m")
        sequence: Optional sequence for getting canonical base
        threshold_enabled: Whether to apply probability threshold
        threshold: Probability threshold (only show mods with prob >= threshold)

    Returns:
        Bokeh figure with modification track, or None if no modifications
    """
    if not show_overlay or not modifications:
        return None

    # Prepare data for modification rectangles
    mod_data = {
        "x": [],  # Center time
        "y": [],  # Y position (always 0.5 for single row)
        "left": [],  # Left edge time
        "right": [],  # Right edge time
        "width": [],  # Width in time
        "mod_type": [],  # Modification type code
        "mod_name": [],  # Modification name
        "probability": [],  # Modification probability
        "position": [],  # Base position
        "color": [],  # Color for each modification
    }

    for mod in modifications:
        # Get time range for this modification
        # signal_start and signal_end are in signal sample indices
        if mod.signal_start >= len(time_ms) or mod.signal_end >= len(time_ms):
            continue  # Skip if indices out of bounds

        # Apply modification type filter
        if mod_type_filter != "all" and sequence:
            # Get canonical base from sequence
            if mod.position < len(sequence):
                canonical_base = sequence[mod.position]
                mod_key = f"{canonical_base}+{mod.mod_code}"
                if mod_key != mod_type_filter:
                    continue

        # Apply probability threshold filter
        if threshold_enabled and mod.probability < threshold:
            continue

        left_time = time_ms[mod.signal_start]
        right_time = time_ms[min(mod.signal_end, len(time_ms) - 1)]

        # Get modification color
        mod_color = MODIFICATION_COLORS.get(
            mod.mod_code, MODIFICATION_COLORS["default"]
        )

        # Get modification name
        mod_name = MODIFICATION_CODES.get(mod.mod_code, str(mod.mod_code))

        # Add data point
        mod_data["x"].append((left_time + right_time) / 2)
        mod_data["y"].append(0.5)  # Single row
        mod_data["left"].append(left_time)
        mod_data["right"].append(right_time)
        mod_data["width"].append(right_time - left_time)
        mod_data["mod_type"].append(str(mod.mod_code))
        mod_data["mod_name"].append(mod_name)
        mod_data["probability"].append(mod.probability)
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
            ("Modification", "@mod_name (@mod_type)"),
            ("Probability", "@probability{0.3f}"),
            ("Time", "@left{0.2f} - @right{0.2f} ms"),
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


def add_modification_overlays(
    p,
    modifications: list,
    time_ms: np.ndarray,
    show_overlay: bool = True,
    overlay_opacity: float = DEFAULT_MOD_OVERLAY_OPACITY,
):
    """Add vertical shaded regions for base modifications

    DEPRECATED: This function creates full-height vertical overlays.
    Use create_modification_track_single() instead for a separate track.

    Args:
        p: Bokeh figure
        modifications: List of ModificationAnnotation objects
        time_ms: Time array in milliseconds
        show_overlay: Whether to show modification overlays
        overlay_opacity: Opacity for modification shading (0-1)
    """
    if not show_overlay or not modifications:
        return

    for mod in modifications:
        # Get time range for this modification
        # signal_start and signal_end are in signal sample indices
        if mod.signal_start >= len(time_ms) or mod.signal_end >= len(time_ms):
            continue  # Skip if indices out of bounds

        left_time = time_ms[mod.signal_start]
        right_time = time_ms[min(mod.signal_end, len(time_ms) - 1)]

        # Get modification color
        mod_color = MODIFICATION_COLORS.get(
            mod.mod_code, MODIFICATION_COLORS["default"]
        )

        # Scale opacity by probability
        alpha = overlay_opacity * mod.probability

        # Add vertical box annotation
        box = BoxAnnotation(
            left=left_time,
            right=right_time,
            fill_color=mod_color,
            fill_alpha=alpha,
            line_width=0,
            level="underlay",  # Draw behind signal
        )
        p.add_layout(box)


def plot_single_read(
    signal: np.ndarray,
    read_id: str,
    sample_rate: int,
    sequence: str | None = None,
    seq_to_sig_map: list[int] | None = None,
    normalization: NormalizationMethod = NormalizationMethod.NONE,
    downsample: int = 1,
    show_dwell_time: bool = False,
    show_labels: bool = True,
    show_signal_points: bool = False,
    position_label_interval: int = DEFAULT_POSITION_LABEL_INTERVAL,
    use_reference_positions: bool = False,
    theme: Theme = Theme.LIGHT,
    modifications: list | None = None,
    show_modification_overlay: bool = True,
    modification_overlay_opacity: float = DEFAULT_MOD_OVERLAY_OPACITY,
    modification_type_filter: str = "all",
    modification_threshold_enabled: bool = False,
    modification_threshold: float = 0.5,
) -> tuple[str, figure]:
    """
    Plot a single nanopore read with optional base annotations

    Args:
        signal: Raw signal array
        read_id: Read identifier
        sample_rate: Sampling rate in Hz
        sequence: Optional basecall sequence
        seq_to_sig_map: Optional mapping from sequence positions to signal indices
        normalization: Signal normalization method
        downsample: Downsampling factor (1 = no downsampling, 10 = every 10th point)
        show_dwell_time: Color bases by dwell time instead of base type
        show_labels: Show base labels on plot
        show_signal_points: Show individual signal points as circles
        position_label_interval: Show position number every N bases
        use_reference_positions: Use reference genome positions (requires alignment data)
        theme: Color theme (LIGHT or DARK)
        modifications: Optional list of ModificationAnnotation objects
        show_modification_overlay: Whether to show modification overlays
        modification_overlay_opacity: Opacity for modification shading (0-1)

    Returns:
        Tuple[str, figure]: (HTML string, Bokeh figure object)
    """

    # Process signal (normalize and downsample)
    signal, seq_to_sig_map = process_signal(
        signal, normalization, downsample, seq_to_sig_map
    )

    # Create time axis and figure with status information
    time_ms = np.arange(len(signal)) * 1000 / sample_rate
    reads_data = [(read_id, signal, sample_rate)]
    title = format_plot_title("Single", reads_data, normalization, downsample)
    p = create_figure(
        title=title,
        x_label="Time (ms)",
        y_label=f"Signal ({normalization.value})",
        theme=theme,
    )

    # Add base annotations if available (returns color_mapper)
    color_mapper, _ = add_base_annotations_single_read(
        p,
        signal,
        time_ms,
        sequence,
        seq_to_sig_map,
        sample_rate,
        show_dwell_time,
        show_labels,
        theme,
    )

    # Create data source and add signal renderers
    signal_source = ColumnDataSource(
        data={"time": time_ms, "signal": signal, "sample": np.arange(len(signal))}
    )
    signal_color = get_signal_line_color(theme)
    renderers = add_signal_renderers(
        p,
        signal_source,
        signal_color,
        show_signal_points,
        x_field="time",
        y_field="signal",
    )

    # Add hover tool
    add_hover_tool(
        p,
        renderers,
        [
            ("Time", "@time{0.2f} ms"),
            ("Signal", "@signal{0.2f}"),
            ("Sample", "@sample"),
        ],
    )

    # Add color bar if showing dwell time
    if color_mapper is not None:
        color_bar = ColorBar(
            color_mapper=color_mapper,
            label_standoff=12,
            location=(0, 0),
            title="Dwell Time (ms)",
            title_standoff=15,
        )
        p.add_layout(color_bar, "right")

    # Create modification track if available
    p_mod = None
    if modifications:
        p_mod = create_modification_track_single(
            modifications,
            time_ms,
            theme,
            show_modification_overlay,
            modification_overlay_opacity,
            modification_type_filter,
            sequence,
            modification_threshold_enabled,
            modification_threshold,
        )

    # Generate HTML - either single plot or gridplot with modification track
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
        html = file_html(layout, CDN, title=f"Squiggy: {read_id}")
        return html, layout
    else:
        # No modifications - return single plot
        html = file_html(p, CDN, title=f"Squiggy: {read_id}")
        return html, p
