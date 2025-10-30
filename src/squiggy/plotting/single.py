"""Single read plotting with base annotations"""

import numpy as np
from bokeh.embed import file_html
from bokeh.models import ColorBar, ColumnDataSource
from bokeh.plotting import figure
from bokeh.resources import CDN

from ..constants import (
    DEFAULT_POSITION_LABEL_INTERVAL,
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

    # Note: Not using column/row layout to avoid JavaScript navigation issues
    # The toggle controls are embedded in the figure via JS callbacks
    # Generate HTML
    html = file_html(p, CDN, title=f"Squiggy: {read_id}")
    return html, p
