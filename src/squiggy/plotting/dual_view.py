"""Dual-view plot mode with synchronized signal and sequence panels"""

import numpy as np
from bokeh.embed import file_html
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, CrosshairTool, HoverTool
from bokeh.plotting import figure
from bokeh.resources import CDN

from ..alignment import (
    AlignedRead,
    calculate_base_dwell_times,
    calculate_base_mean_currents,
    get_base_transitions,
)
from ..constants import (
    NormalizationMethod,
    Theme,
)
from .base import (
    add_signal_renderers,
    create_figure,
    create_signal_data_source,
    format_html_title,
    format_plot_title,
    get_base_colors,
    get_signal_line_color,
    process_signal,
)


def create_dual_view(
    read_id: str,
    signal: np.ndarray,
    sample_rate: int,
    normalization: NormalizationMethod,
    aligned_read: AlignedRead,
    show_transitions: bool = False,
    downsample: int = 1,
    theme: Theme = Theme.LIGHT,
) -> tuple[str, object]:
    """Create dual-view visualization with synchronized signal and sequence panels

    Args:
        read_id: Read identifier
        signal: Raw signal array
        sample_rate: Sampling rate in Hz
        normalization: Normalization method to apply
        aligned_read: AlignedRead object with base annotations
        show_transitions: Whether to show base transition lines
        downsample: Downsampling factor (1 = no downsampling)
        theme: Color theme (light or dark)

    Returns:
        Tuple of (html_string, bokeh_figure)
    """
    # Process signal
    processed_signal, _ = process_signal(signal, normalization, downsample)

    # Create top panel: signal trace
    p_signal = _create_signal_panel(
        read_id,
        processed_signal,
        sample_rate,
        normalization,
        aligned_read,
        show_transitions,
        theme,
    )

    # Create bottom panel: sequence track
    p_sequence = _create_sequence_panel(
        read_id, processed_signal, aligned_read, show_transitions, theme
    )

    # Synchronize x-ranges between panels
    p_sequence.x_range = p_signal.x_range

    # Set explicit heights for each panel
    p_signal.sizing_mode = "stretch_width"
    p_signal.height = 350
    p_sequence.sizing_mode = "stretch_width"
    p_sequence.height = 150

    # Add synchronized crosshair tool
    crosshair = CrosshairTool(dimensions="both")
    p_signal.add_tools(crosshair)
    p_sequence.add_tools(crosshair)

    # Create grid layout with two panels stacked vertically
    grid = gridplot(
        [[p_signal], [p_sequence]],
        sizing_mode="stretch_width",
        toolbar_location="right",
    )

    # Generate HTML
    html_title = format_html_title(
        "Dual View",
        read_id,
        normalization,
        single_read=True,
    )
    html = file_html(grid, CDN, title=html_title)

    return html, grid


def _create_signal_panel(
    read_id: str,
    signal: np.ndarray,
    sample_rate: int,
    normalization: NormalizationMethod,
    aligned_read: AlignedRead,
    show_transitions: bool,
    theme: Theme,
) -> figure:
    """Create top panel with signal trace and optional transition lines

    Args:
        read_id: Read identifier
        signal: Processed signal array
        sample_rate: Sampling rate in Hz
        normalization: Normalization method applied
        aligned_read: AlignedRead object with base annotations
        show_transitions: Whether to show base transition lines
        theme: Color theme

    Returns:
        Bokeh figure for signal panel
    """
    # Create figure
    title = format_plot_title(
        read_id=read_id,
        normalization=normalization,
        mode_label="Dual View - Signal",
    )

    p = create_figure(
        title=title,
        x_label="Signal Sample",
        y_label="Current (pA)"
        if normalization == NormalizationMethod.NONE
        else "Signal",
        theme=theme,
    )

    # Create x-axis (sample indices)
    x = np.arange(len(signal))

    # Add signal trace
    source = create_signal_data_source(x, signal, read_id)
    color = get_signal_line_color(theme)
    renderers = add_signal_renderers(
        p,
        source,
        color,
        show_signal_points=False,
        line_width=1,
        alpha=0.8,
    )

    # Add hover tool for signal
    hover = HoverTool(
        renderers=renderers,
        tooltips=[
            ("Sample", "@sample"),
            ("Signal", "@y{0.2f}"),
        ],
        mode="mouse",
        point_policy="snap_to_data",
    )
    p.add_tools(hover)

    # Add base transition lines if requested
    if show_transitions:
        _add_transition_lines(p, aligned_read, signal, sample_rate, theme)

    return p


def _create_sequence_panel(
    read_id: str,
    signal: np.ndarray,
    aligned_read: AlignedRead,
    show_transitions: bool,
    theme: Theme,
) -> figure:
    """Create bottom panel with sequence track as colored blocks

    Args:
        read_id: Read identifier
        signal: Processed signal array (for y-axis height reference)
        aligned_read: AlignedRead object with base annotations
        show_transitions: Whether to show base transition lines
        theme: Color theme

    Returns:
        Bokeh figure for sequence panel
    """
    # Create figure
    p = create_figure(
        title="Base Sequence",
        x_label="Signal Sample",
        y_label="",
        theme=theme,
    )

    # Get base colors
    base_colors = get_base_colors(theme)

    # Prepare data for base blocks
    block_x_starts = []
    block_x_ends = []
    block_y_bottoms = []
    block_y_tops = []
    block_colors = []
    block_bases = []
    block_positions = []

    # Fixed height for sequence track (y-axis from 0 to 1)
    y_bottom = 0.0
    y_top = 1.0

    for base_annotation in aligned_read.bases:
        block_x_starts.append(base_annotation.signal_start)
        block_x_ends.append(base_annotation.signal_end)
        block_y_bottoms.append(y_bottom)
        block_y_tops.append(y_top)
        block_colors.append(base_colors.get(base_annotation.base, "#808080"))
        block_bases.append(base_annotation.base)
        block_positions.append(base_annotation.position)

    # Create data source for base blocks
    source = ColumnDataSource(
        data={
            "left": block_x_starts,
            "right": block_x_ends,
            "bottom": block_y_bottoms,
            "top": block_y_tops,
            "color": block_colors,
            "base": block_bases,
            "position": block_positions,
        }
    )

    # Add base blocks as quads
    quad_renderer = p.quad(
        left="left",
        right="right",
        bottom="bottom",
        top="top",
        source=source,
        fill_color="color",
        line_color="white",
        line_width=0.5,
        alpha=0.8,
    )

    # Add text labels for bases (centered in blocks)
    text_x = [
        (start + end) / 2
        for start, end in zip(block_x_starts, block_x_ends, strict=True)
    ]
    text_y = [(y_bottom + y_top) / 2] * len(block_bases)

    text_source = ColumnDataSource(
        data={
            "x": text_x,
            "y": text_y,
            "text": block_bases,
        }
    )

    p.text(
        x="x",
        y="y",
        text="text",
        source=text_source,
        text_align="center",
        text_baseline="middle",
        text_font_size="10pt",
        text_color="black" if theme == Theme.LIGHT else "white",
    )

    # Add hover tool for base blocks
    hover = HoverTool(
        renderers=[quad_renderer],
        tooltips=[
            ("Base", "@base"),
            ("Position", "@position"),
            ("Start", "@left"),
            ("End", "@right"),
        ],
        mode="mouse",
    )
    p.add_tools(hover)

    # Set y-axis range (fixed for sequence track)
    p.y_range.start = -0.1
    p.y_range.end = 1.1
    p.yaxis.visible = False  # Hide y-axis (not meaningful for sequence track)

    # Add vertical transition lines if requested
    if show_transitions:
        transitions = get_base_transitions(aligned_read)
        for transition_pos in transitions:
            p.line(
                [transition_pos, transition_pos],
                [y_bottom, y_top],
                line_color="gray",
                line_width=1,
                line_alpha=0.5,
                line_dash="dashed",
            )

    return p


def _add_transition_lines(
    p: figure,
    aligned_read: AlignedRead,
    signal: np.ndarray,
    sample_rate: float,
    theme: Theme,
):
    """Add vertical transition lines to signal panel with hover tooltips

    Args:
        p: Bokeh figure (signal panel)
        aligned_read: AlignedRead object with base annotations
        signal: Signal array for calculating mean currents
        sample_rate: Sampling rate in Hz
        theme: Color theme
    """
    # Calculate dwell times and mean currents for tooltips
    dwell_times = calculate_base_dwell_times(aligned_read, sample_rate)
    mean_currents = calculate_base_mean_currents(aligned_read, signal)

    # Get signal min/max for line height
    signal_min = np.min(signal)
    signal_max = np.max(signal)

    # Prepare data for transition lines
    line_x0 = []
    line_x1 = []
    line_y0 = []
    line_y1 = []
    line_bases = []
    line_dwell_times = []
    line_mean_currents = []

    # Add transition line for each base (except the last transition which is just the end)
    for i, base_annotation in enumerate(aligned_read.bases):
        transition_x = base_annotation.signal_start

        line_x0.append(transition_x)
        line_x1.append(transition_x)
        line_y0.append(signal_min)
        line_y1.append(signal_max)
        line_bases.append(base_annotation.base)
        line_dwell_times.append(dwell_times[i])
        line_mean_currents.append(mean_currents[i])

    # Create data source for transition lines
    source = ColumnDataSource(
        data={
            "x0": line_x0,
            "x1": line_x1,
            "y0": line_y0,
            "y1": line_y1,
            "base": line_bases,
            "dwell_time": line_dwell_times,
            "mean_current": line_mean_currents,
        }
    )

    # Add segment glyphs for transition lines
    segment_renderer = p.segment(
        x0="x0",
        y0="y0",
        x1="x1",
        y1="y1",
        source=source,
        line_color="red",
        line_width=2,
        line_alpha=0.6,
        line_dash="solid",
    )

    # Add hover tool for transition lines
    transition_hover = HoverTool(
        renderers=[segment_renderer],
        tooltips=[
            ("Base", "@base"),
            ("Dwell Time", "@dwell_time{0.2f} ms"),
            ("Mean Current", "@mean_current{0.2f} pA"),
            ("Position", "@x0"),
        ],
        mode="mouse",
    )
    p.add_tools(transition_hover)
