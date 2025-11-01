"""Base plotting utilities and shared functions for nanopore squiggle visualization"""

import numpy as np
from bokeh.models import ColumnDataSource, HoverTool, WheelZoomTool
from bokeh.plotting import figure

from ..constants import (
    BASE_COLORS,
    BASE_COLORS_DARK,
    DARK_THEME,
    LIGHT_THEME,
    SIGNAL_LINE_COLOR,
    SIGNAL_POINT_ALPHA,
    SIGNAL_POINT_COLOR,
    SIGNAL_POINT_SIZE,
    NormalizationMethod,
    Theme,
)

# Color palette for multi-read plots
MULTI_READ_COLORS = [
    SIGNAL_LINE_COLOR,
    "red",
    "green",
    "orange",
    "purple",
    "brown",
    "pink",
    "gray",
]


def normalize_signal(signal: np.ndarray, method: NormalizationMethod) -> np.ndarray:
    """Normalize signal data using specified method"""
    if method == NormalizationMethod.NONE:
        return signal
    elif method == NormalizationMethod.ZNORM:
        # Z-score normalization
        return (signal - np.mean(signal)) / np.std(signal)
    elif method == NormalizationMethod.MEDIAN:
        # Median normalization
        return signal - np.median(signal)
    elif method == NormalizationMethod.MAD:
        # Median absolute deviation (robust)
        median = np.median(signal)
        mad = np.median(np.abs(signal - median))
        return (signal - median) / (mad if mad > 0 else 1)
    else:
        return signal


def process_signal(
    signal: np.ndarray,
    normalization: NormalizationMethod,
    downsample: int = 1,
    seq_to_sig_map: list[int] | None = None,
) -> tuple[np.ndarray, list[int] | None]:
    """Process signal: normalize and optionally downsample

    Args:
        signal: Raw signal array
        normalization: Normalization method to apply
        downsample: Downsampling factor (1 = no downsampling)
        seq_to_sig_map: Optional sequence-to-signal mapping to downsample

    Returns:
        Tuple of (processed_signal, downsampled_seq_to_sig_map)
    """
    signal = normalize_signal(signal, normalization)
    if downsample > 1:
        signal = signal[::downsample]
        if seq_to_sig_map is not None:
            seq_to_sig_map = [idx // downsample for idx in seq_to_sig_map]
    return signal, seq_to_sig_map


def create_signal_data_source(
    x: np.ndarray,
    signal: np.ndarray,
    read_id: str | None = None,
    base_labels: list[str] | None = None,
) -> ColumnDataSource:
    """Create a standard signal data source with common fields

    Args:
        x: X-axis values (time, sample, or position)
        signal: Y-axis signal values
        read_id: Optional read identifier (repeated for all samples)
        base_labels: Optional base labels for each sample

    Returns:
        ColumnDataSource with standardized fields
    """
    data = {"x": x, "y": signal, "sample": np.arange(len(signal))}
    if read_id:
        data["read_id"] = [read_id] * len(signal)
    if base_labels:
        data["base"] = base_labels
    return ColumnDataSource(data=data)


def add_hover_tool(p, renderers: list, tooltip_fields: list[tuple[str, str]]):
    """Add a hover tool with specified tooltips

    Args:
        p: Bokeh figure
        renderers: List of renderers to attach hover to
        tooltip_fields: List of (label, field) tuples for tooltips
                       e.g., [("Time", "@time{0.2f} ms"), ("Signal", "@signal{0.2f}")]
    """
    hover = HoverTool(
        renderers=renderers,
        tooltips=tooltip_fields,
        mode="mouse",
        point_policy="snap_to_data",
    )
    p.add_tools(hover)


def configure_legend(p):
    """Configure standard legend appearance and behavior

    Creates a compact, horizontal legend at the bottom right with transparency.
    """
    p.legend.click_policy = "hide"
    p.legend.location = "bottom_right"
    p.legend.orientation = "horizontal"

    # Add transparency and reduce visual footprint
    p.legend.background_fill_alpha = 0.6
    p.legend.border_line_alpha = 0.5

    # Reduce spacing and padding for even more compact appearance
    p.legend.label_text_font_size = "8pt"
    p.legend.spacing = 1  # Space between legend items
    p.legend.padding = 2  # Padding inside legend box
    p.legend.margin = 3  # Margin around legend box
    p.legend.glyph_width = 12  # Width of color boxes
    p.legend.glyph_height = 8  # Height of color boxes
    p.legend.label_height = 8  # Height of labels
    p.legend.label_standoff = 2  # Space between glyph and label


def add_signal_renderers(
    p,
    source: ColumnDataSource,
    color: str,
    show_signal_points: bool = False,
    legend_label: str | None = None,
    x_field: str = "x",
    y_field: str = "y",
    line_width: int = 1,
    alpha: float = 0.8,
) -> list:
    """Add signal line and optional scatter points

    Args:
        p: Bokeh figure
        source: Data source
        color: Line color
        show_signal_points: Whether to add scatter points
        legend_label: Optional legend label
        x_field: Name of x data field in source
        y_field: Name of y data field in source
        line_width: Width of the line
        alpha: Transparency of the line

    Returns:
        List of renderers [line_renderer] or [line_renderer, scatter_renderer]
    """
    renderers = []

    # Add line
    line_kwargs = {
        "x": x_field,
        "y": y_field,
        "source": source,
        "line_width": line_width,
        "color": color,
        "alpha": alpha,
    }
    if legend_label:
        line_kwargs["legend_label"] = legend_label
    line = p.line(**line_kwargs)
    renderers.append(line)

    # Add scatter points if requested
    if show_signal_points:
        scatter_kwargs = {
            "x": x_field,
            "y": y_field,
            "source": source,
            "size": SIGNAL_POINT_SIZE,
            "color": SIGNAL_POINT_COLOR,
            "alpha": SIGNAL_POINT_ALPHA,
        }
        if legend_label:
            scatter_kwargs["legend_label"] = legend_label
        circle = p.scatter(**scatter_kwargs)
        renderers.append(circle)

    return renderers


def get_base_colors(theme: Theme = Theme.LIGHT) -> dict:
    """Get base colors appropriate for the current theme"""
    return BASE_COLORS_DARK if theme == Theme.DARK else BASE_COLORS


def get_signal_line_color(theme: Theme = Theme.LIGHT) -> str:
    """Get signal line color appropriate for the current theme"""
    theme_colors = DARK_THEME if theme == Theme.DARK else LIGHT_THEME
    return theme_colors["signal_line"]


def create_figure(title: str, x_label: str, y_label: str, theme: Theme = Theme.LIGHT):
    """Create a standard Bokeh figure with common settings"""
    # Get theme colors
    theme_colors = DARK_THEME if theme == Theme.DARK else LIGHT_THEME

    p = figure(
        title=title,
        x_axis_label=x_label,
        y_axis_label=y_label,
        tools="xpan,xbox_zoom,box_zoom,wheel_zoom,reset,save",
        active_drag="xbox_zoom",
        active_scroll="wheel_zoom",
        sizing_mode="stretch_both",
        background_fill_color=theme_colors["plot_bg"],
        border_fill_color=theme_colors["plot_border"],
    )

    # Apply theme to title
    p.title.text_color = theme_colors["title_text"]

    # Apply theme to axes
    p.xaxis.axis_label_text_color = theme_colors["axis_text"]
    p.xaxis.major_label_text_color = theme_colors["axis_text"]
    p.xaxis.axis_line_color = theme_colors["axis_line"]
    p.xaxis.major_tick_line_color = theme_colors["axis_line"]
    p.xaxis.minor_tick_line_color = theme_colors["axis_line"]

    p.yaxis.axis_label_text_color = theme_colors["axis_text"]
    p.yaxis.major_label_text_color = theme_colors["axis_text"]
    p.yaxis.axis_line_color = theme_colors["axis_line"]
    p.yaxis.major_tick_line_color = theme_colors["axis_line"]
    p.yaxis.minor_tick_line_color = theme_colors["axis_line"]

    # Apply theme to grid
    p.xgrid.grid_line_color = None  # Keep vertical grid lines off
    p.ygrid.grid_line_color = theme_colors["grid_line"]

    # Add x-only wheel zoom
    wheel_zoom = WheelZoomTool(dimensions="width")
    p.add_tools(wheel_zoom)
    p.toolbar.active_scroll = wheel_zoom

    return p


def format_plot_title(
    mode_name: str,
    reads_data: list[tuple[str, np.ndarray, int]],
    normalization: NormalizationMethod = None,
    downsample: int = 1,
) -> str:
    """Generate a consistent plot title with status information"""
    if len(reads_data) == 1:
        read_id, signal, sample_rate = reads_data[0]
        signal_range = f"[{int(np.min(signal))}-{int(np.max(signal))}]"
        norm_str = f" | norm: {normalization.value}" if normalization else ""
        ds_str = f" | downsample: 1/{downsample}" if downsample > 1 else ""
        return f"{read_id} | {mode_name} | signal: {signal_range}{norm_str}{ds_str}"
    else:
        return f"{mode_name}: {len(reads_data)} reads"


def format_html_title(
    mode_name: str, reads_data: list[tuple[str, np.ndarray, int]]
) -> str:
    """Generate a consistent HTML title"""
    if len(reads_data) == 1:
        return f"Squiggy: {mode_name} (1 read - {reads_data[0][0]})"
    else:
        return f"Squiggy: {mode_name} ({len(reads_data)} reads)"
