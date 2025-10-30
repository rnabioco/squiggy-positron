"""Stacked plot mode for multiple reads"""

import numpy as np
from bokeh.embed import file_html
from bokeh.plotting import figure
from bokeh.resources import CDN

from ..constants import NormalizationMethod, Theme
from .base import (
    MULTI_READ_COLORS,
    add_hover_tool,
    add_signal_renderers,
    configure_legend,
    create_figure,
    create_signal_data_source,
    format_html_title,
    format_plot_title,
    process_signal,
)


def plot_stacked(
    reads_data: list[tuple[str, np.ndarray, int]],
    normalization: NormalizationMethod,
    downsample: int = 1,
    show_signal_points: bool = False,
    theme: Theme = Theme.LIGHT,
) -> tuple[str, figure]:
    """Plot multiple reads stacked vertically with offset"""
    # Create figure with status information
    title = format_plot_title("Stacked", reads_data, normalization, downsample)
    p = create_figure(
        title=title,
        x_label="Sample",
        y_label=f"Signal ({normalization.value}) + offset",
        theme=theme,
    )

    # First pass: process all signals and determine offset
    offset_step = 0
    processed_signals = []
    for read_id, signal, sample_rate in reads_data:
        signal, _ = process_signal(signal, normalization, downsample)
        processed_signals.append((read_id, signal, sample_rate))
        signal_range = np.ptp(signal)
        offset_step = max(offset_step, signal_range * 1.2)

    # Second pass: plot with offsets
    all_renderers = []
    for idx, (read_id, signal, _sample_rate) in enumerate(processed_signals):
        offset = idx * offset_step
        x = np.arange(len(signal))
        y_offset = signal + offset
        source = create_signal_data_source(x, y_offset, read_id)

        # Add signal renderers with color cycling
        color = MULTI_READ_COLORS[idx % len(MULTI_READ_COLORS)]
        renderers = add_signal_renderers(
            p, source, color, show_signal_points, read_id[:12]
        )
        all_renderers.extend(renderers)

    # Add hover tool and configure legend
    add_hover_tool(
        p,
        all_renderers,
        [("Read", "@read_id"), ("Sample", "@x"), ("Signal", "@y{0.2f}")],
    )
    configure_legend(p)

    # Generate HTML
    html_title = format_html_title("Stacked", reads_data)
    html = file_html(p, CDN, title=html_title)
    return html, p
