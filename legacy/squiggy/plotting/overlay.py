"""Overlay plot mode for multiple reads"""

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


def plot_overlay(
    reads_data: list[tuple[str, np.ndarray, int]],
    normalization: NormalizationMethod,
    downsample: int = 1,
    show_signal_points: bool = False,
    theme: Theme = Theme.LIGHT,
    aligned_reads: list | None = None,
) -> tuple[str, figure]:
    """Plot multiple reads overlaid on same axes

    Note:
        Read subsetting by modification status should be performed by the caller
        before passing reads_data to this function. The aligned_reads parameter
        is accepted for API consistency but not currently used in this plot mode.
    """
    # Create figure with status information
    title = format_plot_title("Overlay", reads_data, normalization, downsample)
    p = create_figure(
        title=title,
        x_label="Sample",
        y_label=f"Signal ({normalization.value})",
        theme=theme,
    )

    all_renderers = []

    for idx, (read_id, signal, _sample_rate) in enumerate(reads_data):
        # Process signal (normalize and downsample)
        signal, _ = process_signal(signal, normalization, downsample)

        # Create data source
        x = np.arange(len(signal))
        source = create_signal_data_source(x, signal, read_id)

        # Add signal renderers with color cycling
        color = MULTI_READ_COLORS[idx % len(MULTI_READ_COLORS)]
        renderers = add_signal_renderers(
            p, source, color, show_signal_points, read_id[:12], alpha=0.7
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
    html_title = format_html_title("Overlay", reads_data)
    html = file_html(p, CDN, title=html_title)
    return html, p
