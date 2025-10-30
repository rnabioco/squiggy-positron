"""Overlay plot mode for multiple reads"""

import numpy as np
from bokeh.embed import file_html
from bokeh.plotting import figure
from bokeh.resources import CDN

from ..constants import CoordinateSpace, NormalizationMethod, Theme
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
    coordinate_space: CoordinateSpace = CoordinateSpace.SIGNAL,
    aligned_reads: list | None = None,
    show_signal_points: bool = False,
    theme: Theme = Theme.LIGHT,
) -> tuple[str, figure]:
    """Plot multiple reads overlaid on same axes

    Args:
        reads_data: List of (read_id, signal, sample_rate) tuples
        normalization: Signal normalization method
        downsample: Downsampling factor
        coordinate_space: SIGNAL (sample indices) or SEQUENCE (reference positions)
        aligned_reads: List of AlignedRead objects (required for SEQUENCE space)
        show_signal_points: Whether to show individual signal points
        theme: Color theme

    Returns:
        Tuple of (HTML string, Bokeh figure)
    """
    # Determine x-axis label based on coordinate space
    x_label = "Sample" if coordinate_space == CoordinateSpace.SIGNAL else "Reference Position"

    # Create figure with status information
    title = format_plot_title("Overlay", reads_data, normalization, downsample)
    p = create_figure(
        title=title,
        x_label=x_label,
        y_label=f"Signal ({normalization.value})",
        theme=theme,
    )

    all_renderers = []

    if coordinate_space == CoordinateSpace.SIGNAL:
        # Signal space: plot by sample index (original behavior)
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

        # Add hover tool
        add_hover_tool(
            p,
            all_renderers,
            [("Read", "@read_id"), ("Sample", "@x"), ("Signal", "@y{0.2f}")],
        )
    else:
        # Sequence space: plot aligned to reference positions
        if not aligned_reads:
            raise ValueError("aligned_reads required for SEQUENCE coordinate space")

        # Create a mapping from read_id to aligned_read
        aligned_dict = {ar.read_id: ar for ar in aligned_reads}

        for idx, (read_id, signal, _sample_rate) in enumerate(reads_data):
            # Get alignment for this read
            aligned_read = aligned_dict.get(read_id)
            if not aligned_read or not aligned_read.bases:
                # Skip reads without alignment
                continue

            # Process signal (normalize and downsample)
            signal, _ = process_signal(signal, normalization, downsample)

            # Map signal to reference positions using base annotations
            ref_positions = []
            signal_values = []

            for base in aligned_read.bases:
                if base.genomic_pos is None:
                    continue

                # Get signal samples for this base
                start_idx = base.signal_start // downsample
                end_idx = base.signal_end // downsample

                # Ensure indices are within bounds
                if start_idx >= len(signal) or end_idx > len(signal):
                    continue

                # Map all signal samples in this base to the genomic position
                for sig_idx in range(start_idx, min(end_idx, len(signal))):
                    ref_positions.append(base.genomic_pos)
                    signal_values.append(signal[sig_idx])

            if not ref_positions:
                # Skip if no valid positions
                continue

            # Create data source
            x = np.array(ref_positions)
            y = np.array(signal_values)
            source = create_signal_data_source(x, y, read_id)

            # Add signal renderers with color cycling
            color = MULTI_READ_COLORS[idx % len(MULTI_READ_COLORS)]
            renderers = add_signal_renderers(
                p, source, color, show_signal_points, read_id[:12], alpha=0.7
            )
            all_renderers.extend(renderers)

        # Add hover tool
        add_hover_tool(
            p,
            all_renderers,
            [("Read", "@read_id"), ("Position", "@x"), ("Signal", "@y{0.2f}")],
        )

    configure_legend(p)

    # Generate HTML
    html_title = format_html_title("Overlay", reads_data)
    html = file_html(p, CDN, title=html_title)
    return html, p
