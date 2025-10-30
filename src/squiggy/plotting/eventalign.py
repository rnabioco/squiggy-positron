"""Event-aligned plot mode with base annotations"""

import numpy as np
from bokeh.embed import file_html
from bokeh.plotting import figure
from bokeh.resources import CDN

from ..constants import (
    DEFAULT_POSITION_LABEL_INTERVAL,
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

    # Generate HTML
    html_title = format_html_title("Event-Aligned", reads_data)
    html = file_html(p, CDN, title=html_title)
    return html, p
