"""Base annotation utilities for nanopore signal visualization"""

import numpy as np
from bokeh.models import ColumnDataSource, LabelSet, LinearColorMapper
from bokeh.transform import transform

from ..constants import (
    BASE_ANNOTATION_ALPHA,
    DARK_THEME,
    DEFAULT_POSITION_LABEL_INTERVAL,
    LIGHT_THEME,
    Theme,
)


def calculate_base_regions_time_mode(
    sequence: str,
    seq_to_sig_map: list[int],
    time_ms: np.ndarray,
    signal: np.ndarray,
    signal_min: float,
    signal_max: float,
    sample_rate: int,
    show_dwell_time: bool,
    base_colors: dict,
):
    """
    Calculate base regions for time-based plots (single read mode).

    Returns:
        If show_dwell_time: (all_regions, all_dwell_times, all_labels_data)
        Else: (base_regions, base_labels_data)
    """
    if show_dwell_time:
        all_regions = []
        all_dwell_times = []
        all_labels_data = []

        for seq_pos in range(len(sequence)):
            if seq_pos >= len(seq_to_sig_map):
                break

            base = sequence[seq_pos]
            if base not in base_colors:
                continue

            sig_idx = seq_to_sig_map[seq_pos]
            if sig_idx >= len(signal):
                continue

            start_time = time_ms[sig_idx]

            # Calculate end time and dwell time
            if seq_pos + 1 < len(seq_to_sig_map):
                next_sig_idx = seq_to_sig_map[seq_pos + 1]
                end_time = (
                    time_ms[next_sig_idx]
                    if next_sig_idx < len(time_ms)
                    else time_ms[-1]
                )
            else:
                end_time = time_ms[-1]

            dwell_time = end_time - start_time

            all_regions.append(
                {
                    "left": start_time,
                    "right": end_time,
                    "top": signal_max,
                    "bottom": signal_min,
                    "dwell": dwell_time,
                }
            )
            all_dwell_times.append(dwell_time)

            # Store label data
            mid_time = (start_time + end_time) / 2
            all_labels_data.append(
                {"time": mid_time, "y": signal[sig_idx], "text": f"{base}{seq_pos}"}
            )

        return all_regions, all_dwell_times, all_labels_data

    else:
        # Normal mode: group by base type
        base_regions = {base: [] for base in ["A", "C", "G", "T"]}
        base_labels_data = {base: [] for base in ["A", "C", "G", "T"]}

        for seq_pos in range(len(sequence)):
            if seq_pos >= len(seq_to_sig_map):
                break

            base = sequence[seq_pos]
            if base not in base_colors:
                continue

            sig_idx = seq_to_sig_map[seq_pos]
            if sig_idx >= len(signal):
                continue

            start_time = time_ms[sig_idx]

            # Calculate end time
            if seq_pos + 1 < len(seq_to_sig_map):
                next_sig_idx = seq_to_sig_map[seq_pos + 1]
                end_time = (
                    time_ms[next_sig_idx]
                    if next_sig_idx < len(time_ms)
                    else time_ms[-1]
                )
            else:
                end_time = time_ms[-1]

            base_regions[base].append(
                {
                    "left": start_time,
                    "right": end_time,
                    "top": signal_max,
                    "bottom": signal_min,
                }
            )

            mid_time = (start_time + end_time) / 2
            base_labels_data[base].append(
                {"time": mid_time, "y": signal[sig_idx], "text": f"{base}{seq_pos}"}
            )

        return base_regions, base_labels_data


def calculate_base_regions_position_mode(
    base_annotations: list,
    signal_min: float,
    signal_max: float,
    sample_rate: int,
    signal_length: int,
    show_dwell_time: bool,
    base_colors: dict,
):
    """
    Calculate base regions for position-based plots (event-aligned mode).

    Returns:
        (base_regions,) - dict of base regions grouped by base type (A, C, G, T, U)
                         with time-scaled or position-based coordinates depending on show_dwell_time
    """
    # Group by base type (A, C, G, T)
    base_regions = {base: [] for base in ["A", "C", "G", "T", "U"]}

    if show_dwell_time:
        # Use cumulative time for x-coordinates (time-scaled axis)
        cumulative_time = 0.0

        for i, base_annotation in enumerate(base_annotations):
            base = base_annotation.base
            if base not in base_colors:
                continue

            # Calculate dwell time from signal indices
            if i + 1 < len(base_annotations):
                next_annotation = base_annotations[i + 1]
                dwell_samples = (
                    next_annotation.signal_start - base_annotation.signal_start
                )
            else:
                dwell_samples = signal_length - base_annotation.signal_start

            dwell_time = (dwell_samples / sample_rate) * 1000

            # Add region with time-based coordinates
            base_regions[base].append(
                {
                    "left": cumulative_time,
                    "right": cumulative_time + dwell_time,
                    "top": signal_max,
                    "bottom": signal_min,
                }
            )
            cumulative_time += dwell_time  # Advance by dwell time

    else:
        # Use base position for x-coordinates (evenly spaced)
        for i, base_annotation in enumerate(base_annotations):
            base = base_annotation.base
            if base not in base_colors:
                continue

            base_regions[base].append(
                {
                    "left": i - 0.5,
                    "right": i + 0.5,
                    "top": signal_max,
                    "bottom": signal_min,
                }
            )

    return (base_regions,)


def add_dwell_time_patches(p, all_regions: list[dict], all_dwell_times: list[float]):
    """Add background patches colored by dwell time"""
    if not all_regions:
        return None

    dwell_array = np.array(all_dwell_times)
    color_mapper = LinearColorMapper(
        palette="Viridis256",
        low=np.percentile(dwell_array, 5),
        high=np.percentile(dwell_array, 95),
    )

    patch_source = ColumnDataSource(
        data={
            "left": [r["left"] for r in all_regions],
            "right": [r["right"] for r in all_regions],
            "top": [r["top"] for r in all_regions],
            "bottom": [r["bottom"] for r in all_regions],
            "dwell": [r["dwell"] for r in all_regions],
        }
    )

    p.quad(
        left="left",
        right="right",
        top="top",
        bottom="bottom",
        source=patch_source,
        fill_color=transform("dwell", color_mapper),
        line_color=None,
        alpha=0.3,
    )

    return color_mapper


def add_base_type_patches(p, base_regions: dict, base_colors: dict):
    """Add background patches grouped by base type (shared by single and eventalign)"""
    for base in ["A", "C", "G", "T"]:
        if base_regions[base]:
            patch_source = ColumnDataSource(
                data={
                    "left": [r["left"] for r in base_regions[base]],
                    "right": [r["right"] for r in base_regions[base]],
                    "top": [r["top"] for r in base_regions[base]],
                    "bottom": [r["bottom"] for r in base_regions[base]],
                }
            )

            p.quad(
                left="left",
                right="right",
                top="top",
                bottom="bottom",
                source=patch_source,
                color=base_colors[base],
                alpha=BASE_ANNOTATION_ALPHA,
                legend_label=base,
            )


def add_base_labels_time_mode(
    p, base_labels_data, show_dwell_time: bool, base_colors: dict
):
    """Add base labels for time-based plots"""
    base_sources = []

    if show_dwell_time:
        # Single label source for all bases
        if base_labels_data:
            label_source = ColumnDataSource(
                data={
                    "time": [d["time"] for d in base_labels_data],
                    "y": [d["y"] for d in base_labels_data],
                    "text": [d["text"] for d in base_labels_data],
                }
            )
            base_sources.append(("all", label_source))
    else:
        # Separate label sources by base type
        for base in ["A", "C", "G", "T"]:
            if base_labels_data[base]:
                label_source = ColumnDataSource(
                    data={
                        "time": [d["time"] for d in base_labels_data[base]],
                        "y": [d["y"] for d in base_labels_data[base]],
                        "text": [d["text"] for d in base_labels_data[base]],
                    }
                )
                base_sources.append((base, label_source))

    return base_sources


def add_base_labels_position_mode(
    p,
    base_annotations: list,
    signal_max: float,
    show_dwell_time: bool,
    base_colors: dict,
    sample_rate: int = None,
    signal_length: int = None,
    position_label_interval: int = DEFAULT_POSITION_LABEL_INTERVAL,
    use_reference_positions: bool = False,
):
    """Add base labels for position-based plots (event-aligned mode)

    Args:
        show_dwell_time: If True, position labels using cumulative time
        base_colors: Dict of base colors
        sample_rate: Required if show_dwell_time=True
        signal_length: Required if show_dwell_time=True
        position_label_interval: Show position number every N bases
        use_reference_positions: Use reference positions (currently not implemented)
    """
    label_data = []
    position_number_data = []

    if show_dwell_time and sample_rate is not None and signal_length is not None:
        # Use cumulative time for label positioning
        cumulative_time = 0.0
        for i, base_annotation in enumerate(base_annotations):
            base = base_annotation.base
            if base in base_colors:
                # Calculate dwell time
                if i + 1 < len(base_annotations):
                    dwell_samples = (
                        base_annotations[i + 1].signal_start
                        - base_annotation.signal_start
                    )
                else:
                    dwell_samples = signal_length - base_annotation.signal_start
                dwell_time = (dwell_samples / sample_rate) * 1000

                # Position label at center of dwell time range
                label_x = cumulative_time + (dwell_time / 2)
                label_data.append(
                    {
                        "x": label_x,
                        "y": signal_max,
                        "text": base,
                        "color": base_colors[base],
                    }
                )

                # Add position number at intervals
                if i % position_label_interval == 0:
                    position_number_data.append(
                        {
                            "x": label_x,
                            "y": signal_max,
                            "text": str(i),
                        }
                    )

                cumulative_time += dwell_time
    else:
        # Use base position for label positioning (current behavior)
        for i, base_annotation in enumerate(base_annotations):
            base = base_annotation.base
            if base in base_colors:
                label_data.append(
                    {
                        "x": i,
                        "y": signal_max,
                        "text": base,
                        "color": base_colors[base],
                    }
                )

                # Add position number at intervals
                if i % position_label_interval == 0:
                    position_number_data.append(
                        {
                            "x": i,
                            "y": signal_max,
                            "text": str(i),
                        }
                    )

    # Add base letters
    if label_data:
        label_source = ColumnDataSource(
            data={
                "x": [d["x"] for d in label_data],
                "y": [d["y"] for d in label_data],
                "text": [d["text"] for d in label_data],
                "color": [d["color"] for d in label_data],
            }
        )
        labels = LabelSet(
            x="x",
            y="y",
            text="text",
            source=label_source,
            text_font_size="10pt",
            text_color="color",
            text_alpha=0.8,
            text_align="center",
            text_baseline="bottom",
            y_offset=5,
        )
        p.add_layout(labels)

    # Add position numbers
    if position_number_data:
        position_source = ColumnDataSource(
            data={
                "x": [d["x"] for d in position_number_data],
                "y": [d["y"] for d in position_number_data],
                "text": [d["text"] for d in position_number_data],
            }
        )
        position_labels = LabelSet(
            x="x",
            y="y",
            text="text",
            source=position_source,
            text_font_size="8pt",
            text_color="black",
            text_alpha=0.6,
            text_align="center",
            text_baseline="bottom",
            y_offset=20,  # Position above base letters
        )
        p.add_layout(position_labels)


def add_simple_labels(
    p,
    base_sources: list[tuple[str, ColumnDataSource]],
    base_colors: dict,
    theme: Theme = Theme.LIGHT,
):
    """Add base labels without toggle controls (always visible)"""
    if not base_sources:
        return

    # Get theme-appropriate text color for "all" labels
    theme_colors = DARK_THEME if theme == Theme.DARK else LIGHT_THEME
    default_text_color = theme_colors["axis_text"]

    for base, source in base_sources:
        labels = LabelSet(
            x="time",
            y="y",
            text="text",
            source=source,
            text_font_size="10pt",
            text_color=base_colors[base] if base != "all" else default_text_color,
            text_alpha=0.8,
            name=f"labels_{base}",
        )
        p.add_layout(labels)
