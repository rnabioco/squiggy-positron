"""Plotting for nanopore squiggle visualization"""

from typing import List, Optional, Tuple

import numpy as np
from bokeh.embed import file_html
from bokeh.models import (
    ColorBar,
    ColumnDataSource,
    HoverTool,
    LabelSet,
    LinearColorMapper,
    WheelZoomTool,
)
from bokeh.plotting import figure
from bokeh.resources import CDN
from bokeh.transform import transform

from .constants import (
    BASE_ANNOTATION_ALPHA,
    BASE_COLORS,
    BASE_COLORS_DARK,
    DARK_THEME,
    DEFAULT_POSITION_LABEL_INTERVAL,
    LIGHT_THEME,
    SIGNAL_LINE_COLOR,
    SIGNAL_POINT_ALPHA,
    SIGNAL_POINT_COLOR,
    SIGNAL_POINT_SIZE,
    NormalizationMethod,
    PlotMode,
    Theme,
)


class SquigglePlotter:
    """Interactive plotter for nanopore squiggle visualization"""

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

    @staticmethod
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

    @staticmethod
    def _get_base_colors(theme: Theme = Theme.LIGHT) -> dict:
        """Get base colors appropriate for the current theme"""
        return BASE_COLORS_DARK if theme == Theme.DARK else BASE_COLORS

    @staticmethod
    def _get_signal_line_color(theme: Theme = Theme.LIGHT) -> str:
        """Get signal line color appropriate for the current theme"""
        theme_colors = DARK_THEME if theme == Theme.DARK else LIGHT_THEME
        return theme_colors["signal_line"]

    @staticmethod
    def _create_figure(
        title: str, x_label: str, y_label: str, theme: Theme = Theme.LIGHT
    ):
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

    @staticmethod
    def _format_plot_title(
        mode_name: str,
        reads_data: List[Tuple[str, np.ndarray, int]],
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

    @staticmethod
    def _format_html_title(
        mode_name: str, reads_data: List[Tuple[str, np.ndarray, int]]
    ) -> str:
        """Generate a consistent HTML title"""
        if len(reads_data) == 1:
            return f"Squiggy: {mode_name} (1 read - {reads_data[0][0]})"
        else:
            return f"Squiggy: {mode_name} ({len(reads_data)} reads)"

    @staticmethod
    def _calculate_base_regions_time_mode(
        sequence: str,
        seq_to_sig_map: List[int],
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

    @staticmethod
    def _calculate_base_regions_position_mode(
        base_annotations: List,
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

    @staticmethod
    def _add_dwell_time_patches(
        p, all_regions: List[dict], all_dwell_times: List[float]
    ):
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

    @staticmethod
    def _add_base_type_patches(p, base_regions: dict, base_colors: dict):
        """Add background patches grouped by base type"""
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
                    legend_label=f"Base {base}",
                )

    @staticmethod
    def _add_base_labels_time_mode(
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

    @staticmethod
    def _add_base_labels_position_mode(
        p,
        base_annotations: List,
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

    @staticmethod
    def _add_simple_labels(
        p,
        base_sources: List[Tuple[str, ColumnDataSource]],
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

    @staticmethod
    def plot_single_read(
        signal: np.ndarray,
        read_id: str,
        sample_rate: int,
        sequence: Optional[str] = None,
        seq_to_sig_map: Optional[List[int]] = None,
        normalization: NormalizationMethod = NormalizationMethod.NONE,
        downsample: int = 1,
        show_dwell_time: bool = False,
        show_labels: bool = True,
        show_signal_points: bool = False,
        position_label_interval: int = DEFAULT_POSITION_LABEL_INTERVAL,
        use_reference_positions: bool = False,
        theme: Theme = Theme.LIGHT,
    ) -> Tuple[str, figure]:
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

        # Normalize and downsample signal
        signal = SquigglePlotter.normalize_signal(signal, normalization)
        if downsample > 1:
            signal = signal[::downsample]
            if seq_to_sig_map is not None:
                seq_to_sig_map = [idx // downsample for idx in seq_to_sig_map]

        # Create time axis and figure with status information
        time_ms = np.arange(len(signal)) * 1000 / sample_rate
        reads_data = [(read_id, signal, sample_rate)]
        title = SquigglePlotter._format_plot_title(
            "Single", reads_data, normalization, downsample
        )
        p = SquigglePlotter._create_figure(
            title=title,
            x_label="Time (ms)",
            y_label=f"Signal ({normalization.value})",
            theme=theme,
        )

        # Add base annotations if available (returns color_mapper)
        color_mapper, _ = SquigglePlotter._add_base_annotations_single_read(
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

        # Add signal line and points
        line_renderer, signal_source = SquigglePlotter._add_signal_line(
            p, time_ms, signal, return_source=True, theme=theme
        )

        # Add signal points if requested
        renderers_for_hover = [line_renderer]
        if show_signal_points:
            circle_renderer = p.scatter(
                "time",
                "signal",
                source=signal_source,
                size=SIGNAL_POINT_SIZE,
                color=SIGNAL_POINT_COLOR,
                alpha=SIGNAL_POINT_ALPHA,
            )
            renderers_for_hover.append(circle_renderer)

        # Add hover tool
        hover = HoverTool(
            renderers=renderers_for_hover,
            tooltips=[
                ("Time", "@time{0.2f} ms"),
                ("Signal", "@signal{0.2f}"),
                ("Sample", "@sample"),
            ],
            mode="mouse",
            point_policy="snap_to_data",
        )
        p.add_tools(hover)

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

    @staticmethod
    def _add_base_annotations_single_read(
        p,
        signal: np.ndarray,
        time_ms: np.ndarray,
        sequence: Optional[str],
        seq_to_sig_map: Optional[List[int]],
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

        base_colors = SquigglePlotter._get_base_colors(theme)
        signal_min = np.min(signal)
        signal_max = np.max(signal)
        color_mapper = None

        if show_dwell_time:
            # Calculate and add dwell time patches
            all_regions, all_dwell_times, all_labels_data = (
                SquigglePlotter._calculate_base_regions_time_mode(
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
            color_mapper = SquigglePlotter._add_dwell_time_patches(
                p, all_regions, all_dwell_times
            )

            # Add labels if requested (always visible, no toggle)
            if show_labels:
                base_sources = SquigglePlotter._add_base_labels_time_mode(
                    p, all_labels_data, show_dwell_time, base_colors
                )
                SquigglePlotter._add_simple_labels(p, base_sources, base_colors, theme)
        else:
            # Calculate and add base type patches
            base_regions, base_labels_data = (
                SquigglePlotter._calculate_base_regions_time_mode(
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
            SquigglePlotter._add_base_type_patches(p, base_regions, base_colors)

            # Add labels if requested (always visible, no toggle)
            if show_labels:
                base_sources = SquigglePlotter._add_base_labels_time_mode(
                    p, base_labels_data, show_dwell_time, base_colors
                )
                SquigglePlotter._add_simple_labels(p, base_sources, base_colors, theme)

            p.legend.click_policy = "hide"
            p.legend.location = "top_right"

        return color_mapper, None

    @staticmethod
    def _add_signal_line(
        p,
        time_ms: np.ndarray,
        signal: np.ndarray,
        return_source: bool = False,
        theme: Theme = Theme.LIGHT,
    ):
        """Add signal line to plot

        Args:
            p: Bokeh figure
            time_ms: Time array in milliseconds
            signal: Signal array
            return_source: If True, return (line_renderer, signal_source) tuple

        Returns:
            line_renderer if return_source=False, else (line_renderer, signal_source)
        """
        signal_source = ColumnDataSource(
            data={"time": time_ms, "signal": signal, "sample": np.arange(len(signal))}
        )

        signal_color = SquigglePlotter._get_signal_line_color(theme)
        line_renderer = p.line(
            "time",
            "signal",
            source=signal_source,
            line_width=1,
            color=signal_color,
            alpha=0.8,
        )

        if return_source:
            return line_renderer, signal_source
        return line_renderer

    @staticmethod
    def plot_multiple_reads(
        reads_data: List[Tuple[str, np.ndarray, int]],
        mode: PlotMode,
        normalization: NormalizationMethod = NormalizationMethod.NONE,
        aligned_reads: Optional[List] = None,
        downsample: int = 1,
        show_dwell_time: bool = False,
        show_labels: bool = True,
        show_signal_points: bool = False,
        position_label_interval: int = DEFAULT_POSITION_LABEL_INTERVAL,
        use_reference_positions: bool = False,
        theme: Theme = Theme.LIGHT,
    ) -> Tuple[str, figure]:
        """
        Plot multiple reads in overlay or stacked mode

        Args:
            reads_data: List of (read_id, signal, sample_rate) tuples
            mode: Plot mode (OVERLAY or STACKED)
            normalization: Signal normalization method
            aligned_reads: Optional list of aligned read objects for EVENTALIGN mode
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
        if mode == PlotMode.OVERLAY:
            return SquigglePlotter._plot_overlay(
                reads_data, normalization, downsample, show_signal_points, theme
            )
        elif mode == PlotMode.STACKED:
            return SquigglePlotter._plot_stacked(
                reads_data, normalization, downsample, show_signal_points, theme
            )
        elif mode == PlotMode.EVENTALIGN:
            return SquigglePlotter._plot_eventalign(
                reads_data,
                normalization,
                aligned_reads,
                downsample,
                show_dwell_time,
                show_labels,
                show_signal_points,
                position_label_interval,
                use_reference_positions,
                theme,
            )
        else:
            raise ValueError(f"Unsupported plot mode: {mode}")

    @staticmethod
    def _plot_overlay(
        reads_data: List[Tuple[str, np.ndarray, int]],
        normalization: NormalizationMethod,
        downsample: int = 1,
        show_signal_points: bool = False,
        theme: Theme = Theme.LIGHT,
    ) -> Tuple[str, figure]:
        """Plot multiple reads overlaid on same axes"""
        # Create figure with status information
        title = SquigglePlotter._format_plot_title(
            "Overlay", reads_data, normalization, downsample
        )
        p = SquigglePlotter._create_figure(
            title=title,
            x_label="Sample",
            y_label=f"Signal ({normalization.value})",
            theme=theme,
        )

        line_renderers = []

        for idx, (read_id, signal, _sample_rate) in enumerate(reads_data):
            # Normalize and downsample signal
            signal = SquigglePlotter.normalize_signal(signal, normalization)
            if downsample > 1:
                signal = signal[::downsample]

            # Create data source and plot
            x = np.arange(len(signal))
            source = ColumnDataSource(
                data={"x": x, "y": signal, "read_id": [read_id] * len(signal)}
            )

            color = SquigglePlotter.MULTI_READ_COLORS[
                idx % len(SquigglePlotter.MULTI_READ_COLORS)
            ]
            line = p.line(
                "x",
                "y",
                source=source,
                line_width=1,
                color=color,
                alpha=0.7,
                legend_label=read_id[:12],
            )
            line_renderers.append(line)

            # Add signal points if requested
            if show_signal_points:
                circle = p.scatter(
                    "x",
                    "y",
                    source=source,
                    size=SIGNAL_POINT_SIZE,
                    color=SIGNAL_POINT_COLOR,
                    alpha=SIGNAL_POINT_ALPHA,
                    legend_label=read_id[:12],
                )
                line_renderers.append(circle)

        # Add hover tool
        hover = HoverTool(
            renderers=line_renderers,
            tooltips=[("Read", "@read_id"), ("Sample", "@x"), ("Signal", "@y{0.2f}")],
            mode="mouse",
            point_policy="snap_to_data",
        )
        p.add_tools(hover)

        p.legend.click_policy = "hide"
        p.legend.location = "top_right"

        # Generate HTML
        html_title = SquigglePlotter._format_html_title("Overlay", reads_data)
        html = file_html(p, CDN, title=html_title)
        return html, p

    @staticmethod
    def _plot_stacked(
        reads_data: List[Tuple[str, np.ndarray, int]],
        normalization: NormalizationMethod,
        downsample: int = 1,
        show_signal_points: bool = False,
        theme: Theme = Theme.LIGHT,
    ) -> Tuple[str, figure]:
        """Plot multiple reads stacked vertically with offset"""
        # Create figure with status information
        title = SquigglePlotter._format_plot_title(
            "Stacked", reads_data, normalization, downsample
        )
        p = SquigglePlotter._create_figure(
            title=title,
            x_label="Sample",
            y_label=f"Signal ({normalization.value}) + offset",
            theme=theme,
        )

        # First pass: normalize all signals and determine offset
        offset_step = 0
        normalized_signals = []
        for read_id, signal, sample_rate in reads_data:
            norm_signal = SquigglePlotter.normalize_signal(signal, normalization)
            if downsample > 1:
                norm_signal = norm_signal[::downsample]

            normalized_signals.append((read_id, norm_signal, sample_rate))
            signal_range = np.ptp(norm_signal)
            offset_step = max(offset_step, signal_range * 1.2)

        # Second pass: plot with offsets
        line_renderers = []
        for idx, (read_id, signal, _sample_rate) in enumerate(normalized_signals):
            offset = idx * offset_step
            x = np.arange(len(signal))
            y_offset = signal + offset
            source = ColumnDataSource(
                data={"x": x, "y": y_offset, "read_id": [read_id] * len(signal)}
            )

            color = SquigglePlotter.MULTI_READ_COLORS[
                idx % len(SquigglePlotter.MULTI_READ_COLORS)
            ]
            line = p.line(
                "x",
                "y",
                source=source,
                line_width=1,
                color=color,
                alpha=0.8,
                legend_label=read_id[:12],
            )
            line_renderers.append(line)

            # Add signal points if requested
            if show_signal_points:
                circle = p.scatter(
                    "x",
                    "y",
                    source=source,
                    size=SIGNAL_POINT_SIZE,
                    color=SIGNAL_POINT_COLOR,
                    alpha=SIGNAL_POINT_ALPHA,
                    legend_label=read_id[:12],
                )
                line_renderers.append(circle)

        # Add hover tool
        hover = HoverTool(
            renderers=line_renderers,
            tooltips=[("Read", "@read_id"), ("Sample", "@x"), ("Signal", "@y{0.2f}")],
            mode="mouse",
            point_policy="snap_to_data",
        )
        p.add_tools(hover)

        p.legend.click_policy = "hide"
        p.legend.location = "top_right"

        # Generate HTML
        html_title = SquigglePlotter._format_html_title("Stacked", reads_data)
        html = file_html(p, CDN, title=html_title)
        return html, p

    @staticmethod
    def _plot_eventalign(
        reads_data: List[Tuple[str, np.ndarray, int]],
        normalization: NormalizationMethod,
        aligned_reads: Optional[List],
        downsample: int = 1,
        show_dwell_time: bool = False,
        show_labels: bool = True,
        show_signal_points: bool = False,
        position_label_interval: int = DEFAULT_POSITION_LABEL_INTERVAL,
        use_reference_positions: bool = False,
        theme: Theme = Theme.LIGHT,
    ) -> Tuple[str, figure]:
        """Plot event-aligned reads with base annotations"""
        if not aligned_reads:
            raise ValueError("Event-aligned mode requires aligned_reads data")

        # Create figure with conditional x-axis label and status information
        title = SquigglePlotter._format_plot_title(
            "Event-Aligned", reads_data, normalization, downsample
        )
        x_label = "Time (ms)" if show_dwell_time else "Base Position"
        p = SquigglePlotter._create_figure(
            title=title,
            x_label=x_label,
            y_label=f"Signal ({normalization.value})",
            theme=theme,
        )

        # Add base annotations
        SquigglePlotter._add_base_annotations_eventalign(
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
        line_renderers = SquigglePlotter._plot_eventalign_signals(
            p,
            reads_data,
            normalization,
            aligned_reads,
            show_dwell_time,
            downsample,
            show_signal_points,
            theme,
        )

        # Add hover tool with conditional tooltip
        x_tooltip = (
            ("Time (ms)", "@x{0.2f}") if show_dwell_time else ("Base Position", "@x")
        )
        hover = HoverTool(
            renderers=line_renderers,
            tooltips=[
                ("Read", "@read_id"),
                x_tooltip,
                ("Base", "@base"),
                ("Signal", "@y{0.2f}"),
            ],
            mode="mouse",
            point_policy="snap_to_data",
        )
        p.add_tools(hover)

        p.legend.click_policy = "hide"
        p.legend.location = "top_right"

        # Generate HTML
        html_title = SquigglePlotter._format_html_title("Event-Aligned", reads_data)
        html = file_html(p, CDN, title=html_title)
        return html, p

    @staticmethod
    def _add_base_annotations_eventalign(
        p,
        reads_data: List[Tuple[str, np.ndarray, int]],
        normalization: NormalizationMethod,
        aligned_reads: List,
        show_dwell_time: bool,
        show_labels: bool,
        position_label_interval: int = DEFAULT_POSITION_LABEL_INTERVAL,
        use_reference_positions: bool = False,
        theme: Theme = Theme.LIGHT,
    ):
        """Add base annotations for event-aligned plots"""
        if not reads_data or not aligned_reads:
            return None

        base_colors = SquigglePlotter._get_base_colors(theme)
        first_aligned = aligned_reads[0]
        base_annotations = first_aligned.bases

        # Calculate signal range across all reads
        all_signals = [
            SquigglePlotter.normalize_signal(signal, normalization)
            for _, signal, _ in reads_data
        ]

        if not all_signals:
            return None

        signal_min = min(np.min(s) for s in all_signals)
        signal_max = max(np.max(s) for s in all_signals)
        sample_rate = reads_data[0][2]
        signal_length = len(all_signals[0])

        # Calculate and add base patches (time-scaled or position-based)
        (base_regions,) = SquigglePlotter._calculate_base_regions_position_mode(
            base_annotations,
            signal_min,
            signal_max,
            sample_rate,
            signal_length,
            show_dwell_time,
            base_colors,
        )
        SquigglePlotter._add_base_type_patches(p, base_regions, base_colors)

        # Add labels if requested
        if show_labels:
            SquigglePlotter._add_base_labels_position_mode(
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

    @staticmethod
    def _plot_eventalign_signals(
        p,
        reads_data: List[Tuple[str, np.ndarray, int]],
        normalization: NormalizationMethod,
        aligned_reads: List,
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
            signal = SquigglePlotter.normalize_signal(signal, normalization)
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
                                position_offset = -0.5 + (
                                    sample_offset / (num_samples - 1)
                                )
                            else:
                                position_offset = 0.0
                            signal_x.append(i + position_offset)
                            signal_y.append(signal[sample_idx])
                            signal_base_labels.append(base)

            # Create data source and plot
            source = ColumnDataSource(
                data={
                    "x": signal_x,
                    "y": signal_y,
                    "base": signal_base_labels,
                    "read_id": [read_id] * len(signal_x),
                }
            )

            # Use theme-aware signal color for first read, then use multi-read colors
            if idx == 0:
                color = SquigglePlotter._get_signal_line_color(theme)
            else:
                color = SquigglePlotter.MULTI_READ_COLORS[
                    idx % len(SquigglePlotter.MULTI_READ_COLORS)
                ]
            line = p.line(
                "x",
                "y",
                source=source,
                line_width=2,
                color=color,
                alpha=0.8,
                legend_label=read_id[:12],
            )
            line_renderers.append(line)

            # Add signal points if requested
            if show_signal_points:
                circle = p.scatter(
                    "x",
                    "y",
                    source=source,
                    size=SIGNAL_POINT_SIZE,
                    color=SIGNAL_POINT_COLOR,
                    alpha=SIGNAL_POINT_ALPHA,
                    legend_label=read_id[:12],
                )
                line_renderers.append(circle)

        return line_renderers
