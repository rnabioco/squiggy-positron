"""Bokeh-based plotting for nanopore squiggle visualization"""

from typing import List, Optional, Tuple

import numpy as np
from bokeh.embed import file_html
from bokeh.layouts import column, row
from bokeh.models import (
    ColorBar,
    ColumnDataSource,
    CustomJS,
    HoverTool,
    LabelSet,
    LinearColorMapper,
    Toggle,
    WheelZoomTool,
)
from bokeh.transform import transform
from bokeh.plotting import figure
from bokeh.resources import CDN

from .constants import (
    BASE_ANNOTATION_ALPHA,
    BASE_COLORS,
    PLOT_DPI,
    PLOT_HEIGHT,
    PLOT_WIDTH,
    NormalizationMethod,
    PlotMode,
)


class BokehSquigglePlotter:
    """Bokeh-based plotter for nanopore squiggle visualization"""

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
    def plot_single_read(
        signal: np.ndarray,
        read_id: str,
        sample_rate: int,
        sequence: Optional[str] = None,
        seq_to_sig_map: Optional[List[int]] = None,
        normalization: NormalizationMethod = NormalizationMethod.NONE,
        downsample: int = 1,
        show_dwell_time: bool = False,
    ) -> str:
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

        Returns:
            HTML string containing the bokeh plot
        """
        # Normalize signal
        signal = BokehSquigglePlotter.normalize_signal(signal, normalization)

        # Apply downsampling if requested
        if downsample > 1:
            signal = signal[::downsample]
            # Adjust seq_to_sig_map if present
            if seq_to_sig_map is not None:
                seq_to_sig_map = [idx // downsample for idx in seq_to_sig_map]

        # Create time axis in milliseconds
        time_ms = np.arange(len(signal)) * 1000 / sample_rate

        # Create figure with responsive sizing
        p = figure(
            title=f"Read: {read_id}",
            x_axis_label="Time (ms)",
            y_axis_label=f"Signal ({normalization.value})",
            tools="pan,xbox_zoom,box_zoom,reset,save",
            active_drag="xbox_zoom",
            sizing_mode="stretch_both",
        )

        # Add x-only wheel zoom
        wheel_zoom = WheelZoomTool(dimensions="width")
        p.add_tools(wheel_zoom)
        p.toolbar.active_scroll = wheel_zoom

        # Add base annotations as background patches if available
        base_sources = []
        color_mapper = None
        if sequence and seq_to_sig_map is not None and len(seq_to_sig_map) > 0:
            # Calculate signal range for background patches
            signal_min = np.min(signal)
            signal_max = np.max(signal)

            if show_dwell_time:
                # Calculate dwell times for each base
                all_regions = []
                all_dwell_times = []
                all_labels_data = []

                for seq_pos in range(len(sequence)):
                    if seq_pos >= len(seq_to_sig_map):
                        break

                    base = sequence[seq_pos]
                    if base not in BASE_COLORS:
                        continue

                    sig_idx = seq_to_sig_map[seq_pos]
                    if sig_idx >= len(signal):
                        continue

                    # Calculate start time for this base
                    start_time = time_ms[sig_idx]

                    # Calculate end time and dwell time
                    if seq_pos + 1 < len(seq_to_sig_map):
                        next_sig_idx = seq_to_sig_map[seq_pos + 1]
                        if next_sig_idx < len(time_ms):
                            end_time = time_ms[next_sig_idx]
                        else:
                            end_time = time_ms[-1]
                        # Dwell time in milliseconds
                        dwell_time = end_time - start_time
                    else:
                        end_time = time_ms[-1]
                        dwell_time = end_time - start_time

                    # Store region data with dwell time
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

                if all_regions:
                    # Create color mapper for dwell time
                    dwell_array = np.array(all_dwell_times)
                    color_mapper = LinearColorMapper(
                        palette="Viridis256",
                        low=np.percentile(dwell_array, 5),
                        high=np.percentile(dwell_array, 95),
                    )

                    # Create datasource for all patches
                    patch_source = ColumnDataSource(
                        data={
                            "left": [r["left"] for r in all_regions],
                            "right": [r["right"] for r in all_regions],
                            "top": [r["top"] for r in all_regions],
                            "bottom": [r["bottom"] for r in all_regions],
                            "dwell": [r["dwell"] for r in all_regions],
                        }
                    )

                    # Draw background patches colored by dwell time
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

                    # Create single label source
                    if all_labels_data:
                        label_source = ColumnDataSource(
                            data={
                                "time": [d["time"] for d in all_labels_data],
                                "y": [d["y"] for d in all_labels_data],
                                "text": [d["text"] for d in all_labels_data],
                            }
                        )
                        base_sources.append(("all", label_source))

            else:
                # Group bases by type for legend (original behavior)
                base_regions = {base: [] for base in ["A", "C", "G", "T"]}
                base_labels_data = {base: [] for base in ["A", "C", "G", "T"]}

                # Process each base and calculate its region
                for seq_pos in range(len(sequence)):
                    if seq_pos >= len(seq_to_sig_map):
                        break

                    base = sequence[seq_pos]
                    if base not in BASE_COLORS:
                        continue

                    sig_idx = seq_to_sig_map[seq_pos]
                    if sig_idx >= len(signal):
                        continue

                    # Calculate start time for this base
                    start_time = time_ms[sig_idx]

                    # Calculate end time (use next base's start or signal end)
                    if seq_pos + 1 < len(seq_to_sig_map):
                        next_sig_idx = seq_to_sig_map[seq_pos + 1]
                        if next_sig_idx < len(time_ms):
                            end_time = time_ms[next_sig_idx]
                        else:
                            end_time = time_ms[-1]
                    else:
                        end_time = time_ms[-1]

                    # Store region data
                    base_regions[base].append(
                        {
                            "left": start_time,
                            "right": end_time,
                            "top": signal_max,
                            "bottom": signal_min,
                        }
                    )

                    # Store label data (center of region)
                    mid_time = (start_time + end_time) / 2
                    base_labels_data[base].append(
                        {"time": mid_time, "y": signal[sig_idx], "text": f"{base}{seq_pos}"}
                    )

                # Draw background patches for each base type
                for base in ["A", "C", "G", "T"]:
                    if base_regions[base]:
                        # Create datasource for quad patches
                        patch_source = ColumnDataSource(
                            data={
                                "left": [r["left"] for r in base_regions[base]],
                                "right": [r["right"] for r in base_regions[base]],
                                "top": [r["top"] for r in base_regions[base]],
                                "bottom": [r["bottom"] for r in base_regions[base]],
                            }
                        )

                        # Draw shaded background regions
                        p.quad(
                            left="left",
                            right="right",
                            top="top",
                            bottom="bottom",
                            source=patch_source,
                            color=BASE_COLORS[base],
                            alpha=BASE_ANNOTATION_ALPHA,
                            legend_label=f"Base {base}",
                        )

                        # Create datasource for labels
                        if base_labels_data[base]:
                            label_source = ColumnDataSource(
                                data={
                                    "time": [d["time"] for d in base_labels_data[base]],
                                    "y": [d["y"] for d in base_labels_data[base]],
                                    "text": [d["text"] for d in base_labels_data[base]],
                                }
                            )
                            base_sources.append((base, label_source))

        # Plot signal line on top of background patches
        signal_source = ColumnDataSource(
            data={"time": time_ms, "signal": signal, "sample": np.arange(len(signal))}
        )

        line_renderer = p.line(
            "time",
            "signal",
            source=signal_source,
            line_width=1,
            color="navy",
            alpha=0.8,
        )

        # Add hover tool - only for the signal line
        hover = HoverTool(
            renderers=[line_renderer],
            tooltips=[
                ("Time", "@time{0.2f} ms"),
                ("Signal", "@signal{0.2f}"),
                ("Sample", "@sample"),
            ],
            mode="mouse",
            point_policy="snap_to_data",
        )
        p.add_tools(hover)

        # Add base labels (if annotations were added)
        if base_sources:
            # Add base labels (initially hidden)
            for base, source in base_sources:
                labels = LabelSet(
                    x="time",
                    y="y",
                    text="text",
                    source=source,
                    text_font_size="10pt",
                    text_color=BASE_COLORS[base],
                    text_alpha=0.0,  # Initially hidden
                    name=f"labels_{base}",
                )
                p.add_layout(labels)

            # Add toggle button for base labels
            toggle_labels = Toggle(
                label="Show Base Labels", active=False, button_type="primary"
            )

            # JavaScript callback to toggle label visibility
            callback_code = """
            const alpha = cb_obj.active ? 0.8 : 0.0;
            for (const renderer of fig.renderers) {
                if (renderer.name && renderer.name.startsWith('labels_')) {
                    renderer.text_alpha = alpha;
                }
            }
            """
            toggle_labels.js_on_click(CustomJS(args={"fig": p}, code=callback_code))

            # Add automatic zoom-based label visibility
            # Show labels when zoomed in to < 5000 ms window
            zoom_callback_code = """
            const x_range = fig.x_range;
            const visible_range = x_range.end - x_range.start;
            const threshold = 5000;  // milliseconds

            // Only auto-show if toggle is not manually activated
            if (!toggle.active) {
                const alpha = visible_range < threshold ? 0.8 : 0.0;
                for (const renderer of fig.renderers) {
                    if (renderer.name && renderer.name.startsWith('labels_')) {
                        renderer.text_alpha = alpha;
                    }
                }
            }
            """
            p.x_range.js_on_change(
                "end",
                CustomJS(
                    args={"fig": p, "toggle": toggle_labels}, code=zoom_callback_code
                ),
            )

            if not show_dwell_time:
                p.legend.click_policy = "hide"
                p.legend.location = "top_right"

            # Return layout with controls (also responsive)
            layout = column(row(toggle_labels), p, sizing_mode="stretch_both")
        else:
            layout = p

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

        # Generate HTML string
        html = file_html(layout, CDN, title=f"Squiggy: {read_id}")
        return html

    @staticmethod
    def plot_multiple_reads(
        reads_data: List[Tuple[str, np.ndarray, int]],
        mode: PlotMode,
        normalization: NormalizationMethod = NormalizationMethod.NONE,
        aligned_reads: Optional[List] = None,
        downsample: int = 1,
        show_dwell_time: bool = False,
    ) -> str:
        """
        Plot multiple reads in overlay or stacked mode

        Args:
            reads_data: List of (read_id, signal, sample_rate) tuples
            mode: Plot mode (OVERLAY or STACKED)
            normalization: Signal normalization method
            aligned_reads: Optional list of aligned read objects for EVENTALIGN mode
            downsample: Downsampling factor (1 = no downsampling, 10 = every 10th point)
            show_dwell_time: Color bases by dwell time instead of base type

        Returns:
            HTML string containing the bokeh plot
        """
        if mode == PlotMode.OVERLAY:
            return BokehSquigglePlotter._plot_overlay(reads_data, normalization, downsample)
        elif mode == PlotMode.STACKED:
            return BokehSquigglePlotter._plot_stacked(reads_data, normalization, downsample)
        elif mode == PlotMode.EVENTALIGN:
            return BokehSquigglePlotter._plot_eventalign(
                reads_data, normalization, aligned_reads, downsample, show_dwell_time
            )
        else:
            raise ValueError(f"Unsupported plot mode: {mode}")

    @staticmethod
    def _plot_overlay(
        reads_data: List[Tuple[str, np.ndarray, int]],
        normalization: NormalizationMethod,
        downsample: int = 1,
    ) -> str:
        """Plot multiple reads overlaid on same axes"""
        # Create figure with responsive sizing
        p = figure(
            title=f"Overlay: {len(reads_data)} reads",
            x_axis_label="Sample",
            y_axis_label=f"Signal ({normalization.value})",
            tools="pan,xbox_zoom,box_zoom,reset,save",
            active_drag="xbox_zoom",
            sizing_mode="stretch_both",
        )

        # Add x-only wheel zoom
        wheel_zoom = WheelZoomTool(dimensions="width")
        p.add_tools(wheel_zoom)
        p.toolbar.active_scroll = wheel_zoom

        colors = ["navy", "red", "green", "orange", "purple", "brown", "pink", "gray"]
        line_renderers = []

        for idx, (read_id, signal, _sample_rate) in enumerate(reads_data):
            # Normalize signal
            signal = BokehSquigglePlotter.normalize_signal(signal, normalization)

            # Apply downsampling if requested
            if downsample > 1:
                signal = signal[::downsample]

            # Create data source
            x = np.arange(len(signal))
            source = ColumnDataSource(
                data={"x": x, "y": signal, "read_id": [read_id] * len(signal)}
            )

            # Plot line
            color = colors[idx % len(colors)]
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

        # Add hover tool - only for the signal lines
        hover = HoverTool(
            renderers=line_renderers,
            tooltips=[("Read", "@read_id"), ("Sample", "@x"), ("Signal", "@y{0.2f}")],
            mode="mouse",
            point_policy="snap_to_data",
        )
        p.add_tools(hover)

        p.legend.click_policy = "hide"
        p.legend.location = "top_right"

        # Generate HTML string
        html = file_html(p, CDN, title=f"Squiggy: Overlay ({len(reads_data)} reads)")
        return html

    @staticmethod
    def _plot_stacked(
        reads_data: List[Tuple[str, np.ndarray, int]],
        normalization: NormalizationMethod,
        downsample: int = 1,
    ) -> str:
        """Plot multiple reads stacked vertically with offset"""
        # Create figure with responsive sizing
        p = figure(
            title=f"Stacked: {len(reads_data)} reads",
            x_axis_label="Sample",
            y_axis_label=f"Signal ({normalization.value}) + offset",
            tools="pan,xbox_zoom,box_zoom,reset,save",
            active_drag="xbox_zoom",
            sizing_mode="stretch_both",
        )

        # Add x-only wheel zoom
        wheel_zoom = WheelZoomTool(dimensions="width")
        p.add_tools(wheel_zoom)
        p.toolbar.active_scroll = wheel_zoom

        colors = ["navy", "red", "green", "orange", "purple", "brown", "pink", "gray"]
        offset_step = 0  # Will be calculated based on signal range

        # First pass: normalize all signals and determine offset
        normalized_signals = []
        for read_id, signal, sample_rate in reads_data:
            norm_signal = BokehSquigglePlotter.normalize_signal(signal, normalization)

            # Apply downsampling if requested
            if downsample > 1:
                norm_signal = norm_signal[::downsample]

            normalized_signals.append((read_id, norm_signal, sample_rate))

            # Calculate offset based on signal range
            signal_range = np.ptp(norm_signal)
            offset_step = max(offset_step, signal_range * 1.2)

        # Second pass: plot with offsets
        line_renderers = []
        for idx, (read_id, signal, _sample_rate) in enumerate(normalized_signals):
            offset = idx * offset_step

            # Create data source
            x = np.arange(len(signal))
            y_offset = signal + offset
            source = ColumnDataSource(
                data={"x": x, "y": y_offset, "read_id": [read_id] * len(signal)}
            )

            # Plot line
            color = colors[idx % len(colors)]
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

        # Add hover tool - only for the signal lines
        hover = HoverTool(
            renderers=line_renderers,
            tooltips=[("Read", "@read_id"), ("Sample", "@x"), ("Signal", "@y{0.2f}")],
            mode="mouse",
            point_policy="snap_to_data",
        )
        p.add_tools(hover)

        p.legend.click_policy = "hide"
        p.legend.location = "top_right"

        # Generate HTML string
        html = file_html(p, CDN, title=f"Squiggy: Stacked ({len(reads_data)} reads)")
        return html

    @staticmethod
    def _plot_eventalign(
        reads_data: List[Tuple[str, np.ndarray, int]],
        normalization: NormalizationMethod,
        aligned_reads: Optional[List],
        downsample: int = 1,
        show_dwell_time: bool = False,
    ) -> str:
        """Plot event-aligned reads with base annotations"""
        if not aligned_reads:
            raise ValueError("Event-aligned mode requires aligned_reads data")

        # Create figure with responsive sizing
        p = figure(
            title=f"Event-Aligned: {len(reads_data)} reads",
            x_axis_label="Base Position",
            y_axis_label=f"Signal ({normalization.value})",
            tools="pan,xbox_zoom,box_zoom,reset,save",
            active_drag="xbox_zoom",
            sizing_mode="stretch_both",
        )

        # Add x-only wheel zoom
        wheel_zoom = WheelZoomTool(dimensions="width")
        p.add_tools(wheel_zoom)
        p.toolbar.active_scroll = wheel_zoom

        colors = ["navy", "red", "green", "orange", "purple", "brown", "pink", "gray"]
        color_mapper = None

        # First pass: add background patches for base annotations
        # We'll do this once based on the first read's base annotations
        if len(reads_data) > 0 and len(aligned_reads) > 0:
            first_aligned = aligned_reads[0]
            base_annotations = first_aligned.bases

            # Calculate signal range for background patches (across all reads)
            all_signals = []
            for _, signal, _ in reads_data:
                norm_signal = BokehSquigglePlotter.normalize_signal(signal, normalization)
                # Note: Not applying downsampling for range calculation in event-aligned mode
                all_signals.append(norm_signal)

            if all_signals:
                signal_min = min(np.min(s) for s in all_signals)
                signal_max = max(np.max(s) for s in all_signals)

                if show_dwell_time:
                    # Calculate dwell times based on signal differences between bases
                    all_regions = []
                    all_dwell_times = []

                    for i, base_annotation in enumerate(base_annotations):
                        base = base_annotation.base
                        if base not in BASE_COLORS:
                            continue

                        # Calculate dwell time from signal indices
                        if i + 1 < len(base_annotations):
                            next_annotation = base_annotations[i + 1]
                            dwell_samples = next_annotation.signal_start - base_annotation.signal_start
                        else:
                            # Last base - estimate from remaining signal
                            dwell_samples = len(all_signals[0]) - base_annotation.signal_start

                        # Convert to milliseconds (sample_rate from first read)
                        sample_rate = reads_data[0][2]
                        dwell_time = (dwell_samples / sample_rate) * 1000

                        # Store region data with dwell time
                        all_regions.append(
                            {
                                "left": i - 0.5,
                                "right": i + 0.5,
                                "top": signal_max,
                                "bottom": signal_min,
                                "dwell": dwell_time,
                            }
                        )
                        all_dwell_times.append(dwell_time)

                    if all_regions:
                        # Create color mapper for dwell time
                        dwell_array = np.array(all_dwell_times)
                        color_mapper = LinearColorMapper(
                            palette="Viridis256",
                            low=np.percentile(dwell_array, 5),
                            high=np.percentile(dwell_array, 95),
                        )

                        # Create datasource for all patches
                        patch_source = ColumnDataSource(
                            data={
                                "left": [r["left"] for r in all_regions],
                                "right": [r["right"] for r in all_regions],
                                "top": [r["top"] for r in all_regions],
                                "bottom": [r["bottom"] for r in all_regions],
                                "dwell": [r["dwell"] for r in all_regions],
                            }
                        )

                        # Draw background patches colored by dwell time
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

                else:
                    # Group bases by type for patches (original behavior)
                    base_regions = {base: [] for base in ["A", "C", "G", "T"]}

                    # Process each base position
                    for i, base_annotation in enumerate(base_annotations):
                        base = base_annotation.base
                        if base not in BASE_COLORS:
                            continue

                        # Calculate left and right boundaries (base position coordinates)
                        left = i - 0.5
                        right = i + 0.5

                        # Store region data
                        base_regions[base].append(
                            {
                                "left": left,
                                "right": right,
                                "top": signal_max,
                                "bottom": signal_min,
                            }
                        )

                    # Draw background patches for each base type
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

                            # Draw shaded background regions
                            p.quad(
                                left="left",
                                right="right",
                                top="top",
                                bottom="bottom",
                                source=patch_source,
                                color=BASE_COLORS[base],
                                alpha=BASE_ANNOTATION_ALPHA,
                                legend_label=f"Base {base}",
                            )

        # Second pass: plot signal lines on top of background patches
        line_renderers = []
        for idx, (read_id, signal, _sample_rate) in enumerate(reads_data):
            aligned_read = aligned_reads[idx]

            # Normalize signal
            signal = BokehSquigglePlotter.normalize_signal(signal, normalization)

            # Extract base-to-signal mapping
            base_annotations = aligned_read.bases

            # Create base-aligned x-coordinates
            base_x = []
            base_y = []
            base_labels = []

            # Note: Downsampling is not applied in event-aligned mode
            # because we're already plotting one point per base
            for i, base_annotation in enumerate(base_annotations):
                base = base_annotation.base
                start_idx = base_annotation.signal_start
                if start_idx < len(signal):
                    base_x.append(i)
                    base_y.append(signal[start_idx])
                    base_labels.append(base)

            # Create data source
            source = ColumnDataSource(
                data={
                    "x": base_x,
                    "y": base_y,
                    "base": base_labels,
                    "read_id": [read_id] * len(base_x),
                }
            )

            # Plot line on top of background patches
            color = colors[idx % len(colors)]
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

        # Add hover tool - only for the signal lines
        hover = HoverTool(
            renderers=line_renderers,
            tooltips=[
                ("Read", "@read_id"),
                ("Base Position", "@x"),
                ("Base", "@base"),
                ("Signal", "@y{0.2f}"),
            ],
            mode="mouse",
            point_policy="snap_to_data",
        )
        p.add_tools(hover)

        if not show_dwell_time:
            p.legend.click_policy = "hide"
            p.legend.location = "top_right"

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

        # Generate HTML string
        html = file_html(
            p, CDN, title=f"Squiggy: Event-Aligned ({len(reads_data)} reads)"
        )
        return html
