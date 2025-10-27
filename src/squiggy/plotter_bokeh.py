"""Bokeh-based plotting for nanopore squiggle visualization"""

from typing import List, Optional, Tuple

import numpy as np
from bokeh.embed import file_html
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource,
    CustomJS,
    HoverTool,
    LabelSet,
    Range1d,
    Toggle,
)
from bokeh.plotting import figure
from bokeh.resources import CDN

from .constants import (
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

        Returns:
            HTML string containing the bokeh plot
        """
        # Normalize signal
        signal = BokehSquigglePlotter.normalize_signal(signal, normalization)

        # Create time axis in milliseconds
        time_ms = np.arange(len(signal)) * 1000 / sample_rate

        # Create figure
        p = figure(
            width=int(PLOT_WIDTH * PLOT_DPI),
            height=int(PLOT_HEIGHT * PLOT_DPI),
            title=f"Read: {read_id}",
            x_axis_label="Time (ms)",
            y_axis_label=f"Signal ({normalization.value})",
            tools="pan,box_zoom,wheel_zoom,reset,save",
            active_scroll="wheel_zoom",
        )

        # Plot signal
        signal_source = ColumnDataSource(
            data=dict(time=time_ms, signal=signal, sample=np.arange(len(signal)))
        )

        p.line("time", "signal", source=signal_source, line_width=1, color="navy", alpha=0.8)

        # Add hover tool
        hover = HoverTool(
            tooltips=[
                ("Time", "@time{0.2f} ms"),
                ("Signal", "@signal{0.2f}"),
                ("Sample", "@sample"),
            ]
        )
        p.add_tools(hover)

        # Add base annotations if available
        if sequence and seq_to_sig_map:
            base_sources = []
            base_labels = []

            for base in ["A", "C", "G", "T"]:
                base_x = []
                base_y = []
                base_text = []
                base_time = []

                for i, (seq_pos, sig_idx) in enumerate(zip(range(len(sequence)), seq_to_sig_map)):
                    if sequence[seq_pos] == base and sig_idx < len(signal):
                        t = time_ms[sig_idx]
                        base_x.append(sig_idx)
                        base_time.append(t)
                        base_y.append(signal[sig_idx])
                        base_text.append(f"{base}{seq_pos}")

                if base_x:
                    source = ColumnDataSource(
                        data=dict(
                            x=base_x,
                            time=base_time,
                            y=base_y,
                            text=base_text,
                        )
                    )
                    base_sources.append((base, source))

                    # Plot base markers
                    p.circle(
                        "time",
                        "y",
                        source=source,
                        size=8,
                        color=BASE_COLORS[base],
                        alpha=0.6,
                        legend_label=f"Base {base}",
                    )

            # Add base labels (initially hidden)
            for base, source in base_sources:
                labels = LabelSet(
                    x="time",
                    y="y",
                    text="text",
                    source=source,
                    text_font_size="8pt",
                    text_color=BASE_COLORS[base],
                    text_alpha=0.0,  # Initially hidden
                    name=f"labels_{base}",
                )
                p.add_layout(labels)

            # Add toggle button for base labels
            toggle_labels = Toggle(label="Show Base Labels", active=False, button_type="primary")

            # JavaScript callback to toggle label visibility
            callback_code = """
            const alpha = cb_obj.active ? 0.8 : 0.0;
            for (const renderer of fig.renderers) {
                if (renderer.name && renderer.name.startsWith('labels_')) {
                    renderer.text_alpha = alpha;
                }
            }
            """
            toggle_labels.js_on_click(
                CustomJS(args=dict(fig=p), code=callback_code)
            )

            p.legend.click_policy = "hide"
            p.legend.location = "top_right"

            # Return layout with controls
            layout = column(row(toggle_labels), p)
        else:
            layout = p

        # Generate HTML string
        html = file_html(layout, CDN, title=f"Squiggy: {read_id}")
        return html

    @staticmethod
    def plot_multiple_reads(
        reads_data: List[Tuple[str, np.ndarray, int]],
        mode: PlotMode,
        normalization: NormalizationMethod = NormalizationMethod.NONE,
        aligned_reads: Optional[List] = None,
    ) -> str:
        """
        Plot multiple reads in overlay or stacked mode

        Args:
            reads_data: List of (read_id, signal, sample_rate) tuples
            mode: Plot mode (OVERLAY or STACKED)
            normalization: Signal normalization method
            aligned_reads: Optional list of aligned read objects for EVENTALIGN mode

        Returns:
            HTML string containing the bokeh plot
        """
        if mode == PlotMode.OVERLAY:
            return BokehSquigglePlotter._plot_overlay(reads_data, normalization)
        elif mode == PlotMode.STACKED:
            return BokehSquigglePlotter._plot_stacked(reads_data, normalization)
        elif mode == PlotMode.EVENTALIGN:
            return BokehSquigglePlotter._plot_eventalign(reads_data, normalization, aligned_reads)
        else:
            raise ValueError(f"Unsupported plot mode: {mode}")

    @staticmethod
    def _plot_overlay(
        reads_data: List[Tuple[str, np.ndarray, int]],
        normalization: NormalizationMethod,
    ) -> str:
        """Plot multiple reads overlaid on same axes"""
        # Create figure
        p = figure(
            width=int(PLOT_WIDTH * PLOT_DPI),
            height=int(PLOT_HEIGHT * PLOT_DPI),
            title=f"Overlay: {len(reads_data)} reads",
            x_axis_label="Sample",
            y_axis_label=f"Signal ({normalization.value})",
            tools="pan,box_zoom,wheel_zoom,reset,save",
            active_scroll="wheel_zoom",
        )

        colors = ["navy", "red", "green", "orange", "purple", "brown", "pink", "gray"]

        for idx, (read_id, signal, sample_rate) in enumerate(reads_data):
            # Normalize signal
            signal = BokehSquigglePlotter.normalize_signal(signal, normalization)

            # Create data source
            x = np.arange(len(signal))
            source = ColumnDataSource(data=dict(x=x, y=signal, read_id=[read_id] * len(signal)))

            # Plot line
            color = colors[idx % len(colors)]
            p.line(
                "x",
                "y",
                source=source,
                line_width=1,
                color=color,
                alpha=0.7,
                legend_label=read_id[:12],
            )

        # Add hover tool
        hover = HoverTool(tooltips=[("Read", "@read_id"), ("Sample", "@x"), ("Signal", "@y{0.2f}")])
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
    ) -> str:
        """Plot multiple reads stacked vertically with offset"""
        # Create figure
        p = figure(
            width=int(PLOT_WIDTH * PLOT_DPI),
            height=int(PLOT_HEIGHT * PLOT_DPI),
            title=f"Stacked: {len(reads_data)} reads",
            x_axis_label="Sample",
            y_axis_label=f"Signal ({normalization.value}) + offset",
            tools="pan,box_zoom,wheel_zoom,reset,save",
            active_scroll="wheel_zoom",
        )

        colors = ["navy", "red", "green", "orange", "purple", "brown", "pink", "gray"]
        offset_step = 0  # Will be calculated based on signal range

        # First pass: normalize all signals and determine offset
        normalized_signals = []
        for read_id, signal, sample_rate in reads_data:
            norm_signal = BokehSquigglePlotter.normalize_signal(signal, normalization)
            normalized_signals.append((read_id, norm_signal, sample_rate))

            # Calculate offset based on signal range
            signal_range = np.ptp(norm_signal)
            offset_step = max(offset_step, signal_range * 1.2)

        # Second pass: plot with offsets
        for idx, (read_id, signal, sample_rate) in enumerate(normalized_signals):
            offset = idx * offset_step

            # Create data source
            x = np.arange(len(signal))
            y_offset = signal + offset
            source = ColumnDataSource(
                data=dict(x=x, y=y_offset, read_id=[read_id] * len(signal))
            )

            # Plot line
            color = colors[idx % len(colors)]
            p.line(
                "x",
                "y",
                source=source,
                line_width=1,
                color=color,
                alpha=0.8,
                legend_label=read_id[:12],
            )

        # Add hover tool
        hover = HoverTool(tooltips=[("Read", "@read_id"), ("Sample", "@x"), ("Signal", "@y{0.2f}")])
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
    ) -> str:
        """Plot event-aligned reads with base annotations"""
        if not aligned_reads:
            raise ValueError("Event-aligned mode requires aligned_reads data")

        # Create figure
        p = figure(
            width=int(PLOT_WIDTH * PLOT_DPI),
            height=int(PLOT_HEIGHT * PLOT_DPI),
            title=f"Event-Aligned: {len(reads_data)} reads",
            x_axis_label="Base Position",
            y_axis_label=f"Signal ({normalization.value})",
            tools="pan,box_zoom,wheel_zoom,reset,save",
            active_scroll="wheel_zoom",
        )

        colors = ["navy", "red", "green", "orange", "purple", "brown", "pink", "gray"]

        for idx, (read_id, signal, sample_rate) in enumerate(reads_data):
            aligned_read = aligned_reads[idx]

            # Normalize signal
            signal = BokehSquigglePlotter.normalize_signal(signal, normalization)

            # Extract base-to-signal mapping
            bases = aligned_read.bases
            move_starts = aligned_read.move_starts

            # Create base-aligned x-coordinates
            base_x = []
            base_y = []
            base_labels = []

            for i, (base, start_idx) in enumerate(zip(bases, move_starts)):
                if start_idx < len(signal):
                    base_x.append(i)
                    base_y.append(signal[start_idx])
                    base_labels.append(base)

            # Create data source
            source = ColumnDataSource(
                data=dict(
                    x=base_x,
                    y=base_y,
                    base=base_labels,
                    read_id=[read_id] * len(base_x),
                )
            )

            # Plot line
            color = colors[idx % len(colors)]
            p.line(
                "x",
                "y",
                source=source,
                line_width=2,
                color=color,
                alpha=0.8,
                legend_label=read_id[:12],
            )

            # Add base markers colored by base type
            for base_char in ["A", "C", "G", "T"]:
                base_indices = [i for i, b in enumerate(base_labels) if b == base_char]
                if base_indices:
                    base_source = ColumnDataSource(
                        data=dict(
                            x=[base_x[i] for i in base_indices],
                            y=[base_y[i] for i in base_indices],
                            base=[base_labels[i] for i in base_indices],
                        )
                    )
                    p.circle(
                        "x",
                        "y",
                        source=base_source,
                        size=6,
                        color=BASE_COLORS[base_char],
                        alpha=0.6,
                    )

        # Add hover tool
        hover = HoverTool(
            tooltips=[("Read", "@read_id"), ("Base Position", "@x"), ("Base", "@base"), ("Signal", "@y{0.2f}")]
        )
        p.add_tools(hover)

        p.legend.click_policy = "hide"
        p.legend.location = "top_right"

        # Generate HTML string
        html = file_html(p, CDN, title=f"Squiggy: Event-Aligned ({len(reads_data)} reads)")
        return html
