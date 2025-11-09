"""
Event-aligned plot strategy implementation

This module implements the Strategy Pattern for event-aligned nanopore reads
with base annotations.
"""

import numpy as np
from bokeh.embed import file_html
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.resources import CDN

from ..constants import (
    DEFAULT_POSITION_LABEL_INTERVAL,
    MULTI_READ_COLORS,
    NormalizationMethod,
    Theme,
)
from ..rendering import BaseAnnotationRenderer, ReferenceTrackRenderer, ThemeManager
from .base import PlotStrategy


class EventAlignPlotStrategy(PlotStrategy):
    """
    Strategy for plotting event-aligned nanopore reads

    This strategy plots reads aligned to their basecalls, showing the
    relationship between signal and sequence. Supports both base position
    and cumulative dwell time x-axes.

    Examples:
        >>> from squiggy.plot_strategies.eventalign import EventAlignPlotStrategy
        >>> from squiggy.constants import Theme, NormalizationMethod
        >>>
        >>> strategy = EventAlignPlotStrategy(Theme.LIGHT)
        >>>
        >>> data = {
        ...     'reads': [
        ...         ('read_001', signal1, 4000),
        ...         ('read_002', signal2, 4000),
        ...     ],
        ...     'aligned_reads': [aligned1, aligned2],
        ... }
        >>>
        >>> options = {
        ...     'normalization': NormalizationMethod.ZNORM,
        ...     'show_dwell_time': False,
        ...     'show_labels': True,
        ... }
        >>>
        >>> html, fig = strategy.create_plot(data, options)
    """

    def __init__(self, theme: Theme):
        """
        Initialize event-aligned plot strategy

        Args:
            theme: Theme enum (LIGHT or DARK)
        """
        super().__init__(theme)
        self.theme_manager = ThemeManager(theme)

    def validate_data(self, data: dict) -> None:
        """
        Validate that required data is present

        Args:
            data: Plot data dictionary

        Raises:
            ValueError: If required keys are missing

        Required keys:
            - reads: list of (read_id, signal, sample_rate) tuples
            - aligned_reads: list of AlignedRead objects with .bases attribute
        """
        if "reads" not in data:
            raise ValueError("Missing required data for event-aligned plot: reads")

        if "aligned_reads" not in data:
            raise ValueError(
                "Missing required data for event-aligned plot: aligned_reads"
            )

        reads = data["reads"]
        aligned_reads = data["aligned_reads"]

        if not isinstance(reads, list) or len(reads) == 0:
            raise ValueError("reads must be a non-empty list")

        if not isinstance(aligned_reads, list) or len(aligned_reads) == 0:
            raise ValueError("aligned_reads must be a non-empty list")

        # Validate read tuples
        self._validate_read_tuples(reads)

        if len(reads) != len(aligned_reads):
            raise ValueError(
                f"reads and aligned_reads must have same length "
                f"(got {len(reads)} and {len(aligned_reads)})"
            )

        # Validate aligned reads have bases
        for idx, aligned_read in enumerate(aligned_reads):
            if not hasattr(aligned_read, "bases"):
                raise ValueError(f"Aligned read {idx} must have 'bases' attribute")

    def create_plot(self, data: dict, options: dict) -> tuple[str, any]:
        """
        Generate Bokeh plot HTML and figure for event-aligned reads

        Args:
            data: Plot data dictionary containing:
                - reads (required): list of (read_id, signal, sample_rate) tuples
                - aligned_reads (required): list of AlignedRead objects

            options: Plot options dictionary containing:
                - normalization: NormalizationMethod enum (default: NONE)
                - downsample: int downsampling factor (default: 1)
                - show_dwell_time: bool use time x-axis (default: False)
                - show_labels: bool show base labels (default: True)
                - show_signal_points: bool show individual points (default: False)
                - position_label_interval: int label interval (default: 50)

        Returns:
            Tuple of (html_string, bokeh_figure)

        Raises:
            ValueError: If required data is missing
        """
        # Validate data
        self.validate_data(data)

        # Extract data
        reads_data = data["reads"]
        aligned_reads = data["aligned_reads"]
        reference_sequence = data.get("reference_sequence", "")

        from ..constants import DEFAULT_DOWNSAMPLE

        # Extract options with defaults
        normalization = options.get("normalization", NormalizationMethod.NONE)
        downsample = options.get("downsample", DEFAULT_DOWNSAMPLE)
        show_dwell_time = options.get("show_dwell_time", False)
        show_labels = options.get("show_labels", True)
        show_signal_points = options.get("show_signal_points", False)
        position_label_interval = options.get(
            "position_label_interval", DEFAULT_POSITION_LABEL_INTERVAL
        )
        clip_x_to_alignment = options.get("clip_x_to_alignment", True)

        # Create figure
        title = self._format_title(reads_data, normalization, downsample)
        x_label = "Time (ms)" if show_dwell_time else "Base Position"
        fig = self.theme_manager.create_figure(
            title=title,
            x_label=x_label,
            y_label=f"Signal ({normalization.value})",
            height=400,
        )

        # Process all signals first to get global min/max
        all_processed = []
        for read_id, signal, sample_rate in reads_data:
            processed, _ = self._process_signal(signal, normalization)
            all_processed.append((read_id, processed, sample_rate))

        signal_min = min(np.min(s) for _, s, _ in all_processed)
        signal_max = max(np.max(s) for _, s, _ in all_processed)

        # Add base annotations using first aligned read
        first_aligned = aligned_reads[0]
        sample_rate = reads_data[0][2]
        signal_length = len(all_processed[0][1])

        self._add_base_annotations(
            fig=fig,
            base_annotations=first_aligned.bases,
            sample_rate=sample_rate,
            signal_length=signal_length,
            signal_min=signal_min,
            signal_max=signal_max,
            show_dwell_time=show_dwell_time,
            show_labels=show_labels,
            position_label_interval=position_label_interval,
        )

        # Plot signals
        all_renderers = []
        for idx, ((read_id, signal, sample_rate), aligned_read) in enumerate(
            zip(all_processed, aligned_reads, strict=False)
        ):
            renderers = self._plot_aligned_signal(
                fig=fig,
                read_id=read_id,
                signal=signal,
                sample_rate=sample_rate,
                base_annotations=aligned_read.bases,
                show_dwell_time=show_dwell_time,
                downsample=downsample,
                show_signal_points=show_signal_points,
                color_index=idx,
            )
            all_renderers.extend(renderers)

        # Add hover tool
        x_tooltip = (
            ("Time (ms)", "@x{0.2f}") if show_dwell_time else ("Base Position", "@x")
        )
        hover = HoverTool(
            renderers=all_renderers,
            tooltips=[
                ("Read", "@read_id"),
                x_tooltip,
                ("Base", "@base"),
                ("Signal", "@y{0.2f}"),
            ],
            mode="mouse",
        )
        fig.add_tools(hover)

        # Apply x-axis clipping if requested
        if clip_x_to_alignment and len(first_aligned.bases) > 0:
            from bokeh.models import Range1d

            # X-axis is in base position units (0, 1, 2, ...) or time units
            num_bases = len(first_aligned.bases)
            if show_dwell_time:
                # For dwell time mode, calculate total dwell time
                # Only calculate if bases have dwell_time attribute
                try:
                    total_dwell = sum(base.dwell_time for base in first_aligned.bases)
                    fig.x_range = Range1d(start=-0.5, end=total_dwell + 0.5)
                except AttributeError:
                    # If dwell_time not available, fall back to position mode
                    fig.x_range = Range1d(start=-0.5, end=num_bases - 0.5)
            else:
                # For position mode, set range to [0, num_bases]
                fig.x_range = Range1d(start=-0.5, end=num_bases - 0.5)

        # Configure legend
        self.theme_manager.configure_legend(fig)

        # Create reference track if reference sequence available
        ref_fig = None
        if reference_sequence and len(first_aligned.bases) > 0:
            # Create reference track
            ref_renderer = ReferenceTrackRenderer(self.theme_manager)

            # Get positions for reference bases
            positions = list(range(len(first_aligned.bases)))

            # Get query sequence for mismatch highlighting
            query_sequence = "".join([base.base for base in first_aligned.bases])

            try:
                ref_fig = ref_renderer.create_reference_track(
                    reference_sequence=reference_sequence,
                    positions=positions,
                    x_label="",  # No x-label (shared with main plot)
                    title="Reference",
                    height=60,
                    query_sequence=query_sequence,  # Enable mismatch highlighting
                )

                # Link x-range for synchronized zoom/pan
                ref_fig.x_range = fig.x_range

                # Minimize borders
                ref_fig.min_border_top = 0
                ref_fig.min_border_bottom = 0
                ref_fig.min_border_left = 5
                ref_fig.min_border_right = 5

            except Exception:
                # If reference track creation fails, continue without it
                ref_fig = None

        # Generate HTML
        html_title = self._format_html_title(reads_data)

        if ref_fig is not None:
            # Create column layout with reference track above signal
            from bokeh.layouts import column

            fig.min_border_top = 0
            fig.min_border_left = 5
            fig.min_border_right = 5

            layout = column(
                ref_fig,
                fig,
                sizing_mode="stretch_width",
                spacing=0,
            )

            # Store reference to main plot
            object.__setattr__(layout, "main_plot", fig)

            html = file_html(layout, CDN, title=html_title)
            return html, layout
        else:
            # No reference track - return single plot
            html = file_html(fig, CDN, title=html_title)
            return html, fig

    # =========================================================================
    # Private Methods: Signal Processing
    # =========================================================================

    # =========================================================================
    # Private Methods: Rendering
    # =========================================================================

    def _add_base_annotations(
        self,
        fig,
        base_annotations: list,
        sample_rate: int,
        signal_length: int,
        signal_min: float,
        signal_max: float,
        show_dwell_time: bool,
        show_labels: bool,
        position_label_interval: int,
    ):
        """Add base annotations using BaseAnnotationRenderer"""
        base_colors = self.theme_manager.get_base_colors()

        renderer = BaseAnnotationRenderer(
            base_colors=base_colors,
            show_dwell_time=show_dwell_time,
            show_labels=show_labels,
        )

        renderer.render_position_based(
            fig=fig,
            base_annotations=base_annotations,
            sample_rate=sample_rate,
            signal_length=signal_length,
            signal_min=signal_min,
            signal_max=signal_max,
            position_label_interval=position_label_interval,
            theme=self.theme,
        )

    def _plot_aligned_signal(
        self,
        fig,
        read_id: str,
        signal: np.ndarray,
        sample_rate: int,
        base_annotations: list,
        show_dwell_time: bool,
        downsample: int,
        show_signal_points: bool,
        color_index: int,
    ) -> list:
        """Plot signal for one aligned read"""
        # Create signal coordinates
        signal_x = []
        signal_y = []
        signal_base_labels = []

        if show_dwell_time:
            # Use cumulative time for x-axis
            cumulative_time = 0.0

            for i, base_annotation in enumerate(base_annotations):
                base = base_annotation.base
                start_idx = base_annotation.signal_start

                # Determine end index
                if i + 1 < len(base_annotations):
                    end_idx = base_annotations[i + 1].signal_start
                else:
                    end_idx = len(signal)

                dwell_samples = end_idx - start_idx
                dwell_time = (dwell_samples / sample_rate) * 1000  # ms

                # Plot signal samples with downsampling
                for sample_offset in range(0, dwell_samples, downsample):
                    sample_idx = start_idx + sample_offset
                    if sample_idx < len(signal):
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
            # Use base position for x-axis
            for i, base_annotation in enumerate(base_annotations):
                base = base_annotation.base
                start_idx = base_annotation.signal_start

                # Determine end index
                if i + 1 < len(base_annotations):
                    end_idx = base_annotations[i + 1].signal_start
                else:
                    end_idx = len(signal)

                num_samples = end_idx - start_idx

                # Plot signal samples with downsampling
                for sample_offset in range(0, num_samples, downsample):
                    sample_idx = start_idx + sample_offset
                    if sample_idx < len(signal):
                        # Map to position: evenly distribute from i-0.5 to i+0.5
                        if num_samples > 1:
                            position_offset = -0.5 + (sample_offset / (num_samples - 1))
                        else:
                            position_offset = 0.0
                        signal_x.append(i + position_offset)
                        signal_y.append(signal[sample_idx])
                        signal_base_labels.append(base)

        # Create data source
        source = ColumnDataSource(
            data={
                "x": np.array(signal_x),
                "y": np.array(signal_y),
                "read_id": [read_id] * len(signal_x),
                "base": signal_base_labels,
            }
        )

        # Get color
        if color_index == 0:
            color = self.theme_manager.get_signal_color()
        else:
            color = MULTI_READ_COLORS[color_index % len(MULTI_READ_COLORS)]

        # Add renderers
        renderers = []

        line_renderer = fig.line(
            x="x",
            y="y",
            source=source,
            color=color,
            line_width=2,
            legend_label=read_id[:12],
        )
        renderers.append(line_renderer)

        if show_signal_points:
            circle_renderer = fig.scatter(
                x="x",
                y="y",
                source=source,
                size=3,
                color=color,
                alpha=0.5,
                legend_label=read_id[:12],
            )
            renderers.append(circle_renderer)

        return renderers

    # =========================================================================
    # Private Methods: Utilities
    # =========================================================================

    def _format_title(
        self,
        reads_data: list,
        normalization: NormalizationMethod,
        downsample: int,
    ) -> str:
        """Format plot title"""
        return self._build_title(
            f"Event-Aligned: {len(reads_data)} reads", normalization, downsample
        )

    def _format_html_title(self, reads_data: list) -> str:
        """Format HTML page title"""
        return self._build_html_title("Event-Aligned", f"{len(reads_data)} reads")
