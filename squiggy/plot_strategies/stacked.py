"""
Stacked plot strategy implementation

This module implements the Strategy Pattern for stacking multiple nanopore reads
vertically with offsets.
"""

import numpy as np
from bokeh.embed import file_html
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.resources import CDN

from ..constants import MULTI_READ_COLORS, NormalizationMethod, Theme
from ..rendering import ThemeManager
from .base import PlotStrategy


class StackedPlotStrategy(PlotStrategy):
    """
    Strategy for stacking multiple nanopore reads vertically

    This strategy plots multiple reads vertically offset to prevent overlap,
    making it easy to compare signals across many reads.

    Examples:
        >>> from squiggy.plot_strategies.stacked import StackedPlotStrategy
        >>> from squiggy.constants import Theme, NormalizationMethod
        >>>
        >>> strategy = StackedPlotStrategy(Theme.LIGHT)
        >>>
        >>> data = {
        ...     'reads': [
        ...         ('read_001', signal1, 4000),
        ...         ('read_002', signal2, 4000),
        ...         ('read_003', signal3, 4000),
        ...     ]
        ... }
        >>>
        >>> options = {
        ...     'normalization': NormalizationMethod.ZNORM,
        ...     'downsample': 5,
        ...     'show_signal_points': False,
        ... }
        >>>
        >>> html, fig = strategy.create_plot(data, options)
    """

    def __init__(self, theme: Theme):
        """
        Initialize stacked plot strategy

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
        """
        if "reads" not in data:
            raise ValueError("Missing required data for stacked plot: reads")

        # Validate read tuples
        reads = data["reads"]
        self._validate_read_tuples(reads)

    def create_plot(self, data: dict, options: dict) -> tuple[str, any]:
        """
        Generate Bokeh plot HTML and figure for stacked reads

        Args:
            data: Plot data dictionary containing:
                - reads (required): list of (read_id, signal, sample_rate) tuples
                - aligned_reads (optional): list of AlignedRead objects for sequence space

            options: Plot options dictionary containing:
                - normalization: NormalizationMethod enum (default: NONE)
                - downsample: int downsampling factor (default: 1)
                - show_signal_points: bool show individual points (default: False)
                - coordinate_space: str ('signal' or 'sequence', default: 'signal')

        Returns:
            Tuple of (html_string, bokeh_figure)

        Raises:
            ValueError: If required data is missing
        """
        # Validate data
        self.validate_data(data)

        # Extract data
        reads_data = data["reads"]
        aligned_reads = data.get("aligned_reads", None)

        from ..constants import DEFAULT_DOWNSAMPLE

        # Extract options with defaults
        normalization = options.get("normalization", NormalizationMethod.NONE)
        downsample = options.get("downsample", DEFAULT_DOWNSAMPLE)
        show_signal_points = options.get("show_signal_points", False)
        read_colors = options.get("read_colors", None)  # Optional: per-read colors
        coordinate_space = options.get("coordinate_space", "signal")

        # First pass: process all signals and determine offset
        processed_reads = []
        offset_step = 0

        for read_id, signal, sample_rate in reads_data:
            processed_signal, _ = self._process_signal(
                signal, normalization, downsample
            )
            processed_reads.append((read_id, processed_signal, sample_rate))

            # Calculate offset based on signal range
            signal_range = np.ptp(processed_signal)  # Peak-to-peak
            offset_step = max(offset_step, signal_range * 1.2)

        # Create figure
        title = self._format_title(reads_data, normalization, downsample)
        x_label = "Reference Position" if coordinate_space == "sequence" else "Sample"
        fig = self.theme_manager.create_figure(
            title=title,
            x_label=x_label,
            y_label=f"Signal ({normalization.value}) + offset",
            height=400,
        )

        # Second pass: plot with offsets
        all_renderers = []
        for idx, (read_id, signal, _sample_rate) in enumerate(processed_reads):
            # Apply vertical offset
            offset = idx * offset_step
            y_offset = signal + offset

            # Determine x-coordinates based on coordinate space
            if coordinate_space == "sequence" and aligned_reads:
                # Use BAM alignment positions
                aligned_read = aligned_reads[idx]
                # Extract query-to-reference mapping from move table
                if aligned_read.moves is None:
                    raise ValueError(
                        f"Read {read_id} has no move table. Cannot use sequence space."
                    )

                # Build position array: maps signal index -> reference position
                # moves table: 1 = base call (increment ref pos), 0 = stay (same ref pos)
                ref_positions = []
                current_ref_pos = aligned_read.query_alignment_start
                for move in aligned_read.moves:
                    ref_positions.append(current_ref_pos)
                    if move == 1:
                        current_ref_pos += 1

                # Ensure we have positions for all signal points
                ref_positions = np.array(ref_positions)
                if len(ref_positions) < len(signal):
                    # Pad with last position if needed
                    ref_positions = np.pad(
                        ref_positions,
                        (0, len(signal) - len(ref_positions)),
                        mode="edge",
                    )
                elif len(ref_positions) > len(signal):
                    # Truncate if too long
                    ref_positions = ref_positions[: len(signal)]

                x = ref_positions
            else:
                # Use raw sample indices (signal space)
                x = np.arange(len(signal))

            source = ColumnDataSource(
                data={
                    "x": x,
                    "y": y_offset,
                    "read_id": [read_id] * len(x),
                }
            )

            # Get color for this read - use read_colors if provided, otherwise cycle through defaults
            if read_colors and read_id in read_colors:
                color = read_colors[read_id]
            else:
                color = MULTI_READ_COLORS[idx % len(MULTI_READ_COLORS)]

            # Add renderers
            renderers = self._add_signal_renderers(
                fig=fig,
                source=source,
                color=color,
                show_signal_points=show_signal_points,
                legend_label=read_id[:12],  # Truncate long read IDs
            )
            all_renderers.extend(renderers)

        # Add hover tool
        hover = HoverTool(
            renderers=all_renderers,
            tooltips=[
                ("Read", "@read_id"),
                ("Sample", "@x"),
                ("Signal", "@y{0.2f}"),
            ],
            mode="mouse",
        )
        fig.add_tools(hover)

        # Configure legend
        self.theme_manager.configure_legend(fig)

        # Generate HTML
        html_title = self._format_html_title(reads_data)
        html = file_html(fig, CDN, title=html_title)
        return html, fig

    # =========================================================================
    # Private Methods: Signal Processing
    # =========================================================================

    # =========================================================================
    # Private Methods: Rendering
    # =========================================================================

    def _add_signal_renderers(
        self,
        fig,
        source: ColumnDataSource,
        color: str,
        show_signal_points: bool,
        legend_label: str,
    ) -> list:
        """Add signal line and optional points"""
        renderers = []

        # Add line
        line_renderer = fig.line(
            x="x",
            y="y",
            source=source,
            color=color,
            line_width=1,
            alpha=1.0,  # Full opacity for stacked (not overlapping)
            legend_label=legend_label,
        )
        renderers.append(line_renderer)

        # Add points if requested
        if show_signal_points:
            circle_renderer = fig.scatter(
                x="x",
                y="y",
                source=source,
                size=3,
                color=color,
                alpha=0.5,
                legend_label=legend_label,
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
            f"Stacked: {len(reads_data)} reads", normalization, downsample
        )

    def _format_html_title(self, reads_data: list) -> str:
        """Format HTML page title"""
        return self._build_html_title("Stacked", f"{len(reads_data)} reads")
