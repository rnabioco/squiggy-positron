"""
Overlay plot strategy implementation

This module implements the Strategy Pattern for overlaying multiple nanopore reads
on the same axes.
"""

import numpy as np
from bokeh.embed import file_html
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.resources import CDN

from ..constants import NormalizationMethod, Theme
from ..normalization import normalize_signal
from ..theme_manager import ThemeManager
from .base import PlotStrategy


class OverlayPlotStrategy(PlotStrategy):
    """
    Strategy for overlaying multiple nanopore reads on same axes

    This strategy plots multiple reads with different colors on the same
    figure, allowing easy comparison of signal patterns across reads.

    Example:
        >>> from squiggy.plot_strategies.overlay import OverlayPlotStrategy
        >>> from squiggy.constants import Theme, NormalizationMethod
        >>>
        >>> strategy = OverlayPlotStrategy(Theme.LIGHT)
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

    # Color palette for multiple reads
    MULTI_READ_COLORS = [
        "#3c8dbc",  # Blue (matches SIGNAL_LINE_COLOR)
        "#d62728",  # Red
        "#2ca02c",  # Green
        "#ff7f0e",  # Orange
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
    ]

    def __init__(self, theme: Theme):
        """
        Initialize overlay plot strategy

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
        if 'reads' not in data:
            raise ValueError("Missing required data for overlay plot: reads")

        reads = data['reads']
        if not isinstance(reads, list):
            raise ValueError("reads must be a list of (read_id, signal, sample_rate) tuples")

        if len(reads) == 0:
            raise ValueError("reads list cannot be empty")

        # Validate each read tuple
        for idx, read_tuple in enumerate(reads):
            if not isinstance(read_tuple, tuple) or len(read_tuple) != 3:
                raise ValueError(
                    f"Read {idx} must be a tuple of (read_id, signal, sample_rate)"
                )

            read_id, signal, sample_rate = read_tuple
            if not isinstance(read_id, str):
                raise ValueError(f"Read {idx}: read_id must be a string")
            if not isinstance(signal, np.ndarray):
                raise ValueError(f"Read {idx}: signal must be a numpy array")
            if not isinstance(sample_rate, (int, float)):
                raise ValueError(f"Read {idx}: sample_rate must be a number")

    def create_plot(self, data: dict, options: dict) -> tuple[str, any]:
        """
        Generate Bokeh plot HTML and figure for overlaid reads

        Args:
            data: Plot data dictionary containing:
                - reads (required): list of (read_id, signal, sample_rate) tuples

            options: Plot options dictionary containing:
                - normalization: NormalizationMethod enum (default: NONE)
                - downsample: int downsampling factor (default: 1)
                - show_signal_points: bool show individual points (default: False)

        Returns:
            Tuple of (html_string, bokeh_figure)

        Raises:
            ValueError: If required data is missing
        """
        # Validate data
        self.validate_data(data)

        # Extract data
        reads_data = data['reads']

        # Extract options with defaults
        normalization = options.get('normalization', NormalizationMethod.NONE)
        downsample = options.get('downsample', 1)
        show_signal_points = options.get('show_signal_points', False)

        # Create figure
        title = self._format_title(reads_data, normalization, downsample)
        fig = self.theme_manager.create_figure(
            title=title,
            x_label="Sample",
            y_label=f"Signal ({normalization.value})",
            height=400,
        )

        # Plot each read with different color
        all_renderers = []
        for idx, (read_id, signal, sample_rate) in enumerate(reads_data):
            # Process signal
            processed_signal = self._process_signal(signal, normalization, downsample)

            # Create data source
            x = np.arange(len(processed_signal))
            source = ColumnDataSource(
                data={
                    "x": x,
                    "y": processed_signal,
                    "read_id": [read_id] * len(x),
                }
            )

            # Get color for this read
            color = self.MULTI_READ_COLORS[idx % len(self.MULTI_READ_COLORS)]

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
        fig.legend.click_policy = "hide"
        fig.legend.location = "top_right"

        # Generate HTML
        html_title = self._format_html_title(reads_data)
        html = file_html(fig, CDN, title=html_title)
        return html, fig

    # =========================================================================
    # Private Methods: Signal Processing
    # =========================================================================

    def _process_signal(
        self,
        signal: np.ndarray,
        normalization: NormalizationMethod,
        downsample: int,
    ) -> np.ndarray:
        """Process signal: normalize and downsample"""
        # Normalize
        if normalization != NormalizationMethod.NONE:
            signal = normalize_signal(signal, method=normalization)

        # Downsample
        if downsample > 1:
            signal = signal[::downsample]

        return signal

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
            alpha=0.7,
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
        parts = [f"Overlay: {len(reads_data)} reads"]

        if normalization != NormalizationMethod.NONE:
            parts.append(f"{normalization.value} normalized")

        if downsample > 1:
            parts.append(f"downsampled {downsample}x")

        return " | ".join(parts)

    def _format_html_title(self, reads_data: list) -> str:
        """Format HTML page title"""
        return f"Squiggy Overlay: {len(reads_data)} reads"
