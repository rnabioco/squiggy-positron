"""
Delta Plot Strategy for comparing two samples

This strategy visualizes differences (deltas) between aggregate statistics
from two samples. It creates a comparison plot showing how sample B differs
from sample A across reference positions.
"""

from typing import Any

import numpy as np
from bokeh.layouts import gridplot
from bokeh.models import Band, ColumnDataSource, HoverTool, NumeralTickFormatter

from ..constants import (
    DELTA_BAND_ALPHA,
    DELTA_LINE_WIDTH,
    DELTA_NEGATIVE_COLOR,
    DELTA_NEUTRAL_COLOR,
    DELTA_POSITIVE_COLOR,
    DELTA_SIGNAL_HEIGHT,
    DELTA_STATS_HEIGHT,
    DELTA_ZERO_LINE_COLOR,
    NormalizationMethod,
    Theme,
)
from ..normalization import normalize_signal
from ..rendering.theme_manager import ThemeManager
from .base import PlotStrategy


class DeltaPlotStrategy(PlotStrategy):
    """
    Plot strategy for comparing delta statistics between two samples

    Creates a visualization of the differences (deltas) between aggregate
    statistics from sample B and sample A. Includes:
    - Delta signal track showing mean signal differences
    - Delta statistics track showing coverage and quality differences
    - Synchronized x-axes for linked zoom/pan

    The delta is computed as: B - A for each statistic.

    Attributes:
        theme: Theme enum (LIGHT or DARK) for plot styling

    Examples:
        >>> from squiggy.plot_strategies.delta import DeltaPlotStrategy
        >>> from squiggy.constants import Theme, NormalizationMethod
        >>>
        >>> strategy = DeltaPlotStrategy(Theme.LIGHT)
        >>> data = {
        ...     'positions': np.array([0, 1, 2, ...]),
        ...     'delta_mean_signal': np.array([5.0, -3.2, 1.5, ...]),
        ...     'delta_std_signal': np.array([0.5, 0.6, 0.4, ...]),
        ...     'sample_a_coverage': [10, 10, 10, ...],
        ...     'sample_b_coverage': [10, 10, 10, ...],
        ...     'sample_a_name': 'model_v4.2',
        ...     'sample_b_name': 'model_v5.0',
        ... }
        >>> options = {'normalization': NormalizationMethod.NONE}
        >>> html, fig = strategy.create_plot(data, options)
    """

    def __init__(self, theme: Theme):
        """
        Initialize delta plot strategy

        Args:
            theme: Theme enum (LIGHT or DARK)
        """
        super().__init__(theme)
        self.theme_manager = ThemeManager(theme)

    def validate_data(self, data: dict[str, Any]) -> None:
        """
        Validate that required delta comparison data is present

        Required keys:
        - positions: np.ndarray of reference positions
        - delta_mean_signal: np.ndarray of delta mean signals (B - A)
        - delta_std_signal: np.ndarray of delta std signals (B - A)
        - sample_a_name: str, name of sample A
        - sample_b_name: str, name of sample B
        - sample_a_coverage: list[int], coverage per position for sample A
        - sample_b_coverage: list[int], coverage per position for sample B

        Args:
            data: Plot data dictionary to validate

        Raises:
            ValueError: If required data is missing
        """
        required = [
            "positions",
            "delta_mean_signal",
            "delta_std_signal",
            "sample_a_name",
            "sample_b_name",
            "sample_a_coverage",
            "sample_b_coverage",
        ]
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"Missing required delta data: {missing}")

        # Validate array shapes match
        positions = data["positions"]
        delta_mean = data["delta_mean_signal"]
        delta_std = data["delta_std_signal"]

        if len(delta_mean) != len(positions):
            raise ValueError(
                f"delta_mean_signal length ({len(delta_mean)}) must match "
                f"positions length ({len(positions)})"
            )

        if len(delta_std) != len(positions):
            raise ValueError(
                f"delta_std_signal length ({len(delta_std)}) must match "
                f"positions length ({len(positions)})"
            )

        # Validate names are strings
        if not isinstance(data["sample_a_name"], str):
            raise ValueError("sample_a_name must be a string")
        if not isinstance(data["sample_b_name"], str):
            raise ValueError("sample_b_name must be a string")

    def create_plot(
        self, data: dict[str, Any], options: dict[str, Any]
    ) -> tuple[str, Any]:
        """
        Generate delta comparison plot

        Creates two synchronized tracks:
        1. Delta Signal Track: Mean delta signal with confidence bands
        2. Delta Stats Track: Coverage differences and statistics

        Args:
            data: Delta comparison data dictionary
            options: Plot options (normalization, downsample, etc.)

        Returns:
            Tuple of (html_string, bokeh_gridplot)
        """
        self.validate_data(data)

        positions = data["positions"]
        delta_mean = data["delta_mean_signal"]
        delta_std = data["delta_std_signal"]
        sample_a_name = data["sample_a_name"]
        sample_b_name = data["sample_b_name"]
        coverage_a = data.get("sample_a_coverage", [1] * len(positions))
        coverage_b = data.get("sample_b_coverage", [1] * len(positions))

        from ..constants import DEFAULT_DOWNSAMPLE

        # Process normalization if requested
        normalization = options.get("normalization", NormalizationMethod.NONE)
        downsample = options.get("downsample", DEFAULT_DOWNSAMPLE)

        if normalization != NormalizationMethod.NONE:
            delta_mean = normalize_signal(delta_mean, method=normalization)
            delta_std = normalize_signal(delta_std, method=normalization)

        # Downsample if requested
        if downsample > 1:
            positions = positions[::downsample]
            delta_mean = delta_mean[::downsample]
            delta_std = delta_std[::downsample]
            coverage_a = coverage_a[::downsample]
            coverage_b = coverage_b[::downsample]

        # Create delta signal track
        p_signal = self._create_signal_track(
            positions, delta_mean, delta_std, sample_a_name, sample_b_name
        )

        # Create delta stats track
        p_stats = self._create_stats_track(
            positions, coverage_a, coverage_b, sample_a_name, sample_b_name
        )

        # Link x-axes for synchronized zoom/pan
        p_stats.x_range = p_signal.x_range

        # Create gridplot layout
        layout_obj = gridplot(
            [[p_signal], [p_stats]],
            toolbar_location="right",
            merge_tools=True,
        )

        # Convert to HTML
        html = self._figure_to_html(layout_obj)

        return html, layout_obj

    def _create_signal_track(
        self,
        positions: np.ndarray,
        delta_mean: np.ndarray,
        delta_std: np.ndarray,
        sample_a_name: str,
        sample_b_name: str,
    ) -> Any:
        """
        Create delta signal track with confidence bands

        Args:
            positions: Reference positions (x-axis)
            delta_mean: Mean delta signal values
            delta_std: Standard deviation of deltas
            sample_a_name: Name of sample A
            sample_b_name: Name of sample B

        Returns:
            Bokeh figure object
        """
        # Create figure
        p = self.theme_manager.create_figure(
            title=f"Delta Signal: {sample_b_name} - {sample_a_name}",
            x_label="Position",
            y_label="Δ Signal (pA)",
            width=900,
            height=DELTA_SIGNAL_HEIGHT,
        )

        # Calculate confidence bands (±1 σ)
        upper_band = delta_mean + delta_std
        lower_band = delta_mean - delta_std

        # Create data source for band
        band_source = ColumnDataSource(
            {"x": positions, "lower": lower_band, "upper": upper_band}
        )

        # Plot confidence band
        band = Band(
            base="x",
            lower="lower",
            upper="upper",
            source=band_source,
            fill_alpha=DELTA_BAND_ALPHA,
            fill_color=DELTA_POSITIVE_COLOR,
            line_width=0,
        )
        p.add_layout(band)

        # Plot mean delta line
        colors = self._color_by_direction(delta_mean)
        p.line(
            positions,
            delta_mean,
            line_width=DELTA_LINE_WIDTH,
            line_color=DELTA_POSITIVE_COLOR,
            alpha=0.8,
            legend_label="Mean Δ Signal",
        )

        # Add scatter for individual deltas (colored by direction)
        for i, (pos, delta) in enumerate(zip(positions, delta_mean, strict=True)):
            color = colors[i]
            p.scatter(
                [pos],
                [delta],
                size=4,
                color=color,
                alpha=0.5,
            )

        # Add zero line
        p.line(
            [positions[0], positions[-1]],
            [0, 0],
            line_dash="dashed",
            line_color=DELTA_ZERO_LINE_COLOR,
            line_width=1,
            alpha=0.5,
        )

        # Add hover tool
        hover = HoverTool(
            tooltips=[
                ("Position", "@x"),
                ("Δ Mean", "@y{0.000}"),
            ]
        )
        p.add_tools(hover)

        # Configure legend
        self.theme_manager.configure_legend(p)

        return p

    def _create_stats_track(
        self,
        positions: np.ndarray,
        coverage_a: list[int],
        coverage_b: list[int],
        sample_a_name: str,
        sample_b_name: str,
    ) -> Any:
        """
        Create delta statistics track showing coverage differences

        Args:
            positions: Reference positions (x-axis)
            coverage_a: Coverage counts for sample A
            coverage_b: Coverage counts for sample B
            sample_a_name: Name of sample A
            sample_b_name: Name of sample B

        Returns:
            Bokeh figure object
        """
        # Create figure
        p = self.theme_manager.create_figure(
            title="Coverage Comparison",
            x_label="Position",
            y_label="Read Count",
            width=900,
            height=DELTA_STATS_HEIGHT,
        )

        # Plot coverage for both samples
        p.line(
            positions,
            coverage_a,
            line_width=DELTA_LINE_WIDTH,
            line_color=DELTA_NEGATIVE_COLOR,
            alpha=0.7,
            legend_label=f"{sample_a_name}",
        )

        p.line(
            positions,
            coverage_b,
            line_width=DELTA_LINE_WIDTH,
            line_color=DELTA_POSITIVE_COLOR,
            alpha=0.7,
            legend_label=f"{sample_b_name}",
        )

        # Add scatter for individual points
        p.scatter(
            positions,
            coverage_a,
            size=3,
            color=DELTA_NEGATIVE_COLOR,
            alpha=0.4,
        )

        p.scatter(
            positions,
            coverage_b,
            size=3,
            color=DELTA_POSITIVE_COLOR,
            alpha=0.4,
        )

        # Format y-axis
        p.yaxis.formatter = NumeralTickFormatter(format="0")

        # Add hover tool
        hover = HoverTool(
            tooltips=[
                ("Position", "@x"),
                ("Coverage", "@y"),
            ]
        )
        p.add_tools(hover)

        # Configure legend
        self.theme_manager.configure_legend(p)

        return p

    @staticmethod
    def _color_by_direction(deltas: np.ndarray) -> list[str]:
        """
        Color deltas based on direction (positive/negative/neutral)

        Args:
            deltas: Array of delta values

        Returns:
            List of color hex strings corresponding to each delta
        """
        colors = []
        for delta in deltas:
            if delta > 0.1:
                colors.append(DELTA_POSITIVE_COLOR)  # Red for positive
            elif delta < -0.1:
                colors.append(DELTA_NEGATIVE_COLOR)  # Blue for negative
            else:
                colors.append(DELTA_NEUTRAL_COLOR)  # Gray for near-zero
        return colors
