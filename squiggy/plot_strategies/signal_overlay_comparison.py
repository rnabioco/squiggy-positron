"""
Signal Overlay Comparison Strategy for comparing multiple samples

This strategy visualizes raw signals from multiple samples overlaid on the same axes,
with each sample color-coded using the Okabe-Ito colorblind-friendly palette.
Includes reference nucleotide sequence annotation.
"""

from typing import Any

import numpy as np
from bokeh.layouts import gridplot
from bokeh.models import HoverTool, NumeralTickFormatter

from ..constants import (
    MULTI_READ_COLORS,
    NormalizationMethod,
    Theme,
)
from ..normalization import normalize_signal
from ..rendering.base_annotation_renderer import BaseAnnotationRenderer
from ..rendering.theme_manager import ThemeManager
from .base import PlotStrategy


class SignalOverlayComparisonStrategy(PlotStrategy):
    """
    Plot strategy for overlaying signals from multiple samples

    Creates a visualization of aligned signals from 2+ samples overlaid on the same axes,
    with each sample assigned a distinct color from the Okabe-Ito palette. Only shows
    signal from regions with alignment data (respects soft-clipped boundaries). Includes:
    - Overlaid signal lines for all samples (color-coded, aligned regions only)
    - Interactive legend to toggle samples on/off
    - Reference nucleotide sequence displayed below x-axis
    - Base coloring using standard nucleotide colors
    - Aligned coverage track showing read count per position per sample (aligned reads only)

    Attributes:
        theme: Theme enum (LIGHT or DARK) for plot styling

    Examples:
        >>> from squiggy.plot_strategies.signal_overlay_comparison import (
        ...     SignalOverlayComparisonStrategy
        ... )
        >>> from squiggy.constants import Theme, NormalizationMethod
        >>>
        >>> strategy = SignalOverlayComparisonStrategy(Theme.LIGHT)
        >>> data = {
        ...     'samples': [
        ...         {
        ...             'name': 'sample_1',
        ...             'positions': np.array([0, 1, 2, ...]),
        ...             'signal': np.array([100.5, 102.3, 99.8, ...]),
        ...         },
        ...         {
        ...             'name': 'sample_2',
        ...             'positions': np.array([0, 1, 2, ...]),
        ...             'signal': np.array([101.2, 103.1, 100.5, ...]),
        ...         },
        ...     ],
        ...     'reference_sequence': 'ACGTACGTACGT...',
        ...     'coverage': {
        ...         'sample_1': [10, 10, 10, ...],
        ...         'sample_2': [10, 10, 10, ...],
        ...     },
        ... }
        >>> options = {'normalization': NormalizationMethod.ZNORM}
        >>> html, fig = strategy.create_plot(data, options)
    """

    def __init__(self, theme: Theme):
        """
        Initialize signal overlay comparison strategy

        Args:
            theme: Theme enum (LIGHT or DARK)
        """
        super().__init__(theme)
        self.theme_manager = ThemeManager(theme)
        self.annotation_renderer = BaseAnnotationRenderer(
            base_colors=self.theme_manager.get_base_colors(),
            show_dwell_time=False,
            show_labels=True,
        )

    def validate_data(self, data: dict[str, Any]) -> None:
        """
        Validate that required signal overlay comparison data is present

        Required keys:
        - samples: list of dicts with 'name', 'positions', 'signal'
        - reference_sequence: str, nucleotide sequence
        - coverage: dict mapping sample names to coverage arrays

        Args:
            data: Plot data dictionary to validate

        Raises:
            ValueError: If required data is missing or invalid
        """
        if "samples" not in data:
            raise ValueError("Missing required key: 'samples'")

        samples = data["samples"]
        if not isinstance(samples, list) or len(samples) < 2:
            raise ValueError(
                f"Need at least 2 samples for comparison, got {len(samples)}"
            )

        # Validate each sample
        for i, sample in enumerate(samples):
            required_keys = ["name", "positions", "signal"]
            missing = [k for k in required_keys if k not in sample]
            if missing:
                raise ValueError(f"Sample {i} missing keys: {missing}")

            # Validate array lengths match
            positions = sample["positions"]
            signal = sample["signal"]
            if len(signal) != len(positions):
                raise ValueError(
                    f"Sample '{sample['name']}': signal length ({len(signal)}) "
                    f"must match positions length ({len(positions)})"
                )

        if "reference_sequence" not in data:
            raise ValueError("Missing required key: 'reference_sequence'")

        if not isinstance(data["reference_sequence"], str):
            raise ValueError("reference_sequence must be a string")

        if "coverage" not in data:
            raise ValueError("Missing required key: 'coverage'")

        coverage = data["coverage"]
        if not isinstance(coverage, dict):
            raise ValueError("coverage must be a dictionary")

        # Validate coverage for each sample
        for sample in samples:
            sample_name = sample["name"]
            if sample_name not in coverage:
                raise ValueError(f"Missing coverage data for sample '{sample_name}'")

    def create_plot(
        self, data: dict[str, Any], options: dict[str, Any]
    ) -> tuple[str, Any]:
        """
        Generate signal overlay comparison plot

        Creates two synchronized tracks:
        1. Signal Overlay Track: All samples overlaid with color per sample
        2. Coverage Track: Read count per position per sample

        Args:
            data: Signal overlay comparison data dictionary
            options: Plot options (normalization, downsample, etc.)

        Returns:
            Tuple of (html_string, bokeh_gridplot)
        """
        self.validate_data(data)

        samples = data["samples"]
        reference_sequence = data["reference_sequence"]
        coverage_data = data["coverage"]
        normalization = options.get("normalization", NormalizationMethod.NONE)
        downsample = options.get("downsample", 1)

        # Prepare plot data for each sample
        plot_data = []
        positions = None
        downsampled_coverage_data = {}

        for i, sample in enumerate(samples):
            sample_positions = np.array(sample["positions"])
            sample_signal = np.array(sample["signal"])

            # Apply normalization if requested
            if normalization != NormalizationMethod.NONE:
                sample_signal = normalize_signal(sample_signal, method=normalization)

            # Apply downsampling if requested
            if downsample > 1:
                sample_positions = sample_positions[::downsample]
                sample_signal = sample_signal[::downsample]

            plot_data.append(
                {
                    "name": sample["name"],
                    "positions": sample_positions,
                    "signal": sample_signal,
                    "color": MULTI_READ_COLORS[i % len(MULTI_READ_COLORS)],
                }
            )

            # Also downsample coverage data for this sample
            sample_name = sample["name"]
            if sample_name in coverage_data:
                coverage = np.array(coverage_data[sample_name])
                if downsample > 1:
                    coverage = coverage[::downsample]
                downsampled_coverage_data[sample_name] = coverage
            else:
                downsampled_coverage_data[sample_name] = coverage_data.get(
                    sample_name, []
                )

            # Store positions from first sample (should be same for all)
            if positions is None:
                positions = sample_positions

        # Create signal overlay track
        p_signal = self._create_signal_track(plot_data, reference_sequence)

        # Create coverage track (use downsampled coverage data)
        p_coverage = self._create_coverage_track(
            positions, downsampled_coverage_data, plot_data, downsample=1
        )

        # Link x-axes for synchronized zoom/pan
        p_coverage.x_range = p_signal.x_range

        # Create gridplot layout
        layout_obj = gridplot(
            [[p_signal], [p_coverage]],
            toolbar_location="right",
            merge_tools=True,
        )

        # Convert to HTML
        html = self._figure_to_html(layout_obj)

        return html, layout_obj

    def _create_signal_track(
        self, plot_data: list[dict[str, Any]], reference_sequence: str
    ) -> Any:
        """
        Create signal overlay track with reference nucleotide display

        Args:
            plot_data: List of dicts with name, positions, signal, color
            reference_sequence: Reference nucleotide sequence

        Returns:
            Bokeh figure object
        """
        # Create figure
        p = self.theme_manager.create_figure(
            title="Signal Overlay Comparison",
            x_label="Position",
            y_label="Signal (pA)",
            width=1000,
            height=500,
        )

        # Plot signal lines for each sample
        for sample_info in plot_data:
            p.line(
                sample_info["positions"],
                sample_info["signal"],
                line_width=2,
                line_color=sample_info["color"],
                alpha=0.8,
                legend_label=sample_info["name"],
            )

        # Add base annotations (reference sequence display)
        if plot_data and reference_sequence:
            plot_data[0]["positions"]
            # Get signal range for positioning annotations
            all_signals = np.concatenate([s["signal"] for s in plot_data])
            signal_min = np.min(all_signals)
            signal_max = np.max(all_signals)

            # Create base annotations for reference
            try:
                base_annotations = [
                    {"position": int(pos), "base": base}
                    for pos, base in enumerate(reference_sequence)
                ]

                self.annotation_renderer.render_position_based(
                    p,
                    base_annotations,
                    sample_rate=1.0,
                    signal_length=len(reference_sequence),
                    signal_min=signal_min,
                    signal_max=signal_max,
                )
            except Exception:
                # Silently skip base annotation rendering if it fails
                pass

        # Configure legend
        p.legend.click_policy = "hide"  # Interactive legend
        self.theme_manager.configure_legend(p)

        # Add hover tool
        hover = HoverTool(
            tooltips=[
                ("Position", "@x"),
                ("Signal", "@y{0.00}"),
            ]
        )
        p.add_tools(hover)

        return p

    def _create_coverage_track(
        self,
        positions: np.ndarray,
        coverage_data: dict[str, list[int]],
        plot_data: list[dict[str, Any]],
        downsample: int,
    ) -> Any:
        """
        Create aligned coverage track showing read count per aligned position per sample

        Shows only reads that have alignment data at each position (excludes soft-clipped regions).

        Args:
            positions: Reference positions (x-axis) - used for axis range
            coverage_data: Dict mapping sample names to coverage arrays (aligned reads only)
            plot_data: List of dicts with sample info (for colors and positions)
            downsample: Downsampling factor

        Returns:
            Bokeh figure object
        """
        # Create figure
        p = self.theme_manager.create_figure(
            title="Aligned Coverage Comparison",
            x_label="Position",
            y_label="Aligned Read Count",
            width=1000,
            height=250,
        )

        # Plot coverage for each sample
        for sample_info in plot_data:
            sample_name = sample_info["name"]
            coverage = coverage_data.get(sample_name, [])

            # Use sample's own positions from plot_data instead of first sample's positions
            # This ensures coverage array length matches positions array length
            sample_positions = sample_info["positions"]

            # Coverage is already downsampled in create_plot(), no need to downsample again
            p.line(
                sample_positions,
                coverage,
                line_width=2,
                line_color=sample_info["color"],
                alpha=0.7,
                legend_label=sample_name,
            )

            # Add scatter for individual points
            p.scatter(
                sample_positions,
                coverage,
                size=4,
                color=sample_info["color"],
                alpha=0.5,
            )

        # Format y-axis
        p.yaxis.formatter = NumeralTickFormatter(format="0")

        # Configure legend
        p.legend.click_policy = "hide"
        self.theme_manager.configure_legend(p)

        # Add hover tool
        hover = HoverTool(
            tooltips=[
                ("Position", "@x"),
                ("Coverage", "@y"),
            ]
        )
        p.add_tools(hover)

        return p
