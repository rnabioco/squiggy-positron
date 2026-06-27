"""
Aggregate Comparison Strategy for comparing aggregate statistics across multiple samples

This strategy visualizes aggregate statistics (signal, dwell time, quality) from
multiple samples overlaid on the same axes, with each sample color-coded using the
Okabe-Ito colorblind-friendly palette.
"""

from typing import Any

import numpy as np
from bokeh.layouts import gridplot
from bokeh.models import Band, ColumnDataSource, HoverTool, Legend, LegendItem

from ..constants import (
    MULTI_READ_COLORS,
    Theme,
)
from ..rendering.theme_manager import ThemeManager
from .base import PlotStrategy


class AggregateComparisonStrategy(PlotStrategy):
    """
    Plot strategy for comparing aggregate statistics across multiple samples

    Creates a visualization of aggregate statistics from 2+ samples overlaid on
    the same axes, with each sample assigned a distinct color. Includes:
    - Signal statistics comparison (mean ± std)
    - Dwell time statistics comparison (mean ± std)
    - Quality statistics comparison (mean ± std)
    - Coverage comparison across samples

    Attributes:
        theme: Theme enum (LIGHT or DARK) for plot styling

    Example:
        >>> from squiggy.plot_strategies.aggregate_comparison import (
        ...     AggregateComparisonStrategy
        ... )
        >>> from squiggy.constants import Theme, NormalizationMethod
        >>>
        >>> strategy = AggregateComparisonStrategy(Theme.LIGHT)
        >>> data = {
        ...     'samples': [
        ...         {
        ...             'name': 'sample_1',
        ...             'signal_stats': {...},
        ...             'dwell_stats': {...},
        ...             'quality_stats': {...},
        ...         },
        ...         {
        ...             'name': 'sample_2',
        ...             'signal_stats': {...},
        ...             'dwell_stats': {...},
        ...             'quality_stats': {...},
        ...         },
        ...     ],
        ...     'reference_name': 'chr1',
        ...     'enabled_metrics': ['signal', 'dwell_time', 'quality'],
        ... }
        >>> options = {'normalization': NormalizationMethod.ZNORM}
        >>> html, grid = strategy.create_plot(data, options)
    """

    def __init__(self, theme: Theme):
        """
        Initialize aggregate comparison strategy

        Args:
            theme: Theme enum (LIGHT or DARK)
        """
        super().__init__(theme)
        self.theme_manager = ThemeManager(theme)

    def validate_data(self, data: dict[str, Any]) -> None:
        """
        Validate that required aggregate comparison data is present

        Required keys:
        - samples: list of dicts with 'name' and statistics dicts
        - reference_name: str, reference identifier
        - enabled_metrics: list of metric names to display

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
            if "name" not in sample:
                raise ValueError(f"Sample {i} missing 'name' key")

        if "reference_name" not in data:
            raise ValueError("Missing required key: 'reference_name'")

        if "enabled_metrics" not in data:
            raise ValueError("Missing required key: 'enabled_metrics'")

    def _create_signal_track(
        self, samples: list[dict], reference_name: str, title: str | None = None
    ):
        """
        Create signal statistics comparison track

        Args:
            samples: List of sample data dicts with 'signal_stats'
            reference_name: Reference name for plot title
            title: Optional explicit title (overrides the default comparison title;
                used by the multi-track layout to label per-sample groups)

        Returns:
            Bokeh figure or None if no signal data
        """
        # Check if any sample has signal stats
        if not any("signal_stats" in s and s["signal_stats"] for s in samples):
            return None

        # Create themed figure
        p = self.theme_manager.create_figure(
            title=title or f"Signal Statistics Comparison - {reference_name}",
            x_label="Reference Position (bp)",
            y_label="Normalized Signal",
        )

        legend_items = []

        # Plot each sample's signal statistics
        for i, sample in enumerate(samples):
            if "signal_stats" not in sample or not sample["signal_stats"]:
                continue

            stats = sample["signal_stats"]
            sample_name = sample["name"]
            color = sample.get("color", MULTI_READ_COLORS[i % len(MULTI_READ_COLORS)])

            positions = stats.get("positions", [])
            mean_signal = stats.get("mean_signal", [])
            std_signal = stats.get("std_signal", [])

            if len(positions) == 0 or len(mean_signal) == 0:
                continue

            # Calculate confidence bands
            upper = np.array(mean_signal) + np.array(std_signal)
            lower = np.array(mean_signal) - np.array(std_signal)

            # Create data source
            source = ColumnDataSource(
                data={
                    "positions": positions,
                    "mean_signal": mean_signal,
                    "upper": upper,
                    "lower": lower,
                    "sample": [sample_name] * len(positions),
                }
            )

            # Add confidence band
            band = Band(
                base="positions",
                lower="lower",
                upper="upper",
                source=source,
                fill_color=color,
                fill_alpha=0.2,
                line_width=0,
            )
            p.add_layout(band)

            # Add mean line
            line = p.line(
                "positions",
                "mean_signal",
                source=source,
                color=color,
                line_width=2,
                alpha=0.8,
            )

            legend_items.append(LegendItem(label=sample_name, renderers=[line]))

            # Add hover tool
            hover = HoverTool(
                renderers=[line],
                tooltips=[
                    ("Sample", "@sample"),
                    ("Position", "@positions"),
                    ("Mean Signal", "@mean_signal{0.00}"),
                ],
            )
            p.add_tools(hover)

        # Add legend
        if legend_items:
            legend = Legend(items=legend_items, location="top_right")
            legend.click_policy = "hide"
            p.add_layout(legend)

        return p

    def _create_dwell_track(
        self, samples: list[dict], reference_name: str, title: str | None = None
    ):
        """
        Create dwell time statistics comparison track

        Args:
            samples: List of sample data dicts with 'dwell_stats'
            reference_name: Reference name for plot title
            title: Optional explicit title (overrides the default comparison title)

        Returns:
            Bokeh figure or None if no dwell data
        """
        # Check if any sample has dwell stats
        if not any("dwell_stats" in s and s["dwell_stats"] for s in samples):
            return None

        # Create themed figure
        p = self.theme_manager.create_figure(
            title=title or f"Dwell Time Statistics Comparison - {reference_name}",
            x_label="Reference Position (bp)",
            y_label="Dwell Time (samples)",
        )

        legend_items = []

        # Plot each sample's dwell statistics
        for i, sample in enumerate(samples):
            if "dwell_stats" not in sample or not sample["dwell_stats"]:
                continue

            stats = sample["dwell_stats"]
            sample_name = sample["name"]
            color = sample.get("color", MULTI_READ_COLORS[i % len(MULTI_READ_COLORS)])

            positions = stats.get("positions", [])
            mean_dwell = stats.get("mean_dwell", [])
            std_dwell = stats.get("std_dwell", [])

            if len(positions) == 0 or len(mean_dwell) == 0:
                continue

            # Calculate confidence bands
            upper = np.array(mean_dwell) + np.array(std_dwell)
            lower = np.array(mean_dwell) - np.array(std_dwell)

            # Create data source
            source = ColumnDataSource(
                data={
                    "positions": positions,
                    "mean_dwell": mean_dwell,
                    "upper": upper,
                    "lower": lower,
                    "sample": [sample_name] * len(positions),
                }
            )

            # Add confidence band
            band = Band(
                base="positions",
                lower="lower",
                upper="upper",
                source=source,
                fill_color=color,
                fill_alpha=0.2,
                line_width=0,
            )
            p.add_layout(band)

            # Add mean line
            line = p.line(
                "positions",
                "mean_dwell",
                source=source,
                color=color,
                line_width=2,
                alpha=0.8,
            )

            legend_items.append(LegendItem(label=sample_name, renderers=[line]))

            # Add hover tool
            hover = HoverTool(
                renderers=[line],
                tooltips=[
                    ("Sample", "@sample"),
                    ("Position", "@positions"),
                    ("Mean Dwell", "@mean_dwell{0.00}"),
                ],
            )
            p.add_tools(hover)

        # Add legend
        if legend_items:
            legend = Legend(items=legend_items, location="top_right")
            legend.click_policy = "hide"
            p.add_layout(legend)

        return p

    def _create_quality_track(
        self, samples: list[dict], reference_name: str, title: str | None = None
    ):
        """
        Create quality statistics comparison track

        Args:
            samples: List of sample data dicts with 'quality_stats'
            reference_name: Reference name for plot title
            title: Optional explicit title (overrides the default comparison title)

        Returns:
            Bokeh figure or None if no quality data
        """
        # Check if any sample has quality stats
        if not any("quality_stats" in s and s["quality_stats"] for s in samples):
            return None

        # Create themed figure
        p = self.theme_manager.create_figure(
            title=title or f"Quality Statistics Comparison - {reference_name}",
            x_label="Reference Position (bp)",
            y_label="Quality Score",
        )

        legend_items = []

        # Plot each sample's quality statistics
        for i, sample in enumerate(samples):
            if "quality_stats" not in sample or not sample["quality_stats"]:
                continue

            stats = sample["quality_stats"]
            sample_name = sample["name"]
            color = sample.get("color", MULTI_READ_COLORS[i % len(MULTI_READ_COLORS)])

            positions = stats.get("positions", [])
            mean_quality = stats.get("mean_quality", [])
            std_quality = stats.get("std_quality", [])

            if len(positions) == 0 or len(mean_quality) == 0:
                continue

            # Calculate confidence bands
            upper = np.array(mean_quality) + np.array(std_quality)
            lower = np.array(mean_quality) - np.array(std_quality)

            # Create data source
            source = ColumnDataSource(
                data={
                    "positions": positions,
                    "mean_quality": mean_quality,
                    "upper": upper,
                    "lower": lower,
                    "sample": [sample_name] * len(positions),
                }
            )

            # Add confidence band
            band = Band(
                base="positions",
                lower="lower",
                upper="upper",
                source=source,
                fill_color=color,
                fill_alpha=0.2,
                line_width=0,
            )
            p.add_layout(band)

            # Add mean line
            line = p.line(
                "positions",
                "mean_quality",
                source=source,
                color=color,
                line_width=2,
                alpha=0.8,
            )

            legend_items.append(LegendItem(label=sample_name, renderers=[line]))

            # Add hover tool
            hover = HoverTool(
                renderers=[line],
                tooltips=[
                    ("Sample", "@sample"),
                    ("Position", "@positions"),
                    ("Mean Quality", "@mean_quality{0.00}"),
                ],
            )
            p.add_tools(hover)

        # Add legend
        if legend_items:
            legend = Legend(items=legend_items, location="top_right")
            legend.click_policy = "hide"
            p.add_layout(legend)

        return p

    def _create_pileup_tracks(
        self, samples: list[dict], rna_mode: bool = False
    ) -> list:
        """
        Create one base-call pileup track per sample

        Reuses the single-sample pileup renderer from AggregatePlotStrategy so the
        stacked-base appearance matches the single-sample composite plot. Each track
        is titled with its sample name.

        Args:
            samples: List of sample data dicts, each optionally with 'pileup_stats'
            rna_mode: If True, display U instead of T for RNA sequences

        Returns:
            List of Bokeh figures (one per sample with pileup data); may be empty
        """
        # Local import avoids a circular import between the two strategy modules
        from .aggregate import AggregatePlotStrategy

        agg = AggregatePlotStrategy(self.theme)

        tracks = []
        for sample in samples:
            pileup_stats = sample.get("pileup_stats")
            if not pileup_stats or len(pileup_stats.get("positions", [])) == 0:
                continue

            fig = agg._create_pileup_track(pileup_stats, rna_mode=rna_mode)
            fig.title.text = f"Base Call Pileup — {sample['name']}"
            tracks.append(fig)

        return tracks

    def _create_multitrack_layout(
        self,
        samples: list[dict],
        reference_name: str,
        enabled_metrics: list[str],
        show_pileup: bool,
        rna_mode: bool,
    ) -> list:
        """
        Build a detailed per-sample layout: one complete group of tracks per sample

        Unlike the overlay layout (which superimposes every sample on shared axes),
        this gives each sample its own signal / pileup / dwell / quality / coverage
        block, stacked vertically and grouped together. This makes it easier to read
        each sample's full profile at the cost of direct visual overlap.

        The per-sample track builders are reused by passing a single-element sample
        list, then the figure title is prefixed with the sample name so the group
        boundaries are obvious in the stacked layout.

        Args:
            samples: List of sample data dicts
            reference_name: Reference name for plot titles
            enabled_metrics: Metrics to render (signal, dwell_time, quality)
            show_pileup: Whether to include a base-call pileup track per sample
            rna_mode: Display U instead of T in pileup tracks for RNA sequences

        Returns:
            Ordered list of Bokeh figures (sample groups concatenated)
        """
        tracks = []

        for sample in samples:
            name = sample["name"]
            group = []

            if "signal" in enabled_metrics:
                signal_track = self._create_signal_track(
                    [sample], reference_name, title=f"{name} — Signal"
                )
                if signal_track:
                    group.append(signal_track)

            if show_pileup:
                pileup_stats = sample.get("pileup_stats")
                if pileup_stats and len(pileup_stats.get("positions", [])) > 0:
                    group.extend(
                        self._create_pileup_tracks([sample], rna_mode=rna_mode)
                    )

            if "dwell_time" in enabled_metrics:
                dwell_track = self._create_dwell_track(
                    [sample], reference_name, title=f"{name} — Dwell Time"
                )
                if dwell_track:
                    group.append(dwell_track)

            if "quality" in enabled_metrics:
                quality_track = self._create_quality_track(
                    [sample], reference_name, title=f"{name} — Quality"
                )
                if quality_track:
                    group.append(quality_track)

            coverage_track = self._create_coverage_track(
                [sample], reference_name, title=f"{name} — Coverage"
            )
            if coverage_track:
                group.append(coverage_track)

            tracks.extend(group)

        return tracks

    def _create_coverage_track(
        self, samples: list[dict], reference_name: str, title: str | None = None
    ):
        """
        Create coverage comparison track

        Args:
            samples: List of sample data dicts with coverage data
            reference_name: Reference name for plot title
            title: Optional explicit title (overrides the default comparison title)

        Returns:
            Bokeh figure or None if no coverage data
        """
        # Check if any sample has coverage data
        if not any("coverage" in s and s["coverage"] for s in samples):
            return None

        # Create themed figure (smaller height for coverage)
        p = self.theme_manager.create_figure(
            title=title or f"Coverage Comparison - {reference_name}",
            x_label="Reference Position (bp)",
            y_label="Read Count",
            height=350,  # Slightly shorter than default
        )

        legend_items = []

        # Plot each sample's coverage
        for i, sample in enumerate(samples):
            if "coverage" not in sample or not sample["coverage"]:
                continue

            sample_name = sample["name"]
            color = sample.get("color", MULTI_READ_COLORS[i % len(MULTI_READ_COLORS)])

            # Coverage data can be in different formats
            coverage_data = sample["coverage"]
            if isinstance(coverage_data, dict):
                positions = coverage_data.get("positions", [])
                coverage = coverage_data.get("coverage", [])
            elif isinstance(coverage_data, list):
                # Assume it's just the coverage array
                positions = list(range(len(coverage_data)))
                coverage = coverage_data
            else:
                continue

            if len(positions) == 0 or len(coverage) == 0:
                continue

            # Create data source
            source = ColumnDataSource(
                data={
                    "positions": positions,
                    "coverage": coverage,
                    "sample": [sample_name] * len(positions),
                }
            )

            # Add coverage line
            line = p.line(
                "positions",
                "coverage",
                source=source,
                color=color,
                line_width=1.5,
                alpha=0.7,
            )

            legend_items.append(LegendItem(label=sample_name, renderers=[line]))

            # Add hover tool
            hover = HoverTool(
                renderers=[line],
                tooltips=[
                    ("Sample", "@sample"),
                    ("Position", "@positions"),
                    ("Coverage", "@coverage"),
                ],
            )
            p.add_tools(hover)

        # Add legend
        if legend_items:
            legend = Legend(items=legend_items, location="top_right")
            legend.click_policy = "hide"
            p.add_layout(legend)

        return p

    def create_plot(
        self, data: dict[str, Any], options: dict[str, Any]
    ) -> tuple[str, Any]:
        """
        Generate aggregate comparison plot

        Creates synchronized tracks for each enabled metric:
        - Signal statistics comparison (if enabled)
        - Dwell time statistics comparison (if enabled)
        - Quality statistics comparison (if enabled)
        - Coverage comparison (always shown if data available)

        Args:
            data: Aggregate comparison data dictionary
            options: Plot options (normalization, etc.)

        Returns:
            Tuple of (html_string, bokeh_gridplot)
        """
        from bokeh.embed import file_html
        from bokeh.resources import CDN

        self.validate_data(data)

        samples = data["samples"]
        reference_name = data.get("reference_name", "Unknown Reference")
        enabled_metrics = data.get(
            "enabled_metrics", ["signal", "dwell_time", "quality"]
        )
        # 'overlay' (default): superimpose all samples on shared axes.
        # 'multi-track': one complete, labelled track group per sample.
        view_style = data.get("view_style", "overlay")
        show_pileup = bool(data.get("show_pileup"))
        rna_mode = data.get("rna_mode", False)

        if view_style == "multi-track":
            tracks = self._create_multitrack_layout(
                samples,
                reference_name,
                enabled_metrics,
                show_pileup,
                rna_mode,
            )
        else:
            # Overlay layout: each track superimposes every sample
            tracks = []

            if "signal" in enabled_metrics:
                signal_track = self._create_signal_track(samples, reference_name)
                if signal_track:
                    tracks.append(signal_track)

            # Per-sample base-call pileup tracks (the most important comparison panel)
            if show_pileup:
                tracks.extend(self._create_pileup_tracks(samples, rna_mode=rna_mode))

            if "dwell_time" in enabled_metrics:
                dwell_track = self._create_dwell_track(samples, reference_name)
                if dwell_track:
                    tracks.append(dwell_track)

            if "quality" in enabled_metrics:
                quality_track = self._create_quality_track(samples, reference_name)
                if quality_track:
                    tracks.append(quality_track)

            # Always try to add coverage track if data available
            coverage_track = self._create_coverage_track(samples, reference_name)
            if coverage_track:
                tracks.append(coverage_track)

        if not tracks:
            raise ValueError("No tracks could be created with the provided data")

        # Apply primer trim bounds to clip x-axis if available
        primer_trim_bounds = data.get("primer_trim_bounds")
        if primer_trim_bounds is not None and tracks:
            from bokeh.models import Range1d

            start_pos, end_pos = primer_trim_bounds
            base_x_range = Range1d(start=start_pos - 0.5, end=end_pos + 0.5)
            for track in tracks:
                track.x_range = base_x_range
        elif tracks:
            # Link x-axes for synchronized zoom/pan
            # All tracks share the x_range from the first track
            base_x_range = tracks[0].x_range
            for track in tracks[1:]:
                track.x_range = base_x_range

        # Create grid layout
        grid = gridplot(
            [[track] for track in tracks],
            sizing_mode="stretch_width",
            toolbar_location="right",
        )

        # Apply theme background to gridplot container
        grid.styles = {"background-color": self.theme_manager.get_color("plot_bg")}

        # Generate HTML
        html = file_html(grid, CDN, title=f"Aggregate Comparison - {reference_name}")

        return html, grid
