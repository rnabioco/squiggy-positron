"""
Aggregate plot strategy implementation

This module implements the Strategy Pattern for aggregate multi-read visualization
with synchronized tracks showing signal statistics, base pileup, and quality.
"""

from bokeh.embed import file_html
from bokeh.layouts import gridplot
from bokeh.models import Band, ColumnDataSource, HoverTool
from bokeh.resources import CDN

from ..constants import NormalizationMethod, Theme
from ..rendering import ThemeManager
from .base import PlotStrategy


class AggregatePlotStrategy(PlotStrategy):
    """
    Strategy for aggregate multi-read visualization

    This strategy plots aggregate statistics across multiple reads with three
    synchronized tracks:
    1. Mean signal with confidence bands
    2. Base call pileup (stacked proportions)
    3. Quality scores by position

    Example:
        >>> from squiggy.plot_strategies.aggregate import AggregatePlotStrategy
        >>> from squiggy.constants import Theme, NormalizationMethod
        >>>
        >>> strategy = AggregatePlotStrategy(Theme.LIGHT)
        >>>
        >>> data = {
        ...     'aggregate_stats': {
        ...         'positions': positions,
        ...         'mean_signal': mean,
        ...         'std_signal': std,
        ...         'median_signal': median,
        ...         'coverage': coverage,
        ...     },
        ...     'pileup_stats': {
        ...         'positions': positions,
        ...         'counts': counts_dict,
        ...     },
        ...     'quality_stats': {
        ...         'positions': positions,
        ...         'mean_quality': mean_q,
        ...         'std_quality': std_q,
        ...     },
        ...     'reference_name': 'chr1:1000-2000',
        ...     'num_reads': 50,
        ... }
        >>>
        >>> options = {'normalization': NormalizationMethod.ZNORM}
        >>>
        >>> html, layout = strategy.create_plot(data, options)
    """

    def __init__(self, theme: Theme):
        """
        Initialize aggregate plot strategy

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
            - aggregate_stats: dict with positions, mean_signal, std_signal, coverage
            - pileup_stats: dict with positions, counts
            - quality_stats: dict with positions, mean_quality, std_quality
            - reference_name: str
            - num_reads: int
        """
        required = [
            "aggregate_stats",
            "pileup_stats",
            "quality_stats",
            "reference_name",
            "num_reads",
        ]
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"Missing required data for aggregate plot: {missing}")

        # Validate aggregate_stats structure
        agg = data["aggregate_stats"]
        required_agg = ["positions", "mean_signal", "std_signal", "coverage"]
        missing_agg = [k for k in required_agg if k not in agg]
        if missing_agg:
            raise ValueError(f"aggregate_stats missing keys: {missing_agg}")

        # Validate pileup_stats structure
        pileup = data["pileup_stats"]
        required_pileup = ["positions", "counts"]
        missing_pileup = [k for k in required_pileup if k not in pileup]
        if missing_pileup:
            raise ValueError(f"pileup_stats missing keys: {missing_pileup}")

        # Validate quality_stats structure
        quality = data["quality_stats"]
        required_quality = ["positions", "mean_quality", "std_quality"]
        missing_quality = [k for k in required_quality if k not in quality]
        if missing_quality:
            raise ValueError(f"quality_stats missing keys: {missing_quality}")

    def create_plot(self, data: dict, options: dict) -> tuple[str, any]:
        """
        Generate Bokeh gridplot HTML and layout for aggregate visualization

        Args:
            data: Plot data dictionary containing:
                - aggregate_stats (required): signal statistics
                - pileup_stats (required): base call pileup
                - quality_stats (required): quality scores
                - reference_name (required): reference identifier
                - num_reads (required): number of reads

            options: Plot options dictionary containing:
                - normalization: NormalizationMethod enum (default: NONE)

        Returns:
            Tuple of (html_string, bokeh_gridplot)

        Raises:
            ValueError: If required data is missing
        """
        # Validate data
        self.validate_data(data)

        # Extract data
        aggregate_stats = data["aggregate_stats"]
        pileup_stats = data["pileup_stats"]
        quality_stats = data["quality_stats"]
        reference_name = data["reference_name"]
        num_reads = data["num_reads"]

        # Extract options
        normalization = options.get("normalization", NormalizationMethod.NONE)

        # Create three synchronized tracks
        p_signal = self._create_signal_track(
            aggregate_stats=aggregate_stats,
            reference_name=reference_name,
            num_reads=num_reads,
            normalization=normalization,
        )

        p_pileup = self._create_pileup_track(pileup_stats=pileup_stats)

        p_quality = self._create_quality_track(quality_stats=quality_stats)

        # Link x-axes for synchronized zoom/pan
        p_pileup.x_range = p_signal.x_range
        p_quality.x_range = p_signal.x_range

        # Create gridplot
        grid = gridplot(
            [[p_signal], [p_pileup], [p_quality]],
            sizing_mode="stretch_width",
            toolbar_location="right",
        )

        # Generate HTML
        html_title = f"Squiggy Aggregate: {reference_name} ({num_reads} reads)"
        html = file_html(grid, CDN, title=html_title)
        return html, grid

    # =========================================================================
    # Private Methods: Track Creation
    # =========================================================================

    def _create_signal_track(
        self,
        aggregate_stats: dict,
        reference_name: str,
        num_reads: int,
        normalization: NormalizationMethod,
    ):
        """Create signal aggregate track with confidence bands"""
        fig = self.theme_manager.create_figure(
            title=f"Aggregate Signal - {reference_name} ({num_reads} reads)",
            x_label="Reference Position",
            y_label=f"Signal ({normalization.value})",
            height=300,
        )

        positions = aggregate_stats["positions"]
        mean_signal = aggregate_stats["mean_signal"]
        std_signal = aggregate_stats["std_signal"]
        coverage = aggregate_stats["coverage"]

        # Create confidence band (mean ± 1 std dev)
        upper = mean_signal + std_signal
        lower = mean_signal - std_signal

        # Data source
        source = ColumnDataSource(
            data={
                "x": positions,
                "mean": mean_signal,
                "upper": upper,
                "lower": lower,
                "std": std_signal,
                "coverage": coverage,
            }
        )

        # Add confidence band
        band_color = self.theme_manager.get_signal_band_color()
        band = Band(
            base="x",
            lower="lower",
            upper="upper",
            source=source,
            level="underlay",
            fill_alpha=0.3,
            fill_color=band_color,
            line_width=0,
        )
        fig.add_layout(band)

        # Add mean line
        signal_color = self.theme_manager.get_signal_color()
        mean_line = fig.line(
            "x",
            "mean",
            source=source,
            line_width=2,
            color=signal_color,
        )

        # Add hover tool
        hover = HoverTool(
            renderers=[mean_line],
            tooltips=[
                ("Position", "@x"),
                ("Mean", "@mean{0.2f}"),
                ("Std Dev", "@std{0.2f}"),
                ("Coverage", "@coverage"),
            ],
            mode="mouse",
        )
        fig.add_tools(hover)

        return fig

    def _create_pileup_track(self, pileup_stats: dict):
        """Create base call pileup track"""
        fig = self.theme_manager.create_figure(
            title="Base Call Pileup",
            x_label="Reference Position",
            y_label="Base Proportion",
            height=250,
        )

        positions = pileup_stats["positions"]
        counts = pileup_stats["counts"]
        reference_bases = pileup_stats.get("reference_bases", {})

        # Prepare data for stacked bars
        pileup_data = {
            "x": [],
            "A": [],
            "C": [],
            "G": [],
            "T": [],
            "total": [],
            "A_bottom": [],
            "A_top": [],
            "C_bottom": [],
            "C_top": [],
            "G_bottom": [],
            "G_top": [],
            "T_bottom": [],
            "T_top": [],
            "ref_base": [],
        }

        for pos in positions:
            pos_counts = counts[pos]
            total = sum(pos_counts.values())
            if total > 0:
                pileup_data["x"].append(pos)
                pileup_data["total"].append(total)
                pileup_data["ref_base"].append(reference_bases.get(pos, ""))

                # Calculate proportions
                proportions = {}
                for base in ["A", "C", "G", "T"]:
                    proportions[base] = pos_counts.get(base, 0) / total
                    pileup_data[base].append(proportions[base])

                # Calculate cumulative positions for stacking
                cumulative = 0.0
                for base in ["A", "C", "G", "T"]:
                    pileup_data[f"{base}_bottom"].append(cumulative)
                    cumulative += proportions[base]
                    pileup_data[f"{base}_top"].append(cumulative)

        # Create source
        source = ColumnDataSource(data=pileup_data)

        # Add stacked bars
        base_colors = self.theme_manager.get_base_colors()
        for base in ["A", "C", "G", "T"]:
            fig.vbar(
                x="x",
                bottom=f"{base}_bottom",
                top=f"{base}_top",
                width=0.8,
                source=source,
                color=base_colors[base],
                legend_label=base,
            )

        # Set y-axis range
        fig.y_range.start = 0
        fig.y_range.end = 1

        # Configure legend
        self.theme_manager.configure_legend(fig)

        return fig

    def _create_quality_track(self, quality_stats: dict):
        """Create quality scores track"""
        fig = self.theme_manager.create_figure(
            title="Base Quality Scores",
            x_label="Reference Position",
            y_label="Mean Quality (Phred)",
            height=200,
        )

        positions = quality_stats["positions"]
        mean_quality = quality_stats["mean_quality"]
        std_quality = quality_stats["std_quality"]

        # Create confidence band
        upper = mean_quality + std_quality
        lower = mean_quality - std_quality

        # Data source
        source = ColumnDataSource(
            data={
                "x": positions,
                "mean": mean_quality,
                "upper": upper,
                "lower": lower,
                "std": std_quality,
            }
        )

        # Add confidence band
        band_color = self.theme_manager.get_quality_band_color()
        band = Band(
            base="x",
            lower="lower",
            upper="upper",
            source=source,
            level="underlay",
            fill_alpha=0.2,
            fill_color=band_color,
            line_width=0,
        )
        fig.add_layout(band)

        # Add mean line
        mean_line = fig.line(
            "x",
            "mean",
            source=source,
            line_width=2,
            color=band_color,
        )

        # Add hover tool
        hover = HoverTool(
            renderers=[mean_line],
            tooltips=[
                ("Position", "@x"),
                ("Mean Quality", "@mean{0.1f}"),
                ("Std Dev", "@std{0.1f}"),
            ],
            mode="mouse",
        )
        fig.add_tools(hover)

        return fig
