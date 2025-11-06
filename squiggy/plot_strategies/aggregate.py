"""
Aggregate plot strategy implementation

This module implements the Strategy Pattern for aggregate multi-read visualization
with synchronized tracks showing signal statistics, base pileup, and quality.
"""

from bokeh.embed import file_html
from bokeh.layouts import gridplot
from bokeh.models import Band, ColumnDataSource, FactorRange, HoverTool, Range1d
from bokeh.resources import CDN

from ..constants import (
    MODIFICATION_CODES,
    MODIFICATION_COLORS,
    NormalizationMethod,
    Theme,
)
from ..rendering import ThemeManager
from .base import PlotStrategy


class AggregatePlotStrategy(PlotStrategy):
    """
    Strategy for aggregate multi-read visualization

    This strategy plots aggregate statistics across multiple reads with up to five
    synchronized tracks:
    1. Base modifications heatmap (optional, if modifications present)
    2. Base call pileup (stacked proportions)
    3. Mean signal with confidence bands
    4. Quality scores by position
    5. Dwell time per base with confidence bands (optional, if data available)

    Examples:
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
                - modification_stats (optional): modification probabilities
                - dwell_stats (optional): dwell time statistics
                - reference_name (required): reference identifier
                - num_reads (required): number of reads

            options: Plot options dictionary containing:
                - normalization: NormalizationMethod enum (default: NONE)
                - show_modifications: bool to show modifications panel (default: True)
                - show_pileup: bool to show pileup panel (default: True)
                - show_dwell_time: bool to show dwell time panel (default: True)
                - show_signal: bool to show signal panel (default: True)
                - show_quality: bool to show quality panel (default: True)
                - motif_positions: Optional set of genomic positions to highlight
                  as motif matches (displayed in bold, larger font)

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
        modification_stats = data.get("modification_stats")
        dwell_stats = data.get("dwell_stats")
        reference_name = data["reference_name"]
        num_reads = data["num_reads"]
        transformation_info = data.get("transformation_info", "")

        # Extract options (transformation now happens in plot_aggregate() before this)
        normalization = options.get("normalization", NormalizationMethod.NONE)
        show_modifications = options.get("show_modifications", True)
        show_pileup = options.get("show_pileup", True)
        show_dwell_time = options.get("show_dwell_time", True)
        show_signal = options.get("show_signal", True)
        show_quality = options.get("show_quality", True)
        motif_positions = options.get("motif_positions", None)
        clip_x_to_alignment = options.get("clip_x_to_alignment", True)

        # Build panel list dynamically based on available data and visibility options
        # Panel order: modifications (optional), pileup, signal, quality, dwell time (optional)
        panels = []
        all_figs = []  # Keep track of all figures for x-range linking

        # Create modification heatmap if data exists and panel is enabled
        if (
            show_modifications
            and modification_stats
            and modification_stats.get("mod_stats")
        ):
            p_mods = self._create_modification_heatmap(
                modification_stats=modification_stats
            )
            panels.append([p_mods])
            all_figs.append(p_mods)

        # Create pileup track if enabled
        if show_pileup:
            p_pileup = self._create_pileup_track(
                pileup_stats=pileup_stats,
                motif_positions=motif_positions,
                transformation_info=transformation_info,
            )
            panels.append([p_pileup])
            all_figs.append(p_pileup)

        # Create signal track if enabled
        if show_signal:
            p_signal = self._create_signal_track(
                aggregate_stats=aggregate_stats,
                reference_name=reference_name,
                num_reads=num_reads,
                normalization=normalization,
            )
            panels.append([p_signal])
            all_figs.append(p_signal)

        # Create quality track if enabled
        if show_quality:
            p_quality = self._create_quality_track(quality_stats=quality_stats)
            panels.append([p_quality])
            all_figs.append(p_quality)

        # Create dwell time track if data exists and panel is enabled
        if (
            show_dwell_time
            and dwell_stats
            and len(dwell_stats.get("positions", [])) > 0
        ):
            p_dwell = self._create_dwell_time_track(dwell_stats=dwell_stats)
            panels.append([p_dwell])
            all_figs.append(p_dwell)

        # Link x-axes for synchronized zoom/pan
        # Use first figure as base for x_range
        if all_figs:
            # Apply x-axis clipping if requested
            if clip_x_to_alignment:
                # Clip to consensus alignment region (where most reads agree)
                # Use pileup coverage to determine the high-coverage region
                all_positions = aggregate_stats.get("positions", [])
                coverage = aggregate_stats.get("coverage", [])

                if len(all_positions) > 0 and len(coverage) > 0:
                    import numpy as np
                    from bokeh.models import Range1d

                    # Find consensus region: positions with >50% of maximum coverage
                    # This filters out sparse regions where only a few reads align
                    max_coverage = np.max(coverage)
                    coverage_threshold = max_coverage * 0.5

                    # Find positions that meet the threshold
                    high_coverage_mask = np.array(coverage) >= coverage_threshold
                    high_coverage_positions = np.array(all_positions)[
                        high_coverage_mask
                    ]

                    if len(high_coverage_positions) > 0:
                        # Check if coordinates were transformed to be reference-anchored
                        is_transformed = bool(transformation_info)

                        if is_transformed:
                            # For transformed coordinates: always start at position 1
                            # Position 1 represents the first base of the reference sequence
                            start_pos = 1
                        else:
                            # For genomic coordinates: use original high-coverage clipping
                            start_pos = high_coverage_positions[0]

                        # End position: clip to last high-coverage position (both modes)
                        end_pos = high_coverage_positions[-1]
                    else:
                        # Fallback to all positions if threshold filters everything
                        start_pos = all_positions[0]
                        end_pos = all_positions[-1]

                    # Add 0.5 padding to prevent bars from being cut off
                    base_x_range = Range1d(start=start_pos - 0.5, end=end_pos + 0.5)
                else:
                    # No positions available, use default
                    base_x_range = all_figs[0].x_range
            else:
                # Use default DataRange1d for auto-scaling (shows full range)
                base_x_range = all_figs[0].x_range

            # Apply the x_range to all figures
            for fig in all_figs:
                fig.x_range = base_x_range

        # Create gridplot
        grid = gridplot(
            panels,
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

    def _create_modification_heatmap(self, modification_stats: dict):
        """Create modification probability heatmap track

        Args:
            modification_stats: Dict with mod_stats and positions from
                                calculate_modification_statistics()

        Returns:
            Bokeh figure with modification heatmap
        """
        mod_stats = modification_stats["mod_stats"]

        # Get all modification types and sort them for consistent ordering
        mod_types = sorted(mod_stats.keys(), key=str)

        # Get unique modification names for categorical y-axis
        unique_mod_names = sorted(
            {MODIFICATION_CODES.get(mod_code, str(mod_code)) for mod_code in mod_types}
        )

        # Create figure with categorical y-axis
        fig = self.theme_manager.create_figure(
            title="Base Modifications (Frequency × Probability)",
            x_label="",  # Shared with other panels
            y_label="Modification",
            height=150,
            y_range=FactorRange(factors=unique_mod_names),
        )

        # Prepare data for heatmap
        heatmap_data = {
            "x": [],  # Position
            "y": [],  # Modification type (human-readable)
            "mod_code": [],  # Raw mod code
            "prob": [],  # Mean probability
            "count": [],  # Number of reads with modification
            "coverage": [],  # Total reads covering position
            "frequency": [],  # Fraction of reads with modification
            "opacity": [],  # Combined frequency × probability for visualization
            "std": [],  # Std dev
            "color": [],  # Color based on mod type
        }

        # Build heatmap data
        # First collect all opacity values to find the range
        all_opacities = []
        for mod_code in mod_types:
            for _pos, stats in mod_stats[mod_code].items():
                frequency = stats.get("frequency", 0.0)
                mean_prob = stats["mean"]
                opacity = frequency * mean_prob
                all_opacities.append(opacity)

        # Calculate min/max for normalization (clip to reasonable range)
        # Ensure max_opacity is never zero to avoid division by zero
        max_opacity = max(all_opacities) if all_opacities else 1.0

        # Handle edge case: all modifications have 0 probability/frequency
        if max_opacity <= 0.0:
            max_opacity = 1.0

        min_opacity = 0.2  # Minimum visible opacity

        for mod_code in mod_types:
            # Get human-readable name
            mod_name = MODIFICATION_CODES.get(mod_code, str(mod_code))

            for pos, stats in mod_stats[mod_code].items():
                frequency = stats.get("frequency", 0.0)
                mean_prob = stats["mean"]

                # Opacity = frequency × mean_probability
                # Scale to [min_opacity, 1.0] range for visibility
                raw_opacity = frequency * mean_prob
                opacity = min_opacity + (raw_opacity / max_opacity) * (
                    1.0 - min_opacity
                )

                heatmap_data["x"].append(pos)
                heatmap_data["y"].append(mod_name)
                heatmap_data["mod_code"].append(str(mod_code))
                heatmap_data["prob"].append(mean_prob)
                heatmap_data["count"].append(stats["count"])
                heatmap_data["coverage"].append(stats.get("total_coverage", 0))
                heatmap_data["frequency"].append(frequency)
                heatmap_data["opacity"].append(opacity)
                heatmap_data["std"].append(stats["std"])

                # Get color for this modification type
                color = MODIFICATION_COLORS.get(mod_code, "#808080")
                heatmap_data["color"].append(color)

        # Create data source
        source = ColumnDataSource(data=heatmap_data)

        # Create rectangles for heatmap
        rects = fig.rect(
            x="x",
            y="y",
            width=1.0,  # One position wide
            height=0.9,  # 90% of row height
            source=source,
            fill_color="color",
            fill_alpha="opacity",  # Opacity = frequency × probability
            line_color=None,
        )

        # Add hover tool
        hover = HoverTool(
            renderers=[rects],
            tooltips=[
                ("Position", "@x"),
                ("Modification", "@y"),
                ("Frequency", "@frequency{0.3f}"),
                ("Mean Probability", "@prob{0.3f}"),
                ("Reads", "@count / @coverage"),
                ("Std Dev", "@std{0.3f}"),
            ],
            mode="mouse",
        )
        fig.add_tools(hover)

        # Hide x-axis labels (will use shared x-axis from other panels)
        fig.xaxis.major_label_text_font_size = "0pt"
        fig.xaxis.major_tick_line_color = None

        return fig

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
            height=200,
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

    def _create_pileup_track(
        self,
        pileup_stats: dict,
        motif_positions: set = None,
        transformation_info: str = "",
    ):
        """
        Create base call pileup track with optional motif highlighting

        Args:
            pileup_stats: Dictionary containing pileup statistics
            motif_positions: Optional set of genomic positions to highlight as motif matches
            transformation_info: Diagnostic info about coordinate transformation
        """
        positions = pileup_stats["positions"]
        counts = pileup_stats["counts"]
        reference_bases = pileup_stats.get("reference_bases", {})

        # Build diagnostic title
        title = "Base Call Pileup"
        if transformation_info:
            title = f"{title} [{transformation_info}]"

        fig = self.theme_manager.create_figure(
            title=title,
            x_label="Reference Position",
            y_label="Base Proportion",
            height=300,
        )

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
            # Ensure position is Python int for dictionary lookup
            pos_key = int(pos)
            pos_counts = counts[pos_key]
            total = sum(pos_counts.values())

            # Only draw bars for positions with coverage
            # The x value (pos_key) is the reference position, ensuring alignment
            if total > 0:
                pileup_data["x"].append(pos_key)
                pileup_data["total"].append(total)
                pileup_data["ref_base"].append(reference_bases.get(pos_key, ""))

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
        renderers = []
        for base in ["A", "C", "G", "T"]:
            r = fig.vbar(
                x="x",
                bottom=f"{base}_bottom",
                top=f"{base}_top",
                width=0.8,
                source=source,
                color=base_colors[base],
                legend_label=base,
            )
            renderers.append(r)

        # Add hover tool to show position and base counts
        hover = HoverTool(
            renderers=renderers,
            tooltips=[
                ("Position", "@x"),
                ("Reference", "@ref_base"),
                ("A", "@A{0.0%} (@{total} reads)"),
                ("C", "@C{0.0%}"),
                ("G", "@G{0.0%}"),
                ("T", "@T{0.0%}"),
                ("Total", "@total reads"),
            ],
            mode="mouse",
        )
        fig.add_tools(hover)

        # Add reference base labels above bars (if available)
        if reference_bases and pileup_data["ref_base"]:
            # Separate motif and non-motif positions for different styling
            motif_label_data = {
                "x": [],
                "y": [],
                "text": [],
                "color": [],
            }
            regular_label_data = {
                "x": [],
                "y": [],
                "text": [],
                "color": [],
            }

            for pos, ref_base in zip(
                pileup_data["x"], pileup_data["ref_base"], strict=True
            ):
                if ref_base:
                    # Use base color if available, otherwise use 'N' (gray) for IUPAC codes
                    base_color = base_colors.get(
                        ref_base, base_colors.get("N", "#808080")
                    )

                    # Determine if this position is part of a motif
                    is_motif = motif_positions and pos in motif_positions

                    target_data = motif_label_data if is_motif else regular_label_data
                    target_data["x"].append(pos)
                    target_data["y"].append(1.05)  # Position just above bars (y=1.0)
                    target_data["text"].append(ref_base)
                    target_data["color"].append(base_color)

            # Add regular (non-motif) text labels
            if regular_label_data["x"]:
                regular_source = ColumnDataSource(data=regular_label_data)
                fig.text(
                    x="x",
                    y="y",
                    text="text",
                    source=regular_source,
                    text_color="color",
                    text_align="center",
                    text_baseline="bottom",
                    text_font_size="10pt",
                )

            # Add motif-highlighted text labels (bold and slightly larger)
            if motif_label_data["x"]:
                motif_source = ColumnDataSource(data=motif_label_data)
                fig.text(
                    x="x",
                    y="y",
                    text="text",
                    source=motif_source,
                    text_color="color",
                    text_align="center",
                    text_baseline="bottom",
                    text_font_size="12pt",
                    text_font_style="bold",
                )

        # Set y-axis range (extend to accommodate labels)
        fig.y_range = Range1d(
            start=0, end=1.15
        )  # Extended from 1.0 to fit labels above bars

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

    def _create_dwell_time_track(self, dwell_stats: dict):
        """Create dwell time track with confidence bands"""
        fig = self.theme_manager.create_figure(
            title="Dwell Time per Base",
            x_label="Reference Position",
            y_label="Mean Dwell Time (ms)",
            height=200,
        )

        positions = dwell_stats["positions"]
        mean_dwell = dwell_stats["mean_dwell"]
        std_dwell = dwell_stats["std_dwell"]
        coverage = dwell_stats["coverage"]

        # Create confidence band
        upper = mean_dwell + std_dwell
        lower = mean_dwell - std_dwell

        # Data source
        source = ColumnDataSource(
            data={
                "x": positions,
                "mean": mean_dwell,
                "upper": upper,
                "lower": lower,
                "std": std_dwell,
                "coverage": coverage,
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
                ("Mean Dwell", "@mean{0.2f} ms"),
                ("Std Dev", "@std{0.2f} ms"),
                ("Coverage", "@coverage"),
            ],
            mode="mouse",
        )
        fig.add_tools(hover)

        # Set initial y-range with some padding
        # Note: Bokeh doesn't have built-in auto-scale on zoom for y-axis
        # Users can use the box zoom or reset tools to adjust view
        import numpy as np

        if len(mean_dwell) > 0:
            y_min = np.min(lower)
            y_max = np.max(upper)
            y_padding = (y_max - y_min) * 0.1
            fig.y_range.start = max(0, y_min - y_padding)
            fig.y_range.end = y_max + y_padding

        return fig
