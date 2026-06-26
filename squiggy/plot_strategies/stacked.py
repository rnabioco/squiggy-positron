"""
Stacked plot strategy implementation

This module implements the Strategy Pattern for stacking multiple nanopore reads
vertically with offsets.
"""

import logging

import numpy as np
from bokeh.embed import file_html
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.resources import CDN

from ..constants import MULTI_READ_COLORS, NormalizationMethod, Theme
from ..rendering import ThemeManager
from .base import PlotStrategy

logger = logging.getLogger(__name__)


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
        title = self._format_title(
            reads_data, normalization, downsample, aligned_reads=aligned_reads
        )
        x_label = "Reference Position" if coordinate_space == "sequence" else "Sample"
        fig = self.theme_manager.create_figure(
            title=title,
            x_label=x_label,
            y_label=f"Signal ({normalization.value}) + offset",
            height=400,
        )

        # Second pass: plot with offsets
        all_renderers = []
        skipped_reads = []
        for idx, (read_id, signal, _sample_rate) in enumerate(processed_reads):
            # Apply vertical offset
            offset = idx * offset_step
            y_offset = signal + offset

            # Determine x-coordinates based on coordinate space
            if coordinate_space == "sequence" and aligned_reads:
                # Use BAM alignment positions (reference-anchored coordinates)
                aligned_read = aligned_reads[idx]

                # Build x-coordinates from genomic positions (shared helper).
                # Handles soft-clipping and insertions.
                ref_positions = self._build_genomic_ref_positions(aligned_read)

                # Check if we got any genomic positions at all
                if not ref_positions:
                    # Skip unmapped/unaligned reads in sequence space mode
                    # This can happen if: read is unmapped, all bases are insertions, or soft-clipped
                    debug_info = (
                        f"Skipping read {read_id} - no genomic positions available. "
                    )
                    debug_info += f"Read has {len(aligned_read.bases)} bases, chromosome: {aligned_read.chromosome}"
                    if aligned_read.bases:
                        first_bases_genomic = [
                            b.genomic_pos for b in aligned_read.bases[:5]
                        ]
                        debug_info += (
                            f". First 5 bases genomic positions: {first_bases_genomic}"
                        )
                    logger.warning(debug_info)
                    skipped_reads.append(read_id)
                    continue

                # Collapse repeated positions (mean per position, NaN at deletions)
                ref_positions, y_offset = self._collapse_to_genomic_positions(
                    ref_positions, y_offset, downsample
                )

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
        html_title = self._format_html_title(reads_data, aligned_reads=aligned_reads)
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
        aligned_reads=None,
    ) -> str:
        """Format plot title with optional reference info"""
        ref_suffix = ""
        if aligned_reads:
            chromosome = getattr(aligned_reads[0], "chromosome", None)
            ref_suffix = self._format_ref_suffix(chromosome)
        return self._build_title(
            f"Stacked: {len(reads_data)} reads{ref_suffix}",
            normalization,
            downsample,
        )

    def _format_html_title(self, reads_data: list, aligned_reads=None) -> str:
        """Format HTML page title with optional reference info"""
        ref_suffix = ""
        if aligned_reads:
            chromosome = getattr(aligned_reads[0], "chromosome", None)
            ref_suffix = self._format_ref_suffix(chromosome)
        return self._build_html_title("Stacked", f"{len(reads_data)} reads{ref_suffix}")
