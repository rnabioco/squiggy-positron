"""
Overlay plot strategy implementation

This module implements the Strategy Pattern for overlaying multiple nanopore reads
on the same axes.
"""

import numpy as np
from bokeh.embed import file_html
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.resources import CDN

from ..constants import MULTI_READ_COLORS, NormalizationMethod, Theme
from ..rendering import ThemeManager
from .base import PlotStrategy


class OverlayPlotStrategy(PlotStrategy):
    """
    Strategy for overlaying multiple nanopore reads on same axes

    This strategy plots multiple reads with different colors on the same
    figure, allowing easy comparison of signal patterns across reads.

    Examples:
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
        if "reads" not in data:
            raise ValueError("Missing required data for overlay plot: reads")

        # Validate read tuples
        reads = data["reads"]
        self._validate_read_tuples(reads)

    def create_plot(self, data: dict, options: dict) -> tuple[str, any]:
        """
        Generate Bokeh plot HTML and figure for overlaid reads

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

        # Calculate alpha blending with a floor to prevent over-transparency
        # Use tiered approach to keep between 0.3 and 0.8 for better visibility
        num_reads = len(reads_data)
        if num_reads == 1:
            alpha = 0.8
        elif num_reads <= 5:
            alpha = 0.7
        elif num_reads <= 20:
            alpha = 0.5  # For ~20 reads, use higher opacity
        elif num_reads <= 50:
            alpha = 0.4
        else:
            alpha = max(0.3, 1.0 / (num_reads**0.5))  # sqrt scaling with floor

        # Create figure
        title = self._format_title(reads_data, normalization, downsample)
        x_label = "Reference Position" if coordinate_space == "sequence" else "Sample"
        fig = self.theme_manager.create_figure(
            title=title,
            x_label=x_label,
            y_label=f"Signal ({normalization.value})",
            height=400,
        )

        # Plot each read with different color
        all_renderers = []
        skipped_reads = []
        for idx, (read_id, signal, _sample_rate) in enumerate(reads_data):
            # Process signal
            processed_signal, _ = self._process_signal(
                signal, normalization, downsample
            )

            # Determine x-coordinates based on coordinate space
            if coordinate_space == "sequence" and aligned_reads:
                # Use BAM alignment positions (reference-anchored coordinates)
                aligned_read = aligned_reads[idx]

                # Build x-coordinates from genomic positions
                # Handle soft-clipping and insertions by using position from aligned bases

                # First, find the first valid genomic position (skip soft-clipped start)
                first_genomic_pos = None
                for base in aligned_read.bases:
                    if base.genomic_pos is not None:
                        first_genomic_pos = base.genomic_pos
                        break

                # Build position array using signal sample indices mapped to genomic coordinates
                ref_positions = []
                for base in aligned_read.bases:
                    num_samples = base.signal_end - base.signal_start
                    if base.genomic_pos is not None:
                        # Mapped base - use genomic position
                        ref_positions.extend([base.genomic_pos] * num_samples)
                    elif first_genomic_pos is not None:
                        # Insertion or soft-clip - use interpolated position
                        # Use the genomic position of the previous mapped base
                        if ref_positions:
                            ref_positions.extend([ref_positions[-1]] * num_samples)
                        else:
                            # Soft-clipped at start - use first genomic position
                            ref_positions.extend([first_genomic_pos] * num_samples)

                # Check if we got any genomic positions at all
                if not ref_positions:
                    # Skip unmapped/unaligned reads in sequence space mode
                    # This can happen if: read is unmapped, all bases are insertions, or soft-clipped
                    print(
                        f"Warning: Skipping read {read_id} - no genomic positions available"
                    )
                    print(
                        f"  Read has {len(aligned_read.bases)} bases, chromosome: {aligned_read.chromosome}"
                    )
                    if aligned_read.bases:
                        first_bases_genomic = [
                            b.genomic_pos for b in aligned_read.bases[:5]
                        ]
                        print(
                            f"  First 5 bases genomic positions: {first_bases_genomic}"
                        )
                    skipped_reads.append(read_id)
                    continue

                # Apply same downsampling to genomic positions as was applied to signal
                # The signal was downsampled, so we need to downsample positions to match
                # Convert to float to allow NaN insertion later (for deletions)
                ref_positions = np.array(ref_positions, dtype=float)

                # Downsample ref_positions if signal was downsampled
                if downsample > 1 and len(ref_positions) > len(processed_signal):
                    # Apply same downsampling stride to genomic positions
                    ref_positions = ref_positions[::downsample]

                # Ensure lengths match after downsampling
                if len(ref_positions) < len(processed_signal):
                    # Pad with last position if needed
                    ref_positions = np.pad(
                        ref_positions,
                        (0, len(processed_signal) - len(ref_positions)),
                        mode="edge",
                    )
                elif len(ref_positions) > len(processed_signal):
                    # Truncate if still too long
                    ref_positions = ref_positions[: len(processed_signal)]

                # For reference-anchored plots, we need to handle repeated positions
                # When multiple samples map to same genomic position, Bokeh draws vertical lines
                # Solution: Keep only unique positions (take mean of signal for each position)
                unique_positions = []
                unique_signals = []

                current_pos = ref_positions[0]
                current_signals = [processed_signal[0]]

                for i in range(1, len(ref_positions)):
                    if ref_positions[i] == current_pos:
                        # Same position - accumulate signal values
                        current_signals.append(processed_signal[i])
                    else:
                        # New position - save mean of accumulated signals
                        unique_positions.append(current_pos)
                        unique_signals.append(np.mean(current_signals))

                        # Check for deletion (position jump > 1)
                        if ref_positions[i] - current_pos > 1:
                            # Insert NaN to break line at deletion
                            unique_positions.append(np.nan)
                            unique_signals.append(np.nan)

                        # Start new position
                        current_pos = ref_positions[i]
                        current_signals = [processed_signal[i]]

                # Don't forget the last position
                unique_positions.append(current_pos)
                unique_signals.append(np.mean(current_signals))

                ref_positions = np.array(unique_positions)
                processed_signal = np.array(unique_signals)

                x = ref_positions
            else:
                # Use raw sample indices (signal space)
                x = np.arange(len(processed_signal))

            source = ColumnDataSource(
                data={
                    "x": x,
                    "y": processed_signal,
                    "read_id": [read_id] * len(x),
                }
            )

            # Get color for this read - use read_colors if provided, otherwise cycle through defaults
            if read_colors and read_id in read_colors:
                color = read_colors[read_id]
            else:
                color = MULTI_READ_COLORS[idx % len(MULTI_READ_COLORS)]

            # Add renderers with calculated alpha
            renderers = self._add_signal_renderers(
                fig=fig,
                source=source,
                color=color,
                alpha=alpha,
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
        alpha: float,
        show_signal_points: bool,
        legend_label: str,
    ) -> list:
        """Add signal line and optional points with configurable alpha"""
        renderers = []

        # Add line
        line_renderer = fig.line(
            x="x",
            y="y",
            source=source,
            color=color,
            line_width=1,
            alpha=alpha,
            legend_label=legend_label,
        )
        renderers.append(line_renderer)

        # Add points if requested (use slightly lower alpha for points)
        if show_signal_points:
            circle_renderer = fig.scatter(
                x="x",
                y="y",
                source=source,
                size=3,
                color=color,
                alpha=alpha * 0.7,  # Slightly more transparent than lines
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
            f"Overlay: {len(reads_data)} reads", normalization, downsample
        )

    def _format_html_title(self, reads_data: list) -> str:
        """Format HTML page title"""
        return self._build_html_title("Overlay", f"{len(reads_data)} reads")
