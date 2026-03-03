"""
Reference-aligned signal overlay plot strategy

This module implements a strategy that superimposes raw signal traces from
multiple reads aligned to the same genomic reference positions. The x-axis
represents genomic coordinates, allowing direct comparison of ionic current
profiles across reads at the same reference location.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
from bokeh.embed import file_html
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.resources import CDN

from ..constants import MULTI_READ_COLORS, NormalizationMethod, Theme
from ..rendering import BaseAnnotationRenderer, ReferenceTrackRenderer, ThemeManager
from .base import PlotStrategy


@dataclass
class DwellXCoord:
    """X-coordinate info for a single genomic position in dwell-scaled space."""

    center: float
    left: float
    right: float


class ReferenceOverlayPlotStrategy(PlotStrategy):
    """
    Strategy for overlaying multiple reads aligned to genomic reference positions

    Unlike EventAlign (which uses read-local base indices on the x-axis) or
    Overlay in sequence space (which collapses signal per position), this
    strategy computes a per-base mean of the raw signal and plots one point
    per genomic position per read. This keeps traces clean even with many
    overlaid reads.

    Insertions (genomic_pos=None) are skipped, and deletions (gaps > 1 in
    genomic_pos) insert NaN to break the line.

    When a ``read_sample_map`` is provided in options, legend entries are
    grouped by sample name instead of showing individual read IDs.

    Examples:
        >>> from squiggy.plot_strategies.reference_overlay import ReferenceOverlayPlotStrategy
        >>> from squiggy.constants import Theme, NormalizationMethod
        >>>
        >>> strategy = ReferenceOverlayPlotStrategy(Theme.LIGHT)
        >>> data = {
        ...     'reads': [('read_001', signal1, 4000), ('read_002', signal2, 4000)],
        ...     'aligned_reads': [aligned1, aligned2],
        ... }
        >>> options = {
        ...     'normalization': NormalizationMethod.ZNORM,
        ...     'downsample': 1,
        ... }
        >>> html, fig = strategy.create_plot(data, options)
    """

    def __init__(self, theme: Theme):
        super().__init__(theme)
        self.theme_manager = ThemeManager(theme)

    def validate_data(self, data: dict[str, Any]) -> None:
        """
        Validate required data for reference overlay plot

        Required keys:
            - reads: list of (read_id, signal, sample_rate) tuples
            - aligned_reads: list of AlignedRead objects with .bases and .chromosome
        """
        if "reads" not in data:
            raise ValueError("Missing required data for reference overlay plot: reads")
        if "aligned_reads" not in data:
            raise ValueError(
                "Missing required data for reference overlay plot: aligned_reads"
            )

        reads = data["reads"]
        aligned_reads = data["aligned_reads"]

        if not isinstance(reads, list) or len(reads) == 0:
            raise ValueError("reads must be a non-empty list")

        if not isinstance(aligned_reads, list) or len(aligned_reads) == 0:
            raise ValueError("aligned_reads must be a non-empty list")

        self._validate_read_tuples(reads)

        if len(reads) != len(aligned_reads):
            raise ValueError(
                f"reads and aligned_reads must have same length "
                f"(got {len(reads)} and {len(aligned_reads)})"
            )

        # Validate aligned reads have bases and chromosome
        for idx, aligned_read in enumerate(aligned_reads):
            if not hasattr(aligned_read, "bases"):
                raise ValueError(f"Aligned read {idx} must have 'bases' attribute")
            if not hasattr(aligned_read, "chromosome"):
                raise ValueError(f"Aligned read {idx} must have 'chromosome' attribute")

        # All reads must share the same chromosome
        chromosomes = {
            ar.chromosome for ar in aligned_reads if ar.chromosome is not None
        }
        if len(chromosomes) > 1:
            raise ValueError(
                f"All reads must map to the same chromosome, got: {sorted(chromosomes)}"
            )

    def create_plot(
        self, data: dict[str, Any], options: dict[str, Any]
    ) -> tuple[str, Any]:
        """
        Generate reference-aligned signal overlay plot

        Args:
            data: Plot data containing:
                - reads: list of (read_id, signal, sample_rate) tuples
                - aligned_reads: list of AlignedRead objects
                - reference_sequence: optional reference string for track

            options: Plot options containing:
                - normalization: NormalizationMethod (default: NONE)
                - downsample: int downsampling factor (default: 1)
                - show_labels: bool show base labels (default: True)
                - show_signal_points: bool show sample points (default: False)
                - read_colors: dict mapping read_id to color (optional)
                - clip_x_to_alignment: bool clip x-range (default: True)
                - read_sample_map: dict mapping read_id to sample name (optional).
                    When provided, legend shows sample names instead of read IDs.
                - scale_x_by_dwell: bool scale base widths by dwell time (default: False)

        Returns:
            Tuple of (html_string, bokeh_figure_or_layout)
        """
        self.validate_data(data)

        reads_data = data["reads"]
        aligned_reads = data["aligned_reads"]
        reference_sequence = data.get("reference_sequence", "")

        from ..constants import DEFAULT_DOWNSAMPLE

        normalization = options.get("normalization", NormalizationMethod.NONE)
        downsample = options.get("downsample", DEFAULT_DOWNSAMPLE)
        show_labels = options.get("show_labels", True)
        show_signal_points = options.get("show_signal_points", False)
        read_colors = options.get("read_colors", None)
        clip_x_to_alignment = options.get("clip_x_to_alignment", True)
        read_sample_map = options.get("read_sample_map", None)
        scale_x_by_dwell = options.get("scale_x_by_dwell", False)

        # Alpha blending tiered by read count (same as Overlay)
        num_reads = len(reads_data)
        if num_reads == 1:
            alpha = 0.8
        elif num_reads <= 5:
            alpha = 0.7
        elif num_reads <= 20:
            alpha = 0.5
        elif num_reads <= 50:
            alpha = 0.4
        else:
            alpha = max(0.3, 1.0 / (num_reads**0.5))

        # Build dwell x-map if scaling by dwell time
        x_map: dict[int, DwellXCoord] | None = None
        if scale_x_by_dwell:
            dwell_map = self._build_consensus_dwell_map(aligned_reads)
            if dwell_map:
                # Determine genomic span for normalization
                all_gpos = sorted(dwell_map.keys())
                genomic_span = all_gpos[-1] - all_gpos[0] + 1
                x_map = self._build_dwell_x_map(dwell_map, genomic_span)

        # Create figure
        x_label = "Dwell-Scaled Genomic Position" if x_map else "Genomic Position"
        title = self._build_title(
            f"Reference Overlay: {num_reads} reads", normalization, downsample
        )
        fig = self.theme_manager.create_figure(
            title=title,
            x_label=x_label,
            y_label=f"Signal ({normalization.value})",
            height=400,
        )

        # Process all signals and build genomic coordinates
        all_renderers = []
        all_genomic_min = []
        all_genomic_max = []
        all_signal_vals = []
        seen_samples: set[str] = set()

        for idx, ((read_id, signal, _sample_rate), aligned_read) in enumerate(
            zip(reads_data, aligned_reads, strict=True)
        ):
            # Normalize signal (downsample=1 here; we handle per-base averaging)
            processed, _ = self._process_signal(signal, normalization, downsample=1)

            # Build genomic signal coordinates (per-base mean)
            x_coords, y_coords = self._build_genomic_signal_coordinates(
                processed, aligned_read.bases, downsample, x_map=x_map
            )

            if len(x_coords) == 0:
                continue

            # Track ranges (ignoring NaN)
            valid_x = x_coords[~np.isnan(x_coords)]
            valid_y = y_coords[~np.isnan(y_coords)]
            if len(valid_x) > 0:
                all_genomic_min.append(np.min(valid_x))
                all_genomic_max.append(np.max(valid_x))
            if len(valid_y) > 0:
                all_signal_vals.extend([np.min(valid_y), np.max(valid_y)])

            # Choose color
            if read_colors and read_id in read_colors:
                color = read_colors[read_id]
            else:
                color = MULTI_READ_COLORS[idx % len(MULTI_READ_COLORS)]

            # Determine legend label: sample name (first occurrence only) or read_id
            legend_kwargs: dict[str, str] = {}
            if read_sample_map and read_id in read_sample_map:
                sample_name = read_sample_map[read_id]
                if sample_name not in seen_samples:
                    legend_kwargs["legend_label"] = sample_name
                    seen_samples.add(sample_name)
                # Subsequent reads in same sample: no legend entry
            else:
                legend_kwargs["legend_label"] = read_id[:12]

            # Create data source
            source = ColumnDataSource(
                data={
                    "x": x_coords,
                    "y": y_coords,
                    "read_id": [read_id] * len(x_coords),
                }
            )

            # Add line renderer
            line_renderer = fig.line(
                x="x",
                y="y",
                source=source,
                color=color,
                line_width=1,
                alpha=alpha,
                **legend_kwargs,
            )
            all_renderers.append(line_renderer)

            if show_signal_points:
                circle_renderer = fig.scatter(
                    x="x",
                    y="y",
                    source=source,
                    size=3,
                    color=color,
                    alpha=alpha * 0.7,
                    **legend_kwargs,
                )
                all_renderers.append(circle_renderer)

        # Add consensus base annotation patches
        if all_signal_vals and show_labels:
            signal_min = min(all_signal_vals)
            signal_max = max(all_signal_vals)
            consensus_map = self._build_consensus_base_map(aligned_reads)
            self._add_genomic_base_annotations(
                fig, consensus_map, signal_min, signal_max, x_map=x_map
            )

        # Add hover tool
        if all_renderers:
            hover = HoverTool(
                renderers=all_renderers,
                tooltips=[
                    ("Read", "@read_id"),
                    ("Genomic Position", "@x{0,0}"),
                    ("Signal", "@y{0.2f}"),
                ],
                mode="mouse",
            )
            fig.add_tools(hover)

        # Clip x-range to genomic span
        if clip_x_to_alignment and all_genomic_min and all_genomic_max:
            from bokeh.models import Range1d

            x_min = min(all_genomic_min)
            x_max = max(all_genomic_max)
            fig.x_range = Range1d(start=x_min - 0.5, end=x_max + 0.5)

        # Configure legend
        self.theme_manager.configure_legend(fig)

        # Optional reference track
        ref_fig = None
        if reference_sequence and all_genomic_min and all_genomic_max:
            ref_fig = self._create_reference_track(
                fig, reference_sequence, aligned_reads
            )

        # Generate HTML
        html_title = self._build_html_title("Reference Overlay", f"{num_reads} reads")

        if ref_fig is not None:
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
            object.__setattr__(layout, "main_plot", fig)
            html = file_html(layout, CDN, title=html_title)
            return html, layout
        else:
            html = file_html(fig, CDN, title=html_title)
            return html, fig

    # =========================================================================
    # Private Methods: Coordinate Building
    # =========================================================================

    def _build_genomic_signal_coordinates(
        self,
        signal: np.ndarray,
        base_annotations: list,
        downsample: int,
        x_map: dict[int, DwellXCoord] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build x/y arrays mapping per-base mean signal to genomic coordinates.

        For each base with genomic_pos, computes the mean of
        signal[signal_start:signal_end:downsample] and places a single point
        at the genomic position (or dwell-scaled center if x_map provided).

        Insertions (genomic_pos=None) are skipped.
        Deletions (gap > 1 in consecutive genomic_pos) insert NaN to break the line.
        """
        x_coords = []
        y_coords = []
        prev_genomic_pos = None

        for base_ann in base_annotations:
            if base_ann.genomic_pos is None:
                # Insertion — skip
                continue

            genomic_pos = base_ann.genomic_pos
            start_idx = base_ann.signal_start
            end_idx = base_ann.signal_end

            # Clamp to signal length
            start_idx = max(0, min(start_idx, len(signal)))
            end_idx = max(start_idx, min(end_idx, len(signal)))

            # Check for deletion gap — insert NaN to break the line
            if prev_genomic_pos is not None:
                gap = abs(genomic_pos - prev_genomic_pos)
                if gap > 1:
                    x_coords.append(np.nan)
                    y_coords.append(np.nan)

            # Get signal samples for this base, with downsampling
            base_signal = signal[start_idx:end_idx:downsample]

            if len(base_signal) == 0:
                prev_genomic_pos = genomic_pos
                continue

            # Per-base mean: use dwell-scaled center or raw genomic position
            if x_map and genomic_pos in x_map:
                x_coords.append(x_map[genomic_pos].center)
            else:
                x_coords.append(float(genomic_pos))
            y_coords.append(float(np.mean(base_signal)))
            prev_genomic_pos = genomic_pos

        return np.array(x_coords, dtype=float), np.array(y_coords, dtype=float)

    # =========================================================================
    # Private Methods: Consensus & Annotations
    # =========================================================================

    def _build_consensus_base_map(self, aligned_reads: list) -> dict[int, str]:
        """
        Build consensus base identity at each genomic position via majority vote.

        Returns dict mapping genomic_pos -> base letter.
        """
        position_bases: dict[int, list[str]] = {}

        for aligned_read in aligned_reads:
            for base_ann in aligned_read.bases:
                if base_ann.genomic_pos is not None:
                    pos = base_ann.genomic_pos
                    if pos not in position_bases:
                        position_bases[pos] = []
                    position_bases[pos].append(base_ann.base)

        consensus: dict[int, str] = {}
        for pos, bases in position_bases.items():
            counter = Counter(bases)
            consensus[pos] = counter.most_common(1)[0][0]

        return consensus

    def _add_genomic_base_annotations(
        self,
        fig,
        consensus_map: dict[int, str],
        signal_min: float,
        signal_max: float,
        x_map: dict[int, DwellXCoord] | None = None,
    ) -> None:
        """
        Draw colored background patches at each genomic position
        using the consensus base for coloring.

        When x_map is provided, patch widths are dwell-scaled.
        """
        base_colors = self.theme_manager.get_base_colors()

        # Group positions by base for batch rendering
        base_regions: dict[str, list[dict]] = {b: [] for b in ["A", "C", "G", "T", "U"]}

        for pos, base in sorted(consensus_map.items()):
            upper_base = base.upper()
            if upper_base in base_regions:
                if x_map and pos in x_map:
                    left = x_map[pos].left
                    right = x_map[pos].right
                else:
                    left = pos - 0.5
                    right = pos + 0.5
                base_regions[upper_base].append(
                    {
                        "left": left,
                        "right": right,
                        "top": signal_max,
                        "bottom": signal_min,
                    }
                )

        # Use BaseAnnotationRenderer's batch rendering approach
        renderer = BaseAnnotationRenderer(
            base_colors=base_colors,
            show_dwell_time=False,
            show_labels=False,
        )
        renderer._add_base_type_patches(fig, base_regions)

    # =========================================================================
    # Private Methods: Dwell Scaling
    # =========================================================================

    def _build_consensus_dwell_map(self, aligned_reads: list) -> dict[int, float]:
        """
        Compute median dwell (signal_end - signal_start) per genomic position
        across all reads.

        Returns dict mapping genomic_pos -> median number of signal samples.
        """
        position_dwells: dict[int, list[int]] = {}

        for aligned_read in aligned_reads:
            for base_ann in aligned_read.bases:
                if base_ann.genomic_pos is not None:
                    pos = base_ann.genomic_pos
                    n_samples = base_ann.signal_end - base_ann.signal_start
                    if n_samples > 0:
                        if pos not in position_dwells:
                            position_dwells[pos] = []
                        position_dwells[pos].append(n_samples)

        return {
            pos: float(np.median(dwells)) for pos, dwells in position_dwells.items()
        }

    def _build_dwell_x_map(
        self, dwell_map: dict[int, float], genomic_span: int
    ) -> dict[int, DwellXCoord]:
        """
        Build cumulative x-coordinate map where each base's width is
        proportional to its median dwell time.

        The total span is normalized to match the genomic span so that
        the visual scale is preserved.

        Returns dict mapping genomic_pos -> DwellXCoord(center, left, right).
        """
        positions = sorted(dwell_map.keys())
        if not positions:
            return {}

        dwells = np.array([dwell_map[p] for p in positions])
        total_dwell = dwells.sum()

        if total_dwell == 0:
            return {}

        # Normalize widths so total equals genomic_span
        widths = dwells * (genomic_span / total_dwell)

        # Build cumulative coordinates starting from the first position
        origin = positions[0] - 0.5  # align left edge with first position
        x_map: dict[int, DwellXCoord] = {}
        cursor = origin

        for pos, width in zip(positions, widths, strict=True):
            left = cursor
            right = cursor + width
            center = (left + right) / 2.0
            x_map[pos] = DwellXCoord(center=center, left=left, right=right)
            cursor = right

        return x_map

    # =========================================================================
    # Private Methods: Reference Track
    # =========================================================================

    def _create_reference_track(
        self, main_fig, reference_sequence: str, aligned_reads: list
    ):
        """Create a reference track sub-figure linked to the main plot x-range."""
        # Collect all unique genomic positions and build consensus
        consensus_map = self._build_consensus_base_map(aligned_reads)
        if not consensus_map:
            return None

        positions = sorted(consensus_map.keys())
        # Build query sequence from consensus for mismatch highlighting
        query_sequence = "".join(consensus_map[p] for p in positions)

        # Determine which part of the reference to show
        min_pos = min(positions)
        max_pos = max(positions)

        # Extract relevant portion of reference sequence
        if len(reference_sequence) > max_pos:
            ref_slice = reference_sequence[min_pos : max_pos + 1]
            ref_positions = list(range(min_pos, max_pos + 1))
        else:
            # Reference shorter than expected — use consensus positions
            ref_slice = query_sequence
            ref_positions = positions

        try:
            ref_renderer = ReferenceTrackRenderer(self.theme_manager)
            ref_fig = ref_renderer.create_reference_track(
                reference_sequence=ref_slice,
                positions=ref_positions,
                x_label="",
                title="Reference",
                height=60,
                query_sequence=query_sequence
                if len(query_sequence) == len(ref_slice)
                else None,
            )

            # Link x-range for synchronized zoom/pan
            ref_fig.x_range = main_fig.x_range

            ref_fig.min_border_top = 0
            ref_fig.min_border_bottom = 0
            ref_fig.min_border_left = 5
            ref_fig.min_border_right = 5

            return ref_fig
        except Exception:
            return None
