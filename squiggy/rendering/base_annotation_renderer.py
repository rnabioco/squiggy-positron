"""
Base annotation rendering for nanopore signal plots

This module handles all base annotation rendering, including:
- Calculating base regions (colored background patches)
- Dwell time coloring
- Base label positioning
- Support for both time-based and position-based modes
"""

import numpy as np
from bokeh.models import ColumnDataSource, LinearColorMapper
from bokeh.transform import transform

from ..constants import BASE_ANNOTATION_ALPHA, DEFAULT_POSITION_LABEL_INTERVAL, Theme


class BaseAnnotationRenderer:
    """
    Centralized base annotation rendering for squiggy plots

    This class handles all aspects of rendering base annotations on Bokeh
    figures, including background patches colored by base type or dwell time,
    and base letter labels.

    Supports two rendering modes:
    - Time-based: Uses time coordinates (single read plots)
    - Position-based: Uses base position coordinates (event-aligned plots)

    Attributes:
        base_colors: Dictionary mapping base letter to hex color
        show_dwell_time: Whether to color by dwell time (vs base type)
        show_labels: Whether to show base letter labels

    Examples:
        >>> from squiggy.base_annotation_renderer import BaseAnnotationRenderer
        >>> from squiggy.theme_manager import ThemeManager
        >>> from squiggy.constants import Theme
        >>>
        >>> # Get base colors from theme
        >>> theme_mgr = ThemeManager(Theme.LIGHT)
        >>> base_colors = theme_mgr.get_base_colors()
        >>>
        >>> # Create renderer
        >>> renderer = BaseAnnotationRenderer(
        ...     base_colors=base_colors,
        ...     show_dwell_time=True,
        ...     show_labels=True
        ... )
        >>>
        >>> # Render annotations on figure
        >>> color_mapper = renderer.render_time_based(
        ...     fig=fig,
        ...     sequence="ACGT",
        ...     seq_to_sig_map=[0, 100, 200, 300],
        ...     time_ms=time_array,
        ...     signal=signal_array,
        ...     sample_rate=4000,
        ...     signal_min=-2.5,
        ...     signal_max=2.5
        ... )
    """

    def __init__(
        self,
        base_colors: dict[str, str],
        show_dwell_time: bool = False,
        show_labels: bool = False,
    ):
        """
        Initialize base annotation renderer

        Args:
            base_colors: Dictionary mapping base letter to hex color
                (e.g., {'A': '#00b388', 'C': '#3c8dbc', ...})
            show_dwell_time: If True, color by dwell time instead of base type
            show_labels: If True, add base letter labels above signal
        """
        self.base_colors = base_colors
        self.show_dwell_time = show_dwell_time
        self.show_labels = show_labels

    def render_time_based(
        self,
        fig,
        sequence: str,
        seq_to_sig_map: list[int],
        time_ms: np.ndarray,
        signal: np.ndarray,
        sample_rate: int,
        signal_min: float,
        signal_max: float,
    ):
        """
        Render base annotations for time-based plots (single read mode)

        This method renders base annotations on a figure where the x-axis
        represents time in milliseconds.

        Args:
            fig: Bokeh figure to render on
            sequence: DNA/RNA sequence string (e.g., "ACGT")
            seq_to_sig_map: List mapping sequence position to signal index
            time_ms: Time array in milliseconds
            signal: Signal array
            sample_rate: Sampling rate (Hz)
            signal_min: Minimum signal value (for patch height)
            signal_max: Maximum signal value (for patch height)

        Returns:
            LinearColorMapper if show_dwell_time=True, else None
                The color mapper can be used to add a colorbar to the plot

        Examples:
            >>> renderer = BaseAnnotationRenderer(
            ...     base_colors={'A': '#00b388', ...},
            ...     show_dwell_time=True
            ... )
            >>> color_mapper = renderer.render_time_based(
            ...     fig=fig,
            ...     sequence="ACGT",
            ...     seq_to_sig_map=[0, 100, 200, 300],
            ...     time_ms=time_array,
            ...     signal=signal_array,
            ...     sample_rate=4000,
            ...     signal_min=-2.5,
            ...     signal_max=2.5
            ... )
        """
        # Calculate base regions
        if self.show_dwell_time:
            regions, dwell_times, labels_data = self._calculate_regions_time_dwell(
                sequence=sequence,
                seq_to_sig_map=seq_to_sig_map,
                time_ms=time_ms,
                signal=signal,
                signal_min=signal_min,
                signal_max=signal_max,
                sample_rate=sample_rate,
            )

            # Add dwell time colored patches
            color_mapper = self._add_dwell_patches(fig, regions, dwell_times)

            # Add labels if requested
            if self.show_labels:
                self._add_labels_time_dwell(fig, labels_data)

            return color_mapper

        else:
            base_regions, base_labels_data = self._calculate_regions_time_base_type(
                sequence=sequence,
                seq_to_sig_map=seq_to_sig_map,
                time_ms=time_ms,
                signal=signal,
                signal_min=signal_min,
                signal_max=signal_max,
            )

            # Add base type colored patches
            self._add_base_type_patches(fig, base_regions)

            # Add labels if requested
            if self.show_labels:
                self._add_labels_time_base_type(fig, base_labels_data)

            return None

    def render_position_based(
        self,
        fig,
        base_annotations: list,
        sample_rate: int,
        signal_length: int,
        signal_min: float,
        signal_max: float,
        position_label_interval: int = DEFAULT_POSITION_LABEL_INTERVAL,
        theme: Theme = Theme.LIGHT,
    ):
        """
        Render base annotations for position-based plots (event-aligned mode)

        This method renders base annotations on a figure where the x-axis
        represents base position (or cumulative dwell time if show_dwell_time=True).

        Args:
            fig: Bokeh figure to render on
            base_annotations: List of BaseAnnotation objects with .base and .signal_start
            sample_rate: Sampling rate (Hz)
            signal_length: Total signal length
            signal_min: Minimum signal value (for patch height)
            signal_max: Maximum signal value (for patch height)
            position_label_interval: Show position number every N bases
            theme: Color theme for position label text

        Returns:
            None

        Examples:
            >>> renderer = BaseAnnotationRenderer(
            ...     base_colors={'A': '#00b388', ...},
            ...     show_dwell_time=False,
            ...     show_labels=True
            ... )
            >>> renderer.render_position_based(
            ...     fig=fig,
            ...     base_annotations=aligned_read.bases,
            ...     sample_rate=4000,
            ...     signal_length=len(signal),
            ...     signal_min=-2.5,
            ...     signal_max=2.5
            ... )
        """
        # Calculate base regions
        base_regions = self._calculate_regions_position(
            base_annotations=base_annotations,
            signal_min=signal_min,
            signal_max=signal_max,
            sample_rate=sample_rate,
            signal_length=signal_length,
        )

        # Add base type patches
        self._add_base_type_patches(fig, base_regions)

        # Add labels if requested
        if self.show_labels:
            self._add_labels_position(
                fig=fig,
                base_annotations=base_annotations,
                signal_max=signal_max,
                sample_rate=sample_rate,
                signal_length=signal_length,
                position_label_interval=position_label_interval,
                theme=theme,
            )

    # =========================================================================
    # Private Methods: Region Calculation
    # =========================================================================

    def _calculate_regions_time_dwell(
        self,
        sequence: str,
        seq_to_sig_map: list[int],
        time_ms: np.ndarray,
        signal: np.ndarray,
        signal_min: float,
        signal_max: float,
        sample_rate: int,
    ) -> tuple[list[dict], list[float], list[dict]]:
        """Calculate base regions with dwell time coloring (time-based mode)"""
        all_regions = []
        all_dwell_times = []
        all_labels_data = []

        for seq_pos in range(len(sequence)):
            if seq_pos >= len(seq_to_sig_map):
                break

            base = sequence[seq_pos]
            if base not in self.base_colors:
                continue

            sig_idx = seq_to_sig_map[seq_pos]
            if sig_idx >= len(signal):
                continue

            start_time = time_ms[sig_idx]

            # Calculate end time (x-coordinate)
            if seq_pos + 1 < len(seq_to_sig_map):
                next_sig_idx = seq_to_sig_map[seq_pos + 1]
                end_time = (
                    time_ms[next_sig_idx]
                    if next_sig_idx < len(time_ms)
                    else time_ms[-1]
                )
            else:
                end_time = time_ms[-1]

            # Calculate actual dwell time in milliseconds
            if seq_pos + 1 < len(seq_to_sig_map):
                next_sig_idx = seq_to_sig_map[seq_pos + 1]
                sig_samples = next_sig_idx - sig_idx
            else:
                sig_samples = len(signal) - sig_idx

            dwell_time = sig_samples * 1000 / sample_rate

            all_regions.append(
                {
                    "left": start_time,
                    "right": end_time,
                    "top": signal_max,
                    "bottom": signal_min,
                    "dwell": dwell_time,
                }
            )
            all_dwell_times.append(dwell_time)

            # Store label data
            mid_time = (start_time + end_time) / 2
            all_labels_data.append({"time": mid_time, "y": signal_max, "text": base})

        return all_regions, all_dwell_times, all_labels_data

    def _calculate_regions_time_base_type(
        self,
        sequence: str,
        seq_to_sig_map: list[int],
        time_ms: np.ndarray,
        signal: np.ndarray,
        signal_min: float,
        signal_max: float,
    ) -> tuple[dict[str, list], dict[str, list]]:
        """Calculate base regions grouped by base type (time-based mode)"""
        base_regions = {base: [] for base in ["A", "C", "G", "T"]}
        base_labels_data = {base: [] for base in ["A", "C", "G", "T"]}

        for seq_pos in range(len(sequence)):
            if seq_pos >= len(seq_to_sig_map):
                break

            base = sequence[seq_pos]
            if base not in self.base_colors:
                continue

            sig_idx = seq_to_sig_map[seq_pos]
            if sig_idx >= len(signal):
                continue

            start_time = time_ms[sig_idx]

            # Calculate end time
            if seq_pos + 1 < len(seq_to_sig_map):
                next_sig_idx = seq_to_sig_map[seq_pos + 1]
                end_time = (
                    time_ms[next_sig_idx]
                    if next_sig_idx < len(time_ms)
                    else time_ms[-1]
                )
            else:
                end_time = time_ms[-1]

            base_regions[base].append(
                {
                    "left": start_time,
                    "right": end_time,
                    "top": signal_max,
                    "bottom": signal_min,
                }
            )

            mid_time = (start_time + end_time) / 2
            base_labels_data[base].append(
                {"time": mid_time, "y": signal_max, "text": base}
            )

        return base_regions, base_labels_data

    def _calculate_regions_position(
        self,
        base_annotations: list,
        signal_min: float,
        signal_max: float,
        sample_rate: int,
        signal_length: int,
    ) -> dict[str, list]:
        """Calculate base regions for position-based mode"""
        base_regions = {base: [] for base in ["A", "C", "G", "T", "U"]}

        if self.show_dwell_time:
            # Use cumulative time for x-coordinates
            cumulative_time = 0.0

            for i, base_annotation in enumerate(base_annotations):
                base = base_annotation.base
                if base not in self.base_colors:
                    continue

                # Calculate dwell time
                if i + 1 < len(base_annotations):
                    dwell_samples = (
                        base_annotations[i + 1].signal_start
                        - base_annotation.signal_start
                    )
                else:
                    dwell_samples = signal_length - base_annotation.signal_start

                dwell_time = (dwell_samples / sample_rate) * 1000

                base_regions[base].append(
                    {
                        "left": cumulative_time,
                        "right": cumulative_time + dwell_time,
                        "top": signal_max,
                        "bottom": signal_min,
                    }
                )
                cumulative_time += dwell_time

        else:
            # Use base position for x-coordinates (evenly spaced)
            for i, base_annotation in enumerate(base_annotations):
                base = base_annotation.base
                if base not in self.base_colors:
                    continue

                base_regions[base].append(
                    {
                        "left": i - 0.5,
                        "right": i + 0.5,
                        "top": signal_max,
                        "bottom": signal_min,
                    }
                )

        return base_regions

    # =========================================================================
    # Private Methods: Patch Rendering
    # =========================================================================

    def _add_dwell_patches(
        self, fig, regions: list[dict], dwell_times: list[float]
    ) -> LinearColorMapper:
        """Add background patches colored by dwell time"""
        if not regions:
            return None

        dwell_array = np.array(dwell_times)
        color_mapper = LinearColorMapper(
            palette="Viridis256",
            low=np.percentile(dwell_array, 5),
            high=np.percentile(dwell_array, 95),
        )

        patch_source = ColumnDataSource(
            data={
                "left": [r["left"] for r in regions],
                "right": [r["right"] for r in regions],
                "top": [r["top"] for r in regions],
                "bottom": [r["bottom"] for r in regions],
                "dwell": [r["dwell"] for r in regions],
            }
        )

        fig.quad(
            left="left",
            right="right",
            top="top",
            bottom="bottom",
            source=patch_source,
            fill_color=transform("dwell", color_mapper),
            line_color=None,
            alpha=0.3,
        )

        return color_mapper

    def _add_base_type_patches(self, fig, base_regions: dict[str, list]):
        """Add background patches grouped by base type"""
        for base in ["A", "C", "G", "T", "U"]:
            if base in base_regions and base_regions[base]:
                patch_source = ColumnDataSource(
                    data={
                        "left": [r["left"] for r in base_regions[base]],
                        "right": [r["right"] for r in base_regions[base]],
                        "top": [r["top"] for r in base_regions[base]],
                        "bottom": [r["bottom"] for r in base_regions[base]],
                    }
                )

                fig.quad(
                    left="left",
                    right="right",
                    top="top",
                    bottom="bottom",
                    source=patch_source,
                    color=self.base_colors[base],
                    alpha=BASE_ANNOTATION_ALPHA,
                    legend_label=base,
                )

    # =========================================================================
    # Private Methods: Label Rendering
    # =========================================================================

    def _add_labels_time_dwell(self, fig, labels_data: list[dict]):
        """Add base labels for dwell time mode (single label source)"""
        if not labels_data:
            return

        label_source = ColumnDataSource(
            data={
                "time": [d["time"] for d in labels_data],
                "y": [d["y"] for d in labels_data],
                "text": [d["text"] for d in labels_data],
            }
        )

        fig.text(
            x="time",
            y="y",
            text="text",
            source=label_source,
            text_align="center",
            text_baseline="bottom",
            text_font_size="8pt",
        )

    def _add_labels_time_base_type(self, fig, base_labels_data: dict[str, list]):
        """Add base labels for base type mode (separate sources per base)"""
        for base in ["A", "C", "G", "T"]:
            if base_labels_data[base]:
                label_source = ColumnDataSource(
                    data={
                        "time": [d["time"] for d in base_labels_data[base]],
                        "y": [d["y"] for d in base_labels_data[base]],
                        "text": [d["text"] for d in base_labels_data[base]],
                    }
                )

                fig.text(
                    x="time",
                    y="y",
                    text="text",
                    source=label_source,
                    text_align="center",
                    text_baseline="bottom",
                    text_font_size="8pt",
                )

    def _add_labels_position(
        self,
        fig,
        base_annotations: list,
        signal_max: float,
        sample_rate: int,
        signal_length: int,
        position_label_interval: int,
        theme: Theme,
    ):
        """Add base labels for position-based mode"""
        label_data = []
        position_number_data = []

        if self.show_dwell_time:
            # Use cumulative time for label positioning
            cumulative_time = 0.0
            for i, base_annotation in enumerate(base_annotations):
                base = base_annotation.base
                if base in self.base_colors:
                    # Calculate dwell time
                    if i + 1 < len(base_annotations):
                        dwell_samples = (
                            base_annotations[i + 1].signal_start
                            - base_annotation.signal_start
                        )
                    else:
                        dwell_samples = signal_length - base_annotation.signal_start
                    dwell_time = (dwell_samples / sample_rate) * 1000

                    # Position label at center
                    label_x = cumulative_time + (dwell_time / 2)
                    label_data.append(
                        {
                            "x": label_x,
                            "y": signal_max,
                            "text": base,
                            "color": self.base_colors[base],
                        }
                    )

                    # Add position number at intervals
                    if i % position_label_interval == 0:
                        position_number_data.append(
                            {"x": label_x, "y": signal_max, "text": str(i)}
                        )

                    cumulative_time += dwell_time
        else:
            # Use base position for label positioning
            for i, base_annotation in enumerate(base_annotations):
                base = base_annotation.base
                if base in self.base_colors:
                    label_data.append(
                        {
                            "x": i,
                            "y": signal_max,
                            "text": base,
                            "color": self.base_colors[base],
                        }
                    )

                    # Add position number at intervals
                    if i % position_label_interval == 0:
                        position_number_data.append(
                            {"x": i, "y": signal_max, "text": str(i)}
                        )

        # Add base letters
        if label_data:
            label_source = ColumnDataSource(
                data={
                    "x": [d["x"] for d in label_data],
                    "y": [d["y"] for d in label_data],
                    "text": [d["text"] for d in label_data],
                    "color": [d["color"] for d in label_data],
                }
            )

            fig.text(
                x="x",
                y="y",
                text="text",
                source=label_source,
                text_color="color",
                text_align="center",
                text_baseline="bottom",
                text_font_size="10pt",
            )

        # Add position numbers
        if position_number_data:
            # Get text color from theme
            from ..constants import DARK_THEME, LIGHT_THEME

            text_color = (
                DARK_THEME["axis_text"]
                if theme == Theme.DARK
                else LIGHT_THEME["axis_text"]
            )

            position_source = ColumnDataSource(
                data={
                    "x": [d["x"] for d in position_number_data],
                    "y": [d["y"] for d in position_number_data],
                    "text": [d["text"] for d in position_number_data],
                }
            )

            fig.text(
                x="x",
                y="y",
                text="text",
                source=position_source,
                text_color=text_color,
                text_align="center",
                text_baseline="top",
                text_font_size="8pt",
            )
