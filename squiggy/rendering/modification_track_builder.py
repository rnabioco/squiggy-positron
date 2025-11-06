"""
Modification track builder for nanopore signal plots

This module handles creation of separate Bokeh figures showing base modifications
(from modBAM files) as colored rectangles with hover tooltips.
"""

import numpy as np
from bokeh.models import ColumnDataSource, HoverTool

from ..constants import (
    MODIFICATION_CODES,
    MODIFICATION_COLORS,
    Theme,
)


class ModificationTrackBuilder:
    """
    Creates separate Bokeh figures for base modification tracks

    This class handles rendering of base modifications (from modBAM files)
    as a separate track that can be composed with main signal plots. Modifications
    are shown as colored rectangles with opacity based on probability.

    Attributes:
        min_probability: Minimum modification probability threshold (0-1)
        enabled_types: List of enabled modification type codes (None = all)
        overlay_opacity: Base opacity for modification rectangles (0-1)
        theme: Color theme (LIGHT or DARK)

    Examples:
        >>> from squiggy.modification_track_builder import ModificationTrackBuilder
        >>> from squiggy.constants import Theme
        >>>
        >>> # Create builder
        >>> builder = ModificationTrackBuilder(
        ...     min_probability=0.7,
        ...     enabled_types=['m', 'h'],  # Only 5mC and 5hmC
        ...     overlay_opacity=0.8,
        ...     theme=Theme.LIGHT
        ... )
        >>>
        >>> # Build modification track
        >>> mod_fig = builder.build_track(
        ...     sequence="ACGT",
        ...     seq_to_sig_map=[0, 100, 200, 300],
        ...     time_ms=time_array,
        ...     modifications=mod_list
        ... )
    """

    def __init__(
        self,
        min_probability: float = 0.5,
        enabled_types: list[str] | None = None,
        overlay_opacity: float = 0.8,
        theme: Theme = Theme.LIGHT,
    ):
        """
        Initialize modification track builder

        Args:
            min_probability: Minimum modification probability threshold (0-1)
                Modifications below this probability are filtered out
            enabled_types: List of enabled modification type codes
                (e.g., ['m', 'h', 'a']). None = all types enabled
            overlay_opacity: Base opacity for modification rectangles (0-1)
                Final alpha = overlay_opacity * probability
            theme: Color theme (LIGHT or DARK)
        """
        self.min_probability = min_probability
        self.enabled_types = enabled_types
        self.overlay_opacity = overlay_opacity
        self.theme = theme

    def build_track(
        self,
        sequence: str | None,
        seq_to_sig_map: list[int] | None,
        time_ms: np.ndarray,
        sample_rate: int,
        modifications: list | None,
    ):
        """
        Build a Bokeh figure showing base modifications as colored rectangles

        This method creates a separate Bokeh figure for displaying base
        modifications. The figure is designed to be composed with main signal
        plots using Bokeh's column layout.

        Args:
            sequence: DNA/RNA sequence string
            seq_to_sig_map: Mapping from sequence positions to signal indices
            time_ms: Time array in milliseconds (x-axis coordinates)
            sample_rate: Sampling rate (Hz)
            modifications: List of ModificationAnnotation objects

        Returns:
            Bokeh figure with modification track, or None if:
            - No modifications provided
            - No modifications pass filters
            - Sequence or mapping is missing

        Examples:
            >>> builder = ModificationTrackBuilder(min_probability=0.7)
            >>> mod_fig = builder.build_track(
            ...     sequence="ACGT",
            ...     seq_to_sig_map=[0, 100, 200, 300],
            ...     time_ms=time_array,
            ...     sample_rate=4000,
            ...     modifications=mod_list
            ... )
            >>> if mod_fig:
            ...     # Compose with main plot
            ...     from bokeh.layouts import column
            ...     layout = column(mod_fig, main_fig)
        """
        # Return None if no modifications or missing data
        if not modifications or len(modifications) == 0:
            return None

        if not sequence or seq_to_sig_map is None:
            return None

        # Prepare modification data
        mod_data = self._prepare_modification_data(
            sequence=sequence,
            seq_to_sig_map=seq_to_sig_map,
            time_ms=time_ms,
            modifications=modifications,
        )

        # If no modifications after filtering, return None
        if not mod_data["x"]:
            return None

        # Create Bokeh figure
        fig = self._create_mod_figure()

        # Add rectangles and hover tool
        self._add_modification_glyphs(fig, mod_data)

        return fig

    # =========================================================================
    # Private Methods: Data Preparation
    # =========================================================================

    def _prepare_modification_data(
        self,
        sequence: str,
        seq_to_sig_map: list[int],
        time_ms: np.ndarray,
        modifications: list,
    ) -> dict:
        """Prepare data dictionary for modification rectangles"""
        mod_data = {
            "x": [],  # Center position
            "y": [],  # Y position (always 0.5 for single row)
            "left": [],  # Left edge
            "right": [],  # Right edge
            "width": [],  # Width
            "mod_type": [],  # Modification type code
            "mod_name": [],  # Modification name
            "probability": [],  # Modification probability
            "base": [],  # Canonical base
            "position": [],  # Base position
            "color": [],  # Color for each modification
        }

        # Create mapping from sequence position to signal indices
        pos_to_signal = dict(enumerate(seq_to_sig_map))

        for mod in modifications:
            # Filter by probability threshold
            if mod.probability < self.min_probability:
                continue

            # Filter by enabled modification types
            if self.enabled_types is not None and len(self.enabled_types) > 0:
                mod_code_str = str(mod.mod_code)
                if mod_code_str not in self.enabled_types:
                    continue

            # Get signal start/end for this modification position
            if mod.position not in pos_to_signal:
                continue

            sig_start_idx = pos_to_signal[mod.position]

            # Find signal end index (next base or end of sequence)
            sig_end_idx = sig_start_idx + 1  # Default: one sample
            if mod.position + 1 < len(seq_to_sig_map):
                sig_end_idx = pos_to_signal[mod.position + 1]

            # Calculate x-axis range using time_ms
            if sig_start_idx < len(time_ms) and sig_end_idx <= len(time_ms):
                left_x = time_ms[sig_start_idx]
                right_x = time_ms[min(sig_end_idx, len(time_ms) - 1)]
            else:
                continue

            # Get modification color and name
            mod_color = MODIFICATION_COLORS.get(
                mod.mod_code, MODIFICATION_COLORS["default"]
            )
            mod_name = MODIFICATION_CODES.get(mod.mod_code, str(mod.mod_code))

            # Add data point
            mod_data["x"].append((left_x + right_x) / 2)
            mod_data["y"].append(0.5)  # Single row
            mod_data["left"].append(left_x)
            mod_data["right"].append(right_x)
            mod_data["width"].append(right_x - left_x)
            mod_data["mod_type"].append(str(mod.mod_code))
            mod_data["mod_name"].append(mod_name)
            mod_data["probability"].append(mod.probability)
            mod_data["base"].append(sequence[mod.position])
            mod_data["position"].append(mod.position)
            mod_data["color"].append(mod_color)

        return mod_data

    # =========================================================================
    # Private Methods: Figure Creation
    # =========================================================================

    def _create_mod_figure(self):
        """Create empty Bokeh figure for modification track"""
        # Import ThemeManager to create themed figure
        from .theme_manager import ThemeManager

        theme_mgr = ThemeManager(self.theme)

        # Create figure with minimal styling
        fig = theme_mgr.create_figure(
            title="Base Modifications (modBAM)",
            x_label="",  # No label - axes shared with main plot
            y_label="",
            height=80,
        )

        # Hide toolbar (main plot will have toolbar)
        fig.toolbar_location = None

        # Hide y-axis (only one row)
        fig.yaxis.visible = False

        # Hide x-axis labels but keep ticks (axes shared with main plot)
        fig.xaxis.major_label_text_font_size = "0pt"

        # Minimize borders to reduce gap with main plot
        fig.min_border_bottom = 0
        fig.min_border_left = 5
        fig.min_border_right = 5
        fig.min_border_top = 5

        # Set y-axis range
        fig.y_range.start = 0
        fig.y_range.end = 1

        return fig

    def _add_modification_glyphs(self, fig, mod_data: dict):
        """Add modification rectangles and hover tool to figure"""
        # Calculate alpha values (base opacity * probability)
        alphas = [self.overlay_opacity * p for p in mod_data["probability"]]
        mod_data["alpha"] = alphas

        # Create data source
        mod_source = ColumnDataSource(data=mod_data)

        # Create rectangles for modifications
        rects = fig.rect(
            x="x",
            y="y",
            width="width",
            height=0.8,  # Height of rectangle (0-1 range)
            source=mod_source,
            fill_color="color",
            fill_alpha="alpha",
            line_color=None,
        )

        # Add hover tool with modification details
        hover_mod = HoverTool(
            renderers=[rects],
            tooltips=[
                ("Position", "@position"),
                ("Base", "@base"),
                ("Modification", "@mod_name (@mod_type)"),
                ("Probability", "@probability{0.3f}"),
            ],
            mode="mouse",
        )
        fig.add_tools(hover_mod)

    def update_filters(
        self,
        min_probability: float | None = None,
        enabled_types: list[str] | None = None,
    ):
        """
        Update modification filters

        This method allows updating filters without recreating the builder.
        Useful for interactive filtering in UI.

        Args:
            min_probability: New minimum probability threshold (None = no change)
            enabled_types: New list of enabled types (None = no change)

        Examples:
            >>> builder = ModificationTrackBuilder()
            >>> builder.update_filters(min_probability=0.8)
            >>> builder.update_filters(enabled_types=['m', 'a'])
        """
        if min_probability is not None:
            self.min_probability = min_probability

        if enabled_types is not None:
            self.enabled_types = enabled_types

    def get_modification_summary(self, modifications: list | None) -> dict:
        """
        Get summary statistics about modifications

        This method provides statistics about modifications before/after filtering,
        useful for debugging and UI display.

        Args:
            modifications: List of ModificationAnnotation objects

        Returns:
            Dictionary with summary statistics:
                - total_mods: Total number of modifications
                - filtered_mods: Number passing probability threshold
                - enabled_mods: Number passing type filter
                - mod_types: Set of unique modification types present

        Examples:
            >>> builder = ModificationTrackBuilder(min_probability=0.7)
            >>> summary = builder.get_modification_summary(mod_list)
            >>> print(f"Showing {summary['enabled_mods']} of {summary['total_mods']} mods")
        """
        if not modifications:
            return {
                "total_mods": 0,
                "filtered_mods": 0,
                "enabled_mods": 0,
                "mod_types": set(),
            }

        total = len(modifications)
        mod_types = {str(m.mod_code) for m in modifications}

        # Count passing probability filter
        filtered = sum(
            1 for m in modifications if m.probability >= self.min_probability
        )

        # Count passing type filter
        if self.enabled_types is not None and len(self.enabled_types) > 0:
            enabled = sum(
                1
                for m in modifications
                if m.probability >= self.min_probability
                and str(m.mod_code) in self.enabled_types
            )
        else:
            enabled = filtered

        return {
            "total_mods": total,
            "filtered_mods": filtered,
            "enabled_mods": enabled,
            "mod_types": mod_types,
        }
