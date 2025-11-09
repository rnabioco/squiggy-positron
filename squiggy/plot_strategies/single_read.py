"""
Single read plot strategy implementation

This module implements the Strategy Pattern for single nanopore read visualization.
"""

import numpy as np
from bokeh.embed import file_html
from bokeh.layouts import column
from bokeh.models import ColorBar, ColumnDataSource, HoverTool
from bokeh.resources import CDN

from ..constants import NormalizationMethod, Theme
from ..rendering import BaseAnnotationRenderer, ModificationTrackBuilder, ThemeManager
from .base import PlotStrategy


class SingleReadPlotStrategy(PlotStrategy):
    """
    Strategy for plotting single nanopore reads

    This strategy handles visualization of individual reads with optional:
    - Base annotations (colored patches + labels)
    - Dwell time coloring
    - Modification tracks (modBAM)
    - Signal normalization
    - Downsampling

    The strategy uses composition of ThemeManager, BaseAnnotationRenderer,
    and ModificationTrackBuilder to create plots.

    Examples:
        >>> from squiggy.plot_strategies.single_read import SingleReadPlotStrategy
        >>> from squiggy.constants import Theme, NormalizationMethod
        >>>
        >>> strategy = SingleReadPlotStrategy(Theme.LIGHT)
        >>>
        >>> data = {
        ...     'signal': read_obj.signal,
        ...     'read_id': 'read_001',
        ...     'sample_rate': 4000,
        ...     'sequence': 'ACGT',
        ...     'seq_to_sig_map': [0, 100, 200, 300],
        ... }
        >>>
        >>> options = {
        ...     'normalization': NormalizationMethod.ZNORM,
        ...     'downsample': 1,
        ...     'show_dwell_time': False,
        ...     'show_labels': True,
        ...     'show_signal_points': False,
        ... }
        >>>
        >>> html, fig = strategy.create_plot(data, options)
    """

    def __init__(self, theme: Theme):
        """
        Initialize single read plot strategy

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
            - signal: np.ndarray of raw signal values
            - read_id: str identifier for the read
            - sample_rate: int sampling rate in Hz
        """
        required = ["signal", "read_id", "sample_rate"]
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"Missing required data for single read plot: {missing}")

        # Validate types
        if not isinstance(data["signal"], np.ndarray):
            raise ValueError("signal must be a numpy array")
        if not isinstance(data["read_id"], str):
            raise ValueError("read_id must be a string")
        if not isinstance(data["sample_rate"], (int, float)):
            raise ValueError("sample_rate must be a number")

    def create_plot(self, data: dict, options: dict) -> tuple[str, any]:
        """
        Generate Bokeh plot HTML and figure for a single read

        Args:
            data: Plot data dictionary containing:
                - signal (required): np.ndarray of raw signal values
                - read_id (required): str identifier for the read
                - sample_rate (required): int sampling rate in Hz
                - sequence (optional): str DNA/RNA sequence
                - seq_to_sig_map (optional): list[int] mapping seq positions to signal
                - modifications (optional): list of ModificationAnnotation objects
                - aligned_read (optional): AlignedRead object for reference-anchored mode

            options: Plot options dictionary containing:
                - normalization: NormalizationMethod enum (default: NONE)
                - downsample: int downsampling factor (default: 1)
                - show_dwell_time: bool color by dwell time (default: False)
                - show_labels: bool show base labels (default: True)
                - show_signal_points: bool show individual points (default: False)
                - scale_dwell_time: bool scale x-axis by dwell time (default: False)
                - show_modification_overlay: bool show mod track (default: True)
                - modification_overlay_opacity: float mod opacity 0-1 (default: 0.6)
                - min_mod_probability: float min mod probability (default: 0.5)
                - enabled_mod_types: list[str] enabled mod types (default: None)
                - coordinate_space: str ('signal' or 'sequence', default: 'signal')

        Returns:
            Tuple of (html_string, bokeh_figure_or_layout)

        Raises:
            ValueError: If required data is missing
        """
        # Validate data
        self.validate_data(data)

        # Extract data
        signal = data["signal"]
        read_id = data["read_id"]
        sample_rate = data["sample_rate"]
        sequence = data.get("sequence")
        seq_to_sig_map = data.get("seq_to_sig_map")
        modifications = data.get("modifications")
        aligned_read = data.get("aligned_read")

        from ..constants import DEFAULT_DOWNSAMPLE

        # Extract options with defaults
        normalization = options.get("normalization", NormalizationMethod.NONE)
        downsample = options.get("downsample", DEFAULT_DOWNSAMPLE)
        show_dwell_time = options.get("show_dwell_time", False)
        show_labels = options.get("show_labels", True)
        show_signal_points = options.get("show_signal_points", False)
        scale_dwell_time = options.get("scale_dwell_time", False)
        show_modification_overlay = options.get("show_modification_overlay", True)
        modification_overlay_opacity = options.get("modification_overlay_opacity", 0.6)
        min_mod_probability = options.get("min_mod_probability", 0.5)
        enabled_mod_types = options.get("enabled_mod_types", None)
        coordinate_space = options.get("coordinate_space", "signal")

        # Process signal (normalize and downsample)
        signal, seq_to_sig_map = self._process_signal(
            signal, normalization, downsample, seq_to_sig_map
        )

        # Create x-axis - use genomic positions if in sequence space with alignment
        if coordinate_space == "sequence" and aligned_read:
            time_ms, signal, x_label = self._create_genomic_position_axis(
                signal=signal,
                aligned_read=aligned_read,
                downsample=downsample,
            )
        else:
            # Original behavior: time-based or sample-based x-axis
            time_ms, x_label = self._create_x_axis(
                signal=signal,
                sample_rate=sample_rate,
                sequence=sequence,
                seq_to_sig_map=seq_to_sig_map,
                scale_dwell_time=scale_dwell_time,
            )

        # Create main figure
        title = self._format_title(read_id, normalization, downsample)
        fig = self.theme_manager.create_figure(
            title=title,
            x_label=x_label,
            y_label=f"Signal ({normalization.value})",
            height=400,
        )

        # Add base annotations if available
        color_mapper = None
        if sequence and seq_to_sig_map is not None:
            color_mapper = self._add_base_annotations(
                fig=fig,
                signal=signal,
                time_ms=time_ms,
                sequence=sequence,
                seq_to_sig_map=seq_to_sig_map,
                sample_rate=sample_rate,
                show_dwell_time=show_dwell_time,
                show_labels=show_labels,
            )

        # Add signal line and points
        self._add_signal_renderers(
            fig=fig,
            time_ms=time_ms,
            signal=signal,
            show_signal_points=show_signal_points,
        )

        # Add color bar if showing dwell time
        if color_mapper is not None:
            color_bar = ColorBar(
                color_mapper=color_mapper,
                label_standoff=12,
                location=(0, 0),
                title="Dwell Time (ms)",
                title_standoff=15,
            )
            fig.add_layout(color_bar, "right")

        # Legend configuration
        if not show_dwell_time and sequence:
            self.theme_manager.configure_legend(fig)

        # Create modification track if available
        mod_fig = self._create_modification_track(
            sequence=sequence,
            seq_to_sig_map=seq_to_sig_map,
            time_ms=time_ms,
            sample_rate=sample_rate,
            modifications=modifications,
            show_modification_overlay=show_modification_overlay,
            modification_overlay_opacity=modification_overlay_opacity,
            min_mod_probability=min_mod_probability,
            enabled_mod_types=enabled_mod_types,
        )

        # Generate HTML
        if mod_fig is not None:
            # Link x-axes for synchronized zoom/pan
            mod_fig.x_range = fig.x_range

            # Minimize borders to reduce gap
            fig.min_border_top = 0
            fig.min_border_left = 5
            fig.min_border_right = 5

            # Create column layout
            layout = column(
                mod_fig,
                fig,
                sizing_mode="stretch_width",
                spacing=0,
            )

            # Store reference to main plot
            object.__setattr__(layout, "main_plot", fig)

            html = file_html(layout, CDN, title=f"Squiggy: {read_id}")
            return html, layout
        else:
            # No modifications - return single plot
            html = file_html(fig, CDN, title=f"Squiggy: {read_id}")
            return html, fig

    # =========================================================================
    # Private Methods: Signal Processing
    # =========================================================================

    def _create_x_axis(
        self,
        signal: np.ndarray,
        sample_rate: int,
        sequence: str | None,
        seq_to_sig_map: list[int] | None,
        scale_dwell_time: bool,
    ) -> tuple[np.ndarray, str]:
        """Create x-axis array and label"""
        if sequence and seq_to_sig_map is not None and len(seq_to_sig_map) > 0:
            if scale_dwell_time:
                # Cumulative dwell time mode
                time_ms, x_label = self._create_dwell_time_axis(
                    signal, sample_rate, sequence, seq_to_sig_map
                )
            else:
                # Base position mode
                time_ms, x_label = self._create_base_position_axis(
                    signal, sequence, seq_to_sig_map
                )
        else:
            # Regular time mode
            time_ms = np.arange(len(signal)) * 1000 / sample_rate
            x_label = "Time (ms)"

        return time_ms, x_label

    def _create_dwell_time_axis(
        self,
        signal: np.ndarray,
        sample_rate: int,
        sequence: str,
        seq_to_sig_map: list[int],
    ) -> tuple[np.ndarray, str]:
        """Create cumulative dwell time x-axis"""
        cumulative_time_ms = np.zeros(len(signal))
        current_time = 0.0

        for seq_pos in range(len(sequence)):
            if seq_pos >= len(seq_to_sig_map):
                break

            sig_start = seq_to_sig_map[seq_pos]

            # Find end of this base's signal region
            if seq_pos + 1 < len(seq_to_sig_map):
                sig_end = seq_to_sig_map[seq_pos + 1]
            else:
                sig_end = len(signal)

            # Calculate dwell time in milliseconds
            dwell_time_ms = (sig_end - sig_start) * 1000 / sample_rate

            # Assign cumulative time to signal samples
            for i in range(sig_start, min(sig_end, len(signal))):
                progress = (i - sig_start) / max(1, sig_end - sig_start)
                cumulative_time_ms[i] = current_time + (progress * dwell_time_ms)

            current_time += dwell_time_ms

        return cumulative_time_ms, "Cumulative Dwell Time (ms)"

    def _create_base_position_axis(
        self,
        signal: np.ndarray,
        sequence: str,
        seq_to_sig_map: list[int],
    ) -> tuple[np.ndarray, str]:
        """Create base position x-axis"""
        base_positions = np.zeros(len(signal))

        for seq_pos in range(len(sequence)):
            if seq_pos >= len(seq_to_sig_map):
                break

            sig_start = seq_to_sig_map[seq_pos]

            # Find end of this base's signal region
            if seq_pos + 1 < len(seq_to_sig_map):
                sig_end = seq_to_sig_map[seq_pos + 1]
            else:
                sig_end = len(signal)

            # Assign base position to signal samples
            for i in range(sig_start, min(sig_end, len(signal))):
                progress = (i - sig_start) / max(1, sig_end - sig_start)
                base_positions[i] = seq_pos + progress

        return base_positions, "Base Position"

    def _create_genomic_position_axis(
        self,
        signal: np.ndarray,
        aligned_read,
        downsample: int,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """
        Create genomic position x-axis for reference-anchored plotting

        Similar to overlay/stacked strategies, collapses multiple samples at same
        genomic position by averaging to avoid vertical lines.

        Returns:
            Tuple of (positions, collapsed_signal, label)
        """
        # Build genomic position array from aligned read
        ref_positions = []

        # Find first valid genomic position
        first_genomic_pos = None
        for base in aligned_read.bases:
            if base.genomic_pos is not None:
                first_genomic_pos = base.genomic_pos
                break

        # Build position array
        for base in aligned_read.bases:
            num_samples = base.signal_end - base.signal_start
            if base.genomic_pos is not None:
                # Mapped base - use genomic position
                ref_positions.extend([base.genomic_pos] * num_samples)
            elif first_genomic_pos is not None:
                # Insertion or soft-clip - use last valid position
                if ref_positions:
                    ref_positions.extend([ref_positions[-1]] * num_samples)
                else:
                    ref_positions.extend([first_genomic_pos] * num_samples)

        if not ref_positions:
            # Fallback to sample indices if no genomic positions available
            return np.arange(len(signal)), signal, "Sample"

        # Convert to float for NaN support
        ref_positions = np.array(ref_positions, dtype=float)

        # Downsample if needed
        if downsample > 1 and len(ref_positions) > len(signal):
            ref_positions = ref_positions[::downsample]

        # Ensure lengths match
        if len(ref_positions) < len(signal):
            ref_positions = np.pad(
                ref_positions,
                (0, len(signal) - len(ref_positions)),
                mode="edge",
            )
        elif len(ref_positions) > len(signal):
            ref_positions = ref_positions[: len(signal)]

        # Collapse repeated positions (average signal at each unique position)
        unique_positions = []
        unique_signals = []

        current_pos = ref_positions[0]
        current_signals = [signal[0]]

        for i in range(1, len(ref_positions)):
            if ref_positions[i] == current_pos:
                # Same position - accumulate signal values
                current_signals.append(signal[i])
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
                current_signals = [signal[i]]

        # Don't forget the last position
        unique_positions.append(current_pos)
        unique_signals.append(np.mean(current_signals))

        # Convert to arrays
        collapsed_positions = np.array(unique_positions)
        collapsed_signal = np.array(unique_signals)

        return collapsed_positions, collapsed_signal, "Reference Position"

    # =========================================================================
    # Private Methods: Rendering
    # =========================================================================

    def _add_base_annotations(
        self,
        fig,
        signal: np.ndarray,
        time_ms: np.ndarray,
        sequence: str,
        seq_to_sig_map: list[int],
        sample_rate: int,
        show_dwell_time: bool,
        show_labels: bool,
    ):
        """Add base annotations using BaseAnnotationRenderer"""
        base_colors = self.theme_manager.get_base_colors()

        renderer = BaseAnnotationRenderer(
            base_colors=base_colors,
            show_dwell_time=show_dwell_time,
            show_labels=show_labels,
        )

        signal_min = np.min(signal)
        signal_max = np.max(signal)

        color_mapper = renderer.render_time_based(
            fig=fig,
            sequence=sequence,
            seq_to_sig_map=seq_to_sig_map,
            time_ms=time_ms,
            signal=signal,
            sample_rate=sample_rate,
            signal_min=signal_min,
            signal_max=signal_max,
        )

        return color_mapper

    def _add_signal_renderers(
        self,
        fig,
        time_ms: np.ndarray,
        signal: np.ndarray,
        show_signal_points: bool,
    ):
        """Add signal line and optional points"""
        # Create data source
        source = ColumnDataSource(
            data={
                "time": time_ms,
                "signal": signal,
                "sample": np.arange(len(signal)),
            }
        )

        # Get signal color
        signal_color = self.theme_manager.get_signal_color()

        # Add line
        line_renderer = fig.line(
            x="time",
            y="signal",
            source=source,
            color=signal_color,
            line_width=1,
            alpha=0.8,
        )

        # Add points if requested
        renderers = [line_renderer]
        if show_signal_points:
            circle_renderer = fig.circle(
                x="time",
                y="signal",
                source=source,
                size=3,
                color=signal_color,
                alpha=0.5,
            )
            renderers.append(circle_renderer)

        # Add hover tool
        hover = HoverTool(
            renderers=renderers,
            tooltips=[
                ("Time", "@time{0.2f} ms"),
                ("Signal", "@signal{0.2f}"),
                ("Sample", "@sample"),
            ],
            mode="mouse",
        )
        fig.add_tools(hover)

    def _create_modification_track(
        self,
        sequence: str | None,
        seq_to_sig_map: list[int] | None,
        time_ms: np.ndarray,
        sample_rate: int,
        modifications: list | None,
        show_modification_overlay: bool,
        modification_overlay_opacity: float,
        min_mod_probability: float,
        enabled_mod_types: list[str] | None,
    ):
        """Create modification track using ModificationTrackBuilder"""
        if not show_modification_overlay:
            return None

        builder = ModificationTrackBuilder(
            min_probability=min_mod_probability,
            enabled_types=enabled_mod_types,
            overlay_opacity=modification_overlay_opacity,
            theme=self.theme,
        )

        mod_fig = builder.build_track(
            sequence=sequence,
            seq_to_sig_map=seq_to_sig_map,
            time_ms=time_ms,
            sample_rate=sample_rate,
            modifications=modifications,
        )

        return mod_fig

    # =========================================================================
    # Private Methods: Utilities
    # =========================================================================

    def _format_title(
        self,
        read_id: str,
        normalization: NormalizationMethod,
        downsample: int,
    ) -> str:
        """Format plot title"""
        return self._build_title(f"Single Read: {read_id}", normalization, downsample)
