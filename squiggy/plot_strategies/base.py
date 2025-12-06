"""
Base classes for plot strategy pattern

This module defines the abstract base class for all plot type implementations.
Each plot mode (SINGLE, EVENTALIGN, AGGREGATE, etc.) implements this interface.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..constants import NormalizationMethod, Theme
from ..normalization import normalize_signal


class PlotStrategy(ABC):
    """
    Abstract base class for all plot type strategies

    Each plot mode (SINGLE, EVENTALIGN, OVERLAY, STACKED, AGGREGATE, COMPARISON)
    implements this interface to provide specialized plotting behavior.

    The strategy pattern allows easy addition of new plot types without modifying
    existing code (Open/Closed Principle).

    Attributes:
        theme: Theme enum (LIGHT or DARK) for plot styling

    Examples:
        >>> from squiggy.plot_strategies.single_read import SingleReadPlotStrategy
        >>> from squiggy.constants import Theme
        >>>
        >>> strategy = SingleReadPlotStrategy(Theme.LIGHT)
        >>> data = {
        ...     'signal': read_obj.signal,
        ...     'read_id': 'read_001',
        ...     'sample_rate': 4000,
        ... }
        >>> options = {
        ...     'normalization': NormalizationMethod.ZNORM,
        ...     'downsample': 1,
        ... }
        >>> html, figure = strategy.create_plot(data, options)
    """

    def __init__(self, theme: Theme):
        """
        Initialize plot strategy with theme

        Args:
            theme: Theme enum (LIGHT or DARK)
        """
        self.theme = theme

    @abstractmethod
    def create_plot(
        self, data: dict[str, Any], options: dict[str, Any]
    ) -> tuple[str, Any]:
        """
        Generate Bokeh plot HTML and figure

        This is the main method that strategies must implement. It takes prepared
        data and options, generates a Bokeh visualization, and returns both HTML
        and the figure object.

        Args:
            data: Plot data dictionary containing:
                - Required keys depend on plot type (validated by validate_data)
                - Common keys: signal, read_id, sample_rate
                - Optional keys: sequence, seq_to_sig_map, modifications, etc.

            options: Plot options dictionary containing:
                - normalization: NormalizationMethod enum
                - downsample: Downsampling factor (int)
                - show_dwell_time: Whether to color by dwell time (bool)
                - show_labels: Whether to show base labels (bool)
                - scale_dwell_time: Whether to scale x-axis by dwell time (bool)
                - min_mod_probability: Minimum probability for mods (float)
                - enabled_mod_types: List of modification types to show
                - show_signal_points: Whether to show individual points (bool)
                - Other plot-specific options

        Returns:
            Tuple of (html_string, bokeh_figure_or_layout)
                - html_string: Complete HTML document with embedded Bokeh plot
                - figure: Bokeh Figure, Column, Row, or GridPlot object

        Raises:
            ValueError: If required data is missing (checked by validate_data)

        Examples:
            >>> data = {'signal': signal_array, 'read_id': 'read_001', 'sample_rate': 4000}
            >>> options = {'normalization': NormalizationMethod.ZNORM, 'downsample': 1}
            >>> html, fig = strategy.create_plot(data, options)
            >>> # html can be displayed in webview or saved to file
            >>> # fig can be further customized if needed
        """
        pass

    @abstractmethod
    def validate_data(self, data: dict[str, Any]) -> None:
        """
        Validate that required data is present for this plot type

        Each strategy must implement this to check for required keys in the
        data dictionary. Should raise ValueError with descriptive message if
        validation fails.

        Args:
            data: Plot data dictionary to validate

        Raises:
            ValueError: If required data is missing, with descriptive message
                indicating which keys are required

        Examples:
            >>> def validate_data(self, data):
            ...     required = ['signal', 'read_id', 'sample_rate']
            ...     missing = [k for k in required if k not in data]
            ...     if missing:
            ...         raise ValueError(f"Missing required data: {missing}")
        """
        pass

    def _figure_to_html(self, figure: Any) -> str:
        """
        Convert Bokeh figure to standalone HTML

        Helper method to convert a Bokeh figure, layout, or gridplot into
        a complete HTML document with embedded resources (JavaScript, CSS).

        This uses Bokeh's CDN resources for smaller file sizes.

        Args:
            figure: Bokeh Figure, Column, Row, or GridPlot object

        Returns:
            Complete HTML document as string

        Examples:
            >>> fig = figure(width=800, height=400)
            >>> fig.line([1, 2, 3], [1, 4, 9])
            >>> html = self._figure_to_html(fig)
            >>> # html contains <html>, <head>, <body> with embedded plot
        """
        from bokeh.embed import file_html
        from bokeh.resources import CDN

        return file_html(figure, CDN, "Squiggy Plot")

    def _process_signal(
        self,
        signal: np.ndarray,
        normalization: NormalizationMethod = NormalizationMethod.NONE,
        downsample: int = None,
        seq_to_sig_map: list[int] | None = None,
    ) -> tuple[np.ndarray, list[int] | None]:
        """
        Process signal: normalize and optionally downsample

        Applies normalization and downsampling to signal data. Optionally
        adjusts sequence-to-signal mapping indices when downsampling.

        When downsample is None, an adaptive downsampling factor is calculated
        to ensure the plot has at most MAX_PLOT_POINTS (50,000) points while
        respecting the minimum DEFAULT_DOWNSAMPLE (5). This provides automatic
        performance optimization for large signals without user intervention.

        Args:
            signal: Raw signal array
            normalization: Normalization method to apply
            downsample: Downsampling factor (1 = no downsampling, None = adaptive)
            seq_to_sig_map: Optional sequence-to-signal index mapping

        Returns:
            Tuple of (processed_signal, adjusted_seq_to_sig_map)
            - If seq_to_sig_map was None, returns (signal, None)
            - If seq_to_sig_map provided and downsample > 1, indices are adjusted

        Examples:
            >>> # Small signal: uses DEFAULT_DOWNSAMPLE (5)
            >>> signal = np.array([1] * 100_000)  # 100K points
            >>> processed, _ = self._process_signal(signal, downsample=None)
            >>> len(processed)  # 100K / 5 = 20K points
            20000

            >>> # Large signal: adaptive downsampling
            >>> signal = np.array([1] * 1_000_000)  # 1M points
            >>> processed, _ = self._process_signal(signal, downsample=None)
            >>> len(processed)  # 1M / 20 = 50K points (auto-adjusted)
            50000

            >>> # User override: explicit downsample
            >>> processed, _ = self._process_signal(signal, downsample=2)
            >>> len(processed)  # 1M / 2 = 500K (user choice preserved)
            500000
        """
        from ..constants import DEFAULT_DOWNSAMPLE, MAX_PLOT_POINTS

        if downsample is None:
            # Adaptive downsampling: ensure we don't exceed MAX_PLOT_POINTS
            # while respecting minimum DEFAULT_DOWNSAMPLE
            adaptive_downsample = max(1, len(signal) // MAX_PLOT_POINTS)
            downsample = max(DEFAULT_DOWNSAMPLE, adaptive_downsample)

        # Normalize
        if normalization != NormalizationMethod.NONE:
            signal = normalize_signal(signal, method=normalization)

        # Downsample
        if downsample > 1:
            signal = signal[::downsample]
            if seq_to_sig_map is not None:
                seq_to_sig_map = [idx // downsample for idx in seq_to_sig_map]

        return signal, seq_to_sig_map

    def _calculate_adapter_trim_indices(
        self,
        aligned_read: Any,
        signal_length: int,
    ) -> tuple[int, int]:
        """
        Calculate signal indices for adapter trimming based on soft-clip info.

        Uses the soft-clip information from the AlignedRead to determine which
        portions of the signal correspond to adapter sequences that should be
        removed.

        The soft-clipped bases map to signal indices through the base annotations:
        - query_start_offset: Number of bases soft-clipped at 5' end (start)
        - query_end_offset: Number of bases soft-clipped at 3' end (end)

        Args:
            aligned_read: AlignedRead object with base annotations and soft-clip info
            signal_length: Total length of the signal array

        Returns:
            Tuple of (trim_start, trim_end) signal indices:
            - trim_start: First signal index to include (0 if no 5' clipping)
            - trim_end: Last signal index to include (signal_length if no 3' clipping)

        Examples:
            >>> # Read with 50bp soft-clipped at start, 30bp at end
            >>> aligned_read.query_start_offset = 50
            >>> aligned_read.query_end_offset = 30
            >>> start, end = self._calculate_adapter_trim_indices(aligned_read, 100000)
            >>> trimmed_signal = signal[start:end]
        """
        from ..alignment import AlignedRead

        # Default to full signal if no alignment or no soft-clips
        if aligned_read is None:
            return 0, signal_length

        # Get soft-clip offsets
        query_start_offset = getattr(aligned_read, "query_start_offset", 0)
        query_end_offset = getattr(aligned_read, "query_end_offset", 0)

        # If no soft-clipping, return full signal range
        if query_start_offset == 0 and query_end_offset == 0:
            return 0, signal_length

        # Get base annotations to map base positions to signal positions
        bases = getattr(aligned_read, "bases", [])
        if not bases:
            # No base annotations, can't determine signal positions
            return 0, signal_length

        # Calculate trim_start: signal index after the soft-clipped 5' bases
        trim_start = 0
        if query_start_offset > 0 and len(bases) > query_start_offset:
            # Signal start is at the first aligned base (after soft-clipped bases)
            trim_start = bases[query_start_offset].signal_start

        # Calculate trim_end: signal index before the soft-clipped 3' bases
        trim_end = signal_length
        if query_end_offset > 0:
            # Find the last aligned base (before soft-clipped bases at end)
            last_aligned_idx = len(bases) - query_end_offset - 1
            if last_aligned_idx >= 0 and last_aligned_idx < len(bases):
                # End at the signal_end of the last aligned base
                trim_end = bases[last_aligned_idx].signal_end

        # Validate bounds
        trim_start = max(0, min(trim_start, signal_length))
        trim_end = max(trim_start, min(trim_end, signal_length))

        return trim_start, trim_end

    def _apply_adapter_trimming(
        self,
        signal: np.ndarray,
        aligned_read: Any,
        seq_to_sig_map: list[int] | None = None,
    ) -> tuple[np.ndarray, list[int] | None, int]:
        """
        Trim adapter regions from signal and adjust sequence-to-signal mapping.

        This method removes the signal portions corresponding to soft-clipped
        (adapter) bases from the read. It uses the AlignedRead's soft-clip
        information to determine trim boundaries.

        Args:
            signal: Raw signal array
            aligned_read: AlignedRead object with soft-clip information
            seq_to_sig_map: Optional sequence-to-signal index mapping

        Returns:
            Tuple of (trimmed_signal, adjusted_seq_to_sig_map, trim_start):
            - trimmed_signal: Signal with adapter regions removed
            - adjusted_seq_to_sig_map: Mapping adjusted for the new signal indices
            - trim_start: The starting index that was trimmed (for coordinate adjustment)

        Examples:
            >>> # Trim adapters from signal
            >>> trimmed, mapping, offset = self._apply_adapter_trimming(
            ...     signal, aligned_read, seq_to_sig_map
            ... )
        """
        if aligned_read is None:
            return signal, seq_to_sig_map, 0

        # Calculate trim indices
        trim_start, trim_end = self._calculate_adapter_trim_indices(
            aligned_read, len(signal)
        )

        # If nothing to trim, return original
        if trim_start == 0 and trim_end == len(signal):
            return signal, seq_to_sig_map, 0

        # Trim the signal
        trimmed_signal = signal[trim_start:trim_end]

        # Adjust sequence-to-signal mapping if provided
        adjusted_mapping = None
        if seq_to_sig_map is not None:
            # Shift indices and filter out any that are now out of bounds
            adjusted_mapping = []
            for idx in seq_to_sig_map:
                new_idx = idx - trim_start
                if 0 <= new_idx < len(trimmed_signal):
                    adjusted_mapping.append(new_idx)
                else:
                    # Keep the mapping but clamp to valid range
                    adjusted_mapping.append(max(0, min(new_idx, len(trimmed_signal) - 1)))

        return trimmed_signal, adjusted_mapping, trim_start

    def _validate_read_tuples(self, reads: list) -> None:
        """
        Validate list of read tuples

        Checks that reads is a non-empty list where each element is a
        3-tuple of (read_id: str, signal: np.ndarray, sample_rate: float).

        Args:
            reads: List of (read_id, signal, sample_rate) tuples to validate

        Raises:
            ValueError: If validation fails with descriptive message

        Examples:
            >>> reads = [
            ...     ("read_001", np.array([1, 2, 3]), 4000),
            ...     ("read_002", np.array([4, 5, 6]), 4000),
            ... ]
            >>> self._validate_read_tuples(reads)  # Passes validation
        """
        if not isinstance(reads, list):
            raise ValueError(
                "reads must be a list of (read_id, signal, sample_rate) tuples"
            )

        if len(reads) == 0:
            raise ValueError("reads list cannot be empty")

        # Validate each read tuple
        for idx, read_tuple in enumerate(reads):
            if not isinstance(read_tuple, tuple) or len(read_tuple) != 3:
                raise ValueError(
                    f"Read {idx} must be a tuple of (read_id, signal, sample_rate)"
                )

            read_id, signal, sample_rate = read_tuple
            if not isinstance(read_id, str):
                raise ValueError(f"Read {idx}: read_id must be a string")
            if not isinstance(signal, np.ndarray):
                raise ValueError(f"Read {idx}: signal must be a numpy array")
            if not isinstance(sample_rate, (int, float)):
                raise ValueError(f"Read {idx}: sample_rate must be a number")

    def _build_title(
        self,
        base_title: str,
        normalization: NormalizationMethod,
        downsample: int,
    ) -> str:
        """
        Build formatted plot title with normalization and downsampling info

        Creates a title string by joining base title with optional normalization
        and downsampling information using " | " separator.

        Args:
            base_title: Base title string (e.g., "Single Read: read_001")
            normalization: Normalization method applied
            downsample: Downsampling factor applied

        Returns:
            Formatted title string

        Examples:
            >>> title = self._build_title(
            ...     "Single Read: read_001",
            ...     NormalizationMethod.ZNORM,
            ...     5
            ... )
            >>> print(title)
            "Single Read: read_001 | znorm normalized | downsampled 5x"
        """
        parts = [base_title]

        if normalization != NormalizationMethod.NONE:
            parts.append(f"{normalization.value} normalized")

        if downsample > 1:
            parts.append(f"downsampled {downsample}x")

        return " | ".join(parts)

    def _build_html_title(self, mode_name: str, description: str) -> str:
        """
        Build HTML page title

        Creates title for HTML document in format "Squiggy {mode}: {description}".

        Args:
            mode_name: Plot mode name (e.g., "Single Read", "Overlay")
            description: Description string (e.g., "read_001", "5 reads")

        Returns:
            Formatted HTML title string

        Examples:
            >>> title = self._build_html_title("Overlay", "5 reads")
            >>> print(title)
            "Squiggy Overlay: 5 reads"
        """
        return f"Squiggy {mode_name}: {description}"
