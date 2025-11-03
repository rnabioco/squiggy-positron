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

    Example:
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

        Example:
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

        Example:
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

        Example:
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
        downsample: int = 1,
        seq_to_sig_map: list[int] | None = None,
    ) -> tuple[np.ndarray, list[int] | None]:
        """
        Process signal: normalize and optionally downsample

        Applies normalization and downsampling to signal data. Optionally
        adjusts sequence-to-signal mapping indices when downsampling.

        Args:
            signal: Raw signal array
            normalization: Normalization method to apply
            downsample: Downsampling factor (1 = no downsampling)
            seq_to_sig_map: Optional sequence-to-signal index mapping

        Returns:
            Tuple of (processed_signal, adjusted_seq_to_sig_map)
            - If seq_to_sig_map was None, returns (signal, None)
            - If seq_to_sig_map provided and downsample > 1, indices are adjusted

        Example:
            >>> signal = np.array([1, 2, 3, 4, 5, 6])
            >>> processed, _ = self._process_signal(
            ...     signal,
            ...     normalization=NormalizationMethod.ZNORM,
            ...     downsample=2
            ... )
            >>> # Returns z-normalized signal downsampled by factor of 2
        """
        # Normalize
        if normalization != NormalizationMethod.NONE:
            signal = normalize_signal(signal, method=normalization)

        # Downsample
        if downsample > 1:
            signal = signal[::downsample]
            if seq_to_sig_map is not None:
                seq_to_sig_map = [idx // downsample for idx in seq_to_sig_map]

        return signal, seq_to_sig_map
