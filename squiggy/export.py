"""
Export functions for Squiggy plots

This module provides functions to export Bokeh plots to static image formats
(PNG, SVG) with dimension control.
"""

import warnings
from typing import Any

from bokeh.io import export_png, export_svgs
from bokeh.models import Plot

# Global storage for the last generated plot figure
# This allows the extension to export the current plot
_last_plot_figure: Any | None = None


def export_plot_to_png(
    plot: Plot,
    output_path: str,
    width: int | None = None,
    height: int | None = None,
    dpi: int = 96,
) -> None:
    """
    Export a Bokeh plot to PNG format

    Args:
        plot: Bokeh Plot object to export
        output_path: Path where PNG file will be saved
        width: Output width in pixels (if None, uses plot's width)
        height: Output height in pixels (if None, uses plot's height)
        dpi: DPI resolution for PNG export (default: 96)

    Raises:
        RuntimeError: If selenium/webdriver is not available
        ValueError: If plot is None or invalid dimensions

    Examples:
        >>> from squiggy import plot_read
        >>> from squiggy.export import export_plot_to_png
        >>> html, fig = plot_read('read_001', return_fig=True)
        >>> export_plot_to_png(fig, 'plot.png', width=1200, height=800, dpi=150)
    """
    if plot is None:
        raise ValueError("Plot object cannot be None")

    if width is not None and width <= 0:
        raise ValueError(f"Width must be positive, got {width}")
    if height is not None and height <= 0:
        raise ValueError(f"Height must be positive, got {height}")
    if dpi <= 0:
        raise ValueError(f"DPI must be positive, got {dpi}")

    # Update plot dimensions if specified
    original_width = plot.width
    original_height = plot.height

    try:
        if width is not None:
            plot.width = width
        if height is not None:
            plot.height = height

        # Export to PNG using Bokeh's export_png
        # This requires selenium and a webdriver (geckodriver for Firefox)
        # Selenium Manager (>=4.6) auto-downloads geckodriver
        with warnings.catch_warnings():
            # Suppress Bokeh warnings about saving to BytesIO
            warnings.filterwarnings("ignore", message="Saving.*in image")
            export_png(plot, filename=output_path)

    except Exception as e:
        if "webdriver" in str(e).lower() or "selenium" in str(e).lower():
            raise RuntimeError(
                "PNG export requires selenium and a webdriver. "
                "Install with: pip install squiggy-positron[export]\n"
                "This will install selenium (>=4.6) with automatic webdriver management."
            ) from e
        raise

    finally:
        # Restore original dimensions
        plot.width = original_width
        plot.height = original_height


def export_plot_to_svg(
    plot: Plot,
    output_path: str,
    width: int | None = None,
    height: int | None = None,
) -> None:
    """
    Export a Bokeh plot to SVG format

    Args:
        plot: Bokeh Plot object to export
        output_path: Path where SVG file will be saved
        width: Output width in pixels (if None, uses plot's width)
        height: Output height in pixels (if None, uses plot's height)

    Raises:
        ValueError: If plot is None or invalid dimensions

    Examples:
        >>> from squiggy import plot_read
        >>> from squiggy.export import export_plot_to_svg
        >>> html, fig = plot_read('read_001', return_fig=True)
        >>> export_plot_to_svg(fig, 'plot.svg', width=1200, height=800)
    """
    if plot is None:
        raise ValueError("Plot object cannot be None")

    if width is not None and width <= 0:
        raise ValueError(f"Width must be positive, got {width}")
    if height is not None and height <= 0:
        raise ValueError(f"Height must be positive, got {height}")

    # Update plot dimensions if specified
    original_width = plot.width
    original_height = plot.height

    try:
        if width is not None:
            plot.width = width
        if height is not None:
            plot.height = height

        # Export to SVG using Bokeh's export_svgs
        # Returns a generator of SVG strings - we take the first one
        plot.output_backend = "svg"

        export_svgs(plot, filename=output_path)

    finally:
        # Restore original dimensions and backend
        plot.width = original_width
        plot.height = original_height
        plot.output_backend = "canvas"


def export_plot_from_html(
    html_content: str,
    output_path: str,
    format: str = "png",
    width: int | None = None,
    height: int | None = None,
    dpi: int = 96,
) -> None:
    """
    Export a Bokeh plot from HTML content to static image format

    This is a convenience function that extracts the plot from HTML and exports it.
    Note: This function currently requires the plot object, not HTML.
    For HTML-based export, the plot object must be passed separately.

    Args:
        html_content: Bokeh HTML string (currently unused - requires plot object)
        output_path: Path where image file will be saved
        format: Output format ('png' or 'svg')
        width: Output width in pixels (if None, uses plot's width)
        height: Output height in pixels (if None, uses plot's height)
        dpi: DPI resolution for PNG export (default: 96)

    Raises:
        NotImplementedError: This function requires the plot object directly
    """
    raise NotImplementedError(
        "HTML-based export not yet implemented. "
        "Use export_plot_to_png() or export_plot_to_svg() with the plot object directly."
    )


def get_plot_dimensions(plot: Plot) -> tuple[int, int]:
    """
    Get the current dimensions of a Bokeh plot

    Args:
        plot: Bokeh Plot object

    Returns:
        Tuple of (width, height) in pixels
    """
    if plot is None:
        raise ValueError("Plot object cannot be None")

    return (plot.width or 800, plot.height or 600)


def set_current_plot(figure: Any) -> None:
    """
    Store the current plot figure for export

    This is called automatically by plotting functions to enable export
    functionality in the Positron extension.

    Args:
        figure: Bokeh Figure, Column, Row, or GridPlot object
    """
    global _last_plot_figure
    _last_plot_figure = figure


def get_current_plot() -> Any:
    """
    Get the currently stored plot figure

    Returns:
        The last plot figure stored via set_current_plot(), or None if no plot available
    """
    return _last_plot_figure


def export_current_plot(
    output_path: str,
    format: str = "png",
    width: int | None = None,
    height: int | None = None,
    dpi: int = 96,
) -> None:
    """
    Export the currently displayed plot to a static image file

    This function exports the last plot that was generated to PNG or SVG format.
    Useful for the Positron extension to export the current plot.

    Args:
        output_path: Path where image file will be saved
        format: Output format ('png' or 'svg')
        width: Output width in pixels (if None, uses plot's width)
        height: Output height in pixels (if None, uses plot's height)
        dpi: DPI resolution for PNG export (default: 96, ignored for SVG)

    Raises:
        ValueError: If no plot is available or invalid parameters
        RuntimeError: If selenium/webdriver is not available (PNG only)

    Examples:
        >>> import squiggy
        >>> squiggy.load_pod5('data.pod5')
        >>> squiggy.plot_read('read_001')  # This stores the plot
        >>> from squiggy.export import export_current_plot
        >>> export_current_plot('plot.png', format='png', width=1200, height=800, dpi=150)
        >>> export_current_plot('plot.svg', format='svg', width=1200, height=800)
    """
    if _last_plot_figure is None:
        raise ValueError(
            "No plot available for export. Generate a plot first using plot_read() or plot_reads()."
        )

    format = format.lower()
    if format not in ("png", "svg"):
        raise ValueError(f"Format must be 'png' or 'svg', got '{format}'")

    if format == "png":
        export_plot_to_png(_last_plot_figure, output_path, width, height, dpi)
    else:  # svg
        export_plot_to_svg(_last_plot_figure, output_path, width, height)
