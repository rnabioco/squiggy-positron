"""Plotting utilities for Squiggy"""


def _route_to_plots_pane(fig) -> None:
    """
    Route Bokeh figure to Positron Plots pane via bokeh.io.show()

    Positron intercepts bokeh.io.show() calls and routes them to Plots pane
    by inspecting the call stack for bokeh.io.showing.show function.

    This ensures plots appear in the Plots pane (with history and navigation)
    rather than the Viewer pane.

    Args:
        fig: Bokeh figure object
    """
    import os
    import sys

    # Skip if running in test environment (pytest)
    if "pytest" in sys.modules or os.getenv("PYTEST_CURRENT_TEST"):
        return

    try:
        from bokeh.io import show

        show(fig)  # Positron intercepts this and routes to Plots pane
    except Exception:
        # Silently fail if bokeh.io not available or not in Positron
        pass


def parse_plot_parameters(
    mode: str | None = None,
    normalization: str = "ZNORM",
    theme: str = "LIGHT",
):
    """
    Parse and validate plot parameters (mode, normalization, theme)

    This utility function eliminates duplicate parameter parsing code across
    plotting functions by centralizing the conversion from string parameters
    to enum values.

    Args:
        mode: Plot mode string (e.g., "SINGLE", "OVERLAY", "EVENTALIGN", "AGGREGATE").
              If None, only normalization and theme are parsed.
        normalization: Normalization method string (default: "ZNORM").
                       Valid values: "NONE", "ZNORM", "MEDIAN", "MAD"
        theme: Color theme string (default: "LIGHT").
               Valid values: "LIGHT", "DARK"

    Returns:
        Dictionary with parsed enum values:
        - "mode": PlotMode enum (if mode parameter provided, else None)
        - "normalization": NormalizationMethod enum
        - "theme": Theme enum

    Raises:
        KeyError: If invalid mode, normalization, or theme string provided

    Examples:
        >>> params = parse_plot_parameters(mode="SINGLE", normalization="ZNORM", theme="LIGHT")
        >>> params["mode"]
        <PlotMode.SINGLE: 'SINGLE'>

        >>> params = parse_plot_parameters(normalization="MEDIAN", theme="DARK")
        >>> params["normalization"]
        <NormalizationMethod.MEDIAN: 'MEDIAN'>
    """
    from ..constants import NormalizationMethod, PlotMode, Theme

    result = {
        "normalization": NormalizationMethod[normalization.upper()],
        "theme": Theme[theme.upper()],
    }

    if mode is not None:
        result["mode"] = PlotMode[mode.upper()]

    return result
