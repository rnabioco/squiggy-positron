"""
Theme management for Bokeh plots

This module centralizes all theme-related functionality, providing consistent
styling across all plot types.
"""

from bokeh.models import WheelZoomTool
from bokeh.plotting import figure

from ..constants import (
    BASE_COLORS,
    BASE_COLORS_DARK,
    DARK_THEME,
    LIGHT_THEME,
    Theme,
)


class ThemeManager:
    """
    Centralized theme management for squiggy plots

    This class handles all theme-related styling for Bokeh figures, providing
    consistent colors, styling, and figure creation across all plot types.

    Attributes:
        theme: Theme enum (LIGHT or DARK)
        colors: Dictionary of theme colors for plots
        base_colors: Dictionary of base nucleotide colors

    Examples:
        >>> from squiggy.theme_manager import ThemeManager
        >>> from squiggy.constants import Theme
        >>>
        >>> manager = ThemeManager(Theme.DARK)
        >>> fig = manager.create_figure(
        ...     title="Signal Plot",
        ...     x_label="Time (ms)",
        ...     y_label="Signal (pA)"
        ... )
        >>> # Figure is already styled with dark theme
    """

    def __init__(self, theme: Theme):
        """
        Initialize theme manager

        Args:
            theme: Theme enum (LIGHT or DARK)
        """
        self.theme = theme
        self.colors = DARK_THEME if theme == Theme.DARK else LIGHT_THEME
        self.base_colors = BASE_COLORS_DARK if theme == Theme.DARK else BASE_COLORS

    def get_signal_color(self) -> str:
        """
        Get signal line color for current theme

        Returns:
            Hex color string for signal line

        Examples:
            >>> manager = ThemeManager(Theme.LIGHT)
            >>> manager.get_signal_color()
            '#000000'
        """
        return self.colors["signal_line"]

    def get_base_colors(self) -> dict[str, str]:
        """
        Get nucleotide base colors for current theme

        Returns:
            Dictionary mapping base letter to hex color
            Keys: 'A', 'C', 'G', 'T', 'U', 'N'

        Examples:
            >>> manager = ThemeManager(Theme.LIGHT)
            >>> colors = manager.get_base_colors()
            >>> colors['A']  # Adenine color
            '#00b388'
        """
        return self.base_colors

    def create_figure(
        self,
        title: str,
        x_label: str,
        y_label: str,
        width: int | None = None,
        height: int = 400,
        tools: str = "xpan,xbox_zoom,box_zoom,wheel_zoom,reset,save",
        **kwargs,
    ):
        """
        Create a themed Bokeh figure with standard settings

        This method creates a Bokeh figure pre-configured with:
        - Theme-appropriate colors
        - Standard toolbar
        - X-only wheel zoom
        - Responsive sizing

        Args:
            title: Plot title
            x_label: X-axis label
            y_label: Y-axis label
            width: Figure width in pixels (None = stretch_width)
            height: Figure height in pixels (default: 400)
            tools: Bokeh tools string (default: standard set)
            **kwargs: Additional arguments passed to bokeh.plotting.figure()

        Returns:
            Themed Bokeh figure ready for plotting

        Examples:
            >>> manager = ThemeManager(Theme.DARK)
            >>> fig = manager.create_figure(
            ...     title="Signal Plot",
            ...     x_label="Time (ms)",
            ...     y_label="Signal (pA)",
            ...     height=500
            ... )
            >>> fig.line([1, 2, 3], [1, 4, 9])
        """
        # Create figure with theme colors
        fig_kwargs = {
            "title": title,
            "x_axis_label": x_label,
            "y_axis_label": y_label,
            "tools": tools,
            "height": height,
            "background_fill_color": self.colors["plot_bg"],
            "border_fill_color": self.colors["plot_border"],
        }

        # Only set active tools if they're in the tools string
        if "xbox_zoom" in tools:
            fig_kwargs["active_drag"] = "xbox_zoom"
        if "wheel_zoom" in tools:
            fig_kwargs["active_scroll"] = "wheel_zoom"

        # Set width or use responsive sizing
        if width is not None:
            fig_kwargs["width"] = width
        else:
            fig_kwargs["sizing_mode"] = "stretch_width"

        # Merge with any additional kwargs
        fig_kwargs.update(kwargs)

        # Create figure
        fig = figure(**fig_kwargs)

        # Apply theme styling
        self.apply_to_figure(fig)

        return fig

    def apply_to_figure(self, fig) -> None:
        """
        Apply theme styling to an existing Bokeh figure

        This method applies theme colors to all elements of a figure:
        title, axes, grid, etc.

        Args:
            fig: Bokeh figure to style

        Examples:
            >>> from bokeh.plotting import figure
            >>> fig = figure(width=800, height=400)
            >>> manager = ThemeManager(Theme.DARK)
            >>> manager.apply_to_figure(fig)
            >>> # Figure now has dark theme colors
        """
        # Apply theme to title
        fig.title.text_color = self.colors["title_text"]

        # Apply theme to x-axis
        fig.xaxis.axis_label_text_color = self.colors["axis_text"]
        fig.xaxis.major_label_text_color = self.colors["axis_text"]
        fig.xaxis.axis_line_color = self.colors["axis_line"]
        fig.xaxis.major_tick_line_color = self.colors["axis_line"]
        fig.xaxis.minor_tick_line_color = self.colors["axis_line"]

        # Apply theme to y-axis
        fig.yaxis.axis_label_text_color = self.colors["axis_text"]
        fig.yaxis.major_label_text_color = self.colors["axis_text"]
        fig.yaxis.axis_line_color = self.colors["axis_line"]
        fig.yaxis.major_tick_line_color = self.colors["axis_line"]
        fig.yaxis.minor_tick_line_color = self.colors["axis_line"]

        # Apply theme to grid
        fig.xgrid.grid_line_color = None  # Keep vertical grid lines off
        fig.ygrid.grid_line_color = self.colors["grid_line"]

        # Add x-only wheel zoom (if not already present)
        has_wheel_zoom = any(isinstance(tool, WheelZoomTool) for tool in fig.tools)
        if not has_wheel_zoom:
            wheel_zoom = WheelZoomTool(dimensions="width")
            fig.add_tools(wheel_zoom)
            fig.toolbar.active_scroll = wheel_zoom

    def configure_legend(self, fig) -> None:
        """
        Configure legend styling for compact, unobtrusive display

        Applies consistent legend styling across all plot types:
        - Horizontal orientation for space efficiency
        - Compact sizing with minimal padding
        - Semi-transparent background
        - Click-to-hide functionality

        Args:
            fig: Bokeh figure with legend to configure

        Examples:
            >>> manager = ThemeManager(Theme.LIGHT)
            >>> fig = manager.create_figure("Plot", "X", "Y")
            >>> fig.line([1, 2, 3], [1, 4, 9], legend_label="Data")
            >>> manager.configure_legend(fig)
        """
        fig.legend.click_policy = "hide"
        fig.legend.location = "bottom_right"
        fig.legend.orientation = "horizontal"
        fig.legend.background_fill_alpha = 0.65
        fig.legend.label_text_font_size = "8pt"
        fig.legend.glyph_width = 15
        fig.legend.glyph_height = 10
        fig.legend.padding = 2
        fig.legend.spacing = 2
        fig.legend.margin = 2

    def get_color(self, color_key: str) -> str:
        """
        Get a specific theme color by key

        Args:
            color_key: Key from LIGHT_THEME or DARK_THEME dict
                (e.g., 'plot_bg', 'grid_line', 'axis_text')

        Returns:
            Hex color string

        Raises:
            KeyError: If color_key not found in theme

        Examples:
            >>> manager = ThemeManager(Theme.LIGHT)
            >>> manager.get_color('plot_bg')
            '#ffffff'
        """
        return self.colors[color_key]

    def get_signal_band_color(self) -> str:
        """
        Get signal confidence band color for current theme

        Returns theme-appropriate color for signal confidence/error bands
        in aggregate plots.

        Returns:
            Hex color string

        Examples:
            >>> manager = ThemeManager(Theme.LIGHT)
            >>> manager.get_signal_band_color()
            '#56B4E9'  # Light blue for light theme
        """
        return self.colors["signal_band"]

    def get_quality_band_color(self) -> str:
        """
        Get quality band color for current theme

        Returns theme-appropriate color for quality score bands
        in aggregate plots.

        Returns:
            Hex color string

        Examples:
            >>> manager = ThemeManager(Theme.DARK)
            >>> manager.get_quality_band_color()
            '#FF8C00'  # Dark orange for dark theme
        """
        return self.colors["quality_band"]
