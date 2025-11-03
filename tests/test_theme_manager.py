"""
Tests for ThemeManager class
"""

import pytest
from bokeh.models.plots import Plot
from bokeh.plotting import figure

from squiggy.constants import Theme
from squiggy.rendering import ThemeManager


class TestThemeManagerInitialization:
    """Tests for ThemeManager initialization"""

    def test_init_with_light_theme(self):
        """Test initialization with LIGHT theme"""
        manager = ThemeManager(Theme.LIGHT)

        assert manager.theme == Theme.LIGHT
        assert manager.colors is not None
        assert manager.base_colors is not None

    def test_init_with_dark_theme(self):
        """Test initialization with DARK theme"""
        manager = ThemeManager(Theme.DARK)

        assert manager.theme == Theme.DARK
        assert manager.colors is not None
        assert manager.base_colors is not None

    def test_light_theme_uses_light_colors(self):
        """Test that LIGHT theme uses LIGHT_THEME colors"""
        from squiggy.constants import BASE_COLORS, LIGHT_THEME

        manager = ThemeManager(Theme.LIGHT)

        assert manager.colors == LIGHT_THEME
        assert manager.base_colors == BASE_COLORS

    def test_dark_theme_uses_dark_colors(self):
        """Test that DARK theme uses DARK_THEME colors"""
        from squiggy.constants import BASE_COLORS_DARK, DARK_THEME

        manager = ThemeManager(Theme.DARK)

        assert manager.colors == DARK_THEME
        assert manager.base_colors == BASE_COLORS_DARK


class TestThemeManagerColorGetters:
    """Tests for color getter methods"""

    def test_get_signal_color_light(self):
        """Test get_signal_color returns correct color for LIGHT theme"""
        manager = ThemeManager(Theme.LIGHT)

        color = manager.get_signal_color()

        assert isinstance(color, str)
        assert color.startswith("#")
        assert color == manager.colors["signal_line"]

    def test_get_signal_color_dark(self):
        """Test get_signal_color returns correct color for DARK theme"""
        manager = ThemeManager(Theme.DARK)

        color = manager.get_signal_color()

        assert isinstance(color, str)
        assert color.startswith("#")
        assert color == manager.colors["signal_line"]

    def test_get_base_colors_light(self):
        """Test get_base_colors returns correct dict for LIGHT theme"""
        manager = ThemeManager(Theme.LIGHT)

        colors = manager.get_base_colors()

        assert isinstance(colors, dict)
        assert "A" in colors
        assert "C" in colors
        assert "G" in colors
        assert "T" in colors
        assert "U" in colors
        assert "N" in colors
        assert all(isinstance(v, str) for v in colors.values())

    def test_get_base_colors_dark(self):
        """Test get_base_colors returns correct dict for DARK theme"""
        manager = ThemeManager(Theme.DARK)

        colors = manager.get_base_colors()

        assert isinstance(colors, dict)
        assert "A" in colors
        assert "C" in colors
        assert "G" in colors
        assert "T" in colors
        assert "U" in colors
        assert "N" in colors
        assert all(isinstance(v, str) for v in colors.values())

    def test_get_color_by_key(self):
        """Test get_color returns correct color for given key"""
        manager = ThemeManager(Theme.LIGHT)

        plot_bg = manager.get_color("plot_bg")
        grid_line = manager.get_color("grid_line")

        assert isinstance(plot_bg, str)
        assert isinstance(grid_line, str)
        assert plot_bg == manager.colors["plot_bg"]
        assert grid_line == manager.colors["grid_line"]

    def test_get_color_raises_on_invalid_key(self):
        """Test get_color raises KeyError for invalid key"""
        manager = ThemeManager(Theme.LIGHT)

        with pytest.raises(KeyError):
            manager.get_color("nonexistent_color_key")


class TestThemeManagerFigureCreation:
    """Tests for create_figure method"""

    def test_create_figure_returns_figure(self):
        """Test that create_figure returns a Bokeh Figure"""
        manager = ThemeManager(Theme.LIGHT)

        fig = manager.create_figure(
            title="Test Plot", x_label="X Axis", y_label="Y Axis"
        )

        assert isinstance(fig, Plot)

    def test_create_figure_applies_labels(self):
        """Test that create_figure applies title and axis labels"""
        manager = ThemeManager(Theme.LIGHT)

        fig = manager.create_figure(
            title="Test Plot", x_label="X Axis", y_label="Y Axis"
        )

        assert fig.title.text == "Test Plot"
        assert fig.xaxis.axis_label == "X Axis"
        assert fig.yaxis.axis_label == "Y Axis"

    def test_create_figure_applies_theme_colors(self):
        """Test that create_figure applies theme colors"""
        manager = ThemeManager(Theme.DARK)

        fig = manager.create_figure(
            title="Test Plot", x_label="X Axis", y_label="Y Axis"
        )

        # Check background colors
        assert fig.background_fill_color == manager.colors["plot_bg"]
        assert fig.border_fill_color == manager.colors["plot_border"]

        # Check title color
        assert fig.title.text_color == manager.colors["title_text"]

    def test_create_figure_with_custom_dimensions(self):
        """Test create_figure with custom width and height"""
        manager = ThemeManager(Theme.LIGHT)

        fig = manager.create_figure(
            title="Test", x_label="X", y_label="Y", width=1000, height=600
        )

        assert fig.width == 1000
        assert fig.height == 600

    def test_create_figure_without_width_uses_stretch(self):
        """Test create_figure without width uses stretch_width"""
        manager = ThemeManager(Theme.LIGHT)

        fig = manager.create_figure(title="Test", x_label="X", y_label="Y", height=400)

        assert fig.sizing_mode == "stretch_width"

    def test_create_figure_with_custom_tools(self):
        """Test create_figure with custom tools string"""
        manager = ThemeManager(Theme.LIGHT)

        fig = manager.create_figure(
            title="Test", x_label="X", y_label="Y", tools="pan,reset"
        )

        # Check that tools were set (exact tool list may vary)
        assert len(fig.tools) > 0


class TestThemeManagerApplyToFigure:
    """Tests for apply_to_figure method"""

    def test_apply_to_figure_styles_existing_figure(self):
        """Test that apply_to_figure styles an existing figure"""
        manager = ThemeManager(Theme.DARK)

        # Create unstyled figure
        fig = figure(width=800, height=400, title="Test")

        # Apply theme
        manager.apply_to_figure(fig)

        # Check that theme was applied
        assert fig.title.text_color == manager.colors["title_text"]
        assert fig.xaxis.axis_label_text_color == manager.colors["axis_text"]
        assert fig.yaxis.axis_label_text_color == manager.colors["axis_text"]

    def test_apply_to_figure_styles_axes(self):
        """Test that apply_to_figure styles axis colors"""
        manager = ThemeManager(Theme.LIGHT)

        fig = figure(width=800, height=400)
        manager.apply_to_figure(fig)

        # X-axis
        assert fig.xaxis.axis_line_color == manager.colors["axis_line"]
        assert fig.xaxis.major_tick_line_color == manager.colors["axis_line"]
        assert fig.xaxis.minor_tick_line_color == manager.colors["axis_line"]

        # Y-axis
        assert fig.yaxis.axis_line_color == manager.colors["axis_line"]
        assert fig.yaxis.major_tick_line_color == manager.colors["axis_line"]
        assert fig.yaxis.minor_tick_line_color == manager.colors["axis_line"]

    def test_apply_to_figure_styles_grid(self):
        """Test that apply_to_figure styles grid lines"""
        manager = ThemeManager(Theme.DARK)

        fig = figure(width=800, height=400)
        manager.apply_to_figure(fig)

        # X-grid should be off
        assert fig.xgrid.grid_line_color is None

        # Y-grid should use theme color
        assert fig.ygrid.grid_line_color == manager.colors["grid_line"]

    def test_apply_to_figure_is_idempotent(self):
        """Test that applying theme multiple times is safe"""
        manager = ThemeManager(Theme.LIGHT)

        fig = figure(width=800, height=400)

        # Apply twice
        manager.apply_to_figure(fig)
        manager.apply_to_figure(fig)

        # Should still have correct colors
        assert fig.title.text_color == manager.colors["title_text"]


class TestThemeManagerIntegration:
    """Integration tests for ThemeManager"""

    def test_light_and_dark_themes_produce_different_colors(self):
        """Test that LIGHT and DARK themes use different colors"""
        light = ThemeManager(Theme.LIGHT)
        dark = ThemeManager(Theme.DARK)

        # Signal colors should be different
        assert light.get_signal_color() != dark.get_signal_color()

        # Background colors should be different
        assert light.get_color("plot_bg") != dark.get_color("plot_bg")

    def test_create_figure_and_apply_produce_same_result(self):
        """Test that create_figure has same styling as apply_to_figure"""
        manager = ThemeManager(Theme.DARK)

        # Method 1: create_figure
        fig1 = manager.create_figure(
            title="Test", x_label="X", y_label="Y", width=800, height=400
        )

        # Method 2: manual creation + apply
        fig2 = figure(
            width=800, height=400, title="Test", x_axis_label="X", y_axis_label="Y"
        )
        manager.apply_to_figure(fig2)

        # Both should have same theme colors
        assert fig1.title.text_color == fig2.title.text_color
        assert fig1.xaxis.axis_label_text_color == fig2.xaxis.axis_label_text_color
        assert fig1.yaxis.axis_label_text_color == fig2.yaxis.axis_label_text_color

    def test_theme_manager_reusable(self):
        """Test that same ThemeManager can style multiple figures"""
        manager = ThemeManager(Theme.LIGHT)

        fig1 = manager.create_figure("Plot 1", "X1", "Y1")
        fig2 = manager.create_figure("Plot 2", "X2", "Y2")

        # Both should have same theme
        assert fig1.background_fill_color == fig2.background_fill_color
        assert fig1.title.text_color == fig2.title.text_color


class TestThemeManagerDocumentation:
    """Tests for ThemeManager documentation"""

    def test_class_has_docstring(self):
        """Test that ThemeManager has comprehensive docstring"""
        assert ThemeManager.__doc__ is not None
        assert len(ThemeManager.__doc__) > 100

    def test_create_figure_has_docstring(self):
        """Test that create_figure has detailed docstring"""
        assert ThemeManager.create_figure.__doc__ is not None
        assert "Args:" in ThemeManager.create_figure.__doc__
        assert "Returns:" in ThemeManager.create_figure.__doc__

    def test_apply_to_figure_has_docstring(self):
        """Test that apply_to_figure has detailed docstring"""
        assert ThemeManager.apply_to_figure.__doc__ is not None
        assert "Args:" in ThemeManager.apply_to_figure.__doc__
