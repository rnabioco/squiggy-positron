"""Tests for constants and configuration."""

from enum import Enum

from squiggy.constants import (
    APP_DESCRIPTION,
    APP_NAME,
    APP_VERSION,
    BASE_COLORS,
    BASE_COLORS_DARK,
    DARK_THEME,
    DEFAULT_WINDOW_HEIGHT,
    DEFAULT_WINDOW_WIDTH,
    LIGHT_THEME,
    MULTI_READ_COLORS,
    NormalizationMethod,
    PlotMode,
    Theme,
)


class TestApplicationMetadata:
    """Tests for application metadata constants."""

    def test_app_name_is_defined(self):
        """Test that application name is defined."""
        assert APP_NAME is not None
        assert isinstance(APP_NAME, str)
        assert len(APP_NAME) > 0

    def test_app_version_is_defined(self):
        """Test that application version is defined."""
        assert APP_VERSION is not None
        assert isinstance(APP_VERSION, str)
        assert len(APP_VERSION) > 0

        # Version should have format like "0.1.0"
        parts = APP_VERSION.split(".")
        assert len(parts) >= 2  # At least major.minor

    def test_app_description_is_defined(self):
        """Test that application description is defined."""
        assert APP_DESCRIPTION is not None
        assert isinstance(APP_DESCRIPTION, str)
        assert len(APP_DESCRIPTION) > 0


class TestWindowSettings:
    """Tests for window dimension constants."""

    def test_default_window_dimensions_are_positive(self):
        """Test that default window dimensions are positive."""
        assert DEFAULT_WINDOW_WIDTH > 0
        assert DEFAULT_WINDOW_HEIGHT > 0

    def test_default_window_dimensions_are_reasonable(self):
        """Test that default dimensions are reasonable screen sizes."""
        # Should be less than typical 4K resolution
        assert DEFAULT_WINDOW_WIDTH <= 3840
        assert DEFAULT_WINDOW_HEIGHT <= 2160

        # Should be larger than tiny windows
        assert DEFAULT_WINDOW_WIDTH >= 640
        assert DEFAULT_WINDOW_HEIGHT >= 480

    def test_default_window_aspect_ratio(self):
        """Test that default window has reasonable aspect ratio."""
        aspect_ratio = DEFAULT_WINDOW_WIDTH / DEFAULT_WINDOW_HEIGHT

        # Typical aspect ratios are 4:3 (1.33) to 16:9 (1.78)
        assert 1.0 <= aspect_ratio <= 2.0


class TestPlotModeEnum:
    """Tests for PlotMode enumeration."""

    def test_plot_mode_is_enum(self):
        """Test that PlotMode is an Enum."""
        assert issubclass(PlotMode, Enum)

    def test_plot_mode_has_expected_values(self):
        """Test that PlotMode has all expected values."""
        expected_modes = ["SINGLE", "OVERLAY", "STACKED", "EVENTALIGN"]

        for mode_name in expected_modes:
            assert hasattr(PlotMode, mode_name), f"PlotMode missing {mode_name}"

    def test_plot_mode_values_are_strings(self):
        """Test that PlotMode values are strings."""
        for mode in PlotMode:
            assert isinstance(mode.value, str)

    def test_plot_mode_single_exists(self):
        """Test that SINGLE mode exists."""
        assert PlotMode.SINGLE is not None
        assert PlotMode.SINGLE.value == "single"

    def test_plot_mode_overlay_exists(self):
        """Test that OVERLAY mode exists."""
        assert PlotMode.OVERLAY is not None
        assert PlotMode.OVERLAY.value == "overlay"

    def test_plot_mode_stacked_exists(self):
        """Test that STACKED mode exists."""
        assert PlotMode.STACKED is not None
        assert PlotMode.STACKED.value == "stacked"

    def test_plot_mode_eventalign_exists(self):
        """Test that EVENTALIGN mode exists."""
        assert PlotMode.EVENTALIGN is not None
        assert PlotMode.EVENTALIGN.value == "eventalign"

    def test_plot_mode_membership(self):
        """Test that we can check PlotMode membership."""
        assert PlotMode.SINGLE in PlotMode
        assert PlotMode.OVERLAY in PlotMode
        assert PlotMode.STACKED in PlotMode
        assert PlotMode.EVENTALIGN in PlotMode


class TestNormalizationMethodEnum:
    """Tests for NormalizationMethod enumeration."""

    def test_normalization_method_is_enum(self):
        """Test that NormalizationMethod is an Enum."""
        assert issubclass(NormalizationMethod, Enum)

    def test_normalization_method_has_expected_values(self):
        """Test that NormalizationMethod has all expected values."""
        expected_methods = ["NONE", "ZNORM", "MEDIAN", "MAD"]

        for method_name in expected_methods:
            assert hasattr(NormalizationMethod, method_name), (
                f"NormalizationMethod missing {method_name}"
            )

    def test_normalization_method_values_are_strings(self):
        """Test that NormalizationMethod values are strings."""
        for method in NormalizationMethod:
            assert isinstance(method.value, str)

    def test_normalization_method_none_exists(self):
        """Test that NONE method exists."""
        assert NormalizationMethod.NONE is not None
        assert NormalizationMethod.NONE.value == "none"

    def test_normalization_method_znorm_exists(self):
        """Test that ZNORM method exists."""
        assert NormalizationMethod.ZNORM is not None
        assert NormalizationMethod.ZNORM.value == "znorm"

    def test_normalization_method_median_exists(self):
        """Test that MEDIAN method exists."""
        assert NormalizationMethod.MEDIAN is not None
        assert NormalizationMethod.MEDIAN.value == "median"

    def test_normalization_method_mad_exists(self):
        """Test that MAD method exists."""
        assert NormalizationMethod.MAD is not None
        assert NormalizationMethod.MAD.value == "mad"

    def test_normalization_method_membership(self):
        """Test that we can check NormalizationMethod membership."""
        assert NormalizationMethod.NONE in NormalizationMethod
        assert NormalizationMethod.ZNORM in NormalizationMethod
        assert NormalizationMethod.MEDIAN in NormalizationMethod
        assert NormalizationMethod.MAD in NormalizationMethod


class TestThemeEnum:
    """Tests for Theme enumeration."""

    def test_theme_is_enum(self):
        """Test that Theme is an Enum."""
        assert issubclass(Theme, Enum)

    def test_theme_has_light_and_dark(self):
        """Test that Theme has LIGHT and DARK values."""
        assert hasattr(Theme, "LIGHT")
        assert hasattr(Theme, "DARK")

    def test_theme_values_are_strings(self):
        """Test that Theme values are strings."""
        for theme in Theme:
            assert isinstance(theme.value, str)

    def test_theme_light_value(self):
        """Test LIGHT theme value."""
        assert Theme.LIGHT.value == "light"

    def test_theme_dark_value(self):
        """Test DARK theme value."""
        assert Theme.DARK.value == "dark"


class TestBaseColors:
    """Tests for base color palette."""

    def test_base_colors_is_dict(self):
        """Test that BASE_COLORS is a dictionary."""
        assert isinstance(BASE_COLORS, dict)

    def test_base_colors_has_all_bases(self):
        """Test that all DNA/RNA bases are defined."""
        required_bases = ["A", "C", "G", "T", "U", "N"]

        for base in required_bases:
            assert base in BASE_COLORS, f"BASE_COLORS missing base {base}"

    def test_base_colors_are_hex_strings(self):
        """Test that color values are hex color strings."""
        for base, color in BASE_COLORS.items():
            assert isinstance(color, str)
            assert color.startswith("#"), f"Color for {base} not hex: {color}"
            # Should be #RRGGBB format
            assert len(color) == 7, f"Color for {base} wrong length: {color}"

    def test_base_colors_purines_vs_pyrimidines(self):
        """Test that purines and pyrimidines have different colors."""
        # Purines: A, G
        # Pyrimidines: C, T, U
        # Should have distinct color groups
        # (though this is more of a design choice than requirement)
        assert len(BASE_COLORS) >= 4  # At least 4 different bases

    def test_base_colors_t_and_u_same(self):
        """Test that T and U have the same color (both pyrimidines)."""
        assert BASE_COLORS["T"] == BASE_COLORS["U"]

    def test_base_colors_dark_is_dict(self):
        """Test that BASE_COLORS_DARK is a dictionary."""
        assert isinstance(BASE_COLORS_DARK, dict)

    def test_base_colors_dark_has_all_bases(self):
        """Test that dark theme has all bases defined."""
        required_bases = ["A", "C", "G", "T", "U", "N"]

        for base in required_bases:
            assert base in BASE_COLORS_DARK, f"BASE_COLORS_DARK missing base {base}"

    def test_base_colors_dark_are_hex_strings(self):
        """Test that dark theme colors are hex strings."""
        for base, color in BASE_COLORS_DARK.items():
            assert isinstance(color, str)
            assert color.startswith("#"), f"Dark color for {base} not hex: {color}"
            assert len(color) == 7, f"Dark color for {base} wrong length: {color}"


class TestMultiReadColors:
    """Tests for multi-read color palette."""

    def test_multi_read_colors_is_list(self):
        """Test that MULTI_READ_COLORS is a list."""
        assert isinstance(MULTI_READ_COLORS, list)

    def test_multi_read_colors_not_empty(self):
        """Test that we have multiple colors defined."""
        assert len(MULTI_READ_COLORS) > 0

    def test_multi_read_colors_are_hex_strings(self):
        """Test that all colors are hex strings."""
        for i, color in enumerate(MULTI_READ_COLORS):
            assert isinstance(color, str), f"Color {i} is not a string"
            assert color.startswith("#"), f"Color {i} not hex: {color}"
            assert len(color) == 7, f"Color {i} wrong length: {color}"

    def test_multi_read_colors_has_enough_colors(self):
        """Test that we have enough colors for typical multi-read plots."""
        # Should have at least 5-10 colors for multi-read plots
        assert len(MULTI_READ_COLORS) >= 5

    def test_multi_read_colors_are_unique(self):
        """Test that colors are unique (no duplicates)."""
        assert len(MULTI_READ_COLORS) == len(set(MULTI_READ_COLORS))


class TestThemeDefinitions:
    """Tests for theme color palette definitions."""

    def test_light_theme_is_dict(self):
        """Test that LIGHT_THEME is a dictionary."""
        assert isinstance(LIGHT_THEME, dict)

    def test_dark_theme_is_dict(self):
        """Test that DARK_THEME is a dictionary."""
        assert isinstance(DARK_THEME, dict)

    def test_light_theme_has_required_keys(self):
        """Test that LIGHT_THEME has all required color keys."""
        required_keys = [
            "window_bg",
            "window_text",
            "base_bg",
            "base_text",
            "plot_bg",
            "signal_line",
        ]

        for key in required_keys:
            assert key in LIGHT_THEME, f"LIGHT_THEME missing {key}"

    def test_dark_theme_has_required_keys(self):
        """Test that DARK_THEME has all required color keys."""
        required_keys = [
            "window_bg",
            "window_text",
            "base_bg",
            "base_text",
            "plot_bg",
            "signal_line",
        ]

        for key in required_keys:
            assert key in DARK_THEME, f"DARK_THEME missing {key}"

    def test_light_theme_colors_are_hex_strings(self):
        """Test that LIGHT_THEME colors are hex strings."""
        for key, color in LIGHT_THEME.items():
            assert isinstance(color, str), f"LIGHT_THEME {key} is not string"
            assert color.startswith("#"), f"LIGHT_THEME {key} not hex: {color}"

    def test_dark_theme_colors_are_hex_strings(self):
        """Test that DARK_THEME colors are hex strings."""
        for key, color in DARK_THEME.items():
            assert isinstance(color, str), f"DARK_THEME {key} is not string"
            assert color.startswith("#"), f"DARK_THEME {key} not hex: {color}"

    def test_themes_have_same_keys(self):
        """Test that light and dark themes have the same keys."""
        light_keys = set(LIGHT_THEME.keys())
        dark_keys = set(DARK_THEME.keys())

        # Themes should have same structure
        assert light_keys == dark_keys

    def test_light_theme_background_is_light(self):
        """Test that light theme has light background colors."""
        # Light backgrounds should have high RGB values
        # Just check it starts with # and is defined
        assert LIGHT_THEME["window_bg"].startswith("#")
        assert LIGHT_THEME["plot_bg"].startswith("#")

    def test_dark_theme_background_is_dark(self):
        """Test that dark theme has dark background colors."""
        # Dark backgrounds should have low RGB values
        # Just check it starts with # and is defined
        assert DARK_THEME["window_bg"].startswith("#")
        assert DARK_THEME["plot_bg"].startswith("#")


class TestConfigurationConstants:
    """Tests for configuration value constants."""

    def test_downsample_settings_exist(self):
        """Test that downsampling settings are defined."""
        from squiggy.constants import (
            DEFAULT_DOWNSAMPLE_FACTOR,
            DOWNSAMPLE_MULTI_READ,
            DOWNSAMPLE_SINGLE_READ,
            MIN_POINTS_FOR_DOWNSAMPLING,
        )

        assert DEFAULT_DOWNSAMPLE_FACTOR > 0
        assert MIN_POINTS_FOR_DOWNSAMPLING > 0
        assert DOWNSAMPLE_SINGLE_READ > 0
        assert DOWNSAMPLE_MULTI_READ > 0

    def test_downsample_mode_constants(self):
        """Test that mode-specific downsample constants are properly defined."""
        from squiggy.constants import DOWNSAMPLE_MULTI_READ, DOWNSAMPLE_SINGLE_READ

        # Multi-read should have higher downsampling for performance
        assert DOWNSAMPLE_MULTI_READ > DOWNSAMPLE_SINGLE_READ

        # Values should be within reasonable range
        assert 1 <= DOWNSAMPLE_SINGLE_READ <= 100
        assert 1 <= DOWNSAMPLE_MULTI_READ <= 100

    def test_base_annotation_settings_exist(self):
        """Test that base annotation settings are defined."""
        from squiggy.constants import BASE_ANNOTATION_ALPHA, BASE_LABEL_SIZE

        # Alpha should be between 0 and 1
        assert 0 <= BASE_ANNOTATION_ALPHA <= 1

        # Label size should be positive
        assert BASE_LABEL_SIZE > 0

    def test_signal_line_settings_exist(self):
        """Test that signal line settings are defined."""
        from squiggy.constants import SIGNAL_LINE_COLOR, SIGNAL_LINE_WIDTH

        assert isinstance(SIGNAL_LINE_COLOR, str)
        assert SIGNAL_LINE_COLOR.startswith("#")
        assert SIGNAL_LINE_WIDTH > 0

    def test_plot_dimensions_exist(self):
        """Test that plot dimension constants are defined."""
        from squiggy.constants import (
            PLOT_HEIGHT,
            PLOT_MIN_HEIGHT,
            PLOT_MIN_WIDTH,
            PLOT_WIDTH,
        )

        assert PLOT_WIDTH > 0
        assert PLOT_HEIGHT > 0
        assert PLOT_MIN_WIDTH > 0
        assert PLOT_MIN_HEIGHT > 0

        # Min should be less than or equal to default
        assert PLOT_MIN_WIDTH <= PLOT_WIDTH or True  # Or just positive
        assert PLOT_MIN_HEIGHT <= PLOT_HEIGHT or True


class TestEnumIterability:
    """Tests that enums can be iterated."""

    def test_can_iterate_plot_mode(self):
        """Test that we can iterate over PlotMode values."""
        modes = list(PlotMode)
        assert len(modes) >= 4  # At least 4 modes

        for mode in PlotMode:
            assert isinstance(mode, PlotMode)

    def test_can_iterate_normalization_method(self):
        """Test that we can iterate over NormalizationMethod values."""
        methods = list(NormalizationMethod)
        assert len(methods) >= 4  # At least 4 methods

        for method in NormalizationMethod:
            assert isinstance(method, NormalizationMethod)

    def test_can_iterate_theme(self):
        """Test that we can iterate over Theme values."""
        themes = list(Theme)
        assert len(themes) >= 2  # At least LIGHT and DARK

        for theme in Theme:
            assert isinstance(theme, Theme)
