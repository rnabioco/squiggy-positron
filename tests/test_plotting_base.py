"""Tests for plotting/base.py - core plotting utilities"""

import numpy as np
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure

from squiggy.constants import NormalizationMethod, Theme
from squiggy.plotting.base import (
    MULTI_READ_COLORS,
    add_hover_tool,
    add_signal_renderers,
    configure_legend,
    create_figure,
    create_signal_data_source,
    format_html_title,
    format_plot_title,
    get_base_colors,
    get_signal_line_color,
    normalize_signal,
    process_signal,
)


class TestNormalization:
    """Tests for signal normalization functions"""

    def test_normalize_signal_none(self):
        """Test NONE normalization returns unchanged signal"""
        signal = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = normalize_signal(signal, NormalizationMethod.NONE)
        np.testing.assert_array_equal(result, signal)

    def test_normalize_signal_znorm(self):
        """Test Z-score normalization produces mean=0, std=1"""
        signal = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = normalize_signal(signal, NormalizationMethod.ZNORM)

        assert np.isclose(np.mean(result), 0.0, atol=1e-10)
        assert np.isclose(np.std(result), 1.0, atol=1e-10)

    def test_normalize_signal_median(self):
        """Test MEDIAN normalization produces median=0"""
        signal = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = normalize_signal(signal, NormalizationMethod.MEDIAN)

        assert np.isclose(np.median(result), 0.0, atol=1e-10)

    def test_normalize_signal_mad(self):
        """Test MAD normalization produces median=0"""
        signal = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = normalize_signal(signal, NormalizationMethod.MAD)

        assert np.isclose(np.median(result), 0.0, atol=1e-10)
        assert not np.array_equal(result, signal)

    def test_normalize_signal_mad_zero_mad(self):
        """Test MAD normalization with constant signal (MAD=0)"""
        signal = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        result = normalize_signal(signal, NormalizationMethod.MAD)

        # Should handle MAD=0 case gracefully
        assert np.all(np.isfinite(result))
        assert np.isclose(np.median(result), 0.0, atol=1e-10)

    def test_normalize_signal_preserves_shape(self):
        """Test normalization preserves array shape"""
        signal = np.random.randn(1000)

        for method in NormalizationMethod:
            result = normalize_signal(signal, method)
            assert result.shape == signal.shape


class TestProcessSignal:
    """Tests for process_signal function"""

    def test_process_signal_no_downsample(self):
        """Test process_signal without downsampling"""
        signal = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result, seq_map = process_signal(
            signal, NormalizationMethod.MEDIAN, downsample=1
        )

        assert result.shape == signal.shape
        assert seq_map is None

    def test_process_signal_with_downsample(self):
        """Test process_signal with downsampling factor"""
        signal = np.arange(100, dtype=float)
        result, seq_map = process_signal(
            signal, NormalizationMethod.NONE, downsample=10
        )

        assert len(result) == 10
        np.testing.assert_array_equal(result, signal[::10])

    def test_process_signal_downsample_with_seq_map(self):
        """Test process_signal downsamples seq_to_sig_map correctly"""
        signal = np.arange(100, dtype=float)
        seq_to_sig_map = [0, 20, 40, 60, 80]

        result_signal, result_map = process_signal(
            signal,
            NormalizationMethod.NONE,
            downsample=10,
            seq_to_sig_map=seq_to_sig_map,
        )

        assert len(result_signal) == 10
        assert result_map == [0, 2, 4, 6, 8]

    def test_process_signal_applies_normalization(self):
        """Test process_signal applies normalization before downsampling"""
        signal = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result, _ = process_signal(signal, NormalizationMethod.ZNORM, downsample=1)

        assert np.isclose(np.mean(result), 0.0, atol=1e-10)


class TestDataSource:
    """Tests for create_signal_data_source"""

    def test_create_signal_data_source_basic(self):
        """Test creating basic signal data source"""
        x = np.arange(10)
        signal = np.random.randn(10)

        source = create_signal_data_source(x, signal)

        assert isinstance(source, ColumnDataSource)
        assert "x" in source.data
        assert "y" in source.data
        assert "sample" in source.data
        np.testing.assert_array_equal(source.data["x"], x)
        np.testing.assert_array_equal(source.data["y"], signal)

    def test_create_signal_data_source_with_read_id(self):
        """Test creating data source with read ID"""
        x = np.arange(10)
        signal = np.random.randn(10)
        read_id = "test_read_123"

        source = create_signal_data_source(x, signal, read_id=read_id)

        assert "read_id" in source.data
        assert len(source.data["read_id"]) == 10
        assert all(rid == read_id for rid in source.data["read_id"])

    def test_create_signal_data_source_with_base_labels(self):
        """Test creating data source with base labels"""
        x = np.arange(5)
        signal = np.random.randn(5)
        base_labels = ["A", "C", "G", "T", "A"]

        source = create_signal_data_source(x, signal, base_labels=base_labels)

        assert "base" in source.data
        assert source.data["base"] == base_labels

    def test_create_signal_data_source_sample_field(self):
        """Test that sample field is 0-indexed array"""
        x = np.arange(10)
        signal = np.random.randn(10)

        source = create_signal_data_source(x, signal)

        np.testing.assert_array_equal(source.data["sample"], np.arange(10))


class TestFigureCreation:
    """Tests for figure creation and configuration"""

    def test_create_figure_basic(self):
        """Test creating basic figure"""
        p = create_figure("Test Title", "X Label", "Y Label")

        assert p is not None
        assert p.title.text == "Test Title"
        assert p.xaxis.axis_label == "X Label"
        assert p.yaxis.axis_label == "Y Label"

    def test_create_figure_has_tools(self):
        """Test figure has expected tools"""
        p = create_figure("Test", "X", "Y")

        tool_names = [tool.__class__.__name__ for tool in p.tools]
        assert "PanTool" in tool_names
        assert "WheelZoomTool" in tool_names
        assert "BoxZoomTool" in tool_names
        assert "ResetTool" in tool_names
        assert "SaveTool" in tool_names

    def test_create_figure_light_theme(self):
        """Test figure creation with light theme"""
        p = create_figure("Test", "X", "Y", theme=Theme.LIGHT)

        # Light theme should have white/light background
        assert p.background_fill_color.lower() == "#ffffff"

    def test_create_figure_dark_theme(self):
        """Test figure creation with dark theme"""
        p = create_figure("Test", "X", "Y", theme=Theme.DARK)

        # Dark theme should have dark background
        assert p.background_fill_color == "#2b2b2b"

    def test_create_figure_sizing_mode(self):
        """Test figure has stretch_both sizing mode"""
        p = create_figure("Test", "X", "Y")

        assert p.sizing_mode == "stretch_both"


class TestRenderers:
    """Tests for signal renderer functions"""

    def test_add_signal_renderers_line_only(self):
        """Test adding line renderer without scatter points"""
        p = figure()
        source = ColumnDataSource(data={"x": [1, 2, 3], "y": [1, 2, 3]})

        renderers = add_signal_renderers(p, source, "blue", show_signal_points=False)

        assert len(renderers) == 1
        assert renderers[0].glyph.line_color == "blue"

    def test_add_signal_renderers_with_scatter(self):
        """Test adding line and scatter renderers"""
        p = figure()
        source = ColumnDataSource(data={"x": [1, 2, 3], "y": [1, 2, 3]})

        renderers = add_signal_renderers(p, source, "red", show_signal_points=True)

        assert len(renderers) == 2  # Line + scatter

    def test_add_signal_renderers_with_legend(self):
        """Test adding renderers with legend label"""
        p = figure()
        source = ColumnDataSource(data={"x": [1, 2, 3], "y": [1, 2, 3]})

        renderers = add_signal_renderers(
            p, source, "green", show_signal_points=False, legend_label="Test Read"
        )

        assert len(renderers) == 1

    def test_add_signal_renderers_custom_fields(self):
        """Test adding renderers with custom data fields"""
        p = figure()
        source = ColumnDataSource(data={"time": [1, 2, 3], "signal": [10, 20, 30]})

        renderers = add_signal_renderers(
            p, source, "purple", x_field="time", y_field="signal"
        )

        assert len(renderers) == 1

    def test_add_signal_renderers_custom_width_alpha(self):
        """Test adding renderers with custom line width and alpha"""
        p = figure()
        source = ColumnDataSource(data={"x": [1, 2, 3], "y": [1, 2, 3]})

        renderers = add_signal_renderers(p, source, "orange", line_width=3, alpha=0.5)

        assert renderers[0].glyph.line_width == 3
        assert renderers[0].glyph.line_alpha == 0.5


class TestHoverAndLegend:
    """Tests for hover tool and legend configuration"""

    def test_add_hover_tool(self):
        """Test adding hover tool with tooltips"""
        p = figure()
        source = ColumnDataSource(data={"x": [1, 2, 3], "y": [1, 2, 3]})
        renderer = p.line(x="x", y="y", source=source)

        tooltip_fields = [("X Value", "@x"), ("Y Value", "@y")]
        add_hover_tool(p, [renderer], tooltip_fields)

        hover_tools = [tool for tool in p.tools if isinstance(tool, HoverTool)]
        assert len(hover_tools) == 1
        assert hover_tools[0].tooltips == tooltip_fields

    def test_configure_legend(self):
        """Test legend configuration"""
        p = figure()
        source = ColumnDataSource(data={"x": [1, 2, 3], "y": [1, 2, 3]})
        p.line(x="x", y="y", source=source, legend_label="Test")

        configure_legend(p)

        assert p.legend.click_policy == "hide"
        assert p.legend.location == "top_right"


class TestThemeColors:
    """Tests for theme color functions"""

    def test_get_base_colors_light(self):
        """Test getting base colors for light theme"""
        colors = get_base_colors(Theme.LIGHT)

        assert isinstance(colors, dict)
        assert "A" in colors
        assert "C" in colors
        assert "G" in colors
        assert "T" in colors

    def test_get_base_colors_dark(self):
        """Test getting base colors for dark theme"""
        colors = get_base_colors(Theme.DARK)

        assert isinstance(colors, dict)
        assert "A" in colors
        # Dark theme colors should be different from light
        light_colors = get_base_colors(Theme.LIGHT)
        # At least some colors should differ
        assert colors != light_colors

    def test_get_signal_line_color_light(self):
        """Test getting signal line color for light theme"""
        color = get_signal_line_color(Theme.LIGHT)

        assert isinstance(color, str)
        assert color.startswith("#") or color.isalpha()

    def test_get_signal_line_color_dark(self):
        """Test getting signal line color for dark theme"""
        color = get_signal_line_color(Theme.DARK)

        assert isinstance(color, str)
        # Dark theme signal color should differ from light
        light_color = get_signal_line_color(Theme.LIGHT)
        assert color != light_color


class TestTitleFormatting:
    """Tests for plot title formatting"""

    def test_format_plot_title_single_read(self):
        """Test formatting title for single read"""
        reads_data = [("read_001", np.array([10, 20, 30]), 4000)]
        title = format_plot_title("SINGLE", reads_data)

        assert "read_001" in title
        assert "SINGLE" in title

    def test_format_plot_title_multiple_reads(self):
        """Test formatting title for multiple reads"""
        reads_data = [
            ("read_001", np.array([10, 20, 30]), 4000),
            ("read_002", np.array([15, 25, 35]), 4000),
        ]
        title = format_plot_title("OVERLAY", reads_data)

        assert "OVERLAY" in title
        assert "2 reads" in title.lower() or "2" in title

    def test_format_plot_title_with_normalization(self):
        """Test title includes normalization method"""
        reads_data = [("read_001", np.array([10, 20, 30]), 4000)]
        title = format_plot_title(
            "SINGLE", reads_data, normalization=NormalizationMethod.ZNORM
        )

        assert "znorm" in title.lower()

    def test_format_plot_title_with_downsample(self):
        """Test title includes downsample factor"""
        reads_data = [("read_001", np.array([10, 20, 30]), 4000)]
        title = format_plot_title("SINGLE", reads_data, downsample=10)

        assert "10" in title or "downsample" in title.lower()

    def test_format_html_title_single_read(self):
        """Test HTML title formatting for single read"""
        reads_data = [("read_001", np.array([10, 20, 30]), 4000)]
        title = format_html_title("SINGLE", reads_data)

        assert "Squiggy" in title
        assert "SINGLE" in title
        assert "1 read" in title
        assert "read_001" in title

    def test_format_html_title_multiple_reads(self):
        """Test HTML title formatting for multiple reads"""
        reads_data = [
            ("read_001", np.array([10, 20, 30]), 4000),
            ("read_002", np.array([15, 25, 35]), 4000),
            ("read_003", np.array([5, 15, 25]), 4000),
        ]
        title = format_html_title("OVERLAY", reads_data)

        assert "Squiggy" in title
        assert "OVERLAY" in title
        assert "3 reads" in title


class TestMultiReadColors:
    """Tests for multi-read color palette"""

    def test_multi_read_colors_is_list(self):
        """Test MULTI_READ_COLORS is a list"""
        assert isinstance(MULTI_READ_COLORS, list)

    def test_multi_read_colors_has_multiple_colors(self):
        """Test palette has multiple colors for multi-read plots"""
        assert len(MULTI_READ_COLORS) >= 3

    def test_multi_read_colors_are_valid(self):
        """Test all colors are valid strings"""
        for color in MULTI_READ_COLORS:
            assert isinstance(color, str)
            assert len(color) > 0
