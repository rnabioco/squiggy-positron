"""
Tests for Overlay and Stacked plot strategies
"""

import numpy as np
import pytest
from bokeh.models.plots import Plot

from squiggy.constants import NormalizationMethod, Theme
from squiggy.plot_strategies.overlay import OverlayPlotStrategy
from squiggy.plot_strategies.stacked import StackedPlotStrategy


class TestOverlayPlotStrategyInitialization:
    """Tests for OverlayPlotStrategy initialization"""

    def test_init_with_light_theme(self):
        """Test initialization with LIGHT theme"""
        strategy = OverlayPlotStrategy(Theme.LIGHT)

        assert strategy.theme == Theme.LIGHT
        assert strategy.theme_manager is not None

    def test_init_with_dark_theme(self):
        """Test initialization with DARK theme"""
        strategy = OverlayPlotStrategy(Theme.DARK)

        assert strategy.theme == Theme.DARK


class TestStackedPlotStrategyInitialization:
    """Tests for StackedPlotStrategy initialization"""

    def test_init_with_light_theme(self):
        """Test initialization with LIGHT theme"""
        strategy = StackedPlotStrategy(Theme.LIGHT)

        assert strategy.theme == Theme.LIGHT
        assert strategy.theme_manager is not None

    def test_init_with_dark_theme(self):
        """Test initialization with DARK theme"""
        strategy = StackedPlotStrategy(Theme.DARK)

        assert strategy.theme == Theme.DARK


class TestOverlayDataValidation:
    """Tests for OverlayPlotStrategy data validation"""

    def test_validate_with_required_data(self):
        """Test validation passes with all required data"""
        strategy = OverlayPlotStrategy(Theme.LIGHT)

        data = {
            "reads": [
                ("read_001", np.random.randn(1000), 4000),
                ("read_002", np.random.randn(1000), 4000),
            ]
        }

        # Should not raise
        strategy.validate_data(data)

    def test_validate_missing_reads(self):
        """Test validation fails when reads is missing"""
        strategy = OverlayPlotStrategy(Theme.LIGHT)

        data = {}

        with pytest.raises(ValueError, match="Missing required data.*reads"):
            strategy.validate_data(data)

    def test_validate_empty_reads_list(self):
        """Test validation fails with empty reads list"""
        strategy = OverlayPlotStrategy(Theme.LIGHT)

        data = {"reads": []}

        with pytest.raises(ValueError, match="reads list cannot be empty"):
            strategy.validate_data(data)

    def test_validate_invalid_read_tuple(self):
        """Test validation fails with invalid read tuple"""
        strategy = OverlayPlotStrategy(Theme.LIGHT)

        data = {
            "reads": [
                ("read_001", np.random.randn(1000)),  # Missing sample_rate
            ]
        }

        with pytest.raises(ValueError, match="must be a tuple"):
            strategy.validate_data(data)

    def test_validate_read_id_wrong_type(self):
        """Test validation fails when read_id is not string"""
        strategy = OverlayPlotStrategy(Theme.LIGHT)

        data = {
            "reads": [
                (123, np.random.randn(1000), 4000),  # Int instead of string
            ]
        }

        with pytest.raises(ValueError, match="read_id must be a string"):
            strategy.validate_data(data)


class TestStackedDataValidation:
    """Tests for StackedPlotStrategy data validation"""

    def test_validate_with_required_data(self):
        """Test validation passes with all required data"""
        strategy = StackedPlotStrategy(Theme.LIGHT)

        data = {
            "reads": [
                ("read_001", np.random.randn(1000), 4000),
                ("read_002", np.random.randn(1000), 4000),
            ]
        }

        # Should not raise
        strategy.validate_data(data)

    def test_validate_missing_reads(self):
        """Test validation fails when reads is missing"""
        strategy = StackedPlotStrategy(Theme.LIGHT)

        data = {}

        with pytest.raises(ValueError, match="Missing required data.*reads"):
            strategy.validate_data(data)


class TestOverlayCreatePlot:
    """Tests for OverlayPlotStrategy create_plot"""

    @pytest.fixture
    def sample_reads_data(self):
        """Sample reads data for testing"""
        return {
            "reads": [
                ("read_001", np.random.randn(1000), 4000),
                ("read_002", np.random.randn(1000), 4000),
                ("read_003", np.random.randn(1000), 4000),
            ]
        }

    def test_create_plot_returns_tuple(self, sample_reads_data):
        """Test that create_plot returns (html, figure) tuple"""
        strategy = OverlayPlotStrategy(Theme.LIGHT)

        result = strategy.create_plot(sample_reads_data, {})

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_create_plot_html_is_string(self, sample_reads_data):
        """Test that returned HTML is a string"""
        strategy = OverlayPlotStrategy(Theme.LIGHT)

        html, _ = strategy.create_plot(sample_reads_data, {})

        assert isinstance(html, str)
        assert len(html) > 0
        assert "<html" in html
        assert "bokeh" in html.lower()

    def test_create_plot_returns_figure(self, sample_reads_data):
        """Test that returned figure is a Bokeh Plot"""
        strategy = OverlayPlotStrategy(Theme.LIGHT)

        _, fig = strategy.create_plot(sample_reads_data, {})

        assert isinstance(fig, Plot)

    def test_create_plot_with_normalization(self, sample_reads_data):
        """Test plot creation with normalization"""
        strategy = OverlayPlotStrategy(Theme.LIGHT)

        options = {"normalization": NormalizationMethod.ZNORM}

        html, fig = strategy.create_plot(sample_reads_data, options)

        assert isinstance(html, str)
        assert isinstance(fig, Plot)
        assert "znorm" in fig.title.text.lower()

    def test_create_plot_with_downsample(self, sample_reads_data):
        """Test plot creation with downsampling"""
        strategy = OverlayPlotStrategy(Theme.LIGHT)

        options = {"downsample": 10}

        html, fig = strategy.create_plot(sample_reads_data, options)

        assert isinstance(html, str)
        assert "10x" in fig.title.text

    def test_create_plot_with_signal_points(self, sample_reads_data):
        """Test plot creation with signal points"""
        strategy = OverlayPlotStrategy(Theme.LIGHT)

        options = {"show_signal_points": True}

        html, fig = strategy.create_plot(sample_reads_data, options)

        assert isinstance(html, str)
        assert isinstance(fig, Plot)

    def test_create_plot_has_legend(self, sample_reads_data):
        """Test that plot has legend for multiple reads"""
        strategy = OverlayPlotStrategy(Theme.LIGHT)

        _, fig = strategy.create_plot(sample_reads_data, {})

        # Should have legend
        assert fig.legend is not None
        assert fig.legend.click_policy == "hide"

    def test_create_plot_title_shows_read_count(self, sample_reads_data):
        """Test that title shows number of reads"""
        strategy = OverlayPlotStrategy(Theme.LIGHT)

        _, fig = strategy.create_plot(sample_reads_data, {})

        assert "3 reads" in fig.title.text


class TestStackedCreatePlot:
    """Tests for StackedPlotStrategy create_plot"""

    @pytest.fixture
    def sample_reads_data(self):
        """Sample reads data for testing"""
        return {
            "reads": [
                ("read_001", np.random.randn(1000), 4000),
                ("read_002", np.random.randn(1000), 4000),
                ("read_003", np.random.randn(1000), 4000),
            ]
        }

    def test_create_plot_returns_tuple(self, sample_reads_data):
        """Test that create_plot returns (html, figure) tuple"""
        strategy = StackedPlotStrategy(Theme.LIGHT)

        result = strategy.create_plot(sample_reads_data, {})

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_create_plot_html_is_string(self, sample_reads_data):
        """Test that returned HTML is a string"""
        strategy = StackedPlotStrategy(Theme.LIGHT)

        html, _ = strategy.create_plot(sample_reads_data, {})

        assert isinstance(html, str)
        assert len(html) > 0

    def test_create_plot_returns_figure(self, sample_reads_data):
        """Test that returned figure is a Bokeh Plot"""
        strategy = StackedPlotStrategy(Theme.LIGHT)

        _, fig = strategy.create_plot(sample_reads_data, {})

        assert isinstance(fig, Plot)

    def test_create_plot_with_normalization(self, sample_reads_data):
        """Test plot creation with normalization"""
        strategy = StackedPlotStrategy(Theme.LIGHT)

        options = {"normalization": NormalizationMethod.MEDIAN}

        html, fig = strategy.create_plot(sample_reads_data, options)

        assert isinstance(html, str)
        assert "median" in fig.title.text.lower()

    def test_create_plot_ylabel_has_offset(self, sample_reads_data):
        """Test that y-axis label mentions offset"""
        strategy = StackedPlotStrategy(Theme.LIGHT)

        _, fig = strategy.create_plot(sample_reads_data, {})

        assert "offset" in fig.yaxis.axis_label.lower()

    def test_create_plot_title_shows_read_count(self, sample_reads_data):
        """Test that title shows number of reads"""
        strategy = StackedPlotStrategy(Theme.LIGHT)

        _, fig = strategy.create_plot(sample_reads_data, {})

        assert "3 reads" in fig.title.text


class TestOverlayPrivateMethods:
    """Tests for OverlayPlotStrategy private methods"""

    def test_process_signal_no_normalization_no_downsample(self):
        """Test signal processing with no changes"""
        strategy = OverlayPlotStrategy(Theme.LIGHT)

        signal = np.random.randn(1000)
        processed, seq_map = strategy._process_signal(
            signal=signal.copy(),
            normalization=NormalizationMethod.NONE,
            downsample=1,
        )

        np.testing.assert_array_equal(processed, signal)
        assert seq_map is None

    def test_process_signal_with_normalization(self):
        """Test signal processing with normalization"""
        strategy = OverlayPlotStrategy(Theme.LIGHT)

        signal = np.random.randn(1000) * 10 + 50
        processed, seq_map = strategy._process_signal(
            signal=signal,
            normalization=NormalizationMethod.ZNORM,
            downsample=1,
        )

        # Should be z-normalized
        assert abs(np.mean(processed)) < 0.1
        assert abs(np.std(processed) - 1.0) < 0.1
        assert seq_map is None

    def test_process_signal_with_downsample(self):
        """Test signal processing with downsampling"""
        strategy = OverlayPlotStrategy(Theme.LIGHT)

        signal = np.random.randn(1000)
        processed, seq_map = strategy._process_signal(
            signal=signal,
            normalization=NormalizationMethod.NONE,
            downsample=10,
        )

        assert len(processed) == len(signal) // 10
        assert seq_map is None


class TestStackedPrivateMethods:
    """Tests for StackedPlotStrategy private methods"""

    def test_process_signal_no_normalization_no_downsample(self):
        """Test signal processing with no changes"""
        strategy = StackedPlotStrategy(Theme.LIGHT)

        signal = np.random.randn(1000)
        processed, seq_map = strategy._process_signal(
            signal=signal.copy(),
            normalization=NormalizationMethod.NONE,
            downsample=1,
        )

        np.testing.assert_array_equal(processed, signal)
        assert seq_map is None

    def test_process_signal_with_normalization(self):
        """Test signal processing with normalization"""
        strategy = StackedPlotStrategy(Theme.LIGHT)

        signal = np.random.randn(1000) * 10 + 50
        processed, seq_map = strategy._process_signal(
            signal=signal,
            normalization=NormalizationMethod.ZNORM,
            downsample=1,
        )

        # Should be z-normalized
        assert abs(np.mean(processed)) < 0.1
        assert abs(np.std(processed) - 1.0) < 0.1
        assert seq_map is None


class TestMultiReadIntegration:
    """Integration tests for multi-read strategies"""

    def test_overlay_complete_workflow(self):
        """Test complete overlay workflow"""
        strategy = OverlayPlotStrategy(Theme.DARK)

        data = {
            "reads": [
                ("read_A", np.random.randn(500), 4000),
                ("read_B", np.random.randn(600), 4000),
            ]
        }

        options = {
            "normalization": NormalizationMethod.MEDIAN,
            "downsample": 2,
            "show_signal_points": False,
        }

        html, fig = strategy.create_plot(data, options)

        assert isinstance(html, str)
        assert isinstance(fig, Plot)
        assert "2 reads" in fig.title.text

    def test_stacked_complete_workflow(self):
        """Test complete stacked workflow"""
        strategy = StackedPlotStrategy(Theme.LIGHT)

        data = {
            "reads": [
                ("read_A", np.random.randn(500), 4000),
                ("read_B", np.random.randn(600), 4000),
                ("read_C", np.random.randn(550), 4000),
            ]
        }

        options = {
            "normalization": NormalizationMethod.ZNORM,
            "downsample": 5,
        }

        html, fig = strategy.create_plot(data, options)

        assert isinstance(html, str)
        assert isinstance(fig, Plot)
        assert "3 reads" in fig.title.text

    def test_reuse_overlay_strategy(self):
        """Test that same strategy can create multiple plots"""
        strategy = OverlayPlotStrategy(Theme.LIGHT)

        data1 = {
            "reads": [
                ("read_1", np.random.randn(400), 4000),
                ("read_2", np.random.randn(400), 4000),
            ]
        }

        data2 = {
            "reads": [
                ("read_3", np.random.randn(500), 4000),
                ("read_4", np.random.randn(500), 4000),
                ("read_5", np.random.randn(500), 4000),
            ]
        }

        html1, fig1 = strategy.create_plot(data1, {})
        html2, fig2 = strategy.create_plot(data2, {})

        assert "2 reads" in fig1.title.text
        assert "3 reads" in fig2.title.text

    def test_reuse_stacked_strategy(self):
        """Test that same strategy can create multiple plots"""
        strategy = StackedPlotStrategy(Theme.LIGHT)

        data1 = {
            "reads": [
                ("read_1", np.random.randn(400), 4000),
                ("read_2", np.random.randn(400), 4000),
            ]
        }

        data2 = {
            "reads": [
                ("read_3", np.random.randn(500), 4000),
            ]
        }

        html1, fig1 = strategy.create_plot(data1, {})
        html2, fig2 = strategy.create_plot(data2, {})

        assert "2 reads" in fig1.title.text
        assert "1 reads" in fig2.title.text

    def test_many_reads_overlay(self):
        """Test overlay with many reads"""
        strategy = OverlayPlotStrategy(Theme.LIGHT)

        # Create 10 reads
        reads = [(f"read_{i:03d}", np.random.randn(300), 4000) for i in range(10)]

        data = {"reads": reads}

        html, fig = strategy.create_plot(data, {})

        assert isinstance(html, str)
        assert "10 reads" in fig.title.text

    def test_many_reads_stacked(self):
        """Test stacked with many reads"""
        strategy = StackedPlotStrategy(Theme.LIGHT)

        # Create 10 reads
        reads = [(f"read_{i:03d}", np.random.randn(300), 4000) for i in range(10)]

        data = {"reads": reads}

        html, fig = strategy.create_plot(data, {})

        assert isinstance(html, str)
        assert "10 reads" in fig.title.text
