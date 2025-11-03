"""
Tests for SingleReadPlotStrategy
"""

import numpy as np
import pytest
from bokeh.layouts import LayoutDOM
from bokeh.models.plots import Plot

from squiggy.constants import NormalizationMethod, Theme
from squiggy.plot_strategies.single_read import SingleReadPlotStrategy


class TestSingleReadPlotStrategyInitialization:
    """Tests for SingleReadPlotStrategy initialization"""

    def test_init_with_light_theme(self):
        """Test initialization with LIGHT theme"""
        strategy = SingleReadPlotStrategy(Theme.LIGHT)

        assert strategy.theme == Theme.LIGHT
        assert strategy.theme_manager is not None
        assert strategy.theme_manager.theme == Theme.LIGHT

    def test_init_with_dark_theme(self):
        """Test initialization with DARK theme"""
        strategy = SingleReadPlotStrategy(Theme.DARK)

        assert strategy.theme == Theme.DARK
        assert strategy.theme_manager.theme == Theme.DARK


class TestDataValidation:
    """Tests for validate_data method"""

    def test_validate_with_required_data(self):
        """Test validation passes with all required data"""
        strategy = SingleReadPlotStrategy(Theme.LIGHT)

        data = {
            "signal": np.random.randn(1000),
            "read_id": "test_read",
            "sample_rate": 4000,
        }

        # Should not raise
        strategy.validate_data(data)

    def test_validate_missing_signal(self):
        """Test validation fails when signal is missing"""
        strategy = SingleReadPlotStrategy(Theme.LIGHT)

        data = {
            "read_id": "test_read",
            "sample_rate": 4000,
        }

        with pytest.raises(ValueError, match="Missing required data.*signal"):
            strategy.validate_data(data)

    def test_validate_missing_read_id(self):
        """Test validation fails when read_id is missing"""
        strategy = SingleReadPlotStrategy(Theme.LIGHT)

        data = {
            "signal": np.random.randn(1000),
            "sample_rate": 4000,
        }

        with pytest.raises(ValueError, match="Missing required data.*read_id"):
            strategy.validate_data(data)

    def test_validate_missing_sample_rate(self):
        """Test validation fails when sample_rate is missing"""
        strategy = SingleReadPlotStrategy(Theme.LIGHT)

        data = {
            "signal": np.random.randn(1000),
            "read_id": "test_read",
        }

        with pytest.raises(ValueError, match="Missing required data.*sample_rate"):
            strategy.validate_data(data)

    def test_validate_signal_wrong_type(self):
        """Test validation fails when signal is not numpy array"""
        strategy = SingleReadPlotStrategy(Theme.LIGHT)

        data = {
            "signal": [1, 2, 3],  # List instead of numpy array
            "read_id": "test_read",
            "sample_rate": 4000,
        }

        with pytest.raises(ValueError, match="signal must be a numpy array"):
            strategy.validate_data(data)

    def test_validate_read_id_wrong_type(self):
        """Test validation fails when read_id is not string"""
        strategy = SingleReadPlotStrategy(Theme.LIGHT)

        data = {
            "signal": np.random.randn(1000),
            "read_id": 12345,  # Int instead of string
            "sample_rate": 4000,
        }

        with pytest.raises(ValueError, match="read_id must be a string"):
            strategy.validate_data(data)


class TestCreatePlotBasic:
    """Tests for basic plot creation"""

    @pytest.fixture
    def minimal_data(self):
        """Minimal data for creating a plot"""
        return {
            "signal": np.random.randn(1000),
            "read_id": "test_read_001",
            "sample_rate": 4000,
        }

    @pytest.fixture
    def minimal_options(self):
        """Minimal options for creating a plot"""
        return {}

    def test_create_plot_returns_tuple(self, minimal_data, minimal_options):
        """Test that create_plot returns (html, figure) tuple"""
        strategy = SingleReadPlotStrategy(Theme.LIGHT)

        result = strategy.create_plot(minimal_data, minimal_options)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_create_plot_html_is_string(self, minimal_data, minimal_options):
        """Test that returned HTML is a string"""
        strategy = SingleReadPlotStrategy(Theme.LIGHT)

        html, _ = strategy.create_plot(minimal_data, minimal_options)

        assert isinstance(html, str)
        assert len(html) > 0
        assert "<html" in html
        assert "bokeh" in html.lower()

    def test_create_plot_returns_figure(self, minimal_data, minimal_options):
        """Test that returned figure is a Bokeh Plot"""
        strategy = SingleReadPlotStrategy(Theme.LIGHT)

        _, fig = strategy.create_plot(minimal_data, minimal_options)

        assert isinstance(fig, Plot)

    def test_create_plot_with_normalization(self, minimal_data):
        """Test plot creation with normalization"""
        strategy = SingleReadPlotStrategy(Theme.LIGHT)

        options = {"normalization": NormalizationMethod.ZNORM}

        html, fig = strategy.create_plot(minimal_data, options)

        assert isinstance(html, str)
        assert isinstance(fig, Plot)
        # Title should mention normalization
        assert "znorm" in fig.title.text.lower()

    def test_create_plot_with_downsample(self, minimal_data):
        """Test plot creation with downsampling"""
        strategy = SingleReadPlotStrategy(Theme.LIGHT)

        options = {"downsample": 10}

        html, fig = strategy.create_plot(minimal_data, options)

        assert isinstance(html, str)
        assert isinstance(fig, Plot)
        # Title should mention downsampling
        assert "10x" in fig.title.text


class TestCreatePlotWithAnnotations:
    """Tests for plot creation with base annotations"""

    @pytest.fixture
    def annotated_data(self):
        """Data with sequence and mapping for annotations"""
        return {
            "signal": np.random.randn(1000),
            "read_id": "test_read_002",
            "sample_rate": 4000,
            "sequence": "ACGTACGTACGT",
            "seq_to_sig_map": list(range(0, 1000, 83)),  # ~12 bases
        }

    def test_create_plot_with_sequence(self, annotated_data):
        """Test plot creation with sequence annotations"""
        strategy = SingleReadPlotStrategy(Theme.LIGHT)

        options = {"show_labels": True}

        html, fig = strategy.create_plot(annotated_data, options)

        assert isinstance(html, str)
        assert isinstance(fig, Plot)
        # Should have more renderers (annotations + signal)
        assert len(fig.renderers) > 1

    def test_create_plot_with_dwell_time_coloring(self, annotated_data):
        """Test plot creation with dwell time coloring"""
        strategy = SingleReadPlotStrategy(Theme.LIGHT)

        options = {"show_dwell_time": True}

        html, fig = strategy.create_plot(annotated_data, options)

        assert isinstance(html, str)
        assert isinstance(fig, Plot)
        # Should have color bar for dwell time
        assert any("ColorBar" in str(type(layout)) for layout in fig.right)

    def test_create_plot_without_labels(self, annotated_data):
        """Test plot creation without labels"""
        strategy = SingleReadPlotStrategy(Theme.LIGHT)

        options = {"show_labels": False}

        html, fig = strategy.create_plot(annotated_data, options)

        assert isinstance(html, str)
        assert isinstance(fig, Plot)

    def test_create_plot_with_signal_points(self, annotated_data):
        """Test plot creation with signal points enabled"""
        strategy = SingleReadPlotStrategy(Theme.LIGHT)

        options = {"show_signal_points": True}

        html, fig = strategy.create_plot(annotated_data, options)

        assert isinstance(html, str)
        assert isinstance(fig, Plot)


class TestXAxisModes:
    """Tests for different x-axis scaling modes"""

    @pytest.fixture
    def annotated_data(self):
        return {
            "signal": np.random.randn(400),
            "read_id": "test_read_003",
            "sample_rate": 4000,
            "sequence": "ACGT",
            "seq_to_sig_map": [0, 100, 200, 300],
        }

    def test_create_plot_regular_time_axis(self, annotated_data):
        """Test plot with regular time x-axis"""
        strategy = SingleReadPlotStrategy(Theme.LIGHT)

        options = {"scale_dwell_time": False}

        _, fig = strategy.create_plot(annotated_data, options)

        # X-axis label should be "Base Position" (with sequence but no scale_dwell_time)
        assert "Position" in fig.xaxis.axis_label

    def test_create_plot_dwell_time_axis(self, annotated_data):
        """Test plot with dwell time scaled x-axis"""
        strategy = SingleReadPlotStrategy(Theme.LIGHT)

        options = {"scale_dwell_time": True}

        _, fig = strategy.create_plot(annotated_data, options)

        # X-axis label should mention dwell time
        assert "Dwell" in fig.xaxis.axis_label

    def test_create_plot_base_position_axis(self, annotated_data):
        """Test plot with base position x-axis"""
        strategy = SingleReadPlotStrategy(Theme.LIGHT)

        options = {"scale_dwell_time": False}

        _, fig = strategy.create_plot(annotated_data, options)

        assert "Position" in fig.xaxis.axis_label


class TestModificationTrack:
    """Tests for modification track creation"""

    @pytest.fixture
    def data_with_mods(self):
        """Data with modifications"""

        # Mock modification
        class MockMod:
            def __init__(self, position, mod_code, probability):
                self.position = position
                self.mod_code = mod_code
                self.canonical_base = "C"
                self.probability = probability
                self.signal_start = position * 100
                self.signal_end = (position + 1) * 100

        return {
            "signal": np.random.randn(400),
            "read_id": "test_read_004",
            "sample_rate": 4000,
            "sequence": "ACGT",
            "seq_to_sig_map": [0, 100, 200, 300],
            "modifications": [
                MockMod(0, "m", 0.9),
                MockMod(2, "h", 0.8),
            ],
        }

    def test_create_plot_with_modifications(self, data_with_mods):
        """Test plot creation with modification track"""
        strategy = SingleReadPlotStrategy(Theme.LIGHT)

        options = {"show_modification_overlay": True}

        html, layout = strategy.create_plot(data_with_mods, options)

        assert isinstance(html, str)
        # Should return layout (not just figure) when mods present
        assert isinstance(layout, LayoutDOM)

    def test_create_plot_without_modification_overlay(self, data_with_mods):
        """Test plot creation with modifications disabled"""
        strategy = SingleReadPlotStrategy(Theme.LIGHT)

        options = {"show_modification_overlay": False}

        html, fig = strategy.create_plot(data_with_mods, options)

        # Should return just figure (not layout)
        assert isinstance(fig, Plot)
        assert not isinstance(fig, LayoutDOM) or isinstance(fig, Plot)

    def test_create_plot_modification_filters(self, data_with_mods):
        """Test plot creation with modification filters"""
        strategy = SingleReadPlotStrategy(Theme.LIGHT)

        options = {
            "show_modification_overlay": True,
            "min_mod_probability": 0.85,
            "enabled_mod_types": ["m"],
        }

        html, layout = strategy.create_plot(data_with_mods, options)

        assert isinstance(html, str)
        assert isinstance(layout, LayoutDOM)


class TestPrivateProcessSignal:
    """Tests for _process_signal private method"""

    def test_process_signal_no_normalization_no_downsample(self):
        """Test signal processing with no changes"""
        strategy = SingleReadPlotStrategy(Theme.LIGHT)

        signal = np.random.randn(1000)
        processed, mapping = strategy._process_signal(
            signal=signal.copy(),
            normalization=NormalizationMethod.NONE,
            downsample=1,
            seq_to_sig_map=None,
        )

        # Signal should be unchanged
        np.testing.assert_array_equal(processed, signal)

    def test_process_signal_with_normalization(self):
        """Test signal processing with normalization"""
        strategy = SingleReadPlotStrategy(Theme.LIGHT)

        signal = np.random.randn(1000) * 10 + 50  # Not normalized
        processed, _ = strategy._process_signal(
            signal=signal,
            normalization=NormalizationMethod.ZNORM,
            downsample=1,
            seq_to_sig_map=None,
        )

        # Should be z-normalized (mean~0, std~1)
        assert abs(np.mean(processed)) < 0.1
        assert abs(np.std(processed) - 1.0) < 0.1

    def test_process_signal_with_downsample(self):
        """Test signal processing with downsampling"""
        strategy = SingleReadPlotStrategy(Theme.LIGHT)

        signal = np.random.randn(1000)
        processed, _ = strategy._process_signal(
            signal=signal,
            normalization=NormalizationMethod.NONE,
            downsample=10,
            seq_to_sig_map=None,
        )

        # Should be 10x smaller
        assert len(processed) == len(signal) // 10

    def test_process_signal_updates_mapping(self):
        """Test that downsampling updates seq_to_sig_map"""
        strategy = SingleReadPlotStrategy(Theme.LIGHT)

        signal = np.random.randn(1000)
        mapping = [0, 100, 200, 300]

        _, processed_mapping = strategy._process_signal(
            signal=signal,
            normalization=NormalizationMethod.NONE,
            downsample=10,
            seq_to_sig_map=mapping,
        )

        # Mapping should be scaled down
        expected = [0, 10, 20, 30]
        assert processed_mapping == expected


class TestPrivateCreateXAxis:
    """Tests for _create_x_axis private methods"""

    def test_create_x_axis_regular_time(self):
        """Test creation of regular time axis"""
        strategy = SingleReadPlotStrategy(Theme.LIGHT)

        signal = np.random.randn(1000)
        sample_rate = 4000

        time_ms, x_label = strategy._create_x_axis(
            signal=signal,
            sample_rate=sample_rate,
            sequence=None,
            seq_to_sig_map=None,
            scale_dwell_time=False,
        )

        assert len(time_ms) == len(signal)
        assert "Time" in x_label
        # Check time is in milliseconds
        assert time_ms[-1] == (len(signal) - 1) * 1000 / sample_rate

    def test_create_x_axis_base_position(self):
        """Test creation of base position axis"""
        strategy = SingleReadPlotStrategy(Theme.LIGHT)

        signal = np.random.randn(400)
        sequence = "ACGT"
        seq_to_sig_map = [0, 100, 200, 300]

        time_ms, x_label = strategy._create_x_axis(
            signal=signal,
            sample_rate=4000,
            sequence=sequence,
            seq_to_sig_map=seq_to_sig_map,
            scale_dwell_time=False,
        )

        assert len(time_ms) == len(signal)
        assert "Position" in x_label

    def test_create_x_axis_dwell_time(self):
        """Test creation of dwell time axis"""
        strategy = SingleReadPlotStrategy(Theme.LIGHT)

        signal = np.random.randn(400)
        sequence = "ACGT"
        seq_to_sig_map = [0, 100, 200, 300]

        time_ms, x_label = strategy._create_x_axis(
            signal=signal,
            sample_rate=4000,
            sequence=sequence,
            seq_to_sig_map=seq_to_sig_map,
            scale_dwell_time=True,
        )

        assert len(time_ms) == len(signal)
        assert "Dwell" in x_label


class TestSingleReadPlotStrategyIntegration:
    """Integration tests for SingleReadPlotStrategy"""

    def test_complete_workflow_minimal(self):
        """Test complete workflow with minimal data"""
        strategy = SingleReadPlotStrategy(Theme.LIGHT)

        data = {
            "signal": np.random.randn(1000),
            "read_id": "integration_test_001",
            "sample_rate": 4000,
        }

        options = {
            "normalization": NormalizationMethod.MEDIAN,
            "downsample": 2,
        }

        html, fig = strategy.create_plot(data, options)

        assert isinstance(html, str)
        assert isinstance(fig, Plot)
        assert "integration_test_001" in fig.title.text

    def test_complete_workflow_with_annotations(self):
        """Test complete workflow with full annotations"""
        strategy = SingleReadPlotStrategy(Theme.DARK)

        data = {
            "signal": np.random.randn(800),
            "read_id": "integration_test_002",
            "sample_rate": 4000,
            "sequence": "ACGTACGT",
            "seq_to_sig_map": list(range(0, 800, 100)),
        }

        options = {
            "normalization": NormalizationMethod.ZNORM,
            "downsample": 1,
            "show_dwell_time": True,
            "show_labels": True,
            "show_signal_points": True,
            "scale_dwell_time": False,
        }

        html, fig = strategy.create_plot(data, options)

        assert isinstance(html, str)
        assert isinstance(fig, Plot)
        # Should have color bar for dwell time
        assert len(fig.right) > 0

    def test_reuse_strategy_multiple_plots(self):
        """Test that same strategy can create multiple plots"""
        strategy = SingleReadPlotStrategy(Theme.LIGHT)

        data1 = {
            "signal": np.random.randn(500),
            "read_id": "read_1",
            "sample_rate": 4000,
        }

        data2 = {
            "signal": np.random.randn(600),
            "read_id": "read_2",
            "sample_rate": 4000,
        }

        html1, fig1 = strategy.create_plot(data1, {})
        html2, fig2 = strategy.create_plot(data2, {})

        assert isinstance(html1, str)
        assert isinstance(html2, str)
        assert isinstance(fig1, Plot)
        assert isinstance(fig2, Plot)
        assert "read_1" in fig1.title.text
        assert "read_2" in fig2.title.text
