"""
Tests for EventAlignPlotStrategy
"""

import numpy as np
import pytest
from bokeh.models.plots import Plot

from squiggy.constants import NormalizationMethod, Theme
from squiggy.plot_strategies.eventalign import EventAlignPlotStrategy


# Mock BaseAnnotation class
class MockBaseAnnotation:
    """Mock base annotation for testing"""

    def __init__(self, base: str, signal_start: int):
        self.base = base
        self.signal_start = signal_start


# Mock AlignedRead class
class MockAlignedRead:
    """Mock aligned read for testing"""

    def __init__(self, read_id: str, bases: list):
        self.read_id = read_id
        self.bases = bases


class TestEventAlignPlotStrategyInitialization:
    """Tests for EventAlignPlotStrategy initialization"""

    def test_init_with_light_theme(self):
        """Test initialization with LIGHT theme"""
        strategy = EventAlignPlotStrategy(Theme.LIGHT)

        assert strategy.theme == Theme.LIGHT
        assert strategy.theme_manager is not None

    def test_init_with_dark_theme(self):
        """Test initialization with DARK theme"""
        strategy = EventAlignPlotStrategy(Theme.DARK)

        assert strategy.theme == Theme.DARK


class TestDataValidation:
    """Tests for EventAlignPlotStrategy data validation"""

    @pytest.fixture
    def valid_data(self):
        """Valid data for testing"""
        bases = [
            MockBaseAnnotation("A", 0),
            MockBaseAnnotation("C", 100),
            MockBaseAnnotation("G", 200),
            MockBaseAnnotation("T", 300),
        ]
        aligned = MockAlignedRead("read_001", bases)

        return {
            "reads": [("read_001", np.random.randn(400), 4000)],
            "aligned_reads": [aligned],
        }

    def test_validate_with_required_data(self, valid_data):
        """Test validation passes with all required data"""
        strategy = EventAlignPlotStrategy(Theme.LIGHT)

        # Should not raise
        strategy.validate_data(valid_data)

    def test_validate_missing_reads(self):
        """Test validation fails when reads is missing"""
        strategy = EventAlignPlotStrategy(Theme.LIGHT)

        data = {"aligned_reads": []}

        with pytest.raises(ValueError, match="Missing required data.*reads"):
            strategy.validate_data(data)

    def test_validate_missing_aligned_reads(self):
        """Test validation fails when aligned_reads is missing"""
        strategy = EventAlignPlotStrategy(Theme.LIGHT)

        data = {"reads": [("read_001", np.random.randn(400), 4000)]}

        with pytest.raises(ValueError, match="Missing required data.*aligned_reads"):
            strategy.validate_data(data)

    def test_validate_empty_reads(self):
        """Test validation fails with empty reads list"""
        strategy = EventAlignPlotStrategy(Theme.LIGHT)

        data = {
            "reads": [],
            "aligned_reads": [],
        }

        with pytest.raises(ValueError, match="reads must be a non-empty list"):
            strategy.validate_data(data)

    def test_validate_mismatched_lengths(self, valid_data):
        """Test validation fails when reads and aligned_reads have different lengths"""
        strategy = EventAlignPlotStrategy(Theme.LIGHT)

        # Add extra read but not aligned read
        valid_data["reads"].append(("read_002", np.random.randn(400), 4000))

        with pytest.raises(ValueError, match="must have same length"):
            strategy.validate_data(valid_data)

    def test_validate_aligned_read_without_bases(self):
        """Test validation fails when aligned read doesn't have bases"""
        strategy = EventAlignPlotStrategy(Theme.LIGHT)

        class NoBasesRead:
            pass

        data = {
            "reads": [("read_001", np.random.randn(400), 4000)],
            "aligned_reads": [NoBasesRead()],
        }

        with pytest.raises(ValueError, match="must have 'bases' attribute"):
            strategy.validate_data(data)


class TestCreatePlot:
    """Tests for EventAlignPlotStrategy create_plot"""

    @pytest.fixture
    def sample_data(self):
        """Sample data for testing"""
        bases = [
            MockBaseAnnotation("A", 0),
            MockBaseAnnotation("C", 100),
            MockBaseAnnotation("G", 200),
            MockBaseAnnotation("T", 300),
        ]
        aligned = MockAlignedRead("read_001", bases)

        return {
            "reads": [("read_001", np.random.randn(400), 4000)],
            "aligned_reads": [aligned],
        }

    def test_create_plot_returns_tuple(self, sample_data):
        """Test that create_plot returns (html, figure) tuple"""
        strategy = EventAlignPlotStrategy(Theme.LIGHT)

        result = strategy.create_plot(sample_data, {})

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_create_plot_html_is_string(self, sample_data):
        """Test that returned HTML is a string"""
        strategy = EventAlignPlotStrategy(Theme.LIGHT)

        html, _ = strategy.create_plot(sample_data, {})

        assert isinstance(html, str)
        assert len(html) > 0
        assert "<html" in html

    def test_create_plot_returns_figure(self, sample_data):
        """Test that returned figure is a Bokeh Plot"""
        strategy = EventAlignPlotStrategy(Theme.LIGHT)

        _, fig = strategy.create_plot(sample_data, {})

        assert isinstance(fig, Plot)

    def test_create_plot_with_normalization(self, sample_data):
        """Test plot creation with normalization"""
        strategy = EventAlignPlotStrategy(Theme.LIGHT)

        options = {"normalization": NormalizationMethod.ZNORM}

        html, fig = strategy.create_plot(sample_data, options)

        assert isinstance(html, str)
        assert "znorm" in fig.title.text.lower()

    def test_create_plot_base_position_mode(self, sample_data):
        """Test plot creation with base position x-axis"""
        strategy = EventAlignPlotStrategy(Theme.LIGHT)

        options = {"show_dwell_time": False}

        _, fig = strategy.create_plot(sample_data, options)

        assert "Position" in fig.xaxis.axis_label

    def test_create_plot_time_mode(self, sample_data):
        """Test plot creation with time x-axis"""
        strategy = EventAlignPlotStrategy(Theme.LIGHT)

        options = {"show_dwell_time": True}

        _, fig = strategy.create_plot(sample_data, options)

        assert "Time" in fig.xaxis.axis_label

    def test_create_plot_with_labels(self, sample_data):
        """Test plot creation with labels enabled"""
        strategy = EventAlignPlotStrategy(Theme.LIGHT)

        options = {"show_labels": True}

        html, fig = strategy.create_plot(sample_data, options)

        assert isinstance(html, str)
        assert isinstance(fig, Plot)

    def test_create_plot_without_labels(self, sample_data):
        """Test plot creation with labels disabled"""
        strategy = EventAlignPlotStrategy(Theme.LIGHT)

        options = {"show_labels": False}

        html, fig = strategy.create_plot(sample_data, options)

        assert isinstance(html, str)
        assert isinstance(fig, Plot)

    def test_create_plot_with_signal_points(self, sample_data):
        """Test plot creation with signal points"""
        strategy = EventAlignPlotStrategy(Theme.LIGHT)

        options = {"show_signal_points": True}

        html, fig = strategy.create_plot(sample_data, options)

        assert isinstance(html, str)

    def test_create_plot_with_downsample(self, sample_data):
        """Test plot creation with downsampling"""
        strategy = EventAlignPlotStrategy(Theme.LIGHT)

        options = {"downsample": 5}

        html, fig = strategy.create_plot(sample_data, options)

        assert isinstance(html, str)
        assert "5x" in fig.title.text


class TestMultipleAlignedReads:
    """Tests for plotting multiple aligned reads"""

    @pytest.fixture
    def multi_read_data(self):
        """Data with multiple aligned reads"""
        bases1 = [
            MockBaseAnnotation("A", 0),
            MockBaseAnnotation("C", 100),
            MockBaseAnnotation("G", 200),
            MockBaseAnnotation("T", 300),
        ]
        bases2 = [
            MockBaseAnnotation("A", 0),
            MockBaseAnnotation("C", 100),
            MockBaseAnnotation("G", 200),
            MockBaseAnnotation("T", 300),
        ]

        aligned1 = MockAlignedRead("read_001", bases1)
        aligned2 = MockAlignedRead("read_002", bases2)

        return {
            "reads": [
                ("read_001", np.random.randn(400), 4000),
                ("read_002", np.random.randn(400), 4000),
            ],
            "aligned_reads": [aligned1, aligned2],
        }

    def test_create_plot_multiple_reads(self, multi_read_data):
        """Test plot creation with multiple reads"""
        strategy = EventAlignPlotStrategy(Theme.LIGHT)

        html, fig = strategy.create_plot(multi_read_data, {})

        assert isinstance(html, str)
        assert isinstance(fig, Plot)
        assert "2 reads" in fig.title.text

    def test_create_plot_multiple_reads_has_legend(self, multi_read_data):
        """Test that multiple reads produce a legend"""
        strategy = EventAlignPlotStrategy(Theme.LIGHT)

        _, fig = strategy.create_plot(multi_read_data, {})

        assert fig.legend is not None
        assert fig.legend.click_policy == "hide"

    def test_create_plot_multiple_reads_time_mode(self, multi_read_data):
        """Test multiple reads with time mode"""
        strategy = EventAlignPlotStrategy(Theme.LIGHT)

        options = {"show_dwell_time": True}

        _, fig = strategy.create_plot(multi_read_data, options)

        assert "Time" in fig.xaxis.axis_label


class TestPrivateMethods:
    """Tests for private methods"""

    def test_process_signal_no_normalization(self):
        """Test signal processing without normalization"""
        strategy = EventAlignPlotStrategy(Theme.LIGHT)

        signal = np.random.randn(1000)
        processed, seq_map = strategy._process_signal(
            signal=signal.copy(),
            normalization=NormalizationMethod.NONE,
            downsample=1,  # Explicitly set to 1 to test no downsampling
        )

        np.testing.assert_array_equal(processed, signal)
        assert seq_map is None  # No seq_to_sig_map provided

    def test_process_signal_with_normalization(self):
        """Test signal processing with normalization"""
        strategy = EventAlignPlotStrategy(Theme.LIGHT)

        # Set seed for reproducible test
        np.random.seed(42)
        signal = np.random.randn(1000) * 10 + 50
        processed, seq_map = strategy._process_signal(
            signal=signal,
            normalization=NormalizationMethod.ZNORM,
        )

        # Should be z-normalized
        assert abs(np.mean(processed)) < 0.1
        assert abs(np.std(processed) - 1.0) < 0.1
        assert seq_map is None  # No seq_to_sig_map provided


class TestEventAlignIntegration:
    """Integration tests for EventAlignPlotStrategy"""

    def test_complete_workflow_single_read(self):
        """Test complete workflow with single read"""
        strategy = EventAlignPlotStrategy(Theme.DARK)

        bases = [
            MockBaseAnnotation("A", 0),
            MockBaseAnnotation("C", 100),
            MockBaseAnnotation("G", 200),
            MockBaseAnnotation("T", 300),
        ]
        aligned = MockAlignedRead("test_read", bases)

        data = {
            "reads": [("test_read", np.random.randn(400), 4000)],
            "aligned_reads": [aligned],
        }

        options = {
            "normalization": NormalizationMethod.MEDIAN,
            "show_dwell_time": False,
            "show_labels": True,
        }

        html, fig = strategy.create_plot(data, options)

        assert isinstance(html, str)
        assert isinstance(fig, Plot)
        assert "test_read" in html

    def test_complete_workflow_multiple_reads(self):
        """Test complete workflow with multiple reads"""
        strategy = EventAlignPlotStrategy(Theme.LIGHT)

        bases1 = [MockBaseAnnotation(b, i * 50) for i, b in enumerate("ACGTACGT")]
        bases2 = [MockBaseAnnotation(b, i * 50) for i, b in enumerate("ACGTACGT")]

        aligned1 = MockAlignedRead("read_A", bases1)
        aligned2 = MockAlignedRead("read_B", bases2)

        data = {
            "reads": [
                ("read_A", np.random.randn(400), 4000),
                ("read_B", np.random.randn(400), 4000),
            ],
            "aligned_reads": [aligned1, aligned2],
        }

        options = {
            "normalization": NormalizationMethod.ZNORM,
            "show_dwell_time": True,
            "show_labels": True,
            "downsample": 2,
        }

        html, fig = strategy.create_plot(data, options)

        assert isinstance(html, str)
        assert isinstance(fig, Plot)
        assert "2 reads" in fig.title.text

    def test_reuse_strategy_multiple_plots(self):
        """Test that same strategy can create multiple plots"""
        strategy = EventAlignPlotStrategy(Theme.LIGHT)

        bases = [MockBaseAnnotation(b, i * 50) for i, b in enumerate("ACGT")]
        aligned1 = MockAlignedRead("read_1", bases)
        aligned2 = MockAlignedRead("read_2", bases)

        data1 = {
            "reads": [("read_1", np.random.randn(200), 4000)],
            "aligned_reads": [aligned1],
        }

        data2 = {
            "reads": [("read_2", np.random.randn(200), 4000)],
            "aligned_reads": [aligned2],
        }

        html1, fig1 = strategy.create_plot(data1, {})
        html2, fig2 = strategy.create_plot(data2, {})

        assert isinstance(html1, str)
        assert isinstance(html2, str)
        assert isinstance(fig1, Plot)
        assert isinstance(fig2, Plot)

    def test_long_sequence(self):
        """Test with longer sequence"""
        strategy = EventAlignPlotStrategy(Theme.LIGHT)

        # Create 100 bases
        bases = [MockBaseAnnotation("ACGT"[i % 4], i * 10) for i in range(100)]
        aligned = MockAlignedRead("long_read", bases)

        data = {
            "reads": [("long_read", np.random.randn(1000), 4000)],
            "aligned_reads": [aligned],
        }

        options = {"show_labels": False}  # Too many labels

        html, fig = strategy.create_plot(data, options)

        assert isinstance(html, str)
        assert isinstance(fig, Plot)
