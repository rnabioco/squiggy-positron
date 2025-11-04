"""Tests for Phase 3 - Delta plotting functionality"""

import numpy as np
import pytest


class TestDeltaPlotStrategy:
    """Tests for DeltaPlotStrategy class"""

    def test_delta_plot_strategy_initialization(self):
        """Test DeltaPlotStrategy initialization"""
        from squiggy.constants import Theme
        from squiggy.plot_strategies.delta import DeltaPlotStrategy

        strategy = DeltaPlotStrategy(Theme.LIGHT)
        assert strategy.theme == Theme.LIGHT

    def test_delta_plot_strategy_validation_required_data(self):
        """Test that validate_data checks for required keys"""
        from squiggy.constants import Theme
        from squiggy.plot_strategies.delta import DeltaPlotStrategy

        strategy = DeltaPlotStrategy(Theme.LIGHT)

        # Missing required keys
        data = {
            "delta_mean_signal": np.array([1.0, 2.0]),
        }

        with pytest.raises(ValueError, match="Missing required delta data"):
            strategy.validate_data(data)

    def test_delta_plot_strategy_validation_complete(self):
        """Test validation with complete data"""
        from squiggy.constants import Theme
        from squiggy.plot_strategies.delta import DeltaPlotStrategy

        strategy = DeltaPlotStrategy(Theme.LIGHT)

        positions = np.array([0, 1, 2, 3])
        delta_mean = np.array([1.0, 2.0, 3.0, 4.0])
        delta_std = np.array([0.1, 0.2, 0.1, 0.2])

        data = {
            "positions": positions,
            "delta_mean_signal": delta_mean,
            "delta_std_signal": delta_std,
            "sample_a_name": "sample_a",
            "sample_b_name": "sample_b",
            "sample_a_coverage": [10, 10, 10, 10],
            "sample_b_coverage": [10, 10, 10, 10],
        }

        # Should not raise
        strategy.validate_data(data)

    def test_delta_plot_strategy_create_plot(self):
        """Test that create_plot generates HTML and figure"""
        from squiggy.constants import NormalizationMethod, Theme
        from squiggy.plot_strategies.delta import DeltaPlotStrategy

        strategy = DeltaPlotStrategy(Theme.LIGHT)

        positions = np.array([0, 1, 2, 3])
        delta_mean = np.array([1.0, 2.0, 3.0, 4.0])
        delta_std = np.array([0.1, 0.2, 0.1, 0.2])

        data = {
            "positions": positions,
            "delta_mean_signal": delta_mean,
            "delta_std_signal": delta_std,
            "sample_a_name": "sample_a",
            "sample_b_name": "sample_b",
            "sample_a_coverage": [10, 10, 10, 10],
            "sample_b_coverage": [12, 11, 13, 10],
        }

        options = {"normalization": NormalizationMethod.NONE}

        html, fig = strategy.create_plot(data, options)

        # Verify HTML is generated
        assert isinstance(html, str)
        assert "<html>" in html or "<!DOCTYPE" in html
        assert "bokeh" in html.lower()

        # Verify figure object is returned
        assert fig is not None

    def test_delta_plot_strategy_with_downsampling(self):
        """Test delta plot with downsampling"""
        from squiggy.constants import NormalizationMethod, Theme
        from squiggy.plot_strategies.delta import DeltaPlotStrategy

        strategy = DeltaPlotStrategy(Theme.LIGHT)

        # Create larger data
        positions = np.array(range(100))
        delta_mean = np.random.randn(100)
        delta_std = np.abs(np.random.randn(100)) + 0.1

        data = {
            "positions": positions,
            "delta_mean_signal": delta_mean,
            "delta_std_signal": delta_std,
            "sample_a_name": "sample_a",
            "sample_b_name": "sample_b",
            "sample_a_coverage": [10] * 100,
            "sample_b_coverage": [10] * 100,
        }

        options = {
            "normalization": NormalizationMethod.NONE,
            "downsample": 5,
        }

        html, fig = strategy.create_plot(data, options)

        # Should succeed with downsampling
        assert isinstance(html, str)
        assert fig is not None

    def test_delta_plot_color_by_direction(self):
        """Test color assignment based on delta direction"""
        from squiggy.constants import (
            DELTA_NEGATIVE_COLOR,
            DELTA_NEUTRAL_COLOR,
            DELTA_POSITIVE_COLOR,
        )
        from squiggy.plot_strategies.delta import DeltaPlotStrategy

        deltas = np.array([1.0, -2.0, 0.05, -0.05, 5.0])
        colors = DeltaPlotStrategy._color_by_direction(deltas)

        assert len(colors) == 5
        assert colors[0] == DELTA_POSITIVE_COLOR  # 1.0 > 0.1
        assert colors[1] == DELTA_NEGATIVE_COLOR  # -2.0 < -0.1
        assert colors[2] == DELTA_NEUTRAL_COLOR  # 0.05, between -0.1 and 0.1
        assert colors[3] == DELTA_NEUTRAL_COLOR  # -0.05, between -0.1 and 0.1
        assert colors[4] == DELTA_POSITIVE_COLOR  # 5.0 > 0.1

    def test_delta_plot_dark_theme(self):
        """Test delta plot with dark theme"""
        from squiggy.constants import NormalizationMethod, Theme
        from squiggy.plot_strategies.delta import DeltaPlotStrategy

        strategy = DeltaPlotStrategy(Theme.DARK)

        positions = np.array([0, 1, 2])
        data = {
            "positions": positions,
            "delta_mean_signal": np.array([1.0, 2.0, 3.0]),
            "delta_std_signal": np.array([0.1, 0.1, 0.1]),
            "sample_a_name": "v4.2",
            "sample_b_name": "v5.0",
            "sample_a_coverage": [10, 10, 10],
            "sample_b_coverage": [10, 10, 10],
        }

        options = {"normalization": NormalizationMethod.NONE}

        html, fig = strategy.create_plot(data, options)

        assert isinstance(html, str)
        assert fig is not None
        assert "v4.2" in html or "v5.0" in html


class TestDeltaPlotFactory:
    """Tests for delta plot factory integration"""

    def test_delta_plot_mode_in_factory(self):
        """Test that DELTA mode is available in plot factory"""
        from squiggy.constants import PlotMode, Theme
        from squiggy.plot_factory import create_plot_strategy
        from squiggy.plot_strategies.delta import DeltaPlotStrategy

        strategy = create_plot_strategy(PlotMode.DELTA, Theme.LIGHT)

        assert isinstance(strategy, DeltaPlotStrategy)

    def test_delta_plot_mode_enum_exists(self):
        """Test that DELTA mode exists in PlotMode enum"""
        from squiggy.constants import PlotMode

        assert hasattr(PlotMode, "DELTA")
        assert PlotMode.DELTA.value == "delta"

    def test_create_delta_strategy_light_theme(self):
        """Test creating delta strategy with light theme"""
        from squiggy.constants import PlotMode, Theme
        from squiggy.plot_factory import create_plot_strategy

        strategy = create_plot_strategy(PlotMode.DELTA, Theme.LIGHT)

        assert strategy.theme == Theme.LIGHT

    def test_create_delta_strategy_dark_theme(self):
        """Test creating delta strategy with dark theme"""
        from squiggy.constants import PlotMode, Theme
        from squiggy.plot_factory import create_plot_strategy

        strategy = create_plot_strategy(PlotMode.DELTA, Theme.DARK)

        assert strategy.theme == Theme.DARK


class TestDeltaConstants:
    """Tests for delta-related constants"""

    def test_delta_color_constants_exist(self):
        """Test that delta color constants are defined"""
        from squiggy.constants import (
            DELTA_NEGATIVE_COLOR,
            DELTA_NEUTRAL_COLOR,
            DELTA_POSITIVE_COLOR,
            DELTA_ZERO_LINE_COLOR,
        )

        # Verify they are valid hex colors
        assert DELTA_POSITIVE_COLOR.startswith("#")
        assert DELTA_NEGATIVE_COLOR.startswith("#")
        assert DELTA_NEUTRAL_COLOR.startswith("#")
        assert DELTA_ZERO_LINE_COLOR.startswith("#")

    def test_delta_dimension_constants(self):
        """Test that delta dimension constants are defined"""
        from squiggy.constants import DELTA_SIGNAL_HEIGHT, DELTA_STATS_HEIGHT

        assert DELTA_SIGNAL_HEIGHT > 0
        assert DELTA_STATS_HEIGHT > 0
        assert isinstance(DELTA_SIGNAL_HEIGHT, int)
        assert isinstance(DELTA_STATS_HEIGHT, int)

    def test_delta_alpha_constant(self):
        """Test that delta alpha constant is valid"""
        from squiggy.constants import DELTA_BAND_ALPHA

        assert 0 <= DELTA_BAND_ALPHA <= 1

    def test_delta_line_width_constant(self):
        """Test that delta line width constant is valid"""
        from squiggy.constants import DELTA_LINE_WIDTH

        assert DELTA_LINE_WIDTH > 0


class TestDeltaIntegration:
    """Integration tests for delta plotting"""

    def test_delta_plot_with_sample_comparison(self, sample_pod5_file):
        """Test delta plotting with actual samples"""
        import numpy as np

        from squiggy import compare_samples, load_sample
        from squiggy.constants import NormalizationMethod, PlotMode, Theme
        from squiggy.plot_factory import create_plot_strategy

        # Load two samples
        load_sample("a", str(sample_pod5_file))
        load_sample("b", str(sample_pod5_file))

        # Get comparison data
        compare_samples(["a", "b"])

        # Create minimal delta data
        positions = np.arange(10)
        delta_mean = np.random.randn(10)
        delta_std = np.abs(np.random.randn(10)) + 0.1

        data = {
            "positions": positions,
            "delta_mean_signal": delta_mean,
            "delta_std_signal": delta_std,
            "sample_a_name": "a",
            "sample_b_name": "b",
            "sample_a_coverage": [10] * 10,
            "sample_b_coverage": [10] * 10,
        }

        options = {"normalization": NormalizationMethod.NONE}

        strategy = create_plot_strategy(PlotMode.DELTA, Theme.LIGHT)
        html, fig = strategy.create_plot(data, options)

        assert html is not None
        assert len(html) > 0
        assert fig is not None

    def test_plot_delta_comparison_function_exists(self):
        """Test that plot_delta_comparison function is exported"""
        from squiggy import plot_delta_comparison

        assert callable(plot_delta_comparison)

    def test_plot_delta_comparison_insufficient_samples(self, sample_pod5_file):
        """Test that plot_delta_comparison requires at least 2 samples"""
        from squiggy import load_sample, plot_delta_comparison

        load_sample("only_one", str(sample_pod5_file))

        with pytest.raises(ValueError, match="at least 2 samples"):
            plot_delta_comparison(["only_one"])

    def test_plot_delta_comparison_nonexistent_sample(self):
        """Test plot_delta_comparison with nonexistent sample"""
        from squiggy import plot_delta_comparison

        with pytest.raises(ValueError, match="not found"):
            plot_delta_comparison(["nonexistent_a", "nonexistent_b"])

    def test_plot_delta_comparison_requires_bam(self, sample_pod5_file):
        """Test that plot_delta_comparison requires BAM files"""
        from squiggy import load_sample, plot_delta_comparison

        # Load samples without BAM
        load_sample("sample_a", str(sample_pod5_file))
        load_sample("sample_b", str(sample_pod5_file))

        # Should fail because no BAM files loaded
        with pytest.raises(ValueError, match="BAM files"):
            plot_delta_comparison(["sample_a", "sample_b"])

    def test_plot_delta_comparison_with_bam(self, sample_pod5_file, sample_bam_file):
        """Test plot_delta_comparison with proper BAM files"""
        from squiggy import load_sample, plot_delta_comparison

        # Load samples with BAM files (bam_path is positional parameter)
        load_sample("sample_a", str(sample_pod5_file), str(sample_bam_file))
        load_sample("sample_b", str(sample_pod5_file), str(sample_bam_file))

        # Should succeed with BAM files
        html = plot_delta_comparison(["sample_a", "sample_b"], normalization="ZNORM")

        assert html is not None
        assert len(html) > 0
        assert "Delta Signal" in html
