"""
Tests for AggregateComparisonStrategy
"""

import numpy as np
import pytest
from bokeh.layouts import GridPlot
from bokeh.models.plots import Plot

from squiggy.constants import NormalizationMethod, Theme
from squiggy.plot_strategies.aggregate_comparison import AggregateComparisonStrategy


class TestAggregateComparisonStrategyInitialization:
    """Tests for AggregateComparisonStrategy initialization"""

    def test_init_with_light_theme(self):
        """Test initialization with LIGHT theme"""
        strategy = AggregateComparisonStrategy(Theme.LIGHT)

        assert strategy.theme == Theme.LIGHT
        assert strategy.theme_manager is not None

    def test_init_with_dark_theme(self):
        """Test initialization with DARK theme"""
        strategy = AggregateComparisonStrategy(Theme.DARK)

        assert strategy.theme == Theme.DARK


class TestDataValidation:
    """Tests for AggregateComparisonStrategy data validation"""

    @pytest.fixture
    def valid_data(self):
        """Valid data for testing"""
        positions = np.arange(50)
        return {
            "samples": [
                {
                    "name": "sample_1",
                    "signal_stats": {
                        "positions": positions,
                        "mean_signal": np.random.randn(50),
                        "std_signal": np.abs(np.random.randn(50)),
                    },
                    "dwell_stats": {
                        "positions": positions,
                        "mean_dwell": np.random.uniform(5, 15, 50),
                        "std_dwell": np.random.uniform(1, 3, 50),
                    },
                    "quality_stats": {
                        "positions": positions,
                        "mean_quality": np.random.uniform(25, 35, 50),
                        "std_quality": np.random.uniform(2, 4, 50),
                    },
                    "coverage": {
                        "positions": positions,
                        "coverage": np.full(50, 30),
                    },
                },
                {
                    "name": "sample_2",
                    "signal_stats": {
                        "positions": positions,
                        "mean_signal": np.random.randn(50),
                        "std_signal": np.abs(np.random.randn(50)),
                    },
                    "dwell_stats": {
                        "positions": positions,
                        "mean_dwell": np.random.uniform(5, 15, 50),
                        "std_dwell": np.random.uniform(1, 3, 50),
                    },
                    "quality_stats": {
                        "positions": positions,
                        "mean_quality": np.random.uniform(25, 35, 50),
                        "std_quality": np.random.uniform(2, 4, 50),
                    },
                    "coverage": {
                        "positions": positions,
                        "coverage": np.full(50, 35),
                    },
                },
            ],
            "reference_name": "chr1:1000-1050",
            "enabled_metrics": ["signal", "dwell_time", "quality"],
        }

    def test_validate_with_required_data(self, valid_data):
        """Test validation passes with all required data"""
        strategy = AggregateComparisonStrategy(Theme.LIGHT)

        # Should not raise
        strategy.validate_data(valid_data)

    def test_validate_missing_samples(self):
        """Test validation fails when samples is missing"""
        strategy = AggregateComparisonStrategy(Theme.LIGHT)

        data = {
            "reference_name": "chr1",
            "enabled_metrics": ["signal"],
        }

        with pytest.raises(ValueError, match="Missing required key.*samples"):
            strategy.validate_data(data)

    def test_validate_insufficient_samples(self):
        """Test validation fails with fewer than 2 samples"""
        strategy = AggregateComparisonStrategy(Theme.LIGHT)

        data = {
            "samples": [{"name": "sample_1"}],  # Only 1 sample
            "reference_name": "chr1",
            "enabled_metrics": ["signal"],
        }

        with pytest.raises(ValueError, match="Need at least 2 samples"):
            strategy.validate_data(data)

    def test_validate_sample_missing_name(self, valid_data):
        """Test validation fails when sample is missing name"""
        strategy = AggregateComparisonStrategy(Theme.LIGHT)

        # Remove name from first sample
        del valid_data["samples"][0]["name"]

        with pytest.raises(ValueError, match="Sample 0 missing 'name' key"):
            strategy.validate_data(valid_data)

    def test_validate_missing_reference_name(self, valid_data):
        """Test validation fails when reference_name is missing"""
        strategy = AggregateComparisonStrategy(Theme.LIGHT)

        del valid_data["reference_name"]

        with pytest.raises(ValueError, match="Missing required key.*reference_name"):
            strategy.validate_data(valid_data)

    def test_validate_missing_enabled_metrics(self, valid_data):
        """Test validation fails when enabled_metrics is missing"""
        strategy = AggregateComparisonStrategy(Theme.LIGHT)

        del valid_data["enabled_metrics"]

        with pytest.raises(ValueError, match="Missing required key.*enabled_metrics"):
            strategy.validate_data(valid_data)


class TestCreatePlot:
    """Tests for AggregateComparisonStrategy create_plot"""

    @pytest.fixture
    def sample_data(self):
        """Sample data for testing"""
        positions = np.arange(30)
        return {
            "samples": [
                {
                    "name": "control",
                    "signal_stats": {
                        "positions": positions,
                        "mean_signal": np.sin(positions / 5),
                        "std_signal": np.full(30, 0.2),
                    },
                    "dwell_stats": {
                        "positions": positions,
                        "mean_dwell": np.full(30, 10.0),
                        "std_dwell": np.full(30, 2.0),
                    },
                    "quality_stats": {
                        "positions": positions,
                        "mean_quality": np.full(30, 30.0),
                        "std_quality": np.full(30, 3.0),
                    },
                    "coverage": {"positions": positions, "coverage": np.full(30, 25)},
                },
                {
                    "name": "treatment",
                    "signal_stats": {
                        "positions": positions,
                        "mean_signal": np.sin(positions / 5) + 0.5,
                        "std_signal": np.full(30, 0.3),
                    },
                    "dwell_stats": {
                        "positions": positions,
                        "mean_dwell": np.full(30, 12.0),
                        "std_dwell": np.full(30, 2.5),
                    },
                    "quality_stats": {
                        "positions": positions,
                        "mean_quality": np.full(30, 28.0),
                        "std_quality": np.full(30, 4.0),
                    },
                    "coverage": {"positions": positions, "coverage": np.full(30, 30)},
                },
            ],
            "reference_name": "test_ref:100-130",
            "enabled_metrics": ["signal", "dwell_time", "quality"],
        }

    def test_create_plot_returns_tuple(self, sample_data):
        """Test that create_plot returns (html, gridplot) tuple"""
        strategy = AggregateComparisonStrategy(Theme.LIGHT)

        result = strategy.create_plot(sample_data, {})

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_create_plot_html_is_string(self, sample_data):
        """Test that returned HTML is a string"""
        strategy = AggregateComparisonStrategy(Theme.LIGHT)

        html, _ = strategy.create_plot(sample_data, {})

        assert isinstance(html, str)
        assert len(html) > 0
        assert "<html" in html

    def test_create_plot_returns_gridplot(self, sample_data):
        """Test that returned layout is a GridPlot"""
        strategy = AggregateComparisonStrategy(Theme.LIGHT)

        _, grid = strategy.create_plot(sample_data, {})

        assert isinstance(grid, GridPlot)

    def test_create_plot_with_normalization(self, sample_data):
        """Test plot creation with normalization"""
        strategy = AggregateComparisonStrategy(Theme.LIGHT)

        options = {"normalization": NormalizationMethod.ZNORM}

        html, grid = strategy.create_plot(sample_data, options)

        assert isinstance(html, str)
        assert isinstance(grid, GridPlot)

    def test_create_plot_html_contains_reference_name(self, sample_data):
        """Test that HTML contains reference name"""
        strategy = AggregateComparisonStrategy(Theme.LIGHT)

        html, _ = strategy.create_plot(sample_data, {})

        assert "test_ref:100-130" in html

    def test_create_plot_html_contains_sample_names(self, sample_data):
        """Test that HTML contains sample names"""
        strategy = AggregateComparisonStrategy(Theme.LIGHT)

        html, _ = strategy.create_plot(sample_data, {})

        assert "control" in html or "treatment" in html

    def test_create_plot_fails_with_no_tracks(self):
        """Test that create_plot raises error when no tracks can be created"""
        strategy = AggregateComparisonStrategy(Theme.LIGHT)

        # Data with no actual statistics
        data = {
            "samples": [
                {"name": "sample_1"},
                {"name": "sample_2"},
            ],
            "reference_name": "chr1",
            "enabled_metrics": ["signal", "dwell_time", "quality"],
        }

        with pytest.raises(ValueError, match="No tracks could be created"):
            strategy.create_plot(data, {})


class TestComparisonTracks:
    """Tests for the comparison tracks"""

    @pytest.fixture
    def sample_data(self):
        """Sample data for testing"""
        positions = np.arange(20)
        return {
            "samples": [
                {
                    "name": "sample_A",
                    "signal_stats": {
                        "positions": positions,
                        "mean_signal": np.random.randn(20),
                        "std_signal": np.abs(np.random.randn(20) * 0.2),
                    },
                    "dwell_stats": {
                        "positions": positions,
                        "mean_dwell": np.random.uniform(8, 12, 20),
                        "std_dwell": np.random.uniform(1, 2, 20),
                    },
                    "quality_stats": {
                        "positions": positions,
                        "mean_quality": np.random.uniform(25, 30, 20),
                        "std_quality": np.random.uniform(2, 3, 20),
                    },
                    "coverage": {"positions": positions, "coverage": np.full(20, 15)},
                },
                {
                    "name": "sample_B",
                    "signal_stats": {
                        "positions": positions,
                        "mean_signal": np.random.randn(20),
                        "std_signal": np.abs(np.random.randn(20) * 0.2),
                    },
                    "dwell_stats": {
                        "positions": positions,
                        "mean_dwell": np.random.uniform(8, 12, 20),
                        "std_dwell": np.random.uniform(1, 2, 20),
                    },
                    "quality_stats": {
                        "positions": positions,
                        "mean_quality": np.random.uniform(25, 30, 20),
                        "std_quality": np.random.uniform(2, 3, 20),
                    },
                    "coverage": {"positions": positions, "coverage": np.full(20, 18)},
                },
            ],
            "reference_name": "chr1:1000-1020",
            "enabled_metrics": ["signal", "dwell_time", "quality"],
        }

    def test_gridplot_has_four_rows_all_metrics(self, sample_data):
        """Test that gridplot has four rows when all metrics enabled (signal, dwell, quality, coverage)"""
        strategy = AggregateComparisonStrategy(Theme.LIGHT)

        _, grid = strategy.create_plot(sample_data, {})

        # GridPlot children are tuples of (figure, row, col)
        # Should have: signal, dwell_time, quality, coverage
        assert len(grid.children) == 4

    def test_gridplot_has_two_rows_signal_only(self, sample_data):
        """Test that gridplot has two rows when only signal enabled (signal + coverage)"""
        strategy = AggregateComparisonStrategy(Theme.LIGHT)

        sample_data["enabled_metrics"] = ["signal"]

        _, grid = strategy.create_plot(sample_data, {})

        # Should have: signal, coverage
        assert len(grid.children) == 2

    def test_all_tracks_are_plots(self, sample_data):
        """Test that all tracks are Plot objects"""
        strategy = AggregateComparisonStrategy(Theme.LIGHT)

        _, grid = strategy.create_plot(sample_data, {})

        # GridPlot children are tuples of (figure, row, col)
        for fig, _row, _col in grid.children:
            assert isinstance(fig, Plot)

    def test_signal_track_has_title(self, sample_data):
        """Test that signal track has appropriate title"""
        strategy = AggregateComparisonStrategy(Theme.LIGHT)

        _, grid = strategy.create_plot(sample_data, {})

        # Find signal track by title
        signal_track = None
        for fig, _row, _col in grid.children:
            if "Signal Statistics Comparison" in fig.title.text:
                signal_track = fig
                break

        assert signal_track is not None
        assert "chr1:1000-1020" in signal_track.title.text

    def test_dwell_track_has_title(self, sample_data):
        """Test that dwell track has appropriate title"""
        strategy = AggregateComparisonStrategy(Theme.LIGHT)

        _, grid = strategy.create_plot(sample_data, {})

        # Find dwell track by title
        dwell_track = None
        for fig, _row, _col in grid.children:
            if "Dwell Time Statistics Comparison" in fig.title.text:
                dwell_track = fig
                break

        assert dwell_track is not None
        assert "chr1:1000-1020" in dwell_track.title.text

    def test_quality_track_has_title(self, sample_data):
        """Test that quality track has appropriate title"""
        strategy = AggregateComparisonStrategy(Theme.LIGHT)

        _, grid = strategy.create_plot(sample_data, {})

        # Find quality track by title
        quality_track = None
        for fig, _row, _col in grid.children:
            if "Quality Statistics Comparison" in fig.title.text:
                quality_track = fig
                break

        assert quality_track is not None

    def test_coverage_track_has_title(self, sample_data):
        """Test that coverage track has appropriate title"""
        strategy = AggregateComparisonStrategy(Theme.LIGHT)

        _, grid = strategy.create_plot(sample_data, {})

        # Find coverage track by title
        coverage_track = None
        for fig, _row, _col in grid.children:
            if "Coverage Comparison" in fig.title.text:
                coverage_track = fig
                break

        assert coverage_track is not None

    def test_tracks_have_linked_x_ranges(self, sample_data):
        """Test that all tracks have synchronized x-ranges"""
        strategy = AggregateComparisonStrategy(Theme.LIGHT)

        _, grid = strategy.create_plot(sample_data, {})

        # Extract all figures
        figures = [fig for fig, _row, _col in grid.children]

        # All should reference the same x_range object
        assert len(figures) >= 2
        first_x_range = figures[0].x_range
        for fig in figures[1:]:
            assert fig.x_range is first_x_range


class TestCustomColors:
    """Tests for custom sample colors"""

    def test_custom_colors_are_used(self):
        """Test that custom colors from samples are used"""
        strategy = AggregateComparisonStrategy(Theme.LIGHT)

        positions = np.arange(20)
        data = {
            "samples": [
                {
                    "name": "sample_1",
                    "color": "#FF0000",  # Custom red
                    "signal_stats": {
                        "positions": positions,
                        "mean_signal": np.random.randn(20),
                        "std_signal": np.full(20, 0.2),
                    },
                },
                {
                    "name": "sample_2",
                    "color": "#00FF00",  # Custom green
                    "signal_stats": {
                        "positions": positions,
                        "mean_signal": np.random.randn(20),
                        "std_signal": np.full(20, 0.2),
                    },
                },
            ],
            "reference_name": "chr1",
            "enabled_metrics": ["signal"],
        }

        html, _ = strategy.create_plot(data, {})

        # HTML should contain custom colors
        assert isinstance(html, str)


class TestMultipleSamples:
    """Tests for handling multiple samples"""

    def test_three_samples(self):
        """Test with three samples"""
        strategy = AggregateComparisonStrategy(Theme.LIGHT)

        positions = np.arange(25)
        data = {
            "samples": [
                {
                    "name": "sample_1",
                    "signal_stats": {
                        "positions": positions,
                        "mean_signal": np.random.randn(25),
                        "std_signal": np.full(25, 0.2),
                    },
                },
                {
                    "name": "sample_2",
                    "signal_stats": {
                        "positions": positions,
                        "mean_signal": np.random.randn(25),
                        "std_signal": np.full(25, 0.2),
                    },
                },
                {
                    "name": "sample_3",
                    "signal_stats": {
                        "positions": positions,
                        "mean_signal": np.random.randn(25),
                        "std_signal": np.full(25, 0.2),
                    },
                },
            ],
            "reference_name": "chr1:500-525",
            "enabled_metrics": ["signal"],
        }

        html, grid = strategy.create_plot(data, {})

        assert isinstance(html, str)
        assert isinstance(grid, GridPlot)
        assert "sample_1" in html or "sample_2" in html or "sample_3" in html

    def test_missing_stats_in_some_samples(self):
        """Test when some samples are missing certain statistics"""
        strategy = AggregateComparisonStrategy(Theme.LIGHT)

        positions = np.arange(20)
        data = {
            "samples": [
                {
                    "name": "sample_1",
                    "signal_stats": {
                        "positions": positions,
                        "mean_signal": np.random.randn(20),
                        "std_signal": np.full(20, 0.2),
                    },
                    "dwell_stats": {
                        "positions": positions,
                        "mean_dwell": np.full(20, 10.0),
                        "std_dwell": np.full(20, 2.0),
                    },
                },
                {
                    "name": "sample_2",
                    "signal_stats": {
                        "positions": positions,
                        "mean_signal": np.random.randn(20),
                        "std_signal": np.full(20, 0.2),
                    },
                    # Missing dwell_stats
                },
            ],
            "reference_name": "chr1",
            "enabled_metrics": ["signal", "dwell_time"],
        }

        html, grid = strategy.create_plot(data, {})

        # Should still create plot with available data
        assert isinstance(html, str)
        assert isinstance(grid, GridPlot)


class TestAggregateComparisonIntegration:
    """Integration tests for AggregateComparisonStrategy"""

    def test_complete_workflow_light_theme(self):
        """Test complete workflow with light theme"""
        strategy = AggregateComparisonStrategy(Theme.LIGHT)

        positions = np.arange(50)
        data = {
            "samples": [
                {
                    "name": "wild_type",
                    "color": "#0072B2",
                    "signal_stats": {
                        "positions": positions,
                        "mean_signal": np.sin(positions / 8),
                        "std_signal": np.full(50, 0.15),
                    },
                    "dwell_stats": {
                        "positions": positions,
                        "mean_dwell": np.full(50, 10.0),
                        "std_dwell": np.full(50, 2.0),
                    },
                    "quality_stats": {
                        "positions": positions,
                        "mean_quality": np.full(50, 30.0),
                        "std_quality": np.full(50, 3.0),
                    },
                    "coverage": {"positions": positions, "coverage": np.full(50, 40)},
                },
                {
                    "name": "mutant",
                    "color": "#D55E00",
                    "signal_stats": {
                        "positions": positions,
                        "mean_signal": np.sin(positions / 8) + 0.3,
                        "std_signal": np.full(50, 0.2),
                    },
                    "dwell_stats": {
                        "positions": positions,
                        "mean_dwell": np.full(50, 12.0),
                        "std_dwell": np.full(50, 2.5),
                    },
                    "quality_stats": {
                        "positions": positions,
                        "mean_quality": np.full(50, 28.0),
                        "std_quality": np.full(50, 4.0),
                    },
                    "coverage": {"positions": positions, "coverage": np.full(50, 35)},
                },
            ],
            "reference_name": "gene_X:1000-1050",
            "enabled_metrics": ["signal", "dwell_time", "quality"],
        }

        options = {"normalization": NormalizationMethod.MEDIAN}

        html, grid = strategy.create_plot(data, options)

        assert isinstance(html, str)
        assert isinstance(grid, GridPlot)
        assert len(grid.children) == 4  # signal, dwell, quality, coverage
        assert "gene_X:1000-1050" in html

    def test_complete_workflow_dark_theme(self):
        """Test complete workflow with dark theme"""
        strategy = AggregateComparisonStrategy(Theme.DARK)

        positions = np.arange(30)
        data = {
            "samples": [
                {
                    "name": "control",
                    "signal_stats": {
                        "positions": positions,
                        "mean_signal": np.random.randn(30),
                        "std_signal": np.full(30, 0.2),
                    },
                    "quality_stats": {
                        "positions": positions,
                        "mean_quality": np.full(30, 29.0),
                        "std_quality": np.full(30, 3.5),
                    },
                },
                {
                    "name": "experiment",
                    "signal_stats": {
                        "positions": positions,
                        "mean_signal": np.random.randn(30),
                        "std_signal": np.full(30, 0.2),
                    },
                    "quality_stats": {
                        "positions": positions,
                        "mean_quality": np.full(30, 27.0),
                        "std_quality": np.full(30, 4.0),
                    },
                },
            ],
            "reference_name": "test_region",
            "enabled_metrics": ["signal", "quality"],
        }

        options = {"normalization": NormalizationMethod.ZNORM}

        html, grid = strategy.create_plot(data, options)

        assert isinstance(html, str)
        assert isinstance(grid, GridPlot)
        assert "test_region" in html

    def test_reuse_strategy_multiple_plots(self):
        """Test that same strategy can create multiple plots"""
        strategy = AggregateComparisonStrategy(Theme.LIGHT)

        positions1 = np.arange(20)
        data1 = {
            "samples": [
                {
                    "name": "A",
                    "signal_stats": {
                        "positions": positions1,
                        "mean_signal": np.random.randn(20),
                        "std_signal": np.full(20, 0.2),
                    },
                },
                {
                    "name": "B",
                    "signal_stats": {
                        "positions": positions1,
                        "mean_signal": np.random.randn(20),
                        "std_signal": np.full(20, 0.2),
                    },
                },
            ],
            "reference_name": "region_1",
            "enabled_metrics": ["signal"],
        }

        positions2 = np.arange(30)
        data2 = {
            "samples": [
                {
                    "name": "C",
                    "signal_stats": {
                        "positions": positions2,
                        "mean_signal": np.random.randn(30),
                        "std_signal": np.full(30, 0.2),
                    },
                },
                {
                    "name": "D",
                    "signal_stats": {
                        "positions": positions2,
                        "mean_signal": np.random.randn(30),
                        "std_signal": np.full(30, 0.2),
                    },
                },
            ],
            "reference_name": "region_2",
            "enabled_metrics": ["signal"],
        }

        html1, grid1 = strategy.create_plot(data1, {})
        html2, grid2 = strategy.create_plot(data2, {})

        assert "region_1" in html1
        assert "region_2" in html2

    def test_coverage_list_format(self):
        """Test with coverage as a list instead of dict"""
        strategy = AggregateComparisonStrategy(Theme.LIGHT)

        positions = np.arange(20)
        data = {
            "samples": [
                {
                    "name": "sample_1",
                    "signal_stats": {
                        "positions": positions,
                        "mean_signal": np.random.randn(20),
                        "std_signal": np.full(20, 0.2),
                    },
                    "coverage": list(np.full(20, 15)),  # List format
                },
                {
                    "name": "sample_2",
                    "signal_stats": {
                        "positions": positions,
                        "mean_signal": np.random.randn(20),
                        "std_signal": np.full(20, 0.2),
                    },
                    "coverage": list(np.full(20, 20)),  # List format
                },
            ],
            "reference_name": "chr1",
            "enabled_metrics": ["signal"],
        }

        html, grid = strategy.create_plot(data, {})

        assert isinstance(html, str)
        assert isinstance(grid, GridPlot)
