"""
Tests for AggregatePlotStrategy
"""

import numpy as np
import pytest
from bokeh.layouts import GridPlot
from bokeh.models.plots import Plot

from squiggy.constants import NormalizationMethod, Theme
from squiggy.plot_strategies.aggregate import AggregatePlotStrategy


class TestAggregatePlotStrategyInitialization:
    """Tests for AggregatePlotStrategy initialization"""

    def test_init_with_light_theme(self):
        """Test initialization with LIGHT theme"""
        strategy = AggregatePlotStrategy(Theme.LIGHT)

        assert strategy.theme == Theme.LIGHT
        assert strategy.theme_manager is not None

    def test_init_with_dark_theme(self):
        """Test initialization with DARK theme"""
        strategy = AggregatePlotStrategy(Theme.DARK)

        assert strategy.theme == Theme.DARK


class TestDataValidation:
    """Tests for AggregatePlotStrategy data validation"""

    @pytest.fixture
    def valid_data(self):
        """Valid data for testing"""
        positions = np.arange(100)
        return {
            "aggregate_stats": {
                "positions": positions,
                "mean_signal": np.random.randn(100),
                "std_signal": np.abs(np.random.randn(100)),
                "median_signal": np.random.randn(100),
                "coverage": np.full(100, 50),
            },
            "pileup_stats": {
                "positions": positions,
                "counts": {
                    pos: {"A": 10, "C": 15, "G": 12, "T": 13} for pos in positions
                },
                "reference_bases": {pos: "ACGT"[pos % 4] for pos in positions},
            },
            "quality_stats": {
                "positions": positions,
                "mean_quality": np.random.uniform(20, 40, 100),
                "std_quality": np.random.uniform(1, 5, 100),
            },
            "reference_name": "chr1:1000-1100",
            "num_reads": 50,
        }

    def test_validate_with_required_data(self, valid_data):
        """Test validation passes with all required data"""
        strategy = AggregatePlotStrategy(Theme.LIGHT)

        # Should not raise
        strategy.validate_data(valid_data)

    def test_validate_missing_aggregate_stats(self):
        """Test validation fails when aggregate_stats is missing"""
        strategy = AggregatePlotStrategy(Theme.LIGHT)

        data = {
            "pileup_stats": {},
            "quality_stats": {},
            "reference_name": "chr1",
            "num_reads": 50,
        }

        with pytest.raises(ValueError, match="Missing required data.*aggregate_stats"):
            strategy.validate_data(data)

    def test_validate_missing_pileup_stats(self):
        """Test validation fails when pileup_stats is missing"""
        strategy = AggregatePlotStrategy(Theme.LIGHT)

        data = {
            "aggregate_stats": {},
            "quality_stats": {},
            "reference_name": "chr1",
            "num_reads": 50,
        }

        with pytest.raises(ValueError, match="Missing required data.*pileup_stats"):
            strategy.validate_data(data)

    def test_validate_missing_quality_stats(self):
        """Test validation fails when quality_stats is missing"""
        strategy = AggregatePlotStrategy(Theme.LIGHT)

        data = {
            "aggregate_stats": {},
            "pileup_stats": {},
            "reference_name": "chr1",
            "num_reads": 50,
        }

        with pytest.raises(ValueError, match="Missing required data.*quality_stats"):
            strategy.validate_data(data)

    def test_validate_missing_reference_name(self):
        """Test validation fails when reference_name is missing"""
        strategy = AggregatePlotStrategy(Theme.LIGHT)

        data = {
            "aggregate_stats": {},
            "pileup_stats": {},
            "quality_stats": {},
            "num_reads": 50,
        }

        with pytest.raises(ValueError, match="Missing required data.*reference_name"):
            strategy.validate_data(data)

    def test_validate_missing_num_reads(self):
        """Test validation fails when num_reads is missing"""
        strategy = AggregatePlotStrategy(Theme.LIGHT)

        data = {
            "aggregate_stats": {},
            "pileup_stats": {},
            "quality_stats": {},
            "reference_name": "chr1",
        }

        with pytest.raises(ValueError, match="Missing required data.*num_reads"):
            strategy.validate_data(data)

    def test_validate_aggregate_stats_missing_positions(self, valid_data):
        """Test validation fails when aggregate_stats missing positions"""
        strategy = AggregatePlotStrategy(Theme.LIGHT)

        del valid_data["aggregate_stats"]["positions"]

        with pytest.raises(ValueError, match="aggregate_stats missing keys.*positions"):
            strategy.validate_data(valid_data)

    def test_validate_aggregate_stats_missing_mean_signal(self, valid_data):
        """Test validation fails when aggregate_stats missing mean_signal"""
        strategy = AggregatePlotStrategy(Theme.LIGHT)

        del valid_data["aggregate_stats"]["mean_signal"]

        with pytest.raises(
            ValueError, match="aggregate_stats missing keys.*mean_signal"
        ):
            strategy.validate_data(valid_data)

    def test_validate_aggregate_stats_missing_std_signal(self, valid_data):
        """Test validation fails when aggregate_stats missing std_signal"""
        strategy = AggregatePlotStrategy(Theme.LIGHT)

        del valid_data["aggregate_stats"]["std_signal"]

        with pytest.raises(
            ValueError, match="aggregate_stats missing keys.*std_signal"
        ):
            strategy.validate_data(valid_data)

    def test_validate_aggregate_stats_missing_coverage(self, valid_data):
        """Test validation fails when aggregate_stats missing coverage"""
        strategy = AggregatePlotStrategy(Theme.LIGHT)

        del valid_data["aggregate_stats"]["coverage"]

        with pytest.raises(ValueError, match="aggregate_stats missing keys.*coverage"):
            strategy.validate_data(valid_data)

    def test_validate_pileup_stats_missing_positions(self, valid_data):
        """Test validation fails when pileup_stats missing positions"""
        strategy = AggregatePlotStrategy(Theme.LIGHT)

        del valid_data["pileup_stats"]["positions"]

        with pytest.raises(ValueError, match="pileup_stats missing keys.*positions"):
            strategy.validate_data(valid_data)

    def test_validate_pileup_stats_missing_counts(self, valid_data):
        """Test validation fails when pileup_stats missing counts"""
        strategy = AggregatePlotStrategy(Theme.LIGHT)

        del valid_data["pileup_stats"]["counts"]

        with pytest.raises(ValueError, match="pileup_stats missing keys.*counts"):
            strategy.validate_data(valid_data)

    def test_validate_quality_stats_missing_positions(self, valid_data):
        """Test validation fails when quality_stats missing positions"""
        strategy = AggregatePlotStrategy(Theme.LIGHT)

        del valid_data["quality_stats"]["positions"]

        with pytest.raises(ValueError, match="quality_stats missing keys.*positions"):
            strategy.validate_data(valid_data)

    def test_validate_quality_stats_missing_mean_quality(self, valid_data):
        """Test validation fails when quality_stats missing mean_quality"""
        strategy = AggregatePlotStrategy(Theme.LIGHT)

        del valid_data["quality_stats"]["mean_quality"]

        with pytest.raises(
            ValueError, match="quality_stats missing keys.*mean_quality"
        ):
            strategy.validate_data(valid_data)

    def test_validate_quality_stats_missing_std_quality(self, valid_data):
        """Test validation fails when quality_stats missing std_quality"""
        strategy = AggregatePlotStrategy(Theme.LIGHT)

        del valid_data["quality_stats"]["std_quality"]

        with pytest.raises(ValueError, match="quality_stats missing keys.*std_quality"):
            strategy.validate_data(valid_data)


class TestCreatePlot:
    """Tests for AggregatePlotStrategy create_plot"""

    @pytest.fixture
    def sample_data(self):
        """Sample data for testing"""
        positions = np.arange(50)
        return {
            "aggregate_stats": {
                "positions": positions,
                "mean_signal": np.random.randn(50),
                "std_signal": np.abs(np.random.randn(50)),
                "median_signal": np.random.randn(50),
                "coverage": np.full(50, 30),
            },
            "pileup_stats": {
                "positions": positions,
                "counts": {pos: {"A": 5, "C": 10, "G": 8, "T": 7} for pos in positions},
            },
            "quality_stats": {
                "positions": positions,
                "mean_quality": np.random.uniform(25, 35, 50),
                "std_quality": np.random.uniform(2, 4, 50),
            },
            "reference_name": "test_ref:100-150",
            "num_reads": 30,
        }

    def test_create_plot_returns_tuple(self, sample_data):
        """Test that create_plot returns (html, gridplot) tuple"""
        strategy = AggregatePlotStrategy(Theme.LIGHT)

        result = strategy.create_plot(sample_data, {})

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_create_plot_html_is_string(self, sample_data):
        """Test that returned HTML is a string"""
        strategy = AggregatePlotStrategy(Theme.LIGHT)

        html, _ = strategy.create_plot(sample_data, {})

        assert isinstance(html, str)
        assert len(html) > 0
        assert "<html" in html

    def test_create_plot_returns_gridplot(self, sample_data):
        """Test that returned layout is a GridPlot"""
        strategy = AggregatePlotStrategy(Theme.LIGHT)

        _, grid = strategy.create_plot(sample_data, {})

        assert isinstance(grid, GridPlot)

    def test_create_plot_with_normalization(self, sample_data):
        """Test plot creation with normalization"""
        strategy = AggregatePlotStrategy(Theme.LIGHT)

        options = {"normalization": NormalizationMethod.ZNORM}

        html, grid = strategy.create_plot(sample_data, options)

        assert isinstance(html, str)
        assert isinstance(grid, GridPlot)
        assert "znorm" in html.lower()

    def test_create_plot_default_normalization(self, sample_data):
        """Test plot creation with default normalization (NONE)"""
        strategy = AggregatePlotStrategy(Theme.LIGHT)

        html, grid = strategy.create_plot(sample_data, {})

        assert isinstance(html, str)
        assert "none" in html.lower() or "raw" in html.lower()

    def test_create_plot_html_contains_reference_name(self, sample_data):
        """Test that HTML contains reference name"""
        strategy = AggregatePlotStrategy(Theme.LIGHT)

        html, _ = strategy.create_plot(sample_data, {})

        assert "test_ref:100-150" in html

    def test_create_plot_html_contains_num_reads(self, sample_data):
        """Test that HTML contains number of reads"""
        strategy = AggregatePlotStrategy(Theme.LIGHT)

        html, _ = strategy.create_plot(sample_data, {})

        assert "30 reads" in html or "30" in html


class TestAggregateTracks:
    """Tests for the three aggregate tracks"""

    @pytest.fixture
    def sample_data(self):
        """Sample data for testing"""
        positions = np.arange(20)
        return {
            "aggregate_stats": {
                "positions": positions,
                "mean_signal": np.random.randn(20),
                "std_signal": np.abs(np.random.randn(20)),
                "median_signal": np.random.randn(20),
                "coverage": np.full(20, 15),
            },
            "pileup_stats": {
                "positions": positions,
                "counts": {pos: {"A": 3, "C": 5, "G": 4, "T": 3} for pos in positions},
                "reference_bases": {pos: "ACGT"[pos % 4] for pos in positions},
            },
            "quality_stats": {
                "positions": positions,
                "mean_quality": np.random.uniform(20, 30, 20),
                "std_quality": np.random.uniform(1, 3, 20),
            },
            "reference_name": "chr1:1000-1020",
            "num_reads": 15,
        }

    def test_gridplot_has_three_rows(self, sample_data):
        """Test that gridplot has three rows (signal, pileup, quality)"""
        strategy = AggregatePlotStrategy(Theme.LIGHT)

        _, grid = strategy.create_plot(sample_data, {})

        # GridPlot children are tuples of (figure, row, col)
        assert len(grid.children) == 3

    def test_all_tracks_are_plots(self, sample_data):
        """Test that all three tracks are Plot objects"""
        strategy = AggregatePlotStrategy(Theme.LIGHT)

        _, grid = strategy.create_plot(sample_data, {})

        # GridPlot children are tuples of (figure, row, col)
        for fig, _row, _col in grid.children:
            assert isinstance(fig, Plot)

    def test_signal_track_has_title(self, sample_data):
        """Test that signal track has appropriate title"""
        strategy = AggregatePlotStrategy(Theme.LIGHT)

        _, grid = strategy.create_plot(sample_data, {})

        # Extract figure from (figure, row, col) tuple
        # Panel order: pileup (0), signal (1), quality (2)
        signal_track, _, _ = grid.children[1]
        assert "Aggregate Signal" in signal_track.title.text
        assert "chr1:1000-1020" in signal_track.title.text
        assert "15 reads" in signal_track.title.text

    def test_pileup_track_has_title(self, sample_data):
        """Test that pileup track has appropriate title"""
        strategy = AggregatePlotStrategy(Theme.LIGHT)

        _, grid = strategy.create_plot(sample_data, {})

        # Extract figure from (figure, row, col) tuple
        # Panel order: pileup (0), signal (1), quality (2)
        pileup_track, _, _ = grid.children[0]
        assert "Base Call Pileup" in pileup_track.title.text

    def test_quality_track_has_title(self, sample_data):
        """Test that quality track has appropriate title"""
        strategy = AggregatePlotStrategy(Theme.LIGHT)

        _, grid = strategy.create_plot(sample_data, {})

        # Extract figure from (figure, row, col) tuple
        quality_track, _, _ = grid.children[2]
        assert "Base Quality Scores" in quality_track.title.text

    def test_pileup_track_y_range(self, sample_data):
        """Test that pileup track y-axis ranges from 0 to 1.15 (extended for labels)"""
        strategy = AggregatePlotStrategy(Theme.LIGHT)

        _, grid = strategy.create_plot(sample_data, {})

        # Extract figure from (figure, row, col) tuple
        # Panel order: pileup (0), signal (1), quality (2)
        pileup_track, _, _ = grid.children[0]
        assert pileup_track.y_range.start == 0
        assert pileup_track.y_range.end == 1.15

    def test_tracks_have_linked_x_ranges(self, sample_data):
        """Test that all three tracks have synchronized x-ranges"""
        strategy = AggregatePlotStrategy(Theme.LIGHT)

        _, grid = strategy.create_plot(sample_data, {})

        # Extract figures from (figure, row, col) tuples
        # Panel order: pileup (0), signal (1), quality (2)
        pileup_track, _, _ = grid.children[0]
        signal_track, _, _ = grid.children[1]
        quality_track, _, _ = grid.children[2]

        # All should reference the same x_range object
        assert pileup_track.x_range is signal_track.x_range
        assert quality_track.x_range is signal_track.x_range


class TestAggregateIntegration:
    """Integration tests for AggregatePlotStrategy"""

    def test_complete_workflow_light_theme(self):
        """Test complete workflow with light theme"""
        strategy = AggregatePlotStrategy(Theme.LIGHT)

        positions = np.arange(100)
        data = {
            "aggregate_stats": {
                "positions": positions,
                "mean_signal": np.sin(positions / 10),
                "std_signal": np.abs(np.random.randn(100) * 0.2),
                "median_signal": np.sin(positions / 10) * 0.9,
                "coverage": np.full(100, 50),
            },
            "pileup_stats": {
                "positions": positions,
                "counts": {
                    pos: {"A": 10, "C": 15, "G": 12, "T": 13} for pos in positions
                },
            },
            "quality_stats": {
                "positions": positions,
                "mean_quality": np.random.uniform(25, 35, 100),
                "std_quality": np.random.uniform(2, 4, 100),
            },
            "reference_name": "chr1:1000-1100",
            "num_reads": 50,
        }

        options = {"normalization": NormalizationMethod.MEDIAN}

        html, grid = strategy.create_plot(data, options)

        assert isinstance(html, str)
        assert isinstance(grid, GridPlot)
        assert len(grid.children) == 3
        assert "chr1:1000-1100" in html
        assert "50 reads" in html

    def test_complete_workflow_dark_theme(self):
        """Test complete workflow with dark theme"""
        strategy = AggregatePlotStrategy(Theme.DARK)

        positions = np.arange(50)
        data = {
            "aggregate_stats": {
                "positions": positions,
                "mean_signal": np.random.randn(50),
                "std_signal": np.abs(np.random.randn(50)),
                "coverage": np.full(50, 30),
            },
            "pileup_stats": {
                "positions": positions,
                "counts": {pos: {"A": 5, "C": 10, "G": 8, "T": 7} for pos in positions},
            },
            "quality_stats": {
                "positions": positions,
                "mean_quality": np.random.uniform(20, 30, 50),
                "std_quality": np.random.uniform(1, 3, 50),
            },
            "reference_name": "test_region",
            "num_reads": 30,
        }

        options = {"normalization": NormalizationMethod.ZNORM}

        html, grid = strategy.create_plot(data, options)

        assert isinstance(html, str)
        assert isinstance(grid, GridPlot)
        assert "test_region" in html

    def test_reuse_strategy_multiple_plots(self):
        """Test that same strategy can create multiple plots"""
        strategy = AggregatePlotStrategy(Theme.LIGHT)

        positions1 = np.arange(30)
        data1 = {
            "aggregate_stats": {
                "positions": positions1,
                "mean_signal": np.random.randn(30),
                "std_signal": np.abs(np.random.randn(30)),
                "coverage": np.full(30, 20),
            },
            "pileup_stats": {
                "positions": positions1,
                "counts": {pos: {"A": 5, "C": 5, "G": 5, "T": 5} for pos in positions1},
            },
            "quality_stats": {
                "positions": positions1,
                "mean_quality": np.random.uniform(25, 30, 30),
                "std_quality": np.random.uniform(2, 3, 30),
            },
            "reference_name": "region_1",
            "num_reads": 20,
        }

        positions2 = np.arange(40)
        data2 = {
            "aggregate_stats": {
                "positions": positions2,
                "mean_signal": np.random.randn(40),
                "std_signal": np.abs(np.random.randn(40)),
                "coverage": np.full(40, 35),
            },
            "pileup_stats": {
                "positions": positions2,
                "counts": {pos: {"A": 8, "C": 9, "G": 9, "T": 9} for pos in positions2},
            },
            "quality_stats": {
                "positions": positions2,
                "mean_quality": np.random.uniform(28, 33, 40),
                "std_quality": np.random.uniform(2, 4, 40),
            },
            "reference_name": "region_2",
            "num_reads": 35,
        }

        html1, grid1 = strategy.create_plot(data1, {})
        html2, grid2 = strategy.create_plot(data2, {})

        assert "region_1" in html1
        assert "region_2" in html2
        assert "20 reads" in html1
        assert "35 reads" in html2

    def test_with_optional_reference_bases(self):
        """Test with optional reference_bases in pileup_stats"""
        strategy = AggregatePlotStrategy(Theme.LIGHT)

        positions = np.arange(20)
        data = {
            "aggregate_stats": {
                "positions": positions,
                "mean_signal": np.random.randn(20),
                "std_signal": np.abs(np.random.randn(20)),
                "coverage": np.full(20, 10),
            },
            "pileup_stats": {
                "positions": positions,
                "counts": {pos: {"A": 2, "C": 3, "G": 2, "T": 3} for pos in positions},
                "reference_bases": {pos: "ACGT"[pos % 4] for pos in positions},
            },
            "quality_stats": {
                "positions": positions,
                "mean_quality": np.random.uniform(20, 25, 20),
                "std_quality": np.random.uniform(1, 2, 20),
            },
            "reference_name": "test",
            "num_reads": 10,
        }

        html, grid = strategy.create_plot(data, {})

        assert isinstance(html, str)
        assert isinstance(grid, GridPlot)

    def test_with_varying_coverage(self):
        """Test with varying coverage across positions"""
        strategy = AggregatePlotStrategy(Theme.LIGHT)

        positions = np.arange(50)
        # Simulate variable coverage
        coverage = np.random.randint(10, 100, 50)

        data = {
            "aggregate_stats": {
                "positions": positions,
                "mean_signal": np.random.randn(50),
                "std_signal": np.abs(np.random.randn(50)),
                "coverage": coverage,
            },
            "pileup_stats": {
                "positions": positions,
                "counts": {
                    pos: {
                        "A": int(coverage[i] * 0.25),
                        "C": int(coverage[i] * 0.25),
                        "G": int(coverage[i] * 0.25),
                        "T": int(coverage[i] * 0.25),
                    }
                    for i, pos in enumerate(positions)
                },
            },
            "quality_stats": {
                "positions": positions,
                "mean_quality": np.random.uniform(20, 35, 50),
                "std_quality": np.random.uniform(2, 5, 50),
            },
            "reference_name": "variable_cov_region",
            "num_reads": 100,
        }

        html, grid = strategy.create_plot(data, {})

        assert isinstance(html, str)
        assert isinstance(grid, GridPlot)
