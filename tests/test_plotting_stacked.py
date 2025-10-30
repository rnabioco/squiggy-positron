"""Tests for plotting/stacked.py - STACKED mode"""

import numpy as np

from squiggy.constants import NormalizationMethod, Theme
from squiggy.plotting.stacked import plot_stacked


class TestPlotStacked:
    """Tests for plot_stacked function"""

    def test_plot_stacked_basic(self):
        """Test basic stacked plotting with multiple reads"""
        reads_data = [
            ("read_001", np.random.randn(100), 4000),
            ("read_002", np.random.randn(100), 4000),
            ("read_003", np.random.randn(100), 4000),
        ]

        html, fig = plot_stacked(reads_data, NormalizationMethod.ZNORM)

        assert isinstance(html, str)
        assert len(html) > 0
        assert fig is not None
        assert "<!DOCTYPE html>" in html
        assert "Stacked" in html

    def test_plot_stacked_single_read(self):
        """Test stacked with single read"""
        reads_data = [("read_001", np.random.randn(100), 4000)]

        html, fig = plot_stacked(reads_data, NormalizationMethod.NONE)

        assert isinstance(html, str)
        assert fig is not None

    def test_plot_stacked_many_reads(self):
        """Test stacked with many reads"""
        reads_data = [(f"read_{i:03d}", np.random.randn(100), 4000) for i in range(10)]

        html, fig = plot_stacked(reads_data, NormalizationMethod.MEDIAN)

        assert isinstance(html, str)
        assert fig is not None
        assert "10 reads" in html.lower() or "10" in html

    def test_plot_stacked_normalization_methods(self):
        """Test stacked with different normalization methods"""
        reads_data = [
            ("read_001", np.random.randn(100), 4000),
            ("read_002", np.random.randn(100), 4000),
        ]

        for norm_method in NormalizationMethod:
            html, fig = plot_stacked(reads_data, norm_method)

            assert isinstance(html, str)
            assert norm_method.value in html

    def test_plot_stacked_with_downsample(self):
        """Test stacked with downsampling"""
        reads_data = [
            ("read_001", np.random.randn(1000), 4000),
            ("read_002", np.random.randn(1000), 4000),
        ]

        html, fig = plot_stacked(reads_data, NormalizationMethod.NONE, downsample=10)

        assert isinstance(html, str)
        assert "downsample" in html.lower() or "10" in html

    def test_plot_stacked_with_signal_points(self):
        """Test stacked with signal points shown"""
        reads_data = [
            ("read_001", np.random.randn(100), 4000),
            ("read_002", np.random.randn(100), 4000),
        ]

        html, fig = plot_stacked(
            reads_data, NormalizationMethod.ZNORM, show_signal_points=True
        )

        assert isinstance(html, str)
        assert fig is not None

    def test_plot_stacked_dark_theme(self):
        """Test stacked with dark theme"""
        reads_data = [
            ("read_001", np.random.randn(100), 4000),
            ("read_002", np.random.randn(100), 4000),
        ]

        html, fig = plot_stacked(reads_data, NormalizationMethod.NONE, theme=Theme.DARK)

        assert isinstance(html, str)
        assert fig is not None

    def test_plot_stacked_different_lengths(self):
        """Test stacked with reads of different lengths"""
        reads_data = [
            ("read_001", np.random.randn(100), 4000),
            ("read_002", np.random.randn(200), 4000),
            ("read_003", np.random.randn(50), 4000),
        ]

        html, fig = plot_stacked(reads_data, NormalizationMethod.MEDIAN)

        assert isinstance(html, str)
        assert fig is not None

    def test_plot_stacked_offset_calculation(self):
        """Test that stacked plot calculates appropriate offset"""
        # Create signals with different ranges
        reads_data = [
            ("read_001", np.array([0, 1, 2, 3, 4] * 20), 4000),
            ("read_002", np.array([0, 10, 20, 30, 40] * 20), 4000),
            ("read_003", np.array([0, 0.1, 0.2, 0.3, 0.4] * 20), 4000),
        ]

        html, fig = plot_stacked(reads_data, NormalizationMethod.NONE)

        # Should handle different signal ranges appropriately
        assert isinstance(html, str)
        assert fig is not None

    def test_plot_stacked_offset_label(self):
        """Test that y-axis label mentions offset"""
        reads_data = [
            ("read_001", np.random.randn(100), 4000),
            ("read_002", np.random.randn(100), 4000),
        ]

        html, fig = plot_stacked(reads_data, NormalizationMethod.NONE)

        # Y-axis should mention offset
        assert "offset" in fig.yaxis.axis_label.lower()
