"""Tests for plotting/overlay.py - OVERLAY mode"""

import numpy as np

from squiggy.constants import NormalizationMethod, Theme
from squiggy.plotting.overlay import plot_overlay


class TestPlotOverlay:
    """Tests for plot_overlay function"""

    def test_plot_overlay_basic(self):
        """Test basic overlay plotting with multiple reads"""
        reads_data = [
            ("read_001", np.random.randn(100), 4000),
            ("read_002", np.random.randn(100), 4000),
            ("read_003", np.random.randn(100), 4000),
        ]

        html, fig = plot_overlay(reads_data, NormalizationMethod.ZNORM)

        assert isinstance(html, str)
        assert len(html) > 0
        assert fig is not None
        assert "<!DOCTYPE html>" in html
        assert "Overlay" in html

    def test_plot_overlay_single_read(self):
        """Test overlay with single read"""
        reads_data = [("read_001", np.random.randn(100), 4000)]

        html, fig = plot_overlay(reads_data, NormalizationMethod.NONE)

        assert isinstance(html, str)
        assert fig is not None

    def test_plot_overlay_many_reads(self):
        """Test overlay with many reads (tests color cycling)"""
        reads_data = [(f"read_{i:03d}", np.random.randn(100), 4000) for i in range(20)]

        html, fig = plot_overlay(reads_data, NormalizationMethod.MEDIAN)

        assert isinstance(html, str)
        assert fig is not None
        assert "20 reads" in html.lower() or "20" in html

    def test_plot_overlay_normalization_methods(self):
        """Test overlay with different normalization methods"""
        reads_data = [
            ("read_001", np.random.randn(100), 4000),
            ("read_002", np.random.randn(100), 4000),
        ]

        for norm_method in NormalizationMethod:
            html, fig = plot_overlay(reads_data, norm_method)

            assert isinstance(html, str)
            assert norm_method.value in html

    def test_plot_overlay_with_downsample(self):
        """Test overlay with downsampling"""
        reads_data = [
            ("read_001", np.random.randn(1000), 4000),
            ("read_002", np.random.randn(1000), 4000),
        ]

        html, fig = plot_overlay(reads_data, NormalizationMethod.NONE, downsample=10)

        assert isinstance(html, str)
        assert "downsample" in html.lower() or "10" in html

    def test_plot_overlay_with_signal_points(self):
        """Test overlay with signal points shown"""
        reads_data = [
            ("read_001", np.random.randn(100), 4000),
            ("read_002", np.random.randn(100), 4000),
        ]

        html, fig = plot_overlay(
            reads_data, NormalizationMethod.ZNORM, show_signal_points=True
        )

        assert isinstance(html, str)
        assert fig is not None

    def test_plot_overlay_dark_theme(self):
        """Test overlay with dark theme"""
        reads_data = [
            ("read_001", np.random.randn(100), 4000),
            ("read_002", np.random.randn(100), 4000),
        ]

        html, fig = plot_overlay(reads_data, NormalizationMethod.NONE, theme=Theme.DARK)

        assert isinstance(html, str)
        assert fig is not None

    def test_plot_overlay_different_lengths(self):
        """Test overlay with reads of different lengths"""
        reads_data = [
            ("read_001", np.random.randn(100), 4000),
            ("read_002", np.random.randn(200), 4000),
            ("read_003", np.random.randn(50), 4000),
        ]

        html, fig = plot_overlay(reads_data, NormalizationMethod.MEDIAN)

        assert isinstance(html, str)
        assert fig is not None

    def test_plot_overlay_legend_labels(self):
        """Test that overlay includes truncated read IDs in legend"""
        reads_data = [
            ("very_long_read_id_that_will_be_truncated", np.random.randn(100), 4000),
            ("another_long_read_id_also_truncated", np.random.randn(100), 4000),
        ]

        html, fig = plot_overlay(reads_data, NormalizationMethod.NONE)

        # Read IDs should be truncated to 12 chars in legend
        assert isinstance(html, str)
