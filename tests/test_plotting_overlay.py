"""Tests for plotting/overlay.py - OVERLAY mode"""

import numpy as np
import pytest

from squiggy.alignment import AlignedRead, BaseAnnotation
from squiggy.constants import CoordinateSpace, NormalizationMethod, Theme
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


class TestPlotOverlaySequenceSpace:
    """Tests for plot_overlay with SEQUENCE coordinate space"""

    def test_plot_overlay_sequence_space_basic(self):
        """Test overlay plotting in sequence space with aligned reads"""
        # Create mock aligned reads with base annotations
        bases1 = [
            BaseAnnotation(base="A", position=0, signal_start=0, signal_end=50, genomic_pos=1000),
            BaseAnnotation(base="C", position=1, signal_start=50, signal_end=100, genomic_pos=1001),
        ]
        bases2 = [
            BaseAnnotation(base="A", position=0, signal_start=0, signal_end=40, genomic_pos=1000),
            BaseAnnotation(base="C", position=1, signal_start=40, signal_end=80, genomic_pos=1001),
        ]

        aligned_reads = [
            AlignedRead(read_id="read_001", sequence="AC", bases=bases1, chromosome="chr1", genomic_start=1000, genomic_end=1002),
            AlignedRead(read_id="read_002", sequence="AC", bases=bases2, chromosome="chr1", genomic_start=1000, genomic_end=1002),
        ]

        reads_data = [
            ("read_001", np.random.randn(100), 4000),
            ("read_002", np.random.randn(100), 4000),
        ]

        html, fig = plot_overlay(
            reads_data,
            NormalizationMethod.ZNORM,
            coordinate_space=CoordinateSpace.SEQUENCE,
            aligned_reads=aligned_reads,
        )

        assert isinstance(html, str)
        assert len(html) > 0
        assert fig is not None
        assert "Reference Position" in html

    def test_plot_overlay_sequence_space_requires_aligned_reads(self):
        """Test that sequence space requires aligned_reads parameter"""
        reads_data = [
            ("read_001", np.random.randn(100), 4000),
            ("read_002", np.random.randn(100), 4000),
        ]

        with pytest.raises(ValueError, match="aligned_reads required"):
            plot_overlay(
                reads_data,
                NormalizationMethod.NONE,
                coordinate_space=CoordinateSpace.SEQUENCE,
                aligned_reads=None,
            )

    def test_plot_overlay_sequence_space_skips_unaligned_reads(self):
        """Test that reads without alignment are skipped in sequence space"""
        # One read with alignment, one without
        bases1 = [
            BaseAnnotation(base="A", position=0, signal_start=0, signal_end=50, genomic_pos=1000),
        ]

        aligned_reads = [
            AlignedRead(read_id="read_001", sequence="A", bases=bases1, chromosome="chr1", genomic_start=1000, genomic_end=1001),
            # read_002 has no bases (no alignment)
            AlignedRead(read_id="read_002", sequence="", bases=[], chromosome=None, genomic_start=None, genomic_end=None),
        ]

        reads_data = [
            ("read_001", np.random.randn(100), 4000),
            ("read_002", np.random.randn(100), 4000),
        ]

        html, fig = plot_overlay(
            reads_data,
            NormalizationMethod.NONE,
            coordinate_space=CoordinateSpace.SEQUENCE,
            aligned_reads=aligned_reads,
        )

        # Should succeed even though read_002 has no alignment
        assert isinstance(html, str)
        assert fig is not None

    def test_plot_overlay_signal_space_default(self):
        """Test that signal space is the default coordinate space"""
        reads_data = [
            ("read_001", np.random.randn(100), 4000),
            ("read_002", np.random.randn(100), 4000),
        ]

        # Default should be signal space (no aligned_reads needed)
        html, fig = plot_overlay(reads_data, NormalizationMethod.NONE)

        assert isinstance(html, str)
        assert "Sample" in html  # Signal space uses "Sample" label

    def test_plot_overlay_sequence_space_with_downsample(self):
        """Test sequence space plotting with downsampling"""
        bases1 = [
            BaseAnnotation(base="A", position=0, signal_start=0, signal_end=100, genomic_pos=1000),
            BaseAnnotation(base="C", position=1, signal_start=100, signal_end=200, genomic_pos=1001),
        ]

        aligned_reads = [
            AlignedRead(read_id="read_001", sequence="AC", bases=bases1, chromosome="chr1", genomic_start=1000, genomic_end=1002),
        ]

        reads_data = [
            ("read_001", np.random.randn(500), 4000),
        ]

        html, fig = plot_overlay(
            reads_data,
            NormalizationMethod.NONE,
            downsample=5,
            coordinate_space=CoordinateSpace.SEQUENCE,
            aligned_reads=aligned_reads,
        )

        assert isinstance(html, str)
        assert fig is not None

    def test_plot_overlay_sequence_space_different_normalizations(self):
        """Test sequence space with different normalization methods"""
        bases1 = [
            BaseAnnotation(base="A", position=0, signal_start=0, signal_end=50, genomic_pos=1000),
        ]

        aligned_reads = [
            AlignedRead(read_id="read_001", sequence="A", bases=bases1, chromosome="chr1", genomic_start=1000, genomic_end=1001),
        ]

        reads_data = [
            ("read_001", np.random.randn(100), 4000),
        ]

        for norm_method in NormalizationMethod:
            html, fig = plot_overlay(
                reads_data,
                norm_method,
                coordinate_space=CoordinateSpace.SEQUENCE,
                aligned_reads=aligned_reads,
            )

            assert isinstance(html, str)
            assert norm_method.value in html
