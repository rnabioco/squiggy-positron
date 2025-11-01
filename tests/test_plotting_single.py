"""Tests for plotting/single.py - SINGLE read mode"""

import numpy as np
from squiggy.plotting.single import add_base_annotations_single_read, plot_single_read

from squiggy.constants import NormalizationMethod, Theme


class TestPlotSingleRead:
    """Tests for plot_single_read function"""

    def test_plot_single_read_basic(self):
        """Test basic single read plotting without annotations"""
        signal = np.random.randn(1000)
        read_id = "test_read_001"
        sample_rate = 4000

        html, fig = plot_single_read(signal, read_id, sample_rate)

        assert isinstance(html, str)
        assert len(html) > 0
        assert fig is not None
        assert "<!DOCTYPE html>" in html
        assert read_id in html

    def test_plot_single_read_with_sequence(self):
        """Test single read with base sequence"""
        signal = np.random.randn(100)
        read_id = "test_read_002"
        sample_rate = 4000
        sequence = "ACGTACGT"
        seq_to_sig_map = [0, 10, 20, 30, 40, 50, 60, 70]

        html, fig = plot_single_read(
            signal,
            read_id,
            sample_rate,
            sequence=sequence,
            seq_to_sig_map=seq_to_sig_map,
        )

        assert isinstance(html, str)
        assert fig is not None

    def test_plot_single_read_normalization(self):
        """Test single read with different normalization methods"""
        signal = np.random.randn(500)
        read_id = "test_read_003"
        sample_rate = 4000

        for norm_method in NormalizationMethod:
            html, fig = plot_single_read(
                signal, read_id, sample_rate, normalization=norm_method
            )

            assert isinstance(html, str)
            assert norm_method.value in html

    def test_plot_single_read_downsampling(self):
        """Test single read with downsampling"""
        signal = np.random.randn(1000)
        read_id = "test_read_004"
        sample_rate = 4000

        html, fig = plot_single_read(signal, read_id, sample_rate, downsample=10)

        assert isinstance(html, str)
        assert "downsample" in html.lower() or "10" in html

    def test_plot_single_read_dwell_time(self):
        """Test single read with dwell time coloring"""
        signal = np.random.randn(100)
        read_id = "test_read_005"
        sample_rate = 4000
        sequence = "ACGT"
        seq_to_sig_map = [0, 20, 40, 60]

        html, fig = plot_single_read(
            signal,
            read_id,
            sample_rate,
            sequence=sequence,
            seq_to_sig_map=seq_to_sig_map,
            show_dwell_time=True,
        )

        assert isinstance(html, str)
        assert "dwell" in html.lower() or "Dwell" in html

    def test_plot_single_read_with_labels(self):
        """Test single read with base labels shown"""
        signal = np.random.randn(100)
        read_id = "test_read_006"
        sample_rate = 4000
        sequence = "ACGT"
        seq_to_sig_map = [0, 20, 40, 60]

        html_with_labels, _ = plot_single_read(
            signal,
            read_id,
            sample_rate,
            sequence=sequence,
            seq_to_sig_map=seq_to_sig_map,
            show_labels=True,
        )

        html_without_labels, _ = plot_single_read(
            signal,
            read_id,
            sample_rate,
            sequence=sequence,
            seq_to_sig_map=seq_to_sig_map,
            show_labels=False,
        )

        assert isinstance(html_with_labels, str)
        assert isinstance(html_without_labels, str)

    def test_plot_single_read_signal_points(self):
        """Test single read with signal points shown"""
        signal = np.random.randn(100)
        read_id = "test_read_007"
        sample_rate = 4000

        html, fig = plot_single_read(
            signal, read_id, sample_rate, show_signal_points=True
        )

        assert isinstance(html, str)
        assert fig is not None

    def test_plot_single_read_dark_theme(self):
        """Test single read with dark theme"""
        signal = np.random.randn(100)
        read_id = "test_read_008"
        sample_rate = 4000

        html, fig = plot_single_read(signal, read_id, sample_rate, theme=Theme.DARK)

        assert isinstance(html, str)
        assert fig is not None

    def test_plot_single_read_empty_sequence(self):
        """Test single read with empty sequence (no annotations)"""
        signal = np.random.randn(100)
        read_id = "test_read_009"
        sample_rate = 4000

        html, fig = plot_single_read(
            signal, read_id, sample_rate, sequence=None, seq_to_sig_map=None
        )

        assert isinstance(html, str)
        assert fig is not None


class TestAddBaseAnnotationsSingleRead:
    """Tests for add_base_annotations_single_read function"""

    def test_add_annotations_no_sequence(self):
        """Test adding annotations with no sequence"""
        from bokeh.plotting import figure

        p = figure()
        signal = np.random.randn(100)
        time_ms = np.arange(100) * 0.25

        color_mapper, toggle = add_base_annotations_single_read(
            p,
            signal,
            time_ms,
            sequence=None,
            seq_to_sig_map=None,
            sample_rate=4000,
            show_dwell_time=False,
            show_labels=False,
        )

        # Should return None, None when no sequence
        assert color_mapper is None
        assert toggle is None

    def test_add_annotations_with_sequence(self):
        """Test adding annotations with valid sequence"""
        from bokeh.plotting import figure

        p = figure()
        signal = np.random.randn(100)
        time_ms = np.arange(100) * 0.25
        sequence = "ACGT"
        seq_to_sig_map = [0, 20, 40, 60]

        color_mapper, toggle = add_base_annotations_single_read(
            p,
            signal,
            time_ms,
            sequence,
            seq_to_sig_map,
            sample_rate=4000,
            show_dwell_time=False,
            show_labels=True,
        )

        # Should not raise error
        assert True

    def test_add_annotations_dwell_time_mode(self):
        """Test adding annotations in dwell time mode"""
        from bokeh.plotting import figure

        p = figure()
        signal = np.random.randn(100)
        time_ms = np.arange(100) * 0.25
        sequence = "ACGT"
        seq_to_sig_map = [0, 20, 40, 60]

        color_mapper, toggle = add_base_annotations_single_read(
            p,
            signal,
            time_ms,
            sequence,
            seq_to_sig_map,
            sample_rate=4000,
            show_dwell_time=True,
            show_labels=True,
        )

        # In dwell time mode, should return a color mapper
        assert color_mapper is not None

    def test_add_annotations_empty_seq_map(self):
        """Test with empty seq_to_sig_map"""
        from bokeh.plotting import figure

        p = figure()
        signal = np.random.randn(100)
        time_ms = np.arange(100) * 0.25

        color_mapper, toggle = add_base_annotations_single_read(
            p,
            signal,
            time_ms,
            sequence="ACGT",
            seq_to_sig_map=[],
            sample_rate=4000,
            show_dwell_time=False,
            show_labels=False,
        )

        # Should return None, None with empty map
        assert color_mapper is None
        assert toggle is None
