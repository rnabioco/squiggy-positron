"""Tests for Bokeh-based plotting functionality."""

import re

import numpy as np
import pytest


class TestBokehHTMLOutput:
    """Tests for Bokeh HTML output generation."""

    def test_plot_single_read_returns_html(self, sample_pod5_file):
        """Test that plot_single_read returns an HTML string."""
        import pod5

        from squiggy.plotter_bokeh import BokehSquigglePlotter

        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())

            html = BokehSquigglePlotter.plot_single_read(
                signal=read.signal,
                read_id=str(read.read_id),
                sample_rate=read.run_info.sample_rate,
            )

            # Verify HTML string is returned
            assert isinstance(html, str)
            assert len(html) > 0

    def test_html_contains_bokeh_elements(self, sample_pod5_file):
        """Test that generated HTML contains Bokeh JavaScript and structure."""
        import pod5

        from squiggy.plotter_bokeh import BokehSquigglePlotter

        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())

            html = BokehSquigglePlotter.plot_single_read(
                signal=read.signal,
                read_id=str(read.read_id),
                sample_rate=read.run_info.sample_rate,
            )

            # Verify HTML structure
            assert "<!DOCTYPE html>" in html
            assert "<html" in html
            assert "</html>" in html

            # Verify Bokeh is included
            assert "Bokeh" in html
            assert "BokehJS" in html or "bokeh" in html.lower()


class TestSingleReadPlotting:
    """Tests for single read plotting functionality."""

    def test_plot_single_read_basic(self, sample_pod5_file):
        """Test basic single read plot without annotations."""
        import pod5

        from squiggy.plotter_bokeh import BokehSquigglePlotter

        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())

            html = BokehSquigglePlotter.plot_single_read(
                signal=read.signal,
                read_id=str(read.read_id),
                sample_rate=read.run_info.sample_rate,
                sequence=None,
                seq_to_sig_map=None,
            )

            assert isinstance(html, str)
            assert len(html) > 0
            # Should contain read ID in title
            assert str(read.read_id) in html

    def test_plot_single_read_with_bases(self, sample_pod5_file, sample_bam_file):
        """Test single read plot with base annotations from BAM."""
        import pod5

        from squiggy.plotter_bokeh import BokehSquigglePlotter
        from squiggy.utils import get_basecall_data

        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())
            read_id = str(read.read_id)

            # Get basecall data from BAM
            sequence, seq_to_sig_map = get_basecall_data(sample_bam_file, read_id)

            if sequence is not None:
                html = BokehSquigglePlotter.plot_single_read(
                    signal=read.signal,
                    read_id=read_id,
                    sample_rate=read.run_info.sample_rate,
                    sequence=sequence,
                    seq_to_sig_map=seq_to_sig_map,
                    show_labels=True,
                )

                assert isinstance(html, str)
                assert len(html) > 0
                # Should contain base-related elements
                assert "base" in html.lower() or sequence[0] in html
            else:
                pytest.skip("No basecall data available for this read")

    def test_plot_single_read_dwell_time(self, sample_pod5_file, sample_bam_file):
        """Test single read plot with dwell time coloring enabled."""
        import pod5

        from squiggy.plotter_bokeh import BokehSquigglePlotter
        from squiggy.utils import get_basecall_data

        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())
            read_id = str(read.read_id)

            # Get basecall data (required for dwell time)
            sequence, seq_to_sig_map = get_basecall_data(sample_bam_file, read_id)

            if sequence is not None:
                html = BokehSquigglePlotter.plot_single_read(
                    signal=read.signal,
                    read_id=read_id,
                    sample_rate=read.run_info.sample_rate,
                    sequence=sequence,
                    seq_to_sig_map=seq_to_sig_map,
                    show_dwell_time=True,
                )

                assert isinstance(html, str)
                assert len(html) > 0
                # Should mention dwell time
                assert "dwell" in html.lower() or "Dwell" in html
            else:
                pytest.skip("No basecall data available for dwell time test")

    def test_plot_single_read_with_labels(self, sample_pod5_file, sample_bam_file):
        """Test single read plot with base labels shown."""
        import pod5

        from squiggy.plotter_bokeh import BokehSquigglePlotter
        from squiggy.utils import get_basecall_data

        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())
            read_id = str(read.read_id)

            sequence, seq_to_sig_map = get_basecall_data(sample_bam_file, read_id)

            if sequence is not None:
                # Plot with labels enabled
                html_with_labels = BokehSquigglePlotter.plot_single_read(
                    signal=read.signal,
                    read_id=read_id,
                    sample_rate=read.run_info.sample_rate,
                    sequence=sequence,
                    seq_to_sig_map=seq_to_sig_map,
                    show_labels=True,
                )

                # Plot with labels disabled
                html_no_labels = BokehSquigglePlotter.plot_single_read(
                    signal=read.signal,
                    read_id=read_id,
                    sample_rate=read.run_info.sample_rate,
                    sequence=sequence,
                    seq_to_sig_map=seq_to_sig_map,
                    show_labels=False,
                )

                assert isinstance(html_with_labels, str)
                assert isinstance(html_no_labels, str)
                # Both should be valid HTML but may differ in content
                assert len(html_with_labels) > 0
                assert len(html_no_labels) > 0
            else:
                pytest.skip("No basecall data available for labels test")


class TestSignalNormalization:
    """Tests for signal normalization methods."""

    def test_normalize_signal_znorm(self):
        """Test Z-score normalization (mean=0, std=1)."""
        from squiggy.constants import NormalizationMethod
        from squiggy.plotter_bokeh import BokehSquigglePlotter

        # Create synthetic signal
        signal = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        normalized = BokehSquigglePlotter.normalize_signal(
            signal, NormalizationMethod.ZNORM
        )

        # Verify mean is close to 0
        assert np.isclose(np.mean(normalized), 0.0, atol=1e-10)
        # Verify std is close to 1
        assert np.isclose(np.std(normalized), 1.0, atol=1e-10)

    def test_normalize_signal_median(self):
        """Test median normalization."""
        from squiggy.constants import NormalizationMethod
        from squiggy.plotter_bokeh import BokehSquigglePlotter

        signal = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        normalized = BokehSquigglePlotter.normalize_signal(
            signal, NormalizationMethod.MEDIAN
        )

        # Verify median is close to 0
        assert np.isclose(np.median(normalized), 0.0, atol=1e-10)

    def test_normalize_signal_mad(self):
        """Test MAD (Median Absolute Deviation) normalization."""
        from squiggy.constants import NormalizationMethod
        from squiggy.plotter_bokeh import BokehSquigglePlotter

        signal = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        normalized = BokehSquigglePlotter.normalize_signal(
            signal, NormalizationMethod.MAD
        )

        # Verify normalization was applied
        assert not np.array_equal(normalized, signal)
        # Median should be close to 0
        assert np.isclose(np.median(normalized), 0.0, atol=1e-10)

    def test_normalize_signal_none(self):
        """Test no normalization (returns original signal)."""
        from squiggy.constants import NormalizationMethod
        from squiggy.plotter_bokeh import BokehSquigglePlotter

        signal = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        normalized = BokehSquigglePlotter.normalize_signal(
            signal, NormalizationMethod.NONE
        )

        # Verify signal is unchanged
        np.testing.assert_array_equal(normalized, signal)


class TestMultipleReadModes:
    """Tests for multiple read plotting modes."""

    def test_plot_overlay_mode(self, sample_pod5_file):
        """Test OVERLAY mode with multiple reads."""
        import pod5

        from squiggy.constants import NormalizationMethod, PlotMode
        from squiggy.plotter_bokeh import BokehSquigglePlotter

        with pod5.Reader(sample_pod5_file) as reader:
            reads = list(reader.reads())[:3]  # Get first 3 reads

            reads_data = [
                (str(r.read_id), r.signal, r.run_info.sample_rate) for r in reads
            ]

            html = BokehSquigglePlotter.plot_multiple_reads(
                reads_data,
                mode=PlotMode.OVERLAY,
                normalization=NormalizationMethod.ZNORM,
            )

            assert isinstance(html, str)
            assert len(html) > 0
            # Should contain overlay-related text
            assert "overlay" in html.lower() or "Overlay" in html

    def test_plot_stacked_mode(self, sample_pod5_file):
        """Test STACKED mode with multiple reads."""
        import pod5

        from squiggy.constants import NormalizationMethod, PlotMode
        from squiggy.plotter_bokeh import BokehSquigglePlotter

        with pod5.Reader(sample_pod5_file) as reader:
            reads = list(reader.reads())[:3]  # Get first 3 reads

            reads_data = [
                (str(r.read_id), r.signal, r.run_info.sample_rate) for r in reads
            ]

            html = BokehSquigglePlotter.plot_multiple_reads(
                reads_data,
                mode=PlotMode.STACKED,
                normalization=NormalizationMethod.MAD,
            )

            assert isinstance(html, str)
            assert len(html) > 0
            # Should contain stacked-related text
            assert "stacked" in html.lower() or "Stacked" in html

    def test_plot_eventalign_mode(self, sample_pod5_file, indexed_bam_file):
        """Test EVENTALIGN mode with aligned reads."""
        import pod5

        from squiggy.alignment import extract_alignment_from_bam
        from squiggy.constants import NormalizationMethod, PlotMode
        from squiggy.plotter_bokeh import BokehSquigglePlotter

        with pod5.Reader(sample_pod5_file) as reader:
            reads = list(reader.reads())[:3]

            reads_data = []
            aligned_reads = []

            for read in reads:
                read_id = str(read.read_id)
                aligned_read = extract_alignment_from_bam(indexed_bam_file, read_id)

                if aligned_read:
                    reads_data.append((read_id, read.signal, read.run_info.sample_rate))
                    aligned_reads.append(aligned_read)

            if len(reads_data) > 0:
                html = BokehSquigglePlotter.plot_multiple_reads(
                    reads_data,
                    mode=PlotMode.EVENTALIGN,
                    normalization=NormalizationMethod.MEDIAN,
                    aligned_reads=aligned_reads,
                )

                assert isinstance(html, str)
                assert len(html) > 0
            else:
                pytest.skip("No aligned reads found in BAM file")


class TestDownsampling:
    """Tests for signal downsampling functionality."""

    def test_downsampling_reduces_signal(self):
        """Test that downsampling reduces the number of data points."""
        from squiggy.plotter_bokeh import BokehSquigglePlotter

        # Create large signal
        signal = np.random.randn(10000)
        downsample_factor = 10

        # Downsample
        downsampled = signal[::downsample_factor]

        assert len(downsampled) == len(signal) // downsample_factor

    def test_plot_with_downsampling(self, sample_pod5_file):
        """Test plot generation with downsampled signal."""
        import pod5

        from squiggy.plotter_bokeh import BokehSquigglePlotter

        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())

            # Plot with downsampling
            html = BokehSquigglePlotter.plot_single_read(
                signal=read.signal,
                read_id=str(read.read_id),
                sample_rate=read.run_info.sample_rate,
                downsample=10,  # Every 10th point
            )

            assert isinstance(html, str)
            assert len(html) > 0


class TestDwellTimeVisualization:
    """Tests for dwell time calculation and visualization."""

    def test_dwell_time_calculation_time_mode(self, sample_pod5_file, sample_bam_file):
        """Test dwell time calculation in time-based plots."""
        import pod5

        from squiggy.plotter_bokeh import BokehSquigglePlotter
        from squiggy.utils import get_basecall_data

        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())
            read_id = str(read.read_id)

            sequence, seq_to_sig_map = get_basecall_data(sample_bam_file, read_id)

            if sequence is not None:
                # Test internal method for calculating base regions
                signal = read.signal
                sample_rate = read.run_info.sample_rate
                time_ms = (np.arange(len(signal)) / sample_rate) * 1000

                signal_min = np.min(signal)
                signal_max = np.max(signal)

                # Call internal method
                result = BokehSquigglePlotter._calculate_base_regions_time_mode(
                    sequence,
                    seq_to_sig_map,
                    time_ms,
                    signal,
                    signal_min,
                    signal_max,
                    sample_rate,
                    show_dwell_time=True,
                )

                # Should return (regions, dwell_times, labels_data)
                assert len(result) == 3
                all_regions, all_dwell_times, all_labels_data = result

                assert isinstance(all_regions, list)
                assert isinstance(all_dwell_times, list)
                assert isinstance(all_labels_data, list)

                # Regions and dwell times should have same length
                assert len(all_regions) == len(all_dwell_times)

                # Each dwell time should be positive
                for dwell in all_dwell_times:
                    assert dwell >= 0
            else:
                pytest.skip("No basecall data available for dwell time test")

    def test_dwell_time_calculation_position_mode(
        self, sample_pod5_file, indexed_bam_file
    ):
        """Test dwell time calculation in position-based plots (EVENTALIGN)."""
        import pod5
        import pysam

        from squiggy.alignment import extract_alignment_from_bam
        from squiggy.plotter_bokeh import BokehSquigglePlotter

        # Get first aligned read from BAM
        with pysam.AlignmentFile(str(indexed_bam_file), "rb") as bam:
            for alignment in bam.fetch(until_eof=True):
                read_id = alignment.query_name
                aligned_read = extract_alignment_from_bam(indexed_bam_file, read_id)

                if aligned_read and len(aligned_read.bases) > 0:
                    # Get actual signal from POD5 to determine real signal length
                    with pod5.Reader(sample_pod5_file) as reader:
                        for pod5_read in reader.reads():
                            if str(pod5_read.read_id) == read_id:
                                signal = pod5_read.signal
                                sample_rate = pod5_read.run_info.sample_rate
                                signal_length = len(signal)

                                # Test internal method
                                signal_min = float(np.min(signal))
                                signal_max = float(np.max(signal))

                                result = BokehSquigglePlotter._calculate_base_regions_position_mode(
                                    aligned_read.bases,
                                    signal_min,
                                    signal_max,
                                    sample_rate,
                                    signal_length,
                                    show_dwell_time=True,
                                )

                                # Should return (regions, dwell_times)
                                assert len(result) == 2
                                all_regions, all_dwell_times = result

                                assert isinstance(all_regions, list)
                                assert isinstance(all_dwell_times, list)

                                # Should have same length
                                assert len(all_regions) == len(all_dwell_times)

                                # Each dwell time should be non-negative
                                for dwell in all_dwell_times:
                                    assert dwell >= 0

                                return  # Test passed, exit

        pytest.skip("No aligned reads found in BAM file")

    def test_dwell_time_patches_created(self, sample_pod5_file, sample_bam_file):
        """Test that dwell time color patches are created."""
        import pod5

        from squiggy.plotter_bokeh import BokehSquigglePlotter
        from squiggy.utils import get_basecall_data

        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())
            read_id = str(read.read_id)

            sequence, seq_to_sig_map = get_basecall_data(sample_bam_file, read_id)

            if sequence is not None:
                # Plot with dwell time enabled
                html = BokehSquigglePlotter.plot_single_read(
                    signal=read.signal,
                    read_id=read_id,
                    sample_rate=read.run_info.sample_rate,
                    sequence=sequence,
                    seq_to_sig_map=seq_to_sig_map,
                    show_dwell_time=True,
                )

                # HTML should mention dwell time
                assert "dwell" in html.lower() or "Dwell" in html
                # Should contain color-related elements
                assert "color" in html.lower() or "Color" in html
            else:
                pytest.skip("No basecall data available for dwell time patches test")


class TestPlotFormatting:
    """Tests for plot title and label formatting."""

    def test_format_plot_title_single_read(self):
        """Test plot title formatting for single read."""
        from squiggy.plotter_bokeh import BokehSquigglePlotter

        reads_data = [("read_001", np.random.randn(1000), 4000)]

        title = BokehSquigglePlotter._format_plot_title("Test Mode", reads_data)

        assert isinstance(title, str)
        assert "Test Mode" in title
        assert "1 read" in title.lower() or "read" in title.lower()

    def test_format_plot_title_multiple_reads(self):
        """Test plot title formatting for multiple reads."""
        from squiggy.plotter_bokeh import BokehSquigglePlotter

        reads_data = [
            ("read_001", np.random.randn(1000), 4000),
            ("read_002", np.random.randn(1000), 4000),
            ("read_003", np.random.randn(1000), 4000),
        ]

        title = BokehSquigglePlotter._format_plot_title("Overlay", reads_data)

        assert isinstance(title, str)
        assert "Overlay" in title
        assert "3" in title or "three" in title.lower()

    def test_create_figure(self):
        """Test figure creation with proper labels."""
        from squiggy.plotter_bokeh import BokehSquigglePlotter

        fig = BokehSquigglePlotter._create_figure(
            title="Test Plot", x_label="Time (ms)", y_label="Signal (pA)"
        )

        assert fig is not None
        assert fig.title.text == "Test Plot"
        assert fig.xaxis.axis_label == "Time (ms)"
        assert fig.yaxis.axis_label == "Signal (pA)"
