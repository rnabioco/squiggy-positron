"""Tests for Bokeh-based plotting functionality."""

import numpy as np
import pytest


class TestBokehHTMLOutput:
    """Tests for Bokeh HTML output generation."""

    def test_plot_single_read_returns_html(self, sample_pod5_file):
        """Test that plot_single_read returns an HTML string and figure."""
        import pod5

        from squiggy.plotter import SquigglePlotter

        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())

            html, figure = SquigglePlotter.plot_single_read(
                signal=read.signal,
                read_id=str(read.read_id),
                sample_rate=read.run_info.sample_rate,
            )

            # Verify HTML string is returned
            assert isinstance(html, str)
            assert len(html) > 0
            # Verify figure object is returned
            assert figure is not None

    def test_html_contains_bokeh_elements(self, sample_pod5_file):
        """Test that generated HTML contains Bokeh JavaScript and structure."""
        import pod5

        from squiggy.plotter import SquigglePlotter

        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())

            html, _figure = SquigglePlotter.plot_single_read(
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

        from squiggy.plotter import SquigglePlotter

        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())

            html, _figure = SquigglePlotter.plot_single_read(
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

        from squiggy.plotter import SquigglePlotter
        from squiggy.utils import get_basecall_data

        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())
            read_id = str(read.read_id)

            # Get basecall data from BAM
            sequence, seq_to_sig_map = get_basecall_data(sample_bam_file, read_id)

            if sequence is not None:
                html, _figure = SquigglePlotter.plot_single_read(
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

        from squiggy.plotter import SquigglePlotter
        from squiggy.utils import get_basecall_data

        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())
            read_id = str(read.read_id)

            # Get basecall data (required for dwell time)
            sequence, seq_to_sig_map = get_basecall_data(sample_bam_file, read_id)

            if sequence is not None:
                html, _figure = SquigglePlotter.plot_single_read(
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

        from squiggy.plotter import SquigglePlotter
        from squiggy.utils import get_basecall_data

        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())
            read_id = str(read.read_id)

            sequence, seq_to_sig_map = get_basecall_data(sample_bam_file, read_id)

            if sequence is not None:
                # Plot with labels enabled
                html_with_labels, _figure = SquigglePlotter.plot_single_read(
                    signal=read.signal,
                    read_id=read_id,
                    sample_rate=read.run_info.sample_rate,
                    sequence=sequence,
                    seq_to_sig_map=seq_to_sig_map,
                    show_labels=True,
                )

                # Plot with labels disabled
                html_no_labels, _figure = SquigglePlotter.plot_single_read(
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
        from squiggy.plotter import SquigglePlotter

        # Create synthetic signal
        signal = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        normalized = SquigglePlotter.normalize_signal(signal, NormalizationMethod.ZNORM)

        # Verify mean is close to 0
        assert np.isclose(np.mean(normalized), 0.0, atol=1e-10)
        # Verify std is close to 1
        assert np.isclose(np.std(normalized), 1.0, atol=1e-10)

    def test_normalize_signal_median(self):
        """Test median normalization."""
        from squiggy.constants import NormalizationMethod
        from squiggy.plotter import SquigglePlotter

        signal = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        normalized = SquigglePlotter.normalize_signal(
            signal, NormalizationMethod.MEDIAN
        )

        # Verify median is close to 0
        assert np.isclose(np.median(normalized), 0.0, atol=1e-10)

    def test_normalize_signal_mad(self):
        """Test MAD (Median Absolute Deviation) normalization."""
        from squiggy.constants import NormalizationMethod
        from squiggy.plotter import SquigglePlotter

        signal = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        normalized = SquigglePlotter.normalize_signal(signal, NormalizationMethod.MAD)

        # Verify normalization was applied
        assert not np.array_equal(normalized, signal)
        # Median should be close to 0
        assert np.isclose(np.median(normalized), 0.0, atol=1e-10)

    def test_normalize_signal_none(self):
        """Test no normalization (returns original signal)."""
        from squiggy.constants import NormalizationMethod
        from squiggy.plotter import SquigglePlotter

        signal = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        normalized = SquigglePlotter.normalize_signal(signal, NormalizationMethod.NONE)

        # Verify signal is unchanged
        np.testing.assert_array_equal(normalized, signal)


class TestMultipleReadModes:
    """Tests for multiple read plotting modes."""

    def test_plot_overlay_mode(self, sample_pod5_file):
        """Test OVERLAY mode with multiple reads."""
        import pod5

        from squiggy.constants import NormalizationMethod, PlotMode
        from squiggy.plotter import SquigglePlotter

        with pod5.Reader(sample_pod5_file) as reader:
            reads = list(reader.reads())[:3]  # Get first 3 reads

            reads_data = [
                (str(r.read_id), r.signal, r.run_info.sample_rate) for r in reads
            ]

            html, _figure = SquigglePlotter.plot_multiple_reads(
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
        from squiggy.plotter import SquigglePlotter

        with pod5.Reader(sample_pod5_file) as reader:
            reads = list(reader.reads())[:3]  # Get first 3 reads

            reads_data = [
                (str(r.read_id), r.signal, r.run_info.sample_rate) for r in reads
            ]

            html, _figure = SquigglePlotter.plot_multiple_reads(
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
        from squiggy.plotter import SquigglePlotter

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
                html, _figure = SquigglePlotter.plot_multiple_reads(
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

        # Create large signal
        signal = np.random.randn(10000)
        downsample_factor = 10

        # Downsample
        downsampled = signal[::downsample_factor]

        assert len(downsampled) == len(signal) // downsample_factor

    def test_plot_with_downsampling(self, sample_pod5_file):
        """Test plot generation with downsampled signal."""
        import pod5

        from squiggy.plotter import SquigglePlotter

        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())

            # Plot with downsampling
            html, _figure = SquigglePlotter.plot_single_read(
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

        from squiggy.plotter import SquigglePlotter
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
                from squiggy.constants import BASE_COLORS

                result = SquigglePlotter._calculate_base_regions_time_mode(
                    sequence,
                    seq_to_sig_map,
                    time_ms,
                    signal,
                    signal_min,
                    signal_max,
                    sample_rate,
                    show_dwell_time=True,
                    base_colors=BASE_COLORS,
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
        from squiggy.plotter import SquigglePlotter

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
                                from squiggy.constants import BASE_COLORS

                                signal_min = float(np.min(signal))
                                signal_max = float(np.max(signal))

                                result = SquigglePlotter._calculate_base_regions_position_mode(
                                    aligned_read.bases,
                                    signal_min,
                                    signal_max,
                                    sample_rate,
                                    signal_length,
                                    show_dwell_time=True,
                                    base_colors=BASE_COLORS,
                                )

                                # Should return (base_regions,)
                                assert len(result) == 1
                                (base_regions,) = result

                                assert isinstance(base_regions, dict)

                                # Should have base type keys
                                assert all(
                                    base in ["A", "C", "G", "T", "U"]
                                    for base in base_regions.keys()
                                )

                                # Each region should have time-based coordinates (left != right)
                                total_regions = 0
                                for _base, regions in base_regions.items():
                                    for region in regions:
                                        assert "left" in region
                                        assert "right" in region
                                        assert (
                                            region["right"] > region["left"]
                                        )  # Time-scaled width
                                        total_regions += 1

                                # Should have some regions
                                assert total_regions > 0

                                return  # Test passed, exit

        pytest.skip("No aligned reads found in BAM file")

    def test_dwell_time_patches_created(self, sample_pod5_file, sample_bam_file):
        """Test that dwell time color patches are created."""
        import pod5

        from squiggy.plotter import SquigglePlotter
        from squiggy.utils import get_basecall_data

        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())
            read_id = str(read.read_id)

            sequence, seq_to_sig_map = get_basecall_data(sample_bam_file, read_id)

            if sequence is not None:
                # Plot with dwell time enabled
                html, _figure = SquigglePlotter.plot_single_read(
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
        from squiggy.plotter import SquigglePlotter

        reads_data = [("read_001", np.random.randn(1000), 4000)]

        title = SquigglePlotter._format_plot_title("Test Mode", reads_data)

        assert isinstance(title, str)
        assert "Test Mode" in title
        assert "1 read" in title.lower() or "read" in title.lower()

    def test_format_plot_title_multiple_reads(self):
        """Test plot title formatting for multiple reads."""
        from squiggy.plotter import SquigglePlotter

        reads_data = [
            ("read_001", np.random.randn(1000), 4000),
            ("read_002", np.random.randn(1000), 4000),
            ("read_003", np.random.randn(1000), 4000),
        ]

        title = SquigglePlotter._format_plot_title("Overlay", reads_data)

        assert isinstance(title, str)
        assert "Overlay" in title
        assert "3" in title or "three" in title.lower()

    def test_create_figure(self):
        """Test figure creation with proper labels."""
        from squiggy.plotter import SquigglePlotter

        fig = SquigglePlotter._create_figure(
            title="Test Plot", x_label="Time (ms)", y_label="Signal (pA)"
        )

        assert fig is not None
        assert fig.title.text == "Test Plot"
        assert fig.xaxis.axis_label == "Time (ms)"
        assert fig.yaxis.axis_label == "Signal (pA)"


class TestStrideAwareDwellTime:
    """Tests for stride-aware dwell time calculations in plotting."""

    def test_dwell_time_respects_stride(self, sample_pod5_file, sample_bam_file):
        """Test that dwell time calculations account for stride."""
        import pod5
        import pysam

        from squiggy.utils import get_basecall_data

        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())
            read_id = str(read.read_id)
            sample_rate = read.run_info.sample_rate

            sequence, seq_to_sig_map = get_basecall_data(sample_bam_file, read_id)

            if sequence is not None and len(sequence) > 1:
                # Check stride from BAM
                with pysam.AlignmentFile(
                    str(sample_bam_file), "rb", check_sq=False
                ) as bam:
                    for alignment in bam.fetch(until_eof=True):
                        if alignment.query_name == read_id and alignment.has_tag("mv"):
                            move_table = np.array(
                                alignment.get_tag("mv"), dtype=np.uint8
                            )
                            stride = int(move_table[0])

                            # Calculate dwell time manually for first base
                            if len(seq_to_sig_map) > 1:
                                signal_samples = seq_to_sig_map[1] - seq_to_sig_map[0]
                                dwell_time_ms = (signal_samples / sample_rate) * 1000

                                # Dwell time should be realistic (not microseconds)
                                # Before stride fix: ~0.2 ms (wrong)
                                # After stride fix: ~1-10 ms (correct)
                                assert dwell_time_ms >= 0.5, (
                                    f"Dwell time {dwell_time_ms} ms is too short. "
                                    f"Expected >= 0.5 ms with stride={stride}"
                                )
                                assert dwell_time_ms <= 50, (
                                    f"Dwell time {dwell_time_ms} ms is too long. "
                                    f"Expected <= 50 ms"
                                )

                                # Signal samples should be multiple of stride
                                assert signal_samples % stride == 0, (
                                    f"Signal samples {signal_samples} not multiple of stride {stride}"
                                )

                            return

            pytest.skip("No basecall data available for stride dwell time test")

    def test_realistic_dwell_times_in_plot(self, sample_pod5_file, sample_bam_file):
        """Test that plotted dwell times are in realistic range."""
        import pod5

        from squiggy.utils import get_basecall_data

        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())
            read_id = str(read.read_id)
            sample_rate = read.run_info.sample_rate

            sequence, seq_to_sig_map = get_basecall_data(sample_bam_file, read_id)

            if sequence is not None and len(seq_to_sig_map) > 10:
                # Calculate dwell times for first 10 bases
                dwell_times = []
                for i in range(min(10, len(seq_to_sig_map) - 1)):
                    signal_samples = seq_to_sig_map[i + 1] - seq_to_sig_map[i]
                    dwell_ms = (signal_samples / sample_rate) * 1000
                    dwell_times.append(dwell_ms)

                # Calculate mean dwell time
                mean_dwell = np.mean(dwell_times)

                # Mean dwell time should be in realistic range
                # Typical: 2-8 ms per base for DNA
                assert mean_dwell >= 0.5, (
                    f"Mean dwell time {mean_dwell:.2f} ms is too short. "
                    "This suggests stride is not being applied correctly."
                )
                assert mean_dwell <= 30, (
                    f"Mean dwell time {mean_dwell:.2f} ms is unusually high"
                )

                # All dwell times should be positive
                assert all(d > 0 for d in dwell_times)

                return

            pytest.skip("Not enough bases for realistic dwell time test")
