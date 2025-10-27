"""Tests for plotting functionality."""

from io import BytesIO

import numpy as np


class TestPlotGeneration:
    """Tests for plot generation and rendering."""

    def test_plot_to_png_buffer(self, sample_pod5_file):
        """Test that plots can be saved to PNG buffers."""
        import pod5

        from squiggy.plotter import SquigglePlotter

        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())

            plot = SquigglePlotter.create_plot(
                signal=read.signal,
                sample_rate=read.run_info.sample_rate,
                read_id=str(read.read_id),
            )

            # Save to buffer
            buffer = BytesIO()
            plot.save(buffer, format="png", dpi=100)

            # Verify buffer has content
            assert buffer.tell() > 0
            buffer.seek(0)

            # Verify it's a valid PNG
            png_signature = buffer.read(8)
            assert png_signature == b"\x89PNG\r\n\x1a\n"

    def test_plot_with_subsampled_signal(self, sample_pod5_file):
        """Test plotting with subsampled signal data."""
        import pod5

        from squiggy.plotter import SquigglePlotter

        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())

            # Subsample the signal (every 10th point)
            subsampled_signal = read.signal[::10]

            plot = SquigglePlotter.create_plot(
                signal=subsampled_signal,
                sample_rate=read.run_info.sample_rate,
                read_id=str(read.read_id),
            )

            assert plot is not None

    def test_signal_dataframe_time_calculation(self):
        """Test that time values are calculated correctly."""
        from squiggy.plotter import SquigglePlotter

        # Create synthetic signal
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sample_rate = 4000  # 4kHz

        df = SquigglePlotter.signal_to_dataframe(signal, sample_rate)

        # Verify time values
        expected_times = np.arange(len(signal)) / sample_rate
        np.testing.assert_array_almost_equal(df["time"].values, expected_times)

        # Verify signal values
        np.testing.assert_array_equal(df["signal"].values, signal)

    def test_multi_read_overlay_plot(self, sample_pod5_file):
        """Test overlay plotting with multiple reads."""
        import pod5

        from squiggy.constants import NormalizationMethod, PlotMode
        from squiggy.plotter import SquigglePlotter

        with pod5.Reader(sample_pod5_file) as reader:
            reads = list(reader.reads())[:3]  # Get first 3 reads

            reads_data = [
                (str(r.read_id), r.signal, r.run_info.sample_rate) for r in reads
            ]

            plot = SquigglePlotter.plot_multiple_reads(
                reads_data,
                mode=PlotMode.OVERLAY,
                normalization=NormalizationMethod.ZNORM,
            )

            assert plot is not None

            # Save to buffer to verify it renders
            buffer = BytesIO()
            plot.save(buffer, format="png", dpi=100)
            assert buffer.tell() > 0

    def test_multi_read_stacked_plot(self, sample_pod5_file):
        """Test stacked plotting with multiple reads."""
        import pod5

        from squiggy.constants import NormalizationMethod, PlotMode
        from squiggy.plotter import SquigglePlotter

        with pod5.Reader(sample_pod5_file) as reader:
            reads = list(reader.reads())[:3]  # Get first 3 reads

            reads_data = [
                (str(r.read_id), r.signal, r.run_info.sample_rate) for r in reads
            ]

            plot = SquigglePlotter.plot_multiple_reads(
                reads_data,
                mode=PlotMode.STACKED,
                normalization=NormalizationMethod.MAD,
            )

            assert plot is not None

            # Save to buffer to verify it renders
            buffer = BytesIO()
            plot.save(buffer, format="png", dpi=100)
            assert buffer.tell() > 0

    def test_signal_normalization_methods(self):
        """Test different normalization methods."""
        from squiggy.constants import NormalizationMethod
        from squiggy.normalization import normalize_signal

        # Create synthetic signal
        signal = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        # Test z-norm
        znorm = normalize_signal(signal, NormalizationMethod.ZNORM)
        assert np.isclose(np.mean(znorm), 0.0, atol=1e-10)
        assert np.isclose(np.std(znorm), 1.0, atol=1e-10)

        # Test median
        median_norm = normalize_signal(signal, NormalizationMethod.MEDIAN)
        assert np.isclose(np.median(median_norm), 0.0, atol=1e-10)

        # Test MAD
        mad_norm = normalize_signal(signal, NormalizationMethod.MAD)
        assert mad_norm is not None

        # Test none
        no_norm = normalize_signal(signal, NormalizationMethod.NONE)
        np.testing.assert_array_equal(no_norm, signal)

    def test_downsampling_functionality(self):
        """Test that downsampling reduces data points correctly."""
        from squiggy.plotter import SquigglePlotter
        from squiggy.utils import downsample_signal

        # Create large synthetic signal (above MIN_POINTS_FOR_DOWNSAMPLING)
        signal = np.random.randn(50000)
        sample_rate = 4000

        # Test manual downsampling
        downsampled = downsample_signal(signal, downsample_factor=100)
        assert len(downsampled) == len(signal) // 100

        # Test automatic downsampling in signal_to_dataframe
        df_downsampled = SquigglePlotter.signal_to_dataframe(
            signal, sample_rate, downsample_factor=100
        )
        assert len(df_downsampled) == len(signal) // 100

        # Test that small signals are not downsampled automatically
        small_signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        df_small = SquigglePlotter.signal_to_dataframe(small_signal, sample_rate)
        assert len(df_small) == len(small_signal)

        # Test auto-downsampling for large signals
        df_auto = SquigglePlotter.signal_to_dataframe(signal, sample_rate)
        assert len(df_auto) < len(signal)  # Should be downsampled automatically
