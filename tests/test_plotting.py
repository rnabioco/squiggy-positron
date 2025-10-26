"""Tests for plotting functionality."""
import pytest
import numpy as np
from io import BytesIO


class TestPlotGeneration:
    """Tests for plot generation and rendering."""

    def test_plot_to_png_buffer(self, sample_pod5_file):
        """Test that plots can be saved to PNG buffers."""
        from squiggy.main import SquigglePlotter
        import pod5

        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())

            plot = SquigglePlotter.create_plot(
                signal=read.signal,
                sample_rate=read.sample_rate,
                read_id=str(read.read_id)
            )

            # Save to buffer
            buffer = BytesIO()
            plot.save(buffer, format='png', dpi=100)

            # Verify buffer has content
            assert buffer.tell() > 0
            buffer.seek(0)

            # Verify it's a valid PNG
            png_signature = buffer.read(8)
            assert png_signature == b'\x89PNG\r\n\x1a\n'

    def test_plot_with_subsampled_signal(self, sample_pod5_file):
        """Test plotting with subsampled signal data."""
        from squiggy.main import SquigglePlotter
        import pod5

        with pod5.Reader(sample_pod5_file) as reader:
            read = next(reader.reads())

            # Subsample the signal (every 10th point)
            subsampled_signal = read.signal[::10]

            plot = SquigglePlotter.create_plot(
                signal=subsampled_signal,
                sample_rate=read.sample_rate,
                read_id=str(read.read_id)
            )

            assert plot is not None

    def test_signal_dataframe_time_calculation(self):
        """Test that time values are calculated correctly."""
        from squiggy.main import SquigglePlotter

        # Create synthetic signal
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sample_rate = 4000  # 4kHz

        df = SquigglePlotter.signal_to_dataframe(signal, sample_rate)

        # Verify time values
        expected_times = np.arange(len(signal)) / sample_rate
        np.testing.assert_array_almost_equal(df['time'].values, expected_times)

        # Verify signal values
        np.testing.assert_array_equal(df['signal'].values, signal)
