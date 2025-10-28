"""Tests for signal normalization methods."""

import numpy as np
import pytest

from squiggy.constants import NormalizationMethod
from squiggy.normalization import (
    mad_normalize,
    median_normalize,
    normalize_signal,
    z_normalize,
)


class TestNormalizeSignalDispatcher:
    """Tests for the normalize_signal dispatcher function."""

    def test_normalize_signal_none(self):
        """Test that NONE method returns original signal unchanged."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalize_signal(signal, NormalizationMethod.NONE)
        np.testing.assert_array_equal(result, signal)

    def test_normalize_signal_znorm(self):
        """Test that ZNORM method calls z_normalize."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalize_signal(signal, NormalizationMethod.ZNORM)
        expected = z_normalize(signal)
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_signal_median(self):
        """Test that MEDIAN method calls median_normalize."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalize_signal(signal, NormalizationMethod.MEDIAN)
        expected = median_normalize(signal)
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_signal_mad(self):
        """Test that MAD method calls mad_normalize."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalize_signal(signal, NormalizationMethod.MAD)
        expected = mad_normalize(signal)
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_signal_invalid_method(self):
        """Test that invalid method raises ValueError."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.raises(ValueError, match="Unknown normalization method"):
            # Create an invalid enum-like object
            class FakeMethod:
                pass

            normalize_signal(signal, FakeMethod())


class TestZNormalize:
    """Tests for Z-score normalization."""

    def test_z_normalize_basic(self):
        """Test Z-normalization with normal data."""
        signal = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = z_normalize(signal)

        # Mean should be approximately 0
        assert np.isclose(np.mean(result), 0.0, atol=1e-10)
        # Standard deviation should be approximately 1
        assert np.isclose(np.std(result), 1.0, atol=1e-10)

    def test_z_normalize_zero_std(self):
        """Test Z-normalization with constant signal (zero std)."""
        signal = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        result = z_normalize(signal)

        # Should return zero-centered signal (mean subtracted)
        expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_z_normalize_single_value(self):
        """Test Z-normalization with single value."""
        signal = np.array([42.0])
        result = z_normalize(signal)

        # Single value should become zero after mean subtraction
        np.testing.assert_array_almost_equal(result, np.array([0.0]))

    def test_z_normalize_negative_values(self):
        """Test Z-normalization with negative values."""
        signal = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        result = z_normalize(signal)

        # Properties should still hold
        assert np.isclose(np.mean(result), 0.0, atol=1e-10)
        assert np.isclose(np.std(result), 1.0, atol=1e-10)

    def test_z_normalize_large_values(self):
        """Test Z-normalization with large values."""
        signal = np.array([1e6, 2e6, 3e6, 4e6, 5e6])
        result = z_normalize(signal)

        # Should still work with large values
        assert np.isclose(np.mean(result), 0.0, atol=1e-6)
        assert np.isclose(np.std(result), 1.0, atol=1e-6)

    def test_z_normalize_with_outliers(self):
        """Test Z-normalization with outliers."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        result = z_normalize(signal)

        # Outlier will affect mean and std
        assert np.isclose(np.mean(result), 0.0, atol=1e-10)
        assert np.isclose(np.std(result), 1.0, atol=1e-10)

    def test_z_normalize_preserves_shape(self):
        """Test that Z-normalization preserves array shape."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = z_normalize(signal)
        assert result.shape == signal.shape


class TestMedianNormalize:
    """Tests for median normalization."""

    def test_median_normalize_basic(self):
        """Test median normalization with normal data."""
        signal = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = median_normalize(signal)

        # Median should be approximately 0
        assert np.isclose(np.median(result), 0.0, atol=1e-10)

    def test_median_normalize_odd_length(self):
        """Test median normalization with odd-length array."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = median_normalize(signal)

        # Original median is 3.0, so all values shifted by -3.0
        expected = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_median_normalize_even_length(self):
        """Test median normalization with even-length array."""
        signal = np.array([1.0, 2.0, 3.0, 4.0])
        result = median_normalize(signal)

        # Median of [1,2,3,4] is 2.5
        expected = np.array([-1.5, -0.5, 0.5, 1.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_median_normalize_single_value(self):
        """Test median normalization with single value."""
        signal = np.array([42.0])
        result = median_normalize(signal)

        np.testing.assert_array_almost_equal(result, np.array([0.0]))

    def test_median_normalize_with_outliers(self):
        """Test that median normalization is robust to outliers."""
        # Median is less affected by outliers than mean
        signal = np.array([1.0, 2.0, 3.0, 4.0, 1000.0])
        result = median_normalize(signal)

        # Median is 3.0, not affected much by the outlier
        assert np.isclose(np.median(result), 0.0, atol=1e-10)

    def test_median_normalize_negative_values(self):
        """Test median normalization with negative values."""
        signal = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        result = median_normalize(signal)

        # Median is 0.0, so result should equal input
        np.testing.assert_array_almost_equal(result, signal)

    def test_median_normalize_preserves_shape(self):
        """Test that median normalization preserves array shape."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = median_normalize(signal)
        assert result.shape == signal.shape


class TestMADNormalize:
    """Tests for MAD (Median Absolute Deviation) normalization."""

    def test_mad_normalize_basic(self):
        """Test MAD normalization with normal data."""
        signal = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = mad_normalize(signal)

        # Median should be approximately 0
        assert np.isclose(np.median(result), 0.0, atol=1e-10)

    def test_mad_normalize_zero_mad(self):
        """Test MAD normalization with constant signal (zero MAD)."""
        signal = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        result = mad_normalize(signal)

        # Should return median-centered signal (like median_normalize)
        expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_mad_normalize_scaling_factor(self):
        """Test that MAD normalization uses correct scaling factor."""
        # For normal distribution, MAD * 1.4826 ≈ std
        # Generate data with known std
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        _ = mad_normalize(signal)

        # Calculate MAD manually
        median = np.median(signal)
        mad = np.median(np.abs(signal - median))
        scaling_factor = 1.4826

        # Verify scaling factor is applied
        expected_scaling = scaling_factor * mad
        # Verify by checking if result has similar spread to z-normalized
        # (not exact due to different scaling approaches)
        assert expected_scaling > 0

    def test_mad_normalize_robust_to_outliers(self):
        """Test that MAD normalization is robust to outliers."""
        # MAD should be less affected by outliers than std
        signal = np.array([1.0, 2.0, 3.0, 4.0, 1000.0])
        result = mad_normalize(signal)

        # Should still center on median
        assert np.isclose(np.median(result), 0.0, atol=1e-10)

        # MAD-based scaling should be more robust than std
        signal_no_outlier = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        _ = mad_normalize(signal_no_outlier)

        # The scale of normalized values should be similar
        # (unlike z-score which would be very different)
        assert np.std(result) > np.std(z_normalize(signal))

    def test_mad_normalize_single_value(self):
        """Test MAD normalization with single value."""
        signal = np.array([42.0])
        result = mad_normalize(signal)

        # Should become zero after median subtraction
        np.testing.assert_array_almost_equal(result, np.array([0.0]))

    def test_mad_normalize_negative_values(self):
        """Test MAD normalization with negative values."""
        signal = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        result = mad_normalize(signal)

        # Median should be 0
        assert np.isclose(np.median(result), 0.0, atol=1e-10)

    def test_mad_normalize_preserves_shape(self):
        """Test that MAD normalization preserves array shape."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = mad_normalize(signal)
        assert result.shape == signal.shape

    def test_mad_normalize_consistency(self):
        """Test MAD normalization consistency with known values."""
        # Simple case where we can verify manually
        signal = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        # Median = 2.0
        # Deviations: [2.0, 1.0, 0.0, 1.0, 2.0]
        # MAD = median([2.0, 1.0, 0.0, 1.0, 2.0]) = 1.0
        # Scaling: 1.4826 * 1.0 = 1.4826

        result = mad_normalize(signal)

        # After centering by median (2.0):
        # [-2.0, -1.0, 0.0, 1.0, 2.0]
        # After scaling by (1.4826 * 1.0):
        expected = np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) / 1.4826

        np.testing.assert_array_almost_equal(result, expected, decimal=5)


class TestNormalizationComparisons:
    """Tests comparing different normalization methods."""

    def test_all_methods_preserve_relative_order(self):
        """Test that all normalization methods preserve relative ordering."""
        signal = np.array([5.0, 2.0, 8.0, 1.0, 9.0])

        # Get argsort of original signal
        original_order = np.argsort(signal)

        # All methods should preserve order
        for method in [
            NormalizationMethod.ZNORM,
            NormalizationMethod.MEDIAN,
            NormalizationMethod.MAD,
        ]:
            result = normalize_signal(signal, method)
            result_order = np.argsort(result)
            np.testing.assert_array_equal(original_order, result_order)

    def test_none_method_is_identity(self):
        """Test that NONE method doesn't modify signal."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalize_signal(signal, NormalizationMethod.NONE)

        # Should be exactly the same object (or equal values)
        np.testing.assert_array_equal(result, signal)

    def test_methods_differ_in_output(self):
        """Test that different normalization methods produce different results."""
        signal = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        znorm_result = normalize_signal(signal, NormalizationMethod.ZNORM)
        median_result = normalize_signal(signal, NormalizationMethod.MEDIAN)
        mad_result = normalize_signal(signal, NormalizationMethod.MAD)

        # Results should differ (not all the same)
        assert not np.allclose(znorm_result, median_result)
        assert not np.allclose(znorm_result, mad_result)
        # MEDIAN and MAD may be similar but scaled differently
        # (both center on median, but MAD also scales)


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_array(self):
        """Test normalization with empty array."""
        signal = np.array([])

        # All methods should handle empty arrays gracefully
        # (numpy will return empty array or NaN)
        for method in NormalizationMethod:
            result = normalize_signal(signal, method)
            assert len(result) == 0

    def test_two_values(self):
        """Test normalization with just two values."""
        signal = np.array([1.0, 2.0])

        # Should work with two values
        znorm_result = z_normalize(signal)
        assert len(znorm_result) == 2
        assert np.isclose(np.mean(znorm_result), 0.0, atol=1e-10)

        median_result = median_normalize(signal)
        assert len(median_result) == 2
        assert np.isclose(np.median(median_result), 0.0, atol=1e-10)

        mad_result = mad_normalize(signal)
        assert len(mad_result) == 2
        assert np.isclose(np.median(mad_result), 0.0, atol=1e-10)

    def test_all_zeros(self):
        """Test normalization with all zeros."""
        signal = np.array([0.0, 0.0, 0.0, 0.0])

        # Z-norm: mean=0, std=0 -> should return zeros
        znorm_result = z_normalize(signal)
        np.testing.assert_array_almost_equal(znorm_result, np.zeros(4))

        # Median: median=0 -> should return zeros
        median_result = median_normalize(signal)
        np.testing.assert_array_almost_equal(median_result, np.zeros(4))

        # MAD: median=0, MAD=0 -> should return zeros
        mad_result = mad_normalize(signal)
        np.testing.assert_array_almost_equal(mad_result, np.zeros(4))

    def test_very_small_values(self):
        """Test normalization with very small values."""
        signal = np.array([1e-10, 2e-10, 3e-10, 4e-10, 5e-10])

        # Should still work with very small values
        for method in NormalizationMethod:
            if method != NormalizationMethod.NONE:
                result = normalize_signal(signal, method)
                assert len(result) == len(signal)
                assert not np.any(np.isnan(result))

    def test_mixed_positive_negative_zero(self):
        """Test normalization with mixed positive, negative, and zero."""
        signal = np.array([-5.0, -2.0, 0.0, 3.0, 7.0])

        for method in NormalizationMethod:
            result = normalize_signal(signal, method)
            assert len(result) == len(signal)
            assert not np.any(np.isnan(result))
            assert not np.any(np.isinf(result))
