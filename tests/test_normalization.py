"""Tests for signal normalization functions"""

import numpy as np
import pytest


class TestNormalizeSignal:
    """Tests for normalize_signal dispatcher function"""

    def test_normalize_signal_none(self):
        """Test that NONE method returns unchanged signal"""
        from squiggy.constants import NormalizationMethod
        from squiggy.normalization import normalize_signal

        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalize_signal(signal, NormalizationMethod.NONE)

        np.testing.assert_array_equal(result, signal)

    def test_normalize_signal_znorm(self):
        """Test that ZNORM method calls z_normalize"""
        from squiggy.constants import NormalizationMethod
        from squiggy.normalization import normalize_signal

        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalize_signal(signal, NormalizationMethod.ZNORM)

        # Should have mean=0 and std=1
        assert np.abs(np.mean(result)) < 1e-10
        assert np.abs(np.std(result) - 1.0) < 1e-10

    def test_normalize_signal_median(self):
        """Test that MEDIAN method calls median_normalize"""
        from squiggy.constants import NormalizationMethod
        from squiggy.normalization import normalize_signal

        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalize_signal(signal, NormalizationMethod.MEDIAN)

        # Should have median=0
        assert np.median(result) == 0.0

    def test_normalize_signal_mad(self):
        """Test that MAD method calls mad_normalize"""
        from squiggy.constants import NormalizationMethod
        from squiggy.normalization import normalize_signal

        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalize_signal(signal, NormalizationMethod.MAD)

        # Should have median=0
        assert np.median(result) == 0.0

    def test_normalize_signal_invalid_method(self):
        """Test that invalid method raises ValueError"""
        from squiggy.normalization import normalize_signal

        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalize_signal(signal, "invalid_method")


class TestZNormalize:
    """Tests for z-score normalization"""

    def test_z_normalize_basic(self):
        """Test basic z-score normalization"""
        from squiggy.normalization import z_normalize

        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = z_normalize(signal)

        # Mean should be ~0, std should be ~1
        assert np.abs(np.mean(result)) < 1e-10
        assert np.abs(np.std(result) - 1.0) < 1e-10

    def test_z_normalize_zero_std(self):
        """Test z-normalize with zero standard deviation (constant signal)"""
        from squiggy.normalization import z_normalize

        signal = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        result = z_normalize(signal)

        # Should subtract mean but not divide (std=0)
        expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_z_normalize_negative_values(self):
        """Test z-normalize with negative values"""
        from squiggy.normalization import z_normalize

        signal = np.array([-5.0, -2.0, 0.0, 2.0, 5.0])
        result = z_normalize(signal)

        # Mean should be ~0, std should be ~1
        assert np.abs(np.mean(result)) < 1e-10
        assert np.abs(np.std(result) - 1.0) < 1e-10

    def test_z_normalize_single_value(self):
        """Test z-normalize with single value"""
        from squiggy.normalization import z_normalize

        signal = np.array([42.0])
        result = z_normalize(signal)

        # Single value should become 0 (std is 0)
        assert result[0] == 0.0

    def test_z_normalize_large_signal(self):
        """Test z-normalize with large array"""
        from squiggy.normalization import z_normalize

        # Create signal with known statistics
        np.random.seed(42)
        signal = np.random.randn(10000) * 100 + 500  # mean=500, std=100

        result = z_normalize(signal)

        # Should be approximately standard normal
        assert np.abs(np.mean(result)) < 0.1
        assert np.abs(np.std(result) - 1.0) < 0.1


class TestMedianNormalize:
    """Tests for median-centered normalization"""

    def test_median_normalize_basic(self):
        """Test basic median normalization"""
        from squiggy.normalization import median_normalize

        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = median_normalize(signal)

        # Median should be 0
        assert np.median(result) == 0.0
        # Values should be shifted by median (3.0)
        expected = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        np.testing.assert_array_equal(result, expected)

    def test_median_normalize_even_length(self):
        """Test median normalization with even-length array"""
        from squiggy.normalization import median_normalize

        signal = np.array([1.0, 2.0, 4.0, 5.0])
        result = median_normalize(signal)

        # Median of [1,2,4,5] is 3.0
        assert np.median(result) == 0.0
        expected = np.array([-2.0, -1.0, 1.0, 2.0])
        np.testing.assert_array_equal(result, expected)

    def test_median_normalize_with_outliers(self):
        """Test median normalization is robust to outliers"""
        from squiggy.normalization import median_normalize

        signal = np.array([1.0, 2.0, 3.0, 4.0, 1000.0])  # Large outlier
        result = median_normalize(signal)

        # Median should still be 0 (centered on 3.0, not affected by outlier)
        assert np.median(result) == 0.0

    def test_median_normalize_negative_values(self):
        """Test median normalization with negative values"""
        from squiggy.normalization import median_normalize

        signal = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        result = median_normalize(signal)

        # Median is 0, so result should equal input
        np.testing.assert_array_equal(result, signal)

    def test_median_normalize_single_value(self):
        """Test median normalization with single value"""
        from squiggy.normalization import median_normalize

        signal = np.array([42.0])
        result = median_normalize(signal)

        # Single value becomes 0
        assert result[0] == 0.0


class TestMADNormalize:
    """Tests for median absolute deviation (MAD) normalization"""

    def test_mad_normalize_basic(self):
        """Test basic MAD normalization"""
        from squiggy.normalization import mad_normalize

        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = mad_normalize(signal)

        # Median should be 0
        assert np.median(result) == 0.0
        # MAD should be scaled appropriately
        # Original: median=3, MAD=median(|[1,2,3,4,5]-3|)=median([2,1,0,1,2])=1
        # Scaling factor: 1.4826
        # Result: (signal - 3) / (1.4826 * 1)

    def test_mad_normalize_zero_mad(self):
        """Test MAD normalization with zero MAD (constant signal)"""
        from squiggy.normalization import mad_normalize

        signal = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        result = mad_normalize(signal)

        # Should subtract median but not divide (MAD=0)
        expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_mad_normalize_with_outliers(self):
        """Test MAD normalization is robust to outliers"""
        from squiggy.normalization import mad_normalize

        # Signal with outlier
        signal = np.array([1.0, 2.0, 3.0, 4.0, 1000.0])
        result = mad_normalize(signal)

        # Median should be 0, and scale should not be dominated by outlier
        assert np.median(result) == 0.0
        # MAD is robust, so first 4 values should have reasonable magnitudes
        assert np.max(np.abs(result[:4])) < 10  # Not huge values

    def test_mad_normalize_negative_values(self):
        """Test MAD normalization with negative values"""
        from squiggy.normalization import mad_normalize

        signal = np.array([-5.0, -2.0, 0.0, 2.0, 5.0])
        result = mad_normalize(signal)

        # Median is 0
        assert np.median(result) == 0.0

    def test_mad_normalize_single_value(self):
        """Test MAD normalization with single value"""
        from squiggy.normalization import mad_normalize

        signal = np.array([42.0])
        result = mad_normalize(signal)

        # Single value should become 0 (MAD is 0)
        assert result[0] == 0.0

    def test_mad_normalize_scaling_factor(self):
        """Test that MAD uses correct scaling factor (1.4826)"""
        from squiggy.normalization import mad_normalize

        # Create signal where we can calculate MAD manually
        signal = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        # Median = 2.0
        # Deviations from median: [-2, -1, 0, 1, 2]
        # Absolute deviations: [2, 1, 0, 1, 2]
        # MAD = median([2, 1, 0, 1, 2]) = 1.0
        # Scaling: 1.4826 * 1.0 = 1.4826
        # Expected result: (signal - 2.0) / 1.4826

        result = mad_normalize(signal)

        expected = (signal - 2.0) / 1.4826
        np.testing.assert_array_almost_equal(result, expected, decimal=10)

    def test_mad_normalize_large_signal(self):
        """Test MAD normalization with large array"""
        from squiggy.normalization import mad_normalize

        # Create signal with known properties
        np.random.seed(42)
        signal = np.random.randn(10000) * 100 + 500

        result = mad_normalize(signal)

        # Median should be ~0
        assert np.abs(np.median(result)) < 0.1
        # For normal distribution, MAD-normalized should have similar scale to z-score
        assert 0.5 < np.std(result) < 1.5


class TestNormalizationEdgeCases:
    """Tests for edge cases and special scenarios"""

    def test_empty_array(self):
        """Test normalization functions with empty arrays"""
        from squiggy.normalization import (
            mad_normalize,
            median_normalize,
            z_normalize,
        )

        signal = np.array([])

        # All should handle empty arrays gracefully
        # NumPy will return NaN for empty arrays in most cases
        z_result = z_normalize(signal)
        assert len(z_result) == 0

        median_result = median_normalize(signal)
        assert len(median_result) == 0

        mad_result = mad_normalize(signal)
        assert len(mad_result) == 0

    def test_nan_handling(self):
        """Test normalization with NaN values"""
        from squiggy.normalization import z_normalize

        signal = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = z_normalize(signal)

        # Result will contain NaN, which is expected behavior
        assert np.isnan(result[2])

    def test_inf_handling(self):
        """Test normalization with infinite values"""
        from squiggy.normalization import z_normalize

        signal = np.array([1.0, 2.0, np.inf, 4.0, 5.0])
        result = z_normalize(signal)

        # NumPy will produce inf or nan, which is expected
        assert np.isinf(result[2]) or np.isnan(result[2])

    def test_very_large_values(self):
        """Test normalization with very large values"""
        from squiggy.normalization import z_normalize

        signal = np.array([1e10, 1e10 + 1, 1e10 + 2, 1e10 + 3, 1e10 + 4])
        result = z_normalize(signal)

        # Should still produce reasonable normalized values
        assert np.abs(np.mean(result)) < 1e-10
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
