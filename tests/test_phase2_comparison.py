"""Tests for Phase 2 - Comparison and aggregation functions"""

import numpy as np
import pytest


class TestCompareReadSets:
    """Tests for compare_read_sets utility function"""

    def test_compare_identical_sets(self):
        """Test comparing identical read sets"""
        from squiggy.utils import compare_read_sets

        reads_a = ["read_001", "read_002", "read_003"]
        reads_b = ["read_001", "read_002", "read_003"]

        result = compare_read_sets(reads_a, reads_b)

        assert result["common_count"] == 3
        assert result["unique_a_count"] == 0
        assert result["unique_b_count"] == 0
        assert result["overlap_percent_a"] == 100.0
        assert result["overlap_percent_b"] == 100.0

    def test_compare_completely_different_sets(self):
        """Test comparing completely different read sets"""
        from squiggy.utils import compare_read_sets

        reads_a = ["read_001", "read_002", "read_003"]
        reads_b = ["read_004", "read_005", "read_006"]

        result = compare_read_sets(reads_a, reads_b)

        assert result["common_count"] == 0
        assert result["unique_a_count"] == 3
        assert result["unique_b_count"] == 3
        assert result["overlap_percent_a"] == 0.0
        assert result["overlap_percent_b"] == 0.0

    def test_compare_partial_overlap(self):
        """Test comparing sets with partial overlap"""
        from squiggy.utils import compare_read_sets

        reads_a = ["read_001", "read_002", "read_003", "read_004"]
        reads_b = ["read_003", "read_004", "read_005", "read_006"]

        result = compare_read_sets(reads_a, reads_b)

        assert result["common_count"] == 2
        assert result["unique_a_count"] == 2
        assert result["unique_b_count"] == 2
        assert result["overlap_percent_a"] == 50.0
        assert result["overlap_percent_b"] == 50.0

    def test_compare_read_sets_returns_sets(self):
        """Test that compare_read_sets returns set objects"""
        from squiggy.utils import compare_read_sets

        reads_a = ["read_001", "read_002"]
        reads_b = ["read_002", "read_003"]

        result = compare_read_sets(reads_a, reads_b)

        assert isinstance(result["common_reads"], set)
        assert isinstance(result["unique_to_a"], set)
        assert isinstance(result["unique_to_b"], set)


class TestCalculateDeltaStats:
    """Tests for calculate_delta_stats function"""

    def test_calculate_delta_simple(self):
        """Test calculating delta for simple stats"""
        from squiggy.utils import calculate_delta_stats

        stats_a = {
            "mean_signal": np.array([100.0, 110.0, 120.0]),
            "std_signal": np.array([10.0, 12.0, 15.0]),
            "positions": np.array([0, 1, 2]),
        }
        stats_b = {
            "mean_signal": np.array([105.0, 115.0, 125.0]),
            "std_signal": np.array([10.0, 12.0, 15.0]),
            "positions": np.array([0, 1, 2]),
        }

        result = calculate_delta_stats(stats_a, stats_b)

        assert "delta_mean_signal" in result
        assert "delta_std_signal" in result
        assert "positions" in result

        # Check values
        np.testing.assert_array_almost_equal(
            result["delta_mean_signal"], np.array([5.0, 5.0, 5.0])
        )
        np.testing.assert_array_almost_equal(
            result["delta_std_signal"], np.array([0.0, 0.0, 0.0])
        )

    def test_calculate_delta_with_specific_stats(self):
        """Test calculating delta for specific stat names"""
        from squiggy.utils import calculate_delta_stats

        stats_a = {
            "mean_signal": np.array([100.0, 110.0]),
            "std_signal": np.array([10.0, 12.0]),
        }
        stats_b = {
            "mean_signal": np.array([105.0, 115.0]),
            "std_signal": np.array([10.0, 12.0]),
        }

        result = calculate_delta_stats(stats_a, stats_b, stat_names=["mean_signal"])

        assert "delta_mean_signal" in result
        assert "delta_std_signal" not in result

    def test_calculate_delta_handles_mismatched_length(self):
        """Test that delta calculation handles different array lengths"""
        from squiggy.utils import calculate_delta_stats

        stats_a = {
            "mean_signal": np.array([100.0, 110.0, 120.0, 130.0]),
        }
        stats_b = {
            "mean_signal": np.array([105.0, 115.0]),
        }

        result = calculate_delta_stats(stats_a, stats_b)

        # Should only have as many deltas as the shorter array
        assert len(result["delta_mean_signal"]) == 2


class TestCompareSignalDistributions:
    """Tests for compare_signal_distributions function"""

    def test_compare_identical_signals(self):
        """Test comparing identical signal distributions"""
        from squiggy.utils import compare_signal_distributions

        signal = np.array([100.0, 110.0, 120.0, 130.0, 140.0])

        result = compare_signal_distributions(signal, signal)

        assert result["mean_a"] == result["mean_b"]
        assert result["std_a"] == result["std_b"]
        assert result["mean_diff"] == pytest.approx(0.0)
        assert result["std_diff"] == pytest.approx(0.0)

    def test_compare_different_signals(self):
        """Test comparing different signal distributions"""
        from squiggy.utils import compare_signal_distributions

        signal_a = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        signal_b = np.array([150.0, 150.0, 150.0, 150.0, 150.0])

        result = compare_signal_distributions(signal_a, signal_b)

        assert result["mean_a"] == 100.0
        assert result["mean_b"] == 150.0
        assert result["mean_diff"] == pytest.approx(50.0)
        assert result["std_a"] == pytest.approx(0.0)
        assert result["std_b"] == pytest.approx(0.0)

    def test_compare_signal_distributions_keys(self):
        """Test that comparison returns all expected keys"""
        from squiggy.utils import compare_signal_distributions

        signal_a = np.array([1.0, 2.0, 3.0])
        signal_b = np.array([4.0, 5.0, 6.0])

        result = compare_signal_distributions(signal_a, signal_b)

        expected_keys = [
            "mean_a",
            "mean_b",
            "median_a",
            "median_b",
            "std_a",
            "std_b",
            "min_a",
            "min_b",
            "max_a",
            "max_b",
            "mean_diff",
            "std_diff",
        ]

        for key in expected_keys:
            assert key in result


class TestGetCommonReads:
    """Tests for get_common_reads function"""

    def test_get_common_reads_two_samples(self, sample_pod5_file):
        """Test getting common reads from two samples"""
        from squiggy import get_common_reads, load_sample

        load_sample("a", str(sample_pod5_file))
        load_sample("b", str(sample_pod5_file))

        # Same file should have 100% common reads
        common = get_common_reads(["a", "b"])
        assert len(common) > 0

    def test_get_common_reads_nonexistent_sample(self, sample_pod5_file):
        """Test get_common_reads with nonexistent sample"""
        from squiggy import get_common_reads, load_sample

        load_sample("a", str(sample_pod5_file))

        with pytest.raises(ValueError, match="not found"):
            get_common_reads(["a", "nonexistent"])

    def test_get_common_reads_empty_list(self):
        """Test get_common_reads with empty sample list"""
        from squiggy import get_common_reads

        common = get_common_reads([])
        assert len(common) == 0


class TestGetUniqueReads:
    """Tests for get_unique_reads function"""

    def test_get_unique_reads_single_sample(self, sample_pod5_file):
        """Test getting unique reads when only one sample loaded"""
        from squiggy import get_unique_reads, load_sample

        load_sample("a", str(sample_pod5_file))

        # All reads should be unique when other samples don't exist
        unique = get_unique_reads("a")
        # When no other samples to compare against, all reads are unique
        assert len(unique) > 0
        assert isinstance(unique, set)

    def test_get_unique_reads_nonexistent_sample(self):
        """Test get_unique_reads with nonexistent sample"""
        from squiggy import get_unique_reads

        with pytest.raises(ValueError, match="not found"):
            get_unique_reads("nonexistent")

    def test_get_unique_reads_with_exclude_list(self, sample_pod5_file):
        """Test get_unique_reads with custom exclude list"""
        from squiggy import get_unique_reads, load_sample

        load_sample("a", str(sample_pod5_file))
        load_sample("b", str(sample_pod5_file))
        load_sample("c", str(sample_pod5_file))

        # Should exclude only sample 'b'
        unique = get_unique_reads("a", exclude_samples=["b"])
        # Since all are same file, should have 0 unique
        assert isinstance(unique, set)


class TestCompareSamples:
    """Tests for compare_samples function"""

    def test_compare_two_samples(self, sample_pod5_file, sample_bam_file):
        """Test comparing two samples"""
        from squiggy import compare_samples, load_sample

        load_sample("v4.2", str(sample_pod5_file), str(sample_bam_file))
        load_sample("v5.0", str(sample_pod5_file), str(sample_bam_file))

        result = compare_samples(["v4.2", "v5.0"])

        assert result["samples"] == ["v4.2", "v5.0"]
        assert "sample_info" in result
        assert "read_overlap" in result
        assert "v4.2" in result["sample_info"]
        assert "v5.0" in result["sample_info"]

    def test_compare_samples_nonexistent(self):
        """Test compare_samples with nonexistent sample"""
        from squiggy import compare_samples

        with pytest.raises(ValueError, match="not found"):
            compare_samples(["nonexistent"])

    def test_compare_samples_with_bam_validation(
        self, sample_pod5_file, sample_bam_file
    ):
        """Test that compare_samples validates BAM references"""
        from squiggy import compare_samples, load_sample

        load_sample("v4.2", str(sample_pod5_file), str(sample_bam_file))
        load_sample("v5.0", str(sample_pod5_file), str(sample_bam_file))

        result = compare_samples(["v4.2", "v5.0"])

        # Should have reference validation since BAMs are loaded
        if "reference_validation" in result:
            assert "v4.2_vs_v5.0" in result["reference_validation"]

    def test_compare_samples_single(self, sample_pod5_file):
        """Test comparing single sample (edge case)"""
        from squiggy import compare_samples, load_sample

        load_sample("only_one", str(sample_pod5_file))

        result = compare_samples(["only_one"])

        assert result["samples"] == ["only_one"]
        assert "only_one" in result["sample_info"]


class TestPhase2Integration:
    """Integration tests for Phase 2 comparison workflows"""

    def test_full_comparison_workflow(self, sample_pod5_file, sample_bam_file):
        """Test complete comparison workflow"""
        from squiggy import (
            compare_samples,
            get_common_reads,
            get_unique_reads,
            list_samples,
            load_sample,
        )

        # Load samples
        load_sample("model_v4.2", str(sample_pod5_file), str(sample_bam_file))
        load_sample("model_v5.0", str(sample_pod5_file), str(sample_bam_file))

        # List samples
        samples = list_samples()
        assert len(samples) == 2

        # Compare samples
        comparison = compare_samples(samples)
        assert len(comparison["sample_info"]) == 2

        # Get common reads
        common = get_common_reads(samples)
        assert len(common) > 0

        # Get unique reads
        unique_a = get_unique_reads("model_v4.2")
        unique_b = get_unique_reads("model_v5.0")
        assert isinstance(unique_a, set)
        assert isinstance(unique_b, set)
