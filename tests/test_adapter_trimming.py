"""Tests for adapter trimming functionality"""

import numpy as np
import pytest

from squiggy.plotting import _apply_adapter_trimming_to_reads


class TestApplyAdapterTrimmingToReads:
    """Tests for _apply_adapter_trimming_to_reads function"""

    def test_no_soft_clipping_unchanged(self):
        """Reads without soft-clipping should be unchanged"""
        read = {
            "signal": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            "sequence": "ACGT",
            "move_table": np.array([1, 0, 1, 0, 1, 1], dtype=np.uint8),
            "stride": 1,
            "query_start_offset": 0,
            "query_end_offset": 0,
            "reference_start": 100,
            "reference_end": 104,
        }

        result = _apply_adapter_trimming_to_reads([read])

        assert len(result) == 1
        trimmed = result[0]
        assert np.array_equal(trimmed["signal"], read["signal"])
        assert trimmed["sequence"] == read["sequence"]
        assert trimmed["reference_start"] == read["reference_start"]
        assert trimmed["reference_end"] == read["reference_end"]

    def test_soft_clip_at_start_only(self):
        """Reads with soft-clipping at start should be trimmed correctly"""
        # 8 bases total: 2 soft-clipped at start, 6 aligned
        read = {
            "signal": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            "sequence": "NNACGTAC",  # NN are adapters, ACGTAC is aligned
            "move_table": np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8),
            "stride": 1,
            "query_start_offset": 2,
            "query_end_offset": 0,
            "reference_start": 100,
            "reference_end": 106,
        }

        result = _apply_adapter_trimming_to_reads([read])

        assert len(result) == 1
        trimmed = result[0]
        assert trimmed["sequence"] == "ACGTAC"
        assert trimmed["query_start_offset"] == 0
        assert trimmed["query_end_offset"] == 0
        assert trimmed["reference_start"] == 102  # Increased by 2
        assert trimmed["reference_end"] == 106  # Unchanged

    def test_soft_clip_at_end_only(self):
        """Reads with soft-clipping at end should be trimmed correctly"""
        # 8 bases total: 6 aligned, 2 soft-clipped at end
        read = {
            "signal": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            "sequence": "ACGTACNN",  # ACGTAC is aligned, NN are adapters
            "move_table": np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8),
            "stride": 1,
            "query_start_offset": 0,
            "query_end_offset": 2,
            "reference_start": 100,
            "reference_end": 106,
        }

        result = _apply_adapter_trimming_to_reads([read])

        assert len(result) == 1
        trimmed = result[0]
        assert trimmed["sequence"] == "ACGTAC"
        assert trimmed["query_start_offset"] == 0
        assert trimmed["query_end_offset"] == 0
        assert trimmed["reference_start"] == 100  # Unchanged
        assert trimmed["reference_end"] == 104  # Decreased by 2

    def test_soft_clip_at_both_ends(self):
        """Reads with soft-clipping at both ends should be trimmed correctly"""
        # 10 bases total: 2 soft-clipped at start, 6 aligned, 2 soft-clipped at end
        read = {
            "signal": np.array(list(range(20))),
            "sequence": "NNACGTACNN",  # NN...NN are adapters
            "move_table": np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8),
            "stride": 1,
            "query_start_offset": 2,
            "query_end_offset": 2,
            "reference_start": 100,
            "reference_end": 106,
        }

        result = _apply_adapter_trimming_to_reads([read])

        assert len(result) == 1
        trimmed = result[0]
        assert trimmed["sequence"] == "ACGTAC"
        assert trimmed["query_start_offset"] == 0
        assert trimmed["query_end_offset"] == 0
        assert trimmed["reference_start"] == 102  # Increased by 2
        assert trimmed["reference_end"] == 104  # Decreased by 2

    def test_quality_scores_trimmed(self):
        """Quality scores should be trimmed along with sequence"""
        read = {
            "signal": np.array(list(range(20))),
            "sequence": "NNACGTACNN",
            "quality_scores": np.array([10, 10, 30, 30, 30, 30, 30, 30, 10, 10]),
            "move_table": np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8),
            "stride": 1,
            "query_start_offset": 2,
            "query_end_offset": 2,
            "reference_start": 100,
            "reference_end": 106,
        }

        result = _apply_adapter_trimming_to_reads([read])
        trimmed = result[0]

        assert len(trimmed["quality_scores"]) == 6
        assert np.array_equal(trimmed["quality_scores"], [30, 30, 30, 30, 30, 30])

    def test_query_to_ref_adjusted(self):
        """query_to_ref mapping should be adjusted for trimmed coordinates"""
        read = {
            "signal": np.array(list(range(20))),
            "sequence": "NNACGTACNN",
            "move_table": np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8),
            "stride": 1,
            "query_start_offset": 2,
            "query_end_offset": 2,
            "reference_start": 100,
            "reference_end": 106,
            # Original mapping: query 2->102, 3->103, 4->104, 5->105, 6->106, 7->107
            "query_to_ref": {2: 102, 3: 103, 4: 104, 5: 105, 6: 106, 7: 107},
        }

        result = _apply_adapter_trimming_to_reads([read])
        trimmed = result[0]

        # After trimming, query indices should be renumbered starting from 0
        # New mapping: 0->102, 1->103, 2->104, 3->105, 4->106, 5->107
        assert 0 in trimmed["query_to_ref"]
        assert trimmed["query_to_ref"][0] == 102
        assert trimmed["query_to_ref"][5] == 107
        # Old keys should not exist
        assert 2 not in trimmed["query_to_ref"] or trimmed["query_to_ref"].get(2) == 104

    def test_move_table_base_count_matches_sequence(self):
        """After trimming, move table should have same base count as sequence"""
        # Create consistent data: 20 bases with move table matching sequence length
        # 4 soft-clipped at start, 12 aligned, 4 soft-clipped at end
        read = {
            "signal": np.array(list(range(40))),  # 40 signal samples
            "sequence": "NNNN" + "ACGT" * 3 + "NNNN",  # 4 + 12 + 4 = 20 bases
            "move_table": np.array([1, 0] * 20, dtype=np.uint8),  # 20 bases (1,0 pattern)
            "stride": 1,
            "query_start_offset": 4,
            "query_end_offset": 4,
            "reference_start": 100,
            "reference_end": 112,
        }

        result = _apply_adapter_trimming_to_reads([read])

        assert len(result) == 1
        trimmed = result[0]
        move_base_count = sum(trimmed["move_table"])
        seq_len = len(trimmed["sequence"])
        # After trimming, move table bases should match sequence length
        assert move_base_count == seq_len, f"Move table bases ({move_base_count}) != sequence length ({seq_len})"
        assert seq_len == 12  # 20 - 4 - 4 = 12 aligned bases

    def test_empty_list_returns_empty(self):
        """Empty input list should return empty output"""
        result = _apply_adapter_trimming_to_reads([])
        assert result == []

    def test_multiple_reads_processed(self):
        """Multiple reads should all be processed"""
        reads = [
            {
                "signal": np.array([1, 2, 3, 4, 5, 6]),
                "sequence": "ACGT",
                "move_table": np.array([1, 0, 1, 0, 1, 1], dtype=np.uint8),
                "stride": 1,
                "query_start_offset": 0,
                "query_end_offset": 0,
                "reference_start": 100,
                "reference_end": 104,
            },
            {
                "signal": np.array([1, 2, 3, 4, 5, 6, 7, 8]),
                "sequence": "NNACGT",  # 2 soft-clipped
                "move_table": np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8),
                "stride": 1,
                "query_start_offset": 2,
                "query_end_offset": 0,
                "reference_start": 100,
                "reference_end": 104,
            },
        ]

        result = _apply_adapter_trimming_to_reads(reads)

        assert len(result) == 2
        # First read unchanged
        assert result[0]["sequence"] == "ACGT"
        # Second read trimmed
        assert result[1]["sequence"] == "ACGT"
        assert result[1]["reference_start"] == 102

    def test_read_without_signal_skipped(self):
        """Reads without signal should be skipped"""
        read = {
            "signal": None,
            "sequence": "ACGT",
            "move_table": np.array([1, 1, 1, 1], dtype=np.uint8),
            "stride": 1,
            "query_start_offset": 1,
            "query_end_offset": 1,
        }

        result = _apply_adapter_trimming_to_reads([read])

        assert len(result) == 0

    def test_signal_trimmed_correctly(self):
        """Signal should be trimmed based on move table base positions"""
        # Create a read where each base corresponds to 2 signal samples (stride=2)
        # 4 bases: A(soft), C, G, T(soft) -> signal indices 0-1, 2-3, 4-5, 6-7
        read = {
            "signal": np.array([1, 1, 2, 2, 3, 3, 4, 4]),  # 8 samples
            "sequence": "ACGT",
            "move_table": np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8),  # 4 bases
            "stride": 1,  # stride=1 means each move entry = 1 sample
            "query_start_offset": 1,  # First base soft-clipped
            "query_end_offset": 1,  # Last base soft-clipped
            "reference_start": 100,
            "reference_end": 102,
        }

        result = _apply_adapter_trimming_to_reads([read])

        assert len(result) == 1
        trimmed = result[0]
        # Should have middle 2 bases worth of signal
        assert trimmed["sequence"] == "CG"
        # Signal should be trimmed to correspond to C and G bases
        assert len(trimmed["signal"]) > 0


class TestAdapterTrimmingIntegration:
    """Integration tests for adapter trimming with real data"""

    def test_trimming_with_real_data(self, sample_pod5_file, indexed_bam_file):
        """Test adapter trimming with actual POD5/BAM data"""
        from squiggy.utils.bam import extract_reads_for_reference

        # Get reads
        reads_data = extract_reads_for_reference(
            pod5_file=str(sample_pod5_file),
            bam_file=str(indexed_bam_file),
            reference_name=None,  # Get all reads
            max_reads=10,
        )

        if not reads_data:
            pytest.skip("No reads with alignments found in test data")

        # Check if any reads have soft-clipping
        reads_with_clips = [
            r
            for r in reads_data
            if r.get("query_start_offset", 0) > 0 or r.get("query_end_offset", 0) > 0
        ]

        if not reads_with_clips:
            pytest.skip("No reads with soft-clipping in test data")

        # Apply trimming
        trimmed_reads = _apply_adapter_trimming_to_reads(reads_data)

        # Verify all soft-clip offsets are now 0
        for trimmed in trimmed_reads:
            assert trimmed.get("query_start_offset", 0) == 0
            assert trimmed.get("query_end_offset", 0) == 0

    def test_trimming_preserves_aligned_region(self, sample_pod5_file, indexed_bam_file):
        """Test that trimming preserves the aligned region correctly"""
        from squiggy.utils.bam import extract_reads_for_reference

        reads_data = extract_reads_for_reference(
            pod5_file=str(sample_pod5_file),
            bam_file=str(indexed_bam_file),
            reference_name=None,
            max_reads=5,
        )

        if not reads_data:
            pytest.skip("No reads with alignments found in test data")

        for original in reads_data:
            soft_start = original.get("query_start_offset", 0)
            soft_end = original.get("query_end_offset", 0)

            if soft_start == 0 and soft_end == 0:
                continue

            trimmed_list = _apply_adapter_trimming_to_reads([original])
            if not trimmed_list:
                continue

            trimmed = trimmed_list[0]

            # Reference start should increase by soft_start
            expected_ref_start = original.get("reference_start", 0) + soft_start
            assert trimmed.get("reference_start") == expected_ref_start

            # Reference end should decrease by soft_end
            expected_ref_end = original.get("reference_end", 0) - soft_end
            assert trimmed.get("reference_end") == expected_ref_end
