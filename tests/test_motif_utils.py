"""Tests for motif-related utility functions in squiggy/utils.py"""

import pytest

from squiggy.motif import MotifMatch
from squiggy.utils import align_reads_to_motif_center, extract_reads_for_motif


@pytest.fixture
def test_data_paths(test_data_dir):
    """Paths to test data files"""
    return {
        "pod5": test_data_dir / "yeast_trna_reads.pod5",
        "bam": test_data_dir / "yeast_trna_mappings.bam",
        "fasta": test_data_dir / "yeast_trna.fa",
    }


class TestExtractReadsForMotif:
    """Tests for extract_reads_for_motif() function"""

    def test_basic_extraction(self, test_data_paths):
        """Test basic read extraction at a motif position"""
        result, motif_match = extract_reads_for_motif(
            pod5_file=str(test_data_paths["pod5"]),
            bam_file=str(test_data_paths["bam"]),
            fasta_file=str(test_data_paths["fasta"]),
            motif="DRACH",
            match_index=0,
            window=50,
            max_reads=10,
        )

        # Should return a list of read data dictionaries
        assert isinstance(result, list)
        assert len(result) <= 10  # Respects max_reads

        # Should return a MotifMatch
        assert isinstance(motif_match, MotifMatch)
        assert motif_match.sequence[0] in ["A", "G", "T"]  # D = A/G/T
        assert len(motif_match.sequence) == 5  # DRACH is 5 bases

    def test_read_structure(self, test_data_paths):
        """Test that extracted reads have correct structure"""
        result, motif_match = extract_reads_for_motif(
            pod5_file=str(test_data_paths["pod5"]),
            bam_file=str(test_data_paths["bam"]),
            fasta_file=str(test_data_paths["fasta"]),
            motif="DRACH",
            match_index=0,
            window=30,
            max_reads=5,
        )

        if len(result) > 0:
            read = result[0]
            # Check required fields
            assert "read_id" in read
            assert "signal" in read
            assert "reference_start" in read
            assert "reference_end" in read
            assert "chrom" in read

            # Signal should be a list/array
            assert hasattr(read["signal"], "__len__")
            assert len(read["signal"]) > 0

            # Coordinates should be integers
            assert isinstance(read["reference_start"], int)
            assert isinstance(read["reference_end"], int)

    def test_window_parameter(self, test_data_paths):
        """Test that window parameter affects region size"""
        result_small, match_small = extract_reads_for_motif(
            pod5_file=str(test_data_paths["pod5"]),
            bam_file=str(test_data_paths["bam"]),
            fasta_file=str(test_data_paths["fasta"]),
            motif="DRACH",
            match_index=0,
            window=10,  # Small window
            max_reads=100,
        )

        result_large, match_large = extract_reads_for_motif(
            pod5_file=str(test_data_paths["pod5"]),
            bam_file=str(test_data_paths["bam"]),
            fasta_file=str(test_data_paths["fasta"]),
            motif="DRACH",
            match_index=0,
            window=100,  # Large window
            max_reads=100,
        )

        # Larger window should capture more reads
        assert len(result_large) >= len(result_small)

    def test_max_reads_limit(self, test_data_paths):
        """Test that max_reads parameter is respected"""
        result, _ = extract_reads_for_motif(
            pod5_file=str(test_data_paths["pod5"]),
            bam_file=str(test_data_paths["bam"]),
            fasta_file=str(test_data_paths["fasta"]),
            motif="DRACH",
            match_index=0,
            window=100,
            max_reads=3,
        )

        # Should not exceed max_reads
        assert len(result) <= 3

    def test_different_match_indices(self, test_data_paths):
        """Test accessing different motif matches"""
        # Get first match
        result_0, match_0 = extract_reads_for_motif(
            pod5_file=str(test_data_paths["pod5"]),
            bam_file=str(test_data_paths["bam"]),
            fasta_file=str(test_data_paths["fasta"]),
            motif="DRACH",
            match_index=0,
            window=50,
            max_reads=10,
        )

        # Get second match
        result_1, match_1 = extract_reads_for_motif(
            pod5_file=str(test_data_paths["pod5"]),
            bam_file=str(test_data_paths["bam"]),
            fasta_file=str(test_data_paths["fasta"]),
            motif="DRACH",
            match_index=1,
            window=50,
            max_reads=10,
        )

        # Different matches should have different positions
        assert (
            match_0.position != match_1.position
            or match_0.chrom != match_1.chrom
            or match_0.strand != match_1.strand
        )

    def test_invalid_match_index(self, test_data_paths):
        """Test that invalid match index raises ValueError"""
        with pytest.raises(ValueError, match="Match index .* out of range"):
            extract_reads_for_motif(
                pod5_file=str(test_data_paths["pod5"]),
                bam_file=str(test_data_paths["bam"]),
                fasta_file=str(test_data_paths["fasta"]),
                motif="DRACH",
                match_index=99999,  # Way too large
                window=50,
                max_reads=10,
            )

    def test_no_reads_in_region(self, test_data_paths):
        """Test handling when no reads overlap the motif region"""
        # Use a very small window to potentially get zero reads
        result, motif_match = extract_reads_for_motif(
            pod5_file=str(test_data_paths["pod5"]),
            bam_file=str(test_data_paths["bam"]),
            fasta_file=str(test_data_paths["fasta"]),
            motif="DRACH",
            match_index=0,
            window=1,  # Tiny window
            max_reads=10,
        )

        # Should still return a list (possibly empty)
        assert isinstance(result, list)

        # Should still return a valid MotifMatch
        assert isinstance(motif_match, MotifMatch)


class TestAlignReadsToMotifCenter:
    """Tests for align_reads_to_motif_center() function"""

    def test_basic_alignment(self):
        """Test basic coordinate transformation"""
        # Mock read data
        reads = [
            {
                "read_id": "read1",
                "reference_start": 950,
                "reference_end": 1050,
                "strand": "+",
            },
            {
                "read_id": "read2",
                "reference_start": 980,
                "reference_end": 1020,
                "strand": "+",
            },
        ]

        motif_center = 1000

        aligned = align_reads_to_motif_center(reads, motif_center)

        # Check first read
        assert aligned[0]["reference_start"] == -50  # 950 - 1000
        assert aligned[0]["reference_end"] == 50  # 1050 - 1000

        # Check second read
        assert aligned[1]["reference_start"] == -20  # 980 - 1000
        assert aligned[1]["reference_end"] == 20  # 1020 - 1000

    def test_preserves_other_fields(self):
        """Test that alignment preserves non-coordinate fields"""
        reads = [
            {
                "read_id": "read1",
                "reference_start": 100,
                "reference_end": 200,
                "strand": "+",
                "signal": [1, 2, 3],
                "quality": 30,
                "chrom": "chr1",
            }
        ]

        motif_center = 150

        aligned = align_reads_to_motif_center(reads, motif_center)

        # Check coordinates were transformed
        assert aligned[0]["reference_start"] == -50
        assert aligned[0]["reference_end"] == 50

        # Check other fields preserved
        assert aligned[0]["read_id"] == "read1"
        assert aligned[0]["strand"] == "+"
        assert aligned[0]["signal"] == [1, 2, 3]
        assert aligned[0]["quality"] == 30
        assert aligned[0]["chrom"] == "chr1"

    def test_empty_list(self):
        """Test that empty list returns empty list"""
        aligned = align_reads_to_motif_center([], motif_center=1000)
        assert aligned == []

    def test_negative_positions(self):
        """Test reads with positions before motif center"""
        reads = [{"read_id": "read1", "reference_start": 50, "reference_end": 100}]

        motif_center = 200

        aligned = align_reads_to_motif_center(reads, motif_center)

        # Both positions should be negative
        assert aligned[0]["reference_start"] == -150  # 50 - 200
        assert aligned[0]["reference_end"] == -100  # 100 - 200

    def test_positive_positions(self):
        """Test reads with positions after motif center"""
        reads = [{"read_id": "read1", "reference_start": 300, "reference_end": 350}]

        motif_center = 200

        aligned = align_reads_to_motif_center(reads, motif_center)

        # Both positions should be positive
        assert aligned[0]["reference_start"] == 100  # 300 - 200
        assert aligned[0]["reference_end"] == 150  # 350 - 200

    def test_multiple_reads(self):
        """Test alignment with multiple reads"""
        reads = [
            {
                "read_id": f"read{i}",
                "reference_start": i * 10,
                "reference_end": i * 10 + 50,
            }
            for i in range(100, 110)
        ]

        motif_center = 1050

        aligned = align_reads_to_motif_center(reads, motif_center)

        # Check all reads were processed
        assert len(aligned) == 10

        # Check each transformation
        for i, read in enumerate(aligned):
            original_start = (i + 100) * 10
            original_end = original_start + 50
            assert read["reference_start"] == original_start - motif_center
            assert read["reference_end"] == original_end - motif_center

    def test_motif_center_zero(self):
        """Test with motif center at position 0"""
        reads = [{"read_id": "read1", "reference_start": 10, "reference_end": 20}]

        aligned = align_reads_to_motif_center(reads, motif_center=0)

        # Coordinates should remain unchanged
        assert aligned[0]["reference_start"] == 10
        assert aligned[0]["reference_end"] == 20

    def test_does_not_modify_original(self):
        """Test that original reads list is not modified"""
        reads = [{"read_id": "read1", "reference_start": 100, "reference_end": 200}]

        original_start = reads[0]["reference_start"]
        original_end = reads[0]["reference_end"]

        align_reads_to_motif_center(reads, motif_center=150)

        # Original should be unchanged
        assert reads[0]["reference_start"] == original_start
        assert reads[0]["reference_end"] == original_end


class TestMotifUtilsIntegration:
    """Integration tests combining extract and align functions"""

    def test_extract_and_align_pipeline(self, test_data_paths):
        """Test that extracted reads are already in motif-relative coordinates"""
        # Extract reads - they come back already clipped and in motif-relative coordinates
        reads, motif_match = extract_reads_for_motif(
            pod5_file=str(test_data_paths["pod5"]),
            bam_file=str(test_data_paths["bam"]),
            fasta_file=str(test_data_paths["fasta"]),
            motif="DRACH",
            match_index=0,
            upstream=50,
            downstream=50,
            max_reads=10,
        )

        # Verify reads are in motif-relative coordinates
        assert len(reads) > 0

        # Reads should be clipped to the window [-50, +50]
        for read in reads:
            # Coordinates should be within the window
            assert read["reference_start"] >= -50
            assert read["reference_end"] <= 50

    def test_aligned_reads_span_window(self, test_data_paths):
        """Test that reads are clipped to the specified window"""
        upstream = 30
        downstream = 30

        reads, motif_match = extract_reads_for_motif(
            pod5_file=str(test_data_paths["pod5"]),
            bam_file=str(test_data_paths["bam"]),
            fasta_file=str(test_data_paths["fasta"]),
            motif="DRACH",
            match_index=0,
            upstream=upstream,
            downstream=downstream,
            max_reads=10,
        )

        # Reads should be clipped to the window
        if len(reads) > 0:
            for read in reads:
                # Read should be within [-upstream, +downstream]
                assert read["reference_start"] >= -upstream
                assert read["reference_end"] <= downstream
