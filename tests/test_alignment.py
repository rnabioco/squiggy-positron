"""Tests for alignment parsing and base annotation extraction"""

import numpy as np
import pysam
import pytest


class TestBaseAnnotation:
    """Tests for BaseAnnotation dataclass"""

    def test_base_annotation_creation(self):
        """Test creating BaseAnnotation with all fields"""
        from squiggy.alignment import BaseAnnotation

        annotation = BaseAnnotation(
            base="A",
            position=0,
            signal_start=0,
            signal_end=100,
            genomic_pos=1000,
            quality=30,
        )

        assert annotation.base == "A"
        assert annotation.position == 0
        assert annotation.signal_start == 0
        assert annotation.signal_end == 100
        assert annotation.genomic_pos == 1000
        assert annotation.quality == 30

    def test_base_annotation_optional_fields(self):
        """Test creating BaseAnnotation with optional fields as None"""
        from squiggy.alignment import BaseAnnotation

        annotation = BaseAnnotation(
            base="C", position=5, signal_start=500, signal_end=550
        )

        assert annotation.base == "C"
        assert annotation.position == 5
        assert annotation.signal_start == 500
        assert annotation.signal_end == 550
        assert annotation.genomic_pos is None
        assert annotation.quality is None


class TestAlignedRead:
    """Tests for AlignedRead dataclass"""

    def test_aligned_read_creation(self):
        """Test creating AlignedRead with all fields"""
        from squiggy.alignment import AlignedRead, BaseAnnotation

        bases = [
            BaseAnnotation(base="A", position=0, signal_start=0, signal_end=100),
            BaseAnnotation(base="T", position=1, signal_start=100, signal_end=200),
        ]

        aligned_read = AlignedRead(
            read_id="test_read_001",
            sequence="AT",
            bases=bases,
            chromosome="chr1",
            genomic_start=1000,
            genomic_end=1002,
            strand="+",
            is_reverse=False,
            modifications=[],
        )

        assert aligned_read.read_id == "test_read_001"
        assert aligned_read.sequence == "AT"
        assert len(aligned_read.bases) == 2
        assert aligned_read.chromosome == "chr1"
        assert aligned_read.genomic_start == 1000
        assert aligned_read.genomic_end == 1002
        assert aligned_read.strand == "+"
        assert aligned_read.is_reverse is False
        assert aligned_read.modifications == []

    def test_aligned_read_minimal(self):
        """Test creating AlignedRead with minimal required fields"""
        from squiggy.alignment import AlignedRead

        aligned_read = AlignedRead(read_id="test_read", sequence="ACGT", bases=[])

        assert aligned_read.read_id == "test_read"
        assert aligned_read.sequence == "ACGT"
        assert aligned_read.bases == []
        assert aligned_read.chromosome is None
        assert aligned_read.genomic_start is None
        assert aligned_read.genomic_end is None
        assert aligned_read.strand is None
        assert aligned_read.is_reverse is False
        assert aligned_read.modifications == []


class TestExtractAlignmentFromBAM:
    """Tests for extract_alignment_from_bam function"""

    def test_extract_alignment_success(self, sample_bam_file):
        """Test extracting alignment for a read that exists in BAM"""
        from squiggy.alignment import extract_alignment_from_bam

        # Find a read ID from the BAM file
        with pysam.AlignmentFile(str(sample_bam_file), "rb", check_sq=False) as bam:
            for alignment in bam.fetch(until_eof=True):
                if alignment.has_tag("mv") and not alignment.is_unmapped:
                    read_id = alignment.query_name
                    break
            else:
                pytest.skip("No alignments with mv tag found")

        # Extract alignment
        aligned_read = extract_alignment_from_bam(sample_bam_file, read_id)

        assert aligned_read is not None
        assert aligned_read.read_id == read_id
        assert len(aligned_read.sequence) > 0
        assert len(aligned_read.bases) > 0
        assert aligned_read.chromosome is not None

    def test_extract_alignment_read_not_found(self, sample_bam_file):
        """Test extracting alignment for non-existent read"""
        from squiggy.alignment import extract_alignment_from_bam

        aligned_read = extract_alignment_from_bam(
            sample_bam_file, "nonexistent_read_id"
        )

        assert aligned_read is None

    def test_extract_alignment_invalid_bam(self):
        """Test extracting from invalid BAM file path"""
        from pathlib import Path

        from squiggy.alignment import extract_alignment_from_bam

        invalid_path = Path("/nonexistent/file.bam")
        aligned_read = extract_alignment_from_bam(invalid_path, "any_read_id")

        assert aligned_read is None


class TestParseAlignment:
    """Tests for _parse_alignment function"""

    def test_parse_alignment_with_move_table(self, sample_bam_file):
        """Test parsing alignment with move table"""
        from squiggy.alignment import _parse_alignment

        # Find an alignment with mv tag
        with pysam.AlignmentFile(str(sample_bam_file), "rb", check_sq=False) as bam:
            for alignment in bam.fetch(until_eof=True):
                if alignment.has_tag("mv") and alignment.query_sequence:
                    aligned_read = _parse_alignment(alignment)

                    assert aligned_read is not None
                    assert aligned_read.read_id == alignment.query_name
                    assert aligned_read.sequence == alignment.query_sequence
                    assert len(aligned_read.bases) > 0

                    # Verify base annotations
                    for base_ann in aligned_read.bases:
                        assert base_ann.base in "ACGT"
                        assert base_ann.signal_start >= 0
                        assert base_ann.signal_end > base_ann.signal_start
                        assert 0 <= base_ann.position < len(aligned_read.sequence)

                    # Verify strand info
                    if alignment.is_reverse:
                        assert aligned_read.strand == "-"
                        assert aligned_read.is_reverse is True
                    else:
                        assert aligned_read.strand == "+"
                        assert aligned_read.is_reverse is False

                    return  # Test passed

        pytest.skip("No alignments with mv tag found")

    def test_parse_alignment_without_move_table(self, sample_bam_file):
        """Test parsing alignment without move table returns None"""
        from unittest.mock import MagicMock

        from squiggy.alignment import _parse_alignment

        # Create a mock alignment without mv tag
        mock_alignment = MagicMock()
        mock_alignment.query_sequence = "ACGT"
        mock_alignment.has_tag.return_value = False

        aligned_read = _parse_alignment(mock_alignment)

        assert aligned_read is None

    def test_parse_alignment_without_sequence(self):
        """Test parsing alignment without sequence returns None"""
        from unittest.mock import MagicMock

        from squiggy.alignment import _parse_alignment

        # Create a mock alignment without sequence
        mock_alignment = MagicMock()
        mock_alignment.query_sequence = None

        aligned_read = _parse_alignment(mock_alignment)

        assert aligned_read is None

    def test_parse_alignment_unmapped_read(self, sample_bam_file):
        """Test parsing unmapped read (should have None genomic positions)"""
        from squiggy.alignment import _parse_alignment

        # Find an unmapped alignment with mv tag (if any)
        with pysam.AlignmentFile(str(sample_bam_file), "rb", check_sq=False) as bam:
            for alignment in bam.fetch(until_eof=True):
                if alignment.is_unmapped and alignment.has_tag("mv"):
                    aligned_read = _parse_alignment(alignment)

                    assert aligned_read is not None
                    assert aligned_read.chromosome is None
                    assert aligned_read.genomic_start is None
                    assert aligned_read.genomic_end is None

                    # Bases should not have genomic positions
                    for base_ann in aligned_read.bases:
                        assert base_ann.genomic_pos is None

                    return  # Test passed

        # If no unmapped reads found, that's okay - skip
        pytest.skip("No unmapped reads with mv tag found")

    def test_parse_alignment_move_table_structure(self, sample_bam_file):
        """Test that move table is parsed correctly (stride + moves)"""
        from squiggy.alignment import _parse_alignment

        # Find an alignment and check move table structure
        with pysam.AlignmentFile(str(sample_bam_file), "rb", check_sq=False) as bam:
            for alignment in bam.fetch(until_eof=True):
                if alignment.has_tag("mv") and alignment.query_sequence:
                    move_table = np.array(alignment.get_tag("mv"), dtype=np.uint8)

                    # First element should be stride (typically 5-12)
                    stride = move_table[0]
                    assert 1 <= stride <= 20  # Reasonable stride values

                    # Parse alignment
                    aligned_read = _parse_alignment(alignment)
                    assert aligned_read is not None

                    # Verify that bases have reasonable signal positions
                    for base_ann in aligned_read.bases:
                        # Signal positions should be non-negative and increasing
                        assert base_ann.signal_start >= 0
                        assert base_ann.signal_end > base_ann.signal_start

                    return  # Test passed

        pytest.skip("No alignments with mv tag found")

    def test_parse_alignment_quality_scores(self, sample_bam_file):
        """Test that quality scores are extracted correctly"""
        from squiggy.alignment import _parse_alignment

        with pysam.AlignmentFile(str(sample_bam_file), "rb", check_sq=False) as bam:
            for alignment in bam.fetch(until_eof=True):
                if (
                    alignment.has_tag("mv")
                    and alignment.query_sequence
                    and alignment.query_qualities is not None
                ):
                    aligned_read = _parse_alignment(alignment)
                    assert aligned_read is not None

                    # Check that quality scores are assigned
                    for base_ann in aligned_read.bases:
                        assert base_ann.quality is not None
                        assert 0 <= base_ann.quality <= 93  # Phred quality score range

                    return  # Test passed

        pytest.skip("No alignments with quality scores found")


class TestGetBaseToSignalMapping:
    """Tests for get_base_to_signal_mapping function"""

    def test_get_base_to_signal_mapping_basic(self):
        """Test basic extraction of sequence and signal mapping"""
        from squiggy.alignment import (
            AlignedRead,
            BaseAnnotation,
            get_base_to_signal_mapping,
        )

        bases = [
            BaseAnnotation(base="A", position=0, signal_start=0, signal_end=100),
            BaseAnnotation(base="C", position=1, signal_start=100, signal_end=200),
            BaseAnnotation(base="G", position=2, signal_start=200, signal_end=300),
            BaseAnnotation(base="T", position=3, signal_start=300, signal_end=400),
        ]

        aligned_read = AlignedRead(read_id="test", sequence="ACGT", bases=bases)

        sequence, seq_to_sig_map = get_base_to_signal_mapping(aligned_read)

        assert sequence == "ACGT"
        assert len(seq_to_sig_map) == 4
        np.testing.assert_array_equal(seq_to_sig_map, [0, 100, 200, 300])

    def test_get_base_to_signal_mapping_empty(self):
        """Test extraction with empty bases list"""
        from squiggy.alignment import AlignedRead, get_base_to_signal_mapping

        aligned_read = AlignedRead(read_id="test", sequence="ACGT", bases=[])

        sequence, seq_to_sig_map = get_base_to_signal_mapping(aligned_read)

        assert sequence == "ACGT"
        assert len(seq_to_sig_map) == 0

    def test_get_base_to_signal_mapping_returns_numpy_array(self):
        """Test that seq_to_sig_map is a numpy array"""
        from squiggy.alignment import (
            AlignedRead,
            BaseAnnotation,
            get_base_to_signal_mapping,
        )

        bases = [BaseAnnotation(base="A", position=0, signal_start=50, signal_end=150)]

        aligned_read = AlignedRead(read_id="test", sequence="A", bases=bases)

        _, seq_to_sig_map = get_base_to_signal_mapping(aligned_read)

        assert isinstance(seq_to_sig_map, np.ndarray)

    def test_get_base_to_signal_mapping_integration(self, sample_bam_file):
        """Test extraction with real alignment from BAM"""
        from squiggy.alignment import (
            extract_alignment_from_bam,
            get_base_to_signal_mapping,
        )

        # Find a read with alignment
        with pysam.AlignmentFile(str(sample_bam_file), "rb", check_sq=False) as bam:
            for alignment in bam.fetch(until_eof=True):
                if alignment.has_tag("mv") and alignment.query_sequence:
                    read_id = alignment.query_name
                    break
            else:
                pytest.skip("No alignments with mv tag found")

        # Extract and test
        aligned_read = extract_alignment_from_bam(sample_bam_file, read_id)
        assert aligned_read is not None

        sequence, seq_to_sig_map = get_base_to_signal_mapping(aligned_read)

        # Verify results
        assert sequence == aligned_read.sequence
        assert len(seq_to_sig_map) == len(aligned_read.bases)
        assert np.all(seq_to_sig_map >= 0)  # All positions should be non-negative
        # Signal positions should be monotonically increasing
        assert np.all(seq_to_sig_map[1:] >= seq_to_sig_map[:-1])


class TestAlignmentEdgeCases:
    """Tests for edge cases and error handling"""

    def test_parse_alignment_with_short_sequence(self, sample_bam_file):
        """Test parsing alignment with very short sequence"""
        from squiggy.alignment import _parse_alignment

        # Find an alignment with short sequence
        with pysam.AlignmentFile(str(sample_bam_file), "rb", check_sq=False) as bam:
            for alignment in bam.fetch(until_eof=True):
                if (
                    alignment.has_tag("mv")
                    and alignment.query_sequence
                    and len(alignment.query_sequence) < 10
                ):
                    aligned_read = _parse_alignment(alignment)

                    assert aligned_read is not None
                    assert len(aligned_read.bases) > 0
                    assert len(aligned_read.bases) <= len(aligned_read.sequence)

                    return  # Test passed

        # If no short sequences found, that's okay
        pytest.skip("No short sequences found")

    def test_extract_multiple_reads_from_same_bam(self, sample_bam_file):
        """Test extracting multiple different reads from same BAM"""
        from squiggy.alignment import extract_alignment_from_bam

        # Get multiple read IDs
        read_ids = []
        with pysam.AlignmentFile(str(sample_bam_file), "rb", check_sq=False) as bam:
            for alignment in bam.fetch(until_eof=True):
                if alignment.has_tag("mv") and alignment.query_sequence:
                    read_ids.append(alignment.query_name)
                    if len(read_ids) >= 3:
                        break

        if len(read_ids) < 2:
            pytest.skip("Not enough reads in BAM")

        # Extract each read
        aligned_reads = []
        for read_id in read_ids:
            aligned_read = extract_alignment_from_bam(sample_bam_file, read_id)
            assert aligned_read is not None
            assert aligned_read.read_id == read_id
            aligned_reads.append(aligned_read)

        # Verify they're different
        assert len({ar.read_id for ar in aligned_reads}) == len(aligned_reads)

    def test_aligned_read_base_positions_sequential(self, sample_bam_file):
        """Test that base positions are sequential (0, 1, 2, ...)"""
        from squiggy.alignment import extract_alignment_from_bam

        # Find a read
        with pysam.AlignmentFile(str(sample_bam_file), "rb", check_sq=False) as bam:
            for alignment in bam.fetch(until_eof=True):
                if alignment.has_tag("mv") and alignment.query_sequence:
                    read_id = alignment.query_name
                    break
            else:
                pytest.skip("No alignments with mv tag found")

        aligned_read = extract_alignment_from_bam(sample_bam_file, read_id)
        assert aligned_read is not None

        # Check positions are sequential
        positions = [base.position for base in aligned_read.bases]
        expected_positions = list(range(len(positions)))
        assert positions == expected_positions

    def test_aligned_read_signal_ranges_non_overlapping(self, sample_bam_file):
        """Test that signal ranges don't overlap (or minimally overlap)"""
        from squiggy.alignment import extract_alignment_from_bam

        # Find a read
        with pysam.AlignmentFile(str(sample_bam_file), "rb", check_sq=False) as bam:
            for alignment in bam.fetch(until_eof=True):
                if alignment.has_tag("mv") and alignment.query_sequence:
                    read_id = alignment.query_name
                    break
            else:
                pytest.skip("No alignments with mv tag found")

        aligned_read = extract_alignment_from_bam(sample_bam_file, read_id)
        assert aligned_read is not None

        # Check signal ranges
        for i in range(len(aligned_read.bases) - 1):
            current = aligned_read.bases[i]
            next_base = aligned_read.bases[i + 1]

            # Current end should be <= next start (allowing for exact match)
            assert current.signal_end <= next_base.signal_start + 1, (
                f"Overlapping signals at position {i}"
            )
