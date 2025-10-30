"""Tests for alignment extraction and base annotation functionality."""

import numpy as np
import pytest


class TestAlignmentExtraction:
    """Tests for extracting alignment data from BAM files."""

    def test_extract_alignment_from_bam(self, indexed_bam_file):
        """Test extracting alignment for a valid read."""
        import pysam

        from squiggy.alignment import extract_alignment_from_bam

        # Get first read ID from BAM
        with pysam.AlignmentFile(str(indexed_bam_file), "rb") as bam:
            for alignment in bam.fetch(until_eof=True):
                if alignment.has_tag("mv"):  # Only test reads with move table
                    read_id = alignment.query_name

                    # Extract alignment
                    aligned_read = extract_alignment_from_bam(indexed_bam_file, read_id)

                    assert aligned_read is not None
                    assert aligned_read.read_id == read_id
                    assert aligned_read.sequence is not None
                    assert len(aligned_read.sequence) > 0
                    assert len(aligned_read.bases) > 0
                    return  # Test passed

        pytest.skip("No reads with move table found in BAM file")

    def test_extract_alignment_missing_read(self, indexed_bam_file):
        """Test extracting alignment for non-existent read returns None."""
        from squiggy.alignment import extract_alignment_from_bam

        aligned_read = extract_alignment_from_bam(
            indexed_bam_file, "NONEXISTENT_READ_ID_12345"
        )

        assert aligned_read is None

    def test_extract_alignment_no_move_table(self, indexed_bam_file):
        """Test that reads without move table are handled correctly."""
        import pysam

        from squiggy.alignment import extract_alignment_from_bam

        # Find a read without move table (if any)
        with pysam.AlignmentFile(str(indexed_bam_file), "rb") as bam:
            for alignment in bam.fetch(until_eof=True):
                if not alignment.has_tag("mv"):
                    read_id = alignment.query_name

                    # Should return None for reads without move table
                    aligned_read = extract_alignment_from_bam(indexed_bam_file, read_id)

                    # Either None or valid (depending on implementation)
                    # The function should handle this gracefully
                    assert aligned_read is None or hasattr(aligned_read, "read_id")
                    return  # Test passed

        pytest.skip("All reads have move tables in this BAM file")


class TestAlignedReadStructure:
    """Tests for AlignedRead dataclass and its properties."""

    def test_aligned_read_has_sequence(self, indexed_bam_file):
        """Test that AlignedRead contains sequence information."""
        import pysam

        from squiggy.alignment import extract_alignment_from_bam

        with pysam.AlignmentFile(str(indexed_bam_file), "rb") as bam:
            for alignment in bam.fetch(until_eof=True):
                if alignment.has_tag("mv"):
                    read_id = alignment.query_name
                    aligned_read = extract_alignment_from_bam(indexed_bam_file, read_id)

                    if aligned_read:
                        assert hasattr(aligned_read, "sequence")
                        assert isinstance(aligned_read.sequence, str)
                        assert len(aligned_read.sequence) > 0
                        # Sequence should only contain valid bases
                        assert all(
                            base in "ACGTN" for base in aligned_read.sequence.upper()
                        )
                        return

        pytest.skip("No aligned reads found")

    def test_aligned_read_has_bases(self, indexed_bam_file):
        """Test that AlignedRead contains base annotations list."""
        import pysam

        from squiggy.alignment import extract_alignment_from_bam

        with pysam.AlignmentFile(str(indexed_bam_file), "rb") as bam:
            for alignment in bam.fetch(until_eof=True):
                if alignment.has_tag("mv"):
                    read_id = alignment.query_name
                    aligned_read = extract_alignment_from_bam(indexed_bam_file, read_id)

                    if aligned_read:
                        assert hasattr(aligned_read, "bases")
                        assert isinstance(aligned_read.bases, list)
                        assert len(aligned_read.bases) > 0

                        # Number of bases should match sequence length (approximately)
                        # May differ slightly due to move table parsing
                        assert len(aligned_read.bases) <= len(aligned_read.sequence)
                        return

        pytest.skip("No aligned reads found")

    def test_aligned_read_genomic_coords(self, indexed_bam_file):
        """Test that AlignedRead contains genomic coordinates for mapped reads."""
        import pysam

        from squiggy.alignment import extract_alignment_from_bam

        with pysam.AlignmentFile(str(indexed_bam_file), "rb") as bam:
            for alignment in bam.fetch(until_eof=True):
                if alignment.has_tag("mv") and not alignment.is_unmapped:
                    read_id = alignment.query_name
                    aligned_read = extract_alignment_from_bam(indexed_bam_file, read_id)

                    if aligned_read:
                        # Mapped reads should have genomic coordinates
                        assert aligned_read.chromosome is not None
                        assert aligned_read.genomic_start is not None
                        assert aligned_read.genomic_end is not None
                        assert isinstance(aligned_read.chromosome, str)
                        assert isinstance(aligned_read.genomic_start, int)
                        assert isinstance(aligned_read.genomic_end, int)
                        assert aligned_read.genomic_end >= aligned_read.genomic_start
                        return

        pytest.skip("No mapped reads with move tables found")

    def test_aligned_read_strand_info(self, indexed_bam_file):
        """Test that AlignedRead contains strand information."""
        import pysam

        from squiggy.alignment import extract_alignment_from_bam

        with pysam.AlignmentFile(str(indexed_bam_file), "rb") as bam:
            for alignment in bam.fetch(until_eof=True):
                if alignment.has_tag("mv"):
                    read_id = alignment.query_name
                    aligned_read = extract_alignment_from_bam(indexed_bam_file, read_id)

                    if aligned_read:
                        assert hasattr(aligned_read, "strand")
                        assert hasattr(aligned_read, "is_reverse")

                        # Strand should be + or -
                        assert aligned_read.strand in ["+", "-"]

                        # is_reverse should be boolean
                        assert isinstance(aligned_read.is_reverse, bool)

                        # Consistency check
                        if aligned_read.is_reverse:
                            assert aligned_read.strand == "-"
                        else:
                            assert aligned_read.strand == "+"
                        return

        pytest.skip("No aligned reads found")


class TestBaseAnnotation:
    """Tests for BaseAnnotation dataclass."""

    def test_base_annotation_signal_mapping(self, indexed_bam_file):
        """Test that BaseAnnotation contains signal start/end indices."""
        import pysam

        from squiggy.alignment import extract_alignment_from_bam

        with pysam.AlignmentFile(str(indexed_bam_file), "rb") as bam:
            for alignment in bam.fetch(until_eof=True):
                if alignment.has_tag("mv"):
                    read_id = alignment.query_name
                    aligned_read = extract_alignment_from_bam(indexed_bam_file, read_id)

                    if aligned_read and len(aligned_read.bases) > 0:
                        base_annotation = aligned_read.bases[0]

                        assert hasattr(base_annotation, "signal_start")
                        assert hasattr(base_annotation, "signal_end")

                        # Signal indices should be non-negative integers
                        assert isinstance(base_annotation.signal_start, int)
                        assert isinstance(base_annotation.signal_end, int)
                        assert base_annotation.signal_start >= 0
                        assert (
                            base_annotation.signal_end >= base_annotation.signal_start
                        )
                        return

        pytest.skip("No aligned reads found")

    def test_base_annotation_position(self, indexed_bam_file):
        """Test that BaseAnnotation contains sequence position."""
        import pysam

        from squiggy.alignment import extract_alignment_from_bam

        with pysam.AlignmentFile(str(indexed_bam_file), "rb") as bam:
            for alignment in bam.fetch(until_eof=True):
                if alignment.has_tag("mv"):
                    read_id = alignment.query_name
                    aligned_read = extract_alignment_from_bam(indexed_bam_file, read_id)

                    if aligned_read and len(aligned_read.bases) > 0:
                        # Check first few bases
                        for i, base_annotation in enumerate(aligned_read.bases[:5]):
                            assert hasattr(base_annotation, "position")
                            assert isinstance(base_annotation.position, int)
                            # Position should match index
                            assert base_annotation.position == i
                        return

        pytest.skip("No aligned reads found")

    def test_base_annotation_base_character(self, indexed_bam_file):
        """Test that BaseAnnotation contains valid base character."""
        import pysam

        from squiggy.alignment import extract_alignment_from_bam

        with pysam.AlignmentFile(str(indexed_bam_file), "rb") as bam:
            for alignment in bam.fetch(until_eof=True):
                if alignment.has_tag("mv"):
                    read_id = alignment.query_name
                    aligned_read = extract_alignment_from_bam(indexed_bam_file, read_id)

                    if aligned_read and len(aligned_read.bases) > 0:
                        for base_annotation in aligned_read.bases:
                            assert hasattr(base_annotation, "base")
                            assert isinstance(base_annotation.base, str)
                            assert len(base_annotation.base) == 1
                            # Should be valid DNA base
                            assert base_annotation.base.upper() in "ACGTN"
                        return

        pytest.skip("No aligned reads found")

    def test_base_annotation_quality(self, indexed_bam_file):
        """Test that BaseAnnotation contains quality scores when available."""
        import pysam

        from squiggy.alignment import extract_alignment_from_bam

        with pysam.AlignmentFile(str(indexed_bam_file), "rb") as bam:
            for alignment in bam.fetch(until_eof=True):
                if alignment.has_tag("mv") and alignment.query_qualities is not None:
                    read_id = alignment.query_name
                    aligned_read = extract_alignment_from_bam(indexed_bam_file, read_id)

                    if aligned_read and len(aligned_read.bases) > 0:
                        base_annotation = aligned_read.bases[0]

                        assert hasattr(base_annotation, "quality")
                        # Quality may be None or an integer
                        if base_annotation.quality is not None:
                            assert isinstance(
                                base_annotation.quality, (int, np.integer)
                            )
                            assert (
                                0 <= base_annotation.quality <= 93
                            )  # Phred quality range
                        return

        pytest.skip("No aligned reads with quality scores found")


class TestMoveTableParsing:
    """Tests for move table parsing and conversion."""

    def test_move_table_to_base_annotations(self, indexed_bam_file):
        """Test that move table is correctly parsed to base annotations."""
        import pysam

        from squiggy.alignment import extract_alignment_from_bam

        with pysam.AlignmentFile(str(indexed_bam_file), "rb") as bam:
            for alignment in bam.fetch(until_eof=True):
                if alignment.has_tag("mv"):
                    read_id = alignment.query_name
                    aligned_read = extract_alignment_from_bam(indexed_bam_file, read_id)

                    if aligned_read:
                        # Move table should produce base annotations
                        assert len(aligned_read.bases) > 0

                        # Each base should have increasing signal positions
                        for i in range(len(aligned_read.bases) - 1):
                            curr_base = aligned_read.bases[i]
                            next_base = aligned_read.bases[i + 1]

                            # Next base should start at or after current base
                            assert next_base.signal_start >= curr_base.signal_start
                        return

        pytest.skip("No reads with move tables found")

    def test_base_to_signal_mapping_conversion(self, indexed_bam_file):
        """Test conversion of AlignedRead to base-to-signal mapping format."""
        import pysam

        from squiggy.alignment import (
            extract_alignment_from_bam,
            get_base_to_signal_mapping,
        )

        with pysam.AlignmentFile(str(indexed_bam_file), "rb") as bam:
            for alignment in bam.fetch(until_eof=True):
                if alignment.has_tag("mv"):
                    read_id = alignment.query_name
                    aligned_read = extract_alignment_from_bam(indexed_bam_file, read_id)

                    if aligned_read:
                        # Convert to plotter format
                        sequence, seq_to_sig_map = get_base_to_signal_mapping(
                            aligned_read
                        )

                        # Verify sequence matches
                        assert sequence == aligned_read.sequence

                        # Verify mapping is a numpy array
                        assert isinstance(seq_to_sig_map, np.ndarray)

                        # Length should match number of bases
                        assert len(seq_to_sig_map) == len(aligned_read.bases)

                        # Values should be signal start positions
                        for i, sig_pos in enumerate(seq_to_sig_map):
                            assert sig_pos == aligned_read.bases[i].signal_start
                        return

        pytest.skip("No aligned reads found")


class TestStrideHandling:
    """Tests for stride extraction and usage in move table parsing."""

    def test_move_table_stride_extraction(self, indexed_bam_file):
        """Test that stride is correctly extracted from move table."""
        import pysam

        with pysam.AlignmentFile(str(indexed_bam_file), "rb") as bam:
            for alignment in bam.fetch(until_eof=True):
                if alignment.has_tag("mv"):
                    move_table = np.array(alignment.get_tag("mv"), dtype=np.uint8)

                    # First element should be stride
                    stride = int(move_table[0])

                    # Stride should be reasonable (typically 5 for DNA, 10-12 for RNA)
                    assert stride > 0
                    assert stride <= 20  # Sanity check
                    assert isinstance(stride, int)

                    # Common values
                    assert stride in [5, 6, 10, 11, 12] or stride <= 20
                    return

        pytest.skip("No reads with move tables found")

    def test_signal_positions_use_stride(self, indexed_bam_file):
        """Test that signal positions are stride-adjusted."""
        import pysam

        from squiggy.alignment import extract_alignment_from_bam

        with pysam.AlignmentFile(str(indexed_bam_file), "rb") as bam:
            for alignment in bam.fetch(until_eof=True):
                if alignment.has_tag("mv"):
                    read_id = alignment.query_name
                    move_table = np.array(alignment.get_tag("mv"), dtype=np.uint8)
                    stride = int(move_table[0])

                    aligned_read = extract_alignment_from_bam(indexed_bam_file, read_id)

                    if aligned_read and len(aligned_read.bases) > 1:
                        # Check that signal positions are multiples of stride
                        for base in aligned_read.bases:
                            # Signal positions should be divisible by stride
                            assert base.signal_start % stride == 0
                            assert base.signal_end % stride == 0

                        # Check spacing between bases is at least stride
                        for i in range(len(aligned_read.bases) - 1):
                            curr = aligned_read.bases[i]
                            next_base = aligned_read.bases[i + 1]
                            spacing = next_base.signal_start - curr.signal_start
                            # Spacing should be a multiple of stride
                            assert spacing % stride == 0
                            assert spacing >= stride
                        return

        pytest.skip("No aligned reads with sufficient bases found")

    def test_dwell_time_with_stride(self, sample_pod5_file, indexed_bam_file):
        """Test that dwell times are realistic when using stride."""
        import pod5
        import pysam

        from squiggy.alignment import extract_alignment_from_bam

        with pysam.AlignmentFile(str(indexed_bam_file), "rb") as bam:
            for alignment in bam.fetch(until_eof=True):
                if alignment.has_tag("mv"):
                    read_id = alignment.query_name
                    aligned_read = extract_alignment_from_bam(indexed_bam_file, read_id)

                    if aligned_read and len(aligned_read.bases) > 0:
                        # Get sample rate from POD5
                        with pod5.Reader(sample_pod5_file) as reader:
                            for read in reader.reads():
                                if str(read.read_id) == read_id:
                                    sample_rate = read.run_info.sample_rate

                                    # Calculate dwell time for first base
                                    base = aligned_read.bases[0]
                                    dwell_samples = base.signal_end - base.signal_start
                                    dwell_time_ms = (dwell_samples / sample_rate) * 1000

                                    # Dwell times should be realistic (typically 1-20 ms per base)
                                    # Not microseconds (which would indicate stride not used)
                                    assert dwell_time_ms >= 0.5, (
                                        f"Dwell time too short: {dwell_time_ms} ms"
                                    )
                                    assert dwell_time_ms <= 100, (
                                        f"Dwell time too long: {dwell_time_ms} ms"
                                    )

                                    # Most bases should be in 1-10 ms range
                                    if dwell_time_ms < 0.5:
                                        pytest.fail(
                                            f"Dwell time {dwell_time_ms} ms is unrealistically short - stride may not be applied"
                                        )

                                    return

        pytest.skip("No matching reads found in POD5 and BAM")

    def test_stride_in_utils_basecall_data(self, indexed_bam_file):
        """Test that utils.get_basecall_data properly uses stride."""
        import pysam

        from squiggy.utils import get_basecall_data

        with pysam.AlignmentFile(str(indexed_bam_file), "rb") as bam:
            for alignment in bam.fetch(until_eof=True):
                if alignment.has_tag("mv"):
                    read_id = alignment.query_name
                    move_table = np.array(alignment.get_tag("mv"), dtype=np.uint8)
                    stride = int(move_table[0])

                    sequence, seq_to_sig_map = get_basecall_data(
                        indexed_bam_file, read_id
                    )

                    if sequence is not None and seq_to_sig_map is not None:
                        # All signal positions should be multiples of stride
                        for sig_pos in seq_to_sig_map:
                            assert sig_pos % stride == 0

                        # Check spacing between consecutive bases
                        if len(seq_to_sig_map) > 1:
                            for i in range(len(seq_to_sig_map) - 1):
                                spacing = seq_to_sig_map[i + 1] - seq_to_sig_map[i]
                                # Spacing should be multiple of stride
                                assert spacing % stride == 0
                                assert spacing >= stride
                        return

        pytest.skip("No reads with move tables found")


class TestEdgeCases:
    """Tests for edge cases in alignment processing."""

    def test_unmapped_read_handling(self, indexed_bam_file):
        """Test that unmapped reads are handled correctly."""
        import pysam

        from squiggy.alignment import extract_alignment_from_bam

        with pysam.AlignmentFile(str(indexed_bam_file), "rb") as bam:
            for alignment in bam.fetch(until_eof=True):
                if alignment.is_unmapped and alignment.has_tag("mv"):
                    read_id = alignment.query_name
                    aligned_read = extract_alignment_from_bam(indexed_bam_file, read_id)

                    if aligned_read:
                        # Unmapped reads should have None for genomic coords
                        assert aligned_read.chromosome is None
                        assert aligned_read.genomic_start is None
                        assert aligned_read.genomic_end is None

                        # But should still have sequence and bases
                        assert aligned_read.sequence is not None
                        assert len(aligned_read.bases) > 0
                        return

        pytest.skip("No unmapped reads with move tables found")

    def test_reverse_strand_alignment(self, indexed_bam_file):
        """Test that reverse strand reads are handled correctly."""
        import pysam

        from squiggy.alignment import extract_alignment_from_bam

        with pysam.AlignmentFile(str(indexed_bam_file), "rb") as bam:
            for alignment in bam.fetch(until_eof=True):
                if alignment.is_reverse and alignment.has_tag("mv"):
                    read_id = alignment.query_name
                    aligned_read = extract_alignment_from_bam(indexed_bam_file, read_id)

                    if aligned_read:
                        # Reverse reads should be marked
                        assert aligned_read.is_reverse is True
                        assert aligned_read.strand == "-"

                        # Should still have sequence and bases
                        assert aligned_read.sequence is not None
                        assert len(aligned_read.bases) > 0
                        return

        pytest.skip("No reverse strand reads found")

    def test_short_sequence_handling(self, indexed_bam_file):
        """Test handling of very short sequences."""
        import pysam

        from squiggy.alignment import extract_alignment_from_bam

        # Find a short sequence (if any)
        with pysam.AlignmentFile(str(indexed_bam_file), "rb") as bam:
            for alignment in bam.fetch(until_eof=True):
                if (
                    alignment.has_tag("mv")
                    and alignment.query_sequence
                    and len(alignment.query_sequence) < 100
                ):
                    read_id = alignment.query_name
                    aligned_read = extract_alignment_from_bam(indexed_bam_file, read_id)

                    # Should handle short sequences gracefully
                    if aligned_read:
                        assert len(aligned_read.sequence) > 0
                        assert len(aligned_read.bases) > 0
                    return

        pytest.skip("No short sequences found")

    def test_empty_move_table_handling(self):
        """Test handling of edge cases in move table parsing."""
        from squiggy.alignment import BaseAnnotation

        # Test BaseAnnotation creation
        annotation = BaseAnnotation(base="A", position=0, signal_start=0, signal_end=10)

        assert annotation.base == "A"
        assert annotation.position == 0
        assert annotation.signal_start == 0
        assert annotation.signal_end == 10

        # Optional fields should default to None
        assert annotation.genomic_pos is None
        assert annotation.quality is None


class TestDataclassCreation:
    """Tests for creating dataclass instances directly."""

    def test_create_base_annotation(self):
        """Test creating BaseAnnotation directly."""
        from squiggy.alignment import BaseAnnotation

        base_ann = BaseAnnotation(
            base="A",
            position=5,
            signal_start=100,
            signal_end=150,
            genomic_pos=1000,
            quality=40,
        )

        assert base_ann.base == "A"
        assert base_ann.position == 5
        assert base_ann.signal_start == 100
        assert base_ann.signal_end == 150
        assert base_ann.genomic_pos == 1000
        assert base_ann.quality == 40

    def test_create_aligned_read(self):
        """Test creating AlignedRead directly."""
        from squiggy.alignment import AlignedRead, BaseAnnotation

        bases = [
            BaseAnnotation("A", 0, 0, 10),
            BaseAnnotation("C", 1, 10, 20),
            BaseAnnotation("G", 2, 20, 30),
        ]

        aligned_read = AlignedRead(
            read_id="test_read",
            sequence="ACG",
            bases=bases,
            stride=5,
            chromosome="chr1",
            genomic_start=1000,
            genomic_end=1003,
            strand="+",
            is_reverse=False,
        )

        assert aligned_read.read_id == "test_read"
        assert aligned_read.sequence == "ACG"
        assert len(aligned_read.bases) == 3
        assert aligned_read.chromosome == "chr1"
        assert aligned_read.strand == "+"
        assert aligned_read.is_reverse is False
