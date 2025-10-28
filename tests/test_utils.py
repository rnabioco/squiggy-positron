"""Tests for utility functions in utils.py"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


class TestDownsampleSignal:
    """Tests for signal downsampling functionality."""

    def test_downsample_no_downsampling(self):
        """Test that factor=1 returns original signal."""
        from squiggy.utils import downsample_signal

        signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = downsample_signal(signal, downsample_factor=1)

        np.testing.assert_array_equal(result, signal)

    def test_downsample_factor_2(self):
        """Test downsampling by factor of 2."""
        from squiggy.utils import downsample_signal

        signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = downsample_signal(signal, downsample_factor=2)

        # Should take every 2nd element
        expected = np.array([1, 3, 5, 7, 9])
        np.testing.assert_array_equal(result, expected)

    def test_downsample_factor_3(self):
        """Test downsampling by factor of 3."""
        from squiggy.utils import downsample_signal

        signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = downsample_signal(signal, downsample_factor=3)

        # Should take every 3rd element
        expected = np.array([1, 4, 7, 10])
        np.testing.assert_array_equal(result, expected)

    def test_downsample_factor_larger_than_length(self):
        """Test downsampling with factor larger than signal length."""
        from squiggy.utils import downsample_signal

        signal = np.array([1, 2, 3, 4, 5])
        result = downsample_signal(signal, downsample_factor=10)

        # Should return just the first element
        expected = np.array([1])
        np.testing.assert_array_equal(result, expected)

    def test_downsample_negative_factor(self):
        """Test that negative factor returns original signal."""
        from squiggy.utils import downsample_signal

        signal = np.array([1, 2, 3, 4, 5])
        result = downsample_signal(signal, downsample_factor=-1)

        # Should treat as no downsampling
        np.testing.assert_array_equal(result, signal)

    def test_downsample_zero_factor(self):
        """Test that zero factor returns original signal."""
        from squiggy.utils import downsample_signal

        signal = np.array([1, 2, 3, 4, 5])
        result = downsample_signal(signal, downsample_factor=0)

        # Should treat as no downsampling
        np.testing.assert_array_equal(result, signal)

    def test_downsample_preserves_dtype(self):
        """Test that downsampling preserves array dtype."""
        from squiggy.utils import downsample_signal

        signal = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float32)
        result = downsample_signal(signal, downsample_factor=2)

        assert result.dtype == signal.dtype

    def test_downsample_empty_array(self):
        """Test downsampling with empty array."""
        from squiggy.utils import downsample_signal

        signal = np.array([])
        result = downsample_signal(signal, downsample_factor=2)

        assert len(result) == 0


class TestReverseComplement:
    """Tests for DNA reverse complement function."""

    def test_reverse_complement_basic(self):
        """Test basic reverse complement."""
        from squiggy.utils import reverse_complement

        seq = "ACGT"
        result = reverse_complement(seq)

        assert result == "ACGT"  # Palindrome

    def test_reverse_complement_simple(self):
        """Test simple reverse complement."""
        from squiggy.utils import reverse_complement

        seq = "AAAA"
        result = reverse_complement(seq)

        assert result == "TTTT"

    def test_reverse_complement_mixed(self):
        """Test reverse complement with mixed bases."""
        from squiggy.utils import reverse_complement

        seq = "ATCGATCG"
        result = reverse_complement(seq)

        # Reverse: GCTAGCTA -> Complement: CGATCGAT
        assert result == "CGATCGAT"

    def test_reverse_complement_with_n(self):
        """Test reverse complement with N (unknown base)."""
        from squiggy.utils import reverse_complement

        seq = "ACGTN"
        result = reverse_complement(seq)

        # N complements to N
        assert result == "NACGT"

    def test_reverse_complement_empty(self):
        """Test reverse complement with empty string."""
        from squiggy.utils import reverse_complement

        seq = ""
        result = reverse_complement(seq)

        assert result == ""

    def test_reverse_complement_single_base(self):
        """Test reverse complement with single base."""
        from squiggy.utils import reverse_complement

        assert reverse_complement("A") == "T"
        assert reverse_complement("T") == "A"
        assert reverse_complement("C") == "G"
        assert reverse_complement("G") == "C"
        assert reverse_complement("N") == "N"

    def test_reverse_complement_preserves_unknown(self):
        """Test that unknown characters are preserved."""
        from squiggy.utils import reverse_complement

        # Characters not in complement dict should be preserved as-is
        seq = "ACGTX"
        result = reverse_complement(seq)

        # X is not defined, will be preserved as X
        assert result == "XACGT"

    def test_reverse_complement_case_sensitive(self):
        """Test that function is case-sensitive."""
        from squiggy.utils import reverse_complement

        # Lowercase bases are not explicitly handled
        # but should work if they're in the dict or preserved
        seq = "acgt"
        result = reverse_complement(seq)

        # Lowercase not in dict, will be preserved
        assert result == "tgca"


class TestValidateBamReadsInPod5:
    """Tests for BAM/POD5 cross-validation."""

    def test_validate_matching_files(self, sample_pod5_file, sample_bam_file):
        """Test validation with matching POD5 and BAM files."""
        from squiggy.utils import validate_bam_reads_in_pod5

        result = validate_bam_reads_in_pod5(sample_bam_file, sample_pod5_file)

        assert isinstance(result, dict)
        assert "is_valid" in result
        assert "bam_read_count" in result
        assert "pod5_read_count" in result
        assert "missing_count" in result
        assert "missing_reads" in result

        # With test data, files should match
        assert result["bam_read_count"] > 0
        assert result["pod5_read_count"] > 0

    def test_validate_reports_read_counts(self, sample_pod5_file, sample_bam_file):
        """Test that validation reports correct read counts."""
        from squiggy.utils import validate_bam_reads_in_pod5

        result = validate_bam_reads_in_pod5(sample_bam_file, sample_pod5_file)

        # Counts should be positive integers
        assert isinstance(result["bam_read_count"], int)
        assert isinstance(result["pod5_read_count"], int)
        assert result["bam_read_count"] >= 0
        assert result["pod5_read_count"] >= 0

    def test_validate_missing_count(self, sample_pod5_file, sample_bam_file):
        """Test that missing_count matches missing_reads set size."""
        from squiggy.utils import validate_bam_reads_in_pod5

        result = validate_bam_reads_in_pod5(sample_bam_file, sample_pod5_file)

        assert result["missing_count"] == len(result["missing_reads"])


class TestGetReferenceSequenceForRead:
    """Tests for extracting reference sequences from BAM."""

    def test_get_reference_for_mapped_read(self, sample_bam_file):
        """Test extracting reference sequence for mapped read."""
        import pysam

        from squiggy.utils import get_reference_sequence_for_read

        # Find a mapped read
        with pysam.AlignmentFile(str(sample_bam_file), "rb", check_sq=False) as bam:
            for read in bam.fetch(until_eof=True):
                if not read.is_unmapped:
                    read_id = read.query_name
                    ref_seq, ref_start, aligned_read = get_reference_sequence_for_read(
                        sample_bam_file, read_id
                    )

                    # Should return valid data for mapped read
                    assert ref_seq is not None
                    assert ref_start is not None
                    assert aligned_read is not None

                    # Reference sequence should be non-empty
                    assert len(ref_seq) > 0

                    # Reference start should be non-negative
                    assert ref_start >= 0

                    # Aligned read should match requested read_id
                    assert aligned_read.query_name == read_id
                    return

        pytest.skip("No mapped reads found in BAM file")

    def test_get_reference_for_unmapped_read(self, sample_bam_file):
        """Test extracting reference for unmapped read returns None."""
        import pysam

        from squiggy.utils import get_reference_sequence_for_read

        # Find an unmapped read
        with pysam.AlignmentFile(str(sample_bam_file), "rb", check_sq=False) as bam:
            for read in bam.fetch(until_eof=True):
                if read.is_unmapped:
                    read_id = read.query_name
                    ref_seq, ref_start, aligned_read = get_reference_sequence_for_read(
                        sample_bam_file, read_id
                    )

                    # Should return None for unmapped read
                    assert ref_seq is None
                    assert ref_start is None
                    assert aligned_read is None
                    return

        pytest.skip("No unmapped reads found in BAM file")

    def test_get_reference_for_nonexistent_read(self, sample_bam_file):
        """Test extracting reference for non-existent read returns None."""
        from squiggy.utils import get_reference_sequence_for_read

        ref_seq, ref_start, aligned_read = get_reference_sequence_for_read(
            sample_bam_file, "NONEXISTENT_READ_ID_12345"
        )

        # Should return None for non-existent read
        assert ref_seq is None
        assert ref_start is None
        assert aligned_read is None

    def test_get_reference_sequence_contains_valid_bases(self, sample_bam_file):
        """Test that extracted reference contains valid DNA bases."""
        import pysam

        from squiggy.utils import get_reference_sequence_for_read

        # Find a mapped read
        with pysam.AlignmentFile(str(sample_bam_file), "rb", check_sq=False) as bam:
            for read in bam.fetch(until_eof=True):
                if not read.is_unmapped:
                    read_id = read.query_name
                    ref_seq, _ref_start, _aligned_read = (
                        get_reference_sequence_for_read(sample_bam_file, read_id)
                    )

                    if ref_seq:
                        # Should only contain valid DNA bases (or N for deletions)
                        valid_bases = set("ACGTN")
                        assert all(base in valid_bases for base in ref_seq.upper())
                        return

        pytest.skip("No mapped reads with reference sequence found")


class TestPathUtilities:
    """Tests for icon and data path resolution."""

    def test_get_icon_path_returns_path_or_none(self):
        """Test that get_icon_path returns Path or None."""
        from squiggy.utils import get_icon_path

        result = get_icon_path()

        # Should return Path object or None
        assert result is None or isinstance(result, Path)

    def test_get_logo_path_returns_path_or_none(self):
        """Test that get_logo_path returns Path or None."""
        from squiggy.utils import get_logo_path

        result = get_logo_path()

        # Should return Path object or None
        assert result is None or isinstance(result, Path)

    def test_get_sample_data_path_returns_path(self):
        """Test that get_sample_data_path returns a Path."""
        from squiggy.utils import get_sample_data_path

        try:
            result = get_sample_data_path()
            assert isinstance(result, Path)
        except FileNotFoundError:
            # OK if sample data not found in test environment
            pytest.skip("Sample data not available in test environment")

    @patch("sys._MEIPASS", "/fake/pyinstaller/path", create=True)
    def test_get_icon_path_checks_pyinstaller_location(self):
        """Test that get_icon_path checks PyInstaller bundle location."""
        from squiggy.utils import get_icon_path

        # Should not crash when sys._MEIPASS is set
        result = get_icon_path()
        assert result is None or isinstance(result, Path)

    def test_get_sample_data_path_error_message(self):
        """Test that get_sample_data_path has helpful error when not found."""
        import sys

        from squiggy.utils import get_sample_data_path

        # Mock the file search to fail
        with patch("pathlib.Path.exists", return_value=False):
            # Mock the appropriate module based on Python version
            if sys.version_info >= (3, 9):
                # Mock importlib.resources.files for Python 3.9+
                with patch(
                    "importlib.resources.files", side_effect=Exception("Not found")
                ):
                    with pytest.raises(
                        FileNotFoundError, match="Sample data not found"
                    ):
                        get_sample_data_path()
            else:
                # Mock pkg_resources.resource_filename for Python 3.8
                with patch(
                    "pkg_resources.resource_filename",
                    side_effect=Exception("Not found"),
                ):
                    with pytest.raises(
                        FileNotFoundError, match="Sample data not found"
                    ):
                        get_sample_data_path()


class TestGetBasecallDataIntegration:
    """Integration tests for get_basecall_data (already partially tested elsewhere)."""

    def test_get_basecall_data_with_move_table(self, sample_bam_file):
        """Test getting basecall data for read with move table."""
        import pysam

        from squiggy.utils import get_basecall_data

        # Find a read with move table
        with pysam.AlignmentFile(str(sample_bam_file), "rb", check_sq=False) as bam:
            for read in bam.fetch(until_eof=True):
                if read.has_tag("mv"):
                    read_id = read.query_name
                    sequence, seq_to_sig_map = get_basecall_data(
                        sample_bam_file, read_id
                    )

                    # Should return valid data
                    assert sequence is not None
                    assert seq_to_sig_map is not None
                    assert len(sequence) > 0
                    assert len(seq_to_sig_map) > 0
                    # Sequence and map should have related lengths
                    assert len(seq_to_sig_map) <= len(sequence)
                    return

        pytest.skip("No reads with move table found in BAM")

    def test_get_basecall_data_without_move_table(self, sample_bam_file):
        """Test getting basecall data for read without move table."""
        import pysam

        from squiggy.utils import get_basecall_data

        # Find a read without move table
        with pysam.AlignmentFile(str(sample_bam_file), "rb", check_sq=False) as bam:
            for read in bam.fetch(until_eof=True):
                if not read.has_tag("mv"):
                    read_id = read.query_name
                    sequence, seq_to_sig_map = get_basecall_data(
                        sample_bam_file, read_id
                    )

                    # Should return None for reads without move table
                    assert sequence is None
                    assert seq_to_sig_map is None
                    return

        pytest.skip("All reads have move tables in this BAM file")

    def test_get_basecall_data_nonexistent_read(self, sample_bam_file):
        """Test getting basecall data for non-existent read."""
        from squiggy.utils import get_basecall_data

        sequence, seq_to_sig_map = get_basecall_data(
            sample_bam_file, "NONEXISTENT_READ_12345"
        )

        # Should return None for non-existent read
        assert sequence is None
        assert seq_to_sig_map is None

    def test_get_basecall_data_none_bam_file(self):
        """Test getting basecall data with None BAM file."""
        from squiggy.utils import get_basecall_data

        sequence, seq_to_sig_map = get_basecall_data(None, "any_read_id")

        # Should return None when BAM file is None
        assert sequence is None
        assert seq_to_sig_map is None


class TestIndexBamFileEdgeCases:
    """Additional tests for BAM indexing edge cases."""

    def test_index_bam_file_none_path(self):
        """Test that indexing None path raises FileNotFoundError."""
        from squiggy.utils import index_bam_file

        with pytest.raises(FileNotFoundError, match="BAM file not found"):
            index_bam_file(None)

    def test_index_bam_file_invalid_path(self):
        """Test that indexing invalid path raises FileNotFoundError."""
        from squiggy.utils import index_bam_file

        with pytest.raises(FileNotFoundError, match="BAM file not found"):
            index_bam_file(Path("/nonexistent/path/file.bam"))


class TestGetBamReferencesEdgeCases:
    """Additional tests for BAM reference extraction edge cases."""

    def test_get_bam_references_none_path(self):
        """Test that getting references from None path raises error."""
        from squiggy.utils import get_bam_references

        with pytest.raises(FileNotFoundError, match="BAM file not found"):
            get_bam_references(None)

    def test_get_bam_references_invalid_path(self):
        """Test that getting references from invalid path raises error."""
        from squiggy.utils import get_bam_references

        with pytest.raises(FileNotFoundError, match="BAM file not found"):
            get_bam_references(Path("/nonexistent/path/file.bam"))


class TestGetReadsInRegionEdgeCases:
    """Additional tests for region query edge cases."""

    def test_get_reads_in_region_none_path(self):
        """Test that querying None path raises error."""
        from squiggy.utils import get_reads_in_region

        with pytest.raises(FileNotFoundError, match="BAM file not found"):
            get_reads_in_region(None, "chr1")

    def test_get_reads_in_region_invalid_path(self):
        """Test that querying invalid path raises error."""
        from squiggy.utils import get_reads_in_region

        with pytest.raises(FileNotFoundError, match="BAM file not found"):
            get_reads_in_region(Path("/nonexistent/path/file.bam"), "chr1")
