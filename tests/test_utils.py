"""Tests for utility functions"""

import os
from pathlib import Path

import numpy as np
import pytest


class TestDownsampleSignal:
    """Tests for downsample_signal function"""

    def test_downsample_no_factor(self):
        """Test that downsample_factor=1 returns original signal"""
        from squiggy.utils import downsample_signal

        signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = downsample_signal(signal, downsample_factor=1)

        np.testing.assert_array_equal(result, signal)

    def test_downsample_factor_2(self):
        """Test downsampling by factor of 2"""
        from squiggy.utils import downsample_signal

        signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = downsample_signal(signal, downsample_factor=2)

        expected = np.array([1, 3, 5, 7, 9])
        np.testing.assert_array_equal(result, expected)

    def test_downsample_factor_5(self):
        """Test downsampling by factor of 5"""
        from squiggy.utils import downsample_signal

        signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = downsample_signal(signal, downsample_factor=5)

        expected = np.array([1, 6])
        np.testing.assert_array_equal(result, expected)

    def test_downsample_factor_larger_than_length(self):
        """Test downsampling with factor larger than array length"""
        from squiggy.utils import downsample_signal

        signal = np.array([1, 2, 3, 4, 5])
        result = downsample_signal(signal, downsample_factor=10)

        # Should return just first element
        expected = np.array([1])
        np.testing.assert_array_equal(result, expected)

    def test_downsample_empty_signal(self):
        """Test downsampling empty signal"""
        from squiggy.utils import downsample_signal

        signal = np.array([])
        result = downsample_signal(signal, downsample_factor=2)

        assert len(result) == 0

    def test_downsample_single_element(self):
        """Test downsampling signal with single element"""
        from squiggy.utils import downsample_signal

        signal = np.array([42])
        result = downsample_signal(signal, downsample_factor=5)

        np.testing.assert_array_equal(result, signal)

    def test_downsample_preserves_dtype(self):
        """Test that downsampling preserves array dtype"""
        from squiggy.utils import downsample_signal

        signal = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
        result = downsample_signal(signal, downsample_factor=2)

        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(
            result, np.array([1.5, 3.5], dtype=np.float32)
        )

    def test_downsample_zero_factor(self):
        """Test that downsample_factor=0 returns original signal"""
        from squiggy.utils import downsample_signal

        signal = np.array([1, 2, 3, 4, 5])
        result = downsample_signal(signal, downsample_factor=0)

        np.testing.assert_array_equal(result, signal)

    def test_downsample_negative_factor(self):
        """Test that negative downsample_factor returns original signal"""
        from squiggy.utils import downsample_signal

        signal = np.array([1, 2, 3, 4, 5])
        result = downsample_signal(signal, downsample_factor=-5)

        np.testing.assert_array_equal(result, signal)


class TestParseRegion:
    """Tests for parse_region function"""

    def test_parse_chromosome_only(self):
        """Test parsing chromosome name without coordinates"""
        from squiggy.utils import parse_region

        chrom, start, end = parse_region("chr1")

        assert chrom == "chr1"
        assert start is None
        assert end is None

    def test_parse_region_with_range(self):
        """Test parsing region with start-end range"""
        from squiggy.utils import parse_region

        chrom, start, end = parse_region("chr1:1000-2000")

        assert chrom == "chr1"
        assert start == 1000
        assert end == 2000

    def test_parse_region_with_single_position(self):
        """Test parsing region with single position"""
        from squiggy.utils import parse_region

        chrom, start, end = parse_region("chr1:1000")

        assert chrom == "chr1"
        assert start == 1000
        assert end == 1000

    def test_parse_region_with_commas(self):
        """Test parsing region with comma-separated coordinates"""
        from squiggy.utils import parse_region

        chrom, start, end = parse_region("chr1:1,000-2,000")

        assert chrom == "chr1"
        assert start == 1000
        assert end == 2000

    def test_parse_region_with_spaces(self):
        """Test parsing region with extra spaces"""
        from squiggy.utils import parse_region

        chrom, start, end = parse_region("  chr1 : 1000 - 2000  ")

        assert chrom == "chr1"
        assert start == 1000
        assert end == 2000

    def test_parse_empty_string(self):
        """Test parsing empty string"""
        from squiggy.utils import parse_region

        chrom, start, end = parse_region("")

        assert chrom is None
        assert start is None
        assert end is None

    def test_parse_whitespace_only(self):
        """Test parsing whitespace-only string"""
        from squiggy.utils import parse_region

        chrom, start, end = parse_region("   ")

        assert chrom is None
        assert start is None
        assert end is None

    def test_parse_invalid_format_multiple_colons(self):
        """Test parsing invalid format with multiple colons"""
        from squiggy.utils import parse_region

        chrom, start, end = parse_region("chr1:1000:2000")

        assert chrom is None
        assert start is None
        assert end is None

    def test_parse_invalid_format_multiple_dashes(self):
        """Test parsing invalid format with multiple dashes"""
        from squiggy.utils import parse_region

        chrom, start, end = parse_region("chr1:1000-2000-3000")

        assert chrom is None
        assert start is None
        assert end is None

    def test_parse_invalid_coordinates_non_numeric(self):
        """Test parsing with non-numeric coordinates"""
        from squiggy.utils import parse_region

        chrom, start, end = parse_region("chr1:abc-def")

        assert chrom is None
        assert start is None
        assert end is None

    def test_parse_different_chromosome_formats(self):
        """Test parsing various chromosome name formats"""
        from squiggy.utils import parse_region

        # Standard format
        chrom, _, _ = parse_region("chr1")
        assert chrom == "chr1"

        # Without 'chr' prefix
        chrom, _, _ = parse_region("1")
        assert chrom == "1"

        # Mitochondrial
        chrom, _, _ = parse_region("chrM")
        assert chrom == "chrM"

        # Sex chromosomes
        chrom, _, _ = parse_region("chrX")
        assert chrom == "chrX"

    def test_parse_none_input(self):
        """Test parsing None input"""
        from squiggy.utils import parse_region

        chrom, start, end = parse_region(None)

        assert chrom is None
        assert start is None
        assert end is None


class TestBAMIndexing:
    """Tests for BAM indexing functions"""

    def test_index_bam_file_success(self, sample_bam_file):
        """Test successful BAM indexing"""
        from squiggy.utils import index_bam_file

        # Remove existing index if present
        index_path = Path(str(sample_bam_file) + ".bai")
        if index_path.exists():
            index_path.unlink()

        # Index the file
        index_bam_file(sample_bam_file)

        # Check that index was created
        assert index_path.exists()

    def test_index_bam_file_nonexistent(self):
        """Test indexing nonexistent BAM file"""
        from squiggy.utils import index_bam_file

        with pytest.raises(FileNotFoundError, match="BAM file not found"):
            index_bam_file("/nonexistent/file.bam")

    def test_index_bam_file_none_input(self):
        """Test indexing with None input"""
        from squiggy.utils import index_bam_file

        with pytest.raises(FileNotFoundError):
            index_bam_file(None)


class TestBAMReferences:
    """Tests for BAM reference extraction"""

    def test_get_bam_references(self, indexed_bam_file):
        """Test extracting references from BAM file"""
        from squiggy.utils import get_bam_references

        references = get_bam_references(indexed_bam_file)

        assert isinstance(references, list)
        assert len(references) > 0

        # Check structure of first reference
        ref = references[0]
        assert "name" in ref
        assert "length" in ref
        assert "read_count" in ref

        assert isinstance(ref["name"], str)
        assert isinstance(ref["length"], int)
        assert isinstance(ref["read_count"], int)
        assert ref["length"] > 0
        assert ref["read_count"] >= 0

    def test_get_bam_references_nonexistent_file(self):
        """Test getting references from nonexistent file"""
        from squiggy.utils import get_bam_references

        with pytest.raises(FileNotFoundError, match="BAM file not found"):
            get_bam_references("/nonexistent/file.bam")

    def test_get_bam_references_none_input(self):
        """Test getting references with None input"""
        from squiggy.utils import get_bam_references

        with pytest.raises(FileNotFoundError):
            get_bam_references(None)


class TestWritableWorkingDirectory:
    """Tests for writable_working_directory context manager"""

    def test_writable_working_directory_changes_cwd(self):
        """Test that context manager changes working directory"""
        from squiggy.utils import writable_working_directory

        original_cwd = os.getcwd()

        with writable_working_directory():
            current_cwd = os.getcwd()
            # Should be in temp directory
            assert current_cwd != original_cwd
            assert Path(current_cwd).exists()
            assert os.access(current_cwd, os.W_OK)  # Check writable

        # Should restore original CWD
        assert os.getcwd() == original_cwd

    def test_writable_working_directory_creates_temp_dir(self):
        """Test that context manager creates temp directory"""
        from squiggy.utils import writable_working_directory

        with writable_working_directory() as temp_dir:
            assert temp_dir.exists()
            assert temp_dir.is_dir()
            assert "squiggy_workdir" in str(temp_dir)

    def test_writable_working_directory_restores_on_exception(self):
        """Test that CWD is restored even on exception"""
        from squiggy.utils import writable_working_directory

        original_cwd = os.getcwd()

        try:
            with writable_working_directory():
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should restore original CWD even after exception
        assert os.getcwd() == original_cwd


class TestGetIconPath:
    """Tests for get_icon_path function"""

    def test_get_icon_path_returns_path_or_none(self):
        """Test that get_icon_path returns Path or None"""
        from squiggy.utils import get_icon_path

        result = get_icon_path()

        # Should return either a Path object or None
        assert result is None or isinstance(result, Path)

    def test_get_logo_path_returns_path_or_none(self):
        """Test that get_logo_path returns Path or None"""
        from squiggy.utils import get_logo_path

        result = get_logo_path()

        # Should return either a Path object or None
        assert result is None or isinstance(result, Path)


class TestGetBasecallData:
    """Tests for get_basecall_data function"""

    def test_get_basecall_data_with_valid_read(self, sample_bam_file):
        """Test extracting basecall data from BAM"""
        import pysam

        from squiggy.utils import get_basecall_data

        # Find a read ID with move table
        with pysam.AlignmentFile(str(sample_bam_file), "rb", check_sq=False) as bam:
            for alignment in bam.fetch(until_eof=True):
                if alignment.has_tag("mv"):
                    read_id = alignment.query_name
                    break
            else:
                pytest.skip("No reads with move table found")

        sequence, seq_to_sig_map = get_basecall_data(sample_bam_file, read_id)

        assert sequence is not None
        assert seq_to_sig_map is not None
        assert isinstance(sequence, str)
        assert isinstance(seq_to_sig_map, np.ndarray)
        assert len(seq_to_sig_map) > 0
        assert all(base in "ACGT" for base in sequence)

    def test_get_basecall_data_nonexistent_read(self, sample_bam_file):
        """Test extracting data for nonexistent read"""
        from squiggy.utils import get_basecall_data

        sequence, seq_to_sig_map = get_basecall_data(
            sample_bam_file, "NONEXISTENT_READ"
        )

        assert sequence is None
        assert seq_to_sig_map is None

    def test_get_basecall_data_no_bam(self):
        """Test extracting data when no BAM provided"""
        from squiggy.utils import get_basecall_data

        sequence, seq_to_sig_map = get_basecall_data(None, "any_read")

        assert sequence is None
        assert seq_to_sig_map is None

    def test_get_basecall_data_invalid_bam_path(self):
        """Test extracting data from invalid BAM path"""
        from squiggy.utils import get_basecall_data

        sequence, seq_to_sig_map = get_basecall_data(
            "/nonexistent/file.bam", "any_read"
        )

        # Should handle error gracefully and return None
        assert sequence is None
        assert seq_to_sig_map is None


class TestCalculateAlignedMoveIndices:
    """Tests for calculate_aligned_move_indices function (Issue #88)"""

    def test_no_soft_clips(self):
        """Test with no soft-clipping (all bases aligned)"""
        from squiggy.utils import calculate_aligned_move_indices

        # Move table with 3 bases at indices [1, 3, 5]
        moves = np.array([0, 1, 0, 1, 0, 1])
        aligned, aligned_set = calculate_aligned_move_indices(moves, 0, 0)

        # All 3 bases should be included
        expected = np.array([1, 3, 5])
        np.testing.assert_array_equal(aligned, expected)
        assert aligned_set == {1, 3, 5}
        assert len(aligned) == 3

    def test_start_soft_clip_only(self):
        """Test with soft-clipping at start"""
        from squiggy.utils import calculate_aligned_move_indices

        # 4 bases total, first 2 soft-clipped
        moves = np.array([1, 0, 1, 0, 1, 0, 1])
        aligned, aligned_set = calculate_aligned_move_indices(moves, 2, 0)

        # Only last 2 bases should be included
        # Bases are at indices [0, 2, 4, 6], skip first 2 -> [4, 6]
        expected = np.array([4, 6])
        np.testing.assert_array_equal(aligned, expected)
        assert aligned_set == {4, 6}
        assert len(aligned) == 2

    def test_end_soft_clip_only(self):
        """Test with soft-clipping at end"""
        from squiggy.utils import calculate_aligned_move_indices

        # 3 bases total, last 1 soft-clipped
        moves = np.array([1, 0, 1, 0, 1])
        aligned, aligned_set = calculate_aligned_move_indices(moves, 0, 1)

        # Only first 2 bases should be included
        # Bases are at indices [0, 2, 4], skip last 1 -> [0, 2]
        expected = np.array([0, 2])
        np.testing.assert_array_equal(aligned, expected)
        assert aligned_set == {0, 2}
        assert len(aligned) == 2

    def test_both_soft_clips(self):
        """Test with soft-clipping at both start and end"""
        from squiggy.utils import calculate_aligned_move_indices

        # 4 bases total, 1 at start and 1 at end soft-clipped
        moves = np.array([1, 0, 1, 0, 1, 0, 1])
        aligned, aligned_set = calculate_aligned_move_indices(moves, 1, 1)

        # Only middle 2 bases should be included
        # Bases are at indices [0, 2, 4, 6], skip first and last -> [2, 4]
        expected = np.array([2, 4])
        np.testing.assert_array_equal(aligned, expected)
        assert aligned_set == {2, 4}
        assert len(aligned) == 2

    def test_all_soft_clipped(self):
        """Test with all bases soft-clipped (edge case)"""
        from squiggy.utils import calculate_aligned_move_indices

        # 2 bases total, both soft-clipped
        moves = np.array([1, 0, 1])
        aligned, aligned_set = calculate_aligned_move_indices(moves, 2, 0)

        # No aligned bases should remain
        assert len(aligned) == 0
        assert len(aligned_set) == 0

    def test_no_moves(self):
        """Test with empty move table"""
        from squiggy.utils import calculate_aligned_move_indices

        moves = np.array([])
        aligned, aligned_set = calculate_aligned_move_indices(moves, 0, 0)

        # Should return empty arrays
        assert len(aligned) == 0
        assert len(aligned_set) == 0

    def test_no_bases_in_moves(self):
        """Test with move table containing no bases (all 0s)"""
        from squiggy.utils import calculate_aligned_move_indices

        moves = np.array([0, 0, 0, 0])
        aligned, aligned_set = calculate_aligned_move_indices(moves, 0, 0)

        # No bases found, should return empty
        assert len(aligned) == 0
        assert len(aligned_set) == 0

    def test_realistic_cigar_4s7m3s(self):
        """Test realistic case matching CIGAR 4S7M3S"""
        from squiggy.utils import calculate_aligned_move_indices

        # Simulate 14 bases total: 4 soft-clipped start, 7 matched, 3 soft-clipped end
        # Create move table with stride pattern
        moves = np.array(
            [1]
            + [0] * 4  # Base 1 + stride
            + [1]
            + [0] * 4  # Base 2
            + [1]
            + [0] * 4  # Base 3
            + [1]
            + [0] * 4  # Base 4 (soft-clip end)
            + [1]
            + [0] * 4  # Base 5 (aligned start)
            + [1]
            + [0] * 4  # Base 6
            + [1]
            + [0] * 4  # Base 7
            + [1]
            + [0] * 4  # Base 8
            + [1]
            + [0] * 4  # Base 9
            + [1]
            + [0] * 4  # Base 10
            + [1]
            + [0] * 4  # Base 11 (aligned end)
            + [1]
            + [0] * 4  # Base 12 (soft-clip start)
            + [1]
            + [0] * 4  # Base 13
            + [1]
        )  # Base 14

        aligned, aligned_set = calculate_aligned_move_indices(moves, 4, 3)

        # Should have 7 aligned bases (indices of the 1s for bases 5-11)
        assert len(aligned) == 7


class TestIterAlignedBases:
    """Tests for iter_aligned_bases generator (Issue #88)"""

    def test_iter_no_soft_clips(self):
        """Test iteration with no soft-clipping"""
        from squiggy.utils import iter_aligned_bases

        # Create mock read with 3 bases, no soft-clips
        read = {
            "move_table": np.array([1, 0, 1, 0, 1]),
            "reference_start": 100,
            "query_start_offset": 0,
            "query_end_offset": 0,
        }

        result = list(iter_aligned_bases(read))

        # Should yield all 3 bases
        assert len(result) == 3

        # Check first base
        move_idx, base_idx, seq_idx, ref_pos = result[0]
        assert move_idx == 0  # First move=1 is at index 0
        assert base_idx == 0  # First aligned base
        assert seq_idx == 0  # First in sequence
        assert ref_pos == 100  # Reference start

        # Check second base
        move_idx, base_idx, seq_idx, ref_pos = result[1]
        assert move_idx == 2
        assert base_idx == 1
        assert seq_idx == 1
        assert ref_pos == 101

        # Check third base
        move_idx, base_idx, seq_idx, ref_pos = result[2]
        assert move_idx == 4
        assert base_idx == 2
        assert seq_idx == 2
        assert ref_pos == 102

    def test_iter_with_start_soft_clip(self):
        """Test iteration with start soft-clipping"""
        from squiggy.utils import iter_aligned_bases

        # 3 bases total, first 1 soft-clipped
        read = {
            "move_table": np.array([1, 0, 1, 0, 1]),
            "reference_start": 100,
            "query_start_offset": 1,  # First base soft-clipped
            "query_end_offset": 0,
        }

        result = list(iter_aligned_bases(read))

        # Should yield only 2 aligned bases (skipping first)
        assert len(result) == 2

        # First yielded base is actually second base in sequence
        move_idx, base_idx, seq_idx, ref_pos = result[0]
        assert move_idx == 2  # Second move=1
        assert base_idx == 0  # First aligned base
        assert seq_idx == 1  # Second in sequence (first is soft-clipped)
        assert ref_pos == 100  # Reference start (aligned portion)

    def test_iter_with_end_soft_clip(self):
        """Test iteration with end soft-clipping"""
        from squiggy.utils import iter_aligned_bases

        # 3 bases total, last 1 soft-clipped
        read = {
            "move_table": np.array([1, 0, 1, 0, 1]),
            "reference_start": 100,
            "query_start_offset": 0,
            "query_end_offset": 1,  # Last base soft-clipped
        }

        result = list(iter_aligned_bases(read))

        # Should yield only 2 aligned bases (skipping last)
        assert len(result) == 2

        # Last yielded base is second-to-last in sequence
        move_idx, base_idx, seq_idx, ref_pos = result[1]
        assert move_idx == 2
        assert base_idx == 1
        assert seq_idx == 1
        assert ref_pos == 101

    def test_iter_all_soft_clipped(self):
        """Test iteration when all bases are soft-clipped"""
        from squiggy.utils import iter_aligned_bases

        read = {
            "move_table": np.array([1, 0, 1]),
            "reference_start": 100,
            "query_start_offset": 2,  # Both bases soft-clipped
            "query_end_offset": 0,
        }

        result = list(iter_aligned_bases(read))

        # Should yield nothing
        assert len(result) == 0

    def test_iter_missing_offsets_defaults_to_zero(self):
        """Test that missing offset keys default to 0 (no soft-clipping)"""
        from squiggy.utils import iter_aligned_bases

        # Don't include offset keys - should default to 0
        read = {
            "move_table": np.array([1, 0, 1]),
            "reference_start": 100,
        }

        result = list(iter_aligned_bases(read))

        # Should treat as no soft-clipping (all 2 bases aligned)
        assert len(result) == 2
