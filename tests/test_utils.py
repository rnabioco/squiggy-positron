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


class TestModelProvenance:
    """Tests for ModelProvenance dataclass"""

    def test_repr_with_all_fields(self):
        """Test __repr__ with all fields populated"""
        from squiggy.utils import ModelProvenance

        prov = ModelProvenance(
            model_name="dorado",
            model_version="0.4.0",
            basecalling_model="dna_r10.4.1_e8.2_400bps_hac",
        )

        repr_str = repr(prov)
        assert "dorado" in repr_str
        assert "0.4.0" in repr_str
        assert "dna_r10.4.1_e8.2_400bps_hac" in repr_str

    def test_repr_with_partial_fields(self):
        """Test __repr__ with only some fields populated"""
        from squiggy.utils import ModelProvenance

        prov = ModelProvenance(model_name="guppy")

        repr_str = repr(prov)
        assert "guppy" in repr_str
        assert "ModelProvenance" in repr_str

    def test_repr_with_no_fields(self):
        """Test __repr__ with no fields populated"""
        from squiggy.utils import ModelProvenance

        prov = ModelProvenance()

        repr_str = repr(prov)
        assert "Unknown" in repr_str

    def test_matches_same_model(self):
        """Test matches() returns True for same model"""
        from squiggy.utils import ModelProvenance

        prov_a = ModelProvenance(
            model_name="dorado", basecalling_model="dna_r10.4.1_e8.2_400bps_hac"
        )
        prov_b = ModelProvenance(
            model_name="dorado", basecalling_model="dna_r10.4.1_e8.2_400bps_hac"
        )

        assert prov_a.matches(prov_b)

    def test_matches_different_versions_same_model(self):
        """Test matches() returns True for different versions of same model"""
        from squiggy.utils import ModelProvenance

        prov_a = ModelProvenance(
            model_name="dorado",
            model_version="0.4.0",
            basecalling_model="dna_r10.4.1_e8.2_400bps_hac",
        )
        prov_b = ModelProvenance(
            model_name="dorado",
            model_version="0.5.0",
            basecalling_model="dna_r10.4.1_e8.2_400bps_hac",
        )

        # Should match despite different versions
        assert prov_a.matches(prov_b)

    def test_matches_different_model_name(self):
        """Test matches() returns False for different model names"""
        from squiggy.utils import ModelProvenance

        prov_a = ModelProvenance(
            model_name="dorado", basecalling_model="dna_r10.4.1_e8.2_400bps_hac"
        )
        prov_b = ModelProvenance(
            model_name="guppy", basecalling_model="dna_r10.4.1_e8.2_400bps_hac"
        )

        assert not prov_a.matches(prov_b)

    def test_matches_different_basecalling_model(self):
        """Test matches() returns False for different basecalling models"""
        from squiggy.utils import ModelProvenance

        prov_a = ModelProvenance(
            model_name="dorado", basecalling_model="dna_r10.4.1_e8.2_400bps_hac"
        )
        prov_b = ModelProvenance(
            model_name="dorado", basecalling_model="rna004_130bps_hac"
        )

        assert not prov_a.matches(prov_b)

    def test_matches_with_none(self):
        """Test matches() returns False when comparing to None"""
        from squiggy.utils import ModelProvenance

        prov = ModelProvenance(model_name="dorado")

        assert not prov.matches(None)


class TestReverseComplement:
    """Tests for reverse_complement function"""

    def test_simple_sequence(self):
        """Test reverse complement of simple sequence"""
        from squiggy.utils import reverse_complement

        assert reverse_complement("ATCG") == "CGAT"

    def test_palindrome(self):
        """Test reverse complement of palindromic sequence"""
        from squiggy.utils import reverse_complement

        assert reverse_complement("ATAT") == "ATAT"

    def test_all_a(self):
        """Test reverse complement of all A's"""
        from squiggy.utils import reverse_complement

        assert reverse_complement("AAAA") == "TTTT"

    def test_all_t(self):
        """Test reverse complement of all T's"""
        from squiggy.utils import reverse_complement

        assert reverse_complement("TTTT") == "AAAA"

    def test_with_n(self):
        """Test reverse complement with N (any base)"""
        from squiggy.utils import reverse_complement

        assert reverse_complement("ATNGC") == "GCNAT"

    def test_empty_sequence(self):
        """Test reverse complement of empty sequence"""
        from squiggy.utils import reverse_complement

        assert reverse_complement("") == ""

    def test_single_base(self):
        """Test reverse complement of single base"""
        from squiggy.utils import reverse_complement

        assert reverse_complement("A") == "T"
        assert reverse_complement("T") == "A"
        assert reverse_complement("C") == "G"
        assert reverse_complement("G") == "C"


class TestValidateBAMReadsInPOD5:
    """Tests for validate_bam_reads_in_pod5 function"""

    def test_validate_matching_files(self, sample_pod5_file, sample_bam_file):
        """Test validation with matching POD5 and BAM files"""
        from squiggy.utils import validate_bam_reads_in_pod5

        result = validate_bam_reads_in_pod5(sample_bam_file, sample_pod5_file)

        assert isinstance(result, dict)
        assert "is_valid" in result
        assert "bam_read_count" in result
        assert "pod5_read_count" in result
        assert "missing_count" in result
        assert "missing_reads" in result

        # The sample files should match
        assert result["is_valid"] is True
        assert result["missing_count"] == 0
        assert len(result["missing_reads"]) == 0

    def test_validate_counts(self, sample_pod5_file, sample_bam_file):
        """Test that read counts are reported correctly"""
        from squiggy.utils import validate_bam_reads_in_pod5

        result = validate_bam_reads_in_pod5(sample_bam_file, sample_pod5_file)

        assert result["bam_read_count"] > 0
        assert result["pod5_read_count"] > 0
        assert result["bam_read_count"] <= result["pod5_read_count"]


class TestGetReadToReferenceMapping:
    """Tests for get_read_to_reference_mapping function"""

    def test_get_mapping(self, sample_bam_file):
        """Test getting read to reference mapping"""
        import pysam

        from squiggy.utils import get_read_to_reference_mapping

        # Get some read IDs from the BAM file
        read_ids = []
        with pysam.AlignmentFile(str(sample_bam_file), "rb", check_sq=False) as bam:
            for read in bam.fetch(until_eof=True):
                if not read.is_unmapped:
                    read_ids.append(read.query_name)
                    if len(read_ids) >= 10:
                        break

        mapping = get_read_to_reference_mapping(sample_bam_file, read_ids)

        assert isinstance(mapping, dict)
        assert len(mapping) > 0

        # Check that mapped reads are in the result
        for read_id, ref_name in mapping.items():
            assert read_id in read_ids
            assert isinstance(ref_name, str)
            assert len(ref_name) > 0

    def test_get_mapping_nonexistent_file(self):
        """Test with nonexistent BAM file"""
        from squiggy.utils import get_read_to_reference_mapping

        mapping = get_read_to_reference_mapping("/nonexistent/file.bam", ["read1"])

        assert mapping == {}

    def test_get_mapping_empty_read_list(self, sample_bam_file):
        """Test with empty read ID list"""
        from squiggy.utils import get_read_to_reference_mapping

        mapping = get_read_to_reference_mapping(sample_bam_file, [])

        assert mapping == {}

    def test_get_mapping_none_bam(self):
        """Test with None BAM file"""
        from squiggy.utils import get_read_to_reference_mapping

        mapping = get_read_to_reference_mapping(None, ["read1"])

        assert mapping == {}


class TestGetReadsInRegion:
    """Tests for get_reads_in_region function"""

    def test_get_reads_entire_chromosome(self, indexed_bam_file):
        """Test querying entire chromosome"""
        import pysam

        from squiggy.utils import get_reads_in_region

        # Find a reference that actually has reads
        ref_name = None
        with pysam.AlignmentFile(str(indexed_bam_file), "rb") as bam:
            for read in bam.fetch(until_eof=True):
                if not read.is_unmapped:
                    ref_name = bam.get_reference_name(read.reference_id)
                    break

        if ref_name is None:
            pytest.skip("No reads found in BAM file")

        reads = get_reads_in_region(indexed_bam_file, ref_name)

        assert isinstance(reads, dict)
        assert len(reads) > 0

        # Check structure of first read
        for _read_id, read_info in reads.items():
            assert "read_id" in read_info
            assert "chromosome" in read_info
            assert "start" in read_info
            assert "end" in read_info
            assert "strand" in read_info
            assert "is_reverse" in read_info

            assert read_info["chromosome"] == ref_name
            assert read_info["strand"] in ["+", "-"]
            assert isinstance(read_info["is_reverse"], bool)
            break

    def test_get_reads_specific_region(self, indexed_bam_file):
        """Test querying specific region"""
        import pysam

        from squiggy.utils import get_reads_in_region

        # Get first reference name and a region within it
        with pysam.AlignmentFile(str(indexed_bam_file), "rb") as bam:
            ref_name = bam.references[0]
            ref_length = bam.lengths[0]

        # Query a region in the middle
        start = ref_length // 4
        end = ref_length // 2

        reads = get_reads_in_region(indexed_bam_file, ref_name, start, end)

        assert isinstance(reads, dict)

        # All reads should overlap the queried region
        for read_info in reads.values():
            assert read_info["start"] < end
            assert read_info["end"] > start

    def test_get_reads_nonexistent_file(self):
        """Test with nonexistent BAM file"""
        from squiggy.utils import get_reads_in_region

        with pytest.raises(FileNotFoundError, match="BAM file not found"):
            get_reads_in_region("/nonexistent/file.bam", "chr1")

    def test_get_reads_not_indexed(self, sample_bam_file):
        """Test with BAM file that is not indexed"""
        from pathlib import Path

        from squiggy.utils import get_reads_in_region

        # Remove index if it exists
        bai_path = Path(str(sample_bam_file) + ".bai")
        if bai_path.exists():
            bai_path.unlink()

        with pytest.raises(ValueError, match="BAM index file not found"):
            get_reads_in_region(sample_bam_file, "chr1")

    def test_get_reads_invalid_chromosome(self, indexed_bam_file):
        """Test with invalid chromosome name"""
        from squiggy.utils import get_reads_in_region

        with pytest.raises(ValueError, match="not found in BAM file"):
            get_reads_in_region(indexed_bam_file, "INVALID_CHR")


class TestGetAvailableReadsForReference:
    """Tests for get_available_reads_for_reference function"""

    def test_count_reads(self, indexed_bam_file):
        """Test counting reads for a reference"""
        import pysam

        from squiggy.utils import get_available_reads_for_reference

        # Find a reference that actually has reads
        ref_name = None
        with pysam.AlignmentFile(str(indexed_bam_file), "rb", check_sq=False) as bam:
            for read in bam.fetch(until_eof=True):
                if not read.is_unmapped:
                    ref_name = bam.get_reference_name(read.reference_id)
                    break

        if ref_name is None:
            pytest.skip("No reads found in BAM file")

        count = get_available_reads_for_reference(indexed_bam_file, ref_name)

        assert isinstance(count, int)
        assert count > 0

    def test_count_reads_nonexistent_reference(self, sample_bam_file):
        """Test counting reads for nonexistent reference"""
        from squiggy.utils import get_available_reads_for_reference

        # Count should be 0 for nonexistent reference
        count = get_available_reads_for_reference(sample_bam_file, "NONEXISTENT_REF")

        assert count == 0


class TestExtractModelProvenance:
    """Tests for extract_model_provenance function"""

    def test_extract_from_sample_bam(self, sample_bam_file):
        """Test extracting model provenance from sample BAM file"""
        from squiggy.utils import extract_model_provenance

        prov = extract_model_provenance(str(sample_bam_file))

        assert isinstance(prov, object)
        assert hasattr(prov, "model_name")
        assert hasattr(prov, "model_version")
        assert hasattr(prov, "basecalling_model")

    def test_extract_nonexistent_file(self):
        """Test extracting from nonexistent file"""
        from squiggy.utils import extract_model_provenance

        with pytest.raises(FileNotFoundError, match="BAM file not found"):
            extract_model_provenance("/nonexistent/file.bam")


class TestValidateSQHeaders:
    """Tests for validate_sq_headers function"""

    def test_validate_same_file(self, sample_bam_file):
        """Test validating same file against itself"""
        from squiggy.utils import validate_sq_headers

        result = validate_sq_headers(str(sample_bam_file), str(sample_bam_file))

        assert isinstance(result, dict)
        assert result["is_valid"] is True
        assert len(result["missing_in_a"]) == 0
        assert len(result["missing_in_b"]) == 0
        assert result["matching_count"] > 0

    def test_validate_structure(self, sample_bam_file):
        """Test that validation result has correct structure"""
        from squiggy.utils import validate_sq_headers

        result = validate_sq_headers(str(sample_bam_file), str(sample_bam_file))

        assert "is_valid" in result
        assert "references_a" in result
        assert "references_b" in result
        assert "missing_in_a" in result
        assert "missing_in_b" in result
        assert "matching_count" in result

    def test_validate_nonexistent_file_a(self, sample_bam_file):
        """Test with nonexistent file A"""
        from squiggy.utils import validate_sq_headers

        with pytest.raises(FileNotFoundError, match="BAM file A not found"):
            validate_sq_headers("/nonexistent/file.bam", str(sample_bam_file))

    def test_validate_nonexistent_file_b(self, sample_bam_file):
        """Test with nonexistent file B"""
        from squiggy.utils import validate_sq_headers

        with pytest.raises(FileNotFoundError, match="BAM file B not found"):
            validate_sq_headers(str(sample_bam_file), "/nonexistent/file.bam")


class TestCompareReadSets:
    """Tests for compare_read_sets function"""

    def test_compare_identical_sets(self):
        """Test comparing identical read sets"""
        from squiggy.utils import compare_read_sets

        reads_a = ["read1", "read2", "read3"]
        reads_b = ["read1", "read2", "read3"]

        result = compare_read_sets(reads_a, reads_b)

        assert result["common_count"] == 3
        assert result["unique_a_count"] == 0
        assert result["unique_b_count"] == 0
        assert result["overlap_percent_a"] == 100.0
        assert result["overlap_percent_b"] == 100.0

    def test_compare_disjoint_sets(self):
        """Test comparing completely different read sets"""
        from squiggy.utils import compare_read_sets

        reads_a = ["read1", "read2", "read3"]
        reads_b = ["read4", "read5", "read6"]

        result = compare_read_sets(reads_a, reads_b)

        assert result["common_count"] == 0
        assert result["unique_a_count"] == 3
        assert result["unique_b_count"] == 3
        assert result["overlap_percent_a"] == 0.0
        assert result["overlap_percent_b"] == 0.0

    def test_compare_partial_overlap(self):
        """Test comparing sets with partial overlap"""
        from squiggy.utils import compare_read_sets

        reads_a = ["read1", "read2", "read3", "read4"]
        reads_b = ["read3", "read4", "read5", "read6"]

        result = compare_read_sets(reads_a, reads_b)

        assert result["common_count"] == 2
        assert result["unique_a_count"] == 2
        assert result["unique_b_count"] == 2
        assert result["overlap_percent_a"] == 50.0
        assert result["overlap_percent_b"] == 50.0

    def test_compare_empty_sets(self):
        """Test comparing empty read sets"""
        from squiggy.utils import compare_read_sets

        result = compare_read_sets([], [])

        assert result["common_count"] == 0
        assert result["unique_a_count"] == 0
        assert result["unique_b_count"] == 0
        assert result["overlap_percent_a"] == 0.0
        assert result["overlap_percent_b"] == 0.0


class TestCalculateDeltaStats:
    """Tests for calculate_delta_stats function"""

    def test_calculate_delta_mean_signal(self):
        """Test calculating delta for mean signal"""
        from squiggy.utils import calculate_delta_stats

        stats_a = {
            "positions": np.array([1, 2, 3]),
            "mean_signal": np.array([10.0, 20.0, 30.0]),
        }
        stats_b = {
            "positions": np.array([1, 2, 3]),
            "mean_signal": np.array([15.0, 25.0, 35.0]),
        }

        result = calculate_delta_stats(stats_a, stats_b, ["mean_signal"])

        assert "delta_mean_signal" in result
        np.testing.assert_array_almost_equal(
            result["delta_mean_signal"], np.array([5.0, 5.0, 5.0])
        )

    def test_calculate_delta_auto_detect(self):
        """Test auto-detecting stats to calculate deltas for"""
        from squiggy.utils import calculate_delta_stats

        stats_a = {
            "positions": np.array([1, 2, 3]),
            "mean_signal": np.array([10.0, 20.0, 30.0]),
            "std_signal": np.array([1.0, 2.0, 3.0]),
        }
        stats_b = {
            "positions": np.array([1, 2, 3]),
            "mean_signal": np.array([15.0, 25.0, 35.0]),
            "std_signal": np.array([1.5, 2.5, 3.5]),
        }

        result = calculate_delta_stats(stats_a, stats_b)

        assert "delta_mean_signal" in result
        assert "delta_std_signal" in result

    def test_calculate_delta_different_lengths(self):
        """Test delta calculation with different array lengths"""
        from squiggy.utils import calculate_delta_stats

        stats_a = {
            "mean_signal": np.array([10.0, 20.0, 30.0, 40.0]),
        }
        stats_b = {
            "mean_signal": np.array([15.0, 25.0]),
        }

        result = calculate_delta_stats(stats_a, stats_b, ["mean_signal"])

        # Should only calculate delta for overlapping region
        assert len(result["delta_mean_signal"]) == 2


class TestCompareSignalDistributions:
    """Tests for compare_signal_distributions function"""

    def test_compare_identical_distributions(self):
        """Test comparing identical signal distributions"""
        from squiggy.utils import compare_signal_distributions

        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = compare_signal_distributions(signal, signal)

        assert result["mean_diff"] == 0.0
        assert result["std_diff"] == 0.0
        assert result["mean_a"] == result["mean_b"]
        assert result["std_a"] == result["std_b"]

    def test_compare_different_distributions(self):
        """Test comparing different signal distributions"""
        from squiggy.utils import compare_signal_distributions

        signal_a = np.array([1.0, 2.0, 3.0])
        signal_b = np.array([2.0, 3.0, 4.0])

        result = compare_signal_distributions(signal_a, signal_b)

        assert result["mean_diff"] == pytest.approx(1.0)
        assert result["mean_b"] > result["mean_a"]

    def test_compare_distribution_structure(self):
        """Test that result has all required fields"""
        from squiggy.utils import compare_signal_distributions

        signal_a = np.array([1.0, 2.0, 3.0])
        signal_b = np.array([2.0, 3.0, 4.0])

        result = compare_signal_distributions(signal_a, signal_b)

        required_fields = [
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

        for field in required_fields:
            assert field in result
            assert isinstance(result[field], float)
