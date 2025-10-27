"""Tests for region-based read search functionality."""

from pathlib import Path

import pytest


class TestRegionParsing:
    """Tests for genomic region string parsing."""

    def test_parse_chromosome_only(self):
        """Test parsing chromosome name without coordinates."""
        from squiggy.utils import parse_region

        chrom, start, end = parse_region("chr1")
        assert chrom == "chr1"
        assert start is None
        assert end is None

    def test_parse_chromosome_with_range(self):
        """Test parsing chromosome with coordinate range."""
        from squiggy.utils import parse_region

        chrom, start, end = parse_region("chr1:1000-2000")
        assert chrom == "chr1"
        assert start == 1000
        assert end == 2000

    def test_parse_chromosome_with_single_position(self):
        """Test parsing chromosome with single position."""
        from squiggy.utils import parse_region

        chrom, start, end = parse_region("chr1:1000")
        assert chrom == "chr1"
        assert start == 1000
        assert end == 1000

    def test_parse_region_with_commas(self):
        """Test parsing region with comma separators."""
        from squiggy.utils import parse_region

        chrom, start, end = parse_region("chr1:1,000-2,000")
        assert chrom == "chr1"
        assert start == 1000
        assert end == 2000

    def test_parse_invalid_region(self):
        """Test parsing invalid region strings."""
        from squiggy.utils import parse_region

        # Invalid format
        assert parse_region("chr1:abc-def") == (None, None, None)

        # Empty string
        assert parse_region("") == (None, None, None)

        # None input
        assert parse_region(None) == (None, None, None)

    def test_parse_region_with_whitespace(self):
        """Test parsing region with extra whitespace."""
        from squiggy.utils import parse_region

        chrom, start, end = parse_region("  chr1 : 1000 - 2000  ")
        assert chrom == "chr1"
        assert start == 1000
        assert end == 2000


class TestBAMIndexing:
    """Tests for BAM file indexing."""

    def test_bam_indexing(self, tmp_path, sample_bam_file):
        """Test creating BAM index file."""
        import shutil

        from squiggy.utils import index_bam_file

        # Copy BAM to temp location
        temp_bam = tmp_path / "test.bam"
        shutil.copy(sample_bam_file, temp_bam)

        # Ensure no index exists
        index_file = Path(str(temp_bam) + ".bai")
        if index_file.exists():
            index_file.unlink()

        # Create index
        index_bam_file(temp_bam)

        # Verify index was created
        assert index_file.exists()

    def test_index_missing_bam_fails(self):
        """Test that indexing non-existent BAM raises error."""
        from squiggy.utils import index_bam_file

        with pytest.raises(FileNotFoundError):
            index_bam_file(Path("/nonexistent/file.bam"))


class TestReferenceBrowsing:
    """Tests for browsing BAM file references."""

    def test_get_bam_references(self, indexed_bam_file):
        """Test getting list of references from BAM file."""
        from squiggy.utils import get_bam_references

        refs = get_bam_references(indexed_bam_file)

        # Should return a list
        assert isinstance(refs, list)
        assert len(refs) > 0

        # Check structure of first reference
        first_ref = refs[0]
        assert "name" in first_ref
        assert "length" in first_ref
        assert "read_count" in first_ref

        # Name and length should be populated
        assert isinstance(first_ref["name"], str)
        assert isinstance(first_ref["length"], int)
        assert first_ref["length"] > 0

        # Read count should be None or int (may be None if not indexed)
        assert first_ref["read_count"] is None or isinstance(
            first_ref["read_count"], int
        )

    def test_get_bam_references_with_counts(self, indexed_bam_file):
        """Test that references include read counts when indexed."""
        from squiggy.utils import get_bam_references

        refs = get_bam_references(indexed_bam_file)

        # At least some references should have read counts
        refs_with_counts = [r for r in refs if r["read_count"] is not None]

        # Should have at least one reference with reads
        refs_with_reads = [r for r in refs_with_counts if r["read_count"] > 0]
        assert len(refs_with_reads) > 0

    def test_get_bam_references_missing_file(self):
        """Test that get_bam_references raises error for missing file."""
        from squiggy.utils import get_bam_references

        with pytest.raises(FileNotFoundError):
            get_bam_references(Path("/nonexistent/file.bam"))


class TestRegionQuery:
    """Tests for querying reads by genomic region."""

    def test_query_chromosome(self, indexed_bam_file):
        """Test querying all reads on a chromosome."""
        import pysam

        from squiggy.utils import get_reads_in_region

        # First, find a chromosome that has reads
        with pysam.AlignmentFile(str(indexed_bam_file), "rb") as bam:
            chroms_with_reads = set()
            for read in bam.fetch(until_eof=True):
                if not read.is_unmapped:
                    chroms_with_reads.add(read.reference_name)
                    if len(chroms_with_reads) >= 1:
                        break

        # Query that chromosome (should return reads)
        if chroms_with_reads:
            test_chrom = list(chroms_with_reads)[0]
            reads = get_reads_in_region(indexed_bam_file, test_chrom)
            assert isinstance(reads, dict)
            assert len(reads) > 0

            # Check read info structure
            for _read_id, info in reads.items():
                assert "read_id" in info
                assert "chromosome" in info
                assert "start" in info
                assert "end" in info
                assert "strand" in info
                assert info["strand"] in ["+", "-"]

    def test_query_region_with_coordinates(self, indexed_bam_file):
        """Test querying reads in specific coordinate range."""
        import pysam

        from squiggy.utils import get_reads_in_region

        # First, find a chromosome with reads
        with pysam.AlignmentFile(str(indexed_bam_file), "rb") as bam:
            test_chrom = None
            for read in bam.fetch(until_eof=True):
                if not read.is_unmapped:
                    test_chrom = read.reference_name
                    break

        if test_chrom:
            # Get all reads from that chromosome
            all_reads = get_reads_in_region(indexed_bam_file, test_chrom)

            if len(all_reads) > 0:
                # Get first read's coordinates
                first_read = list(all_reads.values())[0]
                start = first_read["start"]
                end = first_read["end"]

                # Query overlapping region
                reads_in_region = get_reads_in_region(
                    indexed_bam_file, test_chrom, start, end
                )
                assert len(reads_in_region) >= 1

    def test_query_invalid_chromosome(self, indexed_bam_file):
        """Test querying non-existent chromosome raises error."""
        from squiggy.utils import get_reads_in_region

        with pytest.raises(ValueError, match="not found in BAM file"):
            get_reads_in_region(indexed_bam_file, "chrNONEXISTENT")

    def test_query_requires_index(self, sample_bam_file):
        """Test that querying without index raises error."""
        import pysam

        from squiggy.utils import get_reads_in_region

        # Ensure no index exists
        index_file = Path(str(sample_bam_file) + ".bai")
        if index_file.exists():
            index_file.unlink()

        # Find any valid chromosome name from the BAM
        with pysam.AlignmentFile(str(sample_bam_file), "rb", check_sq=False) as bam:
            test_chrom = bam.references[0] if bam.references else "chr1"

        with pytest.raises(ValueError, match="BAM index file not found"):
            get_reads_in_region(sample_bam_file, test_chrom)

    def test_query_empty_region(self, indexed_bam_file):
        """Test querying region with no reads."""
        import pysam

        from squiggy.utils import get_reads_in_region

        # Find any valid chromosome name from the BAM
        with pysam.AlignmentFile(str(indexed_bam_file), "rb", check_sq=False) as bam:
            test_chrom = bam.references[0] if bam.references else "chr1"

        # Query very large coordinates unlikely to have reads
        reads = get_reads_in_region(indexed_bam_file, test_chrom, 999999999, 1000000000)
        assert isinstance(reads, dict)
        # May or may not have reads, but should not error
