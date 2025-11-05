"""Tests for motif search functionality"""

import pytest

from squiggy.motif import (
    IUPAC_CODES,
    MotifMatch,
    count_motifs,
    iupac_to_regex,
    parse_region,
    reverse_complement,
    search_motif,
)


class TestIupacToRegex:
    """Tests for IUPAC to regex conversion"""

    def test_simple_bases(self):
        """Test conversion of simple bases"""
        assert iupac_to_regex("A") == "A"
        assert iupac_to_regex("C") == "C"
        assert iupac_to_regex("G") == "G"
        assert iupac_to_regex("T") == "T"

    def test_u_treated_as_t(self):
        """Test that U is treated as T"""
        assert iupac_to_regex("U") == "T"

    def test_ambiguity_codes(self):
        """Test ambiguity codes"""
        assert iupac_to_regex("R") == "[AG]"  # Purine
        assert iupac_to_regex("Y") == "[CT]"  # Pyrimidine
        assert iupac_to_regex("N") == "[ACGT]"  # Any

    def test_drach_motif(self):
        """Test DRACH motif (m6A consensus)"""
        result = iupac_to_regex("DRACH")
        assert result == "[AGT][AG]AC[ACT]"

    def test_ygcy_motif(self):
        """Test YGCY motif (m5C consensus)"""
        result = iupac_to_regex("YGCY")
        assert result == "[CT]GC[CT]"

    def test_case_insensitive(self):
        """Test that conversion is case insensitive"""
        assert iupac_to_regex("drach") == "[AGT][AG]AC[ACT]"
        assert iupac_to_regex("DrAcH") == "[AGT][AG]AC[ACT]"

    def test_all_iupac_codes(self):
        """Test all IUPAC codes are supported"""
        for code in IUPAC_CODES.keys():
            result = iupac_to_regex(code)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_invalid_code_raises_error(self):
        """Test that invalid codes raise ValueError"""
        with pytest.raises(ValueError, match="Invalid IUPAC code"):
            iupac_to_regex("Z")

        with pytest.raises(ValueError, match="Invalid IUPAC code"):
            iupac_to_regex("DRXCH")


class TestReverseComplement:
    """Tests for reverse complement function"""

    def test_simple_sequence(self):
        """Test simple DNA sequence"""
        assert reverse_complement("ATCG") == "CGAT"
        assert reverse_complement("AAAA") == "TTTT"
        assert reverse_complement("GCGC") == "GCGC"

    def test_iupac_codes(self):
        """Test reverse complement with IUPAC codes"""
        # DRACH reversed is HCARD, complemented is DGTYH
        assert reverse_complement("DRACH") == "DGTYH"
        # YGCY reversed is YCGY, complemented is RGCR
        assert reverse_complement("YGCY") == "RGCR"

    def test_case_insensitive(self):
        """Test case handling"""
        assert reverse_complement("atcg") == "CGAT"
        assert reverse_complement("AtCg") == "CGAT"

    def test_empty_sequence(self):
        """Test empty sequence"""
        assert reverse_complement("") == ""

    def test_n_handling(self):
        """Test N handling"""
        assert reverse_complement("ANCG") == "CGNT"


class TestParseRegion:
    """Tests for region parsing"""

    def test_chromosome_only(self):
        """Test parsing chromosome only"""
        chrom, start, end = parse_region("chr1")
        assert chrom == "chr1"
        assert start is None
        assert end is None

    def test_chromosome_with_range(self):
        """Test parsing chromosome with range"""
        chrom, start, end = parse_region("chr1:1000-2000")
        assert chrom == "chr1"
        assert start == 999  # Converted to 0-based
        assert end == 2000  # End is exclusive

    def test_chromosome_with_start_only(self):
        """Test parsing chromosome with start only"""
        chrom, start, end = parse_region("chr1:1000")
        assert chrom == "chr1"
        assert start == 999  # Converted to 0-based
        assert end is None

    def test_complex_chromosome_names(self):
        """Test complex chromosome names"""
        chrom, start, end = parse_region("tRNA-Ala-AGC-1-1")
        assert chrom == "tRNA-Ala-AGC-1-1"
        assert start is None
        assert end is None

        chrom, start, end = parse_region("tRNA-Ala-AGC-1-1:10-20")
        assert chrom == "tRNA-Ala-AGC-1-1"
        assert start == 9
        assert end == 20

    def test_invalid_format(self):
        """Test invalid region formats"""
        with pytest.raises(ValueError, match="Invalid region format"):
            parse_region("chr1:1000:2000")

    def test_invalid_coordinates(self):
        """Test invalid coordinates"""
        with pytest.raises(ValueError, match="Invalid coordinates"):
            parse_region("chr1:abc-def")

    def test_start_less_than_one(self):
        """Test start position less than 1"""
        with pytest.raises(ValueError, match="Start position must be >= 1"):
            parse_region("chr1:0-100")

    def test_end_before_start(self):
        """Test end before start"""
        with pytest.raises(ValueError, match="End must be greater than start"):
            parse_region("chr1:2000-1000")


class TestMotifMatch:
    """Tests for MotifMatch dataclass"""

    def test_motif_match_properties(self):
        """Test MotifMatch properties"""
        match = MotifMatch(chrom="chr1", position=100, sequence="GGACA", strand="+")

        assert match.chrom == "chr1"
        assert match.position == 100
        assert match.sequence == "GGACA"
        assert match.strand == "+"
        assert match.length == 5
        assert match.end == 105

    def test_repr(self):
        """Test string representation"""
        match = MotifMatch(chrom="chr1", position=100, sequence="GGACA", strand="+")
        repr_str = repr(match)

        assert "MotifMatch" in repr_str
        assert "chr1" in repr_str
        assert "100" in repr_str


class TestSearchMotif:
    """Tests for motif search functionality"""

    @pytest.fixture
    def fasta_file(self, test_data_dir):
        """Path to test FASTA file"""
        return test_data_dir / "yeast_trna.fa"

    def test_file_not_found(self):
        """Test that missing file raises error"""
        with pytest.raises(FileNotFoundError, match="FASTA file not found"):
            list(search_motif("nonexistent.fa", "DRACH"))

    def test_missing_index_raises_error(self, tmp_path):
        """Test that missing index raises error"""
        fasta_path = tmp_path / "test.fa"
        fasta_path.write_text(">chr1\nATCG\n")

        with pytest.raises(FileNotFoundError, match="FASTA index not found"):
            list(search_motif(fasta_path, "AT"))

    def test_simple_motif_search(self, fasta_file):
        """Test simple motif search"""
        # Search for GGG (should find multiple)
        matches = list(search_motif(fasta_file, "GGG"))
        assert len(matches) > 0

        for match in matches:
            assert isinstance(match, MotifMatch)
            assert match.sequence == "GGG"
            assert match.strand in ("+", "-")

    def test_search_with_region_filter(self, fasta_file):
        """Test search with region filter"""
        # Search only in tRNA-Ala-AGC-1-1-uncharged
        matches = list(
            search_motif(fasta_file, "GGG", region="tRNA-Ala-AGC-1-1-uncharged")
        )

        assert len(matches) > 0
        for match in matches:
            assert match.chrom == "tRNA-Ala-AGC-1-1-uncharged"

    def test_search_with_position_range(self, fasta_file):
        """Test search with position range"""
        # Search in specific region
        matches = list(
            search_motif(fasta_file, "GGG", region="tRNA-Ala-AGC-1-1-uncharged:1-50")
        )

        for match in matches:
            assert match.chrom == "tRNA-Ala-AGC-1-1-uncharged"
            # match.position is 0-based
            assert match.position < 50

    def test_search_forward_strand_only(self, fasta_file):
        """Test search on forward strand only"""
        matches_both = list(search_motif(fasta_file, "GGG", strand="both"))
        matches_forward = list(search_motif(fasta_file, "GGG", strand="+"))

        assert len(matches_forward) > 0
        assert len(matches_forward) < len(matches_both)

        for match in matches_forward:
            assert match.strand == "+"

    def test_search_reverse_strand_only(self, fasta_file):
        """Test search on reverse strand only"""
        matches_both = list(search_motif(fasta_file, "GGG", strand="both"))
        matches_reverse = list(search_motif(fasta_file, "GGG", strand="-"))

        assert len(matches_reverse) > 0
        assert len(matches_reverse) < len(matches_both)

        for match in matches_reverse:
            assert match.strand == "-"

    def test_drach_motif(self, fasta_file):
        """Test DRACH motif search (m6A consensus)"""
        matches = list(search_motif(fasta_file, "DRACH"))

        # Should find at least some matches
        assert len(matches) > 0

        for match in matches:
            # Verify sequence matches DRACH pattern
            seq = match.sequence
            assert len(seq) == 5
            assert seq[0] in "AGT"  # D
            assert seq[1] in "AG"  # R
            assert seq[2] == "A"
            assert seq[3] == "C"
            assert seq[4] in "ACT"  # H

    def test_invalid_chromosome_raises_error(self, fasta_file):
        """Test that invalid chromosome raises error"""
        with pytest.raises(ValueError, match="Chromosome.*not found"):
            list(search_motif(fasta_file, "GGG", region="nonexistent_chr"))

    def test_case_insensitive_search(self, fasta_file):
        """Test that search is case insensitive"""
        matches_upper = list(search_motif(fasta_file, "GGG"))
        matches_lower = list(search_motif(fasta_file, "ggg"))

        assert len(matches_upper) == len(matches_lower)

    def test_lazy_iteration(self, fasta_file):
        """Test that search uses lazy iteration"""
        # Create iterator but don't consume
        iterator = search_motif(fasta_file, "GGG")

        # Should be able to get first match without processing all
        first_match = next(iterator)
        assert isinstance(first_match, MotifMatch)

    def test_invalid_iupac_in_search(self, fasta_file):
        """Test that invalid IUPAC code raises error"""
        # Z is not a valid IUPAC code
        with pytest.raises(ValueError, match="Invalid IUPAC code"):
            list(search_motif(fasta_file, "ZZZZZZZZZZ"))

    def test_n_in_motif(self, fasta_file):
        """Test motif with N (any base)"""
        matches = list(search_motif(fasta_file, "GNG"))

        assert len(matches) > 0
        for match in matches:
            assert len(match.sequence) == 3
            assert match.sequence[0] == "G"
            assert match.sequence[2] == "G"


class TestCountMotifs:
    """Tests for motif counting"""

    @pytest.fixture
    def fasta_file(self, test_data_dir):
        """Path to test FASTA file"""
        return test_data_dir / "yeast_trna.fa"

    def test_count_matches_search(self, fasta_file):
        """Test that count matches search results"""
        matches = list(search_motif(fasta_file, "GGG"))
        count = count_motifs(fasta_file, "GGG")

        assert count == len(matches)

    def test_count_with_region(self, fasta_file):
        """Test counting with region filter"""
        count = count_motifs(fasta_file, "GGG", region="tRNA-Ala-AGC-1-1-uncharged")
        assert count > 0

    def test_count_with_strand(self, fasta_file):
        """Test counting with strand filter"""
        count_both = count_motifs(fasta_file, "GGG", strand="both")
        count_forward = count_motifs(fasta_file, "GGG", strand="+")
        count_reverse = count_motifs(fasta_file, "GGG", strand="-")

        assert count_both == count_forward + count_reverse
        assert count_forward > 0
        assert count_reverse > 0


class TestFastaFileIntegration:
    """Integration tests with FastaFile class"""

    @pytest.fixture
    def fasta_file(self, test_data_dir):
        """Path to test FASTA file"""
        return test_data_dir / "yeast_trna.fa"

    def test_fasta_file_class(self, fasta_file):
        """Test FastaFile class"""
        from squiggy import FastaFile

        with FastaFile(fasta_file) as fasta:
            # Check references
            assert len(fasta.references) > 0
            assert "tRNA-Ala-AGC-1-1-uncharged" in fasta.references

            # Test fetch
            seq = fasta.fetch("tRNA-Ala-AGC-1-1-uncharged", 0, 10)
            assert len(seq) == 10

            # Test motif search
            matches = list(fasta.search_motif("GGG"))
            assert len(matches) > 0

            # Test count
            count = fasta.count_motifs("GGG")
            assert count == len(matches)

    def test_fasta_file_missing_file(self):
        """Test FastaFile with missing file"""
        from squiggy import FastaFile

        with pytest.raises(FileNotFoundError, match="FASTA file not found"):
            FastaFile("nonexistent.fa")

    def test_fasta_file_missing_index(self, tmp_path):
        """Test FastaFile with missing index"""
        from squiggy import FastaFile

        fasta_path = tmp_path / "test.fa"
        fasta_path.write_text(">chr1\nATCG\n")

        with pytest.raises(FileNotFoundError, match="FASTA index not found"):
            FastaFile(fasta_path)


class TestBamFileMotifIntegration:
    """Integration tests for BamFile.get_reads_overlapping_motif()"""

    @pytest.fixture
    def fasta_file(self, test_data_dir):
        """Path to test FASTA file"""
        return test_data_dir / "yeast_trna.fa"

    @pytest.fixture
    def bam_file(self, test_data_dir):
        """Path to test BAM file"""
        return test_data_dir / "yeast_trna_mappings.bam"

    def test_get_reads_overlapping_motif(self, fasta_file, bam_file):
        """Test finding reads overlapping motif positions"""
        from squiggy import BamFile, FastaFile

        with FastaFile(fasta_file) as fasta, BamFile(bam_file) as bam:
            # Find reads overlapping GGG motifs
            overlaps = bam.get_reads_overlapping_motif(
                fasta, "GGG", region="tRNA-Ala-AGC-1-1-uncharged"
            )

            # Should find some overlapping reads
            # (depends on test data, may be 0 if no reads overlap)
            assert isinstance(overlaps, dict)

            # Check structure
            for position_key, reads in overlaps.items():
                # Position key format: "chrom:position:strand"
                parts = position_key.split(":")
                assert len(parts) == 3
                assert parts[0] == "tRNA-Ala-AGC-1-1-uncharged"

                # Should have list of reads
                assert isinstance(reads, list)
                assert len(reads) > 0

    def test_get_reads_with_path_string(self, fasta_file, bam_file):
        """Test get_reads_overlapping_motif with path string instead of FastaFile"""
        from squiggy import BamFile

        with BamFile(bam_file) as bam:
            # Pass path as string
            overlaps = bam.get_reads_overlapping_motif(
                str(fasta_file), "GGG", region="tRNA-Ala-AGC-1-1-uncharged"
            )

            assert isinstance(overlaps, dict)
