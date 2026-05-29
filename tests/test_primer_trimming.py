"""Tests for PT tag parsing and primer trimming."""

from unittest.mock import MagicMock

import numpy as np
import pysam
import pytest

from squiggy.alignment import (
    AlignedRead,
    BaseAnnotation,
    PrimerRegion,
    _parse_pt_tag,
    find_body_bounds,
    has_both_adapters,
    trim_primers,
)

# ============================================================================
# Tests for _parse_pt_tag
# ============================================================================


class TestParsePtTag:
    """Tests for _parse_pt_tag() helper."""

    def test_no_pt_tag(self):
        """Returns empty list when no PT/pt tag present."""
        alignment = MagicMock()
        alignment.has_tag.return_value = False
        assert _parse_pt_tag(alignment) == []

    def test_single_adapter(self):
        """Parses single adapter entry."""
        alignment = MagicMock()
        alignment.has_tag.side_effect = lambda tag: tag == "PT"
        alignment.get_tag.return_value = "0;10;+;5p_adapter"

        regions = _parse_pt_tag(alignment)
        assert len(regions) == 1
        assert regions[0] == PrimerRegion(
            start=0, end=10, strand="+", name="5p_adapter"
        )

    def test_multiple_adapters(self):
        """Parses multiple adapter entries separated by |."""
        alignment = MagicMock()
        alignment.has_tag.side_effect = lambda tag: tag == "PT"
        alignment.get_tag.return_value = "0;6;+;5p_adapter|63;109;+;3p_adapter_charged"

        regions = _parse_pt_tag(alignment)
        assert len(regions) == 2
        assert regions[0].start == 0
        assert regions[0].end == 6
        assert regions[0].name == "5p_adapter"
        assert regions[1].start == 63
        assert regions[1].end == 109
        assert regions[1].name == "3p_adapter_charged"

    def test_lowercase_pt_tag(self):
        """Parses lowercase pt tag."""
        alignment = MagicMock()

        def has_tag(tag):
            return tag == "pt"

        alignment.has_tag.side_effect = has_tag
        alignment.get_tag.return_value = "0;5;+;adapter"

        regions = _parse_pt_tag(alignment)
        assert len(regions) == 1
        assert regions[0].start == 0
        assert regions[0].end == 5

    def test_malformed_entry_skipped(self):
        """Malformed entries are skipped gracefully."""
        alignment = MagicMock()
        alignment.has_tag.side_effect = lambda tag: tag == "PT"
        alignment.get_tag.return_value = "0;10;+;5p_adapter|bad_entry|5;20;-;3p_adapter"

        regions = _parse_pt_tag(alignment)
        assert len(regions) == 2
        assert regions[0].name == "5p_adapter"
        assert regions[1].name == "3p_adapter"

    def test_real_bam_data(self, sample_bam_file):
        """PT tag is parsed from real BAM data."""
        with pysam.AlignmentFile(str(sample_bam_file), "rb", check_sq=False) as bam:
            for read in bam.fetch(until_eof=True):
                if read.has_tag("PT"):
                    regions = _parse_pt_tag(read)
                    assert len(regions) > 0
                    assert all(isinstance(r, PrimerRegion) for r in regions)
                    assert all(r.start < r.end for r in regions)
                    return
        pytest.skip("No reads with PT tag found in test data")


# ============================================================================
# Tests for trim_primers
# ============================================================================


class TestTrimPrimers:
    """Tests for trim_primers() function."""

    @staticmethod
    def _make_aligned_read(
        num_bases: int = 10,
        primer_regions: list[PrimerRegion] | None = None,
        signal_per_base: int = 100,
    ) -> tuple[AlignedRead, np.ndarray]:
        """Helper to create a test AlignedRead with matching signal."""
        bases = [
            BaseAnnotation(
                base="ACGT"[i % 4],
                position=i,
                signal_start=i * signal_per_base,
                signal_end=(i + 1) * signal_per_base,
                genomic_pos=1000 + i,
                quality=30,
            )
            for i in range(num_bases)
        ]
        signal = np.arange(num_bases * signal_per_base, dtype=np.float64)
        sequence = "".join(b.base for b in bases)

        read = AlignedRead(
            read_id="test_read",
            sequence=sequence,
            bases=bases,
            chromosome="chr1",
            genomic_start=1000,
            genomic_end=1000 + num_bases,
            strand="+",
            primer_regions=primer_regions or [],
        )
        return read, signal

    def test_no_primer_regions_passthrough(self):
        """Returns unchanged when no primer regions."""
        read, signal = self._make_aligned_read(num_bases=10)
        trimmed_read, trimmed_signal = trim_primers(read, signal)
        assert trimmed_read is read
        assert trimmed_signal is signal

    def test_trim_5p_adapter(self):
        """Trims 5' adapter at start of read."""
        primers = [PrimerRegion(start=0, end=3, strand="+", name="5p_adapter")]
        read, signal = self._make_aligned_read(num_bases=10, primer_regions=primers)

        trimmed_read, trimmed_signal = trim_primers(read, signal)

        # Should have 7 bases (10 - 3)
        assert len(trimmed_read.bases) == 7
        # Positions should be rebased 0-6
        assert [b.position for b in trimmed_read.bases] == list(range(7))
        # First base should be what was at position 3
        assert trimmed_read.bases[0].genomic_pos == 1003
        # Signal should be trimmed
        assert len(trimmed_signal) == 7 * 100
        # Signal should start contiguous from 0
        assert trimmed_read.bases[0].signal_start == 0
        assert trimmed_read.bases[0].signal_end == 100

    def test_trim_both_adapters(self):
        """Trims both 5' and 3' adapters."""
        primers = [
            PrimerRegion(start=0, end=2, strand="+", name="5p_adapter"),
            PrimerRegion(start=8, end=10, strand="+", name="3p_adapter"),
        ]
        read, signal = self._make_aligned_read(num_bases=10, primer_regions=primers)

        trimmed_read, trimmed_signal = trim_primers(read, signal)

        # Should have 6 bases (10 - 2 - 2)
        assert len(trimmed_read.bases) == 6
        assert [b.position for b in trimmed_read.bases] == list(range(6))
        # Genomic positions preserved
        assert trimmed_read.bases[0].genomic_pos == 1002
        assert trimmed_read.bases[-1].genomic_pos == 1007
        # Signal length matches
        assert len(trimmed_signal) == 6 * 100

    def test_signal_contiguous_after_trim(self):
        """Signal ranges are contiguous after trimming."""
        primers = [PrimerRegion(start=0, end=3, strand="+", name="5p")]
        read, signal = self._make_aligned_read(num_bases=10, primer_regions=primers)

        trimmed_read, trimmed_signal = trim_primers(read, signal)

        # Signal ranges should be contiguous
        for i in range(len(trimmed_read.bases) - 1):
            assert (
                trimmed_read.bases[i].signal_end
                == trimmed_read.bases[i + 1].signal_start
            )

        # Last base signal_end should equal total signal length
        assert trimmed_read.bases[-1].signal_end == len(trimmed_signal)

    def test_sequence_rebuilt(self):
        """Sequence string is rebuilt from remaining bases."""
        primers = [PrimerRegion(start=0, end=2, strand="+", name="5p")]
        read, signal = self._make_aligned_read(num_bases=10, primer_regions=primers)

        trimmed_read, trimmed_signal = trim_primers(read, signal)

        expected_seq = "".join(b.base for b in trimmed_read.bases)
        assert trimmed_read.sequence == expected_seq

    def test_primer_regions_preserved(self):
        """Original primer regions are preserved on trimmed read."""
        primers = [PrimerRegion(start=0, end=3, strand="+", name="5p")]
        read, signal = self._make_aligned_read(num_bases=10, primer_regions=primers)

        trimmed_read, trimmed_signal = trim_primers(read, signal)

        assert trimmed_read.primer_regions == primers

    def test_trim_all_bases_returns_unchanged(self):
        """If trimming would remove all bases, return unchanged."""
        primers = [PrimerRegion(start=0, end=10, strand="+", name="full_adapter")]
        read, signal = self._make_aligned_read(num_bases=10, primer_regions=primers)

        trimmed_read, trimmed_signal = trim_primers(read, signal)

        # Should return original, not empty
        assert trimmed_read is read
        assert trimmed_signal is signal

    def test_modifications_filtered(self):
        """Modifications in primer regions are removed, others rebased."""
        primers = [PrimerRegion(start=0, end=3, strand="+", name="5p")]
        read, signal = self._make_aligned_read(num_bases=10, primer_regions=primers)

        # Add modifications at positions 1 (in primer) and 5 (kept)
        mod_in_primer = MagicMock()
        mod_in_primer.position = 1
        mod_kept = MagicMock()
        mod_kept.position = 5
        read.modifications = [mod_in_primer, mod_kept]

        trimmed_read, _ = trim_primers(read, signal)

        # Only the kept mod should remain, rebased from pos 5 -> 2
        assert len(trimmed_read.modifications) == 1
        assert trimmed_read.modifications[0].position == 2


# ============================================================================
# Tests for has_both_adapters
# ============================================================================


class TestHasBothAdapters:
    """Tests for has_both_adapters() helper."""

    def test_empty_list(self):
        assert has_both_adapters([]) is False

    def test_only_5p(self):
        regions = [PrimerRegion(start=0, end=6, strand="+", name="5p_adapter")]
        assert has_both_adapters(regions) is False

    def test_only_3p(self):
        regions = [
            PrimerRegion(start=60, end=100, strand="+", name="3p_adapter_charged")
        ]
        assert has_both_adapters(regions) is False

    def test_both_adapters(self):
        regions = [
            PrimerRegion(start=0, end=6, strand="+", name="5p_adapter"),
            PrimerRegion(start=60, end=100, strand="+", name="3p_adapter_charged"),
        ]
        assert has_both_adapters(regions) is True


# ============================================================================
# Tests for aggregate primer trimming
# ============================================================================


class TestAggregatePrimerTrimming:
    """Tests for primer trimming in aggregate plot path."""

    def test_aggregate_primer_filtering(self, sample_bam_file, sample_pod5_file):
        """Reads extracted for aggregate include primer_regions."""
        from squiggy.utils.bam import extract_reads_for_reference

        reads = extract_reads_for_reference(
            pod5_file=sample_pod5_file,
            bam_file=sample_bam_file,
            reference_name="host-tRNA-Ala-TGC-1-1",
            max_reads=10,
        )
        assert len(reads) > 0
        # All demo reads should have primer_regions
        for rd in reads:
            assert "primer_regions" in rd
            assert len(rd["primer_regions"]) > 0

    def test_aggregate_primer_filtering_both_adapters(
        self, sample_bam_file, sample_pod5_file
    ):
        """All demo reads should have both 5' and 3' adapters."""
        from squiggy.utils.bam import extract_reads_for_reference

        reads = extract_reads_for_reference(
            pod5_file=sample_pod5_file,
            bam_file=sample_bam_file,
            reference_name="host-tRNA-Ala-TGC-1-1",
            max_reads=60,
        )
        for rd in reads:
            assert has_both_adapters(rd["primer_regions"]), (
                f"Read {rd['read_id']} missing adapter(s)"
            )

    def test_fasta_body_bounds(self, sample_fasta_file):
        """FASTA-based body bounds should give 76-77 nt for all demo tRNAs."""
        for ref in [
            "host-tRNA-Ala-TGC-1-1",
            "host-tRNA-Gly-GCC-1-1",
            "host-tRNA-Arg-ACG-1-1",
        ]:
            bounds = find_body_bounds(sample_fasta_file, ref)
            assert bounds is not None, f"No body bounds found for {ref}"
            body_start, body_end = bounds
            body_width = body_end - body_start
            assert 73 <= body_width <= 78, (
                f"{ref}: body width {body_width} (pos {body_start}-{body_end}) "
                "outside expected range for tRNA"
            )

    def test_fasta_body_bounds_missing_reference(self, sample_fasta_file):
        """Returns None for references not in the FASTA."""
        assert find_body_bounds(sample_fasta_file, "nonexistent-ref") is None

    def test_fasta_body_bounds_no_primer(self, tmp_path):
        """Returns None when reference doesn't contain primer sequences."""
        import pysam

        # Create a FASTA without primer sequences
        fasta_path = tmp_path / "no_primer.fa"
        fasta_path.write_text(">test_ref\nACGTACGTACGT\n")
        pysam.faidx(str(fasta_path))

        assert find_body_bounds(str(fasta_path), "test_ref") is None


# ============================================================================
# Integration test with real BAM data
# ============================================================================


class TestPrimerTrimmingIntegration:
    """Integration tests using real BAM data."""

    def test_alignment_has_primer_regions(self, sample_bam_file):
        """AlignedRead objects from real BAM have primer_regions populated."""
        from squiggy.alignment import extract_alignment_from_bam

        with pysam.AlignmentFile(str(sample_bam_file), "rb", check_sq=False) as bam:
            for read in bam.fetch(until_eof=True):
                if read.has_tag("PT") and read.has_tag("mv"):
                    aligned = extract_alignment_from_bam(
                        sample_bam_file, read.query_name
                    )
                    assert aligned is not None
                    assert len(aligned.primer_regions) > 0
                    # Verify primer regions are within sequence bounds
                    for pr in aligned.primer_regions:
                        assert pr.start >= 0
                        assert pr.end <= len(aligned.sequence)
                    return
        pytest.skip("No reads with both PT and mv tags found")


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_bam_file():
    """Path to sample BAM file with PT tags."""
    from pathlib import Path

    return (
        Path(__file__).parent.parent / "squiggy" / "data" / "ecoli_trna_wt_mappings.bam"
    )


@pytest.fixture
def sample_pod5_file():
    """Path to sample POD5 file."""
    from pathlib import Path

    return (
        Path(__file__).parent.parent / "squiggy" / "data" / "ecoli_trna_wt_reads.pod5"
    )


@pytest.fixture
def sample_fasta_file():
    """Path to sample FASTA file with primer-flanked tRNA references."""
    from pathlib import Path

    return Path(__file__).parent.parent / "squiggy" / "data" / "ecoli_trna.fa"
