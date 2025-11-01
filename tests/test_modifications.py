"""Tests for base modification parsing"""

from pathlib import Path

import pytest


class TestModificationAnnotation:
    """Tests for ModificationAnnotation dataclass"""

    def test_modification_annotation_creation(self):
        """Test that ModificationAnnotation can be created"""
        from squiggy.modifications import ModificationAnnotation

        mod = ModificationAnnotation(
            position=10,
            genomic_pos=1000,
            mod_code="m",
            canonical_base="C",
            probability=0.95,
            signal_start=100,
            signal_end=150,
        )

        assert mod.position == 10
        assert mod.genomic_pos == 1000
        assert mod.mod_code == "m"
        assert mod.canonical_base == "C"
        assert mod.probability == 0.95
        assert mod.signal_start == 100
        assert mod.signal_end == 150

    def test_modification_annotation_with_chebi_code(self):
        """Test ModificationAnnotation with ChEBI numeric code"""
        from squiggy.modifications import ModificationAnnotation

        mod = ModificationAnnotation(
            position=5,
            genomic_pos=None,
            mod_code=17596,  # ChEBI code for inosine
            canonical_base="A",
            probability=0.85,
            signal_start=50,
            signal_end=75,
        )

        assert mod.mod_code == 17596
        assert isinstance(mod.mod_code, int)


class TestExtractModificationsFromAlignment:
    """Tests for extract_modifications_from_alignment function"""

    def test_extract_modifications_no_mods(self):
        """Test that function returns empty list when no modifications"""
        from squiggy.alignment import BaseAnnotation
        from squiggy.modifications import extract_modifications_from_alignment

        # Create a mock alignment without modifications
        class MockAlignment:
            modified_bases = None

        bases = [
            BaseAnnotation(
                position=0, base="A", signal_start=0, signal_end=10, genomic_pos=100
            )
        ]

        mods = extract_modifications_from_alignment(MockAlignment(), bases)
        assert mods == []

    def test_extract_modifications_empty_dict(self):
        """Test that function handles empty modified_bases dict"""
        from squiggy.alignment import BaseAnnotation
        from squiggy.modifications import extract_modifications_from_alignment

        # Create a mock alignment with empty modifications
        class MockAlignment:
            modified_bases = {}

        bases = [
            BaseAnnotation(
                position=0, base="A", signal_start=0, signal_end=10, genomic_pos=100
            )
        ]

        mods = extract_modifications_from_alignment(MockAlignment(), bases)
        assert mods == []

    def test_extract_modifications_with_mods(self):
        """Test that function extracts modifications correctly"""
        from squiggy.alignment import BaseAnnotation
        from squiggy.modifications import extract_modifications_from_alignment

        # Create mock alignment with modifications
        class MockAlignment:
            # Format: {(canonical_base, strand, mod_code): [(position, quality), ...]}
            modified_bases = {
                ("C", 0, "m"): [(0, 243), (2, 230)],  # quality = 256 * probability
            }

        bases = [
            BaseAnnotation(
                position=0, base="C", signal_start=0, signal_end=10, genomic_pos=100
            ),
            BaseAnnotation(
                position=1, base="A", signal_start=10, signal_end=20, genomic_pos=101
            ),
            BaseAnnotation(
                position=2, base="C", signal_start=20, signal_end=30, genomic_pos=102
            ),
        ]

        mods = extract_modifications_from_alignment(MockAlignment(), bases)

        assert len(mods) == 2
        assert all(mod.mod_code == "m" for mod in mods)
        assert all(mod.canonical_base == "C" for mod in mods)
        assert mods[0].position == 0
        assert mods[1].position == 2

        # Check probability conversion (quality / 256)
        assert abs(mods[0].probability - (243 / 256)) < 0.01
        assert abs(mods[1].probability - (230 / 256)) < 0.01

    def test_extract_modifications_skips_unknown_probability(self):
        """Test that modifications with quality=-1 are skipped"""
        from squiggy.alignment import BaseAnnotation
        from squiggy.modifications import extract_modifications_from_alignment

        class MockAlignment:
            modified_bases = {
                ("C", 0, "m"): [(0, -1), (1, 200)],  # -1 = unknown probability
            }

        bases = [
            BaseAnnotation(
                position=0, base="C", signal_start=0, signal_end=10, genomic_pos=100
            ),
            BaseAnnotation(
                position=1, base="C", signal_start=10, signal_end=20, genomic_pos=101
            ),
        ]

        mods = extract_modifications_from_alignment(MockAlignment(), bases)

        # Should only get the one with known probability
        assert len(mods) == 1
        assert mods[0].position == 1

    def test_extract_modifications_caps_probability(self):
        """Test that probability is capped at 1.0"""
        from squiggy.alignment import BaseAnnotation
        from squiggy.modifications import extract_modifications_from_alignment

        class MockAlignment:
            modified_bases = {
                ("C", 0, "m"): [(0, 260)],  # Would give probability > 1.0
            }

        bases = [
            BaseAnnotation(
                position=0, base="C", signal_start=0, signal_end=10, genomic_pos=100
            ),
        ]

        mods = extract_modifications_from_alignment(MockAlignment(), bases)

        assert len(mods) == 1
        assert mods[0].probability <= 1.0

    def test_extract_modifications_with_chebi_code(self):
        """Test that function handles ChEBI numeric codes"""
        from squiggy.alignment import BaseAnnotation
        from squiggy.modifications import extract_modifications_from_alignment

        class MockAlignment:
            modified_bases = {
                ("A", 0, 17596): [(0, 200)],  # ChEBI code for inosine
            }

        bases = [
            BaseAnnotation(
                position=0, base="A", signal_start=0, signal_end=10, genomic_pos=100
            ),
        ]

        mods = extract_modifications_from_alignment(MockAlignment(), bases)

        assert len(mods) == 1
        assert mods[0].mod_code == 17596
        assert isinstance(mods[0].mod_code, int)


class TestDetectModificationProvenance:
    """Tests for detect_modification_provenance function"""

    def test_detect_provenance_nonexistent_file(self):
        """Test that function handles nonexistent file gracefully"""
        from squiggy.modifications import detect_modification_provenance

        result = detect_modification_provenance(Path("/nonexistent/file.bam"))

        # Should return unknown result, not raise error
        assert result["unknown"] is True
        assert result["basecaller"] == "Unknown"

    def test_detect_provenance_returns_expected_structure(self, indexed_bam_file):
        """Test that function returns expected dict structure"""
        from squiggy.modifications import detect_modification_provenance

        result = detect_modification_provenance(indexed_bam_file)

        assert isinstance(result, dict)
        assert "basecaller" in result
        assert "version" in result
        assert "model" in result
        assert "full_info" in result
        assert "unknown" in result

        assert isinstance(result["basecaller"], str)
        assert isinstance(result["version"], str)
        assert isinstance(result["model"], str)
        assert isinstance(result["full_info"], str)
        assert isinstance(result["unknown"], bool)


class TestModificationIntegration:
    """Integration tests using real BAM file"""

    def test_extract_from_real_bam_file(self, indexed_bam_file):
        """Test extracting modifications from a real BAM file"""
        import pysam

        from squiggy.alignment import extract_alignment_from_bam
        from squiggy.modifications import extract_modifications_from_alignment

        # Open BAM and find a read with modifications
        with pysam.AlignmentFile(str(indexed_bam_file), "rb") as bam:
            for alignment in bam.fetch(until_eof=True):
                # Check if this read has modifications
                if hasattr(alignment, "modified_bases") and alignment.modified_bases:
                    read_id = alignment.query_name

                    # Extract alignment with base annotations
                    aligned_read = extract_alignment_from_bam(indexed_bam_file, read_id)

                    if aligned_read and aligned_read.bases:
                        # Extract modifications
                        mods = extract_modifications_from_alignment(
                            alignment, aligned_read.bases
                        )

                        # If modifications found, verify structure
                        if len(mods) > 0:
                            mod = mods[0]
                            assert hasattr(mod, "position")
                            assert hasattr(mod, "mod_code")
                            assert hasattr(mod, "probability")
                            assert hasattr(mod, "signal_start")
                            assert hasattr(mod, "signal_end")
                            assert 0.0 <= mod.probability <= 1.0
                            return  # Test passed

        # If no modifications found in BAM, that's okay - skip test
        pytest.skip("No reads with modifications found in BAM file")
