"""Tests for base modification (modBAM) parsing and analysis"""

from pathlib import Path

import numpy as np
import pytest

from squiggy.alignment import AlignedRead, BaseAnnotation, extract_alignment_from_bam
from squiggy.modifications import (
    ModificationAnnotation,
    ModPositionStats,
    calculate_modification_pileup,
    detect_modification_provenance,
)

# Skip UI tests if Qt is not available
try:
    from PySide6.QtWidgets import QApplication

    from squiggy.ui import ModificationsPanel

    HAVE_QT = True
except ImportError:
    HAVE_QT = False


class TestModificationAnnotation:
    """Test ModificationAnnotation dataclass"""

    def test_modification_annotation_creation(self):
        """Test creating a modification annotation"""
        mod = ModificationAnnotation(
            position=10,
            genomic_pos=1000,
            mod_code="m",
            canonical_base="C",
            probability=0.85,
            signal_start=500,
            signal_end=550,
        )

        assert mod.position == 10
        assert mod.genomic_pos == 1000
        assert mod.mod_code == "m"
        assert mod.canonical_base == "C"
        assert mod.probability == 0.85
        assert mod.signal_start == 500
        assert mod.signal_end == 550


class TestModificationPileup:
    """Test modification pileup calculation"""

    def create_mock_read(
        self, read_id: str, mods: list[tuple[int, str, str, float]]
    ) -> AlignedRead:
        """Create a mock AlignedRead with modifications

        Args:
            read_id: Read identifier
            mods: List of (genomic_pos, mod_code, canonical, probability) tuples

        Returns:
            AlignedRead with modifications
        """
        # Create base annotations (mock)
        bases = [
            BaseAnnotation(
                base="A",
                position=i,
                signal_start=i * 10,
                signal_end=(i + 1) * 10,
                genomic_pos=i + 1000,
            )
            for i in range(50)
        ]

        # Create modifications
        modifications = [
            ModificationAnnotation(
                position=pos - 1000,  # Convert genomic to read position
                genomic_pos=pos,
                mod_code=mod_code,
                canonical_base=canonical,
                probability=prob,
                signal_start=(pos - 1000) * 10,
                signal_end=((pos - 1000) + 1) * 10,
            )
            for pos, mod_code, canonical, prob in mods
        ]

        return AlignedRead(
            read_id=read_id,
            sequence="A" * 50,
            bases=bases,
            modifications=modifications,
            chromosome="chr1",
            genomic_start=1000,
            genomic_end=1050,
        )

    def test_pileup_no_threshold(self):
        """Test pileup calculation without threshold (continuous mode)"""
        # Create mock reads with modifications at same position
        read1 = self.create_mock_read("read1", [(1010, "m", "C", 0.8)])
        read2 = self.create_mock_read("read2", [(1010, "m", "C", 0.9)])
        read3 = self.create_mock_read("read3", [(1010, "m", "C", 0.6)])

        # Calculate pileup without threshold
        pileup = calculate_modification_pileup([read1, read2, read3], tau=None)

        # Check results
        assert (1010, "m") in pileup
        stats = pileup[(1010, "m")]

        assert stats.ref_pos == 1010
        assert stats.mod_type == "m"
        assert stats.canonical_base == "C"
        assert stats.coverage == 3
        assert len(stats.probs) == 3
        assert 0.8 in stats.probs
        assert 0.9 in stats.probs
        assert 0.6 in stats.probs
        assert np.isclose(stats.mean_prob, (0.8 + 0.9 + 0.6) / 3)

        # Threshold-based stats should be None
        assert stats.n_mod_tau is None
        assert stats.n_unmod_tau is None
        assert stats.frequency is None
        assert stats.mean_conf_modified is None

    def test_pileup_with_threshold_position_scope(self):
        """Test pileup calculation with threshold (per-position classification)"""
        # Create mock reads with modifications
        read1 = self.create_mock_read("read1", [(1010, "m", "C", 0.8)])
        read2 = self.create_mock_read("read2", [(1010, "m", "C", 0.9)])
        read3 = self.create_mock_read("read3", [(1010, "m", "C", 0.3)])
        read4 = self.create_mock_read("read4", [(1010, "m", "C", 0.2)])

        # Calculate pileup with threshold = 0.5, position scope
        pileup = calculate_modification_pileup(
            [read1, read2, read3, read4], tau=0.5, scope="position"
        )

        # Check results
        stats = pileup[(1010, "m")]

        assert stats.coverage == 4
        assert stats.n_mod_tau == 2  # 0.8 and 0.9 >= 0.5
        assert stats.n_unmod_tau == 2  # 0.3 and 0.2 < 0.5
        assert np.isclose(stats.frequency, 0.5)  # 2/4
        assert np.isclose(
            stats.mean_conf_modified, (0.8 + 0.9) / 2
        )  # Mean of modified only

    def test_pileup_with_threshold_any_scope(self):
        """Test pileup calculation with 'any' scope (read-level classification)"""
        # Create mock reads
        # read1 has high prob at 1010
        # read2 has low prob at 1010 but high prob at 1020
        # read3 has low prob everywhere
        read1 = self.create_mock_read("read1", [(1010, "m", "C", 0.9)])
        read2 = self.create_mock_read(
            "read2", [(1010, "m", "C", 0.2), (1020, "m", "C", 0.8)]
        )
        read3 = self.create_mock_read("read3", [(1010, "m", "C", 0.3)])

        # Calculate pileup with threshold = 0.5, any scope
        pileup = calculate_modification_pileup(
            [read1, read2, read3], tau=0.5, scope="any"
        )

        # At position 1010, all 3 reads have coverage
        # But read1 and read2 are classified as "modified" (have >=0.5 somewhere)
        # Only read3 is "unmodified"
        stats = pileup[(1010, "m")]

        assert stats.coverage == 3
        assert stats.n_mod_tau == 2  # read1 and read2
        assert stats.n_unmod_tau == 1  # read3
        assert np.isclose(stats.frequency, 2 / 3)

    def test_pileup_multiple_mod_types(self):
        """Test pileup with multiple modification types"""
        # Create reads with different mod types
        read1 = self.create_mock_read("read1", [(1010, "m", "C", 0.8)])
        read2 = self.create_mock_read("read2", [(1010, "h", "C", 0.7)])
        read3 = self.create_mock_read("read3", [(1015, "a", "A", 0.9)])

        pileup = calculate_modification_pileup([read1, read2, read3])

        # Should have 3 distinct pileup entries
        assert len(pileup) == 3
        assert (1010, "m") in pileup
        assert (1010, "h") in pileup
        assert (1015, "a") in pileup

        # Each should have coverage of 1
        assert pileup[(1010, "m")].coverage == 1
        assert pileup[(1010, "h")].coverage == 1
        assert pileup[(1015, "a")].coverage == 1

    def test_pileup_empty_reads(self):
        """Test pileup with reads that have no modifications"""
        read1 = self.create_mock_read("read1", [])
        read2 = self.create_mock_read("read2", [])

        pileup = calculate_modification_pileup([read1, read2])

        assert len(pileup) == 0

    def test_pileup_unmapped_modifications_ignored(self):
        """Test that modifications without genomic positions are ignored"""
        # Create a read with an unmapped modification
        read = AlignedRead(
            read_id="read1",
            sequence="ACGT",
            bases=[],
            modifications=[
                ModificationAnnotation(
                    position=0,
                    genomic_pos=None,  # Unmapped
                    mod_code="m",
                    canonical_base="C",
                    probability=0.8,
                    signal_start=0,
                    signal_end=10,
                )
            ],
        )

        pileup = calculate_modification_pileup([read])

        # Should be empty because unmapped mods are ignored
        assert len(pileup) == 0


class TestModificationProbabilityEncoding:
    """Test probability encoding/decoding"""

    def test_ml_tag_probability_conversion(self):
        """Test ML tag quality to probability conversion

        ML tags encode probabilities as uint8 (0-255)
        Probability = (quality + 0.5) / 256
        """
        # Quality 0 -> ~0.002 probability
        qual_0 = 0
        prob_0 = qual_0 / 256.0
        assert np.isclose(prob_0, 0.0, atol=0.01)

        # Quality 128 -> ~0.5 probability
        qual_128 = 128
        prob_128 = qual_128 / 256.0
        assert np.isclose(prob_128, 0.5, atol=0.01)

        # Quality 255 -> ~0.996 probability (near 1.0)
        qual_255 = 255
        prob_255 = qual_255 / 256.0
        assert np.isclose(prob_255, 1.0, atol=0.01)

        # Verify the formula matches pysam's encoding
        # pysam encodes as (256 * probability)
        # We decode as quality / 256
        test_prob = 0.75
        encoded = int(256 * test_prob)
        decoded = encoded / 256.0
        assert np.isclose(decoded, test_prob, atol=0.01)


class TestModPositionStats:
    """Test ModPositionStats dataclass"""

    def test_mod_position_stats_creation(self):
        """Test creating a ModPositionStats object"""
        stats = ModPositionStats(
            ref_pos=1000,
            mod_type="m",
            canonical_base="C",
            coverage=10,
            probs=[0.8, 0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
            mean_prob=0.5,
        )

        assert stats.ref_pos == 1000
        assert stats.mod_type == "m"
        assert stats.canonical_base == "C"
        assert stats.coverage == 10
        assert len(stats.probs) == 10
        assert stats.mean_prob == 0.5
        assert stats.n_mod_tau is None  # Not set without threshold

    def test_mod_position_stats_with_threshold(self):
        """Test ModPositionStats with threshold-based statistics"""
        stats = ModPositionStats(
            ref_pos=1000,
            mod_type="m",
            canonical_base="C",
            coverage=10,
            probs=[0.8, 0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
            mean_prob=0.5,
            n_mod_tau=5,
            n_unmod_tau=5,
            frequency=0.5,
            mean_conf_modified=0.7,
        )

        assert stats.n_mod_tau == 5
        assert stats.n_unmod_tau == 5
        assert stats.frequency == 0.5
        assert stats.mean_conf_modified == 0.7


class TestRealModBAMData:
    """Test with real modBAM data from yeast tRNA dataset"""

    @pytest.fixture
    def yeast_bam_path(self):
        """Path to yeast tRNA BAM file"""
        return Path("tests/data/yeast_trna_mappings.bam")

    def test_parse_real_modbam(self, yeast_bam_path):
        """Test parsing modifications from real yeast tRNA BAM file"""
        if not yeast_bam_path.exists():
            pytest.skip("Test data not available")

        # Extract first read with modifications
        aligned_read = extract_alignment_from_bam(
            yeast_bam_path, "cf27d81e-2dba-489a-a88c-e768a51e998d"
        )

        assert aligned_read is not None
        assert aligned_read.modifications is not None
        assert len(aligned_read.modifications) > 0

        # Check that we have expected modification types
        # Both string codes and ChEBI numeric codes
        mod_types = set(mod.mod_code for mod in aligned_read.modifications)
        assert "m" in mod_types  # 5mC (string code)
        assert "a" in mod_types  # 6mA (string code)
        assert 17596 in mod_types  # inosine (ChEBI code)
        assert 17802 in mod_types  # pseudouridine (ChEBI code)

        # Verify probabilities are in valid range
        for mod in aligned_read.modifications:
            assert 0.0 <= mod.probability <= 1.0
            assert mod.signal_start >= 0
            assert mod.signal_end > mod.signal_start
            assert mod.canonical_base in ["A", "C", "G", "T", "U"]

    def test_detect_provenance_real_bam(self, yeast_bam_path):
        """Test provenance detection on real BAM file"""
        if not yeast_bam_path.exists():
            pytest.skip("Test data not available")

        provenance = detect_modification_provenance(yeast_bam_path)

        # Check that we got some provenance info
        assert provenance is not None
        assert "basecaller" in provenance
        assert "version" in provenance
        assert "model" in provenance

        # Print for debugging (helpful to see what basecaller was used)
        print(f"\nProvenance: {provenance}")

    def test_modification_pileup_real_data(self, yeast_bam_path):
        """Test modification pileup calculation with real data"""
        if not yeast_bam_path.exists():
            pytest.skip("Test data not available")

        # Extract several reads from the same reference
        import pysam

        reads = []
        with pysam.AlignmentFile(str(yeast_bam_path), "rb", check_sq=False) as bam:
            for i, alignment in enumerate(bam):
                if i >= 5:  # Get first 5 reads
                    break
                read_id = alignment.query_name
                aligned_read = extract_alignment_from_bam(yeast_bam_path, read_id)
                if aligned_read and aligned_read.modifications:
                    reads.append(aligned_read)

        assert len(reads) > 0, "Should have found reads with modifications"

        # Calculate pileup without threshold
        pileup = calculate_modification_pileup(reads, tau=None)

        assert len(pileup) > 0, "Should have modification positions"

        # Check some basic properties
        for (ref_pos, mod_type), stats in pileup.items():
            assert stats.coverage > 0
            assert len(stats.probs) == stats.coverage
            assert 0.0 <= stats.mean_prob <= 1.0
            assert stats.n_mod_tau is None  # No threshold

        # Calculate pileup with threshold
        pileup_thresh = calculate_modification_pileup(reads, tau=0.5, scope="position")

        for (ref_pos, mod_type), stats in pileup_thresh.items():
            assert stats.n_mod_tau is not None
            assert stats.n_unmod_tau is not None
            assert stats.n_mod_tau + stats.n_unmod_tau == stats.coverage
            assert 0.0 <= stats.frequency <= 1.0


@pytest.mark.skipif(not HAVE_QT, reason="Qt not available")
class TestModificationsPanel:
    """Test ModificationsPanel UI component"""

    @pytest.fixture
    def qapp(self):
        """Create QApplication instance for testing"""
        # Check if QApplication already exists
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        yield app

    def test_panel_creation(self, qapp):
        """Test that ModificationsPanel can be created"""
        panel = ModificationsPanel()
        assert panel is not None
        assert not panel.isVisible()  # Initially hidden

    def test_color_legend_display(self, qapp):
        """Test that color legend displays detected modifications"""
        panel = ModificationsPanel()
        panel.show()  # Show panel so isVisible() works correctly

        # Set some detected modifications
        mods = {
            ("C", "m"),  # 5mC
            ("A", "a"),  # 6mA
            ("A", 17596),  # inosine (ChEBI)
            ("T", 17802),  # pseudouridine (ChEBI)
        }
        panel.set_detected_modifications(mods)

        # Color legend should be visible
        assert panel.color_legend_widget.isVisible()

        # Should have 4 modifications in the legend (4 rows * 2 columns = 8 widgets)
        # Each mod has a color box + label = 2 widgets
        legend_widget_count = panel.color_legend_layout.count()
        assert legend_widget_count == 8  # 4 mods × 2 widgets each

    def test_color_legend_empty(self, qapp):
        """Test that color legend is hidden when no mods detected"""
        panel = ModificationsPanel()
        panel.show()  # Show panel so isVisible() works correctly

        # Set empty modifications
        panel.set_detected_modifications(set())

        # Color legend should be hidden
        assert not panel.color_legend_widget.isVisible()

    def test_panel_signals(self, qapp):
        """Test that panel emits signals correctly"""
        panel = ModificationsPanel()

        # Track signal emissions
        overlay_toggled = []
        threshold_changed = []

        panel.modification_overlay_toggled.connect(lambda x: overlay_toggled.append(x))
        panel.threshold_changed.connect(lambda x: threshold_changed.append(x))

        # Toggle overlay
        panel.show_overlay_checkbox.setChecked(False)
        assert len(overlay_toggled) == 1
        assert overlay_toggled[0] is False

        # Change threshold value (threshold is now always enabled)
        panel.threshold_slider.setValue(80)  # 0.80
        assert len(threshold_changed) >= 1
        assert threshold_changed[-1] == 0.80
