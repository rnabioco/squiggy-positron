"""Integration tests for modBAM loading and visualization

Tests the complete workflow from loading modBAM files to visualization,
including:
- Loading POD5 files with corresponding modBAM files
- Detecting modifications from BAM alignments
- Generating plots with modification overlays
- Modification filtering and thresholding
"""

from pathlib import Path

import pod5
import pytest
from squiggy.plotting.aggregate import plot_aggregate
from squiggy.plotting.eventalign import plot_eventalign
from squiggy.plotting.single import plot_single_read

from squiggy.alignment import extract_alignment_from_bam
from squiggy.constants import NormalizationMethod
from squiggy.modifications import (
    calculate_modification_pileup,
    detect_modification_provenance,
)
from squiggy.utils import (
    calculate_aggregate_signal,
    calculate_base_pileup,
    calculate_quality_by_position,
    extract_reads_for_reference,
)


def get_signal_from_pod5(pod5_path, read_id):
    """Helper function to extract signal from POD5 file for a specific read

    Args:
        pod5_path: Path to POD5 file
        read_id: Read ID to extract

    Returns:
        tuple: (signal, sample_rate) or (None, None) if not found
    """
    with pod5.Reader(pod5_path) as reader:
        for read in reader.reads():
            if str(read.read_id) == read_id:
                return read.signal, read.run_info.sample_rate
    return None, None


@pytest.fixture
def yeast_pod5_path():
    """Path to yeast tRNA POD5 test file"""
    path = Path(__file__).parent / "data" / "yeast_trna_reads.pod5"
    if not path.exists():
        pytest.skip(f"Test data not found: {path}")
    return path


@pytest.fixture
def yeast_bam_path():
    """Path to yeast tRNA BAM test file with modifications"""
    path = Path(__file__).parent / "data" / "yeast_trna_mappings.bam"
    if not path.exists():
        pytest.skip(f"Test data not found: {path}")
    return path


@pytest.fixture
def test_read_id():
    """A read ID known to have modifications in the yeast test data"""
    return "cf27d81e-2dba-489a-a88c-e768a51e998d"


class TestModBAMLoading:
    """Test loading and parsing of modBAM files"""

    def test_load_bam_with_modifications(self, yeast_bam_path, test_read_id):
        """Test loading a BAM file and extracting modifications"""
        aligned_read = extract_alignment_from_bam(yeast_bam_path, test_read_id)

        assert aligned_read is not None
        assert aligned_read.read_id == test_read_id
        assert aligned_read.modifications is not None
        assert len(aligned_read.modifications) > 0

        # Check that we have expected modification types
        mod_types = {mod.mod_code for mod in aligned_read.modifications}
        # Yeast tRNA data should have 5mC (m), 6mA (a), inosine (17596), pseudouridine (17802)
        assert len(mod_types) > 0

    def test_modification_probabilities(self, yeast_bam_path, test_read_id):
        """Test that modification probabilities are in valid range"""
        aligned_read = extract_alignment_from_bam(yeast_bam_path, test_read_id)

        for mod in aligned_read.modifications:
            assert 0.0 <= mod.probability <= 1.0
            assert mod.position >= 0
            assert mod.signal_start >= 0
            assert mod.signal_end > mod.signal_start

    def test_detect_provenance(self, yeast_bam_path):
        """Test detecting basecaller provenance from BAM header"""
        provenance = detect_modification_provenance(yeast_bam_path)

        assert provenance is not None
        assert "basecaller" in provenance
        assert "version" in provenance
        assert "model" in provenance
        assert "unknown" in provenance
        # Note: Some test BAM files may not have provenance info, which is OK


class TestModificationPileup:
    """Test modification aggregation and pileup statistics"""

    def test_pileup_from_aligned_reads(self, yeast_bam_path):
        """Test calculating modification pileup from multiple reads"""
        # Load multiple aligned reads
        test_reads = [
            "cf27d81e-2dba-489a-a88c-e768a51e998d",
            "9c89c5b7-95e0-402b-bf70-7ff8e42a63e5",
        ]

        aligned_reads = []
        for read_id in test_reads:
            aligned_read = extract_alignment_from_bam(yeast_bam_path, read_id)
            if aligned_read and aligned_read.modifications:
                aligned_reads.append(aligned_read)

        assert len(aligned_reads) > 0

        # Calculate pileup without threshold (continuous probabilities)
        pileup = calculate_modification_pileup(aligned_reads, tau=None)

        assert len(pileup) > 0
        for (_pos, _mod_type), stats in pileup.items():
            assert stats.coverage > 0
            assert 0.0 <= stats.mean_prob <= 1.0
            assert len(stats.probs) == stats.coverage
            # Threshold-based stats should be None
            assert stats.n_mod_tau is None
            assert stats.frequency is None

    def test_pileup_with_threshold(self, yeast_bam_path):
        """Test modification pileup with probability threshold"""
        # Load multiple aligned reads
        test_reads = [
            "cf27d81e-2dba-489a-a88c-e768a51e998d",
            "9c89c5b7-95e0-402b-bf70-7ff8e42a63e5",
        ]

        aligned_reads = []
        for read_id in test_reads:
            aligned_read = extract_alignment_from_bam(yeast_bam_path, read_id)
            if aligned_read and aligned_read.modifications:
                aligned_reads.append(aligned_read)

        # Calculate pileup with threshold
        tau = 0.5
        pileup = calculate_modification_pileup(aligned_reads, tau=tau, scope="position")

        assert len(pileup) > 0
        for (_pos, _mod_type), stats in pileup.items():
            # Threshold-based stats should be populated
            assert stats.n_mod_tau is not None
            assert stats.n_unmod_tau is not None
            assert stats.frequency is not None
            assert stats.n_mod_tau + stats.n_unmod_tau == stats.coverage
            assert 0.0 <= stats.frequency <= 1.0


class TestSingleReadVisualization:
    """Test single read plotting with modifications"""

    def test_plot_single_read_with_modifications(
        self, yeast_pod5_path, yeast_bam_path, test_read_id
    ):
        """Test generating a single read plot with modification overlays"""
        # Load signal data
        signal, sample_rate = get_signal_from_pod5(yeast_pod5_path, test_read_id)
        assert signal is not None

        # Load alignment with modifications
        aligned_read = extract_alignment_from_bam(yeast_bam_path, test_read_id)
        assert aligned_read is not None
        assert aligned_read.modifications is not None

        # Generate plot with modifications
        html, fig = plot_single_read(
            signal=signal,
            read_id=test_read_id,
            sample_rate=sample_rate,
            modifications=aligned_read.modifications,
            show_modification_overlay=True,
            modification_overlay_opacity=0.6,
        )

        assert html is not None
        assert fig is not None
        assert "bokeh" in html.lower()

    def test_plot_single_read_no_modifications(self, yeast_pod5_path, test_read_id):
        """Test generating a single read plot without modifications"""
        # Load signal data
        signal, sample_rate = get_signal_from_pod5(yeast_pod5_path, test_read_id)
        assert signal is not None

        # Generate plot without modifications
        html, fig = plot_single_read(
            signal=signal,
            read_id=test_read_id,
            sample_rate=sample_rate,
            modifications=None,
            show_modification_overlay=False,
        )

        assert html is not None
        assert fig is not None


class TestEventalignVisualization:
    """Test event-aligned plotting with modifications"""

    def test_plot_eventalign_with_modifications(
        self, yeast_pod5_path, yeast_bam_path, test_read_id
    ):
        """Test generating event-aligned plot with modification overlays"""
        # Load signal data
        signal, sample_rate = get_signal_from_pod5(yeast_pod5_path, test_read_id)
        assert signal is not None

        # Load alignment with modifications
        aligned_read = extract_alignment_from_bam(yeast_bam_path, test_read_id)
        assert aligned_read is not None

        # Prepare reads_data
        reads_data = [(test_read_id, signal, sample_rate)]

        # Generate plot with modifications
        html, fig = plot_eventalign(
            reads_data=reads_data,
            normalization=NormalizationMethod.NONE,
            aligned_reads=[aligned_read],
            show_modification_overlay=True,
            modification_overlay_opacity=0.6,
        )

        assert html is not None
        assert fig is not None
        assert "bokeh" in html.lower()

    def test_plot_eventalign_dwell_time_mode(
        self, yeast_pod5_path, yeast_bam_path, test_read_id
    ):
        """Test event-aligned plot with dwell time and modifications"""
        # Load signal data
        signal, sample_rate = get_signal_from_pod5(yeast_pod5_path, test_read_id)
        assert signal is not None

        # Load alignment with modifications
        aligned_read = extract_alignment_from_bam(yeast_bam_path, test_read_id)
        assert aligned_read is not None

        # Prepare reads_data
        reads_data = [(test_read_id, signal, sample_rate)]

        # Generate plot with dwell time mode
        html, fig = plot_eventalign(
            reads_data=reads_data,
            normalization=NormalizationMethod.NONE,
            aligned_reads=[aligned_read],
            show_dwell_time=True,
            show_modification_overlay=True,
            modification_overlay_opacity=0.6,
        )

        assert html is not None
        assert fig is not None


class TestAggregateVisualization:
    """Test aggregate plotting with modification heatmap"""

    def test_plot_aggregate_with_modifications(self, yeast_pod5_path, yeast_bam_path):
        """Test generating aggregate plot with modification heatmap track"""
        reference_name = "tRNA-Ala-AGC-1-1"

        # Extract reads for reference (includes signal and alignment data)
        reads_data_dicts = extract_reads_for_reference(
            yeast_pod5_path, yeast_bam_path, reference_name, max_reads=5
        )
        if len(reads_data_dicts) == 0:
            pytest.skip(f"No reads found for reference {reference_name}")

        # Load aligned reads with modifications
        aligned_reads = []
        for read_dict in reads_data_dicts:
            aligned_read = extract_alignment_from_bam(
                yeast_bam_path, read_dict["read_id"]
            )
            if aligned_read:
                aligned_reads.append(aligned_read)

        assert len(aligned_reads) > 0

        # Calculate modification pileup
        mod_pileup = calculate_modification_pileup(aligned_reads, tau=None)
        assert len(mod_pileup) > 0

        # Prepare reads_data for aggregate functions
        reads_data = [
            (d["read_id"], d["signal"], d["sample_rate"]) for d in reads_data_dicts
        ]

        # Calculate aggregate statistics
        agg_stats = calculate_aggregate_signal(reads_data, NormalizationMethod.NONE)
        pileup_stats = calculate_base_pileup(aligned_reads)
        quality_stats = calculate_quality_by_position(aligned_reads)

        # Generate plot with modification heatmap
        html, grid = plot_aggregate(
            aggregate_stats=agg_stats,
            pileup_stats=pileup_stats,
            quality_stats=quality_stats,
            reference_name=reference_name,
            num_reads=len(reads_data),
            modification_pileup_stats=mod_pileup,
        )

        assert html is not None
        assert grid is not None
        assert "bokeh" in html.lower()

    def test_plot_aggregate_without_modifications(
        self, yeast_pod5_path, yeast_bam_path
    ):
        """Test generating aggregate plot without modification track"""
        reference_name = "tRNA-Ala-AGC-1-1"

        # Extract reads for reference
        reads_data_dicts = extract_reads_for_reference(
            yeast_pod5_path, yeast_bam_path, reference_name, max_reads=5
        )

        # Load aligned reads
        aligned_reads = []
        for read_dict in reads_data_dicts:
            aligned_read = extract_alignment_from_bam(
                yeast_bam_path, read_dict["read_id"]
            )
            if aligned_read:
                aligned_reads.append(aligned_read)

        # Prepare reads_data
        reads_data = [
            (d["read_id"], d["signal"], d["sample_rate"]) for d in reads_data_dicts
        ]

        # Calculate statistics
        agg_stats = calculate_aggregate_signal(reads_data, NormalizationMethod.NONE)
        pileup_stats = calculate_base_pileup(aligned_reads)
        quality_stats = calculate_quality_by_position(aligned_reads)

        # Generate plot without modifications (should have 3 tracks, not 4)
        html, grid = plot_aggregate(
            aggregate_stats=agg_stats,
            pileup_stats=pileup_stats,
            quality_stats=quality_stats,
            reference_name=reference_name,
            num_reads=len(reads_data),
            modification_pileup_stats=None,  # No modifications
        )

        assert html is not None
        assert grid is not None


class TestModificationFiltering:
    """Test modification-based read filtering"""

    def test_filter_reads_by_modification_status(self, yeast_bam_path):
        """Test filtering reads based on modification presence"""
        test_reads = [
            "cf27d81e-2dba-489a-a88c-e768a51e998d",
            "9c89c5b7-95e0-402b-bf70-7ff8e42a63e5",
        ]

        # Load aligned reads
        aligned_reads = []
        for read_id in test_reads:
            aligned_read = extract_alignment_from_bam(yeast_bam_path, read_id)
            if aligned_read:
                aligned_reads.append(aligned_read)

        # Filter reads with modifications
        reads_with_mods = [read for read in aligned_reads if read.modifications]

        # Should have at least some reads with modifications
        assert len(reads_with_mods) > 0

    def test_filter_by_modification_threshold(self, yeast_bam_path):
        """Test filtering reads by modification probability threshold"""
        test_reads = [
            "cf27d81e-2dba-489a-a88c-e768a51e998d",
            "9c89c5b7-95e0-402b-bf70-7ff8e42a63e5",
        ]

        aligned_reads = []
        for read_id in test_reads:
            aligned_read = extract_alignment_from_bam(yeast_bam_path, read_id)
            if aligned_read and aligned_read.modifications:
                aligned_reads.append(aligned_read)

        # Calculate pileup with threshold
        tau = 0.5
        pileup = calculate_modification_pileup(aligned_reads, tau=tau, scope="any")

        # Check that read IDs are tracked
        for (_pos, _mod_type), stats in pileup.items():
            if stats.read_ids_modified:
                assert isinstance(stats.read_ids_modified, set)
                assert len(stats.read_ids_modified) <= stats.coverage
