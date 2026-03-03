"""
Tests for ReferenceOverlayPlotStrategy
"""

import numpy as np
import pytest
from bokeh.models.plots import Plot

from squiggy.constants import NormalizationMethod, Theme
from squiggy.plot_strategies.reference_overlay import ReferenceOverlayPlotStrategy

# =============================================================================
# Mock Classes
# =============================================================================


class MockBaseAnnotation:
    """Mock base annotation with genomic position and signal boundaries"""

    def __init__(
        self,
        base: str,
        signal_start: int,
        signal_end: int,
        genomic_pos: int | None = None,
    ):
        self.base = base
        self.signal_start = signal_start
        self.signal_end = signal_end
        self.genomic_pos = genomic_pos


class MockAlignedRead:
    """Mock aligned read with bases and chromosome"""

    def __init__(
        self,
        read_id: str,
        bases: list,
        chromosome: str | None = "chr1",
    ):
        self.read_id = read_id
        self.bases = bases
        self.chromosome = chromosome


# =============================================================================
# Helper to build consistent test data
# =============================================================================


def make_bases(
    sequence: str,
    start_pos: int = 0,
    samples_per_base: int = 100,
    genomic_start: int = 1000,
):
    """Build MockBaseAnnotation list with even signal distribution."""
    bases = []
    for i, base_char in enumerate(sequence):
        sig_start = start_pos + i * samples_per_base
        sig_end = sig_start + samples_per_base
        bases.append(
            MockBaseAnnotation(
                base=base_char,
                signal_start=sig_start,
                signal_end=sig_end,
                genomic_pos=genomic_start + i,
            )
        )
    return bases


def make_data(
    num_reads: int = 1,
    sequence: str = "ACGT",
    samples_per_base: int = 100,
    genomic_start: int = 1000,
    chromosome: str = "chr1",
):
    """Build complete data dict for testing."""
    reads = []
    aligned_reads = []
    total_samples = len(sequence) * samples_per_base

    for idx in range(num_reads):
        read_id = f"read_{idx:03d}"
        signal = np.random.randn(total_samples).astype(np.float32)
        bases = make_bases(sequence, 0, samples_per_base, genomic_start)
        reads.append((read_id, signal, 4000))
        aligned_reads.append(MockAlignedRead(read_id, bases, chromosome))

    return {"reads": reads, "aligned_reads": aligned_reads}


# =============================================================================
# Test: Initialization
# =============================================================================


class TestInitialization:
    def test_init_light_theme(self):
        strategy = ReferenceOverlayPlotStrategy(Theme.LIGHT)
        assert strategy.theme == Theme.LIGHT
        assert strategy.theme_manager is not None

    def test_init_dark_theme(self):
        strategy = ReferenceOverlayPlotStrategy(Theme.DARK)
        assert strategy.theme == Theme.DARK


# =============================================================================
# Test: Data Validation
# =============================================================================


class TestDataValidation:
    def test_valid_data(self):
        strategy = ReferenceOverlayPlotStrategy(Theme.LIGHT)
        data = make_data(num_reads=2)
        strategy.validate_data(data)  # should not raise

    def test_missing_reads(self):
        strategy = ReferenceOverlayPlotStrategy(Theme.LIGHT)
        data = {"aligned_reads": []}
        with pytest.raises(ValueError, match="Missing required data.*reads"):
            strategy.validate_data(data)

    def test_missing_aligned_reads(self):
        strategy = ReferenceOverlayPlotStrategy(Theme.LIGHT)
        data = {"reads": [("r", np.array([1.0]), 4000)]}
        with pytest.raises(ValueError, match="Missing required data.*aligned_reads"):
            strategy.validate_data(data)

    def test_empty_reads(self):
        strategy = ReferenceOverlayPlotStrategy(Theme.LIGHT)
        data = {"reads": [], "aligned_reads": []}
        with pytest.raises(ValueError, match="reads must be a non-empty list"):
            strategy.validate_data(data)

    def test_mismatched_lengths(self):
        strategy = ReferenceOverlayPlotStrategy(Theme.LIGHT)
        data = make_data(num_reads=1)
        data["reads"].append(("extra", np.array([1.0]), 4000))
        with pytest.raises(ValueError, match="must have same length"):
            strategy.validate_data(data)

    def test_missing_bases_attribute(self):
        strategy = ReferenceOverlayPlotStrategy(Theme.LIGHT)

        class NoBases:
            chromosome = "chr1"

        data = {
            "reads": [("r", np.array([1.0]), 4000)],
            "aligned_reads": [NoBases()],
        }
        with pytest.raises(ValueError, match="must have 'bases' attribute"):
            strategy.validate_data(data)

    def test_missing_chromosome_attribute(self):
        strategy = ReferenceOverlayPlotStrategy(Theme.LIGHT)

        class NoChrom:
            bases = []

        data = {
            "reads": [("r", np.array([1.0]), 4000)],
            "aligned_reads": [NoChrom()],
        }
        with pytest.raises(ValueError, match="must have 'chromosome' attribute"):
            strategy.validate_data(data)

    def test_chromosome_mismatch(self):
        strategy = ReferenceOverlayPlotStrategy(Theme.LIGHT)
        data = make_data(num_reads=2)
        data["aligned_reads"][0].chromosome = "chr1"
        data["aligned_reads"][1].chromosome = "chr2"
        with pytest.raises(ValueError, match="same chromosome"):
            strategy.validate_data(data)


# =============================================================================
# Test: Single Read
# =============================================================================


class TestSingleRead:
    def test_returns_html_and_figure(self):
        strategy = ReferenceOverlayPlotStrategy(Theme.LIGHT)
        data = make_data(num_reads=1)
        html, fig = strategy.create_plot(data, {})

        assert isinstance(html, str)
        assert len(html) > 0
        assert "<html" in html
        assert isinstance(fig, Plot)

    def test_normalization_in_title(self):
        strategy = ReferenceOverlayPlotStrategy(Theme.LIGHT)
        data = make_data(num_reads=1)
        _, fig = strategy.create_plot(
            data, {"normalization": NormalizationMethod.ZNORM}
        )
        assert "znorm" in fig.title.text.lower()

    def test_x_label_is_genomic(self):
        strategy = ReferenceOverlayPlotStrategy(Theme.LIGHT)
        data = make_data(num_reads=1)
        _, fig = strategy.create_plot(data, {})
        assert "Genomic" in fig.xaxis.axis_label


# =============================================================================
# Test: Multi-Read Overlay
# =============================================================================


class TestMultiReadOverlay:
    def test_multi_read_creates_plot(self):
        strategy = ReferenceOverlayPlotStrategy(Theme.LIGHT)
        data = make_data(num_reads=5)
        html, fig = strategy.create_plot(data, {})

        assert isinstance(html, str)
        assert "5 reads" in fig.title.text

    def test_legend_present(self):
        strategy = ReferenceOverlayPlotStrategy(Theme.LIGHT)
        data = make_data(num_reads=3)
        _, fig = strategy.create_plot(data, {})

        assert fig.legend is not None
        assert fig.legend.click_policy == "hide"

    def test_signal_points(self):
        strategy = ReferenceOverlayPlotStrategy(Theme.LIGHT)
        data = make_data(num_reads=2)
        html, _ = strategy.create_plot(data, {"show_signal_points": True})
        assert isinstance(html, str)


# =============================================================================
# Test: Partially Overlapping Reads
# =============================================================================


class TestPartiallyOverlappingReads:
    def test_different_spans(self):
        """Two reads covering different (overlapping) genomic regions."""
        strategy = ReferenceOverlayPlotStrategy(Theme.LIGHT)

        # Read 1: genomic 1000-1003
        bases1 = make_bases("ACGT", 0, 100, genomic_start=1000)
        # Read 2: genomic 1002-1005
        bases2 = make_bases("GTAC", 0, 100, genomic_start=1002)

        data = {
            "reads": [
                ("r1", np.random.randn(400).astype(np.float32), 4000),
                ("r2", np.random.randn(400).astype(np.float32), 4000),
            ],
            "aligned_reads": [
                MockAlignedRead("r1", bases1),
                MockAlignedRead("r2", bases2),
            ],
        }

        html, fig = strategy.create_plot(data, {})
        assert isinstance(html, str)
        assert isinstance(fig, Plot)


# =============================================================================
# Test: Insertions and Deletions
# =============================================================================


class TestInsertionsAndDeletions:
    def test_insertions_skipped(self):
        """Bases with genomic_pos=None (insertions) should be skipped."""
        strategy = ReferenceOverlayPlotStrategy(Theme.LIGHT)

        bases = [
            MockBaseAnnotation("A", 0, 100, genomic_pos=1000),
            MockBaseAnnotation("C", 100, 200, genomic_pos=None),  # insertion
            MockBaseAnnotation("G", 200, 300, genomic_pos=1001),
            MockBaseAnnotation("T", 300, 400, genomic_pos=1002),
        ]

        data = {
            "reads": [("r1", np.random.randn(400).astype(np.float32), 4000)],
            "aligned_reads": [MockAlignedRead("r1", bases)],
        }

        html, fig = strategy.create_plot(data, {})
        assert isinstance(html, str)

    def test_deletions_create_nan_gaps(self):
        """Gaps > 1 in genomic_pos should insert NaN to break the line."""
        strategy = ReferenceOverlayPlotStrategy(Theme.LIGHT)

        bases = [
            MockBaseAnnotation("A", 0, 100, genomic_pos=1000),
            # deletion at 1001
            MockBaseAnnotation("G", 100, 200, genomic_pos=1002),
            MockBaseAnnotation("T", 200, 300, genomic_pos=1003),
        ]

        # Directly test the coordinate builder
        signal = np.random.randn(300).astype(np.float32)
        x, y = strategy._build_genomic_signal_coordinates(signal, bases, downsample=1)

        # Should contain NaN at the deletion boundary
        assert np.any(np.isnan(x))
        assert np.any(np.isnan(y))


# =============================================================================
# Test: Consensus Base Map
# =============================================================================


class TestConsensusBaseMap:
    def test_majority_vote(self):
        """Consensus should pick the most common base at each position."""
        strategy = ReferenceOverlayPlotStrategy(Theme.LIGHT)

        bases1 = [MockBaseAnnotation("A", 0, 100, genomic_pos=1000)]
        bases2 = [MockBaseAnnotation("A", 0, 100, genomic_pos=1000)]
        bases3 = [MockBaseAnnotation("C", 0, 100, genomic_pos=1000)]

        aligned_reads = [
            MockAlignedRead("r1", bases1),
            MockAlignedRead("r2", bases2),
            MockAlignedRead("r3", bases3),
        ]

        consensus = strategy._build_consensus_base_map(aligned_reads)
        assert consensus[1000] == "A"  # 2 A's vs 1 C

    def test_multiple_positions(self):
        strategy = ReferenceOverlayPlotStrategy(Theme.LIGHT)

        bases = make_bases("ACGT", 0, 100, 500)
        aligned_reads = [MockAlignedRead("r1", bases)]

        consensus = strategy._build_consensus_base_map(aligned_reads)
        assert consensus[500] == "A"
        assert consensus[501] == "C"
        assert consensus[502] == "G"
        assert consensus[503] == "T"


# =============================================================================
# Test: Alpha Blending
# =============================================================================


class TestAlphaBlending:
    def test_single_read_alpha(self):
        """Single read should have high alpha (0.8)."""
        strategy = ReferenceOverlayPlotStrategy(Theme.LIGHT)
        data = make_data(num_reads=1)
        _, fig = strategy.create_plot(data, {})
        # Verify plot was created (alpha is internal to renderers)
        assert isinstance(fig, Plot)

    def test_many_reads_still_creates_plot(self):
        """Many reads should still produce a valid plot with reduced alpha."""
        strategy = ReferenceOverlayPlotStrategy(Theme.LIGHT)
        data = make_data(num_reads=25, sequence="AC", samples_per_base=20)
        html, fig = strategy.create_plot(data, {})
        assert isinstance(html, str)
        assert "25 reads" in fig.title.text


# =============================================================================
# Test: Read Colors
# =============================================================================


class TestReadColors:
    def test_custom_read_colors(self):
        strategy = ReferenceOverlayPlotStrategy(Theme.LIGHT)
        data = make_data(num_reads=2)
        options = {
            "read_colors": {
                "read_000": "#FF0000",
                "read_001": "#0000FF",
            }
        }
        html, fig = strategy.create_plot(data, options)
        assert isinstance(html, str)

    def test_partial_read_colors_falls_back(self):
        """Reads not in read_colors dict should use default palette."""
        strategy = ReferenceOverlayPlotStrategy(Theme.LIGHT)
        data = make_data(num_reads=2)
        options = {"read_colors": {"read_000": "#FF0000"}}
        html, fig = strategy.create_plot(data, options)
        assert isinstance(html, str)


# =============================================================================
# Test: Reference Track
# =============================================================================


class TestReferenceTrack:
    def test_with_reference_sequence(self):
        """Reference sequence should produce a column layout."""
        strategy = ReferenceOverlayPlotStrategy(Theme.LIGHT)
        data = make_data(num_reads=2, genomic_start=0)
        # Provide a long enough reference sequence
        data["reference_sequence"] = "ACGTACGTACGTACGT" * 100
        html, layout = strategy.create_plot(data, {})
        assert isinstance(html, str)
        # Layout should be a column (not just a figure)
        assert hasattr(layout, "children")

    def test_without_reference_sequence(self):
        """No reference sequence should return a plain figure."""
        strategy = ReferenceOverlayPlotStrategy(Theme.LIGHT)
        data = make_data(num_reads=1)
        _, fig = strategy.create_plot(data, {})
        assert isinstance(fig, Plot)


# =============================================================================
# Test: Downsample
# =============================================================================


class TestDownsample:
    def test_downsample_produces_fewer_points(self):
        """Higher downsample should produce fewer coordinate points."""
        strategy = ReferenceOverlayPlotStrategy(Theme.LIGHT)

        bases = make_bases("ACGT", 0, 100, 1000)
        signal = np.random.randn(400).astype(np.float32)

        x1, y1 = strategy._build_genomic_signal_coordinates(signal, bases, downsample=1)
        x5, y5 = strategy._build_genomic_signal_coordinates(signal, bases, downsample=5)

        # Downsample=5 should produce roughly 1/5 the points
        valid_1 = x1[~np.isnan(x1)]
        valid_5 = x5[~np.isnan(x5)]
        assert len(valid_5) < len(valid_1)

    def test_downsample_in_title(self):
        strategy = ReferenceOverlayPlotStrategy(Theme.LIGHT)
        data = make_data(num_reads=1)
        _, fig = strategy.create_plot(data, {"downsample": 10})
        assert "10x" in fig.title.text


# =============================================================================
# Test: Integration
# =============================================================================


class TestIntegration:
    def test_complete_workflow(self):
        strategy = ReferenceOverlayPlotStrategy(Theme.DARK)
        data = make_data(num_reads=3, sequence="ACGTACGT")
        options = {
            "normalization": NormalizationMethod.ZNORM,
            "show_labels": True,
            "show_signal_points": True,
            "downsample": 2,
        }
        html, fig = strategy.create_plot(data, options)
        assert isinstance(html, str)
        assert "3 reads" in fig.title.text

    def test_reuse_strategy(self):
        strategy = ReferenceOverlayPlotStrategy(Theme.LIGHT)

        data1 = make_data(num_reads=1, sequence="AC")
        data2 = make_data(num_reads=2, sequence="GT")

        html1, fig1 = strategy.create_plot(data1, {})
        html2, fig2 = strategy.create_plot(data2, {})

        assert isinstance(html1, str)
        assert isinstance(html2, str)
        assert "1 reads" in fig1.title.text
        assert "2 reads" in fig2.title.text
