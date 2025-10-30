"""Tests for plotting/eventalign.py - EVENTALIGN mode"""

import numpy as np
import pytest

from squiggy.alignment import AlignedRead, BaseAnnotation
from squiggy.constants import NormalizationMethod, Theme
from squiggy.plotting.eventalign import (
    add_base_annotations_eventalign,
    plot_eventalign,
    plot_eventalign_signals,
)


class TestPlotEventalign:
    """Tests for plot_eventalign function"""

    def test_plot_eventalign_basic(self):
        """Test basic eventalign plotting"""
        reads_data = [("read_001", np.random.randn(100), 4000)]

        aligned_reads = [
            AlignedRead(
                read_id="read_001",
                sequence="AAAAAAAAAA",
                chromosome="chr1",
                genomic_start=1000,
                genomic_end=1010,
                bases=[
                    BaseAnnotation(
                        base="A",
                        position=i,
                        signal_start=i * 10,
                        signal_end=(i + 1) * 10,
                    )
                    for i in range(10)
                ],
            )
        ]

        html, fig = plot_eventalign(reads_data, NormalizationMethod.NONE, aligned_reads)

        assert isinstance(html, str)
        assert len(html) > 0
        assert fig is not None
        assert "<!DOCTYPE html>" in html

    def test_plot_eventalign_no_aligned_reads(self):
        """Test eventalign raises error with no aligned reads"""
        reads_data = [("read_001", np.random.randn(100), 4000)]

        with pytest.raises(ValueError, match="requires aligned_reads"):
            plot_eventalign(reads_data, NormalizationMethod.NONE, None)

    def test_plot_eventalign_multiple_reads(self):
        """Test eventalign with multiple reads"""
        reads_data = [
            ("read_001", np.random.randn(100), 4000),
            ("read_002", np.random.randn(100), 4000),
        ]

        aligned_reads = [
            AlignedRead(
                read_id="read_001",
                sequence="AAAAAAAAAA",
                chromosome="chr1",
                genomic_start=1000,
                genomic_end=1010,
                bases=[
                    BaseAnnotation(
                        base="A",
                        position=i,
                        signal_start=i * 10,
                        signal_end=(i + 1) * 10,
                    )
                    for i in range(10)
                ],
            ),
            AlignedRead(
                read_id="read_002",
                sequence="CCCCCCCCCC",
                chromosome="chr1",
                genomic_start=1000,
                genomic_end=1010,
                bases=[
                    BaseAnnotation(
                        base="C",
                        position=i,
                        signal_start=i * 10,
                        signal_end=(i + 1) * 10,
                    )
                    for i in range(10)
                ],
            ),
        ]

        html, fig = plot_eventalign(
            reads_data, NormalizationMethod.ZNORM, aligned_reads
        )

        assert isinstance(html, str)
        assert fig is not None

    def test_plot_eventalign_with_dwell_time(self):
        """Test eventalign with dwell time enabled"""
        reads_data = [("read_001", np.random.randn(100), 4000)]

        aligned_reads = [
            AlignedRead(
                read_id="read_001",
                sequence="AAAAAAAAAA",
                chromosome="chr1",
                genomic_start=1000,
                genomic_end=1010,
                bases=[
                    BaseAnnotation(
                        base="A",
                        position=i,
                        signal_start=i * 10,
                        signal_end=(i + 1) * 10,
                    )
                    for i in range(10)
                ],
            )
        ]

        html, fig = plot_eventalign(
            reads_data, NormalizationMethod.NONE, aligned_reads, show_dwell_time=True
        )

        assert isinstance(html, str)
        # With dwell time, x-axis should be "Time (ms)"
        assert fig.xaxis.axis_label == "Time (ms)"

    def test_plot_eventalign_without_dwell_time(self):
        """Test eventalign without dwell time (base position mode)"""
        reads_data = [("read_001", np.random.randn(100), 4000)]

        aligned_reads = [
            AlignedRead(
                read_id="read_001",
                sequence="AAAAAAAAAA",
                chromosome="chr1",
                genomic_start=1000,
                genomic_end=1010,
                bases=[
                    BaseAnnotation(
                        base="A",
                        position=i,
                        signal_start=i * 10,
                        signal_end=(i + 1) * 10,
                    )
                    for i in range(10)
                ],
            )
        ]

        html, fig = plot_eventalign(
            reads_data, NormalizationMethod.NONE, aligned_reads, show_dwell_time=False
        )

        assert isinstance(html, str)
        # Without dwell time, x-axis should be "Base Position"
        assert fig.xaxis.axis_label == "Base Position"

    def test_plot_eventalign_with_labels(self):
        """Test eventalign with labels shown"""
        reads_data = [("read_001", np.random.randn(100), 4000)]

        aligned_reads = [
            AlignedRead(
                read_id="read_001",
                sequence="ACGTACGTAC",
                chromosome="chr1",
                genomic_start=1000,
                genomic_end=1010,
                bases=[
                    BaseAnnotation(
                        base=base,
                        position=i,
                        signal_start=i * 10,
                        signal_end=(i + 1) * 10,
                    )
                    for i, base in enumerate("ACGTACGTAC")
                ],
            )
        ]

        html, fig = plot_eventalign(
            reads_data, NormalizationMethod.NONE, aligned_reads, show_labels=True
        )

        assert isinstance(html, str)
        assert fig is not None

    def test_plot_eventalign_without_labels(self):
        """Test eventalign without labels"""
        reads_data = [("read_001", np.random.randn(100), 4000)]

        aligned_reads = [
            AlignedRead(
                read_id="read_001",
                sequence="AAAAAAAAAA",
                chromosome="chr1",
                genomic_start=1000,
                genomic_end=1010,
                bases=[
                    BaseAnnotation(
                        base="A",
                        position=i,
                        signal_start=i * 10,
                        signal_end=(i + 1) * 10,
                    )
                    for i in range(10)
                ],
            )
        ]

        html, fig = plot_eventalign(
            reads_data, NormalizationMethod.NONE, aligned_reads, show_labels=False
        )

        assert isinstance(html, str)
        assert fig is not None

    def test_plot_eventalign_with_downsample(self):
        """Test eventalign with downsampling"""
        reads_data = [("read_001", np.random.randn(1000), 4000)]

        aligned_reads = [
            AlignedRead(
                read_id="read_001",
                sequence="A" * 100,
                chromosome="chr1",
                genomic_start=1000,
                genomic_end=1100,
                bases=[
                    BaseAnnotation(
                        base="A",
                        position=i,
                        signal_start=i * 10,
                        signal_end=(i + 1) * 10,
                    )
                    for i in range(100)
                ],
            )
        ]

        html, fig = plot_eventalign(
            reads_data, NormalizationMethod.NONE, aligned_reads, downsample=5
        )

        assert isinstance(html, str)

    def test_plot_eventalign_dark_theme(self):
        """Test eventalign with dark theme"""
        reads_data = [("read_001", np.random.randn(100), 4000)]

        aligned_reads = [
            AlignedRead(
                read_id="read_001",
                sequence="AAAAAAAAAA",
                chromosome="chr1",
                genomic_start=1000,
                genomic_end=1010,
                bases=[
                    BaseAnnotation(
                        base="A",
                        position=i,
                        signal_start=i * 10,
                        signal_end=(i + 1) * 10,
                    )
                    for i in range(10)
                ],
            )
        ]

        html, fig = plot_eventalign(
            reads_data, NormalizationMethod.NONE, aligned_reads, theme=Theme.DARK
        )

        assert isinstance(html, str)
        assert fig is not None


class TestPlotEventalignSignals:
    """Tests for plot_eventalign_signals function"""

    def test_plot_eventalign_signals_basic(self):
        """Test plotting signals in eventalign mode"""
        from bokeh.plotting import figure

        p = figure()
        reads_data = [("read_001", np.random.randn(100), 4000)]

        aligned_reads = [
            AlignedRead(
                read_id="read_001",
                sequence="AAAAAAAAAA",
                chromosome="chr1",
                genomic_start=1000,
                genomic_end=1010,
                bases=[
                    BaseAnnotation(
                        base="A",
                        position=i,
                        signal_start=i * 10,
                        signal_end=(i + 1) * 10,
                    )
                    for i in range(10)
                ],
            )
        ]

        renderers = plot_eventalign_signals(
            p, reads_data, NormalizationMethod.NONE, aligned_reads
        )

        assert isinstance(renderers, list)
        assert len(renderers) > 0

    def test_plot_eventalign_signals_dwell_time(self):
        """Test signal plotting with dwell time enabled"""
        from bokeh.plotting import figure

        p = figure()
        reads_data = [("read_001", np.random.randn(100), 4000)]

        aligned_reads = [
            AlignedRead(
                read_id="read_001",
                sequence="AAAAAAAAAA",
                chromosome="chr1",
                genomic_start=1000,
                genomic_end=1010,
                bases=[
                    BaseAnnotation(
                        base="A",
                        position=i,
                        signal_start=i * 10,
                        signal_end=(i + 1) * 10,
                    )
                    for i in range(10)
                ],
            )
        ]

        renderers = plot_eventalign_signals(
            p, reads_data, NormalizationMethod.NONE, aligned_reads, show_dwell_time=True
        )

        assert isinstance(renderers, list)


class TestAddBaseAnnotationsEventalign:
    """Tests for add_base_annotations_eventalign function"""

    def test_add_annotations_basic(self):
        """Test adding base annotations to eventalign plot"""
        from bokeh.plotting import figure

        p = figure()
        reads_data = [("read_001", np.random.randn(100), 4000)]

        aligned_reads = [
            AlignedRead(
                read_id="read_001",
                sequence="AAAAAAAAAA",
                chromosome="chr1",
                genomic_start=1000,
                genomic_end=1010,
                bases=[
                    BaseAnnotation(
                        base="A",
                        position=i,
                        signal_start=i * 10,
                        signal_end=(i + 1) * 10,
                    )
                    for i in range(10)
                ],
            )
        ]

        add_base_annotations_eventalign(
            p,
            reads_data,
            NormalizationMethod.NONE,
            aligned_reads,
            show_dwell_time=False,
            show_labels=True,
        )

        # Should not raise error
        assert True

    def test_add_annotations_no_reads(self):
        """Test adding annotations with no reads"""
        from bokeh.plotting import figure

        p = figure()

        result = add_base_annotations_eventalign(
            p,
            [],
            NormalizationMethod.NONE,
            [],
            show_dwell_time=False,
            show_labels=False,
        )

        # Should return None when no reads
        assert result is None

    def test_add_annotations_with_dwell_time(self):
        """Test adding annotations in dwell time mode"""
        from bokeh.plotting import figure

        p = figure()
        reads_data = [("read_001", np.random.randn(100), 4000)]

        aligned_reads = [
            AlignedRead(
                read_id="read_001",
                sequence="AAAAAAAAAA",
                chromosome="chr1",
                genomic_start=1000,
                genomic_end=1010,
                bases=[
                    BaseAnnotation(
                        base="A",
                        position=i,
                        signal_start=i * 10,
                        signal_end=(i + 1) * 10,
                    )
                    for i in range(10)
                ],
            )
        ]

        add_base_annotations_eventalign(
            p,
            reads_data,
            NormalizationMethod.NONE,
            aligned_reads,
            show_dwell_time=True,
            show_labels=False,
        )

        # Should not raise error
        assert True
