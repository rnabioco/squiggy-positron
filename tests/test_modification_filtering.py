"""Tests for modification filtering and visualization features"""

import numpy as np
from squiggy.plotting.eventalign import (
    create_modification_track_eventalign,
    plot_eventalign,
)
from squiggy.plotting.single import create_modification_track_single, plot_single_read

from squiggy.alignment import AlignedRead, BaseAnnotation
from squiggy.constants import NormalizationMethod, Theme
from squiggy.modifications import ModificationAnnotation


class TestModificationTrackSingle:
    """Tests for create_modification_track_single function"""

    def test_create_mod_track_basic(self):
        """Test basic modification track creation"""
        time_ms = np.arange(100) * 0.25
        modifications = [
            ModificationAnnotation(
                position=5,
                genomic_pos=1005,
                mod_code="m",
                canonical_base="C",
                probability=0.8,
                signal_start=20,
                signal_end=30,
            ),
            ModificationAnnotation(
                position=15,
                genomic_pos=1015,
                mod_code="a",
                canonical_base="A",
                probability=0.7,
                signal_start=60,
                signal_end=70,
            ),
        ]
        sequence = "A" * 10 + "C" * 10 + "A" * 10

        p_mod = create_modification_track_single(
            modifications=modifications,
            time_ms=time_ms,
            theme=Theme.LIGHT,
            show_overlay=True,
            sequence=sequence,
        )

        assert p_mod is not None
        assert p_mod.height == 80
        assert p_mod.yaxis.visible is False
        assert p_mod.toolbar_location is None

    def test_create_mod_track_no_overlay(self):
        """Test that no track is created when show_overlay=False"""
        time_ms = np.arange(100) * 0.25
        modifications = [
            ModificationAnnotation(
                position=5,
                genomic_pos=1005,
                mod_code="m",
                canonical_base="C",
                probability=0.8,
                signal_start=20,
                signal_end=30,
            )
        ]

        p_mod = create_modification_track_single(
            modifications=modifications,
            time_ms=time_ms,
            theme=Theme.LIGHT,
            show_overlay=False,
        )

        assert p_mod is None

    def test_create_mod_track_no_modifications(self):
        """Test that no track is created when no modifications"""
        time_ms = np.arange(100) * 0.25

        p_mod = create_modification_track_single(
            modifications=None,
            time_ms=time_ms,
            theme=Theme.LIGHT,
            show_overlay=True,
        )

        assert p_mod is None

    def test_create_mod_track_type_filter(self):
        """Test modification type filtering"""
        time_ms = np.arange(100) * 0.25
        # Create sequence where positions match canonical bases
        sequence = "A" * 10 + "C" * 10 + "A" * 10
        modifications = [
            ModificationAnnotation(
                position=15,  # Position 15 is in C region (indices 10-19)
                genomic_pos=1015,
                mod_code="m",
                canonical_base="C",  # Matches sequence[15] = 'C'
                probability=0.8,
                signal_start=45,
                signal_end=48,
            ),
            ModificationAnnotation(
                position=5,  # Position 5 is in A region (indices 0-9)
                genomic_pos=1005,
                mod_code="a",
                canonical_base="A",  # Matches sequence[5] = 'A'
                probability=0.7,
                signal_start=15,
                signal_end=18,
            ),
        ]

        # Filter for C+m only (should show 1 modification)
        p_mod = create_modification_track_single(
            modifications=modifications,
            time_ms=time_ms,
            theme=Theme.LIGHT,
            show_overlay=True,
            mod_type_filter="C+m",
            sequence=sequence,
        )

        assert p_mod is not None

        # Filter for G+m (should show no modifications)
        p_mod_none = create_modification_track_single(
            modifications=modifications,
            time_ms=time_ms,
            theme=Theme.LIGHT,
            show_overlay=True,
            mod_type_filter="G+m",
            sequence=sequence,
        )

        assert p_mod_none is None

    def test_create_mod_track_threshold_filter(self):
        """Test probability threshold filtering"""
        time_ms = np.arange(100) * 0.25
        modifications = [
            ModificationAnnotation(
                position=5,
                genomic_pos=1005,
                mod_code="m",
                canonical_base="C",
                probability=0.8,
                signal_start=20,
                signal_end=30,
            ),
            ModificationAnnotation(
                position=15,
                genomic_pos=1015,
                mod_code="m",
                canonical_base="C",
                probability=0.3,
                signal_start=60,
                signal_end=70,
            ),
        ]
        sequence = "A" * 10 + "C" * 10 + "A" * 10

        # With threshold=0.5, should only show first modification (prob=0.8)
        p_mod = create_modification_track_single(
            modifications=modifications,
            time_ms=time_ms,
            theme=Theme.LIGHT,
            show_overlay=True,
            threshold_enabled=True,
            threshold=0.5,
            sequence=sequence,
        )

        assert p_mod is not None

        # With threshold=0.9, should show no modifications
        p_mod_none = create_modification_track_single(
            modifications=modifications,
            time_ms=time_ms,
            theme=Theme.LIGHT,
            show_overlay=True,
            threshold_enabled=True,
            threshold=0.9,
            sequence=sequence,
        )

        assert p_mod_none is None

    def test_create_mod_track_combined_filters(self):
        """Test combined type and threshold filtering"""
        time_ms = np.arange(100) * 0.25
        # Create sequence where positions match canonical bases
        sequence = "A" * 10 + "C" * 10 + "A" * 10
        modifications = [
            ModificationAnnotation(
                position=15,  # Position 15 is in C region (indices 10-19)
                genomic_pos=1015,
                mod_code="m",
                canonical_base="C",  # Matches sequence[15] = 'C'
                probability=0.8,
                signal_start=45,
                signal_end=48,
            ),
            ModificationAnnotation(
                position=12,  # Position 12 is in C region (indices 10-19)
                genomic_pos=1012,
                mod_code="m",
                canonical_base="C",  # Matches sequence[12] = 'C'
                probability=0.3,
                signal_start=36,
                signal_end=39,
            ),
            ModificationAnnotation(
                position=5,  # Position 5 is in A region (indices 0-9)
                genomic_pos=1005,
                mod_code="a",
                canonical_base="A",  # Matches sequence[5] = 'A'
                probability=0.9,
                signal_start=15,
                signal_end=18,
            ),
        ]

        # Filter for C+m with threshold=0.5 (should show only first C+m mod)
        p_mod = create_modification_track_single(
            modifications=modifications,
            time_ms=time_ms,
            theme=Theme.LIGHT,
            show_overlay=True,
            mod_type_filter="C+m",
            threshold_enabled=True,
            threshold=0.5,
            sequence=sequence,
        )

        assert p_mod is not None

    def test_create_mod_track_dark_theme(self):
        """Test modification track with dark theme"""
        time_ms = np.arange(100) * 0.25
        modifications = [
            ModificationAnnotation(
                position=5,
                genomic_pos=1005,
                mod_code="m",
                canonical_base="C",
                probability=0.8,
                signal_start=20,
                signal_end=30,
            )
        ]
        sequence = "A" * 10 + "C" * 10

        p_mod = create_modification_track_single(
            modifications=modifications,
            time_ms=time_ms,
            theme=Theme.DARK,
            show_overlay=True,
            sequence=sequence,
        )

        assert p_mod is not None


class TestModificationTrackEventalign:
    """Tests for create_modification_track_eventalign function"""

    def test_create_mod_track_eventalign_basic(self):
        """Test basic modification track creation in eventalign mode"""
        aligned_reads = [
            AlignedRead(
                read_id="read_001",
                sequence="ACGTACGT",
                chromosome="chr1",
                genomic_start=1000,
                genomic_end=1008,
                bases=[
                    BaseAnnotation(
                        base=base,
                        position=i,
                        signal_start=i * 10,
                        signal_end=(i + 1) * 10,
                    )
                    for i, base in enumerate("ACGTACGT")
                ],
                modifications=[
                    ModificationAnnotation(
                        position=1,
                        genomic_pos=1001,
                        mod_code="m",
                        canonical_base="C",
                        probability=0.8,
                        signal_start=10,
                        signal_end=20,
                    )
                ],
            )
        ]

        p_mod = create_modification_track_eventalign(
            aligned_reads=aligned_reads,
            show_dwell_time=False,
            sample_rate=4000,
            theme=Theme.LIGHT,
            show_overlay=True,
        )

        assert p_mod is not None
        assert p_mod.height == 80
        assert p_mod.yaxis.visible is False
        assert p_mod.toolbar_location is None

    def test_create_mod_track_eventalign_dwell_time(self):
        """Test modification track in dwell time mode"""
        aligned_reads = [
            AlignedRead(
                read_id="read_001",
                sequence="ACGT",
                chromosome="chr1",
                genomic_start=1000,
                genomic_end=1004,
                bases=[
                    BaseAnnotation(
                        base=base,
                        position=i,
                        signal_start=i * 10,
                        signal_end=(i + 1) * 10,
                    )
                    for i, base in enumerate("ACGT")
                ],
                modifications=[
                    ModificationAnnotation(
                        position=1,
                        genomic_pos=1001,
                        mod_code="m",
                        canonical_base="C",
                        probability=0.8,
                        signal_start=10,
                        signal_end=20,
                    )
                ],
            )
        ]

        p_mod = create_modification_track_eventalign(
            aligned_reads=aligned_reads,
            show_dwell_time=True,
            sample_rate=4000,
            theme=Theme.LIGHT,
            show_overlay=True,
        )

        assert p_mod is not None

    def test_create_mod_track_eventalign_type_filter(self):
        """Test modification type filtering in eventalign mode"""
        aligned_reads = [
            AlignedRead(
                read_id="read_001",
                sequence="ACGTACGT",
                chromosome="chr1",
                genomic_start=1000,
                genomic_end=1008,
                bases=[
                    BaseAnnotation(
                        base=base,
                        position=i,
                        signal_start=i * 10,
                        signal_end=(i + 1) * 10,
                    )
                    for i, base in enumerate("ACGTACGT")
                ],
                modifications=[
                    ModificationAnnotation(
                        position=1,
                        genomic_pos=1001,
                        mod_code="m",
                        canonical_base="C",
                        probability=0.8,
                        signal_start=10,
                        signal_end=20,
                    ),
                    ModificationAnnotation(
                        position=4,
                        genomic_pos=1004,
                        mod_code="a",
                        canonical_base="A",
                        probability=0.7,
                        signal_start=40,
                        signal_end=50,
                    ),
                ],
            )
        ]

        # Filter for C+m only
        p_mod = create_modification_track_eventalign(
            aligned_reads=aligned_reads,
            show_dwell_time=False,
            sample_rate=4000,
            theme=Theme.LIGHT,
            show_overlay=True,
            mod_type_filter="C+m",
        )

        assert p_mod is not None

        # Filter for G+m (should show no modifications)
        p_mod_none = create_modification_track_eventalign(
            aligned_reads=aligned_reads,
            show_dwell_time=False,
            sample_rate=4000,
            theme=Theme.LIGHT,
            show_overlay=True,
            mod_type_filter="G+m",
        )

        assert p_mod_none is None

    def test_create_mod_track_eventalign_threshold_filter(self):
        """Test probability threshold filtering in eventalign mode"""
        aligned_reads = [
            AlignedRead(
                read_id="read_001",
                sequence="ACGT",
                chromosome="chr1",
                genomic_start=1000,
                genomic_end=1004,
                bases=[
                    BaseAnnotation(
                        base=base,
                        position=i,
                        signal_start=i * 10,
                        signal_end=(i + 1) * 10,
                    )
                    for i, base in enumerate("ACGT")
                ],
                modifications=[
                    ModificationAnnotation(
                        position=1,
                        genomic_pos=1001,
                        mod_code="m",
                        canonical_base="C",
                        probability=0.8,
                        signal_start=10,
                        signal_end=20,
                    ),
                    ModificationAnnotation(
                        position=2,
                        genomic_pos=1002,
                        mod_code="m",
                        canonical_base="G",
                        probability=0.3,
                        signal_start=20,
                        signal_end=30,
                    ),
                ],
            )
        ]

        # With threshold=0.5, should show only first modification
        p_mod = create_modification_track_eventalign(
            aligned_reads=aligned_reads,
            show_dwell_time=False,
            sample_rate=4000,
            theme=Theme.LIGHT,
            show_overlay=True,
            threshold_enabled=True,
            threshold=0.5,
        )

        assert p_mod is not None

        # With threshold=0.9, should show no modifications
        p_mod_none = create_modification_track_eventalign(
            aligned_reads=aligned_reads,
            show_dwell_time=False,
            sample_rate=4000,
            theme=Theme.LIGHT,
            show_overlay=True,
            threshold_enabled=True,
            threshold=0.9,
        )

        assert p_mod_none is None

    def test_create_mod_track_eventalign_no_modifications(self):
        """Test eventalign with no modifications"""
        aligned_reads = [
            AlignedRead(
                read_id="read_001",
                sequence="ACGT",
                chromosome="chr1",
                genomic_start=1000,
                genomic_end=1004,
                bases=[
                    BaseAnnotation(
                        base=base,
                        position=i,
                        signal_start=i * 10,
                        signal_end=(i + 1) * 10,
                    )
                    for i, base in enumerate("ACGT")
                ],
                modifications=[],
            )
        ]

        p_mod = create_modification_track_eventalign(
            aligned_reads=aligned_reads,
            show_dwell_time=False,
            sample_rate=4000,
            theme=Theme.LIGHT,
            show_overlay=True,
        )

        assert p_mod is None


class TestPlotSingleReadWithModifications:
    """Integration tests for plot_single_read with modifications"""

    def test_plot_single_with_modifications(self):
        """Test plotting single read with modification track"""
        signal = np.random.randn(100)
        read_id = "test_read_001"
        sample_rate = 4000
        sequence = "A" * 10 + "C" * 10 + "A" * 10
        seq_to_sig_map = list(range(0, 100, 3))[:30]

        modifications = [
            ModificationAnnotation(
                position=15,
                genomic_pos=1015,
                mod_code="m",
                canonical_base="C",
                probability=0.8,
                signal_start=45,
                signal_end=48,
            )
        ]

        html, layout = plot_single_read(
            signal=signal,
            read_id=read_id,
            sample_rate=sample_rate,
            sequence=sequence,
            seq_to_sig_map=seq_to_sig_map,
            modifications=modifications,
            show_modification_overlay=True,
        )

        assert isinstance(html, str)
        assert layout is not None
        # Should be a column layout (not a single figure)
        assert hasattr(layout, "children")

    def test_plot_single_with_filtered_modifications(self):
        """Test plotting single read with filtered modifications"""
        signal = np.random.randn(100)
        read_id = "test_read_002"
        sample_rate = 4000
        sequence = "A" * 10 + "C" * 10 + "A" * 10
        seq_to_sig_map = list(range(0, 100, 3))[:30]

        modifications = [
            ModificationAnnotation(
                position=15,
                genomic_pos=1015,
                mod_code="m",
                canonical_base="C",
                probability=0.3,
                signal_start=45,
                signal_end=48,
            )
        ]

        # With threshold=0.5, modification should be filtered out
        html, fig = plot_single_read(
            signal=signal,
            read_id=read_id,
            sample_rate=sample_rate,
            sequence=sequence,
            seq_to_sig_map=seq_to_sig_map,
            modifications=modifications,
            show_modification_overlay=True,
            modification_threshold_enabled=True,
            modification_threshold=0.5,
        )

        assert isinstance(html, str)
        # Should return single figure (no mod track)
        assert not hasattr(fig, "children")


class TestPlotEventalignWithModifications:
    """Integration tests for plot_eventalign with modifications"""

    def test_plot_eventalign_with_modifications(self):
        """Test plotting eventalign with modification track"""
        reads_data = [("read_001", np.random.randn(100), 4000)]

        aligned_reads = [
            AlignedRead(
                read_id="read_001",
                sequence="ACGTACGT",
                chromosome="chr1",
                genomic_start=1000,
                genomic_end=1008,
                bases=[
                    BaseAnnotation(
                        base=base,
                        position=i,
                        signal_start=i * 10,
                        signal_end=(i + 1) * 10,
                    )
                    for i, base in enumerate("ACGTACGT")
                ],
                modifications=[
                    ModificationAnnotation(
                        position=1,
                        genomic_pos=1001,
                        mod_code="m",
                        canonical_base="C",
                        probability=0.8,
                        signal_start=10,
                        signal_end=20,
                    )
                ],
            )
        ]

        html, layout = plot_eventalign(
            reads_data=reads_data,
            normalization=NormalizationMethod.NONE,
            aligned_reads=aligned_reads,
            show_modification_overlay=True,
        )

        assert isinstance(html, str)
        assert layout is not None
        # Should be a column layout (not a single figure)
        assert hasattr(layout, "children")

    def test_plot_eventalign_with_filtered_modifications(self):
        """Test plotting eventalign with filtered modifications"""
        reads_data = [("read_001", np.random.randn(100), 4000)]

        aligned_reads = [
            AlignedRead(
                read_id="read_001",
                sequence="ACGT",
                chromosome="chr1",
                genomic_start=1000,
                genomic_end=1004,
                bases=[
                    BaseAnnotation(
                        base=base,
                        position=i,
                        signal_start=i * 10,
                        signal_end=(i + 1) * 10,
                    )
                    for i, base in enumerate("ACGT")
                ],
                modifications=[
                    ModificationAnnotation(
                        position=1,
                        genomic_pos=1001,
                        mod_code="m",
                        canonical_base="C",
                        probability=0.2,
                        signal_start=10,
                        signal_end=20,
                    )
                ],
            )
        ]

        # With threshold=0.5, modification should be filtered out
        html, fig = plot_eventalign(
            reads_data=reads_data,
            normalization=NormalizationMethod.NONE,
            aligned_reads=aligned_reads,
            show_modification_overlay=True,
            modification_threshold_enabled=True,
            modification_threshold=0.5,
        )

        assert isinstance(html, str)
        # Should return single figure (no mod track)
        assert not hasattr(fig, "children")

    def test_plot_eventalign_modification_type_filter(self):
        """Test plotting eventalign with modification type filter"""
        reads_data = [("read_001", np.random.randn(100), 4000)]

        aligned_reads = [
            AlignedRead(
                read_id="read_001",
                sequence="ACGT",
                chromosome="chr1",
                genomic_start=1000,
                genomic_end=1004,
                bases=[
                    BaseAnnotation(
                        base=base,
                        position=i,
                        signal_start=i * 10,
                        signal_end=(i + 1) * 10,
                    )
                    for i, base in enumerate("ACGT")
                ],
                modifications=[
                    ModificationAnnotation(
                        position=0,
                        genomic_pos=1000,
                        mod_code="a",
                        canonical_base="A",
                        probability=0.9,
                        signal_start=0,
                        signal_end=10,
                    ),
                    ModificationAnnotation(
                        position=1,
                        genomic_pos=1001,
                        mod_code="m",
                        canonical_base="C",
                        probability=0.8,
                        signal_start=10,
                        signal_end=20,
                    ),
                ],
            )
        ]

        # Filter for A+a only
        html, layout = plot_eventalign(
            reads_data=reads_data,
            normalization=NormalizationMethod.NONE,
            aligned_reads=aligned_reads,
            show_modification_overlay=True,
            modification_type_filter="A+a",
        )

        assert isinstance(html, str)
        assert layout is not None
