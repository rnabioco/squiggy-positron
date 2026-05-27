"""Tests for plotting functions with OO API"""

import pytest
from bokeh.models import LayoutDOM


class TestPlotRead:
    """Tests for plot_read() function"""

    def test_plot_read_invalid_read_id(self, sample_pod5_file):
        from squiggy import Pod5File, plot_read

        pod5 = Pod5File(sample_pod5_file)
        with pytest.raises(ValueError, match="Read not found"):
            plot_read("NONEXISTENT_READ_ID", pod5_file=pod5)
        pod5.close()

    def test_plot_read_single_mode(self, sample_pod5_file):
        from squiggy import Pod5File, plot_read

        pod5 = Pod5File(sample_pod5_file)
        fig = plot_read(pod5.read_ids[0], pod5_file=pod5, mode="SINGLE")
        assert isinstance(fig, LayoutDOM)
        pod5.close()

    def test_plot_read_normalization_options(self, sample_pod5_file):
        from squiggy import Pod5File, plot_read

        pod5 = Pod5File(sample_pod5_file)
        read_id = pod5.read_ids[0]

        for norm in ["NONE", "ZNORM", "MEDIAN", "MAD"]:
            fig = plot_read(read_id, pod5_file=pod5, normalization=norm)
            assert isinstance(fig, LayoutDOM)
        pod5.close()

    def test_plot_read_eventalign_requires_bam(self, sample_pod5_file):
        from squiggy import Pod5File, plot_read

        pod5 = Pod5File(sample_pod5_file)
        with pytest.raises(ValueError, match="requires a bam_file"):
            plot_read(pod5.read_ids[0], pod5_file=pod5, mode="EVENTALIGN")
        pod5.close()

    def test_plot_read_eventalign_mode(self, sample_pod5_file, indexed_bam_file):
        from squiggy import BamFile, Pod5File, plot_read

        pod5 = Pod5File(sample_pod5_file)
        bam = BamFile(indexed_bam_file)
        read_id = pod5.read_ids[0]

        fig = plot_read(read_id, pod5_file=pod5, bam_file=bam, mode="EVENTALIGN")
        assert isinstance(fig, LayoutDOM)
        pod5.close()
        bam.close()

    def test_plot_read_sequence_coordinate_space(
        self, sample_pod5_file, indexed_bam_file
    ):
        from squiggy import BamFile, Pod5File, plot_read

        pod5 = Pod5File(sample_pod5_file)
        bam = BamFile(indexed_bam_file)
        read_id = pod5.read_ids[0]

        fig = plot_read(
            read_id,
            pod5_file=pod5,
            bam_file=bam,
            mode="SINGLE",
            coordinate_space="sequence",
        )
        assert isinstance(fig, LayoutDOM)
        pod5.close()
        bam.close()


class TestPlotReads:
    """Tests for plot_reads() function"""

    def test_plot_reads_overlay(self, sample_pod5_file):
        from squiggy import Pod5File, plot_reads

        pod5 = Pod5File(sample_pod5_file)
        ids = pod5.read_ids[:3]

        fig = plot_reads(ids, pod5_file=pod5, mode="OVERLAY")
        assert isinstance(fig, LayoutDOM)
        pod5.close()

    def test_plot_reads_stacked(self, sample_pod5_file):
        from squiggy import Pod5File, plot_reads

        pod5 = Pod5File(sample_pod5_file)
        ids = pod5.read_ids[:3]

        fig = plot_reads(ids, pod5_file=pod5, mode="STACKED")
        assert isinstance(fig, LayoutDOM)
        pod5.close()

    def test_plot_reads_eventalign(self, sample_pod5_file, indexed_bam_file):
        from squiggy import BamFile, Pod5File, plot_reads

        pod5 = Pod5File(sample_pod5_file)
        bam = BamFile(indexed_bam_file)
        ref = list(bam.ref_mapping.keys())[0]
        ids = bam.get_reads_for_reference(ref, limit=2)

        fig = plot_reads(ids, pod5_file=pod5, bam_file=bam, mode="EVENTALIGN")
        assert isinstance(fig, LayoutDOM)
        pod5.close()
        bam.close()

    def test_plot_reads_reference_overlay(self, sample_pod5_file, indexed_bam_file):
        from squiggy import BamFile, Pod5File, plot_reads

        pod5 = Pod5File(sample_pod5_file)
        bam = BamFile(indexed_bam_file)
        ref = list(bam.ref_mapping.keys())[0]
        ids = bam.get_reads_for_reference(ref, limit=3)

        fig = plot_reads(ids, pod5_file=pod5, bam_file=bam, mode="REFERENCE_OVERLAY")
        assert isinstance(fig, LayoutDOM)
        pod5.close()
        bam.close()

    def test_plot_reads_empty_list_raises(self, sample_pod5_file):
        from squiggy import Pod5File, plot_reads

        pod5 = Pod5File(sample_pod5_file)
        with pytest.raises(ValueError, match="No read IDs"):
            plot_reads([], pod5_file=pod5)
        pod5.close()


class TestPlotAggregate:
    """Tests for plot_aggregate() function"""

    def test_plot_aggregate_basic(self, sample_pod5_file, indexed_bam_file):
        from squiggy import BamFile, Pod5File, plot_aggregate

        pod5 = Pod5File(sample_pod5_file)
        bam = BamFile(indexed_bam_file)
        ref = list(bam.ref_mapping.keys())[0]

        fig = plot_aggregate(ref, pod5_file=pod5, bam_file=bam, max_reads=10)
        assert isinstance(fig, LayoutDOM)
        pod5.close()
        bam.close()

    def test_plot_aggregate_with_options(self, sample_pod5_file, indexed_bam_file):
        from squiggy import BamFile, Pod5File, plot_aggregate

        pod5 = Pod5File(sample_pod5_file)
        bam = BamFile(indexed_bam_file)
        ref = list(bam.ref_mapping.keys())[0]

        fig = plot_aggregate(
            ref,
            pod5_file=pod5,
            bam_file=bam,
            max_reads=10,
            show_modifications=False,
            show_dwell_time=False,
            transform_coordinates=True,
        )
        assert isinstance(fig, LayoutDOM)
        pod5.close()
        bam.close()


class TestPlotPileup:
    """Tests for plot_pileup() function"""

    def test_plot_pileup_bam_only(self, indexed_bam_file):
        from squiggy import BamFile, plot_pileup

        bam = BamFile(indexed_bam_file)
        ref = list(bam.ref_mapping.keys())[0]

        fig = plot_pileup(ref, bam_file=bam, max_reads=10)
        assert isinstance(fig, LayoutDOM)
        bam.close()


class TestSample:
    """Tests for Sample class"""

    def test_sample_creation(self, sample_pod5_file, indexed_bam_file):
        from squiggy import Sample

        sample = Sample("test", sample_pod5_file, indexed_bam_file)
        assert sample.name == "test"
        assert len(sample.read_ids) > 0
        assert len(sample.references) > 0
        sample.close()

    def test_sample_context_manager(self, sample_pod5_file, indexed_bam_file):
        from squiggy import Sample

        with Sample("test", sample_pod5_file, indexed_bam_file) as s:
            assert len(s.read_ids) > 0

    def test_sample_pod5_only(self, sample_pod5_file):
        from squiggy import Sample

        sample = Sample("test", sample_pod5_file)
        assert sample.bam is None
        assert sample.fasta is None
        assert len(sample.read_ids) > 0
        sample.close()


class TestBamFileExtensions:
    """Tests for new BamFile methods"""

    def test_bam_info_property(self, indexed_bam_file):
        from squiggy import BamFile

        bam = BamFile(indexed_bam_file)
        info = bam.info
        assert "references" in info
        assert "has_modifications" in info
        assert "has_event_alignment" in info
        assert "ref_mapping" in info
        bam.close()

    def test_bam_ref_mapping(self, indexed_bam_file):
        from squiggy import BamFile

        bam = BamFile(indexed_bam_file)
        mapping = bam.ref_mapping
        assert isinstance(mapping, dict)
        assert len(mapping) > 0
        for ref, read_ids in mapping.items():
            assert isinstance(ref, str)
            assert isinstance(read_ids, list)
        bam.close()

    def test_bam_get_reads_for_reference(self, indexed_bam_file):
        from squiggy import BamFile

        bam = BamFile(indexed_bam_file)
        ref = list(bam.ref_mapping.keys())[0]

        # Get all reads
        all_reads = bam.get_reads_for_reference(ref)
        assert len(all_reads) > 0

        # Get paginated
        page = bam.get_reads_for_reference(ref, offset=0, limit=2)
        assert len(page) <= 2
        assert page == all_reads[:2]
        bam.close()
