"""Tests for object-oriented API (Pod5File, Read, BamFile classes)"""

import os

import pytest


class TestPod5File:
    """Tests for Pod5File class"""

    def test_pod5file_constructor(self, sample_pod5_file):
        """Test creating Pod5File object"""
        from squiggy import Pod5File

        pod5 = Pod5File(sample_pod5_file)
        assert pod5 is not None
        assert os.path.exists(pod5.path)
        pod5.close()

    def test_pod5file_nonexistent_file(self):
        """Test that Pod5File raises error for nonexistent file"""
        from squiggy import Pod5File

        with pytest.raises(FileNotFoundError, match="POD5 file not found"):
            Pod5File("/nonexistent/file.pod5")

    def test_pod5file_read_ids_property(self, sample_pod5_file):
        """Test getting read IDs from Pod5File"""
        from squiggy import Pod5File

        pod5 = Pod5File(sample_pod5_file)
        read_ids = pod5.read_ids

        assert isinstance(read_ids, list)
        assert len(read_ids) > 0
        assert all(isinstance(rid, str) for rid in read_ids)

        pod5.close()

    def test_pod5file_len(self, sample_pod5_file):
        """Test __len__ returns number of reads"""
        from squiggy import Pod5File

        pod5 = Pod5File(sample_pod5_file)
        num_reads = len(pod5)

        assert num_reads > 0
        assert num_reads == len(pod5.read_ids)

        pod5.close()

    def test_pod5file_get_read(self, sample_pod5_file):
        """Test getting a single read by ID"""
        from squiggy import Pod5File

        pod5 = Pod5File(sample_pod5_file)
        read_id = pod5.read_ids[0]
        read = pod5.get_read(read_id)

        assert read is not None
        assert read.read_id == read_id
        assert hasattr(read, "signal")

        pod5.close()

    def test_pod5file_get_read_invalid_id(self, sample_pod5_file):
        """Test that get_read raises error for invalid ID"""
        from squiggy import Pod5File

        pod5 = Pod5File(sample_pod5_file)

        with pytest.raises(ValueError, match="Read not found"):
            pod5.get_read("INVALID_READ_ID_12345")

        pod5.close()

    def test_pod5file_iter_reads(self, sample_pod5_file):
        """Test iterating over reads"""
        from squiggy import Pod5File

        pod5 = Pod5File(sample_pod5_file)

        # Iterate over first 5 reads
        reads = list(pod5.iter_reads(limit=5))

        assert len(reads) == 5
        assert all(hasattr(read, "signal") for read in reads)
        assert all(hasattr(read, "read_id") for read in reads)

        pod5.close()

    def test_pod5file_iter_reads_no_limit(self, sample_pod5_file):
        """Test iterating over all reads"""
        from squiggy import Pod5File

        pod5 = Pod5File(sample_pod5_file)

        reads = list(pod5.iter_reads())
        assert len(reads) == len(pod5)

        pod5.close()

    def test_pod5file_context_manager(self, sample_pod5_file):
        """Test Pod5File as context manager"""
        from squiggy import Pod5File

        with Pod5File(sample_pod5_file) as pod5:
            assert len(pod5) > 0
            read_ids = pod5.read_ids
            assert len(read_ids) > 0

        # File should be closed after exiting context
        # (No exception should be raised)

    def test_pod5file_repr(self, sample_pod5_file):
        """Test Pod5File string representation"""
        from squiggy import Pod5File

        pod5 = Pod5File(sample_pod5_file)
        repr_str = repr(pod5)

        assert "Pod5File" in repr_str
        assert str(pod5.path) in repr_str
        assert "num_reads" in repr_str

        pod5.close()


class TestRead:
    """Tests for Read class"""

    def test_read_properties(self, sample_pod5_file):
        """Test Read object properties"""
        from squiggy import Pod5File

        pod5 = Pod5File(sample_pod5_file)
        read = pod5.get_read(pod5.read_ids[0])

        # Check properties exist and have correct types
        assert isinstance(read.read_id, str)
        assert isinstance(read.signal, object)  # numpy array
        assert isinstance(read.sample_rate, int)
        assert read.sample_rate > 0

        pod5.close()

    def test_read_signal_is_numpy_array(self, sample_pod5_file):
        """Test that signal property returns numpy array"""
        import numpy as np

        from squiggy import Pod5File

        pod5 = Pod5File(sample_pod5_file)
        read = pod5.get_read(pod5.read_ids[0])

        signal = read.signal
        assert isinstance(signal, np.ndarray)
        assert len(signal) > 0

        pod5.close()

    def test_read_get_normalized_znorm(self, sample_pod5_file):
        """Test signal normalization with ZNORM"""
        import numpy as np

        from squiggy import Pod5File

        pod5 = Pod5File(sample_pod5_file)
        read = pod5.get_read(pod5.read_ids[0])

        normalized = read.get_normalized("ZNORM")

        assert isinstance(normalized, np.ndarray)
        assert len(normalized) == len(read.signal)
        # ZNORM should have mean ~0 and std ~1
        assert abs(np.mean(normalized)) < 0.1
        assert abs(np.std(normalized) - 1.0) < 0.1

        pod5.close()

    def test_read_get_normalized_all_methods(self, sample_pod5_file):
        """Test all normalization methods"""
        import numpy as np

        from squiggy import Pod5File

        pod5 = Pod5File(sample_pod5_file)
        read = pod5.get_read(pod5.read_ids[0])

        for method in ["NONE", "ZNORM", "MEDIAN", "MAD"]:
            normalized = read.get_normalized(method)
            assert isinstance(normalized, np.ndarray)
            assert len(normalized) == len(read.signal)

        pod5.close()

    def test_read_get_alignment_no_bam(self, sample_pod5_file):
        """Test that get_alignment requires BAM file"""
        from squiggy import Pod5File

        pod5 = Pod5File(sample_pod5_file)
        read = pod5.get_read(pod5.read_ids[0])

        with pytest.raises(
            ValueError, match="Must provide either bam_file or bam_path"
        ):
            read.get_alignment()

        pod5.close()

    def test_read_get_alignment_with_bam(self, sample_pod5_file, indexed_bam_file):
        """Test getting alignment from BAM file"""
        import pysam

        from squiggy import BamFile, Pod5File

        pod5 = Pod5File(sample_pod5_file)
        bam = BamFile(indexed_bam_file)

        # Find a read that has alignment with move table
        with pysam.AlignmentFile(str(indexed_bam_file), "rb") as pysam_bam:
            for alignment in pysam_bam.fetch(until_eof=True):
                if alignment.has_tag("mv"):
                    read_id = alignment.query_name
                    read = pod5.get_read(read_id)

                    aligned_read = read.get_alignment(bam)
                    if aligned_read is not None:
                        assert aligned_read.read_id == read_id
                        assert aligned_read.sequence is not None
                        assert len(aligned_read.bases) > 0
                        pod5.close()
                        bam.close()
                        return

        pod5.close()
        bam.close()
        pytest.skip("No reads with move table found")

    def test_read_plot_single_mode(self, sample_pod5_file):
        """Test plotting read in SINGLE mode"""
        from bokeh.models import LayoutDOM

        from squiggy import Pod5File

        pod5 = Pod5File(sample_pod5_file)
        read = pod5.get_read(pod5.read_ids[0])

        fig = read.plot(mode="SINGLE")

        # Should return Bokeh figure or layout
        assert isinstance(fig, LayoutDOM)

        pod5.close()

    def test_read_plot_eventalign_requires_bam(self, sample_pod5_file):
        """Test that EVENTALIGN mode requires bam_file parameter"""
        from squiggy import Pod5File

        pod5 = Pod5File(sample_pod5_file)
        read = pod5.get_read(pod5.read_ids[0])

        with pytest.raises(ValueError, match="EVENTALIGN mode requires bam_file"):
            read.plot(mode="EVENTALIGN")

        pod5.close()

    def test_read_plot_eventalign_with_bam(self, sample_pod5_file, indexed_bam_file):
        """Test plotting read in EVENTALIGN mode with BAM"""
        import pysam
        from bokeh.models import LayoutDOM

        from squiggy import BamFile, Pod5File

        pod5 = Pod5File(sample_pod5_file)
        bam = BamFile(indexed_bam_file)

        # Find aligned read with move table
        with pysam.AlignmentFile(str(indexed_bam_file), "rb") as pysam_bam:
            for alignment in pysam_bam.fetch(until_eof=True):
                if alignment.has_tag("mv"):
                    read_id = alignment.query_name
                    read = pod5.get_read(read_id)

                    fig = read.plot(mode="EVENTALIGN", bam_file=bam)
                    assert isinstance(fig, LayoutDOM)

                    pod5.close()
                    bam.close()
                    return

        pod5.close()
        bam.close()
        pytest.skip("No aligned reads with move table found")

    def test_read_plot_all_options(self, sample_pod5_file):
        """Test plotting with all options specified"""
        from squiggy import Pod5File

        pod5 = Pod5File(sample_pod5_file)
        read = pod5.get_read(pod5.read_ids[0])

        fig = read.plot(
            mode="SINGLE",
            normalization="MEDIAN",
            theme="DARK",
            downsample=10,
            show_signal_points=True,
        )

        assert fig is not None

        pod5.close()

    def test_read_repr(self, sample_pod5_file):
        """Test Read string representation"""
        from squiggy import Pod5File

        pod5 = Pod5File(sample_pod5_file)
        read = pod5.get_read(pod5.read_ids[0])

        repr_str = repr(read)
        assert "Read" in repr_str
        assert read.read_id in repr_str
        assert "signal_length" in repr_str

        pod5.close()


class TestBamFile:
    """Tests for BamFile class"""

    def test_bamfile_constructor(self, indexed_bam_file):
        """Test creating BamFile object"""
        from squiggy import BamFile

        bam = BamFile(indexed_bam_file)
        assert bam is not None
        assert os.path.exists(bam.path)
        bam.close()

    def test_bamfile_nonexistent_file(self):
        """Test that BamFile raises error for nonexistent file"""
        from squiggy import BamFile

        with pytest.raises(FileNotFoundError, match="BAM file not found"):
            BamFile("/nonexistent/file.bam")

    def test_bamfile_references_property(self, indexed_bam_file):
        """Test getting reference names from BAM"""
        from squiggy import BamFile

        bam = BamFile(indexed_bam_file)
        references = bam.references

        assert isinstance(references, list)
        # May be empty if BAM has no references
        assert all(isinstance(ref, str) for ref in references)

        bam.close()

    def test_bamfile_get_alignment(self, indexed_bam_file):
        """Test getting alignment for a specific read"""
        import pysam

        from squiggy import BamFile

        bam = BamFile(indexed_bam_file)

        # Find a read with move table
        with pysam.AlignmentFile(str(indexed_bam_file), "rb") as pysam_bam:
            for alignment in pysam_bam.fetch(until_eof=True):
                if alignment.has_tag("mv"):
                    read_id = alignment.query_name

                    aligned_read = bam.get_alignment(read_id)
                    if aligned_read is not None:
                        assert aligned_read.read_id == read_id
                        assert aligned_read.sequence is not None
                        bam.close()
                        return

        bam.close()
        pytest.skip("No reads with move table found")

    def test_bamfile_get_alignment_invalid_id(self, indexed_bam_file):
        """Test that get_alignment returns None for invalid read ID"""
        from squiggy import BamFile

        bam = BamFile(indexed_bam_file)
        aligned_read = bam.get_alignment("INVALID_READ_ID_12345")

        assert aligned_read is None

        bam.close()

    def test_bamfile_iter_region(self, indexed_bam_file):
        """Test iterating over alignments in a region"""
        import pysam

        from squiggy import BamFile

        bam = BamFile(indexed_bam_file)

        # Get first reference
        if len(bam.references) == 0:
            bam.close()
            pytest.skip("No references in BAM file")

        ref_name = bam.references[0]

        # Get length of reference
        with pysam.AlignmentFile(str(indexed_bam_file), "rb") as pysam_bam:
            ref_lengths = dict(
                zip(pysam_bam.references, pysam_bam.lengths, strict=True)
            )
            if ref_name not in ref_lengths:
                bam.close()
                pytest.skip(f"Reference {ref_name} not found")

            ref_length = ref_lengths[ref_name]

        # Iterate over region
        alignments = list(bam.iter_region(ref_name, 0, min(1000, ref_length)))

        # Should return list of AlignedRead objects (may be empty)
        assert isinstance(alignments, list)

        bam.close()

    def test_bamfile_get_modifications_info(self, indexed_bam_file):
        """Test getting modification information from BAM"""
        from squiggy import BamFile

        bam = BamFile(indexed_bam_file)
        mod_info = bam.get_modifications_info()

        assert isinstance(mod_info, dict)
        assert "has_modifications" in mod_info
        assert "modification_types" in mod_info
        assert "sample_count" in mod_info
        assert "has_probabilities" in mod_info

        bam.close()

    def test_bamfile_context_manager(self, indexed_bam_file):
        """Test BamFile as context manager"""
        from squiggy import BamFile

        with BamFile(indexed_bam_file) as bam:
            assert os.path.exists(bam.path)
            references = bam.references
            assert isinstance(references, list)

        # File should be closed after exiting context

    def test_bamfile_repr(self, indexed_bam_file):
        """Test BamFile string representation"""
        from squiggy import BamFile

        bam = BamFile(indexed_bam_file)
        repr_str = repr(bam)

        assert "BamFile" in repr_str
        assert str(bam.path) in repr_str
        assert "num_references" in repr_str

        bam.close()


class TestFigureToHtml:
    """Tests for figure_to_html utility function"""

    def test_figure_to_html_returns_html(self, sample_pod5_file):
        """Test that figure_to_html converts Bokeh figure to HTML"""
        from squiggy import Pod5File, figure_to_html

        pod5 = Pod5File(sample_pod5_file)
        read = pod5.get_read(pod5.read_ids[0])

        fig = read.plot(mode="SINGLE")
        html = figure_to_html(fig)

        assert isinstance(html, str)
        assert len(html) > 0
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "bokeh" in html.lower()

        pod5.close()

    def test_figure_to_html_custom_title(self, sample_pod5_file):
        """Test figure_to_html with custom title"""
        from squiggy import Pod5File, figure_to_html

        pod5 = Pod5File(sample_pod5_file)
        read = pod5.get_read(pod5.read_ids[0])

        fig = read.plot(mode="SINGLE")
        html = figure_to_html(fig, title="My Custom Title")

        assert "My Custom Title" in html

        pod5.close()


class TestIntegrationWorkflows:
    """Integration tests for complete workflows using OO API"""

    def test_workflow_single_read_analysis(self, sample_pod5_file):
        """Test complete workflow: open file, get read, analyze signal"""
        import numpy as np

        from squiggy import Pod5File

        with Pod5File(sample_pod5_file) as pod5:
            # Get first read
            read = pod5.get_read(pod5.read_ids[0])

            # Access signal data
            signal = read.signal
            assert isinstance(signal, np.ndarray)
            assert len(signal) > 0

            # Get normalized signal
            znorm = read.get_normalized("ZNORM")
            assert len(znorm) == len(signal)

            # Plot
            fig = read.plot(mode="SINGLE")
            assert fig is not None

    def test_workflow_iterate_and_analyze(self, sample_pod5_file):
        """Test workflow: iterate over reads and analyze"""
        from squiggy import Pod5File

        with Pod5File(sample_pod5_file) as pod5:
            # Iterate over first 10 reads
            for i, read in enumerate(pod5.iter_reads(limit=10)):
                # Access data
                assert len(read.signal) > 0
                assert read.sample_rate > 0

                # Get normalized signal
                normalized = read.get_normalized("ZNORM")
                assert len(normalized) == len(read.signal)

                if i >= 9:
                    break

    def test_workflow_read_with_alignment(self, sample_pod5_file, indexed_bam_file):
        """Test workflow: read with alignment and plotting"""
        import pysam

        from squiggy import BamFile, Pod5File

        pod5 = Pod5File(sample_pod5_file)
        bam = BamFile(indexed_bam_file)

        # Find aligned read with move table
        with pysam.AlignmentFile(str(indexed_bam_file), "rb") as pysam_bam:
            for alignment in pysam_bam.fetch(until_eof=True):
                if alignment.has_tag("mv"):
                    read_id = alignment.query_name
                    read = pod5.get_read(read_id)

                    # Get alignment
                    aligned_read = read.get_alignment(bam)
                    if aligned_read is not None:
                        assert aligned_read.sequence is not None
                        assert len(aligned_read.bases) > 0

                        # Plot with alignment
                        fig = read.plot(mode="EVENTALIGN", bam_file=bam)
                        assert fig is not None

                        pod5.close()
                        bam.close()
                        return

        pod5.close()
        bam.close()
        pytest.skip("No aligned reads with move table found")

    def test_workflow_multiple_files_no_global_state(
        self, sample_pod5_file, indexed_bam_file
    ):
        """Test that OO API doesn't pollute global state"""
        from squiggy import BamFile, Pod5File
        from squiggy import io as squiggy_io

        # Verify no global state before
        assert squiggy_io.squiggy_kernel._reader is None
        assert squiggy_io.squiggy_kernel._bam_path is None

        # Use OO API
        pod5 = Pod5File(sample_pod5_file)
        bam = BamFile(indexed_bam_file)

        # Access data
        _ = pod5.read_ids
        _ = bam.references

        # Verify global state is still None (OO API doesn't touch it)
        assert squiggy_io.squiggy_kernel._reader is None
        assert squiggy_io.squiggy_kernel._bam_path is None

        pod5.close()
        bam.close()
