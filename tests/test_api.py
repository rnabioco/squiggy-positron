"""Tests for public API (plot_read, plot_reads functions)"""

import pytest


class TestPlotReadFunction:
    """Tests for plot_read() function"""

    def test_plot_read_requires_pod5_loaded(self):
        """Test that plot_read requires POD5 file to be loaded"""
        from squiggy import close_pod5, plot_read

        close_pod5()

        with pytest.raises(ValueError, match="No POD5 file loaded"):
            plot_read("fake_read_id")

    def test_plot_read_invalid_read_id(self, sample_pod5_file):
        """Test that plot_read raises error for invalid read ID"""
        from squiggy import load_pod5, plot_read

        load_pod5(str(sample_pod5_file))

        with pytest.raises(ValueError, match="Read not found"):
            plot_read("NONEXISTENT_READ_ID_12345")

    def test_plot_read_single_mode_returns_html(self, sample_pod5_file):
        """Test that plot_read in SINGLE mode returns HTML"""
        from squiggy import load_pod5, plot_read

        _, read_ids = load_pod5(str(sample_pod5_file))
        read_id = read_ids[0]

        html = plot_read(read_id, mode="SINGLE")

        assert isinstance(html, str)
        assert len(html) > 0
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "bokeh" in html.lower()

    def test_plot_read_with_normalization_options(self, sample_pod5_file):
        """Test plot_read with different normalization methods"""
        from squiggy import load_pod5, plot_read

        _, read_ids = load_pod5(str(sample_pod5_file))
        read_id = read_ids[0]

        # Test each normalization method
        for norm in ["NONE", "ZNORM", "MEDIAN", "MAD"]:
            html = plot_read(read_id, mode="SINGLE", normalization=norm)
            assert isinstance(html, str)
            assert len(html) > 0

    def test_plot_read_with_theme_options(self, sample_pod5_file):
        """Test plot_read with different themes"""
        from squiggy import load_pod5, plot_read

        _, read_ids = load_pod5(str(sample_pod5_file))
        read_id = read_ids[0]

        # Test each theme
        for theme in ["LIGHT", "DARK"]:
            html = plot_read(read_id, mode="SINGLE", theme=theme)
            assert isinstance(html, str)
            assert len(html) > 0

    def test_plot_read_with_downsample(self, sample_pod5_file):
        """Test plot_read with downsampling enabled/disabled"""
        from squiggy import load_pod5, plot_read

        _, read_ids = load_pod5(str(sample_pod5_file))
        read_id = read_ids[0]

        # With downsampling
        html1 = plot_read(read_id, downsample=True)
        assert isinstance(html1, str)

        # Without downsampling
        html2 = plot_read(read_id, downsample=False)
        assert isinstance(html2, str)

    def test_plot_read_eventalign_requires_bam(self, sample_pod5_file):
        """Test that EVENTALIGN mode requires BAM file"""
        from squiggy import load_pod5, plot_read

        _, read_ids = load_pod5(str(sample_pod5_file))
        read_id = read_ids[0]

        # Without BAM loaded, EVENTALIGN should raise error
        with pytest.raises(ValueError, match="EVENTALIGN mode requires a BAM file"):
            plot_read(read_id, mode="EVENTALIGN")

    def test_plot_read_eventalign_with_bam(self, sample_pod5_file, indexed_bam_file):
        """Test plot_read in EVENTALIGN mode with BAM loaded"""
        import pysam

        from squiggy import load_bam, load_pod5, plot_read

        _, read_ids = load_pod5(str(sample_pod5_file))
        load_bam(str(indexed_bam_file))

        # Find a read that has alignment in BAM
        with pysam.AlignmentFile(str(indexed_bam_file), "rb") as bam:
            for alignment in bam.fetch(until_eof=True):
                if alignment.has_tag("mv"):  # Has move table
                    read_id = alignment.query_name

                    # Should work now
                    html = plot_read(read_id, mode="EVENTALIGN")
                    assert isinstance(html, str)
                    assert len(html) > 0
                    return  # Test passed

        pytest.skip("No reads with move table found in BAM")

    def test_plot_read_with_all_options(self, sample_pod5_file, indexed_bam_file):
        """Test plot_read with all options specified"""
        import pysam

        from squiggy import load_bam, load_pod5, plot_read

        _, read_ids = load_pod5(str(sample_pod5_file))
        load_bam(str(indexed_bam_file))

        # Find aligned read
        with pysam.AlignmentFile(str(indexed_bam_file), "rb") as bam:
            for alignment in bam.fetch(until_eof=True):
                if alignment.has_tag("mv"):
                    read_id = alignment.query_name

                    html = plot_read(
                        read_id,
                        mode="EVENTALIGN",
                        normalization="ZNORM",
                        theme="DARK",
                        downsample=True,
                        show_dwell_time=True,
                        show_labels=True,
                        position_label_interval=50,
                        scale_dwell_time=False,
                        min_mod_probability=0.5,
                        enabled_mod_types=None,
                    )

                    assert isinstance(html, str)
                    assert len(html) > 0
                    return

        pytest.skip("No aligned reads found")

    def test_plot_read_unsupported_mode(self, sample_pod5_file):
        """Test that unsupported plot modes raise error"""
        from squiggy import load_pod5, plot_read

        _, read_ids = load_pod5(str(sample_pod5_file))
        read_id = read_ids[0]

        # OVERLAY and STACKED are not supported for single reads
        with pytest.raises(ValueError, match="not yet supported"):
            plot_read(read_id, mode="OVERLAY")

    def test_plot_read_invalid_normalization(self, sample_pod5_file):
        """Test that invalid normalization method raises error"""
        from squiggy import load_pod5, plot_read

        _, read_ids = load_pod5(str(sample_pod5_file))
        read_id = read_ids[0]

        # Invalid normalization should raise KeyError
        with pytest.raises(KeyError):
            plot_read(read_id, normalization="INVALID")

    def test_plot_read_invalid_theme(self, sample_pod5_file):
        """Test that invalid theme raises error"""
        from squiggy import load_pod5, plot_read

        _, read_ids = load_pod5(str(sample_pod5_file))
        read_id = read_ids[0]

        # Invalid theme should raise KeyError
        with pytest.raises(KeyError):
            plot_read(read_id, theme="INVALID")

    def test_plot_read_with_downsample_factor(self, sample_pod5_file):
        """Test plot_read with downsampling"""
        from squiggy import load_pod5, plot_read

        _, read_ids = load_pod5(str(sample_pod5_file))
        read_id = read_ids[0]

        # Downsample should work
        html = plot_read(read_id, downsample=10)
        assert isinstance(html, str)
        assert len(html) > 0

    def test_plot_read_with_signal_points(self, sample_pod5_file):
        """Test plot_read with signal points enabled"""
        from squiggy import load_pod5, plot_read

        _, read_ids = load_pod5(str(sample_pod5_file))
        read_id = read_ids[0]

        # Signal points should work
        html = plot_read(read_id, show_signal_points=True)
        assert isinstance(html, str)
        assert len(html) > 0


class TestPlotReadsFunction:
    """Tests for plot_reads() function (multiple reads)"""

    def test_plot_reads_requires_pod5_loaded(self):
        """Test that plot_reads requires POD5 file to be loaded"""
        from squiggy import close_pod5, plot_reads

        close_pod5()

        with pytest.raises(ValueError, match="No POD5 file loaded"):
            plot_reads(["fake_read_1", "fake_read_2"])

    def test_plot_reads_empty_list(self, sample_pod5_file):
        """Test that plot_reads handles empty read list"""
        from squiggy import load_pod5, plot_reads

        load_pod5(str(sample_pod5_file))

        # Empty list should return empty string
        result = plot_reads([])
        assert result == ""

    def test_plot_reads_nonexistent_ids(self, sample_pod5_file):
        """Test that plot_reads handles nonexistent read IDs"""
        from squiggy import load_pod5, plot_reads

        load_pod5(str(sample_pod5_file))

        # Nonexistent read IDs should raise ValueError from plot_read
        with pytest.raises(ValueError, match="Read not found"):
            plot_reads(["NONEXISTENT_1", "NONEXISTENT_2"])

    def test_plot_reads_overlay_mode(self, sample_pod5_file):
        """Test plot_reads in OVERLAY mode (returns first plot for now)"""
        from squiggy import load_pod5, plot_reads

        _, read_ids = load_pod5(str(sample_pod5_file))

        # Take first 2 reads
        if len(read_ids) < 2:
            pytest.skip("Need at least 2 reads for overlay test")

        html = plot_reads(read_ids[:2], mode="OVERLAY")

        # Should return valid HTML (currently returns first plot only)
        assert isinstance(html, str)
        assert len(html) > 0
        assert "bokeh" in html.lower()

    def test_plot_reads_stacked_mode(self, sample_pod5_file):
        """Test plot_reads in STACKED mode (not yet implemented)"""
        from squiggy import load_pod5, plot_reads

        _, read_ids = load_pod5(str(sample_pod5_file))

        if len(read_ids) < 2:
            pytest.skip("Need at least 2 reads for stacked test")

        # STACKED mode is not yet fully implemented
        with pytest.raises(ValueError, match="not yet fully implemented"):
            plot_reads(read_ids[:2], mode="STACKED")

    def test_plot_reads_with_options(self, sample_pod5_file):
        """Test plot_reads with various options"""
        from squiggy import load_pod5, plot_reads

        _, read_ids = load_pod5(str(sample_pod5_file))

        if len(read_ids) < 2:
            pytest.skip("Need at least 2 reads")

        html = plot_reads(
            read_ids[:2],
            mode="OVERLAY",
            normalization="MEDIAN",
            theme="DARK",
            downsample=True,
        )

        assert isinstance(html, str)
        assert len(html) > 0
        assert "bokeh" in html.lower()

    def test_plot_reads_unsupported_mode(self, sample_pod5_file):
        """Test that unsupported plot modes raise error"""
        from squiggy import load_pod5, plot_reads

        _, read_ids = load_pod5(str(sample_pod5_file))

        if len(read_ids) < 2:
            pytest.skip("Need at least 2 reads")

        # SINGLE and EVENTALIGN are not supported for multiple reads
        with pytest.raises(ValueError, match="not yet fully implemented"):
            plot_reads(read_ids[:2], mode="SINGLE")


class TestAPIStateManagement:
    """Tests for global state management in API"""

    def test_loading_multiple_files_sequentially(
        self, sample_pod5_file, indexed_bam_file
    ):
        """Test loading multiple files in sequence"""
        from squiggy import get_current_files, load_bam, load_pod5

        # Load POD5
        load_pod5(str(sample_pod5_file))
        files = get_current_files()
        assert files["pod5_path"] is not None
        assert files["bam_path"] is None

        # Load BAM
        load_bam(str(indexed_bam_file))
        files = get_current_files()
        assert files["pod5_path"] is not None
        assert files["bam_path"] is not None

    def test_reloading_pod5_updates_state(self, sample_pod5_file):
        """Test that reloading POD5 file updates read IDs"""
        from squiggy import get_read_ids, load_pod5

        # Load first time
        _, ids1 = load_pod5(str(sample_pod5_file))

        # Load second time (same file)
        _, ids2 = load_pod5(str(sample_pod5_file))

        # IDs should be the same
        assert ids1 == ids2

        # get_read_ids() should return the same
        ids3 = get_read_ids()
        assert ids3 == ids1


class TestAPIIntegration:
    """End-to-end integration tests for the API"""

    def test_full_workflow_single_read(self, sample_pod5_file):
        """Test complete workflow: load POD5 → plot single read"""
        from squiggy import close_pod5, load_pod5, plot_read

        # Clean state
        close_pod5()

        # Load file
        reader, read_ids = load_pod5(str(sample_pod5_file))
        assert len(read_ids) > 0

        # Plot first read
        html = plot_read(read_ids[0], mode="SINGLE", normalization="ZNORM")
        assert "bokeh" in html.lower()

        # Clean up
        close_pod5()

    def test_full_workflow_event_aligned(self, sample_pod5_file, indexed_bam_file):
        """Test complete workflow: load POD5 + BAM → plot event-aligned"""
        import pysam

        from squiggy import close_pod5, load_bam, load_pod5, plot_read

        # Clean state
        close_pod5()

        # Load files
        reader, read_ids = load_pod5(str(sample_pod5_file))
        load_bam(str(indexed_bam_file))

        # Find aligned read with move table
        with pysam.AlignmentFile(str(indexed_bam_file), "rb") as bam:
            for alignment in bam.fetch(until_eof=True):
                if alignment.has_tag("mv"):
                    read_id = alignment.query_name

                    # Plot with event alignment
                    html = plot_read(
                        read_id,
                        mode="EVENTALIGN",
                        normalization="ZNORM",
                        show_labels=True,
                    )
                    assert "bokeh" in html.lower()

                    # Clean up
                    close_pod5()
                    return

        pytest.skip("No aligned reads with move table found")
