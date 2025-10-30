"""Tests for aggregate plotting functionality."""


class TestAggregateDataExtraction:
    """Tests for extracting reads and calculating aggregate statistics."""

    def test_extract_reads_for_reference(self, sample_pod5_file, indexed_bam_file):
        """Test extracting reads mapping to a reference sequence."""
        from squiggy.utils import extract_reads_for_reference, get_bam_references

        # Get a reference from the BAM file
        references = get_bam_references(indexed_bam_file)
        assert len(references) > 0, "BAM file should have at least one reference"

        ref_name = references[0]["name"]

        # Extract reads for this reference
        reads_data = extract_reads_for_reference(
            sample_pod5_file,
            indexed_bam_file,
            ref_name,
            max_reads=10,
            random_sample=True,
        )

        # Verify we got some reads
        assert len(reads_data) > 0, f"Should extract reads for reference {ref_name}"

        # Verify each read has expected fields
        for read in reads_data:
            assert "read_id" in read
            assert "signal" in read
            assert "sample_rate" in read
            assert "reference_start" in read
            assert "reference_end" in read
            assert "sequence" in read
            assert "move_table" in read
            assert "stride" in read
            assert read["stride"] > 0

    def test_calculate_aggregate_signal(self, sample_pod5_file, indexed_bam_file):
        """Test calculating aggregate signal statistics."""
        from squiggy.constants import NormalizationMethod
        from squiggy.utils import (
            calculate_aggregate_signal,
            extract_reads_for_reference,
            get_bam_references,
        )

        # Get reads for first reference
        references = get_bam_references(indexed_bam_file)
        ref_name = references[0]["name"]

        reads_data = extract_reads_for_reference(
            sample_pod5_file, indexed_bam_file, ref_name, max_reads=10
        )

        # Calculate aggregate statistics
        aggregate_stats = calculate_aggregate_signal(
            reads_data, NormalizationMethod.MEDIAN
        )

        # Verify output structure
        assert "positions" in aggregate_stats
        assert "mean_signal" in aggregate_stats
        assert "std_signal" in aggregate_stats
        assert "median_signal" in aggregate_stats
        assert "coverage" in aggregate_stats

        # Verify arrays have same length
        n_positions = len(aggregate_stats["positions"])
        assert len(aggregate_stats["mean_signal"]) == n_positions
        assert len(aggregate_stats["std_signal"]) == n_positions
        assert len(aggregate_stats["median_signal"]) == n_positions
        assert len(aggregate_stats["coverage"]) == n_positions

        # Verify positions are sorted
        positions = aggregate_stats["positions"]
        assert all(positions[i] <= positions[i + 1] for i in range(len(positions) - 1))

        # Verify coverage values are reasonable (at least 1)
        # Note: coverage can be > num_reads because multiple signal points
        # from same read can map to same reference position (due to stride)
        coverage = aggregate_stats["coverage"]
        assert all(c >= 1 for c in coverage)

    def test_calculate_base_pileup(self, sample_pod5_file, indexed_bam_file):
        """Test calculating base pileup statistics."""
        from squiggy.utils import (
            calculate_base_pileup,
            extract_reads_for_reference,
            get_bam_references,
        )

        # Get reads for first reference
        references = get_bam_references(indexed_bam_file)
        ref_name = references[0]["name"]

        reads_data = extract_reads_for_reference(
            sample_pod5_file, indexed_bam_file, ref_name, max_reads=10
        )

        # Calculate base pileup
        pileup_stats = calculate_base_pileup(reads_data)

        # Verify output structure
        assert "positions" in pileup_stats
        assert "counts" in pileup_stats

        # Verify counts dict has entries for each position
        positions = pileup_stats["positions"]
        counts = pileup_stats["counts"]
        assert len(positions) > 0
        assert len(counts) == len(positions)

        # Verify each position has base counts
        for pos in positions:
            assert pos in counts
            base_counts = counts[pos]
            assert isinstance(base_counts, dict)
            # Should have at least one base
            assert len(base_counts) > 0
            # All bases should be A, C, G, T, or U
            for base in base_counts.keys():
                assert base in ["A", "C", "G", "T", "U"]
            # All counts should be positive
            for count in base_counts.values():
                assert count > 0

    def test_calculate_quality_by_position(self, sample_pod5_file, indexed_bam_file):
        """Test calculating quality scores by position."""
        from squiggy.utils import (
            calculate_quality_by_position,
            extract_reads_for_reference,
            get_bam_references,
        )

        # Get reads for first reference
        references = get_bam_references(indexed_bam_file)
        ref_name = references[0]["name"]

        reads_data = extract_reads_for_reference(
            sample_pod5_file, indexed_bam_file, ref_name, max_reads=10
        )

        # Calculate quality statistics
        quality_stats = calculate_quality_by_position(reads_data)

        # Verify output structure
        assert "positions" in quality_stats
        assert "mean_quality" in quality_stats
        assert "std_quality" in quality_stats

        # Verify arrays have same length
        n_positions = len(quality_stats["positions"])
        assert len(quality_stats["mean_quality"]) == n_positions
        assert len(quality_stats["std_quality"]) == n_positions

        # Verify quality values are in reasonable range
        # Note: Phred scores typically 0-60, but can occasionally be higher
        mean_quality = quality_stats["mean_quality"]
        assert all(q >= 0 for q in mean_quality)
        assert all(q < 100 for q in mean_quality)  # Sanity check upper bound


class TestAggregatePlotting:
    """Tests for aggregate plot generation."""

    def test_plot_aggregate_returns_html(self, sample_pod5_file, indexed_bam_file):
        """Test that plot_aggregate returns HTML and a grid object."""
        from squiggy.constants import NormalizationMethod
        from squiggy.plotting import SquigglePlotter
        from squiggy.utils import (
            calculate_aggregate_signal,
            calculate_base_pileup,
            calculate_quality_by_position,
            extract_reads_for_reference,
            get_bam_references,
        )

        # Get reads for first reference
        references = get_bam_references(indexed_bam_file)
        ref_name = references[0]["name"]

        reads_data = extract_reads_for_reference(
            sample_pod5_file, indexed_bam_file, ref_name, max_reads=10
        )

        # Calculate statistics
        aggregate_stats = calculate_aggregate_signal(
            reads_data, NormalizationMethod.MEDIAN
        )
        pileup_stats = calculate_base_pileup(reads_data)
        quality_stats = calculate_quality_by_position(reads_data)

        # Generate plot
        html, grid = SquigglePlotter.plot_aggregate(
            aggregate_stats,
            pileup_stats,
            quality_stats,
            ref_name,
            len(reads_data),
            NormalizationMethod.MEDIAN,
        )

        # Verify HTML string is returned
        assert isinstance(html, str)
        assert len(html) > 0

        # Verify grid object is returned
        assert grid is not None

    def test_aggregate_html_contains_bokeh(self, sample_pod5_file, indexed_bam_file):
        """Test that aggregate HTML contains Bokeh elements."""
        from squiggy.constants import NormalizationMethod
        from squiggy.plotting import SquigglePlotter
        from squiggy.utils import (
            calculate_aggregate_signal,
            calculate_base_pileup,
            calculate_quality_by_position,
            extract_reads_for_reference,
            get_bam_references,
        )

        # Get reads for first reference
        references = get_bam_references(indexed_bam_file)
        ref_name = references[0]["name"]

        reads_data = extract_reads_for_reference(
            sample_pod5_file, indexed_bam_file, ref_name, max_reads=10
        )

        # Calculate statistics
        aggregate_stats = calculate_aggregate_signal(
            reads_data, NormalizationMethod.MEDIAN
        )
        pileup_stats = calculate_base_pileup(reads_data)
        quality_stats = calculate_quality_by_position(reads_data)

        # Generate plot
        html, _grid = SquigglePlotter.plot_aggregate(
            aggregate_stats,
            pileup_stats,
            quality_stats,
            ref_name,
            len(reads_data),
            NormalizationMethod.MEDIAN,
        )

        # Verify HTML structure
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html

        # Verify Bokeh is included
        assert "Bokeh" in html or "bokeh" in html.lower()

        # Verify reference name appears in plot
        assert ref_name in html

    def test_aggregate_with_different_normalizations(
        self, sample_pod5_file, indexed_bam_file
    ):
        """Test aggregate plotting with different normalization methods."""
        from squiggy.constants import NormalizationMethod
        from squiggy.plotting import SquigglePlotter
        from squiggy.utils import (
            calculate_aggregate_signal,
            calculate_base_pileup,
            calculate_quality_by_position,
            extract_reads_for_reference,
            get_bam_references,
        )

        # Get reads for first reference
        references = get_bam_references(indexed_bam_file)
        ref_name = references[0]["name"]

        reads_data = extract_reads_for_reference(
            sample_pod5_file, indexed_bam_file, ref_name, max_reads=10
        )

        # Test each normalization method
        for norm_method in [
            NormalizationMethod.NONE,
            NormalizationMethod.ZNORM,
            NormalizationMethod.MEDIAN,
            NormalizationMethod.MAD,
        ]:
            aggregate_stats = calculate_aggregate_signal(reads_data, norm_method)
            pileup_stats = calculate_base_pileup(reads_data)
            quality_stats = calculate_quality_by_position(reads_data)

            html, grid = SquigglePlotter.plot_aggregate(
                aggregate_stats,
                pileup_stats,
                quality_stats,
                ref_name,
                len(reads_data),
                norm_method,
            )

            assert isinstance(html, str)
            assert len(html) > 0
            assert grid is not None
            # Verify normalization method appears in HTML
            assert norm_method.value in html
