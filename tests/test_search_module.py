"""Tests for search.py module - SearchManager functionality"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest


class TestSearchManager:
    """Tests for SearchManager class"""

    @pytest.fixture
    def mock_parent_window(self):
        """Create mock parent window for SearchManager"""
        return MagicMock()

    @pytest.fixture
    def search_manager(self, mock_parent_window):
        """Create SearchManager instance for testing"""
        from squiggy.search import SearchManager

        return SearchManager(mock_parent_window)

    def test_search_manager_init(self, mock_parent_window):
        """Test SearchManager initialization"""
        from squiggy.search import SearchManager

        manager = SearchManager(mock_parent_window)

        assert manager.parent == mock_parent_window

    def test_filter_by_read_id(self, search_manager):
        """Test filtering read tree by read ID"""
        mock_read_tree = MagicMock()

        search_manager.filter_by_read_id(mock_read_tree, "test_read")

        # Should call the read tree's filter method
        mock_read_tree.filter_by_read_id.assert_called_once_with("test_read")

    def test_filter_by_region_empty_string(self, search_manager):
        """Test filtering by region with empty string shows all reads"""
        mock_read_tree = MagicMock()

        # Run async function
        result = asyncio.run(search_manager.filter_by_region(None, mock_read_tree, ""))

        success, visible_count, message, alignment_info = result
        assert success is True
        assert message == "Ready"
        mock_read_tree.show_all_reads.assert_called_once()

    @patch("squiggy.search.QMessageBox")
    def test_filter_by_region_no_bam_file(self, mock_qmsg, search_manager):
        """Test filtering by region without BAM file shows warning"""
        mock_read_tree = MagicMock()

        result = asyncio.run(
            search_manager.filter_by_region(None, mock_read_tree, "chr1:1000-2000")
        )

        success, visible_count, message, alignment_info = result
        assert success is False
        assert message == "BAM file required"
        mock_qmsg.warning.assert_called_once()

    @patch("squiggy.search.QMessageBox")
    def test_filter_by_region_invalid_format(self, mock_qmsg, search_manager, tmp_path):
        """Test filtering by region with invalid region format"""
        mock_read_tree = MagicMock()
        bam_file = tmp_path / "test.bam"

        result = asyncio.run(
            search_manager.filter_by_region(bam_file, mock_read_tree, "invalid::")
        )

        success, visible_count, message, alignment_info = result
        assert success is False
        assert message == "Invalid region format"
        mock_qmsg.warning.assert_called_once()

    @patch("squiggy.search.QMessageBox")
    def test_filter_by_region_valid(self, mock_qmsg, search_manager, indexed_bam_file):
        """Test filtering by region with valid reference name"""
        from squiggy.utils import get_bam_references

        mock_read_tree = MagicMock()
        mock_read_tree.filter_by_region.return_value = 5  # 5 visible reads

        # Get a real reference name from the BAM file
        references = get_bam_references(indexed_bam_file)
        if references:
            ref_name = references[0]["name"]
            result = asyncio.run(
                search_manager.filter_by_region(
                    indexed_bam_file, mock_read_tree, ref_name
                )
            )

            success, visible_count, message, alignment_info = result
            assert success is True
            assert visible_count == 5
            assert "5 reads" in message
        else:
            pytest.skip("No references in BAM file")

    @patch("squiggy.search.QMessageBox")
    def test_browse_references_no_bam(self, mock_qmsg, search_manager):
        """Test browsing references without BAM file"""
        result = asyncio.run(search_manager.browse_references(None))

        success, references = result
        assert success is False
        assert references is None
        mock_qmsg.warning.assert_called_once()

    def test_browse_references_valid(self, search_manager, indexed_bam_file):
        """Test browsing references with valid BAM file"""
        result = asyncio.run(search_manager.browse_references(indexed_bam_file))

        success, references = result
        assert success is True
        assert references is not None
        assert isinstance(references, list)

    def test_search_sequence_empty_query(self, search_manager):
        """Test sequence search with empty query"""
        result = asyncio.run(
            search_manager.search_sequence(None, "read_001", "", include_revcomp=True)
        )

        success, matches, message = result
        assert success is True
        assert matches == []
        assert message == "Ready"

    @patch("squiggy.search.QMessageBox")
    def test_search_sequence_no_bam(self, mock_qmsg, search_manager):
        """Test sequence search without BAM file"""
        result = asyncio.run(
            search_manager.search_sequence(
                None, "read_001", "ACGT", include_revcomp=True
            )
        )

        success, matches, message = result
        assert success is False
        assert matches == []
        assert message == "BAM file required"
        mock_qmsg.warning.assert_called_once()

    @patch("squiggy.search.QMessageBox")
    def test_search_sequence_invalid_bases(
        self, mock_qmsg, search_manager, indexed_bam_file
    ):
        """Test sequence search with invalid DNA bases"""
        result = asyncio.run(
            search_manager.search_sequence(
                indexed_bam_file, "read_001", "ACGTX", include_revcomp=True
            )
        )

        success, matches, message = result
        assert success is False
        assert matches == []
        assert message == "Invalid sequence"
        mock_qmsg.warning.assert_called_once()

    def test_search_sequence_valid_no_matches(
        self, search_manager, indexed_bam_file, sample_pod5_file
    ):
        """Test sequence search with valid sequence but no matches"""
        import pod5

        # Get a real read ID from the POD5 file
        with pod5.Reader(sample_pod5_file) as reader:
            first_read = next(reader.reads())
            read_id = str(first_read.read_id)

        # Search for a sequence that likely won't match
        result = asyncio.run(
            search_manager.search_sequence(
                indexed_bam_file, read_id, "AAAAAAAAAA", include_revcomp=False
            )
        )

        success, matches, message = result
        # Success should be True even with no matches
        assert success is True
        # Message should indicate no matches found
        assert "No matches" in message or "match" in message.lower()

    def test_search_sequence_with_revcomp(
        self, search_manager, indexed_bam_file, sample_pod5_file
    ):
        """Test sequence search with reverse complement enabled"""
        import pod5

        with pod5.Reader(sample_pod5_file) as reader:
            first_read = next(reader.reads())
            read_id = str(first_read.read_id)

        # Try searching for a short sequence
        result = asyncio.run(
            search_manager.search_sequence(
                indexed_bam_file, read_id, "ACGT", include_revcomp=True
            )
        )

        success, matches, message = result
        # Should complete successfully even if no matches
        assert success is True
        assert isinstance(matches, list)

    def test_search_sequence_without_revcomp(
        self, search_manager, indexed_bam_file, sample_pod5_file
    ):
        """Test sequence search with reverse complement disabled"""
        import pod5

        with pod5.Reader(sample_pod5_file) as reader:
            first_read = next(reader.reads())
            read_id = str(first_read.read_id)

        result = asyncio.run(
            search_manager.search_sequence(
                indexed_bam_file, read_id, "ACGT", include_revcomp=False
            )
        )

        success, matches, message = result
        assert success is True
        assert isinstance(matches, list)


class TestSearchSequenceInReference:
    """Tests for _search_sequence_in_reference method"""

    @pytest.fixture
    def search_manager(self):
        """Create SearchManager instance"""
        from squiggy.search import SearchManager

        return SearchManager(MagicMock())

    def test_search_sequence_in_reference_basic(
        self, search_manager, indexed_bam_file, sample_pod5_file
    ):
        """Test basic sequence search in reference"""
        import pod5

        with pod5.Reader(sample_pod5_file) as reader:
            first_read = next(reader.reads())
            read_id = str(first_read.read_id)

        # This is a blocking function, not async
        try:
            matches = search_manager._search_sequence_in_reference(
                indexed_bam_file, read_id, "ACGT", include_revcomp=False
            )
            assert isinstance(matches, list)
        except ValueError:
            # If the read doesn't have reference sequence, that's OK
            pass

    def test_search_sequence_finds_matches(
        self, search_manager, indexed_bam_file, sample_pod5_file
    ):
        """Test that search finds matches when sequence exists"""
        import pod5

        with pod5.Reader(sample_pod5_file) as reader:
            first_read = next(reader.reads())
            read_id = str(first_read.read_id)

        try:
            # Get the reference sequence first to find a real pattern
            from squiggy.utils import get_reference_sequence_for_read

            ref_seq, ref_start, aligned_read = get_reference_sequence_for_read(
                indexed_bam_file, read_id
            )

            if ref_seq and len(ref_seq) >= 4:
                # Search for a substring that definitely exists
                query = ref_seq[:4]
                matches = search_manager._search_sequence_in_reference(
                    indexed_bam_file, read_id, query, include_revcomp=False
                )

                # Should find at least one match (the one we took the substring from)
                assert len(matches) >= 1
                assert matches[0]["sequence"] == query
                assert matches[0]["strand"] == "Forward"
        except ValueError:
            # If no reference sequence, skip test
            pytest.skip("No reference sequence available for this read")

    def test_search_sequence_with_reverse_complement(
        self, search_manager, indexed_bam_file, sample_pod5_file
    ):
        """Test search with reverse complement enabled"""
        import pod5

        from squiggy.utils import reverse_complement

        with pod5.Reader(sample_pod5_file) as reader:
            first_read = next(reader.reads())
            read_id = str(first_read.read_id)

        try:
            from squiggy.utils import get_reference_sequence_for_read

            ref_seq, ref_start, aligned_read = get_reference_sequence_for_read(
                indexed_bam_file, read_id
            )

            if ref_seq and len(ref_seq) >= 4:
                # Search for a sequence
                query = "ACGT"
                matches = search_manager._search_sequence_in_reference(
                    indexed_bam_file, read_id, query, include_revcomp=True
                )

                # Matches can be on either strand
                for match in matches:
                    assert match["strand"] in ["Forward", "Reverse"]
                    if match["strand"] == "Reverse":
                        # Reverse matches should have the reverse complement sequence
                        assert match["sequence"] == reverse_complement(query)
        except ValueError:
            pytest.skip("No reference sequence available for this read")

    def test_search_sequence_returns_correct_positions(
        self, search_manager, indexed_bam_file, sample_pod5_file
    ):
        """Test that match positions are correct"""
        import pod5

        with pod5.Reader(sample_pod5_file) as reader:
            first_read = next(reader.reads())
            read_id = str(first_read.read_id)

        try:
            from squiggy.utils import get_reference_sequence_for_read

            ref_seq, ref_start, aligned_read = get_reference_sequence_for_read(
                indexed_bam_file, read_id
            )

            if ref_seq and len(ref_seq) >= 4:
                query = ref_seq[:4]
                matches = search_manager._search_sequence_in_reference(
                    indexed_bam_file, read_id, query, include_revcomp=False
                )

                for match in matches:
                    # Check required fields
                    assert "ref_start" in match
                    assert "ref_end" in match
                    assert "base_start" in match
                    assert "base_end" in match
                    assert "sequence" in match
                    assert "strand" in match

                    # Check position logic
                    assert match["ref_end"] > match["ref_start"]
                    assert match["base_end"] > match["base_start"]
                    assert match["ref_end"] - match["ref_start"] == len(query)
        except ValueError:
            pytest.skip("No reference sequence available for this read")
