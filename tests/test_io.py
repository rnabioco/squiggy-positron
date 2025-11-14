"""Tests for I/O functions (load_pod5, load_bam, etc.)"""

from pathlib import Path

import pytest


class TestLoadPOD5:
    """Tests for load_pod5 function"""

    def test_load_pod5_returns_reader_and_ids(self, sample_pod5_file):
        """Test that load_pod5 populates global session with reader and read IDs"""
        from squiggy import load_pod5
        from squiggy.io import squiggy_kernel

        result = load_pod5(str(sample_pod5_file))

        # Should return None (void function)
        assert result is None

        # Should populate global session
        assert squiggy_kernel._reader is not None
        # read_ids can be either list or LazyReadList (optimized)
        from squiggy.io import LazyReadList

        assert isinstance(squiggy_kernel._read_ids, (list, LazyReadList))
        assert len(squiggy_kernel._read_ids) > 0
        assert all(isinstance(rid, str) for rid in squiggy_kernel._read_ids)

    def test_load_pod5_stores_global_state(self, sample_pod5_file):
        """Test that load_pod5 stores global state"""
        from squiggy import get_current_files, get_read_ids, load_pod5

        load_pod5(str(sample_pod5_file))

        # Check current files
        current_files = get_current_files()
        assert current_files["pod5_path"] is not None
        assert str(sample_pod5_file) in current_files["pod5_path"]

        # Check read IDs are accessible
        read_ids = get_read_ids()
        assert len(read_ids) > 0

    def test_load_pod5_converts_to_absolute_path(self, sample_pod5_file):
        """Test that load_pod5 converts relative paths to absolute"""
        import os

        from squiggy import get_current_files, load_pod5

        # Get relative path
        original_dir = os.getcwd()
        try:
            os.chdir(sample_pod5_file.parent)
            rel_path = sample_pod5_file.name

            load_pod5(rel_path)

            current_files = get_current_files()
            # Should be absolute path
            assert Path(current_files["pod5_path"]).is_absolute()
        finally:
            os.chdir(original_dir)

    def test_load_pod5_nonexistent_file(self):
        """Test that load_pod5 raises error for nonexistent file"""
        from squiggy import load_pod5

        with pytest.raises(FileNotFoundError):
            load_pod5("/nonexistent/path/file.pod5")

    def test_load_pod5_closes_previous_reader(self, sample_pod5_file):
        """Test that loading a new file closes the previous reader"""
        from squiggy import load_pod5
        from squiggy.io import squiggy_kernel

        # Load first time
        load_pod5(str(sample_pod5_file))
        assert squiggy_kernel._reader is not None

        # Load second time
        load_pod5(str(sample_pod5_file))

        # Should still have a reader
        assert squiggy_kernel._reader is not None
        # Previous reader should be closed (we can't easily test this, but no errors should occur)


class TestLoadBAM:
    """Tests for load_bam function"""

    def test_load_bam_returns_metadata(self, indexed_bam_file):
        """Test that load_bam populates global session with BAM metadata"""
        from squiggy import load_bam
        from squiggy.io import squiggy_kernel

        result = load_bam(str(indexed_bam_file))

        # Should return None (void function)
        assert result is None

        # Should populate global session
        assert squiggy_kernel._bam_info is not None
        assert isinstance(squiggy_kernel._bam_info, dict)
        assert "file_path" in squiggy_kernel._bam_info
        assert "num_reads" in squiggy_kernel._bam_info
        assert "references" in squiggy_kernel._bam_info
        assert "has_modifications" in squiggy_kernel._bam_info
        assert "modification_types" in squiggy_kernel._bam_info
        assert "has_event_alignment" in squiggy_kernel._bam_info

    def test_load_bam_stores_global_state(self, indexed_bam_file):
        """Test that load_bam stores global path"""
        from squiggy import get_current_files, load_bam

        load_bam(str(indexed_bam_file))

        current_files = get_current_files()
        assert current_files["bam_path"] is not None
        assert str(indexed_bam_file) in current_files["bam_path"]

    def test_load_bam_nonexistent_file(self):
        """Test that load_bam raises error for nonexistent file"""
        from squiggy import load_bam

        with pytest.raises(FileNotFoundError):
            load_bam("/nonexistent/path/file.bam")

    def test_load_bam_references_structure(self, indexed_bam_file):
        """Test that references have expected structure"""
        from squiggy import load_bam
        from squiggy.io import squiggy_kernel

        load_bam(str(indexed_bam_file))
        references = squiggy_kernel._bam_info["references"]

        assert isinstance(references, list)
        if len(references) > 0:
            ref = references[0]
            assert "name" in ref
            assert "length" in ref
            assert "read_count" in ref


class TestGetBAMModificationInfo:
    """Tests for get_bam_modification_info function"""

    def test_get_bam_modification_info_returns_dict(self, indexed_bam_file):
        """Test that function returns properly structured dict"""
        from squiggy.io import get_bam_modification_info

        mod_info = get_bam_modification_info(str(indexed_bam_file))

        assert isinstance(mod_info, dict)
        assert "has_modifications" in mod_info
        assert "modification_types" in mod_info
        assert "sample_count" in mod_info
        assert "has_probabilities" in mod_info

        assert isinstance(mod_info["has_modifications"], bool)
        assert isinstance(mod_info["modification_types"], list)
        assert isinstance(mod_info["sample_count"], int)
        assert isinstance(mod_info["has_probabilities"], bool)

    def test_get_bam_modification_info_nonexistent_file(self):
        """Test that function raises error for nonexistent file"""
        from squiggy.io import get_bam_modification_info

        with pytest.raises(FileNotFoundError):
            get_bam_modification_info("/nonexistent/path/file.bam")


class TestGetBAMEventAlignmentStatus:
    """Tests for get_bam_event_alignment_status function"""

    def test_get_bam_event_alignment_status_returns_bool(self, indexed_bam_file):
        """Test that function returns boolean"""
        from squiggy.io import get_bam_event_alignment_status

        has_events = get_bam_event_alignment_status(str(indexed_bam_file))

        assert isinstance(has_events, bool)

    def test_get_bam_event_alignment_status_nonexistent_file(self):
        """Test that function raises error for nonexistent file"""
        from squiggy.io import get_bam_event_alignment_status

        with pytest.raises(FileNotFoundError):
            get_bam_event_alignment_status("/nonexistent/path/file.bam")


class TestGetReadToReferenceMapping:
    """Tests for get_read_to_reference_mapping function"""

    def test_get_mapping_requires_loaded_bam(self, sample_pod5_file):
        """Test that function requires BAM to be loaded"""
        from squiggy import close_pod5, load_pod5
        from squiggy.io import get_read_to_reference_mapping

        # Close any existing files
        close_pod5()

        # Load only POD5
        load_pod5(str(sample_pod5_file))

        # Should raise error since no BAM loaded
        with pytest.raises(RuntimeError, match="No BAM file"):
            get_read_to_reference_mapping()

    def test_get_mapping_returns_dict(self, sample_pod5_file, indexed_bam_file):
        """Test that function returns reference to read mapping"""
        from squiggy import load_bam, load_pod5
        from squiggy.io import get_read_to_reference_mapping

        load_pod5(str(sample_pod5_file))
        load_bam(str(indexed_bam_file))

        mapping = get_read_to_reference_mapping()

        assert isinstance(mapping, dict)
        # Keys should be reference names (strings)
        for ref_name, read_ids in mapping.items():
            assert isinstance(ref_name, str)
            assert isinstance(read_ids, list)
            assert all(isinstance(rid, str) for rid in read_ids)


class TestGetCurrentFiles:
    """Tests for get_current_files function"""

    def test_get_current_files_initially_none(self):
        """Test that get_current_files returns None when nothing loaded"""
        from squiggy import close_pod5, get_current_files

        close_pod5()

        current = get_current_files()
        assert current["pod5_path"] is None
        assert current["bam_path"] is None

    def test_get_current_files_after_loading(self, sample_pod5_file, indexed_bam_file):
        """Test that get_current_files returns paths after loading"""
        from squiggy import get_current_files, load_bam, load_pod5

        load_pod5(str(sample_pod5_file))
        load_bam(str(indexed_bam_file))

        current = get_current_files()
        assert current["pod5_path"] is not None
        assert current["bam_path"] is not None
        assert str(sample_pod5_file) in current["pod5_path"]
        assert str(indexed_bam_file) in current["bam_path"]


class TestGetReadIDs:
    """Tests for get_read_ids function"""

    def test_get_read_ids_requires_loaded_pod5(self):
        """Test that function requires POD5 to be loaded"""
        from squiggy import close_pod5, get_read_ids

        close_pod5()

        with pytest.raises(ValueError, match="No POD5 file"):
            get_read_ids()

    def test_get_read_ids_returns_list(self, sample_pod5_file):
        """Test that function returns list of read IDs"""
        from squiggy import get_read_ids, load_pod5

        load_pod5(str(sample_pod5_file))

        read_ids = get_read_ids()
        assert isinstance(read_ids, list)
        assert len(read_ids) > 0
        assert all(isinstance(rid, str) for rid in read_ids)


class TestClosePOD5:
    """Tests for close_pod5 function"""

    def test_close_pod5_clears_state(self, sample_pod5_file):
        """Test that close_pod5 clears global state"""
        from squiggy import close_pod5, get_current_files, load_pod5

        # Load file
        load_pod5(str(sample_pod5_file))

        # Verify it's loaded
        current = get_current_files()
        assert current["pod5_path"] is not None

        # Close
        close_pod5()

        # Verify it's cleared
        current = get_current_files()
        assert current["pod5_path"] is None

    def test_close_pod5_when_none_loaded(self):
        """Test that close_pod5 handles case when nothing is loaded"""
        from squiggy import close_pod5

        # Should not raise error
        close_pod5()


class TestCloseBAM:
    """Tests for close_bam function"""

    def test_close_bam_clears_state(self, indexed_bam_file):
        """Test that close_bam clears global BAM state"""
        from squiggy import close_bam, get_current_files, load_bam

        # Load file
        load_bam(str(indexed_bam_file))

        # Verify it's loaded
        current = get_current_files()
        assert current["bam_path"] is not None

        # Close
        close_bam()

        # Verify it's cleared
        current = get_current_files()
        assert current["bam_path"] is None

    def test_close_bam_when_none_loaded(self):
        """Test that close_bam handles case when nothing is loaded"""
        from squiggy import close_bam

        # Should not raise error
        close_bam()


class TestSquiggyKernel:
    """Tests for SquiggyKernel class"""

    def test_session_repr_no_files(self):
        """Test that session repr shows no files loaded"""
        from squiggy.io import SquiggyKernel

        session = SquiggyKernel()
        repr_str = repr(session)

        assert "No files loaded" in repr_str

    def test_session_repr_pod5_only(self, sample_pod5_file):
        """Test that session repr shows POD5 file info"""
        from squiggy import load_pod5
        from squiggy.io import squiggy_kernel

        load_pod5(str(sample_pod5_file))

        repr_str = repr(squiggy_kernel)

        assert "POD5:" in repr_str
        assert sample_pod5_file.name in repr_str
        assert "reads" in repr_str

    def test_session_repr_both_files(self, sample_pod5_file, indexed_bam_file):
        """Test that session repr shows both POD5 and BAM info"""
        from squiggy import load_bam, load_pod5
        from squiggy.io import squiggy_kernel

        load_pod5(str(sample_pod5_file))
        load_bam(str(indexed_bam_file))

        repr_str = repr(squiggy_kernel)

        assert "POD5:" in repr_str
        assert "BAM:" in repr_str
        assert sample_pod5_file.name in repr_str
        assert indexed_bam_file.name in repr_str

    def test_session_stores_pod5_data(self, sample_pod5_file):
        """Test that session stores POD5 data correctly"""
        from squiggy import load_pod5
        from squiggy.io import squiggy_kernel

        load_pod5(str(sample_pod5_file))

        # Check session was populated
        assert squiggy_kernel._reader is not None
        assert squiggy_kernel._pod5_path is not None
        assert len(squiggy_kernel._read_ids) > 0

    def test_session_stores_bam_data(self, indexed_bam_file):
        """Test that session stores BAM data correctly"""
        from squiggy import load_bam
        from squiggy.io import get_read_to_reference_mapping, squiggy_kernel

        load_bam(str(indexed_bam_file))
        mapping = get_read_to_reference_mapping()

        # Check session was populated
        assert squiggy_kernel._bam_path is not None
        assert squiggy_kernel._bam_info is not None
        assert squiggy_kernel._ref_mapping is not None
        assert squiggy_kernel._ref_mapping == mapping

    def test_session_close_pod5(self, sample_pod5_file):
        """Test that session.close_pod5() clears POD5 state"""
        from squiggy import load_pod5
        from squiggy.io import squiggy_kernel

        load_pod5(str(sample_pod5_file))

        # Verify loaded
        assert squiggy_kernel._reader is not None
        assert squiggy_kernel._pod5_path is not None
        assert len(squiggy_kernel._read_ids) > 0

        # Close
        squiggy_kernel.close_pod5()

        # Verify cleared
        assert squiggy_kernel._reader is None
        assert squiggy_kernel._pod5_path is None
        assert len(squiggy_kernel._read_ids) == 0

    def test_session_close_bam(self, indexed_bam_file):
        """Test that session.close_bam() clears BAM state"""
        from squiggy import load_bam
        from squiggy.io import get_read_to_reference_mapping, squiggy_kernel

        load_bam(str(indexed_bam_file))
        get_read_to_reference_mapping()

        # Verify loaded
        assert squiggy_kernel._bam_path is not None
        assert squiggy_kernel._bam_info is not None
        assert squiggy_kernel._ref_mapping is not None

        # Close
        squiggy_kernel.close_bam()

        # Verify cleared
        assert squiggy_kernel._bam_path is None
        assert squiggy_kernel._bam_info is None
        assert squiggy_kernel._ref_mapping is None

    def test_session_close_all(self, sample_pod5_file, indexed_bam_file):
        """Test that session.close_all() clears all state"""
        from squiggy import load_bam, load_pod5
        from squiggy.io import get_read_to_reference_mapping, squiggy_kernel

        load_pod5(str(sample_pod5_file))
        load_bam(str(indexed_bam_file))
        get_read_to_reference_mapping()

        # Verify loaded
        assert squiggy_kernel._reader is not None
        assert squiggy_kernel._bam_path is not None

        # Close all
        squiggy_kernel.close_all()

        # Verify all cleared
        assert squiggy_kernel._reader is None
        assert squiggy_kernel._pod5_path is None
        assert len(squiggy_kernel._read_ids) == 0
        assert squiggy_kernel._bam_path is None
        assert squiggy_kernel._bam_info is None
        assert squiggy_kernel._ref_mapping is None
