"""Tests for cache module"""

import pickle
import time
from pathlib import Path


class TestSquiggyCacheInitialization:
    """Tests for SquiggyCache initialization"""

    def test_init_default_cache_dir(self):
        """Test initialization with default cache directory"""
        from squiggy.cache import SquiggyCache

        cache = SquiggyCache()

        assert cache.enabled is True
        assert cache.cache_dir == Path.home() / ".squiggy" / "cache"
        assert cache.cache_dir.exists()

    def test_init_custom_cache_dir(self, tmp_path):
        """Test initialization with custom cache directory"""
        from squiggy.cache import SquiggyCache

        custom_dir = tmp_path / "custom_cache"
        cache = SquiggyCache(cache_dir=custom_dir)

        assert cache.cache_dir == custom_dir
        assert custom_dir.exists()

    def test_init_disabled_cache(self, tmp_path):
        """Test initialization with caching disabled"""
        from squiggy.cache import SquiggyCache

        custom_dir = tmp_path / "disabled_cache"
        cache = SquiggyCache(cache_dir=custom_dir, enabled=False)

        assert cache.enabled is False
        # Directory should not be created when disabled
        assert not custom_dir.exists()

    def test_cache_dir_created_recursively(self, tmp_path):
        """Test that cache directory is created with parent directories"""
        from squiggy.cache import SquiggyCache

        deep_dir = tmp_path / "level1" / "level2" / "cache"
        SquiggyCache(cache_dir=deep_dir)

        assert deep_dir.exists()


class TestGetCachePath:
    """Tests for _get_cache_path method"""

    def test_get_cache_path_pod5(self, tmp_path):
        """Test cache path generation for POD5 files"""
        from squiggy.cache import SquiggyCache

        cache = SquiggyCache(cache_dir=tmp_path)
        file_path = Path("/data/reads.pod5")

        cache_path = cache._get_cache_path(file_path, ".pod5.cache")

        assert cache_path.parent == tmp_path
        assert cache_path.name.startswith("reads_")
        assert cache_path.name.endswith(".pod5.cache")

    def test_get_cache_path_bam(self, tmp_path):
        """Test cache path generation for BAM files"""
        from squiggy.cache import SquiggyCache

        cache = SquiggyCache(cache_dir=tmp_path)
        file_path = Path("/data/alignments.bam")

        cache_path = cache._get_cache_path(file_path, ".bam.cache")

        assert cache_path.parent == tmp_path
        assert cache_path.name.startswith("alignments_")
        assert cache_path.name.endswith(".bam.cache")

    def test_get_cache_path_collision_avoidance(self, tmp_path):
        """Test that different paths with same filename get different cache paths"""
        from squiggy.cache import SquiggyCache

        cache = SquiggyCache(cache_dir=tmp_path)
        file1 = Path("/data1/reads.pod5")
        file2 = Path("/data2/reads.pod5")

        cache1 = cache._get_cache_path(file1, ".pod5.cache")
        cache2 = cache._get_cache_path(file2, ".pod5.cache")

        # Should have different hashes despite same filename
        assert cache1 != cache2


class TestFileHash:
    """Tests for _file_hash method"""

    def test_file_hash_consistency(self, tmp_path):
        """Test that file hash is consistent for same content"""
        from squiggy.cache import SquiggyCache

        cache = SquiggyCache(cache_dir=tmp_path)
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        hash1 = cache._file_hash(test_file)
        hash2 = cache._file_hash(test_file)

        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 32  # MD5 hex digest length

    def test_file_hash_detects_changes(self, tmp_path):
        """Test that file hash changes when content changes"""
        from squiggy.cache import SquiggyCache

        cache = SquiggyCache(cache_dir=tmp_path)
        test_file = tmp_path / "test.txt"

        test_file.write_text("original content")
        hash1 = cache._file_hash(test_file)

        test_file.write_text("modified content")
        hash2 = cache._file_hash(test_file)

        assert hash1 != hash2

    def test_file_hash_custom_chunk_size(self, tmp_path):
        """Test file hash with custom chunk size"""
        from squiggy.cache import SquiggyCache

        cache = SquiggyCache(cache_dir=tmp_path)
        test_file = tmp_path / "test.txt"
        test_file.write_text("x" * 1000)

        hash1 = cache._file_hash(test_file, chunk_size=100)
        hash2 = cache._file_hash(test_file, chunk_size=500)

        # Different chunk sizes produce different hashes since they read
        # different amounts of the file
        assert hash1 != hash2
        assert isinstance(hash1, str)
        assert isinstance(hash2, str)


class TestPOD5IndexCache:
    """Tests for POD5 index caching"""

    def test_save_and_load_pod5_index(self, tmp_path, sample_pod5_file):
        """Test saving and loading POD5 index"""
        from squiggy.cache import SquiggyCache

        cache = SquiggyCache(cache_dir=tmp_path)
        test_index = {"read1": 0, "read2": 100, "read3": 200}

        # Save index
        cache.save_pod5_index(sample_pod5_file, test_index)

        # Load index
        loaded = cache.load_pod5_index(sample_pod5_file)

        assert loaded == test_index

    def test_load_pod5_index_cache_miss(self, tmp_path):
        """Test loading when no cache exists"""
        from squiggy.cache import SquiggyCache

        cache = SquiggyCache(cache_dir=tmp_path)
        fake_file = Path("/nonexistent/file.pod5")

        result = cache.load_pod5_index(fake_file)

        assert result is None

    def test_load_pod5_index_disabled_cache(self, tmp_path, sample_pod5_file):
        """Test loading with cache disabled"""
        from squiggy.cache import SquiggyCache

        cache = SquiggyCache(cache_dir=tmp_path, enabled=False)

        result = cache.load_pod5_index(sample_pod5_file)

        assert result is None

    def test_save_pod5_index_disabled_cache(self, tmp_path, sample_pod5_file):
        """Test saving with cache disabled does nothing"""
        from squiggy.cache import SquiggyCache

        cache = SquiggyCache(cache_dir=tmp_path, enabled=False)
        test_index = {"read1": 0}

        cache.save_pod5_index(sample_pod5_file, test_index)

        # Should not create any cache files
        assert len(list(tmp_path.glob("*.cache"))) == 0

    def test_pod5_index_invalidation_on_file_change(self, tmp_path):
        """Test that cache is invalidated when file changes"""
        from squiggy.cache import SquiggyCache

        cache = SquiggyCache(cache_dir=tmp_path)

        # Create a test file
        test_file = tmp_path / "test.pod5"
        test_file.write_bytes(b"original content")

        # Save index
        test_index = {"read1": 0}
        cache.save_pod5_index(test_file, test_index)

        # Verify it loads
        loaded = cache.load_pod5_index(test_file)
        assert loaded == test_index

        # Modify the file
        test_file.write_bytes(b"modified content")

        # Should return None due to hash mismatch
        loaded = cache.load_pod5_index(test_file)
        assert loaded is None

    def test_pod5_index_with_num_reads(self, tmp_path, sample_pod5_file):
        """Test saving index with explicit num_reads"""
        from squiggy.cache import SquiggyCache

        cache = SquiggyCache(cache_dir=tmp_path)
        test_index = {"read1": 0, "read2": 100}

        cache.save_pod5_index(sample_pod5_file, test_index, num_reads=2)

        # Should still load correctly
        loaded = cache.load_pod5_index(sample_pod5_file)
        assert loaded == test_index


class TestPOD5ReadIDsCache:
    """Tests for POD5 read IDs caching"""

    def test_save_and_load_pod5_read_ids(self, tmp_path, sample_pod5_file):
        """Test saving and loading read IDs"""
        from squiggy.cache import SquiggyCache

        cache = SquiggyCache(cache_dir=tmp_path)
        test_ids = ["read1", "read2", "read3"]

        # Save
        cache.save_pod5_read_ids(sample_pod5_file, test_ids)

        # Load
        loaded = cache.load_pod5_read_ids(sample_pod5_file)

        assert loaded == test_ids

    def test_load_pod5_read_ids_cache_miss(self, tmp_path):
        """Test loading when no cache exists"""
        from squiggy.cache import SquiggyCache

        cache = SquiggyCache(cache_dir=tmp_path)
        fake_file = Path("/nonexistent/file.pod5")

        result = cache.load_pod5_read_ids(fake_file)

        assert result is None

    def test_load_pod5_read_ids_disabled_cache(self, tmp_path, sample_pod5_file):
        """Test loading with cache disabled"""
        from squiggy.cache import SquiggyCache

        cache = SquiggyCache(cache_dir=tmp_path, enabled=False)

        result = cache.load_pod5_read_ids(sample_pod5_file)

        assert result is None

    def test_save_pod5_read_ids_disabled_cache(self, tmp_path, sample_pod5_file):
        """Test saving with cache disabled does nothing"""
        from squiggy.cache import SquiggyCache

        cache = SquiggyCache(cache_dir=tmp_path, enabled=False)
        test_ids = ["read1"]

        cache.save_pod5_read_ids(sample_pod5_file, test_ids)

        # Should not create any cache files
        assert len(list(tmp_path.glob("*.cache"))) == 0

    def test_pod5_read_ids_invalidation_on_file_change(self, tmp_path):
        """Test that cache is invalidated when file changes"""
        from squiggy.cache import SquiggyCache

        cache = SquiggyCache(cache_dir=tmp_path)

        # Create a test file
        test_file = tmp_path / "test.pod5"
        test_file.write_bytes(b"original content")

        # Save read IDs
        test_ids = ["read1", "read2"]
        cache.save_pod5_read_ids(test_file, test_ids)

        # Verify it loads
        loaded = cache.load_pod5_read_ids(test_file)
        assert loaded == test_ids

        # Modify the file
        test_file.write_bytes(b"modified content")

        # Should return None due to hash mismatch
        loaded = cache.load_pod5_read_ids(test_file)
        assert loaded is None


class TestBAMRefMappingCache:
    """Tests for BAM reference mapping caching"""

    def test_save_and_load_bam_ref_mapping(self, tmp_path, sample_bam_file):
        """Test saving and loading BAM reference mapping"""
        from squiggy.cache import SquiggyCache

        cache = SquiggyCache(cache_dir=tmp_path)
        test_mapping = {
            "chr1": ["read1", "read2"],
            "chr2": ["read3", "read4", "read5"],
        }

        # Save
        cache.save_bam_ref_mapping(sample_bam_file, test_mapping)

        # Load
        loaded = cache.load_bam_ref_mapping(sample_bam_file)

        assert loaded == test_mapping

    def test_load_bam_ref_mapping_cache_miss(self, tmp_path):
        """Test loading when no cache exists"""
        from squiggy.cache import SquiggyCache

        cache = SquiggyCache(cache_dir=tmp_path)
        fake_file = Path("/nonexistent/file.bam")

        result = cache.load_bam_ref_mapping(fake_file)

        assert result is None

    def test_load_bam_ref_mapping_disabled_cache(self, tmp_path, sample_bam_file):
        """Test loading with cache disabled"""
        from squiggy.cache import SquiggyCache

        cache = SquiggyCache(cache_dir=tmp_path, enabled=False)

        result = cache.load_bam_ref_mapping(sample_bam_file)

        assert result is None

    def test_save_bam_ref_mapping_disabled_cache(self, tmp_path, sample_bam_file):
        """Test saving with cache disabled does nothing"""
        from squiggy.cache import SquiggyCache

        cache = SquiggyCache(cache_dir=tmp_path, enabled=False)
        test_mapping = {"chr1": ["read1"]}

        cache.save_bam_ref_mapping(sample_bam_file, test_mapping)

        # Should not create any cache files
        assert len(list(tmp_path.glob("*.cache"))) == 0

    def test_bam_ref_mapping_invalidation_on_file_change(self, tmp_path):
        """Test that cache is invalidated when file changes"""
        from squiggy.cache import SquiggyCache

        cache = SquiggyCache(cache_dir=tmp_path)

        # Create a test file
        test_file = tmp_path / "test.bam"
        test_file.write_bytes(b"original content")

        # Save mapping
        test_mapping = {"chr1": ["read1"]}
        cache.save_bam_ref_mapping(test_file, test_mapping)

        # Verify it loads
        loaded = cache.load_bam_ref_mapping(test_file)
        assert loaded == test_mapping

        # Modify the file
        test_file.write_bytes(b"modified content")

        # Should return None due to hash mismatch
        loaded = cache.load_bam_ref_mapping(test_file)
        assert loaded is None


class TestBAMMetadataCache:
    """Tests for BAM metadata caching"""

    def test_save_and_load_bam_metadata(self, tmp_path, sample_bam_file):
        """Test saving and loading complete BAM metadata"""
        from squiggy.cache import SquiggyCache

        cache = SquiggyCache(cache_dir=tmp_path)
        test_metadata = {
            "references": [
                {"name": "chr1", "length": 1000, "read_count": 5},
                {"name": "chr2", "length": 2000, "read_count": 10},
            ],
            "has_modifications": True,
            "modification_types": ["5mC", "6mA"],
            "has_probabilities": True,
            "has_event_alignment": False,
            "ref_mapping": {"chr1": ["read1"], "chr2": ["read2"]},
            "num_reads": 15,
        }

        # Save
        cache.save_bam_metadata(sample_bam_file, test_metadata)

        # Load
        loaded = cache.load_bam_metadata(sample_bam_file)

        assert loaded == test_metadata
        assert loaded["num_reads"] == 15
        assert len(loaded["references"]) == 2

    def test_load_bam_metadata_cache_miss(self, tmp_path):
        """Test loading when no cache exists"""
        from squiggy.cache import SquiggyCache

        cache = SquiggyCache(cache_dir=tmp_path)
        fake_file = Path("/nonexistent/file.bam")

        result = cache.load_bam_metadata(fake_file)

        assert result is None

    def test_load_bam_metadata_disabled_cache(self, tmp_path, sample_bam_file):
        """Test loading with cache disabled"""
        from squiggy.cache import SquiggyCache

        cache = SquiggyCache(cache_dir=tmp_path, enabled=False)

        result = cache.load_bam_metadata(sample_bam_file)

        assert result is None

    def test_save_bam_metadata_disabled_cache(self, tmp_path, sample_bam_file):
        """Test saving with cache disabled does nothing"""
        from squiggy.cache import SquiggyCache

        cache = SquiggyCache(cache_dir=tmp_path, enabled=False)
        test_metadata = {"num_reads": 10, "references": []}

        cache.save_bam_metadata(sample_bam_file, test_metadata)

        # Should not create any cache files
        assert len(list(tmp_path.glob("*.cache"))) == 0

    def test_bam_metadata_invalidation_on_file_change(self, tmp_path):
        """Test that cache is invalidated when file modified"""
        from squiggy.cache import SquiggyCache

        cache = SquiggyCache(cache_dir=tmp_path)

        # Create a test file
        test_file = tmp_path / "test.bam"
        test_file.write_bytes(b"original content")

        # Save metadata
        test_metadata = {"num_reads": 5, "references": []}
        cache.save_bam_metadata(test_file, test_metadata)

        # Verify it loads
        loaded = cache.load_bam_metadata(test_file)
        assert loaded == test_metadata

        # Wait a bit to ensure mtime changes
        time.sleep(0.01)

        # Modify the file
        test_file.write_bytes(b"modified content")

        # Should return None due to mtime mismatch
        loaded = cache.load_bam_metadata(test_file)
        assert loaded is None


class TestClearCache:
    """Tests for clear_cache method"""

    def test_clear_cache_removes_all_files(self, tmp_path, sample_pod5_file):
        """Test that clear_cache removes all cache files"""
        from squiggy.cache import SquiggyCache

        cache = SquiggyCache(cache_dir=tmp_path)

        # Create multiple cache files
        cache.save_pod5_index(sample_pod5_file, {"read1": 0})
        cache.save_pod5_read_ids(sample_pod5_file, ["read1"])

        # Verify files exist
        assert len(list(tmp_path.glob("*.cache"))) >= 2

        # Clear cache
        count = cache.clear_cache()

        # Should have removed files
        assert count >= 2
        assert len(list(tmp_path.glob("*.cache"))) == 0

    def test_clear_cache_empty_cache(self, tmp_path):
        """Test clear_cache with no cache files"""
        from squiggy.cache import SquiggyCache

        cache = SquiggyCache(cache_dir=tmp_path)

        count = cache.clear_cache()

        assert count == 0

    def test_clear_cache_disabled(self, tmp_path):
        """Test clear_cache with caching disabled"""
        from squiggy.cache import SquiggyCache

        cache = SquiggyCache(cache_dir=tmp_path, enabled=False)

        count = cache.clear_cache()

        assert count == 0

    def test_clear_cache_nonexistent_dir(self, tmp_path):
        """Test clear_cache when cache directory doesn't exist"""
        from squiggy.cache import SquiggyCache

        nonexistent = tmp_path / "nonexistent"
        cache = SquiggyCache(cache_dir=nonexistent, enabled=False)

        count = cache.clear_cache()

        assert count == 0


class TestCacheCorruption:
    """Tests for handling corrupted cache files"""

    def test_load_pod5_index_corrupted_pickle(self, tmp_path, sample_pod5_file):
        """Test loading POD5 index with corrupted cache file"""
        from squiggy.cache import SquiggyCache

        cache = SquiggyCache(cache_dir=tmp_path)

        # Create a corrupted cache file
        cache_path = cache._get_cache_path(sample_pod5_file, ".pod5.cache")
        cache_path.write_bytes(b"corrupted data")

        # Should return None on corruption
        result = cache.load_pod5_index(sample_pod5_file)

        assert result is None

    def test_load_pod5_read_ids_missing_keys(self, tmp_path, sample_pod5_file):
        """Test loading read IDs with malformed cache data"""
        from squiggy.cache import SquiggyCache

        cache = SquiggyCache(cache_dir=tmp_path)

        # Create cache with missing keys
        cache_path = cache._get_cache_path(sample_pod5_file, ".pod5.ids.cache")
        with open(cache_path, "wb") as f:
            pickle.dump({"wrong_key": "value"}, f)

        # Should return None when keys are missing
        result = cache.load_pod5_read_ids(sample_pod5_file)

        assert result is None

    def test_load_bam_ref_mapping_corrupted(self, tmp_path, sample_bam_file):
        """Test loading BAM mapping with corrupted data"""
        from squiggy.cache import SquiggyCache

        cache = SquiggyCache(cache_dir=tmp_path)

        # Create corrupted cache
        cache_path = cache._get_cache_path(sample_bam_file, ".bam.cache")
        cache_path.write_text("not a pickle file")

        # Should return None
        result = cache.load_bam_ref_mapping(sample_bam_file)

        assert result is None


class TestCacheIntegration:
    """Integration tests for cache workflow"""

    def test_full_pod5_workflow(self, tmp_path, sample_pod5_file):
        """Test complete POD5 caching workflow"""
        from squiggy.cache import SquiggyCache

        cache = SquiggyCache(cache_dir=tmp_path)

        # Initial load should be cache miss
        index = cache.load_pod5_index(sample_pod5_file)
        assert index is None

        # Save index
        test_index = {"read1": 0, "read2": 100}
        cache.save_pod5_index(sample_pod5_file, test_index)

        # Second load should hit cache
        index = cache.load_pod5_index(sample_pod5_file)
        assert index == test_index

        # Clear and verify
        count = cache.clear_cache()
        assert count >= 1

        # After clear, should be cache miss again
        index = cache.load_pod5_index(sample_pod5_file)
        assert index is None

    def test_full_bam_workflow(self, tmp_path, sample_bam_file):
        """Test complete BAM caching workflow"""
        from squiggy.cache import SquiggyCache

        cache = SquiggyCache(cache_dir=tmp_path)

        # Initial load should be cache miss
        metadata = cache.load_bam_metadata(sample_bam_file)
        assert metadata is None

        # Save metadata
        test_metadata = {
            "references": [],
            "num_reads": 10,
            "has_modifications": False,
        }
        cache.save_bam_metadata(sample_bam_file, test_metadata)

        # Second load should hit cache
        metadata = cache.load_bam_metadata(sample_bam_file)
        assert metadata == test_metadata

        # Clear and verify
        count = cache.clear_cache()
        assert count >= 1

    def test_multiple_files_in_cache(self, tmp_path, sample_pod5_file, sample_bam_file):
        """Test caching multiple different files"""
        from squiggy.cache import SquiggyCache

        cache = SquiggyCache(cache_dir=tmp_path)

        # Cache POD5 data
        pod5_index = {"read1": 0}
        cache.save_pod5_index(sample_pod5_file, pod5_index)

        # Cache BAM data
        bam_mapping = {"chr1": ["read1"]}
        cache.save_bam_ref_mapping(sample_bam_file, bam_mapping)

        # Both should load independently
        loaded_pod5 = cache.load_pod5_index(sample_pod5_file)
        loaded_bam = cache.load_bam_ref_mapping(sample_bam_file)

        assert loaded_pod5 == pod5_index
        assert loaded_bam == bam_mapping

        # Clear should remove both
        count = cache.clear_cache()
        assert count >= 2
