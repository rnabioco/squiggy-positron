"""
Persistent caching layer for POD5 and BAM metadata

Provides pickle-based caching for read indices and BAM reference mappings,
enabling instant subsequent loads of large files.
"""

import hashlib
import pickle
from datetime import datetime
from pathlib import Path


class SquiggyCache:
    """
    Pickle-based persistent cache for POD5/BAM metadata

    Caches expensive-to-compute data structures like read ID indices and BAM
    reference mappings. Uses file hashing for cache invalidation.

    Attributes:
        cache_dir: Directory to store cache files (default: ~/.squiggy/cache)
        enabled: Whether caching is enabled

    Examples:
        >>> cache = SquiggyCache()
        >>> # Try to load cached index
        >>> index = cache.load_pod5_index(Path('file.pod5'))
        >>> if index is None:
        ...     # Build index fresh
        ...     index = build_index()
        ...     cache.save_pod5_index(Path('file.pod5'), index)
    """

    def __init__(self, cache_dir: Path | None = None, enabled: bool = True):
        """
        Initialize cache manager

        Args:
            cache_dir: Directory for cache files (default: ~/.squiggy/cache)
            enabled: Whether caching is enabled
        """
        self.cache_dir = cache_dir or Path.home() / ".squiggy" / "cache"
        self.enabled = enabled

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        """Return informative summary of cache state"""
        if not self.enabled:
            return "<SquiggyCache: disabled>"

        # Count cache files if directory exists
        num_files = 0
        if self.cache_dir.exists():
            num_files = len(list(self.cache_dir.glob("*.cache")))

        return f"<SquiggyCache: {num_files} cached files in {self.cache_dir}>"

    def _get_cache_path(self, file_path: Path, suffix: str) -> Path:
        """
        Get cache file path for given data file

        Uses content-based naming (hash of file path) to avoid collisions.

        Args:
            file_path: Path to data file (POD5/BAM)
            suffix: Cache file suffix (e.g., '.pod5.cache', '.bam.cache')

        Returns:
            Path to cache file
        """
        path_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:16]
        return self.cache_dir / f"{file_path.stem}_{path_hash}{suffix}"

    def _file_hash(self, file_path: Path, chunk_size: int = 10_000_000) -> str:
        """
        Fast hash of first chunk of file (enough to detect changes)

        Args:
            file_path: File to hash
            chunk_size: Bytes to read (default: 10MB)

        Returns:
            MD5 hex digest
        """
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            hasher.update(f.read(chunk_size))
        return hasher.hexdigest()

    def load_pod5_index(self, file_path: Path) -> dict[str, int] | None:
        """
        Load POD5 read index from cache

        Args:
            file_path: Path to POD5 file

        Returns:
            Dict mapping read_id to file position, or None if cache miss/invalid
        """
        if not self.enabled:
            return None

        cache_path = self._get_cache_path(file_path, ".pod5.cache")
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)

            # Validate file hasn't changed
            current_hash = self._file_hash(file_path)
            if cached["file_hash"] != current_hash:
                return None

            return cached["index"]

        except (FileNotFoundError, pickle.UnpicklingError, KeyError):
            return None

    def save_pod5_index(
        self, file_path: Path, index: dict[str, int], num_reads: int | None = None
    ) -> None:
        """
        Save POD5 read index to cache

        Args:
            file_path: Path to POD5 file
            index: Dict mapping read_id to file position
            num_reads: Total number of reads (optional, for metadata)
        """
        if not self.enabled:
            return

        cache_path = self._get_cache_path(file_path, ".pod5.cache")

        try:
            cached = {
                "file_path": str(file_path),
                "file_hash": self._file_hash(file_path),
                "file_size": file_path.stat().st_size,
                "num_reads": num_reads or len(index),
                "index": index,
                "cached_at": datetime.now().isoformat(),
            }

            with open(cache_path, "wb") as f:
                pickle.dump(cached, f, protocol=pickle.HIGHEST_PROTOCOL)

        except (OSError, pickle.PicklingError):
            pass

    def load_pod5_read_ids(self, file_path: Path) -> list[str] | None:
        """
        Load materialized read ID list from cache

        Args:
            file_path: Path to POD5 file

        Returns:
            List of read IDs, or None if cache miss/invalid
        """
        if not self.enabled:
            return None

        cache_path = self._get_cache_path(file_path, ".pod5.ids.cache")
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)

            # Validate file hasn't changed
            current_hash = self._file_hash(file_path)
            if cached["file_hash"] != current_hash:
                return None

            return cached["read_ids"]

        except (FileNotFoundError, pickle.UnpicklingError, KeyError):
            return None

    def save_pod5_read_ids(self, file_path: Path, read_ids: list[str]) -> None:
        """
        Save materialized read ID list to cache

        Args:
            file_path: Path to POD5 file
            read_ids: List of read IDs
        """
        if not self.enabled:
            return

        cache_path = self._get_cache_path(file_path, ".pod5.ids.cache")

        try:
            cached = {
                "file_path": str(file_path),
                "file_hash": self._file_hash(file_path),
                "num_reads": len(read_ids),
                "read_ids": read_ids,
                "cached_at": datetime.now().isoformat(),
            }

            with open(cache_path, "wb") as f:
                pickle.dump(cached, f, protocol=pickle.HIGHEST_PROTOCOL)

        except (OSError, pickle.PicklingError):
            pass

    def load_bam_ref_mapping(self, file_path: Path) -> dict[str, list[str]] | None:
        """
        Load BAM reference mapping from cache

        Args:
            file_path: Path to BAM file

        Returns:
            Dict mapping reference name to list of read IDs, or None if cache miss/invalid
        """
        if not self.enabled:
            return None

        cache_path = self._get_cache_path(file_path, ".bam.cache")
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)

            # Validate file hasn't changed
            current_hash = self._file_hash(file_path)
            if cached["file_hash"] != current_hash:
                return None

            return cached["ref_mapping"]

        except (FileNotFoundError, pickle.UnpicklingError, KeyError):
            return None

    def save_bam_ref_mapping(
        self, file_path: Path, ref_mapping: dict[str, list[str]]
    ) -> None:
        """
        Save BAM reference mapping to cache

        Args:
            file_path: Path to BAM file
            ref_mapping: Dict mapping reference name to list of read IDs
        """
        if not self.enabled:
            return

        cache_path = self._get_cache_path(file_path, ".bam.cache")

        try:
            total_reads = sum(len(reads) for reads in ref_mapping.values())
            cached = {
                "file_path": str(file_path),
                "file_hash": self._file_hash(file_path),
                "num_references": len(ref_mapping),
                "num_reads": total_reads,
                "ref_mapping": ref_mapping,
                "cached_at": datetime.now().isoformat(),
            }

            with open(cache_path, "wb") as f:
                pickle.dump(cached, f, protocol=pickle.HIGHEST_PROTOCOL)

        except (OSError, pickle.PicklingError):
            pass

    def load_bam_metadata(self, file_path: Path) -> dict | None:
        """
        Load complete BAM metadata from cache (Phase 2 optimization)

        Caches all metadata collected by _collect_bam_metadata_single_pass():
        - references (list of dicts with name/length/read_count)
        - has_modifications (bool)
        - modification_types (list)
        - has_probabilities (bool)
        - has_event_alignment (bool)
        - ref_mapping (dict[str, list[str]])
        - num_reads (int)

        Args:
            file_path: Path to BAM file

        Returns:
            Complete metadata dict, or None if cache miss/invalid

        Examples:
            >>> cache = SquiggyCache()
            >>> metadata = cache.load_bam_metadata(Path('alignments.bam'))
            >>> if metadata:
            ...     print(f"Loaded {metadata['num_reads']} reads from cache")
        """
        if not self.enabled:
            return None

        cache_path = self._get_cache_path(file_path, ".bam.metadata.cache")
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)

            # Validate file hasn't changed (using mtime for faster check)
            current_mtime = file_path.stat().st_mtime
            if abs(cached["file_mtime"] - current_mtime) > 0.001:
                return None

            return cached["metadata"]

        except (FileNotFoundError, pickle.UnpicklingError, KeyError):
            return None

    def save_bam_metadata(self, file_path: Path, metadata: dict) -> None:
        """
        Save complete BAM metadata to cache (Phase 2 optimization)

        Caches the full metadata dict from _collect_bam_metadata_single_pass().
        This replaces the need for separate caching of individual components.

        Args:
            file_path: Path to BAM file
            metadata: Complete metadata dict with keys:
                - references
                - has_modifications
                - modification_types
                - has_probabilities
                - has_event_alignment
                - ref_mapping
                - num_reads

        Examples:
            >>> cache = SquiggyCache()
            >>> metadata = _collect_bam_metadata_single_pass(bam_path)
            >>> cache.save_bam_metadata(bam_path, metadata)
        """
        if not self.enabled:
            return

        cache_path = self._get_cache_path(file_path, ".bam.metadata.cache")

        try:
            cached = {
                "file_path": str(file_path),
                "file_mtime": file_path.stat().st_mtime,
                "file_size": file_path.stat().st_size,
                "metadata": metadata,
                "cached_at": datetime.now().isoformat(),
            }

            with open(cache_path, "wb") as f:
                pickle.dump(cached, f, protocol=pickle.HIGHEST_PROTOCOL)

        except (OSError, pickle.PicklingError):
            pass

    def clear_cache(self) -> int:
        """
        Clear all cache files

        Returns:
            Number of files deleted
        """
        if not self.enabled or not self.cache_dir.exists():
            return 0

        count = 0
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                cache_file.unlink()
                count += 1
            except OSError:
                pass

        return count
