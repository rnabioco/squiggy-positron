"""
I/O functions for loading POD5 and BAM files

These functions are called from the Positron extension via the Jupyter kernel.
"""

import logging
import os
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import pod5
import pysam

from .utils import get_bam_references

logger = logging.getLogger(__name__)


# ============================================================================
# Performance Optimization Classes
# ============================================================================


class LazyReadList:
    """
    Virtual list of read IDs - only materializes requested slices

    Provides O(1) memory overhead instead of O(n) by lazily loading read IDs
    from POD5 file on demand. Works seamlessly with TypeScript's pagination
    pattern (offset/limit slicing).

    Attributes:
        _reader: POD5 Reader instance
        _cached_length: Cached total read count (computed once)
        _materialized_ids: Optional fully materialized list (for caching)

    Examples:
        >>> reader = pod5.Reader('file.pod5')
        >>> lazy_list = LazyReadList(reader)
        >>> len(lazy_list)  # Computes length once
        1000000
        >>> lazy_list[0:100]  # Only loads first 100 IDs
        ['read1', 'read2', ...]
        >>> lazy_list[500000]  # Loads single ID at position 500000
        'read500001'
    """

    def __init__(self, reader: pod5.Reader):
        self._reader = reader
        self._cached_length: int | None = None
        self._materialized_ids: list[str] | None = None

    def __len__(self) -> int:
        """Compute total read count (cached after first call)"""
        if self._cached_length is None:
            if self._materialized_ids is not None:
                self._cached_length = len(self._materialized_ids)
            else:
                logger.info("Computing read count...")
                self._cached_length = sum(1 for _ in self._reader.reads())
                logger.info(f"Found {self._cached_length:,} reads")
        return self._cached_length

    def __getitem__(self, key: int | slice) -> str | list[str]:
        """Get read ID(s) at index/slice - lazy loading"""
        # If fully materialized, use it
        if self._materialized_ids is not None:
            return self._materialized_ids[key]

        if isinstance(key, slice):
            # Handle slice - only load requested range
            start, stop, step = key.indices(len(self))
            result = []
            for i, read in enumerate(self._reader.reads()):
                if i >= stop:
                    break
                if start <= i < stop:
                    result.append(str(read.read_id))
            return result[::step] if step != 1 else result
        else:
            # Single index lookup
            if key < 0:
                key = len(self) + key
            for i, read in enumerate(self._reader.reads()):
                if i == key:
                    return str(read.read_id)
            raise IndexError(f"Read index out of range: {key}")

    def __iter__(self) -> Iterator[str]:
        """Iterate over all read IDs"""
        if self._materialized_ids is not None:
            yield from self._materialized_ids
        else:
            for read in self._reader.reads():
                yield str(read.read_id)

    def materialize(self) -> list[str]:
        """
        Fully materialize the list (for caching)

        Returns:
            Complete list of all read IDs
        """
        if self._materialized_ids is None:
            logger.info("Materializing all read IDs...")
            self._materialized_ids = [
                str(read.read_id) for read in self._reader.reads()
            ]
            self._cached_length = len(self._materialized_ids)
            logger.info(f"Materialized {self._cached_length:,} read IDs")
        return self._materialized_ids


class Pod5Index:
    """
    Fast O(1) read lookup via read_id → file position mapping

    Builds an index mapping read IDs to their position in the POD5 file,
    enabling constant-time lookups instead of O(n) linear scans.

    Attributes:
        _index: Dict mapping read_id (str) to file position (int)

    Examples:
        >>> reader = pod5.Reader('file.pod5')
        >>> index = Pod5Index()
        >>> index.build(reader)
        >>> position = index.get_position('read_abc123')
        >>> if position is not None:
        ...     # Use position for fast retrieval
        ...     pass
    """

    def __init__(self):
        self._index: dict[str, int] = {}

    def build(self, reader: pod5.Reader) -> None:
        """
        Build index by scanning file once

        Args:
            reader: POD5 Reader to index
        """
        logger.info("Building POD5 index...")
        for idx, read in enumerate(reader.reads()):
            self._index[str(read.read_id)] = idx
        logger.info(f"Indexed {len(self._index):,} reads")

    def get_position(self, read_id: str) -> int | None:
        """
        Get file position for read_id (O(1) lookup)

        Args:
            read_id: Read ID to look up

        Returns:
            File position or None if not found
        """
        return self._index.get(read_id)

    def has_read(self, read_id: str) -> bool:
        """
        Check if read exists (O(1))

        Args:
            read_id: Read ID to check

        Returns:
            True if read exists in index
        """
        return read_id in self._index

    def __len__(self) -> int:
        """Number of indexed reads"""
        return len(self._index)


def get_reads_batch(read_ids: list[str]) -> dict[str, pod5.ReadRecord]:
    """
    Fetch multiple reads in a single pass (O(n) instead of O(m×n))

    This replaces the nested loop pattern where each read_id triggers a full
    file scan. Instead, we scan the file once and collect all requested reads.

    Args:
        read_ids: List of read IDs to fetch

    Returns:
        Dict mapping read_id to ReadRecord for found reads

    Raises:
        RuntimeError: If no POD5 file is loaded

    Examples:
        >>> from squiggy import load_pod5
        >>> from squiggy.io import get_reads_batch
        >>> load_pod5('file.pod5')
        >>> reads = get_reads_batch(['read1', 'read2', 'read3'])
        >>> for read_id, read_obj in reads.items():
        ...     print(f"{read_id}: {len(read_obj.signal)} samples")
    """
    if _squiggy_session.reader is None:
        raise RuntimeError("No POD5 file is currently loaded")

    needed = set(read_ids)
    found = {}

    logger.debug(f"Batch fetching {len(needed)} reads...")
    for read in _squiggy_session.reader.reads():
        read_id = str(read.read_id)
        if read_id in needed:
            found[read_id] = read
            if len(found) == len(needed):
                logger.debug(f"Found all {len(found)} reads")
                break  # Early exit once all found

    if len(found) < len(needed):
        missing = needed - set(found.keys())
        logger.warning(f"Could not find {len(missing)} read(s): {list(missing)[:5]}...")

    return found


def get_read_by_id(read_id: str) -> pod5.ReadRecord | None:
    """
    Get a single read by ID using index if available

    Uses Pod5Index for O(1) lookup if index is built, otherwise falls back
    to linear scan.

    Args:
        read_id: Read ID to fetch

    Returns:
        ReadRecord or None if not found

    Raises:
        RuntimeError: If no POD5 file is loaded

    Examples:
        >>> from squiggy import load_pod5
        >>> from squiggy.io import get_read_by_id
        >>> load_pod5('file.pod5')
        >>> read = get_read_by_id('read_abc123')
        >>> if read:
        ...     print(f"Signal length: {len(read.signal)}")
    """
    if _squiggy_session.reader is None:
        raise RuntimeError("No POD5 file is currently loaded")

    # Use index if available
    if (
        hasattr(_squiggy_session, "pod5_index")
        and _squiggy_session.pod5_index is not None
    ):
        position = _squiggy_session.pod5_index.get_position(read_id)
        if position is None:
            return None

        # Use indexed access
        for idx, read in enumerate(_squiggy_session.reader.reads()):
            if idx == position:
                return read
        return None

    # Fallback to linear scan
    logger.debug(f"No index available, using linear scan for {read_id}")
    for read in _squiggy_session.reader.reads():
        if str(read.read_id) == read_id:
            return read
    return None


class Sample:
    """
    Represents a single POD5/BAM file pair (sample/experiment)

    This class encapsulates all data for one sequencing run or basecalling model,
    allowing multiple samples to be loaded and compared simultaneously.

    Attributes:
        name: Unique identifier for this sample (e.g., 'basecaller_v4.2')
        pod5_path: Path to POD5 file
        pod5_reader: Open POD5 file reader
        read_ids: List of read IDs in this sample
        bam_path: Path to BAM file (optional)
        bam_info: Metadata about BAM file
        model_provenance: Model/basecaller information extracted from BAM header
        fasta_path: Path to FASTA reference file (optional)
        fasta_info: Metadata about FASTA file

    Examples:
        >>> sample = Sample('model_v4.2')
        >>> sample.load_pod5('data_v4.2.pod5')
        >>> sample.load_bam('align_v4.2.bam')
        >>> print(f"{sample.name}: {len(sample.read_ids)} reads")
    """

    def __init__(self, name: str):
        """Initialize a new sample with the given name"""
        self.name = name
        self.pod5_path: str | None = None
        self.pod5_reader: pod5.Reader | None = None
        self.read_ids: list[str] = []
        self.bam_path: str | None = None
        self.bam_info: dict | None = None
        self.model_provenance: dict | None = None
        self.fasta_path: str | None = None
        self.fasta_info: dict | None = None

    def __repr__(self) -> str:
        """Return informative summary of sample state"""
        parts = [f"Sample({self.name})"]

        if self.pod5_path:
            filename = os.path.basename(self.pod5_path)
            parts.append(f"POD5: {filename} ({len(self.read_ids):,} reads)")

        if self.bam_path:
            filename = os.path.basename(self.bam_path)
            num_reads = self.bam_info.get("num_reads", 0) if self.bam_info else 0
            parts.append(f"BAM: {filename} ({num_reads:,} reads)")

        if self.fasta_path:
            filename = os.path.basename(self.fasta_path)
            num_refs = (
                len(self.fasta_info.get("references", [])) if self.fasta_info else 0
            )
            parts.append(f"FASTA: {filename} ({num_refs:,} references)")

        if len(parts) == 1:
            return f"<Sample({self.name}): No files loaded>"

        return f"<{' | '.join(parts)}>"

    def close(self):
        """Close all resources and clear sample state"""
        if self.pod5_reader is not None:
            self.pod5_reader.close()
            self.pod5_reader = None
        self.pod5_path = None
        self.read_ids = []
        self.bam_path = None
        self.bam_info = None
        self.model_provenance = None
        self.fasta_path = None
        self.fasta_info = None


class SquiggySession:
    """
    Manages state for loaded POD5 and BAM files, supporting multiple samples

    This enhanced session manages multiple POD5/BAM pairs (samples) simultaneously,
    enabling comparison workflows. Maintains backward compatibility with single-sample
    API by delegating to the first loaded sample.

    Attributes:
        samples: Dict of Sample objects, keyed by sample name
        reader: POD5 file reader (first sample, for backward compat)
        pod5_path: Path to loaded POD5 file (first sample, for backward compat)
        read_ids: List of read IDs (first sample, for backward compat)
        bam_path: Path to loaded BAM file (first sample, for backward compat)
        bam_info: Metadata about loaded BAM file (first sample, for backward compat)
        ref_mapping: Mapping of reference names to read IDs
        fasta_path: Path to loaded FASTA file (first sample, for backward compat)
        fasta_info: Metadata about loaded FASTA file (first sample, for backward compat)

    Examples:
        >>> from squiggy import load_pod5, load_bam, load_sample
        >>> # Single sample (backward compatible)
        >>> load_pod5('data.pod5')
        >>> # Multiple samples
        >>> load_sample('model_v4.2', 'data_v4.2.pod5', 'align_v4.2.bam')
        >>> load_sample('model_v5.0', 'data_v5.0.pod5', 'align_v5.0.bam')
        >>> # Access
        >>> sample = get_sample('model_v5.0')
        >>> print(sample)
    """

    def __init__(self, cache_dir: str | None = None, use_cache: bool = True):
        # Multi-sample support (NEW)
        self.samples: dict[str, Sample] = {}

        # Single-sample properties (for backward compatibility)
        self.reader: pod5.Reader | None = None
        self.pod5_path: str | None = None
        self.read_ids: list[str] | LazyReadList = []
        self.bam_path: str | None = None
        self.bam_info: dict | None = None
        self.ref_mapping: dict[str, list[str]] | None = None
        self.fasta_path: str | None = None
        self.fasta_info: dict | None = None

        # Performance optimization attributes (NEW)
        self.pod5_index: Pod5Index | None = None

        # Cache integration (NEW)
        from .cache import SquiggyCache

        cache_path = Path(cache_dir) if cache_dir else None
        self.cache = SquiggyCache(cache_path, enabled=use_cache) if use_cache else None

    def __repr__(self) -> str:
        """Return informative summary of loaded files"""
        if self.samples:
            # Multi-sample mode
            parts = [f"SquiggySession: {len(self.samples)} sample(s)"]
            for name in sorted(self.samples.keys()):
                sample = self.samples[name]
                if sample.pod5_path:
                    num_reads = len(sample.read_ids)
                    parts.append(f"  {name}: {num_reads:,} reads")
            return "<" + "\n".join(parts) + ">"
        else:
            # Single-sample backward compat mode
            parts = []

            if self.pod5_path:
                filename = os.path.basename(self.pod5_path)
                parts.append(f"POD5: {filename} ({len(self.read_ids):,} reads)")

            if self.bam_path:
                filename = os.path.basename(self.bam_path)
                num_reads = self.bam_info.get("num_reads", 0) if self.bam_info else 0
                parts.append(f"BAM: {filename} ({num_reads:,} reads)")

                if self.bam_info:
                    if self.bam_info.get("has_modifications"):
                        mod_types = ", ".join(
                            str(m) for m in self.bam_info["modification_types"]
                        )
                        parts.append(f"Modifications: {mod_types}")
                    if self.bam_info.get("has_event_alignment"):
                        parts.append("Event alignment: yes")

            if self.fasta_path:
                filename = os.path.basename(self.fasta_path)
                num_refs = (
                    len(self.fasta_info.get("references", [])) if self.fasta_info else 0
                )
                parts.append(f"FASTA: {filename} ({num_refs:,} references)")

            if not parts:
                return "<SquiggySession: No files loaded>"

            return f"<SquiggySession: {' | '.join(parts)}>"

    # Multi-sample API methods (NEW)

    def load_sample(
        self,
        name: str,
        pod5_path: str,
        bam_path: str | None = None,
        fasta_path: str | None = None,
    ) -> Sample:
        """
        Load a POD5/BAM/FASTA sample set into this session

        Args:
            name: Unique identifier for this sample (e.g., 'model_v4.2')
            pod5_path: Path to POD5 file
            bam_path: Path to BAM file (optional)
            fasta_path: Path to FASTA file (optional)

        Returns:
            The created Sample object

        Examples:
            >>> session = SquiggySession()
            >>> sample = session.load_sample('v4.2', 'data_v4.2.pod5', 'align_v4.2.bam')
            >>> print(f"Loaded {len(sample.read_ids)} reads")
        """
        # Close existing sample with this name if any
        if name in self.samples:
            self.samples[name].close()

        # Create new sample
        sample = Sample(name)

        # Load POD5
        abs_pod5_path = os.path.abspath(pod5_path)
        if not os.path.exists(abs_pod5_path):
            raise FileNotFoundError(f"POD5 file not found: {abs_pod5_path}")

        reader = pod5.Reader(abs_pod5_path)
        read_ids = [str(read.read_id) for read in reader.reads()]

        sample.pod5_path = abs_pod5_path
        sample.pod5_reader = reader
        sample.read_ids = read_ids

        # Load BAM if provided
        if bam_path:
            abs_bam_path = os.path.abspath(bam_path)
            if not os.path.exists(abs_bam_path):
                raise FileNotFoundError(f"BAM file not found: {abs_bam_path}")

            # Get references
            references = get_bam_references(abs_bam_path)

            # Check for base modifications
            mod_info = get_bam_modification_info(abs_bam_path)

            # Check for event alignment data
            has_event_alignment = get_bam_event_alignment_status(abs_bam_path)

            # Build metadata dict
            bam_info = {
                "file_path": abs_bam_path,
                "num_reads": sum(ref["read_count"] for ref in references),
                "references": references,
                "has_modifications": mod_info["has_modifications"],
                "modification_types": mod_info["modification_types"],
                "has_probabilities": mod_info["has_probabilities"],
                "has_event_alignment": has_event_alignment,
            }

            sample.bam_path = abs_bam_path
            sample.bam_info = bam_info

        # Load FASTA if provided
        if fasta_path:
            abs_fasta_path = os.path.abspath(fasta_path)
            if not os.path.exists(abs_fasta_path):
                raise FileNotFoundError(f"FASTA file not found: {abs_fasta_path}")

            # Check for index
            fai_path = abs_fasta_path + ".fai"
            if not os.path.exists(fai_path):
                raise FileNotFoundError(
                    f"FASTA index not found: {fai_path}. "
                    f"Create index with: samtools faidx {abs_fasta_path}"
                )

            # Open FASTA file to get metadata
            fasta = pysam.FastaFile(abs_fasta_path)

            try:
                # Extract reference information
                references = list(fasta.references)
                lengths = list(fasta.lengths)

                # Build metadata dict
                fasta_info = {
                    "file_path": abs_fasta_path,
                    "references": references,
                    "num_references": len(references),
                    "reference_lengths": dict(zip(references, lengths, strict=True)),
                }

            finally:
                fasta.close()

            sample.fasta_path = abs_fasta_path
            sample.fasta_info = fasta_info

        # Store sample
        self.samples[name] = sample

        return sample

    def get_sample(self, name: str) -> Sample | None:
        """
        Get a loaded sample by name

        Args:
            name: Sample name

        Returns:
            Sample object or None if not found

        Examples:
            >>> session = SquiggySession()
            >>> sample = session.get_sample('model_v4.2')
        """
        return self.samples.get(name)

    def list_samples(self) -> list[str]:
        """
        List all loaded sample names

        Returns:
            List of sample names in order they were loaded

        Examples:
            >>> session = SquiggySession()
            >>> names = session.list_samples()
            >>> print(f"Loaded samples: {names}")
        """
        return list(self.samples.keys())

    def remove_sample(self, name: str) -> None:
        """
        Unload a sample and free its resources

        Args:
            name: Sample name to remove

        Examples:
            >>> session = SquiggySession()
            >>> session.remove_sample('model_v4.2')
        """
        if name in self.samples:
            self.samples[name].close()
            del self.samples[name]

    # Backward compatibility methods

    def close_pod5(self):
        """Close POD5 reader and clear POD5 state (backward compat mode)"""
        if self.reader is not None:
            self.reader.close()
            self.reader = None
        self.pod5_path = None
        self.read_ids = []
        self.pod5_index = None  # Clear index

    def close_bam(self):
        """Clear BAM state (backward compat mode)"""
        self.bam_path = None
        self.bam_info = None
        self.ref_mapping = None

    def close_fasta(self):
        """Clear FASTA state (backward compat mode)"""
        self.fasta_path = None
        self.fasta_info = None

    def close_all(self):
        """Close all resources and clear all state"""
        # Close all samples
        for sample in list(self.samples.values()):
            sample.close()
        self.samples.clear()

        # Clear backward compat properties
        self.close_pod5()
        self.close_bam()
        self.close_fasta()


# Global session instance (single source of truth for kernel state)
_squiggy_session = SquiggySession()


def load_pod5(file_path: str, build_index: bool = True, use_cache: bool = True) -> None:
    """
    Load a POD5 file into the global kernel session (OPTIMIZED)

    This function mutates the global _squiggy_session object, making
    POD5 data available for subsequent plotting and analysis calls.

    Performance optimizations:
    - Lazy read ID loading (O(1) memory vs. O(n))
    - Optional index building for O(1) lookups
    - Persistent caching for instant subsequent loads

    Args:
        file_path: Path to POD5 file
        build_index: Whether to build read ID index (default: True)
        use_cache: Whether to use persistent cache (default: True)

    Returns:
        None (mutates global _squiggy_session)

    Examples:
        >>> from squiggy import load_pod5
        >>> from squiggy.io import _squiggy_session
        >>> load_pod5('data.pod5')
        >>> print(f"Loaded {len(_squiggy_session.read_ids)} reads")
        >>> # Session is available as _squiggy_session in kernel
        >>> first_read = next(_squiggy_session.reader.reads())
    """
    # Convert to absolute path
    abs_path = Path(file_path).resolve()

    if not abs_path.exists():
        raise FileNotFoundError(f"Failed to open pod5 file at: {abs_path}")

    # Close previous reader if exists
    _squiggy_session.close_pod5()

    # Open new reader
    reader = pod5.Reader(str(abs_path))

    # Create lazy read list (O(1) memory overhead)
    lazy_read_list = LazyReadList(reader)

    # Try to load index from cache
    cached_index = None
    if use_cache and _squiggy_session.cache:
        logger.info("Checking cache for POD5 index...")
        cached_index = _squiggy_session.cache.load_pod5_index(abs_path)

    # Build or restore index
    if build_index:
        if cached_index:
            # Restore from cache
            pod5_index = Pod5Index()
            pod5_index._index = cached_index
            logger.info(f"Restored index from cache ({len(pod5_index):,} reads)")
        else:
            # Build fresh
            pod5_index = Pod5Index()
            pod5_index.build(reader)

            # Save to cache for next time
            if use_cache and _squiggy_session.cache:
                _squiggy_session.cache.save_pod5_index(abs_path, pod5_index._index)

        _squiggy_session.pod5_index = pod5_index
    else:
        _squiggy_session.pod5_index = None

    # Store state in session
    _squiggy_session.reader = reader
    _squiggy_session.pod5_path = str(abs_path)
    _squiggy_session.read_ids = lazy_read_list

    logger.info(f"Loaded POD5: {abs_path.name} ({len(lazy_read_list):,} reads)")


def get_bam_event_alignment_status(file_path: str) -> bool:
    """
    Check if BAM file contains event alignment data (mv tag)

    The mv tag contains the move table from basecalling, which maps
    nanopore signal events to basecalled nucleotides. This is required
    for event-aligned plotting mode.

    Args:
        file_path: Path to BAM file

    Returns:
        True if mv tag is found in sampled reads

    Examples:
        >>> from squiggy import get_bam_event_alignment_status
        >>> has_events = get_bam_event_alignment_status('alignments.bam')
        >>> if has_events:
        ...     print("BAM contains event alignment data")
    """
    from .constants import BAM_SAMPLE_SIZE

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"BAM file not found: {file_path}")

    max_reads_to_check = BAM_SAMPLE_SIZE  # Sample first N reads

    try:
        bam = pysam.AlignmentFile(file_path, "rb", check_sq=False)

        for i, read in enumerate(bam.fetch(until_eof=True)):
            if i >= max_reads_to_check:
                break

            # Check for mv tag (move table)
            if read.has_tag("mv"):
                bam.close()
                return True

        bam.close()

    except Exception as e:
        print(f"Warning: Error checking BAM event alignment: {e}")
        return False

    return False


def get_bam_modification_info(file_path: str) -> dict:
    """
    Check if BAM file contains base modification tags (MM/ML)

    Args:
        file_path: Path to BAM file

    Returns:
        Dict with:
            - has_modifications: bool
            - modification_types: list of modification codes (e.g., ['m', 'h'])
            - sample_count: number of reads checked

    Examples:
        >>> from squiggy import get_bam_modification_info
        >>> mod_info = get_bam_modification_info('alignments.bam')
        >>> if mod_info['has_modifications']:
        ...     print(f"Found modifications: {mod_info['modification_types']}")
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"BAM file not found: {file_path}")

    from .constants import BAM_SAMPLE_SIZE

    modification_types = set()
    has_modifications = False
    has_ml = False
    reads_checked = 0
    max_reads_to_check = BAM_SAMPLE_SIZE  # Sample first N reads

    try:
        bam = pysam.AlignmentFile(file_path, "rb", check_sq=False)

        for read in bam.fetch(until_eof=True):
            if reads_checked >= max_reads_to_check:
                break

            reads_checked += 1

            # Check for modifications using pysam's modified_bases property
            # This is cleaner and more reliable than parsing MM tags manually
            if hasattr(read, "modified_bases") and read.modified_bases:
                has_modifications = True

                # modified_bases returns dict with format:
                # {(canonical_base, strand, mod_code): [(position, quality), ...]}
                for (
                    _canonical_base,
                    _strand,
                    mod_code,
                ), _mod_list in read.modified_bases.items():
                    # mod_code can be str (e.g., 'm') or int (e.g., 17596)
                    # Store as-is to preserve type
                    modification_types.add(mod_code)

            # Check for ML tag (modification probabilities)
            if read.has_tag("ML"):
                has_ml = True

        bam.close()

    except Exception as e:
        print(f"Warning: Error checking BAM modifications: {e}")
        return {
            "has_modifications": False,
            "modification_types": [],
            "sample_count": 0,
            "has_probabilities": False,
        }

    # Convert modification_types to list, handling mixed str/int types
    # Sort with custom key that converts to string for comparison
    mod_types_list = sorted(modification_types, key=str)

    return {
        "has_modifications": has_modifications,
        "modification_types": mod_types_list,
        "sample_count": reads_checked,
        "has_probabilities": has_ml,
    }


def load_bam(
    file_path: str, build_ref_mapping: bool = True, use_cache: bool = True
) -> None:
    """
    Load a BAM file into the global kernel session (OPTIMIZED)

    This function mutates the global _squiggy_session object, making
    BAM alignment data available for subsequent plotting and analysis calls.

    Performance optimizations:
    - Eager reference mapping (transparent cost, eliminates UI freezes)
    - Persistent caching for instant subsequent loads

    Args:
        file_path: Path to BAM file
        build_ref_mapping: Whether to build reference→reads mapping (default: True)
        use_cache: Whether to use persistent cache (default: True)

    Returns:
        None (mutates global _squiggy_session)

    Examples:
        >>> from squiggy import load_bam
        >>> from squiggy.io import _squiggy_session
        >>> load_bam('alignments.bam')
        >>> print(_squiggy_session.bam_info['references'])
        >>> if _squiggy_session.bam_info['has_modifications']:
        ...     print(f"Modifications: {_squiggy_session.bam_info['modification_types']}")
        >>> if _squiggy_session.bam_info['has_event_alignment']:
        ...     print("Event alignment data available")
    """
    # Convert to absolute path
    abs_path = Path(file_path).resolve()

    if not abs_path.exists():
        raise FileNotFoundError(f"BAM file not found: {abs_path}")

    # Get references
    references = get_bam_references(str(abs_path))

    # Check for base modifications
    mod_info = get_bam_modification_info(str(abs_path))

    # Check for event alignment data
    has_event_alignment = get_bam_event_alignment_status(str(abs_path))

    # Build metadata dict
    bam_info = {
        "file_path": str(abs_path),
        "num_reads": sum(ref["read_count"] for ref in references),
        "references": references,
        "has_modifications": mod_info["has_modifications"],
        "modification_types": mod_info["modification_types"],
        "has_probabilities": mod_info["has_probabilities"],
        "has_event_alignment": has_event_alignment,
    }

    # Build reference mapping (eager, with caching)
    ref_mapping = None
    if build_ref_mapping:
        # Try cache first
        if use_cache and _squiggy_session.cache:
            logger.info("Checking cache for BAM reference mapping...")
            ref_mapping = _squiggy_session.cache.load_bam_ref_mapping(abs_path)

        if ref_mapping is None:
            # Build fresh
            logger.info("Building BAM reference mapping...")
            ref_mapping = _build_ref_mapping_immediate(abs_path)
            logger.info(f"Mapped {len(ref_mapping)} references")

            # Save to cache
            if use_cache and _squiggy_session.cache:
                _squiggy_session.cache.save_bam_ref_mapping(abs_path, ref_mapping)

    # Store state in session
    _squiggy_session.bam_path = str(abs_path)
    _squiggy_session.bam_info = bam_info
    _squiggy_session.ref_mapping = ref_mapping

    logger.info(f"Loaded BAM: {abs_path.name} ({bam_info['num_reads']:,} reads)")


def _build_ref_mapping_immediate(bam_path: Path) -> dict[str, list[str]]:
    """
    Build reference→reads mapping immediately (called from load_bam)

    This is the eager version of get_read_to_reference_mapping(), built
    during load to avoid surprise costs later.

    Args:
        bam_path: Path to BAM file

    Returns:
        Dict mapping reference name to list of read IDs
    """
    mapping = defaultdict(list)

    with pysam.AlignmentFile(str(bam_path), "rb", check_sq=False) as bam:
        for read in bam.fetch(until_eof=True):
            if not read.is_unmapped:
                ref_name = bam.get_reference_name(read.reference_id)
                mapping[ref_name].append(read.query_name)

    return dict(mapping)


def get_reads_for_reference_paginated(
    reference_name: str, offset: int = 0, limit: int | None = None
) -> list[str]:
    """
    Get reads for a specific reference with pagination support

    This function enables lazy loading of reads by reference for the UI.
    Returns a slice of read IDs for the specified reference, supporting
    incremental data fetching.

    Args:
        reference_name: Name of reference sequence (e.g., 'chr1', 'contig_42')
        offset: Starting index in the read list (default: 0)
        limit: Maximum number of reads to return (default: None = all remaining)

    Returns:
        List of read IDs for the specified reference, sliced by offset/limit

    Raises:
        RuntimeError: If no BAM file is loaded in the session
        KeyError: If reference_name is not found in the BAM file

    Examples:
        >>> from squiggy import load_bam, load_pod5
        >>> from squiggy.io import get_reads_for_reference_paginated
        >>> load_pod5('reads.pod5')
        >>> load_bam('alignments.bam')
        >>> # Get first 500 reads for chr1
        >>> reads = get_reads_for_reference_paginated('chr1', offset=0, limit=500)
        >>> len(reads)
        500
        >>> # Get next 500 reads
        >>> more_reads = get_reads_for_reference_paginated('chr1', offset=500, limit=500)
    """
    if _squiggy_session.ref_mapping is None:
        raise RuntimeError(
            "No BAM file loaded. Call load_bam() before accessing reference reads."
        )

    if reference_name not in _squiggy_session.ref_mapping:
        available_refs = list(_squiggy_session.ref_mapping.keys())
        raise KeyError(
            f"Reference '{reference_name}' not found. "
            f"Available references: {available_refs[:5]}..."
        )

    all_reads = _squiggy_session.ref_mapping[reference_name]

    if limit is None:
        return all_reads[offset:]
    return all_reads[offset : offset + limit]


def load_fasta(file_path: str) -> None:
    """
    Load a FASTA file into the global kernel session

    This function mutates the global _squiggy_session object, making
    FASTA reference sequences available for subsequent motif search and
    analysis calls.

    If a FASTA index (.fai) doesn't exist, it will be automatically created
    using pysam.faidx().

    Args:
        file_path: Path to FASTA file (index will be created if missing)

    Returns:
        None (mutates global _squiggy_session)

    Examples:
        >>> from squiggy import load_fasta
        >>> from squiggy.io import _squiggy_session
        >>> load_fasta('genome.fa')  # Creates .fai index if needed
        >>> print(_squiggy_session.fasta_info['references'])
        >>> # Use with motif search
        >>> from squiggy.motif import search_motif
        >>> matches = list(search_motif(_squiggy_session.fasta_path, "DRACH"))
    """
    # Convert to absolute path
    abs_path = os.path.abspath(file_path)

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"FASTA file not found: {abs_path}")

    # Check for index, create if missing
    fai_path = abs_path + ".fai"
    if not os.path.exists(fai_path):
        logger.info(f"FASTA index not found, creating: {fai_path}")
        try:
            pysam.faidx(abs_path)
            logger.info(f"Successfully created FASTA index: {fai_path}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to create FASTA index for {abs_path}. "
                f"Error: {e}. You can also create it manually with: samtools faidx {abs_path}"
            ) from e

    # Open FASTA file to get metadata
    fasta = pysam.FastaFile(abs_path)

    try:
        # Extract reference information
        references = list(fasta.references)
        lengths = list(fasta.lengths)

        # Build metadata dict
        fasta_info = {
            "file_path": abs_path,
            "references": references,
            "num_references": len(references),
            "reference_lengths": dict(zip(references, lengths, strict=True)),
        }

    finally:
        fasta.close()

    # Store state in session (no global keyword needed - just mutating object!)
    _squiggy_session.fasta_path = abs_path
    _squiggy_session.fasta_info = fasta_info


def get_read_to_reference_mapping() -> dict[str, list[str]]:
    """
    Get mapping of reference names to read IDs from currently loaded BAM

    Returns:
        Dict mapping reference name to list of read IDs

    Raises:
        RuntimeError: If no BAM file is loaded

    Examples:
        >>> from squiggy import load_bam, get_read_to_reference_mapping
        >>> load_bam('alignments.bam')
        >>> mapping = get_read_to_reference_mapping()
        >>> print(f"References: {list(mapping.keys())}")
    """
    if _squiggy_session.bam_path is None:
        raise RuntimeError("No BAM file is currently loaded")

    if not os.path.exists(_squiggy_session.bam_path):
        raise FileNotFoundError(f"BAM file not found: {_squiggy_session.bam_path}")

    # Open BAM file
    bam = pysam.AlignmentFile(_squiggy_session.bam_path, "rb", check_sq=False)

    # Map reference to read IDs
    ref_to_reads: dict[str, list[str]] = {}

    try:
        for read in bam.fetch(until_eof=True):
            if read.is_unmapped:
                continue

            ref_name = bam.get_reference_name(read.reference_id)
            read_id = read.query_name

            if ref_name not in ref_to_reads:
                ref_to_reads[ref_name] = []

            ref_to_reads[ref_name].append(read_id)

    finally:
        bam.close()

    # Store in session (no global keyword needed!)
    _squiggy_session.ref_mapping = ref_to_reads

    return ref_to_reads


def get_current_files() -> dict[str, str | None]:
    """
    Get paths of currently loaded files

    Returns:
        Dict with pod5_path and bam_path (may be None)
    """
    return {
        "pod5_path": _squiggy_session.pod5_path,
        "bam_path": _squiggy_session.bam_path,
    }


def get_read_ids() -> list[str]:
    """
    Get list of read IDs from currently loaded POD5 file

    Returns:
        List of read ID strings (materialized from lazy list if needed)
    """
    if not _squiggy_session.read_ids:
        raise ValueError("No POD5 file is currently loaded")

    # Convert LazyReadList to list if needed
    if isinstance(_squiggy_session.read_ids, LazyReadList):
        return list(_squiggy_session.read_ids)
    return _squiggy_session.read_ids


def close_pod5():
    """
    Close the currently open POD5 reader

    Call this to free resources when done.

    Examples:
        >>> from squiggy import load_pod5, close_pod5
        >>> load_pod5('data.pod5')
        >>> # ... work with data ...
        >>> close_pod5()
    """
    # Clear session (no global keyword needed!)
    _squiggy_session.close_pod5()


def close_bam():
    """
    Clear the currently loaded BAM file state

    Call this to free BAM-related resources when done. Unlike close_pod5(),
    this doesn't need to close a file handle since BAM files are opened
    and closed per-operation.

    Examples:
        >>> from squiggy import load_bam, close_bam
        >>> load_bam('alignments.bam')
        >>> # ... work with alignments ...
        >>> close_bam()
    """
    # Clear session (no global keyword needed!)
    _squiggy_session.close_bam()


def close_fasta():
    """
    Clear the currently loaded FASTA file state

    Call this to free FASTA-related resources when done. Unlike close_pod5(),
    this doesn't need to close a file handle since FASTA files are opened
    and closed per-operation.

    Examples:
        >>> from squiggy import load_fasta, close_fasta
        >>> load_fasta('genome.fa')
        >>> # ... work with sequences ...
        >>> close_fasta()
    """
    # Clear session (no global keyword needed!)
    _squiggy_session.close_fasta()


# Public API convenience function for multi-sample loading
def load_sample(
    name: str,
    pod5_path: str,
    bam_path: str | None = None,
    fasta_path: str | None = None,
) -> Sample:
    """
    Load a POD5/BAM/FASTA sample set into the global session

    Convenience function that loads a named sample into the global _squiggy_session.

    Args:
        name: Unique identifier for this sample (e.g., 'model_v4.2')
        pod5_path: Path to POD5 file
        bam_path: Path to BAM file (optional)
        fasta_path: Path to FASTA file (optional)

    Returns:
        The created Sample object

    Examples:
        >>> from squiggy import load_sample
        >>> sample = load_sample('v4.2', 'data_v4.2.pod5', 'align_v4.2.bam')
        >>> print(f"Loaded {len(sample.read_ids)} reads")
    """
    return _squiggy_session.load_sample(name, pod5_path, bam_path, fasta_path)


def get_sample(name: str) -> Sample | None:
    """
    Get a loaded sample by name from the global session

    Args:
        name: Sample name

    Returns:
        Sample object or None if not found

    Examples:
        >>> from squiggy import get_sample
        >>> sample = get_sample('model_v4.2')
    """
    return _squiggy_session.get_sample(name)


def list_samples() -> list[str]:
    """
    List all loaded sample names in the global session

    Returns:
        List of sample names

    Examples:
        >>> from squiggy import list_samples
        >>> names = list_samples()
        >>> print(f"Loaded samples: {names}")
    """
    return _squiggy_session.list_samples()


def remove_sample(name: str) -> None:
    """
    Unload a sample from the global session and free its resources

    Args:
        name: Sample name to remove

    Examples:
        >>> from squiggy import remove_sample
        >>> remove_sample('model_v4.2')
    """
    _squiggy_session.remove_sample(name)


def close_all_samples() -> None:
    """
    Close all samples and clear the global session

    Examples:
        >>> from squiggy import close_all_samples
        >>> close_all_samples()
    """
    _squiggy_session.close_all()


# Phase 2: Comparison Functions


def get_common_reads(sample_names: list[str]) -> set[str]:
    """
    Get reads that are present in all specified samples

    Finds the intersection of read IDs across multiple samples.

    Args:
        sample_names: List of sample names to compare

    Returns:
        Set of read IDs present in all samples

    Raises:
        ValueError: If any sample name not found

    Examples:
        >>> from squiggy import get_common_reads
        >>> common = get_common_reads(['model_v4.2', 'model_v5.0'])
        >>> print(f"Common reads: {len(common)}")
    """
    if not sample_names:
        return set()

    # Get first sample
    first_sample = _squiggy_session.get_sample(sample_names[0])
    if first_sample is None:
        raise ValueError(f"Sample '{sample_names[0]}' not found")

    # Start with reads from first sample
    common = set(first_sample.read_ids)

    # Intersect with remaining samples
    for name in sample_names[1:]:
        sample = _squiggy_session.get_sample(name)
        if sample is None:
            raise ValueError(f"Sample '{name}' not found")
        common &= set(sample.read_ids)

    return common


def get_unique_reads(
    sample_name: str, exclude_samples: list[str] | None = None
) -> set[str]:
    """
    Get reads unique to a sample (not in other samples)

    Finds reads that are only in the specified sample.

    Args:
        sample_name: Sample to find unique reads for
        exclude_samples: Samples to exclude from (default: all other samples)

    Returns:
        Set of read IDs unique to the sample

    Raises:
        ValueError: If sample not found

    Examples:
        >>> from squiggy import get_unique_reads
        >>> unique_a = get_unique_reads('model_v4.2')
        >>> unique_b = get_unique_reads('model_v5.0')
    """
    sample = _squiggy_session.get_sample(sample_name)
    if sample is None:
        raise ValueError(f"Sample '{sample_name}' not found")

    sample_reads = set(sample.read_ids)

    # Determine which samples to exclude
    if exclude_samples is None:
        # Exclude all other samples
        exclude_samples = [
            name for name in _squiggy_session.list_samples() if name != sample_name
        ]

    # Remove reads that appear in any excluded sample
    for exclude_name in exclude_samples:
        exclude_sample = _squiggy_session.get_sample(exclude_name)
        if exclude_sample is None:
            raise ValueError(f"Sample '{exclude_name}' not found")
        sample_reads -= set(exclude_sample.read_ids)

    return sample_reads


def compare_samples(sample_names: list[str]) -> dict:
    """
    Compare multiple samples and return analysis

    Generates a comprehensive comparison of samples including read overlap,
    reference validation, and model provenance information.

    Args:
        sample_names: List of sample names to compare

    Returns:
        Dict with comparison results:
            - samples: List of sample names
            - read_overlap: Read ID overlap analysis
            - reference_validation: Reference compatibility (if BAM files loaded)
            - sample_info: Basic info about each sample

    Examples:
        >>> from squiggy import compare_samples
        >>> result = compare_samples(['model_v4.2', 'model_v5.0'])
        >>> print(result['read_overlap'])
    """
    from .utils import compare_read_sets, validate_sq_headers

    # Validate samples exist
    for name in sample_names:
        if _squiggy_session.get_sample(name) is None:
            raise ValueError(f"Sample '{name}' not found")

    result = {
        "samples": sample_names,
        "sample_info": {},
        "read_overlap": {},
    }

    # Add basic info about each sample
    for name in sample_names:
        sample = _squiggy_session.get_sample(name)
        result["sample_info"][name] = {
            "num_reads": len(sample.read_ids),
            "pod5_path": sample.pod5_path,
            "bam_path": sample.bam_path,
        }

    # Compare read sets for all pairs
    if len(sample_names) >= 2:
        for i, name_a in enumerate(sample_names):
            for name_b in sample_names[i + 1 :]:
                sample_a = _squiggy_session.get_sample(name_a)
                sample_b = _squiggy_session.get_sample(name_b)
                pair_key = f"{name_a}_vs_{name_b}"
                result["read_overlap"][pair_key] = compare_read_sets(
                    sample_a.read_ids, sample_b.read_ids
                )

    # Validate references if BAM files are loaded
    if len(sample_names) >= 2:
        bam_pairs = []
        for i, name_a in enumerate(sample_names):
            for name_b in sample_names[i + 1 :]:
                sample_a = _squiggy_session.get_sample(name_a)
                sample_b = _squiggy_session.get_sample(name_b)
                if sample_a.bam_path and sample_b.bam_path:
                    bam_pairs.append(
                        (name_a, name_b, sample_a.bam_path, sample_b.bam_path)
                    )

        if bam_pairs:
            result["reference_validation"] = {}
            for name_a, name_b, bam_a, bam_b in bam_pairs:
                pair_key = f"{name_a}_vs_{name_b}"
                try:
                    validation = validate_sq_headers(bam_a, bam_b)
                    result["reference_validation"][pair_key] = validation
                except Exception as e:
                    result["reference_validation"][pair_key] = {"error": str(e)}

    return result
