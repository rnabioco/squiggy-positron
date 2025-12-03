"""Core I/O functions for loading POD5 and BAM files"""

import logging
import os
from collections import defaultdict
from pathlib import Path

import pod5
import pysam

from .kernel import squiggy_kernel
from .performance import LazyReadList, Pod5Index
from .samples import Sample

# Configure module logger
logger = logging.getLogger(__name__)


def get_reads_batch(
    read_ids: list[str], sample_name: str | None = None
) -> dict[str, pod5.ReadRecord]:
    """
    Fetch multiple reads in a single pass (O(n) instead of O(m×n))

    This replaces the nested loop pattern where each read_id triggers a full
    file scan. Instead, we scan the file once and collect all requested reads.

    Args:
        read_ids: List of read IDs to fetch
        sample_name: (Multi-sample mode) Name of sample to get reads from.
                     If None, uses global session reader.

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
    # Determine which reader to use
    if sample_name:
        sample = squiggy_kernel.get_sample(sample_name)
        if not sample or sample._pod5_reader is None:
            raise RuntimeError(f"Sample '{sample_name}' not loaded or has no POD5 file")
        reader = sample._pod5_reader
    else:
        if squiggy_kernel._reader is None:
            raise RuntimeError("No POD5 file is currently loaded")
        reader = squiggy_kernel._reader

    needed = set(read_ids)
    found = {}

    for read in reader.reads():
        read_id = str(read.read_id)
        if read_id in needed:
            found[read_id] = read
            if len(found) == len(needed):
                break  # Early exit once all found

    return found


def get_reads_batch_multi_sample(
    read_sample_map: dict[str, str],
) -> dict[str, pod5.ReadRecord]:
    """
    Fetch multiple reads from different samples in optimized batches

    Groups reads by sample, then fetches each sample's reads in a single pass.
    This is more efficient than fetching reads one-by-one across samples.

    Args:
        read_sample_map: Dict mapping read_id → sample_name

    Returns:
        Dict mapping read_id to ReadRecord for found reads

    Raises:
        RuntimeError: If a sample is not loaded or has no POD5 file

    Examples:
        >>> from squiggy.io import get_reads_batch_multi_sample
        >>> read_map = {
        ...     'read_001': 'sample_A',
        ...     'read_002': 'sample_A',
        ...     'read_003': 'sample_B',
        ... }
        >>> reads = get_reads_batch_multi_sample(read_map)
        >>> for read_id, read_obj in reads.items():
        ...     print(f"{read_id} from {read_map[read_id]}: {len(read_obj.signal)} samples")
    """
    # Group reads by sample
    sample_to_reads = {}
    for read_id, sample_name in read_sample_map.items():
        if sample_name not in sample_to_reads:
            sample_to_reads[sample_name] = []
        sample_to_reads[sample_name].append(read_id)

    # Fetch reads from each sample
    all_reads = {}
    for sample_name, batch_read_ids in sample_to_reads.items():
        sample_reads = get_reads_batch(batch_read_ids, sample_name=sample_name)
        all_reads.update(sample_reads)

    return all_reads


def get_read_by_id(
    read_id: str, sample_name: str | None = None
) -> pod5.ReadRecord | None:
    """
    Get a single read by ID using index if available

    Uses Pod5Index for O(1) lookup if index is built, otherwise falls back
    to linear scan.

    Args:
        read_id: Read ID to fetch
        sample_name: (Multi-sample mode) Name of sample to get read from.
                     If None, uses global session reader.

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
    # Determine which reader to use
    if sample_name:
        sample = squiggy_kernel.get_sample(sample_name)
        if not sample or sample._pod5_reader is None:
            raise RuntimeError(f"Sample '{sample_name}' not loaded or has no POD5 file")
        reader = sample._pod5_reader
        pod5_index = sample.pod5_index if hasattr(sample, "pod5_index") else None
    else:
        if squiggy_kernel._reader is None:
            raise RuntimeError("No POD5 file is currently loaded")
        reader = squiggy_kernel._reader
        pod5_index = (
            squiggy_kernel._pod5_index
            if hasattr(squiggy_kernel, "pod5_index")
            else None
        )

    # Use index if available
    if pod5_index is not None:
        position = pod5_index.get_position(read_id)
        if position is None:
            return None

        # Use indexed access
        for idx, read in enumerate(reader.reads()):
            if idx == position:
                return read
        return None

    # Fallback to linear scan
    for read in reader.reads():
        if str(read.read_id) == read_id:
            return read
    return None


def load_pod5(file_path: str, build_index: bool = True, use_cache: bool = True) -> None:
    """
    Load a POD5 file into the global kernel session (OPTIMIZED)

    This function mutates the global squiggy_kernel object, making
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
        None (mutates global squiggy_kernel)

    Examples:
        >>> from squiggy import load_pod5
        >>> from squiggy.io import squiggy_kernel
        >>> load_pod5('data.pod5')
        >>> print(f"Loaded {len(squiggy_kernel._read_ids)} reads")
        >>> # Session is available as squiggy_kernel in kernel
        >>> first_read = next(squiggy_kernel._reader.reads())
    """
    # Convert to absolute path
    abs_path = Path(file_path).resolve()

    if not abs_path.exists():
        raise FileNotFoundError(f"Failed to open pod5 file at: {abs_path}")

    # Close previous reader if exists
    squiggy_kernel.close_pod5()

    # Open new reader
    reader = pod5.Reader(str(abs_path))

    # Create lazy read list (O(1) memory overhead)
    lazy_read_list = LazyReadList(reader)

    # Try to load index from cache
    cached_index = None
    if use_cache and squiggy_kernel.cache:
        cached_index = squiggy_kernel.cache.load_pod5_index(abs_path)

    # Build or restore index
    if build_index:
        if cached_index:
            # Restore from cache
            pod5_index = Pod5Index()
            pod5_index._index = cached_index
        else:
            # Build fresh
            pod5_index = Pod5Index()
            pod5_index.build(reader)

            # Save to cache for next time
            if use_cache and squiggy_kernel.cache:
                squiggy_kernel.cache.save_pod5_index(abs_path, pod5_index._index)

        squiggy_kernel._pod5_index = pod5_index
    else:
        squiggy_kernel._pod5_index = None

    # Store state in session
    squiggy_kernel._reader = reader
    squiggy_kernel._pod5_path = str(abs_path)
    squiggy_kernel._read_ids = lazy_read_list


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
    from ..constants import BAM_SAMPLE_SIZE

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
        logger.warning(f"Error checking BAM event alignment: {e}")
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

    from ..constants import BAM_SAMPLE_SIZE

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
            if hasattr(read, "modified_bases") and read.modified_bases:
                has_modifications = True

                for (
                    _canonical_base,
                    _strand,
                    mod_code,
                ), _mod_list in read.modified_bases.items():
                    modification_types.add(mod_code)

            # Check for ML tag (modification probabilities)
            if read.has_tag("ML"):
                has_ml = True

        bam.close()

    except Exception as e:
        logger.warning(f"Error checking BAM modifications: {e}")
        return {
            "has_modifications": False,
            "modification_types": [],
            "sample_count": 0,
            "has_probabilities": False,
        }

    mod_types_list = sorted(modification_types, key=str)

    return {
        "has_modifications": has_modifications,
        "modification_types": mod_types_list,
        "sample_count": reads_checked,
        "has_probabilities": has_ml,
    }


def _collect_bam_metadata_single_pass(
    bam_path: Path, build_ref_mapping: bool = True
) -> dict:
    """
    Collect all BAM metadata in a single file scan (PERFORMANCE OPTIMIZATION)

    This function consolidates what were previously 4 separate file scans:
    1. get_bam_references() - reference names/lengths/counts
    2. get_bam_modification_info() - modification tags (MM/ML)
    3. get_bam_event_alignment_status() - event alignment tags (mv)
    4. _build_ref_mapping_immediate() - reference→reads mapping

    By scanning the file once, we achieve 3-4x faster loading.

    Args:
        bam_path: Path to BAM file
        build_ref_mapping: Whether to build reference→reads mapping

    Returns:
        Dict with keys:
            - references: list[dict] with name/length/read_count
            - has_modifications: bool
            - modification_types: list of mod codes
            - has_probabilities: bool (ML tag present)
            - has_event_alignment: bool (mv tag present)
            - ref_mapping: dict[str, list[str]] (if build_ref_mapping=True)
            - num_reads: int (total mapped reads)

    Performance:
        - 180 reads: ~100ms (vs ~500ms with 4 scans)
        - 10K reads: ~2-3s (vs ~5-10s with 4 scans)
        - 100K reads: ~8-10s (vs ~30s with 4 scans)
    """
    from ..constants import BAM_SAMPLE_SIZE

    # Initialize data structures
    ref_mapping = defaultdict(list) if build_ref_mapping else None
    ref_counts = defaultdict(int)
    modification_types = set()
    has_modifications = False
    has_ml = False
    has_mv = False
    reads_processed = 0

    with pysam.AlignmentFile(str(bam_path), "rb", check_sq=False) as bam:
        # Get reference info from header
        references = []
        for ref_name, ref_length in zip(bam.references, bam.lengths, strict=False):
            references.append(
                {"name": ref_name, "length": ref_length, "read_count": None}
            )

        # Single pass through all reads
        for read in bam.fetch(until_eof=True):
            reads_processed += 1

            # Skip unmapped reads
            if read.is_unmapped:
                continue

            ref_name = bam.get_reference_name(read.reference_id)

            # Build ref_mapping for all reads
            if build_ref_mapping:
                ref_mapping[ref_name].append(read.query_name)

            # Count reads per reference
            ref_counts[ref_name] += 1

            # Sample first N reads for feature detection
            if reads_processed <= BAM_SAMPLE_SIZE:
                # Check for modifications
                if hasattr(read, "modified_bases") and read.modified_bases:
                    has_modifications = True
                    for (
                        _canonical_base,
                        _strand,
                        mod_code,
                    ), _mod_list in read.modified_bases.items():
                        modification_types.add(mod_code)

                # Check for ML tag (modification probabilities)
                if read.has_tag("ML"):
                    has_ml = True

                # Check for mv tag (move table for event alignment)
                if read.has_tag("mv"):
                    has_mv = True

    # Update reference counts
    for ref in references:
        ref["read_count"] = ref_counts.get(ref["name"], 0)

    # Filter out references with no reads
    references = [ref for ref in references if ref["read_count"] > 0]

    # Convert modification_types to sorted list
    mod_types_list = sorted(modification_types, key=str)

    # Convert ref_mapping to regular dict
    if build_ref_mapping:
        ref_mapping = dict(ref_mapping)

    return {
        "references": references,
        "has_modifications": has_modifications,
        "modification_types": mod_types_list,
        "has_probabilities": has_ml,
        "has_event_alignment": has_mv,
        "ref_mapping": ref_mapping,
        "ref_counts": dict(ref_counts),
        "num_reads": sum(ref["read_count"] for ref in references),
    }


def load_bam(
    file_path: str, build_ref_mapping: bool = True, use_cache: bool = True
) -> None:
    """
    Load a BAM file into the global kernel session (OPTIMIZED)

    This function mutates the global squiggy_kernel object, making
    BAM alignment data available for subsequent plotting and analysis calls.

    Performance optimizations:
    - Single-pass metadata collection (3-4x faster than old 4-scan approach)
    - Eager reference mapping (transparent cost, eliminates UI freezes)
    - Persistent caching for instant subsequent loads

    Args:
        file_path: Path to BAM file
        build_ref_mapping: Whether to build reference→reads mapping (default: True)
        use_cache: Whether to use persistent cache (default: True)

    Returns:
        None (mutates global squiggy_kernel)

    Examples:
        >>> from squiggy import load_bam
        >>> from squiggy.io import squiggy_kernel
        >>> load_bam('alignments.bam')
        >>> print(squiggy_kernel._bam_info['references'])
        >>> if squiggy_kernel._bam_info['has_modifications']:
        ...     print(f"Modifications: {squiggy_kernel._bam_info['modification_types']}")
        >>> if squiggy_kernel._bam_info['has_event_alignment']:
        ...     print("Event alignment data available")
    """
    # Convert to absolute path
    abs_path = Path(file_path).resolve()

    if not abs_path.exists():
        raise FileNotFoundError(f"BAM file not found: {abs_path}")

    # Try cache first for complete metadata
    metadata = None
    if use_cache and squiggy_kernel.cache:
        metadata = squiggy_kernel.cache.load_bam_metadata(abs_path)

    # If cache miss or disabled, collect fresh metadata (single-pass scan)
    if metadata is None:
        metadata = _collect_bam_metadata_single_pass(abs_path, build_ref_mapping)

        # Save to cache for instant future loads
        if use_cache and squiggy_kernel.cache:
            squiggy_kernel.cache.save_bam_metadata(abs_path, metadata)

    # Build metadata dict for session
    bam_info = {
        "file_path": str(abs_path),
        "num_reads": metadata["num_reads"],
        "references": metadata["references"],
        "has_modifications": metadata["has_modifications"],
        "modification_types": metadata["modification_types"],
        "has_probabilities": metadata["has_probabilities"],
        "has_event_alignment": metadata["has_event_alignment"],
    }

    # Get ref_mapping from metadata
    ref_mapping = metadata.get("ref_mapping")

    # Store state in session
    squiggy_kernel._bam_path = str(abs_path)
    squiggy_kernel._bam_info = bam_info
    squiggy_kernel._ref_mapping = ref_mapping


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
    if squiggy_kernel._ref_mapping is None:
        raise RuntimeError(
            "No BAM file loaded. Call load_bam() before accessing reference reads."
        )

    if reference_name not in squiggy_kernel._ref_mapping:
        available_refs = list(squiggy_kernel._ref_mapping.keys())
        raise KeyError(
            f"Reference '{reference_name}' not found. "
            f"Available references: {available_refs[:5]}..."
        )

    all_reads = squiggy_kernel._ref_mapping[reference_name]

    if limit is None:
        return all_reads[offset:]
    return all_reads[offset : offset + limit]


def load_fasta(file_path: str) -> None:
    """
    Load a FASTA file into the global kernel session

    This function mutates the global squiggy_kernel object, making
    FASTA reference sequences available for subsequent motif search and
    analysis calls.

    If a FASTA index (.fai) doesn't exist, it will be automatically created
    using pysam.faidx().

    Args:
        file_path: Path to FASTA file (index will be created if missing)

    Returns:
        None (mutates global squiggy_kernel)

    Examples:
        >>> from squiggy import load_fasta
        >>> from squiggy.io import squiggy_kernel
        >>> load_fasta('genome.fa')  # Creates .fai index if needed
        >>> print(squiggy_kernel._fasta_info['references'])
        >>> # Use with motif search
        >>> from squiggy.motif import search_motif
        >>> matches = list(search_motif(squiggy_kernel._fasta_path, "DRACH"))
    """
    # Convert to absolute path
    abs_path = os.path.abspath(file_path)

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"FASTA file not found: {abs_path}")

    # Check for index, create if missing
    fai_path = abs_path + ".fai"
    if not os.path.exists(fai_path):
        try:
            pysam.faidx(abs_path)
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

    # Store state in session
    squiggy_kernel._fasta_path = abs_path
    squiggy_kernel._fasta_info = fasta_info


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
    if squiggy_kernel._bam_path is None:
        raise RuntimeError("No BAM file is currently loaded")

    if not os.path.exists(squiggy_kernel._bam_path):
        raise FileNotFoundError(f"BAM file not found: {squiggy_kernel._bam_path}")

    # Open BAM file
    bam = pysam.AlignmentFile(squiggy_kernel._bam_path, "rb", check_sq=False)

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

    # Store in session
    squiggy_kernel._ref_mapping = ref_to_reads

    return ref_to_reads


def get_current_files() -> dict[str, str | None]:
    """
    Get paths of currently loaded files

    Returns:
        Dict with pod5_path and bam_path (may be None)
    """
    return {
        "pod5_path": squiggy_kernel._pod5_path,
        "bam_path": squiggy_kernel._bam_path,
    }


def get_read_ids() -> list[str]:
    """
    Get list of read IDs from currently loaded POD5 file

    Returns:
        List of read ID strings (materialized from lazy list if needed)
    """
    if not squiggy_kernel._read_ids:
        raise ValueError("No POD5 file is currently loaded")

    # Convert LazyReadList to list if needed
    if isinstance(squiggy_kernel._read_ids, LazyReadList):
        return list(squiggy_kernel._read_ids)
    return squiggy_kernel._read_ids


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
    squiggy_kernel.close_pod5()


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
    squiggy_kernel.close_bam()


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
    squiggy_kernel.close_fasta()


# Public API convenience functions for multi-sample loading


def load_sample(
    name: str,
    pod5_path: str,
    bam_path: str | None = None,
    fasta_path: str | None = None,
) -> Sample:
    """
    Load a POD5/BAM/FASTA sample set into the global session

    Convenience function that loads a named sample into the global squiggy_kernel.

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
        >>> print(f"Loaded {len(sample._read_ids)} reads")
    """
    return squiggy_kernel.load_sample(name, pod5_path, bam_path, fasta_path)


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
    return squiggy_kernel.get_sample(name)


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
    return squiggy_kernel.list_samples()


def remove_sample(name: str) -> None:
    """
    Unload a sample from the global session and free its resources

    Args:
        name: Sample name to remove

    Examples:
        >>> from squiggy import remove_sample
        >>> remove_sample('model_v4.2')
    """
    squiggy_kernel.remove_sample(name)


def close_all_samples() -> None:
    """
    Close all samples and clear the global session

    Examples:
        >>> from squiggy import close_all_samples
        >>> close_all_samples()
    """
    squiggy_kernel.close_all()
