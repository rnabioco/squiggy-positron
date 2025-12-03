"""SquiggyKernel - manages kernel state for loaded POD5 and BAM files"""

import os
from pathlib import Path

import pod5

from .performance import LazyReadList, Pod5Index
from .samples import Sample, _load_sample_files


class SquiggyKernel:
    """
    Manages kernel state for loaded POD5 and BAM files, supporting multiple samples

    This kernel state manager handles multiple POD5/BAM pairs (samples) simultaneously,
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
        # Multi-sample support (NEW) - PUBLIC
        self.samples: dict[str, Sample] = {}

        # Single-sample properties (for backward compatibility) - INTERNAL
        self._reader: pod5.Reader | None = None
        self._pod5_path: str | None = None
        self._read_ids: list[str] | LazyReadList = []
        self._bam_path: str | None = None
        self._bam_info: dict | None = None
        self._ref_mapping: dict[str, list[str]] | None = None
        self._fasta_path: str | None = None
        self._fasta_info: dict | None = None

        # Performance optimization attributes (NEW) - INTERNAL
        self._pod5_index: Pod5Index | None = None

        # Cache integration (NEW) - PUBLIC
        from ..cache import SquiggyCache

        cache_path = Path(cache_dir) if cache_dir else None
        self.cache = SquiggyCache(cache_path, enabled=use_cache) if use_cache else None

    def __dir__(self):
        """Control what appears in Variables pane - only show public API"""
        return [
            "samples",
            "cache",
            "load_sample",
            "get_sample",
            "list_samples",
            "remove_sample",
            "close_all",
            "close_bam",
            "close_fasta",
            "close_pod5",
        ]

    def __repr__(self) -> str:
        """Return informative summary of loaded files"""
        if self.samples:
            # Multi-sample mode
            parts = [f"SquiggyKernel: {len(self.samples)} sample(s)"]
            for name in sorted(self.samples.keys()):
                sample = self.samples[name]
                if sample._pod5_path:
                    num_reads = len(sample._read_ids)
                    parts.append(f"  {name}: {num_reads:,} reads")
            return "<" + "\n".join(parts) + ">"
        else:
            # Single-sample backward compat mode
            parts = []

            if self._pod5_path:
                filename = os.path.basename(self._pod5_path)
                parts.append(f"POD5: {filename} ({len(self._read_ids):,} reads)")

            if self._bam_path:
                filename = os.path.basename(self._bam_path)
                num_reads = self._bam_info.get("num_reads", 0) if self._bam_info else 0
                parts.append(f"BAM: {filename} ({num_reads:,} reads)")

                if self._bam_info:
                    if self._bam_info.get("has_modifications"):
                        mod_types = ", ".join(
                            str(m) for m in self._bam_info["modification_types"]
                        )
                        parts.append(f"Modifications: {mod_types}")
                    if self._bam_info.get("has_event_alignment"):
                        parts.append("Event alignment: yes")

            if self._fasta_path:
                filename = os.path.basename(self._fasta_path)
                num_refs = (
                    len(self._fasta_info.get("references", []))
                    if self._fasta_info
                    else 0
                )
                parts.append(f"FASTA: {filename} ({num_refs:,} references)")

            if not parts:
                return "<SquiggyKernel: No files loaded>"

            return f"<SquiggyKernel: {' | '.join(parts)}>"

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
            >>> sk = SquiggyKernel()
            >>> sample = sk.load_sample('v4.2', 'data_v4.2.pod5', 'align_v4.2.bam')
            >>> print(f"Loaded {len(sample._read_ids)} reads")
        """
        # Close existing sample with this name if any
        if name in self.samples:
            self.samples[name].close()

        # Create new sample
        sample = Sample(name)

        # Load files into sample
        _load_sample_files(sample, pod5_path, bam_path, fasta_path)

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
            >>> sk = SquiggyKernel()
            >>> sample = sk.get_sample('model_v4.2')
        """
        sample = self.samples.get(name)
        return sample

    def list_samples(self) -> list[str]:
        """
        List all loaded sample names

        Returns:
            List of sample names in order they were loaded

        Examples:
            >>> sk = SquiggyKernel()
            >>> names = sk.list_samples()
            >>> print(f"Loaded samples: {names}")
        """
        return list(self.samples.keys())

    def remove_sample(self, name: str) -> None:
        """
        Unload a sample and free its resources

        Args:
            name: Sample name to remove

        Examples:
            >>> sk = SquiggyKernel()
            >>> sk.remove_sample('model_v4.2')
        """
        if name in self.samples:
            self.samples[name].close()
            del self.samples[name]

    # Backward compatibility methods

    def close_pod5(self):
        """Close POD5 reader and clear POD5 state (backward compat mode)"""
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        self._pod5_path = None
        self._read_ids = []
        self._pod5_index = None  # Clear index

    def close_bam(self):
        """Clear BAM state (backward compat mode)"""
        self._bam_path = None
        self._bam_info = None
        self._ref_mapping = None

    def close_fasta(self):
        """Clear FASTA state (backward compat mode)"""
        self._fasta_path = None
        self._fasta_info = None

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


# Global kernel state instance (single source of truth for kernel state)
squiggy_kernel = SquiggyKernel()
