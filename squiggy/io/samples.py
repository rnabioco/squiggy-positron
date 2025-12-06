"""Sample management for multi-sample workflows"""

import os
from pathlib import Path

import pod5
import pysam


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
        >>> print(f"{sample.name}: {len(sample._read_ids)} reads")
    """

    def __init__(self, name: str):
        """Initialize a new sample with the given name"""
        self.name = name
        self._pod5_path: str | None = None
        self._pod5_reader: pod5.Reader | None = None
        self._read_ids: list[str] = []
        self._bam_path: str | None = None
        self._bam_info: dict | None = None
        self._model_provenance: dict | None = None
        self._fasta_path: str | None = None
        self._fasta_info: dict | None = None

    def __repr__(self) -> str:
        """Return informative summary of sample state"""
        parts = [f"Sample({self.name})"]

        if self._pod5_path:
            filename = os.path.basename(self._pod5_path)
            parts.append(f"POD5: {filename} ({len(self._read_ids):,} reads)")

        if self._bam_path:
            filename = os.path.basename(self._bam_path)
            num_reads = self._bam_info.get("num_reads", 0) if self._bam_info else 0
            parts.append(f"BAM: {filename} ({num_reads:,} reads)")

        if self._fasta_path:
            filename = os.path.basename(self._fasta_path)
            num_refs = (
                len(self._fasta_info.get("references", [])) if self._fasta_info else 0
            )
            parts.append(f"FASTA: {filename} ({num_refs:,} references)")

        if len(parts) == 1:
            return f"<Sample({self.name}): No files loaded>"

        return f"<{' | '.join(parts)}>"

    def close(self):
        """Close all resources and clear sample state"""
        if self._pod5_reader is not None:
            self._pod5_reader.close()
            self._pod5_reader = None
        self._pod5_path = None
        self._read_ids = []
        self._bam_path = None
        self._bam_info = None
        self._model_provenance = None
        self._fasta_path = None
        self._fasta_info = None

    @property
    def pod5_path(self) -> str | None:
        """Path to loaded POD5 file"""
        return self._pod5_path

    @property
    def bam_path(self) -> str | None:
        """Path to loaded BAM file"""
        return self._bam_path

    @property
    def fasta_path(self) -> str | None:
        """Path to loaded FASTA file"""
        return self._fasta_path


def _load_sample_files(
    sample: Sample,
    pod5_path: str,
    bam_path: str | None = None,
    fasta_path: str | None = None,
) -> None:
    """
    Load POD5/BAM/FASTA files into a Sample object

    This is an internal function used by SquiggyKernel.load_sample().

    Args:
        sample: Sample object to populate
        pod5_path: Path to POD5 file
        bam_path: Path to BAM file (optional)
        fasta_path: Path to FASTA file (optional)
    """
    from .core import _collect_bam_metadata_single_pass

    # Load POD5
    abs_pod5_path = os.path.abspath(pod5_path)
    if not os.path.exists(abs_pod5_path):
        raise FileNotFoundError(f"POD5 file not found: {abs_pod5_path}")

    reader = pod5.Reader(abs_pod5_path)
    read_ids = [str(read.read_id) for read in reader.reads()]

    sample._pod5_path = abs_pod5_path
    sample._pod5_reader = reader
    sample._read_ids = read_ids

    # Load BAM if provided
    if bam_path:
        abs_bam_path = os.path.abspath(bam_path)
        if not os.path.exists(abs_bam_path):
            raise FileNotFoundError(f"BAM file not found: {abs_bam_path}")

        # Collect all metadata in a single pass
        metadata = _collect_bam_metadata_single_pass(
            Path(abs_bam_path), build_ref_mapping=True
        )

        # Build metadata dict
        bam_info = {
            "file_path": abs_bam_path,
            "num_reads": metadata["num_reads"],
            "references": metadata["references"],
            "has_modifications": metadata["has_modifications"],
            "modification_types": metadata["modification_types"],
            "has_probabilities": metadata["has_probabilities"],
            "has_event_alignment": metadata["has_event_alignment"],
            "ref_counts": metadata["ref_counts"],
            "ref_mapping": metadata["ref_mapping"],
        }

        sample._bam_path = abs_bam_path
        sample._bam_info = bam_info

        # Validate that POD5 and BAM have overlapping read IDs
        bam_read_ids = set()
        for ref_read_ids in metadata["ref_mapping"].values():
            bam_read_ids.update(ref_read_ids)

        pod5_read_ids = set(sample._read_ids)
        overlap = pod5_read_ids & bam_read_ids

        if len(overlap) == 0:
            raise ValueError(
                f"No overlapping read IDs found between POD5 ({len(pod5_read_ids)} reads) "
                f"and BAM ({len(bam_read_ids)} reads). These files appear to be mismatched. "
                f"Please verify that the BAM file contains alignments for reads in the POD5 file."
            )

    # Load FASTA if provided
    if fasta_path:
        abs_fasta_path = os.path.abspath(fasta_path)
        if not os.path.exists(abs_fasta_path):
            raise FileNotFoundError(f"FASTA file not found: {abs_fasta_path}")

        # Check for index, create if missing
        fai_path = abs_fasta_path + ".fai"
        if not os.path.exists(fai_path):
            try:
                pysam.faidx(abs_fasta_path)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to create FASTA index for {abs_fasta_path}. "
                    f"Error: {e}. You can also create it manually with: samtools faidx {abs_fasta_path}"
                ) from e

        # Open FASTA file to get metadata
        fasta = pysam.FastaFile(abs_fasta_path)

        try:
            references = list(fasta.references)
            lengths = list(fasta.lengths)

            fasta_info = {
                "file_path": abs_fasta_path,
                "references": references,
                "num_references": len(references),
                "reference_lengths": dict(zip(references, lengths, strict=True)),
            }
        finally:
            fasta.close()

        sample._fasta_path = abs_fasta_path
        sample._fasta_info = fasta_info

        # Validate that FASTA and BAM have matching references (if BAM is loaded)
        if sample._bam_info:
            bam_refs = {ref["name"] for ref in sample._bam_info["references"]}
            fasta_refs = set(references)
            overlap_refs = bam_refs & fasta_refs

            if len(overlap_refs) == 0:
                raise ValueError(
                    f"No overlapping reference names found between BAM ({len(bam_refs)} refs) "
                    f"and FASTA ({len(fasta_refs)} refs). These files appear to be mismatched. "
                    f"BAM references: {sorted(bam_refs)[:3]}, "
                    f"FASTA references: {sorted(fasta_refs)[:3]}."
                )
