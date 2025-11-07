"""
Object-oriented API for squiggy

This module provides notebook-friendly classes for working with POD5 and BAM files.
Unlike the global state functions in io.py, these classes allow:
- Direct access to signals, alignments, and modifications
- Lazy iteration over reads
- Customization of Bokeh plots before display
- Multiple file instances in the same session

Example usage:
    >>> import squiggy
    >>> pod5 = squiggy.Pod5File('data.pod5')
    >>> bam = squiggy.BamFile('alignments.bam')
    >>>
    >>> # Get a read and access its data
    >>> read = pod5.get_read(pod5.read_ids[0])
    >>> signal = read.signal  # numpy array
    >>> normalized = read.get_normalized('ZNORM')
    >>>
    >>> # Get alignment
    >>> alignment = read.get_alignment(bam)
    >>>
    >>> # Plot and customize
    >>> fig = read.plot(mode='EVENTALIGN', bam_file=bam)
    >>> fig.title.text = "Custom Title"
    >>> from bokeh.plotting import show
    >>> show(fig)
    >>>
    >>> # Lazy iteration
    >>> for read in pod5.iter_reads(limit=10):
    >>>     print(f"{read.read_id}: {len(read.signal)} samples")
    >>>
    >>> pod5.close()
    >>> bam.close()
"""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pod5
import pysam
from bokeh.plotting import figure as BokehFigure

from .alignment import AlignedRead, extract_alignment_from_bam
from .constants import NormalizationMethod, PlotMode, Theme
from .motif import MotifMatch, search_motif
from .normalization import normalize_signal
from .plot_factory import create_plot_strategy


class Pod5File:
    """
    POD5 file reader with lazy loading

    Provides object-oriented interface to POD5 files without global state.
    Supports context manager protocol for automatic cleanup.

    Args:
        path: Path to POD5 file

    Examples:
        >>> with Pod5File('data.pod5') as pod5:
        ...     for read in pod5.iter_reads(limit=5):
        ...         print(read.read_id)
    """

    def __init__(self, path: str | Path):
        """Open POD5 file for reading"""
        resolved_path = Path(path).resolve()

        if not resolved_path.exists():
            raise FileNotFoundError(f"POD5 file not found: {resolved_path}")

        # Store as string to avoid Path object in variables pane
        self.path = str(resolved_path)

        # Open reader
        self._reader = pod5.Reader(self.path)

        # Cache read IDs (lazy loaded on first access)
        self._read_ids: list[str] | None = None

    @property
    def read_ids(self) -> list[str]:
        """Get list of all read IDs in the file"""
        if self._read_ids is None:
            self._read_ids = [str(read.read_id) for read in self._reader.reads()]
        return self._read_ids

    def __len__(self) -> int:
        """Return number of reads in file"""
        return len(self.read_ids)

    def get_read(self, read_id: str) -> "Read":
        """
        Get a single read by ID

        Args:
            read_id: Read identifier

        Returns:
            Read object

        Raises:
            ValueError: If read ID not found
        """
        for read_obj in self._reader.reads():
            if str(read_obj.read_id) == read_id:
                return Read(read_obj, self)

        raise ValueError(f"Read not found: {read_id}")

    def iter_reads(self, limit: int | None = None) -> Iterator["Read"]:
        """
        Iterate over reads (lazy loading)

        Args:
            limit: Maximum number of reads to return (None = all)

        Yields:
            Read objects

        Examples:
            >>> for read in pod5.iter_reads(limit=100):
            ...     print(f"{read.read_id}: {len(read.signal)} samples")
        """
        count = 0
        for read_obj in self._reader.reads():
            if limit is not None and count >= limit:
                break
            yield Read(read_obj, self)
            count += 1

    def close(self):
        """Close the POD5 file"""
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
        return False

    def __repr__(self) -> str:
        return f"Pod5File(path='{self.path}', num_reads={len(self)})"


class Read:
    """
    A single POD5 read with signal data

    Provides access to raw signal, normalization, alignment, and plotting.

    Attributes:
        read_id: Read identifier
        signal: Raw signal data (numpy array)
        sample_rate: Sampling rate in Hz
    """

    def __init__(self, pod5_read: pod5.ReadRecord, parent_file: Pod5File):
        """
        Initialize Read from pod5.Read object

        Args:
            pod5_read: pod5.ReadRecord object
            parent_file: Parent Pod5File object
        """
        self._parent = parent_file

        # Cache all properties immediately (pod5 read handle is temporary)
        self._read_id = str(pod5_read.read_id)
        self._signal = pod5_read.signal  # Must cache now, handle closes later
        self._sample_rate = pod5_read.run_info.sample_rate

    @property
    def read_id(self) -> str:
        """Read identifier"""
        return self._read_id

    @property
    def signal(self) -> np.ndarray:
        """Raw signal data as numpy array"""
        return self._signal

    @property
    def sample_rate(self) -> int:
        """Sampling rate in Hz"""
        return self._sample_rate

    def get_normalized(self, method: str | NormalizationMethod = "ZNORM") -> np.ndarray:
        """
        Get normalized signal

        Args:
            method: Normalization method ('NONE', 'ZNORM', 'MEDIAN', 'MAD')

        Returns:
            Normalized signal as numpy array

        Examples:
            >>> read = pod5.get_read('read_001')
            >>> znorm_signal = read.get_normalized('ZNORM')
            >>> mad_signal = read.get_normalized('MAD')
        """
        if isinstance(method, str):
            method = NormalizationMethod[method.upper()]

        return normalize_signal(self.signal, method)

    def get_alignment(
        self, bam_file: "BamFile | None" = None, bam_path: str | Path | None = None
    ) -> AlignedRead | None:
        """
        Get alignment information from BAM file

        Args:
            bam_file: BamFile object (recommended)
            bam_path: Path to BAM file (alternative to bam_file)

        Returns:
            AlignedRead object or None if not found or no move table

        Examples:
            >>> bam = BamFile('alignments.bam')
            >>> alignment = read.get_alignment(bam)
            >>> if alignment:
            ...     print(f"Aligned to {alignment.chromosome}:{alignment.genomic_start}")
        """
        if bam_file is None and bam_path is None:
            raise ValueError("Must provide either bam_file or bam_path")

        path = bam_path if bam_path is not None else bam_file.path
        return extract_alignment_from_bam(Path(path), self.read_id)

    def plot(
        self,
        mode: str = "SINGLE",
        normalization: str = "ZNORM",
        theme: str = "LIGHT",
        downsample: int = None,
        show_dwell_time: bool = False,
        show_labels: bool = True,
        position_label_interval: int = None,
        scale_dwell_time: bool = False,
        min_mod_probability: float = 0.5,
        enabled_mod_types: list | None = None,
        show_signal_points: bool = False,
        bam_file: "BamFile | None" = None,
    ) -> BokehFigure:
        """
        Generate Bokeh plot for this read

        Args:
            mode: Plot mode ('SINGLE', 'EVENTALIGN')
            normalization: Normalization method ('NONE', 'ZNORM', 'MEDIAN', 'MAD')
            theme: Color theme ('LIGHT', 'DARK')
            downsample: Downsampling factor (1 = no downsampling)
            show_dwell_time: Color bases by dwell time (EVENTALIGN mode)
            show_labels: Show base labels (EVENTALIGN mode)
            position_label_interval: Interval for position labels
            scale_dwell_time: Scale x-axis by cumulative dwell time
            min_mod_probability: Minimum probability for showing modifications
            enabled_mod_types: List of modification types to show (None = all)
            show_signal_points: Show individual signal points
            bam_file: BamFile object (required for EVENTALIGN mode)

        Returns:
            Bokeh Figure object (can be customized before display)

        Examples:
            >>> fig = read.plot(mode='EVENTALIGN', bam_file=bam)
            >>> fig.title.text = "My Custom Title"
            >>> from bokeh.plotting import show
            >>> show(fig)
        """
        from .constants import DEFAULT_DOWNSAMPLE, DEFAULT_POSITION_LABEL_INTERVAL

        # Apply defaults if not specified
        if downsample is None:
            downsample = DEFAULT_DOWNSAMPLE
        if position_label_interval is None:
            position_label_interval = DEFAULT_POSITION_LABEL_INTERVAL

        # Parse parameters
        plot_mode = PlotMode[mode.upper()]
        norm_method = NormalizationMethod[normalization.upper()]
        theme_enum = Theme[theme.upper()]

        # Get alignment if needed
        aligned_read = None

        if plot_mode == PlotMode.EVENTALIGN:
            if bam_file is None:
                raise ValueError("EVENTALIGN mode requires bam_file parameter")

            aligned_read = self.get_alignment(bam_file)
            if aligned_read is None:
                raise ValueError(
                    f"Read {self.read_id} not found in BAM or has no move table. "
                    f"Read may be unmapped or BAM may not contain event alignment data."
                )

        # Generate plot using plot strategy
        if plot_mode == PlotMode.SINGLE:
            data = {
                "signal": self.signal,
                "read_id": self.read_id,
                "sample_rate": self.sample_rate,
            }
            options = {
                "normalization": norm_method,
                "downsample": downsample,
                "show_signal_points": show_signal_points,
                "x_axis_mode": "dwell_time" if scale_dwell_time else "regular_time",
            }
        elif plot_mode == PlotMode.EVENTALIGN:
            data = {
                "reads": [(self.read_id, self.signal, self.sample_rate)],
                "aligned_reads": [aligned_read],
            }
            options = {
                "normalization": norm_method,
                "downsample": downsample,
                "show_dwell_time": scale_dwell_time,
                "show_labels": show_labels,
                "show_signal_points": show_signal_points,
            }
        else:
            raise ValueError(
                f"Unsupported plot mode for single read: {plot_mode}. "
                f"Supported modes: SINGLE, EVENTALIGN"
            )

        strategy = create_plot_strategy(plot_mode, theme_enum)
        _, fig = strategy.create_plot(data, options)

        return fig

    def __repr__(self) -> str:
        return f"Read(read_id='{self.read_id}', signal_length={len(self.signal)})"


class BamFile:
    """
    BAM alignment file reader

    Provides access to alignments, references, and base modifications.
    Supports context manager protocol for automatic cleanup.

    Args:
        path: Path to BAM file (must be indexed with .bai)

    Examples:
        >>> with BamFile('alignments.bam') as bam:
        ...     alignment = bam.get_alignment('read_001')
        ...     print(alignment.sequence)
    """

    def __init__(self, path: str | Path):
        """Open BAM file for reading"""
        resolved_path = Path(path).resolve()

        if not resolved_path.exists():
            raise FileNotFoundError(f"BAM file not found: {resolved_path}")

        # Store as string to avoid Path object in variables pane
        self.path = str(resolved_path)

        # Check for index (silently - user will get error if region queries fail)
        bai_path = Path(self.path + ".bai")
        if not bai_path.exists():
            # Try alternate index location
            alt_bai = Path(self.path).with_suffix(".bam.bai")
            if not alt_bai.exists():
                pass  # Index not found - queries may not work

        # Open BAM file
        self._bam = pysam.AlignmentFile(self.path, "rb", check_sq=False)

        # Cache references
        self._references: list[str] | None = None
        self._mod_info: dict | None = None

    @property
    def references(self) -> list[str]:
        """Get list of reference sequence names"""
        if self._references is None:
            self._references = (
                list(self._bam.references) if self._bam.references else []
            )
        return self._references

    def get_alignment(self, read_id: str) -> AlignedRead | None:
        """
        Get alignment for a specific read

        Args:
            read_id: Read identifier

        Returns:
            AlignedRead object or None if not found or no move table

        Examples:
            >>> alignment = bam.get_alignment('read_001')
            >>> if alignment:
            ...     for base in alignment.bases:
            ...         print(f"{base.base} at signal {base.signal_start}-{base.signal_end}")
        """
        return extract_alignment_from_bam(self.path, read_id)

    def iter_region(
        self, chrom: str, start: int | None = None, end: int | None = None
    ) -> Iterator[AlignedRead]:
        """
        Iterate over alignments in a genomic region

        Args:
            chrom: Chromosome/reference name
            start: Start position (0-based, inclusive)
            end: End position (0-based, exclusive)

        Yields:
            AlignedRead objects that have move tables

        Examples:
            >>> for alignment in bam.iter_region('chr1', 1000, 2000):
            ...     print(f"{alignment.read_id} at {alignment.genomic_start}")
        """
        from .alignment import _parse_alignment

        for pysam_read in self._bam.fetch(chrom, start, end):
            aligned_read = _parse_alignment(pysam_read)
            if aligned_read is not None:  # Only yield if move table present
                yield aligned_read

    def get_modifications_info(self) -> dict:
        """
        Check if BAM contains base modification tags

        Returns:
            Dict with:
                - has_modifications: bool
                - modification_types: list of modification codes
                - sample_count: number of reads checked
                - has_probabilities: bool (ML tag present)

        Examples:
            >>> mod_info = bam.get_modifications_info()
            >>> if mod_info['has_modifications']:
            ...     print(f"Found modifications: {mod_info['modification_types']}")
        """
        if self._mod_info is not None:
            return self._mod_info

        # Import here to avoid circular dependency
        from .io import get_bam_modification_info

        self._mod_info = get_bam_modification_info(self.path)
        return self._mod_info

    def get_reads_overlapping_motif(
        self,
        fasta_file: "FastaFile | str | Path",
        motif: str,
        region: str | None = None,
        strand: str = "both",
    ) -> dict[str, list[AlignedRead]]:
        """
        Find reads overlapping motif positions

        Args:
            fasta_file: FastaFile object or path to indexed FASTA file
            motif: IUPAC nucleotide pattern (e.g., "DRACH")
            region: Optional region filter ("chrom:start-end")
            strand: Motif search strand ('+', '-', or 'both')

        Returns:
            Dict mapping motif position keys to lists of AlignedRead objects
            Position key format: "chrom:position:strand"

        Examples:
            >>> bam = BamFile('alignments.bam')
            >>> fasta = FastaFile('genome.fa')
            >>> overlaps = bam.get_reads_overlapping_motif(fasta, 'DRACH', region='chr1:1000-2000')
            >>> for position, reads in overlaps.items():
            ...     print(f"{position}: {len(reads)} reads")
            ...     for read in reads:
            ...         print(f"  {read.read_id}")
        """
        from .alignment import _parse_alignment

        # Handle fasta_file parameter
        if isinstance(fasta_file, FastaFile):
            fasta_path = fasta_file.path
        else:
            fasta_path = Path(fasta_file).resolve()

        # Search for motif matches
        matches = list(search_motif(fasta_path, motif, region, strand))

        # Build dict of motif position -> overlapping reads
        overlaps: dict[str, list[AlignedRead]] = {}

        for match in matches:
            # Create position key
            position_key = f"{match.chrom}:{match.position}:{match.strand}"

            # Find reads overlapping this position
            reads_at_position = []

            # Query BAM for reads overlapping motif region
            for pysam_read in self._bam.fetch(match.chrom, match.position, match.end):
                aligned_read = _parse_alignment(pysam_read)
                if aligned_read is not None:  # Only include if has move table
                    reads_at_position.append(aligned_read)

            if reads_at_position:
                overlaps[position_key] = reads_at_position

        return overlaps

    def close(self):
        """Close the BAM file"""
        if self._bam is not None:
            self._bam.close()
            self._bam = None

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
        return False

    def __repr__(self) -> str:
        num_refs = len(self.references)
        return f"BamFile(path='{self.path}', num_references={num_refs})"


class FastaFile:
    """
    FASTA reference file reader with motif search capabilities

    Provides access to reference sequences and motif searching.
    Requires indexed FASTA file (.fai).

    Args:
        path: Path to FASTA file (must be indexed with .fai)

    Examples:
        >>> with FastaFile('genome.fa') as fasta:
        ...     # Search for DRACH motif
        ...     for match in fasta.search_motif('DRACH', region='chr1:1000-2000'):
        ...         print(f"{match.chrom}:{match.position} {match.sequence}")
    """

    def __init__(self, path: str | Path):
        """Open FASTA file for reading"""
        resolved_path = Path(path).resolve()

        if not resolved_path.exists():
            raise FileNotFoundError(f"FASTA file not found: {resolved_path}")

        # Store as string to avoid Path object in variables pane
        self.path = str(resolved_path)

        # Check for index
        fai_path = Path(self.path + ".fai")
        if not fai_path.exists():
            raise FileNotFoundError(
                f"FASTA index not found: {fai_path}. "
                f"Create with: samtools faidx {self.path}"
            )

        # Open FASTA file
        self._fasta = pysam.FastaFile(self.path)

        # Cache references
        self._references: list[str] | None = None

    @property
    def references(self) -> list[str]:
        """Get list of reference sequence names"""
        if self._references is None:
            self._references = list(self._fasta.references)
        return self._references

    def fetch(
        self, chrom: str, start: int | None = None, end: int | None = None
    ) -> str:
        """
        Fetch sequence from reference

        Args:
            chrom: Chromosome/reference name
            start: Start position (0-based, inclusive)
            end: End position (0-based, exclusive)

        Returns:
            DNA sequence string

        Examples:
            >>> fasta = FastaFile('genome.fa')
            >>> seq = fasta.fetch('chr1', 1000, 1100)
            >>> print(seq)  # 100 bp sequence
        """
        return self._fasta.fetch(chrom, start, end)

    def search_motif(
        self,
        motif: str,
        region: str | None = None,
        strand: str = "both",
    ) -> Iterator[MotifMatch]:
        """
        Search for motif matches in FASTA file

        Args:
            motif: IUPAC nucleotide pattern (e.g., "DRACH", "YGCY")
            region: Optional region filter ("chrom", "chrom:start", "chrom:start-end")
                    Positions are 1-based in input
            strand: Search strand ('+', '-', or 'both')

        Yields:
            MotifMatch objects for each match found

        Examples:
            >>> fasta = FastaFile('genome.fa')
            >>> matches = list(fasta.search_motif('DRACH', region='chr1:1000-2000'))
            >>> for match in matches:
            ...     print(f"{match.chrom}:{match.position+1} {match.sequence} ({match.strand})")
        """
        return search_motif(self.path, motif, region, strand)

    def count_motifs(
        self,
        motif: str,
        region: str | None = None,
        strand: str = "both",
    ) -> int:
        """
        Count total motif matches

        Args:
            motif: IUPAC nucleotide pattern
            region: Optional region filter
            strand: Search strand ('+', '-', or 'both')

        Returns:
            Total number of matches

        Examples:
            >>> fasta = FastaFile('genome.fa')
            >>> count = fasta.count_motifs('DRACH', region='chr1')
            >>> print(f"Found {count} DRACH motifs")
        """
        return sum(1 for _ in self.search_motif(motif, region, strand))

    def close(self):
        """Close the FASTA file"""
        if self._fasta is not None:
            self._fasta.close()
            self._fasta = None

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
        return False

    def __repr__(self) -> str:
        num_refs = len(self.references)
        return f"FastaFile(path='{self.path}', num_references={num_refs})"


def figure_to_html(fig: BokehFigure, title: str = "Squiggy Plot") -> str:
    """
    Convert Bokeh Figure to HTML string

    Utility function for converting Bokeh figures to standalone HTML.
    Useful when you need HTML output instead of interactive display.

    Args:
        fig: Bokeh Figure object
        title: HTML document title

    Returns:
        HTML string with embedded Bokeh plot

    Examples:
        >>> fig = read.plot()
        >>> html = figure_to_html(fig)
        >>> with open('plot.html', 'w') as f:
        ...     f.write(html)
    """
    from bokeh.embed import file_html
    from bokeh.resources import CDN

    return file_html(fig, CDN, title)


__all__ = [
    "Pod5File",
    "Read",
    "BamFile",
    "FastaFile",
    "figure_to_html",
]
