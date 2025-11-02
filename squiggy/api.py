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
from .normalization import normalize_signal
from .plotter import SquigglePlotter


class Pod5File:
    """
    POD5 file reader with lazy loading

    Provides object-oriented interface to POD5 files without global state.
    Supports context manager protocol for automatic cleanup.

    Args:
        path: Path to POD5 file

    Example:
        >>> with Pod5File('data.pod5') as pod5:
        ...     for read in pod5.iter_reads(limit=5):
        ...         print(read.read_id)
    """

    def __init__(self, path: str | Path):
        """Open POD5 file for reading"""
        self.path = Path(path).resolve()

        if not self.path.exists():
            raise FileNotFoundError(f"POD5 file not found: {self.path}")

        # Open reader
        self._reader = pod5.Reader(str(self.path))

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

        Example:
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

        Example:
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

        Example:
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
        downsample: int = 1,
        show_dwell_time: bool = False,
        show_labels: bool = True,
        position_label_interval: int = 100,
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

        Example:
            >>> fig = read.plot(mode='EVENTALIGN', bam_file=bam)
            >>> fig.title.text = "My Custom Title"
            >>> from bokeh.plotting import show
            >>> show(fig)
        """
        # Parse parameters
        plot_mode = PlotMode[mode.upper()]
        norm_method = NormalizationMethod[normalization.upper()]
        theme_enum = Theme[theme.upper()]

        # Get alignment if needed
        aligned_read = None
        sequence = None
        seq_to_sig_map = None
        modifications = None

        if plot_mode == PlotMode.EVENTALIGN:
            if bam_file is None:
                raise ValueError("EVENTALIGN mode requires bam_file parameter")

            aligned_read = self.get_alignment(bam_file)
            if aligned_read is None:
                raise ValueError(
                    f"Read {self.read_id} not found in BAM or has no move table"
                )

            sequence = aligned_read.sequence
            if aligned_read.bases:
                seq_to_sig_map = [ann.signal_start for ann in aligned_read.bases]
            if hasattr(aligned_read, "modifications"):
                modifications = aligned_read.modifications

        # Generate plot - returns (html, figure) tuple
        _, fig = SquigglePlotter.plot_single_read(
            signal=self.signal,
            read_id=self.read_id,
            sample_rate=self.sample_rate,
            sequence=sequence,
            seq_to_sig_map=seq_to_sig_map,
            normalization=norm_method,
            downsample=downsample,
            show_dwell_time=show_dwell_time,
            show_labels=show_labels,
            show_signal_points=show_signal_points,
            modifications=modifications,
            scale_dwell_time=scale_dwell_time,
            min_mod_probability=min_mod_probability,
            enabled_mod_types=enabled_mod_types,
            theme=theme_enum,
        )

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

    Example:
        >>> with BamFile('alignments.bam') as bam:
        ...     alignment = bam.get_alignment('read_001')
        ...     print(alignment.sequence)
    """

    def __init__(self, path: str | Path):
        """Open BAM file for reading"""
        self.path = Path(path).resolve()

        if not self.path.exists():
            raise FileNotFoundError(f"BAM file not found: {self.path}")

        # Check for index
        bai_path = Path(str(self.path) + ".bai")
        if not bai_path.exists():
            # Try alternate index location
            alt_bai = self.path.with_suffix(".bam.bai")
            if not alt_bai.exists():
                print("Warning: BAM index not found. Region queries may not work.")

        # Open BAM file
        self._bam = pysam.AlignmentFile(str(self.path), "rb", check_sq=False)

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

        Example:
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

        Example:
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

        Example:
            >>> mod_info = bam.get_modifications_info()
            >>> if mod_info['has_modifications']:
            ...     print(f"Found modifications: {mod_info['modification_types']}")
        """
        if self._mod_info is not None:
            return self._mod_info

        # Import here to avoid circular dependency
        from .io import get_bam_modification_info

        self._mod_info = get_bam_modification_info(str(self.path))
        return self._mod_info

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

    Example:
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
    "figure_to_html",
]
