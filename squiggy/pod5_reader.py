"""POD5 access via escapepod

squiggy reads POD5 files with `escapepod <https://github.com/rnabioco/escapepod-rs>`_,
a Rust-backed POD5 reader. escapepod's read records (``escapepod.ReadData``)
carry metadata only: signal is fetched from the reader and the sample rate
lives on the run-info table. This module wraps that in a single
:class:`SignalRead` object (``read_id``, ``signal``, ``run_info``,
``sample_rate``) so the rest of the package does not depend on reader
internals.

Examples:
    >>> from squiggy.pod5_reader import open_reader, iter_reads
    >>> reader = open_reader("reads.pod5")
    >>> for read in iter_reads(reader, selection=["<read-uuid>"], missing_ok=True):
    ...     print(read.read_id, len(read.signal), read.sample_rate)
"""

import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from os import PathLike

import escapepod
import numpy as np

Reader = escapepod.Reader
ReadData = escapepod.ReadData
RunInfo = escapepod.RunInfo


@dataclass(frozen=True, eq=False)
class SignalRead:
    """A POD5 read with its raw signal loaded

    Attributes:
        read_id: Read UUID as a string
        signal: Raw (uncalibrated ADC) signal, int16
        run_info: Run information for the acquisition that produced the read
        record: Underlying escapepod read metadata
    """

    read_id: str
    signal: np.ndarray
    run_info: RunInfo
    record: ReadData

    @property
    def sample_rate(self) -> int:
        """Sampling rate in Hz"""
        return self.run_info.sample_rate


def open_reader(path: str | PathLike[str]) -> Reader:
    """Open a POD5 file for reading

    Raises:
        OSError: If the file cannot be opened
    """
    return escapepod.Reader(str(path))


def close_reader(reader: Reader | None) -> None:
    """Exit a reader opened with :func:`open_reader`

    escapepod readers have no ``close()``. This calls the context-manager
    ``__exit__``, which does not by itself release the file handle: the handle
    is released when the last reference to the reader is dropped, so callers
    must also discard their own reference. Safe to call on ``None``.
    """
    if reader is not None:
        reader.__exit__(None, None, None)


def read_ids(reader: Reader) -> list[str]:
    """All read IDs in file order (metadata only, no signal decoding)"""
    return reader.read_ids()


def read_count(reader: Reader) -> int:
    """Number of reads in the file (O(1))"""
    return reader.read_count


def _is_read_id(value: str) -> bool:
    try:
        uuid.UUID(value)
    except ValueError:
        return False
    return True


def iter_reads(
    reader: Reader,
    selection: Iterable[str] | None = None,
    missing_ok: bool = False,
) -> Iterator[SignalRead]:
    """Iterate over reads, loading each read's signal lazily

    Args:
        reader: An open reader
        selection: Read IDs to fetch (indexed lookup). ``None`` iterates the
            whole file in storage order.
        missing_ok: Skip IDs that are absent from the file (or are not valid
            read UUIDs) instead of raising.

    Yields:
        SignalRead objects, in file storage order

    Raises:
        KeyError: If ``missing_ok`` is False and a requested ID is absent
        ValueError: If ``missing_ok`` is False and an ID is not a valid UUID
    """
    if selection is not None:
        selection = [str(rid) for rid in selection]
        if missing_ok:
            selection = [rid for rid in selection if _is_read_id(rid)]
        if not selection:
            return

    records = reader.reads(selection=selection, missing_ok=missing_ok)
    run_infos = reader.run_infos
    for record in records:
        yield SignalRead(
            read_id=record.read_id,
            signal=reader.get_signal(record),
            run_info=run_infos[record.run_info_index],
            record=record,
        )


def get_read(reader: Reader, read_id: str) -> SignalRead | None:
    """Fetch a single read by ID, or ``None`` if it is not in the file"""
    for read in iter_reads(reader, selection=[read_id], missing_ok=True):
        return read
    return None
