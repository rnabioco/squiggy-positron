"""Performance optimization classes for Squiggy I/O operations"""

from collections.abc import Iterator

import pod5


class LazyReadList:
    """
    Virtual list of read IDs backed by pod5.Reader's Arrow metadata

    Uses pod5.Reader.num_reads (O(1)) for length and reader.read_ids
    (Arrow column read, no signal decompression) for ID access. This is
    orders of magnitude faster than iterating reader.reads() which
    decompresses signal data.

    Attributes:
        _reader: POD5 Reader instance
        _materialized_ids: Optional fully materialized list (for search)

    Examples:
        >>> reader = pod5.Reader('file.pod5')
        >>> lazy_list = LazyReadList(reader)
        >>> len(lazy_list)  # O(1) via reader.num_reads
        1000000
        >>> lazy_list[0:100]  # Arrow metadata read
        ['read1', 'read2', ...]
    """

    def __init__(self, reader: pod5.Reader):
        self._reader = reader
        self._materialized_ids: list[str] | None = None

    def __len__(self) -> int:
        """Total read count via reader.num_reads (O(1))"""
        if self._materialized_ids is not None:
            return len(self._materialized_ids)
        return self._reader.num_reads

    def __getitem__(self, key: int | slice) -> str | list[str]:
        """Get read ID(s) at index/slice via Arrow metadata"""
        if self._materialized_ids is not None:
            return self._materialized_ids[key]

        # Use reader.read_ids (Arrow metadata, no signal decompression)
        all_ids = self._reader.read_ids
        if isinstance(key, slice):
            return [str(rid) for rid in all_ids[key]]
        else:
            return str(all_ids[key])

    def __iter__(self) -> Iterator[str]:
        """Iterate over all read IDs"""
        if self._materialized_ids is not None:
            yield from self._materialized_ids
        else:
            for rid in self._reader.read_ids:
                yield str(rid)

    def materialize(self) -> list[str]:
        """
        Fully materialize the list (for search/caching)

        Uses reader.read_ids (Arrow metadata) instead of iterating reads().

        Returns:
            Complete list of all read IDs as strings
        """
        if self._materialized_ids is None:
            self._materialized_ids = [str(rid) for rid in self._reader.read_ids]
        return self._materialized_ids

    def __repr__(self) -> str:
        """Return informative summary of lazy read list"""
        count = len(self)
        materialized = " (materialized)" if self._materialized_ids is not None else ""
        return f"<LazyReadList: {count:,} reads{materialized}>"
