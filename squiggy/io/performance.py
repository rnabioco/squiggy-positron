"""Performance optimization classes for Squiggy I/O operations"""

from collections.abc import Iterator

from ..pod5_reader import Reader, read_count, read_ids


class LazyReadList:
    """
    Virtual list of read IDs backed by a POD5 reader's metadata

    Uses the reader's O(1) read count for ``len()`` and its read-ID metadata
    (no signal decompression) for ID access. The ID list is fetched once on
    first access and reused for subsequent slices.

    Attributes:
        _reader: POD5 Reader instance
        _materialized_ids: Optional fully materialized list (for search)

    Examples:
        >>> from squiggy.pod5_reader import open_reader
        >>> reader = open_reader('file.pod5')
        >>> lazy_list = LazyReadList(reader)
        >>> len(lazy_list)  # O(1) read count
        1000000
        >>> lazy_list[0:100]  # metadata only
        ['read1', 'read2', ...]
    """

    def __init__(self, reader: Reader):
        self._reader = reader
        self._materialized_ids: list[str] | None = None
        self._id_cache: list[str] | None = None

    def _ids(self) -> list[str]:
        if self._materialized_ids is not None:
            return self._materialized_ids
        if self._id_cache is None:
            self._id_cache = read_ids(self._reader)
        return self._id_cache

    def __len__(self) -> int:
        """Total read count (O(1))"""
        if self._materialized_ids is not None:
            return len(self._materialized_ids)
        return read_count(self._reader)

    def __getitem__(self, key: int | slice) -> str | list[str]:
        """Get read ID(s) at index/slice"""
        return self._ids()[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over all read IDs"""
        yield from self._ids()

    def materialize(self) -> list[str]:
        """
        Fully materialize the list (for search/caching)

        Returns:
            Complete list of all read IDs as strings
        """
        if self._materialized_ids is None:
            self._materialized_ids = list(self._ids())
            self._id_cache = None
        return self._materialized_ids

    def __repr__(self) -> str:
        """Return informative summary of lazy read list"""
        count = len(self)
        materialized = " (materialized)" if self._materialized_ids is not None else ""
        return f"<LazyReadList: {count:,} reads{materialized}>"
