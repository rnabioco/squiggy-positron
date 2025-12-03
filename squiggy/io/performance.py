"""Performance optimization classes for Squiggy I/O operations"""

from collections.abc import Iterator

import pod5


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
                self._cached_length = sum(1 for _ in self._reader.reads())
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
            self._materialized_ids = [
                str(read.read_id) for read in self._reader.reads()
            ]
            self._cached_length = len(self._materialized_ids)
        return self._materialized_ids

    def __repr__(self) -> str:
        """Return informative summary of lazy read list"""
        count = len(self)
        materialized = " (materialized)" if self._materialized_ids is not None else ""
        return f"<LazyReadList: {count:,} reads{materialized}>"


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
        for idx, read in enumerate(reader.reads()):
            self._index[str(read.read_id)] = idx

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

    def __repr__(self) -> str:
        """Return informative summary of index"""
        count = len(self._index)
        return f"<Pod5Index: {count:,} reads indexed>"
