"""Tests for the escapepod-backed POD5 adapter (squiggy.pod5_reader)"""

import numpy as np
import pytest

from squiggy.io import LazyReadList
from squiggy.pod5_reader import (
    SignalRead,
    close_reader,
    get_read,
    iter_reads,
    open_reader,
    read_count,
    read_ids,
)

MISSING_ID = "00000000-0000-0000-0000-000000000000"


@pytest.fixture
def reader(sample_pod5_file):
    r = open_reader(sample_pod5_file)
    yield r
    close_reader(r)


class TestReaderBasics:
    def test_read_ids_and_count(self, reader):
        ids = read_ids(reader)
        assert len(ids) == read_count(reader) > 0
        assert all(isinstance(rid, str) for rid in ids)

    def test_open_missing_file_raises(self, tmp_path):
        with pytest.raises(OSError):
            open_reader(tmp_path / "missing.pod5")

    def test_close_reader_accepts_none(self):
        close_reader(None)


class TestIterReads:
    def test_selection_returns_signal_reads(self, reader):
        wanted = read_ids(reader)[:3]
        reads = list(iter_reads(reader, selection=wanted))

        assert {r.read_id for r in reads} == set(wanted)
        for r in reads:
            assert isinstance(r, SignalRead)
            assert r.signal.dtype == np.int16
            assert len(r.signal) == r.record.num_samples
            assert r.sample_rate == r.run_info.sample_rate > 0

    def test_missing_ok_skips_absent_and_malformed_ids(self, reader):
        first = read_ids(reader)[0]
        reads = list(
            iter_reads(
                reader,
                selection=[first, MISSING_ID, "not-a-uuid"],
                missing_ok=True,
            )
        )
        assert [r.read_id for r in reads] == [first]

    def test_missing_ok_with_only_invalid_ids_yields_nothing(self, reader):
        assert list(iter_reads(reader, selection=["bad"], missing_ok=True)) == []

    def test_missing_id_raises_without_missing_ok(self, reader):
        with pytest.raises(KeyError):
            list(iter_reads(reader, selection=[MISSING_ID]))

    def test_full_iteration_matches_read_ids(self, reader):
        assert [r.read_id for r in iter_reads(reader)] == read_ids(reader)


class TestGetRead:
    def test_found(self, reader):
        rid = read_ids(reader)[0]
        read = get_read(reader, rid)
        assert read is not None and read.read_id == rid

    @pytest.mark.parametrize("rid", [MISSING_ID, "INVALID_READ_ID_12345"])
    def test_not_found_returns_none(self, reader, rid):
        assert get_read(reader, rid) is None


class TestLazyReadList:
    def test_len_index_slice_iter(self, reader):
        ids = read_ids(reader)
        lazy = LazyReadList(reader)

        assert len(lazy) == len(ids)
        assert lazy[0] == ids[0]
        assert lazy[1:4] == ids[1:4]
        assert list(lazy) == ids
        assert "materialized" not in repr(lazy)

    def test_materialize(self, reader):
        lazy = LazyReadList(reader)
        ids = lazy.materialize()

        assert ids == read_ids(reader)
        assert lazy.materialize() is ids
        assert "(materialized)" in repr(lazy)
