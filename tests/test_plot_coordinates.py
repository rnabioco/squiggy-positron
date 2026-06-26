"""Tests for the shared genomic-coordinate helpers on PlotStrategy (Issue #188)

These helpers were extracted from near-identical inline blocks in the
single_read / overlay / stacked strategies. The tests pin the behavior so the
shared implementation stays equivalent to the originals.
"""

from types import SimpleNamespace

import numpy as np

from squiggy.constants import Theme
from squiggy.plot_strategies.base import PlotStrategy


class _DummyStrategy(PlotStrategy):
    """Minimal concrete strategy so the protected helpers can be exercised."""

    def create_plot(self, data, options):
        return "", None

    def validate_data(self, data):
        return None


def _base(genomic_pos, signal_start, signal_end):
    return SimpleNamespace(
        genomic_pos=genomic_pos, signal_start=signal_start, signal_end=signal_end
    )


def _read(bases):
    return SimpleNamespace(bases=bases, chromosome="chr1")


def _strategy():
    return _DummyStrategy(Theme.LIGHT)


class TestBuildGenomicRefPositions:
    def test_basic_mapping(self):
        read = _read([_base(100, 0, 2), _base(101, 2, 4)])
        assert _strategy()._build_genomic_ref_positions(read) == [100, 100, 101, 101]

    def test_start_soft_clip_reuses_first_position(self):
        # First base unmapped (soft-clip): reuse the first mapped genomic pos
        read = _read([_base(None, 0, 2), _base(100, 2, 4)])
        assert _strategy()._build_genomic_ref_positions(read) == [100, 100, 100, 100]

    def test_insertion_reuses_previous_position(self):
        read = _read([_base(100, 0, 2), _base(None, 2, 3), _base(102, 3, 5)])
        assert _strategy()._build_genomic_ref_positions(read) == [
            100,
            100,
            100,
            102,
            102,
        ]

    def test_no_genomic_positions_returns_empty(self):
        read = _read([_base(None, 0, 2), _base(None, 2, 4)])
        assert _strategy()._build_genomic_ref_positions(read) == []


class TestCollapseToGenomicPositions:
    def test_collapses_repeats_by_mean(self):
        positions, values = _strategy()._collapse_to_genomic_positions(
            [100, 100, 101, 101], np.array([1.0, 3.0, 5.0, 7.0]), downsample=1
        )
        assert list(positions) == [100, 101]
        assert list(values) == [2.0, 6.0]

    def test_inserts_nan_at_deletion(self):
        positions, values = _strategy()._collapse_to_genomic_positions(
            [100, 100, 102], np.array([1.0, 3.0, 5.0]), downsample=1
        )
        assert positions[0] == 100
        assert np.isnan(positions[1])  # NaN break at the deletion gap
        assert positions[2] == 102
        assert values[0] == 2.0
        assert np.isnan(values[1])
        assert values[2] == 5.0

    def test_downsamples_positions_to_match_signal(self):
        positions, values = _strategy()._collapse_to_genomic_positions(
            [100, 100, 101, 101, 102, 102],
            np.array([1.0, 2.0, 3.0]),
            downsample=2,
        )
        assert list(positions) == [100, 101, 102]
        assert list(values) == [1.0, 2.0, 3.0]

    def test_pads_positions_to_match_signal(self):
        positions, values = _strategy()._collapse_to_genomic_positions(
            [100, 100], np.array([1.0, 2.0, 3.0]), downsample=1
        )
        # Padded with the edge position, then collapsed to one point
        assert list(positions) == [100]
        assert list(values) == [2.0]
