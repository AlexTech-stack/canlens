# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""The columnar trace representation."""
from __future__ import annotations

import numpy as np
import pytest

from canlens.decode import CanFrame, FrameSet, from_frames
from canlens.decode.frameset import _ranges


def frames(spec):
    """spec: (mono_ns, bus, address, payload, echo)"""
    return [CanFrame(*row) for row in spec]


@pytest.fixture
def sample():
    return from_frames(frames([
        (10, 0, 0x100, b"\x01\x02", False),
        (20, 2, 0x200, b"\xaa\xbb\xcc\xdd", True),
        (30, 0, 0x100, b"\x03\x04", False),
        (40, 1, 0x100, b"\x09", False),
    ]))


class TestRanges:
    def test_covers_each_block(self):
        got = _ranges(np.array([10, 100]), np.array([3, 2]))
        assert got.tolist() == [10, 11, 12, 100, 101]

    def test_a_zero_length_block_contributes_nothing(self):
        # A cumulative-sum construction has no sensible answer here.
        got = _ranges(np.array([10, 50, 100]), np.array([2, 0, 1]))
        assert got.tolist() == [10, 11, 100]

    def test_empty(self):
        assert _ranges(np.array([], dtype=np.int64), np.array([], dtype=np.int64)).size == 0

    def test_blocks_out_of_order(self):
        got = _ranges(np.array([100, 10]), np.array([2, 2]))
        assert got.tolist() == [100, 101, 10, 11]


class TestColumns:
    def test_length(self, sample):
        assert len(sample) == 4

    def test_bus_strips_the_echo_flag(self, sample):
        assert sample.bus.tolist() == [0, 2, 0, 1]

    def test_echo_is_recovered(self, sample):
        assert sample.echo.tolist() == [False, True, False, False]

    def test_offsets_track_payload_starts(self, sample):
        assert sample.offsets.tolist() == [0, 2, 6, 8, 9]

    def test_keys_are_sorted(self, sample):
        assert sample.keys() == [(0, 0x100), (1, 0x100), (2, 0x200)]


class TestSelection:
    def test_dropping_echoes(self, sample):
        kept = sample.without_echoes()
        assert len(kept) == 3
        assert not kept.echo.any()

    def test_dropping_echoes_keeps_payloads_aligned(self, sample):
        kept = sample.without_echoes()
        assert kept.payloads(np.arange(len(kept))) == [b"\x01\x02", b"\x03\x04", b"\x09"]

    def test_take_with_an_index_array(self, sample):
        taken = sample.take(np.array([3, 0]))
        assert taken.payloads(np.arange(2)) == [b"\x09", b"\x01\x02"]

    def test_taking_nothing(self, sample):
        assert len(sample.take(np.zeros(4, dtype=bool))) == 0


class TestPayloads:
    def test_matrix_is_gathered_without_bytes(self, sample):
        index = sample.indices_for(0, 0x100)
        assert sample.payload_matrix(index, 2).tolist() == [[1, 2], [3, 4]]

    def test_matrix_of_nothing_keeps_the_width(self, sample):
        assert sample.payload_matrix(np.array([], dtype=np.int64), 8).shape == (0, 8)

    def test_payloads_as_bytes(self, sample):
        assert sample.payloads(sample.indices_for(0, 0x100)) == [b"\x01\x02", b"\x03\x04"]

    def test_raw_is_materialised_once(self, sample):
        assert sample.raw is sample.raw


class TestGrouping:
    def test_one_message_per_bus_and_address(self, sample):
        assert [m.key for m in sample.group()] == [(0, 0x100), (1, 0x100), (2, 0x200)]

    def test_counts_and_widths(self, sample):
        message = next(m for m in sample.group() if m.key == (0, 0x100))
        assert message.count == 2 and message.width == 2
        assert not message.multi_length

    def test_the_dominant_width_wins_and_the_rest_are_recorded(self):
        trace = from_frames(frames(
            [(i, 0, 0x100, b"\xff\xff", False) for i in range(9)]
            + [(99, 0, 0x100, b"\xff", False)]
        ))
        message = trace.group()[0]
        assert message.width == 2
        assert message.multi_length and message.lengths == {1: 1, 2: 9}
        assert message.index.size == 9      # the odd one out is excluded
        assert message.count == 10          # but still counted

    def test_min_frames_skips_thin_messages(self, sample):
        assert sample.group(min_frames=3) == []

    def test_stamps_follow_the_message(self, sample):
        message = next(m for m in sample.group() if m.key == (0, 0x100))
        assert message.stamps.tolist() == [10, 30]


class TestRoundTrip:
    def test_frames_come_back_unchanged(self, sample):
        original = frames([
            (10, 0, 0x100, b"\x01\x02", False),
            (20, 2, 0x200, b"\xaa\xbb\xcc\xdd", True),
            (30, 0, 0x100, b"\x03\x04", False),
            (40, 1, 0x100, b"\x09", False),
        ])
        assert list(sample) == original

    def test_an_empty_trace_is_safe(self):
        empty = FrameSet(
            mono_ns=np.zeros(0, np.int64), src=np.zeros(0, np.uint8),
            address=np.zeros(0, np.uint32), lengths=np.zeros(0, np.uint8),
            blob=np.zeros(0, np.uint8),
        )
        assert len(empty) == 0 and empty.keys() == [] and empty.group() == []
        assert list(empty) == []
