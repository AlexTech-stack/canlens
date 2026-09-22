# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""The columnar path must agree with the object path, exactly.

Everything above `decode` now has two routes through it: the original one over
CanFrame objects, and the columnar one the cache feeds. They are only worth
having if they cannot disagree, so this compares them on the same input.
"""
from __future__ import annotations

import random

import pytest

from canlens.analyze import analyze_frames, analyze_frameset
from canlens.decode import CanFrame, from_frames
from canlens.infer import infer_frames, infer_frameset
from canlens.infer.checksums import sum8_addr_len


def build(seed: int = 0, count: int = 400) -> list[CanFrame]:
    """A trace with a counter, a Toyota checksum, mixed widths and echoes."""
    rng = random.Random(seed)
    out = []
    for i in range(count):
        body = bytearray([i % 256, *(rng.randrange(256) for _ in range(6)), 0])
        body[7] = sum8_addr_len(bytes(body), 0x210, 7)
        out.append(CanFrame(i * 10_000_000, 1, 0x210, bytes(body), False))
        out.append(CanFrame(i * 10_000_000, 0, 0x120, bytes([i % 7, 0xAA]), False))
        out.append(CanFrame(i * 10_000_000, 2, 0x120, bytes(body), True))  # echo
        if i % 50 == 0:  # a stray short frame, to exercise mixed widths
            out.append(CanFrame(i * 10_000_000, 1, 0x210, b"\x01", False))
    return out


@pytest.fixture(scope="module")
def both():
    objects = build()
    return objects, from_frames(objects)


class TestAnalyze:
    def test_same_messages(self, both):
        objects, columns = both
        assert set(analyze_frames(objects).messages) == set(analyze_frameset(columns).messages)

    def test_same_frame_count_and_duration(self, both):
        objects, columns = both
        a, b = analyze_frames(objects), analyze_frameset(columns)
        assert a.frames == b.frames
        assert a.duration_s == pytest.approx(b.duration_s)

    def test_same_bit_measurements(self, both):
        objects, columns = both
        a, b = analyze_frames(objects), analyze_frameset(columns)
        for key, message in a.messages.items():
            other = b[key]
            assert message.bits.kinds == other.bits.kinds, key
            assert message.bits.payload_entropy == pytest.approx(other.bits.payload_entropy)
            assert message.width == other.width
            assert message.count == other.count
            assert message.lengths == other.lengths
            assert message.analysed == other.analysed

    def test_same_timing(self, both):
        objects, columns = both
        a, b = analyze_frames(objects), analyze_frameset(columns)
        for key, message in a.messages.items():
            assert message.timing.cadence is b[key].timing.cadence
            assert message.timing.period_ms == pytest.approx(b[key].timing.period_ms)

    def test_mixed_widths_are_handled_the_same(self, both):
        objects, columns = both
        a, b = analyze_frames(objects), analyze_frameset(columns)
        key = (1, 0x210)
        assert a[key].multi_length and b[key].multi_length
        assert a[key].lengths == b[key].lengths

    def test_a_dispatched_frameset_takes_the_columnar_path(self, both):
        _objects, columns = both
        assert analyze_frames(columns).messages.keys() == analyze_frameset(columns).messages.keys()


class TestInfer:
    def test_same_messages(self, both):
        objects, columns = both
        assert [m.key for m in infer_frames(objects)] == [m.key for m in infer_frameset(columns)]

    def test_same_counters(self, both):
        objects, columns = both
        a = {m.key: [(c.start_bit, c.length, c.stride) for c in m.counters]
             for m in infer_frames(objects)}
        b = {m.key: [(c.start_bit, c.length, c.stride) for c in m.counters]
             for m in infer_frameset(columns)}
        assert a == b

    def test_same_checksums(self, both):
        objects, columns = both
        a = {m.key: [(s.byte_index, s.algorithm) for s in m.checksums]
             for m in infer_frames(objects)}
        b = {m.key: [(s.byte_index, s.algorithm) for s in m.checksums]
             for m in infer_frameset(columns)}
        assert a == b
        assert any(v for v in a.values()), "the fixture should contain a checksum"

    def test_same_match_rates(self, both):
        objects, columns = both
        a = sorted(c.match_rate for m in infer_frames(objects) for c in m.counters)
        b = sorted(c.match_rate for m in infer_frameset(columns) for c in m.counters)
        assert a == pytest.approx(b)

    def test_a_dispatched_frameset_takes_the_columnar_path(self, both):
        _objects, columns = both
        assert [m.key for m in infer_frames(columns)] == [
            m.key for m in infer_frameset(columns)
        ]


class TestEchoHandling:
    def test_echoes_are_excluded_by_both_paths(self, both):
        objects, columns = both
        kept = [f for f in objects if not f.echo]
        assert analyze_frames(kept).frames == analyze_frameset(columns.without_echoes()).frames
