# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Grouping adjacent moving bits into fields, on payloads with known layout."""
from __future__ import annotations

import random

import numpy as np
import pytest

from canlens.analyze.bits import BitOrder, bit_matrix
from canlens.infer import infer_message
from canlens.infer.signals import MIN_RATE, find_signals


def matrix_of(payloads):
    return bit_matrix(payloads, len(payloads[0]), BitOrder.INTEL)


def ramp(n=400, width=4, fields=((0, 12), (16, 10))):
    """Payloads carrying smooth little-endian ramps at the given (start, bits)."""
    rng = random.Random(0)
    phase = {start: rng.randrange(1 << bits) for start, bits in fields}
    out = []
    for _ in range(n):
        value = 0
        for start, bits in fields:
            phase[start] = (phase[start] + rng.randrange(1, 4)) % (1 << bits)
            value |= phase[start] << start
        out.append(value.to_bytes(width, "little"))
    return out


class TestFindSignals:
    def test_two_ramps_are_two_fields(self):
        found = find_signals(matrix_of(ramp()))
        starts = {s.start_bit for s in found}
        assert 0 in starts and 16 in starts

    def test_a_field_stops_where_its_bits_stop_moving(self):
        """A 12-bit ramp that only reaches 10 bits is claimed as what moved."""
        found = find_signals(matrix_of(ramp(fields=((0, 12),))))
        first = next(s for s in found if s.start_bit == 0)
        assert 2 <= first.length <= 12

    def test_a_field_stopped_by_something_is_bounded(self):
        """Bounded means the width is the width: a claim, not a floor."""
        payloads = ramp(fields=((0, 16),))
        found = find_signals(matrix_of(payloads), claimed_bits={4, 5, 6, 7})
        low = next(s for s in found if s.start_bit == 0)
        assert low.bounded and low.end_bit == 4
        assert "+ bits" not in str(low)

    def test_a_field_fading_into_still_bits_is_only_a_lower_bound(self):
        payloads = ramp(width=4, fields=((0, 6),))
        found = find_signals(matrix_of(payloads))
        first = next(s for s in found if s.start_bit == 0)
        assert not first.bounded
        assert "+ bits" in str(first)

    def test_constant_payloads_have_no_signals(self):
        assert find_signals(matrix_of([bytes([0xAA, 0xBB])] * 200)) == []

    def test_a_single_flickering_bit_is_not_a_field(self):
        """One bit is a flag, and the bit map already shows flags."""
        payloads = [bytes([i % 2, 0]) for i in range(200)]
        assert find_signals(matrix_of(payloads)) == []

    def test_a_bit_that_moves_twice_in_a_long_trace_is_not_a_field(self):
        payloads = [bytes([0, 0])] * 300
        payloads[100] = bytes([0x03, 0])
        assert find_signals(matrix_of(payloads)) == []

    def test_claimed_bits_end_a_field_rather_than_joining_it(self):
        payloads = ramp(fields=((0, 16),))
        whole = find_signals(matrix_of(payloads))
        assert max(s.length for s in whole) > 4
        split = find_signals(matrix_of(payloads), claimed_bits={4, 5, 6, 7})
        assert all(s.end_bit <= 4 or s.start_bit >= 8 for s in split)

    def test_the_observed_range_is_reported(self):
        """A byte counting 0..199 reads as seven bits: bit 6 turns over three
        times in 200 frames, which the floor accepts, and bit 7 turns over
        once, which is not evidence of anything."""
        payloads = [bytes([v, 0]) for v in range(200)]
        found = find_signals(matrix_of(payloads))
        first = next(s for s in found if s.start_bit == 0)
        assert (first.length, first.minimum, first.maximum) == (7, 0, 127)
        assert not first.bounded and first.span == 127

    def test_two_fast_fields_side_by_side_are_read_as_one(self):
        """A known limit, recorded rather than papered over.

        The boundary is visible only where the rate *rises*, and between two
        fields that both move on most frames it does not. Separating those
        needs something this detector does not have.
        """
        import random as _random

        rng = _random.Random(1)
        low = high = 0
        payloads = []
        for _ in range(400):
            low = (low + rng.randrange(1, 40)) % 64
            high = (high + rng.randrange(1, 4)) % 1024
            payloads.append(((high << 6) | low).to_bytes(2, "little"))
        found = find_signals(matrix_of(payloads))
        assert len(found) == 1 and found[0].start_bit == 0

    def test_a_slow_narrow_enum_is_out_of_reach(self):
        """A second known limit, and the reason lowering the floor cannot fix it.

        A 2-bit enum that changes state five times in a drive puts its low bit
        at 0.010 and its high bit at 0.005, because the high bit of any field
        moves half as often as the low one. Whatever floor admits the first
        drops the second, leaving one surviving bit -- a flag by definition,
        not a field. Narrow fields are not lost to a threshold being too
        strict; they are lost to having no rate profile to segment.
        """
        payloads, value = [], 0
        for i in range(400):
            if i and i % 80 == 0:
                value = (value + 1) % 4
            payloads.append(bytes([value, 0]))
        assert find_signals(matrix_of(payloads)) == []

    def test_the_top_of_a_wide_field_survives_the_floor(self):
        """What lowering the floor actually bought: not narrow fields but the
        slow upper bits of wide ones, where the rate decay runs out."""
        payloads = [bytes([v, 0]) for v in range(200)]
        found = find_signals(matrix_of(payloads))
        first = next(s for s in found if s.start_bit == 0)
        assert first.length == 7, "bit 6 turns over three times and must count"
        assert find_signals(matrix_of(payloads), min_rate=0.02)[0].length == 6

    def test_an_empty_trace_yields_nothing(self):
        assert find_signals(np.zeros((0, 16), dtype=np.uint8)) == []
        assert find_signals(np.zeros((1, 16), dtype=np.uint8)) == []

    @pytest.mark.parametrize("rise", [2.0, 3.0, 6.0])
    def test_a_steeper_rise_never_claims_more(self, rise):
        """Raising the bar can only cut fewer fields, never invent them."""
        payloads = ramp()
        loose = find_signals(matrix_of(payloads), rise=1.5)
        strict = find_signals(matrix_of(payloads), rise=rise)
        assert len(strict) <= len(loose)


class TestIntegration:
    def test_signals_never_take_bits_a_counter_owns(self):
        """A verified claim outranks bits that merely move together."""
        payloads = []
        rng = random.Random(4)
        value = 0
        for i in range(400):
            value = (value + rng.randrange(1, 4)) % 4096
            body = bytearray(8)
            body[0] = i % 256                       # a counter
            body[2:4] = value.to_bytes(2, "little")  # a ramp
            payloads.append(bytes(body))
        found = infer_message(payloads, bus=0, address=0x120)
        assert any(c.start_bit == 0 and c.length == 8 for c in found.counters)
        for signal in found.signals:
            assert signal.start_bit >= 8, f"{signal} overlaps the counter"

    def test_a_message_with_only_a_signal_still_counts_as_a_finding(self):
        found = infer_message(ramp(width=8), bus=0, address=0x200)
        assert found.signals and found.found_anything

    def test_the_lowest_bit_carries_the_rate(self):
        found = find_signals(matrix_of(ramp(fields=((0, 10),))))
        first = next(s for s in found if s.start_bit == 0)
        assert first.rate >= MIN_RATE
