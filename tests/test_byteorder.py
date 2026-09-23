# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Deciding a bus's bit order, on payloads built in a known order.

Synthetic buses with a decided wiring, so what is under test is the decision
and the mapping -- not whether a particular car happens to agree.
"""
from __future__ import annotations

import random

import numpy as np

from canlens.analyze.bits import BitOrder
from canlens.infer.byteorder import (
    LONG_FIELD,
    MIN_MARGIN,
    BusOrder,
    intel_to_column,
    motorola_columns,
    to_intel_bit,
)
from canlens.infer.byteorder import (
    decide_byte_order as decide,
)
from canlens.truth.dbc import signal_bits


def ramp(n=400, width=8, bits=12, start=0, big_endian=False, seed=0, step=40):
    """Payloads carrying one wide ramp laid out in the given byte order.

    `step` has to be big enough that the value wraps inside `n` frames; a
    12-bit field advancing by one or two per frame never reaches its top bits
    in 400, and then there is no long field for either order to see.
    """
    rng = random.Random(seed)
    positions = signal_bits(start, bits, "big_endian" if big_endian else "little_endian")
    out = np.zeros((n, width), dtype=np.uint8)
    value = 0
    for row in range(n):
        value = (value + rng.randrange(1, step)) % (1 << bits)
        for k, position in enumerate(positions):
            # positions[0] is the most significant bit for a big-endian signal
            shift = (bits - 1 - k) if big_endian else k
            if (value >> shift) & 1:
                out[row, position // 8] |= 1 << (position % 8)
    return out


class TestMapping:
    def test_the_two_mappings_are_inverse(self):
        for width_bits in (16, 64):
            for bit in range(width_bits):
                column = intel_to_column(bit, width_bits)
                assert to_intel_bit(column, width_bits) == bit

    def test_a_big_endian_field_is_contiguous_in_motorola_columns(self):
        width_bits = 64
        for start in (7, 15, 23, 39):
            for length in (10, 12, 16):
                bits = signal_bits(start, length, "big_endian")
                if max(bits) >= width_bits:
                    continue
                columns = sorted(intel_to_column(b, width_bits) for b in bits)
                assert columns == list(range(columns[0], columns[0] + length))

    def test_the_least_significant_bit_lands_lowest(self):
        """So the rate decays upward, as the cut rule assumes."""
        width_bits = 64
        bits = signal_bits(23, 12, "big_endian")
        columns = [intel_to_column(b, width_bits) for b in bits]
        assert columns[-1] == min(columns)  # bits[-1] is the LSB

    def test_motorola_columns_only_reorders(self):
        payloads = ramp()
        m = motorola_columns(payloads)
        assert m.shape == (payloads.shape[0], payloads.shape[1] * 8)
        assert m.sum() == np.unpackbits(payloads, axis=1).sum()


class TestBusOrder:
    def test_a_clear_majority_decides(self):
        order = BusOrder(bus=0, intel=30, motorola=10, messages=20)
        assert order.decided and order.order is BitOrder.INTEL
        assert order.margin == 0.5

    def test_the_other_way_round(self):
        order = BusOrder(bus=1, intel=8, motorola=24, messages=20)
        assert order.decided and order.order is BitOrder.MOTOROLA

    def test_a_narrow_margin_is_left_undecided(self):
        order = BusOrder(bus=0, intel=100, motorola=97, messages=50)
        assert order.margin < MIN_MARGIN and not order.decided
        assert "undecided" in str(order)

    def test_a_bus_with_no_long_fields_is_undecided_rather_than_intel(self):
        order = BusOrder(bus=0, intel=0, motorola=0, messages=5)
        assert order.margin == 0.0 and not order.decided


class TestDecide:
    def test_a_little_endian_bus_reads_as_intel(self):
        messages = [(0, ramp(seed=i, big_endian=False), set()) for i in range(6)]
        found = decide(messages)[0]
        assert found.order is BitOrder.INTEL and found.decided

    def test_a_big_endian_bus_reads_as_motorola(self):
        messages = [(0, ramp(seed=i, big_endian=True, start=7), set())
                    for i in range(6)]
        found = decide(messages)[0]
        assert found.order is BitOrder.MOTOROLA and found.decided

    def test_two_buses_are_decided_separately(self):
        messages = ([(0, ramp(seed=i, big_endian=False), set()) for i in range(6)]
                    + [(1, ramp(seed=i, big_endian=True, start=7), set())
                       for i in range(6)])
        found = decide(messages)
        assert found[0].order is BitOrder.INTEL
        assert found[1].order is BitOrder.MOTOROLA

    def test_claimed_bits_count_for_neither_order(self):
        """A counter must not be mistaken for a long field on either side."""
        payloads = ramp()
        everything = set(range(payloads.shape[1] * 8))
        found = decide([(0, payloads, everything)])[0]
        assert found.intel == 0 and found.motorola == 0

    def test_a_message_too_short_to_measure_is_skipped(self):
        assert decide([(0, np.zeros((1, 8), dtype=np.uint8), set())]) == {}
        assert decide([(0, np.zeros((0, 8), dtype=np.uint8), set())]) == {}

    def test_messages_are_counted(self):
        messages = [(0, ramp(seed=i), set()) for i in range(4)]
        assert decide(messages)[0].messages == 4

    def test_no_messages_decide_nothing(self):
        assert decide([]) == {}


def test_the_long_field_floor_exceeds_one_byte():
    """An 8-bit field on a byte boundary reads the same either way."""
    assert LONG_FIELD > 8


class TestAppliedToDetection:
    """The decided order has to reach the claims, not just the report."""

    @staticmethod
    def frames(big_endian, count=400):
        from canlens.decode import CanFrame
        payloads = ramp(n=count, big_endian=big_endian, start=7 if big_endian else 0)
        return [
            CanFrame(i * 10_000_000, 0, 0x210, bytes(payloads[i]), False)
            for i in range(count)
        ]

    def test_a_big_endian_bus_yields_big_endian_claims(self):
        from canlens.decode import from_frames
        from canlens.infer import infer_frameset

        found = infer_frameset(from_frames(self.frames(big_endian=True)))
        signals = [s for m in found for s in m.signals]
        assert signals, "no signal found at all"
        assert all(s.byte_order is BitOrder.MOTOROLA for s in signals)

    def test_a_little_endian_bus_is_left_alone(self):
        from canlens.decode import from_frames
        from canlens.infer import infer_frameset

        found = infer_frameset(from_frames(self.frames(big_endian=False)))
        signals = [s for m in found for s in m.signals]
        assert signals and all(s.byte_order is BitOrder.INTEL for s in signals)

    def test_a_big_endian_claim_reports_the_bits_it_actually_covers(self):
        from canlens.decode import from_frames
        from canlens.infer import infer_frameset

        found = infer_frameset(from_frames(self.frames(big_endian=True)))
        signal = next(s for m in found for s in m.signals)
        assert len(signal.bits) == signal.length
        # a wide big-endian field is not contiguous in Intel numbering
        if signal.length > 8:
            assert sorted(signal.bits) != list(range(min(signal.bits),
                                                     max(signal.bits) + 1))

    def test_both_inference_paths_still_agree(self):
        from canlens.decode import from_frames
        from canlens.infer import infer_frames, infer_frameset

        objects = self.frames(big_endian=True)
        a = infer_frameset(from_frames(objects))
        b = infer_frames(iter(objects))
        assert [(m.bus, m.address) for m in a] == [(m.bus, m.address) for m in b]
        assert [[s.bits for s in m.signals] for m in a] == \
               [[s.bits for s in m.signals] for m in b]
