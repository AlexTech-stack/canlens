# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Measurement tests on synthetic frames -- no network, no corpus data."""
from __future__ import annotations

import numpy as np
import pytest

from canlens.analyze import (
    BitKind,
    Cadence,
    analyze_frames,
    bit_entropy,
    bit_matrix,
    classify_bits,
    profile_bits,
    profile_timing,
    transition_rate,
)
from canlens.decode import CanFrame


def frames(payloads, *, bus=0, address=0x123, period_ns=10_000_000, start=1_000):
    return [
        CanFrame(start + i * period_ns, bus, address, p, False)
        for i, p in enumerate(payloads)
    ]


class TestBitMatrix:
    def test_shape_and_msb_first_ordering(self):
        m = bit_matrix([b"\x80\x01"], width=2)
        assert m.shape == (1, 16)
        # MSB of byte 0 is bit 0; LSB of byte 1 is bit 15.
        assert m[0, 0] == 1
        assert m[0, 15] == 1
        assert m[0, 1:15].sum() == 0

    def test_empty_input_keeps_the_width(self):
        assert bit_matrix([], width=8).shape == (0, 64)


class TestEntropy:
    def test_constant_bits_carry_no_information(self):
        m = bit_matrix([b"\xff", b"\xff", b"\xff"], width=1)
        assert bit_entropy(m).tolist() == [0.0] * 8

    def test_evenly_split_bit_carries_one_bit(self):
        m = bit_matrix([b"\x00", b"\x01"], width=1)
        h = bit_entropy(m)
        assert h[7] == pytest.approx(1.0)
        assert h[:7].tolist() == [0.0] * 7

    def test_skewed_bit_is_between_zero_and_one(self):
        m = bit_matrix([b"\x01"] + [b"\x00"] * 7, width=1)
        assert 0.0 < bit_entropy(m)[7] < 1.0

    def test_no_warning_on_degenerate_columns(self):
        with np.errstate(all="raise"):
            bit_entropy(bit_matrix([b"\x00", b"\x00"], width=1))


class TestTransitionRate:
    def test_alternating_bit_flips_every_frame(self):
        m = bit_matrix([b"\x00", b"\x01", b"\x00", b"\x01"], width=1)
        assert transition_rate(m)[7] == pytest.approx(1.0)

    def test_constant_bit_never_flips(self):
        m = bit_matrix([b"\x00"] * 4, width=1)
        assert transition_rate(m).tolist() == [0.0] * 8

    def test_single_frame_has_no_transitions(self):
        assert transition_rate(bit_matrix([b"\x00"], width=1)).tolist() == [0.0] * 8


class TestClassifyBits:
    def test_zero_entropy_is_constant(self):
        assert classify_bits(np.array([0.0]), np.array([0.0])) == [BitKind.CONSTANT]

    def test_fast_flipping_bit_is_noisy(self):
        assert classify_bits(np.array([0.9]), np.array([1.0])) == [BitKind.NOISY]

    def test_rarely_flipping_bit_is_slow(self):
        assert classify_bits(np.array([0.001]), np.array([0.5])) == [BitKind.SLOW]

    def test_middling_bit_is_active(self):
        assert classify_bits(np.array([0.1]), np.array([0.5])) == [BitKind.ACTIVE]


class TestProfileBits:
    def test_counter_low_bit_is_noisy_and_high_bit_is_not(self):
        # An 8-bit counter: bit 7 flips every frame, bit 0 every 128.
        payloads = [bytes([i % 256]) for i in range(256)]
        profile = profile_bits(payloads, width=1)
        assert profile.kinds[7] is BitKind.NOISY
        assert profile.kinds[0] is not BitKind.NOISY

    def test_constant_payload_has_no_entropy(self):
        profile = profile_bits([b"\xa5" * 8] * 50, width=8)
        assert profile.payload_entropy == 0.0
        assert profile.constant_bits == 64


class TestTiming:
    def test_regular_gaps_are_cyclic(self):
        stamps = [i * 10_000_000 for i in range(50)]
        p = profile_timing(stamps)
        assert p.cadence is Cadence.CYCLIC
        assert p.period_ms == pytest.approx(10.0)
        assert p.rate_hz == pytest.approx(100.0)

    def test_irregular_gaps_are_sporadic(self):
        stamps = [0, 5_000_000, 400_000_000, 410_000_000, 900_000_000, 2_000_000_000]
        assert profile_timing(stamps).cadence is Cadence.SPORADIC

    def test_too_few_samples_says_so_instead_of_guessing(self):
        p = profile_timing([0, 10_000_000])
        assert p.cadence is Cadence.SINGLE
        assert p.period_ms == 0.0

    def test_empty_is_safe(self):
        assert profile_timing([]).count == 0


class TestAnalyzeFrames:
    def test_groups_by_bus_and_address(self):
        profile = analyze_frames(
            frames([b"\x00"] * 5, bus=0, address=0x100)
            + frames([b"\x00"] * 5, bus=1, address=0x100)
        )
        assert len(profile) == 2
        assert (0, 0x100) in profile.messages and (1, 0x100) in profile.messages

    def test_counts_frames_and_duration(self):
        profile = analyze_frames(frames([b"\x00"] * 11, period_ns=100_000_000))
        assert profile.frames == 11
        assert profile.duration_s == pytest.approx(1.0)

    def test_by_entropy_puts_the_richest_message_first(self):
        varied = frames([bytes([i]) for i in range(64)], address=0x1)
        flat = frames([b"\x00"] * 64, address=0x2)
        order = [m.address for m in analyze_frames(varied + flat).by_entropy()]
        assert order[0] == 0x1

    def test_mixed_lengths_are_flagged_not_padded(self):
        # Padding the short frames would invent constant bits.
        profile = analyze_frames(frames([b"\xff\xff"] * 9 + [b"\xff"]))
        message = profile[(0, 0x123)]
        assert message.multi_length
        assert message.lengths == {1: 1, 2: 9}
        assert message.width == 2
        assert message.analysed == 9
        assert message.bits.bits == 16

    def test_single_length_is_not_flagged(self):
        profile = analyze_frames(frames([b"\x01\x02"] * 4))
        assert not profile[(0, 0x123)].multi_length

    def test_empty_trace_is_safe(self):
        profile = analyze_frames([])
        assert len(profile) == 0 and profile.frames == 0
