# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Value jumpiness as a filter on signal claims, on payloads with known layout."""
from __future__ import annotations

import random

import numpy as np
import pytest

from canlens.analyze.bits import BitOrder, bit_matrix, bit_matrix_from_bytes
from canlens.infer import infer_message
from canlens.infer.byteorder import (
    find_signals_in,
    intel_to_column,
    motorola_columns,
)
from canlens.infer.counters import field_values
from canlens.infer.signals import SignalHypothesis, find_signals
from canlens.infer.smoothness import (
    JUMP_FRACTION,
    MAX_JUMP_SHARE,
    filter_signals,
    jump_share,
    v_jump_share,
)
from canlens.truth.dbc import signal_bits


def matrix_of(payloads):
    return bit_matrix(payloads, len(payloads[0]), BitOrder.INTEL)


def smooth_walk(n=400, width=2, seed=0):
    """Byte 0 is a bounded random walk (smooth, but not a counter); byte 1 is
    uniformly random, so it lurches across its whole range every frame."""
    rng = random.Random(seed)
    value = 128
    out = []
    for _ in range(n):
        value = max(0, min(255, value + rng.choice([-1, 0, 1])))
        out.append(bytes([value, rng.randrange(256)]) + bytes(width - 2))
    return out


class TestJumpShare:
    def test_a_smooth_ramp_has_no_jumps(self):
        assert jump_share(list(range(200))) == 0.0

    def test_a_value_leaping_across_its_range_jumps_every_frame(self):
        values = [0, 255] * 100
        assert jump_share(values) == 1.0

    def test_constant_values_have_no_range_to_jump_across(self):
        assert jump_share([7] * 50) == 0.0

    def test_too_few_values(self):
        assert jump_share([]) == 0.0
        assert jump_share([3]) == 0.0

    def test_one_large_step_is_one_jump(self):
        values = [0] * 50 + [255] + [0] * 49
        assert jump_share(values) == pytest.approx(2 / 99)

    def test_the_threshold_scales_with_the_observed_range(self):
        """Twice the value, twice the range: the same share."""
        small = [0, 1, 2, 3] * 25
        large = [v * 100 for v in small]
        assert jump_share(small) == jump_share(large)


class TestVectorisedMatchesScalar:
    """The vectorised form must agree with the readable scalar reference."""

    @staticmethod
    def sample(width=4, count=200, seed=5):
        rng = random.Random(seed)
        return np.array(
            [[rng.randrange(256) for _ in range(width)] for _ in range(count)],
            dtype=np.uint8,
        )

    @pytest.mark.parametrize("start", [0, 3, 8, 15])
    @pytest.mark.parametrize("length", [2, 5, 8])
    def test_jump_share(self, start, length):
        matrix = bit_matrix_from_bytes(self.sample(), BitOrder.INTEL)
        if start + length > matrix.shape[1]:
            pytest.skip("field past the end")
        scalar = jump_share(field_values(matrix, start, length).tolist())
        assert v_jump_share(matrix, start, length) == scalar

    def test_across_byte_boundaries(self):
        matrix = bit_matrix_from_bytes(self.sample(width=8, count=100), BitOrder.INTEL)
        for start, length in ((4, 12), (6, 16), (1, 24)):
            scalar = jump_share(field_values(matrix, start, length).tolist())
            assert v_jump_share(matrix, start, length) == scalar


class TestFilterSignals:
    def test_a_coherent_field_survives_and_a_noisy_one_is_dropped(self):
        payloads = smooth_walk()
        matrix = matrix_of(payloads)
        found = find_signals(matrix)
        starts = {s.start_bit for s in found}
        assert 0 in starts and 8 in starts  # both were found
        kept = {s.start_bit for s in filter_signals(matrix, found)}
        assert 0 in kept
        assert 8 not in kept

    def test_the_filter_uses_the_matrix_it_is_given(self):
        """A random-only message is all jumps, so nothing coherent survives."""
        rng = random.Random(3)
        payloads = [bytes(rng.randrange(256) for _ in range(4)) for _ in range(300)]
        matrix = matrix_of(payloads)
        assert filter_signals(matrix, find_signals(matrix)) != find_signals(matrix)

    def test_an_empty_list_stays_empty(self):
        assert filter_signals(matrix_of(smooth_walk()), []) == []

    def test_a_field_that_never_moves_is_kept_by_the_filter(self):
        """Range zero is not a jump; such a shape is caught by the rate floor."""
        matrix = matrix_of([bytes([0, 0])] * 10)
        signal = SignalHypothesis(
            start_bit=0, length=8, frames=10, rate=0.0, minimum=0, maximum=0
        )
        assert filter_signals(matrix, [signal]) == [signal]


class TestDetectionAppliesTheFilter:
    def test_inference_over_a_coherent_and_a_noisy_region(self):
        payloads = smooth_walk()
        found = infer_message(payloads, bus=0, address=0x200)
        starts = {s.start_bit for s in found.signals}
        assert 0 in starts
        assert 8 not in starts

    def test_a_message_of_pure_noise_yields_no_signal(self):
        rng = random.Random(11)
        payloads = [bytes(rng.randrange(256) for _ in range(4)) for _ in range(300)]
        found = infer_message(payloads, bus=0, address=0x201)
        assert found.signals == []

    def test_a_counter_is_still_claimed_beside_a_signal(self):
        rng = random.Random(21)
        value = 128
        payloads = []
        for i in range(400):
            value = max(0, min(255, value + rng.choice([-1, 0, 1])))
            payloads.append(bytes([value, i % 256, rng.randrange(256), 0]))
        found = infer_message(payloads, bus=0, address=0x202)
        assert any(c.start_bit == 8 and c.length == 8 for c in found.counters)
        assert any(s.start_bit == 0 for s in found.signals)
        assert all(not (s.start_bit < 16 < s.start_bit + s.length) for s in found.signals)


class TestMotorolaBranch:
    @staticmethod
    def big_endian_ramp(n=400, width=8, bits=12, start=7, seed=0, step=40):
        rng = random.Random(seed)
        positions = signal_bits(start, bits, "big_endian")
        out = np.zeros((n, width), dtype=np.uint8)
        value = 0
        for row in range(n):
            value = (value + rng.randrange(1, step)) % (1 << bits)
            for k, position in enumerate(positions):
                if (value >> (bits - 1 - k)) & 1:
                    out[row, position // 8] |= 1 << (position % 8)
        return out

    def test_a_big_endian_signal_survives_the_filter(self):
        byte_matrix = self.big_endian_ramp()
        found = find_signals_in(byte_matrix, BitOrder.MOTOROLA)
        assert found, "the coherent big-endian field should survive"
        assert all(s.byte_order is BitOrder.MOTOROLA for s in found)

    def test_every_motorola_claim_obeys_the_filter(self):
        byte_matrix = self.big_endian_ramp()
        found = find_signals_in(byte_matrix, BitOrder.MOTOROLA)
        matrix = motorola_columns(byte_matrix)
        width_bits = byte_matrix.shape[1] * 8
        for signal in found:
            end_column = intel_to_column(signal.start_bit, width_bits)
            start_column = end_column - (signal.length - 1)
            assert (
                v_jump_share(matrix, start_column, signal.length) <= MAX_JUMP_SHARE
            ), signal


def test_the_constants_are_ordered_sensibly():
    assert 0.0 < JUMP_FRACTION < MAX_JUMP_SHARE <= 1.0
