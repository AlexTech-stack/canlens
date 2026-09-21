# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Multiplexor detection on synthetic payloads with known ground truth."""
from __future__ import annotations

import random

import numpy as np
import pytest

from canlens.analyze.bits import BitOrder, bit_matrix
from canlens.infer import ALGORITHMS, find_counters, infer_message
from canlens.infer.multiplex import (
    MIN_LAYOUTS,
    MultiplexHypothesis,
    dependent_bits,
    find_multiplexor,
    live_layouts,
    plain_counter,
    regular,
    selector_candidates,
    trim_to_moving,
)


def matrix_of(payloads: list[bytes]) -> np.ndarray:
    return bit_matrix(payloads, len(payloads[0]), BitOrder.INTEL)


def vin_like(n: int = 600, slices=(b"1C4RJFJ", b"T7LC163", b"7634\x00\x00\x00")) -> list[bytes]:
    """Byte 0 cycles through the slice index; bytes 1-7 carry that slice."""
    return [bytes([i % len(slices)]) + slices[i % len(slices)] for i in range(n)]


def two_layouts(n: int = 600, seed: int = 0) -> list[bytes]:
    """Byte 0 alternates 0/1; layout 0 carries a moving signal where layout
    1 carries constants, and the other way round in another byte."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        if i % 2 == 0:
            out.append(bytes([0, rng.randrange(256), rng.randrange(256), 0x11, 0x22, 0, 0, 0]))
        else:
            out.append(bytes([1, 0xAA, 0xBB, rng.randrange(256), rng.randrange(256), 0, 0, 0]))
    return out


class TestRegular:
    def test_a_strict_cycle_is_regular(self):
        mask = np.array([i % 3 == 0 for i in range(60)])
        assert regular(mask)

    def test_a_schedule_with_a_favoured_value_is_regular(self):
        seq = [0, 1, 0, 2] * 30
        assert regular(np.array([v == 0 for v in seq]))
        assert regular(np.array([v == 1 for v in seq]))

    def test_a_value_that_sits_still_is_not_regular(self):
        """Gap one between nearly all its frames, then a long absence."""
        mask = np.array([True] * 100 + [False] * 300 + [True] * 100 + [False] * 300)
        assert not regular(mask)

    def test_visits_that_stop_for_a_stretch_are_not_regular(self):
        """Regular while a flag is set, absent while it is not."""
        mask = np.array((([True, False, False, False] * 25) + [False] * 200) * 3)
        assert not regular(mask)

    def test_fewer_than_three_visits_cannot_be_regular(self):
        assert not regular(np.array([False] * 10 + [True] + [False] * 10 + [True]))


class TestPlainCounter:
    def test_a_full_nibble_count_is_plain(self):
        values = np.arange(200) % 16
        assert plain_counter(values, np.unique(values))

    def test_three_values_are_not(self):
        values = np.arange(200) % 3
        assert not plain_counter(values, np.unique(values))

    def test_a_sparse_range_is_not(self):
        values = (np.arange(200) % 16) * 16
        assert not plain_counter(values, np.unique(values))


class TestSelectorCandidates:
    def test_byte_nibble_and_short_fields_per_byte(self):
        fields = selector_candidates(16)
        assert (0, 8) in fields and (8, 8) in fields
        assert (0, 4) in fields and (4, 4) in fields and (12, 4) in fields
        assert (0, 2) in fields and (6, 2) in fields and (0, 3) in fields and (13, 3) in fields
        assert all(start + length <= 16 for start, length in fields)


class TestFindMultiplexor:
    def test_finds_a_vin_style_table(self):
        found = find_multiplexor(matrix_of(vin_like()))
        assert found is not None
        # Byte 0 holds 0, 1, 2, so only its low two bits ever move.
        assert (found.start_bit, found.length) == (0, 2)
        assert found.values == (0, 1, 2)
        assert found.frames_per_value == (200, 200, 200)
        assert found.coverage == 1.0
        # Every slice byte differs between at least two slices.
        assert {b // 8 for b in found.dependent_bits} == {1, 2, 3, 4, 5, 6, 7}

    def test_reports_only_the_bits_that_move(self):
        """Claiming the whole byte would claim six constant bits as selector.

        Volkswagen declares `VIN_01_MUX` as two bits and Tesla declares
        `VCFRONT_LVPowerStateIndex` as five; neither is a byte.
        """
        found = find_multiplexor(matrix_of(vin_like()))
        assert found is not None and found.length == 2

    def test_finds_layouts_with_moving_signals(self):
        found = find_multiplexor(matrix_of(two_layouts()))
        assert found is not None
        # Byte 0 alternates 0 and 1, so the selector is its lowest bit alone.
        assert (found.start_bit, found.length, found.values) == (0, 1, (0, 1))
        assert {b // 8 for b in found.dependent_bits} == {1, 2, 3, 4}

    def test_random_payloads_have_no_multiplexor(self):
        rng = random.Random(3)
        payloads = [bytes(rng.randrange(256) for _ in range(8)) for _ in range(600)]
        assert find_multiplexor(matrix_of(payloads)) is None

    def test_a_counter_is_not_a_multiplexor(self):
        rng = random.Random(4)
        payloads = [
            bytes([i % 16, rng.randrange(256), rng.randrange(256), 0, 0, 0, 0, 0])
            for i in range(600)
        ]
        assert find_multiplexor(matrix_of(payloads)) is None

    def test_a_crc_of_a_counter_is_not_a_layout(self):
        """Static message: byte 7 is a CRC of the counter, constant per count.

        Under the counter as selector that byte is a perfect table. With the
        checksum unproven (nothing else moves) the byte must not turn the
        counter into a multiplexor, and a 3-bit window over the counter --
        whose groups see the CRC alternate strictly -- must not either.
        """
        crc = ALGORITHMS["crc8_j1850"]
        payloads = []
        for i in range(600):
            frame = bytes([i % 16, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0])
            payloads.append(frame[:7] + bytes([crc(frame, 0x15A, 7)]))
        assert find_multiplexor(matrix_of(payloads)) is None

    def test_a_sign_extension_is_not_a_layout(self):
        """A signed 16-bit value crossing zero: byte 1 follows byte 0's top bits."""
        rng = random.Random(5)
        value = 0
        payloads = []
        for _ in range(1000):
            value = max(-300, min(300, value + rng.randrange(-40, 41)))
            payloads.append(value.to_bytes(2, "little", signed=True) + bytes(6))
        assert find_multiplexor(matrix_of(payloads)) is None

    def test_a_flag_that_changes_rarely_is_not_a_schedule(self):
        rng = random.Random(6)
        payloads = []
        for i in range(1200):
            flag = (i // 300) % 2
            body = bytes([rng.randrange(256), rng.randrange(256)]) if flag else bytes([0xAA, 0xBB])
            payloads.append(bytes([flag]) + body + bytes(5))
        assert find_multiplexor(matrix_of(payloads)) is None

    def test_a_checksum_byte_is_never_the_selector_nor_dependent(self):
        found = find_multiplexor(matrix_of(vin_like()), skip_bytes={7})
        assert found is not None
        assert 7 not in {b // 8 for b in found.dependent_bits}

    def test_short_traces_are_declined(self):
        assert find_multiplexor(matrix_of(vin_like(12))) is None

    def test_str_lists_the_values(self):
        h = MultiplexHypothesis(0, 8, (0, 1, 2), (200, 200, 200), tuple(range(8, 40)), 600)
        assert str(h) == "8-bit multiplexor @ bit 0: values 0, 1, 2; 32 dependent bits (100.0% of frames)"


def pulsing(n: int = 800) -> list[bytes]:
    """The Audi A3 0x0AF shape: one 16-bit value, idle at zero, then active.

    Bytes 2-3 are a little-endian value that is 0 for two frames and then
    somewhere in 256-511 for two. Bit 8 of that value correlates perfectly
    with idle-versus-active, and its low byte is constant through the idle
    frames and moving through the active ones -- the "still here, moving
    there" signature exactly.

    The real message also carries a counter in byte 1, left out here on
    purpose. With a perfectly periodic synthetic trace the counter's low bits
    lock to the pulse and become a selector in their own right, which is a
    different failure mode from the one under test; on the real trace they do
    not, because the phase drifts and the grouping stops being regular.
    """
    rng = random.Random(7)
    out = []
    for i in range(n):
        value = 0 if (i // 2) % 2 == 0 else 256 + rng.randrange(256)
        out.append(bytes([0x52, 0x5A]) + value.to_bytes(2, "little"))
    return out


class TestDependentBitsUseOnlyGroupedFrames:
    """Evidence has to come from the frames being explained."""

    def test_a_bit_moving_only_outside_every_group_is_not_dependent(self):
        """Rivian 0x247: 38 bits zero in all 15 layouts, data only outside.

        The inference the detector makes is that a bit constant inside every
        group must differ *between* groups. That only follows when "overall"
        means the grouped frames.
        """
        n = 200
        matrix = np.zeros((n, 16), dtype=np.uint8)
        matrix[:, 0] = [i % 2 for i in range(n)]          # the selector bit
        matrix[-6:, 8:] = 1                               # moves only in stray frames
        groups = [np.array([i % 2 == v and i < n - 6 for i in range(n)]) for v in (0, 1)]
        assert dependent_bits(matrix, groups, exclude={0}).bits.size == 0

    def test_a_bit_that_differs_between_groups_is_still_dependent(self):
        n = 200
        matrix = np.zeros((n, 16), dtype=np.uint8)
        matrix[:, 0] = [i % 2 for i in range(n)]
        matrix[1::2, 8:] = 1                              # one value's layout
        groups = [np.array([i % 2 == v for i in range(n)]) for v in (0, 1)]
        found = dependent_bits(matrix, groups, exclude={0})
        assert set(found.bits.tolist()) == set(range(8, 16))
        assert found.table == 8 and found.gated == 0


class TestLiveLayouts:
    """A selector value carrying nothing is an idle state, not a layout."""

    def test_counts_only_the_values_that_carry_something(self):
        matrix = np.zeros((40, 16), dtype=np.uint8)
        matrix[20:, 8:12] = 1                     # only the second group has content
        groups = [np.arange(40) < 20, np.arange(40) >= 20]
        assert live_layouts(matrix, groups, np.arange(8, 16)) == 1

    def test_two_layouts_that_both_carry_something(self):
        matrix = np.zeros((40, 16), dtype=np.uint8)
        matrix[:20, 8:10] = 1
        matrix[20:, 10:12] = 1
        groups = [np.arange(40) < 20, np.arange(40) >= 20]
        assert live_layouts(matrix, groups, np.arange(8, 16)) == MIN_LAYOUTS

    def test_no_dependent_bits_means_no_layouts(self):
        groups = [np.arange(40) < 20, np.arange(40) >= 20]
        assert live_layouts(np.zeros((40, 16), dtype=np.uint8), groups, np.array([])) == 0


class TestIdleIsNotALayout:
    def test_a_pulsing_value_is_not_a_multiplexor(self):
        """The Audi A3 0x0AF false positive, reduced to its essentials."""
        assert find_multiplexor(matrix_of(pulsing())) is None

    def test_a_blank_layout_among_several_real_ones_is_tolerated(self):
        """VW 0x3FB has 20 selector values, 12 of which carry nothing."""
        slices = (b"AAAAAAA", b"BBBBBBB", bytes(7), b"CCCCCCC")
        payloads = [bytes([i % 4]) + slices[i % 4] for i in range(800)]
        found = find_multiplexor(matrix_of(payloads))
        assert found is not None
        assert len(found.values) == 4


class TestACounterIsNotASelector:
    """A counter advancing every frame is a relabelling of the frame index."""

    @staticmethod
    def periodic_with_counter(n: int = 800) -> list[bytes]:
        """Byte 0 counts 0-255; bytes 2-3 pulse with period 4, locked to it."""
        rng = random.Random(11)
        out = []
        for i in range(n):
            value = 0 if (i // 2) % 2 == 0 else 256 + rng.randrange(256)
            out.append(bytes([i % 256, 0x5A]) + value.to_bytes(2, "little"))
        return out

    def test_a_signal_merely_in_phase_with_a_counter_is_not_a_layout(self):
        payloads = self.periodic_with_counter()
        counters = find_counters(matrix_of(payloads))
        assert counters, "the message really does carry a counter"
        bits = {b for c in counters for b in range(c.start_bit, c.end_bit)}
        assert find_multiplexor(matrix_of(payloads), counter_bits=bits) is None

    def test_without_telling_it_about_the_counter_it_is_fooled(self):
        """The rule needs the counter; it cannot be inferred from the window."""
        payloads = self.periodic_with_counter()
        assert find_multiplexor(matrix_of(payloads)) is not None

    def test_a_counter_that_selects_fixed_slices_is_still_a_multiplexor(self):
        """The VIN shape: the selector is a counter, and it decides content."""
        payloads = vin_like()
        counters = find_counters(matrix_of(payloads))
        bits = {b for c in counters for b in range(c.start_bit, c.end_bit)}
        found = find_multiplexor(matrix_of(payloads), counter_bits=bits)
        assert found is not None
        assert (found.start_bit, found.length) == (0, 2)


class TestTrimToMoving:
    """Narrowing a selector to the bits the trace actually justifies."""

    def test_a_byte_holding_three_values_trims_to_two_bits(self):
        matrix = matrix_of(vin_like())
        wide = MultiplexHypothesis(0, 8, (0, 1, 2), (200, 200, 200), (8, 9), 600)
        tight = trim_to_moving(matrix, wide)
        assert (tight.start_bit, tight.length) == (0, 2)
        assert tight.values == (0, 1, 2)

    def test_the_partition_survives_the_trim(self):
        matrix = matrix_of(vin_like())
        wide = MultiplexHypothesis(0, 8, (0, 1, 2), (200, 200, 200), (8, 9), 600)
        tight = trim_to_moving(matrix, wide)
        assert tight.frames_per_value == (200, 200, 200)
        assert tight.coverage == 1.0

    def test_dependent_bits_are_carried_through_untouched(self):
        matrix = matrix_of(vin_like())
        wide = MultiplexHypothesis(0, 8, (0, 1, 2), (200, 200, 200), (8, 9, 10), 600)
        assert trim_to_moving(matrix, wide).dependent_bits == (8, 9, 10)

    def test_a_selector_already_the_moving_span_is_left_alone(self):
        matrix = matrix_of(vin_like())
        tight = MultiplexHypothesis(0, 2, (0, 1, 2), (200, 200, 200), (8, 9), 600)
        assert trim_to_moving(matrix, tight) is tight

    def test_a_high_selector_keeps_its_offset(self):
        """Values 0 and 0x10 in a byte: only bit 4 moves."""
        payloads = [bytes([0x10 * (i % 2), 0xAA, 0xBB]) for i in range(200)]
        matrix = matrix_of(payloads)
        wide = MultiplexHypothesis(0, 8, (0, 16), (100, 100), (8, 9), 200)
        tight = trim_to_moving(matrix, wide)
        assert (tight.start_bit, tight.length) == (4, 1)
        assert tight.values == (0, 1)

    def test_a_gap_between_moving_bits_keeps_the_enclosing_span(self):
        """Bits 0 and 2 move, bit 1 does not; the field is still three bits."""
        payloads = [bytes([(0, 1, 4, 5)[i % 4], 0xAA]) for i in range(200)]
        matrix = matrix_of(payloads)
        wide = MultiplexHypothesis(0, 8, (0, 1, 4, 5), (50, 50, 50, 50), (8, 9), 200)
        tight = trim_to_moving(matrix, wide)
        assert (tight.start_bit, tight.length) == (0, 3)


class TestInferMessageIntegration:
    def test_the_selector_and_its_slices_are_not_counters(self):
        """The VIN selector counts 0, 1, 2 and each ASCII slice cycles with it."""
        payloads = vin_like()
        raw = find_counters(matrix_of(payloads))
        assert raw, "the scan alone does see the selector as a counter"
        found = infer_message(payloads, bus=0, address=0x3E0)
        assert found.multiplexor is not None
        assert found.counters == []
        assert found.found_anything

    def test_an_alive_counter_on_a_multiplexed_message_survives(self):
        """A counter advancing every frame keeps only its lowest bit locked
        to a two-frame schedule; the counter is still reported."""
        layouts = (bytes([0x11, 0x22, 0x33, 0x44]), bytes([0xEE, 0xDD, 0xCC, 0xBB]))
        payloads = [
            bytes([i % 2]) + layouts[i % 2] + bytes([0, i % 16, 0]) for i in range(600)
        ]
        found = infer_message(payloads, bus=0, address=0x221)
        assert found.multiplexor is not None
        assert [(c.start_bit, c.length) for c in found.counters] == [(48, 4)]


@pytest.mark.parametrize("width", [8, 16, 64])
def test_wide_payloads_are_searched_at_every_byte(width):
    """A VIN-style table placed at the end of a wide frame is still found."""
    n = 600
    payloads = []
    for i in range(n):
        head = bytes(width - 8)
        tail = bytes([i % 3]) + [b"AAAAAAA", b"BBBBBBB", b"CCCCCCC"][i % 3]
        payloads.append(head + tail)
    found = find_multiplexor(matrix_of(payloads))
    assert found is not None
    assert found.start_bit == (width - 8) * 8
