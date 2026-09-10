# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Inference tests on synthetic payloads with known ground truth."""
from __future__ import annotations

import random

import numpy as np
import pytest

from canlens.analyze.bits import BitOrder, bit_matrix
from canlens.decode import CanFrame
from canlens.infer import (
    ALGORITHMS,
    checksum_candidate_bytes,
    field_values,
    find_checksums,
    find_counters,
    infer_frames,
    infer_message,
    score_algorithm,
    score_counter,
)
from canlens.infer.checksums import sum8, toyota, xor8


def matrix_of(payloads: list[bytes]) -> np.ndarray:
    return bit_matrix(payloads, len(payloads[0]), BitOrder.INTEL)


def counting_payloads(n: int = 512, width: int = 2, seed: int = 0) -> list[bytes]:
    """Byte 0 counts 0..255 and wraps; the rest is random."""
    rng = random.Random(seed)
    return [bytes([i % 256] + [rng.randrange(256) for _ in range(width - 1)]) for i in range(n)]


class TestFieldValues:
    def test_reads_a_byte_as_an_integer(self):
        m = matrix_of([bytes([0xA5, 0x00])])
        assert field_values(m, 0, 8).tolist() == [0xA5]

    def test_reads_a_nibble(self):
        m = matrix_of([bytes([0x3C, 0x00])])
        assert field_values(m, 0, 4).tolist() == [0xC]  # Intel: low nibble first
        assert field_values(m, 4, 4).tolist() == [0x3]

    def test_spans_a_byte_boundary(self):
        m = matrix_of([bytes([0xFF, 0x01])])
        assert field_values(m, 4, 8).tolist() == [0x1F]


class TestScoreCounter:
    def test_perfect_increment(self):
        assert score_counter(np.arange(100) % 16, 4) == (1, 1.0)

    def test_constant_field_is_not_a_counter(self):
        stride, rate = score_counter(np.full(50, 7), 4)
        assert stride == 0 and rate == 0.0

    def test_reports_the_dominant_stride(self):
        stride, rate = score_counter(np.arange(0, 200, 2) % 16, 4)
        assert stride == 2 and rate == 1.0

    def test_too_few_samples(self):
        assert score_counter(np.array([3]), 4) == (0, 0.0)


class TestFindCounters:
    def test_finds_a_byte_counter(self):
        found = find_counters(matrix_of(counting_payloads()))
        assert any(c.start_bit == 0 and c.length == 8 and c.stride == 1 for c in found)

    def test_prefers_the_longest_field(self):
        # The low nibble of a byte counter is itself a valid nibble counter.
        found = find_counters(matrix_of(counting_payloads()))
        starts = [(c.start_bit, c.length) for c in found]
        assert (0, 8) in starts
        assert (0, 4) not in starts

    def test_rejects_a_single_toggling_bit(self):
        # Bit 7 alternating gives a constant difference of 128 over 8 bits --
        # arithmetically perfect, but the field only ever holds two values.
        payloads = [bytes([(i % 2) << 7]) for i in range(200)]
        assert find_counters(matrix_of(payloads)) == []

    def test_rejects_every_even_stride_on_a_power_of_two_field(self):
        # Even strides never walk the whole range; they are toggles in disguise.
        for stride in (2, 4, 8, 16):
            payloads = [bytes([(i * stride) % 256]) for i in range(400)]
            for c in find_counters(matrix_of(payloads)):
                assert np.gcd(c.stride, 1 << c.length) == 1

    def test_rejects_a_field_that_barely_moves(self):
        payloads = [bytes([i % 3]) for i in range(200)]
        assert all(c.length <= 2 for c in find_counters(payloads and matrix_of(payloads)))

    def test_random_data_yields_no_counter(self):
        rng = random.Random(7)
        payloads = [bytes([rng.randrange(256), rng.randrange(256)]) for _ in range(400)]
        assert find_counters(matrix_of(payloads)) == []

    def test_too_few_frames(self):
        assert find_counters(matrix_of([b"\x00"])) == []


class TestChecksumAlgorithms:
    @pytest.mark.parametrize("name", list(ALGORITHMS))
    def test_every_algorithm_is_deterministic_and_a_byte(self, name):
        fn = ALGORITHMS[name]
        payload = bytes(range(8))
        first = fn(payload, 0x123, 7)
        assert first == fn(payload, 0x123, 7)
        assert 0 <= first <= 0xFF

    def test_sum8_ignores_the_checksum_byte_itself(self):
        assert sum8(b"\x01\x02\xff", 0, 2) == 3

    def test_xor8_ignores_the_checksum_byte_itself(self):
        assert xor8(b"\x0f\xf0\xaa", 0, 2) == 0xFF

    def test_toyota_folds_in_the_address_and_length(self):
        # Same bytes, different address -> different checksum.
        assert toyota(b"\x00" * 8, 0x100, 7) != toyota(b"\x00" * 8, 0x200, 7)


class TestFindChecksums:
    @staticmethod
    def build(algorithm: str, address: int, n: int = 200, index: int = 7) -> list[bytes]:
        fn = ALGORITHMS[algorithm]
        rng = random.Random(3)
        out = []
        for _ in range(n):
            body = bytearray(rng.randrange(256) for _ in range(8))
            body[index] = 0
            body[index] = fn(bytes(body), address, index)
            out.append(bytes(body))
        return out

    @pytest.mark.parametrize("name", ["sum8", "toyota", "crc8_j1850", "crc8_2f"])
    def test_recovers_the_algorithm_that_built_the_data(self, name):
        found = find_checksums(self.build(name, 0x2C1), 0x2C1)
        assert [(f.byte_index, f.algorithm) for f in found] == [(7, name)]
        assert found[0].match_rate == 1.0
        assert not found[0].ambiguous

    def test_xor_is_reported_once_and_flagged_ambiguous(self):
        # XOR is self-inverse: if byte 7 is the XOR of the rest, then so is
        # every other byte. One finding, not eight, and the ambiguity is said
        # out loud rather than resolved by guessing.
        found = find_checksums(self.build("xor8", 0x2C1), 0x2C1)
        assert len(found) == 1
        assert found[0].algorithm == "xor8" and found[0].ambiguous
        assert found[0].ambiguous_positions == tuple(range(8))
        assert found[0].byte_index == 7  # convention: last byte

    def test_score_algorithm_reports_a_fraction(self):
        payloads = self.build("sum8", 0x1)
        assert score_algorithm(payloads, 0x1, 7, ALGORITHMS["sum8"]) == 1.0
        assert score_algorithm(payloads, 0x1, 0, ALGORITHMS["toyota"]) < 0.5

    def test_reports_nothing_for_random_payloads(self):
        rng = random.Random(11)
        payloads = [bytes(rng.randrange(256) for _ in range(8)) for _ in range(200)]
        assert find_checksums(payloads, 0x123) == []

    def test_respects_the_candidate_byte_filter(self):
        payloads = self.build("sum8", 0x123)
        assert find_checksums(payloads, 0x123, candidate_bytes=[0, 1]) == []
        assert find_checksums(payloads, 0x123, candidate_bytes=[7])

    def test_out_of_range_candidates_are_ignored(self):
        assert find_checksums(self.build("sum8", 0x1), 0x1, candidate_bytes=[99]) == []

    def test_empty_input(self):
        assert find_checksums([], 0x123) == []

    def test_match_rate_is_reported_not_rounded_away(self):
        payloads = self.build("sum8", 0x123, n=200)
        broken = list(payloads)
        broken[5] = bytes([*broken[5][:7], (broken[5][7] + 1) & 0xFF])
        found = find_checksums(broken, 0x123, min_match=0.99)
        assert found and found[0].match_rate < 1.0


class TestInferMessage:
    def test_finds_a_counter_and_a_checksum_together(self):
        rng = random.Random(5)
        payloads = []
        for i in range(400):
            body = bytearray([i % 256, *(rng.randrange(256) for _ in range(6)), 0])
            body[7] = toyota(bytes(body), 0x210, 7)
            payloads.append(bytes(body))
        result = infer_message(payloads, bus=1, address=0x210)
        assert result.found_anything
        assert any(c.start_bit == 0 and c.length == 8 for c in result.counters)
        assert [(s.byte_index, s.algorithm) for s in result.checksums] == [(7, "toyota")]

    def test_candidate_bytes_track_movement(self):
        result = infer_message(counting_payloads(width=2), bus=0, address=1)
        assert 0 in checksum_candidate_bytes(result.bits)


class TestInferFrames:
    @staticmethod
    def frames(payloads, address=0x210, bus=1):
        return [CanFrame(i * 10_000_000, bus, address, p, False) for i, p in enumerate(payloads)]

    def test_skips_messages_with_too_few_samples(self):
        assert infer_frames(self.frames(counting_payloads(n=10))) == []

    def test_analyses_messages_above_the_threshold(self):
        results = infer_frames(self.frames(counting_payloads(n=200)))
        assert len(results) == 1 and results[0].key == (1, 0x210)

    def test_groups_by_bus_and_address(self):
        payloads = counting_payloads(n=100)
        results = infer_frames(
            self.frames(payloads, address=0x100, bus=0) + self.frames(payloads, address=0x100, bus=1)
        )
        assert {r.key for r in results} == {(0, 0x100), (1, 0x100)}

    def test_uses_only_the_dominant_payload_length(self):
        payloads = counting_payloads(n=200) + [b"\x00"] * 5
        results = infer_frames(self.frames(payloads))
        assert results[0].width == 2
        assert results[0].frames == 200


class TestCounterPrefilter:
    """A counter's step is odd, so its lowest bit must flip every frame."""

    def test_prefilter_does_not_lose_a_real_counter(self):
        from canlens.infer.counters import find_counters as fc

        m = matrix_of(counting_payloads())
        assert fc(m) == fc(m, min_lsb_rate=0.0)

    def test_a_field_whose_lsb_is_quiet_is_not_a_counter(self):
        # High nibble ramps, low nibble constant: the low bit never moves.
        payloads = [bytes([(i % 16) << 4]) for i in range(300)]
        assert all(c.start_bit != 0 for c in find_counters(matrix_of(payloads)))


class TestEarlyExit:
    """Abandoning hopeless candidates must not change any verdict."""

    def test_score_algorithm_matches_the_exhaustive_count(self):
        payloads = TestFindChecksums.build("sum8", 0x123, n=100)
        fn = ALGORITHMS["sum8"]
        assert score_algorithm(payloads, 0x123, 7, fn, 0.99) == 1.0
        # A hopeless candidate returns something below the threshold, which is
        # all the caller uses it for.
        assert score_algorithm(payloads, 0x123, 7, ALGORITHMS["xor8"], 0.99) < 0.99
