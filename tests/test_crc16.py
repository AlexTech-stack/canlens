# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""16-bit CRC tests, including AUTOSAR E2E Profile 5.

Parameters come from AUTOSAR_PRS_E2EProtocol (FO R19-11): CRC16 is CCITT-FALSE
with polynomial 0x1021 and start value 0xFFFF [PRS_E2E_00400], Profile 5
appends the Data ID low byte then high byte [PRS_E2E_00401, Figure 6.54], and
the CRC is stored little-endian [Figure 6.55].
"""
from __future__ import annotations

import random

import pytest

from canlens.infer.crc16 import (
    CCITT_INDEX_BY_LOW,
    CCITT_TABLE,
    crc16_arc,
    crc16_autosar,
    crc16_ccitt_zero,
    e2e_p05,
    find_crc16,
    read_crc,
    solve_data_id,
    without,
)

CHECK = b"123456789"  # the standard CRC check string


class TestKnownAnswers:
    def test_autosar_crc16_is_ccitt_false(self):
        assert crc16_autosar(CHECK) == 0x29B1

    def test_ccitt_zero_start_is_xmodem(self):
        assert crc16_ccitt_zero(CHECK) == 0x31C3

    def test_arc(self):
        assert crc16_arc(CHECK) == 0xBB3D

    def test_empty_input_returns_the_start_value(self):
        assert crc16_autosar(b"") == 0xFFFF


class TestTable:
    def test_low_bytes_are_a_permutation(self):
        # This is what makes a CRC step invertible. The high bytes are NOT --
        # only 128 are distinct -- and indexing by them silently returns the
        # wrong entry, which is exactly how the first inversion was broken.
        assert len({v & 0xFF for v in CCITT_TABLE}) == 256
        assert len({v >> 8 for v in CCITT_TABLE}) < 256

    def test_index_lookup_round_trips(self):
        for index, value in enumerate(CCITT_TABLE):
            assert CCITT_INDEX_BY_LOW[value & 0xFF] == index


class TestHelpers:
    def test_without_removes_two_bytes(self):
        assert without(bytes(range(8)), 2) == bytes([0, 1, 4, 5, 6, 7])

    def test_read_crc_respects_byte_order(self):
        assert read_crc(b"\x34\x12", 0, "little") == 0x1234
        assert read_crc(b"\x12\x34", 0, "big") == 0x1234


def p05_message(data_id: int, start: int = 0, n: int = 120, width: int = 8, seed: int = 4):
    """Build frames carrying a valid Profile 5 CRC at `start`."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        body = bytearray(rng.randrange(256) for _ in range(width))
        body[start : start + 2] = b"\x00\x00"
        body[start + 2] = i % 256  # a counter, as Profile 5 also carries one
        crc = e2e_p05(bytes(body), start, data_id)
        body[start : start + 2] = crc.to_bytes(2, "little")
        out.append(bytes(body))
    return out


class TestSolveDataId:
    def test_recovers_the_identifier_that_built_the_frame(self):
        payloads = p05_message(0x0A5F)
        assert 0x0A5F in solve_data_id(payloads[0], 0, "little")

    def test_agrees_with_exhaustive_search(self):
        # The inversion must find exactly what a sweep over all 65536 would.
        payload = p05_message(0x1234)[0]
        target = read_crc(payload, 0, "little")
        brute = {i for i in range(1 << 16) if e2e_p05(payload, 0, i) == target}
        assert solve_data_id(payload, 0, "little") == brute

    def test_one_frame_does_not_pin_it_down_alone(self):
        # A single frame leaves several candidates; frames disagree except on
        # the real Data ID, which is why find_crc16 intersects them.
        solutions = [solve_data_id(p, 0, "little") for p in p05_message(0x0A5F)[:6]]
        assert len(set.intersection(*solutions)) == 1


class TestFindCrc16:
    def test_recovers_profile_5_and_its_implicit_data_id(self):
        found = find_crc16(p05_message(0x0A5F))
        assert len(found) == 1
        assert found[0].algorithm == "e2e_p05"
        assert found[0].data_id == 0x0A5F
        assert found[0].byteorder == "little"
        assert found[0].match_rate == 1.0
        assert found[0].start_bit == 0 and found[0].length == 16

    def test_finds_a_crc_that_is_not_at_the_start(self):
        found = find_crc16(p05_message(0x0111, start=4))
        assert found and found[0].start_byte == 4

    @pytest.mark.parametrize("order", ["little", "big"])
    def test_handles_both_byte_orders(self, order):
        rng = random.Random(9)
        payloads = []
        for _ in range(80):
            body = bytearray(rng.randrange(256) for _ in range(8))
            body[0:2] = crc16_autosar(bytes(body[2:])).to_bytes(2, order)
            payloads.append(bytes(body))
        found = find_crc16(payloads, search_data_id=False)
        assert found and found[0].byteorder == order
        assert found[0].algorithm == "crc16_autosar"

    def test_plain_crc_is_preferred_over_a_data_id_fit(self):
        rng = random.Random(10)
        payloads = []
        for _ in range(80):
            body = bytearray(rng.randrange(256) for _ in range(8))
            body[0:2] = crc16_autosar(bytes(body[2:])).to_bytes(2, "little")
            payloads.append(bytes(body))
        assert find_crc16(payloads)[0].algorithm == "crc16_autosar"

    def test_reports_nothing_for_random_payloads(self):
        rng = random.Random(12)
        payloads = [bytes(rng.randrange(256) for _ in range(8)) for _ in range(120)]
        assert find_crc16(payloads) == []

    def test_respects_the_candidate_filter(self):
        payloads = p05_message(0x0A5F)
        assert find_crc16(payloads, candidate_bytes=[4, 5]) == []
        assert find_crc16(payloads, candidate_bytes=[0])

    def test_too_few_frames(self):
        assert find_crc16(p05_message(0x1, n=1)) == []

    def test_data_id_search_can_be_switched_off(self):
        assert find_crc16(p05_message(0x0A5F), search_data_id=False) == []


class TestMatchRateHonesty:
    def test_a_corrupted_frame_lowers_the_reported_rate(self):
        payloads = p05_message(0x0A5F, n=200)
        payloads[7] = bytes([payloads[7][0] ^ 0xFF, *payloads[7][1:]])
        found = find_crc16(payloads, min_match=0.99)
        assert found and found[0].match_rate < 1.0
