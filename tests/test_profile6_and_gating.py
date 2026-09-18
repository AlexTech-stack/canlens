# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Profile 6, and the applicability gate seen end to end through infer."""
from __future__ import annotations

import random

import pytest

from canlens.infer import infer_message
from canlens.infer.checksums import as_matrix
from canlens.infer.profiles import HEADER_BYTES, P04, P07, find_p06, p06_crc


def p06_frames(data_id=0x2B1C, n=200, width=16, offset=0, length=None, seed=3):
    rng = random.Random(seed)
    length = width if length is None else length
    out = []
    for i in range(n):
        body = bytearray(rng.randrange(256) for _ in range(width))
        body[offset + 2 : offset + 4] = length.to_bytes(2, "big")
        body[offset + 4] = i % 256
        crc = p06_crc(bytes(body), offset, data_id)
        body[offset : offset + 2] = crc.to_bytes(2, "big")
        out.append(bytes(body))
    return out


class TestProfile6:
    def test_recovers_the_data_id_high_byte_first(self):
        found = find_p06(as_matrix(p06_frames()))
        assert len(found) == 1
        hit = found[0]
        assert (hit.algorithm, hit.start_byte, hit.byteorder, hit.nbytes) == ("e2e_p06", 0, "big", 2)
        assert hit.data_id == 0x2B1C
        assert hit.match_rate == pytest.approx(1.0)

    def test_the_appended_order_is_not_profile_5s(self):
        # Built with the ID appended high-then-low; a Profile 5 reading of the
        # same frames would return the bytes swapped. Pinning the order.
        found = find_p06(as_matrix(p06_frames(data_id=0x0102)))
        assert found[0].data_id == 0x0102

    def test_padding_beyond_length_is_outside_the_crc(self):
        found = find_p06(as_matrix(p06_frames(width=32, length=12)))
        assert found and found[0].data_id == 0x2B1C

    def test_header_at_an_offset(self):
        found = find_p06(as_matrix(p06_frames(width=24, offset=3)))
        assert found and found[0].start_byte == 3

    def test_an_implausible_length_stops_before_any_crc(self):
        assert find_p06(as_matrix(p06_frames(length=900))) == []

    def test_too_narrow(self):
        assert find_p06(as_matrix(p06_frames(width=16))[:, :4]) == []

    def test_random_payloads_yield_nothing(self):
        rng = random.Random(8)
        payloads = [bytes(rng.randrange(256) for _ in range(16)) for _ in range(200)]
        assert find_p06(as_matrix(payloads)) == []


class TestGatingEndToEnd:
    """What infer reports must respect what a payload can hold."""

    def test_an_8_byte_message_never_yields_a_wide_profile(self):
        rng = random.Random(10)
        payloads = [bytes(rng.randrange(256) for _ in range(8)) for _ in range(300)]
        result = infer_message(payloads, bus=0, address=0x123)
        names = {c.algorithm for c in result.crc16s} | {s.algorithm for s in result.checksums}
        assert not names & {"e2e_p06", "e2e_p04", "e2e_p07"}

    def test_a_profile_4_frame_is_found_only_on_the_wide_path(self):
        from tests.test_profiles import wide_frames

        payloads = wide_frames(P04, width=32)
        result = infer_message(payloads, bus=0, address=0x300)
        hits = [c for c in result.crc16s if c.algorithm == "e2e_p04"]
        assert len(hits) == 1 and hits[0].data_id == 0x1234ABCD and hits[0].nbytes == 4

    def test_a_profile_7_frame_needs_64_bytes(self):
        from tests.test_profiles import wide_frames

        payloads = wide_frames(P07, width=64)
        result = infer_message(payloads, bus=0, address=0x300)
        assert any(c.algorithm == "e2e_p07" and c.nbytes == 8 for c in result.crc16s)

    def test_a_profile_6_frame_end_to_end(self):
        result = infer_message(p06_frames(), bus=0, address=0x300)
        assert any(c.algorithm == "e2e_p06" and c.data_id == 0x2B1C for c in result.crc16s)

    def test_the_header_table_is_the_single_statement_of_applicability(self):
        assert HEADER_BYTES == {
            "e2e_p11": 2, "e2e_p22": 2, "e2e_p05": 3, "e2e_p06": 5, "e2e_p04": 12, "e2e_p07": 20
        }

    def test_export_length_follows_the_crc_width(self):
        from canlens.infer.crc16 import Crc16Hypothesis

        assert Crc16Hypothesis(8, "e2e_p04", "big", 1.0, 100, 1, nbytes=4).length == 32
        assert Crc16Hypothesis(0, "e2e_p07", "big", 1.0, 100, 1, nbytes=8).length == 64
        assert Crc16Hypothesis(0, "e2e_p05", "little", 1.0, 100, 1).length == 16


class TestEvidenceGate:
    """A solved secret needs data that constrains it."""

    @staticmethod
    def static_with_counter(width=8, n=320):
        # Constant payload, alive counter in byte 1, a byte-0 that follows the
        # counter one-to-one: sixteen distinct contents, sixteen "CRC" values.
        table = [((v * 37) ^ 0x5A) & 0xFF for v in range(16)]
        return [bytes([table[i % 16], i % 16] + [0x42] * (width - 2)) for i in range(n)]

    def test_no_profile_is_claimed_on_a_static_message(self):
        result = infer_message(self.static_with_counter(), bus=0, address=0x2A0)
        names = {c.algorithm for c in result.crc16s} | {s.algorithm for s in result.checksums}
        assert not {n for n in names if n.startswith("e2e_")}

    def test_distinct_contents_counts_what_is_left_after_the_crc(self):
        from canlens.infer.checksums import distinct_contents, enough_evidence

        matrix = as_matrix(self.static_with_counter())
        assert distinct_contents(matrix, [0]) == 16
        assert not enough_evidence(matrix, [0], 16)   # Profile 22: 16 equations, 16 unknowns
        assert enough_evidence(matrix, [0], 1)        # Profile 11: 16 equations, 1 unknown

    def test_a_genuinely_protected_message_still_passes(self):
        found = find_p06(as_matrix(p06_frames()))
        assert found and found[0].data_id == 0x2B1C
