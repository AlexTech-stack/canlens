# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""E2E Profiles 22, 4 and 7, on frames built exactly as the spec constructs them."""
from __future__ import annotations

import random

import pytest

from canlens.analyze.bits import BitOrder, bit_matrix
from canlens.infer.checksums import as_matrix
from canlens.infer.counters import CounterHypothesis, find_counters
from canlens.infer.profiles import (
    P04,
    P07,
    Profile,
    applicable,
    find_p22,
    find_wide,
    p22_constant,
    p22_crc,
    p22_id_list,
    wide_crc,
)


def p22_frames(id_list, n=400, width=8, crc_index=0, counter_byte=1, seed=0):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        body = bytearray(rng.randrange(256) for _ in range(width))
        counter = i % 16
        body[counter_byte] = (body[counter_byte] & 0xF0) | counter
        body[crc_index] = p22_crc(bytes(body), crc_index, id_list[counter])
        out.append(bytes(body))
    return out


def wide_frames(profile, n=200, width=32, offset=0, data_id=0x1234ABCD, seed=1, length=None):
    rng = random.Random(seed)
    length = width if length is None else length
    out = []
    for i in range(n):
        body = bytearray(rng.randrange(256) for _ in range(width))
        h = offset
        body[h + profile.length_at : h + profile.length_at + profile.length_bytes] = length.to_bytes(
            profile.length_bytes, "big"
        )
        body[h + profile.counter_at : h + profile.counter_at + profile.counter_bytes] = (
            i % (1 << (8 * profile.counter_bytes))
        ).to_bytes(profile.counter_bytes, "big")
        body[h + profile.data_id_at : h + profile.data_id_at + 4] = data_id.to_bytes(4, "big")
        crc = wide_crc(profile, bytes(body), offset)
        body[h + profile.crc_at : h + profile.crc_at + profile.crc_bytes] = crc.to_bytes(
            profile.crc_bytes, "big"
        )
        out.append(bytes(body))
    return out


ID_LIST = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x10]


class TestProfile22:
    def counters(self, payloads):
        return find_counters(bit_matrix(payloads, len(payloads[0]), BitOrder.INTEL))

    def test_recovers_all_sixteen_data_ids(self):
        payloads = p22_frames(ID_LIST)
        found = find_p22(as_matrix(payloads), self.counters(payloads))
        assert len(found) == 1
        hit = found[0]
        assert (hit.byte_index, hit.algorithm) == (0, "e2e_p22")
        assert p22_id_list(hit.data_id) == ID_LIST
        assert hit.match_rate == pytest.approx(1.0)

    def test_the_counter_it_relies_on_is_the_full_0_to_15_one(self):
        # Profile 22's counter has no reserved value, unlike Profile 11's.
        payloads = p22_frames(ID_LIST)
        c = next(c for c in self.counters(payloads) if c.start_bit == 8)
        assert (c.length, c.modulus) == (4, 0)

    def test_a_crc_elsewhere_than_byte_zero(self):
        payloads = p22_frames(ID_LIST, crc_index=7, counter_byte=1)
        found = find_p22(as_matrix(payloads), self.counters(payloads))
        assert found and found[0].byte_index == 7

    def test_random_payloads_yield_nothing(self):
        rng = random.Random(4)
        payloads = [bytes(rng.randrange(256) for _ in range(8)) for _ in range(400)]
        counters = [CounterHypothesis(8, 4, 1, 1.0, 400)]  # pretend, to reach the search
        assert find_p22(as_matrix(payloads), counters) == []

    def test_a_constant_xor_on_the_crc_is_absorbed_into_the_list(self):
        # The table is linear over GF(2): flipping every group-7 CRC by the
        # same constant is indistinguishable from a different list entry.
        # Not a weakness of the detector -- a property of the construction.
        good = p22_frames(ID_LIST)
        shifted = [(p if (p[1] & 0xF) != 7 else bytes([p[0] ^ 0x01]) + p[1:]) for p in good]
        found = find_p22(as_matrix(shifted), self.counters(shifted))
        assert found
        ids = p22_id_list(found[0].data_id)
        assert ids[:7] + ids[8:] == ID_LIST[:7] + ID_LIST[8:]
        assert ids[7] != ID_LIST[7]

    def test_a_frame_dependent_corruption_fails_the_whole_list(self):
        # Corruption that varies per frame cannot be a list entry.
        good = p22_frames(ID_LIST)
        broken = [(p if (p[1] & 0xF) != 7 else bytes([p[0] ^ p[3]]) + p[1:]) for p in good]
        assert find_p22(as_matrix(broken), self.counters(broken)) == []

    def test_needs_a_counter(self):
        assert find_p22(as_matrix(p22_frames(ID_LIST)), []) == []

    def test_needs_enough_frames_per_counter_value(self):
        payloads = p22_frames(ID_LIST, n=40)
        assert find_p22(as_matrix(payloads), self.counters(payloads)) == []


class TestWideProfiles:
    @pytest.mark.parametrize("profile,width", [(P04, 32), (P04, 64), (P07, 64)])
    def test_recovers_the_header(self, profile, width):
        payloads = wide_frames(profile, width=width)
        found = find_wide(profile, as_matrix(payloads))
        assert len(found) == 1
        hit = found[0]
        assert hit.algorithm == profile.name
        assert hit.start_byte == profile.crc_at
        assert hit.nbytes == profile.crc_bytes
        assert hit.length == profile.crc_bytes * 8
        assert hit.byteorder == "big"
        assert hit.data_id == 0x1234ABCD
        assert hit.match_rate == pytest.approx(1.0)

    def test_padding_beyond_length_is_outside_the_crc(self):
        # A 32-byte frame carrying 20 protected bytes: Length says so, and the
        # CRC must be computed over those 20 only.
        payloads = wide_frames(P04, width=32, length=20)
        found = find_wide(P04, as_matrix(payloads))
        assert found and found[0].data_id == 0x1234ABCD

    def test_a_header_at_an_offset(self):
        payloads = wide_frames(P04, width=32, offset=4)
        found = find_wide(P04, as_matrix(payloads))
        assert found and found[0].start_byte == 4 + P04.crc_at

    def test_too_narrow_for_the_header(self):
        assert find_wide(P04, as_matrix(wide_frames(P04, width=32))[:, :8]) == []
        assert find_wide(P07, as_matrix(wide_frames(P04, width=16))) == []

    def test_an_implausible_length_field_stops_before_any_crc(self):
        payloads = wide_frames(P04, width=32, length=200)  # longer than the frame
        assert find_wide(P04, as_matrix(payloads)) == []

    def test_random_payloads_yield_nothing(self):
        rng = random.Random(6)
        payloads = [bytes(rng.randrange(256) for _ in range(32)) for _ in range(200)]
        assert find_wide(P04, as_matrix(payloads)) == []

    def test_scalar_reference_skips_only_the_crc(self):
        payload = wide_frames(P04, n=1, width=16)[0]
        crc = int.from_bytes(payload[8:12], "big")
        assert wide_crc(P04, payload) == crc


class TestApplicability:
    def test_only_profiles_that_fit_are_offered(self):
        profiles = [Profile("a", 2, list), Profile("b", 5, list), Profile("c", 12, list),
                    Profile("d", 20, list)]
        assert [p.name for p in applicable(8, profiles)] == ["a", "b"]
        assert [p.name for p in applicable(12, profiles)] == ["a", "b", "c"]
        assert [p.name for p in applicable(64, profiles)] == ["a", "b", "c", "d"]
        assert applicable(1, profiles) == []


class TestP22Constant:
    """Volkswagen MQB: Profile 22 whose sixteen Data IDs are one repeated byte."""

    @staticmethod
    def frames(magic=0xFB, n=400, width=8, crc_index=0, distinct=True, seed=3):
        rng = random.Random(seed)
        out = []
        for i in range(n):
            body = bytearray(rng.randrange(256) if distinct else 0 for _ in range(width))
            body[1] = (body[1] & 0xF0) | (i % 16)
            body[crc_index] = p22_crc(bytes(body), crc_index, magic)
            out.append(bytes(body))
        return as_matrix(out)

    def test_solves_the_repeated_data_id(self):
        found = p22_constant(self.frames(magic=0xFB), 0, 0.99)
        assert found is not None
        assert found.algorithm == "e2e_p22" and found.match_rate == 1.0
        assert p22_id_list(found.data_id) == [0xFB] * 16

    def test_found_without_any_counter_being_known(self):
        """One constant has to reproduce every frame, so no counter is needed."""
        found = find_p22(self.frames(), [], candidate_bytes=[0])
        assert [f.byte_index for f in found] == [0]
        assert p22_id_list(found[0].data_id) == [0xFB] * 16

    def test_a_short_trace_is_still_enough(self):
        """The list form needs 64 frames; one constant does not."""
        found = find_p22(self.frames(n=40), [], candidate_bytes=[0])
        assert len(found) == 1

    def test_a_message_with_too_few_distinct_payloads_is_refused(self):
        """Nine distinct contents are the bar for a one-byte secret."""
        matrix = self.frames(n=400, distinct=False)  # only the counter moves: 16 contents
        assert p22_constant(matrix[:8], 0, 0.99) is None

    def test_the_wrong_constant_does_not_fit(self):
        found = p22_constant(self.frames(magic=0x11), 0, 0.99)
        assert found is not None and p22_id_list(found.data_id) == [0x11] * 16

    def test_random_bytes_are_not_a_profile_22(self):
        rng = random.Random(9)
        matrix = as_matrix([bytes(rng.randrange(256) for _ in range(8)) for _ in range(400)])
        assert p22_constant(matrix, 0, 0.99) is None

    def test_a_real_sixteen_entry_list_still_needs_the_list_form(self):
        """A constant must not be reported where the entries genuinely differ."""
        ids = [0x10 + i for i in range(16)]
        payloads = p22_frames(ids)
        matrix = as_matrix(payloads)
        assert p22_constant(matrix, 0, 0.99) is None
        counters = find_counters(bit_matrix(payloads, 8, BitOrder.INTEL))
        found = find_p22(matrix, counters)
        assert found and p22_id_list(found[0].data_id) == ids
