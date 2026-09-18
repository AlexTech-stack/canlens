# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""AUTOSAR E2E Profile 1/11 CRC-8 detection, and the counter that wraps at 15.

Frames are built exactly as AUTOSAR_PRS_E2EProtocol describes: Data ID bytes
first, then every byte but the CRC, register from 0x00, result XOR 0xFF
([PRS_E2E_00082], [PRS_E2E_00505], [PRS_E2E_00506]); the alive counter runs
0..14 with 0x0F reserved ([PRS_E2E_00504]).
"""
from __future__ import annotations

import random

from canlens.analyze.bits import BitOrder, bit_matrix
from canlens.infer import e2e_crc8, find_counters, find_e2e_crc8, infer_message
from canlens.infer.checksums import E2E_XOR, J1850_TABLE, _crc8


def frames(mode: str, data_id: int, n: int = 240, seed: int = 0, width: int = 8) -> list[bytes]:
    """Profile 11 frames: CRC at byte 0, 0..14 alive counter in byte 1's low nibble."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        body = bytearray(rng.randrange(256) for _ in range(width))
        counter = i % 15
        body[1] = counter  # high nibble constant 0, as on the wire
        low, high = data_id & 0xFF, (data_id >> 8) & 0xFF
        ident = {
            "nibble": bytes([low, 0x00]),
            "both": bytes([low, high]),
            "low": bytes([low]),
            "alt": bytes([low]) if counter % 2 == 0 else bytes([high]),
        }[mode]
        body[0] = e2e_crc8(bytes(body), 0, ident)
        out.append(bytes(body))
    return out


class TestReference:
    def test_table_is_sae_j1850(self):
        # The scalar CRC8 in ALGORITHMS uses the same polynomial; feeding the
        # table one byte from a zero register must agree with it.
        for byte in (0x00, 0x01, 0x80, 0xFF):
            assert int(J1850_TABLE[byte]) == _crc8(bytes([byte]), 0x1D, 0x00, 0x00)

    def test_id_bytes_come_first_and_the_crc_byte_is_skipped(self):
        # The final XOR is whatever the data settled on (see checksums.py);
        # the test pins the *ordering*, not a particular XOR value.
        payload = bytes([0xAA, 1, 2, 3])
        direct = _crc8(bytes([0x42, 0x00, 1, 2, 3]), 0x1D, 0x00, 0x00) ^ E2E_XOR
        assert e2e_crc8(payload, 0, bytes([0x42, 0x00])) == direct

    def test_result_is_a_byte(self):
        assert 0 <= e2e_crc8(b"\x00" * 8, 0, b"\x00\x00") <= 0xFF


class TestDetector:
    def test_a_data_id_equal_to_the_can_identifier_is_recognised(self):
        # BOTH mode with the identifier itself as the Data ID -- what Rivian does.
        rng = random.Random(4)
        addr = 0x135
        out = []
        for i in range(240):
            body = bytearray(rng.randrange(256) for _ in range(8))
            body[1] = i % 15
            body[0] = e2e_crc8(bytes(body), 0, bytes([addr & 0xFF, addr >> 8]))
            out.append(bytes(body))
        found = find_e2e_crc8(out, address=addr)
        assert [(f.algorithm, f.data_id, f.match_rate) for f in found] == [("e2e_p11", addr, 1.0)]

    def test_the_identifier_form_is_not_claimed_when_it_does_not_hold(self):
        # A zero-high-byte message under an address whose low byte differs:
        # the address form's recovered byte will not equal addr & 0xFF, so the
        # spec's NIBBLE reading is reported instead.
        found = find_e2e_crc8(frames("nibble", 0x0037), address=0x2C1)
        assert found[0].data_id == 0x37

    def test_recovers_a_profile_11_data_id(self):
        found = find_e2e_crc8(frames("nibble", 0x0037))
        assert [(f.byte_index, f.algorithm, f.data_id) for f in found] == [(0, "e2e_p11", 0x37)]
        assert found[0].match_rate == 1.0

    def test_both_mode_with_a_zero_high_byte_is_the_same_hypothesis(self):
        assert find_e2e_crc8(frames("both", 0x0037))[0].algorithm == "e2e_p11"

    def test_profile_1_low_mode_is_the_same_hypothesis_with_a_convertible_id(self):
        # One trace cannot tell LOW from NIBBLE: both map 256 IDs onto the same
        # 256 register states. The message is found; the LOW reading is derived.
        from canlens.infer import p01_low_id

        found = find_e2e_crc8(frames("low", 0x5A))
        assert found[0].algorithm == "e2e_p11" and found[0].match_rate == 1.0
        assert p01_low_id(found[0].data_id) == 0x5A

    def test_profile_1_alt_mode_needs_the_counter(self):
        payloads = frames("alt", 0x2B1C)
        assert find_e2e_crc8(payloads) == []
        found = find_e2e_crc8(payloads, counter=(8, 4))
        assert (found[0].algorithm, found[0].data_id) == ("e2e_p01_alt", 0x2B1C)
        assert found[0].match_rate == 1.0

    def test_random_payloads_yield_nothing(self):
        rng = random.Random(9)
        payloads = [bytes(rng.randrange(256) for _ in range(8)) for _ in range(240)]
        assert find_e2e_crc8(payloads) == []

    def test_a_crc_elsewhere_than_byte_zero(self):
        rng = random.Random(3)
        out = []
        for _ in range(200):
            body = bytearray(rng.randrange(256) for _ in range(8))
            body[5] = e2e_crc8(bytes(body), 5, bytes([0x10, 0x00]))
            out.append(bytes(body))
        found = find_e2e_crc8(out)
        assert [(f.byte_index, f.data_id) for f in found] == [(5, 0x10)]

    def test_candidate_filter_is_respected(self):
        assert find_e2e_crc8(frames("nibble", 0x37), candidate_bytes=[3, 4]) == []

    def test_too_few_frames(self):
        assert find_e2e_crc8(frames("nibble", 0x37, n=4)) == []


class TestAliveCounter:
    """0..14 in four bits: one odd step per cycle under the natural modulus."""

    def payload_matrix(self):
        return bit_matrix(frames("nibble", 0x37), 8, BitOrder.INTEL)

    def test_the_alive_counter_is_found_with_its_true_wrap(self):
        found = [c for c in find_counters(self.payload_matrix()) if c.start_bit == 8]
        assert len(found) == 1
        c = found[0]
        assert (c.length, c.stride, c.modulus, c.wraps_at) == (4, 1, 15, 15)
        assert c.match_rate == 1.0
        assert "mod 15" in str(c)

    def test_a_full_range_counter_keeps_the_natural_wrap(self):
        payloads = [bytes([i % 256, 0]) for i in range(300)]
        c = find_counters(bit_matrix(payloads, 2, BitOrder.INTEL))[0]
        assert c.modulus == 0 and c.wraps_at == 256 and "mod" not in str(c)

    def test_constant_bits_above_the_counter_are_not_claimed(self):
        # byte 1 = 0b00cccc with the alive counter in the low nibble: a 6-bit
        # window mod 15 fits perfectly, but the counter is four bits wide.
        found = [c for c in find_counters(self.payload_matrix()) if c.start_bit == 8]
        assert [(c.length, c.modulus) for c in found] == [(4, 15)]

    def test_period_follows_the_wrap(self):
        from canlens.infer.counters import CounterHypothesis

        assert CounterHypothesis(8, 4, 1, 1.0, 100, modulus=15).period == 15
        assert CounterHypothesis(8, 4, 1, 1.0, 100).period == 16

    def test_noise_does_not_get_to_choose_its_own_modulus(self):
        # Only the natural wrap and the one the data's maximum implies are
        # tried; a random field must not be fitted by a convenient modulus.
        rng = random.Random(5)
        payloads = [bytes([rng.randrange(256), rng.randrange(256)]) for _ in range(300)]
        assert find_counters(bit_matrix(payloads, 2, BitOrder.INTEL)) == []


class TestEndToEnd:
    def test_infer_message_reports_both_the_crc_and_the_alive_counter(self):
        payloads = frames("nibble", 0x0037)
        result = infer_message(payloads, bus=0, address=0x102)
        assert any(c.start_bit == 8 and c.modulus == 15 for c in result.counters)
        assert any(s.algorithm == "e2e_p11" and s.data_id == 0x37 for s in result.checksums)

    def test_the_columnar_path_agrees(self):
        from canlens.decode import CanFrame, from_frames
        from canlens.infer import infer_frameset

        payloads = frames("nibble", 0x0037)
        fs = from_frames([CanFrame(i * 10_000_000, 0, 0x102, p, False) for i, p in enumerate(payloads)])
        result = infer_frameset(fs)[0]
        assert any(s.algorithm == "e2e_p11" and s.data_id == 0x37 for s in result.checksums)
        assert any(c.modulus == 15 for c in result.counters)
