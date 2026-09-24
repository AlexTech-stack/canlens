# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Reassembling segmented ISO-TP transfers, on exchanges built to the spec.

Synthetic conversations with a known payload, so what is under test is the
reassembly and the abort rules -- not whether a particular car happens to run
a diagnostic session while it was being recorded.
"""
from __future__ import annotations

import pytest

from canlens.infer.isotp import (
    MAX_GAP_NS,
    MIN_MESSAGES,
    IsotpHypothesis,
    find_isotp,
    ordered_frames,
    transport_bits,
)

STEP = 10_000_000  # 10 ms between frames


def exchange(payload, *, address=0x7E0, bus=0, peer=None, start_ns=0, step=STEP,
             break_at=None, gap_before=None):
    """A FirstFrame and its ConsecutiveFrames, optionally answered.

    `break_at` corrupts that ConsecutiveFrame's SequenceNumber; `gap_before`
    delays it, both to exercise the abort rules in 9.6.4.
    """
    out = []
    now = start_ns
    head = bytes([0x10 | (len(payload) >> 8), len(payload) & 0xFF]) + payload[:6]
    out.append((now, bus, address, head.ljust(8, b"\x00")))
    now += step
    if peer is not None:
        out.append((now, bus, peer, b"\x30\x00\x00\x00\x00\x00\x00\x00"))
        now += step
    rest, sn, index = payload[6:], 1, 1
    while rest:
        chunk, rest = rest[:7], rest[7:]
        if gap_before == index:
            now += MAX_GAP_NS + step
        wire = (sn + 1) if break_at == index else sn
        out.append((now, bus, address,
                    bytes([0x20 | (wire & 0x0F)]) + chunk.ljust(7, b"\x00")))
        sn += 1
        index += 1
        now += step
    return out


def repeated(payload, *, times=4, start_ns=0, **kwargs):
    """Several transfers back to back, so the address clears MIN_MESSAGES."""
    out, now = [], start_ns
    for _ in range(times):
        frames = exchange(payload, start_ns=now, **kwargs)
        out.extend(frames)
        now = frames[-1][0] + STEP
    return out


VIN = b"WVWZZZ1KZAW000001"  # 17 bytes: a FirstFrame plus two ConsecutiveFrames


class TestReassembly:
    def test_a_segmented_transfer_comes_back_whole(self):
        found = find_isotp(repeated(VIN))
        hypothesis = found[(0, 0x7E0)]
        assert hypothesis.messages == 4
        assert all(t.data == VIN for t in hypothesis.transfers)

    def test_the_payload_is_trimmed_to_ff_dl_not_to_the_frame(self):
        """The last ConsecutiveFrame is padded; the promise is FF_DL."""
        found = find_isotp(repeated(VIN))
        assert {t.length for t in found[(0, 0x7E0)].transfers} == {len(VIN)}

    def test_a_long_transfer_wraps_the_sequence_number(self):
        payload = bytes(range(200))  # 6 + 28 CFs, so SN runs past 15 twice
        found = find_isotp(repeated(payload, times=2))
        hypothesis = found[(0, 0x7E0)]
        assert hypothesis.messages == 2
        assert all(t.data == payload for t in hypothesis.transfers)
        assert hypothesis.transfers[0].consecutive_frames == 28

    def test_the_match_rate_is_the_share_of_first_frames_that_completed(self):
        frames = repeated(VIN, times=3) + exchange(VIN, start_ns=10**12, break_at=1)
        hypothesis = find_isotp(frames)[(0, 0x7E0)]
        assert hypothesis.started == 4 and hypothesis.messages == 3
        assert hypothesis.match_rate == pytest.approx(0.75)


class TestAbortRules:
    def test_a_wrong_sequence_number_aborts_the_transfer(self):
        assert find_isotp(repeated(VIN, times=4, break_at=1)) == {}

    def test_a_gap_longer_than_the_timeout_aborts(self):
        assert find_isotp(repeated(VIN, times=4, gap_before=2)) == {}

    def test_a_first_frame_promising_no_more_than_it_carries_is_not_one(self):
        """FF_DL <= 6 would have been sent as a SingleFrame."""
        frames = [(0, 0, 0x7E0, b"\x10\x05" + b"\x01" * 6)] * 8
        assert find_isotp(frames) == {}

    def test_consecutive_frames_without_a_first_frame_are_ignored(self):
        frames = [(i * STEP, 0, 0x7E0, bytes([0x20 | (i % 16)]) + b"\x00" * 7)
                  for i in range(64)]
        assert find_isotp(frames) == {}

    def test_a_second_first_frame_abandons_the_one_in_flight(self):
        partial = exchange(VIN)[:2]        # FirstFrame and one ConsecutiveFrame
        found = find_isotp(partial + repeated(VIN, start_ns=10**12))
        assert found[(0, 0x7E0)].started == 5
        assert found[(0, 0x7E0)].messages == 4

    def test_a_frame_too_short_to_hold_a_header_is_skipped(self):
        assert find_isotp([(0, 0, 0x7E0, b"\x10")]) == {}


class TestFlowControl:
    def test_a_peer_answering_is_recorded(self):
        found = find_isotp(repeated(VIN, peer=0x7E8))
        hypothesis = found[(0, 0x7E0)]
        assert hypothesis.flow_control_rate == 1.0
        assert hypothesis.peers == (0x7E8,)

    def test_it_is_corroboration_and_never_a_criterion(self):
        """The same transfers are found whether or not anyone answered."""
        with_fc = find_isotp(repeated(VIN, peer=0x7E8))[(0, 0x7E0)]
        without = find_isotp(repeated(VIN))[(0, 0x7E0)]
        assert with_fc.messages == without.messages
        assert without.flow_control_rate == 0.0 and without.peers == ()

    def test_a_reserved_flow_status_is_not_a_flow_control(self):
        frames = []
        for chunk in repeated(VIN, peer=0x7E8):
            if chunk[2] == 0x7E8:
                chunk = (chunk[0], chunk[1], chunk[2], b"\x39" + chunk[3][1:])
            frames.append(chunk)
        assert find_isotp(frames)[(0, 0x7E0)].flow_control_rate == 0.0

    def test_the_sender_answering_itself_does_not_count(self):
        found = find_isotp(repeated(VIN, peer=0x7E0))
        assert found[(0, 0x7E0)].flow_control_rate == 0.0

    def test_a_peer_on_another_bus_does_not_count(self):
        frames = [(t, b, a, d) if a != 0x7E8 else (t, 1, a, d)
                  for t, b, a, d in repeated(VIN, peer=0x7E8)]
        assert find_isotp(frames)[(0, 0x7E0)].flow_control_rate == 0.0


class TestEvidenceBar:
    def test_a_single_transfer_is_not_enough(self):
        assert find_isotp(exchange(VIN)) == {}

    def test_the_bar_is_the_stated_one(self):
        assert find_isotp(repeated(VIN, times=MIN_MESSAGES)) != {}
        assert find_isotp(repeated(VIN, times=MIN_MESSAGES - 1)) == {}


class TestTransportBits:
    def test_a_verified_transfer_explains_the_pci_byte(self):
        hypothesis = find_isotp(repeated(VIN))[(0, 0x7E0)]
        assert transport_bits(hypothesis) == set(range(8))

    def test_an_unverified_one_explains_nothing(self):
        assert transport_bits(IsotpHypothesis(bus=0, address=0x7E0)) == set()


class TestOrderedFrames:
    def test_echoes_are_excluded(self):
        from canlens.decode import CanFrame, from_frames

        frames = [CanFrame(i * STEP, 0, 0x7E0, b"\x01\x02", False) for i in range(4)]
        frames += [CanFrame(i * STEP, 2, 0x7E0, b"\x01\x02", True) for i in range(4)]
        rows = list(ordered_frames(from_frames(frames)))
        assert len(rows) == 4 and {r[1] for r in rows} == {0}

    def test_rows_come_back_in_time_order(self):
        from canlens.decode import CanFrame, from_frames

        frames = [CanFrame(t, 0, 0x7E0, b"\x01\x02", False)
                  for t in (30 * STEP, 10 * STEP, 20 * STEP)]
        stamps = [row[0] for row in ordered_frames(from_frames(frames))]
        assert stamps == sorted(stamps)


class TestDetectionIntegration:
    """The transport has to reach the inference, and take its bits with it."""

    @staticmethod
    def frames():
        from canlens.decode import CanFrame

        rows = repeated(VIN, times=40, address=0x7E0, peer=0x7E8)
        return [CanFrame(t, b, a, d, False) for t, b, a, d in rows]

    def test_inference_reports_the_conversation(self):
        from canlens.decode import from_frames
        from canlens.infer import infer_frameset

        found = {m.key: m for m in infer_frameset(from_frames(self.frames()))}
        hypothesis = found[(0, 0x7E0)].isotp
        assert hypothesis is not None and hypothesis.messages == 40
        assert found[(0, 0x7E0)].found_anything

    def test_no_counter_is_claimed_inside_the_pci_byte(self):
        """The SequenceNumber is a counter, and it is not a vehicle signal."""
        from canlens.decode import from_frames
        from canlens.infer import infer_frameset

        found = {m.key: m for m in infer_frameset(from_frames(self.frames()))}
        message = found[(0, 0x7E0)]
        assert message.isotp is not None
        pci = set(range(8))
        assert not any(pci.issuperset(range(c.start_bit, c.end_bit))
                       for c in message.counters)

    def test_both_inference_paths_agree(self):
        from canlens.decode import from_frames
        from canlens.infer import infer_frames, infer_frameset

        objects = self.frames()
        a = infer_frameset(from_frames(objects))
        b = infer_frames(iter(objects))
        assert [(m.key, m.isotp.messages if m.isotp else None) for m in a] == \
               [(m.key, m.isotp.messages if m.isotp else None) for m in b]

    def test_an_ordinary_trace_reports_no_transport(self):
        from canlens.decode import CanFrame, from_frames
        from canlens.infer import infer_frameset

        frames = [CanFrame(i * STEP, 0, 0x120, bytes([i % 256, 0xAA]), False)
                  for i in range(400)]
        found = infer_frameset(from_frames(frames))
        assert all(m.isotp is None for m in found)


class TestMatchRateFloor:
    """An endpoint completes what it starts; a coincidence does not."""

    def test_an_address_that_rarely_completes_is_not_an_endpoint(self):
        """VW's 0x101 opened ~200 accidental FirstFrames and completed two."""
        noise = [(i * STEP, 0, 0x101, bytes([0x10, 0x20, *range(6)]))
                 for i in range(200)]
        frames = sorted(noise + repeated(VIN, times=2, address=0x101,
                                         start_ns=10**12),
                        key=lambda row: row[0])
        assert find_isotp(frames) == {}

    def test_the_same_transfers_are_kept_when_nothing_else_opens_frames(self):
        assert find_isotp(repeated(VIN, times=2)) != {}

    def test_the_floor_is_the_stated_one(self):
        from canlens.infer.isotp import MIN_MATCH_RATE

        hypothesis = IsotpHypothesis(bus=0, address=0x7E0)
        hypothesis.started = 100
        hypothesis.transfers = find_isotp(repeated(VIN, times=4))[(0, 0x7E0)].transfers
        assert hypothesis.match_rate < MIN_MATCH_RATE
        assert not hypothesis.verified
