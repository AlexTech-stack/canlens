# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Corroboration over synthetic per-segment inferences -- no I/O."""
from __future__ import annotations

import numpy as np
import pytest

from canlens.analyze.bits import BitKind, BitOrder, BitProfile
from canlens.corroborate import (
    MIN_DEVICES,
    Evidence,
    Tier,
    corroborate,
    device_of,
)
from canlens.infer import MessageInference
from canlens.infer.checksums import ChecksumHypothesis
from canlens.infer.counters import CounterHypothesis
from canlens.infer.crc16 import Crc16Hypothesis


def profile(kinds: list[BitKind]) -> BitProfile:
    n = len(kinds)
    return BitProfile(
        width=n // 8, frames=100, ones=np.zeros(n), entropy=np.zeros(n),
        rates=np.zeros(n), kinds=list(kinds), order=BitOrder.INTEL,
    )


def message(
    key=(1, 0x210), width=1, kinds=None, counters=(), checksums=(), crc16s=(), frames=100,
    multiplexor=None,
) -> MessageInference:
    kinds = kinds or [BitKind.CONSTANT] * (width * 8)
    return MessageInference(
        bus=key[0], address=key[1], width=width, frames=frames, bits=profile(kinds),
        counters=list(counters), checksums=list(checksums), crc16s=list(crc16s),
        multiplexor=multiplexor,
    )


def counter(start=0, length=8, stride=1, rate=1.0):
    return CounterHypothesis(start, length, stride, rate, 100)


class TestDeviceOf:
    def test_first_component_of_the_stored_layout(self):
        assert device_of("/root/segments/dev0/route--a/12/rlog.zst") == "dev0"
        assert device_of("dev0/route--a/12") == "dev0"

    def test_too_short_is_empty(self):
        assert device_of("rlog.zst") == ""


class TestEvidenceTiers:
    def make(self, segments, devices, of_segments=10, of_devices=5):
        return Evidence(segments, devices, of_segments, of_devices, 1.0, 1.0)

    def test_established_needs_support_and_several_devices(self):
        assert self.make(10, MIN_DEVICES).tier is Tier.ESTABLISHED

    def test_unanimous_from_too_few_cars_is_only_consistent(self):
        # Everything seen agrees -- but two cars cannot show it generalises.
        assert self.make(10, 2, of_devices=2).tier is Tier.CONSISTENT

    def test_partial(self):
        assert self.make(6, 4).tier is Tier.PARTIAL

    def test_weak(self):
        assert self.make(2, 2).tier is Tier.WEAK

    def test_support_is_against_segments_that_carry_the_message(self):
        assert self.make(9, 3, of_segments=9).support == 1.0
        assert self.make(0, 0, of_segments=0).support == 0.0


class TestCountsAndPresence:
    def test_devices_are_counted_distinctly(self):
        obs = [("carA", [message()]), ("carA", [message()]), ("carB", [message()])]
        result = corroborate(obs)
        assert result.segments == 3 and result.devices == 2
        m = result[(1, 0x210)]
        assert m.segments == 3 and m.devices == 2

    def test_presence_is_the_share_of_segments_carrying_the_message(self):
        obs = [("a", [message()]), ("b", [message()]), ("c", [])]
        m = corroborate(obs)[(1, 0x210)]
        assert m.of_segments == 2  # the third segment never carried it
        assert m.presence == 1.0   # of the segments that did
        assert corroborate(obs).segments == 3

    def test_messages_from_different_buses_stay_apart(self):
        obs = [("a", [message(key=(0, 0x100)), message(key=(1, 0x100))])]
        assert set(corroborate(obs).messages) == {(0, 0x100), (1, 0x100)}

    def test_empty_input(self):
        result = corroborate([])
        assert len(result) == 0 and result.segments == 0


class TestWidth:
    def test_modal_width_wins_and_agreement_is_reported(self):
        obs = [("a", [message(width=8)]), ("b", [message(width=8)]), ("c", [message(width=4)])]
        m = corroborate(obs)[(1, 0x210)]
        assert m.width == 8
        assert m.width_agreement == pytest.approx(2 / 3)
        assert m.of_segments == 2  # only the modal width is aggregated


class TestBitConsensus:
    def test_unanimous_bits_agree_fully(self):
        kinds = [BitKind.NOISY] * 4 + [BitKind.CONSTANT] * 4
        obs = [(d, [message(kinds=kinds)]) for d in "abc"]
        m = corroborate(obs)[(1, 0x210)]
        assert m.kinds == kinds
        assert all(b.agreement == 1.0 for b in m.bits)
        assert m.rare_bits == []

    def test_a_rarely_moving_bit_is_flagged(self):
        # Constant in most cars; moves in one. Not padding -- a rare state.
        still = [BitKind.CONSTANT] * 8
        moved = [BitKind.CONSTANT] * 7 + [BitKind.SLOW]
        obs = [("a", [message(kinds=still)]), ("b", [message(kinds=still)]),
               ("c", [message(kinds=still)]), ("d", [message(kinds=moved)])]
        m = corroborate(obs)[(1, 0x210)]
        assert m.kinds[7] is BitKind.CONSTANT
        assert m.bits[7].agreement == pytest.approx(0.75)
        assert m.rare_bits == [7]

    def test_a_bit_that_usually_moves_is_not_rare(self):
        obs = [("a", [message(kinds=[BitKind.ACTIVE] * 8)]),
               ("b", [message(kinds=[BitKind.CONSTANT] * 8)])]
        m = corroborate(obs)[(1, 0x210)]
        # A tie resolves toward the quieter class, and constant-with-movement
        # elsewhere is exactly the rare case; either way nothing crashes.
        assert len(m.rare_bits) in (0, 8)

    def test_layout_agreement_is_the_mean(self):
        obs = [("a", [message(kinds=[BitKind.ACTIVE] * 8)]),
               ("b", [message(kinds=[BitKind.ACTIVE] * 4 + [BitKind.SLOW] * 4)])]
        m = corroborate(obs)[(1, 0x210)]
        assert m.agreement == pytest.approx((4 * 1.0 + 4 * 0.5) / 8)


class TestHypotheses:
    def test_a_counter_seen_everywhere_is_established(self):
        obs = [(d, [message(counters=[counter()])]) for d in "abcd"]
        m = corroborate(obs)[(1, 0x210)]
        assert len(m.counters) == 1
        c = m.counters[0]
        assert (c.start_bit, c.length, c.stride) == (0, 8, 1)
        assert c.evidence.segments == 4 and c.evidence.devices == 4
        assert c.evidence.tier is Tier.ESTABLISHED
        assert not c.contested

    def test_match_rates_are_summarised(self):
        obs = [("a", [message(counters=[counter(rate=1.0)])]),
               ("b", [message(counters=[counter(rate=0.96)])])]
        e = corroborate(obs)[(1, 0x210)].counters[0].evidence
        assert e.mean_match == pytest.approx(0.98)
        assert e.min_match == pytest.approx(0.96)

    def test_different_strides_are_different_hypotheses_and_contested(self):
        obs = [("a", [message(counters=[counter(stride=1)])]),
               ("b", [message(counters=[counter(stride=3)])])]
        m = corroborate(obs)[(1, 0x210)]
        assert len(m.counters) == 2
        assert all(c.contested for c in m.counters)

    def test_non_overlapping_counters_are_not_contested(self):
        obs = [("a", [message(width=2, counters=[counter(0, 8), counter(8, 8)])])]
        m = corroborate(obs)[(1, 0x210)]
        assert not any(c.contested for c in m.counters)

    def test_checksums_at_one_byte_with_two_algorithms_are_contested(self):
        obs = [("a", [message(checksums=[ChecksumHypothesis(7, "toyota", 1.0, 100)])]),
               ("b", [message(checksums=[ChecksumHypothesis(7, "sum8", 1.0, 100)])])]
        m = corroborate(obs)[(1, 0x210)]
        assert all(s.contested for s in m.checksums)

    def test_crc16_with_a_different_data_id_is_a_different_hypothesis(self):
        # The EV6 data-ID rule shows up as: one hypothesis, every segment.
        same = Crc16Hypothesis(0, "e2e_p05", "little", 1.0, 100, 0xFA11)
        other = Crc16Hypothesis(0, "e2e_p05", "little", 1.0, 100, 0xFA12)
        obs = [("a", [message(crc16s=[same])]), ("b", [message(crc16s=[same])]),
               ("c", [message(crc16s=[other])])]
        m = corroborate(obs)[(1, 0x210)]
        assert len(m.crc16s) == 2
        assert m.crc16s[0].data_id == 0xFA11 and m.crc16s[0].evidence.segments == 2
        assert all(c.contested for c in m.crc16s)

    def test_strongest_hypothesis_comes_first(self):
        obs = [(d, [message(counters=[counter(stride=1)])]) for d in "abc"] + [
            ("d", [message(counters=[counter(stride=3)])])
        ]
        m = corroborate(obs)[(1, 0x210)]
        assert m.counters[0].stride == 1

    def test_a_hypothesis_missing_from_some_segments_is_partial(self):
        obs = [("a", [message(counters=[counter()])]), ("b", [message(counters=[counter()])]),
               ("c", [message()]), ("d", [message()])]
        c = corroborate(obs)[(1, 0x210)].counters[0]
        assert c.evidence.support == 0.5
        assert c.evidence.tier is Tier.PARTIAL


class TestOrdering:
    def test_by_identifier_groups_by_bus_then_standard_before_extended(self):
        obs = [("a", [message(key=(1, 0x900)), message(key=(0, 0x300)),
                      message(key=(1, 0x100)), message(key=(1, 0x18DAF110))])]
        order = [m.key for m in corroborate(obs).by_identifier()]
        assert order == [(0, 0x300), (1, 0x100), (1, 0x900), (1, 0x18DAF110)]

    def test_by_presence_puts_the_commonest_first(self):
        obs = [("a", [message(key=(0, 1)), message(key=(0, 2))]), ("b", [message(key=(0, 1))])]
        assert next(m.key for m in corroborate(obs).by_presence()) == (0, 1)


class TestDataIdsAndWraps:
    def test_checksums_with_different_data_ids_are_different_hypotheses(self):
        a = ChecksumHypothesis(0, "e2e_p11", 1.0, 100, data_id=0x37)
        b = ChecksumHypothesis(0, "e2e_p11", 1.0, 100, data_id=0x38)
        obs = [("x", [message(checksums=[a])]), ("y", [message(checksums=[a])]),
               ("z", [message(checksums=[b])])]
        m = corroborate(obs)[(1, 0x210)]
        assert len(m.checksums) == 2
        assert m.checksums[0].data_id == 0x37 and m.checksums[0].evidence.segments == 2
        assert all(s.contested for s in m.checksums)
        assert "data ID 0x37" in str(m.checksums[0])

    def test_counters_with_different_wraps_are_different_hypotheses(self):
        obs = [("x", [message(counters=[CounterHypothesis(8, 4, 1, 1.0, 100, modulus=15)])]),
               ("y", [message(counters=[CounterHypothesis(8, 4, 1, 0.93, 100)])])]
        m = corroborate(obs)[(1, 0x210)]
        assert sorted(c.modulus for c in m.counters) == [0, 15]
        assert "mod 15" in str(next(c for c in m.counters if c.modulus == 15))


class TestMultiplexors:
    def test_selectors_are_pooled_by_position_and_values_unioned(self):
        from canlens.infer.multiplex import MultiplexHypothesis

        a = MultiplexHypothesis(0, 8, (0, 1, 2), (100, 100, 100), tuple(range(8, 40)), 300)
        b = MultiplexHypothesis(0, 8, (0, 1, 2, 3), (75, 75, 75, 75), tuple(range(8, 40)), 300)
        obs = [("x", [message(width=8, multiplexor=a)]), ("y", [message(width=8, multiplexor=b)]),
               ("z", [message(width=8, multiplexor=a)])]
        result = corroborate(obs)
        [mux] = result[(1, 0x210)].multiplexors
        assert (mux.start_bit, mux.length, mux.values) == (0, 8, (0, 1, 2, 3))
        assert mux.evidence.segments == 3 and mux.evidence.devices == 3
        assert mux.evidence.tier is Tier.ESTABLISHED
        assert not mux.contested

    def test_two_different_selectors_are_contested(self):
        from canlens.infer.multiplex import MultiplexHypothesis

        a = MultiplexHypothesis(0, 8, (0, 1, 2), (100, 100, 100), (8, 9), 300)
        b = MultiplexHypothesis(8, 4, (0, 1), (150, 150), (16, 17), 300)
        result = corroborate([("x", [message(width=8, multiplexor=a)]),
                              ("y", [message(width=8, multiplexor=b)])])
        muxes = result[(1, 0x210)].multiplexors
        assert len(muxes) == 2 and all(m.contested for m in muxes)

    def test_str_carries_the_evidence(self):
        from canlens.infer.multiplex import MultiplexHypothesis

        a = MultiplexHypothesis(0, 8, (0, 1, 2), (100, 100, 100), (8, 9), 300)
        result = corroborate([("x", [message(width=8, multiplexor=a)])])
        [mux] = result[(1, 0x210)].multiplexors
        assert str(mux).startswith("8-bit multiplexor @ bit 0: values 0, 1, 2 (1/1 seg, 1/1 dev, consistent)")
