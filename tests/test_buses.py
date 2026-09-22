# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Identifying a bus by its traffic rather than by the port it was logged on.

Synthetic platforms with a known wiring, so the thing under test is whether
the labels come out right -- not whether the corpus happens to contain a car
that was rewired.
"""
from __future__ import annotations

import numpy as np
import pytest

from canlens.analyze.bits import BitKind, BitOrder, BitProfile
from canlens.corroborate import corroborate
from canlens.corroborate.buses import (
    MIN_OVERLAP,
    BusSignature,
    assign,
    identify_from,
    overlap,
    signatures,
    signatures_from,
)
from canlens.decode import CanFrame, from_frames
from canlens.infer import MessageInference

POWERTRAIN = frozenset(range(0x100, 0x140))
CHASSIS = frozenset(range(0x200, 0x230))
CAMERA = frozenset(range(0x300, 0x310))


def sig(bus, addresses, frames=1000):
    return BusSignature(bus, frozenset(addresses), frames, len(addresses))


def inference(bus, address):
    n = 64
    profile = BitProfile(
        width=8, frames=100, ones=np.zeros(n), entropy=np.zeros(n),
        rates=np.zeros(n), kinds=[BitKind.CONSTANT] * n, order=BitOrder.INTEL,
    )
    return MessageInference(bus=bus, address=address, width=8, frames=100, bits=profile)


class TestOverlap:
    def test_identical_sets_are_one(self):
        assert overlap(POWERTRAIN, POWERTRAIN) == 1.0

    def test_disjoint_sets_are_zero(self):
        assert overlap(POWERTRAIN, CHASSIS) == 0.0

    def test_two_empty_sets_are_zero_rather_than_undefined(self):
        assert overlap([], []) == 0.0

    def test_partial_overlap_is_the_jaccard_ratio(self):
        assert overlap({1, 2, 3, 4}, {3, 4, 5, 6}) == pytest.approx(2 / 6)


class TestSignatures:
    def test_one_signature_per_bus_of_a_decoded_segment(self):
        frames = from_frames(
            [CanFrame(i * 1000, 0, 0x100, b"\x00" * 8, False) for i in range(40)]
            + [CanFrame(i * 1000, 2, 0x300, b"\x00" * 8, False) for i in range(40)]
        )
        found = signatures(frames)
        assert [s.bus for s in found] == [0, 2]
        assert found[0].addresses == frozenset({0x100})
        assert found[0].frames == 40

    def test_rare_messages_are_left_out_of_a_signature(self):
        """A one-off message makes two recordings of a bus look less alike.

        Including everything split Rivian into seven buses where it has six.
        The cutoff matches the one `infer` uses, so identifying buses from
        frames and from inference results cannot disagree.
        """
        frames = from_frames(
            [CanFrame(i * 1000, 0, 0x100, b"\x00" * 8, False) for i in range(40)]
            + [CanFrame(i * 1000, 0, 0x999, b"\x00" * 8, False) for i in range(3)]
        )
        assert signatures(frames)[0].addresses == frozenset({0x100})
        assert signatures(frames, min_frames=1)[0].addresses == frozenset({0x100, 0x999})

    def test_signatures_from_inferences_need_no_decode(self):
        found = signatures_from(
            [inference(0, 0x100), inference(0, 0x101), inference(2, 0x300)]
        )
        assert [s.bus for s in found] == [0, 2]
        assert found[0].addresses == frozenset({0x100, 0x101})
        assert found[0].frames == 200


class TestAssign:
    def test_pairs_each_bus_with_its_own_traffic(self):
        left = [POWERTRAIN, CHASSIS, CAMERA]
        right = [CAMERA, POWERTRAIN, CHASSIS]
        assert assign(left, right) == {0: 1, 1: 2, 2: 0}

    def test_no_two_buses_take_the_same_partner(self):
        left = [POWERTRAIN, POWERTRAIN | {0x999}]
        right = [POWERTRAIN, CHASSIS]
        result = assign(left, right)
        assert len(set(result.values())) == len(result)

    def test_the_best_pairing_wins_not_the_first_greedy_one(self):
        """Greedy takes the single best pair and then has no good options."""
        a, b = frozenset({1, 2, 3, 4}), frozenset({3, 4, 5, 6})
        left = [a, b]
        right = [frozenset({1, 2, 3, 4, 5, 6}), b]
        assert assign(left, right) == {0: 0, 1: 1}

    def test_a_bus_with_no_counterpart_is_left_out(self):
        assert assign([POWERTRAIN, CAMERA], [POWERTRAIN]) == {0: 0}

    def test_unrelated_buses_are_not_forced_together(self):
        assert assign([CAMERA], [POWERTRAIN]) == {}

    def test_an_empty_side_pairs_nothing(self):
        assert assign([], [POWERTRAIN]) == {}
        assert assign([POWERTRAIN], []) == {}

    def test_the_floor_is_respected(self):
        near = frozenset(list(POWERTRAIN)[:4]) | frozenset(range(0x900, 0x960))
        assert 0 < overlap(POWERTRAIN, near) < MIN_OVERLAP
        assert assign([POWERTRAIN], [near]) == {}


class TestIdentifyFrom:
    @staticmethod
    def platform(wirings):
        """Each wiring is {logged bus: identifier set} for one segment."""
        return [
            (f"dev{i}/route/1", [sig(bus, ids) for bus, ids in sorted(wiring.items())])
            for i, wiring in enumerate(wirings)
        ]

    def test_a_consistent_platform_is_left_alone(self):
        wiring = {0: POWERTRAIN, 1: CHASSIS, 2: CAMERA}
        result = identify_from("X", self.platform([wiring] * 5))
        assert len(result.identities) == 3
        assert result.relabelled == 0
        assert all(i.stable for i in result.identities)

    def test_a_swap_is_undone(self):
        """Two cars wired the other way round still pool together."""
        straight = {0: POWERTRAIN, 1: CHASSIS, 2: CAMERA}
        swapped = {0: CHASSIS, 1: POWERTRAIN, 2: CAMERA}
        result = identify_from("X", self.platform([straight] * 3 + [swapped] * 2))
        assert len(result.identities) == 3
        assert result.relabelled == 4  # two buses in each of the two odd segments
        labels = {i.label: i.addresses for i in result.identities}
        assert labels[0] == set(POWERTRAIN) and labels[1] == set(CHASSIS)

    def test_the_majority_keeps_the_number_it_was_logged_as(self):
        """Otherwise one odd segment renames every other one."""
        usual = {0: POWERTRAIN, 1: CHASSIS, 2: CAMERA}
        odd = {0: POWERTRAIN, 1: frozenset(range(0x800, 0x830)), 2: CAMERA}
        result = identify_from("X", self.platform([odd] + [usual] * 9))
        assert result.relabelled == 1
        chassis = next(i for i in result.identities if i.addresses == set(CHASSIS))
        assert chassis.label == 1

    def test_one_number_naming_two_buses_is_contested(self):
        usual = {0: POWERTRAIN, 1: CHASSIS, 2: CAMERA}
        odd = {0: POWERTRAIN, 1: frozenset(range(0x800, 0x830)), 2: CAMERA}
        result = identify_from("X", self.platform([usual] * 5 + [odd] * 3))
        assert sorted(result.contested) == [1]
        assert len(result.contested[1]) == 2

    def test_an_extra_bus_becomes_its_own_identity(self):
        three = {0: POWERTRAIN, 1: CHASSIS, 2: CAMERA}
        six = dict(three) | {4: frozenset(range(0x400, 0x410))}
        result = identify_from("X", self.platform([three] * 3 + [six] * 2))
        assert len(result.identities) == 4
        extra = next(i for i in result.identities if i.label == 4)
        assert extra.segments == 2

    def test_canonical_falls_back_to_the_logged_number(self):
        result = identify_from("X", self.platform([{0: POWERTRAIN}]))
        assert result.canonical("nowhere/at/all", 7) == 7

    def test_a_platform_with_no_segments_maps_nothing(self):
        result = identify_from("X", [])
        assert result.identities == [] and result.relabelled == 0
        assert "0 buses" in str(result)


class TestCorroborationUsesCanonicalBuses:
    """The point of the exercise: pooling must not mix two buses."""

    @staticmethod
    def observations(swap_from):
        """Segments where, after `swap_from`, buses 0 and 1 are exchanged."""
        out = []
        for i in range(6):
            flip = i >= swap_from
            found = []
            for address in (0x100, 0x101):
                found.append(inference(1 if flip else 0, address))
            for address in (0x200, 0x201):
                found.append(inference(0 if flip else 1, address))
            out.append((f"dev{i}", found))
        return out

    def test_logged_numbers_split_one_bus_into_two(self):
        """Without reconciliation each message is seen in only half the drives."""
        result = corroborate(self.observations(3), platform="X")
        assert len(result) == 8  # four messages, each under two numbers
        assert max(m.of_segments for m in result.messages.values()) == 3

    def test_reconciled_numbers_pool_every_segment(self):
        from canlens.corroborate.buses import identify_from, signatures_from

        raw = self.observations(3)
        bus_map = identify_from(
            "X", ((device, signatures_from(found)) for device, found in raw)
        )
        from dataclasses import replace

        fixed = [
            (device, [replace(m, bus=bus_map.canonical(device, m.bus)) for m in found])
            for device, found in raw
        ]
        result = corroborate(fixed, platform="X")
        assert len(result) == 4
        assert all(m.of_segments == 6 for m in result.messages.values())
        assert all(m.of_devices == 6 for m in result.messages.values())
