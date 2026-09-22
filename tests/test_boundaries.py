# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Boundaries several platforms agree on, tallied from known per-platform input.

Synthetic platforms with a decided layout, so what is under test is the tally
and the tiering -- not whether `find_signals` happened to be right on a car.
"""
from __future__ import annotations

from canlens.corroborate.boundaries import (
    MIN_PLATFORMS,
    BoundarySupport,
    agree,
    boundaries_across,
)
from canlens.corroborate.consensus import Tier
from canlens.infer.signals import SignalHypothesis


def sig(start, length=8):
    return SignalHypothesis(
        start_bit=start, length=length, frames=500, rate=0.2,
        minimum=0, maximum=255, bounded=True,
    )


def platforms(*layouts, address=0x120, bus=0):
    """One entry per platform, each giving that platform's start bits."""
    return {
        f"CAR_{i}": {(bus, address): [sig(s) for s in starts]}
        for i, starts in enumerate(layouts)
    }


class TestSupport:
    def test_support_is_the_share_of_platforms_proposing_it(self):
        s = BoundarySupport(start_bit=0, platforms=3, of_platforms=4, widths=(8, 8, 8))
        assert s.support == 0.75

    def test_a_message_no_platform_carries_has_no_support(self):
        assert BoundarySupport(0, 0, 0, ()).support == 0.0

    def test_unanimous_across_enough_platforms_is_established(self):
        s = BoundarySupport(0, 4, 4, (8, 8, 8, 8))
        assert s.tier is Tier.ESTABLISHED

    def test_unanimous_across_too_few_platforms_is_only_consistent(self):
        """Two agreeing may be two recordings of one design decision."""
        s = BoundarySupport(0, 2, 2, (8, 8))
        assert s.tier is Tier.CONSISTENT

    def test_half_the_group_is_partial(self):
        assert BoundarySupport(0, 2, 4, (8, 8)).tier is Tier.PARTIAL

    def test_a_lone_proposal_among_many_is_weak(self):
        assert BoundarySupport(0, 1, 5, (8,)).tier is Tier.WEAK

    def test_an_agreed_width_is_reported_and_a_disputed_one_is_not(self):
        assert BoundarySupport(0, 3, 3, (8, 8, 8)).agreed_width == 8
        assert BoundarySupport(0, 3, 3, (8, 8, 6)).agreed_width is None


class TestAgree:
    def test_a_boundary_every_platform_proposes_is_established(self):
        found = agree(platforms([0, 8], [0, 8], [0, 8]))
        supports = {s.start_bit: s for s in found[(0, 0x120)].supports}
        assert supports[0].tier is Tier.ESTABLISHED
        assert supports[0].platforms == 3

    def test_a_boundary_only_one_platform_proposes_is_weak(self):
        found = agree(platforms([0, 8], [0, 8], [0, 8], [0, 8], [0, 8, 32]))
        odd = next(s for s in found[(0, 0x120)].supports if s.start_bit == 32)
        assert odd.tier is Tier.WEAK and odd.platforms == 1

    def test_a_message_too_few_platforms_carry_is_left_out(self):
        assert agree(platforms([0], [0])) == {}
        assert agree(platforms([0], [0]), min_platforms=2) != {}

    def test_only_platforms_carrying_the_message_are_counted(self):
        """A platform without the message must not dilute the support."""
        group = platforms([0], [0], [0])
        group["CAR_3"] = {(0, 0x999): [sig(0)]}
        found = agree(group)
        assert found[(0, 0x120)].of_platforms == 3
        assert found[(0, 0x120)].supports[0].support == 1.0

    def test_the_same_address_on_another_bus_is_another_message(self):
        group = platforms([0], [0], [0])
        group["CAR_0"][(1, 0x120)] = [sig(16)]
        found = agree(group, min_platforms=1)
        assert (0, 0x120) in found and (1, 0x120) in found
        assert found[(1, 0x120)].of_platforms == 1

    def test_supports_come_back_strongest_first(self):
        found = agree(platforms([0, 8], [0, 8], [0], [0]))
        supports = found[(0, 0x120)].supports
        assert [s.start_bit for s in supports] == [0, 8]
        assert supports[0].support > supports[1].support

    def test_widths_are_collected_per_proposing_platform(self):
        group = {
            "A": {(0, 0x120): [sig(0, 8)]},
            "B": {(0, 0x120): [sig(0, 8)]},
            "C": {(0, 0x120): [sig(0, 6)]},
        }
        support = agree(group)[(0, 0x120)].supports[0]
        assert sorted(support.widths) == [6, 8, 8]
        assert support.agreed_width is None

    def test_an_empty_group_agrees_on_nothing(self):
        assert agree({}) == {}


class TestAtLeast:
    def test_filtering_keeps_the_strong_and_drops_the_weak(self):
        found = agree(platforms([0, 8], [0, 8], [0, 8], [0, 8], [0, 8, 32]))
        message = found[(0, 0x120)]
        kept = {s.start_bit for s in message.at_least(Tier.PARTIAL)}
        assert kept == {0, 8} and 32 not in kept

    def test_the_weak_tier_keeps_everything(self):
        found = agree(platforms([0], [0], [0, 32]))
        assert len(found[(0, 0x120)].at_least(Tier.WEAK)) == 2


class TestBoundariesAcross:
    def test_a_platform_with_no_signals_is_not_counted(self, monkeypatch):
        from canlens.corroborate import pooled

        table = {"A": {(0, 1): [sig(0)]}, "B": {(0, 1): [sig(0)]},
                 "C": {(0, 1): [sig(0)]}, "D": {}}
        monkeypatch.setattr(
            pooled, "signals_across", lambda p, **kw: table.get(p, {})
        )
        found = boundaries_across(["A", "B", "C", "D"], root="/nowhere")
        assert found[(0, 1)].of_platforms == 3


def test_the_platform_floor_matches_the_device_floor():
    """Both answer 'how many independent observers before this generalises'."""
    from canlens.corroborate.consensus import MIN_DEVICES

    assert MIN_PLATFORMS >= MIN_DEVICES


class TestWidest:
    """Widths disagree because each car exercised a different range."""

    def test_the_widest_claim_is_the_strongest_lower_bound(self):
        assert BoundarySupport(0, 3, 3, (6, 9, 7)).widest == 9

    def test_a_support_with_no_widths_has_no_bound(self):
        assert BoundarySupport(0, 0, 3, ()).widest == 0

    def test_a_disputed_width_reads_as_a_lower_bound_not_a_shrug(self):
        assert "at least 9 bits" in str(BoundarySupport(0, 3, 3, (6, 9, 7)))

    def test_an_agreed_width_is_stated_flatly(self):
        assert "8 bits," in str(BoundarySupport(0, 3, 3, (8, 8, 8)))
        assert "at least" not in str(BoundarySupport(0, 3, 3, (8, 8, 8)))
