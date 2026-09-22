# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Signal boundaries that several platforms agree on.

Every other corroboration here asks whether a hypothesis holds across the
*segments* of one platform. This asks something stronger: whether it holds
across *different vehicles that share a message*. Volkswagen's MQB platform
puts address 0x120 on a Golf, a Tiguan, an Audi Q3 and a Skoda Superb alike,
and the pooled Profile 22 work already showed those really are the same
message -- 25 addresses solved on more than one platform agreed on all sixteen
Data ID bytes, with 0x120 agreeing across seventeen.

If the message is the same, its layout is the same, and a boundary that only
one platform proposes is probably an artefact of that platform's trace.

**What would falsify this.** If agreement carried no information, precision
would be flat across support levels. It is not. Over 386 start bits proposed
on the 176 MQB messages that 3 or more platforms carry, scored against
`vw_mqb.dbc`:

======================  ========  =======  =========
support across platforms  proposed  correct  precision
======================  ========  =======  =========
one platform only              104       21        20%
under half                     103       30        29%
half to 89%                     67       34        51%
90% but not all                 13       11        85%
every platform                  99       84        85%
**all**                        386      180        47%
======================  ========  =======  =========

Precision rises monotonically from 20% to 85%, so a boundary's support is
worth roughly as much as the boundary itself. Keeping only what half the
platforms agree on discards 207 of 386 claims and 51 of 180 correct ones,
which is the trade the tiers exist to make explicit rather than to make
silently.

Filtering on the tier buys precision at a predictable cost. Over the same
seventeen platforms, scored per platform against `vw_mqb.dbc`:

====================  ==========  =======  ====
kept                  precision   recall   F1
====================  ==========  =======  ====
everything                  53%      47%   0.500
partial and better          61%      45%   0.518
established only            73%      38%   0.498
====================  ==========  =======  ====

The F1 gain is small. The point is the operating point: three claims in four
being right is a different kind of artefact from one in two, and no amount of
single-platform tuning reached it.

**What this does not do.** It does not invent boundaries. A cut no platform
proposed is not discovered by agreement, so recall is bounded above by what
`find_signals` already produces; this only sorts that output by how much of
the fleet stands behind each piece of it.

Messages are matched by `(bus, address)` as logged. Bus numbers are reconciled
*within* a platform by :mod:`canlens.corroborate.buses`, but nothing reconciles
them *between* platforms yet, so a group wired differently across models would
match fewer messages than it should -- failing towards fewer comparisons, not
towards wrong ones.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .consensus import ESTABLISHED_SUPPORT, PARTIAL_SUPPORT, Tier

if TYPE_CHECKING:  # pragma: no cover
    from ..infer.signals import SignalHypothesis

# Platforms needed before near-unanimity is called established rather than
# merely consistent. Three matches the device floor the segment-level
# corroboration uses, and for the same reason: two agreeing could be two
# recordings of one design decision.
MIN_PLATFORMS = 3


@dataclass(frozen=True)
class BoundarySupport:
    """One proposed start bit, and how much of the group proposed it."""

    start_bit: int
    platforms: int  # platforms whose own analysis put a field here
    of_platforms: int  # platforms carrying this message at all
    widths: tuple[int, ...]  # the widths claimed, one per proposing platform

    @property
    def support(self) -> float:
        return self.platforms / self.of_platforms if self.of_platforms else 0.0

    @property
    def agreed_width(self) -> int | None:
        """The width every proposing platform claimed, when they all agree."""
        return self.widths[0] if self.widths and len(set(self.widths)) == 1 else None

    @property
    def widest(self) -> int:
        """The strongest lower bound the group offers for this field's width.

        Widths disagree across platforms far more often than start bits do,
        and for a reason that is not disagreement at all: a field's high bits
        only move once some car drives it far enough, so each platform reports
        what its own drivers happened to exercise. Every claim is a lower
        bound, so the widest one is the best lower bound.

        It is not an estimate of the true width, and the numbers say so.
        Against `vw_mqb.dbc`, over the 129 corroborated boundaries whose start
        bit is right, the widest claim is exactly right 20% of the time
        (against 17% for the median claim and 5% for the narrowest) and still
        short 67% of the time, by a median of 3 bits. It overshoots the
        declared width 13% of the time, which is the rate at which treating it
        as a bound is wrong.
        """
        return max(self.widths) if self.widths else 0

    @property
    def tier(self) -> Tier:
        if self.support >= ESTABLISHED_SUPPORT:
            return (
                Tier.ESTABLISHED
                if self.of_platforms >= MIN_PLATFORMS
                else Tier.CONSISTENT
            )
        if self.support >= PARTIAL_SUPPORT:
            return Tier.PARTIAL
        return Tier.WEAK

    def __str__(self) -> str:
        width = (
            f"{self.agreed_width} bits"
            if self.agreed_width
            else f"at least {self.widest} bits"
        )
        return (
            f"bit {self.start_bit}: {width}, "
            f"{self.platforms}/{self.of_platforms} platforms, {self.tier}"
        )


@dataclass(frozen=True)
class MessageBoundaries:
    """Every start bit proposed for one message, by anyone in the group."""

    bus: int
    address: int
    of_platforms: int
    supports: tuple[BoundarySupport, ...]

    def at_least(self, tier: Tier) -> tuple[BoundarySupport, ...]:
        """The supports at `tier` or better, strongest first."""
        rank = {Tier.WEAK: 0, Tier.PARTIAL: 1, Tier.CONSISTENT: 2, Tier.ESTABLISHED: 3}
        return tuple(s for s in self.supports if rank[s.tier] >= rank[tier])

    def __str__(self) -> str:
        return (
            f"bus {self.bus} 0x{self.address:03X}: {len(self.supports)} start bits "
            f"proposed across {self.of_platforms} platforms"
        )


def agree(
    by_platform: dict[str, dict[tuple[int, int], list[SignalHypothesis]]],
    *,
    min_platforms: int = MIN_PLATFORMS,
) -> dict[tuple[int, int], MessageBoundaries]:
    """Tally start bits across platforms that already had signals extracted.

    `by_platform` maps a platform key to that platform's pooled signals, as
    :func:`canlens.corroborate.pooled.signals_across` returns them. Kept
    separate from the reading so a caller can supply cached results rather than
    decoding every segment of every platform again.
    """
    carried: dict[tuple[int, int], list[str]] = defaultdict(list)
    proposed: dict[tuple[int, int], dict[int, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for platform, messages in by_platform.items():
        for key, signals in messages.items():
            carried[key].append(platform)
            for signal in signals:
                proposed[key][signal.start_bit].append(signal.length)

    out: dict[tuple[int, int], MessageBoundaries] = {}
    for key, platforms in carried.items():
        if len(platforms) < min_platforms:
            continue
        supports = tuple(
            sorted(
                (
                    BoundarySupport(
                        start_bit=start,
                        platforms=len(widths),
                        of_platforms=len(platforms),
                        widths=tuple(widths),
                    )
                    for start, widths in proposed[key].items()
                ),
                key=lambda s: (-s.support, s.start_bit),
            )
        )
        bus, address = key
        out[key] = MessageBoundaries(bus, address, len(platforms), supports)
    return out


def boundaries_across(
    platforms: list[str],
    *,
    root: str,
    limit: int | None = None,
    min_platforms: int = MIN_PLATFORMS,
) -> dict[tuple[int, int], MessageBoundaries]:
    """Read signals from every platform in the group, then tally them.

    One pooled read per platform, which is the expensive part: every local
    segment of every platform is decoded. Pass a group that actually shares an
    architecture -- the whole point is that the message is the same message.
    """
    from .pooled import signals_across

    by_platform = {}
    for platform in platforms:
        found = signals_across(platform, root=root, limit=limit)
        if found:
            by_platform[platform] = found
    return agree(by_platform, min_platforms=min_platforms)
