# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Identifying a bus by what is on it, rather than by where it was logged.

A segment's bus numbers come from the logger's port assignment, and nothing
guarantees that the port a wire was plugged into is the same from drive to
drive. Measured across the corpus by asking which bus in another segment of
the same car shares the most identifiers, **293 of 5868 comparisons name a
different number than the one logged**, over 12 of the 31 platforms holding
more than one segment. The KIA EV6 disagrees with itself 42% of the time, the
NMS Passat 33%, the Taos 24%, and several platforms have segment pairs where
the same number shares no identifiers at all.

So grouping by the logged number silently pools two different buses. What this
module provides instead is a *canonical* label per platform, derived from the
traffic.

**The signal is the identifier set.** Two recordings of one bus carry mostly
the same identifiers; two different buses carry almost none in common. Over
2548 same-numbered and 5047 differently-numbered comparisons the medians are
0.99 and 0.04.

**Payload width was tried as a second signal and dropped.** Counting only the
shared identifiers that also agree on width gives a curve indistinguishable
from plain overlap -- 92.8% against 92.9% of pairs retained at the same
threshold. A message keeps its length wherever it appears, so width says
nothing that the identifier had not already said. It is recorded here because
the obvious next idea is to add more features, and this one measurably does
not help.

**Matching is an assignment, not a threshold.** One bus cannot be two buses,
so the buses of one segment are matched to another's one-to-one, maximising
total overlap, rather than each picking its own best partner. That also sides
steps a problem with calibration: a threshold would have to be tuned against
ground truth, and the only ground truth available is the logged number, which
is the thing in doubt. Under assignment the question is which pairing is best
overall, and the best beats the runner-up by a median of 0.65.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from itertools import permutations
from typing import TYPE_CHECKING

from .consensus import device_of

if TYPE_CHECKING:
    from ..decode import FrameSet

# Overlap below which two buses are held to be unrelated, so a bus with no
# counterpart becomes an identity of its own rather than being forced onto the
# nearest one. Set well below the same-bus distribution (5th percentile 0.28)
# and well above the different-bus one (median 0.04): the assignment does the
# discriminating, and this only has to catch "there is nothing to match".
MIN_OVERLAP = 0.15

# Buses per segment run from one to six, so an exhaustive assignment is at
# most a few hundred permutations. The guard is for a corpus that grows
# stranger than this one, not for anything present in it.
MAX_EXHAUSTIVE = 8


@dataclass(frozen=True)
class BusSignature:
    """What one bus of one segment carries."""

    bus: int  # as logged
    addresses: frozenset[int]
    frames: int
    messages: int

    def __str__(self) -> str:
        return f"bus {self.bus}: {len(self.addresses)} identifiers, {self.frames} frames"


def overlap(a: Iterable[int], b: Iterable[int]) -> float:
    """Jaccard similarity of two identifier sets."""
    first, second = set(a), set(b)
    union = first | second
    return len(first & second) / len(union) if union else 0.0


def signatures(frames: FrameSet, *, min_frames: int = 1) -> list[BusSignature]:
    """One signature per bus present in a decoded segment."""
    addresses: dict[int, set[int]] = {}
    counts: dict[int, int] = {}
    for message in frames.group(min_frames=min_frames):
        addresses.setdefault(message.bus, set()).add(message.address)
        counts[message.bus] = counts.get(message.bus, 0) + message.count
    return [
        BusSignature(
            bus=bus,
            addresses=frozenset(found),
            frames=counts.get(bus, 0),
            messages=len(found),
        )
        for bus, found in sorted(addresses.items())
    ]


def signatures_from(inferences: Iterable) -> list[BusSignature]:
    """Signatures from inference results rather than from decoded frames.

    `corroborate` already reads every segment's cached inference, and those
    carry the bus, the identifier and the frame count -- everything a
    signature needs. Deriving them here rather than decoding again keeps a
    platform pass to the cache reads it already made.
    """
    addresses: dict[int, set[int]] = {}
    counts: dict[int, int] = {}
    for message in inferences:
        addresses.setdefault(message.bus, set()).add(message.address)
        counts[message.bus] = counts.get(message.bus, 0) + message.frames
    return [
        BusSignature(
            bus=bus,
            addresses=frozenset(found),
            frames=counts.get(bus, 0),
            messages=len(found),
        )
        for bus, found in sorted(addresses.items())
    ]


def assign(
    left: Sequence[frozenset[int]],
    right: Sequence[frozenset[int]],
    *,
    min_overlap: float = MIN_OVERLAP,
) -> dict[int, int]:
    """Best one-to-one pairing of `left` onto `right`, by index.

    Returns only pairs clearing `min_overlap`; a bus with no counterpart is
    simply absent from the result. Exhaustive because the sides are tiny, and
    exhaustive is the only way to be sure the pairing is the best one rather
    than the first one a greedy pass happened to take.
    """
    if not left or not right:
        return {}
    if max(len(left), len(right)) > MAX_EXHAUSTIVE:
        return _greedy(left, right, min_overlap)

    short, long_ = (left, right) if len(left) <= len(right) else (right, left)
    flip = short is right
    best_total = -1.0
    best: tuple[int, ...] = ()
    for candidate in permutations(range(len(long_)), len(short)):
        total = sum(overlap(short[i], long_[j]) for i, j in enumerate(candidate))
        if total > best_total:
            best_total, best = total, candidate

    out: dict[int, int] = {}
    for i, j in enumerate(best):
        if overlap(short[i], long_[j]) < min_overlap:
            continue
        out[j if flip else i] = i if flip else j
    return out


def _greedy(
    left: Sequence[frozenset[int]], right: Sequence[frozenset[int]], min_overlap: float
) -> dict[int, int]:
    """Fallback for implausibly many buses: best pair first, then the rest."""
    scored = sorted(
        ((overlap(a, b), i, j) for i, a in enumerate(left) for j, b in enumerate(right)),
        reverse=True,
    )
    out: dict[int, int] = {}
    taken: set[int] = set()
    for score, i, j in scored:
        if score < min_overlap or i in out or j in taken:
            continue
        out[i] = j
        taken.add(j)
    return out


@dataclass
class BusIdentity:
    """One physical bus of a platform, as seen across its segments."""

    label: int  # canonical
    addresses: set[int] = field(default_factory=set)
    segments: int = 0
    devices: set[str] = field(default_factory=set)
    logged_as: dict[int, int] = field(default_factory=dict)

    @property
    def stable(self) -> bool:
        """Whether every segment logged this bus under the same number."""
        return len(self.logged_as) <= 1

    def __str__(self) -> str:
        numbers = ", ".join(
            f"{bus} in {count}" for bus, count in sorted(self.logged_as.items())
        )
        return (
            f"bus {self.label}: {len(self.addresses)} identifiers over "
            f"{self.segments} segments, {len(self.devices)} devices; logged as {numbers}"
        )


@dataclass
class BusMap:
    """Canonical bus labels for one platform, and how to apply them."""

    platform: str
    identities: list[BusIdentity] = field(default_factory=list)
    # path -> {logged bus number: canonical label}
    per_segment: dict[str, dict[int, int]] = field(default_factory=dict)

    def canonical(self, path: str, bus: int) -> int:
        """The canonical label for a logged bus, or the logged number itself.

        Falling back to the logged number rather than raising keeps a caller
        that has one unmapped segment working; it is the same behaviour as
        before this module existed.
        """
        return self.per_segment.get(path, {}).get(bus, bus)

    @property
    def relabelled(self) -> int:
        """How many (segment, bus) pairs get a number they were not logged as."""
        return sum(
            1
            for mapping in self.per_segment.values()
            for logged, label in mapping.items()
            if logged != label
        )

    @property
    def contested(self) -> dict[int, list[BusIdentity]]:
        """Logged numbers that turn out to name more than one bus.

        Two identities sharing a number and sharing no identifiers is the
        signature of a platform recorded under two harness configurations:
        the same port carried a different bus on different cars. The KIA EV6,
        Audi A3 and Seat Ateca all show it.
        """
        by_number: dict[int, list[BusIdentity]] = {}
        for identity in self.identities:
            for logged in identity.logged_as:
                by_number.setdefault(logged, []).append(identity)
        return {number: found for number, found in by_number.items() if len(found) > 1}

    @property
    def unstable(self) -> list[BusIdentity]:
        """Identities that appeared under more than one logged number."""
        return [identity for identity in self.identities if not identity.stable]

    def __str__(self) -> str:
        return (
            f"{self.platform}: {len(self.identities)} buses across "
            f"{len(self.per_segment)} segments, {self.relabelled} relabelled"
        )


def identify_from(
    platform: str,
    observed: Iterable[tuple[str, list[BusSignature]]],
    *,
    min_overlap: float = MIN_OVERLAP,
) -> BusMap:
    """Give every bus of a platform a label derived from its traffic.

    Segments are folded in one at a time. Each one's buses are assigned to the
    identities already known, and anything left over starts a new identity.
    An identity accumulates the union of the identifiers ever seen on it, so
    matching gets easier as evidence arrives rather than depending on whichever
    drive happened to come first.

    The label given to an identity is the number it was *most often* logged
    under, so the output still reads like the bus numbers everyone knows; only
    the segments that disagree are moved.
    """
    identities: list[BusIdentity] = []
    assigned: dict[str, dict[int, int]] = {}
    for path, found in observed:
        if not found:
            continue
        pairing = assign(
            [s.addresses for s in found],
            [frozenset(i.addresses) for i in identities],
            min_overlap=min_overlap,
        )
        mapping: dict[int, int] = {}
        for index, signature in enumerate(found):
            slot = pairing.get(index)
            if slot is None:
                identities.append(BusIdentity(label=signature.bus))
                slot = len(identities) - 1
            identity = identities[slot]
            identity.addresses |= signature.addresses
            identity.segments += 1
            identity.devices.add(device_of(path))
            identity.logged_as[signature.bus] = identity.logged_as.get(signature.bus, 0) + 1
            mapping[signature.bus] = slot
        assigned[path] = mapping

    # Name each identity after the number it was most often logged as, keeping
    # names unique so two identities can never collapse into one label.
    #
    # Best-supported first, which matters whenever two identities want the
    # same number. On the Audi A3 one segment's "bus 1" shares no identifiers
    # at all with the nineteen others', and claiming the number in order of
    # appearance handed it to the singleton and renamed the other nineteen --
    # nineteen relabellings to describe one odd segment. The majority keeps
    # the name it was logged under and the minority moves.
    used: set[int] = set()
    labels: list[int] = [0] * len(identities)
    order = sorted(
        range(len(identities)),
        key=lambda slot: (-identities[slot].segments, slot),
    )
    for slot in order:
        identity = identities[slot]
        modal = max(identity.logged_as.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        while modal in used:
            modal += 1
        used.add(modal)
        identity.label = modal
        labels[slot] = modal

    return BusMap(
        platform=platform,
        identities=identities,
        per_segment={
            path: {bus: labels[slot] for bus, slot in mapping.items()}
            for path, mapping in assigned.items()
        },
    )


def identify_buses(
    platform: str,
    *,
    root: str,
    limit: int | None = None,
    min_overlap: float = MIN_OVERLAP,
    load: Callable[..., FrameSet] | None = None,
    paths: Sequence[str] | None = None,
) -> BusMap:
    """`identify_from`, reading the segments of one platform off disk."""
    from ..corpus import Manifest, inventory
    from ..decode import load_frames

    loader = load or load_frames
    if paths is None:
        held = inventory(root, Manifest.load(f"{root}/database.json")).get(platform)
        paths = list(held.paths if held else [])[:limit]

    def observed():
        for path in paths or ():
            try:
                yield path, signatures(loader(path, root=root))
            except (OSError, ValueError):
                continue

    return identify_from(platform, observed(), min_overlap=min_overlap)
