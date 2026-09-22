# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Score what canlens inferred against what a DBC says.

This is the module that turns "the engine produced an answer" into "the engine
is measurably right". Every threshold in :mod:`canlens.infer` was chosen by
reading corpus surveys and judging which findings looked like artefacts. That
is an argument. What follows is a measurement.

**A disagreement is evidence, not a verdict.** A community DBC is itself
reverse-engineered: it goes stale, it disagrees with other DBCs, and its
authors routinely leave counters unnamed because naming them was not the point
of the exercise. So a canlens finding the DBC does not have is not
automatically wrong, and the output says so. Both directions are reported with
enough detail to look at the message and decide.

**What is scored.** Only the three kinds of field canlens actually claims:
counters, checksums (8-bit and wider, pooled, since the DBC does not
distinguish them) and multiplexors. Ordinary signals are counted for context
and not scored -- canlens does not yet infer signal boundaries, so scoring them
would report a recall of zero and say nothing.

**What is scorable.** Recall is computed only over messages present in *both*
the reference and the trace, at a payload long enough to have been inferred. A
DBC message the drive never exercised is not a miss; it is absent data, and it
is counted separately so the denominator stays honest.

The same care applies in the other direction. A reference that names no field
of a kind at all cannot evaluate claims of that kind: the older Honda and
Acura DBCs annotate neither counters nor checksums, so scoring canlens' 44
counter findings against `acura_ilx_2016_nidec.dbc` would report a precision of
0% when the truthful answer is "this reference cannot say". Such a kind is left
out of the rates entirely and its claims are reported as unevaluated.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from ..infer import MessageInference
from .dbc import FieldKind, Reference, ReferenceMessage

# The kinds canlens makes claims about, and therefore the kinds worth scoring.
SCORED_KINDS: tuple[FieldKind, ...] = (
    FieldKind.COUNTER,
    FieldKind.CHECKSUM,
    FieldKind.MULTIPLEXOR,
)


@dataclass(frozen=True)
class Tally:
    """Counts for one kind of field, and the rates derived from them."""

    hits: int = 0
    false_alarms: int = 0  # claimed by canlens, not named in the reference
    missed: int = 0  # named in the reference, not claimed by canlens
    near: int = 0  # overlapped the right bits without matching exactly

    @property
    def claimed(self) -> int:
        return self.hits + self.false_alarms

    @property
    def expected(self) -> int:
        return self.hits + self.missed

    @property
    def precision(self) -> float:
        """Share of what canlens claimed that the reference agrees with."""
        return self.hits / self.claimed if self.claimed else 0.0

    @property
    def recall(self) -> float:
        """Share of what the reference names that canlens found."""
        return self.hits / self.expected if self.expected else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def __add__(self, other: Tally) -> Tally:
        return Tally(
            self.hits + other.hits,
            self.false_alarms + other.false_alarms,
            self.missed + other.missed,
            self.near + other.near,
        )

    def __str__(self) -> str:
        return (
            f"{self.hits} hit, {self.missed} missed, {self.false_alarms} extra "
            f"(precision {self.precision:.0%}, recall {self.recall:.0%})"
        )


@dataclass(frozen=True)
class Disagreement:
    """One place canlens and the reference do not say the same thing."""

    address: int
    message: str
    kind: FieldKind
    claimed: str | None  # None when canlens claimed nothing
    expected: str | None  # None when the reference names nothing
    overlapping: bool = False  # the two describe overlapping bits

    @property
    def verdict(self) -> str:
        if self.claimed is None:
            return "missed"
        if self.expected is None:
            return "extra"
        return "moved"

    def __str__(self) -> str:
        where = f"0x{self.address:03X} {self.message}"
        if self.claimed is None:
            return f"{where}: {self.kind} not found -- reference has {self.expected}"
        if self.expected is None:
            return f"{where}: claimed {self.kind} {self.claimed} -- reference names none"
        return f"{where}: claimed {self.kind} {self.claimed} -- reference has {self.expected}"


@dataclass
class Score:
    """The result of comparing one segment against one reference."""

    reference: str
    bus: int
    tallies: dict[FieldKind, Tally] = field(default_factory=dict)
    disagreements: list[Disagreement] = field(default_factory=list)
    scored_messages: int = 0  # in both the reference and the trace
    reference_messages: int = 0
    trace_messages: int = 0  # on the scored bus
    absent_from_trace: int = 0  # reference messages the drive never exercised
    unnamed_signals: int = 0  # ordinary signals, counted but not scored
    # Kinds the reference names at least once, and so can evaluate.
    scorable: set[FieldKind] = field(default_factory=set)
    # Claims left unevaluated because the reference names no field of that kind.
    unevaluated: dict[FieldKind, int] = field(default_factory=dict)

    @property
    def overall(self) -> Tally:
        total = Tally()
        for kind, tally in self.tallies.items():
            if kind in self.scorable:
                total = total + tally
        return total

    def __str__(self) -> str:
        if not self.scorable:
            return (
                f"{self.reference} vs bus {self.bus}: {self.scored_messages} messages scored, "
                f"nothing scorable -- the reference names no counter, checksum or multiplexor"
            )
        return (
            f"{self.reference} vs bus {self.bus}: {self.scored_messages} messages scored, "
            f"{self.overall}"
        )


def claimed_fields(inference: MessageInference) -> dict[FieldKind, list[tuple[tuple[int, ...], str]]]:
    """What canlens claims about one message, as (bit positions, description).

    Checksums and 16/32/64-bit CRCs are pooled into one kind: a DBC names both
    `CHECKSUM` and `CRC` without distinguishing the width, so keeping them
    apart here would invent a distinction the reference cannot answer.
    """
    out: dict[FieldKind, list[tuple[tuple[int, ...], str]]] = {k: [] for k in SCORED_KINDS}

    for counter in inference.counters:
        bits = tuple(range(counter.start_bit, counter.start_bit + counter.length))
        out[FieldKind.COUNTER].append((bits, f"{counter.length}bit@{counter.start_bit}"))

    for checksum in inference.checksums:
        # start_bit and length rather than the byte index: Honda's checksum is
        # four bits, and claiming the whole byte would claim the counter beside it.
        bits = tuple(range(checksum.start_bit, checksum.start_bit + checksum.length))
        out[FieldKind.CHECKSUM].append(
            (bits, f"{checksum.algorithm}@byte{checksum.byte_index}")
        )
    for crc in inference.crc16s:
        start = crc.start_byte * 8
        out[FieldKind.CHECKSUM].append(
            (
                tuple(range(start, start + crc.nbytes * 8)),
                f"{crc.algorithm}@byte{crc.start_byte}",
            )
        )

    if inference.multiplexor is not None:
        mux = inference.multiplexor
        out[FieldKind.MULTIPLEXOR].append(
            (
                tuple(range(mux.start_bit, mux.end_bit)),
                f"{mux.length}bit@{mux.start_bit}",
            )
        )
    return out


def _compare(
    kind: FieldKind,
    claims: Sequence[tuple[tuple[int, ...], str]],
    expected: Sequence,
    message: ReferenceMessage,
) -> tuple[Tally, list[Disagreement]]:
    """Match claims to reference signals of one kind, by exact bit set.

    Exact equality is the bar deliberately. A counter reported one bit wide of
    the truth is not a hit, and folding it in as one would make the headline
    number meaningless. Overlaps are still counted, as `near`, so that "we are
    looking at the right field and got its extent wrong" stays visible.
    """
    wanted = {frozenset(s.bits): s for s in expected}
    got = {frozenset(bits): text for bits, text in claims}

    hits = sorted(wanted.keys() & got.keys(), key=lambda s: min(s) if s else 0)
    only_claimed = sorted(got.keys() - wanted.keys(), key=lambda s: min(s) if s else 0)
    only_expected = sorted(wanted.keys() - got.keys(), key=lambda s: min(s) if s else 0)

    disagreements: list[Disagreement] = []
    near = 0
    unpaired = list(only_expected)
    for claim in only_claimed:
        partner = next((e for e in unpaired if e & claim), None)
        if partner is not None:
            unpaired.remove(partner)
            near += 1
            disagreements.append(
                Disagreement(
                    message.address, message.name, kind, got[claim],
                    str(wanted[partner]), overlapping=True,
                )
            )
        else:
            disagreements.append(
                Disagreement(message.address, message.name, kind, got[claim], None)
            )
    for leftover in unpaired:
        disagreements.append(
            Disagreement(message.address, message.name, kind, None, str(wanted[leftover]))
        )

    return (
        Tally(hits=len(hits), false_alarms=len(only_claimed), missed=len(only_expected), near=near),
        disagreements,
    )


def pick_bus(inferences: Iterable[MessageInference], reference: Reference) -> int:
    """The bus whose identifiers overlap the reference most.

    A DBC describes one bus; a corpus segment carries several, and their
    numbering is the logger's rather than the DBC author's. Rather than ask
    the caller to know the mapping, the overlap picks it -- and the count is
    reported, so a bad guess is visible instead of silent.
    """
    per_bus: dict[int, set[int]] = {}
    for inference in inferences:
        per_bus.setdefault(inference.bus, set()).add(inference.address)
    if not per_bus:
        return 0
    return max(per_bus, key=lambda bus: (len(per_bus[bus] & reference.addresses), -bus))


def score(
    inferences: Sequence[MessageInference],
    reference: Reference,
    *,
    bus: int | None = None,
) -> Score:
    """Compare one segment's inference against one DBC."""
    if bus is None:
        bus = pick_bus(inferences, reference)
    on_bus = {m.address: m for m in inferences if m.bus == bus}

    # A kind the reference never names cannot evaluate claims of that kind.
    # Counted over the whole DBC rather than the scored subset, because the
    # question it answers is "do these authors annotate counters at all".
    scorable = {kind for kind in SCORED_KINDS if reference.count(kind) > 0}

    result = Score(
        reference=reference.name,
        bus=bus,
        tallies={kind: Tally() for kind in SCORED_KINDS},
        reference_messages=len(reference),
        trace_messages=len(on_bus),
        scorable=scorable,
        unevaluated={kind: 0 for kind in SCORED_KINDS if kind not in scorable},
    )

    for address, expected in sorted(reference.messages.items()):
        inference = on_bus.get(address)
        if inference is None:
            result.absent_from_trace += 1
            continue
        result.scored_messages += 1
        result.unnamed_signals += len(expected.of_kind(FieldKind.SIGNAL))
        claims = claimed_fields(inference)
        for kind in SCORED_KINDS:
            if kind not in scorable:
                result.unevaluated[kind] += len(claims[kind])
                continue
            tally, disagreements = _compare(
                kind, claims[kind], expected.of_kind(kind), expected
            )
            result.tallies[kind] = result.tallies[kind] + tally
            result.disagreements.extend(disagreements)

    result.disagreements.sort(key=lambda d: (d.address, str(d.kind)))
    return result
