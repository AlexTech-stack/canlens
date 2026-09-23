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

**What is scored.** The four kinds of field canlens claims: counters,
checksums (8-bit and wider, pooled, since the DBC does not distinguish them),
multiplexors and ordinary signals.

Signals need one allowance the others do not. A numeric field's high bits stop
moving when the value never grew large enough to reach them, and a trace cannot
tell that from the field ending there, so `find_signals` reports an unbounded
claim as a *lower bound* -- these bits and possibly more above. Judging an
explicit lower bound by exact equality would measure something nobody claimed,
so an unbounded claim also matches a reference signal starting at the same bit
and running wider. A bounded claim gets no such allowance, and neither does
any other kind of field.

They also need the same exemption a message gets. A reference signal whose
bits never move in the trace cannot be found: there is nothing to see. Scoring
them as misses measures how much of the car the driver exercised, not how well
the detector works -- of 4566 signals the DBCs name on messages that were
recorded, 4307 sat completely still. Those are counted separately, like a
message the drive never carried.

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
    FieldKind.SIGNAL,
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
    still_signals: int = 0  # reference signals whose bits never moved
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


@dataclass(frozen=True)
class Claim:
    """One field canlens claims, and how exactly it claims it."""

    bits: tuple[int, ...]
    text: str
    # False for a signal whose width is a lower bound; see the module docstring.
    exact: bool = True
    # The selector value a layout field belongs to; None for a static claim.
    # Two layouts can claim the same bits under different values, so this is
    # half of the identity a match is judged on.
    mux_value: int | None = None


def claimed_fields(inference: MessageInference) -> dict[FieldKind, list[Claim]]:
    """What canlens claims about one message.

    Checksums and 16/32/64-bit CRCs are pooled into one kind: a DBC names both
    `CHECKSUM` and `CRC` without distinguishing the width, so keeping them
    apart here would invent a distinction the reference cannot answer.
    """
    out: dict[FieldKind, list[Claim]] = {k: [] for k in SCORED_KINDS}

    for counter in inference.counters:
        bits = tuple(range(counter.start_bit, counter.start_bit + counter.length))
        out[FieldKind.COUNTER].append(Claim(bits, f"{counter.length}bit@{counter.start_bit}"))

    for checksum in inference.checksums:
        # start_bit and length rather than the byte index: Honda's checksum is
        # four bits, and claiming the whole byte would claim the counter beside it.
        bits = tuple(range(checksum.start_bit, checksum.start_bit + checksum.length))
        out[FieldKind.CHECKSUM].append(
            Claim(bits, f"{checksum.algorithm}@byte{checksum.byte_index}")
        )
    for crc in inference.crc16s:
        start = crc.start_byte * 8
        out[FieldKind.CHECKSUM].append(
            Claim(
                tuple(range(start, start + crc.nbytes * 8)),
                f"{crc.algorithm}@byte{crc.start_byte}",
            )
        )

    if inference.multiplexor is not None:
        mux = inference.multiplexor
        out[FieldKind.MULTIPLEXOR].append(
            Claim(
                tuple(range(mux.start_bit, mux.end_bit)),
                f"{mux.length}bit@{mux.start_bit}",
            )
        )

    for signal in inference.signals:
        width = f"{signal.length}bit" if signal.bounded else f"{signal.length}+bit"
        out[FieldKind.SIGNAL].append(
            Claim(
                signal.bits,
                f"{width}@{signal.start_bit}",
                exact=signal.bounded,
            )
        )

    # A multiplexed layout's constants are ordinary reference signals to the
    # DBC, but they only exist under one selector value, so the value is part
    # of what a match has to agree on.
    for layout in inference.layout_fields:
        out[FieldKind.SIGNAL].append(
            Claim(
                layout.bits,
                f"layout{layout.mux_value}@byte{layout.start_bit // 8}",
                mux_value=layout.mux_value,
            )
        )
    return out


def _compare(
    kind: FieldKind,
    claims: Sequence[Claim],
    expected: Sequence,
    message: ReferenceMessage,
) -> tuple[Tally, list[Disagreement]]:
    """Match claims to reference signals of one kind, by bit set and mux value.

    Exact equality is the bar deliberately. A counter reported one bit wide of
    the truth is not a hit, and folding it in as one would make the headline
    number meaningless. Overlaps are still counted, as `near`, so that "we are
    looking at the right field and got its extent wrong" stays visible.

    The selector value joins the bit set in the identity. A multiplexed
    message puts *different* fields on the same bits under different values,
    so matching on bits alone would call a layout-0 field a hit for a layout-1
    reference and vice versa.
    """
    wanted = {(frozenset(s.bits), getattr(s, "mux_value", None)): s for s in expected}
    got = {(frozenset(c.bits), c.mux_value): c.text for c in claims}
    loose = {frozenset(c.bits) for c in claims if not c.exact and c.mux_value is None}

    def order(key: tuple[frozenset[int], int | None]) -> int:
        return min(key[0]) if key[0] else 0

    hits = sorted(wanted.keys() & got.keys(), key=order)
    only_claimed = sorted(got.keys() - wanted.keys(), key=order)
    only_expected = sorted(wanted.keys() - got.keys(), key=order)

    # A lower bound matches a reference field that starts where it does and
    # runs wider. Only static claims can be lower bounds.
    matched: list[tuple[frozenset[int], int | None]] = []
    for claim in only_claimed:
        bits, mux = claim
        if mux is not None or bits not in loose or not bits:
            continue
        partner = next(
            (
                e
                for e in only_expected
                if e[1] is None and e[0] >= bits and min(e[0]) == min(bits) and e not in matched
            ),
            None,
        )
        if partner is not None:
            matched.append(partner)
            hits.append(claim)
    hit_keys = set(hits)
    only_claimed = [c for c in only_claimed if c not in hit_keys]
    only_expected = [e for e in only_expected if e not in matched]

    disagreements: list[Disagreement] = []
    near = 0
    unpaired = list(only_expected)
    for claim in only_claimed:
        partner = next((e for e in unpaired if e[0] & claim[0] and e[1] == claim[1]), None)
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


def _moves(inference: MessageInference, signal) -> bool:
    """Whether any bit of a reference signal changed during the trace.

    Read off the bit profile `analyze` already measured, using the same
    threshold `find_signals` uses, so the scorer and the detector agree on
    what counts as movement.
    """
    from ..infer.signals import MIN_RATE

    rates = inference.bits.rates
    return any(
        bit < len(rates) and rates[bit] >= MIN_RATE for bit in signal.bits
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
            wanted = expected.of_kind(kind)
            if kind is FieldKind.SIGNAL:
                # A static signal that never moved cannot be found: there is
                # nothing to see, so counting it as a miss measures the driver.
                # A multiplexed signal is different -- it may be a layout
                # constant, which `layout_fields` does claim -- so the mux ones
                # are scored whether or not their bits moved.
                static = [s for s in wanted if s.mux_value is None]
                mux_refs = [s for s in wanted if s.mux_value is not None]
                moving = [s for s in static if _moves(inference, s)]
                result.still_signals += len(static) - len(moving)
                wanted = moving + mux_refs
            tally, disagreements = _compare(kind, claims[kind], wanted, expected)
            result.tallies[kind] = result.tallies[kind] + tally
            result.disagreements.extend(disagreements)

    result.disagreements.sort(key=lambda d: (d.address, str(d.kind)))
    return result
