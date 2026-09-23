# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Which bit order a bus is wired in, decided once for the whole bus.

Byte order is not a property of a signal. It is a property of the bus, and
that is what makes it decidable at all. Measured across the 43 opendbc
databases with twenty or more signals, the median share of signals in their
database's dominant order is **100%**, and 37 of 43 are at least 90% one
order.

More to the point, the exceptions almost never cost anything. A field inside
one byte occupies the same bits under either convention -- bits 3 to 6 of a
byte are a little-endian field at `StartPos` 3 and, identically, a big-endian
field at `StartPos` 6 -- so only a minority-order field that *wraps* a byte
boundary is mishandled by assuming one order throughout. That costs a median
of 0.00% of signals, a mean of 0.10%, and **41 of the 43 databases pay
nothing at all**.

**Why deciding per message fails and per bus works.** Within a byte the two
orders produce the same adjacencies, so only the byte seams differ, and one
message rarely has enough seams to settle anything -- a seam-based vote picks
big-endian buses right 38% of the time, worse than a coin. A bus offers
dozens to hundreds of messages, and the evidence adds up.

**The evidence.** A field of nine bits or more cannot fit inside a byte, so it
only appears as one contiguous run of decaying transition rate in the order
that is actually correct; in the wrong order it is split at the seam into
shorter pieces. Counting long fields under each order and taking the larger
count picks the right order on 8 of 9 buses, against 38% for the per-message
test. The one it gets wrong is also the least decided, at a 6% margin where
the narrowest correct answer sits at 12%, which is what `MIN_MARGIN` exists
to catch.

Scored on those buses, running detection in the decided order rather than
always Intel takes a Rivian from 2% precision and 3% recall to 14% and 18%,
and a Prius from nothing at all to 13% and 23%. Volkswagen platforms get
worse under Motorola, as they should, and that is the check that this is
measuring the bus rather than flattering the detector.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..analyze.bits import BitOrder, bit_matrix_from_bytes
from .signals import SignalHypothesis, find_signals

# A field this wide cannot fit inside one byte, so it can only be read as one
# run in the order that is correct. Nine rather than eight because an 8-bit
# field aligned to a byte reads identically either way and settles nothing.
LONG_FIELD = 9

# Below this relative margin the bus is left undecided rather than guessed.
# Calibrated on nine buses: the one wrong answer had a 6% margin and the
# narrowest correct one 12%.
MIN_MARGIN = 0.10


def motorola_columns(byte_matrix: np.ndarray) -> np.ndarray:
    """Unpack MSB-first, then reverse end to end.

    This is the ordering in which a big-endian signal behaves like an
    Intel-order one: contiguous, with its transition rate decaying upward from
    the least significant bit. Plain MSB-first is *not* enough and fails
    quietly -- a big-endian field runs from its most significant bit, so the
    rate climbs along the field and a rule that cuts on a rise fires inside
    fields instead of between them. Verified on all 344 big-endian layouts
    that fit in eight bytes.
    """
    return bit_matrix_from_bytes(byte_matrix, BitOrder.MOTOROLA)[:, ::-1]


def to_intel_bit(column: int, width_bits: int) -> int:
    """Map a column of :func:`motorola_columns` back to a flat Intel index."""
    byte, offset = divmod(width_bits - 1 - column, 8)
    return byte * 8 + (7 - offset)


def intel_to_column(bit: int, width_bits: int) -> int:
    """The inverse of :func:`to_intel_bit`."""
    return width_bits - 1 - ((bit // 8) * 8 + (7 - bit % 8))


def long_fields(
    byte_matrix: np.ndarray,
    *,
    claimed_bits: set[int] | frozenset[int] = frozenset(),
    long_field: int = LONG_FIELD,
) -> tuple[int, int]:
    """Long fields visible in one message under (Intel, Motorola) order."""
    width_bits = byte_matrix.shape[1] * 8
    intel = bit_matrix_from_bytes(byte_matrix, BitOrder.INTEL)
    motorola = motorola_columns(byte_matrix)
    mapped = {intel_to_column(b, width_bits) for b in claimed_bits}
    return (
        sum(1 for s in find_signals(intel, claimed_bits=claimed_bits)
            if s.length >= long_field),
        sum(1 for s in find_signals(motorola, claimed_bits=mapped)
            if s.length >= long_field),
    )


@dataclass(frozen=True)
class BusOrder:
    """What the traffic on one bus says about its bit order."""

    bus: int
    intel: int  # long fields readable as Intel
    motorola: int  # long fields readable as Motorola
    messages: int

    @property
    def margin(self) -> float:
        total = self.intel + self.motorola
        return abs(self.intel - self.motorola) / total if total else 0.0

    @property
    def decided(self) -> bool:
        return self.margin >= MIN_MARGIN

    @property
    def order(self) -> BitOrder:
        """The better-supported order. Meaningless unless `decided`."""
        return BitOrder.INTEL if self.intel >= self.motorola else BitOrder.MOTOROLA

    def __str__(self) -> str:
        verdict = f"{self.order}" if self.decided else "undecided"
        return (
            f"bus {self.bus}: {verdict} ({self.intel} intel / {self.motorola} "
            f"motorola long fields over {self.messages} messages, "
            f"margin {self.margin:.0%})"
        )


def decide_byte_order(
    messages: list[tuple[int, np.ndarray, set[int]]],
    *,
    long_field: int = LONG_FIELD,
) -> dict[int, BusOrder]:
    """Decide each bus's bit order from `(bus, byte_matrix, claimed_bits)`.

    `claimed_bits` are the Intel-order positions other detectors already
    explained on that message; they are excluded from both counts so a counter
    or checksum cannot be mistaken for a long field in either order.
    """
    tally: dict[int, list[int]] = {}
    for bus, byte_matrix, claimed in messages:
        if byte_matrix.size == 0 or byte_matrix.shape[0] < 2:
            continue
        intel, motorola = long_fields(
            byte_matrix, claimed_bits=claimed, long_field=long_field
        )
        row = tally.setdefault(bus, [0, 0, 0])
        row[0] += intel
        row[1] += motorola
        row[2] += 1
    return {
        bus: BusOrder(bus=bus, intel=i, motorola=m, messages=n)
        for bus, (i, m, n) in sorted(tally.items())
    }


def find_signals_in(
    byte_matrix: np.ndarray,
    order: BitOrder,
    *,
    claimed_bits: set[int] | frozenset[int] = frozenset(),
    **options: object,
) -> list[SignalHypothesis]:
    """Detect signals in one message, reading it in the bus's bit order.

    Only the signal pass moves. Counters, checksums and multiplexors are
    verified arithmetic over bytes and are unaffected by bit numbering, so
    they stay where they were found; `claimed_bits` arrives in Intel indices
    either way and is mapped across for the Motorola pass.

    A Motorola claim comes back labelled as a DBC big-endian field: its
    `start_bit` is the *most* significant bit, which is the run's top column
    mapped back, so `SignalHypothesis.bits` reproduces the wrapped positions.
    """
    if order is BitOrder.INTEL:
        return find_signals(
            bit_matrix_from_bytes(byte_matrix, BitOrder.INTEL),
            claimed_bits=claimed_bits,
            **options,  # type: ignore[arg-type]
        )
    width_bits = byte_matrix.shape[1] * 8
    found = find_signals(
        motorola_columns(byte_matrix),
        claimed_bits={intel_to_column(b, width_bits) for b in claimed_bits},
        **options,  # type: ignore[arg-type]
    )
    return [
        replace(
            hypothesis,
            start_bit=to_intel_bit(hypothesis.end_bit - 1, width_bits),
            byte_order=BitOrder.MOTOROLA,
        )
        for hypothesis in found
    ]
