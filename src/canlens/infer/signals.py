# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Where one numeric field ends and the next begins.

Every other detector here claims a field because it can *reproduce* it: a
counter advances, a checksum recomputes. An ordinary signal offers nothing of
the kind. A wheel speed is just bits that move, and the only question is which
of them move together.

**The start of a field is visible; the end is not.** Inside a numeric field the
transition rate falls away from the least significant bit -- bit 0 flips every
frame, bit 1 every other frame, and so on -- so the boundary between two fields
shows up as the rate *rising* again where the next field's low bit begins.
Measured against opendbc, a rate that rises to three times its neighbour is the
best cut: it puts the start bit right on 210 of 411 real signals.

The far end has no marker at all. A field's high bits stop moving because the
value never grew large enough to reach them, and a trace cannot tell that from
the field ending there. Of 411 signals whose bits move, only 157 have every
declared bit moving; 209 start exactly where the DBC says and simply run out
of evidence early. The rate profile of a genuine 16-bit signal reads

    0.30 0.29 0.30 0.30 0.27 0.20 0.12 0.06 0.03 0.02 0.01 0.01 0.00 0.00 0.00 0.00

and nothing in it says sixteen rather than ten.

So what is claimed is the span the trace justifies, and `bounded` records
whether that is the whole story. A field stopped by another moving field, by a
counter or checksum, or by the end of the payload is bounded and its width is
the width. A field that simply faded into still bits is a **lower bound**: the
real one may be wider, and saying so is the only honest option.

Scored against opendbc over six platforms, counting a claim right when it gets
the start bit and either the exact width or a narrower one whose omitted bits
never moved: 57% precision, 46% recall.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .counters import field_values

# A bit must move at least this often to be part of a signal. Below it there is
# no evidence of a field: a bit that flips twice in six thousand frames is a
# rare state change, which `corroborate` reports as a rare bit rather than
# something to group with its neighbours.
MIN_RATE = 0.02

# The rate must rise by this factor for a bit to start a new field rather than
# continue the one below it. Swept against opendbc: 1.5 gives 40% precision,
# 2.0 gives 52%, 3.0 gives 57%, and beyond that it flattens.
RISE = 3.0

# One bit is a flag, not a field, and flags are what the bit-class map already
# shows. Two is the shortest thing worth calling a signal.
MIN_LENGTH = 2


@dataclass(frozen=True)
class SignalHypothesis:
    """A run of bits that move together as one number."""

    start_bit: int
    length: int
    frames: int
    rate: float  # transition rate of the lowest bit
    minimum: int
    maximum: int
    # Whether something stops the field here, as opposed to it fading into
    # bits that never move. False means the width is a lower bound.
    bounded: bool = True

    @property
    def end_bit(self) -> int:
        return self.start_bit + self.length

    @property
    def span(self) -> int:
        return self.maximum - self.minimum

    def __str__(self) -> str:
        width = f"{self.length} bits" if self.bounded else f"{self.length}+ bits"
        return (
            f"signal @ bit {self.start_bit}, {width} "
            f"(seen {self.minimum}..{self.maximum} over {self.frames} frames)"
        )


def find_signals(
    matrix: np.ndarray,
    *,
    claimed_bits: set[int] | frozenset[int] = frozenset(),
    min_rate: float = MIN_RATE,
    rise: float = RISE,
    min_length: int = MIN_LENGTH,
) -> list[SignalHypothesis]:
    """Group adjacent moving bits into fields.

    `claimed_bits` are positions another detector already explained. They end
    a field rather than joining it: a counter is not the low half of the
    signal beside it, and running through one would claim bits that are
    already spoken for.
    """
    frames, width = matrix.shape
    if frames < 2 or width == 0:
        return []
    if matrix.dtype != np.int64:
        matrix = matrix.astype(np.int64)
    rate = (np.diff(matrix, axis=0) != 0).mean(axis=0)

    found: list[SignalHypothesis] = []
    start: int | None = None
    for bit in range(width + 1):
        dead = bit == width or bit in claimed_bits or rate[bit] < min_rate
        new_field = (
            not dead
            and start is not None
            and bit > start
            and rate[bit] > rate[bit - 1] * rise
        )
        if (dead or new_field) and start is not None and bit - start >= min_length:
            found.append(
                _describe(matrix, rate, start, bit - start, claimed_bits, min_rate)
            )
        if dead:
            start = None
        elif new_field or start is None:
            start = bit
    return found


def _describe(
    matrix: np.ndarray,
    rate: np.ndarray,
    start: int,
    length: int,
    claimed_bits: set[int] | frozenset[int],
    min_rate: float,
) -> SignalHypothesis:
    """Read one field's values, and decide whether its width is the width."""
    values = field_values(matrix, start, length)
    above = start + length
    # Bounded when something is actually there: the payload ends, another
    # detector owns the next bit, or the next bit is a field of its own.
    bounded = (
        above >= matrix.shape[1]
        or above in claimed_bits
        or rate[above] >= min_rate
    )
    return SignalHypothesis(
        start_bit=start,
        length=length,
        frames=int(matrix.shape[0]),
        rate=float(rate[start]),
        minimum=int(values.min()),
        maximum=int(values.max()),
        bounded=bool(bounded),
    )
