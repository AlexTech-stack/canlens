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

**The rate floor, and what it cannot see.** Every bit is held to the same
minimum transition rate, and that single number decides what the detector is
even allowed to look at. It is worth being precise about who that excludes,
because it is not the bits one would guess. Measured over seven platforms,
taking each reference signal whose bits move at all:

===============  =================  ===============
declared width   median LSB rate    moves >= 2% of frames
===============  =================  ===============
1 bit            0.004               27%
2 bits           0.010               34%
3 bits           0.004               14%
4 bits           1.000               84%
5-8 bits         0.257               79%
9 bits and up    0.285               89%
===============  =================  ===============

Narrow fields barely move. An ignition switch does not cycle its states, and a
mode selector sits on one value for a whole drive; a bit that flips fast and
narrow is almost always the bottom of a wider number instead. The 4-bit row is
the exception that proves it -- those are alive counters, which the counter
detector claims long before this one runs.

Halving the floor to 0.01 lifted recall from 49% to 54% with no loss of
precision and nothing given up -- 37 signals gained, none lost. But the gain
does not come from the narrow fields the table is about. Of those 37, only 3
are 3 bits or narrower; 28 are 8 bits or wider. What the old floor was really
cutting off was the *top* of wide fields, where the rate decay runs out: bit 6
of a byte turns over a few times in a drive and bit 9 of a 16-bit field fewer,
so a strict floor truncated the claim below the reference width and scored it
as a miss. Lowering the floor lets those fields run to their real end.

Narrow fields stayed where they were, and the reason is structural rather than
a matter of threshold. A 2-bit enum's high bit moves half as often as its low
bit, so any floor that admits the low bit still drops the high one, leaving a
single bit that is a flag by definition. Moving the floor cannot fix that; it
can only move which field it happens to.

**What still cannot be found, and why.** Below that floor the detector does not
merely lose accuracy; it has nothing to work with. Rate segmentation needs a
rate profile, and a 2-bit enum that changes four times in a drive has none. Two
ways of reading such bits were tried against opendbc and both failed:

*Changes that happen in the same frame.* If adjacent slow bits belong to one
field they should move together. They do not. Over 1069 adjacent slow pairs,
bits within one signal share a transition 46% of the time and bits across a
boundary share one 59% of the time -- the test is not weak, it points the wrong
way. Inside an enum only the low bit moves on most steps, while two unrelated
neighbours both react to the ignition, so co-occurrence measures the driver's
actions rather than the layout.

*Refusing to cut below a fast narrow piece.* If a fast 2-bit field is really
the bottom of a wider one, suppressing the cut that created it should recover
the wider field. It recovers nothing: the scores are identical to three
figures at every threshold tried.

What remains is a limit worth stating plainly. Of claims 2 or 3 bits wide, 5%
match a reference signal. A slow narrow field and the flag beside it produce
the same evidence, and nothing in a trace separates them.

Scored against opendbc over eight platforms with the real scorer, counting a
claim right when it gets the start bit and either the exact width or a narrower
one whose omitted bits never moved: 50% precision, 54% recall.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .counters import field_values

# A bit must move at least this often to be part of a signal. Below it there is
# no evidence of a field: a bit that flips twice in six thousand frames is a
# rare state change, which `corroborate` reports as a rare bit rather than
# something to group with its neighbours.
#
# The floor is low because narrow fields are slow (see "The rate floor" above).
# Swept against opendbc over eight platforms: 0.02 scores 50% precision and 49%
# recall, 0.01 scores 50% and 54%, and going further to 0.005 buys one more
# point of recall for two of precision. 0.01 is where recall stops being free.
MIN_RATE = 0.01

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
