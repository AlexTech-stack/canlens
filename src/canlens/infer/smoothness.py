# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Value jumpiness: a signal does not leap across its own range.

Every other statistic in :mod:`canlens.infer` reads *how often a bit flips*.
This one reads *how far the field's value moves between consecutive frames*,
which is an orthogonal axis and the reason it was tried: a counter's bits can
flip as fast as a wheel speed's, so transition rate cannot always tell a real
signal from a region that is really two fields run together.

The test is scale-free. Normalise the absolute frame-to-frame change by the
range the field was actually observed to hold, and call a frame a **jump** when
that change exceeds `JUMP_FRACTION`. A coherent physical quantity moves in
small steps, so only a small share of its frames are jumps. A region straddling
two independent fields, or a fragment of noise, lurches across its range often.
A claim whose jump share exceeds `MAX_JUMP_SHARE` is dropped.

**What would falsify this.** If jumpiness carried no information beyond field
length or transition rate -- if a longer-field or a slower-field filter bought
the same precision at the same recall -- then this statistic would be a
re-description of one already in use. It is not; measured against opendbc over
eleven platform and DBC pairs (ten Volkswagen MQB platforms against
``vw_mqb.dbc``, plus a Tesla Model 3), three segments each, aggregated:

=========================  =========  =======  ====
filter                     precision  recall   F1
=========================  =========  =======  ====
none                            59%     62%   0.604
length >= 4 (control)           61%     53%   0.563
low-bit rate <= 0.2 (control)   67%     44%   0.528
**jump share <= 0.20**          **71%**  **58%**  **0.643**
=========================  =========  =======  ====

The two controls isolate it. A length floor raises precision from 59% to 61%
only by giving up recall, and *lowers* F1 at every threshold tried; a transition
rate floor does the same more sharply. The jump filter trades 4 points of recall
for 12 of precision and improves F1 by 0.04 -- while dropping roughly six times
more false claims than true ones (1057 extras against 175 hits on the sweep).

The two constants are not knife-edge. A 2D sweep of `JUMP_FRACTION` over
(0.05, 0.10, 0.20) against `MAX_JUMP_SHARE` over (0.10 .. 0.50) leaves F1 on a
plateau of 0.637 to 0.645 for every share at or below 0.30; the chosen corner
sits mid-plateau, well clear of the 0.604 unfiltered baseline. Values were
taken from the corpus, not guessed:

    0.10 fraction / 0.20 share  ->  P 0.714  R 0.584  F1 0.643

A move that *reduces* findings can be the correct one, because most false
positives look like extra findings -- so the way to read this result is that a
quarter of the signal claims were noise dressed as extra coverage.

**What this cannot do.** It is a filter, not a segmenter. It ranks and drops
whole claims; it does not split a region that is genuinely two fields, and it
does not recover a field lost below the rate floor. Measured separately, the
best interior split of a false claim improved smoothness by more than 0.2 on
only 12% of them, which is too weak to cut on, and a synthetic test of the
clean carry structure a counter has showed it is destroyed on real signals --
whose values are signed and offset around a midpoint, not natural-binary -- so
no splitter was built.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .counters import field_values
from .signals import SignalHypothesis

# A frame whose value moves by more than this fraction of the field's observed
# range is a jump. Chosen from the 2D sweep in the module docstring; F1 is flat
# between 0.05 and 0.20, so this is a plateau value rather than a peak.
JUMP_FRACTION = 0.10

# The share of jumping frames a field may have and still be called one signal.
# Above this it lurches across its range too often to be a single quantity.
MAX_JUMP_SHARE = 0.20


def jump_share(
    values: Sequence[int] | np.ndarray, fraction: float = JUMP_FRACTION
) -> float:
    """Share of frame pairs where the value moves more than `fraction` of range.

    The scalar reference. Iterates in plain Python so a reader can check the
    definition against :func:`v_jump_share`, the vectorised form that runs.

    A field that never changed has no range to move across; it returns 0.0
    rather than dividing by zero, though such a field is never a signal.
    """
    count = len(values)
    if count < 2:
        return 0.0
    lo = hi = int(values[0])
    for value in values:
        v = int(value)
        if v < lo:
            lo = v
        elif v > hi:
            hi = v
    span = hi - lo
    if span == 0:
        return 0.0
    threshold = fraction * span
    jumps = 0
    for i in range(1, count):
        if abs(int(values[i]) - int(values[i - 1])) > threshold:
            jumps += 1
    return jumps / (count - 1)


def v_jump_share(
    matrix: np.ndarray, start: int, length: int, fraction: float = JUMP_FRACTION
) -> float:
    """Vectorised :func:`jump_share` over one field of a bit matrix.

    `matrix` is the (frames, bits) array the signal was found in, so `start`
    and `length` are in its own bit order.
    """
    values = field_values(matrix, start, length)
    if values.size < 2:
        return 0.0
    lo = int(values.min())
    hi = int(values.max())
    span = hi - lo
    if span == 0:
        return 0.0
    jumps = np.abs(np.diff(values))
    return float((jumps > fraction * span).mean())


def filter_signals(
    matrix: np.ndarray,
    signals: Sequence[SignalHypothesis],
    *,
    fraction: float = JUMP_FRACTION,
    max_share: float = MAX_JUMP_SHARE,
) -> list[SignalHypothesis]:
    """Drop claims whose value lurches across its range too often to be one.

    `matrix` must be the same bit matrix `signals` were found in, in the same
    order, so the values read here are the values the detector segmented.
    """
    return [
        signal
        for signal in signals
        if v_jump_share(matrix, signal.start_bit, signal.length, fraction) <= max_share
    ]
