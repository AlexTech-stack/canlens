# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Finding sequence counters in a payload.

A counter is a small field that advances by a fixed step every message and
wraps. That is a strong, checkable claim: extract the field, difference it
modulo its width, and see whether one step dominates. Fields that merely look
busy do not qualify.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CounterHypothesis:
    """A field that advances by a constant step each frame."""

    start_bit: int
    length: int
    stride: int
    match_rate: float
    frames: int
    # The value the counter wraps at. 0 means the natural 2**length. AUTOSAR
    # E2E alive counters run 0..14 in four bits -- 0x0F is reserved as the
    # invalid value, [PRS_E2E_00504] -- so their modulus is 15, not 16.
    modulus: int = 0

    @property
    def wraps_at(self) -> int:
        return self.modulus or (1 << self.length)

    @property
    def end_bit(self) -> int:
        return self.start_bit + self.length

    @property
    def period(self) -> int:
        """Frames before the counter repeats."""
        return self.wraps_at // int(np.gcd(self.stride, self.wraps_at))

    def contains(self, other: CounterHypothesis) -> bool:
        return self.start_bit <= other.start_bit and other.end_bit <= self.end_bit

    def __str__(self) -> str:
        step = "" if self.stride == 1 else f" step {self.stride}"
        wrap = "" if not self.modulus else f" mod {self.modulus}"
        return (
            f"{self.length}-bit counter @ bit {self.start_bit}{step}{wrap} "
            f"({self.match_rate:.1%})"
        )


def field_values(matrix: np.ndarray, start: int, length: int) -> np.ndarray:
    """Read bits [start, start+length) as an unsigned integer per frame.

    Weighting is little-endian over the matrix's own bit order, so with the
    Intel ordering the analyze layer produces by default this is exactly how a
    DBC Intel signal is laid out: the first bit of the slice is the LSB.
    """
    weights = (1 << np.arange(length)).astype(np.int64)
    window = matrix[:, start : start + length]
    if window.dtype != np.int64:
        window = window.astype(np.int64)
    return window @ weights


def score_counter(
    values: np.ndarray, length: int, modulus: int | None = None
) -> tuple[int, float]:
    """Dominant step and the fraction of frame pairs showing it.

    `modulus` defaults to 2**length. A counter that wraps earlier -- the
    AUTOSAR 0..14 alive counter is the common case -- shows one odd step per
    cycle under the natural modulus, which is exactly enough to fail a 95%
    match: fourteen good steps in fifteen is 93.3%. Scored under its true
    modulus it is perfect.
    """
    if values.size < 2:
        return 0, 0.0
    modulus = modulus or (1 << length)
    deltas = np.diff(values) % modulus
    counts = np.bincount(deltas, minlength=modulus)
    counts[0] = 0  # a field that never moves is not a counter
    stride = int(counts.argmax())
    return stride, float(counts[stride] / deltas.size)


def find_counters(
    matrix: np.ndarray,
    *,
    min_length: int = 2,
    max_length: int = 8,
    min_match: float = 0.95,
    min_coverage: float = 0.5,
    min_lsb_rate: float | None = None,
) -> list[CounterHypothesis]:
    """Scan every field position for one that advances by a constant step.

    Two filters keep this from reporting arithmetic coincidences as findings.

    The step must be **coprime with the field width**, so the counter walks its
    entire range. Without this, a single toggling bit at the top of an n-bit
    window looks like a perfect counter of step 2^(n-1): the difference really
    is constant, but the field only ever holds two values. Every even step on a
    power-of-two modulus is some version of that.

    The field must also have been **seen holding at least `min_coverage` of its
    possible values**, which is the empirical form of the same requirement and
    catches a short trace that has not yet disproved a bad candidate.

    Longer fields win: the low four bits of an eight-bit counter are themselves
    a perfectly good four-bit counter, so a candidate contained within an
    already-accepted longer one is dropped rather than reported twice.

    Start positions are prefiltered on the least significant bit, and the
    bound is exact rather than loose. The step must be coprime with a
    power-of-two width, so it is odd, so the field's lowest bit flips on
    *every* increment -- and a counter accepted at `min_match` must therefore
    have an LSB transition rate of at least `min_match`. The default follows
    that: `min_lsb_rate` is `min_match` unless overridden. An earlier cutoff
    of 0.40 was merely "noisy", admitted 2729 starts on a CAN FD segment where
    0.95 admits 193, and found exactly the same 163 counters four times slower.
    """
    frames, bits = matrix.shape
    if frames < 2:
        return []

    lsb_rates = (np.diff(matrix.astype(np.int8), axis=0) != 0).mean(axis=0)

    # Converted once: field_values used to cast its slice on every call.
    wide = matrix.astype(np.int64)

    # Batching the surviving positions of one length into a single
    # sliding-window matmul was tried and measured slower (3.18s against
    # 2.65s) even with the prefilter applied first. The cause was not pinned
    # down; with the exact LSB bound above leaving under two hundred positions
    # per segment, the per-position loop is cheap enough that it no longer
    # matters.

    found: list[CounterHypothesis] = []
    for length in range(max_length, min_length - 1, -1):
        natural = 1 << length
        # The exact LSB bound, corrected for an early wrap: a counter that
        # wraps below 2**length skips one LSB flip per cycle (14 -> 0 leaves
        # the low bit at 0), so the bound is min_match scaled by the shortest
        # cycle a field of this width could have and still cover half its
        # range. Without this the 0..14 alive counter fails the prefilter at
        # 93.3% before it is ever scored.
        shortest = max(2, int(min_coverage * natural))
        threshold = (min_lsb_rate if min_lsb_rate is not None else min_match) * (
            1 - 1 / shortest
        )
        positions = np.flatnonzero(lsb_rates >= threshold)
        for start in positions[positions + length <= bits]:
            values = field_values(wide, int(start), length)
            hypothesis = None
            for modulus in _moduli(values, natural):
                stride, rate = score_counter(values, length, modulus)
                if stride == 0 or rate < min_match or np.gcd(stride, modulus) != 1:
                    continue
                if np.unique(values).size < min_coverage * modulus:
                    continue
                hypothesis = CounterHypothesis(
                    int(start), length, stride, rate, frames,
                    modulus=0 if modulus == natural else modulus,
                )
                break
            if hypothesis is None or any(a.contains(hypothesis) for a in found):
                continue
            found.append(hypothesis)
    return sorted(found, key=lambda c: c.start_bit)


def _moduli(values: np.ndarray, natural: int) -> list[int]:
    """The natural modulus, then the early wrap the data itself suggests.

    A field that counts 0..N-1 for N below 2**length never shows a value of N
    or more, so its observed maximum plus one is the only other wrap worth
    testing. Trying every modulus would let a noisy field pick whichever one
    happens to fit best.

    An early wrap is only admitted when it uses the field's top bit, i.e.
    N > 2**(length-1). Otherwise a 4-bit 0..14 counter under two constant
    zero bits fits a 6-bit window mod 15 perfectly, and the longer window wins
    the containment rule -- reporting constant padding as part of the counter.
    A counter that never uses its top bit is a shorter counter.
    """
    observed = int(values.max()) + 1 if values.size else natural
    if observed >= natural or observed <= natural // 2:
        return [natural]
    return [natural, observed]
