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

    @property
    def end_bit(self) -> int:
        return self.start_bit + self.length

    @property
    def period(self) -> int:
        """Frames before the counter repeats."""
        return (1 << self.length) // np.gcd(self.stride, 1 << self.length)

    def contains(self, other: CounterHypothesis) -> bool:
        return self.start_bit <= other.start_bit and other.end_bit <= self.end_bit

    def __str__(self) -> str:
        step = "" if self.stride == 1 else f" step {self.stride}"
        return f"{self.length}-bit counter @ bit {self.start_bit}{step} ({self.match_rate:.1%})"


def field_values(matrix: np.ndarray, start: int, length: int) -> np.ndarray:
    """Read bits [start, start+length) as an unsigned integer per frame.

    Weighting is little-endian over the matrix's own bit order, so with the
    Intel ordering the analyze layer produces by default this is exactly how a
    DBC Intel signal is laid out: the first bit of the slice is the LSB.
    """
    weights = (1 << np.arange(length)).astype(np.int64)
    return matrix[:, start : start + length].astype(np.int64) @ weights


def score_counter(values: np.ndarray, length: int) -> tuple[int, float]:
    """Dominant step and the fraction of frame pairs showing it."""
    if values.size < 2:
        return 0, 0.0
    modulus = 1 << length
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
    """
    frames, bits = matrix.shape
    if frames < 2:
        return []

    found: list[CounterHypothesis] = []
    for length in range(max_length, min_length - 1, -1):
        modulus = 1 << length
        for start in range(bits - length + 1):
            values = field_values(matrix, start, length)
            stride, rate = score_counter(values, length)
            if stride == 0 or rate < min_match:
                continue
            if np.gcd(stride, modulus) != 1:
                continue
            if np.unique(values).size < min_coverage * modulus:
                continue
            candidate = CounterHypothesis(start, length, stride, rate, frames)
            if any(accepted.contains(candidate) for accepted in found):
                continue
            found.append(candidate)
    return sorted(found, key=lambda c: c.start_bit)
