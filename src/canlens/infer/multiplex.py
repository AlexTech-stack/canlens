# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Finding a multiplexor: a field whose value decides what the rest means.

A multiplexed message reuses its payload bits for different signals depending
on a selector field. The Jeep Grand Cherokee's 0x3E0 carries its VIN that way:
byte 0 cycles 0, 1, 2 and bytes 1-7 hold a different slice of the ASCII string
under each value. Read without the selector such a message looks like noise;
read with it, every slice is constant.

That is the signature tested here. For a candidate selector, the frames are
grouped by its value and every other bit is asked how it behaves *within* each
group. A bit whose meaning depends on the selector is constant inside every
group while differing between groups, or is still in a good share of the
frames and moving in another good share. A bit that is one signal throughout
-- a counter, a wheel speed, a slowly changing state -- behaves the same in
every group, because the groups interleave in time and each one samples the
whole trace.

A multiplexor is a transmission schedule, not a vehicle state, and that is
the second half of the test: each selector value must recur at a regular
interval. The VIN's byte 0 goes 0, 1, 2, 0, 1, 2; a schedule that sends
layout 0 twice as often as the others goes 0, 1, 0, 2, 0, 1, 0, 2. A field
that holds one value while a signal is valid and another while it reads
0xFFFF, or the sign bits of a value that crosses zero, also sorts the frames
into groups whose contents differ -- and those were the bulk of what an
earlier version reported -- but nothing about *when* such a field changes is
regular. Regularity is also what makes the grouping robust against slow
signals: every group samples the whole trace, so a state that changed once
is not constant in any of them.

Two things are excluded from the evidence by construction. A checksum byte is
a function of everything else, so under a counter used as the selector it is
constant per group -- exactly the first signature -- and it is therefore
neither tried as a selector nor counted as a dependent bit. And a layout
consisting only of per-value constants is accepted only from a selector with
few values: a byte that merely duplicates another byte makes the second a
"function" of the first through 8 constant-per-group bits, and a signal with
fifty values is not what anyone means by a multiplexor.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .counters import field_values, score_counter

# Group sizes below this cannot show a bit moving; the value is set aside.
MIN_PER_VALUE = 8
# Share of the gaps between a value's visits that must equal the most common
# gap, and how far the longest gap may exceed it. A schedule is exactly
# regular; the allowances cover frames the logger dropped. A value whose
# visits stop for a long stretch is not on a schedule, however regular the
# visits are while they last -- that is a counter seen through a flag.
# Regularity needs at least two gaps, so three visits.
MIN_REGULARITY = 0.8
MAX_GAP_RATIO = 4
# Among candidates, the one with the fewest values wins unless another
# explains more than this much more of the payload's movement. A partition
# refined by an unrelated counter explains everything the true one does and
# the counter besides; a little extra is not a better description.
NEAR_BEST = 0.85
# At least a byte's worth of bits must change meaning with the selector. A
# signal that merely correlates with another -- a gear and a derived lamp --
# shows a few dependent bits, not a layout.
MIN_DEPENDENT_BITS = 8
# A bit "moves" inside a group when its transition rate there is at least
# this. Below it a rare state change that happened to land in the group does
# not count as evidence either way.
MOVING_RATE = 0.05
# ...unless it moves on a schedule of its own. A bit that repeats every two,
# three or four frames *within* a group is locked to a finer cycle than the
# selector -- the CRC of a static message under a 3-bit window over its
# 4-bit counter alternates strictly between two values -- and a signal does
# not do that. Such a bit is neither still nor moving.
LOCK_PERIODS = (2, 3, 4)
LOCKED = 0.95
# For the "still here, moving there" signature, each side must hold at least
# this share of the frames. Without it a signal with many values, most of them
# visited briefly, finds some group in which any slow bit sat still.
MIN_SIDE_SHARE = 0.25
# A layout made only of per-value constants (a table, like the VIN) is a
# weaker claim than one with a moving signal in some slice, so it must be
# wider -- a byte that merely follows another byte is not a layout -- and
# must not come from a selector that is a plain counter. A counter whose only
# "dependent" byte is constant per count is a checksum the checksum search
# could not prove on a static message, and there are a great many of those.
MIN_TABLE_BITS = 12
MAX_TABLE_VALUES = 16
# Frames left outside the retained values, as a share of all frames.
MAX_UNEXPLAINED = 0.05
# A selector with more values than this is some other kind of signal. With
# sixty values of ten frames each, any two period-locked signals satisfy the
# "still here, moving there" test.
MAX_VALUES = 32


@dataclass(frozen=True)
class MultiplexHypothesis:
    """A selector field, and which bits change meaning with it."""

    start_bit: int
    length: int
    values: tuple[int, ...]  # selector values, ascending
    frames_per_value: tuple[int, ...]
    dependent_bits: tuple[int, ...]  # bits whose behaviour depends on the value
    frames: int

    @property
    def end_bit(self) -> int:
        return self.start_bit + self.length

    @property
    def coverage(self) -> float:
        """Share of the trace's frames that carry one of the listed values."""
        return sum(self.frames_per_value) / self.frames if self.frames else 0.0

    def __str__(self) -> str:
        shown = ", ".join(str(v) for v in self.values[:8])
        more = f", … ({len(self.values)} values)" if len(self.values) > 8 else ""
        return (
            f"{self.length}-bit multiplexor @ bit {self.start_bit}: values {shown}{more}; "
            f"{len(self.dependent_bits)} dependent bits ({self.coverage:.1%} of frames)"
        )


def regular(mask: np.ndarray) -> bool:
    """Whether the frames flagged in the mask are visited at a steady interval.

    A visit is a run of consecutive flagged frames; what must be regular is
    the spacing between the starts of successive visits. Spacing between
    individual frames would not do: a value that sits still for hundreds of
    frames has a gap of one between nearly all of them and looks perfectly
    regular, and such values -- a validity flag, a sign -- are exactly what
    is to be rejected.
    """
    flags = mask.astype(np.int8)
    starts = np.flatnonzero(np.diff(flags) == 1) + 1
    if flags[0]:
        starts = np.concatenate(([0], starts))
    gaps = np.diff(starts)
    if gaps.size < 2:
        return False
    counts = np.bincount(gaps)
    dominant = int(counts.argmax())
    return (
        float(counts[dominant] / gaps.size) >= MIN_REGULARITY
        and int(gaps.max()) <= MAX_GAP_RATIO * dominant
    )


def plain_counter(values: np.ndarray, kept: np.ndarray) -> bool:
    """Whether the selector just counts through a full power-of-two range."""
    size = int(kept.size)
    if size < 8 or size & (size - 1) or int(kept.max() - kept.min()) + 1 != size:
        return False
    stride, rate = score_counter(values - int(kept.min()), size.bit_length() - 1)
    return stride == 1 and rate >= 0.95


def selector_candidates(bits: int) -> list[tuple[int, int]]:
    """(start, length) fields a selector is looked for in.

    Whole bytes, nibbles, and 2- and 3-bit fields at either end of a byte: the
    positions a DBC's multiplexor signals actually occupy. A wider search
    would mostly rediscover the same partitions through other windows.
    """
    out: list[tuple[int, int]] = []
    for byte in range(bits // 8):
        base = byte * 8
        out.append((base, 8))
        out += [(base, 4), (base + 4, 4)]
        out += [(base, 3), (base + 5, 3), (base, 2), (base + 6, 2)]
    return out


def dependent_bits(
    matrix: np.ndarray,
    groups: list[np.ndarray],
    exclude: set[int],
    scored_exclude: set[int] | frozenset[int] = frozenset(),
) -> tuple[np.ndarray, bool, float]:
    """Bit positions whose behaviour depends on which group a frame is in.

    `groups` are boolean masks over the frames, one per selector value.
    `exclude` holds bits never counted as dependent: the selector itself and
    any checksum; `scored_exclude` the bits left out of the score (the
    checksums only).

    A bit qualifies if it is constant inside every group (and so, being
    non-constant overall, differs between them), or constant in groups holding
    at least MIN_SIDE_SHARE of the frames while moving in groups holding at
    least as many. The second result says whether any bit met the second
    signature, i.e. whether some slice actually carries a moving signal.

    The third is the score the candidates are ranked on: how much of the
    payload's movement the grouping removes, summed over bits as the overall
    transition rate less the frame-weighted within-group rates. Counting
    dependent bits would not do -- a coarser cut that merges two layouts can
    tally one bit more through the "still here, moving there" signature while
    leaving most of the movement unexplained, and a finer cut that also
    splits on an unrelated bit explains nothing extra but claims more values.
    Rates rather than counts, because a group sees a subsequence: splitting a
    group on an unrelated bit leaves the rate of a fast signal unchanged and
    raises that of a slow one, so a needless refinement never scores higher.
    """
    frames, bits = matrix.shape
    overall_constant = matrix.min(axis=0) == matrix.max(axis=0)
    overall_rate = (np.diff(matrix, axis=0) != 0).mean(axis=0)
    constant_in = np.zeros(bits, dtype=np.int64)
    still_frames = np.zeros(bits, dtype=np.int64)
    moving_frames = np.zeros(bits, dtype=np.int64)
    within_rate = np.zeros(bits)
    for mask in groups:
        sub = matrix[mask]
        constant = sub.min(axis=0) == sub.max(axis=0)
        constant_in += constant
        still_frames += constant * sub.shape[0]
        rate = (np.diff(sub, axis=0) != 0).mean(axis=0)
        locked = np.zeros(bits, dtype=bool)
        for period in LOCK_PERIODS:
            if sub.shape[0] > 2 * period:
                locked |= (sub[:-period] == sub[period:]).mean(axis=0) >= LOCKED
        moving_frames += ((rate >= MOVING_RATE) & ~locked) * sub.shape[0]
        within_rate += rate * (sub.shape[0] / frames)
    side = MIN_SIDE_SHARE * frames
    table = constant_in == len(groups)
    mixed = (still_frames >= side) & (moving_frames >= side)
    counted = np.ones(bits, dtype=bool)
    counted[sorted(b for b in exclude if b < bits)] = False
    dependent = ~overall_constant & (table | mixed) & counted
    # Scored over every bit the selector is allowed to explain, its own
    # included: two windows that cut the frames identically then tie, and the
    # tie-break, not which bits each window happened to cover, decides.
    scored = np.ones(bits, dtype=bool)
    scored[sorted(b for b in scored_exclude if b < bits)] = False
    explained = float(np.clip(overall_rate - within_rate, 0, None)[scored].sum())
    return np.flatnonzero(dependent), bool(np.any(dependent & mixed)), explained


def find_multiplexor(
    matrix: np.ndarray,
    *,
    skip_bytes: set[int] | frozenset[int] = frozenset(),
    counter_bits: set[int] | frozenset[int] = frozenset(),
    min_dependent_bits: int = MIN_DEPENDENT_BITS,
) -> MultiplexHypothesis | None:
    """The one selector that best explains the payload, if any does.

    `skip_bytes` are positions a checksum already accounts for. Such a byte
    is never tried as a selector -- it takes many values and everything
    "depends" on it -- and never counted as dependent, because under a
    counter used as the selector it is constant per group. `counter_bits`
    are the bits of any counter found: a selector may well overlap one (the
    VIN's 0, 1, 2 is a counter modulo 3), but a counter's own movement is
    explained by the frame index and earns no candidate credit.

    Candidates are scored by how much of the payload's movement grouping on
    them explains (see `dependent_bits`). The winner is the candidate with
    the fewest values among those within NEAR_BEST of the top score, then
    the widest field, then the earliest.
    """
    frames, bits = matrix.shape
    if frames < 2 * MIN_PER_VALUE:
        return None
    if matrix.dtype != np.uint8:
        matrix = matrix.astype(np.uint8)
    wide = matrix.astype(np.int64)
    skipped_bits = {b for byte in skip_bytes for b in range(byte * 8, byte * 8 + 8)}
    unscored = skipped_bits | set(counter_bits)

    found: list[tuple[float, MultiplexHypothesis]] = []
    seen_partitions: set[bytes] = set()
    # Widest windows first, so that of several windows cutting the frames the
    # same way -- a byte holding 0, 1, 2 and its low two bits -- the one kept
    # is the whole field a DBC would name, not the narrowest slice of it.
    for start, length in sorted(selector_candidates(bits), key=lambda c: (-c[1], c[0])):
        if start // 8 in skip_bytes:
            continue
        values = field_values(wide, start, length)
        distinct, counts = np.unique(values, return_counts=True)
        if distinct.size < 2 or distinct.size > MAX_VALUES:
            continue
        keep = distinct[counts >= MIN_PER_VALUE]
        if keep.size < 2:
            continue
        masks = [values == v for v in keep]
        unexplained = frames - sum(int(m.sum()) for m in masks)
        if unexplained > MAX_UNEXPLAINED * frames:
            continue
        if not all(regular(m) for m in masks):
            continue
        labels = np.full(frames, -1, dtype=np.int16)
        for i, m in enumerate(masks):
            labels[m] = i
        signature = labels.tobytes()
        if signature in seen_partitions:
            continue
        seen_partitions.add(signature)

        dependent, has_moving, explained = dependent_bits(
            matrix, masks, skipped_bits | set(range(start, start + length)), unscored
        )
        if dependent.size < min_dependent_bits:
            continue
        if not has_moving and (
            dependent.size < MIN_TABLE_BITS
            or keep.size > MAX_TABLE_VALUES
            or plain_counter(values, keep)
        ):
            continue
        hypothesis = MultiplexHypothesis(
            start_bit=start,
            length=length,
            values=tuple(int(v) for v in keep),
            frames_per_value=tuple(int(m.sum()) for m in masks),
            dependent_bits=tuple(int(b) for b in dependent),
            frames=frames,
        )
        found.append((explained, hypothesis))
    if not found:
        return None
    top = max(explained for explained, _ in found)
    near = [h for explained, h in found if explained >= NEAR_BEST * top]
    return min(near, key=lambda h: (len(h.values), -h.length, h.start_bit))
