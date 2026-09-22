# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Hypotheses one segment cannot check, settled by pooling several.

:mod:`canlens.corroborate.consensus` pools *conclusions*: it asks how many
segments independently reached the same finding. That works when each segment
could reach the finding on its own. Some cannot.

The case this module exists for is AUTOSAR Profile 22's Data ID list. Sixteen
bytes are never transmitted and have to be solved for, and
:func:`canlens.infer.checksums.enough_evidence` refuses the claim unless the
message shows more distinct payloads than the secret has bytes, with margin.
A static message carrying nothing but its counter shows exactly sixteen --
sixteen unknowns fitted to sixteen equations, a fit with nothing left over and
therefore no evidence at all. Volkswagen's MQB bus is full of these: on a Golf
Mk7 segment, every checksum `vw_mqb.dbc` names that canlens missed had exactly
sixteen distinct payloads.

Pooling conclusions cannot rescue that. Every segment of such a message shows
the *same* sixteen payloads and so solves the *same* sixteen bytes, and sixteen
segments agreeing on a fit that was unfalsifiable in each of them is still
unfalsifiable. What is needed is more *equations*, which means pooling the
payloads themselves and counting the union: segments whose content differs
contribute new constraints, segments that repeat each other contribute none,
and the gate can tell the difference.

That is the whole idea here. Decode several segments of a platform, gather one
message's payloads across all of them, reduce to distinct contents, and put the
union in front of the same detector and the same gate the single-segment path
uses. Nothing is relaxed; the evidence is simply larger.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..infer.checksums import ChecksumHypothesis, distinct_contents, enough_evidence
from ..infer.profiles import find_p22
from .consensus import device_of

if TYPE_CHECKING:
    from ..decode import FrameSet

# A pooled claim is worth no more than the independence behind it. Sixteen
# unknowns solved from one car's data are that car's, and the point of the
# corpus is to say what the platform does.
MIN_DEVICES = 2


@dataclass(frozen=True)
class Pool:
    """One message's payloads gathered across several segments."""

    bus: int
    address: int
    width: int
    matrix: np.ndarray  # (n, width) of distinct contents, first occurrence order
    ordered: np.ndarray  # one segment's frames, still in time order
    frames: int  # frames pooled before reduction
    segments: int
    devices: int
    per_segment: tuple[int, ...]  # distinct contents each segment contributed

    @property
    def distinct(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def new_from_pooling(self) -> int:
        """Distinct contents beyond what the richest single segment offered."""
        return self.distinct - max(self.per_segment, default=0)

    def __str__(self) -> str:
        return (
            f"bus {self.bus} 0x{self.address:03X}: {self.distinct} distinct payloads from "
            f"{self.frames} frames, {self.segments} segments, {self.devices} devices"
        )


@dataclass(frozen=True)
class PooledFinding:
    """A hypothesis the pool supports and no single segment could."""

    pool: Pool
    hypothesis: ChecksumHypothesis
    alone: int  # distinct contents the best single segment had

    @property
    def settled_by_pooling(self) -> bool:
        """Whether one segment's evidence would have been refused on its own."""
        return self.alone < 16 + 8

    def __str__(self) -> str:
        return (
            f"bus {self.pool.bus} 0x{self.pool.address:03X}: {self.hypothesis.algorithm} "
            f"@ byte {self.hypothesis.byte_index} ({self.hypothesis.match_rate:.1%} of "
            f"{self.hypothesis.frames} frames; {self.pool.distinct} distinct payloads "
            f"pooled from {self.pool.segments} segments, {self.pool.devices} devices)"
        )


def unique_rows(matrix: np.ndarray) -> np.ndarray:
    """Distinct rows, in order of first appearance.

    `np.unique` sorts, which would scramble the frame order the counter is
    read from. Order is kept because the Profile 22 search groups frames by
    their counter value and a sorted matrix makes those groups meaningless.
    """
    if matrix.shape[0] == 0:
        return matrix
    contiguous = np.ascontiguousarray(matrix)
    view = contiguous.view(np.dtype((np.void, contiguous.shape[1] * contiguous.dtype.itemsize)))
    _, index = np.unique(view.ravel(), return_index=True)
    return matrix[np.sort(index)]


def pool_messages(
    paths: Sequence[str],
    keys: set[tuple[int, int]] | frozenset[tuple[int, int]],
    *,
    root: str,
    min_frames: int = 32,
    load: Callable[..., FrameSet] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[tuple[int, int], Pool]:
    """Gather several messages' distinct payloads across several segments.

    One pass over the segments, not one per message. Pooling forty segments
    for each of two hundred messages separately would be eight thousand cache
    reads for the forty the work actually needs.

    Segments whose payload width disagrees are skipped rather than padded:
    the same identifier at a different length is a different message, and
    mixing them would manufacture contents that were never on the wire.
    """
    from ..decode import load_frames

    loader = load or load_frames
    blocks: dict[tuple[int, int], list[np.ndarray]] = {}
    per_segment: dict[tuple[int, int], list[int]] = {}
    devices: dict[tuple[int, int], set[str]] = {}
    widths: dict[tuple[int, int], int] = {}
    counts: dict[tuple[int, int], int] = {}

    for done, path in enumerate(paths, start=1):
        if progress is not None:
            progress(done, len(paths))
        try:
            trace = loader(path, root=root)
        except (OSError, ValueError):
            continue
        for message in trace.group(min_frames=min_frames):
            key = (message.bus, message.address)
            if key not in keys:
                continue
            if widths.setdefault(key, message.width) != message.width:
                continue
            block = message.bytes_matrix()
            blocks.setdefault(key, []).append(block)
            per_segment.setdefault(key, []).append(int(unique_rows(block).shape[0]))
            devices.setdefault(key, set()).add(device_of(path))
            counts[key] = counts.get(key, 0) + int(block.shape[0])

    return {
        key: Pool(
            bus=key[0],
            address=key[1],
            width=widths[key],
            matrix=unique_rows(np.vstack(found)),
            ordered=found[0],
            frames=counts[key],
            segments=len(found),
            devices=len(devices[key]),
            per_segment=tuple(per_segment[key]),
        )
        for key, found in blocks.items()
    }


def pool_message(
    paths: Sequence[str],
    key: tuple[int, int],
    *,
    root: str,
    min_frames: int = 32,
    load: Callable[..., FrameSet] | None = None,
) -> Pool | None:
    """Gather one message's distinct payloads across several segments."""
    pools = pool_messages(paths, {key}, root=root, min_frames=min_frames, load=load)
    return pools.get(key)


def pool_frames(
    paths: Sequence[str],
    keys: set[tuple[int, int]] | frozenset[tuple[int, int]],
    *,
    root: str,
    min_frames: int = 32,
    load: Callable[..., FrameSet] | None = None,
) -> dict[tuple[int, int], tuple[np.ndarray, int]]:
    """One message's frames across several segments, still in order.

    The deduplication `pool_messages` does is right for a secret that has to
    be constrained by distinct payloads, and wrong for a signal: transition
    rates are read from consecutive frames, and removing repeats would turn a
    quiet field into a busy one. So the segments are simply laid end to end.
    Each join adds one transition that never happened, which against tens of
    thousands of frames changes no rate that matters.

    Returns the matrix and the number of devices behind it.
    """
    from ..decode import load_frames

    loader = load or load_frames
    blocks: dict[tuple[int, int], list[np.ndarray]] = {}
    devices: dict[tuple[int, int], set[str]] = {}
    widths: dict[tuple[int, int], int] = {}
    for path in paths:
        try:
            trace = loader(path, root=root)
        except (OSError, ValueError):
            continue
        for message in trace.group(min_frames=min_frames):
            key = (message.bus, message.address)
            if key not in keys or widths.setdefault(key, message.width) != message.width:
                continue
            blocks.setdefault(key, []).append(message.bytes_matrix())
            devices.setdefault(key, set()).add(device_of(path))
    return {
        key: (np.vstack(parts), len(devices[key])) for key, parts in blocks.items()
    }


def signals_across(
    platform: str,
    *,
    root: str,
    limit: int | None = None,
    min_devices: int = MIN_DEVICES,
    progress: Callable[[int, int], None] | None = None,
) -> dict[tuple[int, int], list]:
    """Signal boundaries read from every segment of a platform at once.

    A field's high bits only move when the value grows large enough to reach
    them, so one drive shows one drive's worth of range. Twenty segments of
    the Volkswagen group and Rivian, scored against opendbc, take signal
    detection from 68% precision and 58% recall to 70% and 60% -- and, more
    usefully, from 360 correct claims to 460, because fields that never moved
    in a single drive move somewhere across twenty.
    """
    from ..corpus import Manifest, inventory
    from ..infer import infer_cached
    from ..infer.signals import find_signals

    held = inventory(root, Manifest.load(f"{root}/database.json")).get(platform)
    paths = list(held.paths if held else [])[:limit]
    if len(paths) < min_devices:
        return {}

    claimed: dict[tuple[int, int], set[int]] = {}
    for path in paths:
        for message in infer_cached(path, root=root):
            spoken = claimed.setdefault(message.key, set())
            for counter in message.counters:
                spoken.update(range(counter.start_bit, counter.end_bit))
            for check in message.checksums:
                spoken.update(range(check.start_bit, check.start_bit + check.length))
            for crc in message.crc16s:
                spoken.update(range(crc.start_byte * 8, (crc.start_byte + crc.nbytes) * 8))
            if message.multiplexor is not None:
                spoken.update(
                    range(message.multiplexor.start_bit, message.multiplexor.end_bit)
                )

    from ..analyze.bits import BitOrder, bit_matrix_from_bytes

    pooled = pool_frames(paths, set(claimed), root=root, load=None)
    out = {}
    for done, (key, (matrix, devices)) in enumerate(sorted(pooled.items()), start=1):
        if progress is not None:
            progress(done, len(pooled))
        if devices < min_devices:
            continue
        found = find_signals(
            bit_matrix_from_bytes(matrix, BitOrder.INTEL),
            claimed_bits=claimed.get(key, set()),
        )
        if found:
            out[key] = found
    return out


def solve_p22(pool: Pool, *, min_match: float = 0.99, min_devices: int = MIN_DEVICES) -> list[ChecksumHypothesis]:
    """Run the Profile 22 list search over a pool's distinct payloads.

    The same detector and the same evidence gate as the single-segment path.
    What changes is only how much data is in front of them -- which is the
    point, since the gate is a statement about evidence and pooling is how
    more of it is obtained.

    The counter is located on `pool.ordered` rather than on the pool itself.
    Reducing to distinct contents is what supplies the extra equations, and it
    also destroys the one thing a counter is: a sequence. On a Golf's
    GRA_ACC_01 the deduplicated pool of 125 payloads has its counter nibble
    running 4, 5, 6, ... 15, 0, 1, 2, 3, 7, 8, 13, 14 -- no stride survives
    and `find_counters` rightly reports nothing, which silently left the list
    search with no counter to group by. Each deduplicated row still carries
    its own counter value, so grouping works once the field is known; only
    finding the field needs frames in the order they arrived.
    """
    if pool.devices < min_devices or pool.width < 2:
        return []
    from ..analyze.bits import BitOrder, bit_matrix_from_bytes
    from ..infer.counters import find_counters

    counters = find_counters(bit_matrix_from_bytes(pool.ordered, BitOrder.INTEL))
    return find_p22(pool.matrix, counters, min_match=min_match)


def corroborate_p22(
    platform: str,
    *,
    root: str,
    limit: int | None = None,
    min_devices: int = MIN_DEVICES,
    progress: Callable[[int, int], None] | None = None,
) -> list[PooledFinding]:
    """Every Profile 22 list a platform's pooled segments support.

    Only messages that no single segment could settle are reported: a finding
    the ordinary path already makes is not news, and repeating it here would
    make the pooled numbers look better than the pooling actually was.
    """
    from ..corpus import Manifest, inventory
    from ..infer import infer_cached

    held = inventory(root, Manifest.load(f"{root}/database.json")).get(platform)
    paths = list(held.paths if held else [])[:limit]
    if len(paths) < min_devices:
        return []

    # Any message at least one segment failed to explain. Messages another
    # segment did explain are kept rather than filtered out: the pooled answer
    # is the platform's, and whether a single lucky drive could also have
    # reached it is what `PooledFinding.settled_by_pooling` records.
    wanted = {
        message.key
        for path in paths
        for message in infer_cached(path, root=root)
        if not message.checksums
    }

    pools = pool_messages(paths, wanted, root=root, progress=progress)
    found = []
    for key in sorted(pools):
        pool = pools[key]
        if pool.devices < min_devices:
            continue
        for hypothesis in solve_p22(pool, min_devices=min_devices):
            found.append(
                PooledFinding(pool=pool, hypothesis=hypothesis, alone=max(pool.per_segment))
            )
    return found


def would_pass_alone(matrix: np.ndarray, byte_index: int, secret_bytes: int = 16) -> bool:
    """Whether one segment's evidence would have cleared the gate by itself."""
    return enough_evidence(matrix, [byte_index], secret_bytes)


def evidence_growth(pool: Pool, byte_index: int) -> tuple[int, int]:
    """(distinct contents alone, distinct contents pooled) for one CRC byte."""
    return max(pool.per_segment, default=0), distinct_contents(pool.matrix, [byte_index])
