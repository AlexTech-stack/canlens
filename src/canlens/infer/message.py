# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Per-message inference: what the measured bits actually are."""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from ..analyze.bits import BitOrder, BitProfile, bit_matrix, profile_bits
from ..decode import CanFrame, iter_frames
from .checksums import ChecksumHypothesis, find_checksums
from .counters import CounterHypothesis, find_counters

# A checksum byte is indistinguishable from noise by construction, so only
# bytes that move like noise are worth testing. This screen is what keeps a
# whole-segment sweep affordable.
#
# Set below a byte-wide counter's mean transition rate, which is ~0.249: the
# bits halve in rate from the LSB, so 1 + .5 + .25 + ... averages just under a
# quarter. Testing a counter byte and rejecting it costs almost nothing, while
# skipping a real checksum costs a finding, so the cutoff sits under that.
CHECKSUM_BYTE_MIN_RATE = 0.20


@dataclass
class MessageInference:
    """Hypotheses about one message, each already checked against the trace."""

    bus: int
    address: int
    width: int
    frames: int
    bits: BitProfile
    counters: list[CounterHypothesis] = field(default_factory=list)
    checksums: list[ChecksumHypothesis] = field(default_factory=list)

    @property
    def key(self) -> tuple[int, int]:
        return (self.bus, self.address)

    @property
    def found_anything(self) -> bool:
        return bool(self.counters or self.checksums)

    def __str__(self) -> str:
        return f"bus {self.bus} 0x{self.address:03X}"


def checksum_candidate_bytes(bits: BitProfile, *, min_rate: float = CHECKSUM_BYTE_MIN_RATE) -> list[int]:
    """Byte positions whose bits move enough to be worth testing."""
    return [
        index
        for index in range(bits.width)
        if float(np.mean(bits.rates[index * 8 : (index + 1) * 8])) >= min_rate
    ]


def infer_message(
    payloads: Sequence[bytes],
    *,
    bus: int,
    address: int,
    order: BitOrder = BitOrder.INTEL,
    **kwargs,
) -> MessageInference:
    """Look for counters and checksums in one message's payloads."""
    width = len(payloads[0]) if payloads else 0
    bits = profile_bits(list(payloads), width, order)
    matrix = bit_matrix(list(payloads), width, order)
    return MessageInference(
        bus=bus,
        address=address,
        width=width,
        frames=len(payloads),
        bits=bits,
        counters=find_counters(matrix, **kwargs.get("counter_options", {})),
        checksums=find_checksums(
            payloads,
            address,
            candidate_bytes=checksum_candidate_bytes(bits),
            **kwargs.get("checksum_options", {}),
        ),
    )


def infer_frames(
    frames: Iterable[CanFrame], *, order: BitOrder = BitOrder.INTEL, min_frames: int = 32
) -> list[MessageInference]:
    """Group a frame stream by message and infer each one.

    Messages seen fewer than `min_frames` times are skipped: a counter or a
    checksum claimed from a handful of samples is noise dressed as a finding.
    """
    payloads: dict[tuple[int, int], list[bytes]] = defaultdict(list)
    for frame in frames:
        payloads[(frame.bus, frame.address)].append(frame.data)

    results = []
    for (bus, address), blobs in payloads.items():
        if len(blobs) < min_frames:
            continue
        # Pool only the dominant payload length, for the same reason analyze does.
        width = Counter(len(b) for b in blobs).most_common(1)[0][0]
        same = [b for b in blobs if len(b) == width]
        if len(same) < min_frames:
            continue
        results.append(infer_message(same, bus=bus, address=address, order=order))
    return sorted(results, key=lambda m: (m.bus, m.address))


def infer_segment(
    path: str, *, root: str, order: BitOrder = BitOrder.INTEL, **kwargs
) -> list[MessageInference]:
    """Decode one segment and infer every message in it."""
    return infer_frames(iter_frames(path, root=root), order=order, **kwargs)
