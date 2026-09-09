# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Group a trace by message and measure each one."""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from ..decode import CanFrame, iter_frames
from .bits import BitProfile, profile_bits
from .timing import TimingProfile, profile_timing


@dataclass
class MessageProfile:
    """Everything measured about one (bus, address)."""

    bus: int
    address: int
    count: int
    lengths: dict[int, int]
    width: int
    analysed: int
    bits: BitProfile
    timing: TimingProfile

    @property
    def key(self) -> tuple[int, int]:
        return (self.bus, self.address)

    @property
    def multi_length(self) -> bool:
        """True when the payload length is not fixed.

        Worth flagging rather than averaging over: a varying length usually
        means either a multiplexed message or two different senders sharing an
        address, and bit statistics pooled across both are meaningless.
        """
        return len(self.lengths) > 1

    def __str__(self) -> str:
        return f"bus {self.bus} 0x{self.address:03X}"


@dataclass
class TraceProfile:
    """Every message measured in one trace."""

    messages: dict[tuple[int, int], MessageProfile]
    frames: int
    duration_s: float

    def __len__(self) -> int:
        return len(self.messages)

    def __getitem__(self, key: tuple[int, int]) -> MessageProfile:
        return self.messages[key]

    def by_entropy(self) -> list[MessageProfile]:
        """Messages richest in payload entropy first -- where to start looking."""
        return sorted(
            self.messages.values(), key=lambda m: m.bits.payload_entropy, reverse=True
        )

    @property
    def total_entropy(self) -> float:
        return sum(m.bits.payload_entropy for m in self.messages.values())


def analyze_frames(frames: Iterable[CanFrame]) -> TraceProfile:
    """Measure every message in a stream of frames."""
    payloads: dict[tuple[int, int], list[bytes]] = defaultdict(list)
    stamps: dict[tuple[int, int], list[int]] = defaultdict(list)
    lengths: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    total = 0
    first = last = None

    for frame in frames:
        key = (frame.bus, frame.address)
        payloads[key].append(frame.data)
        stamps[key].append(frame.mono_ns)
        lengths[key][len(frame.data)] += 1
        total += 1
        if first is None:
            first = frame.mono_ns
        last = frame.mono_ns

    messages = {}
    for key, blobs in payloads.items():
        # Pool bit statistics only over the dominant payload length. Padding
        # short frames would manufacture constant bits that were never on the
        # wire, which is worse than analysing fewer frames.
        width = lengths[key].most_common(1)[0][0]
        same_width = [b for b in blobs if len(b) == width]
        messages[key] = MessageProfile(
            bus=key[0],
            address=key[1],
            count=len(blobs),
            lengths=dict(sorted(lengths[key].items())),
            width=width,
            analysed=len(same_width),
            bits=profile_bits(same_width, width),
            timing=profile_timing(stamps[key]),
        )

    duration = (last - first) / 1e9 if first is not None and last is not None else 0.0
    return TraceProfile(messages=messages, frames=total, duration_s=duration)


def analyze_segment(path: str, *, root: str, **kwargs) -> TraceProfile:
    """Decode one `rlog.zst` and measure it."""
    return analyze_frames(iter_frames(path, root=root, **kwargs))
