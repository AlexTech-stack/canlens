# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""openpilot `rlog.zst` -> normalised CAN frames.

An rlog is a concatenation of capnp `Event` messages. Only the `can` events
matter here; each carries a list of `CanData` and the event's `logMonoTime`,
which is the only timestamp available -- individual frames within one event
are not separately stamped.

**Echoes.** `CanData.src` is not simply a bus number. Values >= 128 mark a
frame the device put on the wire itself, on bus `src - 128`, as it relays
traffic between the car and the ADAS camera. Measured over corpus segments,
bus 0 and bus 2 are relayed into each other and the echoes reproduce the
source bus's address set exactly -- roughly 30% of all frames in a segment are
these duplicates. Counting them as independent observations would inflate
every frequency, entropy and corroboration statistic built on top, so they are
dropped by default and must be asked for explicitly.
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass

from .schema import load_schema
from .zstd import decompress

ECHO_FLAG = 128


@dataclass(frozen=True, slots=True)
class CanFrame:
    """One CAN frame observed in a trace."""

    mono_ns: int
    bus: int
    address: int
    data: bytes
    echo: bool

    def __len__(self) -> int:
        return len(self.data)


def classify_src(src: int) -> tuple[int, bool]:
    """Split a raw `CanData.src` into (bus, is_echo).

    >>> classify_src(2)
    (2, False)
    >>> classify_src(130)
    (2, True)
    """
    if src >= ECHO_FLAG:
        return src - ECHO_FLAG, True
    return src, False


def iter_frames(
    path: str,
    *,
    root: str,
    include_echo: bool = False,
    include_invalid: bool = False,
) -> Iterator[CanFrame]:
    """Yield every CAN frame in one `rlog.zst`.

    `root` is the corpus root, used to find (or fetch) the capnp schemas.
    Echoes are excluded unless `include_echo`; see the module docstring for
    why that is the right default.
    """
    schema = load_schema(root)
    with open(path, "rb") as handle:
        raw = decompress(handle.read())
    for event in schema.Event.read_multiple_bytes(raw):
        if event.which() != "can":
            continue
        if not include_invalid and not event.valid:
            continue
        mono_ns = event.logMonoTime
        for frame in event.can:
            bus, echo = classify_src(frame.src)
            if echo and not include_echo:
                continue
            yield CanFrame(mono_ns, bus, frame.address, bytes(frame.dat), echo)


def read_frames(path: str, **kwargs) -> list[CanFrame]:
    """`iter_frames` collected into a list."""
    return list(iter_frames(path, **kwargs))


@dataclass
class SegmentSummary:
    """What one segment contains, without keeping the frames around."""

    path: str
    frames: int
    echoes: int
    buses: dict[int, int]
    addresses: dict[tuple[int, int], int]
    duration_s: float

    @property
    def unique_addresses(self) -> int:
        return len(self.addresses)


def summarize(path: str, *, root: str) -> SegmentSummary:
    """Count what is in a segment, including the echoes normally dropped."""
    buses: dict[int, int] = {}
    addresses: dict[tuple[int, int], int] = {}
    frames = echoes = 0
    first = last = None
    for frame in iter_frames(path, root=root, include_echo=True):
        frames += 1
        if frame.echo:
            echoes += 1
            continue
        buses[frame.bus] = buses.get(frame.bus, 0) + 1
        key = (frame.bus, frame.address)
        addresses[key] = addresses.get(key, 0) + 1
        if first is None:
            first = frame.mono_ns
        last = frame.mono_ns
    duration = (last - first) / 1e9 if first is not None and last is not None else 0.0
    return SegmentSummary(
        path=os.fspath(path),
        frames=frames,
        echoes=echoes,
        buses=dict(sorted(buses.items())),
        addresses=addresses,
        duration_s=duration,
    )
