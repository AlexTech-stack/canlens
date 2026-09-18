# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""A trace held as columns rather than as objects.

Decoding a segment costs about 1.5 seconds, and essentially all of it is
~1.2 million field reads across the pycapnp boundary at roughly 0.6 us each.
Caching removes that -- but only if what comes back out of the cache is
columnar. Rebuilding 400000 :class:`CanFrame` objects from arrays costs 0.7s
on its own, which would cap a 95x cache at about 2x.

So this is the shape everything downstream should consume. Payloads are held
as one flat byte array with a length per frame, which wastes nothing on a
trace mixing 8-byte classic frames with 32-byte CAN FD ones, and lets a
message's payloads be gathered into a matrix by index arithmetic instead of
by slicing out several thousand `bytes` objects.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from .rlog import ECHO_FLAG, CanFrame


@dataclass
class FrameSet:
    """Every frame of a trace, in columns."""

    mono_ns: np.ndarray  # int64
    src: np.ndarray  # uint8, raw CanData.src including the echo flag
    address: np.ndarray  # uint32
    lengths: np.ndarray  # uint8
    blob: np.ndarray  # uint8, payloads laid end to end

    def __post_init__(self) -> None:
        self._offsets: np.ndarray | None = None
        self._raw: bytes | None = None

    @property
    def raw(self) -> bytes:
        """The payload blob as bytes, materialised once.

        `payloads()` is called per message; copying an 11 MiB blob each time
        would cost more than the decode this cache exists to avoid.
        """
        if self._raw is None:
            self._raw = self.blob.tobytes()
        return self._raw

    def __len__(self) -> int:
        return int(self.mono_ns.size)

    @property
    def offsets(self) -> np.ndarray:
        """Start of each payload within `blob`, with a trailing end marker."""
        if self._offsets is None:
            offsets = np.zeros(len(self) + 1, dtype=np.int64)
            np.cumsum(self.lengths, out=offsets[1:])
            self._offsets = offsets
        return self._offsets

    @property
    def bus(self) -> np.ndarray:
        return (self.src & (ECHO_FLAG - 1)).astype(np.uint8)

    @property
    def echo(self) -> np.ndarray:
        """Frames the device put on the wire itself; see `canlens.decode.rlog`."""
        return self.src >= ECHO_FLAG

    def without_echoes(self) -> FrameSet:
        return self.take(~self.echo)

    def take(self, mask: np.ndarray) -> FrameSet:
        """A new FrameSet holding only the selected frames."""
        index = np.flatnonzero(mask) if mask.dtype == bool else mask
        starts, lengths = self.offsets[index], self.lengths[index]
        return FrameSet(
            mono_ns=self.mono_ns[index],
            src=self.src[index],
            address=self.address[index],
            lengths=lengths,
            blob=self.blob[_ranges(starts, lengths)],
        )

    def keys(self) -> list[tuple[int, int]]:
        """Every (bus, address) present, in ascending order."""
        pairs = np.unique(np.stack([self.bus, self.address]).T, axis=0)
        return [(int(b), int(a)) for b, a in pairs]

    def indices_for(self, bus: int, address: int) -> np.ndarray:
        return np.flatnonzero((self.bus == bus) & (self.address == address))

    def group(self, min_frames: int = 1) -> list[Message]:
        """Split into messages, each with its dominant payload width.

        Grouping is done with one lexicographic sort rather than a dictionary
        keyed per frame, which matters at four hundred thousand of them.
        """
        order = np.lexsort((self.address, self.bus))
        bus, address = self.bus[order], self.address[order]
        boundaries = np.flatnonzero(
            (bus[1:] != bus[:-1]) | (address[1:] != address[:-1])
        )
        starts = np.concatenate(([0], boundaries + 1))
        ends = np.concatenate((boundaries + 1, [len(order)]))

        out = []
        for begin, finish in zip(starts, ends, strict=True):
            index = order[begin:finish]
            if index.size < min_frames:
                continue
            lengths = self.lengths[index]
            widths, counts = np.unique(lengths, return_counts=True)
            width = int(widths[counts.argmax()])
            same = index[lengths == width]
            if same.size < min_frames:
                continue
            out.append(
                Message(
                    bus=int(bus[begin]),
                    address=int(address[begin]),
                    width=width,
                    count=int(index.size),
                    index=same,
                    lengths={int(w): int(c) for w, c in zip(widths, counts, strict=True)},
                    frames=self,
                )
            )
        return out

    def payload_matrix(self, index: np.ndarray, width: int) -> np.ndarray:
        """Payloads of the given frames as an (n, width) byte matrix.

        Gathered straight out of the flat blob: no intermediate `bytes` object
        is created, which is the whole point of keeping the trace in columns.
        """
        if index.size == 0:
            return np.zeros((0, width), dtype=np.uint8)
        starts = self.offsets[index]
        return self.blob[starts[:, None] + np.arange(width)]

    def payloads(self, index: np.ndarray) -> list[bytes]:
        """Payloads as bytes, for the algorithms that still want them."""
        raw = self.raw
        starts, lengths = self.offsets[index], self.lengths[index]
        return [raw[s : s + n] for s, n in zip(starts, lengths, strict=True)]

    def frames(self) -> Iterator[CanFrame]:
        """Rebuild CanFrame objects. Costs ~0.7s for a full segment."""
        raw = self.raw
        offsets, lengths = self.offsets, self.lengths
        for i in range(len(self)):
            src = int(self.src[i])
            yield CanFrame(
                int(self.mono_ns[i]),
                src & (ECHO_FLAG - 1),
                int(self.address[i]),
                raw[offsets[i] : offsets[i] + lengths[i]],
                src >= ECHO_FLAG,
            )

    def __iter__(self) -> Iterator[CanFrame]:
        return self.frames()


@dataclass
class Message:
    """One (bus, address) within a FrameSet, at its dominant payload width."""

    bus: int
    address: int
    width: int
    count: int
    index: np.ndarray
    lengths: dict[int, int]
    frames: FrameSet

    @property
    def key(self) -> tuple[int, int]:
        return (self.bus, self.address)

    @property
    def multi_length(self) -> bool:
        return len(self.lengths) > 1

    @property
    def stamps(self) -> np.ndarray:
        """Arrival times of every frame of this message, in order."""
        return self.frames.mono_ns[self.index]

    def bytes_matrix(self) -> np.ndarray:
        """Payloads as an (n, width) byte matrix, gathered without copying."""
        return self.frames.payload_matrix(self.index, self.width)

    def payloads(self) -> list[bytes]:
        return self.frames.payloads(self.index)

    def __str__(self) -> str:
        return f"bus {self.bus} 0x{self.address:03X}"


def _ranges(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Indices covering `lengths[i]` bytes from each `starts[i]`.

    Built by repetition rather than by a cumulative-sum trick: the latter has
    to express each block boundary as a jump, which is easy to write in output
    coordinates when it belongs in source ones, and it has no sensible answer
    for a zero-length payload. Here a zero length simply contributes nothing.
    """
    total = int(lengths.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    lengths = lengths.astype(np.int64)
    block_starts = np.cumsum(lengths) - lengths
    return np.repeat(starts - block_starts, lengths) + np.arange(total, dtype=np.int64)


def from_frames(frames: list[CanFrame]) -> FrameSet:
    """Build a FrameSet from CanFrame objects, for callers that have them."""
    count = len(frames)
    return FrameSet(
        mono_ns=np.fromiter((f.mono_ns for f in frames), np.int64, count),
        src=np.fromiter(
            (f.bus + (ECHO_FLAG if f.echo else 0) for f in frames), np.uint8, count
        ),
        address=np.fromiter((f.address for f in frames), np.uint32, count),
        lengths=np.fromiter((len(f.data) for f in frames), np.uint8, count),
        blob=np.frombuffer(b"".join(f.data for f in frames), dtype=np.uint8),
    )
