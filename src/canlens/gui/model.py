# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Turning a segment's analysis into the arrays the workbench draws.

Kept free of Qt so it can be tested without a display.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..analyze import TraceProfile, analyze_frames
from ..decode import iter_frames
from ..infer import MessageInference, infer_frames
from .palette import kinds_to_indices

# CAN's 11-bit identifier space. Anything above it must be a 29-bit extended
# identifier -- but the converse does not hold, and this data cannot tell:
# openpilot's CanData carries `address`, `dat` and `src` and no IDE flag, so an
# extended frame that happens to use a low identifier is indistinguishable from
# a standard one here. The split below is the best the format allows.
STANDARD_ID_MAX = 0x7FF


@dataclass
class MessageRow:
    """One row of the bit matrix, plus whatever was inferred about it."""

    key: tuple[int, int]
    label: str
    width: int
    count: int
    period_ms: float
    entropy: float
    kinds: np.ndarray
    inference: MessageInference | None = None

    @property
    def bits(self) -> int:
        return int(self.kinds.size)

    @property
    def bus(self) -> int:
        return self.key[0]

    @property
    def address(self) -> int:
        return self.key[1]

    @property
    def extended(self) -> bool:
        """Whether this identifier can only be a 29-bit extended one."""
        return self.address > STANDARD_ID_MAX

    @property
    def sort_key(self) -> tuple[int, int, int]:
        """Bus, then standard identifiers ascending, then extended ascending.

        Stable across segments by construction, which entropy ordering is not:
        the same bus recorded twice must lay out identically, or two snippets
        of one drive cannot be compared row against row.
        """
        return (self.bus, int(self.extended), self.address)


@dataclass
class SegmentModel:
    """Everything the workbench needs about one segment."""

    path: str
    root: str
    platform: str | None
    profile: TraceProfile
    rows: list[MessageRow] = field(default_factory=list)

    @property
    def bit_width(self) -> int:
        return max((row.bits for row in self.rows), default=0)

    def matrix(self) -> np.ndarray:
        """Rows x bits of class indices, padded with -1 where a payload ends.

        Padding is a sentinel rather than a class: CAN FD and classic CAN sit
        in the same view, and drawing a short payload as though it were full of
        constant bits would invent structure that is not there.
        """
        grid = np.full((len(self.rows), self.bit_width), -1, dtype=np.int16)
        for i, row in enumerate(self.rows):
            grid[i, : row.bits] = row.kinds
        return grid

    def separators(self) -> list[tuple[int, str]]:
        """Rows a divider should be drawn above, and what it divides.

        Two kinds: a change of bus, and the step from standard identifiers to
        extended ones within a bus.
        """
        marks = []
        for i in range(1, len(self.rows)):
            previous, current = self.rows[i - 1], self.rows[i]
            if current.bus != previous.bus:
                marks.append((i, f"bus {current.bus}"))
            elif current.extended and not previous.extended:
                marks.append((i, "extended"))
        return marks

    def bus_groups(self) -> list[tuple[int, int, int]]:
        """(bus, first row, last row exclusive) for each contiguous bus block."""
        groups: list[tuple[int, int, int]] = []
        for i, row in enumerate(self.rows):
            if groups and groups[-1][0] == row.bus:
                bus, start, _ = groups[-1]
                groups[-1] = (bus, start, i + 1)
            else:
                groups.append((row.bus, i, i + 1))
        return groups

    def field_spans(self, index: int) -> list[tuple[int, int, str]]:
        """(start_bit, length, kind) for the fields inferred in one row."""
        row = self.rows[index]
        if row.inference is None:
            return []
        spans = [(c.start_bit, c.length, "counter") for c in row.inference.counters]
        spans += [(s.start_bit, s.length, "checksum") for s in row.inference.checksums]
        spans += [(c.start_bit, c.length, "checksum") for c in row.inference.crc16s]
        return spans


def load_segment(
    path: str, *, root: str, platform: str | None = None, infer: bool = True
) -> SegmentModel:
    """Decode, measure and (optionally) infer one segment for display."""
    frames = list(iter_frames(path, root=root))
    profile = analyze_frames(frames)
    inferences = {m.key: m for m in infer_frames(frames)} if infer else {}

    rows = [
        MessageRow(
            key=message.key,
            label=str(message),
            width=message.width,
            count=message.count,
            period_ms=message.timing.period_ms,
            entropy=message.bits.payload_entropy,
            kinds=kinds_to_indices(message.bits.kinds),
            inference=inferences.get(message.key),
        )
        for message in profile.messages.values()
    ]
    # Ordered by identifier, never by entropy: see MessageRow.sort_key.
    rows.sort(key=lambda row: row.sort_key)
    return SegmentModel(
        path=path, root=root, platform=platform, profile=profile, rows=rows
    )
