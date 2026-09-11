# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Turning a segment's analysis into the arrays the workbench draws.

Kept free of Qt so it can be tested without a display.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..analyze import TraceProfile, analyze_frames
from ..analyze.bits import BitOrder, bit_matrix
from ..decode import iter_frames
from ..export import MessageEntry, SignalEntry
from ..infer import MessageInference, infer_frames
from ..infer.counters import field_values, score_counter
from .palette import kinds_to_indices

# CAN's 11-bit identifier space. Anything above it must be a 29-bit extended
# identifier -- but the converse does not hold, and this data cannot tell:
# openpilot's CanData carries `address`, `dat` and `src` and no IDE flag, so an
# extended frame that happens to use a low identifier is indistinguishable from
# a standard one here. The split below is the best the format allows.
STANDARD_ID_MAX = 0x7FF


@dataclass
class NamedSignal:
    """A bit range the user has identified and named."""

    name: str
    start_bit: int
    length: int

    @property
    def end_bit(self) -> int:
        return self.start_bit + self.length

    def overlaps(self, other: NamedSignal) -> bool:
        return self.start_bit < other.end_bit and other.start_bit < self.end_bit

    def __str__(self) -> str:
        return f"{self.name}  bits {self.start_bit}–{self.end_bit - 1} ({self.length})"


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
    signals: list[NamedSignal] = field(default_factory=list)

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
    # Payloads are kept so a bit selection can be replotted without going back
    # to the file. Decoding a segment takes seconds; a drag must not.
    payloads: dict[tuple[int, int], list[bytes]] = field(default_factory=dict)
    _matrix_cache: tuple[int, np.ndarray] | None = field(default=None, repr=False)

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

    def matrix_for(self, index: int) -> np.ndarray:
        """Bit matrix for one message, cached for the row being looked at."""
        if self._matrix_cache is not None and self._matrix_cache[0] == index:
            return self._matrix_cache[1]
        row = self.rows[index]
        matrix = bit_matrix(self.payloads.get(row.key, []), row.width, BitOrder.INTEL)
        self._matrix_cache = (index, matrix)
        return matrix

    def field_series(self, index: int, start: int, length: int) -> np.ndarray:
        """Values of an arbitrary bit range, frame by frame."""
        matrix = self.matrix_for(index)
        if matrix.size == 0 or start < 0 or start + length > matrix.shape[1] or length <= 0:
            return np.zeros(0, dtype=np.int64)
        return field_values(matrix, start, length)

    def field_summary(self, index: int, start: int, length: int) -> str:
        """One line describing a selection, including whether it counts.

        Runs the same stride test the inference layer uses, so a range picked
        by hand is judged on exactly the criteria a reported counter was.
        """
        values = self.field_series(index, start, length)
        if values.size == 0:
            return "—"
        stride, rate = score_counter(values, length)
        distinct = int(np.unique(values).size)
        text = (
            f"bits {start}–{start + length - 1} ({length})   "
            f"min {int(values.min())}   max {int(values.max())}   "
            f"{distinct} distinct"
        )
        if stride and rate >= 0.95:
            text += f"   counts by {stride} on {rate:.0%} of frames"
        return text

    def name_selection(self, index: int, name: str, start: int, length: int) -> NamedSignal:
        """Record a named signal, replacing any it overlaps.

        Overlapping definitions of the same bits cannot both be right, and a
        PDU database that contains both is invalid rather than merely untidy,
        so naming a range that covers an existing one supersedes it.
        """
        row = self.rows[index]
        signal = NamedSignal(name.strip(), start, length)
        row.signals = [s for s in row.signals if not s.overlaps(signal)]
        row.signals.append(signal)
        row.signals.sort(key=lambda s: s.start_bit)
        return signal

    def export_messages(self, *, include_inferred: bool = True) -> list[MessageEntry]:
        """Every message carrying something worth exporting."""
        entries = []
        for index, row in enumerate(self.rows):
            signals = [
                self._named_entry(index, s) for s in row.signals
            ]
            if include_inferred:
                signals += self._inferred_entries(index, row)
            if not signals:
                continue
            signals.sort(key=lambda s: s.start_bit)
            timing = self.profile[row.key].timing
            entries.append(
                MessageEntry(
                    bus=row.bus,
                    address=row.address,
                    length=row.width,
                    extended=row.extended,
                    cyclic=str(timing.cadence) == "cyclic",
                    cycle_time_ms=timing.period_ms,
                    e2e_profile=self._e2e_profile(row),
                    comment=f"derived by canlens from {row.count} frames",
                    signals=signals,
                )
            )
        return entries

    @staticmethod
    def _e2e_profile(row: MessageRow) -> int:
        """AUTOSAR E2E profile number, when one was actually identified."""
        if row.inference is None:
            return 0
        return 5 if any(c.algorithm == "e2e_p05" for c in row.inference.crc16s) else 0

    def _named_entry(self, index: int, signal: NamedSignal) -> SignalEntry:
        values = self.field_series(index, signal.start_bit, signal.length)
        return SignalEntry(
            name=signal.name,
            start_bit=signal.start_bit,
            length=signal.length,
            minimum=float(values.min()) if values.size else 0.0,
            maximum=float(values.max()) if values.size else 0.0,
            init_value=float(values[0]) if values.size else 0.0,
        )

    def _inferred_entries(self, index: int, row: MessageRow) -> list[SignalEntry]:
        """Counters and checksums as signals, with their evidence in Comment."""
        if row.inference is None:
            return []
        named = {(s.start_bit, s.length) for s in row.signals}
        out = []
        for counter in row.inference.counters:
            out.append(
                self._derived_entry(
                    index, "Counter", counter.start_bit, counter.length,
                    f"canlens: counts by {counter.stride} on "
                    f"{counter.match_rate:.1%} of {counter.frames} frames",
                    named,
                )
            )
        for checksum in row.inference.checksums:
            out.append(
                self._derived_entry(
                    index, "Checksum", checksum.start_bit, checksum.length,
                    f"canlens: {checksum.algorithm} reproduces "
                    f"{checksum.match_rate:.1%} of {checksum.frames} frames",
                    named,
                )
            )
        for crc in row.inference.crc16s:
            detail = "" if crc.data_id is None else f", data ID 0x{crc.data_id:04X}"
            out.append(
                self._derived_entry(
                    index, "CRC", crc.start_bit, crc.length,
                    f"canlens: {crc.algorithm} {crc.byteorder}-endian{detail}, "
                    f"reproduces {crc.match_rate:.1%} of {crc.frames} frames",
                    named,
                )
            )
        return [entry for entry in out if entry is not None]

    def _derived_entry(
        self, index: int, name: str, start: int, length: int, comment: str,
        named: set[tuple[int, int]],
    ) -> SignalEntry | None:
        # A hand-given name wins: the user looked at it.
        if (start, length) in named:
            return None
        values = self.field_series(index, start, length)
        return SignalEntry(
            name=name,
            start_bit=start,
            length=length,
            minimum=float(values.min()) if values.size else 0.0,
            maximum=float(values.max()) if values.size else 0.0,
            init_value=float(values[0]) if values.size else 0.0,
            comment=comment,
        )

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

    payloads: dict[tuple[int, int], list[bytes]] = {}
    for frame in frames:
        payloads.setdefault((frame.bus, frame.address), []).append(frame.data)

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
    # Only the dominant payload length, matching how the bit stats were pooled.
    for row_ in rows:
        payloads[row_.key] = [p for p in payloads[row_.key] if len(p) == row_.width]

    return SegmentModel(
        path=path,
        root=root,
        platform=platform,
        profile=profile,
        rows=rows,
        payloads=payloads,
    )
