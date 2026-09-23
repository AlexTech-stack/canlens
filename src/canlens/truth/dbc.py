# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""An opendbc DBC, normalised into the terms canlens makes claims in.

A DBC is the closest thing this project has to an answer key: someone
reverse-engineered the bus by hand and wrote down what they found. It is not
infallible -- community DBCs disagree with each other and go stale -- so what
is built here is a *reference*, not a verdict. :mod:`canlens.truth.score`
reports disagreements in both directions and leaves the judgement to a person.

The one thing that has to be exactly right is where a signal sits, because
every comparison downstream is a comparison of bit positions.

**Bit numbering.** cantools reports `Signal.start` in the DBC file's own
numbering, which is byte-major with bit 0 the least significant bit of byte 0 --
the same flat Intel index :mod:`canlens.analyze.bits` produces. For a
little-endian signal that start is the LSB and the signal runs *upward*. For a
big-endian signal it is the MSB and the signal runs *downward*, wrapping to bit
7 of the next byte when it falls off the bottom of the current one.

That wrap is the part worth stating, because the obvious reading is wrong. An
earlier version of this file treated `start` as a "sawtooth" MSB0 index and
converted it; measured against cantools' own decoder -- set one bit, decode,
see which signal moved -- it disagreed on 895 of 1070 signals. The rule below
agrees on all 2925 signals of the 58 DBCs in opendbc, in both byte orders.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from ..analyze.bits import big_endian_bits


class FieldKind(str, Enum):
    """What a reference signal claims to be, as far as its name admits."""

    COUNTER = "counter"
    CHECKSUM = "checksum"
    MULTIPLEXOR = "multiplexor"
    SIGNAL = "signal"  # an ordinary value; canlens does not yet claim these

    def __str__(self) -> str:
        return self.value


# Naming conventions, taken from what opendbc actually contains rather than
# from a convention document. Across the 58 local DBCs the counter-ish names
# are dominated by COUNTER (230), RollingCounter (13), COUNTER1..3 and Ford's
# `*_No_Cnt`; the checksum-ish ones by CHECKSUM (217), CRC (7), XCHECKSUM and
# CRC1..3. `\b` is useless here because the names are CamelCase as often as
# SNAKE_CASE, so the boundary is spelled out.
#
# A bare "count" is deliberately *not* enough on its own. `ESP_wheelPulseCount`
# and `GTW_country` both contain it and neither is a message counter, so the
# word only qualifies when something says which kind of count it is --
# rolling, message, alive, frame or sequence. Audited over every signal name
# in the 58 local DBCs, this classifies every name containing "count", "cnt",
# "checksum" or "crc" the way a human reading it would, with no spurious hits.
_EDGE = r"(?:^|_|(?<=[A-Za-z0-9]))"
_QUALIFIED_COUNT = r"(?:rolling|roll|msg|message|alive|frame|seq)_?count"
COUNTER_PATTERN = re.compile(
    _EDGE + r"(?:counter|" + _QUALIFIED_COUNT + r"|cnt)(?:er)?\d*(?:$|_|(?=[A-Z]))",
    re.IGNORECASE,
)
CHECKSUM_PATTERN = re.compile(
    _EDGE + r"(?:checksum|crc)\d*(?:$|_|(?=[A-Z]))",
    re.IGNORECASE,
)


def classify(name: str, *, is_multiplexer: bool = False) -> FieldKind:
    """What kind of field a reference signal's name says it is.

    The multiplexer flag wins outright because cantools reads it from the
    DBC's own syntax rather than from a naming habit -- it is the only one of
    the three that is declared rather than inferred.
    """
    if is_multiplexer:
        return FieldKind.MULTIPLEXOR
    if CHECKSUM_PATTERN.search(name):
        return FieldKind.CHECKSUM
    if COUNTER_PATTERN.search(name):
        return FieldKind.COUNTER
    return FieldKind.SIGNAL


def signal_bits(start: int, length: int, byte_order: str) -> tuple[int, ...]:
    """The flat Intel bit indices a DBC signal occupies.

    `byte_order` is cantools' spelling: ``"little_endian"`` or
    ``"big_endian"``. See the module docstring for why big-endian walks
    downward rather than being converted from a sawtooth index.
    """
    if length <= 0:
        return ()
    if byte_order == "little_endian":
        return tuple(range(start, start + length))
    return big_endian_bits(start, length)


@dataclass(frozen=True)
class ReferenceSignal:
    """One signal a DBC declares, in canlens' bit numbering."""

    name: str
    bits: tuple[int, ...]
    byte_order: str
    kind: FieldKind
    mux_value: int | None = None  # set only for a signal inside a mux group

    @property
    def length(self) -> int:
        return len(self.bits)

    @property
    def start_bit(self) -> int:
        """Lowest bit index the signal covers.

        For a little-endian signal this is its LSB and so is directly
        comparable to a canlens `start_bit`. For a big-endian one it is
        simply the bottom of the span; comparisons elsewhere use `bits`.
        """
        return min(self.bits) if self.bits else 0

    @property
    def contiguous(self) -> bool:
        return bool(self.bits) and max(self.bits) - min(self.bits) + 1 == len(self.bits)

    def __str__(self) -> str:
        span = f"bit {self.start_bit}" if self.length == 1 else (
            f"bits {min(self.bits)}-{max(self.bits)}"
        )
        return f"{self.name} ({self.kind}, {span}, {self.length} bits)"


@dataclass(frozen=True)
class ReferenceMessage:
    """One message a DBC declares."""

    address: int
    name: str
    length: int  # payload bytes
    signals: tuple[ReferenceSignal, ...]

    def of_kind(self, kind: FieldKind) -> list[ReferenceSignal]:
        return [s for s in self.signals if s.kind is kind]

    @property
    def counters(self) -> list[ReferenceSignal]:
        return self.of_kind(FieldKind.COUNTER)

    @property
    def checksums(self) -> list[ReferenceSignal]:
        return self.of_kind(FieldKind.CHECKSUM)

    @property
    def multiplexors(self) -> list[ReferenceSignal]:
        return self.of_kind(FieldKind.MULTIPLEXOR)

    def __str__(self) -> str:
        return f"{self.name} (0x{self.address:03X})"


@dataclass(frozen=True)
class Reference:
    """Every message a DBC declares, keyed by CAN identifier."""

    name: str
    messages: dict[int, ReferenceMessage]

    def __len__(self) -> int:
        return len(self.messages)

    def __contains__(self, address: int) -> bool:
        return address in self.messages

    def get(self, address: int) -> ReferenceMessage | None:
        return self.messages.get(address)

    def count(self, kind: FieldKind) -> int:
        return sum(len(m.of_kind(kind)) for m in self.messages.values())

    @property
    def addresses(self) -> set[int]:
        return set(self.messages)

    def summary(self) -> str:
        return (
            f"{self.name}: {len(self.messages)} messages, "
            f"{self.count(FieldKind.COUNTER)} counters, "
            f"{self.count(FieldKind.CHECKSUM)} checksums, "
            f"{self.count(FieldKind.MULTIPLEXOR)} multiplexors"
        )


def load_dbc(path: str) -> Reference:
    """Read a `.dbc` file into a `Reference`.

    cantools is an optional dependency (the `truth` extra), so it is imported
    here rather than at module scope: `import canlens.truth` must work in an
    environment that only ever decodes traces.

    Loaded with `strict=False` deliberately. Several opendbc files overlap
    signals or exceed the declared message length, and cantools' strict mode
    refuses the whole file for it. A reference that is mostly right is worth
    more than no reference, and every comparison downstream is per signal.
    """
    import os

    try:
        import cantools.database
        from cantools.database.can.database import Database as CanDatabase
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "scoring against a DBC needs cantools: pip install 'canlens[truth]'"
        ) from exc

    try:
        database = cantools.database.load_file(path, strict=False)
    except Exception as exc:
        # `strict=False` relaxes overlapping signals and over-long messages,
        # but not a malformed identifier -- cantools refuses the whole file
        # for one bad line, and opendbc has such files (hyundai_2015_ccan.dbc
        # declares BSM_LEFT at 0x62cc033, which does not fit 11 bits). There
        # is nothing to do but say which file and why.
        raise ValueError(f"{os.path.basename(path)} could not be read: {exc}") from exc
    # load_file can also return a diagnostics database; only a CAN one has
    # messages, and anything else here is the wrong kind of file.
    if not isinstance(database, CanDatabase):
        raise TypeError(f"{os.path.basename(path)} is not a CAN database")
    messages = {}
    for message in database.messages:
        signals = []
        for signal in message.signals:
            bits = signal_bits(signal.start, signal.length, signal.byte_order)
            mux = signal.multiplexer_ids
            signals.append(
                ReferenceSignal(
                    name=signal.name,
                    bits=bits,
                    byte_order=signal.byte_order,
                    kind=classify(signal.name, is_multiplexer=bool(signal.is_multiplexer)),
                    mux_value=int(mux[0]) if mux else None,
                )
            )
        messages[int(message.frame_id)] = ReferenceMessage(
            address=int(message.frame_id),
            name=message.name,
            length=int(message.length),
            signals=tuple(signals),
        )
    return Reference(name=os.path.basename(path), messages=messages)
