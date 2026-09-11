# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Export findings as a BoAt PDU-database JSON document.

The format is BoAt's `pdu_db.schema.json` (schema_version 1.0), so anything
canlens derives can be loaded by the PDU editor, replayed by the gateway, or
diffed against a database built from a real DBC.

The one thing worth stating: `StartPos` needs no translation here. The DBC and
Vector convention numbers an Intel signal's start bit as
``byte_index * 8 + offset_from_that_byte's_LSB``, and a Motorola signal's as
its MSB position numbered byte-major from MSB0 -- which is exactly what
:class:`canlens.analyze.BitOrder` produces in each mode. A bit index taken
from the workbench is already a `StartPos`.

Fields the schema requires but a trace cannot supply are written with explicit
neutral values rather than guessed: no routing is known from a single bus, so
`Direction` is 0 and both routing columns are null.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = "1.0"

INTEL = 0
MOTOROLA = 1


@dataclass
class SignalEntry:
    """One signal to export, named by hand or derived by inference."""

    name: str
    start_bit: int
    length: int
    byte_order: int = INTEL
    value_type: str = "Unsigned"
    factor: float = 1.0
    offset: float = 0.0
    minimum: float = 0.0
    maximum: float = 0.0
    unit: str = ""
    init_value: float = 0.0
    comment: str = ""

    def to_json(self, signal_id: int) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "id": signal_id,
            "SignalName": self.name,
            "Length": self.length,
            # No conversion: canlens bit indices already follow the DBC
            # StartPos convention for the byte order they were produced in.
            "StartPos": self.start_bit,
            "ByteOrder": self.byte_order,
            "ValueType": self.value_type,
            "SigSendType": False,
            "Repetitions": 0,
            "InitValue": self.init_value,
            "Factor": self.factor,
            "Offset": self.offset,
            "Min": self.minimum,
            "Max": self.maximum,
            "Unit": self.unit,
            "EnumValues": None,
        }
        if self.comment:
            entry["Comment"] = self.comment
        return entry


@dataclass
class MessageEntry:
    """One CAN message to export."""

    bus: int
    address: int
    length: int
    extended: bool = False
    cyclic: bool = False
    cycle_time_ms: float = 0.0
    e2e_profile: int = 0
    name: str = ""
    comment: str = ""
    signals: list[SignalEntry] = field(default_factory=list)

    @property
    def message_name(self) -> str:
        return self.name or f"MSG_{self.address:03X}"

    @property
    def bus_name(self) -> str:
        return f"CAN_{self.bus}"

    @property
    def is_fd(self) -> bool:
        """Classic CAN carries at most 8 bytes; anything longer is CAN FD."""
        return self.length > 8

    def to_json(self, db_id: int) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "DbId": db_id,
            "MessageName": self.message_name,
            "Bus": self.bus_name,
            "BusType": "CANFD" if self.is_fd else "CAN",
            "MessageType": 0,
            # A single bus tells us nothing about routing, so this is recorded
            # as a source with no copies rather than guessed at.
            "Direction": 0,
            "RoutingType": 0,
            "TargetDbIds": None,
            "SourceDbId": None,
            "isE2E": self.e2e_profile,
            "SendType": "Cyclic" if self.cyclic else "Spontaneous",
            "CycleTime": round(self.cycle_time_ms) if self.cyclic else 0,
            "CycleTimeFast": 0,
            "NrOfRepetitions": 0,
            "Identifier": self.address,
            "FrameType": 1 if self.extended else 0,
            "Length": self.length,
            # Bit Rate Switch is not observable in a decoded trace; CAN FD
            # frames in practice use it, and classic CAN cannot.
            "BRS": self.is_fd,
            "signalcount": len(self.signals),
            "signals": [s.to_json(i) for i, s in enumerate(self.signals, start=1)],
            "Node": "",
        }
        if self.comment:
            entry["Comment"] = self.comment
        return entry


def to_pdu_db(messages: list[MessageEntry]) -> dict[str, Any]:
    """Build the PDU-database document.

    DbIds are sequential from 1 rather than derived from the identifier, so
    adding or removing a signal does not renumber unrelated messages.
    """
    ordered = sorted(messages, key=lambda m: (m.bus, m.extended, m.address))
    return {
        "schema_version": SCHEMA_VERSION,
        "messages": [m.to_json(i) for i, m in enumerate(ordered, start=1)],
        # Cross-bus signal mappings require observing both sides of a gateway;
        # a single trace cannot establish one.
        "signal_routes": [],
    }


def save_pdu_db(messages: list[MessageEntry], path: str) -> str:
    import json

    with open(path, "w") as handle:
        json.dump(to_pdu_db(messages), handle, indent=2)
    return path
