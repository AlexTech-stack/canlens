# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""PDU-database export.

The structural tests are self-contained. A further test validates against
BoAt's real `pdu_db.schema.json` when that repository happens to be present,
which is the only check that can catch the schema moving underneath us.
"""
from __future__ import annotations

import json
import os

import pytest

from canlens.export import (
    INTEL,
    MOTOROLA,
    SCHEMA_VERSION,
    MessageEntry,
    SignalEntry,
    save_pdu_db,
    to_pdu_db,
)

BOAT_SCHEMA = "/home/testuser/BoAt/boat-platform/config/pdu_db.schema.json"

MESSAGE_REQUIRED = {
    "DbId", "MessageName", "Bus", "BusType", "MessageType", "Direction",
    "RoutingType", "TargetDbIds", "SourceDbId", "isE2E", "SendType",
    "CycleTime", "CycleTimeFast", "NrOfRepetitions", "signalcount", "signals",
}
SIGNAL_REQUIRED = {
    "id", "SignalName", "Length", "StartPos", "ByteOrder", "ValueType",
    "SigSendType", "Repetitions", "InitValue", "Factor", "Offset",
    "Min", "Max", "Unit", "EnumValues",
}


def message(**kwargs) -> MessageEntry:
    base = {"bus": 0, "address": 0x210, "length": 8, "signals": [SignalEntry("S", 0, 8)]}
    base.update(kwargs)
    return MessageEntry(**base)


class TestDocument:
    def test_top_level_shape(self):
        db = to_pdu_db([message()])
        assert set(db) == {"schema_version", "messages", "signal_routes"}
        assert db["schema_version"] == SCHEMA_VERSION

    def test_signal_routes_is_empty_because_one_bus_cannot_show_routing(self):
        assert to_pdu_db([message()])["signal_routes"] == []

    def test_empty_input(self):
        db = to_pdu_db([])
        assert db["messages"] == []

    def test_round_trips_through_a_file(self, tmp_path):
        path = tmp_path / "db.json"
        save_pdu_db([message()], str(path))
        assert json.loads(path.read_text()) == to_pdu_db([message()])


class TestRequiredFields:
    def test_every_message_carries_the_required_keys(self):
        db = to_pdu_db([message(), message(address=0x300)])
        for entry in db["messages"]:
            assert MESSAGE_REQUIRED <= set(entry)

    def test_every_signal_carries_the_required_keys(self):
        db = to_pdu_db([message()])
        for signal in db["messages"][0]["signals"]:
            assert SIGNAL_REQUIRED <= set(signal)

    def test_signalcount_matches_the_signal_list(self):
        entry = message(signals=[SignalEntry("A", 0, 4), SignalEntry("B", 4, 4)])
        exported = to_pdu_db([entry])["messages"][0]
        assert exported["signalcount"] == len(exported["signals"]) == 2


class TestIdentity:
    def test_db_ids_are_sequential_from_one(self):
        db = to_pdu_db([message(address=a) for a in (0x300, 0x100, 0x200)])
        assert [m["DbId"] for m in db["messages"]] == [1, 2, 3]

    def test_messages_are_ordered_by_identifier(self):
        db = to_pdu_db([message(address=a) for a in (0x300, 0x100, 0x200)])
        assert [m["Identifier"] for m in db["messages"]] == [0x100, 0x200, 0x300]

    def test_extended_identifiers_sort_after_standard_ones(self):
        db = to_pdu_db([message(address=0x900, extended=True), message(address=0x100)])
        assert [m["Identifier"] for m in db["messages"]] == [0x100, 0x900]

    def test_signal_ids_are_unique_within_a_message(self):
        entry = message(signals=[SignalEntry(f"S{i}", i * 8, 8) for i in range(4)])
        ids = [s["id"] for s in to_pdu_db([entry])["messages"][0]["signals"]]
        assert ids == [1, 2, 3, 4]

    def test_default_name_encodes_the_identifier(self):
        assert to_pdu_db([message(address=0x2C1)])["messages"][0]["MessageName"] == "MSG_2C1"

    def test_an_explicit_name_wins(self):
        assert to_pdu_db([message(name="Motor_1")])["messages"][0]["MessageName"] == "Motor_1"


class TestFrameProperties:
    @pytest.mark.parametrize(
        ("length", "bus_type", "brs"), [(8, "CAN", False), (16, "CANFD", True), (64, "CANFD", True)]
    )
    def test_payload_length_decides_can_versus_canfd(self, length, bus_type, brs):
        entry = to_pdu_db([message(length=length)])["messages"][0]
        assert entry["BusType"] == bus_type
        assert entry["BRS"] is brs

    def test_frame_type_follows_the_extended_flag(self):
        assert to_pdu_db([message()])["messages"][0]["FrameType"] == 0
        assert to_pdu_db([message(extended=True)])["messages"][0]["FrameType"] == 1

    def test_cyclic_messages_carry_their_period(self):
        entry = to_pdu_db([message(cyclic=True, cycle_time_ms=49.6)])["messages"][0]
        assert entry["SendType"] == "Cyclic"
        assert entry["CycleTime"] == 50

    def test_sporadic_messages_report_no_cycle_time(self):
        entry = to_pdu_db([message(cyclic=False, cycle_time_ms=37.0)])["messages"][0]
        assert entry["SendType"] == "Spontaneous"
        assert entry["CycleTime"] == 0

    def test_routing_columns_are_neutral_not_guessed(self):
        entry = to_pdu_db([message()])["messages"][0]
        assert entry["Direction"] == 0
        assert entry["RoutingType"] == 0
        assert entry["TargetDbIds"] is None
        assert entry["SourceDbId"] is None

    def test_e2e_profile_is_carried_through(self):
        assert to_pdu_db([message(e2e_profile=5)])["messages"][0]["isE2E"] == 5
        assert to_pdu_db([message()])["messages"][0]["isE2E"] == 0


class TestSignalFields:
    def test_start_pos_needs_no_translation(self):
        # canlens bit indices already follow the DBC StartPos convention for
        # the byte order they were produced in, which is the whole reason the
        # analyze layer defaults to Intel numbering.
        entry = to_pdu_db([message(signals=[SignalEntry("S", 27, 5)])])["messages"][0]
        assert entry["signals"][0]["StartPos"] == 27

    @pytest.mark.parametrize("order", [INTEL, MOTOROLA])
    def test_byte_order_is_recorded(self, order):
        entry = message(signals=[SignalEntry("S", 0, 8, byte_order=order)])
        assert to_pdu_db([entry])["messages"][0]["signals"][0]["ByteOrder"] == order

    def test_observed_range_is_exported(self):
        entry = message(signals=[SignalEntry("S", 0, 8, minimum=3.0, maximum=250.0)])
        signal = to_pdu_db([entry])["messages"][0]["signals"][0]
        assert signal["Min"] == 3.0 and signal["Max"] == 250.0

    def test_comment_is_omitted_when_empty(self):
        assert "Comment" not in to_pdu_db([message()])["messages"][0]["signals"][0]

    def test_comment_is_kept_when_given(self):
        entry = message(signals=[SignalEntry("S", 0, 8, comment="canlens: counts by 1")])
        assert to_pdu_db([entry])["messages"][0]["signals"][0]["Comment"] == "canlens: counts by 1"


@pytest.mark.skipif(not os.path.exists(BOAT_SCHEMA), reason="BoAt schema not present")
class TestAgainstBoatSchema:
    """The only check that notices if the real schema changes."""

    @pytest.fixture(scope="class")
    @classmethod
    def validator(cls):
        jsonschema = pytest.importorskip("jsonschema")
        with open(BOAT_SCHEMA) as handle:
            return jsonschema.Draft202012Validator(json.load(handle))

    def test_a_minimal_document_validates(self, validator):
        validator.validate(to_pdu_db([message()]))

    def test_every_frame_variety_validates(self, validator):
        validator.validate(to_pdu_db([
            message(address=0x100, length=8, cyclic=True, cycle_time_ms=10),
            message(address=0x900, extended=True, length=32, e2e_profile=5),
            message(bus=1, address=0x200, length=64, signals=[
                SignalEntry("Counter", 0, 8, comment="canlens: counts by 1"),
                SignalEntry("CRC", 16, 16, byte_order=MOTOROLA),
            ]),
        ]))

    def test_an_empty_document_validates(self, validator):
        validator.validate(to_pdu_db([]))
