# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Ground truth: reading a DBC, and scoring inference against it.

No corpus and no opendbc checkout -- the reference is a DBC written here, so
the expected bit positions are known rather than looked up.
"""
from __future__ import annotations

import numpy as np
import pytest

from canlens.analyze.bits import BitKind, BitOrder, BitProfile
from canlens.infer import MessageInference
from canlens.infer.checksums import ChecksumHypothesis
from canlens.infer.counters import CounterHypothesis
from canlens.infer.crc16 import Crc16Hypothesis
from canlens.infer.multiplex import MultiplexHypothesis
from canlens.truth import (
    FieldKind,
    Reference,
    ReferenceMessage,
    ReferenceSignal,
    Score,
    Tally,
    claimed_fields,
    classify,
    load_dbc,
    pick_bus,
    score,
    signal_bits,
)

cantools = pytest.importorskip("cantools", reason="needs the 'truth' extra")


DBC = """VERSION ""

NS_ :

BS_:

BU_: ECU

BO_ 528 ENGINE: 8 ECU
 SG_ COUNTER : 8|4@1+ (1,0) [0|15] "" ECU
 SG_ CHECKSUM : 56|8@1+ (1,0) [0|255] "" ECU
 SG_ ENGINE_RPM : 16|16@1+ (0.25,0) [0|16383] "rpm" ECU

BO_ 100 WHEELS: 8 ECU
 SG_ WHEEL_SPEED : 12|10@0+ (0.01,0) [0|10] "kph" ECU

BO_ 1000 VIN_BLOCK: 8 ECU
 SG_ VIN_MUX M : 0|2@1+ (1,0) [0|3] "" ECU
 SG_ VIN_A m0 : 8|8@1+ (1,0) [0|255] "" ECU
 SG_ VIN_B m1 : 8|8@1+ (1,0) [0|255] "" ECU
"""


@pytest.fixture
def reference(tmp_path):
    path = tmp_path / "synthetic.dbc"
    path.write_text(DBC)
    return load_dbc(str(path))


def profile(width: int = 8, moving: bool = True) -> BitProfile:
    """A bit profile whose bits move, unless asked otherwise.

    The scorer skips reference signals whose bits never moved -- there is
    nothing in the trace to find them with -- so a profile of all-zero rates
    would quietly exempt every signal from being scored at all.
    """
    n = width * 8
    rates = np.full(n, 0.5) if moving else np.zeros(n)
    return BitProfile(
        width=width, frames=100, ones=np.zeros(n), entropy=np.zeros(n),
        rates=rates, kinds=[BitKind.CONSTANT] * n, order=BitOrder.INTEL,
    )


def inference(address=528, bus=0, **kwargs) -> MessageInference:
    return MessageInference(
        bus=bus, address=address, width=8, frames=100, bits=profile(), **kwargs
    )


class TestSignalBits:
    def test_little_endian_runs_upward_from_the_lsb(self):
        assert signal_bits(8, 4, "little_endian") == (8, 9, 10, 11)

    def test_big_endian_runs_downward_within_a_byte(self):
        assert signal_bits(15, 3, "big_endian") == (15, 14, 13)

    def test_big_endian_wraps_to_the_top_of_the_next_byte(self):
        """The case that makes the naive sawtooth conversion wrong.

        Measured against cantools' own decoder on ESR.dbc CAN_TX_TRACK_ANGLE.
        """
        assert set(signal_bits(12, 10, "big_endian")) == {12, 11, 10, 9, 8, 23, 22, 21, 20, 19}

    def test_a_single_bit_is_its_own_position_either_way(self):
        assert signal_bits(6, 1, "little_endian") == (6,)
        assert signal_bits(6, 1, "big_endian") == (6,)

    def test_zero_length_occupies_nothing(self):
        assert signal_bits(0, 0, "little_endian") == ()


class TestClassify:
    @pytest.mark.parametrize("name", [
        "COUNTER", "COUNTER2", "Counter", "counter",
        "RollingCounter", "AliveCounter", "CF_Clu_AliveCnt1",
        "AbsMduleAlive_No_Cnt", "CAN_TX_TRACK_ROLLING_COUNT", "CF_Lkas_MsgCount",
        "Msg_Count", "DAS_controlCounter",
    ])
    def test_counter_names(self, name):
        assert classify(name) is FieldKind.COUNTER

    @pytest.mark.parametrize("name", [
        "CHECKSUM", "Checksum", "CRC", "CRC2", "XCHECKSUM",
        "Motor_Hybrid_01_CRC", "SCCM_steeringAngleCrc", "LKASteeringCmdChecksum",
    ])
    def test_checksum_names(self, name):
        assert classify(name) is FieldKind.CHECKSUM

    @pytest.mark.parametrize("name", [
        "GTW_country", "CF_Gway_CountryCfg", "ESP_wheelPulseCountFrL",
        "VEHICLE_SPEED", "STEER_ANGLE", "GEAR",
    ])
    def test_a_count_of_something_real_is_not_a_message_counter(self, name):
        """`wheelPulseCount` counts wheel pulses and `country` is not a count."""
        assert classify(name) is FieldKind.SIGNAL

    def test_the_declared_multiplexer_flag_wins(self):
        """It is read from the DBC's syntax, not guessed from a naming habit."""
        assert classify("COUNTER", is_multiplexer=True) is FieldKind.MULTIPLEXOR


class TestLoadDbc:
    def test_reads_every_message(self, reference):
        assert set(reference.messages) == {528, 100, 1000}
        assert reference.addresses == {528, 100, 1000}
        assert 528 in reference

    def test_classifies_the_fields_it_names(self, reference):
        engine = reference.get(528)
        assert [s.name for s in engine.counters] == ["COUNTER"]
        assert [s.name for s in engine.checksums] == ["CHECKSUM"]
        assert [s.name for s in engine.of_kind(FieldKind.SIGNAL)] == ["ENGINE_RPM"]

    def test_positions_follow_canlens_bit_numbering(self, reference):
        engine = reference.get(528)
        assert engine.counters[0].bits == (8, 9, 10, 11)
        assert engine.checksums[0].bits == tuple(range(56, 64))

    def test_a_big_endian_signal_keeps_its_real_positions(self, reference):
        wheel = reference.get(100).signals[0]
        assert wheel.byte_order == "big_endian"
        assert set(wheel.bits) == {12, 11, 10, 9, 8, 23, 22, 21, 20, 19}

    def test_finds_the_declared_multiplexor_and_its_groups(self, reference):
        block = reference.get(1000)
        assert [s.name for s in block.multiplexors] == ["VIN_MUX"]
        assert block.multiplexors[0].bits == (0, 1)
        groups = {s.name: s.mux_value for s in block.signals if s.mux_value is not None}
        assert groups == {"VIN_A": 0, "VIN_B": 1}

    def test_counts_across_the_whole_reference(self, reference):
        assert reference.count(FieldKind.COUNTER) == 1
        assert reference.count(FieldKind.CHECKSUM) == 1
        assert reference.count(FieldKind.MULTIPLEXOR) == 1
        assert "3 messages" in reference.summary()

    def test_a_file_cantools_refuses_names_itself(self, tmp_path):
        bad = tmp_path / "bad.dbc"
        bad.write_text('VERSION ""\n\nNS_ :\n\nBS_:\n\nBU_: ECU\n\nBO_ 103546931 X: 8 ECU\n')
        with pytest.raises(ValueError, match="bad.dbc could not be read"):
            load_dbc(str(bad))


class TestClaimedFields:
    def test_a_counter_claims_its_own_bits(self):
        claims = claimed_fields(inference(counters=[CounterHypothesis(8, 4, 1, 1.0, 100)]))
        [claim] = claims[FieldKind.COUNTER]
        assert (claim.bits, claim.text, claim.exact) == ((8, 9, 10, 11), "4bit@8", True)

    def test_a_checksum_claims_its_whole_byte(self):
        claims = claimed_fields(
            inference(checksums=[ChecksumHypothesis(7, "sum8_addr_len", 1.0, 100)])
        )
        claim = claims[FieldKind.CHECKSUM][0]
        assert claim.bits == tuple(range(56, 64)) and claim.text == "sum8_addr_len@byte7"

    def test_a_wide_crc_claims_every_byte_it_covers(self):
        claims = claimed_fields(
            inference(crc16s=[Crc16Hypothesis(2, "e2e_p05", "little", 1.0, 100, 0xFA10)])
        )
        assert claims[FieldKind.CHECKSUM][0].bits == tuple(range(16, 32))

    def test_checksums_and_wide_crcs_pool_into_one_kind(self):
        """A DBC names both CHECKSUM and CRC without distinguishing width."""
        claims = claimed_fields(inference(
            checksums=[ChecksumHypothesis(7, "sum8_addr_len", 1.0, 100)],
            crc16s=[Crc16Hypothesis(2, "e2e_p05", "little", 1.0, 100)],
        ))
        assert len(claims[FieldKind.CHECKSUM]) == 2

    def test_a_multiplexor_claims_its_selector(self):
        mux = MultiplexHypothesis(0, 8, (0, 1, 2), (10, 10, 10), (8, 9), 30)
        claims = claimed_fields(inference(multiplexor=mux))
        [claim] = claims[FieldKind.MULTIPLEXOR]
        assert (claim.bits, claim.text) == (tuple(range(8)), "8bit@0")

    def test_a_message_with_no_findings_claims_nothing(self):
        claims = claimed_fields(inference())
        assert all(not v for v in claims.values())


class TestTally:
    def test_rates_come_from_the_counts(self):
        tally = Tally(hits=3, false_alarms=1, missed=1)
        assert tally.claimed == 4 and tally.expected == 4
        assert tally.precision == 0.75 and tally.recall == 0.75
        assert tally.f1 == pytest.approx(0.75)

    def test_an_empty_tally_has_no_rates_rather_than_dividing_by_zero(self):
        assert Tally().precision == 0.0 and Tally().recall == 0.0 and Tally().f1 == 0.0

    def test_tallies_add(self):
        assert Tally(1, 2, 3, 4) + Tally(10, 20, 30, 40) == Tally(11, 22, 33, 44)


class TestScore:
    def test_an_exact_match_is_a_hit(self, reference):
        result = score([inference(counters=[CounterHypothesis(8, 4, 1, 1.0, 100)])], reference)
        assert result.tallies[FieldKind.COUNTER] == Tally(hits=1)
        assert not [d for d in result.disagreements if d.kind is FieldKind.COUNTER]

    def test_a_field_the_reference_names_but_canlens_did_not_find_is_missed(self, reference):
        result = score([inference()], reference)
        assert result.tallies[FieldKind.COUNTER].missed == 1
        missed = [d for d in result.disagreements if d.verdict == "missed"]
        assert any("COUNTER" in str(d) for d in missed)

    def test_a_field_the_reference_does_not_name_is_extra(self, reference):
        result = score([inference(counters=[CounterHypothesis(24, 8, 1, 1.0, 100)])], reference)
        counter = result.tallies[FieldKind.COUNTER]
        assert counter.false_alarms == 1 and counter.hits == 0
        assert any(d.verdict == "extra" for d in result.disagreements)

    def test_the_wrong_extent_is_near_rather_than_a_hit(self, reference):
        """Right field, wrong width: counted, but never as agreement."""
        result = score([inference(counters=[CounterHypothesis(8, 8, 1, 1.0, 100)])], reference)
        counter = result.tallies[FieldKind.COUNTER]
        assert counter.hits == 0 and counter.near == 1
        moved = [d for d in result.disagreements if d.verdict == "moved"]
        assert len(moved) == 1 and moved[0].overlapping

    def test_a_reference_message_the_drive_never_carried_is_not_a_miss(self, reference):
        result = score([inference(address=528)], reference)
        assert result.scored_messages == 1
        assert result.absent_from_trace == 2
        # WHEELS and VIN_BLOCK contribute no missed multiplexor.
        assert result.tallies[FieldKind.MULTIPLEXOR].missed == 0

    def test_a_kind_the_reference_never_names_is_left_out_of_the_rates(self, tmp_path):
        """An older Honda DBC names no counters; 44 claims is not 0% precision."""
        path = tmp_path / "bare.dbc"
        path.write_text(
            'VERSION ""\n\nNS_ :\n\nBS_:\n\nBU_: ECU\n\nBO_ 528 M: 8 ECU\n'
            ' SG_ SPEED : 0|8@1+ (1,0) [0|255] "" ECU\n'
        )
        bare = load_dbc(str(path))
        result = score([inference(counters=[CounterHypothesis(8, 4, 1, 1.0, 100)])], bare)
        assert FieldKind.COUNTER not in result.scorable
        assert result.unevaluated[FieldKind.COUNTER] == 1
        assert result.tallies[FieldKind.COUNTER] == Tally()
        # SPEED is an ordinary signal, so that kind is scorable and unfound.
        assert result.scorable == {FieldKind.SIGNAL}

    def test_scoring_covers_every_kind_at_once(self, reference):
        mux = MultiplexHypothesis(0, 2, (0, 1), (10, 10), (8, 9), 20)
        result = score(
            [
                inference(
                    address=528,
                    counters=[CounterHypothesis(8, 4, 1, 1.0, 100)],
                    checksums=[ChecksumHypothesis(7, "sum8_addr_len", 1.0, 100)],
                ),
                inference(address=1000, multiplexor=mux),
            ],
            reference,
        )
        assert result.tallies[FieldKind.COUNTER].hits == 1
        assert result.tallies[FieldKind.CHECKSUM].hits == 1
        assert result.tallies[FieldKind.MULTIPLEXOR].hits == 1
        # Nothing claimed an ordinary signal, so the reference's are missed.
        assert result.tallies[FieldKind.SIGNAL] == Tally(missed=2)
        assert result.overall.hits == 3 and result.overall.false_alarms == 0
        assert result.overall.precision == 1.0

    def test_ordinary_signals_are_scored_now_that_canlens_claims_them(self, reference):
        result = score([inference(address=528)], reference)
        assert result.unnamed_signals == 1  # ENGINE_RPM
        assert FieldKind.SIGNAL in result.tallies
        assert result.tallies[FieldKind.SIGNAL].missed == 1

    def test_a_reference_signal_that_never_moved_is_not_a_miss(self):
        """Nothing in the trace could have found it."""
        still = MessageInference(
            bus=0, address=528, width=8, frames=100, bits=profile(moving=False)
        )
        result = score([still], self.reference_for())
        assert result.tallies[FieldKind.SIGNAL] == Tally()
        assert result.still_signals == 1

    def test_a_lower_bound_matches_a_wider_reference_field(self):
        """ENGINE_RPM is 16 bits at 16; claiming "10+ bits at 16" is right."""
        from canlens.infer.signals import SignalHypothesis

        found = score(
            [inference(address=528, signals=[
                SignalHypothesis(16, 10, 100, 0.5, 0, 900, bounded=False)])],
            self.reference_for(),
        )
        assert found.tallies[FieldKind.SIGNAL].hits == 1

    def test_a_bounded_claim_gets_no_such_allowance(self):
        from canlens.infer.signals import SignalHypothesis

        found = score(
            [inference(address=528, signals=[
                SignalHypothesis(16, 10, 100, 0.5, 0, 900, bounded=True)])],
            self.reference_for(),
        )
        assert found.tallies[FieldKind.SIGNAL].hits == 0

    @staticmethod
    def reference_for():
        import pathlib as _p
        import tempfile
        path = _p.Path(tempfile.mkdtemp()) / "s.dbc"
        path.write_text(DBC)
        return load_dbc(str(path))


class TestPickBus:
    def test_picks_the_bus_that_overlaps_the_reference(self, reference):
        inferences = [
            inference(address=0x111, bus=0),
            inference(address=0x222, bus=0),
            inference(address=528, bus=2),
            inference(address=100, bus=2),
        ]
        assert pick_bus(inferences, reference) == 2

    def test_an_empty_trace_picks_bus_zero(self, reference):
        assert pick_bus([], reference) == 0

    def test_an_explicit_bus_overrides_the_guess(self, reference):
        result = score([inference(address=528, bus=3)], reference, bus=3)
        assert result.bus == 3 and result.scored_messages == 1

    def test_scoring_the_wrong_bus_scores_nothing(self, reference):
        result = score([inference(address=528, bus=3)], reference, bus=0)
        assert result.scored_messages == 0 and result.trace_messages == 0


class TestReferenceSignalRendering:
    def test_a_span_reads_as_a_range(self):
        signal = ReferenceSignal("COUNTER", (8, 9, 10, 11), "little_endian", FieldKind.COUNTER)
        assert str(signal) == "COUNTER (counter, bits 8-11, 4 bits)"
        assert signal.start_bit == 8 and signal.contiguous

    def test_a_single_bit_reads_as_one_bit(self):
        signal = ReferenceSignal("FLAG", (6,), "little_endian", FieldKind.SIGNAL)
        assert str(signal) == "FLAG (signal, bit 6, 1 bits)"

    def test_a_big_endian_span_that_wraps_is_not_contiguous(self):
        signal = ReferenceSignal(
            "WIDE", signal_bits(12, 10, "big_endian"), "big_endian", FieldKind.SIGNAL
        )
        assert not signal.contiguous


def test_an_empty_reference_scores_nothing():
    result = score([inference()], Reference(name="empty.dbc", messages={}))
    assert result.scored_messages == 0
    assert isinstance(result, Score)
    assert result.reference == "empty.dbc"


def test_a_reference_message_renders_with_its_identifier():
    message = ReferenceMessage(address=0x210, name="ENGINE", length=8, signals=())
    assert str(message) == "ENGINE (0x210)"
