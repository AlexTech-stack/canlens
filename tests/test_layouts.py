# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Reading what a multiplexor's selector value selects, on known layouts."""
from __future__ import annotations

import random
from dataclasses import replace

import numpy as np

from canlens.analyze.bits import BitOrder, bit_matrix
from canlens.infer import infer_message
from canlens.infer.layouts import LayoutField, read_layout_fields
from canlens.infer.multiplex import find_multiplexor


def matrix_of(payloads):
    return bit_matrix(payloads, len(payloads[0]), BitOrder.INTEL)


def slice_values(payloads: bytes) -> bytes:
    return payloads


VIN_SLICES = (b"1C4RJFJ", b"T7LC163", b"7634\x00\x00\x00")


def vin_like(n: int = 600) -> list[bytes]:
    return [bytes([i % 3]) + VIN_SLICES[i % 3] for i in range(n)]


def vin_held(n: int = 600, hold: int = 40) -> list[bytes]:
    """A VIN whose selector holds each value for `hold` frames in a row.

    The slow switch is what makes the byte-sharing visible: a layout byte
    then sits still for 40 frames and moves twice in 120, which is smooth
    enough to survive the value-jumpiness filter and be claimed as a signal
    as well. Cycling every frame, as `vin_like` does, is filtered out and the
    collision never appears.
    """
    out = []
    for i in range(n):
        value = (i // hold) % 3
        out.append(bytes([value]) + VIN_SLICES[value] + bytes([0x30 + value * 0x40]))
    return out


def vin_with_padding(n: int = 600) -> list[bytes]:
    """As `vin_like`, plus a ninth byte holding 0x5A under every value."""
    return [bytes([i % 3]) + VIN_SLICES[i % 3] + b"\x5a" for i in range(n)]


class TestReadLayoutFields:
    def test_recovers_every_byte_of_every_slice(self):
        matrix = matrix_of(vin_like())
        mux = find_multiplexor(matrix)
        assert mux is not None
        fields = read_layout_fields(matrix, mux)
        by_value: dict[int, dict[int, int]] = {}
        for field in fields:
            by_value.setdefault(field.mux_value, {})[field.start_bit // 8] = field.value
        assert set(by_value) == {0, 1, 2}
        for value, slice_ in enumerate(VIN_SLICES):
            for offset, byte in enumerate(slice_, start=1):
                assert by_value[value][offset] == byte, (value, offset)

    def test_only_bytes_the_selector_chose_are_reported(self):
        """A byte constant in every value's frames is fixing, not laying out."""
        matrix = matrix_of(vin_like())
        mux = find_multiplexor(matrix)
        assert mux is not None
        dependent_bytes = {b // 8 for b in mux.dependent_bits}
        reported = {f.start_bit // 8 for f in read_layout_fields(matrix, mux)}
        assert reported == dependent_bytes

    def test_a_byte_holding_one_constant_under_every_value_is_padding(self):
        """The message fixes it; the selector never chose it.

        Holding a dependent bit is not enough on its own, so the dependency
        is forced here rather than waited for: a byte constant over the whole
        trace still has to be dropped once something calls its bits dependent.
        """
        matrix = matrix_of(vin_with_padding())
        mux = find_multiplexor(matrix)
        assert mux is not None
        claimed = set(mux.dependent_bits) | set(range(64, 72))
        forced = replace(mux, dependent_bits=tuple(sorted(claimed)))
        reported = {f.start_bit // 8 for f in read_layout_fields(matrix, forced)}
        assert 8 not in reported
        assert reported == {1, 2, 3, 4, 5, 6, 7}

    def test_a_byte_moving_inside_a_value_is_reported_only_where_it_is_constant(self):
        """Byte 2 is constant under value 0 but noise under values 1 and 2."""
        rng = random.Random(1)
        payloads = []
        for i in range(600):
            body = 0x55 if i % 3 == 0 else rng.randrange(256)
            payloads.append(bytes([i % 3, 0xAA, body, 0x11]))
        matrix = matrix_of(payloads)
        mux = find_multiplexor(matrix)
        assert mux is not None
        fields = [f for f in read_layout_fields(matrix, mux) if f.start_bit == 16]
        assert {f.mux_value for f in fields} == {0}
        assert fields[0].value == 0x55

    def test_a_layout_field_carries_its_value_and_frame_count(self):
        matrix = matrix_of(vin_like())
        mux = find_multiplexor(matrix)
        assert mux is not None
        fields = read_layout_fields(matrix, mux)
        assert fields
        assert all(f.frames == 200 for f in fields)
        assert all(f.length == 8 and f.byte_order is BitOrder.INTEL for f in fields)

    def test_no_multiplexor_means_no_fields(self):
        matrix = matrix_of([bytes([0, 1, 2, 3])] * 4)
        mux = type("M", (), {"dependent_bits": (), "values": (), "start_bit": 0, "length": 0})()
        assert read_layout_fields(matrix, mux) == []

    def test_an_empty_matrix_yields_nothing(self):
        matrix = np.zeros((0, 16), dtype=np.uint8)
        mux = find_multiplexor(matrix_of(vin_like()))
        assert mux is not None
        assert read_layout_fields(matrix, mux) == []

    def test_str_names_the_value_and_byte(self):
        field = LayoutField(start_bit=8, length=8, mux_value=1, value=0x54, frames=200)
        assert str(field) == "value 1: byte 1 = 84 (0x54, 200 frames)"
        assert field.bits == tuple(range(8, 16))
        assert field.end_bit == 16


class TestDetectionIntegration:
    def test_inference_reports_layout_fields_beside_the_selector(self):
        found = infer_message(vin_like(), bus=0, address=0x6B4)
        assert found.multiplexor is not None
        assert found.layout_fields
        assert found.found_anything
        assert all(f.mux_value in found.multiplexor.values for f in found.layout_fields)

    def test_a_layout_byte_is_not_also_claimed_as_a_signal(self):
        """Two stages cannot both explain the same bits; see invariant 3.

        Before layout fields fed `spoken_for` the signal pass re-claimed 40
        of these bits, reading the same bytes as moving fields that the
        layouts had just read as per-value constants.
        """
        found = infer_message(vin_held(), bus=0, address=0x6B4)
        layout_bits = {b for f in found.layout_fields for b in f.bits}
        signal_bits = {b for s in found.signals for b in s.bits}
        assert layout_bits
        assert not layout_bits & signal_bits

    def test_a_message_without_a_multiplexor_has_no_layout_fields(self):
        payloads = [bytes([i % 256, 0x11, 0x22, 0x33]) for i in range(400)]
        found = infer_message(payloads, bus=0, address=0x100)
        assert found.layout_fields == []
