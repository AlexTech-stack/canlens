# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Workbench data model and palette -- no Qt, no display."""
from __future__ import annotations

import numpy as np
import pytest

from canlens.analyze.bits import KIND_ORDER, BitKind
from canlens.gui.axes import byte_ticks
from canlens.gui.model import MessageRow, SegmentModel
from canlens.gui.palette import KIND_INDEX, KIND_RGB, kinds_to_indices, lookup_table


class TestPalette:
    def test_every_kind_has_a_colour(self):
        assert set(KIND_RGB) == set(BitKind)

    def test_lookup_table_follows_kind_order(self):
        table = lookup_table()
        assert table.shape == (len(KIND_ORDER), 3)
        for i, kind in enumerate(KIND_ORDER):
            assert tuple(table[i]) == KIND_RGB[kind]

    def test_indices_match_the_canonical_order(self):
        assert [KIND_INDEX[k] for k in KIND_ORDER] == list(range(len(KIND_ORDER)))

    def test_kinds_to_indices_round_trips(self):
        kinds = list(KIND_ORDER)
        assert kinds_to_indices(kinds).tolist() == list(range(len(kinds)))

    def test_constant_is_light_and_noisy_is_red(self):
        # The GUI must not teach a different colour language from the terminal.
        assert min(KIND_RGB[BitKind.CONSTANT]) > 200
        red, green, blue = KIND_RGB[BitKind.NOISY]
        assert red > green and red > blue


def row(key, kinds, width=8, count=100):
    return MessageRow(
        key=key,
        label=f"bus {key[0]} 0x{key[1]:03X}",
        width=width,
        count=count,
        period_ms=10.0,
        entropy=1.0,
        kinds=np.array(kinds, dtype=np.uint8),
    )


class TestSegmentMatrix:
    def test_width_is_the_widest_message(self):
        model = SegmentModel("p", "r", None, None, [row((0, 1), [0] * 8), row((0, 2), [0] * 64)])
        assert model.bit_width == 64

    def test_short_payloads_are_padded_with_a_sentinel_not_a_class(self):
        # Padding with "constant" would invent structure that was never sent.
        model = SegmentModel("p", "r", None, None, [row((0, 1), [1] * 8), row((0, 2), [2] * 16)])
        grid = model.matrix()
        assert grid.shape == (2, 16)
        assert (grid[0, :8] == 1).all()
        assert (grid[0, 8:] == -1).all()
        assert (grid[1] == 2).all()

    def test_empty_model_is_safe(self):
        model = SegmentModel("p", "r", None, None, [])
        assert model.bit_width == 0
        assert model.matrix().shape == (0, 0)

    def test_field_spans_is_empty_without_inference(self):
        model = SegmentModel("p", "r", None, None, [row((0, 1), [0] * 8)])
        assert model.field_spans(0) == []


class TestFieldSpans:
    @pytest.fixture
    def model(self):
        from canlens.infer import MessageInference
        from canlens.infer.checksums import ChecksumHypothesis
        from canlens.infer.counters import CounterHypothesis
        from canlens.infer.crc16 import Crc16Hypothesis

        r = row((1, 0x210), [0] * 64)
        r.inference = MessageInference(
            bus=1,
            address=0x210,
            width=8,
            frames=100,
            bits=None,  # type: ignore[arg-type]
            counters=[CounterHypothesis(0, 8, 1, 1.0, 100)],
            checksums=[ChecksumHypothesis(7, "toyota", 1.0, 100)],
            crc16s=[Crc16Hypothesis(2, "e2e_p05", "little", 1.0, 100, 0xFA10)],
        )
        return SegmentModel("p", "r", None, None, [r])

    def test_reports_every_kind_of_field(self, model):
        assert set(model.field_spans(0)) == {
            (0, 8, "counter"),
            (56, 8, "checksum"),
            (16, 16, "checksum"),
        }

    def test_spans_stay_inside_the_payload(self, model):
        for start, length, _ in model.field_spans(0):
            assert 0 <= start and start + length <= model.bit_width


class TestByteTicks:
    """Payload bits are read in bytes; decimal ticks cut bytes in half."""

    def test_labels_land_on_byte_boundaries(self):
        major, _ = byte_ticks(64)
        assert [v for v, _ in major] == [0, 8, 16, 24, 32, 40, 48, 56, 64]

    def test_labels_are_the_bit_index(self):
        major, _ = byte_ticks(32)
        assert [text for _, text in major] == ["0", "8", "16", "24", "32"]

    def test_wide_payloads_thin_the_labels_but_keep_the_grid(self):
        major, minor = byte_ticks(256)
        assert all(v % 8 == 0 for v, _ in major)
        assert len(major) <= 19
        # Every byte boundary survives as an unlabelled minor tick.
        assert all(v % 8 == 0 and text == "" for v, text in minor)
        assert sorted(v for v, _ in major + minor) == list(range(0, 257, 8))

    def test_never_produces_a_decimal_step(self):
        for bits in (8, 16, 64, 128, 256, 512):
            major, _ = byte_ticks(bits)
            assert all(v % 8 == 0 for v, _ in major), bits

    def test_empty(self):
        assert byte_ticks(0) == [[], []]


class TestIdentifierOrder:
    """Row order must be stable across segments, which entropy order is not."""

    @staticmethod
    def model(keys):
        return SegmentModel(
            "p", "r", None, None,
            sorted((row(k, [0] * 8) for k in keys), key=lambda r: r.sort_key),
        )

    def test_standard_identifiers_ascend(self):
        m = self.model([(0, 0x200), (0, 0x024), (0, 0x100)])
        assert [r.address for r in m.rows] == [0x024, 0x100, 0x200]

    def test_extended_follow_every_standard_identifier(self):
        m = self.model([(0, 0x18DAF110), (0, 0x7FF), (0, 0x800), (0, 0x001)])
        assert [r.address for r in m.rows] == [0x001, 0x7FF, 0x800, 0x18DAF110]

    def test_the_boundary_is_0x7ff(self):
        m = self.model([(0, 0x7FF), (0, 0x800)])
        assert m.rows[0].extended is False
        assert m.rows[1].extended is True

    def test_buses_stay_grouped(self):
        m = self.model([(1, 0x010), (0, 0x900), (0, 0x010), (1, 0x009)])
        assert [(r.bus, r.address) for r in m.rows] == [
            (0, 0x010), (0, 0x900), (1, 0x009), (1, 0x010)
        ]

    def test_order_does_not_depend_on_entropy(self):
        # Same identifiers, wildly different bit activity: identical order.
        quiet = self.model([(0, 0x300), (0, 0x100)])
        busy = SegmentModel("p", "r", None, None, sorted(
            [row((0, 0x300), [4] * 8), row((0, 0x100), [0] * 8)],
            key=lambda r: r.sort_key,
        ))
        assert [r.address for r in quiet.rows] == [r.address for r in busy.rows]

    def test_a_missing_message_does_not_reorder_the_rest(self):
        # One snippet lacking a message must not shift the others' order.
        full = self.model([(0, 0x100), (0, 0x200), (0, 0x300)])
        partial = self.model([(0, 0x100), (0, 0x300)])
        assert [r.address for r in partial.rows] == [
            a for a in (r.address for r in full.rows) if a != 0x200
        ]


class TestSeparators:
    @staticmethod
    def model(keys):
        return SegmentModel(
            "p", "r", None, None,
            sorted((row(k, [0] * 8) for k in keys), key=lambda r: r.sort_key),
        )

    def test_marks_where_extended_identifiers_begin(self):
        m = self.model([(0, 0x100), (0, 0x7FF), (0, 0x800), (0, 0x900)])
        assert m.separators() == [(2, "extended")]

    def test_marks_a_change_of_bus(self):
        m = self.model([(0, 0x100), (1, 0x100)])
        assert m.separators() == [(1, "bus 1")]

    def test_a_bus_change_is_not_also_reported_as_extended(self):
        m = self.model([(0, 0x100), (1, 0x900)])
        assert m.separators() == [(1, "bus 1")]

    def test_both_kinds_in_one_segment(self):
        m = self.model([(0, 0x100), (0, 0x800), (1, 0x100), (1, 0x800)])
        assert m.separators() == [(1, "extended"), (2, "bus 1"), (3, "extended")]

    def test_none_when_everything_is_standard_on_one_bus(self):
        assert self.model([(0, 0x100), (0, 0x200)]).separators() == []
