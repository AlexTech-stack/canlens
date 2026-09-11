# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Workbench data model and palette -- no Qt, no display."""
from __future__ import annotations

import numpy as np
import pytest

from canlens.analyze.bits import KIND_ORDER, BitKind
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
