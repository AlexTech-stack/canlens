# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Bit-strip rendering -- no terminal required."""
from __future__ import annotations

import pytest

from canlens.analyze.bits import BitKind as K
from canlens.render import (
    BACKGROUNDS,
    GLYPHS,
    RESET,
    TRUNCATED,
    bit_strip,
    bits_that_fit,
    legend,
    strip_width,
    supports_color,
)

FOUR = [K.CONSTANT, K.SLOW, K.ACTIVE, K.NOISY]


class TestBitStrip:
    def test_one_glyph_per_bit(self):
        assert bit_strip(FOUR, color=False, group=0) == ".:+#"

    def test_byte_boundaries_are_spaced(self):
        strip = bit_strip(FOUR * 4, color=False, group=8)
        assert strip == ".:+#.:+# .:+#.:+#"

    def test_colour_uses_background_blocks(self):
        # A coloured space, not a glyph: survives any terminal theme.
        assert bit_strip([K.NOISY], color=True) == f"{BACKGROUNDS[K.NOISY]} {RESET}"

    def test_every_kind_has_both_a_colour_and_a_glyph(self):
        for kind in K:
            assert kind in BACKGROUNDS and kind in GLYPHS

    def test_truncation_is_marked(self):
        strip = bit_strip(FOUR * 4, color=False, max_bits=4, group=0)
        assert strip == ".:+#" + TRUNCATED

    def test_no_marker_when_everything_fits(self):
        assert TRUNCATED not in bit_strip(FOUR, color=False, max_bits=4)

    def test_empty_input_renders_nothing(self):
        assert bit_strip([], color=False) == ""


class TestWidths:
    @pytest.mark.parametrize(
        ("bits", "expected"), [(0, 0), (1, 1), (8, 8), (9, 10), (16, 17), (64, 71)]
    )
    def test_strip_width_counts_separators(self, bits, expected):
        assert strip_width(bits) == expected

    def test_width_matches_what_is_actually_printed(self):
        kinds = FOUR * 4
        assert len(bit_strip(kinds, color=False)) == strip_width(len(kinds))

    def test_bits_that_fit_is_the_inverse(self):
        for available in range(90):
            fits = bits_that_fit(available)
            assert strip_width(fits) <= available
            assert strip_width(fits + 1) > available


class TestColorDetection:
    def test_no_color_wins_over_force_color(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.setenv("FORCE_COLOR", "1")
        assert supports_color() is False

    def test_force_color_without_a_tty(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("FORCE_COLOR", "1")
        assert supports_color() is True

    def test_non_tty_stream_gets_no_colour(self, monkeypatch):
        import io

        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("FORCE_COLOR", raising=False)
        assert supports_color(io.StringIO()) is False


class TestLegend:
    def test_names_every_kind(self):
        text = legend(color=False)
        for kind in K:
            assert str(kind) in text
