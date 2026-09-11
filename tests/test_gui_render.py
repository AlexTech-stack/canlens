# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Rendering regressions that only show up once pixels exist.

Runs headless via the offscreen Qt platform, the same way BoAt verifies its
own Qt client, and skips entirely when the gui extra is not installed.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pg = pytest.importorskip("pyqtgraph", reason="needs the 'gui' extra")
pytest.importorskip("PySide6", reason="needs the 'gui' extra")

from canlens.analyze.bits import BitKind
from canlens.gui.palette import KIND_RGB


@pytest.fixture(scope="module")
def qt_app():
    from PySide6 import QtWidgets

    yield QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def render_first_pixel(rgba: np.ndarray, **kwargs) -> tuple[int, int, int]:
    item = pg.ImageItem(axisOrder="row-major")
    item.setImage(rgba, **kwargs)
    colour = item.getPixmap().toImage().pixel(0, 0)
    return ((colour >> 16) & 0xFF, (colour >> 8) & 0xFF, colour & 0xFF)


def uniform(rgb: tuple[int, int, int], bits: int = 8) -> np.ndarray:
    rgba = np.zeros((1, bits, 4), dtype=np.ubyte)
    rgba[0, :, :3] = rgb
    rgba[0, :, 3] = 255
    return rgba


class TestUniformPayloadIsNotBlack:
    """A wholly static message must render as constant, not as missing.

    pyqtgraph derives levels from the data unless told otherwise. For a payload
    whose bits are all one class the array is uniform, auto-levels come out as
    (238, 255), and every pixel maps to black -- so a frame that never changes
    looked identical to no frame at all.
    """

    def test_auto_levels_reproduce_the_bug(self, qt_app):
        assert render_first_pixel(uniform(KIND_RGB[BitKind.CONSTANT])) == (0, 0, 0)

    def test_pinned_levels_render_the_real_colour(self, qt_app):
        constant = KIND_RGB[BitKind.CONSTANT]
        assert render_first_pixel(uniform(constant), levels=(0, 255)) == constant

    @pytest.mark.parametrize("kind", list(BitKind))
    def test_every_class_survives_a_uniform_payload(self, qt_app, kind):
        assert render_first_pixel(uniform(KIND_RGB[kind]), levels=(0, 255)) == KIND_RGB[kind]

    def test_mixed_payloads_were_never_affected(self, qt_app):
        rgba = uniform(KIND_RGB[BitKind.CONSTANT])
        rgba[0, 4, :3] = KIND_RGB[BitKind.NOISY]
        assert render_first_pixel(rgba, levels=(0, 255)) == KIND_RGB[BitKind.CONSTANT]


class TestViewsPinLevels:
    """The fix must live in the widgets, not just in this test file."""

    def test_strip_and_matrix_both_pin_levels(self):
        import inspect

        from canlens.gui import window

        source = inspect.getsource(window)
        assert source.count("levels=(0, 255)") == 2
