# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""The per-value layout view under the bit strip -- offscreen Qt."""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6", reason="needs the 'gui' extra")
pytest.importorskip("pyqtgraph", reason="needs the 'gui' extra")

from PySide6 import QtWidgets

from canlens.gui.model import MessageRow, SegmentModel
from canlens.gui.window import LAYOUT_MAX_ROWS, LAYOUT_ROW_PX, DetailPanel, LayoutView
from canlens.infer import MessageInference
from canlens.infer.multiplex import MultiplexHypothesis


@pytest.fixture(scope="module")
def qt_app():
    yield QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def model_with(values: int, frames: int = 300) -> SegmentModel:
    payloads = [bytes([i % values, 0x10 * (i % values)]) + bytes(6) for i in range(frames)]
    r = MessageRow(key=(0, 0x3E0), label="bus 0 0x3E0", width=8, count=frames, period_ms=100.0,
                   entropy=0.0, kinds=np.zeros(64, dtype=np.uint8))
    r.inference = MessageInference(
        bus=0, address=0x3E0, width=8, frames=frames, bits=None,  # type: ignore[arg-type]
        multiplexor=MultiplexHypothesis(
            0, 8, tuple(range(values)), tuple([frames // values] * values), tuple(range(8, 16)), frames
        ),
    )
    return SegmentModel("p", "r", None, None, [r], payloads={(0, 0x3E0): payloads})


def plain_model() -> SegmentModel:
    r = MessageRow(key=(0, 0x100), label="bus 0 0x100", width=8, count=10, period_ms=100.0,
                   entropy=0.0, kinds=np.zeros(64, dtype=np.uint8))
    return SegmentModel("p", "r", None, None, [r], payloads={(0, 0x100): [bytes(8)] * 10})


class TestLayoutView:
    def test_hidden_until_a_multiplexed_row_is_shown(self, qt_app):
        view = LayoutView()
        assert not view.isVisible()
        view.show_row(plain_model(), 0)
        assert not view.isVisibleTo(view.parentWidget()) and view.isHidden()

    def test_one_row_per_value_labelled_with_its_frame_count(self, qt_app):
        view = LayoutView()
        view.show_row(model_with(3), 0)
        assert not view.isHidden()
        assert view._image.image.shape == (3, 64, 4)
        labels = [text for _pos, text in view.getPlotItem().getAxis("left")._tickLevels[0]]
        assert labels == ["0 (100)", "1 (100)", "2 (100)"]

    def test_height_follows_the_row_count_up_to_a_cap(self, qt_app):
        view = LayoutView()
        view.show_row(model_with(3), 0)
        three = view.height()
        view.show_row(model_with(16, frames=640), 0)
        sixteen = view.height()
        # The bottom axis adds a platform-dependent amount on top of the rows.
        assert 3 * LAYOUT_ROW_PX < three < 3 * LAYOUT_ROW_PX + 80
        assert LAYOUT_MAX_ROWS * LAYOUT_ROW_PX < sixteen < LAYOUT_MAX_ROWS * LAYOUT_ROW_PX + 80
        assert sixteen > three

    def test_showing_a_plain_row_hides_it_again(self, qt_app):
        view = LayoutView()
        view.show_row(model_with(3), 0)
        view.show_row(plain_model(), 0)
        assert view.isHidden()


class TestDetailPanelWiring:
    def test_show_row_drives_the_layout_view(self, qt_app):
        panel = DetailPanel()
        panel.show_row(model_with(2), 0)
        assert not panel.layouts.isHidden()
        panel.show_row(plain_model(), 0)
        assert panel.layouts.isHidden()
