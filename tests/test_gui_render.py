# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Rendering regressions that only show up once pixels exist.

Runs headless via the offscreen Qt platform, the same way BoAt verifies its
own Qt client, and skips entirely when the gui extra is not installed.
"""
from __future__ import annotations

import os
from itertools import pairwise

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
        # Matrix, strip, and the per-value layout rows.
        assert source.count("levels=(0, 255)") == 3


class TestBitSelection:
    """Dragging a range must describe whole bits and nothing else."""

    @pytest.fixture
    def strip(self, qt_app):
        from canlens.gui.model import MessageRow, SegmentModel
        from canlens.gui.window import BitStripView

        key = (0, 0x100)
        payloads = [bytes([i, 0]) for i in range(64)]
        row = MessageRow(
            key=key, label="bus 0 0x100", width=2, count=64,
            period_ms=10.0, entropy=1.0, kinds=np.zeros(16, dtype=np.uint8),
        )
        model = SegmentModel("p", "r", None, None, [row], payloads={key: payloads})
        view = BitStripView()
        view.show_row(model, 0)
        return view

    def test_fractional_drags_snap_to_whole_bits(self, strip):
        strip.selector.setRegion((3.7, 8.2))
        assert strip.selector.getRegion() == (4, 8)

    def test_emits_start_and_length_not_start_and_end(self, strip):
        seen = []
        strip.selection_changed.connect(lambda s, n: seen.append((s, n)))
        strip.selector.setRegion((2.4, 9.6))
        # Snaps to bits 2..10 exclusive, which is a length of 8.
        assert strip.selector.getRegion() == (2, 10)
        assert seen[-1] == (2, 8)

    def test_a_collapsed_selection_keeps_one_bit(self, strip):
        strip.selector.setRegion((5.0, 5.0))
        start, end = strip.selector.getRegion()
        assert end - start == 1

    def test_selection_cannot_leave_the_payload(self, strip):
        strip.selector.setRegion((-20, 500))
        start, end = strip.selector.getRegion()
        assert start >= 0 and end <= 16

    def test_set_selection_round_trips(self, strip):
        strip.set_selection(3, 5)
        assert strip.selector.getRegion() == (3, 8)


class TestLabelLanes:
    """Field labels must never overlap each other, nor cover the bits."""

    @pytest.fixture
    def strip(self, qt_app):
        from canlens.analyze import analyze_frames
        from canlens.decode import CanFrame
        from canlens.gui.model import MessageRow, SegmentModel
        from canlens.gui.palette import kinds_to_indices
        from canlens.gui.window import BitStripView

        # A 32-byte CAN FD payload: 256 bits across a few hundred pixels is
        # where labels genuinely collide, and it is what the EV6 looks like.
        key = (0, 0x210)
        width = 32
        payloads = [
            bytes((i * (k + 1)) & 0xFF for k in range(width)) for i in range(128)
        ]
        frames = [CanFrame(i * 10_000_000, *key, p, False) for i, p in enumerate(payloads)]
        profile = analyze_frames(frames)
        row = MessageRow(
            key=key, label="bus 0 0x210", width=width, count=len(payloads),
            period_ms=10.0, entropy=1.0,
            kinds=kinds_to_indices(profile[key].bits.kinds),
        )
        model = SegmentModel("p", "r", None, profile, [row], payloads={key: payloads})
        # Deliberately crowded: long names a few bits apart.
        for name, start, length in (
            ("VehicleSpeed", 0, 4), ("SteeringAngle", 5, 4),
            ("BrakePressure", 10, 4), ("GearPosition", 16, 4),
        ):
            model.name_selection(0, name, start, length)
        view = BitStripView()
        view.resize(900, 90)
        view.show()
        view.show_row(model, 0)
        # Geometry only becomes real once the widget is laid out: setFixedHeight
        # on an unshown widget changes the request, not height().
        for _ in range(3):
            qt_app.processEvents()
        view.show_row(model, 0)
        for _ in range(3):
            qt_app.processEvents()
        return view

    @staticmethod
    def lanes(strip) -> dict[int, list[tuple[float, float]]]:
        """Label extents in bits, grouped by the lane they were placed in."""
        from canlens.gui.window import LABEL_TOP, LANE_HEIGHT

        (left, right), _ = strip.getPlotItem().vb.viewRange()
        per_bit = max(strip.getPlotItem().vb.width(), 1) / (right - left)
        out: dict[int, list[tuple[float, float]]] = {}
        for item, start in strip._labels:
            lane = round((LABEL_TOP - item.pos().y()) / LANE_HEIGHT)
            width = item.boundingRect().width() / per_bit
            out.setdefault(lane, []).append((start, start + width))
        return out

    def test_no_two_labels_overlap_within_a_lane(self, strip):
        for lane, extents in self.lanes(strip).items():
            extents.sort()
            for (_, end), (next_start, _) in pairwise(extents):
                assert next_start >= end, f"overlap in lane {lane}"

    def test_crowded_fields_need_more_than_one_lane(self, strip):
        # At roughly three pixels per bit a name is twenty-odd bits wide, so
        # fields five bits apart cannot share a line.
        assert len(self.lanes(strip)) > 1

    def test_labels_sit_below_the_strip_not_on_it(self, strip):
        # The bits occupy y 0..1; a label over them hides what it describes.
        for item, _ in strip._labels:
            assert item.pos().y() < 0

    def test_every_field_still_gets_a_label(self, strip):
        names = {item.toPlainText() for item, _ in strip._labels}
        assert {"VehicleSpeed", "SteeringAngle", "BrakePressure", "GearPosition"} <= names

    def test_zooming_in_frees_lanes(self, strip):
        before = len(self.lanes(strip))
        strip.setXRange(0, 24, padding=0)   # far more pixels per bit
        after = len(self.lanes(strip))
        assert after < before
        for lane, extents in self.lanes(strip).items():
            extents.sort()
            for (_, end), (next_start, _) in pairwise(extents):
                assert next_start >= end, f"overlap after zoom in lane {lane}"

    def test_the_height_is_fixed_not_merely_capped(self, strip):
        # A maximum alone lets the layout hand back less than asked for, and
        # the lanes squash back on top of each other.
        assert strip.minimumHeight() == strip.maximumHeight()

    def test_the_height_tracks_the_lane_count(self, qt_app, strip):
        from canlens.gui.window import LANE_PIXELS

        crowded_lanes, crowded_height = strip._lanes, strip.minimumHeight()
        strip.setXRange(0, 24, padding=0)   # more pixels per bit, fewer lanes
        for _ in range(3):
            qt_app.processEvents()
        assert strip._lanes < crowded_lanes
        assert strip.minimumHeight() < crowded_height
        assert crowded_height - strip.minimumHeight() == pytest.approx(
            (crowded_lanes - strip._lanes) * LANE_PIXELS, abs=2
        )

    def test_a_lane_is_always_the_same_number_of_pixels(self, strip):
        from canlens.gui.window import LANE_HEIGHT, LANE_PIXELS

        view = strip.getPlotItem().vb
        bottom, top = view.viewRange()[1]
        per_unit = max(view.height(), 1) / (top - bottom)
        assert LANE_HEIGHT * per_unit == pytest.approx(LANE_PIXELS, rel=0.15)

    def test_a_message_without_fields_uses_a_single_lane(self, qt_app, strip):
        from canlens.analyze import analyze_frames
        from canlens.decode import CanFrame
        from canlens.gui.model import MessageRow, SegmentModel
        from canlens.gui.palette import kinds_to_indices
        from canlens.gui.window import BitStripView

        key = (0, 0x111)
        payloads = [b"\x00\x00"] * 16
        frames = [CanFrame(i * 10_000_000, *key, p, False) for i, p in enumerate(payloads)]
        profile = analyze_frames(frames)
        row = MessageRow(
            key=key, label="x", width=2, count=16, period_ms=10.0, entropy=0.0,
            kinds=kinds_to_indices(profile[key].bits.kinds),
        )
        view = BitStripView()
        view.show_row(SegmentModel("p", "r", None, profile, [row], payloads={key: payloads}), 0)
        assert view._labels == []
        assert view._lanes == 1
        assert view.minimumHeight() < strip.minimumHeight()

    def test_the_selection_band_does_not_cover_the_labels(self, strip):
        # The band must stop at the bits; spanning the whole view strikes
        # through whichever name sits beneath it.
        low, high = strip.selector.span
        bottom, top = strip.getPlotItem().vb.viewRange()[1]
        assert bottom + low * (top - bottom) == pytest.approx(0.0, abs=0.05)
        assert bottom + high * (top - bottom) == pytest.approx(1.0, abs=0.05)
