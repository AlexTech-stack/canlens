# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""The Corroborate screen, driven with synthetic consensus -- no decoding."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6", reason="needs the 'gui' extra")
pytest.importorskip("pyqtgraph", reason="needs the 'gui' extra")

from PySide6 import QtWidgets

from canlens.analyze.bits import BitKind, BitOrder, BitProfile
from canlens.corpus import Manifest
from canlens.corpus.local import SEGMENT_FILE
from canlens.corroborate import corroborate
from canlens.gui.corroborate import ALPHA_FLOOR, ALPHA_SPAN, CorroborateScreen
from canlens.infer import MessageInference
from canlens.infer.counters import CounterHypothesis
from canlens.infer.crc16 import Crc16Hypothesis


@pytest.fixture(scope="module")
def qt_app():
    yield QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def profile(kinds):
    n = len(kinds)
    return BitProfile(width=n // 8, frames=100, ones=np.zeros(n), entropy=np.zeros(n),
                      rates=np.zeros(n), kinds=list(kinds), order=BitOrder.INTEL)


def message(key, kinds, counters=(), crc16s=()):
    return MessageInference(bus=key[0], address=key[1], width=len(kinds) // 8, frames=100,
                            bits=profile(kinds), counters=list(counters), crc16s=list(crc16s))


def consensus():
    """Four cars; 0x210 has an established counter, 0x300 a rare bit, 0x900 is extended."""
    still = [BitKind.CONSTANT] * 8
    moved = [BitKind.CONSTANT] * 7 + [BitKind.SLOW]
    ctr = CounterHypothesis(0, 8, 1, 1.0, 100)
    crc = Crc16Hypothesis(0, "e2e_p05", "little", 1.0, 100, 0xFA10)
    obs = []
    for i, device in enumerate("abcd"):
        obs.append((device, [
            message((0, 0x210), [BitKind.NOISY] * 8, counters=[ctr], crc16s=[crc]),
            message((0, 0x300), moved if i == 3 else still),
            message((1, 0x900), [BitKind.ACTIVE] * 8),
        ]))
    return corroborate(obs, platform="TEST_CAR")


@pytest.fixture
def screen(qt_app, tmp_path, monkeypatch):
    layout = {"TEST_CAR": ["dev/route--a/1"], "EMPTY_CAR": ["dev9/route--z/1"]}
    (tmp_path / "database.json").write_text(json.dumps({k: [f"{s}/s" for s in v] for k, v in layout.items()}))
    d = tmp_path / "segments" / "dev" / "route--a" / "1"
    d.mkdir(parents=True)
    (d / SEGMENT_FILE).write_bytes(b"stand-in")
    import canlens.gui.corroborate as module

    monkeypatch.setattr(module, "corroborate_platform", lambda *a, **k: consensus())
    view = CorroborateScreen(str(tmp_path), Manifest.load(str(tmp_path / "database.json")))
    view.resize(1000, 700)
    return view


def row_texts(screen, column=0):
    return [screen.table.item(r, column).text() for r in range(screen.table.rowCount())]


class TestPlatforms:
    def test_only_platforms_with_local_segments_are_offered(self, screen):
        assert [screen.platform.itemData(i) for i in range(screen.platform.count())] == ["TEST_CAR"]

    def test_the_label_says_how_many_are_local(self, screen):
        assert "(1 local)" in screen.platform.currentText()

    def test_refresh_keeps_the_current_choice(self, screen):
        screen.refresh_platforms()
        assert screen.selected_platform() == "TEST_CAR"


class TestRun:
    def test_completion_fills_the_table_by_identifier(self, screen):
        screen._on_complete(consensus())
        assert row_texts(screen) == ["bus 0 0x210", "bus 0 0x300", "bus 1 0x900"]

    def test_the_status_line_reports_segments_and_devices(self, screen):
        screen._on_complete(consensus())
        assert "4 segments from 4 devices" in screen.status.text()

    def test_findings_column_carries_the_tier(self, screen):
        screen._on_complete(consensus())
        assert row_texts(screen, 6)[0] == "ctr 8bit@0 [established], e2e_p05@0/idFA10 [established]"

    def test_rare_column(self, screen):
        screen._on_complete(consensus())
        assert row_texts(screen, 5) == ["0", "1", "0"]

    def test_the_worker_path_end_to_end(self, screen, qt_app):
        screen._run()
        assert screen._worker is not None
        assert screen._worker.wait(5000)
        for _ in range(5):
            qt_app.processEvents()
        assert screen.table.rowCount() == 3
        assert screen.run_button.isEnabled()

    def test_a_failure_is_reported_not_raised(self, screen, monkeypatch):
        import canlens.gui.corroborate as module

        def boom(*a, **k):
            raise RuntimeError("no such cache")

        monkeypatch.setattr(module, "corroborate_platform", boom)
        screen._run()
        assert screen._worker.wait(5000)
        for _ in range(5):
            QtWidgets.QApplication.processEvents()
        assert "failed" in screen.status.text() and "no such cache" in screen.status.text()
        assert screen.run_button.isEnabled()


class TestFilter:
    def test_filters_by_identifier_prefix(self, screen):
        screen._on_complete(consensus())
        screen.filter_field.setText("*,0x2*")
        screen._apply_filter()
        assert row_texts(screen) == ["bus 0 0x210"]

    def test_filters_by_bus(self, screen):
        screen._on_complete(consensus())
        screen.filter_field.setText("CAN1,*")
        screen._apply_filter()
        assert row_texts(screen) == ["bus 1 0x900"]

    def test_clearing_restores_everything(self, screen):
        screen._on_complete(consensus())
        screen.filter_field.setText("CAN1,*")
        screen._apply_filter()
        screen.filter_field.clear()
        screen._apply_filter()
        assert screen.table.rowCount() == 3


class TestDetail:
    def test_selecting_a_row_shows_its_hypotheses(self, screen):
        screen._on_complete(consensus())
        screen.table.selectRow(0)
        text = screen.findings.toPlainText()
        assert "8-bit counter @ bit 0" in text and "established" in text
        assert "data ID 0xFA10" in text

    def test_rare_bits_are_listed(self, screen):
        screen._on_complete(consensus())
        screen.table.selectRow(1)
        assert "rare bits: [7]" in screen.findings.toPlainText()

    def test_heading_carries_the_device_counts(self, screen):
        screen._on_complete(consensus())
        screen.table.selectRow(0)
        assert "4/4 segments, 4/4 devices" in screen.heading.text()

    def test_the_strip_fades_by_agreement_and_outlines_rare_bits(self, screen):
        screen._on_complete(consensus())
        screen.table.selectRow(1)   # 0x300: bit 7 rare, 75% agreement
        image = screen.strip._image.image
        assert image.shape == (1, 8, 4)
        assert image[0, 0, 3] == ALPHA_FLOOR + ALPHA_SPAN          # unanimous bit: solid
        assert image[0, 7, 3] == int(ALPHA_FLOOR + ALPHA_SPAN * 0.75)
        assert len(screen.strip._marks) == 1

    def test_a_message_with_nothing_says_so(self, screen):
        screen._on_complete(consensus())
        screen.table.selectRow(2)
        assert "no counter or checksum" in screen.findings.toPlainText()


class TestNavigation:
    def test_the_window_gains_a_third_tab(self, qt_app, screen, tmp_path):
        from canlens.gui.window import CORROBORATE_TAB, Workbench

        window = Workbench(str(tmp_path))
        try:
            labels = [window.screens.tabText(i) for i in range(window.screens.count())]
            assert labels == ["Data", "Heat Map", "Corroborate"]
            assert window.screens.tabText(CORROBORATE_TAB) == "Corroborate"
        finally:
            window.close()
