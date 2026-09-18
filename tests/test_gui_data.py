# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""The Data screen and the navigation bar.

Builds a throwaway corpus under tmp_path; nothing here touches a real one.
"""
from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="needs the 'gui' extra")
pytest.importorskip("pyqtgraph", reason="needs the 'gui' extra")

from PySide6 import QtWidgets

from canlens.corpus import Manifest
from canlens.corpus.local import SEGMENT_FILE
from canlens.gui.data import DataScreen


@pytest.fixture(scope="module")
def qt_app():
    yield QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def screen(qt_app, tmp_path):
    layout = {
        "TOYOTA_PRIUS": ["dev0/route0--aa/1", "dev0/route0--aa/2"],
        "KIA_EV6": ["dev1/route1--bb/7"],
        "HONDA_ACCORD": ["dev9/route9--zz/3"],   # nothing local
    }
    (tmp_path / "database.json").write_text(
        json.dumps({k: [f"{s}/s" for s in v] for k, v in layout.items()})
    )
    for key in ("TOYOTA_PRIUS", "KIA_EV6"):
        for rel in layout[key]:
            directory = tmp_path / "segments" / rel
            directory.mkdir(parents=True)
            (directory / SEGMENT_FILE).write_bytes(b"x" * 2048)
    manifest = Manifest.load(str(tmp_path / "database.json"))
    view = DataScreen(str(tmp_path), manifest)
    view.resize(900, 600)
    return view


def row_for(screen, platform: str) -> int:
    for row in range(screen.table.rowCount()):
        if screen.table.item(row, 0).text() == platform:
            return row
    raise AssertionError(f"{platform} not listed")


class TestListing:
    def test_lists_every_platform_in_the_manifest(self, screen):
        assert screen.table.rowCount() == 3

    def test_platforms_held_locally_come_first(self, screen):
        names = [screen.table.item(r, 0).text() for r in range(screen.table.rowCount())]
        assert names.index("TOYOTA_PRIUS") < names.index("HONDA_ACCORD")
        assert names.index("KIA_EV6") < names.index("HONDA_ACCORD")

    def test_local_counts_are_shown(self, screen):
        assert screen.table.item(row_for(screen, "TOYOTA_PRIUS"), 3).text() == "2"

    def test_platforms_with_nothing_local_show_no_count(self, screen):
        assert screen.table.item(row_for(screen, "HONDA_ACCORD"), 3).text() == ""

    def test_the_summary_reports_holdings_and_free_space(self, screen):
        text = screen.summary.text()
        assert "3 platforms" in text and "3 local" in text and "free" in text


class TestSearch:
    def test_wildcards_apply_to_the_platform_name(self, screen, qt_app):
        screen.search.setText("*EV6")
        assert [screen.table.item(r, 0).text() for r in range(screen.table.rowCount())] == [
            "KIA_EV6"
        ]

    def test_a_prefix_pattern(self, screen):
        screen.search.setText("TOYOTA*")
        assert screen.table.rowCount() == 1

    def test_clearing_restores_everything(self, screen):
        screen.search.setText("*EV6")
        screen.search.clear()
        assert screen.table.rowCount() == 3

    def test_matching_ignores_case(self, screen):
        screen.search.setText("kia*")
        assert screen.table.rowCount() == 1


class TestSelection:
    def test_selecting_a_platform_lists_its_local_segments(self, screen):
        screen.table.selectRow(row_for(screen, "TOYOTA_PRIUS"))
        assert screen.segments.count() == 2

    def test_a_platform_with_nothing_local_lists_none(self, screen):
        screen.table.selectRow(row_for(screen, "HONDA_ACCORD"))
        assert screen.segments.count() == 0

    def test_double_clicking_a_segment_asks_for_it_to_be_opened(self, screen):
        screen.table.selectRow(row_for(screen, "KIA_EV6"))
        seen = []
        screen.open_segment.connect(seen.append)
        screen._open(screen.segments.item(0))
        assert seen and seen[0].endswith(SEGMENT_FILE)


class TestDeleteTargets:
    def test_a_selected_platform_targets_all_of_its_segments(self, screen):
        screen.table.selectRow(row_for(screen, "TOYOTA_PRIUS"))
        assert len(screen.targets_for_delete()) == 2

    def test_selected_segments_win_over_the_platform(self, screen):
        # So that a handful can be dropped without emptying the platform.
        screen.table.selectRow(row_for(screen, "TOYOTA_PRIUS"))
        screen.segments.setCurrentRow(0)
        assert len(screen.targets_for_delete()) == 1

    def test_nothing_selected_targets_nothing(self, screen):
        assert screen.targets_for_delete() == []

    def test_deleting_updates_the_listing(self, screen):
        screen.table.selectRow(row_for(screen, "KIA_EV6"))
        targets = screen.targets_for_delete()
        from canlens.corpus import delete_segments

        delete_segments(targets)
        screen.reload()
        assert screen.table.item(row_for(screen, "KIA_EV6"), 3).text() == ""


class TestFetchGuards:
    def test_fetching_with_no_selection_says_so(self, screen):
        screen._fetch()
        assert "select" in screen.status.text().lower()

    def test_a_fetch_larger_than_the_disk_is_refused(self, screen, monkeypatch):
        import canlens.gui.data as module

        monkeypatch.setattr(module, "disk_free", lambda root: 1)
        screen.table.selectRow(row_for(screen, "HONDA_ACCORD"))
        screen._fetch()
        assert "refusing" in screen.status.text()
        assert screen._worker is None


class TestNavigationBar:
    @pytest.fixture
    def window(self, qt_app, screen, tmp_path):
        from canlens.gui.window import Workbench

        view = Workbench(str(tmp_path))
        yield view
        view.close()

    def test_both_screens_are_present_with_data_first(self, window):
        from canlens.gui.window import DATA_TAB

        labels = [window.screens.tabText(i) for i in range(window.screens.count())]
        assert labels == ["Data", "Heat Map"]
        assert window.screens.currentIndex() == DATA_TAB

    def test_opening_the_window_decodes_nothing(self, window):
        # Selecting a row is not a request to decode it. This regressed once:
        # setCurrentRow fires currentRowChanged, which decoded a segment and
        # fetched the capnp schemas before anyone had opened the heat map --
        # 73 seconds of a test suite that now takes half of one.
        assert window._model is None

    def test_opening_the_window_touches_the_network_for_nothing(
        self, qt_app, screen, tmp_path, monkeypatch
    ):
        # `screen` builds the corpus, so the manifest is already on disk and
        # any remaining request would be a decode fetching capnp schemas.
        import urllib.request

        from canlens.gui.window import Workbench

        def refuse(*args, **kwargs):
            raise AssertionError("constructing the window must not fetch anything")

        monkeypatch.setattr(urllib.request, "urlopen", refuse)
        view = Workbench(str(tmp_path))
        try:
            assert view._model is None
        finally:
            view.close()

    def test_the_segment_list_is_populated_even_though_nothing_is_decoded(self, window):
        assert window.segments.count() == 3


class TestColumnWidths:
    """Refreshing must not shove the columns against the left edge."""

    def test_widths_survive_a_refresh(self, screen, qt_app):
        screen.show()
        for _ in range(3):
            qt_app.processEvents()
        before = [screen.table.columnWidth(c) for c in range(screen.table.columnCount())]
        screen.reload()
        for _ in range(3):
            qt_app.processEvents()
        assert [screen.table.columnWidth(c) for c in range(screen.table.columnCount())] == before

    def test_the_columns_fill_the_table(self, screen, qt_app):
        screen.show()
        for _ in range(3):
            qt_app.processEvents()
        screen.reload()
        for _ in range(3):
            qt_app.processEvents()
        total = sum(screen.table.columnWidth(c) for c in range(screen.table.columnCount()))
        assert abs(total - screen.table.viewport().width()) <= 2

    def test_the_name_column_takes_the_slack(self, screen, qt_app):
        screen.show()
        for _ in range(3):
            qt_app.processEvents()
        widths = [screen.table.columnWidth(c) for c in range(screen.table.columnCount())]
        assert widths[0] > sum(widths[1:])


class TestRefreshKeepsContext:
    def test_the_selected_platform_survives_a_refresh(self, screen):
        screen.table.selectRow(row_for(screen, "TOYOTA_PRIUS"))
        screen.reload()
        assert screen.selected_platforms() == ["TOYOTA_PRIUS"]

    def test_the_segment_list_follows_a_delete(self, screen):
        from canlens.corpus import delete_segments

        screen.table.selectRow(row_for(screen, "TOYOTA_PRIUS"))
        assert screen.segments.count() == 2
        delete_segments(screen.targets_for_delete())
        screen.reload()
        # It used to keep listing segments that had just been deleted.
        assert screen.segments.count() == 0


class TestHeatMapFollowsTheCorpus:
    @pytest.fixture
    def window(self, qt_app, screen, tmp_path):
        from canlens.gui.window import Workbench

        view = Workbench(str(tmp_path))
        yield view
        view.close()

    def test_deleted_segments_leave_the_heat_map_list(self, window, screen):
        from canlens.corpus import delete_segments

        before = window.segments.count()
        screen.table.selectRow(row_for(screen, "TOYOTA_PRIUS"))
        delete_segments(screen.targets_for_delete())
        window.data.reload()
        # Before the fix the list kept them, and opening one raised
        # FileNotFoundError from deep inside the decoder.
        assert window.segments.count() == before - 2

    @staticmethod
    def _on_heat_map(window):
        from canlens.gui.window import HEAT_MAP_TAB

        # Reach the tab without the tab-change hook opening a segment first.
        window.screens.blockSignals(True)
        window.screens.setCurrentIndex(HEAT_MAP_TAB)
        window.screens.blockSignals(False)

    def test_opening_a_segment_that_is_gone_reports_instead_of_raising(self, window):
        os.unlink(window._paths[0][1])   # vanished without the Data screen knowing
        self._on_heat_map(window)
        window._open_selected(0)
        assert "no longer on disk" in window.statusBar().currentMessage()

    def test_a_vanished_segment_is_dropped_from_the_list(self, window):
        before = window.segments.count()
        os.unlink(window._paths[0][1])
        self._on_heat_map(window)
        window._open_selected(0)
        assert window.segments.count() == before - 1

    def test_a_file_that_is_not_a_segment_reports_instead_of_raising(self, window):
        # The fixture writes filler bytes, not real zstd. A corrupt or
        # truncated download raises from inside zstd, which is not an OSError.
        self._on_heat_map(window)
        window._open_selected(0)
        assert "could not read" in window.statusBar().currentMessage()
        assert window._model is None
