# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""The Data screen: what the corpus holds, and what of it is on this disk.

Everything `canlens corpus` does from the terminal -- list, plan, fetch,
delete, and say what is local -- with the addition that a fetch here runs on a
worker thread. Downloading a platform takes minutes; doing that on the Qt
thread would freeze the window for the duration and make cancelling
impossible, which is precisely when someone wants to cancel.
"""
from __future__ import annotations

import os

from PySide6 import QtCore, QtWidgets

from ..corpus import (
    AVG_SEGMENT_MB,
    FetchResult,
    LocalPlatform,
    Manifest,
    delete_segments,
    disk_free,
    fetch_all,
    inventory,
    select,
)
from ..filters import matches

COLUMNS = ("Platform", "Segments", "Full size", "Local", "Local size")


class FetchWorker(QtCore.QThread):
    """Runs a fetch off the UI thread, reporting progress as it goes."""

    progressed = QtCore.Signal(int, int, float)
    completed = QtCore.Signal(object)

    def __init__(self, segments: list[str], root: str, jobs: int = 8) -> None:
        super().__init__()
        self._segments = segments
        self._root = root
        self._jobs = jobs
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        def progress(done: int, total: int, result: FetchResult) -> None:
            self.progressed.emit(done, total, result.bytes_new / 2**30)

        result = fetch_all(
            self._segments,
            self._root,
            jobs=self._jobs,
            progress=progress,
            should_stop=lambda: self._stop,
            report_every=5,
        )
        self.completed.emit(result)


class CacheWorker(QtCore.QThread):
    """Warms both caches for a list of segments on a process pool.

    A QThread only hosts the pool and relays progress; the decoding and
    inference happen in worker processes, one segment each, so the window
    stays responsive and all physical cores are used.
    """

    progressed = QtCore.Signal(int, int)
    completed = QtCore.Signal(int, int)

    def __init__(self, paths: list[str], root: str, jobs: int) -> None:
        super().__init__()
        self._paths = paths
        self._root = root
        self._jobs = jobs
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        from ..infer import warm_segment

        done = failed = 0
        with ProcessPoolExecutor(max_workers=self._jobs) as pool:
            futures = [pool.submit(warm_segment, path, self._root) for path in self._paths]
            for future in as_completed(futures):
                done += 1
                try:
                    future.result()
                except Exception:  # noqa: BLE001 - one bad file must not stop the rest
                    failed += 1
                self.progressed.emit(done, len(self._paths))
                if self._stop:
                    for pending in futures:
                        pending.cancel()
                    break
        self.completed.emit(done - failed, failed)


class DataScreen(QtWidgets.QWidget):
    """Browse the corpus, fetch what is wanted, delete what is not."""

    open_segment = QtCore.Signal(str)
    corpus_changed = QtCore.Signal()

    def __init__(self, root: str, manifest: Manifest) -> None:
        super().__init__()
        self.root = root
        self.manifest = manifest
        self._local: dict[str, LocalPlatform] = {}
        self._worker: FetchWorker | None = None
        self._cache_worker: CacheWorker | None = None

        layout = QtWidgets.QVBoxLayout(self)

        self.summary = QtWidgets.QLabel()
        self.summary.setStyleSheet("font-weight: 600;")
        layout.addWidget(self.summary)

        controls = QtWidgets.QHBoxLayout()
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("filter platforms, e.g. TOYOTA* or *EV6")
        self.search.textChanged.connect(self._refill)
        controls.addWidget(self.search, stretch=1)
        controls.addWidget(QtWidgets.QLabel("segments"))
        self.limit = QtWidgets.QSpinBox()
        self.limit.setRange(0, 20000)
        self.limit.setValue(200)
        self.limit.setSpecialValueText("all")
        self.limit.setToolTip("how many segments per platform to fetch; 0 means all")
        controls.addWidget(self.limit)
        self.fetch_button = QtWidgets.QPushButton("Fetch")
        self.fetch_button.clicked.connect(self._fetch)
        controls.addWidget(self.fetch_button)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)
        controls.addWidget(self.cancel_button)
        self.delete_button = QtWidgets.QPushButton("Delete local")
        self.delete_button.clicked.connect(self._delete)
        controls.addWidget(self.delete_button)
        self.cache_button = QtWidgets.QPushButton("Build cache")
        self.cache_button.setToolTip(
            "decode and infer the selected platforms' local segments once, "
            "so opening them later is instant"
        )
        self.cache_button.clicked.connect(self._build_cache)
        controls.addWidget(self.cache_button)
        refresh = QtWidgets.QPushButton("Refresh")
        refresh.clicked.connect(self.reload)
        controls.addWidget(refresh)
        layout.addLayout(controls)

        self.table = QtWidgets.QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        # Resize modes are set once, per column, and never followed by
        # resizeColumnsToContents(): that call overrides the modes, which is
        # why refreshing used to shove every column against the left edge and
        # leave the rest of the row empty. The name absorbs the slack; the
        # numbers size to their contents and stay put.
        head = self.table.horizontalHeader()
        head.setStretchLastSection(False)
        head.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for column in range(1, len(COLUMNS)):
            head.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._show_segments)
        layout.addWidget(self.table, stretch=3)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.status = QtWidgets.QLabel("—")
        layout.addWidget(self.status)

        layout.addWidget(QtWidgets.QLabel("local segments  (double-click to open)"))
        self.segments = QtWidgets.QListWidget()
        self.segments.itemDoubleClicked.connect(self._open)
        layout.addWidget(self.segments, stretch=2)

        self.reload()

    # ----- data -------------------------------------------------------------

    def reload(self) -> None:
        self._local = inventory(self.root, self.manifest)
        self._refill()
        self.corpus_changed.emit()

    def _rows(self) -> list[tuple[str, int, float, int, float]]:
        """Platform rows under the current search, local holdings first."""
        pattern = self.search.text().strip()
        rows = []
        for key in self.manifest:
            if not matches(pattern, key):
                continue
            platform = self.manifest[key]
            held = self._local.get(key)
            rows.append(
                (
                    key,
                    platform.count,
                    platform.count * AVG_SEGMENT_MB / 1024,
                    held.count if held else 0,
                    held.gib if held else 0.0,
                )
            )
        rows.sort(key=lambda r: (-r[3], -r[1]))
        return rows

    def _refill(self) -> None:
        # Rebuilding the table drops the selection, which would leave the
        # segment list below showing a platform that is no longer selected --
        # and, after a delete, segments that no longer exist.
        previously = set(self.selected_platforms())
        rows = self._rows()
        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))
        for index, (key, count, gib, local, local_gib) in enumerate(rows):
            cells = (
                key,
                f"{count}",
                f"{gib:.2f} GB",
                f"{local}" if local else "",
                f"{local_gib:.2f} GiB" if local else "",
            )
            for column, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                if column:
                    item.setTextAlignment(
                        QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
                    )
                self.table.setItem(index, column, item)
        self.table.blockSignals(False)
        for index, row in enumerate(rows):
            if row[0] in previously:
                self.table.selectRow(index)
        self._show_segments()
        self._update_summary()

    def _update_summary(self) -> None:
        held = sum(p.count for p in self._local.values())
        used = sum(p.bytes_used for p in self._local.values())
        self.summary.setText(
            f"{len(self.manifest)} platforms, {self.manifest.total_segments} segments upstream  ·  "
            f"{held} local ({used / 2**30:.2f} GiB)  ·  "
            f"{disk_free(self.root) / 2**30:.0f} GiB free in {self.root}"
        )

    def selected_platforms(self) -> list[str]:
        rows = {index.row() for index in self.table.selectedIndexes()}
        return [
            item.text()
            for item in (self.table.item(row, 0) for row in sorted(rows))
            if item is not None
        ]

    def _show_segments(self) -> None:
        self.segments.clear()
        for key in self.selected_platforms():
            held = self._local.get(key)
            for path in held.paths if held else []:
                item = QtWidgets.QListWidgetItem(
                    f"{key}   " + "/".join(path.split("/")[-4:-1])
                )
                item.setData(QtCore.Qt.ItemDataRole.UserRole, path)
                self.segments.addItem(item)

    def _open(self, item: QtWidgets.QListWidgetItem) -> None:
        self.open_segment.emit(item.data(QtCore.Qt.ItemDataRole.UserRole))

    # ----- fetching ---------------------------------------------------------

    def _fetch(self) -> None:
        platforms = self.selected_platforms()
        if not platforms:
            self.status.setText("select one or more platforms first")
            return
        limit = self.limit.value() or None
        segments = select(self.manifest, platforms, limit)
        estimate = len(segments) * AVG_SEGMENT_MB / 1024
        free = disk_free(self.root) / 2**30
        if estimate > free:
            self.status.setText(
                f"refusing: {estimate:.1f} GB wanted, only {free:.1f} GiB free"
            )
            return

        self.status.setText(
            f"fetching {len(segments)} segments (~{estimate:.2f} GB) "
            f"from {len(platforms)} platform(s)"
        )
        self.progress.setRange(0, len(segments))
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self._set_busy(True)

        self._worker = FetchWorker(segments, self.root)
        self._worker.progressed.connect(self._on_progress)
        self._worker.completed.connect(self._on_complete)
        self._worker.start()

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self.status.setText("cancelling — letting transfers in flight finish")
        if self._cache_worker is not None:
            self._cache_worker.stop()
            self.status.setText("cancelling — letting segments in progress finish")

    # ----- caching ----------------------------------------------------------

    def _build_cache(self) -> None:
        paths: list[str] = []
        for key in self.selected_platforms():
            held = self._local.get(key)
            if held:
                paths += held.paths
        if not paths:
            self.status.setText("select platforms with local segments first")
            return
        from ..cli import default_jobs

        jobs = default_jobs()
        self.status.setText(f"caching {len(paths)} segments on {jobs} workers")
        self.progress.setRange(0, len(paths))
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self._set_busy(True)
        self._cache_worker = CacheWorker(paths, self.root, jobs)
        self._cache_worker.progressed.connect(self._on_cache_progress)
        self._cache_worker.completed.connect(self._on_cache_complete)
        self._cache_worker.start()

    def _on_cache_progress(self, done: int, total: int) -> None:
        self.progress.setValue(done)
        self.status.setText(f"cached {done}/{total} segments")

    def _on_cache_complete(self, ok: int, failed: int) -> None:
        self._set_busy(False)
        self.progress.setVisible(False)
        self._cache_worker = None
        self.status.setText(
            f"cached {ok} segments" + (f", {failed} failed" if failed else "")
        )

    def _on_progress(self, done: int, total: int, gib: float) -> None:
        self.progress.setValue(done)
        self.status.setText(f"{done}/{total} segments, {gib:.2f} GiB downloaded")

    def _on_complete(self, result: FetchResult) -> None:
        self._set_busy(False)
        self.progress.setVisible(False)
        self._worker = None
        parts = [f"{result.fetched} fetched", f"{result.skipped} already present"]
        if result.failed:
            parts.append(f"{result.failed} failed")
        if result.cancelled:
            parts.append("cancelled")
        self.status.setText(", ".join(parts) + f" — {result.bytes_new / 2**30:.2f} GiB new")
        self.reload()

    def _set_busy(self, busy: bool) -> None:
        self.fetch_button.setEnabled(not busy)
        self.delete_button.setEnabled(not busy)
        self.cache_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)

    # ----- deleting ---------------------------------------------------------

    def targets_for_delete(self) -> list[str]:
        """Local paths the current selection would remove.

        Selected segments win over selected platforms, so it is possible to
        drop a handful without emptying the platform they belong to.
        """
        chosen = [
            item.data(QtCore.Qt.ItemDataRole.UserRole)
            for item in self.segments.selectedItems()
        ]
        if chosen:
            return chosen
        paths: list[str] = []
        for key in self.selected_platforms():
            held = self._local.get(key)
            if held:
                paths += held.paths
        return paths

    def _delete(self) -> None:
        targets = self.targets_for_delete()
        if not targets:
            self.status.setText("nothing local in the selection to delete")
            return
        size = sum(os.path.getsize(p) for p in targets if os.path.exists(p))
        answer = QtWidgets.QMessageBox.question(
            self,
            "Delete local segments",
            f"Delete {len(targets)} segment(s), freeing {size / 2**30:.2f} GiB?\n\n"
            f"They can be fetched again from the bucket.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            self.status.setText("delete cancelled")
            return
        result = delete_segments(targets)
        self.status.setText(
            f"deleted {result.deleted}, freed {result.bytes_freed / 2**30:.2f} GiB"
            + (f", {len(result.failed)} failed" if result.failed else "")
        )
        self.reload()
