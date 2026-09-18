# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""The Corroborate screen: what holds across every local segment of a platform.

The heat map shows one trace. This shows what the corpus agrees on -- each
inferred hypothesis with the number of segments and, more importantly, the
number of distinct cars behind it -- and the bits that only many traces can
reveal: constant almost everywhere, moving somewhere.

Corroboration reads the inference results cache, so it is milliseconds per
segment once `cache build` has run, but a platform can have thousands of
segments; the pass runs on a worker thread with a progress bar.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..corpus import Manifest, inventory
from ..corroborate import MessageConsensus, PlatformConsensus, corroborate_platform
from ..filters import SEPARATORS, TraceFilter
from .axes import byte_ticks
from .palette import kinds_to_indices, lookup_table

COLUMNS = ("Message", "Seg", "Dev", "Width", "Agree", "Rare", "Findings")

RARE_RGB = (255, 159, 67)
# Opacity range for the consensus strip: a bit every segment agrees on is
# solid, one the platform splits over fades toward the background.
ALPHA_FLOOR, ALPHA_SPAN = 70, 185


class CorroborateWorker(QtCore.QThread):
    progressed = QtCore.Signal(int, int)
    completed = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, platform: str, root: str, limit: int | None) -> None:
        super().__init__()
        self._platform, self._root, self._limit = platform, root, limit

    def run(self) -> None:
        try:
            result = corroborate_platform(
                self._platform,
                root=self._root,
                limit=self._limit,
                progress=lambda done, total: self.progressed.emit(done, total),
            )
        except Exception as exc:  # noqa: BLE001 - reported to the status line, not the console
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.completed.emit(result)


class ConsensusStrip(pg.PlotWidget):
    """One message's consensus bits: class as colour, agreement as opacity.

    Rare bits -- constant in most segments, moving in some -- are outlined, since
    their consensus colour is the constant white that hides exactly what makes
    them interesting.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setMenuEnabled(False)
        self.setFixedHeight(96)
        self.getPlotItem().hideAxis("left")
        self.getPlotItem().setLabel("bottom", "payload bit")
        self._image = pg.ImageItem(axisOrder="row-major")
        self.addItem(self._image)
        self._marks: list[QtWidgets.QGraphicsRectItem] = []

    def show_message(self, message: MessageConsensus) -> None:
        for mark in self._marks:
            self.removeItem(mark)
        self._marks.clear()

        table = lookup_table()
        indices = kinds_to_indices(message.kinds)
        agreement = np.array([b.agreement for b in message.bits])
        rgba = np.zeros((1, len(indices), 4), dtype=np.ubyte)
        rgba[0, :, :3] = table[indices]
        rgba[0, :, 3] = (ALPHA_FLOOR + ALPHA_SPAN * agreement).astype(np.ubyte)
        # levels pinned: a uniform payload otherwise renders black (see window.py)
        self._image.setImage(rgba, levels=(0, 255))

        for bit in message.rare_bits:
            box = QtWidgets.QGraphicsRectItem(bit, -0.06, 1, 1.12)
            box.setPen(pg.mkPen(RARE_RGB, width=2))
            box.setBrush(pg.mkBrush(None))
            box.setZValue(5)
            self.addItem(box)
            self._marks.append(box)

        bits = len(indices)
        self.getPlotItem().getAxis("bottom").setTicks(byte_ticks(bits))
        self.getPlotItem().setLimits(xMin=-1, xMax=bits + 1, yMin=-0.5, yMax=1.5)
        self.setXRange(0, bits, padding=0.01)
        self.setYRange(-0.3, 1.3, padding=0)


class CorroborateScreen(QtWidgets.QWidget):
    def __init__(self, root: str, manifest: Manifest) -> None:
        super().__init__()
        self.root = root
        self.manifest = manifest
        self._result: PlatformConsensus | None = None
        self._worker: CorroborateWorker | None = None
        self._filter = TraceFilter()

        layout = QtWidgets.QVBoxLayout(self)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("platform"))
        self.platform = QtWidgets.QComboBox()
        self.platform.setMinimumWidth(260)
        controls.addWidget(self.platform)
        controls.addWidget(QtWidgets.QLabel("segments"))
        self.limit = QtWidgets.QSpinBox()
        self.limit.setRange(0, 200000)
        self.limit.setValue(0)
        self.limit.setSpecialValueText("all")
        self.limit.setToolTip("use at most this many local segments; 0 means all")
        controls.addWidget(self.limit)
        self.run_button = QtWidgets.QPushButton("Corroborate")
        self.run_button.clicked.connect(self._run)
        controls.addWidget(self.run_button)
        controls.addStretch(1)
        controls.addWidget(QtWidgets.QLabel("filter"))
        self.filter_field = QtWidgets.QLineEdit()
        self.filter_field.setPlaceholderText("bus,can id   e.g. CAN0,0x2*")
        self.filter_field.setMinimumWidth(220)
        self.filter_field.returnPressed.connect(self._apply_filter)
        controls.addWidget(self.filter_field)
        layout.addLayout(controls)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self.status = QtWidgets.QLabel("choose a platform and press Corroborate")
        layout.addWidget(self.status)

        self.table = QtWidgets.QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        head = self.table.horizontalHeader()
        head.setStretchLastSection(True)
        for column in range(len(COLUMNS) - 1):
            head.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._show_selected)
        layout.addWidget(self.table, stretch=3)

        self.heading = QtWidgets.QLabel("—")
        self.heading.setStyleSheet("font-weight: 600; font-size: 13px;")
        layout.addWidget(self.heading)
        self.strip = ConsensusStrip()
        layout.addWidget(self.strip)
        self.legend = QtWidgets.QLabel(
            "colour: consensus class   opacity: agreement across segments   "
            "orange outline: rare bit (constant in most segments, moving in some)"
        )
        self.legend.setStyleSheet("color: #9fb4d0;")
        layout.addWidget(self.legend)
        self.findings = QtWidgets.QTextEdit()
        self.findings.setReadOnly(True)
        layout.addWidget(self.findings, stretch=2)

        self.refresh_platforms()

    # ----- platforms ---------------------------------------------------------

    def refresh_platforms(self) -> None:
        """List the platforms that have local segments, most segments first."""
        current = self.platform.currentText()
        held = inventory(self.root, self.manifest)
        names = sorted((k for k in held if k), key=lambda k: -held[k].count)
        self.platform.blockSignals(True)
        self.platform.clear()
        for name in names:
            self.platform.addItem(f"{name}   ({held[name].count} local)", name)
        index = next((i for i in range(self.platform.count())
                      if self.platform.itemData(i) == current), 0)
        self.platform.setCurrentIndex(index if names else -1)
        self.platform.blockSignals(False)

    def selected_platform(self) -> str | None:
        data = self.platform.currentData()
        return str(data) if data else None

    # ----- running -------------------------------------------------------------

    def _run(self) -> None:
        platform = self.selected_platform()
        if platform is None:
            self.status.setText("no platform has local segments -- fetch some first")
            return
        self.run_button.setEnabled(False)
        self.progress.setRange(0, 0)  # busy until the first progress report
        self.progress.setVisible(True)
        self.status.setText(f"corroborating {platform} …")
        self._worker = CorroborateWorker(platform, self.root, self.limit.value() or None)
        self._worker.progressed.connect(self._on_progress)
        self._worker.completed.connect(self._on_complete)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_progress(self, done: int, total: int) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        self.status.setText(f"{done}/{total} segments")

    def _on_failed(self, message: str) -> None:
        self._finish()
        self.status.setText(f"corroboration failed: {message}")

    def _on_complete(self, result: PlatformConsensus) -> None:
        self._finish()
        self._result = result
        self.status.setText(
            f"{result.platform}: {result.segments} segments from {result.devices} devices, "
            f"{len(result)} messages"
        )
        self._refill()

    def _finish(self) -> None:
        self.run_button.setEnabled(True)
        self.progress.setVisible(False)
        self._worker = None

    # ----- table -------------------------------------------------------------------

    def _apply_filter(self) -> None:
        """Read a two-field `bus,can id` filter.

        A platform is already chosen here, so the heat map's four-field form
        would have two dead fields. Two fields are read as bus and identifier;
        a full four-field filter is accepted too, with its first two ignored.
        """
        text = self.filter_field.text().strip()
        try:
            parts = [part.strip() or "*" for part in SEPARATORS.split(text)] if text else []
            if len(parts) > 2:
                parsed = TraceFilter.parse(text)
                self._filter = TraceFilter("*", "*", parsed.bus, parsed.can_id)
            else:
                parts += ["*"] * (2 - len(parts))
                self._filter = TraceFilter("*", "*", parts[0], parts[1])
        except ValueError as exc:
            self.status.setText(f"filter: {exc}")
            return
        self._refill()

    def _rows(self) -> list[MessageConsensus]:
        if self._result is None:
            return []
        return [
            m for m in self._result.by_identifier()
            if self._filter.matches_message(m.bus, m.address)
        ]

    @staticmethod
    def _findings_text(m: MessageConsensus) -> str:
        parts = []
        for ctr in m.counters:
            parts.append(f"ctr {ctr.length}b@{ctr.start_bit} [{ctr.evidence.tier}"
                         + (",contested" if ctr.contested else "") + "]")
        for chk in m.checksums:
            parts.append(f"{chk.algorithm}@{chk.byte_index} [{chk.evidence.tier}"
                         + (",contested" if chk.contested else "") + "]")
        for crc in m.crc16s:
            ident = f"/id{crc.data_id:04X}" if crc.data_id is not None else ""
            parts.append(f"{crc.algorithm}@{crc.start_byte}{ident} [{crc.evidence.tier}"
                         + (",contested" if crc.contested else "") + "]")
        return ", ".join(parts) or "-"

    def _refill(self) -> None:
        rows = self._rows()
        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))
        for index, m in enumerate(rows):
            cells = (
                str(m), str(m.segments), str(m.devices), str(m.width),
                f"{m.agreement:.0%}", str(len(m.rare_bits)), self._findings_text(m),
            )
            for column, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                if 0 < column < 6:
                    item.setTextAlignment(
                        QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
                    )
                item.setData(QtCore.Qt.ItemDataRole.UserRole, index)
                self.table.setItem(index, column, item)
        self.table.blockSignals(False)
        if rows:
            self.table.selectRow(0)
        else:
            self.heading.setText("—")
            self.findings.clear()

    # ----- detail ----------------------------------------------------------------------

    def _show_selected(self) -> None:
        rows = self._rows()
        selected = {i.row() for i in self.table.selectedIndexes()}
        if not selected or not rows:
            return
        m = rows[min(selected)]
        result = self._result
        assert result is not None
        self.heading.setText(
            f"{m}   width {m.width} ({m.width_agreement:.0%} agree)   "
            f"in {m.segments}/{result.segments} segments, {m.devices}/{result.devices} devices   "
            f"layout agreement {m.agreement:.0%}"
        )
        self.strip.show_message(m)
        lines = []
        if m.rare_bits:
            lines.append(f"rare bits: {m.rare_bits}")
            lines.append("")
        lines += [f"counter   {c}" for c in m.counters]
        lines += [f"checksum  {s}" for s in m.checksums]
        lines += [f"crc16     {c}" for c in m.crc16s]
        if not (m.counters or m.checksums or m.crc16s):
            lines.append("no counter or checksum found in any segment")
        self.findings.setPlainText("\n".join(lines))
