# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""The canlens workbench window.

Layout follows the reverse-engineering loop rather than the data model: pick a
segment on the left, read the whole bus as a bit matrix in the middle, and
drill into one message at the bottom. Selecting a row drives the detail views,
so the question "what is this field" is always one click from its answer.
"""
from __future__ import annotations

import os

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..corpus import Manifest, segment_dest
from .axes import bus_ticks, byte_ticks
from .model import SegmentModel, load_segment
from .palette import BACKGROUND, CHECKSUM_RGBA, COUNTER_RGBA, lookup_table

pg.setConfigOption("background", BACKGROUND)
pg.setConfigOption("foreground", "#d7dae0")
pg.setConfigOption("antialias", False)


class BitMatrixView(pg.PlotWidget):
    """The whole bus as one image: rows are messages, columns are payload bits.

    An ImageItem is used rather than a table of widgets because the matrix runs
    to tens of thousands of cells and must stay responsive while panning; this
    hands the class indices straight to the renderer as a numpy array.
    """

    row_selected = QtCore.Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.setMenuEnabled(False)
        self.getPlotItem().invertY(True)
        self.getPlotItem().setLabel("bottom", "payload bit")
        self.getPlotItem().setLabel("left", "")

        self._image = pg.ImageItem(axisOrder="row-major")
        self.addItem(self._image)

        # Marks the selected row without redrawing the image.
        self._highlight = pg.LinearRegionItem(
            orientation="horizontal",
            movable=False,
            brush=pg.mkBrush(255, 255, 255, 55),
            pen=pg.mkPen("#ffffff", width=2),
        )
        self._highlight.setZValue(10)
        self.addItem(self._highlight)

        self._overlays: list[pg.GraphicsObject] = []
        self._dividers: list[pg.GraphicsObject] = []
        self._model: SegmentModel | None = None
        self.scene().sigMouseClicked.connect(self._on_click)

    def set_model(self, model: SegmentModel) -> None:
        self._model = model
        grid = model.matrix()
        # -1 marks "payload ended here". Render RGBA directly so those cells
        # are transparent: a short message must simply stop rather than look
        # like a run of constant bits.
        table = lookup_table()
        rgba = np.zeros((*grid.shape, 4), dtype=np.ubyte)
        present = grid >= 0
        rgba[present, :3] = table[grid[present]]
        rgba[present, 3] = 255
        # levels are pinned because pyqtgraph would otherwise derive them from
        # the data: a message whose bits are all one class gives a uniform
        # array, auto-levels come out as (238, 255), and every pixel maps to
        # black -- a wholly static frame rendered as though it were missing.
        self._image.setImage(rgba, levels=(0, 255))
        self.getPlotItem().getAxis("bottom").setTicks(byte_ticks(model.bit_width))
        self._draw_dividers(model)
        self.getPlotItem().getAxis("left").setTicks(bus_ticks(model.bus_groups()))
        self.getPlotItem().setLimits(
            xMin=0, xMax=model.bit_width, yMin=0, yMax=max(len(model.rows), 1)
        )
        self.autoRange()
        self.select_row(0)

    def _draw_dividers(self, model: SegmentModel) -> None:
        """Rules between buses, and where extended identifiers begin."""
        for item in self._dividers:
            self.removeItem(item)
        self._dividers.clear()
        for index, label in model.separators():
            extended = label == "extended"
            # A bus change is named by the left axis; only the extended split
            # needs a label, and it sits at the right edge clear of the data.
            line = pg.InfiniteLine(
                pos=index,
                angle=0,
                pen=pg.mkPen(
                    "#ff9f43" if extended else "#8b93a1",
                    width=2,
                    style=QtCore.Qt.PenStyle.SolidLine
                    if extended
                    else QtCore.Qt.PenStyle.DashLine,
                ),
                label="extended IDs" if extended else None,
                labelOpts={
                    "position": 0.88,
                    "color": "#ff9f43",
                    "fill": pg.mkBrush(20, 22, 26, 220),
                    "movable": False,
                },
            )
            line.setZValue(8)
            self.addItem(line)
            self._dividers.append(line)

    def select_row(self, index: int) -> None:
        if self._model is None or not self._model.rows:
            return
        index = max(0, min(index, len(self._model.rows) - 1))
        self._highlight.setRegion((index, index + 1))
        self._draw_overlays(index)
        self.row_selected.emit(index)

    def _draw_overlays(self, index: int) -> None:
        for item in self._overlays:
            self.removeItem(item)
        self._overlays.clear()
        if self._model is None:
            return
        for start, length, kind in self._model.field_spans(index):
            colour = COUNTER_RGBA if kind == "counter" else CHECKSUM_RGBA
            box = QtWidgets.QGraphicsRectItem(start, index, length, 1)
            box.setBrush(pg.mkBrush(*colour))
            box.setPen(pg.mkPen(colour[:3] + (220,), width=0))
            box.setZValue(5)
            self.addItem(box)
            self._overlays.append(box)

    def _on_click(self, event) -> None:
        if self._model is None:
            return
        point = self.getPlotItem().vb.mapSceneToView(event.scenePos())
        self.select_row(int(point.y()))


class BitStripView(pg.PlotWidget):
    """One message's bits, magnified.

    The matrix above is a map: at 186 messages a single row is under three
    pixels tall, so a field overlay drawn there is invisible however correct
    its coordinates are. This is the detail counterpart -- the same colours,
    one row, tall enough that a counter or CRC span can actually be seen and
    read off against the bit axis.
    """

    selection_changed = QtCore.Signal(int, int)

    def __init__(self) -> None:
        super().__init__()
        self.setMenuEnabled(False)
        self.setMaximumHeight(96)
        self.getPlotItem().hideAxis("left")
        self.getPlotItem().setLabel("bottom", "payload bit")
        self._image = pg.ImageItem(axisOrder="row-major")
        self.addItem(self._image)
        self._overlays: list[pg.GraphicsObject] = []
        self._bits = 0
        self._snapping = False

        # Drag to define a field by hand. Whole bits only: a selection running
        # from 3.7 to 8.2 describes nothing, so the edges snap on every change
        # rather than only when the drag ends.
        self.selector = pg.LinearRegionItem(
            values=(0, 8),
            brush=pg.mkBrush(255, 255, 255, 38),
            hoverBrush=pg.mkBrush(255, 255, 255, 60),
            pen=pg.mkPen("#ffffff", width=2),
        )
        self.selector.setZValue(20)
        self.addItem(self.selector)
        self.selector.sigRegionChanged.connect(self._snap)

    def _snap(self) -> None:
        """Round the selection to whole bits and announce it."""
        if self._snapping or not self._bits:
            return
        # getRegion returns plain Python numbers, so round() already gives int.
        low, high = self.selector.getRegion()
        start = max(0, min(round(low), self._bits - 1))
        end = max(start + 1, min(round(high), self._bits))
        self._snapping = True
        self.selector.setRegion((start, end))
        self._snapping = False
        self.selection_changed.emit(start, end - start)

    def set_selection(self, start: int, length: int) -> None:
        self.selector.setRegion((start, start + length))

    def show_row(self, model: SegmentModel, index: int) -> None:
        for item in self._overlays:
            self.removeItem(item)
        self._overlays.clear()

        row = model.rows[index]
        self._bits = row.bits
        table = lookup_table()
        rgba = np.zeros((1, row.bits, 4), dtype=np.ubyte)
        rgba[0, :, :3] = table[row.kinds]
        rgba[0, :, 3] = 255
        # Pinned for the same reason as the matrix: a fully static payload is
        # uniform, and auto-levels would render it black instead of constant.
        self._image.setImage(rgba, levels=(0, 255))

        for start, length, kind in model.field_spans(index):
            colour = COUNTER_RGBA if kind == "counter" else CHECKSUM_RGBA
            box = QtWidgets.QGraphicsRectItem(start, -0.35, length, 1.7)
            box.setBrush(pg.mkBrush(colour[0], colour[1], colour[2], 70))
            box.setPen(pg.mkPen(colour[:3], width=2))
            box.setZValue(5)
            self.addItem(box)
            self._overlays.append(box)

            label = pg.TextItem(kind, color=colour[:3], anchor=(0, 1))
            label.setPos(start, -0.35)
            label.setZValue(6)
            self.addItem(label)
            self._overlays.append(label)

        self.selector.setBounds((0, row.bits))
        self.getPlotItem().getAxis("bottom").setTicks(byte_ticks(row.bits))
        self.getPlotItem().setLimits(xMin=-1, xMax=row.bits + 1, yMin=-1.2, yMax=2.2)
        self.setXRange(0, row.bits, padding=0.01)
        self.setYRange(-0.6, 1.6, padding=0)


class DetailPanel(QtWidgets.QWidget):
    """Findings for the selected message, and the counter's values over time."""

    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        self.heading = QtWidgets.QLabel("—")
        self.heading.setStyleSheet("font-weight: 600; font-size: 13px;")
        layout.addWidget(self.heading)

        self.strip = BitStripView()
        self.strip.selection_changed.connect(self._on_selection)
        layout.addWidget(self.strip)

        self.selection = QtWidgets.QLabel("—")
        self.selection.setStyleSheet("color: #9fb4d0; font-family: monospace;")
        layout.addWidget(self.selection)

        self.findings = QtWidgets.QTextEdit()
        self.findings.setReadOnly(True)
        self.findings.setMaximumHeight(110)
        layout.addWidget(self.findings)

        self.plot = pg.PlotWidget()
        self.plot.setMenuEnabled(False)
        self.plot.getPlotItem().setLabel("left", "field value")
        self.plot.getPlotItem().setLabel("bottom", "frame")
        layout.addWidget(self.plot, stretch=1)

        self._model: SegmentModel | None = None
        self._index = 0

    def _on_selection(self, start: int, length: int) -> None:
        """Replot for a hand-picked bit range.

        Cheap enough to run live while dragging because the payloads are held
        in the model; nothing here goes back to the trace file.
        """
        if self._model is None:
            return
        self.selection.setText(self._model.field_summary(self._index, start, length))
        self.plot.clear()
        values = self._model.field_series(self._index, start, length)
        if values.size:
            self.plot.plot(values, pen=pg.mkPen("#5abeff", width=1))

    def show_row(self, model: SegmentModel, index: int) -> None:
        row = model.rows[index]
        self._model = model
        self._index = index
        self.strip.show_row(model, index)
        self.heading.setText(
            f"{row.label}   {row.count} frames x {row.width} bytes"
            f"   {row.period_ms:.1f} ms   {row.entropy:.0f} bits entropy"
        )
        lines = []
        if row.inference is not None:
            lines += [f"counter   {c}" for c in row.inference.counters]
            lines += [f"checksum  {s}" for s in row.inference.checksums]
            lines += [f"crc16     {c}" for c in row.inference.crc16s]
        self.findings.setPlainText("\n".join(lines) or "no counter or checksum reproduced this message")

        # Start on the most interesting known field, so the plot says something
        # before the first drag; any range can be selected from there.
        if row.inference is not None and row.inference.counters:
            counter = row.inference.counters[0]
            start, length = counter.start_bit, counter.length
        else:
            start, length = 0, min(8, row.bits)
        self.strip.set_selection(start, length)
        self._on_selection(start, length)


class Workbench(QtWidgets.QMainWindow):
    def __init__(self, root: str) -> None:
        super().__init__()
        self.root = root
        self.manifest = Manifest.load(os.path.join(root, "database.json"))
        self.setWindowTitle("canlens")
        self.resize(1440, 900)

        self.segments = QtWidgets.QListWidget()
        self.segments.setMaximumWidth(340)
        self.segments.currentRowChanged.connect(self._open_selected)

        self.matrix = BitMatrixView()
        self.details = DetailPanel()
        self.matrix.row_selected.connect(self._on_row)

        right = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        right.addWidget(self.matrix)
        right.addWidget(self.details)
        right.setSizes([560, 340])

        split = QtWidgets.QSplitter()
        split.addWidget(self.segments)
        split.addWidget(right)
        split.setSizes([340, 1100])
        self.setCentralWidget(split)
        self.statusBar().showMessage("ready")

        self._model: SegmentModel | None = None
        self._paths: list[tuple[str, str]] = []
        self._populate()

    def _populate(self) -> None:
        """List every locally present segment, labelled by platform."""
        for entry in self.manifest.by_size():
            for segment in entry.segments:
                path = segment_dest(self.root, segment)
                if os.path.exists(path):
                    self._paths.append((entry.key, path))
        for name, path in self._paths:
            _device, route, index = path.split("/")[-4:-1]
            self.segments.addItem(f"{name}   {route.split('--')[0]}/{index}")
        if self._paths:
            self.segments.setCurrentRow(0)

    def _open_selected(self, index: int) -> None:
        if not (0 <= index < len(self._paths)):
            return
        platform, path = self._paths[index]
        self.statusBar().showMessage(f"loading {platform} …")
        QtWidgets.QApplication.processEvents()
        model = load_segment(path, root=self.root, platform=platform)
        self._model = model
        self.matrix.set_model(model)
        self.statusBar().showMessage(
            f"{platform}  —  {len(model.rows)} messages, {model.profile.frames} frames, "
            f"{model.bit_width} bits wide"
        )

    def _on_row(self, index: int) -> None:
        if self._model is not None:
            self.details.show_row(self._model, index)


def run(root: str) -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = Workbench(root)
    window.show()
    return app.exec()
