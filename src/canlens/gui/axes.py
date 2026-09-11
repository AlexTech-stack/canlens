# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Axis ticks for bit-indexed views.

Payload bits are read in bytes, so the only useful gridlines are multiples of
eight. pyqtgraph's default tick picker chooses round decimal numbers -- 5, 10,
15 -- which is exactly wrong here: it puts lines through the middle of bytes
and never on a boundary.
"""
from __future__ import annotations

BYTE = 8


def byte_ticks(bits: int, *, max_labels: int = 18) -> list[list[tuple[float, str]]]:
    """Major/minor ticks on byte boundaries, thinned to stay readable.

    Labels are always on a multiple of eight. A 32-byte CAN FD payload has 32
    of them, which is too many to print, so the step doubles until the labels
    fit and every byte boundary is kept as an unlabelled minor tick.
    """
    if bits <= 0:
        return [[], []]
    step = BYTE
    while bits // step > max_labels:
        step *= 2
    major = [(float(v), str(v)) for v in range(0, bits + 1, step)]
    minor = (
        [(float(v), "") for v in range(0, bits + 1, BYTE) if v % step]
        if step > BYTE
        else []
    )
    return [major, minor]


def bus_ticks(groups: list[tuple[int, int, int]]) -> list[list[tuple[float, str]]]:
    """Left-axis labels naming each bus, centred on its block of rows.

    The bus name used to ride on an InfiniteLine inside the view, which put
    small text on top of the colour matrix where it was barely readable. In the
    axis margin it has a plain background and room to breathe, and the divider
    line is left to do nothing but divide.

    `groups` is (bus, first_row, last_row_exclusive).
    """
    return [
        [(float(start + end - 1) / 2, f"bus {bus}") for bus, start, end in groups],
        [],
    ]
