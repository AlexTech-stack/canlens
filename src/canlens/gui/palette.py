# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""One palette for the bit classes, shared by the terminal and the workbench.

The colours are the terminal's: white constant, blue slow, green active,
yellow busy, red noisy. Someone who has learned to read the ASCII bit map
should not have to relearn anything in the GUI.
"""
from __future__ import annotations

import numpy as np

from ..analyze.bits import KIND_ORDER, BitKind

# RGB, in KIND_ORDER. Darkened slightly from pure terminal colours so that
# white-on-dark stays comfortable over a full screen of cells.
KIND_RGB: dict[BitKind, tuple[int, int, int]] = {
    BitKind.CONSTANT: (238, 238, 238),
    BitKind.SLOW: (54, 122, 214),
    BitKind.ACTIVE: (56, 176, 94),
    BitKind.BUSY: (226, 182, 44),
    BitKind.NOISY: (214, 66, 58),
}

KIND_INDEX: dict[BitKind, int] = {kind: i for i, kind in enumerate(KIND_ORDER)}

# Field overlay colours, matching the terminal's C / X ruler marks.
COUNTER_RGBA = (90, 190, 255, 90)
CHECKSUM_RGBA = (255, 130, 220, 90)

BACKGROUND = "#14161a"
GRID = "#2a2e36"


def lookup_table() -> np.ndarray:
    """RGB lookup table indexed by KIND_ORDER position, for ImageItem."""
    return np.array([KIND_RGB[kind] for kind in KIND_ORDER], dtype=np.ubyte)


def kinds_to_indices(kinds: list[BitKind]) -> np.ndarray:
    """Bit classes as the small integers ImageItem's colour map expects."""
    return np.array([KIND_INDEX[kind] for kind in kinds], dtype=np.uint8)
