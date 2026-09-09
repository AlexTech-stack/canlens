# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Terminal presentation helpers.

Kept apart from the analysis packages so that measurement code never has to
think about terminals, and so the rendering can be tested without one.
"""
from __future__ import annotations

import os
import sys
from typing import IO

from .analyze.bits import BitKind

RESET = "\x1b[0m"

# Bright background colours: a coloured space renders as a solid block on any
# terminal theme, whereas a foreground glyph disappears when it matches the
# background -- white "constant" bits on a light theme especially.
BACKGROUNDS: dict[BitKind, str] = {
    BitKind.CONSTANT: "\x1b[107m",  # white
    BitKind.SLOW: "\x1b[104m",  # blue
    BitKind.ACTIVE: "\x1b[102m",  # green
    BitKind.NOISY: "\x1b[101m",  # red
}

# Deliberately escalating density, so the shape survives a pipe, a log file or
# a colour-blind reader.
GLYPHS: dict[BitKind, str] = {
    BitKind.CONSTANT: ".",
    BitKind.SLOW: ":",
    BitKind.ACTIVE: "+",
    BitKind.NOISY: "#",
}

TRUNCATED = "›"  # single right angle quote


def supports_color(stream: IO[str] | None = None) -> bool:
    """Whether to emit ANSI colour, honouring NO_COLOR and FORCE_COLOR."""
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    stream = stream or sys.stdout
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


def bit_strip(
    kinds: list[BitKind],
    *,
    color: bool = True,
    max_bits: int | None = None,
    group: int = 8,
) -> str:
    """Render one bit per character, coloured or glyphed by behaviour.

    A space every `group` bits marks the byte boundary, which is what makes a
    field that straddles two bytes visible at a glance.
    """
    shown = kinds if max_bits is None else kinds[:max_bits]
    out = []
    for i, kind in enumerate(shown):
        if i and group and i % group == 0:
            out.append(" ")
        out.append(f"{BACKGROUNDS[kind]} {RESET}" if color else GLYPHS[kind])
    if max_bits is not None and len(kinds) > max_bits:
        out.append(TRUNCATED)
    return "".join(out)


def strip_width(bits: int, *, group: int = 8) -> int:
    """Printed width of a strip of `bits` bits, separators included."""
    if bits <= 0:
        return 0
    return bits + (bits - 1) // group if group else bits


def bits_that_fit(available: int, *, group: int = 8) -> int:
    """Largest whole number of bits whose strip fits in `available` columns."""
    bits = 0
    while strip_width(bits + 1, group=group) <= available:
        bits += 1
    return bits


def legend(color: bool = True) -> str:
    """One-line key for the strip colours."""
    parts = []
    for kind in (BitKind.CONSTANT, BitKind.SLOW, BitKind.ACTIVE, BitKind.NOISY):
        mark = f"{BACKGROUNDS[kind]} {RESET}" if color else GLYPHS[kind]
        parts.append(f"{mark} {kind}")
    return "  ".join(parts)
