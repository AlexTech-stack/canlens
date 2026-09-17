# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Filtering what the workbench shows, by vehicle, segment, bus and CAN ID.

A filter is four comma-separated patterns::

    TOYOTA_PRIUS,*,CAN0,*        every segment and every identifier on bus 0
    *,*,*,0x1*                   identifiers beginning 0x1, on any vehicle
    KIA_EV6,*,*,0x2??            three-digit identifiers 0x200-0x2FF

Two wildcards are supported, and only two: ``*`` matches any run of characters
including none, and ``?`` matches exactly one. Everything else is literal --
notably ``[``, which a glob library would quietly treat as a character class.

Matching is case-insensitive, and each field is compared against several
renderings of the same thing so that a filter can be written the way the value
is naturally spoken. ``CAN0``, ``CAN_0``, ``bus 0`` and ``0`` all select bus 0;
``0x211`` and ``211`` both select that identifier. Identifiers are rendered
without leading zeroes, which is what makes ``0x1*`` mean "0x1, 0x11, 0x123 …"
rather than only the four-digit ones.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

FIELDS = ("vehicle", "segment", "bus", "can_id")
ANY = "*"
SEPARATORS = re.compile(r"[,;]")


def compile_pattern(pattern: str) -> re.Pattern[str]:
    """Translate a `*`/`?` pattern into an anchored, case-insensitive regex."""
    out = []
    for char in pattern:
        if char == "*":
            out.append(".*")
        elif char == "?":
            out.append(".")
        else:
            out.append(re.escape(char))
    return re.compile("".join(out), re.IGNORECASE | re.DOTALL)


def matches(pattern: str, *candidates: str) -> bool:
    """Whether any rendering of a value satisfies the pattern."""
    pattern = pattern.strip()
    if not pattern or pattern == ANY:
        return True
    regex = compile_pattern(pattern)
    return any(regex.fullmatch(candidate) for candidate in candidates)


def bus_names(bus: int) -> tuple[str, ...]:
    """Every way a bus number is reasonably written."""
    return (f"CAN{bus}", f"CAN_{bus}", f"bus {bus}", f"bus{bus}", str(bus))


def can_id_names(address: int) -> tuple[str, ...]:
    """Every way an identifier is reasonably written, without leading zeroes."""
    return (f"0x{address:X}", f"{address:X}")


def segment_names(path: str) -> tuple[str, ...]:
    """Every way a segment is reasonably named, from a stored path."""
    parts = [p for p in path.replace("\\", "/").split("/") if p]
    if parts and parts[-1].endswith(".zst"):
        parts = parts[:-1]
    # Take the last three, then pad on the *left*: padding on the right and
    # slicing the tail hands back the padding instead of the path.
    tail = parts[-3:]
    device, route, index = ["", "", ""][: 3 - len(tail)] + tail
    return (
        f"{device}/{route}/{index}",
        f"{route}/{index}",
        route,
        route.split("--")[0],
        index,
    )


@dataclass(frozen=True)
class TraceFilter:
    """Four patterns: vehicle, segment, bus, CAN identifier."""

    vehicle: str = ANY
    segment: str = ANY
    bus: str = ANY
    can_id: str = ANY

    @classmethod
    def parse(cls, text: str) -> TraceFilter:
        """Read a filter expression.

        Missing trailing fields default to `*`, so `TOYOTA_PRIUS` alone is a
        valid filter meaning that vehicle and everything within it.
        """
        text = (text or "").strip()
        if not text:
            return cls()
        parts = [part.strip() or ANY for part in SEPARATORS.split(text)]
        if len(parts) > len(FIELDS):
            raise ValueError(
                f"a filter has at most {len(FIELDS)} fields "
                f"({', '.join(FIELDS)}); got {len(parts)}"
            )
        parts += [ANY] * (len(FIELDS) - len(parts))
        return cls(*parts)

    def __str__(self) -> str:
        return ",".join(getattr(self, name) for name in FIELDS)

    @property
    def is_empty(self) -> bool:
        """True when this filter would keep everything."""
        return all(getattr(self, name).strip() in ("", ANY) for name in FIELDS)

    @property
    def selects_segments(self) -> bool:
        """Whether the vehicle or segment fields narrow anything."""
        return not (
            self.vehicle.strip() in ("", ANY) and self.segment.strip() in ("", ANY)
        )

    def matches_segment(self, platform: str | None, path: str) -> bool:
        return matches(self.vehicle, platform or "") and matches(
            self.segment, *segment_names(path)
        )

    def matches_message(self, bus: int, address: int) -> bool:
        return matches(self.bus, *bus_names(bus)) and matches(
            self.can_id, *can_id_names(address)
        )
