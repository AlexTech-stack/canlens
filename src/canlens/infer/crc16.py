# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""16-bit CRCs, including the AUTOSAR E2E Profile 5 construction.

Parameters here follow AUTOSAR_PRS_E2EProtocol (FO R19-11). The library's
`Crc_CalculateCRC16` is the CCITT-FALSE 16-bit CRC -- polynomial 0x1021 in
normal form, start value 0xFFFF, no final XOR and no reflection -- and
[PRS_E2E_00400] names it as Profile 5's CRC.

Profile 5 is the interesting case for reverse engineering. Per
[PRS_E2E_00401] the CRC covers the E2E header excluding the CRC bytes
themselves, plus the user data, *extended at the end with the Data ID*; the
flowchart in Figure 6.54 fixes the order as the Data ID's low byte then its
high byte. The Data ID is "implicitly sent" -- it never appears on the wire --
so a trace alone cannot read it. It can, however, be *solved for*: a CRC is
deterministic, so the observed value pins down the two appended bytes, and a
single sweep over the 65536 candidates finds the one Data ID that reproduces
the CRC on every frame. Figure 6.55 stores the CRC little-endian.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

POLY_CCITT = 0x1021
POLY_ARC = 0x8005


def _table(poly: int) -> list[int]:
    table = []
    for byte in range(256):
        crc = byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
        table.append(crc)
    return table


CCITT_TABLE = _table(POLY_CCITT)
ARC_TABLE = _table(POLY_ARC)

# What makes a CRC step invertible is that the table's *low* bytes are a
# permutation of 0..255 -- the high bytes are not (only 128 are distinct), so
# indexing by those silently returns the wrong entry.
CCITT_INDEX_BY_LOW = {value & 0xFF: index for index, value in enumerate(CCITT_TABLE)}


def _reflect(value: int, width: int) -> int:
    out = 0
    for i in range(width):
        if value & (1 << i):
            out |= 1 << (width - 1 - i)
    return out


def crc16_update(crc: int, data: bytes, table: list[int] = CCITT_TABLE) -> int:
    """Feed bytes into a running non-reflected CRC16."""
    for byte in data:
        crc = ((crc << 8) & 0xFFFF) ^ table[((crc >> 8) ^ byte) & 0xFF]
    return crc


def crc16_autosar(data: bytes) -> int:
    """`Crc_CalculateCRC16`: CCITT-FALSE, start 0xFFFF, no final XOR."""
    return crc16_update(0xFFFF, data)


def crc16_ccitt_zero(data: bytes) -> int:
    """Same polynomial, zero start value (XMODEM)."""
    return crc16_update(0x0000, data)


def crc16_arc(data: bytes) -> int:
    """Reflected 0x8005, zero start -- the other CRC16 seen in the wild."""
    crc = 0x0000
    for byte in data:
        crc = ((crc << 8) & 0xFFFF) ^ ARC_TABLE[((crc >> 8) ^ _reflect(byte, 8)) & 0xFF]
    return _reflect(crc, 16)


CRC16_ALGORITHMS = {
    "crc16_autosar": crc16_autosar,
    "crc16_ccitt_zero": crc16_ccitt_zero,
    "crc16_arc": crc16_arc,
}


@dataclass(frozen=True)
class Crc16Hypothesis:
    """A 2-byte field that a 16-bit CRC reproduces."""

    start_byte: int
    algorithm: str
    byteorder: str
    match_rate: float
    frames: int
    data_id: int | None = None

    @property
    def start_bit(self) -> int:
        return self.start_byte * 8

    @property
    def length(self) -> int:
        return 16

    def __str__(self) -> str:
        ident = "" if self.data_id is None else f", data ID 0x{self.data_id:04X}"
        return (
            f"{self.algorithm} @ bytes {self.start_byte}-{self.start_byte + 1} "
            f"({self.byteorder}-endian{ident}, {self.match_rate:.1%})"
        )


def without(payload: bytes, start: int) -> bytes:
    """The payload with the two CRC bytes at `start` removed."""
    return payload[:start] + payload[start + 2 :]


def read_crc(payload: bytes, start: int, byteorder: str) -> int:
    return int.from_bytes(payload[start : start + 2], byteorder)  # type: ignore[arg-type]


def solve_data_id(payload: bytes, start: int, byteorder: str) -> set[int]:
    """Data IDs that would make an E2E Profile 5 CRC come out as observed.

    Solved rather than searched. Each low byte fixes an intermediate CRC, and
    from there the high byte needed to land on the observed value is recovered
    by inverting a single CRC step -- so this costs 256 iterations, not the
    65536 a sweep over every candidate Data ID would take. One frame therefore
    narrows the field to 256 possibilities; intersecting a handful of frames
    leaves the one Data ID that is actually in use.
    """
    target = read_crc(payload, start, byteorder)
    base = crc16_update(0xFFFF, without(payload, start))
    found = set()
    for low in range(256):
        after_low = ((base << 8) & 0xFFFF) ^ CCITT_TABLE[((base >> 8) ^ low) & 0xFF]
        # target == ((after_low << 8) & 0xFFFF) ^ TABLE[(after_low >> 8) ^ high]
        wanted = target ^ ((after_low << 8) & 0xFFFF)
        index = CCITT_INDEX_BY_LOW.get(wanted & 0xFF)
        if index is None or CCITT_TABLE[index] != wanted:
            continue
        high = (index ^ (after_low >> 8)) & 0xFF
        found.add(low | (high << 8))
    return found


def e2e_p05(payload: bytes, start: int, data_id: int) -> int:
    """Profile 5's CRC: data without the CRC bytes, then ID low, then ID high."""
    crc = crc16_update(0xFFFF, without(payload, start))
    return crc16_update(crc, bytes([data_id & 0xFF, (data_id >> 8) & 0xFF]))


def _rate(payloads: Sequence[bytes], predicate, min_match: float = 0.0) -> float:
    """Fraction of payloads satisfying `predicate`, abandoned once hopeless.

    Almost every candidate is wrong and fails on the first frame or two, so
    giving up as soon as `min_match` is unreachable turns a whole-segment
    sweep from tens of seconds into well under one.
    """
    total = len(payloads)
    budget = total - int(min_match * total)  # failures we can still afford
    hits = misses = 0
    for payload in payloads:
        if predicate(payload):
            hits += 1
        else:
            misses += 1
            if misses > budget:
                return hits / total
    return hits / total


def find_crc16(
    payloads: Sequence[bytes],
    *,
    candidate_bytes: Sequence[int] | None = None,
    min_match: float = 0.99,
    screen_frames: int = 64,
    search_data_id: bool = True,
) -> list[Crc16Hypothesis]:
    """Look for a 2-byte CRC field, with or without an E2E Profile 5 Data ID."""
    if len(payloads) < 2:
        return []
    width = len(payloads[0])
    starts = (
        [s for s in candidate_bytes if 0 <= s < width - 1]
        if candidate_bytes is not None
        else range(width - 1)
    )
    screen = list(payloads[:screen_frames])
    found: list[Crc16Hypothesis] = []

    for start in starts:
        for byteorder in ("little", "big"):
            hit = None
            for name, fn in CRC16_ALGORITHMS.items():
                def plain(p: bytes, fn=fn, start=start, byteorder=byteorder) -> bool:
                    return fn(without(p, start)) == read_crc(p, start, byteorder)

                if _rate(screen, plain, min_match) < min_match:
                    continue
                rate = _rate(payloads, plain, min_match)
                if rate >= min_match:
                    hit = Crc16Hypothesis(start, name, byteorder, rate, len(payloads))
                    break

            if hit is None and search_data_id:
                # Intersect the solutions from a few frames: a real Data ID
                # satisfies every frame, a coincidence satisfies one.
                candidates: set[int] | None = None
                for payload in screen[:4]:
                    solutions = solve_data_id(payload, start, byteorder)
                    candidates = solutions if candidates is None else candidates & solutions
                    if not candidates:
                        break
                for data_id in sorted(candidates or ()):
                    def p05(p: bytes, start=start, data_id=data_id, byteorder=byteorder) -> bool:
                        return e2e_p05(p, start, data_id) == read_crc(p, start, byteorder)

                    rate = _rate(payloads, p05, min_match)
                    if rate >= min_match:
                        hit = Crc16Hypothesis(
                            start, "e2e_p05", byteorder, rate, len(payloads), data_id
                        )
                        break

            if hit is not None:
                found.append(hit)
                break  # one byte order is enough for this position
    return found
