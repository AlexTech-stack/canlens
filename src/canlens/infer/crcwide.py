# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""32- and 64-bit CRCs from the AUTOSAR CRC library, scalar and column-wise.

The E2E profiles that use these name the routine, not the parameters --
"Crc_CalculateCRC32P4", "Crc_CalculateCRC64" -- and refer to SWS_CRCLibrary
for start values and XOR values. That document is not in the spec set here,
so the parameters below are the library's published ones and are pinned by
their published check values in the tests rather than trusted:

    CRC32P4  poly 0xF4ACFB13, init 0xFFFFFFFF, reflected, xor 0xFFFFFFFF   check 0x1697D06A
    CRC32    poly 0x04C11DB7, init 0xFFFFFFFF, reflected, xor 0xFFFFFFFF   check 0xCBF43926
    CRC64    poly 0x42F0E1EBA9EA3693, init all ones, reflected, xor all ones  check 0x995DC9BBDF1939FA

All three are reflected, unlike the CRC-8 and CRC-16 forms used by the other
profiles, so they get their own table construction rather than a widened copy
of crc16.py's.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _reflect(value: int, width: int) -> int:
    out = 0
    for i in range(width):
        if value & (1 << i):
            out |= 1 << (width - 1 - i)
    return out


@dataclass(frozen=True)
class WideCrc:
    """A reflected, table-driven CRC of 32 or 64 bits."""

    name: str
    width: int
    poly: int  # normal form, as the spec writes it
    init: int
    xorout: int

    @property
    def mask(self) -> int:
        return (1 << self.width) - 1

    @property
    def dtype(self):
        return np.uint32 if self.width == 32 else np.uint64

    def table(self) -> np.ndarray:
        reflected = _reflect(self.poly, self.width)
        rows = []
        for value in range(256):
            crc = value
            for _ in range(8):
                crc = (crc >> 1) ^ reflected if crc & 1 else crc >> 1
            rows.append(crc)
        return np.array(rows, dtype=self.dtype)

    def compute(self, data: bytes) -> int:
        """Scalar reference."""
        table = self.table()
        crc = self.init
        for byte in data:
            crc = int(table[(crc ^ byte) & 0xFF]) ^ (crc >> 8)
        return (crc ^ self.xorout) & self.mask

    def columns(self, matrix: np.ndarray, skip: tuple[int, int] | None = None) -> np.ndarray:
        """CRC of every row of an (n, width) byte matrix, skipping a column range."""
        table = self.table()
        crc = np.full(matrix.shape[0], self.init, dtype=self.dtype)
        for column in range(matrix.shape[1]):
            if skip is not None and skip[0] <= column < skip[1]:
                continue
            index = (crc ^ matrix[:, column].astype(self.dtype)) & np.uint8(0xFF)
            crc = table[index.astype(np.intp)] ^ (crc >> np.uint8(8))
        return (crc ^ self.dtype(self.xorout)) & self.dtype(self.mask)


CRC32P4 = WideCrc("crc32p4", 32, 0xF4ACFB13, 0xFFFFFFFF, 0xFFFFFFFF)
CRC32 = WideCrc("crc32", 32, 0x04C11DB7, 0xFFFFFFFF, 0xFFFFFFFF)
CRC64 = WideCrc("crc64", 64, 0x42F0E1EBA9EA3693, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF)


def read_be(matrix: np.ndarray, start: int, nbytes: int) -> np.ndarray:
    """Big-endian unsigned field of `nbytes` at `start`, per row."""
    out = np.zeros(matrix.shape[0], dtype=np.uint64)
    for i in range(nbytes):
        out = (out << np.uint8(8)) | matrix[:, start + i].astype(np.uint64)
    return out
