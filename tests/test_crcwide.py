# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""The wide CRCs, pinned to their published check values."""
from __future__ import annotations

import random

import numpy as np
import pytest

from canlens.infer.crcwide import CRC32, CRC32P4, CRC64, read_be

CHECK = b"123456789"


class TestKnownAnswers:
    def test_crc32p4(self):
        assert CRC32P4.compute(CHECK) == 0x1697D06A

    def test_crc32_ieee(self):
        assert CRC32.compute(CHECK) == 0xCBF43926

    def test_crc64(self):
        assert CRC64.compute(CHECK) == 0x995DC9BBDF1939FA

    def test_empty_input_is_init_xor_xorout(self):
        assert CRC32P4.compute(b"") == 0
        assert CRC64.compute(b"") == 0


class TestColumnsMatchScalar:
    @pytest.mark.parametrize("crc", [CRC32P4, CRC32, CRC64])
    def test_every_row(self, crc):
        rng = random.Random(1)
        matrix = np.array([[rng.randrange(256) for _ in range(20)] for _ in range(50)], dtype=np.uint8)
        scalar = [crc.compute(bytes(row)) for row in matrix]
        assert crc.columns(matrix).tolist() == scalar

    @pytest.mark.parametrize("crc", [CRC32P4, CRC64])
    def test_skipping_a_column_range(self, crc):
        rng = random.Random(2)
        matrix = np.array([[rng.randrange(256) for _ in range(20)] for _ in range(30)], dtype=np.uint8)
        scalar = [crc.compute(bytes(row[:8]) + bytes(row[12:])) for row in matrix]
        assert crc.columns(matrix, skip=(8, 12)).tolist() == scalar


class TestReadBigEndian:
    def test_two_and_four_bytes(self):
        m = np.array([[0x12, 0x34, 0x56, 0x78]], dtype=np.uint8)
        assert read_be(m, 0, 2).tolist() == [0x1234]
        assert read_be(m, 0, 4).tolist() == [0x12345678]
