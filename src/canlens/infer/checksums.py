# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Candidate checksum algorithms, and the search that tests them.

A hypothesis here is only ever reported if it *reproduces* the observed byte,
on nearly every frame in the trace. Nothing is inferred from a byte merely
looking random -- that is what :mod:`canlens.analyze` already told us, and it
is exactly the kind of guess this layer exists to replace.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

# A checksum function sees the whole payload, the message address, and which
# byte is under test, and returns the byte it expects to find there.
ChecksumFn = Callable[[bytes, int, int], int]


def _others(payload: bytes, index: int) -> bytes:
    """The payload with the byte under test removed."""
    return payload[:index] + payload[index + 1 :]


def sum8(payload: bytes, address: int, index: int) -> int:
    return sum(_others(payload, index)) & 0xFF


def sum8_complement(payload: bytes, address: int, index: int) -> int:
    return (-sum(_others(payload, index))) & 0xFF


def xor8(payload: bytes, address: int, index: int) -> int:
    acc = 0
    for byte in _others(payload, index):
        acc ^= byte
    return acc


def toyota(payload: bytes, address: int, index: int) -> int:
    """Toyota's checksum: byte sum folded together with the address and length."""
    return (
        sum(_others(payload, index)) + (address & 0xFF) + ((address >> 8) & 0xFF) + len(payload)
    ) & 0xFF


def _crc8(data: bytes, poly: int, init: int, xorout: int) -> int:
    crc = init
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc ^ xorout


def _crc8_factory(poly: int, init: int, xorout: int) -> ChecksumFn:
    def fn(payload: bytes, address: int, index: int) -> int:
        return _crc8(_others(payload, index), poly, init, xorout)

    return fn


def _crc8_table(poly: int) -> np.ndarray:
    """One byte's worth of CRC8 shifting, precomputed for every input."""
    table = np.zeros(256, dtype=np.uint8)
    for value in range(256):
        crc = value
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
        table[value] = crc
    return table


# Vectorised twins of the scalar algorithms above. Each takes an
# (n_frames, width) byte matrix and returns the byte it expects at `index` for
# every frame at once, replacing a Python loop over several hundred thousand
# payloads. `test_vectorised_matches_scalar` holds them to the originals.
VectorFn = Callable[[np.ndarray, int, int], np.ndarray]


def _sum_without(matrix: np.ndarray, index: int) -> np.ndarray:
    return matrix.sum(axis=1, dtype=np.uint32) - matrix[:, index]


def v_sum8(matrix: np.ndarray, address: int, index: int) -> np.ndarray:
    return (_sum_without(matrix, index) & 0xFF).astype(np.uint8)


def v_sum8_complement(matrix: np.ndarray, address: int, index: int) -> np.ndarray:
    return ((-_sum_without(matrix, index).astype(np.int64)) & 0xFF).astype(np.uint8)


def v_xor8(matrix: np.ndarray, address: int, index: int) -> np.ndarray:
    # XOR is self-inverse, so the byte under test removes itself again.
    return np.bitwise_xor.reduce(matrix, axis=1) ^ matrix[:, index]


def v_toyota(matrix: np.ndarray, address: int, index: int) -> np.ndarray:
    folded = (
        _sum_without(matrix, index)
        + (address & 0xFF)
        + ((address >> 8) & 0xFF)
        + matrix.shape[1]
    )
    return (folded & 0xFF).astype(np.uint8)


def _v_crc8_factory(poly: int, init: int, xorout: int) -> VectorFn:
    table = _crc8_table(poly)

    def fn(matrix: np.ndarray, address: int, index: int) -> np.ndarray:
        crc = np.full(matrix.shape[0], init, dtype=np.uint8)
        for column in range(matrix.shape[1]):
            if column == index:
                continue  # the byte under test is not part of its own input
            crc = table[crc ^ matrix[:, column]]
        return crc ^ np.uint8(xorout)

    return fn


# Ordered cheapest-and-commonest first; the search stops at the first algorithm
# that reproduces the byte, so a plain sum is never reported as an exotic CRC.
ALGORITHMS: dict[str, ChecksumFn] = {
    "sum8": sum8,
    "sum8_complement": sum8_complement,
    "xor8": xor8,
    "toyota": toyota,
    "crc8": _crc8_factory(0x07, 0x00, 0x00),
    "crc8_j1850": _crc8_factory(0x1D, 0xFF, 0xFF),
    "crc8_2f": _crc8_factory(0x2F, 0xFF, 0xFF),
    "crc8_autosar": _crc8_factory(0x2F, 0xFF, 0x00),
}


VECTOR_ALGORITHMS: dict[str, VectorFn] = {
    "sum8": v_sum8,
    "sum8_complement": v_sum8_complement,
    "xor8": v_xor8,
    "toyota": v_toyota,
    "crc8": _v_crc8_factory(0x07, 0x00, 0x00),
    "crc8_j1850": _v_crc8_factory(0x1D, 0xFF, 0xFF),
    "crc8_2f": _v_crc8_factory(0x2F, 0xFF, 0xFF),
    "crc8_autosar": _v_crc8_factory(0x2F, 0xFF, 0x00),
}


def as_matrix(payloads: Sequence[bytes]) -> np.ndarray:
    """Equal-width payloads as an (n, width) byte matrix."""
    if not payloads:
        return np.zeros((0, 0), dtype=np.uint8)
    width = len(payloads[0])
    return np.frombuffer(b"".join(payloads), dtype=np.uint8).reshape(len(payloads), width)


@dataclass(frozen=True)
class ChecksumHypothesis:
    """A checksum byte whose value the named algorithm reproduces."""

    byte_index: int
    algorithm: str
    match_rate: float
    frames: int
    # Some algorithms are self-inverse: if byte 7 is the XOR of bytes 0-6 then
    # byte 0 is equally the XOR of bytes 1-7, so every position "verifies" and
    # the trace alone cannot say which byte the protocol calls the checksum.
    ambiguous_positions: tuple[int, ...] = ()

    @property
    def start_bit(self) -> int:
        return self.byte_index * 8

    @property
    def length(self) -> int:
        return 8

    @property
    def ambiguous(self) -> bool:
        return len(self.ambiguous_positions) > 1

    def __str__(self) -> str:
        where = (
            f"any of bytes {list(self.ambiguous_positions)}"
            if self.ambiguous
            else f"byte {self.byte_index}"
        )
        return f"{self.algorithm} @ {where} ({self.match_rate:.1%})"


def score_algorithm(
    payloads: Sequence[bytes], address: int, index: int, fn: ChecksumFn, min_match: float = 0.0
) -> float:
    """Fraction of frames whose byte `index` the algorithm reproduces.

    Abandoned as soon as `min_match` is out of reach. Nearly every candidate is
    wrong and fails on the first frame or two, so scoring the rest is waste --
    this is most of the difference between a segment sweep taking half a minute
    and taking a second.
    """
    total = len(payloads)
    if not total:
        return 0.0
    budget = total - int(min_match * total)
    hits = misses = 0
    for payload in payloads:
        if fn(payload, address, index) == payload[index]:
            hits += 1
        else:
            misses += 1
            if misses > budget:
                return hits / total
    return hits / total


def find_checksums(
    payloads: Sequence[bytes],
    address: int,
    *,
    candidate_bytes: Sequence[int] | None = None,
    min_match: float = 0.99,
    screen_frames: int = 256,
    matrix: np.ndarray | None = None,
) -> list[ChecksumHypothesis]:
    """Search for a byte that a known algorithm reproduces.

    `candidate_bytes` narrows the search to positions that already look like a
    checksum; without it every byte is tried. A cheap screen over the first
    `screen_frames` frames rejects almost everything before the full trace is
    scored, which is what keeps a whole-segment sweep affordable.
    """
    blob = as_matrix(payloads) if matrix is None else matrix
    if blob.shape[0] == 0:
        return []
    # When the matrix is supplied, `payloads` may be only a sample of it; the
    # matrix is what is scored, so the matrix says how many frames there were.
    frames = blob.shape[0]
    width = blob.shape[1]
    positions = candidate_bytes if candidate_bytes is not None else range(width)
    # The screen is now only worth having for very long traces: scoring every
    # frame is one pass over an array rather than a loop in Python.
    screen = blob[:screen_frames]

    hits: dict[str, list[tuple[int, float]]] = {}
    for index in positions:
        if not 0 <= index < width:
            continue
        for name, fn in VECTOR_ALGORITHMS.items():
            if float((fn(screen, address, index) == screen[:, index]).mean()) < min_match:
                continue
            rate = float((fn(blob, address, index) == blob[:, index]).mean())
            if rate >= min_match:
                hits.setdefault(name, []).append((index, rate))
                break  # first (simplest) algorithm that works wins

    found = []
    for name, matches in hits.items():
        indices = tuple(i for i, _ in matches)
        # Convention puts a checksum last, so that is the one reported -- but
        # every position that verified is carried along rather than hidden.
        index, rate = matches[-1]
        found.append(
            ChecksumHypothesis(index, name, rate, frames,
                               ambiguous_positions=indices if len(indices) > 1 else ())
        )
    return sorted(found, key=lambda f: f.byte_index)
