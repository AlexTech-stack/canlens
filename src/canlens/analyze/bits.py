# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Per-bit measurement over a stack of payloads.

Everything here is a *measurement*, not a hypothesis: how often a bit is set,
how often it changes, how much information it carries. Deciding that a run of
bits is a counter or a CRC is inference and belongs in :mod:`canlens.infer`.

Bit ordering is explicit and selectable, because getting it wrong is silent.
Under :data:`BitOrder.INTEL` -- the default, and the numbering DBC files use --
index ``i`` is bit ``i % 8`` of byte ``i // 8`` counting from the *least*
significant bit, so index 0 is the LSB of byte 0 and index 7 its MSB. Under
:data:`BitOrder.MOTOROLA` the walk within each byte is reversed, MSB first.

Only the column order changes: a bit's measurements are the same either way.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np


class BitOrder(str, Enum):
    """How payload bytes are unpacked into a flat bit index.

    INTEL walks each byte LSB-first (DBC bit numbering); MOTOROLA walks it
    MSB-first, as hex is read.
    """

    INTEL = "intel"
    MOTOROLA = "motorola"

    def __str__(self) -> str:
        return self.value

    @property
    def numpy_bitorder(self) -> Literal["big", "little"]:
        return "little" if self is BitOrder.INTEL else "big"


class BitKind(str, Enum):
    """Coarse description of how a bit behaves over a trace.

    Heuristic and threshold-driven -- useful for triage and for narrowing what
    the inference layer has to look at, not a claim about meaning.
    """

    CONSTANT = "constant"  # never changes
    SLOW = "slow"  # changes rarely: state, mode, warning flags
    ACTIVE = "active"  # changes at a moderate rate: physical quantities
    NOISY = "noisy"  # changes almost every frame: counter LSBs, CRCs

    def __str__(self) -> str:
        return self.value


# Transition rate = fraction of consecutive frame pairs where the bit differs.
# A bit flipping on more than 40% of frames is doing so about as often as a
# fair coin, which is what a CRC bit or a counter's low bit looks like. Below
# 1% it is carrying state rather than a value. Both are deliberately loose:
# they exist to partition work, and the inference layer re-tests anything it
# cares about.
SLOW_MAX_RATE = 0.01
NOISY_MIN_RATE = 0.40


def bit_matrix(
    payloads: list[bytes], width: int, order: BitOrder = BitOrder.INTEL
) -> np.ndarray:
    """Stack equal-length payloads into an (n_frames, width * 8) bit matrix."""
    if not payloads:
        return np.zeros((0, width * 8), dtype=np.uint8)
    raw = np.frombuffer(b"".join(payloads), dtype=np.uint8).reshape(len(payloads), width)
    return np.unpackbits(raw, axis=1, bitorder=order.numpy_bitorder)


def bit_entropy(matrix: np.ndarray) -> np.ndarray:
    """Shannon entropy per bit position, in bits (0.0 constant .. 1.0 uniform)."""
    if matrix.shape[0] == 0:
        return np.zeros(matrix.shape[1])
    p = matrix.mean(axis=0)
    # 0 * log2(0) is 0 by convention; np.where alone would still evaluate the
    # log and emit a warning, so clip first and zero the degenerate columns.
    safe = np.clip(p, 1e-12, 1 - 1e-12)
    h = -(safe * np.log2(safe) + (1 - safe) * np.log2(1 - safe))
    return np.where((p == 0) | (p == 1), 0.0, h)


def transition_rate(matrix: np.ndarray) -> np.ndarray:
    """Fraction of consecutive frame pairs in which each bit changes."""
    if matrix.shape[0] < 2:
        return np.zeros(matrix.shape[1])
    return (np.diff(matrix.astype(np.int8), axis=0) != 0).mean(axis=0)


def classify_bits(rates: np.ndarray, entropy: np.ndarray) -> list[BitKind]:
    """Bucket each bit by how much and how fast it moves."""
    kinds = []
    for rate, h in zip(rates, entropy, strict=True):
        if h == 0.0:
            kinds.append(BitKind.CONSTANT)
        elif rate >= NOISY_MIN_RATE:
            kinds.append(BitKind.NOISY)
        elif rate <= SLOW_MAX_RATE:
            kinds.append(BitKind.SLOW)
        else:
            kinds.append(BitKind.ACTIVE)
    return kinds


@dataclass
class BitProfile:
    """Per-bit measurements for one message's payload."""

    width: int
    frames: int
    ones: np.ndarray
    entropy: np.ndarray
    rates: np.ndarray
    kinds: list[BitKind]
    order: BitOrder = BitOrder.INTEL

    @property
    def bits(self) -> int:
        return self.width * 8

    def count(self, kind: BitKind) -> int:
        return sum(1 for k in self.kinds if k is kind)

    @property
    def constant_bits(self) -> int:
        return self.count(BitKind.CONSTANT)

    @property
    def payload_entropy(self) -> float:
        """Total entropy across the payload, in bits.

        This is the headline triage number: a message whose payload carries
        almost no entropy has nothing in it to reverse engineer.
        """
        return float(self.entropy.sum())


def profile_bits(
    payloads: list[bytes], width: int, order: BitOrder = BitOrder.INTEL
) -> BitProfile:
    """Measure every bit position across a stack of same-width payloads."""
    matrix = bit_matrix(payloads, width, order)
    entropy = bit_entropy(matrix)
    rates = transition_rate(matrix)
    ones = matrix.mean(axis=0) if matrix.shape[0] else np.zeros(width * 8)
    return BitProfile(
        width=width,
        frames=matrix.shape[0],
        ones=ones,
        entropy=entropy,
        rates=rates,
        kinds=classify_bits(rates, entropy),
        order=order,
    )
