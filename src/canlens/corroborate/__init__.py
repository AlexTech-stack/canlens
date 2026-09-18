# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Cross-segment and cross-platform corroboration.

The layer a single-trace tool cannot have. A hypothesis that holds across many
segments from many independent cars is established rather than guessed, and
the spread of segments reveals bits -- rarely-changing states -- that no one
trace can show at all.
"""

from .consensus import (
    ESTABLISHED_SUPPORT,
    MIN_DEVICES,
    PARTIAL_SUPPORT,
    BitConsensus,
    ChecksumConsensus,
    CounterConsensus,
    Crc16Consensus,
    Evidence,
    MessageConsensus,
    MultiplexConsensus,
    PlatformConsensus,
    Tier,
    corroborate,
    corroborate_platform,
    device_of,
)

__all__ = [
    "ESTABLISHED_SUPPORT",
    "MIN_DEVICES",
    "PARTIAL_SUPPORT",
    "BitConsensus",
    "ChecksumConsensus",
    "CounterConsensus",
    "Crc16Consensus",
    "Evidence",
    "MessageConsensus",
    "MultiplexConsensus",
    "PlatformConsensus",
    "Tier",
    "corroborate",
    "corroborate_platform",
    "device_of",
]
