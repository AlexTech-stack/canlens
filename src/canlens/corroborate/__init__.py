# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Cross-segment and cross-platform corroboration.

The layer a single-trace tool cannot have. A hypothesis that holds across many
segments from many independent cars is established rather than guessed, and
the spread of segments reveals bits -- rarely-changing states -- that no one
trace can show at all.
"""

from .boundaries import (
    MIN_PLATFORMS,
    BoundarySupport,
    MessageBoundaries,
    agree,
    boundaries_across,
)
from .buses import (
    MIN_OVERLAP,
    BusIdentity,
    BusMap,
    BusSignature,
    assign,
    identify_buses,
    overlap,
    signatures,
)
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
from .pooled import (
    Pool,
    PooledFinding,
    corroborate_p22,
    evidence_growth,
    pool_frames,
    pool_message,
    pool_messages,
    signals_across,
    solve_p22,
    unique_rows,
    would_pass_alone,
)

__all__ = [
    "ESTABLISHED_SUPPORT",
    "MIN_DEVICES",
    "MIN_OVERLAP",
    "MIN_PLATFORMS",
    "PARTIAL_SUPPORT",
    "BitConsensus",
    "BoundarySupport",
    "BusIdentity",
    "BusMap",
    "BusSignature",
    "ChecksumConsensus",
    "CounterConsensus",
    "Crc16Consensus",
    "Evidence",
    "MessageBoundaries",
    "MessageConsensus",
    "MultiplexConsensus",
    "PlatformConsensus",
    "Pool",
    "PooledFinding",
    "Tier",
    "agree",
    "assign",
    "boundaries_across",
    "corroborate",
    "corroborate_p22",
    "corroborate_platform",
    "device_of",
    "evidence_growth",
    "identify_buses",
    "overlap",
    "pool_frames",
    "pool_message",
    "pool_messages",
    "signals_across",
    "signatures",
    "solve_p22",
    "unique_rows",
    "would_pass_alone",
]
