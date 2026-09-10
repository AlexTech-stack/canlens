# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Single-trace measurement: timing, entropy, per-bit classification.

This layer decides whether a trace is worth a full inference pass and hands
:mod:`canlens.infer` the statistics it needs. It measures; it does not
conclude. Anything that amounts to a claim about what a bit *means* belongs a
layer up.
"""

from .bits import (
    KIND_ORDER,
    BitKind,
    BitOrder,
    BitProfile,
    bit_entropy,
    bit_matrix,
    classify_bits,
    profile_bits,
    transition_rate,
)
from .timing import Cadence, TimingProfile, profile_timing
from .trace import MessageProfile, TraceProfile, analyze_frames, analyze_segment

__all__ = [
    "KIND_ORDER",
    "BitKind",
    "BitOrder",
    "BitProfile",
    "Cadence",
    "MessageProfile",
    "TimingProfile",
    "TraceProfile",
    "analyze_frames",
    "analyze_segment",
    "bit_entropy",
    "bit_matrix",
    "classify_bits",
    "profile_bits",
    "profile_timing",
    "transition_rate",
]
