# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Signal inference within a single trace.

Counters and checksums, each stated only when it has been *checked* against
the trace rather than guessed at from how a bit looks. Corroborating a finding
across many traces is :mod:`canlens.corroborate`'s job.
"""

from .checksums import ALGORITHMS, ChecksumHypothesis, find_checksums, score_algorithm
from .counters import CounterHypothesis, field_values, find_counters, score_counter
from .message import (
    MessageInference,
    checksum_candidate_bytes,
    infer_frames,
    infer_message,
    infer_segment,
)

__all__ = [
    "ALGORITHMS",
    "ChecksumHypothesis",
    "CounterHypothesis",
    "MessageInference",
    "checksum_candidate_bytes",
    "field_values",
    "find_checksums",
    "find_counters",
    "infer_frames",
    "infer_message",
    "infer_segment",
    "score_algorithm",
    "score_counter",
]
