# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Signal inference within a single trace.

Counters and checksums, each stated only when it has been *checked* against
the trace rather than guessed at from how a bit looks. Corroborating a finding
across many traces is :mod:`canlens.corroborate`'s job.
"""

from .checksums import (
    ALGORITHMS,
    ChecksumHypothesis,
    e2e_crc8,
    find_checksums,
    find_e2e_crc8,
    p01_low_id,
    score_algorithm,
)
from .counters import (
    CounterHypothesis,
    field_values,
    find_counters,
    score_counter,
)
from .crc16 import (
    CRC16_ALGORITHMS,
    Crc16Hypothesis,
    crc16_autosar,
    e2e_p05,
    find_crc16,
    solve_data_id,
)
from .message import (
    MessageInference,
    checksum_candidate_bytes,
    infer_frames,
    infer_frameset,
    infer_message,
    infer_message_columnar,
    infer_segment,
)
from .results import INFER_VERSION, clear_results, infer_cached, warm_segment

__all__ = [
    "ALGORITHMS",
    "CRC16_ALGORITHMS",
    "INFER_VERSION",
    "ChecksumHypothesis",
    "CounterHypothesis",
    "Crc16Hypothesis",
    "MessageInference",
    "checksum_candidate_bytes",
    "clear_results",
    "crc16_autosar",
    "e2e_crc8",
    "e2e_p05",
    "field_values",
    "find_checksums",
    "find_counters",
    "find_crc16",
    "find_e2e_crc8",
    "infer_cached",
    "infer_frames",
    "infer_frameset",
    "infer_message",
    "infer_message_columnar",
    "infer_segment",
    "p01_low_id",
    "score_algorithm",
    "score_counter",
    "solve_data_id",
    "warm_segment",
]
