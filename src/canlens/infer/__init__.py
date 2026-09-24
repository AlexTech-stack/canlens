# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Signal inference within a single trace.

Counters and checksums, each stated only when it has been *checked* against
the trace rather than guessed at from how a bit looks. Corroborating a finding
across many traces is :mod:`canlens.corroborate`'s job.
"""

from .byteorder import (
    LONG_FIELD,
    MIN_MARGIN,
    BusOrder,
    decide_byte_order,
    motorola_columns,
)
from .checksums import (
    ALGORITHMS,
    ChecksumHypothesis,
    e2e_crc8,
    find_checksums,
    find_e2e_crc8,
    find_honda_nibble,
    honda_nibble,
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
from .isotp import (
    MIN_MESSAGES,
    IsotpHypothesis,
    Transfer,
    find_isotp,
    ordered_frames,
    transport_bits,
)
from .layouts import LayoutField, read_layout_fields
from .message import (
    MessageInference,
    checksum_candidate_bytes,
    infer_frames,
    infer_frameset,
    infer_message,
    infer_message_columnar,
    infer_segment,
)
from .multiplex import MultiplexHypothesis, find_multiplexor
from .results import INFER_VERSION, clear_results, infer_cached, warm_segment
from .signals import SignalHypothesis, find_signals
from .smoothness import (
    JUMP_FRACTION,
    MAX_JUMP_SHARE,
    filter_signals,
    jump_share,
    v_jump_share,
)

__all__ = [
    "ALGORITHMS",
    "CRC16_ALGORITHMS",
    "INFER_VERSION",
    "JUMP_FRACTION",
    "LONG_FIELD",
    "MAX_JUMP_SHARE",
    "MIN_MARGIN",
    "MIN_MESSAGES",
    "BusOrder",
    "ChecksumHypothesis",
    "CounterHypothesis",
    "Crc16Hypothesis",
    "IsotpHypothesis",
    "LayoutField",
    "MessageInference",
    "MultiplexHypothesis",
    "SignalHypothesis",
    "Transfer",
    "checksum_candidate_bytes",
    "clear_results",
    "crc16_autosar",
    "decide_byte_order",
    "e2e_crc8",
    "e2e_p05",
    "field_values",
    "filter_signals",
    "find_checksums",
    "find_counters",
    "find_crc16",
    "find_e2e_crc8",
    "find_honda_nibble",
    "find_isotp",
    "find_multiplexor",
    "find_signals",
    "honda_nibble",
    "infer_cached",
    "infer_frames",
    "infer_frameset",
    "infer_message",
    "infer_message_columnar",
    "infer_segment",
    "jump_share",
    "motorola_columns",
    "ordered_frames",
    "p01_low_id",
    "read_layout_fields",
    "score_algorithm",
    "score_counter",
    "solve_data_id",
    "transport_bits",
    "v_jump_share",
    "warm_segment",
]
