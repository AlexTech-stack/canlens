# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Inter-arrival measurement for one message.

Timing is the cheapest signal in a trace and it constrains everything else: a
strictly cyclic 10 ms message is a different kind of object from one that only
appears when a door opens, and a counter's stride has to agree with the cycle
time to be believable.

The only timestamp available is the enclosing capnp event's `logMonoTime`, so
frames batched into one event share a stamp. That puts a floor on resolvable
jitter and is why `cyclic` is judged on relative spread rather than absolute.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Cadence(str, Enum):
    CYCLIC = "cyclic"  # regular period, low relative spread
    SPORADIC = "sporadic"  # irregular: event-driven, or bursty
    SINGLE = "single"  # too few observations to say anything

    def __str__(self) -> str:
        return self.value


# Relative spread = IQR / median of the inter-arrival gaps. A strictly cyclic
# CAN message on a healthy bus sits far below this; the threshold is loose
# enough to tolerate the shared-timestamp quantisation described above.
CYCLIC_MAX_SPREAD = 0.25
MIN_SAMPLES = 3


@dataclass
class TimingProfile:
    """Inter-arrival statistics for one message."""

    count: int
    cadence: Cadence
    period_ms: float
    jitter_ms: float
    spread: float
    duration_s: float

    @property
    def rate_hz(self) -> float:
        return 1000.0 / self.period_ms if self.period_ms > 0 else 0.0


def profile_timing(stamps_ns: list[int]) -> TimingProfile:
    """Measure the cadence of one message from its arrival timestamps."""
    n = len(stamps_ns)
    duration = (stamps_ns[-1] - stamps_ns[0]) / 1e9 if n >= 2 else 0.0
    if n < MIN_SAMPLES:
        return TimingProfile(n, Cadence.SINGLE, 0.0, 0.0, 0.0, duration)

    gaps = np.diff(np.asarray(stamps_ns, dtype=np.int64)) / 1e6  # ms
    median = float(np.median(gaps))
    q1, q3 = np.percentile(gaps, [25, 75])
    iqr = float(q3 - q1)
    spread = iqr / median if median > 0 else float("inf")
    cadence = Cadence.CYCLIC if spread <= CYCLIC_MAX_SPREAD else Cadence.SPORADIC
    return TimingProfile(
        count=n,
        cadence=cadence,
        period_ms=median,
        jitter_ms=float(np.std(gaps)),
        spread=spread,
        duration_s=duration,
    )
