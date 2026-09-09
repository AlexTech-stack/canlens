# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Decoder tests against real corpus data.

Skipped unless a segment has actually been fetched -- these are the tests that
prove the schema pins and echo handling work on real logs, so they are worth
running whenever data is present.
"""
from __future__ import annotations

import glob
import os

import pytest

from canlens.decode import iter_frames, summarize

ROOT = os.environ.get("CANLENS_DATA", os.path.expanduser("~/data/canlens"))
SEGMENTS = sorted(glob.glob(os.path.join(ROOT, "segments", "*", "*", "*", "rlog.zst")))

pytestmark = pytest.mark.skipif(
    not SEGMENTS, reason=f"no corpus segments under {ROOT}; run 'canlens corpus fetch'"
)

pytest.importorskip("capnp", reason="needs the 'decode' extra")


@pytest.fixture(scope="module")
def segment() -> str:
    return SEGMENTS[0]


def test_decodes_frames(segment):
    frames = list(iter_frames(segment, root=ROOT))
    assert frames, "expected at least one CAN frame"
    assert all(not f.echo for f in frames), "echoes must be excluded by default"


def test_frames_look_like_can(segment):
    for frame in list(iter_frames(segment, root=ROOT))[:5000]:
        assert 0 <= frame.address <= 0x1FFFFFFF
        assert len(frame.data) <= 64  # CAN FD maximum
        assert frame.mono_ns > 0


def test_timestamps_are_monotonic(segment):
    stamps = [f.mono_ns for f in iter_frames(segment, root=ROOT)]
    assert stamps == sorted(stamps)


def test_echoes_are_a_real_share_of_the_log(segment):
    summary = summarize(segment, root=ROOT)
    # Measured at 30-46% across sampled segments; if this ever reads zero the
    # src classification has silently stopped working.
    assert summary.echoes > 0
    assert 0.1 < summary.echoes / summary.frames < 0.7


def test_include_echo_yields_strictly_more(segment):
    without = sum(1 for _ in iter_frames(segment, root=ROOT))
    with_echo = sum(1 for _ in iter_frames(segment, root=ROOT, include_echo=True))
    assert with_echo > without
