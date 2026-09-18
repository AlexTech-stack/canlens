# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""The inference results cache, and the payload sampling it sits on top of."""
from __future__ import annotations

import os
import random

import pytest

from canlens.decode import CanFrame, from_frames
from canlens.decode.cache import Source
from canlens.infer import INFER_VERSION, infer_frameset
from canlens.infer.checksums import toyota
from canlens.infer.results import (
    Key,
    clear_results,
    infer_cached,
    load_results,
    results_path,
    save_results,
)


def trace(count: int = 300) -> list[CanFrame]:
    rng = random.Random(1)
    out = []
    for i in range(count):
        body = bytearray([i % 256, *(rng.randrange(256) for _ in range(6)), 0])
        body[7] = toyota(bytes(body), 0x210, 7)
        out.append(CanFrame(i * 10_000_000, 1, 0x210, bytes(body), False))
    return out


@pytest.fixture
def corpus(tmp_path):
    """A segment file on disk (contents irrelevant: frames are supplied)."""
    path = tmp_path / "segments" / "dev" / "route--a" / "1" / "rlog.zst"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"stand-in")
    return str(tmp_path), str(path)


class TestPayloadSampling:
    """Detectors score the matrix; only a handful of payloads are materialised."""

    def test_frame_counts_come_from_the_matrix_not_the_sample(self):
        frames = from_frames(trace(300))
        result = infer_frameset(frames)[0]
        assert result.frames == 300
        assert all(c.frames == 300 for c in result.counters)
        assert all(s.frames == 300 for s in result.checksums)

    def test_findings_are_unchanged_by_sampling(self):
        frames = from_frames(trace(300))
        result = infer_frameset(frames)[0]
        assert [(c.start_bit, c.length) for c in result.counters] == [(0, 8)]
        assert [(s.byte_index, s.algorithm) for s in result.checksums] == [(7, "toyota")]


class TestProfileReuse:
    def test_a_supplied_profile_gives_the_same_answer(self):
        from canlens.analyze import analyze_frameset

        frames = from_frames(trace(300))
        alone = infer_frameset(frames)
        shared = infer_frameset(frames, profile=analyze_frameset(frames))
        assert [(m.key, m.bits.kinds) for m in alone] == [(m.key, m.bits.kinds) for m in shared]
        assert [c.stride for m in alone for c in m.counters] == [
            c.stride for m in shared for c in m.counters
        ]

    def test_a_mismatched_profile_is_ignored_not_trusted(self):
        from canlens.analyze import analyze_frameset

        frames = from_frames(trace(300))
        other = analyze_frameset(from_frames(trace(120)))  # same key, fewer frames
        result = infer_frameset(frames, profile=other)[0]
        assert result.bits.frames == 300


class TestResultsCache:
    def test_a_miss_infers_and_a_hit_does_not(self, corpus, monkeypatch):
        root, path = corpus
        frames = from_frames(trace())
        first = infer_cached(path, root=root, frames=frames)
        assert os.path.exists(results_path(root, path))

        import canlens.infer.results as module

        def boom(*args, **kwargs):
            raise AssertionError("a cache hit must not infer again")

        monkeypatch.setattr(module, "infer_frameset", boom)
        second = infer_cached(path, root=root, frames=frames)
        assert [(m.key, len(m.counters)) for m in second] == [(m.key, len(m.counters)) for m in first]

    def test_results_survive_the_round_trip_intact(self, corpus):
        root, path = corpus
        frames = from_frames(trace())
        first = infer_cached(path, root=root, frames=frames)
        second = infer_cached(path, root=root, frames=frames)
        assert second[0].counters == first[0].counters
        assert second[0].checksums == first[0].checksums
        assert second[0].bits.kinds == first[0].bits.kinds

    def test_changing_the_segment_invalidates(self, corpus):
        root, path = corpus
        infer_cached(path, root=root, frames=from_frames(trace()))
        with open(path, "wb") as handle:
            handle.write(b"rewritten, and longer than before")
        key = Key(INFER_VERSION, Source.of(path), "intel", 32)
        assert load_results(results_path(root, path), key) is None

    def test_a_version_bump_invalidates(self, corpus):
        root, path = corpus
        infer_cached(path, root=root, frames=from_frames(trace()))
        key = Key(INFER_VERSION + 1, Source.of(path), "intel", 32)
        assert load_results(results_path(root, path), key) is None

    def test_different_parameters_do_not_share_an_entry(self, corpus):
        root, path = corpus
        infer_cached(path, root=root, frames=from_frames(trace()), min_frames=32)
        key = Key(INFER_VERSION, Source.of(path), "intel", 16)
        assert load_results(results_path(root, path), key) is None

    def test_a_corrupt_entry_is_a_miss_not_a_crash(self, corpus):
        root, path = corpus
        infer_cached(path, root=root, frames=from_frames(trace()))
        with open(results_path(root, path), "wb") as handle:
            handle.write(b"\x80\x04not really a pickle")
        assert infer_cached(path, root=root, frames=from_frames(trace()))

    def test_no_part_file_is_left_behind(self, corpus):
        root, path = corpus
        infer_cached(path, root=root, frames=from_frames(trace()))
        directory = os.path.dirname(results_path(root, path))
        assert not [f for f in os.listdir(directory) if ".part" in f]

    def test_use_cache_false_neither_reads_nor_writes(self, corpus):
        root, path = corpus
        infer_cached(path, root=root, frames=from_frames(trace()), use_cache=False)
        assert not os.path.exists(results_path(root, path))

    def test_clear_removes_results_only(self, corpus):
        root, path = corpus
        infer_cached(path, root=root, frames=from_frames(trace()))
        assert clear_results(root) == 1
        assert clear_results(root) == 0
        assert os.path.exists(path)

    def test_the_entry_sits_beside_the_frame_cache(self, corpus):
        from canlens.decode import cache_path

        root, path = corpus
        assert os.path.dirname(results_path(root, path)) == os.path.dirname(cache_path(root, path))
        assert results_path(root, path).endswith(".infer.pkl")

    def test_save_and_load_directly(self, corpus):
        root, path = corpus
        key = Key(INFER_VERSION, Source.of(path), "intel", 32)
        results = infer_frameset(from_frames(trace()))
        save_results(results, results_path(root, path), key)
        back = load_results(results_path(root, path), key)
        # Whole-object equality is not defined for a dataclass holding numpy
        # arrays; the fields that carry the findings are compared instead.
        assert back is not None
        assert [(m.key, m.counters, m.checksums, m.crc16s, m.bits.kinds) for m in back] == [
            (m.key, m.counters, m.checksums, m.crc16s, m.bits.kinds) for m in results
        ]
