# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Pooling several segments to settle what none of them could alone.

Synthetic traces with a known Data ID list, so the thing being tested is
whether pooling supplies the missing equations -- not whether the corpus
happens to contain a suitable message.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from canlens.corroborate.pooled import (
    Pool,
    corroborate_p22,
    evidence_growth,
    pool_message,
    solve_p22,
    unique_rows,
    would_pass_alone,
)
from canlens.decode import CanFrame, from_frames
from canlens.infer.checksums import as_matrix
from canlens.infer.profiles import p22_crc, p22_id_list

IDS = [0x10 + i for i in range(16)]


def p22_payloads(n=64, width=8, crc_index=0, bodies=None, seed=0):
    """Frames carrying a Profile 22 CRC, over a limited set of bodies.

    `bodies` bounds how many distinct payloads exist, which is the whole
    point: sixteen unknowns against sixteen equations is not evidence.
    """
    rng = random.Random(seed)
    pool = bodies if bodies is not None else [
        bytes(rng.randrange(256) for _ in range(width)) for _ in range(n)
    ]
    # Byte 1's high nibble must not be zero, or the counter below it reads as
    # a wider field than four bits and `find_p22` will not accept it -- the
    # same shape real VW traffic has, where that nibble carries other data.
    out = []
    for i in range(n):
        body = bytearray(pool[i % len(pool)])
        body[1] = (body[1] & 0xF0) | (i % 16)
        body[crc_index] = p22_crc(bytes(body), crc_index, IDS[i % 16])
        out.append(bytes(body))
    return out


def frames_of(payloads, bus=0, address=0x120):
    return from_frames(
        [CanFrame(i * 10_000_000, bus, address, p, False) for i, p in enumerate(payloads)]
    )


class TestUniqueRows:
    def test_keeps_first_appearance_order(self):
        matrix = np.array([[3, 3], [1, 1], [3, 3], [2, 2]], dtype=np.uint8)
        assert unique_rows(matrix).tolist() == [[3, 3], [1, 1], [2, 2]]

    def test_an_empty_matrix_stays_empty(self):
        assert unique_rows(np.zeros((0, 4), dtype=np.uint8)).shape == (0, 4)

    def test_all_distinct_is_unchanged(self):
        matrix = np.arange(12, dtype=np.uint8).reshape(3, 4)
        assert unique_rows(matrix).tolist() == matrix.tolist()


class TestPoolMessage:
    def test_gathers_across_segments_and_counts_devices(self, tmp_path):
        traces = {
            "a/r/1": frames_of(p22_payloads(seed=1)),
            "b/r/1": frames_of(p22_payloads(seed=2)),
        }
        pool = pool_message(
            list(traces), (0, 0x120), root=str(tmp_path),
            load=lambda path, root: traces[path],
        )
        assert pool is not None
        assert pool.segments == 2 and pool.devices == 2
        assert pool.frames == 128
        assert pool.distinct > max(pool.per_segment)
        assert pool.new_from_pooling == pool.distinct - max(pool.per_segment)

    def test_a_message_absent_from_a_segment_is_skipped(self, tmp_path):
        traces = {
            "a/r/1": frames_of(p22_payloads(seed=1)),
            "b/r/1": frames_of(p22_payloads(seed=2), address=0x999),
        }
        pool = pool_message(
            list(traces), (0, 0x120), root=str(tmp_path),
            load=lambda path, root: traces[path],
        )
        assert pool is not None and pool.segments == 1

    def test_segments_of_a_different_width_are_not_mixed(self, tmp_path):
        traces = {
            "a/r/1": frames_of(p22_payloads(seed=1, width=8)),
            "b/r/1": frames_of(p22_payloads(seed=2, width=6)),
        }
        pool = pool_message(
            list(traces), (0, 0x120), root=str(tmp_path),
            load=lambda path, root: traces[path],
        )
        assert pool is not None and pool.segments == 1 and pool.width == 8

    def test_nothing_anywhere_gives_nothing(self, tmp_path):
        traces = {"a/r/1": frames_of(p22_payloads(), address=0x999)}
        assert pool_message(
            list(traces), (0, 0x120), root=str(tmp_path),
            load=lambda path, root: traces[path],
        ) is None

    def test_an_unreadable_segment_is_skipped_rather_than_fatal(self, tmp_path):
        traces = {"a/r/1": frames_of(p22_payloads(seed=1))}

        def load(path, root):
            if path == "missing":
                raise OSError("gone")
            return traces[path]

        pool = pool_message(["missing", "a/r/1"], (0, 0x120), root=str(tmp_path), load=load)
        assert pool is not None and pool.segments == 1


class TestEvidence:
    def test_sixteen_contents_never_clears_the_bar_for_sixteen_unknowns(self):
        """The case the whole module exists for."""
        bodies = [bytes([0, 0xA0, 0, 0, 0, 0, 0, 0])]
        matrix = as_matrix(p22_payloads(n=64, bodies=bodies))
        assert not would_pass_alone(matrix, 0)

    def test_plenty_of_contents_clears_it(self):
        matrix = as_matrix(p22_payloads(n=200, seed=4))
        assert would_pass_alone(matrix, 0)

    def test_growth_reports_alone_against_pooled(self, tmp_path):
        traces = {
            "a/r/1": frames_of(p22_payloads(seed=1)),
            "b/r/1": frames_of(p22_payloads(seed=2)),
        }
        pool = pool_message(
            list(traces), (0, 0x120), root=str(tmp_path),
            load=lambda path, root: traces[path],
        )
        alone, pooled = evidence_growth(pool, 0)
        assert pooled > alone


class TestSolveP22:
    @staticmethod
    def thin_pool(segments: int, tmp_path) -> Pool:
        """Segments that each carry too little, and together carry enough.

        Each contributes a different handful of bodies, so the union grows
        rather than repeating -- which is the difference between pooling that
        helps and pooling that does not.
        """
        traces = {}
        for s in range(segments):
            rng = random.Random(100 + s)
            bodies = [
                bytes([0, 0xA0, *(rng.randrange(256) for _ in range(6))]) for _ in range(4)
            ]
            # Long enough that the four-bit counter wins its window. A short
            # trace makes the five bits at 8 look like a counter too, because
            # one wrap in sixteen is forgiven at 95% when there are only a few
            # wraps to see; over 256 frames it is not.
            traces[f"dev{s}/r/1"] = frames_of(p22_payloads(n=256, bodies=bodies))
        return pool_message(
            list(traces), (0, 0x120), root=str(tmp_path),
            load=lambda path, root: traces[path],
        )

    def test_one_segment_is_refused(self, tmp_path):
        pool = self.thin_pool(1, tmp_path)
        assert not would_pass_alone(pool.matrix, 0)
        assert solve_p22(pool, min_devices=1) == []

    def test_the_counter_is_located_on_ordered_frames(self, tmp_path):
        """Reducing to distinct contents can destroy the sequence a counter is.

        On a real Golf the deduplicated pool of GRA_ACC_01 has its counter
        nibble running 4, 5, ... 15, 0, 1, 2, 3, 7, 8, 13, 14 -- no stride
        survives, and the list search is left with no field to group by.
        Shuffling the pool here reproduces that: every equation is still
        present, and only the order is gone.
        """
        import dataclasses

        from canlens.analyze.bits import BitOrder, bit_matrix_from_bytes
        from canlens.infer.counters import find_counters

        def nibble_at_8(matrix):
            return [
                c for c in find_counters(bit_matrix_from_bytes(matrix, BitOrder.INTEL))
                if c.length == 4 and c.start_bit == 8
            ]

        pool = self.thin_pool(10, tmp_path)
        assert nibble_at_8(pool.ordered), "the fixture must present a clean counter"

        rng = np.random.default_rng(0)
        shuffled = pool.matrix[rng.permutation(pool.matrix.shape[0])]
        assert not nibble_at_8(shuffled), "the shuffle must destroy it, or this proves nothing"

        found = solve_p22(dataclasses.replace(pool, matrix=shuffled))
        assert found and p22_id_list(found[0].data_id) == IDS

    def test_enough_segments_settle_the_list(self, tmp_path):
        pool = self.thin_pool(10, tmp_path)
        assert pool.distinct > max(pool.per_segment)
        found = solve_p22(pool)
        assert found, f"{pool.distinct} pooled contents should be enough"
        assert p22_id_list(found[0].data_id) == IDS

    def test_one_device_is_never_enough_however_much_data(self, tmp_path):
        """A platform claim needs more than one car behind it."""
        payloads = p22_payloads(n=400, seed=7)
        traces = {"solo/r/1": frames_of(payloads)}
        pool = pool_message(
            list(traces), (0, 0x120), root=str(tmp_path),
            load=lambda path, root: traces[path],
        )
        assert would_pass_alone(pool.matrix, 0)
        assert solve_p22(pool, min_devices=2) == []
        assert solve_p22(pool, min_devices=1)

    def test_a_pool_of_identical_segments_gains_nothing(self, tmp_path):
        """Repeating the same content adds frames, not equations."""
        bodies = [bytes([0, 0xA0, 0, 0, 0, 0, 0, 0])]
        same = p22_payloads(n=64, bodies=bodies)
        traces = {f"dev{s}/r/1": frames_of(same) for s in range(8)}
        pool = pool_message(
            list(traces), (0, 0x120), root=str(tmp_path),
            load=lambda path, root: traces[path],
        )
        assert pool.new_from_pooling == 0
        assert solve_p22(pool) == []


def test_a_platform_with_too_few_segments_reports_nothing(tmp_path):
    (tmp_path / "database.json").write_text('{"segments": []}')
    assert corroborate_p22("NOT_HELD", root=str(tmp_path)) == []


def test_pool_renders_its_provenance(tmp_path):
    traces = {"a/r/1": frames_of(p22_payloads(seed=1)), "b/r/1": frames_of(p22_payloads(seed=2))}
    pool = pool_message(
        list(traces), (0, 0x120), root=str(tmp_path), load=lambda path, root: traces[path]
    )
    text = str(pool)
    assert "0x120" in text and "2 segments" in text and "2 devices" in text


@pytest.mark.parametrize("segments", [2, 5])
def test_more_segments_never_reduce_the_evidence(segments, tmp_path):
    pool = TestSolveP22.thin_pool(segments, tmp_path)
    smaller = TestSolveP22.thin_pool(segments - 1, tmp_path)
    assert pool.distinct >= smaller.distinct
