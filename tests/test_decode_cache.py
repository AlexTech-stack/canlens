# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""The decode cache: correctness, and knowing when it is stale."""
from __future__ import annotations

import os

import pytest

from canlens.decode import CanFrame, cache_path, clear, from_frames
from canlens.decode.cache import CACHE_DIR, CACHE_VERSION, Source, load, save


@pytest.fixture
def frames():
    return from_frames([
        CanFrame(10, 0, 0x100, b"\x01\x02", False),
        CanFrame(20, 2, 0x200, b"\xaa\xbb\xcc\xdd", True),
        CanFrame(30, 1, 0x300, b"\x09", False),
    ])


@pytest.fixture
def segment(tmp_path):
    path = tmp_path / "segments" / "dev" / "route--a" / "1" / "rlog.zst"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"pretend this is zstd")
    return str(path)


class TestCachePath:
    def test_mirrors_the_stored_layout(self, tmp_path, segment):
        got = cache_path(str(tmp_path), segment)
        assert got.endswith(os.path.join(CACHE_DIR, "dev", "route--a", "1.npz"))

    def test_accepts_a_path_without_the_file(self, tmp_path):
        a = cache_path(str(tmp_path), "dev/route--a/1/rlog.zst")
        b = cache_path(str(tmp_path), "dev/route--a/1")
        assert a == b

    def test_lands_inside_the_root(self, tmp_path, segment):
        assert cache_path(str(tmp_path), segment).startswith(str(tmp_path))


class TestRoundTrip:
    def test_what_goes_in_comes_out(self, tmp_path, frames, segment):
        destination = cache_path(str(tmp_path), segment)
        source = Source.of(segment)
        save(frames, destination, source)
        back = load(destination, source)
        assert back is not None
        assert back.mono_ns.tolist() == frames.mono_ns.tolist()
        assert back.address.tolist() == frames.address.tolist()
        assert back.src.tolist() == frames.src.tolist()
        assert back.lengths.tolist() == frames.lengths.tolist()
        assert list(back) == list(frames)

    def test_echoes_are_stored_not_filtered(self, tmp_path, frames, segment):
        # One cache must serve a caller that wants echoes and one that does not.
        destination = cache_path(str(tmp_path), segment)
        save(frames, destination, Source.of(segment))
        assert load(destination, Source.of(segment)).echo.any()

    def test_no_stray_part_file_is_left(self, tmp_path, frames, segment):
        destination = cache_path(str(tmp_path), segment)
        save(frames, destination, Source.of(segment))
        leftovers = [f for f in os.listdir(os.path.dirname(destination)) if ".part" in f]
        assert leftovers == []


class TestStaleness:
    @pytest.fixture
    def written(self, tmp_path, frames, segment):
        destination = cache_path(str(tmp_path), segment)
        save(frames, destination, Source.of(segment))
        return destination

    def test_a_missing_entry_is_a_miss(self, tmp_path, segment):
        assert load(cache_path(str(tmp_path), segment), Source.of(segment)) is None

    def test_a_changed_size_invalidates(self, written, segment):
        source = Source.of(segment)
        assert load(written, Source(source.size + 1, source.mtime_ns)) is None

    def test_a_changed_mtime_invalidates(self, written, segment):
        source = Source.of(segment)
        assert load(written, Source(source.size, source.mtime_ns + 1)) is None

    def test_rewriting_the_segment_invalidates(self, written, segment):
        with open(segment, "wb") as handle:
            handle.write(b"different contents entirely")
        assert load(written, Source.of(segment)) is None

    def test_an_older_version_is_re_decoded_not_misread(self, written, segment, monkeypatch):
        import canlens.decode.cache as module

        monkeypatch.setattr(module, "CACHE_VERSION", CACHE_VERSION + 1)
        assert module.load(written, Source.of(segment)) is None

    def test_a_corrupt_entry_is_a_miss_not_a_crash(self, written, segment):
        with open(written, "wb") as handle:
            handle.write(b"not an npz at all")
        assert load(written, Source.of(segment)) is None

    def test_a_truncated_entry_is_a_miss(self, written, segment):
        with open(written, "rb") as handle:
            data = handle.read()
        with open(written, "wb") as handle:
            handle.write(data[: len(data) // 2])
        assert load(written, Source.of(segment)) is None


class TestClear:
    def test_removes_entries_and_reports_how_many(self, tmp_path, frames, segment):
        save(frames, cache_path(str(tmp_path), segment), Source.of(segment))
        assert clear(str(tmp_path)) == 1
        assert clear(str(tmp_path)) == 0

    def test_leaves_the_segments_alone(self, tmp_path, frames, segment):
        save(frames, cache_path(str(tmp_path), segment), Source.of(segment))
        clear(str(tmp_path))
        assert os.path.exists(segment)

    def test_clearing_an_empty_root_is_not_an_error(self, tmp_path):
        assert clear(str(tmp_path)) == 0


class TestSourceIdentity:
    def test_reads_size_and_mtime(self, segment):
        source = Source.of(segment)
        assert source.size == os.path.getsize(segment)
        assert source.mtime_ns == os.stat(segment).st_mtime_ns

    def test_equal_for_an_unchanged_file(self, segment):
        assert Source.of(segment) == Source.of(segment)
