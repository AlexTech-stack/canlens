# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Local inventory and deletion.

Every test builds its own corpus under tmp_path. Nothing here may point at a
real fetched corpus: these tests delete what they are given.
"""
from __future__ import annotations

import json
import os

import pytest

from canlens.corpus import Manifest, delete_segments, disk_free, inventory, walk_local
from canlens.corpus.local import SEGMENT_FILE, LocalPlatform


@pytest.fixture
def corpus(tmp_path):
    """A small corpus on disk plus the manifest describing it."""
    layout = {
        "TOYOTA_PRIUS": ["dev0/route0--aa/1", "dev0/route0--aa/2"],
        "KIA_EV6": ["dev1/route1--bb/7"],
    }
    manifest_data = {key: [f"{s}/s" for s in paths] for key, paths in layout.items()}
    manifest_data["HONDA_ACCORD"] = ["dev9/route9--zz/3/s"]  # in the manifest, not on disk
    (tmp_path / "database.json").write_text(json.dumps(manifest_data))

    for paths in layout.values():
        for rel in paths:
            directory = tmp_path / "segments" / rel
            directory.mkdir(parents=True)
            (directory / SEGMENT_FILE).write_bytes(b"x" * 1024)
    return str(tmp_path), Manifest.load(str(tmp_path / "database.json"))


class TestWalk:
    def test_finds_every_stored_segment(self, corpus):
        root, _ = corpus
        assert len(walk_local(root)) == 3

    def test_reports_real_sizes(self, corpus):
        root, _ = corpus
        assert set(walk_local(root).values()) == {1024}

    def test_ignores_other_files(self, corpus):
        root, _ = corpus
        (os.path.join(root, "segments", "stray.txt"))
        open(os.path.join(root, "segments", "stray.txt"), "w").close()
        assert len(walk_local(root)) == 3

    def test_an_empty_root_is_not_an_error(self, tmp_path):
        assert walk_local(str(tmp_path)) == {}


class TestInventory:
    def test_groups_by_platform(self, corpus):
        root, manifest = corpus
        held = inventory(root, manifest)
        assert held["TOYOTA_PRIUS"].count == 2
        assert held["KIA_EV6"].count == 1

    def test_platforms_with_nothing_local_are_absent(self, corpus):
        root, manifest = corpus
        assert "HONDA_ACCORD" not in inventory(root, manifest)

    def test_sizes_are_summed(self, corpus):
        root, manifest = corpus
        assert inventory(root, manifest)["TOYOTA_PRIUS"].bytes_used == 2048

    def test_unknown_segments_are_kept_under_an_empty_key(self, corpus):
        # A stale or hand-copied file must stay visible, and deletable.
        root, manifest = corpus
        stray = os.path.join(root, "segments", "devX", "routeX--cc", "9")
        os.makedirs(stray)
        open(os.path.join(stray, SEGMENT_FILE), "w").close()
        assert inventory(root, manifest)[""].count == 1

    def test_paths_are_sorted(self, corpus):
        root, manifest = corpus
        paths = inventory(root, manifest)["TOYOTA_PRIUS"].paths
        assert paths == sorted(paths)

    def test_gib_is_derived_from_bytes(self):
        assert LocalPlatform("X", ["a"], 2**30).gib == pytest.approx(1.0)


class TestDelete:
    def test_removes_the_files(self, corpus):
        root, manifest = corpus
        targets = inventory(root, manifest)["TOYOTA_PRIUS"].paths
        result = delete_segments(targets)
        assert result.deleted == 2
        assert result.bytes_freed == 2048
        assert result.ok
        assert not any(os.path.exists(p) for p in targets)

    def test_leaves_other_platforms_alone(self, corpus):
        root, manifest = corpus
        delete_segments(inventory(root, manifest)["TOYOTA_PRIUS"].paths)
        assert inventory(root, manifest)["KIA_EV6"].count == 1

    def test_prunes_the_directories_it_empties(self, corpus):
        root, manifest = corpus
        paths = inventory(root, manifest)["KIA_EV6"].paths
        delete_segments(paths)
        assert not os.path.exists(os.path.dirname(paths[0]))

    def test_keeps_a_directory_that_still_holds_something(self, corpus):
        root, manifest = corpus
        paths = inventory(root, manifest)["TOYOTA_PRIUS"].paths
        keep = os.path.join(os.path.dirname(paths[0]), "notes.txt")
        open(keep, "w").close()
        delete_segments([paths[0]])
        assert os.path.exists(keep)

    def test_refuses_anything_that_is_not_a_stored_segment(self, corpus):
        root, _ = corpus
        victim = os.path.join(root, "database.json")
        result = delete_segments([victim])
        assert result.deleted == 0
        assert not result.ok
        assert os.path.exists(victim), "must not delete a file it was not asked to"

    def test_a_missing_file_is_reported_not_raised(self, corpus):
        root, _ = corpus
        result = delete_segments([os.path.join(root, "segments", "gone", SEGMENT_FILE)])
        assert result.deleted == 0 and len(result.failed) == 1

    def test_deleting_nothing_succeeds(self):
        assert delete_segments([]).ok


class TestDiskFree:
    def test_reports_something_plausible(self, corpus):
        root, _ = corpus
        assert disk_free(root) > 0

    def test_works_for_a_path_that_does_not_exist_yet(self, tmp_path):
        assert disk_free(str(tmp_path / "not" / "created" / "yet")) > 0
