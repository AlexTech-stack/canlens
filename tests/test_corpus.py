# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Manifest parsing and fetch planning -- no network required."""
from __future__ import annotations

import json

import pytest

from canlens.corpus import (
    Manifest,
    segment_dest,
    segment_key,
    segment_relpath,
    segment_url,
    select,
)

SAMPLE = {
    "TOYOTA_PRIUS": ["dev0/route0--aa/13/s", "dev1/route1--bb/7/s"],
    "KIA_EV6": ["dev2/route2--cc/1/s"],
}


@pytest.fixture
def manifest(tmp_path):
    path = tmp_path / "database.json"
    path.write_text(json.dumps(SAMPLE))
    return Manifest.load(str(path), download=False)


def test_load_and_index(manifest):
    assert len(manifest) == 2
    assert "TOYOTA_PRIUS" in manifest
    assert manifest["TOYOTA_PRIUS"].count == 2
    assert manifest.total_segments == 3


def test_by_size_orders_largest_first(manifest):
    assert [p.key for p in manifest.by_size()] == ["TOYOTA_PRIUS", "KIA_EV6"]


def test_missing_manifest_is_not_silently_downloaded(tmp_path):
    with pytest.raises(FileNotFoundError):
        Manifest.load(str(tmp_path / "absent.json"), download=False)


def test_segment_relpath_strips_logreader_suffix():
    # Upstream IDs carry a trailing '/s'; the stored object does not.
    assert segment_relpath("dev0/route0--aa/13/s") == "dev0/route0--aa/13"
    assert segment_relpath("dev0/route0--aa/13") == "dev0/route0--aa/13"


def test_segment_url_targets_rlog():
    url = segment_url("dev0/route0--aa/13/s")
    assert url.endswith("/segments/dev0/route0--aa/13/rlog.zst")


def test_segment_dest_mirrors_remote_layout():
    assert segment_dest("/corpus", "dev0/route0--aa/13/s") == (
        "/corpus/segments/dev0/route0--aa/13/rlog.zst"
    )


def test_select_applies_per_platform_limit(manifest):
    assert len(select(manifest, ["TOYOTA_PRIUS", "KIA_EV6"], limit=1)) == 2
    assert len(select(manifest, ["TOYOTA_PRIUS", "KIA_EV6"], limit=None)) == 3


def test_select_rejects_unknown_platform(manifest):
    with pytest.raises(KeyError, match="FORD_MODEL_T"):
        select(manifest, ["FORD_MODEL_T"], limit=None)


class TestSegmentKey:
    """Anything naming a segment normalises to device/route/index."""

    def test_manifest_id_drops_the_logreader_suffix(self):
        assert segment_key("dev0/route0--aa/13/s") == "dev0/route0--aa/13"

    def test_relative_path_passes_through(self):
        assert segment_key("dev0/route0--aa/13") == "dev0/route0--aa/13"

    def test_full_filesystem_path_is_trimmed(self):
        assert segment_key("/data/canlens/segments/dev0/route0--aa/13/rlog.zst") == (
            "dev0/route0--aa/13"
        )

    def test_trailing_and_duplicate_slashes_are_ignored(self):
        assert segment_key("//dev0//route0--aa//13//") == "dev0/route0--aa/13"


class TestPlatformOf:
    """The stored layout says nothing about which car a segment came from."""

    def test_resolves_every_accepted_form(self, manifest):
        for value in (
            "dev0/route0--aa/13/s",
            "dev0/route0--aa/13",
            "/data/canlens/segments/dev0/route0--aa/13/rlog.zst",
        ):
            assert manifest.platform_of(value) == "TOYOTA_PRIUS"

    def test_distinguishes_platforms(self, manifest):
        assert manifest.platform_of("dev2/route2--cc/1/s") == "KIA_EV6"

    def test_unknown_segment_is_none_not_a_guess(self, manifest):
        assert manifest.platform_of("nobody/nothing/0") is None

    def test_repeated_lookups_agree(self, manifest):
        # The reverse index is built lazily and cached; it must not drift.
        first = manifest.platform_of("dev1/route1--bb/7/s")
        assert first == manifest.platform_of("dev1/route1--bb/7") == "TOYOTA_PRIUS"
