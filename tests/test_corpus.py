# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Manifest parsing and fetch planning -- no network required."""
from __future__ import annotations

import json

import pytest

from canlens.corpus import Manifest, segment_dest, segment_relpath, segment_url, select

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
