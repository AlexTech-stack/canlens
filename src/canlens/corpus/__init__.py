# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Corpus access: the upstream manifest and selective local materialisation."""

from .fetch import FetchResult, fetch_all, fetch_one, segment_dest, select
from .local import (
    DeleteResult,
    LocalPlatform,
    delete_segments,
    disk_free,
    inventory,
    walk_local,
)
from .manifest import (
    AVG_SEGMENT_MB,
    BUCKET,
    Manifest,
    Platform,
    segment_key,
    segment_relpath,
    segment_url,
)

__all__ = [
    "AVG_SEGMENT_MB",
    "BUCKET",
    "DeleteResult",
    "FetchResult",
    "LocalPlatform",
    "Manifest",
    "Platform",
    "delete_segments",
    "disk_free",
    "fetch_all",
    "fetch_one",
    "inventory",
    "segment_dest",
    "segment_key",
    "segment_relpath",
    "segment_url",
    "select",
    "walk_local",
]
