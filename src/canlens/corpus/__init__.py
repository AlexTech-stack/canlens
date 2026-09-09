# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Corpus access: the upstream manifest and selective local materialisation."""

from .fetch import FetchResult, fetch_all, fetch_one, segment_dest, select
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
    "FetchResult",
    "Manifest",
    "Platform",
    "fetch_all",
    "fetch_one",
    "segment_dest",
    "segment_key",
    "segment_relpath",
    "segment_url",
    "select",
]
