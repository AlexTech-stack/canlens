# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""What of the corpus is actually on this disk, and removing it again.

Inventory walks the local segment tree once rather than asking the manifest
whether each of its 188883 entries exists: the answer is almost always no, and
a walk costs one pass over what is present instead of a stat per thing that
is not. It also yields real byte counts rather than the 1.62 MB average used
to size a fetch before it happens.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field

from .manifest import Manifest, segment_key

SEGMENT_FILE = "rlog.zst"


@dataclass
class LocalPlatform:
    """What is held locally for one platform."""

    key: str
    paths: list[str] = field(default_factory=list)
    bytes_used: int = 0

    @property
    def count(self) -> int:
        return len(self.paths)

    @property
    def gib(self) -> float:
        return self.bytes_used / 2**30


@dataclass
class DeleteResult:
    deleted: int = 0
    bytes_freed: int = 0
    failed: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed


def segments_root(root: str) -> str:
    return os.path.join(root, "segments")


def walk_local(root: str) -> dict[str, int]:
    """Every stored segment file, mapped to its size in bytes."""
    base = segments_root(root)
    found: dict[str, int] = {}
    for directory, _subdirs, files in os.walk(base):
        if SEGMENT_FILE in files:
            path = os.path.join(directory, SEGMENT_FILE)
            try:
                found[path] = os.path.getsize(path)
            except OSError:  # vanished between walk and stat
                continue
    return found


def inventory(root: str, manifest: Manifest) -> dict[str, LocalPlatform]:
    """Local holdings per platform, keyed by platform name.

    Segments on disk that the manifest does not know about are grouped under
    the empty key rather than dropped, so a stale or hand-copied file is still
    visible and can still be deleted.
    """
    owner = {}
    for key in manifest:
        for segment in manifest[key].segments:
            owner[segment_key(segment)] = key

    out: dict[str, LocalPlatform] = {}
    for path, size in walk_local(root).items():
        platform = owner.get(segment_key(path), "")
        entry = out.setdefault(platform, LocalPlatform(platform))
        entry.paths.append(path)
        entry.bytes_used += size
    for entry in out.values():
        entry.paths.sort()
    return out


def delete_segments(paths: list[str]) -> DeleteResult:
    """Remove stored segments, and the directories left empty behind them.

    Only the segment file and the now-empty directories that held it are
    touched; a directory that still contains anything is left alone.
    """
    result = DeleteResult()
    for path in paths:
        if os.path.basename(path) != SEGMENT_FILE:
            result.failed.append(f"{path}: not a stored segment")
            continue
        try:
            size = os.path.getsize(path)
            os.unlink(path)
        except OSError as exc:
            result.failed.append(f"{path}: {exc}")
            continue
        result.deleted += 1
        result.bytes_freed += size
        _prune_empty(os.path.dirname(path))
    return result


def _prune_empty(directory: str, levels: int = 3) -> None:
    """Remove up to `levels` of directories left empty by a deletion."""
    for _ in range(levels):
        try:
            if os.listdir(directory):
                return
            os.rmdir(directory)
        except OSError:
            return
        directory = os.path.dirname(directory)


def disk_free(root: str) -> int:
    """Bytes available where the corpus is stored."""
    target = root
    while target and not os.path.exists(target):
        target = os.path.dirname(target)
    return shutil.disk_usage(target or "/").free
