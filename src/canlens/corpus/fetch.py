# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Selective, resumable fetching of corpus segments.

The whole bucket is ~299 GB, which is more than most working machines have
spare. Nothing here ever fetches all of it: callers name platforms (and
optionally a per-platform cap) and only those objects are pulled. Already
present files are skipped, so an interrupted run resumes by re-running.
"""
from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .manifest import Manifest, segment_relpath, segment_url


@dataclass
class FetchResult:
    fetched: int = 0
    skipped: int = 0
    failed: int = 0
    bytes_new: int = 0

    @property
    def ok(self) -> bool:
        return self.failed == 0


def segment_dest(out_dir: str, segment_id: str) -> str:
    return os.path.join(out_dir, "segments", segment_relpath(segment_id), "rlog.zst")


def fetch_one(segment_id: str, out_dir: str, *, timeout: int = 60) -> tuple[str, int]:
    """Fetch a single segment. Returns (path_or_error, bytes) with -1 on failure.

    Downloads to a `.part` sidecar and renames on success, so an interrupted
    transfer never leaves a truncated file that a later run would skip.
    """
    dest = segment_dest(out_dir, segment_id)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest, 0
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    try:
        with urllib.request.urlopen(segment_url(segment_id), timeout=timeout) as r:
            written = 0
            with open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    written += len(chunk)
        os.replace(tmp, dest)
        return dest, written
    except (urllib.error.URLError, OSError) as exc:
        if os.path.exists(tmp):
            os.unlink(tmp)
        return f"{segment_relpath(segment_id)}: {exc}", -1


def select(manifest: Manifest, platforms: list[str], limit: int | None) -> list[str]:
    """Resolve platform keys to a flat segment list, capped per platform."""
    unknown = [p for p in platforms if p not in manifest]
    if unknown:
        raise KeyError(f"unknown platform(s): {unknown}")
    out: list[str] = []
    for key in platforms:
        segments = manifest[key].segments
        out.extend(segments[:limit] if limit else segments)
    return out


def fetch_all(
    segments: list[str],
    out_dir: str,
    *,
    jobs: int = 8,
    progress=None,
) -> FetchResult:
    """Fetch every segment into `out_dir`, `jobs` at a time."""
    result = FetchResult()
    done = 0
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for path_or_err, written in pool.map(lambda s: fetch_one(s, out_dir), segments):
            if written < 0:
                result.failed += 1
                print(f"  failed: {path_or_err}", file=sys.stderr)
            elif written == 0:
                result.skipped += 1
            else:
                result.fetched += 1
                result.bytes_new += written
            done += 1
            if progress and (done % 50 == 0 or done == len(segments)):
                progress(done, len(segments), result)
    return result
