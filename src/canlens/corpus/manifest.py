# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""The corpus manifest: which platforms exist and which segments belong to them.

The upstream bucket ships a single `database.json` mapping a platform key
(e.g. `TOYOTA_PRIUS`) to the list of segment IDs recorded on that platform.
That file is the index for everything else -- selection, size estimation and
fetch planning all read it, and nothing else needs the full file listing.
"""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass

BUCKET = "https://huggingface.co/buckets/AlexTech-stack/commaCarSegments-bucket/resolve"

# Measured over the whole bucket: 299 GB across 188883 segments. Good enough
# to size a fetch before committing to it; it is not a per-file guarantee.
AVG_SEGMENT_MB = 1.62


@dataclass(frozen=True)
class Platform:
    """One vehicle platform and the segments recorded on it."""

    key: str
    segments: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.segments)

    @property
    def est_gb(self) -> float:
        return self.count * AVG_SEGMENT_MB / 1024


class Manifest:
    """Loaded `database.json`, keyed by platform."""

    def __init__(self, platforms: dict[str, list[str]]) -> None:
        self._platforms = {k: Platform(k, tuple(v)) for k, v in platforms.items()}
        self._owner: dict[str, str] | None = None

    @classmethod
    def load(cls, path: str, *, download: bool = True) -> Manifest:
        """Load the manifest, fetching it (~9 MB) if absent and allowed."""
        if not os.path.exists(path):
            if not download:
                raise FileNotFoundError(path)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            urllib.request.urlretrieve(f"{BUCKET}/database.json", path)
        with open(path) as f:
            return cls(json.load(f))

    def __contains__(self, key: str) -> bool:
        return key in self._platforms

    def __getitem__(self, key: str) -> Platform:
        return self._platforms[key]

    def __len__(self) -> int:
        return len(self._platforms)

    def keys(self) -> list[str]:
        return list(self._platforms)

    def by_size(self) -> list[Platform]:
        """Platforms, largest segment count first."""
        return sorted(self._platforms.values(), key=lambda p: -p.count)

    @property
    def total_segments(self) -> int:
        return sum(p.count for p in self._platforms.values())

    def platform_of(self, value: str) -> str | None:
        """Which platform a segment belongs to, or None if it is not in the corpus.

        The stored layout mirrors the upstream bucket, which is keyed by device
        and route -- nothing in a path says which car it came from. This is the
        only way back to that, so anything reporting on a segment should use it.
        """
        if self._owner is None:
            self._owner = {
                segment_relpath(s): key
                for key, platform in self._platforms.items()
                for s in platform.segments
            }
        return self._owner.get(segment_key(value))


def segment_key(value: str) -> str:
    """Normalise anything naming a segment to `<device>/<route>/<index>`.

    Accepts a manifest segment ID (`.../s`), a stored relative path, or a full
    filesystem path to an `rlog.zst`, because all three turn up in practice and
    callers should not have to care which they are holding.
    """
    parts = [p for p in os.fspath(value).replace("\\", "/").split("/") if p]
    if parts and (parts[-1].endswith(".zst") or parts[-1] == "s"):
        parts = parts[:-1]
    return "/".join(parts[-3:])


def segment_relpath(segment_id: str) -> str:
    """`<device>/<route>/<index>/s` -> `<device>/<route>/<index>`.

    Upstream segment IDs carry a trailing `/s` (openpilot LogReader syntax);
    the stored object drops it and appends `rlog.zst`.
    """
    return "/".join(segment_id.rstrip("/").split("/")[:3])


def segment_url(segment_id: str) -> str:
    return f"{BUCKET}/segments/{segment_relpath(segment_id)}/rlog.zst"
