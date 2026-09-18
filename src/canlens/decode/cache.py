# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Decode a segment once, keep the columns.

Decoding costs ~1.5 s per segment and almost all of it is field reads across
the pycapnp boundary -- roughly 1.2 million of them at ~0.6 us each. That is
not something Python can be made to do faster, so the answer is to do it once.

The cache is `savez_compressed`, which turns out to be smaller than the source
it was decoded from -- 1.6 MiB against 2.9 MiB of zstd -- because the columns
compress far better than interleaved capnp does. Writing costs about 0.25 s on
top of the decode; reading costs 0.03 s.

Everything cached is *raw*: every frame, echoes included, exactly as the trace
carried it. Filtering happens on the way out, so one cache serves a caller
that wants echoes and one that does not.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .frameset import FrameSet
from .schema import load_schema
from .zstd import decompress

# Bumped whenever what is stored changes meaning. An older file is re-decoded
# rather than misread.
CACHE_VERSION = 1

CACHE_DIR = "cache"


@dataclass(frozen=True)
class Source:
    """What a cache entry was built from, for deciding whether it still holds."""

    size: int
    mtime_ns: int

    @classmethod
    def of(cls, path: str) -> Source:
        stat = os.stat(path)
        return cls(stat.st_size, stat.st_mtime_ns)


def cache_path(root: str, segment: str) -> str:
    """Where the cache for one segment lives, mirroring its stored layout."""
    parts = [p for p in os.path.normpath(segment).split(os.sep) if p]
    if parts and parts[-1].endswith(".zst"):
        parts = parts[:-1]
    return os.path.join(root, CACHE_DIR, *parts[-3:]) + ".npz"


def decode_columnar(path: str, *, root: str) -> FrameSet:
    """Decode a segment straight into columns, building no per-frame object."""
    schema = load_schema(root)
    with open(path, "rb") as handle:
        raw = decompress(handle.read())

    stamps: list[int] = []
    srcs: list[int] = []
    addresses: list[int] = []
    lengths: list[int] = []
    chunks: list[bytes] = []
    for event in schema.Event.read_multiple_bytes(raw):
        if event.which() != "can" or not event.valid:
            continue
        stamp = event.logMonoTime
        for frame in event.can:
            payload = frame.dat
            stamps.append(stamp)
            srcs.append(frame.src)
            addresses.append(frame.address)
            lengths.append(len(payload))
            chunks.append(payload)

    count = len(stamps)
    return FrameSet(
        mono_ns=np.asarray(stamps, dtype=np.int64),
        src=np.asarray(srcs, dtype=np.uint8),
        address=np.asarray(addresses, dtype=np.uint32),
        lengths=np.asarray(lengths, dtype=np.uint8),
        blob=np.frombuffer(b"".join(chunks), dtype=np.uint8)
        if count
        else np.zeros(0, dtype=np.uint8),
    )


def save(frames: FrameSet, destination: str, source: Source) -> None:
    """Write a cache entry, atomically."""
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    partial = destination + ".part"
    np.savez_compressed(
        partial,
        version=np.asarray([CACHE_VERSION], dtype=np.int32),
        source=np.asarray([source.size, source.mtime_ns], dtype=np.int64),
        mono_ns=frames.mono_ns,
        src=frames.src,
        address=frames.address,
        lengths=frames.lengths,
        blob=frames.blob,
    )
    # savez appends .npz to a path that lacks it; rename whatever it wrote.
    written = partial if os.path.exists(partial) else partial + ".npz"
    os.replace(written, destination)


def load(destination: str, source: Source) -> FrameSet | None:
    """Read a cache entry, or None if it is absent, stale or unreadable."""
    if not os.path.exists(destination):
        return None
    try:
        with np.load(destination) as data:
            if int(data["version"][0]) != CACHE_VERSION:
                return None
            size, mtime = (int(v) for v in data["source"])
            if (size, mtime) != (source.size, source.mtime_ns):
                return None
            return FrameSet(
                mono_ns=data["mono_ns"],
                src=data["src"],
                address=data["address"],
                lengths=data["lengths"],
                blob=data["blob"],
            )
    except Exception:  # noqa: BLE001 - any unreadable entry means "decode again"
        # Deliberately broad: a half-written entry surfaces as
        # zipfile.BadZipFile, a foreign one as ValueError, a vanished one as
        # OSError. Every answer is the same and always safe, because the
        # fallback is to decode the segment the cache was standing in for.
        return None


def load_frames(
    path: str, *, root: str, use_cache: bool = True, include_echo: bool = False
) -> FrameSet:
    """Frames for one segment, from the cache when it is valid.

    Echoes are dropped on the way out rather than on the way in, so the stored
    entry stays canonical whatever the caller asked for.
    """
    source = Source.of(path)
    destination = cache_path(root, path)
    frames = load(destination, source) if use_cache else None
    if frames is None:
        frames = decode_columnar(path, root=root)
        if use_cache:
            try:
                save(frames, destination, source)
            except OSError:
                pass  # a read-only or full disk must not stop the analysis
    return frames if include_echo else frames.without_echoes()


def clear(root: str) -> int:
    """Delete every cache entry, returning how many were removed."""
    base = os.path.join(root, CACHE_DIR)
    removed = 0
    for directory, _subdirs, files in os.walk(base, topdown=False):
        for name in files:
            if name.endswith(".npz"):
                os.unlink(os.path.join(directory, name))
                removed += 1
        if directory != base and not os.listdir(directory):
            os.rmdir(directory)
    return removed
