# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Cache inference results, not only decoded frames.

Inference is deterministic in the frames it is given, costs a couple of
seconds per segment, and its complete result for a CAN FD segment is a few
kilobytes. The decode cache was built first because decode was profiled first;
this is the cache that makes re-opening a segment in the workbench instant.

Entries are keyed on the same source identity as the decode cache plus the
parameters inference ran with, and on INFER_VERSION, which must be bumped
whenever a detector or a threshold changes -- a stale result here is not
slower, it is wrong.

The entry is a gzipped pickle. Pickle is acceptable for a cache the user's
own tool writes under the user's own corpus root, and nothing else is ever
unpickled. Gzip because the entry is mostly float64 bit statistics: uncompressed
it was 1 MB per segment, a third of the frame cache it sits beside.
"""
from __future__ import annotations

import gzip
import os
import pickle
from dataclasses import asdict, dataclass

from ..analyze import TraceProfile
from ..analyze.bits import BitOrder
from ..decode import FrameSet, load_frames
from ..decode.cache import Source, cache_path
from .message import MessageInference, infer_frameset

# Bump on any change to what infer would conclude about the same frames.
INFER_VERSION = 14


@dataclass(frozen=True)
class Key:
    version: int
    source: Source
    order: str
    min_frames: int


def results_path(root: str, segment: str) -> str:
    base = cache_path(root, segment)
    return base[: -len(".npz")] + ".infer.pkl" if base.endswith(".npz") else base + ".infer.pkl"


def save_results(results: list[MessageInference], destination: str, key: Key) -> None:
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    partial = destination + ".part"
    with gzip.open(partial, "wb", compresslevel=1) as handle:
        pickle.dump({"key": asdict(key), "results": results}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(partial, destination)


def load_results(destination: str, key: Key) -> list[MessageInference] | None:
    if not os.path.exists(destination):
        return None
    try:
        with gzip.open(destination, "rb") as handle:
            payload = pickle.load(handle)
        if payload.get("key") != asdict(key):
            return None
        results = payload["results"]
        return results if isinstance(results, list) else None
    except Exception:  # noqa: BLE001 - any unreadable entry means "infer again"
        return None


def infer_cached(
    path: str,
    *,
    root: str,
    order: BitOrder = BitOrder.INTEL,
    min_frames: int = 32,
    use_cache: bool = True,
    frames: FrameSet | None = None,
    profile: TraceProfile | None = None,
) -> list[MessageInference]:
    """Inference for one segment, from the results cache when it is valid.

    `frames` and `profile` let a caller that already decoded and analysed the
    segment avoid doing either again on a miss.
    """
    key = Key(INFER_VERSION, Source.of(path), str(order), min_frames)
    destination = results_path(root, path)
    if use_cache:
        cached = load_results(destination, key)
        if cached is not None:
            return cached
    if frames is None:
        frames = load_frames(path, root=root)
    results = infer_frameset(frames, order=order, min_frames=min_frames, profile=profile)
    if use_cache:
        try:
            save_results(results, destination, key)
        except OSError:
            pass  # a read-only or full disk must not stop the analysis
    return results


def warm_segment(path: str, root: str) -> tuple[int, int]:
    """Fill both caches for one segment. Returns (messages, findings).

    Module-level so a process pool can pickle a reference to it.
    """
    frames = load_frames(path, root=root)
    results = infer_cached(path, root=root, frames=frames)
    findings = sum(len(m.counters) + len(m.checksums) + len(m.crc16s) for m in results)
    return len(results), findings


def clear_results(root: str) -> int:
    """Delete every cached inference result, returning how many."""
    from ..decode.cache import CACHE_DIR

    removed = 0
    for directory, _subdirs, files in os.walk(os.path.join(root, CACHE_DIR)):
        for name in files:
            if name.endswith(".infer.pkl"):
                os.unlink(os.path.join(directory, name))
                removed += 1
    return removed
