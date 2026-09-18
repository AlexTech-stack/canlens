# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Raw log formats in, normalised CAN records out.

The corpus ships openpilot `rlog.zst`: zstandard-compressed capnp (`cereal`)
message streams, not `.asc`/`.blf`. Everything downstream consumes
:class:`CanFrame`, so adding readers for other formats later is additive.
"""

from .cache import cache_path, clear, decode_columnar, load_frames
from .frameset import FrameSet, from_frames
from .rlog import CanFrame, SegmentSummary, classify_src, iter_frames, read_frames, summarize
from .schema import ensure_schemas, load_schema, schema_dir

__all__ = [
    "CanFrame",
    "FrameSet",
    "SegmentSummary",
    "cache_path",
    "classify_src",
    "clear",
    "decode_columnar",
    "ensure_schemas",
    "from_frames",
    "iter_frames",
    "load_frames",
    "load_schema",
    "read_frames",
    "schema_dir",
    "summarize",
]
