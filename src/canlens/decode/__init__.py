# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Raw log formats in, normalised CAN records out.

The corpus ships openpilot `rlog.zst`: zstandard-compressed capnp (`cereal`)
message streams, not `.asc`/`.blf`. Everything downstream consumes
:class:`CanFrame`, so adding readers for other formats later is additive.
"""

from .rlog import CanFrame, SegmentSummary, classify_src, iter_frames, read_frames, summarize
from .schema import ensure_schemas, load_schema, schema_dir

__all__ = [
    "CanFrame",
    "SegmentSummary",
    "classify_src",
    "ensure_schemas",
    "iter_frames",
    "load_schema",
    "read_frames",
    "schema_dir",
    "summarize",
]
