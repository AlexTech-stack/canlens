# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Desktop workbench for canlens."""

from .model import MessageRow, SegmentModel, load_segment
from .palette import KIND_RGB, kinds_to_indices, lookup_table

__all__ = [
    "KIND_RGB",
    "MessageRow",
    "SegmentModel",
    "kinds_to_indices",
    "load_segment",
    "lookup_table",
]
