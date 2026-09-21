# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Ground truth and self-evaluation.

opendbc carries community DBCs for many corpus platforms. Scoring inferred
signals against them is what turns "the engine produced an answer" into "the
engine is measurably right", and makes regressions in the inference code
visible.

Two halves. :mod:`canlens.truth.dbc` reads a `.dbc` into a `Reference` stated
in canlens' own bit numbering; :mod:`canlens.truth.score` compares that against
what :mod:`canlens.infer` claimed and reports precision and recall per kind of
field, together with every disagreement in both directions.

A disagreement is evidence, not a verdict. The reference is itself
reverse-engineered, and its authors routinely leave counters unnamed.

Requires the `truth` extra (`pip install 'canlens[truth]'`) for cantools. The
import is lazy, so this package imports cleanly without it.
"""

from .dbc import (
    FieldKind,
    Reference,
    ReferenceMessage,
    ReferenceSignal,
    classify,
    load_dbc,
    signal_bits,
)
from .score import (
    SCORED_KINDS,
    Disagreement,
    Score,
    Tally,
    claimed_fields,
    pick_bus,
    score,
)

__all__ = [
    "SCORED_KINDS",
    "Disagreement",
    "FieldKind",
    "Reference",
    "ReferenceMessage",
    "ReferenceSignal",
    "Score",
    "Tally",
    "claimed_fields",
    "classify",
    "load_dbc",
    "pick_bus",
    "score",
    "signal_bits",
]
