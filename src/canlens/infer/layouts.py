# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""What a multiplexor's selector value actually selects.

:mod:`canlens.infer.multiplex` finds the selector, the values it takes, and
the ``dependent_bits`` whose behaviour changes with it. What it leaves
implicit is the *content*: the VIN message ``0x6B4`` sends seven bytes under
each of its three selector values, and a reader of the trace wants to see
those bytes and know which value carries which. This module turns the
dependent content into addressable fields.

**It is not signal detection, and that was measured before anything was
built.** Running the ordinary signal detector inside each selector value
recovers almost nothing real, for two reasons that are properties of CAN
multiplexing rather than of this code. The flagship VIN is a table of
*constants* -- seven bytes per value, never moving -- and a detector built on
transition rate finds nothing where nothing moves. The genuinely moving mux
signals opendbc declares, such as Tesla ``0x221``'s state fields, are two
bits wide, and a narrow slow field is exactly what the rate floor cannot see
(see :mod:`canlens.infer.signals`). Measured over 24 segments and 8
platforms, 51 multiplexed messages yielded 33 putative layout signals, and
the one message with a DBC to check them against recovered none of its 13
declared moving fields.

**The grouping rule is whole bytes, and it was chosen by scoring three.**
The hard part is not reading values but deciding where a constant field ends,
which a trace cannot see for the same reason it cannot see a signal's far
end. Three rules were scored against the mux fields the DBCs declare, over
the VIN on five MQB platforms and the Tesla state message, matching on exact
bit set and selector value:

===========================  =======  =======  ====
grouping rule                hit      extra    F1
===========================  =======  =======  ====
maximal runs of dependent bits     0      540  0.00
**whole byte**                   315        8  0.95
byte if a single run, else runs   96      495  0.21
===========================  =======  =======  ====

A run of dependent bits is *narrower* than the field a DBC declares, because
a bit that happened not to differ between the selector values in this trace
is not dependent -- so runs never line up with declared fields and the first
rule misses every one. The whole-byte rule wins because a CAN mux field is
usually a whole byte and the byte is fully observed: reporting it claims no
more than the constant value that is there. All 27 reference fields it missed
are sub-byte fields in one Tesla message, which no rule recovered, and that
limit is worth stating plainly rather than papering over: a two-bit selector
field inside a byte that also carries other content is invisible to a trace.

**A byte has to be one the selector actually changes.** Holding a dependent
bit is necessary but not sufficient, because a bit can be dependent by
*behaviour* -- locked to the group's schedule, moving in one value and still
in another -- while the byte around it holds the same constant throughout. So
a byte is reported only if its constant differs between selector values, or
if some value leaves it moving; one constant under every value is the same
thing as a constant over the whole trace, which the message fixes rather than
the selector choosing.

The guard is stated because the rule needs it, not because the corpus needed
it: over 60 segments and 133 multiplexed messages it drops **nothing** --
2750 claims with it and 2750 without. A byte that reaches this point has
always earned it, usually the way a VIN slice does, by holding 0x00 under
some values and letters under others. Keep it anyway: `dependent_bits` is
decided by :mod:`canlens.infer.multiplex` on behaviour, and nothing there
promises the byte's *value* ever changes, so without the guard the module's
output depends on a property its input does not guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..analyze.bits import BitOrder
from .counters import field_values
from .multiplex import MultiplexHypothesis


@dataclass(frozen=True)
class LayoutField:
    """One field a selector value selects, and the constant value it holds."""

    start_bit: int
    length: int
    mux_value: int
    value: int
    frames: int  # frames of this selector value's layout
    # The multiplexor is always read in Intel order (counters, checksums and
    # multiplexors are verified arithmetic over whole bytes and never move
    # with bit numbering), so a layout field is Intel too.
    byte_order: BitOrder = BitOrder.INTEL

    @property
    def end_bit(self) -> int:
        return self.start_bit + self.length

    @property
    def bits(self) -> tuple[int, ...]:
        """The flat Intel indices this field occupies."""
        return tuple(range(self.start_bit, self.start_bit + self.length))

    def __str__(self) -> str:
        return (
            f"value {self.mux_value}: byte {self.start_bit // 8} = {self.value} "
            f"(0x{self.value:02X}, {self.frames} frames)"
        )


def read_layout_fields(matrix: np.ndarray, mux: MultiplexHypothesis) -> list[LayoutField]:
    """The whole-byte constants each selector value selects.

    `matrix` is the Intel bit matrix the selector was found in. A field is
    reported for a byte that is constant over a value's frames, holds at least
    one dependent bit, and that the selector is actually seen to change --
    either its constant differs between values, or some value leaves it
    moving. A byte holding one constant under every value is the message's own
    padding and is dropped. Bytes that move within a value are left to the
    ordinary signal detector.
    """
    if matrix.size == 0 or not mux.dependent_bits:
        return []
    width_bits = matrix.shape[1]
    byte_count = width_bits // 8
    dependent_bytes = sorted({bit // 8 for bit in mux.dependent_bits if bit < width_bits})
    if not dependent_bytes:
        return []

    selector = field_values(matrix, mux.start_bit, mux.length)
    out: list[LayoutField] = []
    # Per byte: the distinct constants it held, and whether any value left it
    # moving. Both are needed to tell a byte the selector *chose* from one the
    # message simply fixes; see below.
    constants: dict[int, set[int]] = {}
    moving: set[int] = set()
    for value in mux.values:
        frames = matrix[selector == value]
        if frames.shape[0] == 0:
            continue
        for byte in dependent_bytes:
            if byte >= byte_count:
                continue
            column = frames[:, byte * 8 : byte * 8 + 8]
            if not bool((column.min(axis=0) == column.max(axis=0)).all()):
                moving.add(byte)
                continue
            field = LayoutField(
                start_bit=byte * 8,
                length=8,
                mux_value=int(value),
                value=int(field_values(frames, byte * 8, 8)[0]),
                frames=int(frames.shape[0]),
            )
            constants.setdefault(byte, set()).add(field.value)
            out.append(field)

    # Holding one constant under every value is the same thing as being
    # constant over the whole trace: the selector did not choose it, the
    # message fixes it. Containing a dependent bit is not enough to rule that
    # out, because a bit can be dependent by *behaviour* -- locked to the
    # group's schedule -- while the byte around it never changes value. This
    # drops nothing on 60 corpus segments (2750 claims either way); it is
    # here because `dependent_bits` does not promise otherwise.
    chosen = {
        byte
        for byte, values in constants.items()
        if len(values) > 1 or byte in moving
    }
    return [field for field in out if field.start_bit // 8 in chosen]
