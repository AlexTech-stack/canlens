# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Segmented ISO-TP transfers, reassembled and checked against themselves.

Every other detector here reads one message's payloads as a block of columns.
This one cannot: a segmented transfer is a *sequence* of frames on one address,
and the evidence for it is the order they arrive in. So it consumes the trace
in time order, before anything is grouped by identifier.

**What is claimed, and what would falsify it.** ISO 15765-2:2016 splits a long
message into a FirstFrame carrying a 12-bit FF_DL, then ConsecutiveFrames whose
SequenceNumbers begin at 1 and increment modulo 16 (9.6.3, 9.6.4). Both halves
are checkable against the trace and neither is a guess: the chain must be
unbroken and the bytes must reach the length the FirstFrame promised. A frame
whose SequenceNumber is not the expected one aborts the transfer, exactly as
[the spec's] 9.6.4.4 requires of a receiver.

**The independent check.** The spec also requires the receiver to answer a
FirstFrame with a FlowControl frame before ConsecutiveFrames may flow. Nothing
here uses that to *find* a transfer -- it is recorded separately, so it stays
an independent confirmation rather than a criterion. Measured over 803 corpus
segments, 98% of 22401 completed reassemblies had a FlowControl from a peer
address on the same bus while the FirstFrame was open.

**Why only segmented transfers.** A SingleFrame's entire header is one nibble,
and ordinary traffic satisfies it constantly: 8216707 matches corpus-wide
against 22401 genuine multi-frame messages. Restricting to addresses this
module has already proven does not rescue it either, lifting the share of
plausible service bytes only from 38% to 45%. Single frames need a padding
check against ISO 15765-2:2016 10.4.2, which is a separate detector and is not
built here.

**What this finds, in the corpus.** Run per segment, as inference here always
is: **9 endpoints over 803 segments, 21 of which carry one, and 20823
transfers, every one FlowControl-backed**, with payloads from 8 to 740 bytes.
Among them the standard diagnostic pair -- 0x7E0 sending nothing but SID 0x22
and 0x7E8 answering 0x62 -- and Toyota's 0x080 carrying FF_DL 740 in chains of
106 ConsecutiveFrames.

The floor was calibrated on figures pooled per address *across* segments, where
it keeps 49 addresses and 21209 transfers against 223 and 22401 without it.
Applying it per segment is stricter and is the right place for it: an address
needs its evidence within one recording, and agreement between recordings is
what :mod:`canlens.corroborate` is for. The difference is 386 transfers spread
thinly enough that no single segment could check them.

A transport is not a vehicle signal, and this module exists partly to say so.
An ISO-TP SequenceNumber advances by one and wraps, so the counter detector
reports Toyota's 0x080 and 0x085 as 4-bit counters at 98%; the arithmetic is
right and the layer is wrong. :func:`transport_bits` names the bits a verified
transfer explains so that claim can be withdrawn.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

import numpy as np

# N_PCItype, the high nibble of the first PCI byte (ISO 15765-2:2016, Table 8).
SINGLE_FRAME = 0x0
FIRST_FRAME = 0x1
CONSECUTIVE_FRAME = 0x2
FLOW_CONTROL = 0x3

# FlowStatus values a real FlowControl carries: ContinueToSend, Wait, Overflow
# (Table 18). Anything above 2 is reserved, and reading it as a FlowControl
# would admit three quarters of the 0x3x byte space for nothing.
MAX_FLOW_STATUS = 0x2

# N_Bs, the sender's timeout waiting for a FlowControl (9.8.2). Used only to
# bound how long a FirstFrame stays open for the corroboration to count.
N_BS_NS = 1_000_000_000

# N_Cr, the receiver's timeout waiting for the next ConsecutiveFrame. A gap
# longer than this means the transfer was abandoned, so the chain cannot be
# continued by a frame that happens to carry the right SequenceNumber much
# later. The spec's performance requirement is 1000 ms; ten times that is used
# here because a logger drops frames and a strict bound would break real
# chains, while still refusing to join two transfers minutes apart.
MAX_GAP_NS = 10 * N_BS_NS

# Completed reassemblies before an address is reported. One completion of a
# short transfer is a weak claim: a FF_DL that fits in two frames needs only
# one correctly numbered ConsecutiveFrame to follow, which ordinary traffic
# supplies by accident. Costs 33 addresses and 33 transfers over the corpus,
# and takes FlowControl backing from 99% to 100%.
MIN_MESSAGES = 2

# Share of FirstFrames that must reach their promised length. Every detector
# here carries a match rate and is held to a floor -- counters 0.95, the rest
# 0.99 -- and this is the same kind of number. It is the constant that matters:
# an ordinary high-rate message opens hundreds of accidental FirstFrames and
# completes a couple of them, which is how VW's 0x101 was reported as an
# endpoint on 2 completions out of ~200 openings.
#
# Measured per address over 803 segments, the distribution is bimodal with
# nothing in the middle, so the floor is a plateau rather than a knife edge:
#
#     match rate   addresses   transfers   FlowControl-backed
#        0% -  5%        137        1141                  68%
#        5% - 25%          4          18                  83%
#       50% - 90%          2           8                 100%
#             100%        80       21234                  99%
#
# Floors of 0.25, 0.50 and 0.75 all keep the same 49 addresses and 21209
# transfers, at 100% backing; 0.50 sits mid-plateau.
MIN_MATCH_RATE = 0.5


@dataclass(frozen=True)
class Transfer:
    """One reassembled ISO-TP message."""

    bus: int
    address: int
    data: bytes
    consecutive_frames: int
    flow_controlled: bool  # a peer answered while the FirstFrame was open
    peer: int | None  # the address that answered, when one did

    @property
    def length(self) -> int:
        return len(self.data)

    def __str__(self) -> str:
        head = " ".join(f"{b:02X}" for b in self.data[:8])
        more = " …" if len(self.data) > 8 else ""
        return f"{self.length} bytes in {self.consecutive_frames} CFs: {head}{more}"


@dataclass
class IsotpHypothesis:
    """The segmented transfers one address was seen to send."""

    bus: int
    address: int
    started: int = 0  # FirstFrames opened
    transfers: list[Transfer] = field(default_factory=list)

    @property
    def messages(self) -> int:
        return len(self.transfers)

    @property
    def match_rate(self) -> float:
        """Share of FirstFrames that reached their promised length.

        The detector's own falsifiable number, in the sense the other
        detectors use it: a FirstFrame whose chain never completes is a
        hypothesis the trace refused.
        """
        return self.messages / self.started if self.started else 0.0

    @property
    def flow_control_rate(self) -> float:
        """Share of completions a peer FlowControl corroborated."""
        if not self.transfers:
            return 0.0
        return sum(t.flow_controlled for t in self.transfers) / self.messages

    @property
    def peers(self) -> tuple[int, ...]:
        """Addresses seen answering this one, ascending."""
        return tuple(sorted({t.peer for t in self.transfers if t.peer is not None}))

    @property
    def lengths(self) -> tuple[int, ...]:
        return tuple(sorted({t.length for t in self.transfers}))

    @property
    def verified(self) -> bool:
        """Enough completions, and nearly all the FirstFrames completing.

        The rate is what separates an endpoint from a coincidence: a real one
        completes every transfer it starts, an ordinary message completes a
        couple of the hundreds its byte 0 opens by accident.
        """
        return self.messages >= MIN_MESSAGES and self.match_rate >= MIN_MATCH_RATE

    @property
    def key(self) -> tuple[int, int]:
        return (self.bus, self.address)

    def __str__(self) -> str:
        lengths = self.lengths
        span = (
            f"{lengths[0]} bytes"
            if len(lengths) == 1
            else f"{lengths[0]}–{lengths[-1]} bytes"
        )
        peers = ", ".join(f"0x{p:03X}" for p in self.peers[:3])
        answered = f", answered by {peers}" if peers else ""
        return (
            f"ISO-TP: {self.messages} transfers of {span} "
            f"({self.match_rate:.0%} of FirstFrames completed, "
            f"{self.flow_control_rate:.0%} flow-controlled){answered}"
        )


def transport_bits(hypothesis: IsotpHypothesis) -> set[int]:
    """Flat Intel bit positions the transport explains on this address.

    The PCI byte under normal addressing, which is byte 0. Extended and mixed
    addressing put it in byte 1 behind an address extension; neither is
    detected here, so neither is claimed.
    """
    return set(range(8)) if hypothesis.verified else set()


def find_isotp(
    frames: Iterable[tuple[int, int, int, bytes]],
) -> dict[tuple[int, int], IsotpHypothesis]:
    """Reassemble every segmented transfer in a trace given in time order.

    `frames` yields `(mono_ns, bus, address, payload)`. Echoes must already be
    excluded by the caller; a device echo reproduces the source bus's traffic
    and would double every transfer on it.
    """
    found: dict[tuple[int, int], IsotpHypothesis] = {}
    # One open transfer per address. ISO-TP allows only one segmented message
    # in flight per address pair, so a second FirstFrame abandons the first.
    open_transfers: dict[tuple[int, int], _Partial] = {}

    for mono_ns, bus, address, payload in frames:
        if len(payload) < 2:
            continue
        key = (bus, address)
        kind = payload[0] >> 4
        low = payload[0] & 0x0F

        if kind == FIRST_FRAME:
            ff_dl = (low << 8) | payload[1]
            # A FirstFrame must promise more than it carries, or the sender
            # would have used a SingleFrame. Without this every 0x1x byte
            # opens a transfer.
            if ff_dl > len(payload) - 2:
                hypothesis = found.setdefault(key, IsotpHypothesis(bus, address))
                hypothesis.started += 1
                open_transfers[key] = _Partial(ff_dl, bytearray(payload[2:]), mono_ns)

        elif kind == FLOW_CONTROL and low <= MAX_FLOW_STATUS:
            # Corroboration only: a FlowControl from another address on this
            # bus, while a transfer there is waiting for one.
            for other, waiting in open_transfers.items():
                if (
                    other[0] == bus
                    and other[1] != address
                    and mono_ns - waiting.opened_ns <= N_BS_NS
                ):
                    waiting.peer = address

        elif kind == CONSECUTIVE_FRAME:
            partial = open_transfers.get(key)
            if partial is None:
                continue
            if low != partial.next_sn & 0x0F or mono_ns - partial.last_ns > MAX_GAP_NS:
                del open_transfers[key]  # 9.6.4.4: a wrong SN aborts
                continue
            partial.data.extend(payload[1:])
            partial.next_sn += 1
            partial.consecutive_frames += 1
            partial.last_ns = mono_ns
            if len(partial.data) >= partial.ff_dl:
                found[key].transfers.append(
                    Transfer(
                        bus=bus,
                        address=address,
                        data=bytes(partial.data[: partial.ff_dl]),
                        consecutive_frames=partial.consecutive_frames,
                        flow_controlled=partial.peer is not None,
                        peer=partial.peer,
                    )
                )
                del open_transfers[key]

    return {key: h for key, h in found.items() if h.verified}


@dataclass
class _Partial:
    """A transfer in flight. Not part of the API."""

    ff_dl: int
    data: bytearray
    opened_ns: int
    next_sn: int = 1
    consecutive_frames: int = 0
    peer: int | None = None

    def __post_init__(self) -> None:
        self.last_ns = self.opened_ns


def ordered_frames(frames) -> Iterator[tuple[int, int, int, bytes]]:
    """A `FrameSet` as `(mono_ns, bus, address, payload)` in time order.

    Sorted explicitly rather than trusted: frames batched into one capnp event
    share a timestamp and the stored order is the decode order, which is close
    to but not guaranteed to be monotonic.
    """
    from ..decode.rlog import ECHO_FLAG

    order = np.argsort(frames.mono_ns, kind="stable")
    offsets = frames.offsets
    blob = frames.blob
    for index in order:
        src = int(frames.src[index])
        if src >= ECHO_FLAG:
            continue  # invariant 4: an echo is not data
        start = int(offsets[index])
        length = int(frames.lengths[index])
        yield (
            int(frames.mono_ns[index]),
            src,
            int(frames.address[index]),
            bytes(blob[start : start + length]),
        )
