# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Per-message inference: what the measured bits actually are."""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from ..analyze import TraceProfile
from ..analyze.bits import (
    BitOrder,
    BitProfile,
    bit_matrix,
    bit_matrix_from_bytes,
    profile_bits,
    profile_bits_from_bytes,
)
from ..decode import CanFrame, iter_frames, load_frames
from ..decode.frameset import FrameSet, Message
from .byteorder import find_signals_in
from .checksums import (
    ChecksumHypothesis,
    find_checksums,
    find_e2e_crc8,
    find_honda_nibble,
)
from .counters import CounterHypothesis, find_counters
from .crc16 import Crc16Hypothesis, find_crc16
from .multiplex import MultiplexHypothesis, find_multiplexor
from .signals import SignalHypothesis

# How many payloads to materialise as bytes for the detectors that still read
# them (the Data-ID solver reads 4, the CRC screen 64). Everything that scores
# frames does so against the byte matrix, so the rest would be built and
# thrown away -- 318,510 objects per segment, measured.
PAYLOAD_SAMPLE = 64

# A checksum byte is indistinguishable from noise by construction, so only
# bytes that move like noise are worth testing. This screen is what keeps a
# whole-segment sweep affordable.
#
# Set below a byte-wide counter's mean transition rate, which is ~0.249: the
# bits halve in rate from the LSB, so 1 + .5 + .25 + ... averages just under a
# quarter. Testing a counter byte and rejecting it costs almost nothing, while
# skipping a real checksum costs a finding, so the cutoff sits under that.
CHECKSUM_BYTE_MIN_RATE = 0.20

# Deliberately the same as the 8-bit screen, not stricter. A CRC byte is
# uniformly random and averages about 0.5, so 0.35 looks safe -- but it drops
# real findings: five EV6 messages carry an E2E Profile 5 CRC verifying at
# 100% whose bytes fall below it. Since the early exit in the scan made the
# extra positions nearly free, recall wins.
CRC16_BYTE_MIN_RATE = CHECKSUM_BYTE_MIN_RATE


@dataclass
class MessageInference:
    """Hypotheses about one message, each already checked against the trace."""

    bus: int
    address: int
    width: int
    frames: int
    bits: BitProfile
    counters: list[CounterHypothesis] = field(default_factory=list)
    checksums: list[ChecksumHypothesis] = field(default_factory=list)
    crc16s: list[Crc16Hypothesis] = field(default_factory=list)
    multiplexor: MultiplexHypothesis | None = None
    signals: list[SignalHypothesis] = field(default_factory=list)

    @property
    def key(self) -> tuple[int, int]:
        return (self.bus, self.address)

    @property
    def found_anything(self) -> bool:
        return bool(
            self.counters
            or self.checksums
            or self.crc16s
            or self.multiplexor
            or self.signals
        )

    def __str__(self) -> str:
        return f"bus {self.bus} 0x{self.address:03X}"


def checksum_candidate_bytes(bits: BitProfile, *, min_rate: float = CHECKSUM_BYTE_MIN_RATE) -> list[int]:
    """Byte positions whose bits move enough to be worth testing."""
    return [
        index
        for index in range(bits.width)
        if float(np.mean(bits.rates[index * 8 : (index + 1) * 8])) >= min_rate
    ]


def infer_message(
    payloads: Sequence[bytes],
    *,
    bus: int,
    address: int,
    order: BitOrder = BitOrder.INTEL,
    **kwargs,
) -> MessageInference:
    """Look for counters and checksums in one message's payloads."""
    width = len(payloads[0]) if payloads else 0
    bits = profile_bits(list(payloads), width, order)
    matrix = bit_matrix(list(payloads), width, order)
    return _build(bus, address, width, list(payloads), bits, matrix, **kwargs)


def _build(
    bus: int,
    address: int,
    width: int,
    payloads: Sequence[bytes],
    bits: BitProfile,
    matrix: np.ndarray,
    byte_matrix: np.ndarray | None = None,
    **kwargs,
) -> MessageInference:
    """Run every detector over one message's payloads.

    `byte_matrix` is the payloads already gathered as an (n, width) array. The
    checksum and CRC searches score every frame at once against it, so handing
    over the one the caller already has avoids rebuilding it per message.
    """
    if byte_matrix is None:
        from .checksums import as_matrix

        byte_matrix = as_matrix(payloads)
    candidates = set(checksum_candidate_bytes(bits))
    crc16_candidates = set(checksum_candidate_bytes(bits, min_rate=CRC16_BYTE_MIN_RATE))
    counters = find_counters(matrix, **kwargs.get("counter_options", {}))
    checksums = find_checksums(
        payloads,
        address,
        candidate_bytes=sorted(candidates),
        matrix=byte_matrix,
        **kwargs.get("checksum_options", {}),
    )
    # Honda's checksum is four bits in the low nibble of the last byte, which
    # no byte-wide search can see. Tried only where nothing wider explained
    # that byte, so the simpler explanation still wins when there is one.
    if width - 1 not in {c.byte_index for c in checksums}:
        nibble = find_honda_nibble(byte_matrix, address)
        if nibble is not None:
            checksums.append(nibble)

    # The E2E forms are tried only where the plain forms found nothing, so
    # that the simplest algorithm which fits is always the one reported.
    explained = {c.byte_index for c in checksums}
    alive = next(((c.start_bit, c.length) for c in counters if c.length == 4), None)
    checksums += find_e2e_crc8(
        payloads,
        address=address,
        candidate_bytes=sorted(candidates - explained),
        matrix=byte_matrix,
        counter=alive,
    )
    # The remaining profiles run only where the payload could hold their
    # header at all -- a Profile 6 is never tried on an 8-byte frame -- and
    # only on bytes nothing simpler has already explained.
    from .profiles import run_profiles

    explained = {c.byte_index for c in checksums}
    eight, wider = run_profiles(
        byte_matrix, counters, candidate_bytes=sorted(candidates - explained)
    )
    checksums += eight
    # A 16-bit CRC needs two adjacent bytes that both move like noise.
    crc16s = find_crc16(
        payloads,
        candidate_bytes=[s for s in crc16_candidates if s + 1 in crc16_candidates],
        matrix=byte_matrix,
        **kwargs.get("crc16_options", {}),
    ) + wider
    explained_bytes = {c.byte_index for c in checksums}
    for crc in crc16s:
        explained_bytes.update(range(crc.start_byte, crc.start_byte + crc.nbytes))
    multiplexor = find_multiplexor(
        matrix,
        skip_bytes=explained_bytes,
        counter_bits={b for c in counters for b in range(c.start_bit, c.end_bit)},
    )
    kept = outside_multiplex(outside_checksums(counters, checksums, crc16s), multiplexor)

    # Last, over whatever nothing else explained. A signal is the weakest
    # claim here -- bits that move together, with no arithmetic to check it
    # against -- so it never takes bits from a detector that can verify one.
    spoken_for = set(explained_bytes_to_bits(checksums, crc16s))
    for counter in kept:
        spoken_for.update(range(counter.start_bit, counter.end_bit))
    if multiplexor is not None:
        spoken_for.update(range(multiplexor.start_bit, multiplexor.end_bit))
    # The bus's bit order, decided once for the whole bus by
    # `infer.byteorder` and passed down. Intel unless something says otherwise.
    signals = find_signals_in(
        byte_matrix,
        kwargs.get("signal_order", BitOrder.INTEL),
        claimed_bits=spoken_for,
        **kwargs.get("signal_options", {}),
    )

    return MessageInference(
        bus=bus,
        address=address,
        width=width,
        frames=matrix.shape[0],
        bits=bits,
        counters=kept,
        checksums=checksums,
        crc16s=crc16s,
        multiplexor=multiplexor,
        signals=signals,
    )


def explained_bytes_to_bits(
    checksums: Sequence[ChecksumHypothesis], crc16s: Sequence[Crc16Hypothesis]
) -> set[int]:
    """Every bit a checksum or CRC accounts for."""
    out: set[int] = set()
    for check in checksums:
        out.update(range(check.start_bit, check.start_bit + check.length))
    for crc in crc16s:
        out.update(range(crc.start_byte * 8, (crc.start_byte + crc.nbytes) * 8))
    return out


def outside_multiplex(
    counters: list[CounterHypothesis], multiplexor: MultiplexHypothesis | None
) -> list[CounterHypothesis]:
    """Drop counters that are really the selector, or a slice it decides.

    A selector that cycles 0, 1, 2 is numerically a counter modulo 3, and the
    bytes it multiplexes cycle with it: the Jeep's VIN message showed seven
    "2-bit counters" that were all the same three-frame cycle seen through
    different windows. A counter that moves inside every group -- an alive
    counter on a multiplexed PDU -- is untouched, because it is not among the
    dependent bits.
    """
    if multiplexor is None:
        return counters
    taken = set(multiplexor.dependent_bits) | set(
        range(multiplexor.start_bit, multiplexor.end_bit)
    )
    # Whole containment, not overlap: a counter advancing once per frame
    # under a two-frame schedule has its lowest bit locked to the selector,
    # and that one bit does not make the counter a slice of the layout.
    return [c for c in counters if not set(range(c.start_bit, c.end_bit)) <= taken]


def outside_checksums(
    counters: list[CounterHypothesis],
    checksums: Sequence[ChecksumHypothesis],
    crc16s: Sequence[Crc16Hypothesis],
) -> list[CounterHypothesis]:
    """Drop counters that live inside a byte a checksum already explains.

    Bit-granular, not byte-granular, because Honda's checksum is half a byte:
    the other nibble of that byte is ordinary data and often carries the
    counter, which must stay reportable.

    A CRC is linear over GF(2), so when the only thing moving in a message is
    its alive counter the CRC byte is an affine image of that counter -- and
    two of its bits can walk 0..3 as faithfully as any real counter. Every
    Jeep Grand Cherokee message with a J1850 CRC showed a phantom 2-bit
    counter in the CRC byte for exactly this reason. The counter scan cannot
    tell the two apart, but once the byte is known to be a checksum it is
    not also a counter. The unfiltered list still feeds the E2E searches
    above, which only ever read the 4-bit alive counter.
    """
    explained: set[int] = set()
    for check in checksums:
        explained.update(range(check.start_bit, check.start_bit + check.length))
    for crc in crc16s:
        explained.update(range(crc.start_byte * 8, (crc.start_byte + crc.nbytes) * 8))
    return [c for c in counters if not explained & set(range(c.start_bit, c.end_bit))]


def infer_message_columnar(
    message: Message,
    *,
    order: BitOrder = BitOrder.INTEL,
    bits: BitProfile | None = None,
    **kwargs,
) -> MessageInference:
    """Infer one message straight from its columns.

    `bits` lets a caller that has already measured the message -- the analyze
    pass does, over exactly the same frames -- hand the profile in rather than
    have it computed a second time.
    """
    byte_matrix = message.bytes_matrix()
    if bits is None or bits.width != message.width or bits.frames != byte_matrix.shape[0]:
        bits = profile_bits_from_bytes(byte_matrix, order)
    matrix = bit_matrix_from_bytes(byte_matrix, order)
    payloads = message.frames.payloads(message.index[:PAYLOAD_SAMPLE])
    return _build(
        message.bus, message.address, message.width, payloads, bits, matrix,
        byte_matrix=byte_matrix, **kwargs,
    )


def claimed_bits_of(inference: MessageInference) -> set[int]:
    """Flat Intel positions every verified detector already explained."""
    bits = set(explained_bytes_to_bits(inference.checksums, inference.crc16s))
    for counter in inference.counters:
        bits.update(range(counter.start_bit, counter.end_bit))
    if inference.multiplexor is not None:
        bits.update(range(inference.multiplexor.start_bit, inference.multiplexor.end_bit))
    return bits


def apply_bus_order(
    found: list[MessageInference], byte_matrices: dict[tuple[int, int], np.ndarray]
) -> list[MessageInference]:
    """Decide each bus's bit order, then re-read signals where it is not Intel.

    Two passes, because the decision needs the other detectors first: a
    counter or a checksum must be excluded before long fields are counted, or
    it would be mistaken for one in whichever order happens to fit it. The
    first pass finds those, the decision follows, and only the signal pass is
    redone -- checksums are arithmetic over bytes and do not move with bit
    numbering.

    A bus below `byteorder.MIN_MARGIN` is left as Intel, which is the default
    and the thing to fall back to when the traffic will not say.
    """
    from .byteorder import decide_byte_order

    claimed = {m.key: claimed_bits_of(m) for m in found}
    decided = decide_byte_order(
        [
            (m.bus, byte_matrices[m.key], claimed[m.key])
            for m in found
            if m.key in byte_matrices
        ]
    )
    motorola = {
        bus
        for bus, verdict in decided.items()
        if verdict.decided and verdict.order is BitOrder.MOTOROLA
    }
    if not motorola:
        return found
    out = []
    for message in found:
        if message.bus in motorola and message.key in byte_matrices:
            message = replace(
                message,
                signals=find_signals_in(
                    byte_matrices[message.key],
                    BitOrder.MOTOROLA,
                    claimed_bits=claimed[message.key],
                ),
            )
        out.append(message)
    return out


def infer_frameset(
    frames: FrameSet,
    *,
    order: BitOrder = BitOrder.INTEL,
    min_frames: int = 32,
    profile: TraceProfile | None = None,
) -> list[MessageInference]:
    """Infer every message of a columnar trace. The fast path.

    Pass the `TraceProfile` from `analyze_frameset` over the same frames and
    each message's bit profile is reused instead of measured again.
    """
    messages = list(frames.group(min_frames=min_frames))
    found = [
        infer_message_columnar(
            message,
            order=order,
            bits=profile[message.key].bits
            if profile is not None and message.key in profile.messages
            else None,
        )
        for message in messages
    ]
    if order is not BitOrder.INTEL:
        return found
    return apply_bus_order(found, {m.key: m.bytes_matrix() for m in messages})


def infer_frames(
    frames: Iterable[CanFrame] | FrameSet,
    *,
    order: BitOrder = BitOrder.INTEL,
    min_frames: int = 32,
) -> list[MessageInference]:
    """Group a frame stream by message and infer each one.

    Messages seen fewer than `min_frames` times are skipped: a counter or a
    checksum claimed from a handful of samples is noise dressed as a finding.
    """
    if isinstance(frames, FrameSet):
        return infer_frameset(frames, order=order, min_frames=min_frames)

    payloads: dict[tuple[int, int], list[bytes]] = defaultdict(list)
    for frame in frames:
        payloads[(frame.bus, frame.address)].append(frame.data)

    from .checksums import as_matrix

    results = []
    matrices: dict[tuple[int, int], np.ndarray] = {}
    for (bus, address), blobs in payloads.items():
        if len(blobs) < min_frames:
            continue
        # Pool only the dominant payload length, for the same reason analyze does.
        width = Counter(len(b) for b in blobs).most_common(1)[0][0]
        same = [b for b in blobs if len(b) == width]
        if len(same) < min_frames:
            continue
        results.append(infer_message(same, bus=bus, address=address, order=order))
        matrices[(bus, address)] = as_matrix(same)
    results = sorted(results, key=lambda m: (m.bus, m.address))
    if order is not BitOrder.INTEL:
        return results
    return apply_bus_order(results, matrices)


def infer_segment(
    path: str, *, root: str, order: BitOrder = BitOrder.INTEL, use_cache: bool = True, **kwargs
) -> list[MessageInference]:
    """Decode one segment and infer every message, through the cache by default."""
    if use_cache:
        return infer_frameset(load_frames(path, root=root), order=order, **kwargs)
    return infer_frames(iter_frames(path, root=root), order=order, **kwargs)
