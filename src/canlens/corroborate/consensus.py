# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Cross-segment corroboration: what holds across a platform, not one trace.

A single trace can attach a confidence to a hypothesis; it cannot check it.
Here the same message is looked at across every local segment of a platform,
and each hypothesis is scored by how many of those segments -- and, more
importantly, how many distinct *devices* -- agree with it.

Devices matter more than segments. The corpus stores segments under
``<device>/<route>/<index>``, and a device is one physical car. Eight hundred
segments from one car establish what that car does; eighty from thirty cars
establish what the platform does. Every piece of evidence therefore carries
both counts, and the strongest tier requires several devices, not merely many
segments.

The bit-level consensus is also where the corpus yields information a single
trace cannot contain at all. A bit that is constant in 95% of segments and
moves in the other 5% is not padding: it is a state that rarely changes -- a
door, a mode, a warning -- and only the spread of segments reveals it. Those
are reported as *rare* bits.

Everything here consumes cached inference results, so corroborating a platform
costs a few milliseconds per segment once the cache is warm.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum

from ..analyze.bits import KIND_ORDER, BitKind
from ..infer import MessageInference

# Tier thresholds -- heuristics for partitioning evidence, documented so they
# can be argued with. A hypothesis is "established" when nearly every segment
# that carries the message agrees with it AND those segments come from at
# least MIN_DEVICES distinct cars. Fewer cars than that can only ever reach
# "consistent": consistent in everything seen, but not shown to generalise.
ESTABLISHED_SUPPORT = 0.90
PARTIAL_SUPPORT = 0.50
MIN_DEVICES = 3


class Tier(str, Enum):
    ESTABLISHED = "established"  # near-unanimous across several cars
    CONSISTENT = "consistent"  # near-unanimous, but too few cars to generalise
    PARTIAL = "partial"  # holds in some segments, not most
    WEAK = "weak"  # a minority; treat as a single-trace guess

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Evidence:
    """How much of the platform stands behind one hypothesis."""

    segments: int  # segments in which the hypothesis was found
    devices: int  # distinct devices among those segments
    of_segments: int  # segments in which the message occurred at all
    of_devices: int  # distinct devices among *those*
    mean_match: float
    min_match: float

    @property
    def support(self) -> float:
        """Share of the message's segments that carry this hypothesis."""
        return self.segments / self.of_segments if self.of_segments else 0.0

    @property
    def tier(self) -> Tier:
        if self.support >= ESTABLISHED_SUPPORT:
            return Tier.ESTABLISHED if self.devices >= MIN_DEVICES else Tier.CONSISTENT
        if self.support >= PARTIAL_SUPPORT:
            return Tier.PARTIAL
        return Tier.WEAK

    def __str__(self) -> str:
        return (
            f"{self.segments}/{self.of_segments} seg, {self.devices}/{self.of_devices} dev, "
            f"{self.tier}"
        )


@dataclass
class CounterConsensus:
    start_bit: int
    length: int
    stride: int
    evidence: Evidence
    contested: bool = False
    modulus: int = 0  # 0 = the natural 2**length

    @property
    def end_bit(self) -> int:
        return self.start_bit + self.length

    def __str__(self) -> str:
        step = "" if self.stride == 1 else f" step {self.stride}"
        wrap = f" mod {self.modulus}" if self.modulus else ""
        flag = " CONTESTED" if self.contested else ""
        return (
            f"{self.length}-bit counter @ bit {self.start_bit}{step}{wrap} "
            f"({self.evidence}){flag}"
        )


@dataclass
class ChecksumConsensus:
    byte_index: int
    algorithm: str
    evidence: Evidence
    contested: bool = False
    data_id: int | None = None

    def __str__(self) -> str:
        flag = " CONTESTED" if self.contested else ""
        ident = "" if self.data_id is None else f", data ID 0x{self.data_id:X}"
        return f"{self.algorithm} @ byte {self.byte_index} ({self.evidence}{ident}){flag}"


@dataclass
class Crc16Consensus:
    start_byte: int
    algorithm: str
    byteorder: str
    data_id: int | None
    evidence: Evidence
    contested: bool = False
    nbytes: int = 2

    def __str__(self) -> str:
        ident = "" if self.data_id is None else f", data ID 0x{self.data_id:X}"
        flag = " CONTESTED" if self.contested else ""
        return (
            f"{self.algorithm} @ bytes {self.start_byte}-{self.start_byte + self.nbytes - 1} "
            f"({self.byteorder}{ident}; {self.evidence}){flag}"
        )


@dataclass
class MultiplexConsensus:
    start_bit: int
    length: int
    values: tuple[int, ...]  # union of the values seen across segments
    evidence: Evidence
    contested: bool = False

    @property
    def end_bit(self) -> int:
        return self.start_bit + self.length

    def __str__(self) -> str:
        shown = ", ".join(str(v) for v in self.values[:8])
        more = f", … ({len(self.values)} values)" if len(self.values) > 8 else ""
        flag = " CONTESTED" if self.contested else ""
        return (
            f"{self.length}-bit multiplexor @ bit {self.start_bit}: values {shown}{more} "
            f"({self.evidence}){flag}"
        )


@dataclass(frozen=True)
class BitConsensus:
    """What a bit position does across segments."""

    kind: BitKind  # the most common class
    agreement: float  # share of segments classifying it that way
    rare: bool  # constant almost everywhere, but seen to move


@dataclass
class MessageConsensus:
    bus: int
    address: int
    width: int
    width_agreement: float
    segments: int
    devices: int
    of_segments: int
    of_devices: int
    bits: list[BitConsensus] = field(default_factory=list)
    counters: list[CounterConsensus] = field(default_factory=list)
    checksums: list[ChecksumConsensus] = field(default_factory=list)
    crc16s: list[Crc16Consensus] = field(default_factory=list)
    multiplexors: list[MultiplexConsensus] = field(default_factory=list)

    @property
    def key(self) -> tuple[int, int]:
        return (self.bus, self.address)

    @property
    def presence(self) -> float:
        """Share of the platform's segments in which this message appears."""
        return self.segments / self.of_segments if self.of_segments else 0.0

    @property
    def kinds(self) -> list[BitKind]:
        return [b.kind for b in self.bits]

    @property
    def agreement(self) -> float:
        """Mean per-bit agreement: how stable the message's layout is."""
        return sum(b.agreement for b in self.bits) / len(self.bits) if self.bits else 0.0

    @property
    def rare_bits(self) -> list[int]:
        return [i for i, b in enumerate(self.bits) if b.rare]

    def __str__(self) -> str:
        return f"bus {self.bus} 0x{self.address:03X}"


@dataclass
class PlatformConsensus:
    platform: str
    segments: int
    devices: int
    messages: dict[tuple[int, int], MessageConsensus]

    def __len__(self) -> int:
        return len(self.messages)

    def __getitem__(self, key: tuple[int, int]) -> MessageConsensus:
        return self.messages[key]

    def by_identifier(self) -> list[MessageConsensus]:
        return sorted(self.messages.values(), key=lambda m: (m.bus, m.address > 0x7FF, m.address))

    def by_presence(self) -> list[MessageConsensus]:
        return sorted(self.messages.values(), key=lambda m: (-m.presence, m.bus, m.address))


def device_of(path: str) -> str:
    """The device a stored segment belongs to: the first path component."""
    parts = [p for p in path.replace("\\", "/").split("/") if p]
    if parts and parts[-1].endswith(".zst"):
        parts = parts[:-1]
    return parts[-3] if len(parts) >= 3 else ""


def _evidence(
    hits: dict[str, list[float]], of_segments: int, of_devices: int
) -> Evidence:
    """Evidence from {device: [match rates in that device's segments]}."""
    rates = [r for per_device in hits.values() for r in per_device]
    return Evidence(
        segments=len(rates),
        devices=len(hits),
        of_segments=of_segments,
        of_devices=of_devices,
        mean_match=sum(rates) / len(rates) if rates else 0.0,
        min_match=min(rates) if rates else 0.0,
    )


def corroborate(
    observations: Iterable[tuple[str, list[MessageInference]]], *, platform: str = ""
) -> PlatformConsensus:
    """Consolidate per-segment inferences into per-message consensus.

    `observations` pairs each segment's device with its inference results.
    Only what the inference layer already produced is consulted: nothing is
    decoded or re-measured here, which is what keeps a platform-wide pass
    cheap once the results cache is warm.
    """
    per_key: dict[tuple[int, int], list[tuple[str, MessageInference]]] = defaultdict(list)
    all_devices: set[str] = set()
    total_segments = 0
    for device, inferences in observations:
        total_segments += 1
        all_devices.add(device)
        for inference in inferences:
            per_key[inference.key].append((device, inference))

    messages = {}
    for key, seen in per_key.items():
        messages[key] = _consolidate(key, seen)
    return PlatformConsensus(
        platform=platform,
        segments=total_segments,
        devices=len(all_devices),
        messages=messages,
    )


def _consolidate(
    key: tuple[int, int], seen: list[tuple[str, MessageInference]]
) -> MessageConsensus:
    widths = Counter(m.width for _, m in seen)
    width = widths.most_common(1)[0][0]
    same = [(d, m) for d, m in seen if m.width == width]
    of_segments = len(same)
    of_devices = len({d for d, _ in same})

    # Bit classes: the mode per position, and how much of the platform agrees.
    n_bits = width * 8
    tallies: list[Counter[BitKind]] = [Counter() for _ in range(n_bits)]
    for _, m in same:
        for i, kind in enumerate(m.bits.kinds[:n_bits]):
            tallies[i][kind] += 1
    bits = []
    for tally in tallies:
        if not tally:
            bits.append(BitConsensus(BitKind.CONSTANT, 0.0, False))
            continue
        kind, count = max(tally.items(), key=lambda kv: (kv[1], -KIND_ORDER.index(kv[0])))
        moved = sum(c for k, c in tally.items() if k is not BitKind.CONSTANT)
        bits.append(
            BitConsensus(
                kind=kind,
                agreement=count / of_segments,
                rare=kind is BitKind.CONSTANT and moved > 0,
            )
        )

    # Hypotheses: keyed on their full parameters, so two segments that agree
    # on a counter but disagree on its stride count as two hypotheses.
    counters: dict[tuple[int, int, int, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    checksums: dict[tuple[int, str, int | None], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    crc16s: dict[tuple[int, str, str, int | None, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    multiplexors: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    mux_values: dict[tuple[int, int], set[int]] = defaultdict(set)
    for device, m in same:
        if m.multiplexor is not None:
            mux = m.multiplexor
            multiplexors[(mux.start_bit, mux.length)][device].append(mux.coverage)
            mux_values[(mux.start_bit, mux.length)].update(mux.values)
        for ctr in m.counters:
            counters[(ctr.start_bit, ctr.length, ctr.stride, ctr.modulus)][device].append(
                ctr.match_rate
            )
        for chk in m.checksums:
            checksums[(chk.byte_index, chk.algorithm, chk.data_id)][device].append(
                chk.match_rate
            )
        for crc in m.crc16s:
            crc16s[(crc.start_byte, crc.algorithm, crc.byteorder, crc.data_id, crc.nbytes)][
                device
            ].append(crc.match_rate)

    counter_out = [
        CounterConsensus(s, n, stride, _evidence(hits, of_segments, of_devices), modulus=mod)
        for (s, n, stride, mod), hits in counters.items()
    ]
    checksum_out = [
        ChecksumConsensus(b, algo, _evidence(hits, of_segments, of_devices), data_id=ident)
        for (b, algo, ident), hits in checksums.items()
    ]
    crc16_out = [
        Crc16Consensus(b, algo, order, ident, _evidence(hits, of_segments, of_devices), nbytes=n)
        for (b, algo, order, ident, n), hits in crc16s.items()
    ]
    mux_out = [
        MultiplexConsensus(
            s, n, tuple(sorted(mux_values[(s, n)])), _evidence(hits, of_segments, of_devices)
        )
        for (s, n), hits in multiplexors.items()
    ]
    _mark_contested(counter_out, checksum_out, crc16_out, mux_out)

    strongest = lambda item: (-item.evidence.support, -item.evidence.devices)
    return MessageConsensus(
        bus=key[0],
        address=key[1],
        width=width,
        width_agreement=of_segments / len(seen),
        segments=len(seen),
        devices=len({d for d, _ in seen}),
        of_segments=of_segments,
        of_devices=of_devices,
        bits=bits,
        counters=sorted(counter_out, key=strongest),
        checksums=sorted(checksum_out, key=strongest),
        crc16s=sorted(crc16_out, key=strongest),
        multiplexors=sorted(mux_out, key=strongest),
    )


def _mark_contested(
    counters: list[CounterConsensus],
    checksums: list[ChecksumConsensus],
    crc16s: list[Crc16Consensus],
    multiplexors: list[MultiplexConsensus] | None = None,
) -> None:
    """Flag hypotheses that claim the same bits with different parameters.

    Two segments proposing different counters over overlapping bits cannot
    both describe the platform; neither is discarded, both are marked, and the
    evidence counts say which one the corpus actually backs.
    """
    for i, ctr in enumerate(counters):
        for other_ctr in counters[i + 1 :]:
            if ctr.start_bit < other_ctr.end_bit and other_ctr.start_bit < ctr.end_bit:
                ctr.contested = other_ctr.contested = True
    for i, chk in enumerate(checksums):
        for other_chk in checksums[i + 1 :]:
            if chk.byte_index == other_chk.byte_index:
                chk.contested = other_chk.contested = True
    for i, crc in enumerate(crc16s):
        for other_crc in crc16s[i + 1 :]:
            if crc.start_byte == other_crc.start_byte:
                crc.contested = other_crc.contested = True
    # A message has one selector; two different ones cannot both be right.
    for i, mux in enumerate(multiplexors or []):
        for other_mux in multiplexors[i + 1 :]:  # type: ignore[index]
            mux.contested = other_mux.contested = True


def corroborate_platform(
    platform: str, *, root: str, limit: int | None = None, progress=None
) -> PlatformConsensus:
    """Corroborate every local segment of one platform, through the caches."""
    from ..corpus import Manifest, inventory
    from ..infer import infer_cached

    manifest = Manifest.load(f"{root}/database.json")
    held = inventory(root, manifest).get(platform)
    paths = (held.paths if held else [])[:limit]

    def observations():
        for done, path in enumerate(paths, start=1):
            yield device_of(path), infer_cached(path, root=root)
            if progress is not None:
                progress(done, len(paths))

    return corroborate(observations(), platform=platform)
