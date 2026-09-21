# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Candidate checksum algorithms, and the search that tests them.

A hypothesis here is only ever reported if it *reproduces* the observed byte,
on nearly every frame in the trace. Nothing is inferred from a byte merely
looking random -- that is what :mod:`canlens.analyze` already told us, and it
is exactly the kind of guess this layer exists to replace.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

# A checksum function sees the whole payload, the message address, and which
# byte is under test, and returns the byte it expects to find there.
ChecksumFn = Callable[[bytes, int, int], int]


def _others(payload: bytes, index: int) -> bytes:
    """The payload with the byte under test removed."""
    return payload[:index] + payload[index + 1 :]


def sum8(payload: bytes, address: int, index: int) -> int:
    return sum(_others(payload, index)) & 0xFF


def sum8_complement(payload: bytes, address: int, index: int) -> int:
    return (-sum(_others(payload, index))) & 0xFF


def xor8(payload: bytes, address: int, index: int) -> int:
    acc = 0
    for byte in _others(payload, index):
        acc ^= byte
    return acc


def toyota(payload: bytes, address: int, index: int) -> int:
    """Toyota's checksum: byte sum folded together with the address and length."""
    return (
        sum(_others(payload, index)) + (address & 0xFF) + ((address >> 8) & 0xFF) + len(payload)
    ) & 0xFF


def tesla(payload: bytes, address: int, index: int) -> int:
    """Tesla's checksum: byte sum folded with the address, and no length term.

    Derived from the corpus rather than from a document: over the eleven
    messages `tesla_model3_party.dbc` names a checksum on, solving for the
    additive constant that reproduces the byte gives exactly
    `(address & 0xFF) + (address >> 8)` every time, at a 100% match rate.

    It differs from :func:`toyota` only by the missing length term, so the two
    can never both fit the same message -- a payload length is between 1 and
    64 and so never vanishes modulo 256.
    """
    return (
        sum(_others(payload, index)) + (address & 0xFF) + ((address >> 8) & 0xFF)
    ) & 0xFF


def _crc8(data: bytes, poly: int, init: int, xorout: int) -> int:
    crc = init
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc ^ xorout


def _crc8_factory(poly: int, init: int, xorout: int) -> ChecksumFn:
    def fn(payload: bytes, address: int, index: int) -> int:
        return _crc8(_others(payload, index), poly, init, xorout)

    return fn


def _crc8_table(poly: int) -> np.ndarray:
    """One byte's worth of CRC8 shifting, precomputed for every input."""
    table = np.zeros(256, dtype=np.uint8)
    for value in range(256):
        crc = value
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
        table[value] = crc
    return table


# Vectorised twins of the scalar algorithms above. Each takes an
# (n_frames, width) byte matrix and returns the byte it expects at `index` for
# every frame at once, replacing a Python loop over several hundred thousand
# payloads. `test_vectorised_matches_scalar` holds them to the originals.
VectorFn = Callable[[np.ndarray, int, int], np.ndarray]


def _sum_without(matrix: np.ndarray, index: int) -> np.ndarray:
    return matrix.sum(axis=1, dtype=np.uint32) - matrix[:, index]


def v_sum8(matrix: np.ndarray, address: int, index: int) -> np.ndarray:
    return (_sum_without(matrix, index) & 0xFF).astype(np.uint8)


def v_sum8_complement(matrix: np.ndarray, address: int, index: int) -> np.ndarray:
    return ((-_sum_without(matrix, index).astype(np.int64)) & 0xFF).astype(np.uint8)


def v_xor8(matrix: np.ndarray, address: int, index: int) -> np.ndarray:
    # XOR is self-inverse, so the byte under test removes itself again.
    return np.bitwise_xor.reduce(matrix, axis=1) ^ matrix[:, index]


def v_toyota(matrix: np.ndarray, address: int, index: int) -> np.ndarray:
    folded = (
        _sum_without(matrix, index)
        + (address & 0xFF)
        + ((address >> 8) & 0xFF)
        + matrix.shape[1]
    )
    return (folded & 0xFF).astype(np.uint8)


def v_tesla(matrix: np.ndarray, address: int, index: int) -> np.ndarray:
    folded = _sum_without(matrix, index) + (address & 0xFF) + ((address >> 8) & 0xFF)
    return (folded & 0xFF).astype(np.uint8)


def _v_crc8_factory(poly: int, init: int, xorout: int) -> VectorFn:
    table = _crc8_table(poly)

    def fn(matrix: np.ndarray, address: int, index: int) -> np.ndarray:
        crc = np.full(matrix.shape[0], init, dtype=np.uint8)
        for column in range(matrix.shape[1]):
            if column == index:
                continue  # the byte under test is not part of its own input
            crc = table[crc ^ matrix[:, column]]
        return crc ^ np.uint8(xorout)

    return fn


# Ordered cheapest-and-commonest first; the search stops at the first algorithm
# that reproduces the byte, so a plain sum is never reported as an exotic CRC.
ALGORITHMS: dict[str, ChecksumFn] = {
    "sum8": sum8,
    "sum8_complement": sum8_complement,
    "xor8": xor8,
    "toyota": toyota,
    "tesla": tesla,
    "crc8": _crc8_factory(0x07, 0x00, 0x00),
    "crc8_j1850": _crc8_factory(0x1D, 0xFF, 0xFF),
    "crc8_2f": _crc8_factory(0x2F, 0xFF, 0xFF),
    "crc8_autosar": _crc8_factory(0x2F, 0xFF, 0x00),
}


VECTOR_ALGORITHMS: dict[str, VectorFn] = {
    "sum8": v_sum8,
    "sum8_complement": v_sum8_complement,
    "xor8": v_xor8,
    "toyota": v_toyota,
    "tesla": v_tesla,
    "crc8": _v_crc8_factory(0x07, 0x00, 0x00),
    "crc8_j1850": _v_crc8_factory(0x1D, 0xFF, 0xFF),
    "crc8_2f": _v_crc8_factory(0x2F, 0xFF, 0xFF),
    "crc8_autosar": _v_crc8_factory(0x2F, 0xFF, 0x00),
}


# A CRC hypothesis with a solved-for secret is only evidence when the data
# constrains the secret. Each distinct payload content (with the CRC bytes
# removed) is one 8-bit equation; a secret of k bytes needs more than k of
# them or any value fits trivially. A static message with an alive counter has
# sixteen distinct contents, which is exactly the size of a Profile 22 list --
# and so, before this gate, Profile 22 was "found" on 136 of 230 platforms and
# Profile 6 on 139, including Toyota, which uses a byte sum. The margin is what
# makes a coincidental fit improbable rather than merely non-trivial.
MIN_EQUATIONS = 8


def distinct_contents(matrix: np.ndarray, skip: Sequence[int]) -> int:
    """How many distinct payloads there are once the CRC bytes are removed."""
    keep = [column for column in range(matrix.shape[1]) if column not in set(skip)]
    if not keep or matrix.shape[0] == 0:
        return 0
    rows = np.ascontiguousarray(matrix[:, keep])
    return int(np.unique(rows.view(np.dtype((np.void, rows.shape[1])))).size)


def enough_evidence(matrix: np.ndarray, skip: Sequence[int], secret_bytes: int) -> bool:
    """Whether the data can constrain a secret of `secret_bytes` bytes."""
    return distinct_contents(matrix, skip) >= secret_bytes + MIN_EQUATIONS


def as_matrix(payloads: Sequence[bytes]) -> np.ndarray:
    """Equal-width payloads as an (n, width) byte matrix."""
    if not payloads:
        return np.zeros((0, 0), dtype=np.uint8)
    width = len(payloads[0])
    return np.frombuffer(b"".join(payloads), dtype=np.uint8).reshape(len(payloads), width)


@dataclass(frozen=True)
class ChecksumHypothesis:
    """A checksum byte whose value the named algorithm reproduces."""

    byte_index: int
    algorithm: str
    match_rate: float
    frames: int
    # Set for the AUTOSAR E2E Profile 1/11 forms, whose CRC covers an implicit
    # Data ID that never appears on the wire and is solved for instead.
    data_id: int | None = None
    # Some algorithms are self-inverse: if byte 7 is the XOR of bytes 0-6 then
    # byte 0 is equally the XOR of bytes 1-7, so every position "verifies" and
    # the trace alone cannot say which byte the protocol calls the checksum.
    ambiguous_positions: tuple[int, ...] = ()

    @property
    def start_bit(self) -> int:
        return self.byte_index * 8

    @property
    def length(self) -> int:
        return 8

    @property
    def ambiguous(self) -> bool:
        return len(self.ambiguous_positions) > 1

    def __str__(self) -> str:
        where = (
            f"any of bytes {list(self.ambiguous_positions)}"
            if self.ambiguous
            else f"byte {self.byte_index}"
        )
        if self.data_id is None:
            ident = ""
        elif self.algorithm == "e2e_p22":
            ids = " ".join(f"{b:02X}" for b in self.data_id.to_bytes(16, "little"))
            ident = f", data ID list [{ids}]"
        else:
            ident = f", data ID 0x{self.data_id:X}"
        return f"{self.algorithm} @ {where} ({self.match_rate:.1%}{ident})"


def score_algorithm(
    payloads: Sequence[bytes], address: int, index: int, fn: ChecksumFn, min_match: float = 0.0
) -> float:
    """Fraction of frames whose byte `index` the algorithm reproduces.

    Abandoned as soon as `min_match` is out of reach. Nearly every candidate is
    wrong and fails on the first frame or two, so scoring the rest is waste --
    this is most of the difference between a segment sweep taking half a minute
    and taking a second.
    """
    total = len(payloads)
    if not total:
        return 0.0
    budget = total - int(min_match * total)
    hits = misses = 0
    for payload in payloads:
        if fn(payload, address, index) == payload[index]:
            hits += 1
        else:
            misses += 1
            if misses > budget:
                return hits / total
    return hits / total


def find_checksums(
    payloads: Sequence[bytes],
    address: int,
    *,
    candidate_bytes: Sequence[int] | None = None,
    min_match: float = 0.99,
    screen_frames: int = 256,
    matrix: np.ndarray | None = None,
) -> list[ChecksumHypothesis]:
    """Search for a byte that a known algorithm reproduces.

    `candidate_bytes` narrows the search to positions that already look like a
    checksum; without it every byte is tried. A cheap screen over the first
    `screen_frames` frames rejects almost everything before the full trace is
    scored, which is what keeps a whole-segment sweep affordable.
    """
    blob = as_matrix(payloads) if matrix is None else matrix
    if blob.shape[0] == 0:
        return []
    # When the matrix is supplied, `payloads` may be only a sample of it; the
    # matrix is what is scored, so the matrix says how many frames there were.
    frames = blob.shape[0]
    width = blob.shape[1]
    positions = candidate_bytes if candidate_bytes is not None else range(width)
    # The screen is now only worth having for very long traces: scoring every
    # frame is one pass over an array rather than a loop in Python.
    screen = blob[:screen_frames]

    hits: dict[str, list[tuple[int, float]]] = {}
    for index in positions:
        if not 0 <= index < width:
            continue
        for name, fn in VECTOR_ALGORITHMS.items():
            if float((fn(screen, address, index) == screen[:, index]).mean()) < min_match:
                continue
            rate = float((fn(blob, address, index) == blob[:, index]).mean())
            if rate >= min_match:
                hits.setdefault(name, []).append((index, rate))
                break  # first (simplest) algorithm that works wins

    found = []
    for name, matches in hits.items():
        indices = tuple(i for i, _ in matches)
        # Convention puts a checksum last, so that is the one reported -- but
        # every position that verified is carried along rather than hidden.
        index, rate = matches[-1]
        found.append(
            ChecksumHypothesis(index, name, rate, frames,
                               ambiguous_positions=indices if len(indices) > 1 else ())
        )
    return sorted(found, key=lambda f: f.byte_index)


# ---------------------------------------------------------------------------
# AUTOSAR E2E Profiles 1 and 11: CRC-8 SAE J1850 over an implicit Data ID.
#
# Per AUTOSAR_PRS_E2EProtocol (FO R19-11): [PRS_E2E_00082] the CRC is computed
# first over the Data ID bytes and then over every transmitted byte except the
# CRC byte; [PRS_E2E_00505] mode BOTH feeds the Data ID low byte then its high
# byte; [PRS_E2E_00506] mode NIBBLE feeds the low byte then a zero byte (the
# high nibble travels in the data, where the CRC covers it as data);
# [PRS_E2E_00163] Profile 1 additionally allows LOW (low byte only) and ALT
# (low byte for even counters, high byte for odd). Every call in the flowcharts
# is Crc_CalculateCRC8(..., Crc_StartValue8: 0xFF, Crc_IsFirstCall: FALSE),
# which under the R4 CRC library puts the register at 0x00 to begin with.
#
# The final XOR is settled by data, not by the flowchart. The library applies
# 0xFF on output, and a first reading of the spec suggested the same here; but
# for a fixed-length message a final XOR is absorbed into an equivalent start
# state, so a single message cannot tell -- only structure across messages of
# *different* lengths can. On a vehicle with 94 such messages of three widths,
# the register-from-0x00, no-final-XOR convention with the Data ID fed as
# [addr & 0xFF, addr >> 8] recovers the CAN identifier as the Data ID for all
# 94; every other convention yields noise. That is the convention used.
#
# The Data ID is never on the wire. It is solved for: the register after the ID
# bytes can only take 256 values, so every single-byte mode is a 256-way search
# vectorised over all frames, and a real ID reproduces the CRC on every frame
# where a coincidence reproduces it on one.

E2E_XOR = 0x00
J1850_TABLE = _crc8_table(0x1D)


def e2e_crc8(payload: bytes, index: int, id_bytes: bytes) -> int:
    """Scalar reference for the Profile 1/11 CRC. `id_bytes` is the Data ID
    as fed: [low], [low, high], or [low, 0x00]."""
    crc = 0
    for byte in id_bytes + _others(payload, index):
        crc = int(J1850_TABLE[crc ^ byte])
    return crc ^ E2E_XOR


def _e2e_over_data(state: np.ndarray, matrix: np.ndarray, index: int) -> np.ndarray:
    """Run `state` (256, n) through every data column but `index`, then XOR out."""
    for column in range(matrix.shape[1]):
        if column == index:
            continue
        state = J1850_TABLE[state ^ matrix[None, :, column]]
    return state ^ np.uint8(E2E_XOR)


def _e2e_candidates(matrix: np.ndarray, index: int, second: int | None) -> np.ndarray:
    """(256, n) CRC for every Data ID low byte, optionally followed by `second`."""
    low = np.arange(256, dtype=np.uint8)
    state = np.repeat(J1850_TABLE[low][:, None], matrix.shape[0], axis=1)
    if second is not None:
        state = J1850_TABLE[state ^ np.uint8(second)]
    return _e2e_over_data(state, matrix, index)


def _e2e_single(blob: np.ndarray, index: int, low: int, second: int | None) -> np.ndarray:
    """(n,) CRC for one Data ID low byte, optionally followed by `second`."""
    state = np.full((1, blob.shape[0]), J1850_TABLE[low], dtype=np.uint8)
    if second is not None:
        state = J1850_TABLE[state ^ np.uint8(second)]
    return _e2e_over_data(state, blob, index)[0]


def _best_e2e(
    blob: np.ndarray, sample: np.ndarray, index: int, second: int | None, min_match: float
) -> tuple[int, float] | None:
    """The one Data ID low byte worth verifying, and its rate over every frame.

    All 256 candidates are scored on the sample only. A true Data ID reproduces
    the whole trace, so it reproduces the sample, so it is the sample's argmax;
    verifying that single candidate on every frame is therefore exact, and it
    costs one pass instead of 256. Measured on a Rivian segment with 159
    protected messages: 4.98 s to 2.87 s. Less than the arithmetic promises,
    so the sample screen itself -- 256 candidates, twice per candidate byte --
    is now where the time goes.
    """
    screened = (_e2e_candidates(sample, index, second) == sample[None, :, index]).mean(axis=1)
    best = int(screened.argmax())
    if screened[best] < min_match:
        return None
    rate = float((_e2e_single(blob, index, best, second) == blob[:, index]).mean())
    return (best, rate) if rate >= min_match else None


def find_e2e_crc8(
    payloads: Sequence[bytes],
    *,
    address: int | None = None,
    candidate_bytes: Sequence[int] | None = None,
    min_match: float = 0.99,
    matrix: np.ndarray | None = None,
    counter: tuple[int, int] | None = None,
    screen_frames: int = 64,
) -> list[ChecksumHypothesis]:
    """Find a Profile 1/11 CRC byte and recover its Data ID.

    One trace cannot tell the single-state modes apart. The register after the
    Data ID bytes takes one of 256 values, and each of NIBBLE, BOTH and LOW
    maps 256 Data IDs onto those 256 states bijectively -- so whichever mode
    the sender used, exactly one Data ID in *every* mode reproduces the CRC.
    They are therefore reported as one hypothesis, `e2e_p11`, with the Data ID
    under the [low, 0x00] convention shared by Profile 11 NIBBLE and BOTH with
    a zero high byte; `p01_low_id` converts it to the Profile 1 LOW reading.
    ALT is different: it alternates between two states by counter parity, so
    no single state fits, and it is falsifiable -- which is why the alive
    counter found earlier is passed in.

    One form *is* checkable from a single message: BOTH with the Data ID equal
    to the CAN identifier, fed as [addr & 0xFF, addr >> 8]. That is tried
    first, and claimed only when the recovered low byte comes out equal to
    addr & 0xFF -- a 1-in-256 coincidence otherwise, which corroboration
    across segments then settles. The reported `data_id` is the full
    identifier in that case.
    """
    blob = as_matrix(payloads) if matrix is None else matrix
    frames, width = blob.shape
    if frames < 8 or width < 2:
        return []
    positions = candidate_bytes if candidate_bytes is not None else range(width)
    sample = blob[:screen_frames]
    found = []
    for index in positions:
        if not 0 <= index < width or not enough_evidence(blob, [index], 1):
            continue
        hit = None
        if address is not None:
            got = _best_e2e(blob, sample, index, (address >> 8) & 0xFF, min_match)
            if got is not None and got[0] == (address & 0xFF):
                hit = ChecksumHypothesis(index, "e2e_p11", got[1], frames, data_id=address)
        if hit is None:
            got = _best_e2e(blob, sample, index, 0x00, min_match)
            if got is not None:
                hit = ChecksumHypothesis(index, "e2e_p11", got[1], frames, data_id=got[0])
        if hit is not None:
            found.append(hit)
        elif counter is not None:
            hit = _e2e_alt(blob, index, counter, min_match)
            if hit is not None:
                found.append(hit)
    return found


def p01_low_id(data_id: int) -> int:
    """The Profile 1 LOW-mode Data ID equivalent to an `e2e_p11` one.

    [low, 0x00] and [low'] leave the register in the same state for exactly
    one low'; this finds it. The two readings are the same wire behaviour.
    """
    state = int(J1850_TABLE[J1850_TABLE[data_id & 0xFF] ^ 0])
    return int(np.flatnonzero(J1850_TABLE == state)[0])


def _e2e_alt(
    blob: np.ndarray, index: int, counter: tuple[int, int], min_match: float
) -> ChecksumHypothesis | None:
    """Profile 1 ALT: low byte on even counters, high byte on odd ones."""
    from ..analyze.bits import BitOrder, bit_matrix_from_bytes
    from .counters import field_values

    bits = bit_matrix_from_bytes(blob, BitOrder.INTEL)
    parity = field_values(bits, counter[0], counter[1]) & 1
    ids = []
    for wanted in (0, 1):
        rows = np.flatnonzero(parity == wanted)
        if rows.size < 4:
            return None
        rate = (_e2e_candidates(blob[rows], index, None) == blob[None, rows, index]).mean(axis=1)
        if rate.max() < min_match:
            return None
        ids.append(int(rate.argmax()))
    low, high = ids
    total = (_e2e_candidates(blob, index, None)[np.where(parity == 0, low, high), np.arange(blob.shape[0])]
             == blob[:, index]).mean()
    return ChecksumHypothesis(
        index, "e2e_p01_alt", float(total), blob.shape[0], data_id=low | (high << 8)
    )
