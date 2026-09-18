# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""AUTOSAR E2E profile detectors, gated by what a message could carry.

Each profile has a fixed header the payload must be able to hold, so a
Profile 6 is never tried on an 8-byte frame and a Profile 7 only on frames of
twenty bytes or more. Where a profile carries an explicit Length, that field
is read first and must be consistent with the payload before any CRC is
computed -- it is the cheapest and most decisive check the spec provides, and
it also bounds the CRC correctly on padded CAN FD frames, where the protected
data ends at Length and padding follows.

Constructions follow AUTOSAR_PRS_E2EProtocol FO R19-11:

Profile 22 / 2  CRC-8 0x2F (Crc_CalculateCRC8H2F, start 0xFF, XOR 0xFF per
                the Profile 2 table). CRC over the whole header except the
                CRC byte and the user data, *then* the Data ID appended
                [PRS_E2E_00527] -- selected from a 16-entry list by the 4-bit
                counter [PRS_E2E_00120]. Counter 0..15, no reserved value.
Profile 4       Length(2) Counter(2) DataID(4) CRC(4), big-endian, CRC-32P4
                over Data[0 .. o+8) and Data[o+12 .. Length). Data ID explicit.
Profile 7       CRC(8) Length(4) Counter(4) DataID(4), big-endian, CRC-64 over
                Data[0 .. o) and Data[o+8 .. Length). Data ID explicit.

Profiles 1/11 and 5 live in checksums.py and crc16.py; Profile 6 is added
alongside 5 since it shares its CRC. This module only knows which of them
apply to a given width.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .checksums import ChecksumHypothesis, _crc8_table, enough_evidence
from .crc16 import Crc16Hypothesis
from .crcwide import CRC32P4, CRC64, WideCrc, read_be

H2F_TABLE = _crc8_table(0x2F)
H2F_INIT, H2F_XOR = 0xFF, 0xFF


# ---------------------------------------------------------------------------
# Profile 22 (and 2): CRC-8 H2F, Data ID from a list indexed by the counter.

def p22_crc(payload: bytes, crc_index: int, data_id: int) -> int:
    """Scalar reference: header and data except the CRC byte, then the ID."""
    crc = H2F_INIT
    for byte in payload[:crc_index] + payload[crc_index + 1 :] + bytes([data_id & 0xFF]):
        crc = int(H2F_TABLE[crc ^ byte])
    return crc ^ H2F_XOR


def _h2f_states(matrix: np.ndarray, crc_index: int) -> np.ndarray:
    """Register after the data, per frame, before the Data ID is appended."""
    crc = np.full(matrix.shape[0], H2F_INIT, dtype=np.uint8)
    for column in range(matrix.shape[1]):
        if column == crc_index:
            continue
        crc = H2F_TABLE[crc ^ matrix[:, column]]
    return crc


def find_p22(
    matrix: np.ndarray,
    counters: Sequence,
    *,
    candidate_bytes: Sequence[int] | None = None,
    min_match: float = 0.99,
    min_per_value: int = 4,
) -> list[ChecksumHypothesis]:
    """Profile 22: one solved Data ID per counter value, all sixteen required.

    The register after the data is computed once per frame; appending a
    candidate ID is then a single table step, so all 256 candidates are scored
    for every frame at once. Frames are grouped by their counter value and each
    group must be reproduced by one ID -- sixteen independent 256-way fits,
    which is a far stronger claim than a single one.

    The recovered IDs are relative to the 0xFF/0xFF convention, as Profile
    11's are to its convention. It was tempting to think the ID coming *last*
    made a final XOR testable; it does not. The CRC table is linear over
    GF(2), so T[s ^ id] ^ X == T[s ^ id'] for the constant id' = id ^ T^-1[X]:
    a constant XOR on the CRC is absorbed into every list entry uniformly.
    Only structure across messages -- or the list being known -- can settle it.
    """
    frames, width = matrix.shape
    if width < 2 or frames < 16 * min_per_value:
        return []
    alive = [c for c in counters if c.length == 4 and c.start_bit % 4 == 0]
    if not alive:
        return []
    positions = candidate_bytes if candidate_bytes is not None else range(width)
    from ..analyze.bits import BitOrder, bit_matrix_from_bytes
    from .counters import field_values

    bits = bit_matrix_from_bytes(matrix, BitOrder.INTEL)
    found = []
    for counter in alive:
        value = field_values(bits, counter.start_bit, 4)
        for index in positions:
            if not 0 <= index < width or not enough_evidence(matrix, [index], 16):
                continue
            states = _h2f_states(matrix, index)
            stored = matrix[:, index]
            ids: list[int] | None = []
            total_ok = 0
            for v in range(16):
                rows = np.flatnonzero(value == v)
                if rows.size < min_per_value:
                    ids = None
                    break
                # (256, n_v): every candidate ID appended to each frame's state
                after = H2F_TABLE[states[rows][None, :] ^ np.arange(256, dtype=np.uint8)[:, None]]
                rate = ((after ^ np.uint8(H2F_XOR)) == stored[rows][None, :]).mean(axis=1)
                best = int(rate.argmax())
                if rate[best] < min_match:
                    ids = None
                    break
                assert ids is not None
                ids.append(best)
                total_ok += int(rate[best] * rows.size)
            if ids is None:
                continue
            found.append(
                ChecksumHypothesis(
                    index, "e2e_p22", total_ok / frames, frames,
                    # the sixteen IDs packed low-to-high by counter value
                    data_id=int.from_bytes(bytes(ids), "little"),
                )
            )
            break  # one CRC byte per counter is enough
    return found


def p22_id_list(data_id: int) -> list[int]:
    """Unpack the 16-entry Data ID list a Profile 22 hypothesis carries."""
    return list(data_id.to_bytes(16, "little"))


# ---------------------------------------------------------------------------
# Profile 6: CRC(2) Length(2) Counter(1), big-endian, Data ID appended high
# byte then low byte -- the reverse of Profile 5 -- over Data[0..o) and
# Data[o+2..Length). Length is the total including the header [6.7.4].

P06_HEADER = 5


def p06_crc(payload: bytes, offset: int, data_id: int) -> int:
    """Scalar reference for the Profile 6 CRC."""
    from .crc16 import crc16_update

    length = int.from_bytes(payload[offset + 2 : offset + 4], "big")
    covered = payload[:offset] + payload[offset + 2 : length]
    crc = crc16_update(0xFFFF, covered)
    return crc16_update(crc, bytes([(data_id >> 8) & 0xFF, data_id & 0xFF]))


def find_p06(
    matrix: np.ndarray,
    *,
    min_match: float = 0.99,
    offsets: Sequence[int] | None = None,
    candidate_bytes: Sequence[int] | None = None,
) -> list[Crc16Hypothesis]:
    """Profile 6 at any header offset the Length field makes consistent.

    Gated on Length first, like the wide profiles. The Data ID is solved as
    for Profile 5 -- the two appended bytes are pinned by the stored CRC and
    intersected across frames -- but read back high byte first.
    """
    from .crc16 import solve_data_id, v_append, v_crc16_autosar, v_read_crc

    frames, width = matrix.shape
    if width < P06_HEADER or frames < 8:
        return []
    found = []
    noisy = None if candidate_bytes is None else set(candidate_bytes)
    for offset in offsets if offsets is not None else range(width - P06_HEADER + 1):
        # The CRC bytes must move like noise, and the data must constrain the
        # two secret bytes, before the Length field is even read.
        if noisy is not None and not {offset, offset + 1} <= noisy:
            continue
        if not enough_evidence(matrix, range(offset, offset + 2), 2):
            continue
        length = read_be(matrix, offset + 2, 2)
        plausible = (length >= offset + P06_HEADER) & (length <= width)
        if plausible.mean() < min_match:
            continue
        fixed = int(np.bincount(length.astype(np.int64)).argmax())
        rows = np.flatnonzero(length == fixed)
        if rows.size < min_match * frames:
            continue
        trimmed = matrix[rows, :fixed]
        stored = v_read_crc(trimmed, offset, "big")
        # Profile 5's solver appends [b0, b1] and returns b0 | b1 << 8; here
        # b0 is the Data ID's high byte, so the reading is byte-swapped.
        candidates: set[int] | None = None
        for row in trimmed[:4]:
            sol = solve_data_id(bytes(row), offset, "big")
            candidates = sol if candidates is None else candidates & sol
            if not candidates:
                break
        if not candidates:
            continue
        base = v_crc16_autosar(trimmed, (offset, offset + 2))
        for packed in sorted(candidates):
            high, low = packed & 0xFF, (packed >> 8) & 0xFF
            computed = v_append(v_append(base, high), low)
            rate = float((computed == stored).mean()) * rows.size / frames
            if rate >= min_match:
                found.append(
                    Crc16Hypothesis(offset, "e2e_p06", "big", rate, frames, (high << 8) | low)
                )
                break
    return found


# ---------------------------------------------------------------------------
# Profiles 4 and 7: explicit header with Length, Counter, Data ID and a wide CRC.

@dataclass(frozen=True)
class WideProfile:
    name: str
    header: int  # total header bytes
    length_at: int  # offsets within the header
    length_bytes: int
    counter_at: int
    counter_bytes: int
    data_id_at: int
    crc_at: int
    crc_bytes: int
    crc: WideCrc


P04 = WideProfile("e2e_p04", 12, 0, 2, 2, 2, 4, 8, 4, CRC32P4)
P07 = WideProfile("e2e_p07", 20, 8, 4, 12, 4, 16, 0, 8, CRC64)


def wide_crc(profile: WideProfile, payload: bytes, offset: int = 0) -> int:
    """Scalar reference for a Profile 4/7 CRC over a payload with the header at `offset`.

    The protected region ends at the Length field, not at the frame's end.
    """
    length = int.from_bytes(
        payload[offset + profile.length_at : offset + profile.length_at + profile.length_bytes], "big"
    )
    crc_start = offset + profile.crc_at
    covered = payload[:crc_start] + payload[crc_start + profile.crc_bytes : length]
    return profile.crc.compute(covered)


def find_wide(
    profile: WideProfile,
    matrix: np.ndarray,
    *,
    min_match: float = 0.99,
    offsets: Sequence[int] | None = None,
    candidate_bytes: Sequence[int] | None = None,
) -> list[Crc16Hypothesis]:
    """Profile 4 or 7 at any header offset that the Length field makes consistent.

    Gate first, compute last: the header must fit, and the big-endian Length
    must be at least the header and at most the payload width on (nearly)
    every frame. Only then is the wide CRC computed, over Data[0 .. crc) and
    Data[crc+n .. Length). Padding beyond Length -- normal on CAN FD -- is
    outside the CRC by construction.
    """
    frames, width = matrix.shape
    if width < profile.header or frames < 8:
        return []
    found = []
    noisy = None if candidate_bytes is None else set(candidate_bytes)
    for offset in offsets if offsets is not None else range(width - profile.header + 1):
        crc_columns = range(offset + profile.crc_at, offset + profile.crc_at + profile.crc_bytes)
        if noisy is not None and not set(crc_columns) <= noisy:
            continue
        if not enough_evidence(matrix, crc_columns, 0):
            continue
        length = read_be(matrix, offset + profile.length_at, profile.length_bytes)
        plausible = (length >= offset + profile.header) & (length <= width)
        if plausible.mean() < min_match:
            continue
        # A variable Length would need a per-frame CRC bound; the common case
        # is a fixed one, and the spec allows either. Only the fixed case is
        # claimed here, which is the honest scope of the check.
        fixed = int(np.bincount(length.astype(np.int64)).argmax())
        rows = np.flatnonzero(length == fixed)
        if rows.size < min_match * frames:
            continue
        crc_start = offset + profile.crc_at
        trimmed = matrix[rows, :fixed]
        computed = profile.crc.columns(trimmed, skip=(crc_start, crc_start + profile.crc_bytes))
        stored = read_be(matrix[rows], crc_start, profile.crc_bytes)
        rate = float((computed.astype(np.uint64) == stored).mean()) * rows.size / frames
        if rate < min_match:
            continue
        ident = read_be(matrix[rows], offset + profile.data_id_at, 4)
        data_id = int(np.bincount(ident.astype(np.int64)).argmax()) if ident.size else None
        found.append(
            Crc16Hypothesis(
                crc_start, profile.name, "big", rate, frames, data_id, nbytes=profile.crc_bytes
            )
        )
    return found


# ---------------------------------------------------------------------------
# Applicability.

@dataclass(frozen=True)
class Profile:
    name: str
    min_width: int
    run: Callable[..., list]


def applicable(width: int, profiles: Sequence[Profile]) -> list[Profile]:
    """The profiles a payload of `width` bytes could carry at all."""
    return [p for p in profiles if width >= p.min_width]


# Minimum payload each profile needs for its header. Profiles 1/11 and 5 are
# listed for completeness -- their detectors run from checksums.py and
# crc16.py -- so this table is the single statement of what applies where.
HEADER_BYTES = {
    "e2e_p11": 2,   # CRC(1) + counter nibble
    "e2e_p22": 2,   # CRC(1) + counter nibble
    "e2e_p05": 3,   # CRC(2) + counter(1)
    "e2e_p06": 5,   # CRC(2) + Length(2) + counter(1)
    "e2e_p04": 12,  # Length(2) + Counter(2) + DataID(4) + CRC(4)
    "e2e_p07": 20,  # CRC(8) + Length(4) + Counter(4) + DataID(4)
}

PROFILES: list[Profile] = [
    Profile("e2e_p22", HEADER_BYTES["e2e_p22"], find_p22),
    Profile("e2e_p06", HEADER_BYTES["e2e_p06"], find_p06),
    Profile("e2e_p04", HEADER_BYTES["e2e_p04"], lambda m, *_, **kw: find_wide(P04, m, **kw)),
    Profile("e2e_p07", HEADER_BYTES["e2e_p07"], lambda m, *_, **kw: find_wide(P07, m, **kw)),
]


def run_profiles(
    matrix: np.ndarray, counters: Sequence, *, candidate_bytes: Sequence[int] | None = None,
    min_match: float = 0.99,
) -> tuple[list[ChecksumHypothesis], list[Crc16Hypothesis]]:
    """Run every profile the payload width admits. Returns (8-bit, wider)."""
    eight: list[ChecksumHypothesis] = []
    wider: list[Crc16Hypothesis] = []
    for profile in applicable(matrix.shape[1], PROFILES):
        if profile.name == "e2e_p22":
            eight += find_p22(matrix, counters, candidate_bytes=candidate_bytes, min_match=min_match)
        elif profile.name == "e2e_p06":
            wider += find_p06(matrix, min_match=min_match, candidate_bytes=candidate_bytes)
        else:
            wider += profile.run(matrix, min_match=min_match, candidate_bytes=candidate_bytes)
    return eight, wider
