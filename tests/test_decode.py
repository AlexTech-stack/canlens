# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Decoder unit tests -- no network, no corpus data."""
from __future__ import annotations

import re

import pytest

from canlens.decode import classify_src
from canlens.decode.rlog import ECHO_FLAG, CanFrame
from canlens.decode.schema import OPENDBC_REF, OPENPILOT_REF, SCHEMA_FILES
from canlens.decode.zstd import ZstdError, decompress


class TestClassifySrc:
    def test_plain_bus_is_not_an_echo(self):
        assert classify_src(0) == (0, False)
        assert classify_src(2) == (2, False)

    def test_high_src_is_an_echo_of_the_offset_bus(self):
        # Bus 0 traffic relayed onto bus 2 comes back as src 130.
        assert classify_src(130) == (2, True)
        assert classify_src(ECHO_FLAG) == (0, True)

    @pytest.mark.parametrize("src", [0, 1, 2, 127, 128, 130, 255])
    def test_bus_is_always_below_the_flag(self, src):
        bus, _ = classify_src(src)
        assert 0 <= bus < ECHO_FLAG


class TestCanFrame:
    def test_len_is_the_payload_length(self):
        assert len(CanFrame(1, 0, 0x123, b"\xde\xad\xbe\xef", False)) == 4

    def test_is_hashable_and_immutable(self):
        frame = CanFrame(1, 0, 0x123, b"\x00", False)
        assert {frame, frame} == {frame}
        with pytest.raises(AttributeError):
            frame.address = 0x124  # type: ignore[misc]


class TestSchemaPins:
    def test_refs_are_commit_shas_not_branches(self):
        # A moving branch would let the same segment decode differently later.
        for ref in (OPENPILOT_REF, OPENDBC_REF):
            assert re.fullmatch(r"[0-9a-f]{40}", ref), ref

    def test_car_capnp_comes_from_opendbc(self):
        # In openpilot it is a git symlink; a raw fetch there returns link text.
        assert "opendbc" in SCHEMA_FILES["car.capnp"]
        assert "openpilot" not in SCHEMA_FILES["car.capnp"]

    def test_every_other_schema_comes_from_openpilot(self):
        for name, url in SCHEMA_FILES.items():
            if name != "car.capnp":
                assert "commaai/openpilot" in url, name

    def test_pinned_ref_appears_in_every_url(self):
        for name, url in SCHEMA_FILES.items():
            expected = OPENDBC_REF if name == "car.capnp" else OPENPILOT_REF
            assert expected in url, name


class TestZstd:
    def test_round_trip(self):
        import compression.zstd as stdlib_zstd

        payload = b"can frames" * 1000
        assert decompress(stdlib_zstd.compress(payload)) == payload

    def test_rejects_garbage(self):
        with pytest.raises(ZstdError):
            decompress(b"not a zstd frame")
