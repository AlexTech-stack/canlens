# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""One `decompress` and one `ZstdError` regardless of interpreter version.

Python 3.14 ships zstd in the stdlib (PEP 784). Below that the third-party
`zstandard` package is pulled in by the `decode` extra. The two APIs differ
in one way that matters: `zstandard.decompress` refuses a frame whose header
omits the content size, which openpilot's writer does not always set, so the
fallback streams instead of asking for the size up front.

Both backends are wrapped in a function of this module's own signature rather
than re-exported, so callers (and type checkers) see one shape either way.
"""
from __future__ import annotations

import io
from collections.abc import Callable
from typing import Any

STDLIB: bool
ZstdError: type[Exception]
_backend_decompress: Callable[[bytes], bytes]


def _load_stdlib() -> tuple[type[Exception], Callable[[bytes], bytes]]:
    from compression.zstd import ZstdError as Error
    from compression.zstd import decompress as raw

    def call(data: bytes) -> bytes:
        return raw(data)

    return Error, call


def _load_zstandard() -> tuple[type[Exception], Callable[[bytes], bytes]]:
    try:
        import zstandard
    except ImportError as exc:  # pragma: no cover - only below 3.14
        raise ImportError(
            "Python < 3.14 needs the zstandard package: pip install 'canlens[decode]'"
        ) from exc

    def call(data: bytes) -> bytes:
        # Streamed on purpose: zstandard.decompress() raises when the frame
        # header carries no content size, which openpilot's writer omits.
        with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(data)) as reader:
            out: bytes = reader.read()
        return out

    error: Any = zstandard.ZstdError
    return error, call


try:  # Python >= 3.14
    ZstdError, _backend_decompress = _load_stdlib()
    STDLIB = True
except ImportError:  # pragma: no cover - exercised only below 3.14
    ZstdError, _backend_decompress = _load_zstandard()
    STDLIB = False


def decompress(data: bytes) -> bytes:
    """Decompress a zstd frame, raising `ZstdError` on malformed input."""
    return _backend_decompress(data)


__all__ = ["STDLIB", "ZstdError", "decompress"]
