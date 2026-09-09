# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""One `decompress` and one `ZstdError` regardless of interpreter version.

Python 3.14 ships zstd in the stdlib (PEP 784). Below that the third-party
`zstandard` package is pulled in by the `decode` extra. The two APIs differ
in one way that matters: `zstandard.decompress` refuses a frame whose header
omits the content size, which openpilot's writer does not always set, so the
fallback streams instead of asking for the size up front.
"""
from __future__ import annotations

try:  # Python >= 3.14
    from compression.zstd import ZstdError
    from compression.zstd import decompress as _decompress

    STDLIB = True
except ImportError:  # pragma: no cover - exercised only below 3.14
    STDLIB = False

    try:
        from zstandard import ZstdError
    except ImportError:
        class ZstdError(Exception):  # type: ignore[no-redef]
            """Raised for malformed zstd input when no backend is installed."""

    def _decompress(data: bytes) -> bytes:
        try:
            import zstandard
        except ImportError as exc:
            raise ImportError(
                "Python < 3.14 needs the zstandard package: pip install 'canlens[decode]'"
            ) from exc
        import io

        with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(data)) as reader:
            return reader.read()


def decompress(data: bytes) -> bytes:
    """Decompress a zstd frame."""
    return _decompress(data)


__all__ = ["STDLIB", "ZstdError", "decompress"]
