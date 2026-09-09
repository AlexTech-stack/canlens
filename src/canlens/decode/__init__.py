# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Raw log formats in, normalised CAN records out.

The corpus ships openpilot `rlog.zst`: zstandard-compressed capnp
(`cereal`) message streams, not `.asc`/`.blf`. Decoding one needs the
cereal schema plus zstd -- see the `decode` extra. Everything downstream
consumes the normalised record type produced here, so adding `.blf`/`.asc`
readers later is purely additive.

Not yet implemented.
"""
