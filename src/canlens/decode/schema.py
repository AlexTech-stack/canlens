# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Fetching and loading the cereal capnp schemas.

The schemas are not distributed on any package index, so they cannot be a pip
dependency -- they are fetched on demand and cached next to the corpus, like
the corpus data itself.

Two traps are encoded here. First, `log.capnp` lives on openpilot's
`release3` branch, not `master`. Second, openpilot's `cereal/car.capnp` is a
git symlink into the opendbc submodule, so fetching it from openpilot yields
the 37-byte link text rather than a schema; it has to come from opendbc
directly.

Both refs are pinned to a commit rather than a branch. A moving branch would
mean the same segment could decode differently next week, which is exactly the
kind of silent drift this project exists to eliminate.
"""
from __future__ import annotations

import os
import urllib.request
from typing import Any

# Pinned deliberately -- see module docstring.
OPENPILOT_REF = "551d088497144425feb14299c6d725b63f1b942e"
OPENDBC_REF = "3e92d112129507debe45364891954db70238997a"

_RAW = "https://raw.githubusercontent.com"

# Local filename -> upstream URL.
SCHEMA_FILES: dict[str, str] = {
    "log.capnp": f"{_RAW}/commaai/openpilot/{OPENPILOT_REF}/cereal/log.capnp",
    "legacy.capnp": f"{_RAW}/commaai/openpilot/{OPENPILOT_REF}/cereal/legacy.capnp",
    "custom.capnp": f"{_RAW}/commaai/openpilot/{OPENPILOT_REF}/cereal/custom.capnp",
    "include/c++.capnp": f"{_RAW}/commaai/openpilot/{OPENPILOT_REF}/cereal/include/c++.capnp",
    # NOT from openpilot: a raw fetch there returns the symlink target text.
    "car.capnp": f"{_RAW}/commaai/opendbc/{OPENDBC_REF}/opendbc/car/car.capnp",
}

# A schema that came back as a symlink target is a few dozen bytes of path and
# will fail to compile with a confusing capnp error much later. Catch it here.
_MIN_SCHEMA_BYTES = 256

_loaded: Any = None


def schema_dir(root: str) -> str:
    return os.path.join(root, "schema")


def ensure_schemas(root: str, *, refresh: bool = False) -> str:
    """Download the schema set into `<root>/schema` if not already there."""
    target = schema_dir(root)
    os.makedirs(os.path.join(target, "include"), exist_ok=True)
    for name, url in SCHEMA_FILES.items():
        dest = os.path.join(target, name)
        if not refresh and os.path.exists(dest) and os.path.getsize(dest) >= _MIN_SCHEMA_BYTES:
            continue
        with urllib.request.urlopen(url, timeout=60) as response:
            body = response.read()
        if len(body) < _MIN_SCHEMA_BYTES:
            raise RuntimeError(
                f"{name} came back as {len(body)} bytes from {url} -- "
                "this is what a git symlink looks like over raw.githubusercontent"
            )
        with open(dest, "wb") as handle:
            handle.write(body)
    return target


def load_schema(root: str) -> Any:
    """Return the compiled `log.capnp` module, fetching the schemas if needed.

    Compilation is process-global and cached: capnp registers schema node IDs
    in a global table, and loading the same file twice in one process is both
    wasted work and a source of duplicate-ID errors.
    """
    global _loaded
    if _loaded is not None:
        return _loaded
    try:
        import capnp
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ImportError(
            "decoding needs pycapnp: pip install 'canlens[decode]'"
        ) from exc
    target = ensure_schemas(root)
    capnp.remove_import_hook()
    _loaded = capnp.load(os.path.join(target, "log.capnp"), imports=[target])
    return _loaded
