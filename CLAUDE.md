# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

> **Sibling file: `AGENTS.md`.** The two are split by *audience*, not by depth: Claude Code
> loads this file, every other agent or model (DeepSeek flash v1, and anything else following
> the AGENTS.md convention) loads `AGENTS.md`. They describe the same repository and must
> never disagree. **Any change you make here to a fact about the codebase has to land in
> `AGENTS.md` too** — otherwise the other tools keep acting on the stale version and nobody
> notices, because neither file's readers see the other. Neither is authoritative over the
> other; **the source code is authoritative over both.**

## What this is

`canlens` reverse-engineers automotive CAN bus traces **at corpus scale**. Most tools analyse
one trace; a checksum hypothesis from a single recording is a guess with a confidence number
attached, and nothing can check it. canlens is built around comma.ai's public commaCarSegments
corpus — 230 vehicle platforms, 188,883 segments — so a hypothesis can be *corroborated* across
thousands of segments from hundreds of independent drivers.

The pipeline is `corpus → decode → analyze → infer → corroborate`, with a PySide6 workbench and
a BoAt PDU-database exporter hanging off the end. `HOWTO.md` is the end-to-end walkthrough and
the closest thing to a design document; read the relevant section before working in a layer.

> **Status: pre-alpha.** Every layer works: `corpus`, `decode`, `analyze`, `infer`,
> `corroborate`, `truth`, `gui` and `export`.

## Environment

Python **3.14.4** in `.venv` at the repo root. The system Python is PEP 668 externally-managed,
so always go through the venv rather than a bare `python3`/`pip`.

```bash
./.venv/bin/python -m canlens --help
./.venv/bin/canlens corpus status
```

Installed: `numpy`, `pycapnp`, `cantools`, `PySide6`, `pyqtgraph`, `jsonschema`, and the dev
tools. **Not** installed: `pyarrow`, `python-can`, `zstandard` — the `store` and `traces`
extras. On 3.14 zstd comes from the stdlib (`compression.zstd`, PEP 784), so the
third-party package is correctly absent. Tests needing an absent extra skip cleanly; keep it
that way.

The corpus root defaults to `~/data/canlens`, overridable with `--root` or `$CANLENS_DATA`. It
holds `database.json`, `segments/` and `cache/`. It is gitignored and **must stay that way** —
the upstream bucket is ~299 GB.

## The gate

Run all three before claiming anything is done:

```bash
./.venv/bin/ruff check . && ./.venv/bin/mypy src && ./.venv/bin/pytest -q
```

684 tests, about 14 seconds, no corpus data required — the suite runs on synthetic payloads
with known ground truth.

**Never pipe a gate command into `tail`/`head` without checking `PIPESTATUS`.** Doing so masked
a real failure three separate times in this project's history, because the pipeline exits with
the status of `tail`, which always succeeds:

```bash
# WRONG — reports success no matter what pytest did
./.venv/bin/pytest -q | tail -1

# Right
./.venv/bin/pytest -q 2>&1 | tail -1; test "${PIPESTATUS[0]}" -eq 0
```

`mypy` takes no `-q` flag. `ruff check --fix` is fine for import sorting.

## Layout

| package | role |
|---|---|
| `corpus/` | manifest, selective fetch, local accounting. **Deliberately stdlib-only** |
| `decode/` | `rlog.zst` → columnar `FrameSet`, plus the npz frame cache |
| `analyze/` | per-bit entropy/rate/classification and inter-arrival timing |
| `infer/` | counters, checksums, CRC-16/32/64, every AUTOSAR E2E profile, multiplexors |
| `corroborate/` | device-weighted agreement; pooled evidence for secrets one segment cannot check |
| `gui/` | PySide6 workbench: Data, Heat Map and Corroborate screens |
| `export/` | BoAt PDU-database JSON |
| `truth/` | opendbc DBCs as a reference, and precision/recall against them |
| `cli.py` | the one `canlens` entry point; `render.py` draws the terminal output |

CLI verbs: `corpus {list,plan,fetch,status,delete,which}`, `decode {schema,summary}`,
`analyze trace`, `infer {trace,message}`, `corroborate`, `corroborate-pooled`, `truth {dbc,score}`,
`export pdu-db`, `cache {build,clear,status}`, `gui`.

## Hard invariants

Violating any of these breaks something *silently*. They are the reason this section exists.

**1. Bump the cache version when behaviour changes.** `INFER_VERSION` in `infer/results.py`
must be incremented on any change to a detector, a threshold, or what `_build()` returns.
`CACHE_VERSION` in `decode/cache.py` likewise when the stored columns change meaning. A stale
entry is not slower, **it is wrong** — it returns yesterday's answer for today's code. After
bumping, rebuild before re-surveying the corpus:

```bash
./.venv/bin/python -c "from canlens.infer import clear_results; print(clear_results('$HOME/data/canlens'))"
./.venv/bin/canlens --root ~/data/canlens cache build
```

**2. A hypothesis is reported only if it was checked.** Nothing in `infer/` may claim a field
because it *looks* like one — that is what `analyze/` already said, and it is exactly the guess
this layer replaces. Every detector verifies against the trace and carries its match rate.
Counters need 0.95, everything else 0.99.

**3. Detector order in `_build()` is load-bearing.** `infer/message.py` runs candidate bytes →
counters → plain checksums → Honda's nibble → E2E 1/11 → width-gated E2E 22/6/4/7 → CRC-16
and E2E 5 → multiplexor → two counter cleanups. Each stage receives only the byte positions nothing simpler
has explained. Reordering changes results; a plain sum would get reported as an exotic CRC.

**4. Echoes are not data.** `CanData.src >= 128` marks a frame the device itself put on the
wire, on bus `src - 128`. They are 30–46% of a segment and reproduce the source bus's address
set exactly. They are dropped by default; only `decode.summarize()` counts them on purpose.
Including them inflates every frequency, entropy and corroboration statistic downstream.

**5. Intel bit order is the default, and a bit index is already a DBC `StartPos`.** No
translation happens on export, by design. Motorola exists as a parameter and is never searched.

**6. Stay columnar.** `FrameSet` holds the trace as arrays plus one flat payload blob.
Rebuilding `CanFrame` objects costs ~0.7 s per segment and would cap a 95× cache at about 2×.
Use `bytes_matrix()` / `payload_matrix()`; never materialise objects in a hot loop.

**7. Corpus data and caches are never committed.** `/data/`, `*.zst` and `database.json` are
gitignored. The capnp schemas and opendbc DBCs are fetched on demand and stay unvendored too.

**8. Evidence gates protect solved-for secrets.** Any detector that solves for a hidden Data ID
must first pass `enough_evidence()`: distinct payload contents ≥ secret bytes + 8. Without it,
a static message with an alive counter offers exactly sixteen distinct contents — the size of a
Profile 22 ID list — and the profile fits trivially. Adding the gate removed Profile 22 from 136
platforms and Profile 6 from 139.

## Conventions

**SPDX header on every new source file**, after any shebang, matching the surrounding files.
All 42 source files and all 27 test files carry it.

```python
# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT
```

**Comments record why, with numbers.** This is the most distinctive thing about the codebase and
the thing most worth preserving. A comment here explains the reasoning, cites the measurement
that settled it, and says what was tried and rejected. `infer/counters.py` records that a 0.40
LSB cutoff admitted 2729 start positions where 0.95 admits 193 and found the same 163 counters
four times slower. `decode/frameset.py` records why the columnar shape exists at all. Match
that register: if you tune a threshold, say what you measured.

**Docstrings state the claim being made**, not the mechanics. Detector docstrings say what would
falsify the hypothesis.

**Commit messages**: `type(scope): summary` (`feat`, `fix`, `perf`, `docs`, `refactor`), body in
prose explaining *why* and what was verified, wrapped at ~72 characters. End with:

```
Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

Commit or push only when asked. The default branch is `master`; the remote is
`git@github.com:AlexTech-stack/canlens.git`.

## Testing

Tests live in `tests/`, one file per module, named after the behaviour being asserted rather
than the function under test. Prefer synthetic payloads with known ground truth over fixtures.

- **GUI tests** set `QT_QPA_PLATFORM=offscreen` and `pytest.importorskip` PySide6 and pyqtgraph
  before importing anything from `canlens.gui`. Keep Qt out of `gui/model.py` so the data model
  stays testable without a display.
- **Corpus tests** (`test_decode_integration.py`) skip unless a segment has actually been
  fetched. They are what prove the schema pins and echo handling work on real logs.
- **Export tests** validate against BoAt's real `pdu_db.schema.json` at
  `/home/testuser/BoAt/boat-platform/config/pdu_db.schema.json` and skip when absent.
- `test_columnar_equivalence.py` holds the object path and the columnar path to identical
  results. If you add a detector to `_build()`, both paths must agree.
- Vectorised detectors are held to their scalar reference implementations by
  `TestVectorisedMatchesScalar` in `tests/test_infer.py`. Keep the scalar version as the readable spec.

## Gotchas found the hard way

- **pyqtgraph auto-levels render a uniform image black.** A fully static payload is uniform, so
  every `setImage` call must pass `levels=(0, 255)`. `test_gui_render.py` asserts the count of
  those call sites, so adding an image view means updating that number.
- **`setMaximumHeight` does not reserve space** — the layout handed back 68px of a requested 96
  and the label lanes collapsed. Use `setFixedHeight`.
- **A CRC is linear over GF(2).** When the only moving content of a message is its alive counter,
  the CRC byte is an affine image of that counter and two of its bits walk 0..3 exactly like a
  real 2-bit counter. This produced phantom counters on 404 messages corpus-wide before
  `outside_checksums()` existed. Expect this class of bug wherever one field is a function of
  another.
- **A selector that cycles 0, 1, 2 is numerically a counter modulo 3**, and every byte it
  multiplexes cycles with it. `outside_multiplex()` drops counters wholly contained in the
  selector or its dependent bits, but keeps one that merely has its lowest bit locked to a
  two-frame schedule.
- **The AUTOSAR E2E Profile 1/11 final XOR cannot be read off the spec.** For a fixed-length
  message a final XOR is absorbed into an equivalent start state, so only structure across
  messages of different lengths settles it. The convention in the code (register from 0x00, no
  final XOR, Data ID fed as `[addr & 0xFF, addr >> 8]`) recovers the CAN identifier as the Data
  ID on all 94 such messages of a Rivian; every other convention yields noise.
- **Reasoning about what a DBC "would name" is not evidence; scoring against one is.** The
  multiplexor detector reported the whole selector byte, justified by the argument that a byte
  is what a DBC names. Scored against opendbc it matched none of the declared selectors, because
  Volkswagen's `VIN_01_MUX` is two bits and Tesla's `VCFRONT_LVPowerStateIndex` is five. It now
  claims the span that actually moves. Run `canlens truth score` before trusting that kind of
  argument.
- **A DBC's big-endian start bit is not a sawtooth index.** cantools reports `Signal.start` in
  the DBC's own numbering, which is already canlens' flat Intel index. A little-endian signal
  runs upward from it; a big-endian one runs *downward* and wraps to bit 7 of the next byte.
  Treating it as an MSB0 sawtooth and converting disagreed with cantools' own decoder on 895 of
  1070 signals. The rule in `truth/dbc.py` agrees on all 2925 signals of the 58 DBCs in opendbc.
- **6 physical cores, 12 logical.** `default_jobs()` returns physical cores on purpose; the
  second thread of a core adds nothing to numpy-bound work.

## Not implemented

Say so plainly rather than implying otherwise:

- **Signal boundaries.** Nothing groups adjacent bits into a field unless a detector claims them.
- **Scaling and units.** No factor, offset or unit is ever inferred.
- **Signals inside a multiplexed layout.** The selector is found and its layouts are drawn, but
  no detector runs separately within each selector value.
- **Motorola search.** Bit order is a parameter, not something searched.
- **Variable-length E2E.** Profiles 4 and 7 are claimed only where Length is fixed across the
  trace.

## Related

- [opendbc](https://github.com/commaai/opendbc) — CAN database definitions, and the reference
  `truth/` scores against. A local clone may exist at `/home/testuser/BoAt/tools/dbc/opendbc`.
  There is no automatic platform-to-DBC mapping; `canlens truth score` takes `--dbc` explicitly.
- [openpilot](https://github.com/commaai/openpilot) — `LogReader`, cabana, and the `cereal`
  schemas `decode/` pins.
- **BoAt** — the deterministic SIL/HIL simulation platform this grew out of, and the consumer of
  `export/`. It has its own `CLAUDE.md`/`AGENTS.md` pair following the same convention. AUTOSAR
  specifications live under `/home/testuser/BoAt/spec/` (gitignored, populated per machine):
  `spec/text/*.txt` flat UTF-8, `spec/search.db` SQLite FTS5. `AUTOSAR_PRS_E2EProtocol.txt` is
  the reference every E2E detector here was built from.
