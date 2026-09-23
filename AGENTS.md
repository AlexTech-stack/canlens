# AGENTS.md

Guidance for coding agents working in this repository.

> **Sibling file: `CLAUDE.md`.** The two are split by *audience*, not by depth: Claude Code
> loads `CLAUDE.md`, every other agent or model (DeepSeek flash v1, and anything else following
> the AGENTS.md convention) loads this file. They describe the same repository and must never
> disagree. **Any change you make here to a fact about the codebase has to land in `CLAUDE.md`
> too** — otherwise the other tools keep acting on the stale version and nobody notices, because
> neither file's readers see the other. Neither is authoritative over the other; **the source
> code is authoritative over both.** When this file and the code disagree, the code is right and
> this file is a bug.

---

## 1. What this project is

`canlens` reverse-engineers automotive CAN bus traces **at corpus scale**.

Most CAN tools analyse one trace at a time. That bounds what they can conclude: a checksum
hypothesis derived from a single recording is a guess with a confidence number attached, and
there is no way to check it. `canlens` is built around comma.ai's public commaCarSegments
corpus — 230 vehicle platforms, 188,883 segments, about 3,148 hours of driving — so a
hypothesis can be **corroborated** across thousands of segments recorded by hundreds of
independent drivers.

The pipeline runs in one direction:

```
corpus  →  decode  →  analyze  →  infer  →  corroborate
                                     ↓
                          gui  /  export (BoAt PDU database)
```

`HOWTO.md` in the repo root is the end-to-end walkthrough and the closest thing this project has
to a design document. **Read the section covering the layer you are about to touch before you
touch it.** It records why thresholds have the values they do.

**Status: pre-alpha.** Every layer works: `corpus`, `decode`, `analyze`, `infer`,
`corroborate`, `truth`, `gui` and `export`.

---

## 2. Environment and commands

### 2.1 Use the virtualenv

Python **3.14.4** lives in `.venv` at the repo root. The system Python is PEP 668
externally-managed, so a bare `pip install` fails and a bare `python3` may not see the package.
**Always call the venv binaries by path:**

```bash
./.venv/bin/python        # not python3
./.venv/bin/pip           # not pip
./.venv/bin/canlens       # the CLI entry point
./.venv/bin/pytest
./.venv/bin/ruff
./.venv/bin/mypy
```

### 2.2 What is installed

Present: `numpy`, `pycapnp`, `cantools`, `PySide6`, `pyqtgraph`, `jsonschema`, `pytest`,
`pytest-cov`, `ruff`, `mypy`.

Absent on purpose: `pyarrow` (`store` extra), `python-can` (`traces` extra), `zstandard` — on Python 3.14 zstd comes from the stdlib as
`compression.zstd` (PEP 784), so the third-party package is correctly not installed.

**Do not install the missing extras to make a test pass.** Tests that need an absent
dependency must skip cleanly via `pytest.importorskip`. That is the designed behaviour.

### 2.3 The corpus root

Segment data, the manifest and both caches live outside the repo, under a root that defaults to
`~/data/canlens` and is overridable with `--root` or the `$CANLENS_DATA` environment variable.
Layout: `database.json`, `segments/<device>/<route>/<index>/rlog.zst`, `cache/`.

It is gitignored and **must stay that way**. The upstream bucket is about 299 GB.

### 2.4 The gate — run all three before saying anything is done

```bash
./.venv/bin/ruff check . && ./.venv/bin/mypy src && ./.venv/bin/pytest -q
```

869 tests, about 17 seconds. **No corpus data is required** — the suite runs on synthetic
payloads with known ground truth.

### 2.5 The pipe trap — read this, it has bitten this project three times

Piping a gate command into `tail` or `head` makes the shell report the status of `tail`, which
always succeeds. Failures then look like passes.

```bash
# WRONG — reports success even when pytest fails
./.venv/bin/pytest -q | tail -1

# Right — check the first element of PIPESTATUS
./.venv/bin/pytest -q 2>&1 | tail -1; test "${PIPESTATUS[0]}" -eq 0
```

Also: `mypy` accepts no `-q` flag (it errors out). `ruff check --fix` is safe for import
ordering.

### 2.6 Running the tool

```bash
./.venv/bin/canlens corpus list                      # 230 platforms with counts and sizes
./.venv/bin/canlens corpus plan TOYOTA_PRIUS         # size it before fetching
./.venv/bin/canlens corpus fetch KIA_EV6 --limit 50  # resumable; re-run to continue
./.venv/bin/canlens corpus status                    # what is local
./.venv/bin/canlens corpus which <segment path>      # which car is this trace?
./.venv/bin/canlens analyze trace <segment>/rlog.zst
./.venv/bin/canlens infer trace <segment>/rlog.zst
./.venv/bin/canlens infer message <segment>/rlog.zst --address 0x210 --bus 1
./.venv/bin/canlens corroborate KIA_EV6
./.venv/bin/canlens corroborate-pooled VOLKSWAGEN_GOLF_MK7  # needs many segments
./.venv/bin/canlens corroborate-boundaries VOLKSWAGEN_GOLF_MK7 AUDI_A3_MK3 SKODA_OCTAVIA_MK3
./.venv/bin/canlens truth dbc <file.dbc>                 # what ground truth it offers
./.venv/bin/canlens truth score <segment>/rlog.zst --dbc <file.dbc>
./.venv/bin/canlens export pdu-db <segment>/rlog.zst -o out.json
./.venv/bin/canlens cache {build,clear,status}
./.venv/bin/canlens gui
```

Segment paths carry no platform name — the layout mirrors the upstream bucket, which is keyed by
device and route. `corpus which` and the `[PLATFORM]` prefix on output are the way back.

---

## 3. Repository layout

| package | role | state |
|---|---|---|
| `src/canlens/corpus/` | manifest, selective fetch, local accounting | working |
| `src/canlens/decode/` | `rlog.zst` → columnar `FrameSet`; npz frame cache | working |
| `src/canlens/analyze/` | per-bit entropy, transition rate, classification; timing | working |
| `src/canlens/infer/` | counters, checksums, CRC-16/32/64, E2E, multiplexors, signals | working |
| `src/canlens/corroborate/` | bus identification, device-weighted agreement, pooled evidence | working |
| `src/canlens/gui/` | PySide6 workbench: Data, Heat Map, Corroborate screens | working |
| `src/canlens/export/` | BoAt PDU-database JSON | working |
| `src/canlens/truth/` | opendbc DBCs as a reference; precision/recall against them | working |
| `src/canlens/cli.py` | the single `canlens` entry point | working |
| `src/canlens/render.py` | terminal bit strips, sparklines, rulers, bars | working |
| `src/canlens/filters.py` | `vehicle,segment,bus,can_id` wildcard filter (`*`, `?`) | working |

`corpus/` is **deliberately stdlib-only** (`urllib`, `json`, `concurrent.futures`) so that
fetching data never drags in a compiler toolchain. Do not add a third-party import to it.

### Key files by task

| Task | File |
|---|---|
| Add or change a detector | `infer/message.py` (`_build`), then the detector's own module |
| Change what a counter claims | `infer/counters.py` |
| Change an 8-bit checksum or E2E Profile 1/11 | `infer/checksums.py` |
| Change a 16-bit CRC or E2E Profile 5 | `infer/crc16.py` |
| Change E2E Profile 22, 6, 4 or 7 | `infer/profiles.py` (+ `infer/crcwide.py` for CRC-32/64) |
| Change multiplexor detection | `infer/multiplex.py` |
| Change signal boundary detection | `infer/signals.py` |
| Change the signal precision filter | `infer/smoothness.py` |
| Change bit classification thresholds | `analyze/bits.py` |
| Change cadence classification | `analyze/timing.py` |
| Change cross-segment tiering | `corroborate/consensus.py` |
| Change pooled-evidence solving | `corroborate/pooled.py` |
| Change how buses are identified | `corroborate/buses.py` |
| Change how a DBC is read | `truth/dbc.py` |
| Change how findings are scored | `truth/score.py` |
| Change what the GUI draws | `gui/window.py` (Qt) and `gui/model.py` (no Qt) |
| Change terminal output | `cli.py` and `render.py` |

---

## 4. Hard invariants

Breaking any of these fails **silently**. Check them explicitly before you finish.

### 4.1 Bump the cache version whenever behaviour changes

`INFER_VERSION` in `infer/results.py` must be incremented on **any** change to a detector, a
threshold, or what `_build()` returns. `CACHE_VERSION` in `decode/cache.py` likewise, whenever
the stored columns change meaning.

A stale cache entry is not slower — **it is wrong.** It hands back yesterday's answer for
today's code. This is the single easiest way to produce a confidently incorrect result here.

After bumping, clear and rebuild before re-surveying the corpus:

```bash
./.venv/bin/python -c "from canlens.infer import clear_results; print(clear_results('$HOME/data/canlens'))"
./.venv/bin/canlens --root ~/data/canlens cache build   # ~200 s for 259 segments, 6 workers
```

### 4.2 Report only what was checked

Nothing in `infer/` may claim a field because it *looks* like one. Looking like one is what
`analyze/` already reported, and replacing that guess with a verified claim is the entire point
of the layer. Every detector reproduces the observed bytes and carries its match rate.

Thresholds: counters **0.95**, everything else **0.99**.

### 4.3 Detector order in `_build()` is load-bearing

`infer/message.py` runs, in this exact order:

1. Candidate bytes — mean bit transition rate ≥ 0.20
2. Counters
3. Plain checksums (9 byte-wide algorithms, first that fits wins), then
   Honda's 4-bit nibble where nothing wider explained the last byte
4. E2E Profile 1 / 11
5. Width-gated E2E Profiles 22, 6, 4, 7
6. CRC-16 (3 variants) and E2E Profile 5
7. Multiplexor
8. Two counter cleanups (`outside_checksums`, `outside_multiplex`)
9. Signals, over bits nothing above explained, then the value-jumpiness
   precision filter (`infer/smoothness.py`)

Each stage receives only the byte positions nothing simpler has already explained. **Reordering
changes results.** Without the ordering a plain byte sum gets reported as an exotic CRC.

### 4.4 Echo frames are not data

`CanData.src >= 128` marks a frame the logging device itself put on the wire, on bus
`src - 128`, while relaying traffic between the car and the ADAS camera. They are **30–46% of a
segment** and reproduce the source bus's address set exactly.

They are dropped by default. Only `decode.summarize()` counts them, deliberately. Including
them inflates every frequency, entropy and corroboration statistic built on top.

### 4.5 Intel bit order is the default and needs no translation

A bit index produced by this codebase is already a DBC `StartPos`. `export/pdu_db.py` writes it
straight through — **do not add a conversion.**

`BitOrder.MOTOROLA` is no longer merely a parameter: `infer/byteorder.py` decides each bus's
order from its own traffic and `apply_bus_order` re-reads signals on the buses that are not
Intel. A `SignalHypothesis` therefore carries a `byte_order`, and **`start_bit` means different
things under each**: the lowest bit under Intel, the *most* significant bit under Motorola, as a
DBC `StartPos` does. Always take positions from `SignalHypothesis.bits`; a big-endian field is
not contiguous in Intel numbering and `range(start_bit, end_bit)` claims the wrong bits.

### 4.6 Stay columnar

`FrameSet` holds a trace as numpy arrays plus one flat payload blob. Rebuilding `CanFrame`
objects costs about 0.7 s per segment, which would cap a 95× cache at roughly 2×.

Use `Message.bytes_matrix()` and `FrameSet.payload_matrix()`. Never materialise per-frame
objects inside a loop over a whole segment.

### 4.7 Never commit corpus data, caches, or fetched schemas

`/data/`, `*.zst`, `*.blf`, `*.asc`, `*.pcap` and `database.json` are gitignored. The capnp
schemas (from `commaai/openpilot` branch `release3`, plus `car.capnp` from `commaai/opendbc`)
and opendbc `.dbc` files are fetched on demand and stay unvendored, exactly like the corpus.

### 4.8 Evidence gates protect solved-for secrets

Any detector that solves for a Data ID never transmitted on the wire must first pass
`enough_evidence()` in `infer/checksums.py`: distinct payload contents ≥ secret bytes + 8.

Without it, a static message carrying an alive counter offers exactly sixteen distinct payload
contents — which is the size of a Profile 22 Data ID list — and the profile fits trivially.
Adding this gate removed Profile 22 from 136 platforms and Profile 6 from 139, including Toyota,
which actually uses a byte sum.

---

## 5. Code conventions

### 5.1 SPDX header on every new source file

Two lines, after any shebang, matching the surrounding files. All 48 source files and all 33
test files carry it; do not create one without it.

```python
# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT
```

### 5.2 Comments record *why*, with the numbers that settled it

This is the most distinctive property of the codebase and the thing most worth preserving. A
comment here is not a restatement of the code. It gives the reasoning, cites the measurement,
and says what was tried and rejected.

Representative examples to read before writing your own:

- `infer/counters.py` — records that a 0.40 LSB cutoff admitted 2729 start positions where 0.95
  admits 193, and found the same 163 counters four times slower.
- `decode/frameset.py` — records why the columnar shape exists and what the object path cost.
- `analyze/bits.py` — records the measured rate distribution that motivated splitting the
  `active` band, and states plainly that the 0.10 boundary is a choice rather than a discovery.
- `infer/checksums.py` — records that appending the Data ID last does *not* make a final XOR
  testable, because the CRC table is linear over GF(2), and corrects an earlier wrong claim.

**If you tune a threshold, state what you measured.** A tuned constant with no recorded
measurement is worse than an untuned one, because the next reader cannot tell whether it was
reasoned about.

### 5.3 Docstrings state the claim, not the mechanics

A detector's docstring should say what would falsify its hypothesis. Line length is 100
characters (`ruff`, configured in `pyproject.toml`). `ruff target-version` is `py311` and guards
the supported floor even though the dev environment runs 3.14 — do not use syntax newer than
3.11 in `src/`.

### 5.4 Commit messages

Format: `type(scope): summary` where type is `feat`, `fix`, `perf`, `docs` or `refactor`. The
body is prose wrapped at about 72 characters, explaining *why* and what was verified — not a
list of files changed. End every commit with the trailer:

```
Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

**Commit or push only when the user asks.** The default branch is `master`; the remote is
`git@github.com:AlexTech-stack/canlens.git`.

---

## 6. Testing

Tests live in `tests/`, one file per module, named for the behaviour being asserted rather than
the function under test. Prefer synthetic payloads with known ground truth over recorded
fixtures — that is why the suite needs no corpus.

| Convention | Detail |
|---|---|
| GUI tests | Set `QT_QPA_PLATFORM=offscreen` and `pytest.importorskip` PySide6 and pyqtgraph **before** importing `canlens.gui`. Keep Qt out of `gui/model.py` so the data model is testable without a display. |
| Corpus tests | `test_decode_integration.py` skips unless a segment has been fetched. These prove the schema pins and echo handling work on real logs. |
| Export tests | Validate against BoAt's real `pdu_db.schema.json` at `/home/testuser/BoAt/boat-platform/config/pdu_db.schema.json`, skipping when that repository is absent. |
| Path equivalence | `test_columnar_equivalence.py` holds the object path and the columnar path to identical results. **If you add a detector to `_build()`, both paths must still agree.** |
| Vectorised code | `TestVectorisedMatchesScalar` in `tests/test_infer.py` holds every vectorised detector to its scalar reference. Keep the scalar version as the readable specification; do not delete it as redundant. |

---

## 7. Recipes

### 7.1 Adding a detector

1. Write the detector in its own module under `infer/`, with a frozen dataclass hypothesis type
   carrying at minimum `match_rate` and `frames`.
2. Keep a scalar reference implementation next to any vectorised one.
3. Wire it into `_build()` in `infer/message.py` at the correct point in the order (§4.3),
   passing it only the bytes nothing simpler explained.
4. **Bump `INFER_VERSION`** in `infer/results.py`.
5. Export the public names from `infer/__init__.py` and add them to `__all__`.
6. Surface the finding in all five places or explain why not: `cli.py` (`infer trace` and
   `infer message`), `gui/window.py`, `gui/model.py` (`field_spans`, `_inferred_entries`),
   `corroborate/consensus.py`, and `export/pdu_db.py`.
7. Add tests including at least one negative case — something that *looks* like the finding but
   is not.
8. Clear and rebuild the results cache, then re-survey the corpus to see what changed.
9. Update `HOWTO.md`, and this file plus `CLAUDE.md` if any stated fact changed.

### 7.2 Surveying the corpus after a change

There is one segment per platform fetched locally (259 segments, 230 platforms). After
rebuilding the cache, iterate `infer_cached` over `inventory()` to count findings per platform.
Compare against the previous numbers before concluding the change was an improvement — a change
that *reduces* findings is often the correct one, because most false positives look like extra
findings.

### 7.3 Scoring against ground truth

opendbc has community DBCs for many corpus platforms; a local clone may exist at
`/home/testuser/BoAt/tools/dbc/opendbc`. There is no automatic platform-to-DBC mapping, so the
file is named explicitly:

```bash
./.venv/bin/canlens truth dbc /path/to/vw_mqb.dbc          # is it worth scoring against?
./.venv/bin/canlens truth score <segment>/rlog.zst --dbc /path/to/vw_mqb.dbc
```

The bus is chosen by identifier overlap and can be forced with `--bus`. Three rules keep the
numbers honest, and changing any of them needs a reason:

- A reference message the drive never carried is **not** a miss. It is counted separately.
- A kind the reference never names at all is **left out of the rates entirely**. Older Honda
  and Acura DBCs annotate no counters, and scoring 44 counter findings against one would report
  0% precision when the truthful answer is "this reference cannot say".
- A claim that overlaps a reference field without matching it exactly is `near`, never a hit.

**A disagreement is evidence, not a verdict.** The reference is itself reverse-engineered and
its authors routinely leave counters unnamed. Read the disagreement list before concluding that
either side is wrong.

---

## 8. Gotchas found the hard way

- **pyqtgraph auto-levels render a uniform image black.** A fully static payload *is* uniform,
  so every `setImage` call must pass `levels=(0, 255)`. `test_gui_render.py` asserts how many
  such call sites exist; adding an image view means updating that count.
- **`setMaximumHeight` does not reserve space.** The layout handed back 68px of a requested 96
  and the label lanes collapsed on top of each other. Use `setFixedHeight`.
- **A CRC is linear over GF(2).** When the only moving content of a message is its alive
  counter, the CRC byte is an affine image of that counter, and two of its bits walk 0..3 exactly
  like a genuine 2-bit counter. This produced phantom counters on 404 messages corpus-wide before
  `outside_checksums()` existed. Expect this class of bug anywhere one field is a function of
  another.
- **A selector cycling 0, 1, 2 is numerically a counter modulo 3**, and every byte it
  multiplexes cycles with it. `outside_multiplex()` drops counters wholly contained in the
  selector or its dependent bits, while keeping one that merely has its lowest bit locked to a
  two-frame schedule.
- **The AUTOSAR E2E Profile 1/11 final XOR cannot be read off the specification.** For a
  fixed-length message a final XOR is absorbed into an equivalent start state, so only structure
  across messages of *different* lengths can settle it. The convention in the code — register
  from 0x00, no final XOR, Data ID fed as `[addr & 0xFF, addr >> 8]` — recovers the CAN
  identifier as the Data ID on all 94 such messages of a Rivian; every other convention yields
  noise.
- **A "multiplexor" that is really a validity flag or a sign bit** also sorts frames into groups
  with different contents. What distinguishes a real selector is that it runs on a *schedule* —
  each value revisited at a steady interval with no long absence. Requiring that removed the
  large majority of early false positives.
- **The machine has 6 physical cores and 12 logical.** `default_jobs()` in `cli.py` returns
  physical cores deliberately; the second thread of a core adds nothing to numpy-bound work.
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
- **A logged bus number is a port, not a bus.** Across the corpus 293 of 5868 comparisons find
  that a different number matches better, over 12 of 31 multi-segment platforms; the KIA EV6
  disagrees with itself 42% of the time. `corroborate/buses.py` reconciles them from identifier
  overlap before pooling, and `corroborate_platform` does so by default. Grouping segments by the
  logged number pools two different buses and calls the disagreement evidence.
- **A narrow field is slow, and no rate floor will find it.** Reference signals of 1 to 3 bits
  sit at a median transition rate near 0.005 while signals of 9 bits and up sit at 0.285: an
  ignition switch does not run through its states, and a bit flipping fast *and* narrow is
  nearly always the bottom of a wider number. Lowering `MIN_RATE` from 0.02 to 0.01 gained 43
  signals, but of 37 measured gains only 3 were 3 bits or narrower and 28 were 8 bits or wider
  — it recovered the *tops of wide fields*, where the rate decay runs out, not narrow ones. A
  2-bit enum's high bit moves half as often as its low bit, so whatever floor admits the first
  drops the second and leaves a flag. Grouping by co-occurring transitions points the wrong way
  (46% agreement inside a signal against 59% across a boundary), because neighbouring slow
  fields both react to the ignition.
- **Entropy answers a different question than transition rate.** Per-bit entropy sees 99% of
  reference signals where the rate floor sees 78%, and 96% of 1-bit fields where the floor
  sees 34% — but it cannot place a boundary: the ratio between adjacent bits has a median of
  1.00 both inside a field and at an edge, because duty-cycle entropy saturates near 1. Using
  entropy for liveness and rate for boundaries scores worse than rate alone at every threshold
  (F1 0.489 against 0.504), because a bit admitted by a statistic that cannot segment it has
  nowhere to be cut. BinaryInferno's byte-value form fails separately: it can only cut on byte
  edges, and 4126 of 4539 reference signals are not byte-aligned.
- **Agreement across platforms is the strongest *ranking* signal found so far.** Over 17 MQB
  platforms, a start bit proposed by one platform is right 20% of the time and one proposed by
  all of them 85% of the time, rising monotonically in between. Filtering to the established
  tier takes signal precision from 53% to 73%; no single-platform *boundary* threshold sweep
  reached past the low fifties. It cannot invent a boundary, so recall is capped by
  `find_signals`. Widths are reported as `at least n bits` from the group's widest claim, which
  is still short 67% of the time and overshoots 13%. See `corroborate/boundaries.py`.
- **How far a value moves is a precision signal that rate, length and entropy all miss.**
  A field whose value lurches across more than 10% of its observed range on more than 20% of
  frames is not one coherent signal. Dropping such claims (`infer/smoothness.py`) trades four
  points of recall for twelve of precision over eleven platform/DBC pairs (P 0.593 -> 0.714,
  F1 0.604 -> 0.643), and removes 29% of all signal claims across 38 corpus segments — six
  times more false than true. Controls rule out the confound: a length floor and a rate floor
  both *lower* F1. It is a filter, not a segmenter; it cannot split a merged field, and a
  carry-based split of one was measured and rejected because real signals are signed and
  offset around a midpoint, not natural-binary. See `infer/smoothness.py`.
- **A better cut rule is not a better field detector.** CAN-D's conditional-flip terms
  (Algorithm 1) raise per-bit boundary recall from 51% to 79% and drop per-bit precision from
  94% to 37%; as extra cut conditions they take whole-field F1 from 0.533 to 0.414. Every
  wrong cut destroys two fields, not one, so boundary recall was never the binding constraint.
  In CAN-D that heuristic feeds a global optimizer that re-decides each cut against a cut
  penalty; used as a standalone rule it only adds cuts. Measured with and without constant-bit
  removal. The existing rate-ratio rule scores 0.660 on the per-bit metric, so the cut rule is
  not the weak part.
- **A field inside one byte reads as the same integer in either bit order.** Not just the same
  bits: a little-endian field runs LSB-first from its start and a big-endian one MSB-first from
  its start, and inside one byte those coincide (all 36 single-byte layouts). This is why a
  Rivian's 169 counters and 159 checksums were all correct while the bus was being read in the
  wrong order — every counter there is a 4-bit E2E alive counter inside one byte, and checksums
  consume whole bytes and never touch bit numbering. Only signals were ever wrong, being the
  only fields wide enough to cross a byte. Residual exposure: a counter spanning a byte on a
  Motorola bus, which is 2 of 598 over a nine-platform sample.
- **Do not trust a timing claim you did not measure in this codebase.** Lazy attribute access
  across the pycapnp boundary made an early "0.02 s parse" claim wrong by two orders of
  magnitude.

---

## 9. Not implemented — say so plainly

- **Scaling and units.** No factor, offset or unit is ever inferred.
- **Signals inside a multiplexed layout.** The selector is found and its per-value layouts are
  drawn, but no detector runs separately within each selector value.
- **Motorola for counters, checksums and multiplexors.** The bus's bit order is decided and
  applied to *signal* detection only. Counters, checksums and multiplexors are verified
  arithmetic over whole bytes, so bit numbering does not move them and their reported
  `StartPos` remains Intel.
- **Variable-length E2E.** Profiles 4 and 7 are claimed only where Length is fixed across the
  trace, which is the honest scope of the check.

---

## 10. Related projects and references

- [opendbc](https://github.com/commaai/opendbc) — CAN database definitions; the ground truth the
  multiplexor detector was verified against.
- [openpilot](https://github.com/commaai/openpilot) — `LogReader`, cabana, and the `cereal`
  capnp schemas `decode/` pins.
- **BoAt** — the deterministic SIL/HIL simulation platform this grew out of, and the consumer of
  `export/`. It maintains its own `CLAUDE.md`/`AGENTS.md` pair under the same convention.
- **AUTOSAR specifications** live under `/home/testuser/BoAt/spec/` (symlinked, gitignored,
  populated per machine): `spec/text/*.txt` as flat UTF-8, `spec/search.db` as SQLite FTS5.
  `AUTOSAR_PRS_E2EProtocol.txt` (FO R19-11) is the document every E2E detector here was built
  from; cite requirement numbers (`[PRS_E2E_00xxx]`) in comments as the existing code does.
