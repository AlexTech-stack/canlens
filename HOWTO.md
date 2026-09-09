# canlens HOWTO

A walkthrough from empty machine to per-bit measurements of a real vehicle's
CAN traffic. Every command and every number below was run against the live
corpus, not invented.

> **Scope.** `corpus`, `decode` and `analyze` work today. `infer`,
> `corroborate` and `truth` are placeholders — see [Where this stops](#where-this-stops).

---

## 1. Setup

The project needs a virtualenv. On Ubuntu the `venv` module ships separately:

```bash
sudo apt-get install -y python3.14-venv
```

```bash
python3 -m venv .venv && ./.venv/bin/pip install -e ".[dev,decode]"
```

`decode` is the extra that pulls `pycapnp`. On Python 3.14 zstd comes from the
stdlib (`compression.zstd`, PEP 784) and nothing extra is installed for it; on
3.11–3.13 the `zstandard` package is pulled in automatically.

Check it took:

```bash
./.venv/bin/canlens --version
```

---

## 2. Choose what to download

The corpus is **299 GB across 188,883 segments** on 230 vehicle platforms.
Downloading all of it is not an option the tool offers, and you almost never
want it: the largest single platform is ~18 GB and the *median* platform is
~0.3 GB.

```bash
canlens corpus list
```

```
TOYOTA_RAV4_TSS2                        11387 segments  ~ 18.01 GB
TOYOTA_COROLLA_TSS2                      8559 segments  ~ 13.54 GB
TOYOTA_PRIUS                             8310 segments  ~ 13.15 GB
CHEVROLET_BOLT_EUV                       7662 segments  ~ 12.12 GB
...
230 platforms, 188883 segments total
```

Size a selection before committing to it:

```bash
canlens corpus plan KIA_EV6 TOYOTA_PRIUS --limit 200
```

```
KIA_EV6                                   200 segments  ~  0.16 GB
TOYOTA_PRIUS                              200 segments  ~  0.16 GB
TOTAL                                     400 segments  ~  0.32 GB
```

Then fetch. Downloads are parallel and **resumable** — files already present
are skipped, and an interrupted transfer leaves a `.part` that is retried
rather than a truncated file that would be silently trusted:

```bash
canlens corpus fetch KIA_EV6 --limit 200
```

```bash
canlens corpus status
```

Data lands in `~/data/canlens` by default. Override with `--root` or
`$CANLENS_DATA`. It is gitignored and must stay that way.

**Start small.** A few hundred segments is enough to develop against; you can
always fetch more of a platform later without re-fetching what you have.

---

## 3. Decode

Segments are openpilot `rlog.zst`: zstd-compressed capnp (`cereal`) message
streams — not `.asc` or `.blf`. Reading them needs schema files that ship on
no package index, so fetch them once:

```bash
canlens decode schema
```

This pulls five `.capnp` files into `~/data/canlens/schema`, pinned to commit
SHAs rather than branches so a segment cannot decode differently next month.

Then look at what a segment holds:

```bash
canlens decode summary ~/data/canlens/segments/*/*/*/rlog.zst
```

```
318551 frames over 60.0s (135358 echoes dropped, 30% of raw)
239 unique (bus, address) pairs
  bus 0:   88767 frames,   70 addresses
  bus 1:  183197 frames,  149 addresses
  bus 2:   46587 frames,   20 addresses
```

### The echo trap — read this before trusting any statistic

`CanData.src` is **not a plain bus number**. Values ≥ 128 mark a frame the
recording device *itself* put on the wire, on bus `src - 128`, as it relays
traffic between the car and the ADAS camera. Buses 0 and 2 are relayed into
each other, and the echo reproduces the source bus's address set exactly.

Across sampled segments this is **30–46% of every log**. Counting echoes as
independent observations inflates every frequency, entropy and corroboration
statistic downstream — and makes the evidence for a hypothesis look roughly
twice as strong as it is.

`canlens` therefore **drops echoes by default**. Ask for them explicitly only
when you are studying the relay itself:

```python
iter_frames(path, root=root, include_echo=True)
```

---

## 4. Analyze

```bash
canlens analyze trace ~/data/canlens/segments/240e.../11/rlog.zst --top 8
```

```
318551 frames, 239 messages, 60.0s, 12289 bits of payload entropy

message           count  len   period   cadence  entropy  bits (const/slow/active/noisy)
bus 0 0x211        1199  32    50.0ms    cyclic    191.8  42/16/168/30
bus 0 0x212        1199  32    50.0ms    cyclic    191.6  40/22/166/28
bus 0 0x210        1199  32    50.0ms    cyclic    190.1  41/13/175/27
bus 2 0x181        1999  32    29.7ms    cyclic    172.6  35/51/148/22
bus 0 0x276        1199  32    50.1ms    cyclic    144.9  99/1/135/21
```

### Reading the columns

| column | meaning |
|---|---|
| `count` | frames observed for this `(bus, address)` |
| `len` | payload length used for bit stats; `*` marks a varying length |
| `period` | median inter-arrival gap |
| `cadence` | `cyclic`, `sporadic`, or `single` (too few samples to say) |
| `entropy` | total Shannon entropy across the payload, in bits |
| `bits` | how many bits fall in each behaviour class |

Rows are sorted by payload entropy, so **the top of the list is where to start
looking**. A message with near-zero entropy has nothing in it to reverse
engineer, however often it arrives.

### The four bit classes

- **constant** — never changes. Padding, reserved fields, or a value fixed for
  this drive.
- **slow** — changes on under 1% of frames. Door open, gear position, warning
  lamps.
- **active** — changes at a moderate rate. Where physical quantities live:
  speed, angle, temperature.
- **noisy** — changes on 40%+ of frames, about as often as a coin flip. Counter
  low bits and CRCs look like this.

That last row above is worth an eye: `0x276` has **99 constant bits** out of
256 and only 1 slow bit. Two-fifths of its payload never moves — a strong hint
of reserved space or a fixed-value field, and a cheap way to shrink the search
before any inference runs.

### These thresholds are heuristics

The cutoffs (`SLOW_MAX_RATE = 0.01`, `NOISY_MIN_RATE = 0.40`,
`CYCLIC_MAX_SPREAD = 0.25`) exist to *partition work*, not to settle
questions. They live in `canlens.analyze.bits` and `canlens.analyze.timing`
and are meant to be tuned. A `noisy` bit is a candidate for a counter or CRC —
confirming which is the inference layer's job, and it re-tests from scratch.

### Timestamp resolution

The only timestamp available is the enclosing capnp event's `logMonoTime`, so
frames batched into one event share a stamp. That puts a floor on resolvable
jitter, which is why `cadence` is judged on *relative* spread rather than an
absolute millisecond threshold.

---

## 5. From Python

```python
from canlens.corpus import Manifest
from canlens.decode import iter_frames
from canlens.analyze import BitKind, analyze_segment

root = "~/data/canlens"

# What is in the corpus
manifest = Manifest.load(f"{root}/database.json")
print(manifest["KIA_EV6"].count, "segments")

# Frame by frame
for frame in iter_frames(path, root=root):
    ...  # frame.mono_ns, frame.bus, frame.address, frame.data

# Measured, grouped by message
profile = analyze_segment(path, root=root)
for message in profile.by_entropy()[:10]:
    print(message, message.timing.period_ms, message.bits.payload_entropy)

# Zoom in on one message's bits
bits = profile[(0, 0x211)].bits
noisy = [i for i, k in enumerate(bits.kinds) if k is BitKind.NOISY]
print("counter/CRC candidates at bit positions:", noisy)
```

Bit numbering is **MSB-first within each byte**, matching how CAN signal
layouts are conventionally written: bit 0 is the most significant bit of
byte 0.

---

## 6. Gotchas

**Mixed payload lengths.** When a message's length varies, bit statistics are
computed over the *dominant* length only and the row is flagged with `*`.
Padding short frames would manufacture constant bits that were never on the
wire. A varying length usually means a multiplexed message, or two senders
sharing an address — worth investigating rather than averaging away.

**The `opendbc` PyPI package is not used.** It declares
`requires-python = <3.13` and pins `pycapnp==2.1.0`, so it cannot be installed
alongside this project on 3.13+. Only its `.dbc` files were ever wanted, and
those are plain data.

**`car.capnp` comes from opendbc, not openpilot.** In openpilot it is a git
symlink, so fetching it from there returns 37 bytes of link text instead of a
schema — which fails much later with a confusing capnp error. `decode/`
fetches it from opendbc and rejects any schema file that arrives suspiciously
short.

**`log.capnp` is on openpilot's `release3` branch**, not `master`.

---

## 7. Development

```bash
./.venv/bin/pytest -q
```

```bash
./.venv/bin/ruff check . && ./.venv/bin/mypy
```

Tests split in two. Unit tests need neither network nor corpus data and always
run. Integration tests (`tests/test_decode_integration.py`) run against real
segments when they are present and skip cleanly when they are not, so a fresh
checkout is green before anything is downloaded.

`ruff` is pinned to `target-version = "py311"` to guard the supported floor
even though development happens on 3.14.

---

## Where this stops

| package | state |
|---|---|
| `corpus/` | working — manifest, selective fetch, local accounting |
| `decode/` | working — `rlog.zst` → `CanFrame` |
| `analyze/` | working — timing, entropy, per-bit classification |
| `infer/` | **not implemented** — signal boundaries, counters, CRCs, multiplexors |
| `corroborate/` | **not implemented** — cross-segment and cross-platform agreement |
| `truth/` | **not implemented** — opendbc ground truth, scoring the engine |

`analyze` deliberately stops at measurement. It will tell you a bit flips on
94% of frames; it will not tell you it is a counter. That claim requires
checking the stride against the cycle time and confirming it across many
segments — which is `infer`'s and `corroborate`'s job, and the reason the
corpus is the point of this project rather than an afterthought.
