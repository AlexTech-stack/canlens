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

### Which car am I looking at?

Nothing in a segment's path says which vehicle it came from. The on-disk
layout mirrors the upstream bucket, which is keyed by *device* and *route*:

```
segments/0045a17309e72a84/00000044--dbc821d8d9/464/rlog.zst
         ^device           ^route              ^index
```

Only `database.json` maps those to a platform, so a glob like
`segments/*/*/*/rlog.zst` happily mixes every car you have fetched. Ask
directly:

```bash
canlens corpus which ~/data/canlens/segments/0045a17309e72a84/*/*/rlog.zst
```

```
TOYOTA_PRIUS   /home/testuser/data/canlens/segments/0045a17309e72a84/00000044--dbc821d8d9/464/rlog.zst
```

`decode summary` and `analyze trace` also print the platform as a `[PLATFORM]`
prefix, so their output is never ambiguous about which car it describes.

Then look at what a segment holds:

```bash
canlens decode summary ~/data/canlens/segments/*/*/*/rlog.zst
```

```
[KIA_EV6] /home/testuser/data/canlens/segments/240e.../11/rlog.zst
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
[KIA_EV6] /home/testuser/data/canlens/segments/240e.../11/rlog.zst
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
| `len` | payload length in bytes — 8 is classic CAN, 32+ means CAN FD |
| `period` | median inter-arrival gap |
| `cadence` | `cyclic`, `sporadic`, or `single` (too few samples to say) |
| `entropy` | total Shannon entropy across the payload, in bits |
| `bits` | how many bits fall in each behaviour class |

Rows are sorted by payload entropy, so **the top of the list is where to start
looking**. A message with near-zero entropy has nothing in it to reverse
engineer, however often it arrives.

### The bit map

The right-hand column draws one character per payload bit, coloured by class —
white `constant`, blue `slow`, green `active`, yellow `busy`, red `noisy` —
with a space at every byte boundary so a field straddling two bytes is obvious.

```
message         count  len   period   cadence  entropy  bits (intel, 64 of 64 shown)
bus 1 0x365       600   8   100.0ms    cyclic     52.7  ##**+++: ##****** ######## ++++++++ #####**+ **#***.. **#***.. ########
bus 1 0x210      1200   8    50.0ms    cyclic     49.2  ##**+++: #***++++ .##**##* *++++::: .:..+#** ++++..++ +:.***** ###**##*
```

Piped output, `NO_COLOR=1`, or `--no-color` falls back to glyphs of escalating
density — `.` constant, `:` slow, `+` active, `*` busy, `#` noisy — so the
shape survives a log file or a colour-blind reader. `FORCE_COLOR=1` forces colour on.

The strip is sized to your terminal; wider terminals show more bits. A `›` at
the end means the payload continues past what fits, which is normal for CAN FD
(32 bytes is 256 bits). Use `--bit-columns N` to pin the width, or
`--no-bitmap` for the old numeric `const/slow/active/noisy` counts.

Reading these pays off quickly. On the EV6, `0x210`–`0x212` all open with
**two solid red bytes** — sixteen bits flipping almost every frame at the head
of the payload, which is what a CRC looks like. On the Prius, every message
above shares the byte-0 pattern `##**+++:` — the same descending gradient,
which is a field boundary showing itself.

### Bit order: Intel by default

`--order intel` (the default) numbers bits the way DBC files do: index 0 is the
**least** significant bit of byte 0, index 7 its MSB, index 8 the LSB of byte 1.
`--order motorola` walks each byte the other way, MSB first, as you read hex.

The two are an exact per-byte reversal of each other and **nothing measured
changes** — same entropies, same classes, same totals, just a different column
order. Only the display and the bit indices move:

```
intel     ##**+++: #***++++ .##**##* *++++::: .:..+#** ++++..++ +:.***** ###**##*
motorola  :+++**## ++++***# *##**##. :::++++* **#+..:. ++..++++ *****.:+ *##**###
```

Pick the one matching the convention you will write your signal definitions
in, so a bit index you note down means the same thing later.

### The five bit classes

Ordered by how much the bit moves:

| class | colour | glyph | transition rate | typically |
|---|---|---|---|---|
| `constant` | white | `.` | never | padding, reserved, fixed for this drive |
| `slow` | blue | `:` | ≤ 1% | door open, gear, warning lamps |
| `active` | green | `+` | 1–10% | upper bits of a numeric value |
| `busy` | yellow | `*` | 10–40% | lower bits of a moving value |
| `noisy` | red | `#` | ≥ 40% | counter LSBs, CRCs |

The `active`/`busy` boundary is what makes **field structure** visible. A
numeric signal's transition rate falls with bit significance, so a field shows
up as a gradient running from noisy at its LSB to slow at its MSB:

```
bus 1 0x210   ##**+++:  #***++++  .##**##*  ...
              ^^^^^^^^
              one field: # # * * + + + :
```

Pooled over 24 corpus segments, mean transition rate by bit position within a
byte runs `.343 .242 .180 .151 .137 .132 .121 .105` — monotonically down from
LSB to MSB, which is why the gradient is readable at all.

That last row above is worth an eye: `0x276` has **99 constant bits** out of
256 and only 1 slow bit. Two-fifths of its payload never moves — a strong hint
of reserved space or a fixed-value field, and a cheap way to shrink the search
before any inference runs.

### These thresholds are heuristics

The cutoffs (`SLOW_MAX_RATE = 0.01`, `BUSY_MIN_RATE = 0.10`,
`NOISY_MIN_RATE = 0.40`, `CYCLIC_MAX_SPREAD = 0.25`) exist to *partition work*, not to settle
questions. They live in `canlens.analyze.bits` and `canlens.analyze.timing`
and are meant to be tuned. A `noisy` bit is a candidate for a counter or CRC —
confirming which is the inference layer's job, and it re-tests from scratch.

### Timestamp resolution

The only timestamp available is the enclosing capnp event's `logMonoTime`, so
frames batched into one event share a stamp. That puts a floor on resolvable
jitter, which is why `cadence` is judged on *relative* spread rather than an
absolute millisecond threshold.

---

## 5. Infer

`analyze` measures. `infer` makes claims — and only ones it has **checked**
against the trace.

```bash
canlens infer trace <segment>/rlog.zst
```

```
[TOYOTA_PRIUS] .../464/rlog.zst
167 messages examined, 69 with findings: 79 counters, 69 checksums

message        frames  counters                           checksums
bus 0 0x024      4984  -                                  toyota@7 100%
bus 1 0x210      1200  8b@0 100%                          toyota@7 100%
```

`8b@0` is an 8-bit counter starting at bit 0; a `/n` suffix marks a step other
than 1. Percentages are the fraction of the trace the hypothesis reproduces —
they are the whole point, and a finding below 100% deserves a look.

### One message, in detail

```bash
canlens infer message <segment>/rlog.zst --address 0x210 --bus 1
```

```
[TOYOTA_PRIUS] 0x210
1200 frames x 8 bytes, intel bit order

  ##**+++: #***++++ .##**##* *++++::: .:..+#** ++++..++ +:.***** ###**##*
  CCCCCCCC                                                       XXXXXXXX

  counter  8 bits @ bit 0, step 1, wraps every 256
           ▁▁▁▁▁▁▁▂▂▂▂▂▂▂▂▂▂▂▂▃▃▃▃▃▃▃▃▃▃▃▃▃▄▄▄▄▄▄▄▄▄▄▄▄▅▅▅▅▅▅▅▅▅▅▅▅▆▆▆▆▆▆▆▆▆▆▆▆▆▇▇▇▇▇▇▇▇▇▇▇▇███████
           ████████████████████ 100.0% of steps

  checksum byte 7, algorithm toyota
           ████████████████████ 100.0% (1200/1200 frames)
```

Three things are stacked deliberately. The **ruler** (`C` counter, `X`
checksum) sits directly under the bit map, character for character, so a
claimed field can be checked against the measured bits above it. The
**sparkline** plots the field's actual values — a counter's saw teeth make it
self-evident, and the first samples are drawn rather than the series
decimated, because decimating a counter aliases it into noise. The **bar** is
the match rate.

Note how `##**+++:` under `CCCCCCCC` is exactly the gradient an incrementing
byte must produce: bit 0 flips every frame, bit 1 every second, and so on down
to bit 7 flipping once per 128. The measurement and the inference were
computed independently and agree.

Widen the plot with `--samples` to see the counter wrap.

### What is and is not claimed

**Counters** must advance by a constant step *and* walk their whole range. Two
filters enforce that. The step must be coprime with the field width — without
it, a single toggling bit at the top of an n-bit window is a mathematically
perfect counter of step 2^(n-1) that only ever holds two values, and every
even step on a power-of-two field is a version of that mistake. The field must
also have been seen holding at least half its possible values.

**Checksums** are only reported when a named algorithm *reproduces* the byte.
The library is `sum8`, `sum8_complement`, `xor8`, `toyota`, and CRC-8 with
polynomials 0x07, 0x1D (SAE J1850), and 0x2F; 16-bit CRCs are searched
separately (see below). The simplest algorithm that
works wins, so a plain sum is never dressed up as a CRC.

`xor8` reports as ambiguous, shown `xor8@?`. XOR is self-inverse: if byte 7 is
the XOR of bytes 0-6, then byte 0 is equally the XOR of bytes 1-7. Every
position verifies, and a single trace cannot say which one the protocol calls
the checksum. The last byte is reported by convention and every candidate
position is kept.

### 16-bit CRCs and AUTOSAR E2E Profile 5

Parameters follow AUTOSAR_PRS_E2EProtocol (FO R19-11). `Crc_CalculateCRC16` is
CCITT-FALSE — polynomial 0x1021, start 0xFFFF, no final XOR — and
[PRS_E2E_00400] names it as Profile 5's CRC. Per [PRS_E2E_00401] the CRC
covers the payload *excluding the CRC bytes*, extended at the end with the
**Data ID**; Figure 6.54 fixes the order as ID low byte then high byte, and
Figure 6.55 stores the CRC little-endian.

The Data ID is the interesting part: it is *implicitly sent* — it never
appears on the wire — so a trace cannot read it. It can be **solved for**. A
CRC is deterministic, so the observed value constrains the two appended bytes;
inverting a single CRC step recovers them in 256 steps rather than sweeping
65536 candidates, and intersecting a handful of frames leaves exactly one
Data ID.

On the EV6 this cracks the whole bus:

```
[KIA_EV6] 0x211
  ######## ######## ##**+++: ...
  XXXXXXXX XXXXXXXX CCCCCCCC

  crc16    bytes 0-1, e2e_p05, little-endian, data ID 0xFA11
           ████████████████████ 100.0% (1199/1199 frames)
```

A 16-bit CRC then an 8-bit counter — textbook Profile 5, and exactly the shape
the bit map showed before anything was inferred. **162 of 237 messages** carry
one, all at 100%.

And the recovered Data IDs are not arbitrary:

| address | Data ID |
|---|---|
| 0x051 | 0xF851 |
| 0x201 | 0xFA01 |
| 0x2A4 | 0xFAA4 |

`data_id == (0xF8 + (addr >> 8)) << 8 | (addr & 0xFF)` — which holds for
**162 of 162**. Each message's Data ID was solved independently, and they all
land on one rule.

This is also a preview of why the corpus matters: `toyota@7 100%` holding
across 69 messages of one segment is suggestive, but holding across thousands
of segments from hundreds of different drivers is proof — and that is
`corroborate`'s job, not `infer`'s.

## 6. From Python

```python
from canlens.corpus import Manifest
from canlens.decode import iter_frames
from canlens.analyze import BitKind, analyze_segment
from canlens.infer import infer_segment

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

# Tested claims, not candidates
for message in infer_segment(path, root=root):
    for counter in message.counters:
        print(message, counter)      # "8-bit counter @ bit 0 (100.0%)"
    for checksum in message.checksums:
        print(message, checksum)     # "toyota @ byte 7 (100.0%)"
```

Bit numbering is **MSB-first within each byte**, matching how CAN signal
layouts are conventionally written: bit 0 is the most significant bit of
byte 0.

---

## 7. The workbench

```bash
pip install -e ".[gui]" && canlens gui
```

Everything the CLI prints, arranged for the reverse-engineering loop:

- **left** — every locally fetched segment, labelled by platform
- **centre** — the whole bus as one bit matrix, rows ordered **by identifier**:
  standard IDs ascending, then extended, grouped by bus. Dividers mark each
  bus change and the step into extended IDs
- **strip** — the selected message magnified, with inferred counter and
  checksum fields outlined and labelled
- **bottom** — findings, and the counter's real values plotted over time

The matrix is drawn with pyqtgraph's `ImageItem`, which takes the bit classes
as a numpy array directly. That is why this is a desktop app rather than a web
one: the analysis already produces the array the renderer wants, so there is no
serialisation step between them.

The magnified strip exists because of a scale problem worth knowing about. At
186 messages a single row of the matrix is under three pixels tall, so a field
overlay drawn there is invisible however correct its coordinates are. The
matrix is the map; the strip is the detail.

**Row order is by identifier, deliberately.** Entropy ordering would put the
most interesting messages on top, but it is not stable: the same bus recorded
twice sorts differently, so two snippets of one drive cannot be read row
against row. Identifier order is fixed by construction — verified across three
real segments, where every shared message keeps the same relative position.
The CLI's `analyze trace` still sorts by entropy, because there the question
is "where do I start", not "how do these compare".

**Extended identifiers are inferred, not read.** openpilot's `CanData` carries
`address`, `dat` and `src` and no IDE flag, so anything above 0x7FF must be a
29-bit identifier — but an extended frame using a low identifier is
indistinguishable from a standard one in this format. The divider marks what
the data can actually support.

Bit axes tick on **byte boundaries** (8, 16, 24 …), never pyqtgraph's default
decimal steps, which put gridlines through the middle of bytes. A 32-byte CAN
FD payload has too many boundaries to label, so the labels thin to every 16 or
32 bits and the rest stay as unlabelled minor ticks.

Headless check, the same way BoAt verifies its Qt client:

```bash
QT_QPA_PLATFORM=offscreen canlens gui
```

## 8. Gotchas

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

## 9. Development

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
| `infer/` | counters, 8-bit checksums, 16-bit CRCs and E2E Profile 5 Data IDs working; signal boundaries and multiplexors still to do |
| `corroborate/` | **not implemented** — cross-segment and cross-platform agreement |
| `truth/` | **not implemented** — opendbc ground truth, scoring the engine |

`analyze` deliberately stops at measurement. It will tell you a bit flips on
94% of frames; it will not tell you it is a counter. That claim requires
checking the stride against the cycle time and confirming it across many
segments — which is `infer`'s and `corroborate`'s job, and the reason the
corpus is the point of this project rather than an afterthought.
