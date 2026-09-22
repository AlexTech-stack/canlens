# canlens HOWTO

A walkthrough from empty machine to per-bit measurements of a real vehicle's
CAN traffic. Every command and every number below was run against the live
corpus, not invented.

> **Scope.** Every layer works today — `corpus`, `decode`, `analyze`, `infer`,
> `corroborate`, `truth`, `gui` and `export`. What each one does *not* yet
> claim is listed under [Where this stops](#where-this-stops).

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

## 5. The decode cache

Decoding a segment costs ~1.5 s, and profiling showed almost all of it is
~1.2 million field reads across the pycapnp boundary at ~0.6 µs each. That is
not something Python can be made to do faster — so it is done once.

```bash
canlens cache build              # every local segment
canlens cache build KIA_EV6      # or just one platform
canlens cache status
canlens cache clear
```

Measured over a whole local corpus:

```
analyze 31 segments
  without cache    64.4s
  with cache       11.9s   -> 5.4x
```

The cache is **smaller than what it was decoded from** — 83 MB against 104 MB
of `.zst` — because columns compress far better than interleaved capnp. It
lives under `<root>/cache/`, mirroring the segment layout, and is rebuilt
automatically when a segment's size or mtime changes, when the cache format
version moves, or when an entry is unreadable for any reason at all.

Everything downstream now runs columnar. `FrameSet` holds a trace as arrays
plus one flat payload blob, and a message's payloads are gathered into a byte
matrix by index arithmetic rather than by slicing out thousands of `bytes`.
That part matters: rebuilding 400,000 `CanFrame` objects from the cache costs
0.7 s on its own, which would have capped a 12× cache at about 2×.

### Vectorised detectors

`infer` was the rest of the cost, and its checksum and CRC searches were
per-frame Python loops — `crc16_update` alone was called 586,000 times. They
now score every frame at once against the byte matrix the cache already
provides. The scalar functions remain as the readable statement of each
algorithm and as what the known-answer tests check; a test holds the two forms
to each other for every algorithm, width and byte position.

Measured over the whole local corpus:

```
full pipeline over 31 segments: 74.3s (2.40s each)
  before this work:            126.6s (4.08s each)
  speedup:                       1.7x
```

**One optimisation was tried and reverted.** Batching every start position of
a counter scan into a single sliding-window matmul sounds obviously better and
measured 3.18 s against 2.65 s — on a 32-byte payload only a few dozen of the
250-odd offsets can begin a counter at all, so computing them all costs more
than the per-position calls it saves. The comment in `find_counters` records
the numbers so nobody tries it twice.

### The second pass: an exact bound, a results cache, and parallelism

A reevaluation of the above found the biggest remaining win was algorithmic,
not numeric. A counter's stride must be coprime with a power-of-two width, so
it is odd, so the field's LSB flips on *every* increment — which means a
counter accepted at 95% match **must** have an LSB transition rate of at
least 95%. The prefilter had been set at a merely "noisy" 0.40:

```
min_lsb_rate=0.40   eligible starts= 2729   counters=163   1.64s
min_lsb_rate=0.95   eligible starts=  193   counters=163   0.43s
```

Same findings, 3.8× faster, one line. It now defaults to `min_match`.

Inference results are cached too — the decode cache was built first only
because decode was profiled first. Inference is deterministic in its frames,
so its result is stored (gzipped, ~200 KB) beside the frame cache and keyed on
the same source identity plus `INFER_VERSION`, which **must be bumped when a
detector or threshold changes**: a stale result here is not slower, it is
wrong. `analyze` and `infer` also share bit profiles rather than measuring
each message twice, and only a 64-frame sample of payloads is materialised as
`bytes`, since every detector scores the byte matrix.

`canlens cache build` now fills both caches over a process pool, defaulting
to **physical** cores — the machine's 12 logical CPUs are 6 cores with
hyperthreading, and the second thread of a core adds nothing to numpy-bound
work. The Data screen has the same as a **Build cache** button.

Measured on the local corpus:

```
                                  per segment
original pipeline                    4.08s
first pass, parallel build           0.68s wall   (32 segments in 21.6s, 6 workers)
every pass after that                0.40s        (10x)
workbench opening a warm segment     0.64s        (was 5.47s)
```

Still no reason to leave Python. If the counter scan ever matters again,
`numba` on that one loop is the middle step.

## 6. Infer

`analyze` measures. `infer` makes claims — and only ones it has **checked**
against the trace.

```bash
canlens infer trace <segment>/rlog.zst
```

```
[TOYOTA_PRIUS] .../464/rlog.zst
167 messages examined, 69 with findings: 79 counters, 69 checksums

message        frames  counters                           checksums
bus 0 0x024      4984  -                                  sum8_addr_len@7 100%
bus 1 0x210      1200  8bit@0 100%                        sum8_addr_len@7 100%
```

`8bit@0` is an 8-bit counter starting at bit 0; a `/n` suffix marks a step other
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

  checksum byte 7, algorithm sum8_addr_len
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

### Signals — where one field ends and the next begins

Every detector above claims a field because it can *reproduce* it. An ordinary
signal offers nothing of the kind: a wheel speed is just bits that move, and
the only question is which of them move together.

**The start of a field is visible.** Inside a numeric field the transition rate
falls away from the least significant bit — bit 0 flips every frame, bit 1
every other, and so on — so a boundary shows up as the rate *rising* again
where the next field's low bit begins. Scored against opendbc, a rate rising to
three times its neighbour puts the start bit right on 210 of 411 real signals.

**The far end has no marker at all.** A field's high bits stop moving because
the value never grew large enough to reach them, and a trace cannot tell that
from the field ending there. Of 411 signals whose bits move, only 157 have
every declared bit moving; 209 start exactly where the DBC says and simply run
out of evidence. The rate profile of a genuine 16-bit signal reads

```
0.30 0.29 0.30 0.30 0.27 0.20 0.12 0.06 0.03 0.02 0.01 0.01 0.00 0.00 0.00 0.00
```

and nothing in it says sixteen rather than ten.

So what is claimed is the span the trace justifies, and a `+` marks the rest:

```
  signal   at least 4 bits @ bit 16, seen 0..15
  signal   at least 3 bits @ bit 30, seen 0..7
```

A field stopped by something — another moving field, a counter, a checksum, the
end of the payload — is *bounded*, and its width is the width. One that faded
into still bits is a lower bound. The distinction earns its keep: bounded
claims have the exact width 89% of the time, and unbounded ones are right as a
lower bound 89% of the time.

Scored against opendbc over nine platform and DBC pairs: **51% precision, 50%
recall**, on 313 hits against 312 missed and 303 extra. The scorer skips
reference signals whose bits never moved, on the same reasoning it already
skips messages the drive never carried — of 4566 signals the DBCs name on
recorded messages, 4307 sat completely still, and counting those as misses
would measure how much of the car the driver exercised rather than how well the
detector works. That is the weakest detector here by some
distance, and it is reported last, over only the bits nothing else explained —
a verified claim always outranks bits that merely move together.

### How fast a field moves, and who the rate floor excludes

Every bit is held to a minimum transition rate before it may join a field, and
that one number decides what the detector is allowed to look at. It excludes
almost exactly the opposite of what one would guess. Over seven platforms,
taking each reference signal whose bits move at all:

| declared width | median rate of its low bit | moves on ≥2% of frames |
|---|---|---|
| 1 bit | 0.004 | 27% |
| 2 bits | 0.010 | 34% |
| 3 bits | 0.004 | 14% |
| 4 bits | 1.000 | 84% |
| 5–8 bits | 0.257 | 79% |
| 9 bits and up | 0.285 | 89% |

Narrow fields barely move. An ignition switch does not run through its states
for the fun of it, and a mode selector holds one value for a whole drive. A bit
that flips *fast* and narrow is nearly always the bottom of a wider number
instead. The 4-bit row is the exception that proves the rule: those are alive
counters, claimed by the counter detector long before this one runs.

So the floor came down from 0.02 to 0.01, which took precision from 48% to 51%
and recall from 49% to 50% — 43 more correct signals and none lost.

**But not for the reason the table suggests.** Of the 37 signals that turn from
miss to hit on one measured set, only 3 are three bits or narrower; 28 are
eight bits or wider. What the old floor was really cutting off was the *top* of
wide fields, where the rate decay runs out — bit 6 of a byte turns over a few
times in a drive, bit 9 of a 16-bit field fewer — so the claim stopped short of
the declared width and scored as a miss.

Narrow fields did not move, and cannot be moved by a threshold. A 2-bit enum's
high bit changes half as often as its low bit, so any floor that admits the
first still drops the second, leaving one surviving bit: a flag by definition.

Two other ways of reading such bits were tried against opendbc and both failed,
which is why neither is in the code:

- **Changes landing in the same frame.** If adjacent slow bits belong to one
  field they ought to move together. Over 1069 adjacent slow pairs, bits inside
  one signal share a transition 46% of the time and bits across a boundary 59%
  of the time. The test does not merely fail, it points the wrong way: inside
  an enum only the low bit moves on most steps, while two unrelated neighbours
  both react to the ignition. It measures the driver, not the layout.
- **Refusing to cut below a fast narrow piece.** If a fast 2-bit field is
  really the bottom of a wider one, suppressing the cut that made it should
  recover the wider field. It recovers nothing — identical scores to three
  figures at every threshold tried.

A slow narrow field and the flag beside it leave the same trace. Separating
them needs something a recording does not contain.

### What it cannot do

**Two fast fields side by side read as one.** The boundary is visible only
where the rate rises, and between two fields that both move on most frames it
does not. There is a test for this, asserting the merge, because the limit is
better recorded than discovered.

**A slow field is cut short.** A byte counting 0 to 199 is reported as seven
bits, because bit 7 turns over once in two hundred frames and once is not
evidence of anything. Longer traces push that out; they do not remove it.

**A narrow field that sits still is invisible.** Of the claims two or three
bits wide, 5% match a reference signal. Unlike the other two this is not a
threshold that could be loosened; the section above measures why.

### More drives, more range

That second limit is the one the corpus can attack. A field's high bits only
move once the value grows large enough to reach them, so one drive shows one
drive's worth of range. `canlens corroborate-pooled` reads a platform's
signals from every local segment at once, laying the frames end to end — not
deduplicating them, which is right for a secret that needs distinct payloads
and wrong for a signal, whose transition rates are read from consecutive
frames.

Over the Volkswagen group and Rivian, twenty segments against one takes
precision from 68% to 70% and recall from 58% to 60%. The headline gain is not
the rates but the coverage: **360 correct claims become 460**, because fields
that never moved on one drive move somewhere across twenty.

### The C-variable prior, tested and rejected

Signals are packed C variables, so widths of 1, 2, 4, 8, 16 and 32 bits should
dominate — and across every local DBC they do, at 86% of all signals. It is
tempting to round an unbounded width up to the next of those.

Measured, it is wrong far more often than right. Rounding up drops the score
from 0.63 to 0.27 on F1; even restricted to gaps of a single bit it is correct
12% of the time. Two reasons. The gap between what moves and what is declared
is usually five to seven bits, not one, so the next C width is rarely the
answer. And the prior does not survive the subset that matters: of the signals
canlens can actually see — the ones whose bits move — only 46% have a C-typical
width, because the wide majority of 1- and 2-bit fields are flags and enums
that sit still, while the ones that move are physical quantities scaled to fit,
at 9, 10, 12 or 13 bits.

The prior is real. It is just not a prior about the signals a trace can show
you.

### Shannon entropy, tested and rejected

BinaryInferno (NDSS 2023) finds field boundaries in general binary protocols by
comparing the Shannon entropy of adjacent bytes rather than how often bits
flip. Entropy is a genuinely different statistic, and different in a promising
direction: a 2-bit enum resting on one value for most of a drive has almost no
transition rate but plenty of entropy. It was worth testing on the field
canlens is worst at.

**As a liveness test it is excellent.** Taking each reference signal whose bits
move at all, across seven platforms:

| declared width | rate ≥ 0.01 sees | entropy ≥ 0.10 sees |
|---|---|---|
| 1 bit | 34% | 96% |
| 2 bits | 39% | 98% |
| 3 bits | 40% | 100% |
| 5–8 bits | 95% | 100% |
| 9 bits and up | 99% | 100% |
| **all** | **78%** | **99%** |

**As a boundary test it is worthless.** Per-bit entropy is the entropy of a
bit's duty cycle, and it saturates: anything not heavily skewed sits near 1.0.
Comparing a bit to the one below it, the ratio has a median of 1.00 both inside
a field and at a boundary. Cutting where it rises by half catches 4% of real
boundaries while splitting 3% of real fields, which is no discrimination at
all. Transition rate, over the same pairs, has a median of 0.58 inside a field
against 0.92 at a boundary and cuts 18% against 3%.

Combining them fails for a reason worth stating: **a bit admitted by a
statistic that cannot segment it has nowhere to be cut.** Using entropy for
liveness and rate for boundaries admits every slow narrow field and then runs
them together into long merged claims. Scored over nine platform and DBC pairs,
every threshold tried is worse than the rate floor alone:

| liveness | boundary | precision | recall | F1 |
|---|---|---|---|---|
| rate ≥ 0.01 | rate ×3.0 | **51%** | 50% | **0.504** |
| entropy ≥ 0.10 | rate ×3.0 | 46% | 52% | 0.489 |
| entropy ≥ 0.50 | rate ×3.0 | 47% | 47% | 0.471 |
| entropy ≥ 0.90 | rate ×3.0 | 43% | 34% | 0.378 |
| entropy ≥ 0.50 | rate ×1.5 | 37% | 44% | 0.399 |

**The faithful form fails for a different reason.** BinaryInferno compares the
entropy of a byte's *value distribution*, which spans 0 to 8 bits rather than
saturating, so it was tested separately. The statistic does carry information,
and in the direction the paper describes: where a field spans a byte edge the
upper byte's entropy is lower, median ratio 0.56, against 1.00 at a real edge
between two fields. But as a cut it is unusable — the best threshold catches
14% of real edges while wrongly splitting 40% of the fields that span one.

The deeper problem is that the rule can only ever produce byte-aligned
boundaries, and CAN signals are not byte-aligned. Of 4539 reference signals,
**413 start on a byte boundary and have a length that is a multiple of eight;
4126 do not**. BinaryInferno's own evaluation notes it does better on payloads
because more of their fields are byte-aligned. That assumption is what makes it
work there and what makes it inapplicable here.

Entropy is not a worse statistic than transition rate. It answers a different
question — *has this bit ever carried information* rather than *where does one
number end* — and only the second question segments a payload.

### What is and is not claimed

**Counters** must advance by a constant step *and* walk their whole range. Two
filters enforce that. The step must be coprime with the field width — without
it, a single toggling bit at the top of an n-bit window is a mathematically
perfect counter of step 2^(n-1) that only ever holds two values, and every
even step on a power-of-two field is a version of that mistake. The field must
also have been seen holding at least half its possible values. A third filter
runs after the checksum search: a counter is never reported inside a byte a
checksum explains. A CRC is linear, so when nothing else in the message moves
the CRC byte is an affine image of the alive counter, and two of its bits walk
0..3 as convincingly as a real counter — every Jeep Grand Cherokee message with
a J1850 CRC showed such a phantom 2-bit counter before this was added.

**Multiplexors** are the one finding that is about the *rest* of the payload.
A multiplexed message reuses its bits for different signals depending on a
selector field — the Jeep Grand Cherokee's 0x3E0 sends its VIN that way, byte
0 cycling 0, 1, 2 and bytes 1-7 carrying a different slice of the string under
each value. Read without the selector it looks like noise; read with it, every
slice is constant. The detector groups the frames by each candidate field
(bytes, nibbles, and 2- and 3-bit fields at either end of a byte) and asks how
every other bit behaves *within* a group: a bit whose meaning depends on the
selector is constant inside every group while differing between them, or is
still in a good share of the frames and moving in another good share. Counters
and slowly changing states behave the same in every group, because the groups
interleave in time. Two things are required beyond that, and both came from
what the first version reported. The selector must be on a **schedule** — each
value revisited at a steady interval, with no long absences — because a
validity flag or the sign bits of a value crossing zero also sort the frames
into groups with different contents, but never regularly. And a bit that
repeats every two, three or four frames *inside* a group is locked to a finer
cycle than the selector, not moving: that is the CRC of a static message seen
through a 3-bit window over its 4-bit counter, and it was most of what the
Volkswagen platforms showed before the check existed.

Two more rules came from scoring against opendbc rather than from reading
output. **Every statistic is taken over the frames that belong to a group**,
never over the whole trace. Up to a twentieth of the frames carry a selector
value too rare to keep, and the inference the detector makes — that a bit
constant inside every group must therefore *differ between* groups — only
follows if "overall" means the grouped frames. Rivian's 0x247 was claimed as a
fifteen-layout message whose thirty-eight dependent bits were zero in all
fifteen and carried data only in the twenty-five frames outside them. And **an
all-zero payload is not a layout**: at least two selector values must carry
something, because zero is the universal idle pattern, and without the rule
any signal that alternates between carrying data and sitting at zero satisfies
the still-here-moving-there test. The Audi A3's 0x0AF holds a 16-bit value
that is zero on 53% of frames and 256-511 on the rest, and bit 8 of that value
was claimed as a selector over its own low byte. Together the two rules took
the corpus from 814 multiplexed messages to 675, and raised agreement with
opendbc from 40% to 67% without losing a single match.

The count is per value rather than a veto, because a wide selector
legitimately leaves most of its values empty: Volkswagen's 0x3FB has twenty
values of which twelve carry nothing and the other eight are a real layout set.

A third rule handles counters. A counter advancing every frame is a
relabelling of the frame index, so grouping by its low bits groups by frame
index modulo something — and any signal whose activity is periodic then looks
selector-dependent. The counter is not a selector; it is merely locked to one.
But counters cannot simply be excluded, because the VIN selector *is* one: it
cycles 0, 1, 2 and the counter scan duly reports it. What separates the two is
that the VIN's selector **determines** the content, each value mapping to a
fixed slice, while a locked counter only **gates** it. So a selector lying
entirely inside a counter is believed only on a byte's worth of
constant-per-group evidence. One such bit is not enough, since an idle/active
flag is constant per group all by itself. The eight opendbc-confirmed
selectors carry between 14 and 45 such bits; 199 of 675 detections had a
selector inside a counter and none at all.

The finding names the selector, its values, and which bits depend on it:

```
  mux      8 bits @ bit 0, 3 values, 35 bits depend on it
           ▁▄█▁▄█▁▄█▁▄█▁▄█▁▄█▁▄█▁▄█
           ████████████████████ 100.0% of frames carry a listed value
           value   0: 200 frames
           value   1: 200 frames
           value   2: 200 frames
```

The ruler marks the selector `M`. Checked against opendbc, where its DBCs
define a multiplexor: the Volkswagen MQB `VIN_01` (0x6B4, a 2-bit selector in
byte 0), Tesla's `VCFRONT_LVPowerState` (0x221, 5 bits in byte 0) and the
Chrysler/Jeep VIN message are all found, and reported as the span that
actually moves rather than the whole byte the search found it in. An earlier
version claimed the byte, on the reasoning that the byte is what a DBC names.
Scoring against those same DBCs showed that reasoning was simply wrong:
Volkswagen declares `VIN_01_MUX` as two bits and Tesla declares
`VCFRONT_LVPowerStateIndex` as five, so the whole-byte form matched none of
them. What a trace cannot recover is how wide the field was *declared* — a
five-bit selector that only ever took two values is indistinguishable from a
one-bit one — so what is claimed is the bits in use.
Hyundai's `EMS12` (0x329, a 2-bit selector switching six bits) is not: six
bits are below the byte's worth of dependent bits the detector demands, and
on a one-minute segment the switched signals never moved. A selector that is
not a bit field at all — Volkswagen's 0x15A alternates two layouts on a
function of its counter — is not claimed either.

A counter that lies entirely inside the selector or its dependent bits is
dropped: the VIN's byte 0 is numerically a counter modulo 3, and each ASCII
slice cycles with it — seven "2-bit counters" on that one message before the
detector existed. A counter that merely has its lowest bit locked to a
two-frame schedule is kept.

**Checksums** are only reported when a named algorithm *reproduces* the byte.
The library is `sum8`, `sum8_complement`, `xor8`, `sum8_addr`,
`sum8_addr_len`, Honda's
4-bit `honda_nibble`, and CRC-8 with
polynomials 0x07, 0x1D (SAE J1850), and 0x2F; 16-bit CRCs and the AUTOSAR
E2E Profile 1/11 form with a solved-for Data ID are searched separately. The simplest algorithm that
works wins, so a plain sum is never dressed up as a CRC.

`xor8` reports as ambiguous, shown `xor8@?`. XOR is self-inverse: if byte 7 is
the XOR of bytes 0-6, then byte 0 is equally the XOR of bytes 1-7. Every
position verifies, and a single trace cannot say which one the protocol calls
the checksum. The last byte is reported by convention and every candidate
position is kept.

### Honda protects a message with four bits

Every search above is byte-wide, and Honda's checksum is half a byte: a sum
over every nibble of the identifier and the payload, subtracted from 8 and
written into the low nibble of the last byte. canlens could not see it however
long it looked, and Honda was the starkest gap in the corpus — 97% of its
messages carried a byte that moved like a checksum with nothing to explain it.

Implemented from openpilot's description, it reproduces Honda traffic on sight.
Coverage across the 25 Honda and Acura platforms goes from 3% to 64%, and on a
single platform tested directly against every message it reaches 95–99%.

It also forced two things wider than itself. A checksum hypothesis now carries
a bit offset and width, because a byte index cannot describe half a byte, and
the filter that drops counters inside checksum bytes became bit-granular — the
*other* nibble of that byte is ordinary data, and Honda often counts there.

### A constant is not a check

The nibble detector immediately claimed messages on Ford, GM and Mazda. Every
one was a message with a single distinct payload: the field never changes, so
a constant matches a constant, and with four bits that happens one time in
sixteen by chance. Corpus-wide, 1350 of 3325 nibble claims rested on fewer
than eight distinct payloads.

Checking the byte-wide algorithms found the same flaw, smaller only because
eight bits make a coincidence sixteen times rarer. Both searches now require
enough distinct payloads for the match to *be* evidence.

How many is enough depends on the width. Each distinct payload a fixed
algorithm reproduces is one independent check worth as many bits as the field,
so the chance of a wrong algorithm surviving k of them is about
2⁻ʷᵏ. Asking for 32 bits of agreement beyond the first check gives five
distinct payloads for a byte and nine for a nibble. `MIN_EQUATIONS` is the
wrong bar here — it is sized for a *solved secret*, and using it cost findings
at eight bits while staying too lenient at four.

The gate costs something real and it is worth stating: five checksums the DBCs
confirm are no longer reported, because those messages carry fewer than five
distinct payloads and the trace genuinely cannot check them. The scored set
under-represents the benefit, since it contains none of the platforms where the
false positives were.

`sum8_addr` was derived from the corpus rather than from a document. Solving
for the additive constant that reproduces the checksum byte on the eleven
messages `tesla_model3_party.dbc` names one on gives exactly `(address & 0xFF)
+ (address >> 8)` every time, at a 100% match rate. It differs from
`sum8_addr_len`
only by the missing length term, and the two can never both fit the same
message, since a payload length is between 1 and 64 and so never vanishes
modulo 256. With it, Tesla scores 100% precision and 100% recall against its
DBC where it previously found nothing at all.

It was called `tesla` until a corpus survey contradicted the name. It fires on
**31 platforms across six makes** — Lexus, Mazda, Nissan, Subaru, Tesla and
Toyota — of which Tesla is two, and Subaru leans on it hardest at 1200 of the
1220 checksums found on an Outback. It had been named after the first car it
was found on, which made the name a claim about origin that the corpus does
not support. A name is a hypothesis like any other here, and this one failed.

`sum8_addr_len` was renamed on the same reasoning, though it failed the test
less badly. It was `toyota`, and it does fire on the Toyota group plus Mazda
and Perodua, both of which build on licensed Toyota platforms — 60 platforms
in all. But the arithmetic carries nothing Toyota-specific: it is `sum8_addr`
with the payload length added. A marque name on a generic sum is a claim about
origin that the arithmetic does not make, and keeping one of the pair named
after a car while its near-twin was renamed would have left the more
misleading half standing.

### Volkswagen, and the difference one unknown makes

Volkswagen's MQB bus is AUTOSAR Profile 22 in structure — CRC-8 0x2F over the
payload but the CRC byte, then a Data ID appended, selected from a sixteen-entry
list by the counter. canlens already searched for exactly that and still missed
half of it: 13 of the 26 checksums `vw_mqb.dbc` names on a Golf Mk7 segment.

Every one of the 13 had exactly **16 distinct payloads**. That is not a
coincidence, it is the shape of a static message whose only moving content is
its counter — and sixteen unknowns fitted to sixteen equations always fit, so
the evidence gate refused them, correctly.

What the gate could not see is that on many of those messages the whole
sixteen-entry list is *one repeated byte*. That is one unknown, not sixteen,
and fifteen equations of confirmation rather than none. Trying the constant
form first recovers them and needs no counter at all, since one constant
reproducing every frame implies it reproduces every counter value's frames.

The messages whose sixteen entries genuinely differ are still refused when all
they offer is sixteen distinct payloads, and that refusal is right: a single
segment cannot check them. Several segments together can — see
[Pooling segments for evidence](#pooling-segments-for-evidence).

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

This is also a preview of why the corpus matters: `sum8_addr_len@7 100%` holding
across 69 messages of one segment is suggestive, but holding across thousands
of segments from hundreds of different drivers is proof — and that is
`corroborate`'s job, not `infer`'s.

## 7. Corroborate

Everything above works on one trace. This is the layer the corpus exists for.

```bash
canlens corroborate KIA_EV6                       # every message, one line each
canlens corroborate KIA_EV6 --address 0x211 --bus 0
```

Each hypothesis `infer` produced is scored by how much of the platform stands
behind it — and **devices count more than segments**. The corpus stores a
segment under `<device>/<route>/<index>`, and a device is one physical car.
Eight hundred segments from one car establish what that car does; eighty from
thirty cars establish what the platform does. Every piece of evidence carries
both counts:

```
crc16     e2e_p05 @ bytes 0-1 (little, data ID 0xFA11; 5/5 seg, 5/5 dev, established)
```

Tiers (`ESTABLISHED_SUPPORT`, `PARTIAL_SUPPORT`, `MIN_DEVICES` in
`corroborate/consensus.py`, heuristics like every other threshold here):

| tier | means |
|---|---|
| `established` | ≥90% of the segments carrying the message agree, from ≥3 cars |
| `consistent` | ≥90% agree, but fewer than 3 cars — true of everything seen, not shown to generalise |
| `partial` | 50–90% |
| `weak` | under 50%; treat as a single-trace guess |

Two hypotheses claiming the same bits with different parameters — two
strides for one counter, two Data IDs for one CRC — are both kept and both
marked **contested**; the evidence counts say which one the corpus backs.

### A bus number is a port, not a bus

Everything pooled above is grouped by `(bus, identifier)`, and the bus number
comes from the logger's port assignment. Nothing guarantees the wire went into
the same socket on the next drive.

It often did not. Asking, for every bus of every segment, which bus in another
segment of the same car shares the most identifiers, **293 of 5868 comparisons
name a different number than the one logged**, across 12 of the 31 platforms
holding more than one segment. The KIA EV6 disagrees with itself 42% of the
time, the NMS Passat 33%, the Taos 24%; several platforms have segment pairs
where the same number shares no identifiers at all.

So `corroborate` reconciles the numbers before it pools anything:

```
KIA_EV6: 11 segments from 10 devices, 249 messages

4 buses, identified by the identifiers they carry:
  bus 0: 151 identifiers over 11 segments, 10 devices; logged as 0 in 6, 1 in 5
  bus 1:  70 identifiers over 11 segments, 10 devices; logged as 0 in 5, 1 in 6
  bus 2:   8 identifiers over  6 segments,  5 devices; logged as 2 in 6
  bus 3:  20 identifiers over  5 segments,  5 devices; logged as 2 in 5
  15 (segment, bus) pairs were logged under another number and have been moved.
```

The EV6 was recorded in two wirings, and the reconciliation finds it without
being told: two buses each appear under both numbers, roughly half the
segments each. `--logged-buses` turns the reconciliation off.

**What it is worth.** On the EV6 the number of messages seen in *every* one of
the eleven segments goes from **1 to 141**, and the median segments behind a
message from 5 to 11. Before, each bus was split in half and every finding
rested on the drives that happened to be wired one way. The message count falls
from 404 to 249 because the 155 extra were the same messages counted twice,
once under each number. On the Golf, where the numbering was always consistent,
nothing changes at all.

### How a bus is identified

**The signal is the identifier set.** Two recordings of one bus carry mostly
the same identifiers and two different buses carry almost none in common:
across the corpus the medians are 0.99 and 0.04.

**Payload width was tried as a second signal and dropped.** Counting only the
shared identifiers that also agree on width gives 92.8% against 92.9% of pairs
retained — indistinguishable. A message keeps its length wherever it appears,
so width says nothing the identifier had not already said. It is recorded
because the obvious next move is to add features, and this one measurably does
not help.

**Matching is an assignment, not a threshold.** One bus cannot be two buses, so
each segment's buses are matched one-to-one against the identities already
known, maximising total overlap rather than each picking its own best partner.
That also avoids a calibration trap: a threshold would need ground truth, and
the only ground truth available is the logged number, which is the thing in
doubt. Under assignment the best pairing beats the runner-up by a median of
0.65.

**Rare messages are left out.** A signature counts only messages seen at least
32 times, the same cutoff `infer` uses. A message that appeared once on one
drive and not on the next makes two recordings of the same bus look less alike
than they are; including everything split Rivian into seven buses where it has
six and invented relabellings on three other platforms. Matching the cutoff
also means identifying buses from frames and from cached inference cannot give
different answers — checked across all 31 multi-segment platforms.

**The majority keeps its name.** An identity is labelled with the number it was
most often logged under, so output still reads like the bus numbers everyone
knows. Sorting by support first matters more than it sounds: on the Audi A3 one
segment's "bus 1" shares no identifiers with the other nineteen, and letting
whichever identity appeared first claim the number renamed nineteen segments to
describe one oddity. Now the nineteen keep the name and the one moves.

### Pooling segments for evidence

The corroboration above pools *conclusions*: how many segments independently
reached the same finding. That works when each segment could reach it alone.
Some cannot, and Profile 22's sixteen-byte Data ID list is the case that
proves it. Sixteen unknowns fitted to sixteen distinct payloads is a fit with
nothing left over, so a static message carrying only its counter can never
settle the claim — and pooling conclusions cannot rescue it, because every
segment of such a message shows the same sixteen payloads and so solves the
same sixteen bytes. Sixteen segments agreeing on a fit that was unfalsifiable
in each of them is still unfalsifiable.

What is needed is more *equations*, which means pooling the payloads
themselves:

```bash
canlens corroborate-pooled VOLKSWAGEN_GOLF_MK7
```

```
VOLKSWAGEN_GOLF_MK7: 30 Profile 22 lists solved from pooled segments,
2 of which no single segment carried enough evidence for

* bus 0 0x65D: e2e_p22 @ byte 0 (100.0% of 87 frames; 87 distinct payloads
  pooled from 40 segments, 24 devices)
    best single segment offered 20 distinct payloads, pooling gives 87
    data IDs by counter value: AC B3 AB EB 7A E1 3B F7 73 BA 7C 9E 06 5F 02 D9
```

Nothing is relaxed. The same detector and the same evidence gate see a larger
matrix, reduced to distinct contents so that segments repeating each other
contribute nothing and segments carrying new content contribute equations.

### Two things this got wrong first

**Deduplication destroys the sequence a counter is.** Reducing to distinct
contents is exactly what supplies the extra equations, and it also scrambles
the frame order. On the Golf's `GRA_ACC_01` the deduplicated pool of 125
payloads has its counter nibble running 4, 5, … 15, 0, 1, 2, 3, 7, 8, 13, 14 —
no stride survives, `find_counters` rightly reports nothing, and the list
search was silently left with no field to group by. Each deduplicated row
still carries its own counter value, so grouping works once the field is
known; only *finding* the field needs frames in the order they arrived. The
counter is therefore located on one segment's ordered frames.

**A message another segment could explain is still worth reporting.** The
first version filtered those out, on the grounds that a finding the ordinary
path already makes is not news. But whether a single segment can settle a
message depends on which drive you happened to analyse: `ESP_33` offers 16
distinct payloads on one Golf segment and 78 on another. Filtering hid
twenty-eight real lists. They are reported, with a mark on the ones pooling
was actually necessary for.

### Whether to believe it

Two checks, neither of which the solver can arrange for itself. Of the 30
lists, 27 are on messages `vw_mqb.dbc` names a checksum on, all at byte 0 and
all reproducing every frame. And twelve of the messages appear on two buses,
because the gateway copies them — twelve independent solves over different
frames, agreeing on all sixteen bytes in every case.

Pooled findings are reported on their own and do not feed back into the
per-segment results or into `truth score`, which still measures what one
segment can do.

### When pooling correctly finds nothing

Run it on the Rivian or the EV6 and it reports no lists at all, over twenty
and eleven segments respectively. That is the right answer, not a shortage of
data, and the tool says so:

```
RIVIAN_R1_GEN1: pooled 20 segments from 19 devices and found no Profile 22 Data ID list.
This platform's checksums are e2e_p11 (4152) -- none of which hides a sixteen-byte
secret, so none of them needs pooling to be checked.
```

Each platform family sticks to one scheme. Rivian is Profile 11 throughout,
the EV6 is Profile 5 with a little Profile 11, and the Golf is Profile 22
throughout. Only Profile 22 hides sixteen bytes; Profile 11 hides one and
Profile 5 two, needing nine and ten distinct payloads against Profile 22's
twenty-four. A single segment clears those easily, which is why Rivian already
scores 100% precision and recall against its DBC without any of this.

So pooling is not a general strengthener. It is the answer to one specific
shape of problem — a secret large enough that an ordinary message cannot
constrain it — and saying "fetch more segments" to a platform that does not
have that problem would be wrong advice.

### What only the corpus can show

The per-bit line under the strip is the agreement decile per bit, and a `!`
marks a **rare bit**: constant in most segments, moving in some.

```
  ######## ######## ##**+++: *+.+:... ****:*.. #**++::: ...
  99999999 99999999 99999999 98966999 889669!9 99999696 ...
  rare bits (constant in most segments, moving in some): [38, 102, 103, 130, 131, 175]
```

A single trace classifies those as padding. Across cars they are states that
rarely change — a door, a mode, a warning — and no one recording contains
enough of them to tell. That list is a to-do list for the bit selector.

### A finding the layer surfaced on its first run

KIA_EV6 has 11 local segments from 10 cars, yet every E2E-protected message
shows `5/11 seg, 5/10 dev`. Splitting the segments by whether bus 0 carries
0x211 and comparing address sets per bus:

```
                B-bus0  B-bus1  B-bus2
  A-bus0 ( 70)    0.00    0.29    0.00
  A-bus1 (149)    0.90    0.00    0.05
  A-bus2 ( 20)    0.01    0.00    0.04
```

The 149-ID main bus is numbered **1** in one group of cars and **0** in the
other — and the 70-ID bus carrying every E2E message exists in only one
group. One platform key, two harness configurations, one of which never sees
the ADAS bus. The presence numbers were honest; the reason is wiring. Keying
on `(bus, address)` is the right behaviour given that, but it means support
for bus-0 messages is capped at the cars wired to carry it. Aligning buses
across configurations by address-set overlap is a natural next feature and
deliberately not done blind.

### The gap it exposed, and what closing it found

RIVIAN_R1_GEN1 (20 segments, 19 cars) at first yielded 287 counters and
**zero checksums against 1,251 checksum-shaped bytes**. Adding the AUTOSAR
E2E Profile 1/11 detector (below) turned that into:

```
e2e_p11 CRCs: 432 of 540 messages; Data ID == CAN identifier for 431
tiers: established 392, consistent 40; contested 0
counters: 432 x (4 bits, wraps at 15); contested 0
```

**Rivian's Data ID is the CAN identifier itself**, fed low byte then high
byte — Profile 11 `DATAID_BOTH`. 431 of 432 messages obey it; the exception
is `bus 5 0x31A` with Data ID 0x84, established across four cars, and worth
a look. That rule was not assumed: the detector tries the identifier form
first and claims it only when the recovered byte comes out equal to
`addr & 0xFF`.

### E2E Profiles 1 and 11: a CRC-8 over a Data ID that is never sent

Per AUTOSAR_PRS_E2EProtocol the CRC is SAE J1850 (0x1D) computed first over
the Data ID bytes — low then high for `BOTH`, low then a zero byte for
`NIBBLE` — and then over every byte but the CRC ([PRS_E2E_00082],
[PRS_E2E_00505], [PRS_E2E_00506]). The Data ID is solved for: the register
after the ID bytes takes one of 256 values, so every candidate is scored on a
sample of frames at once and the one that fits is verified on all of them.

Two honest limits. **One trace cannot tell the single-state modes apart** —
NIBBLE, BOTH and LOW each map 256 IDs onto the same 256 register states, so
whichever the sender used, exactly one ID in every mode fits. They are one
hypothesis, `e2e_p11`, with `p01_low_id()` giving the Profile 1 LOW reading.
Only the identifier form (checkable against the address) and ALT (two states
alternating with counter parity) are individually falsifiable.

**The final XOR was settled by data, not by the flowchart.** The CRC library
XORs 0xFF on output and a first reading of the spec suggested the same. But a
final XOR is absorbed into an equivalent start state for any *fixed* length,
so one message cannot tell — only structure across messages of different
lengths can. On 94 Rivian messages of three widths, register-from-0x00 with
no final XOR recovers the CAN identifier for all 94; every other convention
yields noise. The detector uses what the wire showed.

### Every profile the spec defines, gated by what a frame can hold

All eight profiles of AUTOSAR_PRS_E2EProtocol FO R19-11 are implemented,
each from its own section of the text:

| profile | CRC | header | Data ID | detector |
|---|---|---|---|---|
| 1 / 11 | CRC-8 J1850 | 2 B | implicit, solved | `e2e_p11`, `e2e_p01_alt` |
| 2 / 22 | CRC-8 0x2F | 2 B | implicit list of 16, one per counter value, solved | `e2e_p22` |
| 5 | CRC-16 0x1021 | 3 B | implicit, solved | `e2e_p05` |
| 6 | CRC-16 0x1021 | 5 B + Length | implicit, solved, appended **high byte first** | `e2e_p06` |
| 4 | CRC-32P4 | 12 B + Length | explicit | `e2e_p04` |
| 7 | CRC-64 | 20 B + Length | explicit | `e2e_p07` |

`HEADER_BYTES` in `infer/profiles.py` is the single statement of what
applies where: a profile is never tried on a payload narrower than its
header, so a Profile 6 is not attempted on an 8-byte frame. Profiles carrying
an explicit Length are gated on it before any CRC is computed — it must be
at least the header and at most the payload — and the CRC is bounded by it,
which is what makes padded CAN FD frames come out right.

**The gate that mattered more than the width gate.** On first contact with
the corpus, Profile 6 was "found" on 139 of 230 platforms and Profile 22 on
136 — including Toyota, which uses a byte sum. A CRC hypothesis with a
solved-for secret is only evidence when the data constrains the secret: each
distinct payload content is one 8-bit equation, and a static message with an
alive counter has exactly sixteen — the size of a Profile 22 list. Every
solved-secret detector now requires at least `secret_bytes + 8` distinct
contents. After it:

```
profile     messages  platforms
e2e_p11         4283         27
e2e_p22          993         26
e2e_p05         3152         21
e2e_p06            0          0
e2e_p04            0          0
e2e_p07            0          0
platforms with no E2E profile: 175 of 230
```

Profile 22's 26 platforms are the VW / Audi / Skoda / Seat family, as
expected — plus two outside it, at full match rate:

```
TESLA_MODEL_3: 1 of 90 messages, min match 1.000
TESLA_MODEL_Y: 3 of 388 messages, min match 0.994
JEEP_GRAND_CHEROKEE_2019: 1 of 189 messages, min match 0.998
VOLKSWAGEN_GOLF_MK7: 35 of 227 messages, min match 1.000
```

Profiles 4, 6 and 7 are verified on spec-built frames only; nothing in this
corpus uses them on CAN, which is itself the honest finding.

One claim corrected in the process: appending the Data ID *after* the data
does not make a final-XOR convention testable. The CRC table is linear over
GF(2), so a constant XOR on the CRC is absorbed into every list entry
uniformly. Profile 22's recovered IDs are relative to the 0xFF/0xFF
convention, exactly as Profile 11's are to its own.

### The alive counter my detector rejected

E2E alive counters run **0..14** in four bits — 0x0F is reserved
([PRS_E2E_00504], `Counter %= 15`). Under a natural modulus of 16 the 14→0
wrap is one odd step per cycle: fourteen good steps in fifteen is 93.3%,
just under the 95% bar, and the same wrap skips one LSB flip so the exact
prefilter rejected it too. The most standard counter in automotive was
invisible by construction. Counters may now wrap below 2^length, reported as
`4-bit counter @ bit 8 mod 15`, with two guards: the only early wrap tried is
the one the data's maximum implies, and it is admitted only if it uses the
field's top bit — otherwise a 6-bit window over two constant zero bits fits
mod 15 perfectly and would claim padding as counter.

Corroboration reads the inference results cache, so a platform pass costs
milliseconds per segment once `cache build` has run: 11 EV6 segments in
0.7 s. Building that cache is dearer for a platform where most messages are
E2E-protected, since each is a solved search; the results are versioned
(`INFER_VERSION`) and rebuilt when the detectors change.

## 8. Truth — scoring the engine

Everything up to here produces claims. This scores them.

opendbc carries community DBC files for many corpus platforms: someone
reverse-engineered the bus by hand and wrote down the counters, the checksums
and the multiplexors. That is an answer key, so inference can be *measured*
rather than argued about.

```bash
canlens truth dbc /path/to/opendbc/vw_mqb.dbc
canlens truth score <segment>/rlog.zst --dbc /path/to/opendbc/vw_mqb.dbc
```

```
reference rivian_primary_actuator.dbc: 67 messages, 40 counters, 40 checksums, 0 multiplexors
bus 0 (best identifier overlap): 50 messages in the trace, 30 also in the
reference, 37 reference messages this drive never carried

field           hit  missed  extra  near  precision   recall
counter          26       0      0     0      100%     100%
checksum         26       0      0     0      100%     100%
overall          52       0      0     0      100%     100%
```

There is no automatic mapping from a corpus platform to a DBC file, so the
file is named explicitly. The bus is chosen by identifier overlap, since a DBC
describes one bus and the trace's numbering is the logger's; `--bus` forces it.

### What the numbers are allowed to mean

Three rules keep them honest, and each one exists because the naive version
lies:

**A reference message the drive never carried is not a miss.** It is absent
data. Counting it would make recall a measure of how much of the car the
driver happened to exercise.

**A kind the reference never names is left out of the rates entirely.** The
older Honda and Acura DBCs annotate neither counters nor checksums. Scoring
canlens' 44 counter findings against `acura_ilx_2016_nidec.dbc` would report a
precision of 0% when the truthful answer is "this reference cannot say". Those
claims are reported as unevaluated instead.

**A claim that overlaps a reference field without matching it exactly is
`near`, never a hit.** A counter reported one bit wide of the truth is not
agreement, and folding it in would make the headline number meaningless.

### A disagreement is evidence, not a verdict

The reference is itself reverse-engineered. It goes stale, community DBCs
disagree with each other, and their authors routinely leave counters unnamed
because naming them was not the point. So every disagreement is printed in
both directions, with the message and the bit positions, for a person to
judge:

```
  0x041 Airbag_03: checksum not found -- reference has Airbag_03_CRC (checksum, bits 0-7, 8 bits)
  0x0A7 Motor_11: claimed counter 4bit@8 -- reference names none
```

The first is a real gap: canlens does not implement the CRC-8 variant
Volkswagen uses. The second is almost certainly canlens being right and the
DBC being incomplete, because MQB puts a counter in that nibble on nearly
every message. The tool does not pretend to know which is which.

### The bit numbering that has to be exactly right

Every comparison here is a comparison of bit positions, so reading a DBC's
numbering correctly is the whole game. cantools reports `Signal.start` in the
DBC's own numbering, which is already the flat Intel index `analyze` produces.
A little-endian signal runs *upward* from it. A big-endian one runs
*downward*, and wraps to bit 7 of the next byte when it falls off the bottom
of the current one.

The obvious reading — that a big-endian start is an MSB0 "sawtooth" index
needing conversion — is wrong, and wrong quietly. Measured against cantools'
own decoder by setting one bit at a time and seeing which signal moved, it
disagreed on 895 of 1070 signals. The rule above agrees on all 2925 signals of
the 58 DBCs in opendbc, in both byte orders.

---

## 9. From Python

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
        print(message, checksum)     # "sum8_addr_len @ byte 7 (100.0%)"
```

Bit numbering is **MSB-first within each byte**, matching how CAN signal
layouts are conventionally written: bit 0 is the most significant bit of
byte 0.

---

## 10. The workbench

```bash
pip install -e ".[gui]" && canlens gui
```

Three screens, chosen from the tabs along the top.

### Data — the landing screen

Everything `canlens corpus` does, plus deletion:

- every platform with its upstream size and what is held locally, holdings
  first, filtered by the same `*`/`?` wildcards (`TOYOTA*`, `*EV6`)
- **Fetch** downloads the selected platforms, capped by the segment spinner,
  on a worker thread with a progress bar and a working **Cancel** — the
  queued work stops and transfers already in flight finish rather than
  leaving `.part` files behind
- **Delete local** removes what is selected, after a confirmation naming the
  count and the space it frees. Selected *segments* win over the selected
  platform, so a handful can be dropped without emptying the car
- double-click a local segment to open it in the heat map

A fetch is refused outright if the estimate exceeds free disk, rather than
filling the disk and failing partway.

Nothing is decoded until the heat map is actually opened. Selecting a row is
not a request to decode it — that distinction is worth a few seconds of
startup and one avoided schema fetch.

### Corroborate

The `canlens corroborate` command as a screen. Pick a platform (only those
with local segments are offered), press **Corroborate**, and the pass runs on
a worker thread with a progress bar — milliseconds per segment from the
results cache, but a platform can have thousands.

The table is one row per message, by identifier: segments, devices, width,
layout agreement, rare-bit count, and every hypothesis with its tier. A
`bus,can id` filter takes the same wildcards as everywhere else.

Select a row for the consensus strip. **Colour is the consensus class;
opacity is agreement** — a bit every segment classifies the same way is solid,
one the platform splits over fades toward the background. **Rare bits are
outlined in orange**, because their consensus colour is the constant white
that hides exactly what makes them interesting. Below it, every hypothesis
with its evidence, tier, and contested flag.

### Heat Map

Everything the CLI prints, arranged for the reverse-engineering loop:

- **left** — every locally fetched segment, labelled by platform
- **centre** — the whole bus as one bit matrix, rows ordered **by identifier**:
  standard IDs ascending, then extended, grouped by bus. Dividers mark each
  bus change and the step into extended IDs
- **strip** — the selected message magnified, with inferred counter,
  checksum and multiplexor fields outlined and labelled
- **layouts** — for a multiplexed message only, one strip per selector value
  directly under the main strip, each classified over that value's frames
  alone and labelled with the value and its frame count. The selector's
  column is boxed on every row. The whole-message strip blends the layouts
  together: the VIN's three ASCII slices make bytes 1-7 look busy, while the
  layout rows show three constant slices. A signal that exists under one value
  only is green on that row and white on the others. Beyond twelve values the
  rows pan
- **selection** — drag on the strip to pick any bit range; the plot and the
  summary line follow live
- **bottom** — findings, and the selected field's values over time

The matrix is drawn with pyqtgraph's `ImageItem`, which takes the bit classes
as a numpy array directly. That is why this is a desktop app rather than a web
one: the analysis already produces the array the renderer wants, so there is no
serialisation step between them.

The magnified strip exists because of a scale problem worth knowing about. At
186 messages a single row of the matrix is under three pixels tall, so a field
overlay drawn there is invisible however correct its coordinates are. The
matrix is the map; the strip is the detail.

### Filtering what is shown

The toolbar takes a four-field filter:

```
vehicle , segment , bus , can id
```

```
TOYOTA_PRIUS,*,CAN0,*      every segment and identifier of that car, on bus 0
*,*,*,0x1*                 identifiers beginning 0x1, on any vehicle or bus
KIA_EV6,*,*,0x2??          three-digit identifiers 0x200-0x2FF
*,*,CAN2,0x18DA*           UDS diagnostics on bus 2
TOYOTA_PRIUS               a vehicle; trailing fields default to *
```

Two wildcards, and only two. **`*`** matches any run of characters including
none, so `0x1*` selects 0x1, 0x11, 0x112, 0x123, 0x124 and so on but not
0x211. **`?`** matches exactly one, so `0x2??` is 0x200–0x2FF and nothing
shorter or longer. Everything else is literal — including `[`, which a glob
library would quietly read as a character class.

Vehicle and segment narrow the list on the left; bus and identifier narrow the
rows in the matrix, and the status bar says how many were hidden. `Clear`
restores everything without re-reading the trace.

Matching is case-insensitive, and each field accepts the ways a value is
naturally written: `CAN0`, `CAN_0`, `bus 0` and `0` all mean bus 0, and both
`0x211` and `211` name that identifier. Identifiers carry no leading zeroes,
which is what makes `0x1*` work as a prefix.

### Selecting a field by hand

Drag on the magnified strip to select any range of bits. The line beneath
reports what you picked and what is in it:

```
bits 16–27 (12)   min 2   max 3912   648 distinct
bits 0–7 (8)      min 0   max 255    256 distinct   counts by 1 on 100% of frames
```

The counter test is the same one `infer` uses, so a range picked by hand is
judged on exactly the criteria a reported counter was — it will tell you when
your guess counts, and stay quiet when it does not.

Selection snaps to whole bits: a range from 3.7 to 8.2 describes nothing.
Opening a message seeds the selection from its most interesting inferred
field, so the plot says something before the first drag.

This is the loop for finding signals the inference layer cannot name. Pick a
run of `active` bits, watch the plot, and see whether it moves like a speed, a
temperature or an angle.

Replotting is instant because the payloads are held in memory once the segment
is decoded — a drag never goes back to the trace file.

### Naming a field and exporting it

Type a name for the current selection and press Enter. Named signals appear in
green on the strip, labelled underneath it, and in the list below.

Labels are packed into lanes: each takes the topmost line whose previous label
ends before this one begins. On a 32-byte payload a name is twenty-odd bits
wide at normal zoom, so fields a few bits apart cannot share a line — and the
lanes repack as you zoom, because a label is drawn at a fixed pixel size while
a bit is not. Naming a range that overlaps an
existing one supersedes it — two definitions of the same bits cannot both be
right, and a PDU database containing both is invalid rather than merely untidy.

**Export PDU database…** writes BoAt's `pdu_db.schema.json` format
(schema_version 1.0), so the result loads in BoAt's PDU editor, replays through
the gateway, or diffs against a database built from a real DBC. Also available
without the GUI:

```bash
canlens export pdu-db <segment>/rlog.zst -o prius.json
```

Everything found goes in, not just what you named. Inferred counters and
checksums are exported as signals carrying their evidence:

```json
{
  "SignalName": "CRC", "Length": 16, "StartPos": 0, "ByteOrder": 0,
  "Comment": "canlens: e2e_p05 little-endian, data ID 0xFA01, reproduces 100.0% of 1200 frames"
}
```

and a message whose CRC identified as Profile 5 is exported with `isE2E: 5` —
on the EV6 that is 162 of 163 messages.

**`StartPos` needs no translation.** The DBC and Vector convention numbers an
Intel signal's start bit as `byte_index * 8 + offset_from_that_byte's_LSB`,
which is exactly what `BitOrder.INTEL` produces. A bit index read off the
workbench *is* a `StartPos`. The same holds for Motorola. This is the payoff
from defaulting to Intel numbering earlier.

What a single trace cannot supply is written neutrally rather than guessed:
`Direction` is 0, both routing columns are null, and `signal_routes` is empty,
because establishing a route means observing both sides of a gateway.

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

They are also **rare**, which is why a few spot checks will miss them. Across
211 locally fetched segments, KIA_EV6 has none at all (highest address 0x4FE)
and TOYOTA_PRIUS has 28, concentrated in a handful of segments. Where they do
appear they are UDS diagnostics — `0x18DA<target>F1`, the ISO 15765-4 29-bit
physical addressing scheme, with `0x18DB33F1` as the functional broadcast —
i.e. a scan tool talking to ECUs, not normal traffic. To find one:

```bash
canlens corpus which ~/data/canlens/segments/*/*/*/rlog.zst | head
```

then look for a segment whose bus 2 carries `0x18DAxxF1` addresses.

Bit axes tick on **byte boundaries** (8, 16, 24 …), never pyqtgraph's default
decimal steps, which put gridlines through the middle of bytes. A 32-byte CAN
FD payload has too many boundaries to label, so the labels thin to every 16 or
32 bits and the rest stay as unlabelled minor ticks.

Headless check, the same way BoAt verifies its Qt client:

```bash
QT_QPA_PLATFORM=offscreen canlens gui
```

## 11. Gotchas

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

## 12. Development

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
| `infer/` | counters, 8-bit checksums, 16-bit CRCs, the E2E profiles and multiplexors working; signal boundaries still to do |
| `corroborate/` | cross-segment agreement with device-weighted evidence — **working**; cross-platform and bus alignment to do |
| `truth/` | opendbc DBCs as a reference, precision and recall against them — **working**; no automatic platform-to-DBC mapping |

`analyze` deliberately stops at measurement. It will tell you a bit flips on
94% of frames; it will not tell you it is a counter. That claim requires
checking the stride against the cycle time and confirming it across many
segments — which is `infer`'s and `corroborate`'s job, and the reason the
corpus is the point of this project rather than an afterthought.
