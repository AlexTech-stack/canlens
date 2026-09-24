# Approaches and results

Every idea tried against CAN traces in this project, and what measurement said
about it. Positive results and negative ones, because a negative result costs
the same to obtain and is worth as much the second time somebody has the idea.

This file is about **analysing traces**: reading structure out of payloads.
Corpus plumbing, caching, GUI and export are out of scope except where they
changed what an analysis could see.

## Contents

- [How to read the numbers](#how-to-read-the-numbers)
- [1. Per-bit measurement](#1-per-bit-measurement)
- [2. Signal boundary detection](#2-signal-boundary-detection)
  - [2.1 What works](#21-what-works)
  - [2.2 The narrow-field limit](#22-the-narrow-field-limit)
  - [2.3 Rejected: reading slow bits another way](#23-rejected-reading-slow-bits-another-way)
  - [2.4 Rejected: entropy as a boundary statistic](#24-rejected-entropy-as-a-boundary-statistic)
  - [2.5 Rejected: CAN-D's conditional-flip cut rule](#25-rejected-can-ds-conditional-flip-cut-rule)
  - [2.6 Rejected: splitting a merged claim](#26-rejected-splitting-a-merged-claim)
  - [2.7 Rejected: rounding widths to C types](#27-rejected-rounding-widths-to-c-types)
- [3. Byte order](#3-byte-order)
  - [3.1 What is and is not observable](#31-what-is-and-is-not-observable)
  - [3.2 Rejected: deciding the order per message](#32-rejected-deciding-the-order-per-message)
  - [3.3 Rejected: a second pass in both orders](#33-rejected-a-second-pass-in-both-orders)
  - [3.4 Adopted: decide once per bus](#34-adopted-decide-once-per-bus)
  - [3.5 Corrected: why Rivian and the Prius scored badly](#35-corrected-why-rivian-and-the-prius-scored-badly)
  - [3.6 Why counters and checksums were right anyway](#36-why-counters-and-checksums-were-right-anyway)
- [4. Counters](#4-counters)
- [5. Checksums, CRCs and AUTOSAR E2E](#5-checksums-crcs-and-autosar-e2e)
  - [5.1 Adopted schemes](#51-adopted-schemes)
  - [5.2 The two gates that mattered more than the algorithms](#52-the-two-gates-that-mattered-more-than-the-algorithms)
  - [5.3 Corrected and settled by data](#53-corrected-and-settled-by-data)
- [6. Multiplexors and layouts](#6-multiplexors-and-layouts)
  - [6.1 The signature](#61-the-signature)
  - [6.2 Four false-positive modes, each measured](#62-four-false-positive-modes-each-measured)
  - [6.3 Rejected: running the signal detector inside each layout](#63-rejected-running-the-signal-detector-inside-each-layout)
  - [6.4 Adopted: read the whole-byte constant each value selects](#64-adopted-read-the-whole-byte-constant-each-value-selects)
- [7. Corroboration across segments and platforms](#7-corroboration-across-segments-and-platforms)
  - [7.1 A logged bus number is a port, not a bus](#71-a-logged-bus-number-is-a-port-not-a-bus)
  - [7.2 Pooling conclusions, and where it fails](#72-pooling-conclusions-and-where-it-fails)
  - [7.3 Pooling payloads](#73-pooling-payloads)
  - [7.4 Pooling frames for signal boundaries](#74-pooling-frames-for-signal-boundaries)
  - [7.5 Agreement across platforms: the strongest ranking signal found](#75-agreement-across-platforms-the-strongest-ranking-signal-found)
- [8. Ground truth: getting the reference right](#8-ground-truth-getting-the-reference-right)
- [9. ISO-TP and UDS](#9-iso-tp-and-uds)
  - [9.1 Adopted in principle: multi-frame ISO-TP verifies itself](#91-adopted-in-principle-multi-frame-iso-tp-verifies-itself)
  - [9.2 A misreading this already causes](#92-a-misreading-this-already-causes)
  - [9.3 Rejected: finding SingleFrames from the header](#93-rejected-finding-singleframes-from-the-header)
  - [9.4 Adopted: validate the SingleFrame against the protocol](#94-adopted-validate-the-singleframe-against-the-protocol)
  - [9.5 Where the residual false positives are, and the rule that separates them](#95-where-the-residual-false-positives-are-and-the-rule-that-separates-them)
  - [9.6 The UDS layer, and what it would take](#96-the-uds-layer-and-what-it-would-take)
  - [9.7 The constant that decides it is the match rate, not the reassembly](#97-the-constant-that-decides-it-is-the-match-rate-not-the-reassembly)
- [10. Open, and deliberately not attempted](#10-open-and-deliberately-not-attempted)
- [11. Recurring lessons](#11-recurring-lessons)

## How to read the numbers

Every figure here came from measurement, not from judgement. Two things make
that possible and both bound what the numbers mean.

**Ground truth is opendbc.** `truth/score.py` compares a `MessageInference`
against a DBC and reports precision and recall per field kind. Three rules keep
it honest: a reference message the drive never carried is not a miss; a kind the
reference never names is excluded from the rates entirely (older Honda and Acura
DBCs annotate no counters, and scoring 44 findings against zero would report 0%
precision where the truthful answer is "cannot say"); and a claim that overlaps a
reference field without matching it exactly is *near*, never a hit.

**A reference signal that never moved is not a miss either.** Of 4566 signals the
DBCs name on recorded messages, 4307 sat still. Scoring those would measure the
driver rather than the detector.

So the scores below measure agreement with one human-written database over the
fields a particular drive exercised. They are comparable to each other and are
not an absolute truth about the vehicle.

Status labels: **Adopted** (in the code), **Rejected** (measured, not shipped),
**Partial** (shipped with a stated limit), **Corrected** (a claim this project
made and later disproved).

---

## 1. Per-bit measurement

| Idea | Status | Outcome |
|---|---|---|
| Per-bit transition rate | Adopted | The backbone statistic. Everything downstream is built on it. |
| Per-bit Shannon entropy | Adopted as a measurement | Kept in `analyze/bits.py` for triage. As a *detector* it fails — see §2. |
| Splitting the "active" band at rate 0.10 | Adopted | — |
| Inter-arrival timing | Adopted | — |

**Splitting the active band.** Pooled over 24 corpus segments, the old active
class held 24.7% of all bit positions — more than any other class — spread over a
40× range of transition rates, so it said very little. A bit's rate falls
monotonically with its significance inside a numeric field; measured mean rate by
within-byte position runs `.343 .242 .180 .151 .137 .132 .121 .105`. Cutting at
0.10 separates a value's fast low bits from its slower upper bits. The measured
distribution is smooth with no trough, so the boundary is a choice rather than a
discovery: 0.10 is a tenth of all frames, and lands near the band's median rate
of 0.078, splitting it 56/44. The payoff is that field structure became visible —
byte 0 of the Prius `0x210` went from `##+++++:` to `##**+++:`, a clean descending
gradient that is a numeric field showing its own boundary.

**Timing.** The only timestamp available is the enclosing capnp event's
`logMonoTime`, so frames batched into one event share a stamp. That puts a floor
on resolvable jitter, which is why `cyclic` is judged on relative spread rather
than absolute.

**Echoes are not data.** `CanData.src >= 128` marks a frame the device itself put
on the wire. They are 30–46% of a segment and reproduce the source bus's address
set exactly. Including them inflates every frequency, entropy and corroboration
statistic downstream. Dropped by default.

---

## 2. Signal boundary detection

The hardest problem here, and the one with the most rejected ideas. An ordinary
signal offers no arithmetic to verify — a wheel speed is just bits that move, and
the only question is which move together.

### 2.1 What works

**Cut where the transition rate rises.** Inside a numeric field the rate falls
away from the least significant bit, so a boundary shows up as the rate *rising*
again where the next field's low bit begins. Swept against opendbc: a rise factor
of 1.5 gives 40% precision, 2.0 gives 52%, 3.0 gives 57%, and beyond that it
flattens. `RISE = 3.0` puts the start bit right on 210 of 411 real signals.
**Adopted.**

**The far end has no marker, so say so.** A field's high bits stop moving because
the value never grew large enough to reach them, and a trace cannot tell that
from the field ending there. Of 411 signals whose bits move, only 157 have every
declared bit moving; 209 start exactly where the DBC says and run out of evidence
early. The rate profile of a genuine 16-bit signal reads

```
0.30 0.29 0.30 0.30 0.27 0.20 0.12 0.06 0.03 0.02 0.01 0.01 0.00 0.00 0.00 0.00
```

and nothing in it says sixteen rather than ten. So a claim carries `bounded`:
stopped by another moving field, a counter, a checksum or the payload end means
the width is the width; anything else is an explicit lower bound. The flag earns
its keep — bounded claims have the exact width 89% of the time, and unbounded
ones are right as a lower bound 89% of the time. **Adopted.**

**Lowering the rate floor from 0.02 to 0.01.** Precision 48% → 51%, recall 49% →
50% over nine platform/DBC pairs: 43 more correct signals, none lost. **Adopted**
— but the reason is not the obvious one, see §2.2.

**Value jumpiness as a precision filter.** Every other statistic reads *how often
a bit flips*; this one reads *how far the field's value moves between frames*. A
coherent physical quantity moves in small steps, so a claim whose value moves more
than 10% of its observed range on more than 20% of frames is not one signal.

| filter | precision | recall | F1 |
|---|---|---|---|
| none | 59% | 62% | 0.604 |
| length ≥ 4 (control) | 61% | 53% | 0.563 |
| low-bit rate ≤ 0.2 (control) | 67% | 44% | 0.528 |
| **jump share ≤ 0.20** | **71%** | **58%** | **0.643** |

The two controls are what make it a result rather than a coincidence: a length
floor and a rate floor both *lower* F1 at every threshold tried, so the statistic
is not length or rate in disguise. It removes 29% of all signal claims across 38
corpus segments, dropping roughly six false claims for every true one (1057
extras against 175 hits on the sweep). A 2D sweep leaves F1 on a plateau of 0.637
to 0.645 for every share at or below 0.30, so the constants are not knife-edge.
**Adopted** (`infer/smoothness.py`).

### 2.2 The narrow-field limit

Worth stating separately because it is a property of CAN traffic, not of this
code, and it will otherwise be rediscovered by anyone loosening a threshold.

Measured over seven platforms, taking each reference signal whose bits move:

| declared width | median LSB rate | moves ≥ 2% of frames |
|---|---|---|
| 1 bit | 0.004 | 27% |
| 2 bits | 0.010 | 34% |
| 3 bits | 0.004 | 14% |
| 4 bits | 1.000 | 84% |
| 5–8 bits | 0.257 | 79% |
| 9 bits and up | 0.285 | 89% |

Narrow fields barely move. An ignition switch does not cycle its states; a mode
selector sits on one value for a whole drive. A bit that flips fast *and* narrow
is nearly always the bottom of a wider number instead. The 4-bit row is the
exception that proves it — those are alive counters, which the counter detector
claims long before the signal pass runs.

So the rate-floor gain above did **not** come from narrow fields. Of 37 signals
that turned from miss to hit, only 3 are 3 bits or narrower and 28 are 8 bits or
wider. The old floor was truncating the *tops of wide fields*, where the rate
decay runs out. Narrow fields cannot be recovered by any floor: a 2-bit enum's
high bit moves half as often as its low bit, so whatever admits the first drops
the second and leaves a single bit, which is a flag by definition.

The standing limit: **of claims 2 or 3 bits wide, 5% match a reference signal.**
A slow narrow field and the flag beside it produce the same evidence, and nothing
in a trace separates them.

### 2.3 Rejected: reading slow bits another way

**Co-occurring transitions.** If adjacent slow bits belong to one field they
should move in the same frame. Over 1069 adjacent slow pairs, bits within one
signal share a transition **46%** of the time and bits across a boundary share
one **59%** of the time. The test is not weak — it points the wrong way. Inside
an enum only the low bit moves on most steps, while two unrelated neighbours both
react to the ignition, so co-occurrence measures the driver's actions rather than
the layout. **Rejected.**

**Refusing to cut below a fast narrow piece.** If a fast 2-bit field is really
the bottom of a wider one, suppressing the cut that created it should recover the
wider field. It recovers nothing: scores identical to three figures at every
threshold tried. **Rejected.**

### 2.4 Rejected: entropy as a boundary statistic

Entropy looked like the answer to narrow-field blindness, since a 2-bit enum
resting on one value has no transition rate but plenty of entropy.

As a **liveness** test it is much better than the rate floor: it sees 99% of
reference signals against 78%, and 96% of 1-bit fields against 34%.

As a **boundary** test it is worthless. Per-bit entropy is the entropy of a duty
cycle and saturates near 1, so the ratio between adjacent bits has a median of
**1.00 both inside a field and at an edge**. Cutting on it catches 4% of
boundaries while splitting 3% of real fields; transition rate over the same pairs
cuts 18% against 3%.

Combining them is worse than either, for a reason that generalises: **a bit
admitted by a statistic that cannot segment it has nowhere to be cut.** Entropy
liveness with rate boundaries admits every slow narrow field and runs them
together, taking F1 from 0.504 to 0.489 at best and 0.378 at worst. **Rejected.**

**BinaryInferno's byte-value entropy** was tested separately, since the entropy of
a byte's value distribution does not saturate. It carries real information in the
direction the paper describes — a field spanning a byte edge shows a median ratio
of 0.56 against 1.00 at a true edge — but as a cut, the best threshold catches 14%
of edges while splitting 40% of spanning fields. The rule can only produce
byte-aligned boundaries, and **4126 of 4539 reference signals are not
byte-aligned**. The assumption that makes the method work on general binary
protocols is what makes it inapplicable to bit-packed CAN. **Rejected.**

### 2.5 Rejected: CAN-D's conditional-flip cut rule

CAN-D's Algorithm 1 cuts on conditional flip probabilities rather than the raw
rate ratio, with a second two-bit-lookahead term. Added as extra cut conditions
and scored over six Volkswagen-group platforms (chosen because their DBC is
entirely little-endian, so an Intel-order detector is judged fairly):

| variant | whole-field F1 |
|---|---|
| rate-ratio baseline | 0.533 |
| + term 1 | 0.487 |
| + term 2 | 0.416 |
| + both | 0.414 |

The same holds after removing constant bits first, which is how CAN-D builds its
features; the baseline reproduces to three figures on the condensed vector, so
what changed is the terms and not the plumbing.

On CAN-D's *own* per-bit metric the terms take boundary recall from 51% to 79%
and precision from 94% to 37%. They are recall rules, and **boundary recall was
never the binding constraint** — whole-field precision is, and a wrong cut
destroys the field on each side of it rather than one.

The mismatch is structural. Algorithm 1 is a classifier feeding a global
optimizer, not a segmentation rule: CAN-D's Step 2 re-decides every cut against a
cut penalty, and the paper names the binary output a drawback precisely because
it costs that step its flexibility. Bolted on as extra cut conditions — the only
way to use it without also building Step 2 — it adds cuts with nothing to
counterbalance them. Recorded alongside: the existing rate-ratio rule scores
0.660 on that per-bit metric, **so the cut rule is not the weak part of signal
detection.** **Rejected.**

### 2.6 Rejected: splitting a merged claim

Two follow-ups to the jumpiness filter, both measured and neither shipped.

**Split at the best interior cut.** The best interior split of a false claim
improved its smoothness by more than 0.2 on only **12%** of them — too weak to
cut on. **Rejected.**

**Carry-based boundary test.** Clean on synthetic ramps and destroyed on real
signals, because real values are signed and offset around a midpoint rather than
natural-binary. **Rejected.**

The jumpiness filter therefore stays a *filter*, not a segmenter: it ranks and
drops whole claims, and cannot split a region that is genuinely two fields.

### 2.7 Rejected: rounding widths to C types

Signals are packed C variables, so rounding an unbounded width up to the next of
1, 2, 4, 8, 16, 32 looks free. It drops F1 from **0.63 to 0.27**, and even
restricted to gaps of a single bit it is right 12% of the time. The gap between
what moves and what is declared is usually five to seven bits, so the next C
width is rarely the answer. And while 86% of all DBC signals do have a C-typical
width, only **46% of the ones canlens can see** do — the 1- and 2-bit fields that
dominate that statistic are flags and enums that sit still, while the fields that
move are physical quantities scaled to 9, 10, 12 or 13 bits. **Rejected.**

---

## 3. Byte order

### 3.1 What is and is not observable

**A field inside one byte is byte-order-invariant.** Not merely the same bits:
the same *integer*. A little-endian field runs LSB-first from its start and a
big-endian one MSB-first from its start, and inside one byte those coincide —
verified on all 36 single-byte layouts, and all 224 single-byte layouts have an
identical little-endian twin differing only in which end the DBC calls
`StartPos`. Single-byte fields are **83–91% of every reference DBC**.

Only byte-wrapping fields are genuinely at stake: 0% of Volkswagen's scorable
signals, 8% of Tesla's, 30% of Rivian's, 41% of the Prius'.

**Reversed-Motorola ordering.** Unpack MSB-first, *then reverse the whole vector
end to end*. Every big-endian signal then becomes a contiguous run whose rate
decays upward, exactly like an Intel-order field, with the DBC `StartPos`
recoverable from the run's top column. Verified on all 344 big-endian layouts
that fit in eight bytes. Plain MSB-first is **not** enough and fails quietly —
the rate climbs along the field instead of decaying, so a rule that cuts on a
rise fires *inside* fields rather than between them. **Adopted as the mechanism.**

### 3.2 Rejected: deciding the order per message

Within a byte both orders give the same adjacencies, so only the byte seams
differ, and one message rarely has enough seams. A seam objective gets
little-endian right 97% of the time and big-endian **38%** — worse than a coin.
Whole-message objectives reach 75% and are biased rather than discriminative.
**Rejected.**

### 3.3 Rejected: a second pass in both orders

Running a Motorola pass alongside the Intel one and keeping only the wrapping
claims it alone can express: **10 right against 78 wrong** across five platforms,
including 20 wrong on each Volkswagen platform where there is nothing to find.
Requiring 12 bits or more leaves 7 right against 29 wrong; requiring 16 leaves
none right. **Rejected.**

### 3.4 Adopted: decide once per bus

Byte order is a property of the bus, and that is what makes it decidable.
Measured across the 43 opendbc databases with 20 or more signals, the median
share of signals in their database's dominant order is **100%**, and 37 of 43 are
at least 90% one order. Assuming one order throughout costs a median of 0.00% of
signals, a mean of 0.10%, and **41 of the 43 databases pay nothing at all**.

The evidence is long fields. A field of nine bits or more cannot fit inside a
byte, so it appears as one contiguous run of decaying rate only in the order that
is correct; in the wrong order the seam splits it. Counting long fields under
each order and taking the larger count picks the right order on **8 of 9 buses**,
against 38% for the per-message test. The one it gets wrong is also the least
decided, at a 6% margin where the narrowest correct answer sits at 12% — which is
what `MIN_MARGIN = 0.10` exists to catch, leaving the bus undecided rather than
guessed.

Applied to detection, scored on one segment each:

| platform | precision | recall |
|---|---|---|
| Rivian R1, Intel | 21% | 19% |
| Rivian R1, decided | **43%** | **35%** |
| Toyota Prius, Intel | 12% | 18% |
| Toyota Prius, decided | **42%** | **59%** |

Volkswagen platforms get *worse* under Motorola, as they should, and that control
is what says this measures the bus rather than flattering the detector.
**Adopted** (`infer/byteorder.py`).

### 3.5 Corrected: why Rivian and the Prius scored badly

This project claimed that a DBC's big-endian share was the largest identified
factor in its signal score, based on a correlation across five platforms. That
was wrong. Most signals are single-byte and order-agnostic; Rivian and the Prius
simply have very few signals that move — **29 and 37 scorable against 133 on a
Golf**. The scores were low for want of data, not for want of a bit order.
Deciding the order still helps them (§3.4), but not for the reason first given.

### 3.6 Why counters and checksums were right anyway

A related question: how were a Rivian's counters and CRCs correct while the bus
was being read in the wrong order?

Checksums never touch bit numbering — they consume whole bytes and report a byte
index. Counters do read the bit matrix, but every Rivian counter is inside one
byte (169 counters: 159 four-bit E2E alive counters and 10 two-bit), so the
same-integer property of §3.1 applies. The Prius' 59 counters are likewise all
8-bit within one byte. This is not luck: AUTOSAR E2E puts the alive counter in a
nibble and the CRC in a byte, precisely so their position is unambiguous.

Residual exposure is a counter that *does* cross a byte boundary on a Motorola
bus: **2 of 598** over a nine-platform sample with 13 Motorola buses, both on a
Chevrolet Bolt, and both verified at ≥95% match so only the `StartPos` convention
is in question. **Partial** — signals were the only fields ever mis-read.

---

## 4. Counters

| Idea | Status | Outcome |
|---|---|---|
| Verify by differencing modulo width | Adopted | One step must dominate; `min_match` 0.95. |
| LSB prefilter at 0.95 rather than 0.40 | Adopted | See below. |
| Allow early wrap (mod 15) | Adopted, guarded | See below. |
| Exclude bits a checksum explains | Adopted | 404 phantom messages. |
| Exclude bits a multiplexor explains | Adopted | — |

**The prefilter cutoff.** A cutoff of 0.40, meaning merely "noisy", admitted 2729
start positions on a CAN FD segment where 0.95 admits 193 — and found exactly the
same 163 counters, four times slower.

**Early wrap.** AUTOSAR E2E alive counters run 0..14 (`Counter %= 15`), so under a
natural modulus of 16 the 14→0 wrap is one odd step per cycle: fourteen good
steps in fifteen is 93.3%, *under* the 95% bar, and the same wrap skips one LSB
flip so the exact prefilter rejected them too. Counters may now wrap early, with
two guards: the only early wrap tried is the one the data's maximum implies, and
it is admitted only if it uses the field's top bit. Without the second guard a
6-bit window over two constant zero bits fits mod 15 perfectly and claims padding
as a counter — which the first version did, reporting 432 × `6b@8`, all
contested. With it: 432 × 4 bits mod 15, zero contested.

**Phantom counters from CRC linearity.** A CRC is linear over GF(2), so when the
only moving content of a message is its alive counter, the CRC byte is an affine
image of that counter and two of its bits walk 0..3 exactly like a real 2-bit
counter. Every Jeep Grand Cherokee message with a J1850 CRC reported one; **404
messages corpus-wide** were affected before `outside_checksums()` existed. Expect
this class of bug wherever one field is a function of another.

**Phantom counters inside a multiplexor.** A selector that cycles 0, 1, 2 is
numerically a counter modulo 3, and every byte it multiplexes cycles with it —
the Jeep VIN message produced seven phantom 2-bit counters. `outside_multiplex()`
drops counters wholly inside the selector or its dependent bits, but keeps one
that merely has its lowest bit locked to a two-frame schedule.

---

## 5. Checksums, CRCs and AUTOSAR E2E

The rule throughout: **a hypothesis is reported only if it reproduces the observed
byte**, on nearly every frame. Nothing is inferred from a byte merely looking
random. Counters need a 0.95 match rate, everything else 0.99. The search runs
simplest-algorithm-first, so a plain sum is never dressed up as an exotic CRC.

### 5.1 Adopted schemes

**`sum8_addr` (originally named `tesla`).** Tesla's checksum is not a CRC at all.
Solving for the additive constant that reproduces the byte on the eleven messages
`tesla_model3_party.dbc` names one on gives exactly `(address & 0xFF) + (address
>> 8)` every time, at a 100% match rate. Tesla went from finding nothing to 100%
precision and 100% recall on both Model 3 and Model Y. Later renamed generically,
since it is a default construction rather than a Tesla one (`sum8_addr_len` is
the same form with a length term; the two can never both fit one message, a
payload length being between 1 and 64 and so never vanishing modulo 256).

*A footnote on that rename, because it was itself a measurement.* The scheme was
called `tesla` until a corpus survey contradicted the name: it fires on **31
platforms across six makes** — Lexus, Mazda, Nissan, Subaru, Tesla and Toyota —
of which Tesla is two, and Subaru leans on it hardest at 1200 of the 1220
checksums found on an Outback. A name is a hypothesis like any other, and that
one failed. `sum8_addr_len` (formerly `toyota`, 60 platforms) was renamed on the
same reasoning.

**Honda's nibble checksum.** Four bits, not eight: every nibble of the identifier
and the payload summed, subtracted from 8, written into the low nibble of the
last byte. Every search was byte-wide, so it could not be seen however long one
looked — Honda was the starkest gap in the corpus, 97% of its messages carrying a
byte that moved like a checksum with nothing to explain it. Coverage across 25
Honda and Acura platforms went from **3% to 64%**.

**Profile 22 with a one-constant Data ID list.** MQB is Profile 22 in structure
and was already searched, yet 13 of the 26 checksums `vw_mqb.dbc` names on a Golf
segment were missed. Every one had exactly 16 distinct payloads — sixteen
unknowns fitted to sixteen equations, which always fit, so the evidence gate
refused them and was right to. What the gate could not see is that on many of
those messages the whole sixteen-entry list is **one repeated byte**: one unknown,
needing nine distinct payloads rather than twenty-four, with fifteen equations of
confirmation. Checksum recall over nine scored pairs went from **53% to 78%**,
precision holding at 91%.

**Profile 5 Data ID recovery.** The Data ID is implicitly sent and never appears
on the wire, but a CRC is deterministic, so it can be solved for. A first attempt
swept all 65536 candidates; each low byte instead fixes an intermediate CRC and
the high byte is recovered by inverting one CRC step — 256 iterations rather than
65536. That inversion depends on the table's *low* bytes being a permutation of
0..255; the high bytes are not (only 128 are distinct) and indexing by them
returns the wrong entry silently. Both facts are now asserted. On the KIA EV6,
162 of 237 messages carry a Profile 5 CRC verifying at 100%, every Data ID solved
independently, and all 162 satisfy `data_id == (0xF8 + (addr >> 8)) << 8 | (addr
& 0xFF)`.

### 5.2 The two gates that mattered more than the algorithms

**Solved-secret evidence gate.** On first contact with the corpus, Profile 6 was
reported on **139 of 230 platforms** and Profile 22 on **136** — Toyota included,
which uses a byte sum. A solved-secret CRC hypothesis is only evidence when the
data constrains the secret: each distinct payload content is one 8-bit equation,
and a static message with an alive counter has exactly sixteen, the size of a
Profile 22 list. Requiring `secret_bytes + 8` distinct contents left Profile 6
reporting nothing and Profile 22 reporting 26 platforms — the VW/Audi/Skoda/Seat
family plus Tesla Model 3/Y and one Jeep — with Toyota clean again.

**Width-scaled evidence bar for fixed algorithms.** The Honda nibble detector
immediately claimed messages on Ford, GM and Mazda, every one a message with a
single distinct payload: a constant matches a constant, and at four bits that is
a one-in-sixteen coincidence. **1350 of 3325 nibble claims rested on fewer than
eight distinct payloads.** The byte-wide algorithms had the same flaw, smaller
only because eight bits make it sixteen times rarer. Each distinct payload a
fixed algorithm reproduces is one independent check worth as many bits as the
field, so surviving *k* of them by accident is about `2**(-width*k)`; asking for
32 bits beyond the first gives five distinct payloads for a byte and nine for a
nibble. The gate costs five DBC-confirmed checksums on messages too static to
check, moving measured precision 91% → 90% and recall 78% → 76%, and removes
roughly 1400 claims corpus-wide.

### 5.3 Corrected and settled by data

**The Profile 1/11 final XOR cannot be read off the spec.** The CRC library XORs
0xFF on output and a first reading suggested the same, but for a fixed-length
message a final XOR is absorbed into an equivalent start state — so a single
message cannot tell, and only structure across messages of different lengths can.
On 94 Rivian messages of three widths, register-from-0x00 with no final XOR and
the Data ID fed as `[addr & 0xFF, addr >> 8]` recovers the CAN identifier as the
Data ID for all 94; every other convention yields noise. Rivian's Data ID is its
CAN identifier on **431 of 432** protected messages across 19 cars; the exception
(bus 5 `0x31A` → `0x84`) is established across four of them and remains
unexplained.

**An appended Data ID does not make a final XOR testable.** A claim made along the
way and corrected: the CRC table is linear over GF(2), so a constant XOR is
absorbed into every list entry uniformly. The test that assumed otherwise now
asserts the absorption.

**NIBBLE, BOTH and LOW are indistinguishable in one trace.** Each maps 256 IDs
onto the same 256 register states, so exactly one ID in every mode fits. Reported
as one hypothesis. Only ALT is falsifiable — two states alternating with counter
parity — and is detected using the alive counter found first.

**Profiles 4, 6 and 7 are verified on spec-built frames only.** Nothing in this
corpus uses them on CAN.

---

## 6. Multiplexors and layouts

### 6.1 The signature

Group frames by a candidate selector's value and ask every other bit how it
behaves *within* each group. A dependent bit is constant inside every group while
differing between groups, or still in a good share of frames and moving in
another. A bit that is one signal throughout behaves the same in every group,
because the groups interleave in time and each samples the whole trace.

A multiplexor is a **transmission schedule**, not a vehicle state, and that is the
second half of the test: each value must recur at a regular interval. A field
that holds one value while a signal is valid and another while it reads 0xFFFF,
or the sign bits of a value crossing zero, also sorts frames into groups whose
contents differ — and those were the bulk of what an early version reported — but
nothing about *when* such a field changes is regular.

### 6.2 Four false-positive modes, each measured

**Claiming the whole byte searched in.** Justified by the argument that a byte is
what a DBC names. Scored against those same DBCs, the whole-byte form matched
**none** of the declared selectors, because Volkswagen's `VIN_01_MUX` is two bits
and Tesla's `VCFRONT_LVPowerStateIndex` is five. Trimming to the span that
actually moves took multiplexor agreement from nothing to 4 hits and 3 missed,
57% recall. *Reasoning about what a DBC "would name" is not evidence; scoring
against one is.* **Corrected.**

**Per-group statistics over the wrong frame set.** `dependent_bits` took its
per-group statistics over a group's frames but its overall statistics over the
whole trace. Up to `MAX_UNEXPLAINED` of frames carry a value too rare to keep, and
the inference only follows when "overall" means the *grouped* frames. Rivian's
`0x247` was claimed as a fifteen-layout message whose 38 dependent bits were zero
in all fifteen and carried data only in the 25 frames outside them; **123 of 814
corpus detections rested on this.** **Fixed.**

**An all-zero payload treated as a layout.** Any signal alternating between
carrying data and sitting at zero satisfied the still-here-moving-there test. The
Audi `0x0AF` holds a 16-bit value that is zero on 53% of frames and 256–511 on
the rest, and bit 8 of that value was claimed as a selector over its own low
byte. Requiring at least two selector values to carry something took the corpus
from 814 multiplexed messages on 165 platforms to **675 on 157**, with multiplexor
precision **40% → 67%** at unchanged 57% recall and all eight known ground-truth
matches surviving intact. Counted per value rather than as a veto, because a wide
selector legitimately leaves most values empty — VW's `0x3FB` has twenty values of
which twelve carry nothing and eight are a real layout set. **Fixed.**

**A counter locked to a periodic signal.** A counter advancing every frame is a
relabelling of the frame index, so grouping by its low bits groups by frame index
modulo something, and any signal whose activity is periodic then looks
selector-dependent. Counters cannot simply be barred — the VIN selector *is* one —
so a selector lying entirely inside a counter is believed only on a byte's worth
of constant-per-group evidence. Corpus 675/157 → **485/127**, all eight
ground-truth selectors surviving with identical extents. Evidence for this one is
weaker than the other two and the commit says so: scored precision and recall are
*unchanged* at 67% and 57%, because none of the 190 removed detections fall on a
message any scored DBC covers. What supports it is the structural argument plus
three hand-inspected examples. **Fixed, on weaker evidence.**

**Known and left unfixed:** if a counter's low bits stay perfectly in phase with a
periodic signal they can still be claimed as a selector. Real traces drift enough
that the schedule test catches it; a perfectly periodic one would not.

### 6.3 Rejected: running the signal detector inside each layout

The obvious next step, and it recovers almost nothing real, for two reasons that
are properties of CAN multiplexing rather than of this code. The flagship VIN
message is a table of **constants** — seven bytes per value, never moving — and a
detector built on transition rate finds nothing where nothing moves. The
genuinely moving mux signals opendbc declares, such as Tesla `0x221`'s state
fields, are **two bits wide**, which is exactly what the rate floor cannot see
(§2.2). Measured over 24 segments and 8 platforms, 51 multiplexed messages yielded
33 putative layout signals, and the one message with a DBC to check them against
recovered **none** of its 13 declared moving fields. **Rejected.**

### 6.4 Adopted: read the whole-byte constant each value selects

The hard part is not reading values but deciding where a constant field ends,
which a trace cannot see for the same reason it cannot see a signal's far end.
Three grouping rules were scored against the mux fields the DBCs declare, over
the VIN on five MQB platforms and the Tesla state message, matching on exact bit
set and selector value:

| grouping rule | hit | extra | F1 |
|---|---|---|---|
| maximal runs of dependent bits | 0 | 540 | 0.00 |
| **whole byte** | **315** | **8** | **0.95** |
| byte if a single run, else runs | 96 | 495 | 0.21 |

A run of dependent bits is *narrower* than the field a DBC declares, because a bit
that happened not to differ between values in this trace is not dependent — so
runs never line up and the first rule misses every one. A whole byte is fully
observed, so reporting it claims no more than the constant that is there. All 27
reference fields the winner missed are sub-byte fields in one Tesla message, which
no rule recovered: a two-bit field inside a byte that also carries other content
is invisible to a trace. **Adopted** (`infer/layouts.py`).

**Read the headline gain correctly.** Layout constants take the signal tally over
25 segments from F1 0.622 to 0.700 — but that is two detectors pooled, not
segmentation improving. Split on six VW MQB segments:

| claims scored | hits | extras | misses | P | R | F1 |
|---|---|---|---|---|---|---|
| rate-based signals | 296 | 83 | 259 | 0.781 | 0.533 | 0.634 |
| layout constants | 126 | 0 | 0 | 1.000 | 1.000 | **1.000** |
| combined | 422 | 83 | 259 | 0.836 | 0.620 | 0.712 |

A layout constant is a whole byte, fully observed, with nothing to infer about its
extent — a perfectly solved sub-problem, and 22% of all signal-kind claims across
60 segments. An easier question joined the same tally.

**A layout constant spends its byte.** Its bits must be spoken for before the
signal pass runs, in `_build` and again in `apply_bus_order`'s Motorola re-run.
Without both, the two stages claimed the same bits on **9 of 133 multiplexed
messages**, at worst a byte read as sixteen layout constants *and* as two 4-bit
signals. The byte-order decision is deliberately left out of that exclusion:
adding layout bits to its input moved the verdict on 2 of 176 buses, and
`MIN_MARGIN` is calibrated.

**A guard that fires on nothing, kept anyway.** A byte is reported only where the
selector is actually seen to change it. Holding a dependent bit is not sufficient,
because dependence is decided on *behaviour* and promises nothing about the
byte's value. Measured, the guard drops nothing over 60 corpus segments — 2750
claims either way. It is kept because the input does not promise otherwise, and
the number is recorded so nobody mistakes it for a win.

---

## 7. Corroboration across segments and platforms

The reason this project is built on a corpus at all: a hypothesis from a single
recording is a guess with a confidence number attached, and nothing can check it.

### 7.1 A logged bus number is a port, not a bus

It comes from the logger's wiring and does not always mean the same thing twice.
Asking, for every bus of every segment, which bus in another segment of the same
car shares the most identifiers: **293 of 5868 comparisons name a different number
than the one logged**, across 12 of the 31 platforms holding more than one
segment. The KIA EV6 disagrees with itself 42% of the time, the NMS Passat 33%,
the Taos 24%, and several platforms have segment pairs where the same number
shares no identifiers at all. Grouping by the logged number silently pools two
different buses and calls the disagreement evidence. **Adopted:
`corroborate/buses.py` derives a canonical label from the traffic.**

| sub-idea | Status | Outcome |
|---|---|---|
| Identify by identifier set | Adopted | Same bus across drives has median overlap 0.99; different buses 0.04 (2548 vs 5047 comparisons). |
| Add payload width as a second feature | **Rejected** | 92.8% of pairs retained against 92.9% — indistinguishable. A message keeps its length wherever it appears, so width says nothing the identifier had not. |
| One-to-one assignment rather than a threshold | Adopted | One bus cannot be two buses, and a threshold would need ground truth to calibrate while the only ground truth is the logged number, which is what is in doubt. Best pairing beats the runner-up by median 0.65. |
| Label the best-supported identity first | Adopted | On the Audi A3 one segment's "bus 1" shares nothing with the other nineteen; letting first-seen claim the number renamed nineteen segments to describe one oddity. 19 relabellings → 1. |
| Build signatures from the same 32-frame cutoff inference uses | Adopted | The two paths disagreed on 4 of 31 platforms and the inference-based answer was better every time — Rivian resolved to its actual six buses rather than seven. |

Measured on the EV6: messages seen in every one of eleven segments went from 1 to
141, median segments behind a message from 5 to 11, and the message count fell
404 → 249 because the extra 155 were the same messages counted once under each
number. The Golf, whose numbering was always consistent, is unchanged.

### 7.2 Pooling conclusions, and where it fails

`corroborate/consensus.py` scores each hypothesis by how many segments — and more
importantly how many distinct **devices**, meaning physical cars — agree. Eight
hundred segments from one car establish what that car does; eighty from thirty
cars establish what the platform does.

It also yields information no single trace contains: a bit constant in 95% of
segments and moving in 5% is not padding but a rarely changing state — a door, a
mode, a warning. Reported as *rare bits*. **Adopted.**

**Where pooling conclusions cannot help.** Profile 22 hides sixteen bytes, and a
static message carrying only its counter shows exactly sixteen distinct payloads
— sixteen unknowns fitted to sixteen equations, a fit with nothing left over.
Every segment of such a message shows the *same* sixteen payloads and solves the
*same* sixteen bytes, and sixteen segments agreeing on an unfalsifiable fit leaves
it unfalsifiable. What is needed is more **equations**.

### 7.3 Pooling payloads

`corroborate/pooled.py` gathers one message's payloads across many segments,
reduces them to distinct contents, and puts the union in front of the same
detector and the same gate. Nothing is relaxed; the evidence is simply larger.

On 40 Golf Mk7 segments from 24 devices it solves **30 Data ID lists**, two of them
on a message no single segment could settle — `ESP_20` at `0x65D` on both buses,
20 distinct payloads at best alone against 87 pooled. Two independent checks the
solver could not arrange for itself: 27 of the 30 are on messages `vw_mqb.dbc`
names a checksum on, every one at byte 0 and reproducing every frame; and twelve
messages ride two buses because the gateway copies them, giving twelve independent
solves over different frames that agree on all sixteen bytes every time.
**Adopted.**

Two things this got wrong on the way, both now tested:

- **Deduplication destroys the sequence a counter is.** `GRA_ACC_01`'s pooled
  counter nibble runs 4, 5, … 15, 0, 1, 2, 3, 7, 8, 13, 14, so `find_counters`
  reported nothing and the list search was silently left with no field to group
  by. The counter is now located on one segment's *ordered* frames.
- **Filtering out messages another segment could explain hid 28 real lists**,
  because whether one segment suffices depends on which drive you happened to
  analyse: `ESP_33` offers 16 distinct payloads on one Golf segment and 78 on
  another.

### 7.4 Pooling frames for signal boundaries

A field's high bits only move once the value grows large enough to reach them, so
one drive shows one drive's worth of range. `signals_across` lays a platform's
frames end to end — **not** deduplicated, which is right for a secret constrained
by distinct payloads and wrong for a signal, whose rates come from consecutive
frames. Each join adds one transition that never happened, which against tens of
thousands of frames changes no rate that matters.

Twenty segments against one takes precision 68% → 70% and recall 58% → 60%. The
real gain is coverage: **360 correct claims become 460**, because fields that never
moved on one drive move somewhere across twenty. **Adopted.**

### 7.5 Agreement across platforms: the strongest ranking signal found

MQB puts address `0x120` on a Golf, a Tiguan, an Audi Q3 and a Skoda Superb alike,
and the pooled Profile 22 work already proved those are the same message: 25
addresses solved on more than one platform agreed on all sixteen Data ID bytes.
If the message is the same, its layout is the same, so a boundary only one
platform proposes is probably an artefact of that platform's trace.

Over 386 start bits proposed on the 176 MQB messages that 3 or more of 17
platforms carry, scored against `vw_mqb.dbc`:

| support across platforms | proposed | correct | precision |
|---|---|---|---|
| one platform only | 104 | 21 | 20% |
| under half | 103 | 30 | 29% |
| half to 89% | 67 | 34 | 51% |
| 90% but not all | 13 | 11 | 85% |
| every platform | 99 | 84 | 85% |
| **all** | **386** | **180** | **47%** |

Precision rises **monotonically from 20% to 85%**, which is the whole claim: a
boundary's support is worth roughly as much as the boundary itself. Filtering by
tier:

| kept | precision | recall | F1 |
|---|---|---|---|
| everything | 53% | 47% | 0.500 |
| partial and better | 61% | 45% | 0.518 |
| established only | **73%** | 38% | 0.498 |

The F1 gain is small and is not the point. The point is the operating point:
three claims in four being right is a different kind of artefact from one in two,
and **no single-platform sweep reached past the low fifties** — the rate floor,
the rise factor and the C-width prior were all swept. **Adopted**
(`corroborate/boundaries.py`).

**It cannot invent a boundary.** Recall stays capped by `find_signals`; this ranks
that output rather than extending it.

**Widths get separate treatment**, because they disagree for a reason that is not
disagreement: a field's high bits only move once some driver exercises them, so
each platform reports what its own cars did. Every claim is a lower bound, so the
widest is the best lower bound, reported as "at least *n* bits". It is exactly
right 20% of the time against 17% for the median claim and 5% for the narrowest,
still short 67% of the time by a median of 3 bits, and overshoots 13% of the time
— which is the rate at which treating it as a bound is wrong.

---

## 8. Ground truth: getting the reference right

The scorer is itself an analysis of the DBCs, and it had one hard part.

**A DBC's big-endian start bit is not a sawtooth index.** cantools reports
`Signal.start` in the DBC's own numbering, which is already canlens' flat Intel
index. A little-endian signal runs upward from it; a big-endian one runs
*downward* and wraps to bit 7 of the next byte. Reading it as an MSB0 sawtooth and
converting — the obvious reading — disagreed with cantools' own decoder on **895 of
1070 signals**. The rule now in `truth/dbc.py` agrees on all **2925 signals of the
58 DBCs** in opendbc, in both byte orders.

**The name classifier was audited**, not assumed: over all 19222 distinct signal
names in the 58 local DBCs it agrees with a human reading of every name containing
`count`, `cnt`, `checksum` or `crc`, with no spurious hits. A bare "count" is
deliberately not enough, because `wheelPulseCount` and `country` are not message
counters.

**Matching on `(bits, mux_value)`** rather than bits alone, since a multiplexed
message puts different fields on the same bits under different values. Checked
against all 49 readable opendbc DBCs: every one of the 406 mux reference signals
is of kind *signal*, so no counter or checksum claim is made unmatchable by the
key, and no signal anywhere carries more than one multiplexer id.

A validation worth recording: Rivian scored against `rivian_primary_actuator`
gives **100% precision and 100% recall over 52 fields**, which independently
confirms the E2E Profile 11 work.

---

## 9. ISO-TP and UDS

A diagnostic transport is not a vehicle signal, but it occupies the same
payload bytes and the detectors here read it as one. It is also the one
structure in a CAN trace that carries its own verification, which makes it a
natural fit for this project's bar. Measured over all 803 local segments.

### 9.1 Adopted in principle: multi-frame ISO-TP verifies itself

ISO 15765-2:2016 segments a long message into a FirstFrame carrying a 12-bit
FF_DL, then ConsecutiveFrames whose SequenceNumbers start at 1 and increment
modulo 16 (9.6.3, 9.6.4). Both are checkable against the trace: the SN chain
must be unbroken and the reassembled length must reach FF_DL.

**22 401 complete reassemblies** across 223 `(bus, address)` pairs, in 379 of
803 segments. The confirmation is independent of how they were found: the spec
requires the receiver to answer a FirstFrame with a FlowControl frame before
ConsecutiveFrames may flow, and **98%** of the reassemblies had a FlowControl
from a peer address on the same bus while the FirstFrame was open — a fact the
scanner never used as a criterion.

Those are the numbers *before* an evidence bar, and 174 of the 223 do not
survive one; see §9.7. The shipped detector reports **21 209 transfers over 49
pairs, 100% FlowControl-backed**.

| bus | address | messages | FC-backed | reading |
|---|---|---|---|---|
| 1 | `0x7E8` | 6227 | 100% | UDS response; replies `0x41` and `0x62` |
| 1 | `0x7E0` | 2590 | 100% | UDS request; every one SID `0x22` |
| 1 | `0x085` | 5972 | 100% | Toyota, FF_DL 144 |
| 1 | `0x080` | 5966 | 100% | Toyota, FF_DL **740** — 106 CFs per message |

Audi Q3 carries the most (7529 messages over 40 pairs); six Toyota platforms
carry `0x080`/`0x085`.

### 9.2 A misreading this already causes

Toyota's `0x080` and `0x085` are reported today as
`4-bit counter @ bit 0 (98.1%)`. That nibble is the ISO-TP SequenceNumber. The
counter detector is not malfunctioning — an SN genuinely advances by one and
wraps, at 98% — it is describing a *transport* artefact as a vehicle signal.
Same class as the CRC-linearity phantom in §4: one field being a function of
another. Nothing is currently excluded on this basis.

### 9.3 Rejected: finding SingleFrames from the header

A SingleFrame's entire header is one nibble — high nibble 0, low nibble a
length ≤ CAN_DL − 1 — and ordinary traffic satisfies it constantly. Scanned
without further constraint: **8 216 707** matches corpus-wide against 22 401
genuine multi-frame messages. The service histogram from that scan is noise;
it showed OBD-II services `0x01`–`0x0A` at 100k–277k hits each, which is
simply messages whose byte 0 happens to fall in that range.

**Restricting by address does not fix it**, which is worth recording because it
is the obvious first move and it was measured:

| endpoint set | addresses | messages | named service | share |
|---|---|---|---|---|
| ever sent a `0x3x` byte | 1800 | 5 418 313 | 2 034 226 | 38% |
| proven + FlowControl peers | 1167 | 4 615 292 | 1 831 670 | 40% |
| proven by an FF+CF chain | 223 | 1 772 287 | 804 975 | **45%** |

45% on the strictest possible set. The sender's identity is not the evidence.

### 9.4 Adopted: validate the SingleFrame against the protocol

The frame has to be *consistent with its own SF_DL*, and the spec gives two
mutually exclusive ways for that to hold (10.4.2.1, 10.4.2.2):

- **padded** — DLC forced to 8, unused bytes filled with one repeated value.
  The spec's default is `CC16`, chosen to minimise stuff-bit insertion;
  `0x55`, `0xAA` and `0x00` occur in practice.
- **optimised** — no padding, and then `CAN_DL` must equal `SF_DL + 1` exactly
  (Table 12, normal addressing).

Anything else drops the ISO-TP presumption. Per frame this rejects **77.4%** of
what the header nibble admits (8 216 707 → 1 856 074). Applied to every frame
of an address, with at least one frame carrying real padding so the check has
something to bite on, it cuts **75 addresses to 26**.

**It rediscovers the diagnostic address map on its own.** Told nothing about
ISO 15765-4, the rule surfaced `0x720`–`0x723`, `0x7E0`, `0x7E1`, `0x7EA`,
`0x7EE`, `0x737` and `0x582`. That is the independent confirmation, and it is
what makes the rule credible rather than merely selective.

**It is complementary to §9.1, not an alternative.** None of the 26 overlap the
223 FF+CF-proven endpoints, because "every frame is a SingleFrame" excludes by
construction any address that also sends FirstFrames. One test finds endpoints
that segment, the other finds endpoints that only ever send short messages. A
detector wants the union.

### 9.5 Where the residual false positives are, and the rule that separates them

`0x00` is the weakest padding value, being the commonest byte in ordinary CAN
payloads. A second condition separates the set: **does SF_DL actually vary?** A
real endpoint sends requests of different lengths, so the length nibble moves
and the padding boundary moves with it. An ordinary periodic message whose
byte 0 is a fixed small constant and whose tail is always zero satisfies the
padding rule with a *frozen* SF_DL — a constant matching a constant, the same
trap the checksum evidence bar exists for (§5.2).

Of the 26 survivors: SF_DL varies on 9, padding is not only `0x00` on 6, and 4
satisfy both. Cross-tabulated against the diagnostic address range:

| | in `0x700`–`0x7FF` | outside | share diagnostic |
|---|---|---|---|
| SF_DL varies | 6 | 3 | 67% |
| SF_DL frozen | 4 | 13 | 24% |

The 13 frozen-and-outside are where the false positives live — `0x4D2`, `0x113`
and `0x3EC` all sit at `SF_DL = 1` with six zero bytes behind it, which is an
ordinary one-byte message, not a diagnostic request.

**But a frozen SF_DL is not disproof**, and the counter-example is in the data:
`0x737` sends 33 frames, all `SF_DL = 3`, padded with `0xCC` — spec-default
padding in the diagnostic range. A TesterPresent heartbeat genuinely is one
fixed-length message repeated forever. So variation is *positive evidence when
present*, never a required condition, exactly like the `bounded` flag on
signals (§2.1). Combining as "reject only `0x00`-padded **and** frozen" keeps 11
of 26 and costs three likely-real endpoints (`0x720`–`0x723` on some buses).

### 9.6 The UDS layer, and what it would take

Once a message is reassembled, ISO 14229-1:2013 makes the service byte
readable: a request carries the SID, a positive response carries SID + `0x40`,
and a negative response is `0x7F <SID> <NRC>`. On the FF+CF-proven endpoints
the services below the OBD-II noise floor are the shape of a real diagnostic
session — `0x3D` WriteMemoryByAddress (36 197), `0x22` ReadDataByIdentifier
(12 086), `0x10` DiagnosticSessionControl (7337), `0x11` ECUReset (7149).
Those do not arise by coincidence the way `0x01`–`0x0A` do.

So the UDS layer is reachable, but **only over messages the transport layer has
already verified**. Reading service bytes off unverified frames reproduces the
8.2-million-match failure of §9.3 one level up. The order is: reassemble,
confirm, then interpret — and a service whose SID is unknown is still worth
reporting as a request/response pair, since the `+0x40` relation is checkable
without knowing what the service does.

**Built for the multi-frame layer only** (`infer/isotp.py`); the single-frame
and UDS layers remain measured and unbuilt.

### 9.7 The constant that decides it is the match rate, not the reassembly

Reassembly alone is not an evidence bar, and the first build of the detector
proved it by reporting VW's `0x101` as an endpoint. That address is an ordinary
high-rate message whose byte 0 opens roughly 200 accidental FirstFrames in a
segment and completes two of them — and it even picked up a FlowControl
"answer" from a 29-bit address on an 11-bit conversation.

The fix is the number every other detector here already carries: the share of
FirstFrames that reached their promised length. Per address over 803 segments,
using FlowControl backing as the independent check:

| match rate | addresses | transfers | FC-backed |
|---|---|---|---|
| 0–5% | 137 | 1141 | 68% |
| 5–25% | 4 | 18 | 83% |
| 50–90% | 2 | 8 | 100% |
| **100%** | **80** | **21 234** | **99%** |

The distribution is bimodal with **nothing between 25% and 50%**, so the floor
is a plateau rather than a knife edge: 0.25, 0.50 and 0.75 all keep the same 49
addresses and 21 209 transfers, at 100% backing. `MIN_MATCH_RATE = 0.5` sits
mid-plateau. A second completion is also required (`MIN_MESSAGES = 2`), which
costs 33 addresses carrying one transfer each and takes backing from 99% to
100%.

The lesson generalises past ISO-TP: **a structure that reassembles is not
thereby verified.** What verifies it is the rate at which it reassembles when
it says it will, measured against everything else the same address does.

## 10. Open, and deliberately not attempted

Stated plainly rather than implied away.

**Not implemented:**

- **Scaling and units.** No factor, offset or unit is ever inferred.
- **Motorola for counters, checksums and multiplexors.** The bus's bit order is
  decided and applied to *signal* detection only. The others are verified
  arithmetic over whole bytes, so bit numbering does not move them (§3.6); their
  reported `StartPos` is still Intel.
- **Signals inside a multiplexed layout.** The selector is found, its layouts are
  drawn, and the whole-byte constant each value selects is read — but no signal
  detector runs within a selector value, for the reasons in §6.3. Sub-byte layout
  fields are not recovered.
- **Variable-length E2E.** Profiles 4 and 7 are claimed only where Length is fixed
  across the trace.
- **ISO-TP SingleFrames, and the UDS layer.** The segmented transport is built
  (`infer/isotp.py`, §9) and withdraws the counters it explains. SingleFrames
  and service bytes are measured and unbuilt; see §9.4 and §9.6.

**Known gaps with evidence pointing at them:**

- **Ford, GM and Mazda checksum schemes.** Their messages carry bytes that look
  protected and stay unexplained — the same shape Honda had before the nibble
  scheme was found. The nibble detector reached for them and had to be gated off
  (§5.2), which is a hint that something byte-wide is there and unmodelled. Not
  yet surveyed for a share.
- **Rivian bus 5 `0x31A`, Data ID `0x84`.** The one protected message of 432 whose
  Data ID is not its CAN identifier, established across four cars and unexplained.
- **E2E counters wider than 8 bits.** The counter search caps at `max_length = 8`,
  so 16- and 32-bit E2E counters are under-reported.
- **Odd early-wrap moduli.** Only the one wrap the data's maximum implies is
  tried (§4), so a counter wrapping at some other modulus is missed. Not
  surveyed for a count.
- **Bus alignment across harness configurations.** `KIA_EV6` is one platform key
  with two wirings, one of which never sees the ADAS bus. Bus identity is
  reconciled *within* a platform but not *between* platforms, so
  `corroborate/boundaries.py` matches fewer messages than it should — failing
  towards fewer comparisons, not towards wrong ones.
- **Corroborating layout constants across cars.** `consensus.py` tallies the
  verified arithmetic; `MessageInference.layout_fields` is not pooled yet.

**Not attempted here:** supervised learning on opendbc labels. CAN-D reports a
random forest at F 90.2 against F 89.6 for its heuristic — its largest single
gain — and nothing in this project has tested anything of the kind. Every
threshold here was swept by hand. It is the most promising untried direction,
and is called out as untested rather than as a result.

---

## 11. Recurring lessons

Patterns that showed up more than once, worth having in mind before the next
idea.

1. **A better cut rule is not a better field detector.** Boundary recall was never
   the binding constraint. A wrong cut destroys the field on each side of it, so
   precision is what binds, and rules that add cuts make things worse (§2.5).
2. **A bit admitted by a statistic that cannot segment it has nowhere to be cut.**
   Liveness and segmentation are different problems, and a good liveness test
   makes segmentation *worse* if it admits fields no rule can split (§2.4).
3. **Expect one field to be a function of another.** A CRC over a counter walks
   like a counter; a selector that cycles is a counter modulo its period; a byte
   that duplicates another byte is a "function" of it. Every one of these produced
   a real false-positive class (§4, §6.2).
4. **A fit with nothing left over is not evidence.** Sixteen unknowns fitted to
   sixteen equations always fit. Gate every solved secret on distinct payload
   count, and scale the bar to the field's width (§5.2).
5. **Reasoning about what a DBC "would name" is not evidence; scoring against one
   is.** The whole-byte multiplexor claim was argued convincingly and matched
   nothing (§6.2).
6. **Controls are what turn a measurement into a result.** The jumpiness filter is
   only credible because a length floor and a rate floor were swept alongside it
   and both lowered F1 (§2.1).
7. **A move that reduces findings can be the correct one**, because most false
   positives look like extra coverage. Several of the largest improvements here
   removed claims (§5.2, §6.2, §2.1).
8. **Say which number the headline is.** Pooling an easy sub-problem into a hard
   problem's metric moves the metric without moving the hard problem (§6.4).
9. **A transport layer is not a vehicle signal**, and it does not announce
   itself. An ISO-TP SequenceNumber passes the counter test at 98% because it
   really is a counter; what is wrong is the layer it is attributed to. Where a
   protocol carries its own verification — an SN chain, a length that must be
   reached, a FlowControl that must answer — use it, and do not accept a header
   nibble as a substitute (§9).
