# canlens

Corpus-scale reverse engineering of automotive CAN bus traces.

> **Status: pre-alpha.** `corpus`, `decode`, `analyze` and `infer` work. Everything
> downstream is scaffolding with an honest docstring where the implementation
> will go.

**New here? Read [HOWTO.md](HOWTO.md)** — an end-to-end walkthrough from empty
machine to per-bit measurements of a real vehicle's traffic.

## Why

Most CAN reverse-engineering tools analyse **one trace at a time**. That bounds
what they can conclude: a checksum hypothesis derived from a single recording is
a guess with a confidence score attached, and there is no way to check it.

`canlens` is built around a corpus instead — 230 vehicle platforms, 188,883
segments, ~3,148 hours of driving from openpilot's fleet, all labeled by
platform. That changes what is provable:

- **Cross-segment corroboration** — a hypothesis that holds across thousands of
  segments from hundreds of independent drivers is established, not guessed.
- **Cross-platform inference** — a scheme solved on one platform becomes a prior
  for the rest of that manufacturer's range.
- **Differential analysis** — diff two model years of one platform to see
  exactly what moved.
- **Ground truth** — opendbc has community DBCs for many of these platforms, so
  the inference engine can be *scored* rather than merely trusted.

## Install

```bash
pip install -e ".[dev]"
```

This machine's system Python is PEP 668 externally-managed and lacks
`python3.14-venv`, so neither a plain `pip install` nor `python3 -m venv`
works out of the box. Until `apt install python3.14-venv` is run, everything
works directly from the source tree:

```bash
PYTHONPATH=src python3 -m canlens corpus list
```

```bash
PYTHONPATH=src python3 -m pytest -q
```

## Corpus

Data comes from the public [commaCarSegments bucket][bucket] (comma.ai, MIT).
The full bucket is **~299 GB**, so nothing here fetches all of it. You name
platforms; only those are pulled. Fetches are resumable — re-run to continue.

```bash
canlens corpus list                              # 230 platforms, counts + sizes
canlens corpus plan TOYOTA_PRIUS                 # size it before committing
canlens corpus fetch TOYOTA_PRIUS KIA_EV6 --limit 200
canlens corpus status                            # how much is local
canlens corpus which <segment path>              # which car is this?
```

Segment paths carry no platform name — the layout mirrors the upstream bucket,
which is keyed by device and route — so `corpus which` (and the `[PLATFORM]`
prefix on `decode`/`analyze` output) is the way back to which car a trace is.

The local corpus root defaults to `~/data/canlens`, overridable with
`--root` or `$CANLENS_DATA`. It is gitignored and must stay that way.

For scale: the largest single platform is ~18 GB, and the median platform is
~0.3 GB. A working set of a few hundred segments across a dozen platforms is a
couple of GB, not 299.

## Dependencies

Core is `numpy` only — `canlens.corpus` is deliberately stdlib-only, so pulling
data never needs a compiler. Everything else is an extra:

| extra | pulls | for |
|---|---|---|
| `decode` | `pycapnp`, `zstandard` (only below 3.14) | reading `rlog.zst` |
| `store` | `pyarrow` | decoded-frame cache |
| `truth` | `cantools` | parsing opendbc DBCs |
| `traces` | `python-can` | `.blf` / `.asc` ingest |

Two things worth knowing:

**The capnp schemas are not a pip dependency.** `rlog` is capnp `cereal`, and
those `.capnp` files ship on no package index. `log.capnp`, `legacy.capnp`,
`custom.capnp` and `include/c++.capnp` come from `commaai/openpilot` (branch
`release3`); `car.capnp` must come from `commaai/opendbc`, because in openpilot
it is a git symlink and a raw fetch returns the link text, not the schema.
`decode/` fetches all five on demand — MIT, and unvendored like the corpus.

**The `opendbc` PyPI package is not used.** It declares
`requires-python = <3.13,>=3.11` and pins `pycapnp==2.1.0`, so it cannot coexist
with this project on 3.13+. Only its `.dbc` files were ever wanted; those are
plain data, fetched from the repo and parsed with `cantools`.

On Python 3.14, zstd comes from the stdlib (`compression.zstd`, PEP 784) and the
third-party package is skipped entirely.

## Layout

| package | role |
|---|---|
| `corpus/` | manifest, selective fetch, local accounting — **working** |
| `decode/` | `rlog.zst` → normalised CAN records — **working** |
| `analyze/` | single-trace timing, entropy, per-bit classification — **working** |
| `infer/` | counters + 8-bit checksums — **working**; boundaries, 16-bit CRCs, muxes to do |
| `corroborate/` | cross-segment and cross-platform agreement |
| `truth/` | opendbc ground truth, scoring the inference engine |

## Related

- [opendbc](https://github.com/commaai/opendbc) — CAN database definitions
- [openpilot](https://github.com/commaai/openpilot) — `LogReader`, cabana
- BoAt — the deterministic SIL/HIL simulation platform this grew out of

## License

MIT — see [LICENSE](LICENSE). The upstream dataset is MIT-licensed and is
fetched on demand, never vendored.

[bucket]: https://huggingface.co/buckets/AlexTech-stack/commaCarSegments-bucket
