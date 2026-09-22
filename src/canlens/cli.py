# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""`canlens` command line entry point."""
from __future__ import annotations

import argparse
import os
import sys

from . import __version__
from .analyze.bits import BitOrder
from .corpus import AVG_SEGMENT_MB, FetchResult, Manifest, fetch_all, segment_dest, select

DEFAULT_ROOT = os.environ.get("CANLENS_DATA", os.path.expanduser("~/data/canlens"))


def _manifest(root: str) -> Manifest:
    return Manifest.load(os.path.join(root, "database.json"))


def _plan(manifest: Manifest, platforms: list[str], limit: int | None) -> list[str]:
    segments = select(manifest, platforms, limit)
    for key in platforms:
        n = len(manifest[key].segments[:limit] if limit else manifest[key].segments)
        print(f"{key:<38}{n:>7} segments  ~{n * AVG_SEGMENT_MB / 1024:6.2f} GB")
    if len(platforms) > 1:
        total = len(segments)
        print(f"{'TOTAL':<38}{total:>7} segments  ~{total * AVG_SEGMENT_MB / 1024:6.2f} GB")
    return segments


def _platform_label(root: str, path: str) -> str:
    """`[PLATFORM] ` prefix for a segment, or empty if it cannot be resolved.

    Nothing in the stored path says which car a segment came from -- the layout
    mirrors the upstream bucket, which is keyed by device and route -- so this
    is the only way to tell a Prius trace from an EV6 one.
    """
    try:
        platform = _manifest(root).platform_of(path)
    except (OSError, ValueError):
        return ""
    return f"[{platform}] " if platform else "[unknown platform] "


def cmd_list(args) -> int:
    manifest = _manifest(args.root)
    for platform in manifest.by_size():
        print(f"{platform.key:<38}{platform.count:>7} segments  ~{platform.est_gb:6.2f} GB")
    print(f"\n{len(manifest)} platforms, {manifest.total_segments} segments total")
    return 0


def cmd_plan(args) -> int:
    _plan(_manifest(args.root), args.platform, args.limit)
    return 0


def cmd_fetch(args) -> int:
    manifest = _manifest(args.root)
    segments = _plan(manifest, args.platform, args.limit)
    print(f"-> {args.root}")

    def progress(done: int, total: int, r: FetchResult) -> None:
        print(
            f"  {done}/{total}  {r.bytes_new / 2**30:.2f} GiB new, "
            f"{r.skipped} present, {r.failed} failed",
            flush=True,
        )

    result = fetch_all(segments, args.root, jobs=args.jobs, progress=progress)
    return 0 if result.ok else 1


def cmd_status(args) -> int:
    manifest = _manifest(args.root)
    present = total = 0
    for platform in manifest.by_size():
        have = sum(1 for s in platform.segments if os.path.exists(segment_dest(args.root, s)))
        if have:
            print(f"{platform.key:<38}{have:>7}/{platform.count} segments local")
        present += have
        total += platform.count
    pct = 100 * present / total if total else 0.0
    print(f"\n{present}/{total} segments local ({pct:.2f}%) in {args.root}")
    return 0


def cmd_decode_schema(args) -> int:
    from .decode import ensure_schemas

    target = ensure_schemas(args.root, refresh=args.refresh)
    print(f"schemas ready in {target}")
    return 0


def cmd_decode_summary(args) -> int:
    from .decode import summarize

    for path in args.path:
        s = summarize(path, root=args.root)
        echo_pct = 100 * s.echoes / s.frames if s.frames else 0.0
        print(f"{_platform_label(args.root, path)}{path}")
        print(
            f"  {s.frames - s.echoes} frames over {s.duration_s:.1f}s "
            f"({s.echoes} echoes dropped, {echo_pct:.0f}% of raw)"
        )
        print(f"  {s.unique_addresses} unique (bus, address) pairs")
        for bus, n in s.buses.items():
            addrs = sum(1 for b, _ in s.addresses if b == bus)
            print(f"    bus {bus}: {n:>7} frames, {addrs:>4} addresses")
    return 0


def cmd_analyze_trace(args) -> int:
    import shutil

    from .analyze import KIND_ORDER, BitOrder, analyze_segment
    from .render import bit_strip, bits_that_fit, legend, strip_width, supports_color

    order = BitOrder(args.order)
    profile = analyze_segment(args.path, root=args.root, order=order)
    color = supports_color() and not args.no_color

    print(f"{_platform_label(args.root, args.path)}{args.path}")
    print(
        f"{profile.frames} frames, {len(profile)} messages, "
        f"{profile.duration_s:.1f}s, {profile.total_entropy:.0f} bits of payload entropy"
    )

    rows = profile.by_entropy()[: args.top]
    head = f"{'message':<14}{'count':>7}{'len':>5}{'period':>9}{'cadence':>10}{'entropy':>9}  "

    if args.bitmap:
        widest = max((m.bits.bits for m in rows), default=0)
        if args.bit_columns:
            budget = strip_width(args.bit_columns)
        else:
            budget = max(16, shutil.get_terminal_size((110, 24)).columns - len(head) - 1)
        max_bits = min(widest, bits_that_fit(budget))
        print(f"\n{head}bits ({order}, {max_bits} of {widest} shown)")
    else:
        print(f"\n{head}bits ({'/'.join(str(k) for k in KIND_ORDER)})")
        max_bits = 0

    for message in rows:
        period = f"{message.timing.period_ms:.1f}ms" if message.timing.period_ms else "-"
        flag = "*" if message.multi_length else " "
        line = (
            f"{message!s:<14}{message.count:>7}{message.width:>4}{flag}{period:>9}"
            f"{message.timing.cadence!s:>10}{message.bits.payload_entropy:>9.1f}  "
        )
        if args.bitmap:
            line += bit_strip(message.bits.kinds, color=color, max_bits=max_bits)
        else:
            line += "/".join(str(message.bits.count(k)) for k in KIND_ORDER)
        print(line)

    if args.bitmap:
        print(f"\n{legend(color)}")
    if any(m.multi_length for m in rows):
        print("* payload length varies -- stats cover the dominant length only")
    return 0


def cmd_infer_trace(args) -> int:
    from .analyze import BitOrder
    from .infer import infer_segment

    results = infer_segment(args.path, root=args.root, order=BitOrder(args.order))
    hits = [m for m in results if m.found_anything]
    print(f"{_platform_label(args.root, args.path)}{args.path}")
    print(
        f"{len(results)} messages examined, {len(hits)} with findings: "
        f"{sum(len(m.counters) for m in hits)} counters, "
        f"{sum(len(m.checksums) for m in hits)} checksums, "
        f"{sum(len(m.crc16s) for m in hits)} 16-bit CRCs, "
        f"{sum(m.multiplexor is not None for m in hits)} multiplexors, "
        f"{sum(len(m.signals) for m in hits)} signals"
    )
    if not hits:
        return 0
    print(f"\n{'message':<14}{'frames':>7}  {'counters':<35}{'checksums':<30}signals")
    for m in hits[: args.top]:
        counters = ", ".join(
            f"{c.length}bit@{c.start_bit}"
            + ("" if c.stride == 1 else f"/{c.stride}")
            + f" {c.match_rate:.0%}"
            for c in m.counters
        )
        if m.multiplexor is not None:
            mux = m.multiplexor
            counters = ", ".join(
                filter(None, [f"mux {mux.length}bit@{mux.start_bit} x{len(mux.values)}", counters])
            )
        checks = ", ".join(
            [
                f"{s.algorithm}@{'?' if s.ambiguous else s.byte_index}"
                + ("/idlist" if s.algorithm == "e2e_p22"
                   else f"/id{s.data_id:X}" if s.data_id is not None else "")
                + f" {s.match_rate:.0%}"
                for s in m.checksums
            ]
            + [
                f"{c.algorithm}@{c.start_byte}"
                + (f"/id{c.data_id:04X}" if c.data_id is not None else "")
                + f" {c.match_rate:.0%}"
                for c in m.crc16s
            ]
        )
        if len(counters) > 33:
            counters = counters[:32] + "\u2026"
        if len(checks) > 28:
            checks = checks[:27] + "\u2026"
        fields = ", ".join(
            f"{s.length}{'' if s.bounded else '+'}bit@{s.start_bit}" for s in m.signals
        )
        print(f"{m!s:<14}{m.frames:>7}  {counters or '-':<35}{checks or '-':<30}"
              f"{fields or '-'}")
    if len(hits) > args.top:
        print(f"... and {len(hits) - args.top} more (use --top)")
    return 0


def cmd_infer_message(args) -> int:
    import shutil

    from .analyze import BitOrder
    from .analyze.bits import bit_matrix
    from .decode import iter_frames
    from .infer import infer_message
    from .infer.counters import field_values
    from .render import FIELD_MARKS, bar, bit_strip, field_ruler, legend, sparkline, supports_color

    order = BitOrder(args.order)
    color = supports_color() and not args.no_color
    address = int(args.address, 0)

    payloads = [
        f.data
        for f in iter_frames(args.path, root=args.root)
        if f.address == address and (args.bus is None or f.bus == args.bus)
    ]
    if not payloads:
        where = "" if args.bus is None else f" on bus {args.bus}"
        print(f"canlens: no frames for 0x{address:X}{where} in {args.path}", file=sys.stderr)
        return 1
    width = max(set(map(len, payloads)), key=[len(p) for p in payloads].count)
    payloads = [p for p in payloads if len(p) == width]

    result = infer_message(payloads, bus=args.bus or 0, address=address, order=order)
    print(f"{_platform_label(args.root, args.path)}0x{address:03X}")
    print(f"{result.frames} frames x {width} bytes, {order} bit order\n")

    marks: dict[int, str] = {}
    for c in result.counters:
        marks.update(dict.fromkeys(range(c.start_bit, c.end_bit), FIELD_MARKS["counter"]))
    spans = [(s.start_bit, s.length) for s in result.checksums]
    spans += [(c.start_bit, c.length) for c in result.crc16s]
    for start_bit, length in spans:
        marks.update(
            dict.fromkeys(range(start_bit, start_bit + length), FIELD_MARKS["checksum"])
        )
    if result.multiplexor is not None:
        mux = result.multiplexor
        marks.update(dict.fromkeys(range(mux.start_bit, mux.end_bit), FIELD_MARKS["mux"]))
    for signal in result.signals:
        marks.update(
            dict.fromkeys(range(signal.start_bit, signal.end_bit), FIELD_MARKS["signal"])
        )

    print(f"  {bit_strip(result.bits.kinds, color=color)}")
    if marks:
        print(f"  {field_ruler(marks, result.bits.bits)}")

    # Default the plot to whatever the terminal can actually show, so a long
    # counter wraps visibly instead of spilling over the edge.
    samples = args.samples or max(16, shutil.get_terminal_size((110, 24)).columns - 12)

    matrix = bit_matrix(payloads, width, order)
    if result.multiplexor is not None:
        mux = result.multiplexor
        values = field_values(matrix, mux.start_bit, mux.length).tolist()
        print(f"\n  mux      {mux.length} bits @ bit {mux.start_bit}, "
              f"{len(mux.values)} values, {len(mux.dependent_bits)} bits depend on it")
        print(f"           {sparkline(values, samples)}")
        print(f"           {bar(mux.coverage)} {mux.coverage:.1%} of frames carry a listed value")
        for value, count in zip(mux.values, mux.frames_per_value):
            print(f"           value {value:>3}: {count} frames")
    for c in result.counters:
        values = field_values(matrix, c.start_bit, c.length).tolist()
        print(f"\n  counter  {c.length} bits @ bit {c.start_bit}, step {c.stride}, "
              f"wraps every {c.period}")
        print(f"           {sparkline(values, samples)}")
        print(f"           {bar(c.match_rate)} {c.match_rate:.1%} of steps")
    for s in result.checksums:
        print(f"\n  checksum byte {s.byte_index}, algorithm {s.algorithm}")
        print(f"           {bar(s.match_rate)} {s.match_rate:.1%} "
              f"({round(s.match_rate * s.frames)}/{s.frames} frames)")
    for signal in result.signals:
        extent = (
            f"{signal.length} bits" if signal.bounded else f"at least {signal.length} bits"
        )
        print(f"\n  signal   {extent} @ bit {signal.start_bit}, "
              f"seen {signal.minimum}..{signal.maximum}")
        print(f"           {sparkline(field_values(matrix, signal.start_bit, signal.length).tolist(), samples)}")
    for crc in result.crc16s:
        ident = "" if crc.data_id is None else f", data ID 0x{crc.data_id:04X}"
        print(f"\n  crc16    bytes {crc.start_byte}-{crc.start_byte + 1}, {crc.algorithm}, "
              f"{crc.byteorder}-endian{ident}")
        print(f"           {bar(crc.match_rate)} {crc.match_rate:.1%} "
              f"({round(crc.match_rate * crc.frames)}/{crc.frames} frames)")
    if not result.found_anything:
        print("\n  no counter or checksum reproduced this message")

    marks_key = "   ".join(f"{v} {k}" for k, v in FIELD_MARKS.items())
    print(f"\n{legend(color)}   {marks_key}")
    return 0


def cmd_export_pdu_db(args) -> int:
    from .export import save_pdu_db
    from .gui.model import load_segment

    model = load_segment(
        args.path, root=args.root, platform=_manifest(args.root).platform_of(args.path)
    )
    entries = model.export_messages()
    if not entries:
        print("canlens: nothing inferred in this segment to export", file=sys.stderr)
        return 1
    save_pdu_db(entries, args.output)
    signals = sum(len(e.signals) for e in entries)
    print(f"{len(entries)} messages, {signals} signals -> {args.output}")
    return 0


def cmd_corroborate_pooled(args) -> int:
    """Profile 22 Data ID lists that only several segments together can settle."""
    from .corroborate import corroborate_p22
    from .infer.profiles import p22_id_list

    def progress(done: int, total: int) -> None:
        if done % 10 == 0 or done == total:
            print(f"  {done}/{total} candidate messages", file=sys.stderr, flush=True)

    found = corroborate_p22(
        args.platform, root=args.root, limit=args.limit,
        min_devices=args.min_devices, progress=progress,
    )
    from .corroborate import signals_across

    pooled_signals = signals_across(
        args.platform, root=args.root, limit=args.limit, min_devices=args.min_devices
    )
    if not found:
        code = _no_pooled_findings(args)
        _report_pooled_signals(pooled_signals)
        return code
    needed = [f for f in found if f.settled_by_pooling]
    print(
        f"\n{args.platform}: {len(found)} Profile 22 lists solved from pooled segments, "
        f"{len(needed)} of which no single segment carried enough evidence for\n"
    )
    for finding in sorted(found, key=lambda f: (not f.settled_by_pooling, f.pool.bus, f.pool.address)):
        mark = "*" if finding.settled_by_pooling else " "
        print(f"{mark} {finding}")
        print(f"    best single segment offered {finding.alone} distinct payloads, "
              f"pooling gives {finding.pool.distinct}"
              f"{' -- below the bar of 24 alone' if finding.settled_by_pooling else ''}")
        if finding.hypothesis.data_id is not None:
            ids = p22_id_list(finding.hypothesis.data_id)
            print(f"    data IDs by counter value: {' '.join(f'{i:02X}' for i in ids)}")
    if needed:
        print("\n* marks a list only the pool could settle; the rest a rich enough "
              "single\n  segment could also have reached.")
    _report_pooled_signals(pooled_signals)
    return 0


def _report_pooled_signals(pooled) -> None:
    """Signal boundaries read from every segment at once."""
    if not pooled:
        return
    total = sum(len(v) for v in pooled.values())
    bounded = sum(1 for v in pooled.values() for s in v if s.bounded)
    print(
        f"\n{total} signals over {len(pooled)} messages, from every segment pooled: "
        f"{bounded} with a settled width, {total - bounded} lower bounds."
    )
    print("  A field's high bits only move once the value reaches them, so more")
    print("  drives means more of the range seen and fewer widths left open.")


def _no_pooled_findings(args) -> int:
    """Say which kind of nothing this is: too little data, or a real answer.

    "Pool more segments" is wrong advice for a platform that simply does not
    use Profile 22, and Rivian and the EV6 are both in that position -- one is
    Profile 11 throughout, the other Profile 5. Reporting what the platform
    *does* use turns an empty result into an informative one.
    """
    from collections import Counter

    from .corpus import Manifest, inventory
    from .corroborate import device_of
    from .corroborate.pooled import MIN_DEVICES
    from .infer import infer_cached

    held = inventory(args.root, Manifest.load(f"{args.root}/database.json")).get(args.platform)
    paths = list(held.paths if held else [])[: args.limit]
    devices = {device_of(path) for path in paths}
    if len(devices) < max(args.min_devices, MIN_DEVICES):
        print(
            f"canlens: {args.platform} has {len(paths)} local segment(s) from "
            f"{len(devices)} device(s); pooling needs at least {args.min_devices}. "
            f"Try 'canlens corpus fetch {args.platform}'.",
            file=sys.stderr,
        )
        return 1

    algorithms: Counter[str] = Counter()
    for path in paths:
        for message in infer_cached(path, root=args.root):
            algorithms.update(c.algorithm for c in message.checksums)
            algorithms.update(c.algorithm for c in message.crc16s)
    print(
        f"{args.platform}: pooled {len(paths)} segments from {len(devices)} devices "
        f"and found no Profile 22 Data ID list."
    )
    if algorithms:
        named = ", ".join(f"{name} ({count})" for name, count in algorithms.most_common(4))
        print(f"This platform's checksums are {named} -- none of which hides a sixteen-byte")
        print("secret, so none of them needs pooling to be checked.")
    return 0


def cmd_truth_dbc(args) -> int:
    from .truth import FieldKind, load_dbc

    reference = load_dbc(args.dbc)
    print(reference.summary())
    named = sum(len(m.of_kind(FieldKind.SIGNAL)) for m in reference.messages.values())
    print(f"{named} ordinary signals (not scored -- canlens does not infer boundaries)")
    if not args.verbose:
        return 0
    print(f"\n{'message':<28}{'id':>7}  fields")
    for message in sorted(reference.messages.values(), key=lambda m: m.address):
        fields = [
            f"{s.name}[{s.kind}]"
            for s in message.signals
            if s.kind is not FieldKind.SIGNAL
        ]
        if fields:
            print(f"{message.name[:27]:<28}{message.address:>#7x}  {', '.join(fields)}")
    return 0


def cmd_truth_score(args) -> int:
    from .analyze import BitOrder
    from .infer import infer_cached
    from .truth import load_dbc, score

    reference = load_dbc(args.dbc)
    inferences = infer_cached(args.path, root=args.root, order=BitOrder(args.order))
    result = score(inferences, reference, bus=args.bus)

    print(f"{_platform_label(args.root, args.path)}{args.path}")
    print(f"reference {reference.summary()}")
    print(
        f"bus {result.bus}{'' if args.bus is not None else ' (best identifier overlap)'}: "
        f"{result.trace_messages} messages in the trace, {result.scored_messages} also in the "
        f"reference, {result.absent_from_trace} reference messages this drive never carried"
    )
    if not result.scored_messages:
        print("canlens: no message is in both -- wrong DBC, or try --bus", file=sys.stderr)
        return 1

    print(f"\n{'field':<14}{'hit':>5}{'missed':>8}{'extra':>7}{'near':>6}"
          f"{'precision':>11}{'recall':>9}")
    for kind, tally in result.tallies.items():
        if kind not in result.scorable:
            continue
        print(f"{kind!s:<14}{tally.hits:>5}{tally.missed:>8}{tally.false_alarms:>7}"
              f"{tally.near:>6}{tally.precision:>10.0%}{tally.recall:>9.0%}")
    overall = result.overall
    if len(result.scorable) > 1:
        print(f"{'overall':<14}{overall.hits:>5}{overall.missed:>8}{overall.false_alarms:>7}"
              f"{overall.near:>6}{overall.precision:>10.0%}{overall.recall:>9.0%}")
    for kind, count in result.unevaluated.items():
        if count:
            print(f"{kind!s:<14}{count:>5} claims not evaluated -- "
                  f"this reference names no {kind}")

    if result.disagreements and not args.no_detail:
        shown = [d for d in result.disagreements if args.kind is None or str(d.kind) == args.kind]
        print(f"\ndisagreements ({len(shown)}):")
        for disagreement in shown[: args.top]:
            mark = "~" if disagreement.overlapping else " "
            print(f"  {mark} {disagreement}")
        if len(shown) > args.top:
            print(f"  ... and {len(shown) - args.top} more (use --top)")
        print("\n~ marks a claim that overlaps the reference without matching it exactly.")
    print(
        "\nA disagreement is evidence, not a verdict: the reference is itself "
        "reverse-engineered,\nand its authors often leave counters unnamed."
    )
    return 0


def default_jobs() -> int:
    """Physical cores, not logical ones.

    The pipeline is embarrassingly parallel across segments, and measured 4.0x
    on this machine's 12 logical CPUs -- which are 6 physical cores with
    hyperthreading. The second thread of a core adds nothing to numpy-bound
    work, so the default does not pretend otherwise.
    """
    return max(1, (os.cpu_count() or 2) // 2)


def cmd_corroborate(args) -> int:
    from .corroborate import corroborate_platform
    from .render import bit_strip, field_ruler, legend, supports_color

    def progress(done: int, total: int) -> None:
        if done % 25 == 0 or done == total:
            print(f"  {done}/{total} segments", file=sys.stderr, flush=True)

    result = corroborate_platform(
        args.platform, root=args.root, limit=args.limit, progress=progress,
        canonical_buses=not args.logged_buses,
    )
    if not result.segments:
        print(f"canlens: nothing local for {args.platform}", file=sys.stderr)
        return 1
    print(f"{result.platform}: {result.segments} segments from {result.devices} devices, "
          f"{len(result)} messages")

    buses = result.buses
    if buses is not None and buses.identities:
        print(f"\n{len(buses.identities)} buses, identified by the identifiers they carry:")
        for identity in sorted(buses.identities, key=lambda i: i.label):
            print(f"  {identity}")
        if buses.relabelled:
            print(f"  {buses.relabelled} (segment, bus) pairs were logged under another "
                  f"number and have been moved.")
        for number, sharing in sorted(buses.contested.items()):
            print(f"  bus {number} names {len(sharing)} different buses across these "
                  f"segments -- the car was recorded in more than one wiring.")
    print()

    if args.address is not None:
        address = int(args.address, 0)
        matches = [m for m in result.by_identifier() if m.address == address
                   and (args.bus is None or m.bus == args.bus)]
        if not matches:
            print(f"canlens: 0x{address:X} not seen on {args.platform}", file=sys.stderr)
            return 1
        color = supports_color() and not args.no_color
        for m in matches:
            print(f"\n{m}   width {m.width} ({m.width_agreement:.0%} agree)   "
                  f"in {m.segments}/{result.segments} segments, {m.devices}/{result.devices} devices   "
                  f"layout agreement {m.agreement:.0%}")
            print(f"  {bit_strip(m.kinds, color=color)}")
            # Agreement per bit as a decile digit; ! marks a rarely-moving bit.
            marks = {i: ("!" if b.rare else str(min(9, int(b.agreement * 10))))
                     for i, b in enumerate(m.bits)}
            print(f"  {field_ruler(marks, len(m.bits))}")
            if m.rare_bits:
                print(f"  rare bits (constant in most segments, moving in some): {m.rare_bits}")
            for ctr in m.counters:
                print(f"  counter   {ctr}")
            for chk in m.checksums:
                print(f"  checksum  {chk}")
            for crc in m.crc16s:
                print(f"  crc16     {crc}")
            for mux in m.multiplexors:
                print(f"  mux       {mux}")
            if not (m.counters or m.checksums or m.crc16s or m.multiplexors):
                print("  no counter or checksum found in any segment")
        print(f"\n{legend(color)}   digits: agreement decile   !: rare bit")
        return 0

    rows = result.by_identifier()[: args.top]
    print(f"\n{'message':<14}{'seg':>5}{'dev':>5}{'w':>4}{'agree':>7}{'rare':>6}  findings")
    for m in rows:
        findings = []
        for mux in m.multiplexors:
            findings.append(f"mux {mux.length}bit@{mux.start_bit} [{mux.evidence.tier}"
                            + (",contested" if mux.contested else "") + "]")
        for ctr in m.counters:
            findings.append(f"ctr {ctr.length}bit@{ctr.start_bit} [{ctr.evidence.tier}"
                            + (",contested" if ctr.contested else "") + "]")
        for chk in m.checksums:
            ident = (
                "/idlist" if chk.algorithm == "e2e_p22"
                else f"/id{chk.data_id:X}" if chk.data_id is not None else ""
            )
            findings.append(f"{chk.algorithm}@{chk.byte_index}{ident} [{chk.evidence.tier}"
                            + (",contested" if chk.contested else "") + "]")
        for crc in m.crc16s:
            ident = f"/id{crc.data_id:04X}" if crc.data_id is not None else ""
            findings.append(f"{crc.algorithm}@{crc.start_byte}{ident} [{crc.evidence.tier}"
                            + (",contested" if crc.contested else "") + "]")
        print(f"{m!s:<14}{m.segments:>5}{m.devices:>5}{m.width:>4}{m.agreement:>7.0%}"
              f"{len(m.rare_bits):>6}  {', '.join(findings) or '-'}")
    if len(result) > args.top:
        print(f"... and {len(result) - args.top} more (use --top)")
    return 0


def cmd_cache_build(args) -> int:
    """Decode and infer every local segment once, so later passes are hits.

    Both caches are filled -- decoded frames and inference results -- and the
    work is spread over a process pool, since one segment never needs another.
    """
    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from .corpus import inventory
    from .infer import warm_segment

    manifest = _manifest(args.root)
    held = inventory(args.root, manifest)
    names = args.platform or sorted(k for k in held if k)
    paths = [p for name in names for p in (held[name].paths if name in held else [])]
    if not paths:
        print("canlens: nothing local to cache", file=sys.stderr)
        return 1

    jobs = args.jobs or default_jobs()
    started = time.time()
    done = failed = 0
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(warm_segment, path, args.root): path for path in paths}
        for future in as_completed(futures):
            done += 1
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop the rest
                failed += 1
                print(f"  failed: {futures[future]}: {exc}", file=sys.stderr)
            if done % 10 == 0 or done == len(paths):
                rate = done / max(time.time() - started, 1e-9)
                print(f"  {done}/{len(paths)} segments  ({rate:.1f}/s, {jobs} workers)", flush=True)
    print(f"cached {done - failed} segments in {time.time() - started:.1f}s"
          + (f", {failed} failed" if failed else ""))
    return 1 if failed else 0


def cmd_cache_clear(args) -> int:
    from .decode import clear
    from .infer import clear_results

    print(f"removed {clear(args.root)} frame entries and {clear_results(args.root)} result entries")
    return 0


def cmd_cache_status(args) -> int:
    """How much of what is local is cached -- both frames and inference results."""
    from .corpus import inventory
    from .decode import cache_path
    from .decode.cache import CACHE_DIR
    from .infer.results import results_path

    manifest = _manifest(args.root)
    held = inventory(args.root, manifest)
    total = frames = results = frame_bytes = result_bytes = 0
    for entry in held.values():
        for path in entry.paths:
            total += 1
            frame_entry, result_entry = cache_path(args.root, path), results_path(args.root, path)
            if os.path.exists(frame_entry):
                frames += 1
                frame_bytes += os.path.getsize(frame_entry)
            if os.path.exists(result_entry):
                results += 1
                result_bytes += os.path.getsize(result_entry)
    where = os.path.join(args.root, CACHE_DIR)
    print(f"{total} local segments in {where}")
    print(f"  frames   {frames:>5}/{total}  {frame_bytes / 2**30:.2f} GiB")
    print(f"  results  {results:>5}/{total}  {result_bytes / 2**20:.0f} MiB")
    return 0


def cmd_gui(args) -> int:
    try:
        from .gui.window import run
    except ImportError as exc:
        print(f"canlens: the workbench needs the gui extra: pip install 'canlens[gui]' ({exc})",
              file=sys.stderr)
        return 2
    return run(args.root)


def cmd_corpus_delete(args) -> int:
    from .corpus import delete_segments, inventory

    manifest = _manifest(args.root)
    held = inventory(args.root, manifest)
    targets: list[str] = []
    for name in args.platform:
        entry = held.get(name)
        if entry is None:
            print(f"canlens: nothing local for {name}", file=sys.stderr)
            continue
        targets += entry.paths[: args.limit] if args.limit else entry.paths
    if not targets:
        print("canlens: nothing to delete", file=sys.stderr)
        return 1

    size = sum(os.path.getsize(p) for p in targets if os.path.exists(p))
    print(f"{len(targets)} segments, {size / 2**30:.2f} GiB, under {args.root}")
    if not args.yes:
        # Deleting fetched data is cheap to undo -- it can be fetched again --
        # but it is never what someone wanted by accident.
        reply = input("delete these? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("cancelled")
            return 1
    result = delete_segments(targets)
    print(f"deleted {result.deleted} segments, freed {result.bytes_freed / 2**30:.2f} GiB")
    for failure in result.failed:
        print(f"  failed: {failure}", file=sys.stderr)
    return 0 if result.ok else 1


def cmd_corpus_which(args) -> int:
    manifest = _manifest(args.root)
    for path in args.path:
        platform = manifest.platform_of(path)
        print(f"{platform or 'not in the corpus manifest':<28} {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="canlens", description=__doc__)
    parser.add_argument("--version", action="version", version=f"canlens {__version__}")
    parser.add_argument(
        "--root",
        default=DEFAULT_ROOT,
        help="local corpus directory (default: %(default)s, or $CANLENS_DATA)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    corpus = sub.add_parser("corpus", help="inspect and materialise the segment corpus")
    ops = corpus.add_subparsers(dest="op", required=True)

    p = ops.add_parser("list", help="list platforms with segment counts and sizes")
    p.set_defaults(func=cmd_list)

    for name, func, help_text in (
        ("plan", cmd_plan, "show what a fetch would download, without downloading"),
        ("fetch", cmd_fetch, "download selected platforms (resumable)"),
    ):
        p = ops.add_parser(name, help=help_text)
        p.add_argument("platform", nargs="+", help="platform key(s); see 'canlens corpus list'")
        p.add_argument("--limit", type=int, help="max segments per platform")
        if name == "fetch":
            p.add_argument("--jobs", type=int, default=8, help="parallel downloads (default: 8)")
        p.set_defaults(func=func)

    p = ops.add_parser("status", help="report how much of the corpus is local")
    p.set_defaults(func=cmd_status)

    p = ops.add_parser("delete", help="remove locally stored segments")
    p.add_argument("platform", nargs="+", help="platform key(s); see 'canlens corpus list'")
    p.add_argument("--limit", type=int, help="delete at most this many per platform")
    p.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    p.set_defaults(func=cmd_corpus_delete)

    p = ops.add_parser("which", help="say which platform a segment path belongs to")
    p.add_argument("path", nargs="+", help="segment path(s) or IDs")
    p.set_defaults(func=cmd_corpus_which)

    decode = sub.add_parser("decode", help="turn raw logs into CAN frames")
    dops = decode.add_subparsers(dest="op", required=True)

    p = dops.add_parser("schema", help="fetch the capnp schemas needed to read rlogs")
    p.add_argument("--refresh", action="store_true", help="re-download even if cached")
    p.set_defaults(func=cmd_decode_schema)

    p = dops.add_parser("summary", help="report what a segment contains")
    p.add_argument("path", nargs="+", help="path(s) to rlog.zst")
    p.set_defaults(func=cmd_decode_summary)

    analyze = sub.add_parser("analyze", help="measure timing and payload entropy")
    aops = analyze.add_subparsers(dest="op", required=True)

    p = aops.add_parser("trace", help="per-message measurements for one segment")
    p.add_argument("path", help="path to rlog.zst")
    p.add_argument("--top", type=int, default=20, help="rows to print (default: 20)")
    p.add_argument(
        "--order",
        choices=[o.value for o in BitOrder],
        default=BitOrder.INTEL.value,
        help="payload bit numbering (default: %(default)s, as DBC files use)",
    )
    p.add_argument(
        "--no-bitmap",
        dest="bitmap",
        action="store_false",
        help="print per-class counts instead of the coloured bit map",
    )
    p.add_argument(
        "--bit-columns", type=int, help="bits to show per row (default: fit the terminal)"
    )
    p.add_argument("--no-color", action="store_true", help="never emit ANSI colour")
    p.set_defaults(func=cmd_analyze_trace)

    infer = sub.add_parser("infer", help="find counters and checksums")
    iops = infer.add_subparsers(dest="op", required=True)

    for name, func, help_text in (
        ("trace", cmd_infer_trace, "every finding in one segment"),
        ("message", cmd_infer_message, "one message in detail, with graphs"),
    ):
        p = iops.add_parser(name, help=help_text)
        p.add_argument("path", help="path to rlog.zst")
        p.add_argument(
            "--order",
            choices=[o.value for o in BitOrder],
            default=BitOrder.INTEL.value,
            help="payload bit numbering (default: %(default)s)",
        )
        if name == "trace":
            p.add_argument("--top", type=int, default=30, help="rows to print (default: 30)")
        else:
            p.add_argument("--address", required=True, help="CAN address, e.g. 0x210")
            p.add_argument("--bus", type=int, help="restrict to one bus")
            p.add_argument(
                "--samples", type=int, help="counter samples to plot (default: fit the terminal)"
            )
            p.add_argument("--no-color", action="store_true", help="never emit ANSI colour")
        p.set_defaults(func=func)

    export = sub.add_parser("export", help="write findings in other tools' formats")
    eops = export.add_subparsers(dest="op", required=True)
    p = eops.add_parser("pdu-db", help="BoAt PDU database JSON")
    p.add_argument("path", help="path to rlog.zst")
    p.add_argument("-o", "--output", required=True, help="destination .json")
    p.set_defaults(func=cmd_export_pdu_db)

    truth = sub.add_parser("truth", help="score inference against an opendbc DBC")
    tops = truth.add_subparsers(dest="op", required=True)

    p = tops.add_parser("dbc", help="what ground truth a DBC offers")
    p.add_argument("dbc", help="path to a .dbc file")
    p.add_argument("-v", "--verbose", action="store_true", help="list every named field")
    p.set_defaults(func=cmd_truth_dbc)

    p = tops.add_parser("score", help="compare one segment against a DBC")
    p.add_argument("path", help="path to rlog.zst")
    p.add_argument("--dbc", required=True, help="path to the .dbc to score against")
    p.add_argument("--bus", type=int, help="bus to score (default: best identifier overlap)")
    p.add_argument("--top", type=int, default=25, help="disagreements to print (default: 25)")
    p.add_argument(
        "--kind",
        choices=["counter", "checksum", "multiplexor"],
        help="show disagreements of one kind only",
    )
    p.add_argument("--no-detail", action="store_true", help="rates only, no disagreements")
    p.add_argument(
        "--order",
        choices=[o.value for o in BitOrder],
        default=BitOrder.INTEL.value,
        help="payload bit numbering (default: %(default)s)",
    )
    p.set_defaults(func=cmd_truth_score)

    p = sub.add_parser("corroborate", help="what holds across every local segment of a platform")
    p.add_argument("platform", help="platform key; see 'canlens corpus list'")
    p.add_argument("--address", help="one message in detail, e.g. 0x211")
    p.add_argument("--bus", type=int, help="with --address: restrict to one bus")
    p.add_argument("--limit", type=int, help="use at most this many segments")
    p.add_argument("--top", type=int, default=40, help="rows in the overview (default: 40)")
    p.add_argument("--no-color", action="store_true", help="never emit ANSI colour")
    p.add_argument(
        "--logged-buses",
        action="store_true",
        help="group by the bus number as logged, without reconciling them",
    )
    p.set_defaults(func=cmd_corroborate, op=None)

    p = sub.add_parser(
        "corroborate-pooled",
        help="solve Data ID lists no single segment carries enough evidence for",
    )
    p.add_argument("platform", help="platform key; see 'canlens corpus list'")
    p.add_argument("--limit", type=int, help="segments to pool (default: all local)")
    p.add_argument(
        "--min-devices", type=int, default=2,
        help="distinct cars required (default: %(default)s)",
    )
    p.set_defaults(func=cmd_corroborate_pooled, op=None)

    cache = sub.add_parser("cache", help="decode segments once and keep the columns")
    cops = cache.add_subparsers(dest="op", required=True)
    p = cops.add_parser("build", help="decode, infer and cache local segments in parallel")
    p.add_argument("platform", nargs="*", help="platform key(s); default every local one")
    p.add_argument("--jobs", type=int, help="worker processes (default: physical cores)")
    p.set_defaults(func=cmd_cache_build)
    p = cops.add_parser("clear", help="delete every cache entry")
    p.set_defaults(func=cmd_cache_clear)
    p = cops.add_parser("status", help="how much of what is local is cached")
    p.set_defaults(func=cmd_cache_status)

    p = sub.add_parser("gui", help="open the desktop workbench")
    p.set_defaults(func=cmd_gui, op=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyError as exc:
        print(f"canlens: {exc.args[0]}", file=sys.stderr)
        print("run 'canlens corpus list' to see valid platform keys", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted -- re-run to resume", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
