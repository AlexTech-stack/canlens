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
