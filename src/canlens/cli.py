# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""`canlens` command line entry point."""
from __future__ import annotations

import argparse
import os
import sys

from . import __version__
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
        print(path)
        print(
            f"  {s.frames - s.echoes} frames over {s.duration_s:.1f}s "
            f"({s.echoes} echoes dropped, {echo_pct:.0f}% of raw)"
        )
        print(f"  {s.unique_addresses} unique (bus, address) pairs")
        for bus, n in s.buses.items():
            addrs = sum(1 for b, _ in s.addresses if b == bus)
            print(f"    bus {bus}: {n:>7} frames, {addrs:>4} addresses")
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

    decode = sub.add_parser("decode", help="turn raw logs into CAN frames")
    dops = decode.add_subparsers(dest="op", required=True)

    p = dops.add_parser("schema", help="fetch the capnp schemas needed to read rlogs")
    p.add_argument("--refresh", action="store_true", help="re-download even if cached")
    p.set_defaults(func=cmd_decode_schema)

    p = dops.add_parser("summary", help="report what a segment contains")
    p.add_argument("path", nargs="+", help="path(s) to rlog.zst")
    p.set_defaults(func=cmd_decode_summary)

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
