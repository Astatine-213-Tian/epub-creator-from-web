#!/usr/bin/env python3
"""Auto-detect a supported novel site and build an EPUB.

Usage:
    python book_to_epub.py <url> [-o output.epub]
    python book_to_epub.py 2574 --parser mgsf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from booklib.parser_registry import PARSERS, ParserOptions, find_parser


def main(argv: list[str] | None = None) -> int:
    parser_names = [parser.name for parser in PARSERS]

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("target", nargs="?", help="Book URL, or site-specific id with --parser")
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("--parser", choices=parser_names, help="Force a parser for ids or ambiguous URLs")
    p.add_argument("--delay", type=float, default=None, help="Seconds to wait between requests")
    p.add_argument(
        "--headless",
        action="store_true",
        help="Run browser-backed parsers headless where supported",
    )
    p.add_argument("--list-parsers", action="store_true", help="Show supported parsers")
    args = p.parse_args(argv)

    if args.list_parsers:
        for spec in PARSERS:
            print(f"{spec.name}: {', '.join(spec.domains)} — {spec.description}")
        return 0

    if not args.target:
        p.error("target is required unless --list-parsers is used")

    try:
        spec = find_parser(args.target, args.parser)
        print(f"[+] using parser: {spec.name}", file=sys.stderr)
        out_path = spec.run(
            args.target,
            ParserOptions(output=args.output, delay=args.delay, headless=args.headless),
        )
    except Exception as exc:
        print(f"[!] {exc}", file=sys.stderr)
        return 1

    print(f"[+] wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
