#!/usr/bin/env python3
"""Auto-detect a supported novel site and build an EPUB.

Usage:
    uv run book-to-epub <url> [-o output.epub]
    uv run book-to-epub 2574 --parser mgsf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.providers.registry import PARSERS, ParserOptions, find_parser
from src.runtime.progress import ProgressLogger, configure_progress
from src.search import build_previews, choose_preview, fake_menu_previews, search_all


def main(argv: list[str] | None = None) -> int:
    parser_names = [parser.name for parser in PARSERS]

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("target", nargs="?", help="Book URL, or site-specific id with --parser")
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("--parser", choices=parser_names, help="Force a parser for ids or ambiguous URLs")
    p.add_argument("--search", help="Search supported providers, preview matches, then choose one to parse")
    p.add_argument("--limit", type=int, default=10, help="Maximum search results/previews to show")
    p.add_argument("--first", action="store_true", help="With --search, choose the top ranked result")
    p.add_argument("--delay", type=float, default=None, help="Seconds to wait between requests")
    p.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Maximum chapter fetch concurrency where supported",
    )
    p.add_argument(
        "--headless",
        action="store_true",
        help="Run browser-backed parsers headless where supported",
    )
    p.add_argument("--verbose", action="store_true", help="Show detailed search/preview fetch logs")
    p.add_argument("--list-parsers", action="store_true", help="Show supported parsers")
    p.add_argument(
        "--test-menu",
        action="store_true",
        help="Open the result selection menu with fixed fake preview data, then exit",
    )
    args = p.parse_args(argv)
    configure_progress(debug=args.verbose)
    progress = ProgressLogger()

    if args.list_parsers:
        for spec in PARSERS:
            print(f"{spec.name}: {', '.join(spec.domains)} — {spec.description}")
        return 0

    if args.test_menu:
        selected = choose_preview(fake_menu_previews(), first=args.first)
        if selected is None:
            progress.warning("cancelled")
            return 1
        progress.info(f"selected {selected.parser}: {selected.title} ({selected.url})")
        return 0

    if args.search:
        scope = f" with parser {args.parser}" if args.parser else ""
        progress.section("Search")
        progress.info(f"Query: {args.search!r}{scope}")
        results = search_all(
            args.search,
            parser_name=args.parser,
            limit_per_provider=max(1, args.limit),
            verbose=True,
            debug=args.verbose,
        )
        if not results:
            progress.warning("no search results")
            return 1

        progress.section("Preview")
        previews = build_previews(
            args.search,
            results,
            max_previews=max(1, args.limit),
            verbose=True,
            debug=args.verbose,
        )
        if not previews:
            progress.warning("no previewable search results")
            return 1

        selected = choose_preview(previews, first=args.first)
        if selected is None:
            progress.warning("cancelled")
            return 1

        args.target = selected.url
        args.parser = selected.parser

    if not args.target:
        p.error("target is required unless --list-parsers or --search is used")

    try:
        spec = find_parser(args.target, args.parser)
        progress.info(f"using parser: {spec.name}")
        out_path = spec.run(
            args.target,
            ParserOptions(
                output=args.output,
                delay=args.delay,
                headless=args.headless,
                concurrency=args.concurrency,
            ),
        )
    except Exception as exc:
        progress.warning(str(exc))
        return 1

    progress.info(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
