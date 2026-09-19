#!/usr/bin/env python3
"""Crawl a novel, then upload a CMS draft or write local EPUB/TXT output.

Usage:
    uv run book-to-epub <url> [-o output.epub]
    uv run book-to-epub 2574 --parser mgsf --output-format epub
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.crawler.models import CrawlOptions
from src.crawler.registry import PARSERS, find_parser
from src.crawler.search import (
    build_previews,
    choose_preview,
    fake_menu_previews,
    search_all,
)
from src.runtime.progress import ProgressLogger, configure_progress
from src.workflows.ingest import OutputOptions, ingest, requested_formats


def main(argv: list[str] | None = None) -> int:
    parser_names = [parser.name for parser in PARSERS]

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "target", nargs="?", help="Book URL, or site-specific id with --parser"
    )
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument(
        "--txt-output",
        type=Path,
        default=None,
        help="Explicit TXT output path when --output-format txt is requested",
    )
    p.add_argument(
        "--output-format",
        action="append",
        choices=["notion", "epub", "txt", "both"],
        default=None,
        help="Output destination; repeat to combine epub, notion and txt. No default. both means epub plus txt",
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Write TXT output under author child folders rooted here. Use book-ingest or book-dataset upsert to update the manifest.",
    )
    overwrite_group = p.add_mutually_exclusive_group()
    overwrite_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing requested output files. Default: skip existing outputs",
    )
    overwrite_group.add_argument(
        "--prevent-overwrite",
        action="store_true",
        help="Skip writing any requested output file that already exists. This is the default",
    )
    p.add_argument(
        "--parser",
        choices=parser_names,
        help="Force a parser for ids or ambiguous URLs",
    )
    p.add_argument(
        "--search",
        help="Search supported providers, preview matches, then choose one to parse",
    )
    p.add_argument(
        "--author",
        help="With --search, pass an author hint to providers that support it",
    )
    p.add_argument(
        "--limit", type=int, default=10, help="Maximum search results/previews to show"
    )
    p.add_argument(
        "--first",
        action="store_true",
        help="With --search, choose the top ranked result",
    )
    p.add_argument(
        "--delay", type=float, default=None, help="Seconds to wait between requests"
    )
    p.add_argument(
        "--request-interval",
        type=float,
        default=0.0,
        help="Minimum interval between xfxs chapter requests",
    )
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
    p.add_argument(
        "--verbose", action="store_true", help="Show detailed search/preview fetch logs"
    )
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

    if args.author and not args.search:
        p.error("--author can only be used with --search")

    output_options = OutputOptions(
        output=args.output,
        txt_output=args.txt_output,
        output_formats=tuple(args.output_format or ()),
        dataset_root=args.dataset_root,
        prevent_overwrite=not args.overwrite,
    )
    try:
        requested_formats(output_options)
    except ValueError as error:
        p.error(str(error))

    if args.search:
        scope = f" with parser {args.parser}" if args.parser else ""
        author_scope = f" / author {args.author!r}" if args.author else ""
        progress.section("Search")
        progress.info(f"Query: {args.search!r}{author_scope}{scope}")
        results = search_all(
            args.search,
            parser_name=args.parser,
            author=args.author,
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
        result = ingest(
            args.target,
            parser=spec,
            output_options=output_options,
            crawl_options=CrawlOptions(
                delay=args.delay,
                request_interval=args.request_interval,
                headless=args.headless,
                concurrency=args.concurrency,
            ),
        )
    except Exception as exc:
        progress.warning(str(exc))
        return 1

    for label, path in (
        ("EPUB", result.epub_path),
        ("TXT", result.txt_path),
        ("Notion draft checkpoint", result.notion_state),
    ):
        if path:
            progress.info(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
