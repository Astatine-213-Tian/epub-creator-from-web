#!/usr/bin/env python3
"""Crawl one book and write the selected EPUB, Notion draft and/or TXT outputs."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

from src.crawler.models import CrawlOptions
from src.crawler.registry import PARSERS, find_parser
from src.crawler.search import build_previews, choose_preview, search_all
from src.dataset.library import extract_epub_text, upsert_txt_dataset_entry
from src.epub.validate import validate_epub
from src.metadata.jjwxc import GENRES, TIME_AREAS
from src.runtime.progress import ProgressLogger, configure_progress
from src.workflows.ingest import OutputOptions, ingest, requested_formats


def validate_epub_output(epub_path: Path, *, strict: bool) -> bool:
    with zipfile.ZipFile(epub_path) as zf:
        bad_member = zf.testzip()
    if bad_member:
        print(f"FAIL {epub_path}: corrupt ZIP member {bad_member}", file=sys.stderr)
        return False

    issues, chapter_count = validate_epub(epub_path)
    if issues:
        print(f"WARN {epub_path}: {chapter_count} numbered chapter entries")
        for issue in issues:
            print(f"  - {issue}")
        return not strict

    print(f"OK   {epub_path}: ZIP valid; {chapter_count} numbered chapter entries")
    return True


def select_search_target(
    args: argparse.Namespace, progress: ProgressLogger
) -> tuple[str, str | None]:
    if not args.search:
        if not args.target:
            raise ValueError("target is required unless --search is used")
        return args.target, args.parser

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
        raise ValueError("no search results")

    progress.section("Preview")
    previews = build_previews(
        args.search,
        results,
        max_previews=max(1, args.limit),
        verbose=True,
        debug=args.verbose,
    )
    if not previews:
        raise ValueError("no previewable search results")

    selected = choose_preview(previews, first=args.first)
    if selected is None:
        raise ValueError("cancelled")
    return selected.url, selected.parser


def main(argv: list[str] | None = None) -> int:
    parser_names = [parser.name for parser in PARSERS]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target", nargs="?", help="Book URL, or site-specific id with --parser"
    )
    parser.add_argument(
        "--mode",
        "--output-format",
        action="append",
        choices=["notion", "epub", "both", "txt"],
        default=None,
        help="Output destination; repeat to combine epub, notion and txt. No default. both means epub plus txt",
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="EPUB output path for epub/both modes"
    )
    parser.add_argument(
        "--txt-output", type=Path, help="TXT output path for txt/both modes"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("research/datasets"),
    )
    parser.add_argument(
        "--parser",
        choices=parser_names,
        help="Force a parser for ids or ambiguous URLs",
    )
    parser.add_argument(
        "--search", help="Search supported providers, preview matches, then choose one"
    )
    parser.add_argument(
        "--author",
        help="With --search, pass an author hint to providers that support it",
    )
    parser.add_argument(
        "--limit", type=int, default=10, help="Maximum search results/previews to show"
    )
    parser.add_argument(
        "--first",
        action="store_true",
        help="With --search, choose the top ranked result",
    )
    parser.add_argument(
        "--delay", type=float, default=None, help="Seconds to wait between requests"
    )
    parser.add_argument(
        "--request-interval",
        type=float,
        default=0.0,
        help="Minimum interval between xfxs chapter requests",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Maximum chapter fetch concurrency where supported",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser-backed parsers headless where supported",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace requested output files"
    )
    parser.add_argument("--no-fetch-jjwxc", action="store_true")
    parser.add_argument("--no-codex-classify", action="store_true")
    parser.add_argument(
        "--jjwxc-manifest",
        type=Path,
        default=Path("research/generated/corpus_acquisition/ranking/manifest.json"),
    )
    parser.add_argument(
        "--jjwxc-top50",
        type=Path,
        default=Path("research/generated/corpus_acquisition/ranking/top50.json"),
    )
    parser.add_argument(
        "--time-area", choices=TIME_AREAS, help="Manual dataset time-area override"
    )
    parser.add_argument("--genre", choices=GENRES, help="Manual dataset genre override")
    parser.add_argument(
        "--strict-epub-validation",
        action="store_true",
        help="Return failure on chapter-number validator warnings instead of reporting them as warnings",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed search/preview fetch logs"
    )
    args = parser.parse_args(argv)

    configure_progress(debug=args.verbose)
    progress = ProgressLogger()
    output_options = OutputOptions(
        output=args.output,
        txt_output=args.txt_output,
        output_formats=tuple(args.mode or ()),
        dataset_root=args.dataset_root if not args.txt_output else None,
        prevent_overwrite=not args.overwrite,
    )
    try:
        requested_formats(output_options)
    except ValueError as error:
        parser.error(str(error))
    if args.author and not args.search:
        parser.error("--author can only be used with --search")

    try:
        target, parser_name = select_search_target(args, progress)
        spec = find_parser(target, parser_name)
        progress.info(f"using parser: {spec.name}")
        result = ingest(
            target,
            parser=spec,
            output_options=output_options,
            crawl_options=CrawlOptions(
                delay=args.delay,
                request_interval=args.request_interval,
                headless=args.headless,
                concurrency=args.concurrency,
            ),
        )

        epub_path = result.epub_path
        title: str | None = None
        author: str | None = None
        if epub_path:
            title, author, _ = extract_epub_text(epub_path)
            if not validate_epub_output(epub_path, strict=args.strict_epub_validation):
                return 1

        if result.txt_path:
            entry = upsert_txt_dataset_entry(
                txt_path=result.txt_path,
                output_root=args.dataset_root,
                jjwxc_manifest=args.jjwxc_manifest,
                jjwxc_top50=args.jjwxc_top50,
                fetch_jjwxc=not args.no_fetch_jjwxc,
                codex_classify=not args.no_codex_classify,
                title=title,
                author=author,
                source_epub=epub_path,
                time_area=args.time_area,
                genre=args.genre,
            )
            progress.info(f"dataset upserted: {entry.author}::{entry.title}")
            progress.info(f"txt: {entry.txt_path}")

        for label, path in (
            ("EPUB", result.epub_path),
            ("TXT", result.txt_path),
            ("Notion draft checkpoint", result.notion_state),
        ):
            if path:
                progress.info(f"{label}: {path}")
        return 0
    except Exception as exc:
        progress.warning(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
