#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from src.runtime.progress import ProgressLogger, configure_progress


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Crawl a supported source into a reusable snapshot for later translation."
    )
    parser.add_argument("target", help="Collection/book URL to crawl")
    parser.add_argument("--provider", choices=("patreon",), required=True)
    parser.add_argument("--output", type=Path, required=True, help="Snapshot output directory")
    parser.add_argument("--title", help="Override book title in the snapshot")
    parser.add_argument("--author", help="Override author in the snapshot")
    parser.add_argument("--delay", type=float, default=0.4)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-comments", action="store_true", help="Do not crawl comments")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    configure_progress(debug=args.verbose)
    progress = ProgressLogger()

    try:
        if args.provider == "patreon":
            from src.providers.patreon import parser as patreon

            progress.section("Crawl")
            progress.info(f"provider: patreon")
            progress.info(f"snapshot: {args.output}")
            manifest = asyncio.run(
                patreon.crawl_snapshot(
                    args.target,
                    args.output,
                    title=args.title,
                    author=args.author,
                    headless=args.headless,
                    delay=args.delay,
                    concurrency=max(1, args.concurrency),
                    include_comments=not args.no_comments,
                )
            )
        else:  # pragma: no cover - argparse prevents this today.
            parser.error(f"unsupported provider: {args.provider}")
    except Exception as exc:  # noqa: BLE001
        progress.warning(str(exc))
        return 1

    progress.info(f"wrote crawl snapshot {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
