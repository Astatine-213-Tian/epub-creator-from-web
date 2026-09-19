"""Dataset command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.dataset.library import (
    export_single_epub_txt,
    export_txt_dataset,
    upsert_txt_dataset_entry,
)
from src.metadata.jjwxc import GENRES, TIME_AREAS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build and maintain targeted TXT dataset entries."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser(
        "export-txt",
        help="Export TXT from an explicit EPUB or explicit EPUB tree.",
    )
    export_source = export_parser.add_mutually_exclusive_group(required=True)
    export_source.add_argument(
        "--epub", type=Path, help="Single EPUB to export and upsert"
    )
    export_source.add_argument(
        "--books-root", type=Path, help="Explicit EPUB directory for a bulk export"
    )
    export_parser.add_argument(
        "--output-root", type=Path, default=Path("research/datasets")
    )
    export_parser.add_argument(
        "--jjwxc-manifest",
        type=Path,
        default=Path("research/generated/corpus_acquisition/ranking/manifest.json"),
    )
    export_parser.add_argument(
        "--jjwxc-top50",
        type=Path,
        default=Path("research/generated/corpus_acquisition/ranking/top50.json"),
    )
    export_parser.add_argument("--overwrite", action="store_true")
    export_parser.add_argument("--no-fetch-jjwxc", action="store_true")
    export_parser.add_argument("--no-codex-classify", action="store_true")
    export_parser.add_argument(
        "--exclude-title",
        action="append",
        default=[],
        help="Bulk only: skip EPUBs whose metadata title exactly matches this value; repeatable",
    )
    export_parser.add_argument(
        "--exclude-book",
        action="append",
        default=[],
        metavar="AUTHOR::TITLE",
        help="Bulk only: skip EPUBs whose metadata author/title match this value; repeatable",
    )

    upsert_parser = subparsers.add_parser(
        "upsert",
        help="Upsert one existing TXT file into dataset_manifest.json.",
    )
    upsert_parser.add_argument("--txt", type=Path, required=True)
    upsert_parser.add_argument("--title")
    upsert_parser.add_argument("--author")
    upsert_parser.add_argument("--source-epub", type=Path)
    upsert_parser.add_argument(
        "--output-root", type=Path, default=Path("research/datasets")
    )
    upsert_parser.add_argument(
        "--jjwxc-manifest",
        type=Path,
        default=Path("research/generated/corpus_acquisition/ranking/manifest.json"),
    )
    upsert_parser.add_argument(
        "--jjwxc-top50",
        type=Path,
        default=Path("research/generated/corpus_acquisition/ranking/top50.json"),
    )
    upsert_parser.add_argument("--time-area", choices=TIME_AREAS)
    upsert_parser.add_argument("--genre", choices=GENRES)
    upsert_parser.add_argument("--no-fetch-jjwxc", action="store_true")
    upsert_parser.add_argument("--no-codex-classify", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "upsert":
        entry = upsert_txt_dataset_entry(
            txt_path=args.txt,
            output_root=args.output_root,
            jjwxc_manifest=args.jjwxc_manifest,
            jjwxc_top50=args.jjwxc_top50,
            fetch_jjwxc=not args.no_fetch_jjwxc,
            codex_classify=not args.no_codex_classify,
            title=args.title,
            author=args.author,
            source_epub=args.source_epub,
            time_area=args.time_area,
            genre=args.genre,
        )
        print(f"upserted {entry.author}::{entry.title}")
        print(f"txt: {entry.txt_path}")
        print(f"manifest: {args.output_root / 'dataset_manifest.json'}")
        return 0

    if args.epub:
        entry = export_single_epub_txt(
            epub_path=args.epub,
            output_root=args.output_root,
            jjwxc_manifest=args.jjwxc_manifest,
            jjwxc_top50=args.jjwxc_top50,
            overwrite=args.overwrite,
            fetch_jjwxc=not args.no_fetch_jjwxc,
            codex_classify=not args.no_codex_classify,
        )
        print(
            f"exported 1 EPUB: written={1 if entry.status == 'written' else 0}, "
            f"skipped_existing={1 if entry.status == 'skipped_existing' else 0}, failed=0"
        )
        print(f"txt: {entry.txt_path}")
        print(f"manifest: {args.output_root / 'dataset_manifest.json'}")
        return 0

    entries = export_txt_dataset(
        books_root=args.books_root,
        output_root=args.output_root,
        jjwxc_manifest=args.jjwxc_manifest,
        jjwxc_top50=args.jjwxc_top50,
        overwrite=args.overwrite,
        fetch_jjwxc=not args.no_fetch_jjwxc,
        codex_classify=not args.no_codex_classify,
        exclude_titles=set(args.exclude_title),
        exclude_metadata_keys={f"local::{item}" for item in args.exclude_book},
    )
    written = sum(1 for entry in entries if entry.status == "written")
    skipped = sum(1 for entry in entries if entry.status == "skipped_existing")
    failed = sum(1 for entry in entries if entry.status == "failed")
    print(
        f"exported {len(entries)} EPUBs: written={written}, "
        f"skipped_existing={skipped}, failed={failed}"
    )
    print(f"manifest: {args.output_root / 'dataset_manifest.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
