from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.epub.metadata import enrich_epub_metadata
from src.metadata.catalog import (
    DEFAULT_PRIMARY_SUBJECT,
    MetadataEnrichmentReport,
    MetadataLookup,
)
from src.runtime.files import DEFAULT_BACKUP_DIR


def _epub_paths(values: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        if value.is_dir():
            paths.extend(sorted(value.rglob("*.epub")))
        elif value.suffix.lower() == ".epub":
            paths.append(value)
        else:
            raise ValueError(f"Not an EPUB file or directory: {value}")
    return sorted(dict.fromkeys(paths))


def _author_ids(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        author, separator, author_id = value.partition("=")
        if not separator or not author.strip() or not author_id.strip().isdigit():
            raise ValueError("--jjwxc-author-id must use AUTHOR=NUMERIC_ID")
        result[author.strip()] = author_id.strip()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Enrich EPUB content.opf metadata from Jinjiang, with "
            "KadoKado used only when Jinjiang has no verified match."
        )
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report proposed metadata changes without rewriting EPUB files.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Write the complete machine-readable JSON report to this path.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a pre-enrichment backup.",
    )
    parser.add_argument(
        "--overwrite-backup",
        action="store_true",
        help="Overwrite an existing backup with the current pre-edit EPUB.",
    )
    parser.add_argument(
        "--primary-subject",
        default=DEFAULT_PRIMARY_SUBJECT,
        help=(
            "First dc:subject for Jinjiang matches "
            f"(default: {DEFAULT_PRIMARY_SUBJECT})."
        ),
    )
    parser.add_argument(
        "--jjwxc-author-id",
        action="append",
        default=[],
        metavar="AUTHOR=ID",
        help=("Supply a known Jinjiang author ID; repeat for multiple aliases."),
    )
    parser.add_argument(
        "--no-kadokado",
        action="store_true",
        help="Disable KadoKado fallback after a confirmed Jinjiang miss.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        metavar="SECONDS",
        help="Timeout for each metadata source request (default: 20).",
    )
    parser.add_argument(
        "--fail-on-unmatched",
        action="store_true",
        help="Exit unsuccessfully if a book is unmatched or lookup fails.",
    )
    return parser


def write_report(
    path: Path,
    reports: list[MetadataEnrichmentReport],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reports": [report.to_dict() for report in reports],
        "summary": {
            "books": len(reports),
            "changed": sum(report.changed for report in reports),
            "unmatched": sum(report.status == "unmatched" for report in reports),
            "errors": sum(report.status == "error" for report in reports),
        },
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = build_parser().parse_args()
    try:
        paths = _epub_paths(args.paths)
        author_ids = _author_ids(args.jjwxc_author_id)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not paths:
        raise SystemExit("No EPUB files found")

    lookup = MetadataLookup(
        author_ids=author_ids,
        kadokado=not args.no_kadokado,
        timeout=args.timeout,
    )
    backup_dir = None if args.no_backup or args.check else DEFAULT_BACKUP_DIR
    reports = [
        enrich_epub_metadata(
            path,
            apply=not args.check,
            backup_dir=backup_dir,
            overwrite_backup=args.overwrite_backup,
            primary_subject=args.primary_subject,
            lookup=lookup,
        )
        for path in paths
    ]
    for report in reports:
        print(report.format_text())
    if args.report:
        write_report(args.report, reports)
        print(f"Wrote metadata report: {args.report}")
    print(
        "SUMMARY "
        f"books={len(reports)} "
        f"changed={sum(report.changed for report in reports)} "
        f"unmatched={sum(report.status == 'unmatched' for report in reports)} "
        f"errors={sum(report.status == 'error' for report in reports)}"
    )
    if args.fail_on_unmatched and any(
        report.status in {"unmatched", "error"} for report in reports
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
