from __future__ import annotations

import argparse
from pathlib import Path

from src.core.epub_normalizer import (
    NormalizationReport,
    attach_metadata_enrichment,
    automatic_report_path,
    normalize_and_review_epub,
    write_reports,
)
from src.metadata.epub_enricher import (
    DEFAULT_PRIMARY_SUBJECT,
    MetadataLookup,
    enrich_epub_metadata,
)


DEFAULT_BACKUP_DIR = Path(
    "/private/tmp/epub-creator-from-web-codex-backups"
)


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
            raise ValueError(
                "--jjwxc-author-id must use AUTHOR=NUMERIC_ID"
            )
        result[author.strip()] = author_id.strip()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Enrich EPUB metadata, normalize reader-visible text, and report "
            "unresolved quote or suspicious-character issues."
        )
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Audit and report proposed fixes without rewriting EPUB files.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Write the complete machine-readable JSON report to this path.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a pre-normalization backup for manual rewrites.",
    )
    parser.add_argument(
        "--overwrite-backup",
        action="store_true",
        help="Overwrite an existing backup instead of preserving it.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="List fix counts for every changed archive member.",
    )
    parser.add_argument(
        "--no-codex-review",
        action="store_true",
        help="Skip the automatic second-stage Codex review.",
    )
    parser.add_argument(
        "--codex-review-timeout",
        type=int,
        default=300,
        metavar="SECONDS",
        help="Timeout for each automatic Codex review batch (default: 300).",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Skip Jinjiang-first metadata enrichment.",
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
        help="Supply a known Jinjiang author ID; repeat for aliases.",
    )
    parser.add_argument(
        "--no-kadokado",
        action="store_true",
        help="Disable KadoKado fallback after a confirmed Jinjiang miss.",
    )
    parser.add_argument(
        "--metadata-timeout",
        type=int,
        default=20,
        metavar="SECONDS",
        help="Timeout for each metadata source request (default: 20).",
    )
    parser.add_argument(
        "--fail-on-issues",
        action="store_true",
        help="Exit unsuccessfully when Codex or user review is still required.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        paths = _epub_paths(args.paths)
        author_ids = _author_ids(args.jjwxc_author_id)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not paths:
        raise SystemExit("No EPUB files found")

    metadata_lookup = None
    if not args.no_metadata:
        metadata_lookup = MetadataLookup(
            author_ids=author_ids,
            kadokado=not args.no_kadokado,
            timeout=args.metadata_timeout,
        )
    reports: list[NormalizationReport] = []
    backup_dir = None if args.no_backup or args.check else DEFAULT_BACKUP_DIR
    for path in paths:
        metadata = None
        if metadata_lookup is not None:
            metadata = enrich_epub_metadata(
                path,
                apply=not args.check,
                backup_dir=backup_dir,
                overwrite_backup=args.overwrite_backup,
                primary_subject=args.primary_subject,
                lookup=metadata_lookup,
            )
        normalizer_overwrite = args.overwrite_backup and not (
            metadata is not None and metadata.applied and metadata.changed
        )
        report = normalize_and_review_epub(
            path,
            apply=not args.check,
            backup_dir=backup_dir,
            overwrite_backup=normalizer_overwrite,
            codex_review=not args.no_codex_review,
            review_timeout_seconds=args.codex_review_timeout,
        )
        if metadata is not None:
            attach_metadata_enrichment(report, metadata)
        reports.append(report)
        print(report.format_text(include_members=args.verbose))

    if args.report:
        write_reports(args.report, reports)
        print(f"Wrote normalization report: {args.report}")
    else:
        for report in reports:
            report_path = automatic_report_path(report.path)
            write_reports(report_path, [report])
            print(f"Wrote normalization report: {report_path}")

    metadata_reports = [
        report.metadata_enrichment
        for report in reports
        if report.metadata_enrichment is not None
    ]
    print(
        "SUMMARY "
        f"books={len(reports)} "
        f"fixes={sum(report.total_changes for report in reports)} "
        f"changed_members={sum(report.changed_members for report in reports)} "
        f"metadata_changed={sum(item.changed for item in metadata_reports)} "
        f"metadata_unmatched={sum(item.status == 'unmatched' for item in metadata_reports)} "
        f"metadata_errors={sum(item.status == 'error' for item in metadata_reports)} "
        f"reported_issues={sum(len(report.issues) for report in reports)} "
        f"pending_codex={sum(issue.requires_codex_review for report in reports for issue in report.issues)} "
        f"needs_user={sum(issue.requires_user_review for report in reports for issue in report.issues)}"
    )
    if args.fail_on_issues and any(
        issue.requires_codex_review or issue.requires_user_review
        for report in reports
        for issue in report.issues
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
