"""Explicit maintenance pipeline for existing or downloaded editions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from src.content.normalization import NormalizationReport
from src.epub.metadata import enrich_epub_metadata
from src.epub.normalize import normalize_epub
from src.epub.reports import automatic_report_path, write_reports
from src.epub.review import (
    _load_cached_review_decisions,
    _reuse_cached_reviews,
    review_epub_with_codex,
)
from src.metadata.catalog import MetadataEnrichmentReport


def _codex_review_enabled() -> bool:
    value = os.environ.get("BOOKLIB_CODEX_REVIEW", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def normalize_and_review_epub(
    path: Path,
    *,
    apply: bool = True,
    backup_dir: Path | None = None,
    overwrite_backup: bool = False,
    codex_review: bool | None = None,
    review_runner: Callable[[str], str] | None = None,
    review_timeout_seconds: int = 300,
) -> NormalizationReport:
    report = normalize_epub(
        path,
        apply=apply,
        backup_dir=backup_dir,
        overwrite_backup=overwrite_backup,
    )
    enabled = _codex_review_enabled() if codex_review is None else codex_review
    if not enabled:
        report.codex_review_status = "skipped"
        return report
    if review_runner is None:
        _reuse_cached_reviews(
            report,
            _load_cached_review_decisions(Path(path)),
        )
    return review_epub_with_codex(
        path,
        report,
        apply=apply,
        backup_dir=backup_dir,
        overwrite_backup=overwrite_backup,
        runner=review_runner,
        timeout_seconds=review_timeout_seconds,
    )


def attach_metadata_enrichment(
    report: NormalizationReport,
    metadata: MetadataEnrichmentReport,
) -> NormalizationReport:
    report.metadata_enrichment = metadata
    member = metadata.opf_member or "EPUB/content.opf"
    for field_name in metadata.changed_fields:
        report.record_change(
            f"metadata_{field_name}_updated",
            member,
            1,
            str(metadata.before.get(field_name, "")),
            str(metadata.after.get(field_name, "")),
        )
    return report


def normalize_new_epub(path: Path) -> NormalizationReport:
    metadata = enrich_epub_metadata(path, backup_dir=None)
    report = normalize_and_review_epub(path)
    attach_metadata_enrichment(report, metadata)
    report_path = automatic_report_path(path)
    write_reports(report_path, [report])
    print(report.format_text())
    print(f"Wrote normalization report: {report_path}")
    return report
