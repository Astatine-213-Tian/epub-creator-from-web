"""Collect once, normalize once, then write to explicitly selected destinations."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from src.content.prepare import prepare_source
from src.crawler.models import CrawledBook, CrawlOptions, DownloadedEdition
from src.crawler.registry import ParserSpec
from src.dataset.text import write_txt
from src.runtime.paths import (
    dataset_txt_output_path,
    resolve_output_path,
    resolve_txt_output_path,
)


@dataclass(frozen=True)
class OutputOptions:
    output: Path | None = None
    txt_output: Path | None = None
    output_formats: tuple[str, ...] = ()
    dataset_root: Path | None = None
    prevent_overwrite: bool = True


@dataclass(frozen=True)
class IngestResult:
    epub_path: Path | None = None
    txt_path: Path | None = None
    notion_state: Path | None = None


def requested_formats(options: OutputOptions) -> tuple[str, ...]:
    formats: list[str] = []
    chosen = options.output_formats or tuple(
        name
        for name, path in (("epub", options.output), ("txt", options.txt_output))
        if path is not None
    )
    if not chosen:
        raise ValueError(
            "Choose an output with --mode/--output-format epub, notion or txt; "
            "repeat the option to combine formats, or provide -o/--txt-output"
        )
    for output_format in chosen:
        normalized = output_format.strip().lower()
        if normalized == "both":
            normalized_formats = ("epub", "txt")
        else:
            normalized_formats = (normalized,)
        for item in normalized_formats:
            if item not in {"notion", "epub", "txt"}:
                raise ValueError(f"unsupported output format: {output_format}")
            if item not in formats:
                formats.append(item)
    if options.output is not None and "epub" not in formats:
        raise ValueError("--output requires the epub format")
    if options.txt_output is not None and "txt" not in formats:
        raise ValueError("--txt-output requires the txt format")
    return tuple(formats)


def resolve_requested_txt_output(
    options: OutputOptions,
    *,
    title: str,
    author: str,
) -> Path:
    if options.txt_output is not None:
        return resolve_txt_output_path(options.txt_output, title, author)
    if options.dataset_root:
        return dataset_txt_output_path(
            options.dataset_root,
            title=title,
            author=author,
            time_area="",
            genre="",
        )
    return resolve_txt_output_path(options.txt_output, title, author)


def write_outputs(book: CrawledBook, options: OutputOptions) -> IngestResult:
    formats = requested_formats(options)
    epub_path = (
        resolve_output_path(options.output, book.title, book.author)
        if "epub" in formats
        else None
    )
    txt_path = (
        resolve_requested_txt_output(
            options,
            title=book.title,
            author=book.author,
        )
        if "txt" in formats
        else None
    )

    if epub_path and options.prevent_overwrite and epub_path.exists():
        raise ValueError("Output EPUB exists; choose another path or use --overwrite")
    notion_path = None
    if "notion" in formats or epub_path:
        from src.epub.writer import export_local

        prepared = prepare_source(
            title=book.title,
            author=book.author,
            volumes=book.volumes,
            source_url=book.source_url,
            intro_paragraphs=book.intro_paragraphs,
            intro_html=book.intro_html,
        )
        if "notion" in formats:
            from src.notion.upload import upload_source

            notion_path = upload_source(prepared, cover_bytes=book.cover_bytes)
        if epub_path:
            export_local(
                prepared,
                epub_path,
                cover=book.cover_bytes,
                cover_mime=book.cover_mime,
                prevent_overwrite=options.prevent_overwrite,
            )
    if txt_path and (not options.prevent_overwrite or not txt_path.exists()):
        write_txt(
            title=book.title,
            author=book.author,
            volumes=book.volumes,
            out_path=txt_path,
            intro_paragraphs=book.intro_paragraphs,
            intro_html=book.intro_html,
        )

    return IngestResult(epub_path, txt_path, notion_path)


def write_edition(edition: DownloadedEdition, options: OutputOptions) -> IngestResult:
    """Maintain a downloaded edition without interpreting it as a crawler draft."""
    from src.dataset.library import extract_epub_text
    from src.epub.archive import install_archive, validate_archive
    from src.epub.maintenance import normalize_new_epub
    from src.epub.reports import automatic_report_path, write_reports
    from src.runtime.files import digest

    formats = requested_formats(options)
    if "notion" in formats:
        raise ValueError(
            "Z-Library downloads editions; use --output-format epub or txt explicitly"
        )
    epub_path = (
        resolve_output_path(options.output, edition.title, edition.author)
        if "epub" in formats
        else None
    )
    txt_path = (
        resolve_requested_txt_output(
            options, title=edition.title, author=edition.author
        )
        if "txt" in formats
        else None
    )
    with tempfile.TemporaryDirectory(prefix="book-edition-") as temporary:
        candidate = Path(temporary) / "edition.epub"
        candidate.write_bytes(edition.epub_bytes)
        extraction_path = candidate
        if epub_path and (not options.prevent_overwrite or not epub_path.exists()):
            expected = digest(epub_path.read_bytes()) if epub_path.exists() else None
            report_path = automatic_report_path(epub_path)
            if report_path.exists():
                automatic_report_path(candidate).write_bytes(report_path.read_bytes())
            report = normalize_new_epub(candidate)
            validate_archive(candidate)
            install_archive(candidate, epub_path, expected)
            report.path = epub_path
            if report.metadata_enrichment:
                report.metadata_enrichment.path = epub_path
            write_reports(report_path, [report])
            extraction_path = epub_path
        elif epub_path:
            extraction_path = epub_path
        if txt_path and (not options.prevent_overwrite or not txt_path.exists()):
            _, _, text = extract_epub_text(extraction_path)
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            txt_path.write_text(
                f"{edition.title}\n作者：{edition.author}\n\n{text.strip()}\n",
                encoding="utf-8",
            )
    return IngestResult(epub_path, txt_path)


def ingest(
    target: str,
    *,
    parser: ParserSpec,
    crawl_options: CrawlOptions,
    output_options: OutputOptions,
) -> IngestResult:
    formats = requested_formats(output_options)
    if parser.downloads_edition and "notion" in formats:
        raise ValueError(
            "Z-Library downloads editions; use --output-format epub or txt explicitly"
        )
    collected = parser.crawl(target, crawl_options)
    if isinstance(collected, DownloadedEdition):
        return write_edition(collected, output_options)
    return write_outputs(collected, output_options)
