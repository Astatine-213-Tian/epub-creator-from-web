from __future__ import annotations

import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

from src.cli.ingest import main as ingest_main
from src.cli.main import main as crawl_main
from src.content.models import Chapter, Volume
from src.crawler.models import CrawledBook, CrawlOptions, DownloadedEdition
from src.crawler.registry import ParserSpec
from src.dataset.library import export_txt_dataset, load_manifest
from src.epub.reports import automatic_report_path
from src.metadata.catalog import MetadataEnrichmentReport
from src.workflows.ingest import (
    IngestResult,
    OutputOptions,
    ingest,
    requested_formats,
    write_outputs,
)


def book() -> CrawledBook:
    return CrawledBook(
        "书",
        "作者",
        [Volume("正文", [Chapter("第1章 开始", ["内容。", "第二段。"])])],
        "https://example.org/book",
    )


class IngestWorkflowTests(unittest.TestCase):
    def test_collects_once_and_passes_only_crawl_controls_to_provider(self):
        crawl = Mock(return_value=book())
        parser = ParserSpec("fixture", ("example.org",), "fixture", crawl)
        options = CrawlOptions(delay=0.1, concurrency=2, headless=True)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("src.content.prepare.enrich_source", return_value={}):
                result = ingest(
                    "https://example.org/book",
                    parser=parser,
                    crawl_options=options,
                    output_options=OutputOptions(
                        output=root / "book.epub",
                        txt_output=root / "book.txt",
                        output_formats=("both",),
                    ),
                )
            crawl.assert_called_once_with("https://example.org/book", options)
            self.assertEqual(result.epub_path, root / "book.epub")
            self.assertIn("第二段。", (root / "book.txt").read_text())
            with zipfile.ZipFile(result.epub_path) as archive:
                self.assertIn(
                    "第二段。", archive.read("EPUB/chap_01_001.xhtml").decode()
                )

    def test_invalid_output_and_unsupported_draft_stop_before_collection(self):
        crawl = Mock()
        for formats, is_edition in [
            ((), False),
            (("invalid",), False),
            (("notion",), True),
        ]:
            with self.subTest(formats=formats), self.assertRaises(ValueError):
                ingest(
                    "https://example.org/book",
                    parser=ParserSpec("fixture", (), "fixture", crawl, is_edition),
                    crawl_options=CrawlOptions(),
                    output_options=OutputOptions(output_formats=formats),
                )
        crawl.assert_not_called()

    def test_existing_output_is_not_overwritten_or_uploaded(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "existing.epub"
            target.write_bytes(b"original")
            with (
                patch("src.notion.upload.upload_source") as upload,
                self.assertRaises(ValueError),
            ):
                write_outputs(
                    book(),
                    OutputOptions(output=target, output_formats=("epub", "notion")),
                )
            upload.assert_not_called()
            self.assertEqual(target.read_bytes(), b"original")

    def test_failed_edition_repair_preserves_existing_file(self):
        edition = DownloadedEdition(
            "书", "作者", "https://example.org/book", b"download"
        )
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "existing.epub"
            target.write_bytes(b"original")
            with (
                patch(
                    "src.epub.maintenance.normalize_new_epub",
                    side_effect=ValueError("bad edition"),
                ),
                self.assertRaisesRegex(ValueError, "bad edition"),
            ):
                ingest(
                    edition.source_url,
                    parser=ParserSpec(
                        "edition", (), "fixture", Mock(return_value=edition), True
                    ),
                    crawl_options=CrawlOptions(),
                    output_options=OutputOptions(
                        output=target, output_formats=("epub",), prevent_overwrite=False
                    ),
                )
            self.assertEqual(target.read_bytes(), b"original")

    def test_downloaded_edition_keeps_report_at_final_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.epub"
            target = root / "result.epub"
            with patch("src.content.prepare.enrich_source", return_value={}):
                write_outputs(
                    book(), OutputOptions(output=source, output_formats=("epub",))
                )
            edition = DownloadedEdition(
                "书", "作者", "https://example.org/book", source.read_bytes()
            )
            metadata = MetadataEnrichmentReport(
                path=source, applied=False, status="fixture"
            )
            with (
                patch.dict(os.environ, {"BOOKLIB_CODEX_REVIEW": "0"}),
                patch(
                    "src.epub.maintenance.enrich_epub_metadata", return_value=metadata
                ),
            ):
                result = ingest(
                    edition.source_url,
                    parser=ParserSpec(
                        "edition", (), "fixture", Mock(return_value=edition), True
                    ),
                    crawl_options=CrawlOptions(),
                    output_options=OutputOptions(
                        output=target, output_formats=("epub",)
                    ),
                )
            self.assertEqual(result.epub_path, target)
            report = json.loads(automatic_report_path(target).read_text())["reports"][0]
            self.assertEqual(report["path"], str(target))
            self.assertEqual(report["metadata_enrichment"]["path"], str(target))
            with zipfile.ZipFile(target) as archive:
                self.assertIsNone(archive.testzip())

    def test_txt_only_does_not_prepare_epub_or_contact_notion(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "book.txt"
            with (
                patch(
                    "src.workflows.ingest.prepare_source",
                    side_effect=AssertionError("EPUB preparation called"),
                ),
                patch(
                    "src.notion.upload.upload_source",
                    side_effect=AssertionError("Notion called"),
                ),
            ):
                self.assertEqual(
                    write_outputs(
                        book(),
                        OutputOptions(txt_output=target, output_formats=("txt",)),
                    ).txt_path,
                    target,
                )
            self.assertIn("内容。", target.read_text())

    def test_explicit_txt_path_takes_priority_over_dataset_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "chosen" / "book.txt"
            result = write_outputs(
                book(),
                OutputOptions(
                    txt_output=target,
                    dataset_root=root / "dataset",
                    output_formats=("txt",),
                ),
            )
            self.assertEqual(result.txt_path, target)
            self.assertIn("内容。", target.read_text())
            self.assertFalse((root / "dataset").exists())

    def test_cli_requires_choice_and_paths_can_express_it(self):
        for entry, module in [(crawl_main, "main"), (ingest_main, "ingest")]:
            with self.subTest(command=module):
                with (
                    patch(f"src.cli.{module}.ingest") as run,
                    patch("sys.stderr"),
                    self.assertRaises(SystemExit) as error,
                ):
                    entry(["2574", "--parser", "mgsf"])
                self.assertEqual(error.exception.code, 2)
                run.assert_not_called()
        for args, expected in [
            (["-o", "book.epub"], ("epub",)),
            (["--txt-output", "book.txt"], ("txt",)),
            (["-o", "book.epub", "--txt-output", "book.txt"], ("epub", "txt")),
        ]:
            with (
                self.subTest(args=args),
                patch("src.cli.main.ingest", return_value=IngestResult()) as run,
            ):
                self.assertEqual(crawl_main(["2574", "--parser", "mgsf", *args]), 0)
                self.assertEqual(
                    requested_formats(run.call_args.kwargs["output_options"]), expected
                )

    def test_conflicting_output_path_stops_before_collection(self):
        for entry, module, flag in [
            (crawl_main, "main", "--output-format"),
            (ingest_main, "ingest", "--mode"),
        ]:
            with (
                self.subTest(command=module),
                patch(f"src.cli.{module}.ingest") as run,
                patch("sys.stderr"),
                self.assertRaises(SystemExit),
            ):
                entry(["2574", "--parser", "mgsf", flag, "notion", "-o", "book.epub"])
            run.assert_not_called()

    def test_all_three_outputs_use_one_crawl_and_track_actual_txt_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            crawl = Mock(return_value=book())
            provider = ParserSpec("fixture", (), "fixture", crawl)

            def enriched(source):
                source["metadata"]["title"] = "官方书名"
                return {"status": "matched"}

            with (
                patch("src.cli.ingest.find_parser", return_value=provider),
                patch("src.content.prepare.enrich_source", side_effect=enriched),
                patch(
                    "src.notion.upload.upload_source", return_value=root / "notion.json"
                ) as upload,
            ):
                code = ingest_main(
                    [
                        "https://example.org/book",
                        "--mode",
                        "epub",
                        "--mode",
                        "notion",
                        "--mode",
                        "txt",
                        "--mode",
                        "notion",
                        "-o",
                        str(root / "book.epub"),
                        "--dataset-root",
                        str(root / "dataset"),
                        "--no-fetch-jjwxc",
                        "--no-codex-classify",
                    ]
                )
            self.assertEqual(code, 0)
            crawl.assert_called_once()
            upload.assert_called_once()
            self.assertEqual(upload.call_args.args[0]["metadata"]["title"], "官方书名")
            rows = load_manifest(root / "dataset")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["title"], "官方书名")
            self.assertEqual(rows[0]["txt_path"], "raw/作者/书.txt")
            self.assertIn(
                "第二段。", (root / "dataset" / rows[0]["txt_path"]).read_text()
            )
            with zipfile.ZipFile(root / "book.epub") as archive:
                self.assertIn("官方书名", archive.read("EPUB/content.opf").decode())

    def test_every_output_combination_is_supported(self):
        from itertools import combinations

        for size in (1, 2, 3):
            for formats in combinations(("epub", "notion", "txt"), size):
                with (
                    self.subTest(formats=formats),
                    tempfile.TemporaryDirectory() as temporary,
                ):
                    root = Path(temporary)
                    with (
                        patch("src.content.prepare.enrich_source", return_value={}),
                        patch(
                            "src.notion.upload.upload_source",
                            return_value=root / "notion.json",
                        ) as upload,
                    ):
                        result = write_outputs(
                            book(),
                            OutputOptions(
                                output_formats=formats,
                                output=root / "book.epub"
                                if "epub" in formats
                                else None,
                                txt_output=root / "book.txt"
                                if "txt" in formats
                                else None,
                            ),
                        )
                    self.assertEqual(
                        bool(result.epub_path and result.epub_path.exists()),
                        "epub" in formats,
                    )
                    self.assertEqual(
                        bool(result.txt_path and result.txt_path.exists()),
                        "txt" in formats,
                    )
                    self.assertEqual(bool(result.notion_state), "notion" in formats)
                    self.assertEqual(upload.call_count, int("notion" in formats))

    def test_ingest_both_updates_only_targeted_dataset_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            provider = ParserSpec("fixture", (), "fixture", Mock(return_value=book()))
            with (
                patch("src.cli.ingest.find_parser", return_value=provider),
                patch("src.content.prepare.enrich_source", return_value={}),
            ):
                code = ingest_main(
                    [
                        "https://example.org/book",
                        "--mode",
                        "both",
                        "-o",
                        str(root / "book.epub"),
                        "--dataset-root",
                        str(root / "dataset"),
                        "--no-fetch-jjwxc",
                        "--no-codex-classify",
                    ]
                )
            self.assertEqual(code, 0)
            rows = load_manifest(root / "dataset")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["title"], "书")
            self.assertTrue((root / "dataset" / rows[0]["txt_path"]).is_file())

    def test_explicit_dataset_export_records_relative_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("src.content.prepare.enrich_source", return_value={}):
                write_outputs(
                    book(),
                    OutputOptions(
                        output=root / "books/book.epub", output_formats=("epub",)
                    ),
                )
            rows = export_txt_dataset(
                books_root=root / "books",
                output_root=root / "dataset",
                jjwxc_manifest=None,
                jjwxc_top50=None,
                overwrite=False,
                fetch_jjwxc=False,
                codex_classify=False,
                exclude_titles=set(),
                exclude_metadata_keys=set(),
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].status, "written")
            self.assertEqual(rows[0].source_epub, "../books/book.epub")
            self.assertEqual(
                load_manifest(root / "dataset")[0]["txt_path"], rows[0].txt_path
            )
