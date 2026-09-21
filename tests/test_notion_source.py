from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from lxml import etree as ET
from notion_books import NotionBooks, from_markdown, text_blocks, to_markdown

from src.content.blocks import content_signature
from src.content.formatting import prepare_blocks
from src.content.models import Chapter, Volume
from src.content.prepare import prepare_crawl
from src.content.xhtml import read_xhtml
from src.crawler.models import CrawledBook
from src.epub.archive import install_archive
from src.epub.writer import create_book
from src.notion.upload import upload_row
from src.runtime.files import digest
from src.workflows.ingest import OutputOptions, write_outputs


class SourceTests(unittest.TestCase):
    def test_source_whitespace_becomes_paragraph_layout_and_explicit_line_breaks(self):
        data = '<html xmlns="http://www.w3.org/1999/xhtml"><body><h2>标题</h2><p> A sentence.\u2028</p><p> </p></body></html>'.encode()
        blocks = prepare_blocks(
            data, "chapter.xhtml", {}, read_xhtml(data, title="标题")
        )
        self.assertEqual(text_blocks(blocks), ["A sentence.\n", ""])
        self.assertIn("<br>", to_markdown(blocks))
        self.assertEqual(
            content_signature(from_markdown(to_markdown(blocks))),
            content_signature(blocks),
        )

    def test_notion_autolinks_preserve_bare_urls_without_dropping_named_links(self):
        url = "https://example.org/book?id=1"
        self.assertEqual(from_markdown(f"[{url}]({url})"), from_markdown(url))
        with self.assertRaisesRegex(ValueError, "(?i)unsupported"):
            from_markdown(f"[different label]({url})")

    def test_install_checks_observed_bytes_and_backs_up_absolute_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            candidate, target = folder / "candidate.epub", folder / "target.epub"
            candidate.write_bytes(b"replacement")
            target.write_bytes(b"original")
            with patch("src.epub.archive.DEFAULT_BACKUP_DIR", folder / "backups"):
                with self.assertRaisesRegex(ValueError, "changed"):
                    install_archive(candidate, target, digest(b"outdated"))
                self.assertEqual(target.read_bytes(), b"original")
                backup = install_archive(candidate, target, digest(b"original"))
                self.assertEqual(backup.read_bytes(), b"original")
                self.assertEqual(target.read_bytes(), candidate.read_bytes())
                new_target = folder / "new" / "book.epub"
                self.assertIsNone(install_archive(candidate, new_target, None))
                self.assertEqual(new_target.read_bytes(), candidate.read_bytes())

    def test_crawl_prepares_explicit_layout_and_export_does_not_interpret_words(self):
        source = prepare_crawl(
            title="书",
            author="作者",
            source_url="https://example.org/book",
            volumes=[
                Volume("卷一", [Chapter("第1章 开始", ["普通段落。", "***", "全文完"])])
            ],
            intro_paragraphs=["简介内容。"],
        )
        source["identifier"] = "fixture-source"
        chapter = source["chapters"]["EPUB/chap_01_001.xhtml"]
        self.assertEqual(chapter["blocks"][1]["alignment"], "center")
        # The presentation layer must obey changed layout even for marker words.
        chapter["blocks"][-1].pop("alignment", None)
        chapter["blocks"].append(
            {
                "kind": "heading",
                "level": 3,
                "runs": [{"text": "任意内容", "styles": []}],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "book.epub"
            create_book(source, path)
            with zipfile.ZipFile(path) as archive:
                blocks = read_xhtml(
                    archive.read("EPUB/chap_01_001.xhtml"), title=chapter["title"]
                )
                self.assertEqual(
                    content_signature(blocks), content_signature(chapter["blocks"])
                )
                root = ET.fromstring(archive.read("EPUB/chap_01_001.xhtml"))
                self.assertIn("font-size: 1.1em", root.find(".//{*}h3").get("style"))

    def test_italic_whitespace_and_unicode_line_separator_preserve_prose(self):
        blocks = [
            {
                "kind": "paragraph",
                "runs": [
                    {"text": " A thought. ", "styles": ["italic"]},
                    {"text": "More\u2028words", "styles": []},
                ],
            }
        ]
        actual = from_markdown(to_markdown(blocks))
        self.assertEqual(content_signature(actual), content_signature(blocks))
        self.assertEqual(text_blocks(actual), [" A thought. More\u2028words"])

    def test_selected_notion_output_publishes_without_building_an_epub(self):
        with (
            patch(
                "src.notion.upload.upload_source",
                return_value=Path("source.json"),
            ) as publish_source,
            patch("src.content.prepare.enrich_source", return_value={}),
        ):
            result = write_outputs(
                CrawledBook(
                    title="书",
                    author="作者",
                    volumes=[Volume("", [Chapter("第1章", ["内容"])])],
                    source_url="https://example.org/book",
                ),
                OutputOptions(output_formats=("notion",)),
            )
        self.assertEqual(result.notion_state, Path("source.json"))
        self.assertNotIn("output", publish_source.call_args.kwargs)

    def test_right_alignment_uses_one_occupied_column_and_rejects_multiple(self):
        blocks = [
            {
                "kind": "paragraph",
                "alignment": "right",
                "runs": [{"text": "署名", "styles": []}],
            }
        ]
        markdown = to_markdown(blocks)
        self.assertEqual(from_markdown(markdown), blocks)
        with self.assertRaises(ValueError):
            from_markdown(markdown.replace("<empty-block/>", "额外正文", 1))


class BatchResumeTests(unittest.IsolatedAsyncioTestCase):
    async def test_prepared_crawl_readback_preserves_trailing_spaces(self):
        for html in ("<p>hello  </p>", "<p>hello <strong>world</strong> </p>"):
            for trim_readback in (False, True):
                with self.subTest(html=html, trim_readback=trim_readback):
                    book = prepare_crawl(
                        title="Fixture",
                        author="Author",
                        source_url="https://example.org/book",
                        volumes=[Volume("", [Chapter("Chapter", html_blocks=[html])])],
                    )
                    member, item = next(iter(book["chapters"].items()))
                    self.assertTrue(text_blocks(item["blocks"])[0].endswith(" "))
                    page_id = "11111111-1111-4111-8111-111111111111"
                    calls = []

                    class Tools:
                        async def call(self, name, arguments):
                            calls.append(name)
                            if name == "notion-create-pages":
                                self.page = arguments["pages"][0]
                                return {"pages": [{"id": page_id}]}
                            if name == "notion-fetch" and arguments["id"] == page_id:
                                body = self.page["content"]
                                if trim_readback:
                                    body = body.rstrip(" ")
                                return {
                                    "text": "<properties>\n"
                                    + json.dumps(self.page["properties"])
                                    + "\n</properties>\n<content>\n"
                                    + body
                                    + "\n</content>"
                                }
                            raise AssertionError((name, arguments))

                    with tempfile.TemporaryDirectory() as directory:
                        state = Path(directory) / "import.json"
                        upload = upload_row(
                            item,
                            book,
                            state,
                            data_source="22222222-2222-4222-8222-222222222222",
                            properties={"章节": "Chapter"},
                            title_property="章节",
                            tools=Tools(),
                        )
                        if trim_readback:
                            with self.assertRaisesRegex(ValueError, "readback differs"):
                                await upload
                        else:
                            await upload
                        saved = json.loads(state.read_text())["chapters"][member]
                        self.assertEqual(
                            saved.get("verified", False), not trim_readback
                        )
                        self.assertEqual(saved["page_id"], page_id)
                        self.assertNotIn("pending", saved)
                    self.assertEqual(calls, ["notion-create-pages", "notion-fetch"])

    async def test_title_only_source_page_is_readable(self):
        class Tools:
            async def call(self, *_):
                return {
                    "text": '<properties>\n{"title":"Section"}\n</properties>\n<blank-page>This page is blank and has no content.</blank-page>'
                }

        document = await NotionBooks(Tools()).document("page")

        props, body = document.properties, document.markdown
        self.assertEqual(props["title"], "Section")
        self.assertEqual(from_markdown(body), [])


if __name__ == "__main__":
    unittest.main()
