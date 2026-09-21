from __future__ import annotations

import copy
import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from lxml import etree as ET
from notion_books import NotionError, Page, work_properties
from PIL import Image

from src.content.models import Chapter, Volume
from src.content.prepare import prepare_crawl
from src.crawler.models import CrawledBook
from src.epub.writer import export_local
from src.notion.cms import STORAGE, chapter_entries, ensure_work, ensure_views
from src.notion.cover import upload_cover, validate_cover
from src.notion.upload import upload_draft, upload_row
from src.runtime.files import digest
from src.workflows.ingest import OutputOptions, write_outputs

WORK = "11111111-1111-1111-1111-111111111111"
DS = "22222222-2222-2222-2222-222222222222"


def source():
    book = prepare_crawl(
        title="书",
        author="作者",
        source_url="https://example.org/book",
        intro_paragraphs=["简介内容"],
        volumes=[
            Volume("卷一", [Chapter("第1章", ["正文"]), Chapter("第2章", ["后续"])]),
            Volume("番外", [Chapter("番外篇", ["独立番外"])]),
        ],
    )
    book["identifier"] = "local-book"
    return book


def png():
    data = io.BytesIO()
    Image.new("RGB", (2, 3), "blue").save(data, format="PNG")
    return data.getvalue()


class DestinationTests(unittest.TestCase):
    def test_local_path_never_authenticates_and_retains_thumbnail_only(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "local.epub"
            with (
                patch("src.content.prepare.enrich_source", return_value={}),
                patch(
                    "src.notion.upload.upload_source",
                    side_effect=AssertionError("Notion called"),
                ),
            ):
                write_outputs(
                    CrawledBook(
                        title="书",
                        author="作者",
                        source_url="https://example.org/book",
                        volumes=[
                            Volume("卷一", [Chapter("第1章", ["正文"])]),
                            Volume("番外", [Chapter("番外", ["内容"])]),
                        ],
                        cover_bytes=png(),
                        cover_mime="image/png",
                    ),
                    OutputOptions(output_formats=("epub",), output=out),
                )
            with zipfile.ZipFile(out) as z:
                self.assertEqual(z.read("EPUB/cover.png"), png())
                self.assertFalse(
                    any(
                        Path(n).name in ("cover.xhtml", "cover.html")
                        for n in z.namelist()
                    )
                )
                root = ET.fromstring(z.read("EPUB/content.opf"))
                images = root.xpath(
                    '//*[local-name()="item" and @properties="cover-image"]'
                )
                self.assertEqual(len(images), 1)
                self.assertEqual(images[0].get("href"), "cover.png")
                self.assertEqual(root.find("{*}spine")[0].get("idref"), "nav")
                self.assertIn("番外", z.read("EPUB/nav.xhtml").decode())
                self.assertIn("内容", z.read("EPUB/extra_001.xhtml").decode())

    def test_bilingual_builder_embeds_only_cover_image(self):
        from src.translation.output import build_bilingual_epub

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cover.png").write_bytes(png())
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "title": "Fixture",
                        "author": "Author",
                        "cover": {"path": "cover.png", "mime": "image/png"},
                        "chapters": [{"id": "001", "path": "chapter.json"}],
                    }
                )
            )
            (root / "chapter.json").write_text(
                json.dumps(
                    {
                        "title": "Chapter 001",
                        "paragraphs": [{"index": 0, "english": "Text."}],
                    }
                )
            )
            with patch("src.translation.output.normalize_new_epub"):
                build_bilingual_epub(
                    snapshot_dir=root,
                    translations_dir=root / "translations",
                    output=root / "bilingual.epub",
                )
            for name in ("bilingual.epub",):
                with self.subTest(builder=name), zipfile.ZipFile(root / name) as z:
                    self.assertEqual(z.read("EPUB/cover.png"), png())
                    self.assertFalse(
                        any(
                            Path(n).name in ("cover.xhtml", "cover.html")
                            for n in z.namelist()
                        )
                    )
                    opf = ET.fromstring(z.read("EPUB/content.opf"))
                    self.assertEqual(
                        len(
                            opf.xpath(
                                '//*[local-name()="item" and @properties="cover-image"]'
                            )
                        ),
                        1,
                    )
                    self.assertFalse(
                        opf.xpath('//*[local-name()="itemref" and @idref="cover"]')
                    )

    def test_cms_metadata_has_new_field_types_and_no_service_fields(self):
        self.assertEqual(source()["metadata"]["description"], "简介内容")
        metadata = source()["metadata"] | {
            "date": "2020-02-01",
            "series": "系列",
            "series_position": "1.5",
            "subjects": ["爱情", "耽美"],
            "description": "摘要",
        }
        props = work_properties(metadata, [WORK, DS])
        self.assertEqual(props["作者"], [WORK, DS])
        self.assertEqual(props["系列序号"], 1.5)
        self.assertEqual(props["书籍分类"], ["爱情", "耽美"])
        self.assertEqual(props["date:出版日期:start"], "2020-02-01")
        self.assertEqual(props["来源"], "https://example.org/book")
        self.assertNotIn("发表日期", props)
        self.assertFalse(any("发布" in key or "Bookshelf" in key for key in props))

    def test_cms_rejects_deeper_toc_without_flattening_local_structure(self):
        book = source()
        book["sections"] = [
            {"title": "部", "children": copy.deepcopy(book["sections"])}
        ]
        with self.assertRaisesRegex(ValueError, "one parent"):
            chapter_entries(book)
        with tempfile.TemporaryDirectory() as directory:
            export_local(book, Path(directory) / "book.epub")

    def test_cms_outline_rejects_missing_duplicate_and_shared_chapters(self):
        for kind in ("missing", "duplicate", "shared"):
            with self.subTest(kind=kind):
                book = source()
                member = next(iter(book["chapters"]))
                if kind == "missing":
                    book["chapters"]["unlisted"] = copy.deepcopy(
                        book["chapters"][member]
                    )
                elif kind == "duplicate":
                    book["sections"].append({"member": member})
                else:
                    book["chapters"][member]["shared"] = True
                with self.assertRaises(ValueError):
                    chapter_entries(book)


class UploadTests(unittest.IsolatedAsyncioTestCase):
    async def test_importer_applies_language_default_without_changing_checkpoint_source(
        self,
    ):
        for language in (None, "", "en"):
            with self.subTest(language=language):
                book = source()
                book["metadata"].pop("language", None)
                if language is not None:
                    book["metadata"]["language"] = language
                original = copy.deepcopy(book["metadata"])
                config = {
                    "databases": {
                        "works": {"data_source_id": DS, "view_id": "works"},
                        "authors": {"data_source_id": DS, "view_id": "authors"},
                    }
                }
                tools = AsyncMock()
                tools.call.return_value = {"pages": [{"id": WORK}]}
                with (
                    tempfile.TemporaryDirectory() as directory,
                    patch(
                        "notion_books.NotionBooks.catalog",
                        new=AsyncMock(
                            return_value={"works": {"default_template": WORK}}
                        ),
                    ),
                    patch(
                        "notion_books.NotionBooks.rows",
                        new=AsyncMock(
                            side_effect=[
                                [{"id": WORK, "作者": original["creator"]}],
                                [],
                            ]
                        ),
                    ),
                    patch("notion_books.NotionBooks.ensure_options", new=AsyncMock()),
                ):
                    await ensure_work(
                        book, Path(directory) / "import.json", config, tools=tools
                    )
                properties = tools.call.call_args.args[1]["pages"][0]["properties"]
                self.assertEqual(properties["语言"], language or "zh-CN")
                self.assertEqual(book["metadata"], original)

    async def test_template_discovery_selects_manual_views_over_display_views(self):
        chapter_database = "33333333-3333-4333-8333-333333333333"
        manual = "44444444-4444-4444-8444-444444444444"
        sorted_view = "55555555-5555-4555-8555-555555555555"
        extras_database = "66666666-6666-4666-8666-666666666666"
        extras_source = "77777777-7777-4777-8777-777777777777"
        extras_view = "88888888-8888-4888-8888-888888888888"
        book = {"work_id": WORK}
        config = {
            "databases": {
                "works": {"data_source_id": WORK},
                "extras": {"data_source_id": extras_source},
            }
        }

        def view(data_source, **options):
            return {
                "text": "<view>\n"
                + json.dumps(
                    {"dataSourceUrl": "collection://" + data_source, **options}
                )
                + "\n</view>"
            }

        documents = {
            WORK: {
                "text": f'<parent-data-source url="collection://{WORK}"/>\n<properties>\n{{}}\n</properties>\n<content>\n<database url="{chapter_database}"/>\n<database url="{extras_database}"/>\n</content>'
            },
            chapter_database: {
                "text": f'<parent-page url="{WORK}"/> view://{sorted_view} view://{manual}'
            },
            extras_database: {"text": "view://" + extras_view},
            "collection://" + DS: {
                "url": chapter_database,
                "text": '<data-source-state>\n{"name":"正文","schema":{"章节":{"type":"title"},"所属标题":{"type":"select"}}}\n</data-source-state>',
            },
            "view://" + manual: view(DS),
            "view://" + sorted_view: view(DS, sorts=[{"property": "章节"}]),
            "view://" + extras_view: view(
                extras_source,
                advancedFilter={
                    "type": "property",
                    "property": "涉及作品",
                    "propertyType": "relation",
                    "operator": "relation_contains",
                    "value": {"type": "exact", "value": WORK},
                },
            ),
        }

        class Tools:
            async def call(self, name, arguments):
                if name != "notion-fetch":
                    raise AssertionError("Unexpected write during discovery")
                return documents[arguments["id"]]

        with tempfile.TemporaryDirectory() as directory:
            await ensure_views(
                book, Path(directory) / "import.json", config, tools=Tools()
            )
        self.assertEqual(book["chapters_view_id"], manual)
        self.assertEqual(book["view_id"], extras_view)

    async def test_draft_creation_matches_cms_order_and_resumes_without_overwrite(self):
        book = source()
        book.update(
            storage=STORAGE,
            catalog_id=DS,
            work_id=WORK,
            chapters_data_source_id="chapters",
            chapters_view_id="main",
            view_id="extras",
        )
        config = {
            "databases": {
                "works": {"data_source_id": DS},
                "extras": {"data_source_id": DS, "view_id": "all-extras"},
            }
        }
        rows = {"main": [], "extras": [], "all-extras": []}
        created = {}
        calls = []

        class Tools:
            async def call(self, name, args):
                calls.append((name, args))
                if name != "notion-create-pages":
                    raise AssertionError("Unexpected write: " + name)
                page = args["pages"][0]
                id = f"00000000-0000-0000-0000-{len(created) + 1:012d}"
                created[id] = page
                view = (
                    "main"
                    if args["parent"]["data_source_id"] == "chapters"
                    else "extras"
                )
                rows[view].insert(0, {"id": id})
                return {"pages": [{"id": id}]}

        async def document(_reader, id):
            return Page(
                page_id=id,
                data_source_id=DS,
                title="",
                markdown=created[id]["content"],
                revision="",
                properties=created[id]["properties"],
                blocks=None,
                cover=None,
                cover_known=False,
            )

        async def read_rows(_reader, view):
            return rows[view]

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("src.notion.upload.ensure_work", new=AsyncMock()),
            patch("src.notion.upload.ensure_views", new=AsyncMock()),
            patch("notion_books.NotionBooks.ensure_options", new=AsyncMock()),
            patch("notion_books.NotionBooks.document", document),
            patch("notion_books.NotionBooks.rows", read_rows),
            patch(
                "notion_books.NotionBooks.inventory",
                new=AsyncMock(return_value=[]),
            ),
        ):
            state = Path(directory) / "import.json"
            await upload_draft(book, state, config, tools=Tools())
            self.assertTrue(book["uploaded"])
            self.assertEqual(
                [created[r["id"]]["properties"]["章节"] for r in rows["main"]],
                ["简介", "第1章", "第2章"],
            )
            self.assertEqual(
                created[rows["main"][1]["id"]]["properties"]["所属标题"], "卷一"
            )
            count = len(calls)
            created[rows["main"][1]["id"]]["content"] = "User edit after upload"
            await upload_draft(book, state, config, tools=Tools())
            self.assertEqual(len(calls), count)

    async def test_lost_create_response_stops_retry_without_duplicate(self):
        book = source()
        item = next(iter(book["chapters"].values()))
        tools = AsyncMock()
        tools.call.side_effect = TimeoutError("response lost")
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "import.json"
            kwargs = {
                "data_source": DS,
                "properties": {"章节": item["title"]},
                "title_property": "章节",
                "tools": tools,
            }
            with self.assertRaises(NotionError) as caught:
                await upload_row(item, book, state, **kwargs)
            self.assertTrue(caught.exception.uncertain)
            self.assertIsInstance(caught.exception.__cause__, TimeoutError)
            self.assertTrue(item["pending"])
            with self.assertRaisesRegex(ValueError, "response was lost"):
                await upload_row(item, book, state, **kwargs)
            self.assertEqual(tools.call.await_count, 1)

    async def test_old_catalog_state_cannot_be_written(self):
        with self.assertRaisesRegex(ValueError, "old library"):
            await upload_draft(
                source(),
                Path("unused.json"),
                {"databases": {"works": {"data_source_id": DS}}},
                tools=None,
            )

    async def test_native_cover_uses_only_api_for_file_and_mcp_for_readback(self):
        cover = png()
        requests = []

        async def handler(request):
            requests.append(request)
            if request.url.host == "files.example.org":
                self.assertNotIn("authorization", request.headers)
                return httpx.Response(200, content=cover)
            self.assertEqual(request.headers["authorization"], "Bearer test-token")
            if request.url.path.endswith("/send"):
                self.assertIn(b'filename="cover.png"', request.content)
                return httpx.Response(200, json={"status": "uploaded"})
            if request.method == "PATCH":
                self.assertEqual(
                    json.loads(request.content),
                    {"cover": {"type": "file_upload", "file_upload": {"id": DS}}},
                )
            return httpx.Response(200, json={"id": DS})

        client = httpx.AsyncClient

        def factory(**kwargs):
            return client(transport=httpx.MockTransport(handler), **kwargs)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"NOTION_API_TOKEN": "test-token"}),
            patch("src.notion.cover.httpx.AsyncClient", factory),
        ):
            asset = Path(directory) / "cover.png"
            asset.write_bytes(cover)
            book = {
                "work_id": WORK,
                "cover_asset": str(asset),
                "cover_sha256": digest(cover),
            }
            tools = AsyncMock()
            tools.call.side_effect = [
                {
                    "cover": None,
                    "text": "<properties>\n{}\n</properties>\n<blank-page>",
                },
                {
                    "text": "<properties>\n{}\n</properties>\n<blank-page>",
                    "cover": {
                        "type": "file",
                        "file": {"url": "https://files.example.org/cover"},
                    },
                },
            ]
            await upload_cover(book, Path(directory) / "import.json", tools=tools)
            self.assertTrue(book["cover_uploaded"])
            self.assertNotIn("cover_pending", book)
            self.assertEqual(
                [r.method for r in requests], ["GET", "POST", "POST", "PATCH", "GET"]
            )
            self.assertTrue(
                all(
                    call.args[0] == "notion-fetch" for call in tools.call.call_args_list
                )
            )

    def test_invalid_cover_is_rejected(self):
        self.assertEqual(validate_cover(png()), "png")
        with self.assertRaises(ValueError):
            validate_cover(b"not an image")
