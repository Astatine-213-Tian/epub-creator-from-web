from __future__ import annotations

import contextlib
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from notion_books import NotionBooks, Page, from_markdown

from src.notion.cms import STORAGE
from src.notion.duplicates import (
    TextProfile,
    preflight_extras,
    resolve_extra,
    similarity,
)
from src.notion.upload import upload_draft, upload_source

WORK = "11111111-1111-1111-1111-111111111111"
OTHER = "22222222-2222-2222-2222-222222222222"
CATALOG = "33333333-3333-3333-3333-333333333333"
EXTRAS = "44444444-4444-4444-4444-444444444444"
PAGE = "55555555-5555-5555-5555-555555555555"
SECOND = "66666666-6666-6666-6666-666666666666"
TEXT = (
    "月亮升起来的时候，他们刚刚走到河边。桥上的灯还亮着，远处有人在唱歌。"
    "他把手里的纸袋递过去，笑着说，这是给你的礼物。对方拆开包装，发现里面是一本旧书。\n"
    "书页上留下了许多年前的笔迹，那时候他们尚未见过面，也不知道以后会在这里相遇。"
    "风吹过树梢，树叶落进河中，顺着水流漂向远方。\n"
    "第二天清晨，街上的店铺陆续开门。卖早餐的老人认出了他们，招呼两个人过来坐。"
    "热气从碗里升起，他想起昨天没有说完的话，于是轻轻喊了对方一声。"
    "那人回过头来，眼里映着晨光。\n"
    "回家以后，他们把旧书放在窗前，一起做了午饭。窗外的花已经开了，空气里有雨后的清香。"
    "直到夜里再一次下起雨，门廊下的灯始终亮着。这个假期还有很长的时间。"
)


def profile(text):
    return TextProfile.from_blocks(from_markdown(text))


def different_versions(*, proofread=False):
    """Mostly unrelated editions, sharing two passages in a different order."""
    first, second = TEXT.splitlines()[:2]
    old_only = "".join(chr(n) for n in range(0x5000, 0x5400))
    new_only = "".join(chr(n) for n in range(0x7000, 0x7400))
    corrected = first.replace("纸袋", "布袋") if proofread else first
    left = "\n".join((old_only, first, old_only, second, old_only))
    right = "\n".join((new_only, second, new_only, corrected, new_only))
    return left, right


def source():
    return {
        "storage": STORAGE,
        "catalog_id": CATALOG,
        "metadata": {"title": "新书"},
        "work_id": WORK,
        "chapters_data_source_id": "main-db",
        "chapters_view_id": "main",
        "view_id": "extras",
        "sections": [{"member": "chapter"}],
        "chapters": {"chapter": {"title": "第一章", "blocks": from_markdown("正文")}},
        "extras": [{"title": "原来的番外标题", "blocks": from_markdown(TEXT)}],
    }


CONFIG = {
    "databases": {
        "works": {"data_source_id": CATALOG},
        "extras": {"data_source_id": EXTRAS, "view_id": "all-extras"},
    }
}


class Library:
    def __init__(self):
        self.pages = {
            PAGE: {
                "properties": {"番外": "用户重命名的标题", "涉及作品": [OTHER]},
                "content": TEXT.replace("纸袋", "布袋"),
            }
        }
        self.main = []
        self.shared = [PAGE]
        self.calls = []
        self.reads = []
        self.filter = None

    async def ensure_options(self, data_source, values):
        pass

    async def create_page(self, data_source, properties, **kwargs):
        return await NotionBooks(self).create_page(data_source, properties, **kwargs)

    async def write_properties(self, id, properties):
        return await NotionBooks(self).write_properties(id, properties)

    async def inventory(self, database):
        return await NotionBooks(self).inventory(database)

    async def rows(self, view):
        ids = self.main if view == "main" else self.shared
        if view == "extras":
            ids = [
                id
                for id in ids
                if WORK in self.pages[id]["properties"].get("涉及作品", [])
            ]
        return [{"id": id} for id in ids]

    async def document(self, id):
        self.reads.append(id)
        page = self.pages[id]
        return Page(
            page_id=id,
            data_source_id=EXTRAS,
            title="",
            markdown=page["content"],
            revision="",
            properties=copy.deepcopy(page["properties"]),
            blocks=None,
            cover=None,
            cover_known=False,
        )

    async def call(self, name, args):
        if name == "notion-fetch":
            view = {"dataSourceUrl": f"collection://{EXTRAS}", "filter": self.filter}
            return {"text": "<view>\n" + json.dumps(view) + "\n</view>"}
        if name == "notion-query-data-sources":
            return {"results": [{"url": id} for id in self.shared], "has_more": False}
        self.calls.append((name, args))
        if name == "notion-update-page":
            self.pages[args["page_id"]]["properties"].update(args["properties"])
            return {}
        if name == "notion-create-pages":
            id = f"00000000-0000-0000-0000-{len(self.pages):012d}"
            self.pages[id] = copy.deepcopy(args["pages"][0])
            ids = (
                self.main
                if args["parent"]["data_source_id"] == "main-db"
                else self.shared
            )
            ids.insert(0, id)
            return {"pages": [{"id": id}]}
        raise AssertionError(f"Unexpected mutation: {name}")


class SimilarityTests(unittest.TestCase):
    def test_proofreading_and_format_changes_match(self):
        score = similarity(
            profile(TEXT), profile("### " + TEXT.replace("纸袋", "布袋"))
        )
        self.assertGreater(score["similarity"], 0.95)
        self.assertLess(score["similarity"], 1)
        self.assertEqual(
            similarity(profile("**我們回家**！"), profile("我们 回家。"))["similarity"],
            1,
        )

    def test_containment_detects_part_of_a_merged_extra(self):
        longer = TEXT + "\n" + "".join(chr(n) for n in range(0x5000, 0x5400))
        score = similarity(profile(TEXT), profile(longer))
        self.assertLess(score["similarity"], 0.85)
        self.assertEqual(score["containment"], 1)

    def test_two_shared_passages_warn_even_with_low_total_overlap(self):
        for proofread in (False, True):
            with self.subTest(proofread=proofread):
                left, right = map(profile, different_versions(proofread=proofread))
                score = similarity(left, right)
                self.assertLess(score["similarity"], 0.10)
                self.assertLess(score["containment"], 0.10)
                self.assertEqual(score["reasons"], ["局部文字重合"])
                self.assertGreaterEqual(score["overlap_chars"], 60)
                for snippet in score["overlap_snippets"]:
                    self.assertIn(snippet, left.text)
                    self.assertIn(snippet, right.text)

    def test_split_joined_paragraph_boundaries_do_not_hide_overlap(self):
        left, right = different_versions()
        joined = profile(right.replace("\n", ""))
        paragraphs = from_markdown(left)
        for block in paragraphs:
            for run in block["runs"]:
                run["text"] = "\n".join(
                    run["text"][i : i + 13] for i in range(0, len(run["text"]), 13)
                )
        score = similarity(TextProfile.from_blocks(paragraphs), joined)
        self.assertIn("局部文字重合", score["reasons"])

    def test_two_thirty_character_passages_reach_local_threshold(self):
        text = profile(TEXT).text
        first, second = text[:30], text[140:170]
        for size in (29, 30):
            with self.subTest(second_passage_size=size):
                left = "甲" * 500 + first + "乙" * 500 + second[:size] + "丙" * 500
                right = "丁" * 500 + second[:size] + "戊" * 500 + first + "己" * 500
                score = similarity(profile(left), profile(right))
                if size == 29:
                    self.assertIsNone(score)
                else:
                    self.assertEqual(score["overlap_chars"], 60)
                    self.assertEqual(score["reasons"], ["局部文字重合"])

    def test_full_collection_warns_against_both_separate_halves(self):
        paragraphs = TEXT.splitlines()
        full = profile(TEXT)
        for half in ("\n".join(paragraphs[:2]), "\n".join(paragraphs[2:])):
            with self.subTest(half_length=len(half)):
                self.assertIsNotNone(similarity(full, profile(half)))
                self.assertIsNotNone(similarity(profile(half), full))

    def test_missing_or_added_paragraphs_still_warn(self):
        paragraphs = TEXT.splitlines()
        incomplete = profile("\n".join(paragraphs[::2]))
        expanded = profile(
            "\n".join(p + "\n新增了一段完全不同的情节。" for p in paragraphs)
        )
        self.assertIsNotNone(similarity(incomplete, expanded))
        self.assertIsNotNone(similarity(expanded, incomplete))

    def test_scattered_corrections_can_match_below_old_threshold(self):
        text = "".join(chr(n) for n in range(0x5000, 0x5200))
        edited = "".join("字" if i % 17 == 0 else ch for i, ch in enumerate(text))
        score = similarity(profile(text), profile(edited))
        self.assertGreaterEqual(score["similarity"], 0.70)
        self.assertLess(score["similarity"], 0.85)
        self.assertIn("整体相似", score["reasons"])
        self.assertEqual(score["overlap_chars"], 0)
        padded = edited + "".join(chr(n) for n in range(0x7000, 0x7400))
        score = similarity(profile(text), profile(padded))
        self.assertLess(score["similarity"], 0.70)
        self.assertIn("部分内容包含", score["reasons"])

    def test_short_overlap_is_not_multiplied_by_windows_or_repeated_copies(self):
        common = profile(TEXT).text[:40]
        left_tail = "".join(chr(n) for n in range(0x5000, 0x5200))
        right_tail = "".join(chr(n) for n in range(0x7000, 0x7200))
        for count in (1, 10):
            with self.subTest(count=count):
                self.assertIsNone(
                    similarity(
                        profile(common * count + left_tail),
                        profile(common + right_tail),
                    )
                )

    def test_common_short_phrases_alone_do_not_warn(self):
        phrases = ("他笑了", "你好啊", "故事完结", "月亮升起来了")
        left = "\n".join(p + "旧稿独有的叙述经过" * 30 for p in phrases)
        right = "\n".join(p + "另一篇新的角色不同情节" * 30 for p in phrases)
        self.assertIsNone(similarity(profile(left), profile(right)))

    def test_unrelated_short_and_empty_text_do_not_match(self):
        self.assertIsNone(
            similarity(
                profile(TEXT), profile("".join(chr(n) for n in range(0x5000, 0x5200)))
            )
        )
        self.assertIsNone(
            similarity(profile("第一天，下雨了。"), profile("第二天，下雨了。"))
        )
        self.assertIsNone(similarity(profile("---"), profile("---")))

    def test_ingestion_keeps_warning_visible_through_mcp_exception_groups(self):
        def fail(coroutine):
            coroutine.close()
            raise ExceptionGroup(
                "SDK task group", [ValueError("WARNING: review extra-review.md")]
            )

        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
            config = Path("config.json")
            config.write_text(json.dumps(CONFIG))
            book = source() | {"identifier": "warning-test"}
            with patch("src.notion.upload.asyncio.run", side_effect=fail):
                with self.assertRaisesRegex(
                    ValueError, "^WARNING: review extra-review.md$"
                ):
                    upload_source(book, config_path=config)


class DuplicateUploadTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.state = Path(self.directory.name) / "import.json"
        self.book = source()
        self.library = Library()

    async def preflight(self):
        await preflight_extras(self.book, self.state, CONFIG, self.library)

    async def warn(self):
        with self.assertRaisesRegex(ValueError, "WARNING.*manual review"):
            await self.preflight()

    def choose(self, page=PAGE):
        resolve_extra(self.state, 1, use_existing=page)
        self.book = json.loads(self.state.read_text())

    async def upload(self):
        with (
            patch("src.notion.upload.NotionBooks", return_value=self.library),
            patch("src.notion.upload.ensure_work", new=AsyncMock()) as work,
            patch("src.notion.upload.ensure_views", new=AsyncMock()),
            patch("notion_books.NotionBooks.ensure_options", new=AsyncMock()),
        ):
            try:
                await upload_draft(self.book, self.state, CONFIG, tools=self.library)
            except ValueError:
                work.assert_not_awaited()
                raise

    async def test_warning_before_any_remote_writes_then_reuse_curated_page(self):
        with self.assertRaisesRegex(ValueError, "WARNING"):
            await self.upload()
        self.assertEqual(self.library.calls, [])
        report = (self.state.parent / "extra-review.md").read_text()
        self.assertIn(PAGE, report)
        self.assertIn("布袋", report)
        self.assertIn("--create-new", report)
        self.assertNotIn("page_id", self.book["extras"][0])
        self.choose()
        existing = copy.deepcopy(self.library.pages[PAGE])
        await self.upload()
        self.assertTrue(self.book["uploaded"])
        self.assertEqual(self.library.pages[PAGE]["content"], existing["content"])
        self.assertEqual(
            self.library.pages[PAGE]["properties"]["番外"],
            existing["properties"]["番外"],
        )
        self.assertEqual(
            self.library.pages[PAGE]["properties"]["涉及作品"], [OTHER, WORK]
        )
        self.assertEqual(
            [
                args["command"]
                for name, args in self.library.calls
                if name == "notion-update-page"
            ],
            ["update_properties"],
        )
        count = len(self.library.calls)
        self.library.pages[PAGE]["content"] = "Later editorial change"
        await self.upload()
        self.assertEqual(len(self.library.calls), count)

    async def test_confirm_create_new_preserves_existing(self):
        await self.warn()
        self.choose(None)
        await self.upload()
        self.assertTrue(self.book["uploaded"])
        self.assertNotEqual(self.book["extras"][0]["page_id"], PAGE)
        self.assertEqual(self.library.pages[PAGE]["properties"]["涉及作品"], [OTHER])

    async def test_partial_overlap_stops_upload_and_reports_matching_excerpts(self):
        incoming, existing = different_versions(proofread=True)
        self.book["extras"][0]["blocks"] = from_markdown(incoming)
        self.library.pages[PAGE]["content"] = existing
        with self.assertRaisesRegex(ValueError, "WARNING"):
            await self.upload()
        self.assertEqual(self.library.calls, [])
        report = (self.state.parent / "extra-review.md").read_text()
        self.assertIn("局部文字重合", report)
        self.assertIn("归一化重合片段", report)
        score = self.book["extras"][0]["duplicate_review"]["candidates"][0]
        self.assertLess(score["similarity"], 0.1)
        self.assertIn(score["overlap_snippets"][0], report)
        self.choose(None)
        await self.upload()
        self.assertTrue(self.book["uploaded"])
        self.assertEqual(self.library.pages[PAGE]["content"], existing)

    async def test_exact_existing_still_reuses_without_review(self):
        page = self.library.pages[PAGE]
        page["properties"]["番外"] = self.book["extras"][0]["title"]
        page["content"] = TEXT
        await self.preflight()
        self.assertEqual(self.book["extras"][0]["page_id"], PAGE)
        self.assertTrue(self.book["extras"][0]["reused"])

    async def test_renamed_identical_body_requires_review(self):
        self.library.pages[PAGE]["content"] = TEXT
        await self.warn()
        candidates = self.book["extras"][0]["duplicate_review"]["candidates"]
        self.assertEqual(candidates[0]["similarity"], 1)

    async def test_same_title_unrelated_body_creates_normally(self):
        self.library.pages[PAGE]["properties"]["番外"] = self.book["extras"][0]["title"]
        self.library.pages[PAGE]["content"] = "另一个故事"
        await self.upload()
        self.assertTrue(self.book["uploaded"])
        self.assertNotEqual(self.book["extras"][0]["page_id"], PAGE)

    async def test_changed_candidate_invalidates_reuse_and_create_approval(self):
        for choice in (PAGE, None):
            with self.subTest(choice=choice):
                self.book = source()
                await self.warn()
                self.choose(choice)
                self.library.pages[PAGE]["content"] += "\n又一次校对。"
                await self.warn()
                self.assertNotIn("decision", self.book["extras"][0]["duplicate_review"])
                self.assertEqual(self.library.calls, [])

    async def test_disappeared_candidate_does_not_silently_create(self):
        await self.warn()
        self.choose()
        self.library.shared.clear()
        await self.warn()
        self.assertEqual(self.book["extras"][0]["duplicate_review"]["candidates"], [])
        self.choose(None)
        await self.preflight()

    async def test_new_candidate_requires_new_review(self):
        await self.warn()
        self.choose(None)
        self.library.pages[SECOND] = copy.deepcopy(self.library.pages[PAGE])
        self.library.shared.append(SECOND)
        await self.warn()
        self.assertEqual(
            len(self.book["extras"][0]["duplicate_review"]["candidates"]), 2
        )

    async def test_selected_but_unverified_reuse_is_rechecked_on_resume(self):
        await self.warn()
        self.choose()
        await self.preflight()
        self.assertEqual(self.book["extras"][0]["page_id"], PAGE)
        self.library.pages[PAGE]["content"] += "\n新的校对。"
        await self.warn()
        self.assertNotIn("page_id", self.book["extras"][0])
        self.choose()
        await self.upload()
        self.assertTrue(self.book["uploaded"])

    async def test_source_change_requires_review_again(self):
        await self.warn()
        self.choose()
        self.book["extras"][0]["blocks"] = from_markdown(TEXT + "\n新增的一行。")
        await self.warn()
        self.assertNotIn("decision", self.book["extras"][0]["duplicate_review"])

    async def test_multiple_exact_copies_require_choice(self):
        self.library.pages[PAGE]["properties"]["番外"] = self.book["extras"][0]["title"]
        self.library.pages[PAGE]["content"] = TEXT
        self.library.pages[SECOND] = copy.deepcopy(self.library.pages[PAGE])
        self.library.shared.append(SECOND)
        await self.warn()
        self.assertEqual(
            len(self.book["extras"][0]["duplicate_review"]["candidates"]), 2
        )

    async def test_all_extras_reported_library_read_only_once(self):
        self.book["extras"].append(
            {"title": "另一个标题", "blocks": from_markdown(TEXT)}
        )
        with self.assertRaisesRegex(ValueError, "WARNING: 2"):
            await self.preflight()
        self.assertEqual(self.library.reads, [PAGE])
        self.choose()
        resolve_extra(self.state, 2, use_existing=PAGE)
        self.book = json.loads(self.state.read_text())
        with self.assertRaisesRegex(ValueError, "same page"):
            await self.preflight()

    async def test_unknown_candidate_rejected(self):
        await self.warn()
        with self.assertRaisesRegex(ValueError, "listed"):
            self.choose(SECOND)

    async def test_incomplete_or_filtered_inventory_never_allows_upload(self):
        self.library.filter = {"some": "filter"}
        with self.assertRaisesRegex(ValueError, "unfiltered"):
            await self.upload()
        self.library.filter = None
        with patch.object(
            self.library,
            "document",
            new=AsyncMock(side_effect=ValueError("Incomplete Notion page")),
        ):
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                await self.upload()
        self.assertEqual(self.library.calls, [])
