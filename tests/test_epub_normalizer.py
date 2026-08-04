from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from unittest import mock
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from src.core.epub_normalizer import (
    normalize_and_review_epub,
    normalize_epub,
    write_reports,
)
from src.core.epub_writer import write_epub
from src.core.models import Chapter, Volume
from src.cli.validate_epub_chapters import validate_epub
from src.metadata.epub_enricher import MetadataEnrichmentReport


CHAPTER = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
  <head><title>第1章 小P孩</title></head>
  <body>
    <h2>第1章 小P孩</h2>
    <p>激 情, 下一站.</p>
    <p>主星VCU07， “你好”</p>
    <p>X337,Y160,Z19 γ-B11；2,000,000元。</p>
    <p>"直引号"</p>
    <p>““重复。”</p>
    <p>上一句还没结束，</p><p>下一句。</p>
    <p>“最后一句。</p><p>”</p>
    <p>坏字ㄉ和私用字。</p>
  </body>
</html>
"""
NAV = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
  <head><title>测试A书</title></head>
  <body><nav><ol><li><a href="chap_01_001.xhtml">第1章 小P孩</a></li></ol></nav></body>
</html>
"""
NCX = """<?xml version="1.0" encoding="utf-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/">
  <navMap><navPoint id="n1"><navLabel><text>第1章 小P孩</text></navLabel>
  <content src="chap_01_001.xhtml"/></navPoint></navMap>
</ncx>
"""
OPF = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf"
         xmlns:dc="http://purl.org/dc/elements/1.1/">
  <metadata><dc:title>测试A书</dc:title></metadata>
</package>
"""


class EpubNormalizerTests(unittest.TestCase):
    def test_normalizes_content_titles_structure_and_reports_bad_characters(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            epub_path = Path(temp) / "book.epub"
            self._write_fixture(epub_path, CHAPTER)

            report = normalize_epub(epub_path)

            self.assertGreater(report.total_changes, 0)
            self.assertEqual(
                report.change_counts["mixed_width_space_inserted"],
                14,
            )
            self.assertEqual(
                report.change_counts["comma_paragraph_break_merged"],
                1,
            )
            self.assertEqual(
                report.change_counts["isolated_closing_quote_merged"],
                1,
            )
            issue_kinds = [issue.kind for issue in report.issues]
            self.assertEqual(issue_kinds.count("suspicious_character"), 2)
            self.assertNotIn("quote_mismatch", issue_kinds)
            self.assertTrue(
                all(
                    "Browser Act" in issue.recommended_action
                    for issue in report.issues
                    if issue.kind == "suspicious_character"
                )
            )

            with zipfile.ZipFile(epub_path) as archive:
                infos = archive.infolist()
                self.assertEqual(infos[0].filename, "mimetype")
                self.assertEqual(infos[0].compress_type, zipfile.ZIP_STORED)
                chapter = archive.read("EPUB/chap_01_001.xhtml").decode()
                nav = archive.read("EPUB/nav.xhtml").decode()
                ncx = archive.read("EPUB/toc.ncx").decode()
                opf = archive.read("EPUB/content.opf").decode()
                for member in (chapter, nav, ncx, opf):
                    ET.fromstring(member)

            self.assertIn("第1章 小 P 孩", chapter)
            self.assertIn("激情，下一站。", chapter)
            self.assertIn("主星 VCU07，“你好”", chapter)
            self.assertIn("X337, Y160, Z19 γ-B11；2,000,000 元。", chapter)
            self.assertIn("<p>“直引号”</p>", chapter)
            self.assertIn("<p>“重复。”</p>", chapter)
            self.assertIn("<p>上一句还没结束，下一句。</p>", chapter)
            self.assertIn("<p>“最后一句。”</p>", chapter)
            self.assertIn("第1章 小 P 孩", nav)
            self.assertIn("第1章 小 P 孩", ncx)
            self.assertIn("测试 A 书", opf)

            second = normalize_epub(epub_path)
            self.assertEqual(second.total_changes, 0)
            self.assertEqual(len(second.issues), 2)

            report_path = Path(temp) / "report.json"
            write_reports(report_path, [report])
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["reports"][0]["total_changes"], report.total_changes)
            self.assertEqual(
                payload["cumulative_fixes"]["total_changes"],
                report.total_changes,
            )
            self.assertTrue(
                all(
                    issue["requires_codex_review"]
                    for issue in payload["reports"][0]["issues"]
                )
            )

            write_reports(report_path, [second])
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["fix_history"]), 1)
            self.assertEqual(
                payload["cumulative_fixes"]["total_changes"],
                report.total_changes,
            )

    def test_check_mode_reports_without_rewriting(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            epub_path = Path(temp) / "book.epub"
            self._write_fixture(epub_path, "<html><body><p>激 情.</p></body></html>")
            before = epub_path.read_bytes()

            report = normalize_epub(epub_path, apply=False)

            self.assertGreater(report.total_changes, 0)
            self.assertEqual(epub_path.read_bytes(), before)

    def test_white_circle_in_chinese_numeral_run_becomes_ideographic_zero(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            epub_path = Path(temp) / "unicode-zero.epub"
            self._write_fixture(
                epub_path,
                """<html><body>
                <p>西元二○四六，香港元朗区，夜。</p>
                <p>编号一○，旧年份二○○六。</p>
                <p>几何符号○和装饰○○保留。</p>
                </body></html>""",
            )

            report = normalize_epub(epub_path)

            self.assertEqual(
                report.change_counts["white_circle_to_ideographic_zero"],
                4,
            )
            with zipfile.ZipFile(epub_path) as archive:
                chapter = archive.read("EPUB/chap_01_001.xhtml").decode()
            self.assertIn("西元二〇四六，香港元朗区，夜。", chapter)
            self.assertIn("编号一〇，旧年份二〇〇六。", chapter)
            self.assertIn("几何符号○和装饰○○保留。", chapter)

            second = normalize_epub(epub_path)
            self.assertEqual(second.total_changes, 0)

    def test_unresolved_quote_mismatch_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            epub_path = Path(temp) / "book.epub"
            self._write_fixture(
                epub_path,
                "<html><body><p>“这句话没有收尾。</p></body></html>",
            )

            report = normalize_epub(epub_path)

            mismatch = [
                issue for issue in report.issues if issue.kind == "quote_mismatch"
            ]
            self.assertEqual(len(mismatch), 1)
            self.assertIn("paragraph 1", mismatch[0].message)
            self.assertIn("opening=1, closing=0", mismatch[0].message)
            self.assertIn("这句话没有收尾", mismatch[0].excerpt)


    def test_quote_detection_uses_normalized_local_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            epub_path = Path(temp) / "quotes.epub"
            self._write_fixture(
                epub_path,
                """<html><body>
                <p>周昇回来的时候发现余皓在和自己的妈打电话。 “好……的……好的……”余皓和周昇抢电话线。</p>
                <p>余皓摘下耳机，说：“对啊。“</p>
                <p>“欢迎您进入大型沉浸式情景剧。</p>
                <p>“请完成任务后回到现实世界。”</p>
                <p>他说：“第一半</p>
                <p>第二半。”</p>
                <p>他说：“连续引语第一段。</p>
                <p>“连续引语第二段。</p>
                <p>“连续引语第三段。”</p>
                <p>“歌词第一行；</p>
                <p>歌词中间没有重复开引号；</p>
                <p>歌词最后一行。”</p>
                <p>多个对话：“第一句。”“第二句。”</p>
                <p>听说孙策被唤’美孙郎‘，确实好看。</p>
                <p>用‘你’还是用‘我’，‘清官’一样是‘清官’。</p>
                <p>“Darling, I’ll follow your lead, and you’re ready.”</p>
                <p>贺简躺成_(:3」∠)_的姿势。</p>
                </body></html>""",
            )

            report = normalize_epub(epub_path)

            with zipfile.ZipFile(epub_path) as archive:
                chapter = archive.read("EPUB/chap_01_001.xhtml").decode()
            self.assertIn("余皓摘下耳机，说：“对啊。”", chapter)
            self.assertIn("听说孙策被唤‘美孙郎’，确实好看。", chapter)
            self.assertIn(
                "用‘你’还是用‘我’，‘清官’一样是‘清官’。",
                chapter,
            )
            self.assertEqual(
                report.change_counts["contextual_quote_direction_fixed"],
                1,
            )
            self.assertEqual(
                report.change_counts["single_quote_direction_fixed"],
                1,
            )
            self.assertFalse(
                any(issue.kind == "quote_mismatch" for issue in report.issues)
            )

    def test_chapter_level_quote_parser_still_reports_real_residue(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            epub_path = Path(temp) / "real-quote-errors.epub"
            self._write_fixture(
                epub_path,
                """<html><body>
                <p>前文是正常的。</p>
                <p>“从东南角走是最安全的。”王雷道：“</p>
                <p>迟小多说：“下一句自身完整。”</p>
                </body></html>""",
            )

            report = normalize_epub(epub_path)

            mismatch = [
                issue for issue in report.issues if issue.kind == "quote_mismatch"
            ]
            residue = [
                issue
                for issue in report.issues
                if issue.kind == "orphan_dialogue_quote_residue"
            ]
            self.assertEqual(len(residue), 1)
            self.assertIn("王雷道", residue[0].excerpt)
            self.assertLessEqual(len(mismatch), 1)


    def test_codex_keep_decision_closes_review_queue(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            epub_path = Path(temp) / "review.epub"
            self._write_fixture(
                epub_path,
                (
                    "<html><body><h2>第1章 小P孩</h2>"
                    "<p>他说：“故意保留开放式引语。</p></body></html>"
                ),
            )

            report = normalize_and_review_epub(
                epub_path,
                review_runner=lambda _prompt: json.dumps(
                    {
                        "reviews": [
                            {
                                "issue_index": 0,
                                "verdict": "keep",
                                "confidence": "high",
                                "reason": "Confirmed intentional typography.",
                                "old": "",
                                "new": "",
                            }
                        ]
                    }
                ),
            )

            self.assertEqual(report.codex_review_status, "completed")
            self.assertEqual(len(report.issues), 1)
            self.assertFalse(report.issues[0].requires_codex_review)
            self.assertFalse(report.issues[0].requires_user_review)
            self.assertEqual(report.issues[0].review_verdict, "keep")


    def test_unchanged_codex_keep_decision_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            epub_path = Path(temp) / "cached-review.epub"
            self._write_fixture(
                epub_path,
                (
                    "<html><body><h2>第1章 小P孩</h2>"
                    "<p>他说：“故意保留开放式引语。</p></body></html>"
                ),
            )
            response = json.dumps(
                {
                    "reviews": [
                        {
                            "issue_index": 0,
                            "verdict": "keep",
                            "confidence": "high",
                            "reason": "Confirmed intentional typography.",
                            "old": "",
                            "new": "",
                        }
                    ]
                }
            )
            first = normalize_and_review_epub(
                epub_path,
                review_runner=lambda _prompt: response,
            )
            write_reports(epub_path.with_suffix(".normalization.json"), [first])

            with mock.patch(
                "src.core.epub_normalizer._run_codex_review_prompt"
            ) as review:
                second = normalize_and_review_epub(epub_path)

            review.assert_not_called()
            self.assertEqual(second.codex_review_status, "completed")
            self.assertFalse(second.issues[0].requires_codex_review)
            self.assertTrue(second.codex_review_decisions[0]["cached"])


    def test_codex_review_failure_is_reported_without_blocking_normalization(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            epub_path = Path(temp) / "failed-review.epub"
            self._write_fixture(
                epub_path,
                "<html><body><p>他说：“激 情, 下一站.</p></body></html>",
            )

            def fail_review(_prompt: str) -> str:
                raise RuntimeError("review service unavailable")

            report = normalize_and_review_epub(
                epub_path,
                review_runner=fail_review,
            )

            with zipfile.ZipFile(epub_path) as archive:
                chapter = archive.read("EPUB/chap_01_001.xhtml").decode()
            self.assertIn("他说：“激情，下一站。", chapter)
            self.assertEqual(report.codex_review_status, "failed")
            self.assertIn("review service unavailable", report.codex_review_error or "")
            self.assertTrue(report.issues[0].requires_codex_review)

    def test_dialogue_and_ascii_punctuation_rules_are_context_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            epub_path = Path(temp) / "book.epub"
            self._write_fixture(
                epub_path,
                """<html><body>
                <p>“喝点什么？“曹斌问，”可以吗？”</p>
                <p>Silent night,holy night. I don't know.</p>
                <p>版本2.0，价格2,000元，网址https://example.test/a.</p>
                <p>图片是冷漠.jpg，视频叫睡觉.avi。</p>
                <p>“Loving you……is easy ’cause you’re beautiful……”</p>
                <p>表情：【 ＼／ 】</p>
                <p>中文问句?真的!标签:值;等等...前--后。</p>
                <p>说明(中文内容)，状态[系统提示]。</p>
                <p>提示: "Hello, world!" at 12:30; keep [API] (test).</p>
                <p>倒计时 XX:XX:XX，珍?奥与数值?????/?????需要人工判断。</p>
                </body></html>""",
            )

            report = normalize_epub(epub_path)

            with zipfile.ZipFile(epub_path) as archive:
                chapter = archive.read("EPUB/chap_01_001.xhtml").decode()
            self.assertIn("“喝点什么？”曹斌问，“可以吗？”", chapter)
            self.assertIn("Silent night, holy night. I don't know.", chapter)
            self.assertIn(
                "版本 2.0，价格 2,000 元，网址 https://example.test/a.",
                chapter,
            )
            self.assertIn("图片是冷漠.jpg，视频叫睡觉.avi。", chapter)
            self.assertIn(
                "“Loving you……is easy ’cause you’re beautiful……”",
                chapter,
            )
            self.assertIn("表情：【 ＼／ 】", chapter)
            self.assertIn(
                "中文问句？真的！标签：值；等等……前——后。",
                chapter,
            )
            self.assertIn("说明（中文内容），状态［系统提示］。", chapter)
            self.assertIn(
                '提示："Hello, world!" at 12:30; keep [API] (test).',
                chapter,
            )
            self.assertIn(
                "倒计时 XX:XX:XX，珍?奥与数值?????/?????需要人工判断。",
                chapter,
            )
            self.assertTrue(
                any(
                    issue.kind == "suspicious_ascii_punctuation"
                    for issue in report.issues
                )
            )
            self.assertEqual(
                report.change_counts["dialogue_quote_direction_fixed"],
                1,
            )

    def test_punctuation_normalization_runs_before_quote_direction(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            epub_path = Path(temp) / "punctuation-first.epub"
            self._write_fixture(
                epub_path,
                (
                    "<html><body>"
                    "<p>“喝点什么?“曹斌问,”可以吗?”</p>"
                    "</body></html>"
                ),
            )

            report = normalize_epub(epub_path)

            with zipfile.ZipFile(epub_path) as archive:
                chapter = archive.read("EPUB/chap_01_001.xhtml").decode()
            self.assertIn("“喝点什么？”曹斌问，“可以吗？”", chapter)
            self.assertEqual(
                report.change_counts["ascii_question_to_chinese"],
                2,
            )
            self.assertEqual(
                report.change_counts["ascii_comma_to_chinese"],
                1,
            )
            self.assertEqual(
                report.change_counts["dialogue_quote_direction_fixed"],
                1,
            )

    def test_known_split_han_words_are_joined_before_ambiguous_space_review(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            epub_path = Path(temp) / "known-split-words.epub"
            self._write_fixture(
                epub_path,
                (
                    "<html><body><p>"
                    "洗去体 液，性 梦惊醒，胯 下硬 挺，赤 裸，"
                    "勃 起，性 具，后 庭，阳 具，抽 插，"
                    "坚 挺的下 身与光 裸的屁 股、臀 股，无非 是要 去。"
                    "</p></body></html>"
                ),
            )

            report = normalize_epub(epub_path)

            with zipfile.ZipFile(epub_path) as archive:
                chapter = archive.read("EPUB/chap_01_001.xhtml").decode()
            self.assertIn(
                "洗去体液，性梦惊醒，胯下硬挺，赤裸，"
                "勃起，性具，后庭，阳具，抽插，"
                "坚挺的下身与光裸的屁股、臀股，无非是要去。",
                chapter,
            )
            self.assertEqual(
                report.change_counts["known_split_han_word_joined"],
                17,
            )
            self.assertFalse(
                any(issue.kind == "ambiguous_han_spacing" for issue in report.issues)
            )

    def test_duplicate_quotes_and_han_spacing_use_conservative_contexts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            epub_path = Path(temp) / "book.epub"
            self._write_fixture(
                epub_path,
                """<html><body>
                <h2>第一回：祥瑞</h2>
                <p>：““你好。”</p>
                <p>你好吗？”“下一句。</p>
                <p>结束。””下一句。</p>
                <p>尾声……””</p>
                <p>去光明神殿““圣泉。</p>
                <p>内容标签：强强 天之骄子</p>
                <p>第五回 遇章邯犹似故人面 闻刘恒如同旧日音</p>
                <p>普通文字里有激 情。</p>
                <p>反斜线误码是激\\情，但 C:\\Books\\book.epub 要保留。</p>
                <p>称谓选项本宫\\本座\\哀家\\朕要保留。</p>
                <p>多处空格应该留给人工判断：长歌 当哭 天涯 共此时。</p>
                </body></html>""",
            )

            report = normalize_epub(epub_path)

            with zipfile.ZipFile(epub_path) as archive:
                chapter = archive.read("EPUB/chap_01_001.xhtml").decode()
            self.assertIn("<h2>第一回：祥瑞</h2>", chapter)
            self.assertIn("<p>：“你好。”</p>", chapter)
            self.assertIn("<p>你好吗？”“下一句。</p>", chapter)
            self.assertIn("<p>结束。”“下一句。</p>", chapter)
            self.assertIn("<p>尾声……”</p>", chapter)
            self.assertIn("<p>去光明神殿““圣泉。</p>", chapter)
            self.assertIn("<p>内容标签：强强 天之骄子</p>", chapter)
            self.assertIn(
                "<p>第五回 遇章邯犹似故人面 闻刘恒如同旧日音</p>",
                chapter,
            )
            self.assertIn("<p>普通文字里有激情。</p>", chapter)
            self.assertIn(
                "<p>反斜线误码是激情，但 C:\\Books\\book.epub 要保留。</p>",
                chapter,
            )
            self.assertEqual(
                report.change_counts["backslash_between_han_removed"],
                1,
            )
            self.assertIn("<p>称谓选项本宫\\本座\\哀家\\朕要保留。</p>", chapter)
            self.assertIn(
                "<p>多处空格应该留给人工判断：长歌 当哭 天涯 共此时。</p>",
                chapter,
            )
            self.assertTrue(
                any(
                    issue.kind == "ambiguous_duplicate_quote"
                    for issue in report.issues
                )
            )
            self.assertTrue(
                any(
                    issue.kind == "ambiguous_han_backslash"
                    for issue in report.issues
                )
            )
            self.assertTrue(
                any(
                    issue.kind == "ambiguous_han_spacing"
                    for issue in report.issues
                )
            )

    def test_suspicious_ads_are_reported_but_not_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            epub_path = Path(temp) / "book.epub"
            original = (
                "角色打开了真实网站 https://example.com/story。"
                "关注微信公众号：测试账号，更多精彩请访问备用网址。"
            )
            self._write_fixture(
                epub_path,
                f"<html><body><p>{original}</p></body></html>",
            )

            report = normalize_epub(epub_path)

            with zipfile.ZipFile(epub_path) as archive:
                chapter = archive.read("EPUB/chap_01_001.xhtml").decode()
            self.assertIn("https://example.com/story", chapter)
            self.assertIn("关注微信公众号：测试账号", chapter)
            ads = [
                issue for issue in report.issues
                if issue.kind == "suspicious_ad"
            ]
            self.assertGreaterEqual(len(ads), 2)
            self.assertTrue(
                all("do not delete automatically" in issue.recommended_action for issue in ads)
            )

    def test_author_notes_are_permitted_but_resource_group_ads_are_removed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            epub_path = Path(temp) / "book.epub"
            self._write_fixture(
                epub_path,
                """<html><body>
                <p>正文结束。</p>
                <p>作者有话要说：</p>
                <p>个人志预售详情请关注我的微博@作者。</p>
                <p>晋江同步连载。耽美小说资源群：496267128</p>
                </body></html>""",
            )

            report = normalize_epub(epub_path)

            with zipfile.ZipFile(epub_path) as archive:
                chapter = archive.read("EPUB/chap_01_001.xhtml").decode()
            self.assertIn("个人志预售详情请关注我的微博@作者。", chapter)
            self.assertIn("<p>晋江同步连载。</p>", chapter)
            self.assertNotIn("496267128", chapter)
            self.assertEqual(
                report.change_counts["resource_group_ad_removed"],
                1,
            )
            self.assertFalse(
                any(issue.kind == "suspicious_ad" for issue in report.issues)
            )

    def test_author_note_headings_are_kept_on_separate_paragraphs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            epub_path = Path(temp) / "author-note-break.epub"
            self._write_fixture(
                epub_path,
                """<html><body>
                <p>正文结束。作者有话说：第一条。</p>
                <p>上一句还没结束，</p>
                <p>作者有话要说：第二条。</p>
                <p>这里作者有话说 4000，正文 4000。</p>
                </body></html>""",
            )

            report = normalize_epub(epub_path)

            with zipfile.ZipFile(epub_path) as archive:
                chapter = archive.read("EPUB/chap_01_001.xhtml").decode()
            self.assertIn("<p>正文结束。</p>\n<p>作者有话说：第一条。</p>", chapter)
            self.assertIn("<p>上一句还没结束，</p>", chapter)
            self.assertIn("<p>作者有话要说：第二条。</p>", chapter)
            self.assertIn("这里作者有话说 4000，正文 4000。", chapter)
            self.assertEqual(
                report.change_counts["author_note_paragraph_break_inserted"],
                1,
            )
            self.assertEqual(report.change_counts["comma_paragraph_break_merged"], 0)

            second = normalize_epub(epub_path)
            self.assertEqual(second.total_changes, 0)


    def test_trailing_author_notes_and_emoticons_are_excluded_from_anomalies(
        self,
    ) -> None:
        chapter = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
  <head><title>第1章 测试</title></head>
  <body>
    <h2>第1章 测试</h2>
    <p>正文内容。</p>
    <p>正文颜文字 /(ㄒoㄒ)/~~ 和 emoji 👩‍💻。</p>
    <p>正文仍应报告坏字ㄉ。</p>
    <p>作者有话要说：</p>
    <p>自由格式 ㄒ_____ㄒ 和私用字都不进入报告。</p>
  </body>
</html>
"""
        with tempfile.TemporaryDirectory() as temp:
            epub_path = Path(temp) / "author-note.epub"
            self._write_fixture(epub_path, chapter)
            report = normalize_epub(epub_path, apply=False)

        suspicious = [
            issue
            for issue in report.issues
            if issue.kind == "suspicious_character"
        ]
        self.assertEqual(len(suspicious), 1)
        self.assertIn("U+3109", suspicious[0].message)
        self.assertNotIn("U+3112", suspicious[0].message)
        self.assertNotIn("U+E788", suspicious[0].message)

    def test_intro_is_excluded_from_anomaly_reports_but_still_normalized(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            epub_path = Path(temp) / "intro.epub"
            self._write_fixture(
                epub_path,
                "<html><body><p>正文。</p></body></html>",
                intro=(
                    "<html><body>"
                    "<p>内容标签：都市情缘 情有独钟 因缘邂逅 天作之合</p>"
                    "<p>搜索关键字：主角：余皓 周昇</p>"
                    "<p>故意保留的奇怪格式：“未闭合</p>"
                    "<p>晋江同步连载。耽美小说资源群：496267128</p>"
                    "</body></html>"
                ),
            )

            report = normalize_epub(epub_path)

            with zipfile.ZipFile(epub_path) as archive:
                intro = archive.read("EPUB/intro.xhtml").decode()
            self.assertIn("都市情缘 情有独钟 因缘邂逅 天作之合", intro)
            self.assertIn("故意保留的奇怪格式：“未闭合", intro)
            self.assertNotIn("496267128", intro)
            self.assertEqual(
                report.change_counts["resource_group_ad_removed"],
                1,
            )
            self.assertFalse(
                any(
                    issue.member == "EPUB/intro.xhtml"
                    for issue in report.issues
                )
            )

    def test_grouped_fanwai_titles_drop_number_but_keep_bare_entries(self) -> None:
        cases = (
            ("第114章 沧浪之龙", "沧浪之龙", 4),
            ("第60章", "第60章", 0),
        )
        for source_title, expected_title, prefix_changes in cases:
            with self.subTest(source_title=source_title):
                with tempfile.TemporaryDirectory() as temp:
                    epub_path = Path(temp) / "book.epub"
                    chapter = CHAPTER.replace("第1章 小P孩", source_title)
                    nav = f"""<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
  <body><nav><ol><li><a href="chap_01_001.xhtml">番外</a><ol>
    <li><a href="chap_01_001.xhtml">{source_title}</a></li>
  </ol></li></ol></nav></body>
</html>
"""
                    ncx = f"""<?xml version="1.0" encoding="utf-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/"><navMap>
  <navPoint id="fanwai"><navLabel><text>番外</text></navLabel>
    <content src="chap_01_001.xhtml"/>
    <navPoint id="n1"><navLabel><text>{source_title}</text></navLabel>
      <content src="chap_01_001.xhtml"/></navPoint>
  </navPoint>
</navMap></ncx>
"""
                    self._write_fixture(epub_path, chapter, nav=nav, ncx=ncx)

                    report = normalize_epub(epub_path)

                    self.assertEqual(
                        report.change_counts.get(
                            "fanwai_chapter_number_removed",
                            0,
                        ),
                        prefix_changes,
                    )
                    with zipfile.ZipFile(epub_path) as archive:
                        outputs = (
                            archive.read("EPUB/chap_01_001.xhtml").decode(),
                            archive.read("EPUB/nav.xhtml").decode(),
                            archive.read("EPUB/toc.ncx").decode(),
                        )
                    for output in outputs:
                        self.assertIn(expected_title, output)
                        if expected_title != source_title:
                            self.assertNotIn(source_title, output)

                    second = normalize_epub(epub_path)
                    self.assertEqual(second.total_changes, 0)

    def test_fanwai_colon_and_middle_autumn_year_are_normalized(self) -> None:
        cases = (
            (
                "2021 年中秋节番外：前年风月满江湖",
                "2021 年中秋节番外·前年风月满江湖",
                4,
                0,
                0,
                0,
            ),
            (
                "2022 中秋番外·游园",
                "2022 年中秋番外·游园",
                0,
                4,
                0,
                0,
            ),
            (
                "2017中秋番外：贺中秋",
                "2017 年中秋番外·贺中秋",
                4,
                4,
                0,
                0,
            ),
            (
                "2017 中秋·番外贺中秋·月中霜里斗婵娟",
                "2017 年中秋番外·贺中秋·月中霜里斗婵娟",
                0,
                4,
                4,
                4,
            ),
        )
        for (
            source_title,
            expected_title,
            colon_changes,
            year_changes,
            internal_separator_changes,
            title_separator_changes,
        ) in cases:
            with self.subTest(source_title=source_title):
                with tempfile.TemporaryDirectory() as temp:
                    epub_path = Path(temp) / "book.epub"
                    chapter = CHAPTER.replace("第1章 小P孩", source_title)
                    nav = NAV.replace("第1章 小P孩", source_title)
                    ncx = NCX.replace("第1章 小P孩", source_title)
                    self._write_fixture(epub_path, chapter, nav=nav, ncx=ncx)

                    report = normalize_epub(epub_path)

                    self.assertEqual(
                        report.change_counts.get(
                            "fanwai_colon_to_middle_dot",
                            0,
                        ),
                        colon_changes,
                    )
                    self.assertEqual(
                        report.change_counts.get(
                            "fanwai_middle_autumn_year_added",
                            0,
                        ),
                        year_changes,
                    )
                    self.assertEqual(
                        report.change_counts.get(
                            "fanwai_middle_autumn_internal_separator_removed",
                            0,
                        ),
                        internal_separator_changes,
                    )
                    self.assertEqual(
                        report.change_counts.get(
                            "fanwai_middle_autumn_title_separator_added",
                            0,
                        ),
                        title_separator_changes,
                    )
                    with zipfile.ZipFile(epub_path) as archive:
                        outputs = (
                            archive.read("EPUB/chap_01_001.xhtml").decode(),
                            archive.read("EPUB/nav.xhtml").decode(),
                            archive.read("EPUB/toc.ncx").decode(),
                        )
                    for output in outputs:
                        self.assertIn(expected_title, output)
                        self.assertNotIn(source_title, output)

                    second = normalize_epub(epub_path)
                    self.assertEqual(second.total_changes, 0)

    def test_fanwai_titles_drop_chapter_number_and_use_middle_dot(self) -> None:
        cases = (
            ("第152章 番外一承前启后", "番外一·承前启后", 4, 0, 0),
            ("第155章 番外四", "番外四", 0, 0, 0),
            ("第156章 番外 四", "番外四", 0, 0, 4),
            ("第158章 番外 1 飞天猫", "番外一·飞天猫", 4, 4, 4),
            ("第138章 番外 10", "番外十", 0, 4, 4),
            ("第142章 小番外二则", "小番外二则", 0, 0, 0),
            ("第169章 番外十八·隋州", "番外十八·隋州", 0, 0, 0),
            ("第283章 番外十三（完）", "番外十三（完）", 0, 0, 0),
            ("第292章 番外六一快乐", "番外六一快乐", 0, 0, 0),
        )
        for (
            source_title,
            expected_title,
            separator_changes,
            number_changes,
            number_spacing_changes,
        ) in cases:
            with self.subTest(source_title=source_title):
                with tempfile.TemporaryDirectory() as temp:
                    epub_path = Path(temp) / "book.epub"
                    chapter = CHAPTER.replace("第1章 小P孩", source_title)
                    nav = NAV.replace("第1章 小P孩", source_title)
                    ncx = NCX.replace("第1章 小P孩", source_title)
                    self._write_fixture(epub_path, chapter, nav=nav, ncx=ncx)

                    report = normalize_epub(epub_path)

                    self.assertEqual(
                        report.change_counts["fanwai_chapter_number_removed"],
                        4,
                    )
                    self.assertEqual(
                        report.change_counts.get(
                            "fanwai_title_separator_normalized",
                            0,
                        ),
                        separator_changes,
                    )
                    self.assertEqual(
                        report.change_counts.get(
                            "fanwai_number_to_chinese",
                            0,
                        ),
                        number_changes,
                    )
                    self.assertEqual(
                        report.change_counts.get(
                            "fanwai_number_spacing_removed",
                            0,
                        ),
                        number_spacing_changes,
                    )
                    with zipfile.ZipFile(epub_path) as archive:
                        outputs = (
                            archive.read("EPUB/chap_01_001.xhtml").decode(),
                            archive.read("EPUB/nav.xhtml").decode(),
                            archive.read("EPUB/toc.ncx").decode(),
                        )
                    for output in outputs:
                        self.assertIn(expected_title, output)
                        self.assertNotIn(source_title, output)

                    second = normalize_epub(epub_path)
                    self.assertEqual(second.total_changes, 0)

    def test_grouped_fanwai_titles_drop_a_single_trailing_separator(self) -> None:
        for source_title in (
            "第223章 2018 年戊戌年中秋番外·啷里个啷.",
            "第223章 2018 年戊戌年中秋番外·啷里个啷·",
        ):
            with self.subTest(source_title=source_title):
                with tempfile.TemporaryDirectory() as temp:
                    epub_path = Path(temp) / "book.epub"
                    chapter = CHAPTER.replace("第1章 小P孩", source_title)
                    nav = f"""<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
  <body><nav><ol><li><a href="chap_01_001.xhtml">番外</a><ol>
    <li><a href="chap_01_001.xhtml">{source_title}</a></li>
  </ol></li></ol></nav></body>
</html>
"""
                    ncx = f"""<?xml version="1.0" encoding="utf-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/"><navMap>
  <navPoint id="fanwai"><navLabel><text>番外</text></navLabel>
    <content src="chap_01_001.xhtml"/>
    <navPoint id="n1"><navLabel><text>{source_title}</text></navLabel>
      <content src="chap_01_001.xhtml"/></navPoint>
  </navPoint>
</navMap></ncx>
"""
                    self._write_fixture(epub_path, chapter, nav=nav, ncx=ncx)

                    report = normalize_epub(epub_path)

                    self.assertEqual(
                        report.change_counts["fanwai_trailing_separator_removed"],
                        4,
                    )
                    expected_title = "2018 年戊戌年中秋番外·啷里个啷"
                    with zipfile.ZipFile(epub_path) as archive:
                        outputs = (
                            archive.read("EPUB/chap_01_001.xhtml").decode(),
                            archive.read("EPUB/nav.xhtml").decode(),
                            archive.read("EPUB/toc.ncx").decode(),
                        )
                    for output in outputs:
                        self.assertIn(expected_title, output)
                        self.assertNotIn(source_title, output)

                    second = normalize_epub(epub_path)
                    self.assertEqual(second.total_changes, 0)

    def test_nav_uses_canonical_epub_namespace_prefix(self) -> None:
        nav = NAV.replace(
            '<html xmlns="http://www.w3.org/1999/xhtml">',
            (
                '<ns0:html xmlns:ns0="http://www.w3.org/1999/xhtml" '
                'xmlns:ns1="http://www.idpf.org/2007/ops">'
            ),
        ).replace("</html>", "</ns0:html>")
        for tag in ("head", "title", "body", "nav", "ol", "li", "a"):
            nav = nav.replace(f"<{tag}", f"<ns0:{tag}")
            nav = nav.replace(f"</{tag}>", f"</ns0:{tag}>")
        nav = nav.replace("<ns0:nav>", '<ns0:nav ns1:type="toc">')

        with tempfile.TemporaryDirectory() as temp:
            epub_path = Path(temp) / "book.epub"
            self._write_fixture(epub_path, CHAPTER, nav=nav)

            report = normalize_epub(epub_path)

            self.assertEqual(
                report.change_counts["nav_epub_namespace_prefix_normalized"],
                2,
            )
            with zipfile.ZipFile(epub_path) as archive:
                output = archive.read("EPUB/nav.xhtml").decode()
            self.assertIn(
                'xmlns:epub="http://www.idpf.org/2007/ops"',
                output,
            )
            self.assertIn('epub:type="toc"', output)
            self.assertNotIn("ns1:", output)

            second = normalize_epub(epub_path)
            self.assertEqual(second.total_changes, 0)

    def test_grouped_fanwai_titles_drop_redundant_prefix(self) -> None:
        cases = (
            ("第73章 番外：打飞机奇遇记 1", "打飞机奇遇记 1", 4, 4, 4),
            ("番外: 蜜月流水账", "蜜月流水账", 0, 4, 4),
            ("番外一·故事开始", "一·故事开始", 0, 0, 4),
            ("番外——野旷天低树，江清月近人", "野旷天低树，江清月近人", 0, 0, 4),
            ("加笔番外", "加笔番外", 0, 0, 0),
            ("番外", "番外", 0, 0, 0),
        )
        for (
            source_title,
            expected_title,
            number_changes,
            colon_changes,
            prefix_changes,
        ) in cases:
            with self.subTest(source_title=source_title):
                with tempfile.TemporaryDirectory() as temp:
                    epub_path = Path(temp) / "book.epub"
                    chapter = CHAPTER.replace("第1章 小P孩", source_title)
                    nav = f"""<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
  <body><nav><ol><li><a href="chap_01_001.xhtml">番外</a><ol>
    <li><a href="chap_01_001.xhtml">{source_title}</a></li>
  </ol></li></ol></nav></body>
</html>
"""
                    ncx = f"""<?xml version="1.0" encoding="utf-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/"><navMap>
  <navPoint id="fanwai"><navLabel><text>番外</text></navLabel>
    <content src="chap_01_001.xhtml"/>
    <navPoint id="n1"><navLabel><text>{source_title}</text></navLabel>
      <content src="chap_01_001.xhtml"/></navPoint>
  </navPoint>
</navMap></ncx>
"""
                    self._write_fixture(epub_path, chapter, nav=nav, ncx=ncx)

                    report = normalize_epub(epub_path)

                    self.assertEqual(
                        report.change_counts.get(
                            "fanwai_chapter_number_removed",
                            0,
                        ),
                        number_changes,
                    )
                    self.assertEqual(
                        report.change_counts.get("fanwai_colon_to_middle_dot", 0),
                        colon_changes,
                    )
                    self.assertEqual(
                        report.change_counts.get("fanwai_group_prefix_removed", 0),
                        prefix_changes,
                    )
                    with zipfile.ZipFile(epub_path) as archive:
                        outputs = (
                            archive.read("EPUB/chap_01_001.xhtml").decode(),
                            archive.read("EPUB/nav.xhtml").decode(),
                            archive.read("EPUB/toc.ncx").decode(),
                        )
                    for output in outputs:
                        self.assertIn(expected_title, output)
                        if expected_title != source_title:
                            self.assertNotIn(source_title, output)
                    self.assertIn(">番外</a>", outputs[1])

                    second = normalize_epub(epub_path)
                    self.assertEqual(second.total_changes, 0)

    def test_flat_fanwai_title_uses_middle_dot_after_section_prefix(self) -> None:
        source_title = "番外：蜜月流水账"
        expected_title = "番外·蜜月流水账"
        with tempfile.TemporaryDirectory() as temp:
            epub_path = Path(temp) / "book.epub"
            chapter = CHAPTER.replace("第1章 小P孩", source_title)
            nav = NAV.replace("第1章 小P孩", source_title)
            ncx = NCX.replace("第1章 小P孩", source_title)
            self._write_fixture(epub_path, chapter, nav=nav, ncx=ncx)

            report = normalize_epub(epub_path)

            self.assertEqual(
                report.change_counts.get("fanwai_group_prefix_removed", 0),
                0,
            )
            self.assertEqual(
                report.change_counts["fanwai_colon_to_middle_dot"],
                4,
            )
            with zipfile.ZipFile(epub_path) as archive:
                outputs = (
                    archive.read("EPUB/chap_01_001.xhtml").decode(),
                    archive.read("EPUB/nav.xhtml").decode(),
                    archive.read("EPUB/toc.ncx").decode(),
                )
            for output in outputs:
                self.assertIn(expected_title, output)
                self.assertNotIn(source_title, output)

            second = normalize_epub(epub_path)
            self.assertEqual(second.total_changes, 0)

    def test_reformatter_skill_rules_fix_safe_cases_and_report_structure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            epub_path = Path(temp) / "book.epub"
            chapter = CHAPTER.replace(
                "第1章 小P孩",
                "第1章《上.下》",
            ).replace(
                "</body>",
                (
                    "<p>等等，，快走。</p>"
                    "<p>-----卷四羽觞醉月终-----</p>"
                    "<p>——卷二·如梦·完——</p>"
                    "<p>后记</p></body>"
                ),
            )
            nav = NAV.replace("第1章 小P孩", "第1章《上.下》")
            ncx = NCX.replace("第1章 小P孩", "第1章《上.下》")
            intro = (
                "<html><body><p>《测试 A 书》作者：测试作者</p></body></html>"
            )
            self._write_fixture(
                epub_path,
                chapter,
                nav=nav,
                ncx=ncx,
                intro=intro,
            )

            report = normalize_epub(epub_path)

            with zipfile.ZipFile(epub_path) as archive:
                chapter_output = archive.read(
                    "EPUB/chap_01_001.xhtml"
                ).decode()
                nav_output = archive.read("EPUB/nav.xhtml").decode()
                ncx_output = archive.read("EPUB/toc.ncx").decode()
            for output in (chapter_output, nav_output, ncx_output):
                self.assertIn("第1章 《上·下》", output)
            self.assertIn("<p>等等，快走。</p>", chapter_output)
            self.assertIn("——卷四·羽觞醉月·终——", chapter_output)
            self.assertIn("——卷二·如梦·完——", chapter_output)
            self.assertNotIn("卷二··如梦", chapter_output)
            self.assertIn("text-align: center", chapter_output)
            self.assertIn("text-indent: 0", chapter_output)
            issue_kinds = {issue.kind for issue in report.issues}
            self.assertIn("possible_structural_marker", issue_kinds)
            self.assertNotIn("possible_intro_boilerplate", issue_kinds)

    def test_shared_writer_runs_normalizer_automatically(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            epub_path = Path(temp) / "writer.epub"
            output = io.StringIO()
            metadata = MetadataEnrichmentReport(
                path=epub_path,
                applied=True,
                status="unmatched",
            )
            with mock.patch(
                "src.core.epub_normalizer.enrich_epub_metadata",
                return_value=metadata,
            ) as enrich:
                with contextlib.redirect_stdout(output):
                    write_epub(
                        identifier="test",
                        title="测试A书",
                        author="测试作者",
                        volumes=[
                            Volume(
                                title="",
                                chapters=[
                                    Chapter(
                                        title="第1章 小P孩",
                                        paragraphs=["激 情,下一站."],
                                    )
                                ],
                            )
                        ],
                        out_path=epub_path,
                    )

            with zipfile.ZipFile(epub_path) as archive:
                chapter = next(
                    archive.read(name).decode()
                    for name in archive.namelist()
                    if name.endswith("chap_01_001.xhtml")
                )
            self.assertIn("第1章 小 P 孩", chapter)
            self.assertIn("激情，下一站。", chapter)
            self.assertIn("NORMALIZE", output.getvalue())
            self.assertIn("METADATA", output.getvalue())
            enrich.assert_called_once_with(epub_path, backup_dir=None)


    def test_shared_writer_runs_codex_review_after_punctuation_normalization(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            epub_path = Path(temp) / "reviewed.epub"
            response = json.dumps(
                {
                    "reviews": [
                        {
                            "issue_index": 0,
                            "verdict": "replace",
                            "confidence": "high",
                            "reason": "The dialogue has a locally missing closing quote.",
                            "old": "他说：“激情，下一站。",
                            "new": "他说：“激情，下一站。”",
                        }
                    ]
                },
                ensure_ascii=False,
            )
            metadata = MetadataEnrichmentReport(
                path=epub_path,
                applied=True,
                status="unmatched",
            )
            with mock.patch(
                "src.core.epub_normalizer.enrich_epub_metadata",
                return_value=metadata,
            ):
                with mock.patch(
                    "src.core.epub_normalizer._run_codex_review_prompt",
                    return_value=response,
                ) as review:
                    write_epub(
                        identifier="review-test",
                        title="复核测试",
                        author="测试作者",
                        volumes=[
                            Volume(
                                title="",
                                chapters=[
                                    Chapter(
                                        title="第1章 测试",
                                        paragraphs=["他说：“激 情, 下一站."],
                                    )
                                ],
                            )
                        ],
                        out_path=epub_path,
                    )

            self.assertEqual(review.call_count, 1)
            prompt = review.call_args.args[0]
            self.assertIn("他说：“激情，下一站。", prompt)
            self.assertNotIn("激 情, 下一站.", prompt)
            with zipfile.ZipFile(epub_path) as archive:
                chapter = archive.read("EPUB/chap_01_001.xhtml").decode()
            self.assertIn("他说：“激情，下一站。”", chapter)

            report_path = epub_path.with_suffix(".normalization.json")
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            report = payload["reports"][0]
            self.assertEqual(report["codex_review"]["status"], "completed")
            self.assertEqual(report["codex_review"]["pending"], 0)
            self.assertTrue(report["codex_review"]["decisions"][0]["applied"])

    def test_repairs_epub2_visible_toc_numbering_and_centered_headings(
        self,
    ) -> None:
        def chapter(title: str) -> str:
            return (
                '<?xml version="1.0" encoding="utf-8"?>'
                '<html xmlns="http://www.w3.org/1999/xhtml">'
                f"<head><title>{title}</title></head>"
                f'<body><h1 class="chapter">{title}</h1><p>正文。</p></body>'
                "</html>"
            )

        opf = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0">
  <manifest>
    <item id="prologue" href="text/prologue.html" media-type="application/xhtml+xml"/>
    <item id="chapter1" href="text/chapter1.html" media-type="application/xhtml+xml"/>
    <item id="chapter2" href="text/chapter2.html" media-type="application/xhtml+xml"/>
    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
  </manifest>
  <spine toc="ncx">
    <itemref idref="prologue"/>
    <itemref idref="chapter1"/>
    <itemref idref="chapter2"/>
  </spine>
</package>
"""
        ncx = """<?xml version="1.0" encoding="utf-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/">
  <navMap>
    <navPoint id="n1"><navLabel><text>序章</text></navLabel><content src="text/prologue.html"/></navPoint>
    <navPoint id="n2"><navLabel><text>鱼事</text></navLabel><content src="text/chapter1.html"/></navPoint>
    <navPoint id="n3"><navLabel><text>射天</text></navLabel><content src="text/chapter2.html"/></navPoint>
  </navMap>
</ncx>
"""
        with tempfile.TemporaryDirectory() as temp:
            epub_path = Path(temp) / "epub2.epub"
            with zipfile.ZipFile(epub_path, "w") as archive:
                archive.writestr(
                    "mimetype",
                    "application/epub+zip",
                    compress_type=zipfile.ZIP_STORED,
                )
                archive.writestr("content.opf", opf)
                archive.writestr("toc.ncx", ncx)
                archive.writestr("text/prologue.html", chapter("序章"))
                archive.writestr("text/chapter1.html", chapter("鱼事"))
                archive.writestr("text/chapter2.html", chapter("射天"))

            report = normalize_epub(epub_path)

            self.assertEqual(report.change_counts["visible_toc_page_added"], 1)
            self.assertEqual(report.change_counts["visible_toc_registered"], 3)
            self.assertEqual(report.change_counts["chapter_number_prefix_added"], 6)
            self.assertEqual(report.change_counts["chapter_heading_centered"], 3)
            self.assertFalse(report.issues)
            with zipfile.ZipFile(epub_path) as archive:
                self.assertEqual(archive.namelist()[0], "mimetype")
                self.assertEqual(
                    archive.getinfo("mimetype").compress_type,
                    zipfile.ZIP_STORED,
                )
                nav_output = archive.read("nav.xhtml").decode()
                ncx_output = archive.read("toc.ncx").decode()
                opf_output = archive.read("content.opf").decode()
                chapter_output = archive.read("text/chapter1.html").decode()
            self.assertIn("<title>目录</title>", nav_output)
            self.assertIn(">序章</a>", nav_output)
            self.assertIn(">第1章 鱼事</a>", nav_output)
            self.assertIn(">第2章 射天</a>", nav_output)
            self.assertIn("<text>第1章 鱼事</text>", ncx_output)
            self.assertIn('id="reader_toc" href="nav.xhtml"', opf_output)
            self.assertLess(
                opf_output.index('idref="reader_toc"'),
                opf_output.index('idref="prologue"'),
            )
            self.assertIn(
                '<reference type="toc" href="nav.xhtml" title="目录"/>',
                opf_output,
            )
            self.assertIn("<title>第1章 鱼事</title>", chapter_output)
            self.assertIn(">第1章 鱼事</h1>", chapter_output)
            self.assertIn("text-align: center", chapter_output)

            issues, count = validate_epub(epub_path)
            self.assertEqual(issues, [])
            self.assertEqual(count, 2)
            rerun = normalize_epub(epub_path)
            self.assertEqual(rerun.total_changes, 0)
            self.assertEqual(rerun.issues, [])

    @staticmethod
    def _write_fixture(
        path: Path,
        chapter: str,
        *,
        nav: str = NAV,
        ncx: str = NCX,
        intro: str | None = None,
    ) -> None:
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(
                "mimetype",
                "application/epub+zip",
                compress_type=zipfile.ZIP_STORED,
            )
            archive.writestr("EPUB/chap_01_001.xhtml", chapter)
            if intro is not None:
                archive.writestr("EPUB/intro.xhtml", intro)
            archive.writestr("EPUB/nav.xhtml", nav)
            archive.writestr("EPUB/toc.ncx", ncx)
            archive.writestr("EPUB/content.opf", OPF)


if __name__ == "__main__":
    unittest.main()
