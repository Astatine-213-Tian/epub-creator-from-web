from __future__ import annotations

import unittest

from lxml import etree as ET

from src.content.xhtml import read_xhtml
from src.epub.xhtml import render_blocks
from notion_books import from_markdown, text_blocks, to_markdown


class ContentTests(unittest.TestCase):
    def test_h3_size_is_fixed_when_alignment_is_added_or_removed(self):
        for alignment in (None, "left", "center"):
            blocks = from_markdown("### 分标题\n**普通加粗段落**")
            blocks[0]["attributes"] = {"style": "font-size: 2em; margin-top: 1em;"}
            if alignment:
                blocks[0]["alignment"] = alignment
            root = ET.Element("body")
            render_blocks(root, blocks)
            style = root[0].get("style")
            self.assertIn("font-size: 1.1em;", style)
            self.assertNotIn("2em", style)
            self.assertIn("margin-top: 1em", style)
            self.assertEqual("text-align: center" in style, alignment == "center")
            self.assertEqual(ET.QName(root[0]).localname, "h3")
            self.assertIsNone(root[1].get("style"))

    def test_rich_text_paragraphs_and_literal_punctuation_survive(self):
        xml = """<html xmlns="http://www.w3.org/1999/xhtml"><head/><body>
        <h2>第1章 开始</h2><p>“测试，<strong>保留粗体</strong>……”</p>
        <p>* * * 是正文</p><p>1. 不是列表</p><p>换行<br/>下一行</p>
        <p></p><h3>（一）</h3><p><em>斜体</em>与普通文字</p>
        </body></html>""".encode()
        before = read_xhtml(xml, title="第1章 开始")
        after = from_markdown(to_markdown(before))
        self.assertEqual(text_blocks(before), text_blocks(after))
        self.assertEqual(after[0]["runs"][1]["styles"], ["bold"])
        self.assertEqual(after[-1]["runs"][0]["styles"], ["italic"])

    def test_unknown_source_blocks_stop_instead_of_dropping_text(self):
        with self.assertRaises(ValueError):
            read_xhtml(
                b'<html xmlns="http://www.w3.org/1999/xhtml"><body><h2>Title</h2><table/></body></html>',
                title="Title",
            )
        with self.assertRaises(ValueError):
            from_markdown("![Unconverted image](https://example.com/a.png)")

    def test_literal_rule_text_is_escaped_instead_of_becoming_a_notion_divider(self):
        blocks = [
            {
                "kind": "paragraph",
                "runs": [{"text": "-----", "styles": []}],
                "alignment": "center",
            }
        ]
        markdown = to_markdown(blocks)
        self.assertIn(r"\-\-\-\-\-", markdown)
        self.assertEqual(from_markdown(markdown), blocks)

    def test_center_is_a_paragraph_property_not_a_text_pattern(self):
        xml = """<html xmlns="http://www.w3.org/1999/xhtml"><body><h2>标题</h2>
        <p style="text-align: center; text-indent: 0;">任意一段普通文字</p>
        <p>——本卷完——</p>
        <p style="text-align: center; text-indent: 0;"><strong>（一）</strong></p>
        </body></html>""".encode()
        original = read_xhtml(xml, title="标题")
        blocks = from_markdown(to_markdown(original))
        self.assertEqual(blocks, original)
        self.assertEqual(blocks[0]["alignment"], "center")
        self.assertNotIn("alignment", blocks[1])
        self.assertEqual(blocks[2]["kind"], "paragraph")
        root = ET.Element("body")
        render_blocks(root, blocks)
        self.assertEqual(root[0].get("style"), "text-align: center; text-indent: 0;")
        self.assertIsNone(root[1].get("style"))
        self.assertEqual(ET.QName(root[2]).localname, "p")
        self.assertEqual(ET.QName(root[2][0]).localname, "strong")

    def test_column_wrapper_can_hold_multiple_blocks_but_not_discard_side_content(self):
        markdown = "<columns>\n\t<column>\n\t\t<empty-block/>\n\t</column>\n\t<column>\n\t\t第一段\n\t\t**第二段**\n\t</column>\n\t<column>\n\t</column>\n</columns>"
        blocks = from_markdown(markdown)
        self.assertEqual(text_blocks(blocks), ["第一段", "第二段"])
        self.assertTrue(all(b["alignment"] == "center" for b in blocks))
        for invalid in [
            markdown.replace("<empty-block/>", "左列也有正文"),
            markdown.replace("</columns>", ""),
            markdown.replace("\t<column>\n\t</column>\n", ""),
        ]:
            with self.assertRaises(ValueError):
                from_markdown(invalid)
