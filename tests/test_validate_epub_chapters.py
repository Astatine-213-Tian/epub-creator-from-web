from __future__ import annotations

import unittest
from io import BytesIO
from zipfile import ZipFile

from src.epub.validate import ChapterRef, find_nav_entries, validate_sequence


class ValidateEpubChaptersTests(unittest.TestCase):
    def _chapter(self, number: int, *, fanwai_before: int = 0) -> ChapterRef:
        return ChapterRef(
            source="nav",
            href=f"EPUB/chap_{number:03d}.xhtml",
            title=f"第{number}章",
            number_text=str(number),
            number=number,
            unit="章",
            fanwai_before=fanwai_before,
        )

    def test_fanwai_entries_can_fill_an_intentional_number_gap(self) -> None:
        entries = [
            self._chapter(24),
            self._chapter(26, fanwai_before=1),
        ]

        self.assertEqual(validate_sequence(entries), [])

    def test_number_gap_without_enough_fanwai_entries_is_reported(self) -> None:
        entries = [
            self._chapter(24),
            self._chapter(27, fanwai_before=1),
        ]

        self.assertEqual(len(validate_sequence(entries)), 1)

    def test_renamed_inline_extra_uses_source_marker_without_masking_missing_chapters(self) -> None:
        for marker, next_number, expected_issues in [
            ('<meta name="notion-source" content="https://www.notion.so/story"/>', 3, 0),
            ('<meta name="notion-source" content="https://www.notion.so/story"/>', 4, 1),
            ('<meta name="notion-source" content=""/>', 3, 1),
            ("", 3, 1),
        ]:
            with self.subTest(marker=marker, next_number=next_number):
                buffer = BytesIO()
                with ZipFile(buffer, "w") as archive:
                    archive.writestr("EPUB/nav.xhtml", f'''<html xmlns="http://www.w3.org/1999/xhtml">
<body><nav><ol><li><a href="one.xhtml">第1章 起点</a></li>
<li><a href="extra.xhtml">光与暗的童年</a></li>
<li><a href="next.xhtml">第{next_number}章 后续</a></li></ol></nav></body></html>''')
                    archive.writestr("EPUB/extra.xhtml", f'''<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>光与暗的童年</title>{marker}</head><body><h2>光与暗的童年</h2></body></html>''')
                with ZipFile(buffer) as archive:
                    entries = find_nav_entries(archive)
                self.assertEqual(len(validate_sequence(entries)), expected_issues)


if __name__ == "__main__":
    unittest.main()
