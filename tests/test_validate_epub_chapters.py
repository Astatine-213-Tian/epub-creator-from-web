from __future__ import annotations

import unittest

from src.cli.validate_epub_chapters import ChapterRef, validate_sequence


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


if __name__ == "__main__":
    unittest.main()
