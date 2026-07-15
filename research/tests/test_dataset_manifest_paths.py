from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from workflows.audit_style_dataset import book_record


class HistoricalDatasetManifestPathTests(unittest.TestCase):
    def test_moved_absolute_dataset_path_is_remapped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            research_root = Path(temporary) / "research"
            dataset_root = research_root / "datasets"
            txt_path = dataset_root / "raw/author/book.txt"
            txt_path.parent.mkdir(parents=True)
            txt_path.write_text(
                "Book\nAuthor: author\nChapter 1\nBody text.\n",
                encoding="utf-8",
            )

            record = book_record(
                {
                    "title": "book",
                    "author": "author",
                    "txt_path": "/obsolete/project/datasets/raw/author/book.txt",
                },
                dataset_root,
                research_root / "generated/texts",
            )

            self.assertTrue(record["exists"])
            self.assertEqual(Path(record["txt_path"]), txt_path)


if __name__ == "__main__":
    unittest.main()
