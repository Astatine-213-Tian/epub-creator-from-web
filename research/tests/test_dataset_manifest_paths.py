from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from workflows.audit_style_dataset import book_record, normalize_manifest_paths


class HistoricalDatasetManifestPathTests(unittest.TestCase):
    def test_manifest_paths_are_rewritten_relative_to_manifest_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            research_root = Path(temporary) / "research"
            dataset_root = research_root / "datasets"
            txt_path = dataset_root / "raw/author/book.txt"
            epub_path = research_root.parent / "books/author/book.epub"
            txt_path.parent.mkdir(parents=True)
            epub_path.parent.mkdir(parents=True)
            txt_path.write_text("正文", encoding="utf-8")
            epub_path.write_bytes(b"epub")
            manifest = [
                {
                    "txt_path": str(txt_path),
                    "source_epub": str(epub_path),
                }
            ]

            changed = normalize_manifest_paths(manifest, dataset_root)

            self.assertEqual(changed, 2)
            self.assertEqual(manifest[0]["txt_path"], "raw/author/book.txt")
            self.assertEqual(manifest[0]["source_epub"], "../../books/author/book.epub")

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
            self.assertEqual(Path(record["txt_path"]).resolve(), txt_path.resolve())


if __name__ == "__main__":
    unittest.main()
