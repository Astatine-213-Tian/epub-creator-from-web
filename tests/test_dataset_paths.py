from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.cli.dataset import manifest_path


class DatasetManifestPathTests(unittest.TestCase):
    def test_paths_are_relative_to_dataset_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            production_root = Path(temporary)
            dataset_root = production_root / "research/datasets"
            txt_path = dataset_root / "raw/author/book.txt"
            epub_path = production_root / "books/author/book.epub"
            txt_path.parent.mkdir(parents=True)
            epub_path.parent.mkdir(parents=True)
            txt_path.write_text("text", encoding="utf-8")
            epub_path.write_bytes(b"epub")

            self.assertEqual(
                manifest_path(txt_path, output_root=dataset_root),
                "raw/author/book.txt",
            )
            self.assertEqual(
                manifest_path(epub_path, output_root=dataset_root),
                "../../books/author/book.epub",
            )


if __name__ == "__main__":
    unittest.main()
