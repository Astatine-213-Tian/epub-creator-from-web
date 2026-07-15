from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.validation.meter.validate_dataset_source import analyze_source


class DatasetSourceValidationTests(unittest.TestCase):
    def write_source(self, text: str) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "source.txt"
        path.write_text(text, encoding="utf-8")
        return path

    def test_complete_clean_source_passes(self) -> None:
        body = "正文内容" * 40
        path = self.write_source(f"书名\n作者：作者\n\n第1章 一\n{body}\n第2章 二\n{body}\n")
        result = analyze_source(
            path,
            expected_author="作者",
            accepted_titles={"书名"},
            expected_chapters=2,
            min_cjk=100,
        )
        self.assertEqual(result["status"], "pass")

    def test_malformed_source_fails(self) -> None:
        path = self.write_source("书名\n作者：作者\n\n第1章 一\n正文,?€€�[()]来小说(ｃｏｍ)\n")
        result = analyze_source(
            path,
            expected_author="作者",
            accepted_titles={"书名"},
            expected_chapters=1,
            min_cjk=1,
        )
        self.assertEqual(result["status"], "fail")
        self.assertGreaterEqual(len(result["fatal_reasons"]), 4)


if __name__ == "__main__":
    unittest.main()
