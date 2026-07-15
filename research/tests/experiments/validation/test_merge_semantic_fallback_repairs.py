from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.crawl.snapshot import write_json
from experiments.validation.application.merge_semantic_fallback_repairs import merge_fallback_repairs


class MergeSemanticFallbackRepairsTest(unittest.TestCase):
    def test_only_failed_primary_paragraphs_are_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            primary = root / "primary"
            fallback = root / "fallback"
            write_json(
                primary / "outputs/chunk.json",
                {
                    "translations": [
                        {"index": 1, "zh": "保留原译。"},
                        {"index": 2, "zh": ""},
                    ]
                },
            )
            write_json(
                fallback / "outputs/chunk.json",
                {
                    "translations": [
                        {"index": 1, "zh": "不得覆盖。"},
                        {"index": 2, "zh": "补充译文。"},
                    ]
                },
            )
            summary = merge_fallback_repairs(
                primary_run_dir=primary,
                fallback_run_dir=fallback,
                fallback_model="test-model",
            )
            merged = json.loads(
                (primary / "outputs/chunk.json").read_text(encoding="utf-8")
            )
            self.assertEqual(merged["translations"][0]["zh"], "保留原译。")
            self.assertEqual(merged["translations"][1]["zh"], "补充译文。")
            self.assertEqual(summary["repaired_paragraph_count"], 1)
            self.assertEqual(summary["remaining_failed_paragraph_count"], 0)


if __name__ == "__main__":
    unittest.main()
