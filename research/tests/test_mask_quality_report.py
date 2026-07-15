from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from workflows.mask_quality_report import sample_flags, view_stats


class MaskQualityReportTests(unittest.TestCase):
    def test_flags_selected_view_when_global_term_survives(self) -> None:
        sample = {
            "clean": {"text": "甲里转身。"},
            "train_global_masked": {"text": "甲里转身。"},
            "entity_masked_v3": {"text": "某某转身。"},
            "topic_distorted": {"text": "文文文文。"},
            "structure_only": {"text": "文文文文。"},
        }

        flags = sample_flags(sample, {"entity_terms_v2": ["甲里"]}, ["甲里"])

        self.assertTrue(
            any("train-global mask terms still visible" in flag for flag in flags)
        )

    def test_selected_density_is_grouped_by_split_author_and_book(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "chunks.jsonl"
            rows = [
                {
                    "chunk_id": "a",
                    "author": "甲",
                    "title": "甲书",
                    "split": "train",
                    "text": "某某正文。",
                    "chunk_clean_cjk_count": 4,
                    "chunk_view_cjk_count": 4,
                },
                {
                    "chunk_id": "b",
                    "author": "乙",
                    "title": "乙书",
                    "split": "test",
                    "text": "某正文文。",
                    "chunk_clean_cjk_count": 4,
                    "chunk_view_cjk_count": 4,
                },
            ]
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )

            stats = view_stats(
                "train_global_masked",
                path,
                clean_natural_mou={"a": 1, "b": 0},
            )

        self.assertEqual(
            stats["mask_density_by_split"]["train"]["mask_chars_per_1k_cjk"],
            250.0,
        )
        self.assertEqual(
            stats["mask_density_by_author"]["乙"]["mask_chars"], 1
        )
        self.assertIn("甲 / 甲书", stats["mask_density_by_book"])


if __name__ == "__main__":
    unittest.main()
