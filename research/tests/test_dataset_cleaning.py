from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from workflows.audit_style_dataset import (
    CROSS_BOOK_DECONTAMINATION_VERSION,
    MASKING_POLICY_VERSION,
    cleaned_text,
    compile_term_matcher,
    decontaminate_cross_book_passages,
    mask_terms,
    normalize_punctuation,
    select_mask_terms,
)


class DatasetPunctuationNormalizationTests(unittest.TestCase):
    def test_canonicalizes_typographic_variants_without_changing_numeric_separators(self) -> None:
        source = '他说,"好...really?" 价格3.14, 时间12:30; (注) 「再见」--'

        self.assertEqual(
            normalize_punctuation(source),
            "他说，“好……really？” 价格3.14， 时间12:30； （注） “再见”——",
        )

    def test_normalization_is_idempotent(self) -> None:
        normalized = "他说：“好……” 时间12:30；（注）"

        self.assertEqual(normalize_punctuation(normalized), normalized)

    def test_cleaned_text_applies_the_shared_policy(self) -> None:
        cleaned, metadata = cleaned_text(
            "书名：样书\n作者：甲\n第一章\n他说,‘走吧!’\n",
            title="样书",
        )

        self.assertEqual(cleaned, "他说，‘走吧！’\n")
        self.assertEqual(metadata["punctuation_normalization"], "canonical_zh_v1")
        self.assertTrue(metadata["punctuation_normalized"])


class DatasetLeakageControlTests(unittest.TestCase):
    def test_multi_term_matcher_uses_exact_leftmost_longest_replacement(self) -> None:
        matcher = compile_term_matcher(["甲乙", "甲乙丙", "乙丙"])

        self.assertEqual(
            mask_terms("甲乙丙甲乙", matcher, placeholder="某", preserve_length=True),
            "某某某某某",
        )
        self.assertEqual(
            mask_terms("甲乙丙甲乙", matcher, placeholder="<TERM>"),
            "<TERM><TERM>",
        )

    def test_cross_book_passages_are_removed_only_from_clean_copies(self) -> None:
        repeated = "重复正文" * 30
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            raw_snapshots: dict[Path, str] = {}
            for index, unique in enumerate(("甲独有", "乙独有"), start=1):
                raw_path = root / f"raw-{index}.txt"
                clean_path = root / f"clean-{index}.txt"
                text = f"{unique}\n{repeated}\n"
                raw_path.write_text(text, encoding="utf-8")
                clean_path.write_text(text, encoding="utf-8")
                raw_snapshots[raw_path] = text
                records.append(
                    {
                        "author": "作者",
                        "title": f"书{index}",
                        "exists": True,
                        "txt_path": str(raw_path),
                        "clean_txt_path": str(clean_path),
                        "author_header_conflicts": [],
                    }
                )

            report = decontaminate_cross_book_passages(records)

            self.assertEqual(report["policy"], CROSS_BOOK_DECONTAMINATION_VERSION)
            self.assertGreater(report["removed_line_count"], 0)
            self.assertEqual(report["remaining_fingerprint_count"], 0)
            for record in records:
                clean = Path(record["clean_txt_path"]).read_text(encoding="utf-8")
                self.assertNotIn(repeated, clean)
                raw_path = Path(record["txt_path"])
                self.assertEqual(raw_path.read_text(encoding="utf-8"), raw_snapshots[raw_path])

    def test_global_mask_vocabulary_is_fit_on_train_books_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_path = root / "train.txt"
            heldout_path = root / "heldout.txt"
            train_path.write_text("青龙白虎" * 20, encoding="utf-8")
            heldout_path.write_text("测试词组" * 20, encoding="utf-8")
            records = [
                {
                    "author": "训练作者",
                    "title": "训练书",
                    "exists": True,
                    "clean_txt_path": str(train_path),
                },
                {
                    "author": "测试作者",
                    "title": "测试书",
                    "exists": True,
                    "clean_txt_path": str(heldout_path),
                },
            ]
            splits = {
                "train": [{"author": "训练作者", "title": "训练书"}],
                "dev": [],
                "test": [{"author": "测试作者", "title": "测试书"}],
                "proxy_transfer": [],
                "excluded": [],
            }

            first = select_mask_terms(records, splits)
            heldout_path.write_text("完全不同" * 30, encoding="utf-8")
            records[1]["author"] = "另一个测试标签"
            splits["test"][0]["author"] = "另一个测试标签"
            second = select_mask_terms(records, splits)

            self.assertEqual(first["masking_policy"], MASKING_POLICY_VERSION)
            self.assertEqual(first["global_terms"], second["global_terms"])
            self.assertFalse(first["provenance"]["transform_uses_author_label"])
            self.assertFalse(
                first["provenance"]["held_out_corpus_statistics_used_for_global_terms"]
            )


if __name__ == "__main__":
    unittest.main()
