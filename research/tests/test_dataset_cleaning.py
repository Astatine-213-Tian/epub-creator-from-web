from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from workflows.audit_style_dataset import (
    CROSS_BOOK_DECONTAMINATION_VERSION,
    MASKING_POLICY_VERSION,
    canonicalize_nested_mask_terms,
    cleaned_text,
    compile_term_matcher,
    count_selected_terms,
    decontaminate_cross_book_passages,
    is_concentration_rescue_candidate,
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

    def test_cleaned_text_removes_reader_mode_boilerplate(self) -> None:
        cleaned, _metadata = cleaned_text(
            "\n".join(
                [
                    "第一章",
                    "正文保留。",
                    "这个段落是图片段落，请访问正确的网站且关闭广告拦截功能并且退出浏览器阅读模式",
                    "原版未篡改内容请移至 #官！网。如已在，请，关闭广告拦截功能并且退出浏览器阅读模式",
                    "关闭广告公司后，他回到家。",
                ]
            ),
            title="样书",
        )

        self.assertEqual(cleaned, "正文保留。\n关闭广告公司后，他回到家。\n")


class DatasetLeakageControlTests(unittest.TestCase):
    def test_selected_term_recount_includes_ineligible_subspans(self) -> None:
        counts = count_selected_terms(
            "甲里的肩，甲里的手。",
            {"甲里", "里的", "的肩", "的手"},
        )

        self.assertEqual(counts, {"甲里": 2, "里的": 2, "的肩": 1, "的手": 1})

    def test_concentration_rescue_is_limited_to_short_entity_candidates(self) -> None:
        self.assertTrue(is_concentration_rescue_candidate("甲里"))
        self.assertTrue(is_concentration_rescue_candidate("甲里转"))
        self.assertFalse(is_concentration_rescue_candidate("甲里转身"))

    def test_nested_term_canonicalization_preserves_actions_and_full_names(self) -> None:
        terms, diagnostics = canonicalize_nested_mask_terms(
            {
                "甲里",
                "甲里转",
                "甲里转身",
                "夏习",
                "夏习清",
                "乙里转",
            },
            {
                "甲里": 1000,
                "甲里转": 120,
                "甲里转身": 80,
                "夏习": 1000,
                "夏习清": 990,
                "乙里": 1000,
                "乙里转": 100,
            },
        )

        self.assertIn("甲里", terms)
        self.assertNotIn("甲里转", terms)
        self.assertNotIn("甲里转身", terms)
        self.assertIn("夏习清", terms)
        self.assertNotIn("夏习", terms)
        self.assertNotIn("乙里转", terms)
        self.assertEqual(diagnostics["output_term_count"], 2)

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

    def test_concentration_rescue_recovers_short_names_with_common_characters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            split_rows = []

            entity_path = root / "entity-book.txt"
            entity_path.write_text(
                "甲里转身。甲里开口。甲里点头。甲里离开。" * 10,
                encoding="utf-8",
            )
            records.append(
                {
                    "author": "甲作者",
                    "title": "实体书",
                    "exists": True,
                    "clean_txt_path": str(entity_path),
                }
            )
            split_rows.append({"author": "甲作者", "title": "实体书"})

            concentrated_path = root / "concentrated-book.txt"
            concentrated_path.write_text("丙里转身。" * 25, encoding="utf-8")
            records.append(
                {
                    "author": "集中作者",
                    "title": "集中书",
                    "exists": True,
                    "clean_txt_path": str(concentrated_path),
                }
            )
            split_rows.append({"author": "集中作者", "title": "集中书"})

            collision_path = root / "collision-name-book.txt"
            collision_path.write_text(
                "丁里转身。丁里开口。丁里点头。丁里离开。" * 25,
                encoding="utf-8",
            )
            records.append(
                {
                    "author": "碰撞作者",
                    "title": "碰撞名书",
                    "exists": True,
                    "clean_txt_path": str(collision_path),
                }
            )
            split_rows.append({"author": "碰撞作者", "title": "碰撞名书"})

            fragment_path = root / "single-character-fragment-book.txt"
            fragment_path.write_text(
                (
                    "戊不转身。戊很开心。戊点点头。戊没说话。" * 40
                )
                + ("不是这样。" * 200),
                encoding="utf-8",
            )
            records.append(
                {
                    "author": "片段作者",
                    "title": "片段书",
                    "exists": True,
                    "clean_txt_path": str(fragment_path),
                }
            )
            split_rows.append({"author": "片段作者", "title": "片段书"})

            for index in range(5):
                path = root / f"common-{index}.txt"
                path.write_text(
                    ("小乙转身。" * 30)
                    + ("丙里转身。" * 19)
                    + "丁里转身。",
                    encoding="utf-8",
                )
                author = f"对照作者{index}"
                title = f"对照书{index}"
                records.append(
                    {
                        "author": author,
                        "title": title,
                        "exists": True,
                        "clean_txt_path": str(path),
                    }
                )
                split_rows.append({"author": author, "title": title})

            plan = select_mask_terms(
                records,
                {
                    "train": split_rows,
                    "dev": [],
                    "test": [],
                    "proxy_transfer": [],
                    "excluded": [],
                },
            )

            rescue_terms = plan["global_terms"]["concentration_rescue_terms"]
            self.assertIn("甲里", rescue_terms)
            self.assertIn("丁里", rescue_terms)
            self.assertIn("甲里", plan["global_terms"]["entity_terms_v2"])
            self.assertNotIn("小乙", rescue_terms)
            self.assertNotIn("丙里", rescue_terms)
            self.assertNotIn("戊不", rescue_terms)
            matcher = compile_term_matcher(
                plan["global_terms"]["entity_terms_v2"]
            )
            masked = mask_terms(
                "甲里转身。",
                matcher,
                placeholder="某",
                preserve_length=True,
            )
            self.assertEqual(masked, "某某转身。")
            self.assertTrue(
                plan["provenance"]["concentration_rescue_counts_all_fit_occurrences"]
            )


if __name__ == "__main__":
    unittest.main()
