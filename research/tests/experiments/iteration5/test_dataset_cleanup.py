from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from workflows.audit_style_dataset import (
    filter_body_lines,
    prune_stale_cleaned_texts,
    text_flags,
)


class DatasetCleanupTests(unittest.TestCase):
    def test_obfuscated_inline_ads_are_removed_without_dropping_prose(self) -> None:
        examples = {
            "前文➯(小说)_[()]➯€来➯小说➯？♧？♧➯()•(ｃｏm)后文": "前文后文",
            "前文✾([()])☘来✾♀♡小说✾♀♡♀♡✾()•(ｃｏｍ)后文": "前文后文",
            "前文ℂ[()]ℂ✔来ℂ笔&❂ℂ&❂&❂ℂ()•(ｃｏｍ)后文": "前文后文",
        }
        for source, expected in examples.items():
            with self.subTest(source=source):
                self.assertEqual(filter_body_lines([source]), [expected])

    def test_split_and_nested_symbol_domain_ads_are_removed(self) -> None:
        examples = {
            "你敢不敢那是你的本事。(ｍｉｄｕｘｓ)(ｃｏｍ)": "你敢不敢那是你的本事。",
            "我还得去复命。(ｍｉｄｕｘｓ)(ｃo)": "我还得去复命。",
            "行行行，送送送。ℂ()ℂ✓来ℂ笔?╬ℂ?╬?╬ℂ()•(ｃｏｍ)": "行行行，送送送。",
        }
        for source, expected in examples.items():
            with self.subTest(source=source):
                self.assertEqual(filter_body_lines([source]), [expected])

    def test_systemic_replacement_artifacts_are_fatal_quality_flags(self) -> None:
        text = "\n".join(f"正文{i}€€" for i in range(10))
        flags = text_flags(text, text)
        self.assertIn("clean_replacement_artifacts", flags)
        self.assertIn("systemic_encoding_corruption", flags)

    def test_single_currency_symbol_is_not_treated_as_systemic_corruption(self) -> None:
        flags = text_flags("价格为€5。", "价格为€5。")
        self.assertNotIn("clean_replacement_artifacts", flags)
        self.assertNotIn("systemic_encoding_corruption", flags)

    def test_censorship_boxes_are_not_treated_as_encoding_corruption(self) -> None:
        text = "这个词被□□处理。" * 20
        flags = text_flags(text, text)
        self.assertNotIn("clean_replacement_artifacts", flags)
        self.assertNotIn("systemic_encoding_corruption", flags)

    def test_unicode_decoder_replacement_char_is_removed_and_audited(self) -> None:
        source = "时�安要�求这枚戒指。"
        self.assertEqual(filter_body_lines([source]), ["时安要求这枚戒指。"])
        self.assertIn("raw_unicode_replacement_char", text_flags(source, "时安要求这枚戒指。"))

    def test_stale_generated_cleaned_text_is_pruned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kept = root / "作者/保留.clean.txt"
            stale = root / "作者/过期.clean.txt"
            kept.parent.mkdir(parents=True)
            kept.write_text("保留", encoding="utf-8")
            stale.write_text("过期", encoding="utf-8")
            removed = prune_stale_cleaned_texts(
                root,
                [{"exists": True, "clean_txt_path": str(kept)}],
            )
            self.assertEqual(removed, [str(stale)])
            self.assertTrue(kept.exists())
            self.assertFalse(stale.exists())


if __name__ == "__main__":
    unittest.main()
