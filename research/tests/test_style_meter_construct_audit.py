from __future__ import annotations

import unittest

import numpy as np
from scipy.sparse import csr_matrix

from experiments.iteration1.audit_style_meter_construct import (
    build_interpretation_groups,
    interpretation_group_key,
    is_cjk_core_with_punctuation,
    is_punctuation_only,
    is_punctuation_or_symbol_only,
    zero_columns_and_renormalize,
)


class StyleMeterConstructAuditTests(unittest.TestCase):
    def test_groups_overlapping_speech_tag_punctuation_variants(self) -> None:
        rows = [
            {
                "rank": 1,
                "feature": "，说：",
                "coefficient": 1.9,
                "category": "dialogue_structure",
            },
            {
                "rank": 2,
                "feature": "说：“",
                "coefficient": 1.8,
                "category": "dialogue_structure",
            },
            {
                "rank": 3,
                "feature": "说。",
                "coefficient": 1.4,
                "category": "dialogue_structure",
            },
        ]

        groups = build_interpretation_groups(rows)

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["group_key"], "cjk_core:说")
        self.assertEqual(groups[0]["label"], "说")
        self.assertEqual(groups[0]["member_count"], 3)
        self.assertEqual(groups[0]["strongest_feature"], "，说：")

    def test_groups_pure_punctuation_separately(self) -> None:
        self.assertEqual(interpretation_group_key("。“"), "pure_punctuation")
        self.assertEqual(interpretation_group_key("……”"), "pure_punctuation")
        self.assertEqual(interpretation_group_key("＋＋"), "source_format_symbols")
        self.assertTrue(is_punctuation_or_symbol_only("？！"))
        self.assertTrue(is_punctuation_only("？！"))
        self.assertFalse(is_punctuation_only("＋＋"))
        self.assertFalse(is_punctuation_or_symbol_only("AI"))
        self.assertFalse(is_punctuation_or_symbol_only("说："))

    def test_excludes_mask_and_latin_fragments_from_shuo_punctuation_family(self) -> None:
        self.assertTrue(is_cjk_core_with_punctuation("，说：“", "说"))
        self.assertFalse(is_cjk_core_with_punctuation(">说", "说"))
        self.assertFalse(is_cjk_core_with_punctuation("IN>说", "说"))
        self.assertNotEqual(
            interpretation_group_key(">说"), interpretation_group_key("说：")
        )

    def test_feature_ablation_renormalizes_remaining_columns(self) -> None:
        matrix = csr_matrix([[0.6, 0.8]], dtype=np.float32)

        ablated = zero_columns_and_renormalize(matrix, np.asarray([0]))

        np.testing.assert_allclose(ablated.toarray(), [[0.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
