from __future__ import annotations

import copy
import unittest

from experiments.iteration5.decision import analyze_transfer_lora_reference_benchmark_v2 as analyze
from experiments.iteration5.decision import prepare_transfer_lora_reference_benchmark_v2 as prepare
from experiments.iteration5.decision import run_transfer_lora_blind_ratings_v1 as ratings


class ReferenceAnchoredBenchmarkV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.blocks, cls.eligibility, cls.lineage = prepare.build_blocks()
        cls.controls = prepare.build_calibration_blocks(cls.blocks)

    def test_geometry_and_clean_lora_references(self) -> None:
        self.assertEqual(len(self.blocks), 31)
        self.assertEqual(sum(row["arm"] == "prompt_development" for row in self.blocks), 22)
        self.assertEqual(sum(row["arm"] == "lora_internal_test" for row in self.blocks), 9)
        self.assertEqual(len(self.eligibility), 181)
        self.assertEqual(len(self.lineage), 31)
        lora = [row for row in self.blocks if row["arm"] == "lora_internal_test"]
        self.assertEqual({row["book"] for row in lora}, {"星盘重启", "金牌助理", "鹰奴"})
        self.assertTrue(lora[0]["style_reference"][0]["zh"].startswith("“这是父神的知识库。”"))
        self.assertTrue(lora[3]["style_reference"][0]["zh"].startswith("“老子不干了！”卢舟"))
        self.assertTrue(lora[6]["style_reference"][0]["zh"].startswith("李效：“抬头回话。”"))
        self.assertNotEqual(lora[0]["style_reference"], lora[0]["masked_structure_control"])
        self.assertTrue(all("hidden_original" not in row["candidates"] for row in self.blocks))

    def test_calibration_preserves_lexical_inventory_but_disrupts_order(self) -> None:
        self.assertEqual(len(self.controls), 4)
        for block in self.controls:
            positive = block["candidates"]["calibration_positive"]
            decoy = block["candidates"]["calibration_decoy"]
            positive_text = "".join(row["zh"] for row in positive)
            decoy_text = "".join(row["zh"] for row in decoy)
            self.assertNotEqual(positive_text, decoy_text)
            self.assertEqual(sorted(positive_text), sorted(decoy_text))
            self.assertEqual(
                [row["id"] for row in positive],
                [row["id"] for row in decoy],
            )

    def test_v2_result_validator_requires_all_dimension_scores(self) -> None:
        block = self.blocks[0]
        packet, _ = prepare.packet(
            block=block,
            task="style",
            identity=prepare.STYLE_IDENTITIES[0],
            methods=sorted(block["candidates"]),
        )
        valid = {
            "packet_id": packet["packet_id"],
            "judgments": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "dimension_scores": {
                        dimension: 3 for dimension in prepare.RUBRIC_DIMENSIONS
                    },
                    "style_adherence": 3,
                    "naturalness": 4,
                    "rationale": "结构维度可观察。",
                }
                for candidate in packet["candidates"]
            ],
        }
        self.assertEqual(ratings.validate_result(valid, packet, "style"), [])
        invalid = copy.deepcopy(valid)
        del invalid["judgments"][0]["dimension_scores"][prepare.RUBRIC_DIMENSIONS[0]]
        self.assertTrue(ratings.validate_result(invalid, packet, "style"))

    def test_calibration_gate_is_identity_level_and_binding(self) -> None:
        records = []
        for identity in prepare.STYLE_IDENTITIES:
            for index in range(4):
                records.extend(
                    [
                        {
                            "is_calibration": True,
                            "task": "style",
                            "identity": identity,
                            "block_id": f"calibration.c{index + 1:02d}",
                            "method_id": "calibration_positive",
                            "style_adherence": 4,
                        },
                        {
                            "is_calibration": True,
                            "task": "style",
                            "identity": identity,
                            "block_id": f"calibration.c{index + 1:02d}",
                            "method_id": "calibration_decoy",
                            "style_adherence": 2,
                        },
                    ]
                )
        summary, passed = analyze.calibration_summary(records)
        self.assertTrue(passed)
        self.assertTrue(all(row["passed"] for row in summary))
        records[-2]["style_adherence"] = 1
        records[-4]["style_adherence"] = 1
        summary, passed = analyze.calibration_summary(records)
        self.assertFalse(passed)
        self.assertFalse(summary[-1]["passed"])

    def test_semantic_sensitivity_requires_equal_denominators(self) -> None:
        row = analyze.semantic_sensitivity_row(
            arm="prompt_development",
            method="neutral_only",
            v2_values=[True, False],
            v1_values=[False, False],
        )
        self.assertTrue(row["denominators_match"])
        self.assertEqual(row["v1_sources"], 2)
        self.assertEqual(row["v2_sources"], 2)
        with self.assertRaises(ValueError):
            analyze.semantic_sensitivity_row(
                arm="prompt_development",
                method="neutral_only",
                v2_values=[True, False],
                v1_values=[False],
            )


if __name__ == "__main__":
    unittest.main()
