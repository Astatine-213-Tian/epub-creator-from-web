#!/usr/bin/env python3
from __future__ import annotations

import unittest

from experiments.iteration4 import build_independent_candidate_selector as selector


class IndependentSelectorContractTest(unittest.TestCase):
    def test_blind_request_excludes_method_identity_and_machine_scores(self) -> None:
        request = selector.build_blind_request(
            sample_id="s_0123456789abcdef01234567",
            english=[{"id": "p0001", "en": "He laughed."}],
            style_definition={"dimensions": [{"id": "laughter_pivot"}]},
            eligible=[
                {
                    "candidate_id": "candidate_01",
                    "method_id": "hidden_method",
                    "target_margin": 99.0,
                    "paragraphs": [{"id": "p0001", "zh": "他笑了。"}],
                }
            ],
        )

        serialized = selector.canonical_json(request)
        self.assertNotIn("hidden_method", serialized)
        self.assertNotIn("method_id", serialized)
        self.assertNotIn("target_margin", serialized)
        self.assertIn("candidate_01", serialized)

    def test_selector_result_requires_complete_unique_ranking(self) -> None:
        valid = {
            "selected_candidate_id": "candidate_02",
            "rankings": [
                {
                    "candidate_id": "candidate_02",
                    "rank": 1,
                    "semantic_fidelity": 5,
                    "naturalness": 4,
                    "style_adherence": 4,
                    "hard_semantic_error": False,
                },
                {
                    "candidate_id": "candidate_01",
                    "rank": 2,
                    "semantic_fidelity": 4,
                    "naturalness": 4,
                    "style_adherence": 3,
                    "hard_semantic_error": False,
                },
            ],
        }
        candidate_ids = ["candidate_01", "candidate_02"]
        self.assertEqual(selector.validate_selector_result(valid, candidate_ids), [])

        invalid = {**valid, "selected_candidate_id": "candidate_01"}
        self.assertIn(
            "selected_candidate_not_rank_one",
            selector.validate_selector_result(invalid, candidate_ids),
        )

        selected_hard_error = {
            **valid,
            "rankings": [
                {**valid["rankings"][0], "hard_semantic_error": True},
                valid["rankings"][1],
            ],
        }
        self.assertIn(
            "selected_candidate_hard_semantic_error",
            selector.validate_selector_result(selected_hard_error, candidate_ids),
        )

    def test_hard_gate_rejects_surface_fact_change(self) -> None:
        neutral = [{"id": "p0001", "zh": "他等了12天，然后离开。"}]
        candidate = [{"id": "p0001", "zh": "他等了21天，然后离开。"}]

        result = selector.hard_gate(neutral, candidate, {"reference_examples": []})

        self.assertEqual(result["status"], "fail")
        self.assertIn("p0001:number_surface_mismatch", result["failures"])

    def test_hard_gate_scans_style_definition_example_text(self) -> None:
        neutral = [{"id": "p0001", "zh": "这是完全不同的中性表达内容。"}]
        candidate = [{"id": "p0001", "zh": "这是独特参考句子片段内容。"}]
        request = {
            "reference_examples": [
                {"target_style_text": "独特参考句子片段只用于测试。"}
            ]
        }

        result = selector.hard_gate(neutral, candidate, request)

        self.assertEqual(result["status"], "fail")
        self.assertIn("introduced_reference_8gram", result["failures"])


if __name__ == "__main__":
    unittest.main()
