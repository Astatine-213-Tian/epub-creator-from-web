from __future__ import annotations

import unittest

from experiments.iteration5.application import apply_content_plan_combined_to_translation_run as app
from src.translation import codex_cli


class Method4ApplicationTest(unittest.TestCase):
    def test_stable_sample_id_is_deterministic_and_input_bound(self) -> None:
        first = app.stable_sample_id("09", [0], ["A source."], ["一段译文。"])
        second = app.stable_sample_id("09", [0], ["A source."], ["一段译文。"])
        changed = app.stable_sample_id("09", [1], ["A source."], ["一段译文。"])
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)
        self.assertRegex(first, r"^s_[0-9a-f]{24}$")

    def test_request_uses_frozen_combined_contract(self) -> None:
        request = app.request_for_rows(
            experiment_root=app.DEFAULT_EXPERIMENT_ROOT,
            chapter_id="09",
            rows=[
                {
                    "index": 0,
                    "english": "He opened the door.",
                    "neutral_zh": "他打开了门。",
                }
            ],
        )
        self.assertEqual(request["method_id"], app.METHOD_ID)
        self.assertEqual(request["intensity"], app.INTENSITY)
        self.assertIn("content_plan_contract", request["method_payload"])
        self.assertTrue(request["reference_examples"])

    def test_application_validation_rejects_misaligned_plan(self) -> None:
        request = {
            "sample_id": "s_0123456789abcdef01234567",
            "method_id": app.METHOD_ID,
            "intensity": app.INTENSITY,
            "english_semantic_source": [{"id": "p0001", "en": "He left."}],
            "neutral_zh": [{"id": "p0001", "zh": "他离开了。"}],
            "method_payload": {},
            "reference_examples": [],
        }
        result = {
            "sample_id": request["sample_id"],
            "method_id": app.METHOD_ID,
            "intensity": app.INTENSITY,
            "content_plan": [{"id": "p0002", "facts": [], "constraints": []}],
            "paragraphs": [{"id": "p0001", "zh": "他走了。"}],
            "style_cues_applied": [],
            "style_cues_skipped": [],
            "uncertainties": [],
        }
        errors = app.application_validation_errors(request, result)
        self.assertIn("content_plan paragraph IDs or order do not match input", errors)

    def test_reference_copy_guard_rejects_eight_han_characters(self) -> None:
        request = {
            "reference_examples": [
                {"target_style_text": "他横掌抹过面前虚空，花影出现。"}
            ]
        }
        result = {"paragraphs": [{"zh": "忽然，他横掌抹过面前虚空。"}]}
        self.assertIn("他横掌抹过面前虚", app.copied_reference_spans(request, result))

    def test_deterministic_fidelity_rejects_number_loss(self) -> None:
        request = {"neutral_zh": [{"id": "p0001", "zh": "他等了三天，共30人。"}]}
        result = {"paragraphs": [{"id": "p0001", "zh": "他等了三天。"}]}
        self.assertIn(
            "fidelity:p0001:number_surface_mismatch",
            app.deterministic_fidelity_errors(request, result),
        )

    def test_model_fallback_only_recognizes_capacity_evidence(self) -> None:
        self.assertTrue(
            app.model_capacity_failure(
                {"response_errors": [{"message": "model is at capacity"}]}
            )
        )
        self.assertTrue(
            app.model_capacity_failure(
                {
                    "stderr_tail": (
                        "The 'gpt-5.6' model is not supported when using Codex "
                        "with a ChatGPT account."
                    )
                }
            )
        )
        self.assertFalse(
            app.model_capacity_failure(
                {"validation_errors": ["paragraph IDs or order do not match input"]}
            )
        )

    def test_neutral_leaf_fallback_preserves_valid_plan_and_surface(self) -> None:
        request = {
            "sample_id": "s_0123456789abcdef01234567",
            "method_id": app.METHOD_ID,
            "intensity": app.INTENSITY,
            "english_semantic_source": [{"id": "p0001", "en": "Zion asked."}],
            "neutral_zh": [{"id": "p0001", "zh": "锡安问：\u201c你有什么事？\u201d"}],
            "method_payload": {},
            "reference_examples": [],
        }
        result = {
            "sample_id": request["sample_id"],
            "method_id": app.METHOD_ID,
            "intensity": app.INTENSITY,
            "content_plan": [
                {"id": "p0001", "facts": ["锡安发问"], "constraints": ["保留说话者"]}
            ],
            "paragraphs": [{"id": "p0001", "zh": "\u201c你有什么事？\u201d锡安问。"}],
            "style_cues_applied": ["dialogue_rhythm"],
            "style_cues_skipped": [],
            "uncertainties": [],
        }
        fallback = app.neutral_leaf_fallback(
            request=request,
            artifact={"result": result},
            validation_errors=["fidelity:p0001:dialogue_turn_surface_mismatch"],
        )
        self.assertIsNotNone(fallback)
        fallback_result, rejected_errors = fallback or ({}, [])
        self.assertEqual(fallback_result["paragraphs"], request["neutral_zh"])
        self.assertEqual(fallback_result["content_plan"], result["content_plan"])
        self.assertEqual(
            rejected_errors,
            ["fidelity:p0001:dialogue_turn_surface_mismatch"],
        )
        self.assertIn(
            "neutral_leaf_fallback_after_fidelity_gate",
            fallback_result["style_cues_skipped"],
        )

    def test_neutral_leaf_fallback_rejects_contract_error(self) -> None:
        self.assertIsNone(
            app.neutral_leaf_fallback(
                request={"neutral_zh": [{"id": "p0001", "zh": "原文。"}]},
                artifact={"result": {"content_plan": []}},
                validation_errors=["content_plan paragraph IDs or order do not match input"],
            )
        )

    def test_adaptive_cache_is_model_bound(self) -> None:
        self.assertNotEqual(
            codex_cli._adaptive_model_namespace("gpt-5.5"),
            codex_cli._adaptive_model_namespace("gpt-5.3-codex-spark"),
        )


if __name__ == "__main__":
    unittest.main()
