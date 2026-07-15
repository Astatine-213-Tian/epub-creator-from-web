#!/usr/bin/env python3
from __future__ import annotations

import unittest

from experiments.iteration4 import evaluate_development_amendment as amendment
from experiments.iteration4 import evaluate_style_transfer_methods as iteration4
from experiments.shared.paths import RESEARCH_ROOT


REPO_ROOT = RESEARCH_ROOT
EXPERIMENT_ROOT = (
    REPO_ROOT
    / "generated/style_research/style_transfer_experiments/iterations/full_regeneration_v1"
)


class EvaluationContractTest(unittest.TestCase):
    def load_base(self):
        return iteration4.load_module(
            "iteration4_contract_evaluation_base",
            REPO_ROOT / "experiments/iteration1/evaluate_style_transfer_methods.py",
        )

    def test_frozen_iteration4_geometry(self) -> None:
        base = self.load_base()

        contract = iteration4.load_iteration4_sample_contract(
            base,
            EXPERIMENT_ROOT,
            "iteration4_proxy_v1",
        )

        self.assertEqual(len(contract["method_ids"]), 182)
        self.assertEqual(len(contract["calibration_ids"]), 32)
        self.assertEqual(
            {
                key: len(value["sample_ids"])
                for key, value in contract["frozen_selections"].items()
            },
            {
                "development_v1": 16,
                "screening_v1": 44,
                "confirmation_v1": 122,
            },
        )

    def test_existing_placeholders_survive_entity_masking(self) -> None:
        base = self.load_base()
        masker = base.EntityMasker(REPO_ROOT / "datasets/masked/mask_terms.json")

        masked = masker.mask(
            "已有<NUM>，新增123；已有<LATIN>，新增ABC。",
            author="非天夜翔",
            title="朝圣",
        )

        self.assertEqual(masked.count("<NUM>"), 2)
        self.assertEqual(masked.count("<LATIN>"), 2)

    def test_analysis_lock_routes_to_iteration4_sample_set(self) -> None:
        lock_module = iteration4.load_module(
            "iteration4_contract_style_analysis_lock",
            REPO_ROOT / "experiments/iteration1/style_analysis_lock.py",
        )

        artifact_paths = lock_module.experiment_artifact_paths(EXPERIMENT_ROOT)
        names = {path.name for path in artifact_paths}

        self.assertIn("iteration4_proxy_v1.evaluator_allocation.jsonl", names)
        self.assertIn("iteration4_proxy_v1.hidden_targets.jsonl", names)
        self.assertNotIn("development_proxy_v1.evaluator_allocation.jsonl", names)

    def test_iteration4_asset_loader_matches_shared_runner_contract(self) -> None:
        payload = iteration4.load_module(
            "iteration4_contract_payload_loader",
            REPO_ROOT / "experiments/iteration4/style_transfer_payloads.py",
        )

        frozen = payload.load_frozen_assets(EXPERIMENT_ROOT)

        self.assertEqual(
            frozen["schema_version"],
            "style_transfer_payload_assets.v4.full_regeneration",
        )
        self.assertEqual(
            frozen["assets"]["source_projections"]["style_prompt_sha256"],
            payload.file_sha256(
                EXPERIMENT_ROOT / "prompts/style_transfer_method.v1.md"
            ),
        )

    def test_method_configs_use_evaluator_identity_contract(self) -> None:
        prepare = iteration4.load_module(
            "iteration4_contract_registry_reader",
            REPO_ROOT / "experiments/iteration4/prepare.py",
        )
        registry = prepare.read_json(
            EXPERIMENT_ROOT / "method_registry/style_methods.v1.json"
        )

        for row in registry["methods"]:
            config = prepare.read_json(REPO_ROOT / row["config_path"])
            self.assertEqual(config["method_id"], row["id"])

    def test_development_amendment_requires_exact_frozen_admission(self) -> None:
        valid = {
            "execution_admission": {
                "admission_stage": "preregistered_development_pilot",
                "selection_id": "development_v1",
                "official_efficacy_evidence": False,
            },
            "execution_selection_id": "development_v1",
            "execution_selection_sha256": "selection-hash",
            "execution_selection_sample_count": 16,
        }

        self.assertTrue(
            amendment._has_exact_development_admission(
                valid,
                selection_sha256="selection-hash",
                sample_count=16,
            )
        )
        for field, replacement in (
            ("execution_selection_id", "screening_v1"),
            ("execution_selection_sha256", "wrong"),
            ("execution_selection_sample_count", 15),
        ):
            altered = {**valid, field: replacement}
            self.assertFalse(
                amendment._has_exact_development_admission(
                    altered,
                    selection_sha256="selection-hash",
                    sample_count=16,
                )
            )

        altered = {**valid, "execution_admission": {**valid["execution_admission"]}}
        altered["execution_admission"]["admission_stage"] = "preregistered_screening"
        self.assertFalse(
            amendment._has_exact_development_admission(
                altered,
                selection_sha256="selection-hash",
                sample_count=16,
            )
        )


if __name__ == "__main__":
    unittest.main()
