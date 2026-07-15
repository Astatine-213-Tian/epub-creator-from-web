#!/usr/bin/env python3
from __future__ import annotations

import argparse
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.iteration1 import run_style_transfer_generation as generation

from experiments.shared.paths import RESEARCH_ROOT


REPO_ROOT = RESEARCH_ROOT
EXPERIMENT_ROOT = (
    REPO_ROOT
    / "generated/style_research/style_transfer_experiments/iterations/full_regeneration_v1"
)
DEVELOPMENT_SELECTION = (
    EXPERIMENT_ROOT
    / "sample_sets/iteration4_proxy_v1.development_v1_ids.json"
)
CONFIRMATION_SELECTION = (
    EXPERIMENT_ROOT
    / "sample_sets/iteration4_proxy_v1.confirmation_v1_ids.json"
)


def args_for(method_id: str) -> argparse.Namespace:
    return argparse.Namespace(
        stage="style_transfer",
        analysis_lock=Path("/tmp/nonexistent-rehearsal-lock.json"),
        experiment_root=EXPERIMENT_ROOT,
        method_id=method_id,
        intensity="strong",
        base_method_id=None,
        base_intensity=None,
        provisional_shortlist=None,
        refinement_contract=None,
        promotion_file=None,
        final_lock=None,
    )


def development_selection() -> dict[str, object]:
    return {
        "execution_selection_id": "development_v1",
        "execution_selection_is_frozen_file": True,
        "execution_selection_path": str(DEVELOPMENT_SELECTION.relative_to(REPO_ROOT)),
        "execution_selection_sha256": generation.file_sha256(DEVELOPMENT_SELECTION),
        "execution_selection_sample_count": 16,
        "active_sample_count": 16,
    }


def confirmation_selection() -> dict[str, object]:
    return {
        "execution_selection_id": "confirmation_v1",
        "execution_selection_is_frozen_file": True,
        "execution_selection_path": str(CONFIRMATION_SELECTION.relative_to(REPO_ROOT)),
        "execution_selection_sha256": generation.file_sha256(CONFIRMATION_SELECTION),
        "execution_selection_sample_count": 122,
        "active_sample_count": 122,
    }


class DevelopmentAdmissionContractTest(unittest.TestCase):
    def test_registered_fixed_method_is_admitted_without_outcome_artifact(self) -> None:
        selection = development_selection()
        lock = {"status": "valid", "content_sha256": "rehearsal-lock"}

        with patch.object(generation, "validate_analysis_lock", return_value=lock):
            admission = generation.validate_execution_admission(
                args_for("generic_full_regeneration"), selection
            )

        self.assertEqual(
            admission["admission_stage"], "preregistered_development_pilot"
        )
        self.assertEqual(admission["analysis_lock"], lock)
        self.assertIs(admission["official_efficacy_evidence"], False)

    def test_derived_selector_is_not_admitted_as_fixed_generation(self) -> None:
        selection = development_selection()

        with patch.object(
            generation,
            "validate_analysis_lock",
            return_value={"status": "valid", "content_sha256": "test"},
        ):
            with self.assertRaisesRegex(ValueError, "development-pilot roster"):
                generation.validate_execution_admission(
                    args_for("independent_candidate_selector"), selection
                )

    def test_substituted_development_selection_is_rejected(self) -> None:
        selection = development_selection()
        selection["execution_selection_path"] = "sample_sets/substituted.json"

        with patch.object(
            generation,
            "validate_analysis_lock",
            return_value={"status": "valid", "content_sha256": "test"},
        ):
            with self.assertRaisesRegex(ValueError, "registered selection file"):
                generation.validate_execution_admission(
                    args_for("generic_full_regeneration"), selection
                )

    def test_confirmation_requires_frozen_promotion(self) -> None:
        with patch.object(
            generation,
            "validate_analysis_lock",
            return_value={"status": "valid", "content_sha256": "test"},
        ):
            with self.assertRaisesRegex(ValueError, "requires --promotion-file"):
                generation.validate_execution_admission(
                    args_for("generic_full_regeneration"),
                    confirmation_selection(),
                )


if __name__ == "__main__":
    unittest.main()
