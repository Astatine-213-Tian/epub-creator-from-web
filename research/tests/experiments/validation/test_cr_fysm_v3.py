from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from experiments.validation.meter.benchmark_content_resistant_meter import (
    Record,
    author_book_weights,
)
from experiments.validation.meter.build_cr_fysm_v3 import (
    FEATURE_FNS,
    calibrated_scores,
    canonical_lock_id,
    choose_threshold,
    cross_fitted_platt_scores,
    ensemble,
    feature_dispersion_mask,
    feature_schema_violations,
    fit_platt_calibrator,
    locked_family_contract,
    role_records,
)
from experiments.validation.meter.preregister_cr_fysm_v3 import (
    active_partition,
    content_fingerprint,
    external_prior_exposure_violations,
)


from experiments.shared.paths import RESEARCH_ROOT


REPO_ROOT = RESEARCH_ROOT


class ContentResistantFeatureTests(unittest.TestCase):
    def test_all_families_ignore_equal_shape_open_class_words(self) -> None:
        first = "人性好。\n山河美。"
        second = "草木青。\n风雪寒。"
        for name, feature_fn in FEATURE_FNS.items():
            with self.subTest(family=name):
                self.assertEqual(feature_fn(first), feature_fn(second))

    def test_registered_feature_schema_covers_each_family(self) -> None:
        text = (
            "某问道：“你走吗？”\n"
            "某答道：“不走。”\n"
            "半晌，两人又坐了下来。"
        )
        artifacts = {
            name: SimpleNamespace(feature_names=list(feature_fn(text)))
            for name, feature_fn in FEATURE_FNS.items()
        }
        self.assertEqual(feature_schema_violations(artifacts), [])

    def test_platt_calibration_and_threshold_are_operational(self) -> None:
        records = [
            Record(f"t{i}", "train", "非天夜翔", f"target-{i // 2}", i, "某。")
            for i in range(8)
        ] + [
            Record(f"n{i}", "train", f"author-{i // 2}", f"negative-{i}", i, "某。")
            for i in range(12)
        ]
        truth = np.asarray([1] * 8 + [0] * 12, dtype=np.int8)
        raw = np.asarray(
            [0.61, 0.66, 0.70, 0.74, 0.78, 0.81, 0.85, 0.90]
            + [0.08, 0.12, 0.15, 0.20, 0.25, 0.29, 0.34, 0.38, 0.42, 0.46, 0.50, 0.54]
        )
        calibrator = fit_platt_calibrator(
            raw, truth, records, target_author="非天夜翔"
        )
        scores = calibrated_scores(calibrator, raw)
        weights = np.ones(len(truth), dtype=np.float64)
        selected = choose_threshold(
            scores,
            truth,
            weights,
            sensitivity_min=0.80,
            specificity_min=0.90,
        )
        self.assertTrue(selected["eligible"])
        self.assertGreaterEqual(selected["sensitivity"], 0.80)
        self.assertGreaterEqual(selected["specificity"], 0.90)

    def test_categorical_features_cannot_bypass_dispersion(self) -> None:
        records = [
            Record("t-a", "train", "target", "target-a", 0, "某。"),
            Record("t-b", "train", "target", "target-b", 0, "某。"),
            Record("n-a", "train", "other-a", "negative-a", 0, "某。"),
            Record("n-b", "train", "other-b", "negative-b", 0, "某。"),
        ]
        matrix = np.asarray(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        )
        keep = feature_dispersion_mask(
            sparse.csr_matrix(matrix),
            records,
            [
                "mean_sentence_cjk",
                "group:conjunction",
                "paragraph_type_share:dialogue",
                "paragraph_transition:dialogue>narration",
            ],
            target_author="target",
            min_target_fraction=1.0,
            min_comparison_fraction=1.0,
        )
        self.assertEqual(keep.tolist(), [True, False, False, False])

    def test_ensemble_consumes_locked_order_and_weights(self) -> None:
        contract = {
            "family_order": [
                "paragraph_discourse",
                "sentence_rhythm",
                "punctuation_dialogue",
                "function_grammar",
            ],
            "family_weights": {
                "paragraph_discourse": 0.4,
                "sentence_rhythm": 0.3,
                "punctuation_dialogue": 0.2,
                "function_grammar": 0.1,
            },
        }
        order, weights = locked_family_contract(contract)
        probabilities = {
            "function_grammar": np.asarray([1.0]),
            "punctuation_dialogue": np.asarray([0.0]),
            "sentence_rhythm": np.asarray([0.0]),
            "paragraph_discourse": np.asarray([0.0]),
        }
        self.assertAlmostEqual(float(ensemble(probabilities, order, weights)[0]), 0.1)
        self.assertAlmostEqual(
            float(
                ensemble(
                    probabilities,
                    order,
                    weights,
                    excluded="paragraph_discourse",
                )[0]
            ),
            1.0 / 6.0,
        )

    def test_group_cross_fitted_ablation_path_is_operational(self) -> None:
        records = [
            Record(f"t-{index}", "dev", "target", f"target-{index // 2}", index, "某。")
            for index in range(8)
        ] + [
            Record(
                f"n-{index}",
                "dev",
                f"negative-{index // 2}",
                f"negative-book-{index}",
                index,
                "某。",
            )
            for index in range(8)
        ]
        truth = np.asarray([1] * 8 + [0] * 8, dtype=np.int8)
        raw_scores = np.asarray(
            [0.68, 0.72, 0.74, 0.77, 0.80, 0.83, 0.87, 0.91]
            + [0.08, 0.11, 0.15, 0.18, 0.22, 0.25, 0.29, 0.33]
        )
        scores, _calibrator, fold_sizes = cross_fitted_platt_scores(
            raw_scores,
            truth,
            records,
            target_author="target",
        )
        weights = author_book_weights(records, binary_target="target")
        threshold = choose_threshold(
            scores,
            truth,
            weights,
            sensitivity_min=0.75,
            specificity_min=0.75,
        )
        self.assertEqual(sum(fold_sizes.values()), len(records))
        self.assertTrue(threshold["eligible"])


class ContractTests(unittest.TestCase):
    def test_lock_id_changes_when_payload_changes(self) -> None:
        payload = {"meter_id": "CR-FYSM-v3", "status": "locked_before_model_fit"}
        first = canonical_lock_id(payload)
        payload["status"] = "changed"
        self.assertNotEqual(first, canonical_lock_id(payload))

    def test_role_records_enforces_split_and_external_author_roles(self) -> None:
        active = [
            Record("fit-target", "train", "非天夜翔", "fit-book", 3, "某。"),
            Record("cal-target", "dev", "非天夜翔", "cal-book", 3, "某。"),
            Record("qual-target", "test", "非天夜翔", "qual-book", 3, "某。"),
            Record("fit-negative", "train", "fit-author", "a", 3, "某。"),
            Record("cal-negative", "dev", "cal-author", "b", 3, "某。"),
        ]
        external = [
            Record("external", "excluded", "external-author", "c", 3, "某。")
        ]
        partition = {
            "target": {
                "fit": ["fit-book"],
                "calibration": ["cal-book"],
                "qualification": ["qual-book"],
            },
            "comparison": {
                "fit": ["fit-author"],
                "calibration": ["cal-author"],
                "qualification": ["external-author"],
            },
        }
        roles = role_records(active, external, partition, "非天夜翔")
        self.assertEqual(
            [item.chunk_id for item in roles["fit"]],
            ["fit-negative", "fit-target"],
        )
        self.assertEqual(
            [item.chunk_id for item in roles["calibration"]],
            ["cal-negative", "cal-target"],
        )
        self.assertEqual(
            [item.chunk_id for item in roles["qualification"]],
            ["external", "qual-target"],
        )

    def test_current_v3_partition_has_fresh_external_qualification(self) -> None:
        partition = active_partition(
            REPO_ROOT / "datasets/masked/chunks.entity_masked_v3.jsonl",
            REPO_ROOT
            / "generated/style_research/external_author_challenge_v1/"
            "masked/chunks.entity_masked_v3.jsonl",
            external_clean_path=REPO_ROOT
            / "generated/style_research/external_author_challenge_v1/"
            "unmasked/chunks.clean.jsonl",
            quarantine_manifest_path=REPO_ROOT
            / "datasets/raw/_quarantine/authorship_mismatch/quarantine_manifest.json",
            prior_exposed_clean_paths=[
                REPO_ROOT
                / "generated/style_research/quarantine/authorship_mismatch/"
                "已枯之色.clean.txt"
            ],
        )
        self.assertEqual(len(partition["target"]["fit"]), 28)
        self.assertEqual(len(partition["target"]["calibration"]), 4)
        self.assertEqual(len(partition["target"]["qualification"]), 4)
        self.assertEqual(len(partition["comparison"]["fit"]), 29)
        self.assertEqual(len(partition["comparison"]["calibration"]), 10)
        self.assertEqual(len(partition["comparison"]["qualification"]), 10)
        self.assertFalse(
            set(partition["comparison"]["qualification"])
            & set(partition["comparison"]["fit"])
        )

    def test_prior_exposure_rejects_title_and_content_hash(self) -> None:
        self.assertEqual(content_fingerprint("甲，乙。\n丙"), content_fingerprint("甲乙丙"))
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            external = root / "external.jsonl"
            quarantine = root / "quarantine.json"
            prior = root / "prior.txt"
            external.write_text(
                json.dumps(
                    {
                        "author": "new-author",
                        "title": "old-title",
                        "chunk_index": 1,
                        "text": "甲乙丙",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            quarantine.write_text(
                json.dumps([{"title": "old-title"}], ensure_ascii=False),
                encoding="utf-8",
            )
            prior.write_text("甲，乙。丙", encoding="utf-8")
            violations = external_prior_exposure_violations(
                external, quarantine, [prior]
            )
            self.assertEqual(
                {row["reason"] for row in violations},
                {
                    "quarantined_title_was_observed_before_v3",
                    "content_fingerprint_matches_prior_exposed_text",
                },
            )


if __name__ == "__main__":
    unittest.main()
