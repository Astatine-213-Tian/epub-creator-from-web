from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.iteration5.meter.benchmark_content_resistant_meter import Record, content_counterfactual
from experiments.iteration5.meter.build_cr_fysm_v2 import (
    FEATURE_FNS,
    calibrated_scores,
    choose_threshold,
    feature_schema_violations,
    fit_platt_calibrator,
)
from experiments.iteration5.meter.preregister_cr_fysm_v2 import rotated_partition


from experiments.shared.paths import RESEARCH_ROOT


REPO_ROOT = RESEARCH_ROOT


class ContentResistantFeatureTests(unittest.TestCase):
    def test_all_families_ignore_lexical_substitution(self) -> None:
        text = (
            "某某忽然问道：“你真的要走吗？”\n"
            "某摇了摇头，半晌，只得答道：“不走。”\n"
            "于是两人坐下，安静地等了很久。"
        )
        counterfactual = content_counterfactual(text)
        for name, feature_fn in FEATURE_FNS.items():
            with self.subTest(family=name):
                self.assertEqual(feature_fn(text), feature_fn(counterfactual))

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
        selected = choose_threshold(scores, truth)
        self.assertTrue(selected["eligible"])
        self.assertGreaterEqual(selected["sensitivity"], 0.80)
        self.assertGreaterEqual(selected["specificity"], 0.90)


class RotatedPartitionTests(unittest.TestCase):
    def test_current_partition_is_rotated_and_complete(self) -> None:
        partition = rotated_partition(
            REPO_ROOT / "datasets/masked/chunks.entity_masked_v3.jsonl",
            REPO_ROOT
            / "generated/style_research/style_transfer_experiments/iterations/"
            "content_resistant_v1/cr_fysm_v1/partition.v1.json",
            "非天夜翔",
        )
        self.assertEqual(len(partition["target"]["fit"]), 20)
        self.assertEqual(len(partition["target"]["calibration"]), 4)
        self.assertEqual(len(partition["target"]["qualification"]), 4)
        self.assertEqual(len(partition["comparison"]["fit"]), 29)
        self.assertEqual(len(partition["comparison"]["calibration"]), 10)
        self.assertEqual(len(partition["comparison"]["qualification"]), 10)
        self.assertNotIn("已枯之色", json.dumps(partition, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
