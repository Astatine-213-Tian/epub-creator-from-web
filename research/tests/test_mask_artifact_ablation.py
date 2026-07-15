from __future__ import annotations

import unittest

from workflows.audit_mask_artifacts import (
    collapse_mask_runs,
    mask_stripped_char_ngrams,
    mask_topology_features,
)


class MaskArtifactAblationTests(unittest.TestCase):
    def test_collapses_each_mask_run_without_merging_separate_runs(self) -> None:
        self.assertEqual(collapse_mask_runs("甲某某某乙某某丙"), "甲某乙某丙")

    def test_topology_features_ignore_lexical_identity(self) -> None:
        first = mask_topology_features("甲乙某某丙丁某戊")
        second = mask_topology_features("春夏某某秋冬某雨")

        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mask_fraction"], 3 / 8)
        self.assertEqual(first["run_length_max"], 2.0)

    def test_topology_features_handle_unmasked_text(self) -> None:
        features = mask_topology_features("甲乙丙丁")

        self.assertEqual(features["mask_fraction"], 0.0)
        self.assertEqual(features["run_count_per_1k"], 0.0)

    def test_stripped_ngrams_remove_markers_and_do_not_bridge_spans(self) -> None:
        features = mask_stripped_char_ngrams("甲乙某某某丙丁戊")

        self.assertIn("甲乙", features)
        self.assertIn("丙丁", features)
        self.assertIn("丙丁戊", features)
        self.assertNotIn("乙丙", features)
        self.assertFalse(any("某" in feature for feature in features))


if __name__ == "__main__":
    unittest.main()
