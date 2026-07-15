from __future__ import annotations

import math
import unittest

import numpy as np

from experiments.iteration5.construct import analyze_cr_fysm_v4_construct_v2 as analyze
from experiments.iteration5.construct import run_cr_fysm_v4_construct_ratings_v2 as ratings


class ConstructAnalysisV2Tests(unittest.TestCase):
    def test_schema_validator_enforces_integer_range_and_enum(self) -> None:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["score", "choice"],
            "properties": {
                "score": {"type": "integer", "minimum": 1, "maximum": 5},
                "choice": {"type": "string", "enum": ["A", "B", "tie"]},
            },
        }
        self.assertEqual(
            ratings.validate_schema_instance({"score": 4, "choice": "A"}, schema),
            [],
        )
        self.assertTrue(
            ratings.validate_schema_instance({"score": 7, "choice": "C"}, schema)
        )

    def test_semantic_vote_contract(self) -> None:
        valid = {
            "semantic_fidelity": 4,
            "naturalness": 3,
            "high_severity_semantic_error": False,
            "speaker_dialogue_topology_preserved": True,
        }
        self.assertTrue(ratings.semantic_pass(valid))
        self.assertFalse(ratings.semantic_pass({**valid, "semantic_fidelity": 3}))
        self.assertFalse(
            ratings.semantic_pass({**valid, "high_severity_semantic_error": True})
        )

    def test_average_ranks_preserve_ties(self) -> None:
        observed = analyze.average_ranks(np.asarray([5.0, 1.0, 1.0, 3.0]))
        np.testing.assert_allclose(observed, [4.0, 1.5, 1.5, 3.0])

    def test_spearman_and_constant_contract(self) -> None:
        self.assertAlmostEqual(analyze.spearman([1, 2, 3], [3, 2, 1]), -1.0)
        self.assertTrue(math.isnan(analyze.spearman([1, 1, 1], [1, 2, 3])))

    def test_style_vote_decoding(self) -> None:
        self.assertEqual(
            analyze.decode_style_vote({"choice": "A"}, side_order="style_first"), 1
        )
        self.assertEqual(
            analyze.decode_style_vote({"choice": "A"}, side_order="control_first"), -1
        )
        self.assertEqual(
            analyze.decode_style_vote({"choice": "tie"}, side_order="style_first"), 0
        )

    def test_wilson_lower_is_conservative(self) -> None:
        self.assertGreater(analyze.wilson_lower(25, 25), 0.65)
        self.assertLess(analyze.wilson_lower(20, 25), 0.80)


if __name__ == "__main__":
    unittest.main()
