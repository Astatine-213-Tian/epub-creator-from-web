from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from workflows.author_style_meter_contract import file_sha256
from workflows.report_author_style_meter import load_bound_evaluation


class AuthorStyleReportTests(unittest.TestCase):
    def test_evaluator_review_must_bind_current_benchmark_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = root / "results.json"
            review = root / "review.md"
            benchmark.write_text('{"accuracy": 0.9}\n', encoding="utf-8")
            review.write_text(
                f"<!-- benchmark-result-sha256: {file_sha256(benchmark)} -->\n\nReview.",
                encoding="utf-8",
            )

            self.assertEqual(
                load_bound_evaluation(review, benchmark).splitlines()[-1],
                "Review.",
            )

            benchmark.write_text('{"accuracy": 0.8}\n', encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "not bound"):
                load_bound_evaluation(review, benchmark)

    def test_ablation_review_uses_its_own_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = root / "ablation.json"
            review = root / "review.md"
            result.write_text('{}\n', encoding="utf-8")
            review.write_text(
                f"<!-- ablation-result-sha256: {file_sha256(result)} -->\n\nAudit.",
                encoding="utf-8",
            )

            self.assertEqual(
                load_bound_evaluation(
                    review,
                    result,
                    marker_name="ablation-result-sha256",
                ).splitlines()[-1],
                "Audit.",
            )


if __name__ == "__main__":
    unittest.main()
