from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.validation.application import evaluate_method4_application as evaluation


class EvaluateMethod4ApplicationTest(unittest.TestCase):
    def test_windows_preserve_every_index_once(self) -> None:
        neutral = {index: "中" * 250 for index in range(10)}
        styled = {index: "文" * 240 for index in range(10)}
        windows = evaluation.build_windows(
            chapter_id="09",
            title="Chapter 007",
            neutral=neutral,
            styled=styled,
        )
        indexes = [index for window in windows for index in window["indexes"]]
        self.assertEqual(indexes, list(range(10)))
        self.assertEqual(len(indexes), len(set(indexes)))

    def test_generated_mask_removes_terms_latin_and_numbers(self) -> None:
        masked = evaluation.mask_generated_text("锡安见到Tetsu，共30人。", ["锡安"])
        self.assertNotIn("锡安", masked)
        self.assertIn("<LATIN>", masked)
        self.assertIn("<NUM>", masked)

    def test_svg_renderer_writes_nonempty_chart(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "chart.svg"
            evaluation.render_svg(
                path,
                [
                    {
                        "chapter_title": "Chapter 007",
                        "mean_paired_margin_lift": 0.25,
                    },
                    {
                        "chapter_title": "Chapter 008",
                        "mean_paired_margin_lift": -0.10,
                    },
                ],
            )
            rendered = path.read_text(encoding="utf-8")
            self.assertIn("<svg", rendered)
            self.assertIn("+0.250", rendered)
            self.assertGreater(path.stat().st_size, 500)


if __name__ == "__main__":
    unittest.main()
