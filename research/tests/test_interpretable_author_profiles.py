from __future__ import annotations

import unittest

from workflows.analyze_interpretable_author_profiles import (
    METRIC_LABELS,
    extract_book_metrics,
    render_profile_heatmap,
    summarize_authors,
)


class InterpretableAuthorProfilesTest(unittest.TestCase):
    def test_extract_book_metrics_counts_structure_and_connectives(self) -> None:
        text = (
            "他说：“你好吗？”\n"
            "我很好。\n"
            "因为下雨，所以回家。\n"
        )

        metrics = extract_book_metrics(text)

        self.assertAlmostEqual(metrics["dialogue_paragraph_pct"], 100 / 3)
        self.assertEqual(metrics["questions_per_10k"], 10000 / 16)
        self.assertEqual(metrics["connectives_per_10k"], 20000 / 16)
        self.assertEqual(metrics["short_sentence_pct"], 100.0)

    def test_author_summary_weights_books_before_authors(self) -> None:
        def book(title: str, value: float) -> dict[str, object]:
            return {
                "title": title,
                "clean_cjk_count": 50_000,
                "metrics": {metric: value for metric in METRIC_LABELS},
            }

        summaries = summarize_authors(
            {
                "作者甲": [book("甲一", 1.0), book("甲二", 3.0)],
                "作者乙": [book("乙一", 6.0)],
            }
        )

        self.assertEqual(summaries["作者甲"]["metrics"]["median_sentence_cjk"], 2.0)
        self.assertEqual(summaries["作者乙"]["metrics"]["median_sentence_cjk"], 6.0)
        self.assertLess(
            summaries["作者甲"]["standard_scores"]["median_sentence_cjk"], 0
        )
        self.assertGreater(
            summaries["作者乙"]["standard_scores"]["median_sentence_cjk"], 0
        )

        svg = render_profile_heatmap(summaries, ("作者甲", "作者乙"))
        self.assertIn("作者甲", svg)
        self.assertIn("句长中位数", svg)


if __name__ == "__main__":
    unittest.main()
