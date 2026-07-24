from __future__ import annotations

import unittest

from workflows.analyze_author_style_drift import (
    assign_period,
    assign_style_family,
    benjamini_hochberg,
    count_quoted_cjk,
    count_third_person_pronouns,
    extract_extended_metrics,
)


class AuthorStyleDriftTest(unittest.TestCase):
    def test_style_family_keeps_western_fantasy_and_future_distinct(self) -> None:
        self.assertEqual(
            assign_style_family("逆世界之书", "架空历史"),
            "西方架空／玄幻",
        )
        self.assertEqual(
            assign_style_family("星辰骑士", "幻想未来"),
            "科幻／未来",
        )
        self.assertEqual(
            assign_style_family("相见欢", "古色古香"),
            "东方古代／武侠",
        )

    def test_period_boundaries_match_report(self) -> None:
        self.assertEqual(assign_period(2012), "2008–2012")
        self.assertEqual(assign_period(2013), "2013–2017")
        self.assertEqual(assign_period(2017), "2013–2017")
        self.assertEqual(assign_period(2018), "2018–2025")

    def test_extended_metrics_use_transparent_counts(self) -> None:
        text = "其他人弹吉他。他说：“她去了。”\n甲乙丙丁，戊己。"

        self.assertEqual(count_third_person_pronouns(text), 2)
        self.assertEqual(count_quoted_cjk(text), 3)

        metrics = extract_extended_metrics(text)

        self.assertAlmostEqual(
            metrics["third_person_pronouns_per_1k"], 2000 / 17
        )
        self.assertAlmostEqual(metrics["quoted_cjk_pct"], 300 / 17)
        self.assertAlmostEqual(
            metrics["exact_four_cjk_clause_pct"], 20.0
        )
        self.assertEqual(metrics["dao_speech_tag_share"], 0.0)
        self.assertEqual(metrics["speech_tags_per_quoted_span"], 1.0)

    def test_benjamini_hochberg_is_monotone_in_rank_order(self) -> None:
        adjusted = benjamini_hochberg([0.01, 0.04, 0.03, 0.20])

        self.assertAlmostEqual(adjusted[0], 0.04)
        self.assertAlmostEqual(adjusted[1], 0.05333333333333334)
        self.assertAlmostEqual(adjusted[2], 0.05333333333333334)
        self.assertAlmostEqual(adjusted[3], 0.20)


if __name__ == "__main__":
    unittest.main()
