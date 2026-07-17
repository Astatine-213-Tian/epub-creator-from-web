from __future__ import annotations

import json
import unittest

from src.translation.codex_cli import _candidates_from
from src.translation.comments import selected_comment_notes
from src.translation.glossary import glossary_terms, matched_terms
from src.translation.prompt_builder import build_translation_prompt


class TranslationPromptTests(unittest.TestCase):
    def test_prompt_keeps_payload_and_requires_conservative_candidates(self) -> None:
        payload = {
            "glossary": {"Oracle": "司命"},
            "sentence_translations": [
                {
                    "source": ["A protected sentence."],
                    "zh": ["作者钦定整句。"],
                    "note": "creator translation",
                    "confidence": "high",
                }
            ],
            "comment_notes": ["Risk: Oracle 的固定中译是司命。"],
            "items": [
                {
                    "index": 1,
                    "english": "The Oracle prepared dried ginger and suluo root.",
                    "current_zh": "",
                    "context_before": [],
                    "context_after": [],
                    "glossary_matches": {"Oracle": "司命"},
                }
            ],
        }

        prompt = build_translation_prompt(payload, style_name="test")

        self.assertIn("The default is an empty list", prompt)
        self.assertIn('Ordinary examples to omit include "dried ginger"', prompt)
        self.assertIn('An invented herb such as "suluo root"', prompt)
        self.assertIn("Authoritative comments can establish the Chinese rendering", prompt)
        self.assertIn("a separate protected-quotation layer", prompt)
        self.assertIn("not a source of glossary_candidates", prompt)
        self.assertIn("Omit low-confidence candidates entirely", prompt)
        serialized_payload = prompt.split("INPUT JSON:\n", 1)[1]
        self.assertEqual(json.loads(serialized_payload), payload)

    def test_authoritative_chinese_comments_still_survive_budget_selection(self) -> None:
        comments = [
            {
                "author": "reader",
                "body": "普通读者评论",
                "created": "2026-01-03",
            },
            {
                "author": "Risk",
                "body": "名字翻译为司命。",
                "created": "2026-01-01",
            },
            {
                "author": "creator-account",
                "body": "设定中这个称号就是司命。",
                "is_by_creator": True,
                "created": "2026-01-02",
            },
            {
                "author": "Risk",
                "body": "English-only comments are not translation notes.",
                "created": "2026-01-04",
            },
        ]

        notes = selected_comment_notes(
            comments,
            {"Oracle": "司命"},
            char_budget=1,
        )

        self.assertEqual(
            notes,
            [
                "Risk: 名字翻译为司命。",
                "creator-account: 设定中这个称号就是司命。",
            ],
        )

    def test_glossary_alias_matching_and_candidate_parsing_are_unchanged(self) -> None:
        terms = glossary_terms(
            {
                "terms": {
                    "Oracle": {
                        "zh": "司命",
                        "aliases": ["the Oracle"],
                    },
                    "suluo root": {
                        "zh": "苏罗根",
                    },
                    "Gran": {
                        "zh": "格兰",
                    },
                }
            }
        )

        self.assertEqual(
            matched_terms("The Oracle gathered suluo root.", terms),
            {
                "Oracle": "司命",
                "suluo root": "苏罗根",
            },
        )
        self.assertEqual(
            matched_terms("The grand scholars’ tent was open.", terms),
            {},
        )
        self.assertEqual(
            matched_terms("Gran entered the tent.", terms),
            {"Gran": "格兰"},
        )

        candidates = _candidates_from(
            {
                "glossary_candidates": [
                    {
                        "source": "suluo root",
                        "zh": "苏罗根",
                        "reason": "invented herb",
                        "confidence": "high",
                    },
                    "invalid",
                ]
            },
            fallback_id="fallback-1",
        )
        self.assertEqual(
            candidates,
            [
                {
                    "source": "suluo root",
                    "zh": "苏罗根",
                    "reason": "invented herb",
                    "confidence": "high",
                    "adaptive_fallback_id": "fallback-1",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
