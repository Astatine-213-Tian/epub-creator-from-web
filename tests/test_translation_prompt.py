from __future__ import annotations

import json
import unittest

from src.translation.codex_cli import _candidates_from
from src.translation.comments import authoritative_reply_threads
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
        self.assertIn(
            "maintained glossary and sentence_translations are the only",
            prompt,
        )
        self.assertIn("Raw source comments are reviewed separately", prompt)
        self.assertIn("a separate protected-quotation layer", prompt)
        self.assertIn("not a source of glossary_candidates", prompt)
        self.assertIn("Omit low-confidence candidates entirely", prompt)
        serialized_payload = prompt.split("INPUT JSON:\n", 1)[1]
        self.assertEqual(json.loads(serialized_payload), payload)

    def test_prompt_rejects_raw_comment_context(self) -> None:
        with self.assertRaisesRegex(ValueError, "raw comment context"):
            build_translation_prompt(
                {
                    "comment_notes": ["Risk: Oracle 的固定中译是司命。"],
                    "items": [],
                }
            )

    def test_comment_evidence_keeps_only_chinese_authoritative_reply_threads(
        self,
    ) -> None:
        comments = [
            {
                "id": "reader-question",
                "author": "reader",
                "body": "这个名字的中译是什么？",
                "created": "2026-01-01",
            },
            {
                "id": "risk-reply",
                "parent_id": "reader-question",
                "author": "Risk",
                "body": "名字翻译为司命。",
                "created": "2026-01-02",
            },
            {
                "id": "translation-editor-reply",
                "parent_id": "reader-question",
                "author": "pengiesama",
                "body": "Zion：锡安；Tetsu：闪哲。",
                "created": "2026-01-02T01:00:00",
            },
            {
                "id": "unverified-translator-reply",
                "parent_id": "reader-question",
                "author": "unknown-account",
                "body": "我是翻译，名字是错误答案。",
                "created": "2026-01-02T02:00:00",
            },
            {
                "id": "unanswered-reader",
                "author": "reader",
                "body": "普通读者评论，不应进入提示。",
                "created": "2026-01-03",
            },
            {
                "id": "creator-standalone",
                "author": "creator-account",
                "body": "设定中这个称号就是司命。",
                "is_by_creator": True,
                "created": "2026-01-04",
            },
            {
                "id": "english-risk-reply",
                "parent_id": "unanswered-reader",
                "author": "Risk",
                "body": "English-only comments are not translation notes.",
                "created": "2026-01-05",
            },
        ]

        threads = authoritative_reply_threads(comments)

        self.assertEqual(
            threads,
            [
                {
                    "parent": {
                        "id": "reader-question",
                        "parent_id": None,
                        "author": "reader",
                        "body": "这个名字的中译是什么？",
                        "created": "2026-01-01",
                    },
                    "authoritative_replies": [
                        {
                            "id": "risk-reply",
                            "parent_id": "reader-question",
                            "author": "Risk",
                            "body": "名字翻译为司命。",
                            "created": "2026-01-02",
                        },
                        {
                            "id": "translation-editor-reply",
                            "parent_id": "reader-question",
                            "author": "pengiesama",
                            "body": "Zion：锡安；Tetsu：闪哲。",
                            "created": "2026-01-02T01:00:00",
                        }
                    ],
                }
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
