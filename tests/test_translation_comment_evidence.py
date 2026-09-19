from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.cli.translate import main as translate_main
from src.crawler.snapshot import write_json
from src.translation.comments import write_glossary_comment_evidence
from src.translation.prompt_builder import prepare_prompts


class TranslationCommentEvidenceTests(unittest.TestCase):
    def _write_snapshot(self, root: Path) -> Path:
        snapshot = root / "snapshot"
        write_json(
            snapshot / "manifest.json",
            {
                "schema_version": 1,
                "title": "Test Book",
                "author": "Test Author",
                "chapters": [
                    {
                        "id": "001",
                        "title": "Chapter One",
                        "source_id": "source-1",
                        "source_url": "https://example.test/chapter-1",
                        "path": "chapters/001.json",
                        "comments_path": "comments/001.json",
                    }
                ],
            },
        )
        write_json(
            snapshot / "chapters/001.json",
            {
                "id": "001",
                "title": "Chapter One",
                "paragraphs": [{"index": 0, "english": "The Oracle arrived."}],
            },
        )
        write_json(
            snapshot / "comments/001.json",
            {
                "comments": [
                    {
                        "id": "question",
                        "author": "reader",
                        "body": "What is the official Chinese name?",
                        "created": "2026-01-01",
                    },
                    {
                        "id": "answer",
                        "parent_id": "question",
                        "author": "Risk",
                        "body": "正式中译是司命。",
                        "created": "2026-01-02",
                    },
                    {
                        "id": "unanswered",
                        "author": "reader",
                        "body": "普通评论。",
                        "created": "2026-01-03",
                    },
                ]
            },
        )
        return snapshot

    def test_writes_review_only_authoritative_reply_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = self._write_snapshot(root)
            output = root / "glossary_comment_evidence.json"

            evidence = write_glossary_comment_evidence(
                snapshot_dir=snapshot,
                output_path=output,
            )

            self.assertTrue(evidence["review_required"])
            self.assertFalse(evidence["consumed_by_translation_prompts"])
            self.assertEqual(evidence["purpose"], "review_for_glossary")
            self.assertEqual(evidence["thread_count"], 1)
            self.assertEqual(evidence["authoritative_reply_count"], 1)
            chapter = evidence["chapters"][0]
            self.assertEqual(chapter["chapter_id"], "001")
            self.assertEqual(
                chapter["threads"][0]["parent"]["body"],
                "What is the official Chinese name?",
            )
            self.assertEqual(
                chapter["threads"][0]["authoritative_replies"][0]["body"],
                "正式中译是司命。",
            )
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                evidence,
            )

    def test_comment_changes_do_not_change_translation_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = self._write_snapshot(root)
            first_run = root / "first"
            second_run = root / "second"
            config = {"title": "Test Book", "translation": {"chunk_size": 10}}

            prepare_prompts(
                snapshot_dir=snapshot,
                run_dir=first_run,
                config=config,
                glossary={"terms": {"Oracle": {"zh": "司命"}}},
            )
            write_json(
                snapshot / "comments/001.json",
                {
                    "comments": [
                        {
                            "id": "changed",
                            "author": "Risk",
                            "body": "完全不同的原始评论内容。",
                            "is_by_creator": True,
                        }
                    ]
                },
            )
            prepare_prompts(
                snapshot_dir=snapshot,
                run_dir=second_run,
                config=config,
                glossary={"terms": {"Oracle": {"zh": "司命"}}},
            )

            first_prompt = (first_run / "prompts/001_001.txt").read_text(
                encoding="utf-8"
            )
            second_prompt = (second_run / "prompts/001_001.txt").read_text(
                encoding="utf-8"
            )
            self.assertEqual(first_prompt, second_prompt)
            payload = json.loads(first_prompt.split("INPUT JSON:\n", 1)[1])
            self.assertNotIn("comment_notes", payload)
            self.assertNotIn("正式中译是司命", first_prompt)

    def test_validate_writes_candidates_without_mutating_glossary(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = self._write_snapshot(root)
            glossary_path = root / "glossary.json"
            write_json(
                glossary_path,
                {"terms": {"Oracle": {"zh": "司命"}}, "style_notes": []},
            )
            before = glossary_path.read_text(encoding="utf-8")
            config_path = root / "config.json"
            write_json(config_path, {"glossary_path": str(glossary_path)})
            run_dir = root / "run"
            write_json(
                run_dir / "run_manifest.json",
                {
                    "schema_version": 2,
                    "snapshot_dir": str(snapshot),
                    "chunks": [
                        {
                            "chunk_id": "001_001",
                            "chapter_id": "001",
                            "chapter_title": "Chapter One",
                            "indexes": [0],
                            "raw_output_path": "outputs/001_001.raw.txt",
                            "json_output_path": "outputs/001_001.json",
                        }
                    ],
                },
            )
            write_json(
                run_dir / "outputs/001_001.json",
                {
                    "chapter_title": "Chapter One",
                    "translations": [{"index": 0, "zh": "司命来了。"}],
                    "glossary_candidates": [
                        {
                            "source": "new term",
                            "zh": "新词",
                            "reason": "test candidate",
                            "confidence": "medium",
                        }
                    ],
                },
            )

            return_code = translate_main(
                ["validate", str(run_dir), "--config", str(config_path)]
            )

            self.assertEqual(return_code, 0)
            self.assertEqual(glossary_path.read_text(encoding="utf-8"), before)
            candidates = json.loads(
                (run_dir / "glossary_candidates.json").read_text(encoding="utf-8")
            )
            self.assertEqual(candidates[0]["source"], "new term")


if __name__ == "__main__":
    unittest.main()
