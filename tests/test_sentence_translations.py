from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.crawler.snapshot import write_json
from src.translation.sentence_translations import (
    apply_sentence_translation_overrides,
    sentence_translation_entries,
)


SOURCE = (
    "If ever should I taste of love’s sweet savor,",
    "All things I shall forsake and seek no more the Eternal Gate.",
)
ZH = (
    "若有朝品尝爱情的滋味，",
    "我将舍弃万物，不再奔往永恒之门。",
)


class SentenceTranslationTests(unittest.TestCase):
    def test_sentence_entries_are_separate_from_glossary_terms(self) -> None:
        entries = sentence_translation_entries(
            {
                "terms": {"Oracle": {"zh": "司命"}},
                "sentence_translations": [
                    {
                        "source": list(SOURCE),
                        "zh": list(ZH),
                        "note": "creator-confirmed",
                        "confidence": "high",
                    }
                ],
            }
        )

        self.assertEqual(entries[0].source, SOURCE)
        self.assertEqual(entries[0].zh, ZH)
        self.assertEqual(entries[0].note, "creator-confirmed")

    def test_segment_count_must_match(self) -> None:
        with self.assertRaisesRegex(ValueError, "matching source and zh segment"):
            sentence_translation_entries(
                {
                    "sentence_translations": [
                        {"source": list(SOURCE), "zh": ["只有一段"]}
                    ]
                }
            )

    def test_split_and_embedded_occurrences_are_restored_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = root / "snapshot"
            semantic_run = root / "semantic"
            style_run = root / "style"
            self._write_snapshot(snapshot)
            self._write_translation(
                semantic_run,
                {
                    0: ZH[0],
                    1: ZH[1],
                    2: f"闪哲轻声道：“{''.join(ZH)}”",
                },
            )
            self._write_translation(
                style_run,
                {
                    0: "若有一日尝到爱情甜美，",
                    1: "我便舍尽一切，不再寻门。",
                    2: "闪哲轻声说了另一种译文。",
                    3: "无关段落。",
                },
            )
            write_json(
                style_run / "run_manifest.json",
                {
                    "snapshot_dir": str(snapshot),
                    "semantic_run_dir": str(semantic_run),
                },
            )
            glossary = {
                "sentence_translations": [
                    {
                        "source": list(SOURCE),
                        "zh": list(ZH),
                        "note": "creator-confirmed",
                        "confidence": "high",
                    }
                ]
            }

            first = apply_sentence_translation_overrides(
                run_dir=style_run,
                snapshot_dir=snapshot,
                glossary=glossary,
            )
            translations = self._translation_map(style_run)

            self.assertEqual(translations[0], ZH[0])
            self.assertEqual(translations[1], ZH[1])
            self.assertEqual(translations[2], f"闪哲轻声道：“{''.join(ZH)}”")
            self.assertEqual(translations[3], "无关段落。")
            self.assertEqual(first["occurrence_count"], 2)
            self.assertEqual(first["changed_occurrence_count"], 2)
            self.assertEqual(
                [
                    item["mode"]
                    for item in first["entries"][0]["occurrences"]
                ],
                ["canonical_segments", "semantic_paragraph_restore"],
            )

            second = apply_sentence_translation_overrides(
                run_dir=style_run,
                snapshot_dir=snapshot,
                glossary=glossary,
            )
            self.assertEqual(second["changed_occurrence_count"], 0)

    def test_embedded_occurrence_without_canonical_semantic_text_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = root / "snapshot"
            style_run = root / "style"
            self._write_snapshot(snapshot)
            self._write_translation(
                style_run,
                {0: ZH[0], 1: ZH[1], 2: "错误改写。", 3: "无关段落。"},
            )
            write_json(
                style_run / "run_manifest.json",
                {"snapshot_dir": str(snapshot)},
            )

            with self.assertRaisesRegex(ValueError, "could not resolve 1 occurrence"):
                apply_sentence_translation_overrides(
                    run_dir=style_run,
                    snapshot_dir=snapshot,
                    glossary={
                        "sentence_translations": [
                            {"source": list(SOURCE), "zh": list(ZH)}
                        ]
                    },
                )

            report = json.loads(
                (
                    style_run
                    / "sentence_translations"
                    / "session_summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(len(report["unresolved"]), 1)

    @staticmethod
    def _write_snapshot(snapshot: Path) -> None:
        write_json(
            snapshot / "manifest.json",
            {
                "schema_version": 1,
                "title": "test",
                "author": "test",
                "chapters": [
                    {
                        "id": "01",
                        "title": "Chapter 1",
                        "path": "chapters/01.json",
                    }
                ],
            },
        )
        write_json(
            snapshot / "chapters" / "01.json",
            {
                "title": "Chapter 1",
                "paragraphs": [
                    {"index": 0, "english": SOURCE[0]},
                    {"index": 1, "english": SOURCE[1]},
                    {
                        "index": 2,
                        "english": (
                            "He murmured, “If ever should I taste of love’s sweet "
                            "savor, all things I shall forsake and seek no more "
                            "the Eternal Gate.”"
                        ),
                    },
                    {"index": 3, "english": "Unrelated."},
                ],
            },
        )

    @staticmethod
    def _write_translation(run_dir: Path, translations: dict[int, str]) -> None:
        write_json(
            run_dir / "translations" / "01.json",
            {
                "chapter_id": "01",
                "chapter_title": "Chapter 1",
                "translations": [
                    {"index": index, "zh": zh}
                    for index, zh in sorted(translations.items())
                ],
            },
        )

    @staticmethod
    def _translation_map(run_dir: Path) -> dict[int, str]:
        data = json.loads(
            (run_dir / "translations" / "01.json").read_text(encoding="utf-8")
        )
        return {
            int(item["index"]): str(item["zh"])
            for item in data["translations"]
        }


if __name__ == "__main__":
    unittest.main()
