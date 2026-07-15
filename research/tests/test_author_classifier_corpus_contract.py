from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from workflows.audit_style_dataset import (
    CROSS_BOOK_DECONTAMINATION_VERSION,
    MASKING_POLICY_VERSION,
    PUNCTUATION_NORMALIZATION_VERSION,
)
from workflows.benchmark_author_style_supervised import load_records, validate_view_alignment


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def row(*, chunk_id: str = "author-book-0001", split: str = "train") -> dict[str, str]:
    return {
        "split": split,
        "author": "author",
        "title": "book",
        "chunk_id": chunk_id,
        "punctuation_normalization": PUNCTUATION_NORMALIZATION_VERSION,
        "cross_book_decontamination": CROSS_BOOK_DECONTAMINATION_VERSION,
        "masking_policy": MASKING_POLICY_VERSION,
        "text": "一段规范化文本。",
    }


class AuthorClassifierCorpusContractTests(unittest.TestCase):
    def test_rejects_chunks_without_current_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "chunks.jsonl"
            stale = row()
            stale.pop("punctuation_normalization")
            write_jsonl(path, [stale])

            with self.assertRaisesRegex(ValueError, "rebuild"):
                load_records(path)

    def test_rejects_cross_view_split_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clean_path = root / "clean.jsonl"
            masked_path = root / "masked.jsonl"
            write_jsonl(clean_path, [row(split="train")])
            write_jsonl(masked_path, [row(split="test")])

            with self.assertRaisesRegex(ValueError, "split mismatch"):
                validate_view_alignment(
                    {
                        "clean": load_records(clean_path),
                        "entity_masked_v3": load_records(masked_path),
                    }
                )


if __name__ == "__main__":
    unittest.main()
