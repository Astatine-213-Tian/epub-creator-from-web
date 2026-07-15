#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from contextlib import ExitStack
from itertools import zip_longest
from pathlib import Path
from typing import Any, TextIO


MANIFEST = Path("datasets/dataset_manifest.json")
CLEANED_MANIFEST = Path("generated/style_research/corpus/cleaned_manifest.json")
CLEANING_REPORT = Path("generated/style_research/corpus/cleaning_report.json")
CHUNK_REPORT = Path("generated/style_research/corpus/chunk_report.json")
CLEAN_TEXT_ROOT = Path("generated/style_research/corpus/texts")
CHUNK_PATHS = {
    "clean": Path("datasets/unmasked/chunks.clean.jsonl"),
    "entity_masked": Path("datasets/masked/chunks.entity_masked.jsonl"),
    "entity_masked_v2": Path("datasets/masked/chunks.entity_masked_v2.jsonl"),
    "entity_masked_v3": Path("datasets/masked/chunks.entity_masked_v3.jsonl"),
    "topic_distorted": Path("datasets/masked/chunks.topic_distorted.jsonl"),
    "structure_only": Path("datasets/masked/chunks.structure_only.jsonl"),
}
RETIRED_BOOKS = {
    ("妄鸦", "晚来天欲雪"),
    ("妄鸦", "能饮一杯无"),
}
METADATA_FIELDS = (
    "chunk_id",
    "split",
    "author",
    "title",
    "chunk_index",
    "book_clean_cjk_count",
    "chunk_clean_cjk_count",
    "time_area",
    "genre",
    "article_type",
    "quality_flags",
)
RESIDUE_RE = re.compile(
    r"作者有话要说|晋江文学城|jjwxc|请收藏|霸王票|营养液加更|"
    r"最新网址|返回目录|手机阅读|https?://|www\.|€{2,}|�|"
    r"[（(][ｃcＣ][ｏoＯ](?:[ｍmＭ])?[）)]",
    re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the rebuilt style-research corpus.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("generated/style_research/corpus/rebuild_validation.json"),
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def residue_hits(path: Path, *, limit: int = 20) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            match = RESIDUE_RE.search(line)
            if match:
                hits.append(
                    {
                        "path": str(path),
                        "line": line_number,
                        "match": match.group(0),
                    }
                )
                if len(hits) >= limit:
                    break
    return hits


def validate_chunks() -> tuple[dict[str, int], list[dict[str, Any]], list[str]]:
    counts = Counter[str]()
    residues: list[dict[str, Any]] = []
    errors: list[str] = []
    handles: dict[str, TextIO] = {}
    with ExitStack() as stack:
        for view, path in CHUNK_PATHS.items():
            handles[view] = stack.enter_context(path.open(encoding="utf-8"))
        streams = [handles[view] for view in CHUNK_PATHS]
        for line_number, lines in enumerate(zip_longest(*streams), start=1):
            if any(line is None for line in lines):
                errors.append(f"chunk views have unequal row counts at line {line_number}")
                break
            rows = [json.loads(line) for line in lines if line is not None]
            baseline = rows[0]
            for (view, _), row in zip(CHUNK_PATHS.items(), rows, strict=True):
                counts[view] += 1
                if row.get("view") != view:
                    errors.append(f"line {line_number}: expected view {view}, got {row.get('view')}")
                for field in METADATA_FIELDS:
                    if row.get(field) != baseline.get(field):
                        errors.append(
                            f"line {line_number}: {view}.{field} differs from clean view"
                        )
                        break
                if (row.get("author"), row.get("title")) in RETIRED_BOOKS:
                    errors.append(f"line {line_number}: retired book remains in {view}")
                match = RESIDUE_RE.search(str(row.get("text", "")))
                if match and len(residues) < 20:
                    residues.append(
                        {
                            "path": str(CHUNK_PATHS[view]),
                            "line": line_number,
                            "match": match.group(0),
                        }
                    )
            if len(errors) >= 20:
                break
    return dict(counts), residues, errors[:20]


def main() -> None:
    args = parse_args()
    manifest = read_json(MANIFEST)
    cleaned = read_json(CLEANED_MANIFEST)
    cleaning_report = read_json(CLEANING_REPORT)
    chunk_report = read_json(CHUNK_REPORT)
    authors = Counter(str(row["author"]) for row in manifest)
    cleaned_books = {(str(row["author"]), str(row["title"])) for row in cleaned}
    manifest_books = {(str(row["author"]), str(row["title"])) for row in manifest}
    expected_clean_paths = {
        Path(row["clean_txt_path"])
        for row in cleaned
        if row.get("exists") and row.get("clean_txt_path")
    }
    actual_clean_paths = set(CLEAN_TEXT_ROOT.rglob("*.clean.txt"))
    text_residues = [
        hit
        for path in sorted(actual_clean_paths)
        for hit in residue_hits(path)
    ][:20]
    chunk_counts, chunk_residues, alignment_errors = validate_chunks()
    expected_chunk_count = chunk_report.get("chunks_by_view", {}).get("clean")
    checks = {
        "manifest_is_nonempty": bool(manifest),
        "manifest_has_50_authors": len(authors) == 50,
        "every_author_has_at_least_3_books": bool(authors) and min(authors.values()) >= 3,
        "cleaned_manifest_matches_active_manifest": cleaned_books == manifest_books,
        "retired_corrupt_books_absent": not (cleaned_books & RETIRED_BOOKS),
        "clean_text_projection_exact": actual_clean_paths == expected_clean_paths,
        "clean_text_residue_free": not text_residues,
        "chunk_views_aligned": not alignment_errors and len(set(chunk_counts.values())) == 1,
        "chunk_count_matches_report": bool(chunk_counts)
        and expected_chunk_count is not None
        and set(chunk_counts.values()) == {int(expected_chunk_count)},
        "chunk_text_residue_free": not chunk_residues,
    }
    result = {
        "schema_version": 1,
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "counts": {
            "manifest_books": len(manifest),
            "authors": len(authors),
            "minimum_books_per_author": min(authors.values()) if authors else 0,
            "cleaned_books": len(cleaned_books),
            "clean_text_files": len(actual_clean_paths),
            "chunks_by_view": chunk_counts,
        },
        "errors": {
            "missing_clean_texts": sorted(str(path) for path in expected_clean_paths - actual_clean_paths),
            "stale_clean_texts": sorted(str(path) for path in actual_clean_paths - expected_clean_paths),
            "clean_text_residues": text_residues,
            "chunk_residues": chunk_residues,
            "chunk_alignment": alignment_errors,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
