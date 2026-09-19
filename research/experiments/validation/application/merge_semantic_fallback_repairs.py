#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.crawler.snapshot import write_json
from src.translation.codex_cli import is_failed_translation


SCHEMA = "semantic_fallback_merge.v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_fallback_repairs(
    *,
    primary_run_dir: Path,
    fallback_run_dir: Path,
    fallback_model: str,
) -> dict[str, Any]:
    primary_outputs = primary_run_dir / "outputs"
    fallback_outputs = fallback_run_dir / "outputs"
    if not primary_outputs.exists() or not fallback_outputs.exists():
        raise FileNotFoundError("both primary and fallback output directories must exist")

    repairs: list[dict[str, Any]] = []
    inspected_chunks: list[str] = []
    for fallback_path in sorted(fallback_outputs.glob("*.json")):
        chunk_id = fallback_path.stem
        primary_path = primary_outputs / fallback_path.name
        if not primary_path.exists():
            raise FileNotFoundError(f"primary output is missing chunk {chunk_id}")
        inspected_chunks.append(chunk_id)
        primary_before_sha256 = file_sha256(primary_path)
        fallback_sha256 = file_sha256(fallback_path)
        primary = read_json(primary_path)
        fallback = read_json(fallback_path)
        fallback_by_index = {
            int(item["index"]): str(item.get("zh") or "").strip()
            for item in fallback.get("translations") or []
        }
        repaired_indexes: list[int] = []
        replacement_hashes: dict[str, str] = {}
        for item in primary.get("translations") or []:
            index = int(item["index"])
            if not is_failed_translation(str(item.get("zh") or "")):
                continue
            replacement = fallback_by_index.get(index, "")
            if is_failed_translation(replacement):
                raise ValueError(
                    f"fallback output remains empty/refusal for {chunk_id}:{index}"
                )
            item["zh"] = replacement
            repaired_indexes.append(index)
            replacement_hashes[str(index)] = hashlib.sha256(
                replacement.encode("utf-8")
            ).hexdigest()
        if repaired_indexes:
            write_json(primary_path, primary)
        repairs.append(
            {
                "chunk_id": chunk_id,
                "repaired_indexes": repaired_indexes,
                "replacement_sha256": replacement_hashes,
                "primary_before_sha256": primary_before_sha256,
                "primary_after_sha256": file_sha256(primary_path),
                "fallback_file_sha256": fallback_sha256,
            }
        )

    remaining: list[dict[str, int]] = []
    for primary_path in sorted(primary_outputs.glob("*.json")):
        for item in read_json(primary_path).get("translations") or []:
            if is_failed_translation(str(item.get("zh") or "")):
                remaining.append(
                    {"chunk_id": primary_path.stem, "index": int(item["index"])}
                )
    summary = {
        "schema_version": 1,
        "schema": SCHEMA,
        "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "primary_run_dir": str(primary_run_dir),
        "fallback_run_dir": str(fallback_run_dir),
        "fallback_model": fallback_model,
        "merge_policy": "replace_primary_empty_or_refusal_only",
        "inspected_chunks": inspected_chunks,
        "repaired_paragraph_count": sum(
            len(row["repaired_indexes"]) for row in repairs
        ),
        "remaining_failed_paragraph_count": len(remaining),
        "remaining_failed_paragraphs": remaining,
        "repairs": repairs,
    }
    write_json(primary_run_dir / "semantic_fallback_merge_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge only empty/refusal paragraphs from an isolated fallback run."
    )
    parser.add_argument("--primary-run-dir", type=Path, required=True)
    parser.add_argument("--fallback-run-dir", type=Path, required=True)
    parser.add_argument("--fallback-model", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = merge_fallback_repairs(
        primary_run_dir=args.primary_run_dir.expanduser().resolve(),
        fallback_run_dir=args.fallback_run_dir.expanduser().resolve(),
        fallback_model=args.fallback_model,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["remaining_failed_paragraph_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
