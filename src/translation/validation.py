from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.crawl.snapshot import clean_text, write_json
from src.translation.codex_cli import extract_json_object


def _load_chunk_output(path: Path, raw_path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if raw_path.exists():
        data = extract_json_object(raw_path.read_text(encoding="utf-8"))
        write_json(path, data)
        return data
    raise FileNotFoundError(f"missing translation output: {path}")


def validate_and_merge(run_dir: Path, *, allow_missing: bool = False) -> dict[str, Any]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    translations_by_chapter: dict[str, list[dict[str, Any]]] = {}
    glossary_candidates: list[dict[str, Any]] = []
    chunk_summaries: list[dict[str, Any]] = []

    for chunk in manifest.get("chunks", []):
        chunk_id = str(chunk["chunk_id"])
        expected = [int(index) for index in chunk["indexes"]]
        json_path = run_dir / chunk["json_output_path"]
        raw_path = run_dir / chunk["raw_output_path"]
        data = _load_chunk_output(json_path, raw_path)
        translations = data.get("translations") or []
        actual = [int(item["index"]) for item in translations]
        if actual != expected:
            raise ValueError(f"{chunk_id}: index mismatch expected {expected} got {actual}")
        missing = [int(item["index"]) for item in translations if not str(item.get("zh") or "").strip()]
        if missing and not allow_missing:
            raise ValueError(f"{chunk_id}: empty translations for indexes {missing}")
        normalized = [
            {"index": int(item["index"]), "zh": clean_text(str(item.get("zh") or ""))}
            for item in translations
        ]
        translations_by_chapter.setdefault(str(chunk["chapter_id"]), []).extend(normalized)
        for candidate in data.get("glossary_candidates") or []:
            if isinstance(candidate, dict):
                candidate = dict(candidate)
                candidate["chunk_id"] = chunk_id
                glossary_candidates.append(candidate)
        chunk_summaries.append(
            {
                "chunk_id": chunk_id,
                "chapter_id": chunk["chapter_id"],
                "count": len(normalized),
                "empty": missing,
            }
        )

    translations_dir = run_dir / "translations"
    translations_dir.mkdir(parents=True, exist_ok=True)
    for chapter_id, items in translations_by_chapter.items():
        items.sort(key=lambda item: int(item["index"]))
        title = next(
            (chunk["chapter_title"] for chunk in manifest["chunks"] if chunk["chapter_id"] == chapter_id),
            "",
        )
        write_json(
            translations_dir / f"{chapter_id}.json",
            {"chapter_id": chapter_id, "chapter_title": title, "translations": items},
        )

    summary = {
        "chunks": chunk_summaries,
        "chapter_count": len(translations_by_chapter),
        "glossary_candidate_count": len(glossary_candidates),
    }
    write_json(run_dir / "glossary_candidates.json", glossary_candidates)
    write_json(run_dir / "validation_summary.json", summary)
    return summary
