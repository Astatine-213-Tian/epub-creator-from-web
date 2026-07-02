from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.crawl.snapshot import write_json


def load_glossary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"terms": {}, "style_notes": []}
    return json.loads(path.read_text(encoding="utf-8"))


def glossary_terms(glossary: dict[str, Any]) -> dict[str, str]:
    terms = glossary.get("terms", {})
    return {
        str(key): str(value.get("zh") or "")
        for key, value in terms.items()
        if isinstance(value, dict) and str(value.get("zh") or "").strip()
    }


def matched_terms(text: str, terms: dict[str, str]) -> dict[str, str]:
    lower = text.lower()
    return {source: target for source, target in terms.items() if source.lower() in lower}


def merge_glossary_candidates(
    *,
    glossary_path: Path,
    candidates_path: Path,
) -> dict[str, int]:
    glossary = load_glossary(glossary_path)
    glossary.setdefault("terms", {})
    terms = glossary["terms"]
    if not isinstance(terms, dict):
        raise ValueError(f"glossary terms must be an object: {glossary_path}")

    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    added = 0
    kept = 0
    skipped = 0
    for candidate in candidates:
        if not isinstance(candidate, dict):
            skipped += 1
            continue
        source = str(candidate.get("source") or "").strip()
        zh = str(candidate.get("zh") or "").strip()
        if not source or not zh:
            skipped += 1
            continue
        existing = terms.get(source)
        if isinstance(existing, dict) and str(existing.get("zh") or "").strip():
            kept += 1
            continue
        terms[source] = {
            "zh": zh,
            "note": str(candidate.get("reason") or "translation run candidate").strip(),
            "confidence": str(candidate.get("confidence") or "").strip(),
            "source_chunk": str(candidate.get("chunk_id") or "").strip(),
        }
        added += 1

    glossary_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(glossary_path, glossary)
    return {"added": added, "kept": kept, "skipped": skipped}
