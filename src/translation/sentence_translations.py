from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.crawl.snapshot import (
    load_chapter,
    load_manifest,
    snapshot_chapter_ids,
    write_json,
)


@dataclass(frozen=True)
class SentenceTranslation:
    source: tuple[str, ...]
    zh: tuple[str, ...]
    note: str = ""
    confidence: str = ""


SOURCE_PUNCTUATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }
)


def sentence_translation_entries(
    glossary: dict[str, Any],
) -> list[SentenceTranslation]:
    raw_entries = glossary.get("sentence_translations") or []
    if not isinstance(raw_entries, list):
        raise ValueError("sentence_translations must be a list")

    entries: list[SentenceTranslation] = []
    for position, raw_entry in enumerate(raw_entries, 1):
        if not isinstance(raw_entry, dict):
            raise ValueError(
                f"sentence_translations entry {position} must be an object"
            )
        source = _segments(raw_entry.get("source"))
        zh = _segments(raw_entry.get("zh"))
        if not source or not zh:
            raise ValueError(
                f"sentence_translations entry {position} requires source and zh"
            )
        if len(source) != len(zh):
            raise ValueError(
                f"sentence_translations entry {position} must have matching "
                "source and zh segment counts"
            )
        entries.append(
            SentenceTranslation(
                source=source,
                zh=zh,
                note=str(raw_entry.get("note") or "").strip(),
                confidence=str(raw_entry.get("confidence") or "").strip(),
            )
        )
    return entries


def apply_sentence_translation_overrides(
    *,
    run_dir: Path,
    snapshot_dir: Path,
    glossary: dict[str, Any],
    strict: bool = True,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    snapshot_dir = snapshot_dir.expanduser().resolve()
    entries = sentence_translation_entries(glossary)
    run_manifest = _read_json(run_dir / "run_manifest.json")
    semantic_run_dir = _manifest_path(run_dir, run_manifest.get("semantic_run_dir"))
    snapshot_manifest = load_manifest(snapshot_dir)

    entry_reports = [
        {
            "source": list(entry.source),
            "zh": list(entry.zh),
            "note": entry.note,
            "confidence": entry.confidence,
            "occurrences": [],
        }
        for entry in entries
    ]
    unresolved: list[dict[str, Any]] = []
    changed_count = 0
    occurrence_count = 0

    for chapter_id in snapshot_chapter_ids(snapshot_manifest):
        chapter = load_chapter(snapshot_dir, snapshot_manifest, chapter_id)
        paragraphs = sorted(
            chapter.get("paragraphs") or [],
            key=lambda item: int(item["index"]),
        )
        translation_path = run_dir / "translations" / f"{chapter_id}.json"
        if not translation_path.exists():
            continue
        translation_data = _read_json(translation_path)
        translations = {
            int(item["index"]): item
            for item in translation_data.get("translations") or []
        }
        semantic_translations = _load_translation_map(semantic_run_dir, chapter_id)
        chapter_changed = False

        for entry_index, entry in enumerate(entries):
            occurrences = _find_occurrences(paragraphs, entry)
            desired_joined = "".join(entry.zh)
            for occurrence in occurrences:
                occurrence_count += 1
                indexes = occurrence["indexes"]
                mode = occurrence["mode"]
                changed = False

                if mode == "segmented":
                    missing_indexes = [
                        index for index in indexes if index not in translations
                    ]
                    if missing_indexes:
                        unresolved.append(
                            {
                                "entry": entry_index,
                                "chapter_id": chapter_id,
                                "indexes": indexes,
                                "reason": (
                                    "missing merged translation index(es) "
                                    f"{missing_indexes}"
                                ),
                            }
                        )
                        continue
                    for index, target in zip(indexes, entry.zh, strict=True):
                        item = translations[index]
                        if str(item.get("zh") or "") != target:
                            item["zh"] = target
                            changed = True
                    if changed:
                        changed_count += 1
                        chapter_changed = True
                    entry_reports[entry_index]["occurrences"].append(
                        {
                            "chapter_id": chapter_id,
                            "indexes": indexes,
                            "mode": "canonical_segments",
                            "changed": changed,
                        }
                    )
                    continue

                index = indexes[0]
                item = translations.get(index)
                if item is None:
                    unresolved.append(
                        {
                            "entry": entry_index,
                            "chapter_id": chapter_id,
                            "indexes": indexes,
                            "reason": f"missing merged translation index {index}",
                        }
                    )
                    continue

                if mode == "whole_paragraph":
                    replacement = desired_joined
                    applied_mode = "canonical_whole_paragraph"
                else:
                    current_zh = str(item.get("zh") or "")
                    semantic_zh = semantic_translations.get(index, "")
                    if desired_joined in current_zh:
                        replacement = current_zh
                        applied_mode = "already_canonical_embedded"
                    elif desired_joined in semantic_zh:
                        replacement = semantic_zh
                        applied_mode = "semantic_paragraph_restore"
                    else:
                        unresolved.append(
                            {
                                "entry": entry_index,
                                "chapter_id": chapter_id,
                                "indexes": indexes,
                                "reason": (
                                    "source sentence is embedded in a larger paragraph, "
                                    "but no canonical semantic paragraph is available"
                                ),
                            }
                        )
                        continue

                if str(item.get("zh") or "") != replacement:
                    item["zh"] = replacement
                    changed = True
                    changed_count += 1
                    chapter_changed = True
                entry_reports[entry_index]["occurrences"].append(
                    {
                        "chapter_id": chapter_id,
                        "indexes": indexes,
                        "mode": applied_mode,
                        "changed": changed,
                    }
                )

        if chapter_changed:
            write_json(translation_path, translation_data)

    unmatched_entries = [
        index
        for index, report in enumerate(entry_reports)
        if not report["occurrences"]
        and not any(item["entry"] == index for item in unresolved)
    ]
    summary = {
        "schema_version": 1,
        "session": "authoritative_sentence_translations",
        "run_dir": str(run_dir),
        "snapshot_dir": str(snapshot_dir),
        "entry_count": len(entries),
        "occurrence_count": occurrence_count,
        "changed_occurrence_count": changed_count,
        "unmatched_entries": unmatched_entries,
        "unresolved": unresolved,
        "entries": entry_reports,
    }
    write_json(
        run_dir / "sentence_translations" / "session_summary.json",
        summary,
    )
    if strict and unresolved:
        raise ValueError(
            "authoritative sentence translation protection could not resolve "
            f"{len(unresolved)} occurrence(s); see "
            f"{run_dir / 'sentence_translations' / 'session_summary.json'}"
        )
    return summary


def _segments(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, list):
        segments = tuple(str(item or "").strip() for item in value)
        return segments if all(segments) else ()
    return ()


def _normalize_source(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).translate(SOURCE_PUNCTUATION)
    return " ".join(normalized.casefold().split())


def _find_occurrences(
    paragraphs: list[dict[str, Any]],
    entry: SentenceTranslation,
) -> list[dict[str, Any]]:
    sources = [_normalize_source(item) for item in entry.source]
    joined_source = _normalize_source(" ".join(entry.source))
    normalized_paragraphs = [
        _normalize_source(str(item.get("english") or ""))
        for item in paragraphs
    ]
    occurrences: list[dict[str, Any]] = []
    claimed: set[tuple[int, ...]] = set()

    segment_count = len(sources)
    for start in range(0, len(paragraphs) - segment_count + 1):
        window = normalized_paragraphs[start : start + segment_count]
        if window != sources:
            continue
        indexes = tuple(
            int(item["index"])
            for item in paragraphs[start : start + segment_count]
        )
        occurrences.append({"indexes": list(indexes), "mode": "segmented"})
        claimed.add(indexes)

    for paragraph, normalized in zip(paragraphs, normalized_paragraphs, strict=True):
        index = int(paragraph["index"])
        if (index,) in claimed or not joined_source or joined_source not in normalized:
            continue
        mode = "whole_paragraph" if normalized == joined_source else "embedded"
        occurrences.append({"indexes": [index], "mode": mode})

    return occurrences


def _manifest_path(run_dir: Path, raw_path: Any) -> Path | None:
    if not raw_path:
        return None
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (run_dir / path).resolve()


def _load_translation_map(
    translations_run_dir: Path | None,
    chapter_id: str,
) -> dict[int, str]:
    if translations_run_dir is None:
        return {}
    path = translations_run_dir / "translations" / f"{chapter_id}.json"
    if not path.exists():
        return {}
    data = _read_json(path)
    return {
        int(item["index"]): str(item.get("zh") or "")
        for item in data.get("translations") or []
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
