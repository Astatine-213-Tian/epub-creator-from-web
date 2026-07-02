from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.crawl.snapshot import load_chapter, load_comments, load_manifest, snapshot_chapter_ids, write_json
from src.translation.codex_cli import run_prompt_with_codex
from src.translation.comments import selected_comment_notes
from src.translation.glossary import glossary_terms, load_glossary, matched_terms
from src.translation.prompt_builder import build_translation_prompt
from src.translation.vector_index import resolve_repo_path


FAILURE_MARKERS = (
    "无法翻译",
    "不能翻译",
    "未成年角色",
    "抱歉，我不能",
    "抱歉，我无法",
    "cannot translate",
    "can't translate",
    "unable to translate",
)


def is_failed_translation(value: str | None) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    lower = text.lower()
    return any(marker.lower() in lower for marker in FAILURE_MARKERS)


def _load_reference_matches(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "reference_matches.json"
    if not path.exists():
        return {"chapters": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _build_retry_prompt(payload: dict[str, Any], *, style_name: str) -> str:
    prompt = build_translation_prompt(payload, style_name=style_name)
    retry_note = """RETRY MODE:
- This is a single-paragraph retry after a wider context translation failed.
- Translate from this paragraph itself. Use comments, glossary, and references only for names, terms, and style.
- Do not infer a refusal from omitted neighboring paragraphs.
- If this paragraph itself still cannot be translated, set zh to an empty string.

"""
    return prompt.replace("INPUT JSON:\n", retry_note + "INPUT JSON:\n", 1)


def _chapter_lookup(snapshot_dir: Path, manifest: dict[str, Any]) -> dict[str, dict[int, dict[str, Any]]]:
    lookup: dict[str, dict[int, dict[str, Any]]] = {}
    for chapter_id in snapshot_chapter_ids(manifest):
        chapter = load_chapter(snapshot_dir, manifest, chapter_id)
        lookup[chapter_id] = {
            int(item["index"]): item
            for item in chapter.get("paragraphs") or []
        }
    return lookup


def retry_failed_translations(
    *,
    run_dir: Path,
    config: dict[str, Any],
    model: str | None = None,
    codex_bin: str = "codex",
    max_items: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser()
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    snapshot_dir = Path(manifest["snapshot_dir"]).expanduser()
    snapshot_manifest = load_manifest(snapshot_dir)
    chapter_items = _chapter_lookup(snapshot_dir, snapshot_manifest)
    reference_matches = _load_reference_matches(run_dir)
    glossary_path = config.get("glossary_path")
    glossary = load_glossary(resolve_repo_path(str(glossary_path))) if glossary_path else {}
    terms = glossary_terms(glossary)
    translation_config = config.get("translation", {})
    comment_budget = int(translation_config.get("retry_comment_char_budget") or 4000)
    style_name = str(config.get("style_name") or config.get("title") or "")

    prompt_dir = run_dir / "retry_prompts"
    output_dir = run_dir / "retry_outputs"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    comment_cache: dict[str, list[str]] = {}
    attempted = 0
    recovered = 0
    still_missing = 0
    skipped_existing = 0
    retry_items: list[dict[str, Any]] = []

    for chunk in manifest.get("chunks", []):
        chunk_id = str(chunk["chunk_id"])
        chapter_id = str(chunk["chapter_id"])
        output_path = run_dir / chunk["json_output_path"]
        if not output_path.exists():
            continue
        data = json.loads(output_path.read_text(encoding="utf-8"))
        changed = False
        for translation in data.get("translations") or []:
            index = int(translation["index"])
            if not is_failed_translation(str(translation.get("zh") or "")):
                continue
            if max_items is not None and attempted >= max_items:
                break
            paragraph = chapter_items.get(chapter_id, {}).get(index)
            if paragraph is None:
                continue

            retry_id = f"{chapter_id}_{index:04d}"
            retry_json_path = output_dir / f"{retry_id}.json"
            retry_raw_path = output_dir / f"{retry_id}.raw.txt"
            retry_prompt_path = prompt_dir / f"{retry_id}.txt"

            if chapter_id not in comment_cache:
                comments = load_comments(snapshot_dir, snapshot_manifest, chapter_id)
                comment_cache[chapter_id] = selected_comment_notes(
                    comments,
                    terms,
                    char_budget=comment_budget,
                )

            chapter_refs = (reference_matches.get("chapters") or {}).get(chapter_id) or {}
            paragraph_refs = {
                int(key): value
                for key, value in (chapter_refs.get("paragraph_refs") or {}).items()
            }
            ref_ids = paragraph_refs.get(index, [])
            bank_by_id = {
                str(entry.get("id")): entry
                for entry in chapter_refs.get("reference_bank") or []
                if entry.get("id")
            }
            reference_bank = [
                {"id": ref_id, "book": bank_by_id[ref_id].get("book"), "text": bank_by_id[ref_id].get("text")}
                for ref_id in ref_ids
                if ref_id in bank_by_id
            ]

            prompt_payload = {
                "book_title": config.get("title") or snapshot_manifest.get("title") or "",
                "author": config.get("author") or snapshot_manifest.get("author") or "",
                "chapter_id": chapter_id,
                "chapter_title": chunk.get("chapter_title") or "",
                "glossary": terms,
                "comment_notes": comment_cache[chapter_id],
                "reference_bank": reference_bank,
                "items": [
                    {
                        "index": index,
                        "english": paragraph.get("english") or "",
                        "current_zh": "",
                        "context_before": [],
                        "context_after": [],
                        "glossary_matches": matched_terms(str(paragraph.get("english") or ""), terms),
                        "reference_ids": ref_ids,
                    }
                ],
            }
            retry_prompt_path.write_text(
                _build_retry_prompt(prompt_payload, style_name=style_name),
                encoding="utf-8",
            )

            if retry_json_path.exists() and not overwrite:
                retry_data = json.loads(retry_json_path.read_text(encoding="utf-8"))
                skipped_existing += 1
            else:
                retry_data = run_prompt_with_codex(
                    prompt_path=retry_prompt_path,
                    raw_output_path=retry_raw_path,
                    json_output_path=retry_json_path,
                    model=model,
                    codex_bin=codex_bin,
                )
            attempted += 1

            retry_translations = retry_data.get("translations") or []
            retry_zh = ""
            for retry_translation in retry_translations:
                if int(retry_translation.get("index", -1)) == index:
                    retry_zh = str(retry_translation.get("zh") or "").strip()
                    break
            if retry_zh and not is_failed_translation(retry_zh):
                translation["zh"] = retry_zh
                recovered += 1
                changed = True
            else:
                translation["zh"] = ""
                still_missing += 1
                changed = True

            candidates = data.setdefault("glossary_candidates", [])
            for candidate in retry_data.get("glossary_candidates") or []:
                if isinstance(candidate, dict):
                    candidate = dict(candidate)
                    candidate["chunk_id"] = chunk_id
                    candidate["retry_id"] = retry_id
                    candidates.append(candidate)
            retry_items.append(
                {
                    "chunk_id": chunk_id,
                    "chapter_id": chapter_id,
                    "index": index,
                    "recovered": bool(retry_zh and not is_failed_translation(retry_zh)),
                }
            )

        if changed:
            write_json(output_path, data)
        if max_items is not None and attempted >= max_items:
            break

    summary = {
        "attempted": attempted,
        "recovered": recovered,
        "still_missing": still_missing,
        "skipped_existing": skipped_existing,
        "items": retry_items,
    }
    write_json(run_dir / "retry_summary.json", summary)
    return summary
