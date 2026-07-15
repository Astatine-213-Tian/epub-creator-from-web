from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.crawl.snapshot import load_chapter, load_comments, load_manifest, snapshot_chapter_ids, write_json
from src.core.output import repo_root
from src.translation.comments import selected_comment_notes
from src.translation.glossary import glossary_terms, matched_terms


def chunked(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    return [items[index : index + size] for index in range(0, len(items), size)]


def load_existing_translations(path: Path | None, chapter_id: str) -> dict[int, str]:
    if path is None:
        return {}
    file_path = path / f"{chapter_id}.json"
    if not file_path.exists():
        return {}
    data = json.loads(file_path.read_text(encoding="utf-8"))
    return {
        int(item["index"]): str(item.get("zh") or "")
        for item in data.get("translations", [])
    }


def build_translation_prompt(payload: dict[str, Any], *, style_name: str = "") -> str:
    label = f" for {style_name}" if style_name else ""
    header = f"""You are producing a semantic Chinese draft from an English fantasy/danmei novel{label}.

SCOPE:
- Translate only the JSON items in INPUT JSON.
- Output ONLY valid JSON, no markdown, no explanation.
- JSON shape exactly: {{"chapter_title": string, "translations": [{{"index": number, "zh": string}}], "glossary_candidates": [{{"source": string, "zh": string, "reason": string, "confidence": "high|medium|low"}}]}}
- Include one translation item for every input item index, in the same order.
- Keep paragraph boundaries. Do not merge or split paragraphs.
- Do not include the English original in zh.
- If you cannot translate an item, set zh to an empty string. Do not write a refusal, placeholder, or explanation.

SEMANTIC DRAFT GOAL:
- Treat the English as the semantic source.
- Produce plain, natural Simplified Chinese that is easy to verify against the English.
- This pass is not the author style-transfer pass. Do not imitate retrieved author prose, add literary flourish, or optimize for style.
- Preserve meaning, continuity, names, worldbuilding, paragraph boundaries, speaker turns, numbers, and event order.
- Do not add motives, lore, erotic intensity, injuries, jokes, or setting details absent from the source.

CONTEXT RULES:
- source_context describes the project-level translation premise.
- context_before/context_after are only for understanding local continuity.
- comment_notes contains Chinese comments/replies from the chapter, preserving commenter names.
- Treat comments by Risk, Via Lactea Press Inc., or the creator as authoritative for names, poems, and worldbuilding.
- If a comment gives an explicit Chinese translation for a quoted line or poem, use that wording unless it conflicts with the glossary.
- Preserve glossary terms exactly unless the English context proves a different sense.
- Use Chinese dialogue punctuation.

INPUT JSON:
"""
    return header + json.dumps(payload, ensure_ascii=False, indent=2)


def prepare_prompts(
    *,
    snapshot_dir: Path,
    run_dir: Path,
    config: dict[str, Any],
    glossary: dict[str, Any],
) -> dict[str, Any]:
    manifest = load_manifest(snapshot_dir)
    terms = glossary_terms(glossary)
    translation_config = config.get("translation", {})
    source_context = str(config.get("source_context") or translation_config.get("source_context") or "")
    chunk_size = int(translation_config.get("chunk_size") or 30)
    context_window = int(translation_config.get("context_paragraphs") or 5)
    comment_budget = int(translation_config.get("comment_char_budget") or 10000)
    use_existing_in_prompt = bool(translation_config.get("use_existing_translations_in_prompt"))
    existing_dir = (
        (
            Path(str(config["existing_translations_dir"])).expanduser().resolve()
            if Path(str(config["existing_translations_dir"])).expanduser().is_absolute()
            else (repo_root() / str(config["existing_translations_dir"])).resolve()
        )
        if use_existing_in_prompt and config.get("existing_translations_dir")
        else None
    )

    prompt_dir = run_dir / "prompts"
    output_dir = run_dir / "outputs"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[dict[str, Any]] = []

    for chapter_id in snapshot_chapter_ids(manifest):
        chapter = load_chapter(snapshot_dir, manifest, chapter_id)
        paragraphs = sorted(chapter.get("paragraphs") or [], key=lambda item: int(item["index"]))
        comments = load_comments(snapshot_dir, manifest, chapter_id)
        comment_notes = selected_comment_notes(comments, terms, char_budget=comment_budget)
        current = load_existing_translations(existing_dir, chapter_id)

        by_index = {int(item["index"]): item for item in paragraphs}
        for chunk_number, chunk in enumerate(chunked(paragraphs, chunk_size), 1):
            chunk_id = f"{chapter_id}_{chunk_number:03d}"
            input_items = []
            for item in chunk:
                idx = int(item["index"])
                before = [
                    by_index[pos]["english"]
                    for pos in range(max(0, idx - context_window), idx)
                    if pos in by_index
                ]
                after = [
                    by_index[pos]["english"]
                    for pos in range(idx + 1, idx + context_window + 1)
                    if pos in by_index
                ]
                input_items.append(
                    {
                        "index": idx,
                        "english": item["english"],
                        "current_zh": current.get(idx, ""),
                        "context_before": before,
                        "context_after": after,
                        "glossary_matches": matched_terms(str(item["english"]), terms),
                    }
                )
            prompt_payload = {
                "pass": "semantic_draft",
                "book_title": config.get("title") or manifest.get("title") or "",
                "author": config.get("author") or manifest.get("author") or "",
                "source_context": source_context,
                "chapter_id": chapter_id,
                "chapter_title": chapter.get("title") or "",
                "glossary": terms,
                "comment_notes": comment_notes,
                "items": input_items,
            }
            prompt = build_translation_prompt(
                prompt_payload,
                style_name=str(config.get("style_name") or config.get("title") or ""),
            )
            prompt_path = prompt_dir / f"{chunk_id}.txt"
            prompt_path.write_text(prompt, encoding="utf-8")
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "chapter_id": chapter_id,
                    "chapter_title": chapter.get("title") or "",
                    "indexes": [int(item["index"]) for item in chunk],
                    "prompt_path": str(prompt_path.relative_to(run_dir)),
                    "raw_output_path": f"outputs/{chunk_id}.raw.txt",
                    "json_output_path": f"outputs/{chunk_id}.json",
                }
            )

    run_manifest = {
        "schema_version": 2,
        "pass": "semantic_draft",
        "snapshot_dir": str(snapshot_dir),
        "config": config,
        "chunks": chunks,
    }
    write_json(run_dir / "run_manifest.json", run_manifest)
    return run_manifest
