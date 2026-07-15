from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Sequence

from src.crawl.snapshot import write_json
from src.core.output import repo_root


PROMPT_JSON_MARKER = "INPUT JSON:\n"
FALLBACK_ITEM_COUNT = 5
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
MODEL_CAPACITY_RE = re.compile(
    r"(?:unsupported_with_chatgpt_account|unsupported model|not supported when using codex|"
    r"model[_ -]?not[_ -]?found|model is not available|not available for this account|"
    r"capacity|usage limit|rate limit|rate_limit|quota|too many requests)",
    re.IGNORECASE,
)


class ModelCapacityError(RuntimeError):
    def __init__(self, model: str | None, evidence: str) -> None:
        self.model = model
        self.evidence = evidence[-2000:]
        super().__init__(
            f"model {model or 'default'} is at capacity or unavailable: {self.evidence}"
        )


def is_model_capacity_text(value: str) -> bool:
    return bool(MODEL_CAPACITY_RE.search(value or ""))


def _adaptive_model_namespace(model: str | None) -> str:
    value = str(model or "default").strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", value).strip("-") or "default"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{digest}"


def is_failed_translation(value: str | None) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    lower = text.lower()
    return any(marker.lower() in lower for marker in FAILURE_MARKERS)


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in Codex output")
    candidate = stripped[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        repaired = re.sub(
            r'("zh"\s*:\s*"[^"]*?)(?=\}\s*(?:,\s*\{"index"|\]\s*,\s*"glossary_candidates"))',
            r'\1"',
            candidate,
        )
        if repaired != candidate:
            return json.loads(repaired)
        raise


def _load_prompt_payload(prompt_path: Path) -> tuple[str, dict[str, Any]]:
    prompt = prompt_path.read_text(encoding="utf-8")
    if PROMPT_JSON_MARKER not in prompt:
        raise ValueError(f"prompt missing {PROMPT_JSON_MARKER.strip()}: {prompt_path}")
    payload = json.loads(prompt.split(PROMPT_JSON_MARKER, 1)[1])
    return prompt, payload


def _filtered_reference_bank(base_payload: dict[str, Any], items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ref_ids = {
        str(ref_id)
        for item in items
        for ref_id in (item.get("reference_ids") or [])
    }
    if not ref_ids:
        return []
    return [
        entry
        for entry in (base_payload.get("reference_bank") or [])
        if str(entry.get("id") or "") in ref_ids
    ]


def _subset_payload(
    base_payload: dict[str, Any],
    items: list[dict[str, Any]],
    *,
    fallback_kind: str,
) -> dict[str, Any]:
    payload = dict(base_payload)
    payload["items"] = items
    payload["reference_bank"] = _filtered_reference_bank(base_payload, items)
    payload["adaptive_fallback"] = {
        "kind": fallback_kind,
        "item_indexes": [int(item["index"]) for item in items],
    }
    return payload


def _prompt_for_subset(
    original_prompt: str,
    payload: dict[str, Any],
    *,
    note: str,
) -> str:
    header = original_prompt.split(PROMPT_JSON_MARKER, 1)[0]
    fallback_note = f"""ADAPTIVE FALLBACK MODE:
- {note}
- Translate only the JSON items in this smaller fallback payload.
- Keep the same output JSON shape and include every item index in this payload.
- If an individual item still cannot be translated, set zh to an empty string.

"""
    return header + fallback_note + PROMPT_JSON_MARKER + json.dumps(payload, ensure_ascii=False, indent=2)


def _translation_map(data: dict[str, Any]) -> dict[int, str]:
    translations: dict[int, str] = {}
    for item in data.get("translations") or []:
        if not isinstance(item, dict) or "index" not in item:
            continue
        translations[int(item["index"])] = str(item.get("zh") or "").strip()
    return translations


def _candidates_from(data: dict[str, Any], *, fallback_id: str | None = None) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for candidate in data.get("glossary_candidates") or []:
        if not isinstance(candidate, dict):
            continue
        item = dict(candidate)
        if fallback_id:
            item["adaptive_fallback_id"] = fallback_id
        candidates.append(item)
    return candidates


def _chunk_output_from_translations(
    base_payload: dict[str, Any],
    translations_by_index: dict[int, str],
    glossary_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "chapter_title": base_payload.get("chapter_title") or "",
        "translations": [
            {
                "index": int(item["index"]),
                "zh": translations_by_index.get(int(item["index"]), ""),
            }
            for item in base_payload.get("items") or []
        ],
        "glossary_candidates": glossary_candidates,
    }


def _normalize_chunk_output(data: dict[str, Any], base_payload: dict[str, Any]) -> dict[str, Any]:
    return _chunk_output_from_translations(
        base_payload,
        _translation_map(data),
        _candidates_from(data),
    )


def _run_subset_with_codex(
    *,
    original_prompt: str,
    base_payload: dict[str, Any],
    items: list[dict[str, Any]],
    fallback_id: str,
    note: str,
    prompt_dir: Path,
    output_dir: Path,
    model: str | None,
    codex_bin: str,
    timeout_seconds: int | None,
    overwrite: bool,
) -> dict[str, Any] | None:
    prompt_path = prompt_dir / f"{fallback_id}.txt"
    raw_output_path = output_dir / f"{fallback_id}.raw.txt"
    json_output_path = output_dir / f"{fallback_id}.json"
    if json_output_path.exists() and not overwrite:
        return json.loads(json_output_path.read_text(encoding="utf-8"))

    prompt_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _subset_payload(base_payload, items, fallback_kind=fallback_id)
    prompt_path.write_text(
        _prompt_for_subset(original_prompt, payload, note=note),
        encoding="utf-8",
    )
    try:
        return run_prompt_with_codex(
            prompt_path=prompt_path,
            raw_output_path=raw_output_path,
            json_output_path=json_output_path,
            model=model,
            codex_bin=codex_bin,
            timeout_seconds=timeout_seconds,
        )
    except ModelCapacityError:
        raise
    except Exception as exc:  # noqa: BLE001
        failure_path = output_dir / f"{fallback_id}.failure.txt"
        failure_path.write_text(str(exc), encoding="utf-8")
        return None


def _run_single_item_fallback(
    *,
    original_prompt: str,
    base_payload: dict[str, Any],
    item: dict[str, Any],
    fallback_id: str,
    note: str,
    prompt_dir: Path,
    output_dir: Path,
    model: str | None,
    codex_bin: str,
    timeout_seconds: int | None,
    overwrite: bool,
    summary: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    summary["paragraph_attempts"] += 1
    data = _run_subset_with_codex(
        original_prompt=original_prompt,
        base_payload=base_payload,
        items=[item],
        fallback_id=fallback_id,
        note=note,
        prompt_dir=prompt_dir,
        output_dir=output_dir,
        model=model,
        codex_bin=codex_bin,
        timeout_seconds=timeout_seconds,
        overwrite=overwrite,
    )
    if data is None:
        summary["paragraph_failures"] += 1
        summary["still_missing"] += 1
        return "", []

    index = int(item["index"])
    zh = _translation_map(data).get(index, "")
    candidates = _candidates_from(data, fallback_id=fallback_id)
    if is_failed_translation(zh):
        summary["still_missing"] += 1
        return "", candidates

    summary["recovered"] += 1
    return zh, candidates


def _local_windows_for_failed_indices(
    items: list[dict[str, Any]],
    failed_indices: list[int],
    *,
    max_items: int,
) -> list[list[dict[str, Any]]]:
    if not items or not failed_indices:
        return []

    positions_by_index = {
        int(item["index"]): position
        for position, item in enumerate(items)
    }
    failed_positions = sorted(
        positions_by_index[index]
        for index in failed_indices
        if index in positions_by_index
    )
    covered: set[int] = set()
    windows: list[list[dict[str, Any]]] = []

    while True:
        remaining = [position for position in failed_positions if position not in covered]
        if not remaining:
            break

        span_start = remaining[0]
        span_end = span_start
        for position in remaining[1:]:
            if position - span_start + 1 > max_items:
                break
            span_end = position

        start = span_start
        end = span_end + 1
        prefer_before = True
        while end - start < max_items and (start > 0 or end < len(items)):
            if prefer_before and start > 0:
                start -= 1
            elif end < len(items):
                end += 1
            elif start > 0:
                start -= 1
            prefer_before = not prefer_before

        windows.append(items[start:end])
        covered.update(
            position
            for position in failed_positions
            if start <= position < end
        )

    return windows


def _repair_failed_items_with_local_windows(
    *,
    original_prompt: str,
    base_payload: dict[str, Any],
    items: list[dict[str, Any]],
    translations_by_index: dict[int, str],
    glossary_candidates: list[dict[str, Any]],
    chunk_id: str,
    run_dir: Path,
    model: str | None,
    codex_bin: str,
    timeout_seconds: int | None,
    overwrite: bool,
    summary: dict[str, Any],
) -> None:
    failed_indices = [
        int(item["index"])
        for item in items
        if is_failed_translation(translations_by_index.get(int(item["index"]), ""))
    ]
    if not failed_indices:
        return

    items_by_index = {
        int(item["index"]): item
        for item in items
    }
    model_namespace = _adaptive_model_namespace(model)
    prompt_dir = run_dir / "adaptive_prompts" / model_namespace
    output_dir = run_dir / "adaptive_outputs" / model_namespace
    windows = _local_windows_for_failed_indices(
        items,
        failed_indices,
        max_items=FALLBACK_ITEM_COUNT,
    )

    for window_number, window_items in enumerate(windows, 1):
        window_failed_indices = [
            int(item["index"])
            for item in window_items
            if int(item["index"]) in failed_indices
            and is_failed_translation(translations_by_index.get(int(item["index"]), ""))
        ]
        if not window_failed_indices:
            continue

        fallback_id = f"{chunk_id}_w{window_number:03d}"
        summary["local_window_attempts"] += 1
        data = _run_subset_with_codex(
            original_prompt=original_prompt,
            base_payload=base_payload,
            items=window_items,
            fallback_id=fallback_id,
            note=(
                "The wider chunk returned empty/refusal translations. "
                "This local window keeps neighboring paragraphs together before single-paragraph fallback."
            ),
            prompt_dir=prompt_dir,
            output_dir=output_dir,
            model=model,
            codex_bin=codex_bin,
            timeout_seconds=timeout_seconds,
            overwrite=overwrite,
        )
        if data is None:
            summary["local_window_failures"] += 1
        else:
            glossary_candidates.extend(_candidates_from(data, fallback_id=fallback_id))
            window_translations = _translation_map(data)
            for index in window_failed_indices:
                zh = window_translations.get(index, "")
                if is_failed_translation(zh):
                    continue
                translations_by_index[index] = zh
                summary["recovered"] += 1
                summary["local_window_recovered"] += 1

        for index in window_failed_indices:
            if not is_failed_translation(translations_by_index.get(index, "")):
                continue
            single_id = f"{chunk_id}_i{index:04d}"
            zh, candidates = _run_single_item_fallback(
                original_prompt=original_prompt,
                base_payload=base_payload,
                item=items_by_index[index],
                fallback_id=single_id,
                note="The local fallback window still returned an empty/refusal translation for this paragraph.",
                prompt_dir=prompt_dir,
                output_dir=output_dir,
                model=model,
                codex_bin=codex_bin,
                timeout_seconds=timeout_seconds,
                overwrite=overwrite,
                summary=summary,
            )
            translations_by_index[index] = zh
            glossary_candidates.extend(candidates)


def _repair_failed_items(
    data: dict[str, Any],
    *,
    original_prompt: str,
    base_payload: dict[str, Any],
    chunk_id: str,
    run_dir: Path,
    model: str | None,
    codex_bin: str,
    timeout_seconds: int | None,
    overwrite: bool,
    summary: dict[str, Any],
) -> dict[str, Any]:
    normalized = _normalize_chunk_output(data, base_payload)
    translations_by_index = _translation_map(normalized)
    glossary_candidates = list(normalized.get("glossary_candidates") or [])
    items = list(base_payload.get("items") or [])
    _repair_failed_items_with_local_windows(
        original_prompt=original_prompt,
        base_payload=base_payload,
        items=items,
        translations_by_index=translations_by_index,
        glossary_candidates=glossary_candidates,
        chunk_id=chunk_id,
        run_dir=run_dir,
        model=model,
        codex_bin=codex_bin,
        timeout_seconds=timeout_seconds,
        overwrite=overwrite,
        summary=summary,
    )

    return _chunk_output_from_translations(base_payload, translations_by_index, glossary_candidates)


def _run_adaptive_chunk_fallback(
    *,
    original_prompt: str,
    base_payload: dict[str, Any],
    chunk_id: str,
    run_dir: Path,
    model: str | None,
    codex_bin: str,
    timeout_seconds: int | None,
    overwrite: bool,
    summary: dict[str, Any],
) -> dict[str, Any]:
    model_namespace = _adaptive_model_namespace(model)
    prompt_dir = run_dir / "adaptive_prompts" / model_namespace
    output_dir = run_dir / "adaptive_outputs" / model_namespace
    translations_by_index: dict[int, str] = {}
    glossary_candidates: list[dict[str, Any]] = []
    items = list(base_payload.get("items") or [])

    for piece_number, start in enumerate(range(0, len(items), FALLBACK_ITEM_COUNT), 1):
        piece = items[start : start + FALLBACK_ITEM_COUNT]
        fallback_id = f"{chunk_id}_p{piece_number:03d}"
        summary["fallback_piece_attempts"] += 1
        data = _run_subset_with_codex(
            original_prompt=original_prompt,
            base_payload=base_payload,
            items=piece,
            fallback_id=fallback_id,
            note="The original full chunk failed; this is a smaller fallback group.",
            prompt_dir=prompt_dir,
            output_dir=output_dir,
            model=model,
            codex_bin=codex_bin,
            timeout_seconds=timeout_seconds,
            overwrite=overwrite,
        )
        if data is None:
            summary["fallback_piece_failures"] += 1
            for item in piece:
                index = int(item["index"])
                single_id = f"{chunk_id}_i{index:04d}"
                zh, candidates = _run_single_item_fallback(
                    original_prompt=original_prompt,
                    base_payload=base_payload,
                    item=item,
                    fallback_id=single_id,
                    note="The original full chunk and smaller fallback group both failed; this is a single-paragraph fallback.",
                    prompt_dir=prompt_dir,
                    output_dir=output_dir,
                    model=model,
                    codex_bin=codex_bin,
                    timeout_seconds=timeout_seconds,
                    overwrite=overwrite,
                    summary=summary,
                )
                translations_by_index[index] = zh
                glossary_candidates.extend(candidates)
            continue

        glossary_candidates.extend(_candidates_from(data, fallback_id=fallback_id))
        piece_translations = _translation_map(data)
        for item in piece:
            index = int(item["index"])
            zh = piece_translations.get(index, "")
            if not is_failed_translation(zh):
                translations_by_index[index] = zh
                continue

            single_id = f"{chunk_id}_i{index:04d}"
            zh, candidates = _run_single_item_fallback(
                original_prompt=original_prompt,
                base_payload=base_payload,
                item=item,
                fallback_id=single_id,
                note="The smaller fallback group returned an empty/refusal translation for this paragraph.",
                prompt_dir=prompt_dir,
                output_dir=output_dir,
                model=model,
                codex_bin=codex_bin,
                timeout_seconds=timeout_seconds,
                overwrite=overwrite,
                summary=summary,
            )
            translations_by_index[index] = zh
            glossary_candidates.extend(candidates)

    return _chunk_output_from_translations(base_payload, translations_by_index, glossary_candidates)


def run_prompt_with_codex(
    *,
    prompt_path: Path,
    raw_output_path: Path,
    json_output_path: Path,
    model: str | None = None,
    codex_bin: str = "codex",
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    prompt = prompt_path.read_text(encoding="utf-8")
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        codex_bin,
        "exec",
        "--sandbox",
        "read-only",
        "--output-last-message",
        str(raw_output_path),
        "-C",
        str(repo_root()),
        "-",
    ]
    if model:
        cmd[2:2] = ["--model", model]
    result = subprocess.run(
        cmd,
        input=prompt,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_seconds,
    )
    log_path = raw_output_path.with_name(f"{raw_output_path.stem}.log")
    log_path.write_text(result.stdout or "", encoding="utf-8")
    if result.returncode != 0:
        if is_model_capacity_text(result.stdout or ""):
            raise ModelCapacityError(model, result.stdout or "")
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            output=result.stdout,
        )
    data = extract_json_object(raw_output_path.read_text(encoding="utf-8"))
    write_json(json_output_path, data)
    return data


def run_prompt_to_text(
    *,
    prompt_path: Path,
    output_path: Path,
    model: str | None = None,
    codex_bin: str = "codex",
    log_path: Path | None = None,
    timeout_seconds: int | None = None,
) -> str:
    prompt = prompt_path.read_text(encoding="utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        codex_bin,
        "exec",
        "--sandbox",
        "read-only",
        "--output-last-message",
        str(output_path),
        "-C",
        str(repo_root()),
        "-",
    ]
    if model:
        cmd[2:2] = ["--model", model]
    result = subprocess.run(
        cmd,
        input=prompt,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_seconds,
    )
    if log_path is None:
        log_path = output_path.with_name(f"{output_path.stem}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout or "", encoding="utf-8")
    if result.returncode != 0:
        if is_model_capacity_text(result.stdout or ""):
            raise ModelCapacityError(model, result.stdout or "")
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            output=result.stdout,
        )
    return output_path.read_text(encoding="utf-8")


def run_missing_prompts(
    *,
    run_dir: Path,
    model: str | None = None,
    codex_bin: str = "codex",
    chunk_ids: set[str] | None = None,
    overwrite: bool = False,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    summary: dict[str, Any] = {
        "completed": 0,
        "skipped": 0,
        "full_success": 0,
        "chunk_failures": 0,
        "fallback_piece_attempts": 0,
        "fallback_piece_failures": 0,
        "local_window_attempts": 0,
        "local_window_failures": 0,
        "local_window_recovered": 0,
        "paragraph_attempts": 0,
        "paragraph_failures": 0,
        "recovered": 0,
        "still_missing": 0,
        "items": [],
    }
    for chunk in manifest.get("chunks", []):
        chunk_id = str(chunk["chunk_id"])
        if chunk_ids is not None and chunk_id not in chunk_ids:
            continue
        prompt_path = run_dir / chunk["prompt_path"]
        raw_output_path = run_dir / chunk["raw_output_path"]
        json_output_path = run_dir / chunk["json_output_path"]
        original_prompt, base_payload = _load_prompt_payload(prompt_path)

        if json_output_path.exists() and not overwrite:
            before_windows = int(summary["local_window_attempts"])
            before_attempts = int(summary["paragraph_attempts"])
            before_recovered = int(summary["recovered"])
            before_missing = int(summary["still_missing"])
            data = json.loads(json_output_path.read_text(encoding="utf-8"))
            normalized = _normalize_chunk_output(data, base_payload)
            repaired = _repair_failed_items(
                data,
                original_prompt=original_prompt,
                base_payload=base_payload,
                chunk_id=chunk_id,
                run_dir=run_dir,
                model=model,
                codex_bin=codex_bin,
                timeout_seconds=timeout_seconds,
                overwrite=overwrite,
                summary=summary,
            )
            attempted_repairs = (
                int(summary["local_window_attempts"]) > before_windows
                or int(summary["paragraph_attempts"]) > before_attempts
            )
            if repaired != normalized:
                write_json(json_output_path, repaired)
            if attempted_repairs:
                summary["completed"] += 1
                summary["items"].append(
                    {
                        "chunk_id": chunk_id,
                        "status": "repaired_existing",
                        "local_window_attempts": int(summary["local_window_attempts"]) - before_windows,
                        "paragraph_attempts": int(summary["paragraph_attempts"]) - before_attempts,
                        "recovered": int(summary["recovered"]) - before_recovered,
                        "still_missing": int(summary["still_missing"]) - before_missing,
                    }
                )
                print(f"repaired failed item(s) in chunk {chunk_id}")
            else:
                summary["skipped"] += 1
            continue

        used_adaptive_fallback = False
        before_piece_attempts = int(summary["fallback_piece_attempts"])
        before_window_attempts = int(summary["local_window_attempts"])
        before_paragraph_attempts = int(summary["paragraph_attempts"])
        before_recovered = int(summary["recovered"])
        before_missing = int(summary["still_missing"])
        try:
            data = run_prompt_with_codex(
                prompt_path=prompt_path,
                raw_output_path=raw_output_path,
                json_output_path=json_output_path,
                model=model,
                codex_bin=codex_bin,
                timeout_seconds=timeout_seconds,
            )
            summary["full_success"] += 1
            final_data = _repair_failed_items(
                data,
                original_prompt=original_prompt,
                base_payload=base_payload,
                chunk_id=chunk_id,
                run_dir=run_dir,
                model=model,
                codex_bin=codex_bin,
                timeout_seconds=timeout_seconds,
                overwrite=overwrite,
                summary=summary,
            )
            write_json(json_output_path, final_data)
            print(f"translated chunk {chunk_id}")
        except ModelCapacityError:
            raise
        except Exception as exc:  # noqa: BLE001
            used_adaptive_fallback = True
            summary["chunk_failures"] += 1
            failure_path = raw_output_path.with_name(f"{raw_output_path.stem}.failure.txt")
            failure_path.write_text(str(exc), encoding="utf-8")
            print(f"chunk {chunk_id} failed; running adaptive fallback")
            final_data = _run_adaptive_chunk_fallback(
                original_prompt=original_prompt,
                base_payload=base_payload,
                chunk_id=chunk_id,
                run_dir=run_dir,
                model=model,
                codex_bin=codex_bin,
                timeout_seconds=timeout_seconds,
                overwrite=overwrite,
                summary=summary,
            )
            write_json(json_output_path, final_data)
            print(f"translated chunk {chunk_id} via adaptive fallback")

        summary["completed"] += 1
        summary["items"].append(
            {
                "chunk_id": chunk_id,
                "status": "adaptive_fallback" if used_adaptive_fallback else "full_chunk",
                "fallback_piece_attempts": int(summary["fallback_piece_attempts"]) - before_piece_attempts,
                "local_window_attempts": int(summary["local_window_attempts"]) - before_window_attempts,
                "paragraph_attempts": int(summary["paragraph_attempts"]) - before_paragraph_attempts,
                "recovered": int(summary["recovered"]) - before_recovered,
                "still_missing": int(summary["still_missing"]) - before_missing,
            }
        )

    write_json(run_dir / "adaptive_summary.json", summary)
    return summary


def _merge_run_summaries(
    summaries: Sequence[dict[str, Any]],
    *,
    requested_models: Sequence[str],
    unavailable_models: dict[str, str],
) -> dict[str, Any]:
    if not summaries:
        raise ValueError("at least one run summary is required")
    merged = dict(summaries[-1])
    counter_fields = (
        "completed",
        "full_success",
        "chunk_failures",
        "fallback_piece_attempts",
        "fallback_piece_failures",
        "local_window_attempts",
        "local_window_failures",
        "local_window_recovered",
        "paragraph_attempts",
        "paragraph_failures",
        "recovered",
        "still_missing",
    )
    for field in counter_fields:
        merged[field] = sum(int(summary.get(field) or 0) for summary in summaries)
    merged["items"] = [
        item for summary in summaries for item in (summary.get("items") or [])
    ]
    merged["requested_model_order"] = list(requested_models)
    merged["unavailable_models"] = unavailable_models
    merged["effective_model"] = next(
        (
            str(summary["effective_model"])
            for summary in reversed(summaries)
            if summary.get("effective_model")
        ),
        None,
    )
    return merged


def run_missing_prompts_with_model_fallback(
    *,
    run_dir: Path,
    models: Sequence[str],
    codex_bin: str = "codex",
    chunk_ids: set[str] | None = None,
    overwrite: bool = False,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    requested = list(dict.fromkeys(str(model) for model in models if str(model)))
    if not requested:
        raise ValueError("at least one Codex model is required")
    summaries: list[dict[str, Any]] = []
    unavailable: dict[str, str] = {}
    for model in requested:
        try:
            summary = run_missing_prompts(
                run_dir=run_dir,
                model=model,
                codex_bin=codex_bin,
                chunk_ids=chunk_ids,
                overwrite=overwrite,
                timeout_seconds=timeout_seconds,
            )
        except ModelCapacityError as exc:
            unavailable[model] = exc.evidence
            continue
        summary["effective_model"] = model
        summaries.append(summary)
        merged = _merge_run_summaries(
            summaries,
            requested_models=requested,
            unavailable_models=unavailable,
        )
        write_json(run_dir / "adaptive_summary.json", merged)
        return merged
    raise RuntimeError(f"all requested Codex models are unavailable: {unavailable}")
