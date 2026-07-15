from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from src.crawl.snapshot import clean_text, load_chapter, load_manifest, write_json
from src.core.output import repo_root
from src.translation.codex_cli import (
    ModelCapacityError,
    extract_json_object,
    run_prompt_to_text,
)
from src.translation.style_transfer import (
    is_author_style_transfer_run,
    validate_style_transfer_provenance,
    validate_style_transfer_source_inputs,
)
from src.translation.validation import validate_and_merge


CJK_RE = re.compile(r"[\u4e00-\u9fff]")
FAILURE_MARKERS = ("无法翻译", "抱歉", "cannot comply", "can't comply", "PLACEHOLDER", "TODO")

DEFAULT_MAX_CANDIDATES = 30
DEFAULT_BATCH_SIZE = 8
DEFAULT_MIN_CONFIDENCE = "medium"

CONFIDENCE_RANK = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
}

IMPOSSIBLE_SOURCE = ["不能", "无法", "不敢", "不会", "不可", "绝不", "再也不", "不许", "不准", "不得"]
IMPOSSIBLE_STYLE = IMPOSSIBLE_SOURCE + [
    "没法",
    "无从",
    "不至于",
    "不愿",
    "从不",
    "未能",
    "免",
    "藏也藏不住",
    "怎么也",
    "得不到",
]
NECESSITY_SOURCE = ["必须", "不得不", "只能", "只好", "只准", "注定"]
NECESSITY_STYLE = NECESSITY_SOURCE + ["要", "需", "得以", "方能", "被迫", "只得", "只问"]
WARNING_SOURCE = ["提醒", "警告", "告诫", "警示", "劝诫"]
WARNING_STYLE = WARNING_SOURCE + ["叮嘱"]
ORDER_SOURCE = ["命令", "下令", "吩咐", "嘱咐", "指示"]
ORDER_STYLE = ORDER_SOURCE + ["传令", "奉命", "奉神之命", "命人"]
REFUSAL_SOURCE = ["拒绝", "阻止", "不肯", "不愿"]
REFUSAL_STYLE = REFUSAL_SOURCE + ["拦", "不许", "不让"]
RECIPROCAL_TERMS = ["彼此", "互相", "相互", "两人", "二人", "十指相扣", "交缠"]
ACTION_TERMS = [
    "按在身下",
    "压在身下",
    "按倒",
    "压下",
    "压住",
    "收紧",
    "搂紧",
    "箍住",
    "抱紧",
    "连着",
    "连接",
    "接在",
    "钻进",
    "冲进",
    "跃入",
    "跃进",
    "投身",
    "剥掉",
    "烙上",
    "刺青",
    "刮不掉",
]

NUMERIC_RE = re.compile(
    r"(?:\d+(?:\.\d+)?|[二三四五六七八九十百千万两半]{1,4}|一[百千万十])"
    r"(?:年|岁|天|日|月|时|枚|名|位|人|条|只|匹|艘|座|次|遍|里|尺|寸|斤|两|银币|万人)?"
    r"|[二三四五六七八九十百千万两半]{1,4}"
    r"(?:年|岁|天|日|月|时|枚|名|位|人|条|只|匹|艘|座|次|遍|里|尺|寸|斤|两|银币|万人)"
)

DEFAULT_IGNORED_GLOSSARY_TERMS = {
    "白色雄鹿",
    "格兰情人",
    "大学士坎多",
    "大将军隆让",
    "春泽城",
    "永恒之门",
    "神明降言",
}


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return repo_root() / candidate


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _qa_config_sha256(config: dict[str, Any]) -> str:
    maintained = {key: value for key, value in config.items() if key != "_config_path"}
    glossary_path = config.get("glossary_path")
    glossary_file_sha256 = ""
    if glossary_path:
        path = resolve_path(str(glossary_path))
        if path.exists():
            glossary_file_sha256 = _file_sha256(path)
    return _sha256_json(
        {
            "config": maintained,
            "glossary_file_sha256": glossary_file_sha256,
        }
    )


def _style_run_binding_sha256(run_dir: Path) -> str:
    manifest = _load_json(run_dir / "run_manifest.json")
    transfer = manifest.get("style_transfer") or {}
    return _sha256_json(
        {
            "pass": manifest.get("pass"),
            "snapshot_dir": manifest.get("snapshot_dir"),
            "semantic_run_dir": manifest.get("semantic_run_dir"),
            "style_transfer": {
                key: transfer.get(key)
                for key in (
                    "schema",
                    "method_id",
                    "intensity",
                    "block_size",
                    "asset_file_sha256",
                    "asset_content_sha256",
                    "component_hashes",
                    "prompt_sha256",
                    "schema_sha256",
                    "semantic_manifest_sha256",
                    "snapshot_manifest_sha256",
                )
            },
            "chunks": [
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "request_sha256": chunk.get("request_sha256"),
                    "english_sha256": chunk.get("english_sha256"),
                    "neutral_zh_sha256": chunk.get("neutral_zh_sha256"),
                }
                for chunk in manifest.get("chunks") or []
            ],
        }
    )


def translation_state_sha256(run_dir: Path) -> str:
    run_dir = run_dir.expanduser()
    manifest = _load_json(run_dir / "run_manifest.json")
    state: list[dict[str, str]] = []
    for chunk in manifest.get("chunks") or []:
        output_path = run_dir / str(chunk["json_output_path"])
        if not output_path.exists():
            raise FileNotFoundError(f"style output missing from QA state: {output_path}")
        state.append(
            {
                "chunk_id": str(chunk["chunk_id"]),
                "output_sha256": _file_sha256(output_path),
            }
        )
    encoded = json.dumps(
        state,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def semantic_qa_summary_is_current(
    run_dir: Path,
    summary_path: Path,
    config: dict[str, Any] | None = None,
) -> bool:
    if not summary_path.exists():
        return False
    try:
        summary = _load_json(summary_path)
        result = summary.get("result_summary") or {}
        if bool(summary.get("dry_run")):
            return False
        if int(result.get("failure_count") or 0) > 0:
            return False
        if int(result.get("rejected_count") or 0) > 0:
            return False
        if int(result.get("unresolved_true_loss_count") or 0) > 0:
            return False
        expected = str(summary.get("output_translation_state_sha256") or "")
        if not expected or expected != translation_state_sha256(run_dir):
            return False
        provenance = summary.get("provenance") or {}
        if provenance.get("schema") != "semantic_qa_provenance.v1" or config is None:
            return False
        if provenance.get("qa_config_sha256") != _qa_config_sha256(config):
            return False
        semantic_config = config.get("semantic_compression") or {}
        settings = summary.get("settings") or {}
        if int(settings.get("max_candidates") or 0) != int(
            semantic_config.get("max_candidates") or DEFAULT_MAX_CANDIDATES
        ):
            return False
        if int(settings.get("batch_size") or 0) != int(
            semantic_config.get("batch_size") or DEFAULT_BATCH_SIZE
        ):
            return False
        if bool(settings.get("auto_repair")) != bool(
            semantic_config.get("auto_repair")
        ):
            return False
        if str(settings.get("min_auto_apply_confidence") or "") != str(
            semantic_config.get("min_auto_apply_confidence")
            or DEFAULT_MIN_CONFIDENCE
        ):
            return False
        if provenance.get("review_contract_sha256") != _sha256_json(
            _review_contract()
        ):
            return False
        if provenance.get("style_run_binding_sha256") != _style_run_binding_sha256(
            run_dir
        ):
            return False
        source_state = validate_style_transfer_source_inputs(run_dir)
        if provenance.get("style_source_state_sha256") != source_state.get(
            "state_sha256"
        ):
            return False
        candidate_path = Path(str(provenance.get("candidate_report_path") or ""))
        if (
            not candidate_path.exists()
            or provenance.get("candidate_report_sha256")
            != _file_sha256(candidate_path)
        ):
            return False
        review_artifacts = provenance.get("review_artifacts") or []
        if int(provenance.get("review_artifact_count") or 0) != len(
            review_artifacts
        ):
            return False
        for artifact in review_artifacts:
            prompt_path = Path(str(artifact.get("prompt_path") or ""))
            result_path = Path(str(artifact.get("result_path") or ""))
            if not prompt_path.exists() or not result_path.exists():
                return False
            if artifact.get("prompt_sha256") != _file_sha256(prompt_path):
                return False
            if artifact.get("result_sha256") != _file_sha256(result_path):
                return False
        return True
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _cjk_len(text: str) -> int:
    return len(CJK_RE.findall(text or ""))


def _translation_map(path: Path) -> dict[int, str]:
    data = _load_json(path)
    return {int(item["index"]): str(item.get("zh") or "") for item in data.get("translations") or []}


def _english_map(
    snapshot_dir: Path,
    snapshot_manifest: dict[str, Any],
    chapter_id: str,
) -> dict[int, str]:
    data = load_chapter(snapshot_dir, snapshot_manifest, chapter_id)
    return {int(item["index"]): str(item.get("english") or "") for item in data.get("paragraphs") or []}


def _glossary_terms(config: dict[str, Any], compression_config: dict[str, Any]) -> list[dict[str, str]]:
    glossary_path = config.get("glossary_path")
    if not glossary_path:
        return []
    path = resolve_path(str(glossary_path))
    if not path.exists():
        return []
    glossary = _load_json(path)
    ignored = set(DEFAULT_IGNORED_GLOSSARY_TERMS)
    ignored.update(str(item) for item in compression_config.get("ignored_glossary_terms") or [])
    min_chars = int(compression_config.get("min_glossary_term_chars") or 3)
    terms: dict[str, dict[str, str]] = {}
    for source, entry in (glossary.get("terms") or {}).items():
        if isinstance(entry, dict):
            target = str(entry.get("zh") or entry.get("target") or "")
            confidence = str(entry.get("confidence") or "")
        else:
            target = str(entry or "")
            confidence = ""
        if not target or target in ignored or len(target) < min_chars or not CJK_RE.search(target):
            continue
        terms[target] = {"zh": target, "source": str(source), "confidence": confidence}
    return sorted(terms.values(), key=lambda item: len(item["zh"]), reverse=True)


def _term_present_loose(term: str, text: str) -> bool:
    if term in text:
        return True
    if len(term) >= 4:
        chars = [char for char in term if CJK_RE.match(char)]
        if chars and all(char in text for char in chars):
            return True
    return False


def _glossary_loss(
    *,
    semantic_zh: str,
    style_zh: str,
    terms: list[dict[str, str]],
) -> list[dict[str, str]]:
    losses: list[dict[str, str]] = []
    for item in terms:
        term = item["zh"]
        if term in semantic_zh and not _term_present_loose(term, style_zh):
            losses.append(item)
    return losses[:6]


def _number_loss(semantic_zh: str, style_zh: str) -> list[str]:
    numbers = sorted(set(NUMERIC_RE.findall(semantic_zh)), key=len, reverse=True)
    return [number for number in numbers if number and number not in style_zh]


def _has_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def _category_loss(
    *,
    semantic_zh: str,
    style_zh: str,
    source_terms: list[str],
    style_terms: list[str],
) -> list[str]:
    if not _has_any(semantic_zh, source_terms) or _has_any(style_zh, style_terms):
        return []
    return [term for term in source_terms if term in semantic_zh]


def _candidate_from_pair(
    *,
    ref: str,
    chapter_id: str,
    index: int,
    english: str,
    semantic_zh: str,
    style_zh: str,
    terms: list[dict[str, str]],
) -> dict[str, Any] | None:
    ratio = _cjk_len(style_zh) / max(_cjk_len(semantic_zh), 1)
    signals: list[str] = []
    details: dict[str, Any] = {}
    score = 0

    glossary = _glossary_loss(semantic_zh=semantic_zh, style_zh=style_zh, terms=terms)
    if glossary:
        signals.append("specific_glossary_loss")
        details["specific_glossary_loss"] = glossary
        score += 4

    numbers = _number_loss(semantic_zh, style_zh)
    if numbers:
        signals.append("concrete_number_loss")
        details["concrete_number_loss"] = numbers[:6]
        score += 3

    categories = [
        ("impossibility_negation_loss", IMPOSSIBLE_SOURCE, IMPOSSIBLE_STYLE, 4),
        ("necessity_modal_loss", NECESSITY_SOURCE, NECESSITY_STYLE, 3),
        ("warning_function_loss", WARNING_SOURCE, WARNING_STYLE, 3),
        ("order_function_loss", ORDER_SOURCE, ORDER_STYLE, 3),
        ("refusal_function_loss", REFUSAL_SOURCE, REFUSAL_STYLE, 2),
        ("reciprocal_loss", RECIPROCAL_TERMS, RECIPROCAL_TERMS, 2),
        ("action_direction_loss", ACTION_TERMS, ACTION_TERMS, 3),
    ]
    for signal_name, source_terms, style_terms, weight in categories:
        losses = _category_loss(
            semantic_zh=semantic_zh,
            style_zh=style_zh,
            source_terms=source_terms,
            style_terms=style_terms,
        )
        if losses:
            signals.append(signal_name)
            details[signal_name] = losses
            score += weight

    if not signals:
        return None

    if _cjk_len(semantic_zh) >= 24 and ratio < 0.85:
        signals.append("low_ratio_context")
        details["ratio"] = round(ratio, 3)
        score += 1

    return {
        "ref": ref,
        "chapter_id": chapter_id,
        "index": index,
        "score": score,
        "ratio": round(ratio, 3),
        "signals": signals,
        "details": details,
        "english": english,
        "semantic_zh": semantic_zh,
        "style_zh": style_zh,
    }


def detect_semantic_compression_candidates(
    *,
    run_dir: Path,
    config: dict[str, Any],
    max_candidates: int | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser()
    if not is_author_style_transfer_run(run_dir):
        raise ValueError(
            "semantic compression QA only runs on author_style_transfer run directories"
        )
    manifest = _load_json(run_dir / "run_manifest.json")
    semantic_run_dir = resolve_path(str(manifest["semantic_run_dir"]))
    snapshot_dir = resolve_path(str(manifest["snapshot_dir"]))
    compression_config = config.get("semantic_compression") or {}
    terms = _glossary_terms(config, compression_config)
    snapshot_manifest = load_manifest(snapshot_dir)
    candidates: list[dict[str, Any]] = []
    chapter_ids = sorted({str(chunk["chapter_id"]) for chunk in manifest.get("chunks") or []})
    for chapter_id in chapter_ids:
        english_by_index = _english_map(snapshot_dir, snapshot_manifest, chapter_id)
        semantic_by_index = _translation_map(semantic_run_dir / "translations" / f"{chapter_id}.json")
        style_by_index = _translation_map(run_dir / "translations" / f"{chapter_id}.json")
        for index, style_zh in sorted(style_by_index.items()):
            semantic_zh = semantic_by_index.get(index, "")
            if not semantic_zh or semantic_zh == style_zh:
                continue
            candidate = _candidate_from_pair(
                ref=f"{chapter_id}:{index}",
                chapter_id=chapter_id,
                index=index,
                english=english_by_index.get(index, ""),
                semantic_zh=semantic_zh,
                style_zh=style_zh,
                terms=terms,
            )
            if candidate:
                candidates.append(candidate)

    candidates.sort(key=lambda item: (-int(item["score"]), float(item["ratio"]), str(item["ref"])))
    if max_candidates is not None:
        candidates = candidates[:max_candidates]
    signal_counts = Counter(signal for item in candidates for signal in item.get("signals") or [])
    report = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "semantic_run_dir": str(semantic_run_dir),
        "snapshot_dir": str(snapshot_dir),
        "candidate_count": len(candidates),
        "signal_counts": dict(signal_counts),
        "candidates": candidates,
    }
    if output_dir is not None:
        output_dir = output_dir.expanduser()
        write_json(output_dir / "semantic_compression_candidates.json", report)
    return report


def _review_contract() -> dict[str, Any]:
    return {
        "task": "Review candidate semantic-compression losses in Chinese style reconstruction.",
        "instructions": [
            "English is the semantic authority; semantic_zh is a faithful reference draft; style_zh is the styled candidate.",
            "Classify whether each deterministic signal is a real semantic/style-texture loss or preserved by paraphrase.",
            "Do not reward literal retention. Compact paraphrase is acceptable if semantic force, role, quantity, action direction, and texture are preserved.",
            "Return a targeted repair only when verdict is true_loss.",
            "Keep repair_zh as one Chinese paragraph and preserve the style_zh sentence flow unless the loss requires a small local edit.",
            "Do not mechanically paste a missing term into style_zh; choose natural Chinese phrasing, borrowing local wording from semantic_zh when that reads better.",
        ],
        "verdicts": ["true_loss", "preserved_by_paraphrase", "harmless_compaction", "false_positive"],
        "output_schema": {
            "items": [
                {
                    "ref": "chapter:index",
                    "verdict": "true_loss | preserved_by_paraphrase | harmless_compaction | false_positive",
                    "severity": "high | medium | low | none",
                    "confidence": "high | medium | low",
                    "risk": "low | medium | high",
                    "reason": "brief explanation",
                    "repair_zh": "replacement Chinese only if true_loss, otherwise empty string",
                }
            ],
            "summary": {
                "true_loss": "number",
                "preserved_by_paraphrase": "number",
                "harmless_compaction": "number",
                "false_positive": "number",
                "recommendation": "short recommendation",
            },
        },
    }


def _write_review_prompt(path: Path, candidates: list[dict[str, Any]]) -> None:
    payload = {
        **_review_contract(),
        "items": candidates,
    }
    text = "Return ONLY valid JSON matching output_schema.\n\nINPUT JSON:\n"
    text += json.dumps(payload, ensure_ascii=False, indent=2)
    text += "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _chunked(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    size = max(1, size)
    return [items[index : index + size] for index in range(0, len(items), size)]


def _chunk_by_ref(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_ref: dict[str, dict[str, Any]] = {}
    for chunk in manifest.get("chunks") or []:
        chapter_id = str(chunk["chapter_id"])
        for index in chunk.get("indexes") or []:
            by_ref[f"{chapter_id}:{int(index)}"] = chunk
    return by_ref


def _replace_translation_item(path: Path, index: int, new_zh: str) -> str:
    data = _load_json(path)
    for item in data.get("translations") or []:
        if int(item.get("index")) == index:
            old_zh = str(item.get("zh") or "")
            item["zh"] = new_zh
            write_json(path, data)
            return old_zh
    raise ValueError(f"{path}: translation index {index} not found")


def _confidence_allows(value: str, minimum: str) -> bool:
    return CONFIDENCE_RANK.get(value.lower(), 0) >= CONFIDENCE_RANK.get(minimum.lower(), 2)


def _result_by_ref(review_items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item.get("ref") or ""): item for item in review_items if isinstance(item, dict)}


def _repair_preserves_original_signals(
    *,
    candidate: dict[str, Any],
    repair_zh: str,
    terms: list[dict[str, str]],
) -> tuple[bool, str]:
    original_signals = set(candidate.get("signals") or [])
    checked = _candidate_from_pair(
        ref=str(candidate["ref"]),
        chapter_id=str(candidate["chapter_id"]),
        index=int(candidate["index"]),
        english=str(candidate.get("english") or ""),
        semantic_zh=str(candidate.get("semantic_zh") or ""),
        style_zh=repair_zh,
        terms=terms,
    )
    remaining = set((checked or {}).get("signals") or [])
    remaining.discard("low_ratio_context")
    original_signals.discard("low_ratio_context")
    unresolved = sorted(original_signals & remaining)
    if unresolved:
        return False, f"repair still triggers original signal(s): {', '.join(unresolved)}"
    return True, ""


def _validate_repair(
    *,
    candidate: dict[str, Any],
    result: dict[str, Any],
    min_confidence: str,
    terms: list[dict[str, str]],
) -> tuple[bool, str, str]:
    verdict = str(result.get("verdict") or "").strip().lower()
    confidence = str(result.get("confidence") or "").strip().lower()
    risk = str(result.get("risk") or "").strip().lower()
    repair_zh = clean_text(str(result.get("repair_zh") or ""))
    if verdict != "true_loss":
        return False, "", "not a true_loss verdict"
    if not _confidence_allows(confidence, min_confidence):
        return False, "", f"confidence {confidence or 'none'} below {min_confidence}"
    if risk != "low":
        return False, "", f"risk is {risk or 'unspecified'}"
    if not repair_zh:
        return False, "", "empty repair_zh"
    if any(marker.lower() in repair_zh.lower() for marker in FAILURE_MARKERS):
        return False, "", "repair_zh contains failure marker"
    style_zh = clean_text(str(candidate.get("style_zh") or ""))
    semantic_zh = clean_text(str(candidate.get("semantic_zh") or ""))
    if _cjk_len(repair_zh) < max(1, int(_cjk_len(style_zh) * 0.75)):
        return False, "", "repair_zh is too short relative to style_zh"
    if _cjk_len(repair_zh) < max(1, int(_cjk_len(semantic_zh) * 0.6)):
        return False, "", "repair_zh is too short relative to semantic_zh"
    preserves, reason = _repair_preserves_original_signals(candidate=candidate, repair_zh=repair_zh, terms=terms)
    if not preserves:
        return False, "", reason
    return True, repair_zh, ""


def run_semantic_compression_qa(
    *,
    run_dir: Path,
    config: dict[str, Any],
    output_dir: Path | None = None,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    model: str | None = None,
    models: Sequence[str] | None = None,
    codex_bin: str = "codex",
    timeout_seconds: int | None = None,
    auto_repair: bool = False,
    min_auto_apply_confidence: str = DEFAULT_MIN_CONFIDENCE,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser()
    validate_style_transfer_provenance(run_dir)
    validate_and_merge(run_dir, allow_missing=False)
    if output_dir is None:
        output_dir = run_dir / "semantic_compression"
    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir = output_dir / "review_prompts"
    result_dir = output_dir / "review_results"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    input_translation_state_sha256 = translation_state_sha256(run_dir)
    style_source_state = validate_style_transfer_source_inputs(run_dir)
    style_run_binding_sha256 = _style_run_binding_sha256(run_dir)
    qa_config_sha256 = _qa_config_sha256(config)

    compression_config = config.get("semantic_compression") or {}
    terms = _glossary_terms(config, compression_config)
    candidate_report = detect_semantic_compression_candidates(
        run_dir=run_dir,
        config=config,
        max_candidates=max_candidates,
        output_dir=output_dir,
    )
    candidate_report_path = output_dir / "semantic_compression_candidates.json"
    candidates = list(candidate_report.get("candidates") or [])
    candidates_by_ref = {str(item["ref"]): item for item in candidates}
    requested_models: list[str | None] = list(
        dict.fromkeys(str(value) for value in models if str(value))
    ) if models else [model]
    unavailable_models: dict[str, str] = {}
    effective_model_batches: dict[str, int] = {}

    review_items: list[dict[str, Any]] = []
    review_artifacts: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for batch_number, batch in enumerate(_chunked(candidates, batch_size), 1):
        prompt_path = prompt_dir / f"batch_{batch_number:03d}.txt"
        raw_path = result_dir / f"batch_{batch_number:03d}.raw.txt"
        parsed_path = result_dir / f"batch_{batch_number:03d}.json"
        parsed_meta_path = result_dir / f"batch_{batch_number:03d}.meta.json"
        log_path = result_dir / f"batch_{batch_number:03d}.log"
        _write_review_prompt(prompt_path, batch)
        prompt_sha256 = _file_sha256(prompt_path)
        try:
            reuse_parsed = False
            if parsed_path.exists() and parsed_meta_path.exists() and not overwrite:
                parsed_meta = _load_json(parsed_meta_path)
                reuse_parsed = (
                    parsed_meta.get("prompt_sha256") == prompt_sha256
                    and parsed_meta.get("result_sha256") == _file_sha256(parsed_path)
                )
            if reuse_parsed:
                parsed = _load_json(parsed_path)
            else:
                text = ""
                for candidate_model in requested_models:
                    if candidate_model is not None and candidate_model in unavailable_models:
                        continue
                    try:
                        text = run_prompt_to_text(
                            prompt_path=prompt_path,
                            output_path=raw_path,
                            log_path=log_path,
                            model=candidate_model,
                            codex_bin=codex_bin,
                            timeout_seconds=timeout_seconds,
                        )
                    except ModelCapacityError as exc:
                        unavailable_models[str(candidate_model)] = exc.evidence
                        continue
                    effective_key = str(candidate_model or "default")
                    effective_model_batches[effective_key] = (
                        effective_model_batches.get(effective_key, 0) + 1
                    )
                    break
                else:
                    raise RuntimeError(
                        f"all semantic-QA models are unavailable: {unavailable_models}"
                    )
                parsed = extract_json_object(text)
                write_json(parsed_path, parsed)
                write_json(
                    parsed_meta_path,
                    {
                        "prompt_sha256": prompt_sha256,
                        "result_sha256": _file_sha256(parsed_path),
                    },
                )
            review_artifacts.append(
                {
                    "batch": batch_number,
                    "prompt_path": str(prompt_path.resolve()),
                    "prompt_sha256": prompt_sha256,
                    "result_path": str(parsed_path.resolve()),
                    "result_sha256": _file_sha256(parsed_path),
                }
            )
            for item in parsed.get("items") or []:
                if isinstance(item, dict):
                    review_items.append(item)
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "batch": batch_number,
                    "refs": [item["ref"] for item in batch],
                    "error": str(exc),
                }
            )

    manifest = _load_json(run_dir / "run_manifest.json")
    chunks = _chunk_by_ref(manifest)
    applied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    unresolved_true_losses: list[dict[str, Any]] = []
    review_by_ref = _result_by_ref(review_items)
    review_ref_counts = Counter(
        str(item.get("ref") or "")
        for item in review_items
        if isinstance(item, dict)
    )
    for ref in sorted(set(review_ref_counts) - set(candidates_by_ref)):
        rejected.append({"ref": ref, "reject_reason": "unexpected review ref"})

    for ref, candidate in candidates_by_ref.items():
        if review_ref_counts.get(ref, 0) > 1:
            rejected.append({"ref": ref, "reject_reason": "duplicate review ref"})
            continue
        result = review_by_ref.get(ref)
        if result is None:
            rejected.append({"ref": ref, "reject_reason": "missing from model output"})
            continue
        verdict = str(result.get("verdict") or "").strip().lower()
        severity = str(result.get("severity") or "").strip().lower()
        confidence = str(result.get("confidence") or "").strip().lower()
        risk = str(result.get("risk") or "").strip().lower()
        invalid_fields: list[str] = []
        if verdict not in {
            "true_loss",
            "preserved_by_paraphrase",
            "harmless_compaction",
            "false_positive",
        }:
            invalid_fields.append("verdict")
        if severity not in {"high", "medium", "low", "none"}:
            invalid_fields.append("severity")
        if confidence not in {"high", "medium", "low"}:
            invalid_fields.append("confidence")
        if risk not in {"high", "medium", "low"}:
            invalid_fields.append("risk")
        if invalid_fields:
            rejected.append(
                {
                    "ref": ref,
                    "reject_reason": "invalid review field(s): "
                    + ", ".join(invalid_fields),
                }
            )
            continue
        allowed, repair_zh, reject_reason = _validate_repair(
            candidate=candidate,
            result=result,
            min_confidence=min_auto_apply_confidence,
            terms=terms,
        )
        if not allowed:
            skipped_item = {
                "ref": ref,
                "verdict": result.get("verdict"),
                "severity": result.get("severity"),
                "confidence": result.get("confidence"),
                "risk": result.get("risk"),
                "reason": result.get("reason") or "",
                "skip_reason": reject_reason,
            }
            skipped.append(skipped_item)
            if str(result.get("verdict") or "").lower() == "true_loss":
                unresolved_true_losses.append(dict(skipped_item))
            continue
        if not auto_repair:
            skipped_item = {
                "ref": ref,
                "verdict": result.get("verdict"),
                "severity": result.get("severity"),
                "confidence": result.get("confidence"),
                "risk": result.get("risk"),
                "reason": result.get("reason") or "",
                "repair_zh": repair_zh,
                "skip_reason": "auto_repair disabled",
            }
            skipped.append(skipped_item)
            unresolved_true_losses.append(dict(skipped_item))
            continue
        if dry_run:
            skipped_item = {
                "ref": ref,
                "verdict": result.get("verdict"),
                "severity": result.get("severity"),
                "confidence": result.get("confidence"),
                "risk": result.get("risk"),
                "reason": result.get("reason") or "",
                "repair_zh": repair_zh,
                "skip_reason": "dry run",
            }
            skipped.append(skipped_item)
            unresolved_true_losses.append(dict(skipped_item))
            continue
        chapter_id, index_text = ref.split(":", 1)
        index = int(index_text)
        chunk = chunks.get(ref)
        if chunk is None:
            rejected.append({"ref": ref, "reject_reason": "manifest chunk not found"})
            continue
        output_path = run_dir / str(chunk["json_output_path"])
        translation_path = run_dir / "translations" / f"{chapter_id}.json"
        old_zh = _replace_translation_item(output_path, index, repair_zh)
        _replace_translation_item(translation_path, index, repair_zh)
        applied.append(
            {
                "ref": ref,
                "chapter_id": chapter_id,
                "index": index,
                "chunk_id": chunk.get("chunk_id"),
                "old_zh": old_zh,
                "new_zh": repair_zh,
                "verdict": result.get("verdict"),
                "severity": result.get("severity"),
                "confidence": result.get("confidence"),
                "risk": result.get("risk"),
                "reason": result.get("reason") or "",
                "signals": candidate.get("signals") or [],
            }
        )

    verdict_counts = Counter(str(item.get("verdict") or "missing") for item in review_items)
    unresolved_severity_counts = Counter(
        str(item.get("severity") or "unspecified")
        for item in unresolved_true_losses
    )
    output_translation_state_sha256 = translation_state_sha256(run_dir)
    summary = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "input_translation_state_sha256": input_translation_state_sha256,
        "output_translation_state_sha256": output_translation_state_sha256,
        "dry_run": dry_run,
        "settings": {
            "max_candidates": max_candidates,
            "batch_size": batch_size,
            "model": model,
            "requested_model_order": requested_models,
            "unavailable_models": unavailable_models,
            "effective_model_batches": effective_model_batches,
            "codex_bin": codex_bin,
            "timeout_seconds": timeout_seconds,
            "auto_repair": auto_repair,
            "min_auto_apply_confidence": min_auto_apply_confidence,
            "overwrite": overwrite,
        },
        "candidate_summary": {
            "candidate_count": len(candidates),
            "signal_counts": candidate_report.get("signal_counts") or {},
        },
        "review_summary": {
            "reviewed_count": len(review_items),
            "verdict_counts": dict(verdict_counts),
        },
        "result_summary": {
            "applied_count": len(applied),
            "skipped_count": len(skipped),
            "rejected_count": len(rejected),
            "failure_count": len(failures),
            "unresolved_true_loss_count": len(unresolved_true_losses),
            "unresolved_high_severity_count": unresolved_severity_counts.get(
                "high", 0
            ),
        },
        "provenance": {
            "schema": "semantic_qa_provenance.v1",
            "qa_config_sha256": qa_config_sha256,
            "review_contract_sha256": _sha256_json(_review_contract()),
            "style_run_binding_sha256": style_run_binding_sha256,
            "style_source_state_sha256": style_source_state["state_sha256"],
            "candidate_report_path": str(candidate_report_path.resolve()),
            "candidate_report_sha256": _file_sha256(candidate_report_path),
            "review_artifact_count": len(review_artifacts),
            "review_artifacts": review_artifacts,
        },
        "candidates": candidates,
        "review_items": review_items,
        "applied": applied,
        "skipped": skipped,
        "rejected": rejected,
        "unresolved_true_losses": unresolved_true_losses,
        "failures": failures,
    }
    write_json(output_dir / "semantic_compression_summary.json", summary)
    _write_markdown(output_dir / "semantic_compression_summary.md", summary)
    return summary


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    result = summary["result_summary"]
    candidate = summary["candidate_summary"]
    review = summary["review_summary"]
    lines = [
        "# Semantic Compression QA",
        "",
        f"- Run: `{summary['run_dir']}`",
        f"- Candidates: {candidate['candidate_count']}",
        f"- Reviewed: {review['reviewed_count']}",
        f"- Applied: {result['applied_count']}",
        f"- Skipped: {result['skipped_count']}",
        f"- Rejected: {result['rejected_count']}",
        f"- Failed batches: {result['failure_count']}",
        f"- Unresolved true losses: {result.get('unresolved_true_loss_count', 0)}",
        f"- Unresolved high-severity losses: {result.get('unresolved_high_severity_count', 0)}",
        "",
        "## Verdict Counts",
        "",
    ]
    for verdict, count in sorted((review.get("verdict_counts") or {}).items()):
        lines.append(f"- `{verdict}`: {count}")
    lines.extend(["", "## Applied", ""])
    for item in summary.get("applied") or []:
        lines.extend(
            [
                f"### {item['ref']} `{item.get('severity')}`",
                "",
                f"- Confidence: {item.get('confidence')}",
                f"- Risk: {item.get('risk')}",
                f"- Signals: {', '.join(item.get('signals') or [])}",
                f"- Reason: {item.get('reason') or ''}",
                "",
                f"Before: {item.get('old_zh') or ''}",
                "",
                f"After: {item.get('new_zh') or ''}",
                "",
            ]
        )
    lines.extend(["", "## Skipped", ""])
    for item in summary.get("skipped") or []:
        lines.append(
            f"- {item.get('ref')}: {item.get('verdict')} / {item.get('severity')} / "
            f"{item.get('confidence')} / {item.get('risk')} - {item.get('skip_reason')}; "
            f"{item.get('reason') or ''}"
        )
    lines.extend(["", "## Rejected", ""])
    for item in summary.get("rejected") or []:
        lines.append(f"- {item.get('ref')}: {item.get('reject_reason')}")
    lines.extend(["", "## Unresolved True Losses", ""])
    for item in summary.get("unresolved_true_losses") or []:
        lines.append(
            f"- {item.get('ref')}: {item.get('severity')} / "
            f"{item.get('confidence')} / {item.get('risk')} - "
            f"{item.get('skip_reason')}; {item.get('reason') or ''}"
        )
    lines.extend(["", "## Failures", ""])
    for item in summary.get("failures") or []:
        lines.append(f"- batch {item.get('batch')}: {item.get('error')}")
    path.write_text("\n".join(lines), encoding="utf-8")
