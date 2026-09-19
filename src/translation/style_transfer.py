from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from src.crawler.snapshot import (
    load_chapter,
    load_manifest,
    snapshot_chapter_ids,
    write_json,
)
from src.runtime.paths import repo_root
from src.translation.codex_cli import is_model_capacity_text
from src.translation.style_transfer_assets import (
    INTENSITY,
    METHOD_ID,
    build_method_request,
    canonical_json,
    file_sha256,
    load_assets,
    sha256_json,
)


RUN_SCHEMA = "author_style_transfer_run.v1"
DEFAULT_BLOCK_SIZE = 12
DEFAULT_MODEL_ORDER = (
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
    "gpt-5.3-codex-spark",
    "gpt-5.4",
)
HAN_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]{8,}")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
NUMBER_RE = re.compile(r"\d+(?:[.,:/-]\d+)*")
LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]*")
PLACEHOLDER_RE = re.compile(r"<(?:TERM|CONTENT|NUM|LATIN)>")
JSON_RESIDUE_RE = re.compile(r"(?:\}\s*,\s*\{|```|\[\s*\{|\}\s*\])")
PRINT_LOCK = threading.Lock()


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (repo_root() / candidate).resolve()


def configured_model_order(
    config: dict[str, Any], overrides: Sequence[str] | None = None
) -> list[str]:
    if overrides:
        values = list(overrides)
    else:
        style_config = config.get("style_transfer") or {}
        codex_config = config.get("codex") or {}
        values = list(
            style_config.get("model_order")
            or codex_config.get("model_order")
            or DEFAULT_MODEL_ORDER
        )
    result = list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
    if not result:
        raise ValueError("at least one Codex model is required")
    return result


def stable_sample_id(
    chapter_id: str,
    indexes: Sequence[int],
    english: Sequence[str],
    neutral: Sequence[str],
) -> str:
    digest = sha256_json(
        {
            "chapter_id": chapter_id,
            "indexes": list(indexes),
            "english": list(english),
            "neutral": list(neutral),
        }
    )
    return f"s_{digest[:24]}"


def translation_map(path: Path) -> dict[int, str]:
    data = read_json(path)
    return {
        int(item["index"]): str(item.get("zh") or "").strip()
        for item in data.get("translations") or []
    }


def request_for_rows(
    *,
    asset_path: Path,
    asset_file_sha256: str,
    chapter_id: str,
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    indexes = [int(row["index"]) for row in rows]
    english_texts = [str(row["english"]) for row in rows]
    neutral_texts = [str(row["neutral_zh"]) for row in rows]
    paragraph_ids = [f"p{index + 1:04d}" for index in range(len(rows))]
    return build_method_request(
        asset_path=asset_path,
        asset_file_sha256=asset_file_sha256,
        sample_id=stable_sample_id(chapter_id, indexes, english_texts, neutral_texts),
        english_semantic_source=[
            {"id": paragraph_ids[index], "en": text}
            for index, text in enumerate(english_texts)
        ],
        neutral_zh=[
            {"id": paragraph_ids[index], "zh": text}
            for index, text in enumerate(neutral_texts)
        ],
    )


def build_prompt(template: str, request: dict[str, Any]) -> str:
    return (
        template.rstrip()
        + "\n\n## Concrete Request\n\n"
        + "Do not use tools or inspect the filesystem. Work only from the JSON "
        + "request below.\n\n```json\n"
        + json.dumps(request, ensure_ascii=False, indent=2)
        + "\n```\n"
    )


def prepare_style_transfer_run(
    *,
    semantic_run_dir: Path,
    style_run_dir: Path,
    config: dict[str, Any],
    block_size: int | None = None,
) -> dict[str, Any]:
    style_config = config.get("style_transfer") or {}
    asset_path = resolve_repo_path(str(style_config["asset_path"]))
    asset_file_sha256 = str(style_config.get("asset_file_sha256") or "")
    prompt_path = resolve_repo_path(str(style_config["prompt_path"]))
    schema_path = resolve_repo_path(str(style_config["schema_path"]))
    assets = load_assets(asset_path, expected_file_sha256=asset_file_sha256)
    prompt_file_sha256 = file_sha256(prompt_path)
    schema_file_sha256 = file_sha256(schema_path)
    expected_prompt_sha256 = str(style_config.get("prompt_file_sha256") or "")
    expected_schema_sha256 = str(style_config.get("schema_file_sha256") or "")
    if expected_prompt_sha256 and prompt_file_sha256 != expected_prompt_sha256:
        raise ValueError(
            f"style-transfer prompt hash mismatch: expected {expected_prompt_sha256}, "
            f"got {prompt_file_sha256}"
        )
    if expected_schema_sha256 and schema_file_sha256 != expected_schema_sha256:
        raise ValueError(
            f"style-transfer schema hash mismatch: expected {expected_schema_sha256}, "
            f"got {schema_file_sha256}"
        )
    block_size = int(block_size or style_config.get("block_size") or DEFAULT_BLOCK_SIZE)
    if not 1 <= block_size <= DEFAULT_BLOCK_SIZE:
        raise ValueError(f"style-transfer block size must be between 1 and {DEFAULT_BLOCK_SIZE}")

    semantic_run_dir = semantic_run_dir.expanduser().resolve()
    style_run_dir = style_run_dir.expanduser().resolve()
    semantic_manifest_path = semantic_run_dir / "run_manifest.json"
    semantic_manifest = read_json(semantic_manifest_path)
    semantic_translations_dir = semantic_run_dir / "translations"
    if not semantic_translations_dir.exists():
        raise FileNotFoundError(
            f"validated semantic translations not found: {semantic_translations_dir}"
        )
    snapshot_dir = Path(str(semantic_manifest["snapshot_dir"])).expanduser().resolve()
    snapshot_manifest = load_manifest(snapshot_dir)
    prompt_template = prompt_path.read_text(encoding="utf-8")

    style_run_dir.mkdir(parents=True, exist_ok=True)
    for directory in ("requests", "prompts", "outputs", "method_outputs"):
        (style_run_dir / directory).mkdir(parents=True, exist_ok=True)

    expected_by_chapter: dict[str, list[int]] = {}
    title_by_chapter: dict[str, str] = {}
    for chunk in semantic_manifest.get("chunks") or []:
        chapter_id = str(chunk["chapter_id"])
        expected_by_chapter.setdefault(chapter_id, []).extend(
            int(value) for value in chunk["indexes"]
        )
        title_by_chapter.setdefault(chapter_id, str(chunk.get("chapter_title") or ""))

    chunks: list[dict[str, Any]] = []
    total_paragraphs = 0
    for chapter_id in snapshot_chapter_ids(snapshot_manifest):
        expected_indexes = expected_by_chapter.get(chapter_id)
        if expected_indexes is None:
            raise ValueError(f"semantic run has no chunks for chapter {chapter_id}")
        chapter = load_chapter(snapshot_dir, snapshot_manifest, chapter_id)
        english_by_index = {
            int(item["index"]): str(item.get("english") or "").strip()
            for item in chapter.get("paragraphs") or []
        }
        neutral_path = semantic_translations_dir / f"{chapter_id}.json"
        neutral_by_index = translation_map(neutral_path)
        if sorted(expected_indexes) != sorted(english_by_index):
            raise ValueError(f"English index mismatch in chapter {chapter_id}")
        if sorted(expected_indexes) != sorted(neutral_by_index):
            raise ValueError(f"neutral index mismatch in chapter {chapter_id}")
        if any(not neutral_by_index[index] for index in expected_indexes):
            raise ValueError(f"empty neutral paragraph in chapter {chapter_id}")
        ordered_rows = [
            {
                "index": index,
                "english": english_by_index[index],
                "neutral_zh": neutral_by_index[index],
            }
            for index in sorted(expected_indexes)
        ]
        for block_number, start in enumerate(range(0, len(ordered_rows), block_size), 1):
            rows = ordered_rows[start : start + block_size]
            chunk_id = f"{chapter_id}_style_{block_number:03d}"
            request = request_for_rows(
                asset_path=asset_path,
                asset_file_sha256=asset_file_sha256,
                chapter_id=chapter_id,
                rows=rows,
            )
            request_path = style_run_dir / "requests" / f"{chunk_id}.json"
            chunk_prompt_path = style_run_dir / "prompts" / f"{chunk_id}.md"
            write_json(request_path, request)
            chunk_prompt_path.write_text(
                build_prompt(prompt_template, request), encoding="utf-8"
            )
            indexes = [int(row["index"]) for row in rows]
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "chapter_id": chapter_id,
                    "chapter_title": title_by_chapter.get(chapter_id)
                    or str(chapter.get("title") or ""),
                    "indexes": indexes,
                    "prompt_path": str(chunk_prompt_path.relative_to(style_run_dir)),
                    "request_path": str(request_path.relative_to(style_run_dir)),
                    "raw_output_path": f"outputs/{chunk_id}.raw.txt",
                    "json_output_path": f"outputs/{chunk_id}.json",
                    "method_output_path": f"method_outputs/{chunk_id}.json",
                    "request_sha256": sha256_json(request),
                    "prompt_sha256": file_sha256(chunk_prompt_path),
                    "english_sha256": sha256_json([row["english"] for row in rows]),
                    "neutral_zh_sha256": sha256_json(
                        [row["neutral_zh"] for row in rows]
                    ),
                }
            )
            total_paragraphs += len(rows)

    manifest = {
        "schema_version": 3,
        "pass": "author_style_transfer",
        "snapshot_dir": str(snapshot_dir),
        "semantic_run_dir": str(semantic_run_dir),
        "config": {key: value for key, value in config.items() if key != "_config_path"},
        "chunks": chunks,
        "style_transfer": {
            "schema": RUN_SCHEMA,
            "method_id": METHOD_ID,
            "intensity": INTENSITY,
            "status": "prepared",
            "prepared_at": utc_now(),
            "block_size": block_size,
            "chunk_count": len(chunks),
            "paragraph_count": total_paragraphs,
            "capacity_fallback_policy": {
                "requested_model_order": configured_model_order(config),
                "switch_only_on": "explicit_model_capacity_or_availability_failure",
                "content_or_contract_failure_action": (
                    "same_model_retry_then_recursive_bisection"
                ),
            },
            "asset_path": str(asset_path),
            "asset_file_sha256": assets["file_sha256"],
            "asset_content_sha256": assets["content_sha256"],
            "component_hashes": assets["assets"]["component_hashes"],
            "prompt_path": str(prompt_path),
            "prompt_sha256": prompt_file_sha256,
            "schema_path": str(schema_path),
            "schema_sha256": schema_file_sha256,
            "semantic_manifest_sha256": file_sha256(semantic_manifest_path),
            "snapshot_manifest_sha256": file_sha256(snapshot_dir / "manifest.json"),
        },
    }
    write_json(style_run_dir / "run_manifest.json", manifest)
    return manifest


def reference_texts(request: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for reference in request.get("reference_examples") or []:
        direct = reference.get("target_style_text")
        if isinstance(direct, str):
            texts.append(direct)
        for item in reference.get("target_style_version") or []:
            if isinstance(item, dict) and isinstance(item.get("zh"), str):
                texts.append(item["zh"])
    return texts


def copied_reference_spans(request: dict[str, Any], result: dict[str, Any]) -> list[str]:
    references = reference_texts(request)
    neutral_text = "\n".join(
        str(paragraph.get("zh") or "") for paragraph in request.get("neutral_zh") or []
    )
    copied: set[str] = set()
    for paragraph in result.get("paragraphs") or []:
        output = str(paragraph.get("zh") or "")
        for run in HAN_RUN_RE.findall(output):
            for start in range(0, len(run) - 7):
                phrase = run[start : start + 8]
                if phrase not in neutral_text and any(
                    phrase in reference for reference in references
                ):
                    copied.add(phrase)
    return sorted(copied)


def starts_dialogue(text: str) -> bool:
    return text.lstrip().startswith(("“", "‘", "「", "『", '"'))


def quotes_balanced(text: str) -> bool:
    return (
        text.count("“") == text.count("”")
        and text.count("‘") == text.count("’")
        and text.count("「") == text.count("」")
        and text.count("『") == text.count("』")
    )


def base_result_errors(request: dict[str, Any], result: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if result.get("sample_id") != request["sample_id"]:
        errors.append("sample_id mismatch")
    if result.get("method_id") != METHOD_ID:
        errors.append("method_id mismatch")
    if result.get("intensity") != INTENSITY:
        errors.append("intensity mismatch")
    for field in ("style_cues_applied", "style_cues_skipped", "uncertainties"):
        if not isinstance(result.get(field), list):
            errors.append(f"{field} must be a list")
    paragraphs = result.get("paragraphs")
    if not isinstance(paragraphs, list):
        return errors + ["paragraphs must be a list"]
    expected_ids = [row["id"] for row in request["neutral_zh"]]
    output_ids = [row.get("id") for row in paragraphs]
    if output_ids != expected_ids:
        errors.append("paragraph IDs or order do not match input")
    if any(not isinstance(row.get("zh"), str) or not row["zh"].strip() for row in paragraphs):
        errors.append("one or more zh outputs are empty")
    if len(output_ids) != len(set(output_ids)):
        errors.append("duplicate paragraph IDs")
    return errors


def deterministic_fidelity_errors(
    request: dict[str, Any], result: dict[str, Any]
) -> list[str]:
    neutral = list(request.get("neutral_zh") or [])
    candidate = list(result.get("paragraphs") or [])
    if [row.get("id") for row in neutral] != [row.get("id") for row in candidate]:
        return ["fidelity:paragraph_id_or_order_mismatch"]
    failures: list[str] = []
    for source, output in zip(neutral, candidate):
        source_text = str(source.get("zh") or "")
        output_text = str(output.get("zh") or "")
        prefix = str(output.get("id") or "unknown")
        if sorted(NUMBER_RE.findall(source_text)) != sorted(NUMBER_RE.findall(output_text)):
            failures.append(f"fidelity:{prefix}:number_surface_mismatch")
        if sorted(PLACEHOLDER_RE.findall(source_text)) != sorted(
            PLACEHOLDER_RE.findall(output_text)
        ):
            failures.append(f"fidelity:{prefix}:placeholder_mismatch")
        if sorted(LATIN_RE.findall(source_text)) != sorted(LATIN_RE.findall(output_text)):
            failures.append(f"fidelity:{prefix}:latin_token_mismatch")
        if starts_dialogue(source_text) != starts_dialogue(output_text):
            failures.append(f"fidelity:{prefix}:dialogue_turn_surface_mismatch")
        if not quotes_balanced(output_text):
            failures.append(f"fidelity:{prefix}:unbalanced_dialogue_quotes")
        if JSON_RESIDUE_RE.search(output_text):
            failures.append(f"fidelity:{prefix}:json_residue")
        source_cjk = len(CJK_RE.findall(source_text))
        output_cjk = len(CJK_RE.findall(output_text))
        ratio = output_cjk / max(source_cjk, 1)
        if source_cjk >= 20 and not 0.50 <= ratio <= 1.80:
            failures.append(f"fidelity:{prefix}:paragraph_cjk_ratio_out_of_bounds")
    source_total = sum(len(CJK_RE.findall(str(row.get("zh") or ""))) for row in neutral)
    output_total = sum(
        len(CJK_RE.findall(str(row.get("zh") or ""))) for row in candidate
    )
    if not 0.70 <= output_total / max(source_total, 1) <= 1.35:
        failures.append("fidelity:aggregate_cjk_ratio_out_of_bounds")
    return failures


def application_validation_errors(
    request: dict[str, Any], result: dict[str, Any]
) -> list[str]:
    errors = base_result_errors(request, result)
    expected_ids = [row["id"] for row in request["neutral_zh"]]
    content_plan = result.get("content_plan")
    if not isinstance(content_plan, list):
        errors.append("content_plan must be a list")
    else:
        if [row.get("id") for row in content_plan] != expected_ids:
            errors.append("content_plan paragraph IDs or order do not match input")
        for row in content_plan:
            if not isinstance(row.get("facts"), list) or not isinstance(
                row.get("constraints"), list
            ):
                errors.append("content_plan facts/constraints must be arrays")
                break
    copied = copied_reference_spans(request, result)
    if copied:
        errors.append("copied_reference_span:" + ",".join(copied[:5]))
    errors.extend(deterministic_fidelity_errors(request, result))
    return list(dict.fromkeys(errors))


def _minimal_child_environment(temp_dir: Path) -> dict[str, str]:
    allowed_names = (
        "HOME",
        "USER",
        "LOGNAME",
        "LANG",
        "LC_ALL",
        "CODEX_HOME",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
    )
    environment = {
        name: os.environ[name] for name in allowed_names if name in os.environ
    }
    environment["PATH"] = os.environ.get("PATH", "/usr/bin:/bin")
    environment["TMPDIR"] = str(temp_dir)
    return environment


def _sandbox_profile() -> str:
    root = str(repo_root())
    return "\n".join(
        (
            "(version 1)",
            "(allow default)",
            f'(deny file-read* (subpath "{root}"))',
            f'(deny file-write* (subpath "{root}"))',
            "",
        )
    )


def _codex_binary(codex_bin: str) -> Path:
    located = shutil.which(codex_bin)
    if not located:
        raise FileNotFoundError(f"Codex executable is not available: {codex_bin}")
    return Path(located).resolve()


def _parse_events(
    stdout: str,
) -> tuple[str | None, dict[str, int], list[dict[str, Any]]]:
    response_id: str | None = None
    usage: dict[str, int] = {}
    response_errors: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "thread.started":
            response_id = event.get("thread_id")
        if event.get("type") == "turn.completed":
            usage = {
                key: int(value)
                for key, value in (event.get("usage") or {}).items()
                if isinstance(value, int)
            }
        if event.get("type") in {"turn.failed", "error"}:
            response_errors.append(event)
    return response_id, usage, response_errors


def run_structured_attempt(
    *,
    request: dict[str, Any],
    prompt_template: str,
    schema_path: Path,
    output_path: Path,
    model: str,
    reasoning_effort: str,
    timeout_seconds: int,
    codex_bin: str,
    attempt: int,
) -> dict[str, Any]:
    started_at = utc_now()
    validation_errors: list[str] = []
    stdout = ""
    stderr = ""
    response_id: str | None = None
    usage: dict[str, int] = {}
    response_errors: list[dict[str, Any]] = []
    process_error_tail = ""
    result: dict[str, Any] | None = None
    with tempfile.TemporaryDirectory(prefix="author-style-transfer-") as temp:
        temp_dir = Path(temp)
        response_path = temp_dir / "response.json"
        isolated_schema = temp_dir / "output.schema.json"
        isolated_schema.write_bytes(schema_path.read_bytes())
        profile_path = temp_dir / "isolation.sb"
        profile_path.write_text(_sandbox_profile(), encoding="utf-8")
        command = [
            str(_codex_binary(codex_bin)),
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
            "--disable",
            "shell_tool",
            "--disable",
            "unified_exec",
            "--disable",
            "browser_use",
            "--disable",
            "browser_use_external",
            "--disable",
            "browser_use_full_cdp_access",
            "--disable",
            "in_app_browser",
            "--disable",
            "computer_use",
            "--disable",
            "apps",
            "--disable",
            "multi_agent",
            "--cd",
            str(temp_dir),
            "--model",
            model,
            "--config",
            f'model_reasoning_effort="{reasoning_effort}"',
            "--output-schema",
            str(isolated_schema),
            "--output-last-message",
            str(response_path),
            "--json",
            "-",
        ]
        if platform.system() == "Darwin" and Path("/usr/bin/sandbox-exec").exists():
            command = ["/usr/bin/sandbox-exec", "-f", str(profile_path), *command]
        try:
            completed = subprocess.run(
                command,
                input=build_prompt(prompt_template, request),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=_minimal_child_environment(temp_dir),
                cwd=temp_dir,
            )
        except subprocess.TimeoutExpired:
            validation_errors.append("codex request timed out")
        else:
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            response_id, usage, response_errors = _parse_events(stdout)
            if completed.returncode != 0:
                validation_errors.append(f"codex exited with status {completed.returncode}")
                process_error_tail = stdout[-2000:]
            elif not response_path.exists():
                validation_errors.append("codex did not write a final response")
            else:
                try:
                    parsed = json.loads(response_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    validation_errors.append(f"response is not valid JSON: {exc}")
                else:
                    if isinstance(parsed, dict):
                        result = parsed
                        validation_errors.extend(base_result_errors(request, result))
                    else:
                        validation_errors.append("response is not a JSON object")
    record = {
        "started_at": started_at,
        "completed_at": utc_now(),
        "attempt": attempt,
        "status": "success" if result is not None and not validation_errors else "failed",
        "model": model,
        "reasoning_effort": reasoning_effort,
        "response_id": response_id,
        "response_usage": usage,
        "response_errors": response_errors,
        "validation_errors": validation_errors,
        "stderr_tail": stderr[-2000:],
        "process_error_tail": process_error_tail,
    }
    if record["status"] == "success" and result is not None:
        artifact = {
            "schema_version": 1,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "attempt": attempt,
            "response_id": response_id,
            "response_usage": usage,
            "request_sha256": sha256_json(request),
            "result_sha256": sha256_json(result),
            "result": result,
        }
        write_json(output_path, artifact)
    elif result is not None:
        write_json(
            output_path.parent / "_quarantine" / output_path.name,
            {"record": record, "result": result},
        )
    return record


def model_capacity_failure(record: dict[str, Any]) -> bool:
    evidence = canonical_json(
        {
            "validation_errors": record.get("validation_errors") or [],
            "response_errors": record.get("response_errors") or [],
            "stderr_tail": record.get("stderr_tail") or "",
            "process_error_tail": record.get("process_error_tail") or "",
        }
    )
    return is_model_capacity_text(evidence)


@dataclass(frozen=True)
class SegmentResult:
    indexes: list[int]
    paragraphs: list[str]
    content_plan: list[dict[str, Any]]
    style_cues_applied: list[str]
    style_cues_skipped: list[str]
    uncertainties: list[str]
    artifacts: list[dict[str, Any]]
    usage: dict[str, int]
    models_used: list[str]


def _add_usage(target: dict[str, int], source: dict[str, Any]) -> None:
    for key, value in (source.get("response_usage") or {}).items():
        if isinstance(value, int):
            target[key] = target.get(key, 0) + value


def neutral_leaf_fallback(
    *,
    request: dict[str, Any],
    artifact: dict[str, Any] | None,
    validation_errors: Sequence[str],
) -> tuple[dict[str, Any], list[str]] | None:
    if artifact is None or not isinstance(artifact.get("result"), dict):
        return None
    non_fidelity_errors = [
        error for error in validation_errors if not error.startswith("fidelity:")
    ]
    if non_fidelity_errors:
        return None
    result = artifact["result"]
    neutral = list(request.get("neutral_zh") or [])
    content_plan = result.get("content_plan")
    if len(neutral) != 1 or not isinstance(content_plan, list):
        return None
    fallback = dict(result)
    fallback["paragraphs"] = [dict(neutral[0])]
    fallback["style_cues_skipped"] = list(
        dict.fromkeys(
            [
                *[str(value) for value in result.get("style_cues_skipped") or []],
                "neutral_leaf_fallback_after_fidelity_gate",
            ]
        )
    )
    if application_validation_errors(request, fallback):
        return None
    return fallback, list(validation_errors)


class StyleTransferRunner:
    def __init__(
        self,
        *,
        style_run_dir: Path,
        asset_path: Path,
        asset_file_sha256: str,
        prompt_path: Path,
        schema_path: Path,
        models: Sequence[str],
        reasoning_effort: str,
        timeout_seconds: int,
        max_attempts: int,
        codex_bin: str,
        overwrite: bool,
    ) -> None:
        self.style_run_dir = style_run_dir
        self.asset_path = asset_path
        self.asset_file_sha256 = asset_file_sha256
        self.prompt_path = prompt_path
        self.schema_path = schema_path
        self.models = list(dict.fromkeys(models))
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.codex_bin = codex_bin
        self.overwrite = overwrite
        self.prompt_template = prompt_path.read_text(encoding="utf-8")
        self._disabled_models: dict[str, str] = {}
        self._model_lock = threading.Lock()

    def active_models(self) -> list[str]:
        with self._model_lock:
            active = [model for model in self.models if model not in self._disabled_models]
            disabled = dict(self._disabled_models)
        if not active:
            raise RuntimeError(f"all requested models are unavailable: {disabled}")
        return active

    def disable_model(self, model: str, record: dict[str, Any]) -> None:
        reason = canonical_json(
            {
                "validation_errors": record.get("validation_errors") or [],
                "response_errors": record.get("response_errors") or [],
                "stderr_tail": record.get("stderr_tail") or "",
                "process_error_tail": record.get("process_error_tail") or "",
            }
        )[-2000:]
        with self._model_lock:
            self._disabled_models.setdefault(model, reason)

    def disabled_models(self) -> dict[str, str]:
        with self._model_lock:
            return dict(self._disabled_models)

    def invoke_segment(
        self,
        *,
        chunk_id: str,
        chapter_id: str,
        rows: Sequence[dict[str, Any]],
        segment_name: str,
        depth: int,
    ) -> SegmentResult:
        request = request_for_rows(
            asset_path=self.asset_path,
            asset_file_sha256=self.asset_file_sha256,
            chapter_id=chapter_id,
            rows=rows,
        )
        segment_root = self.style_run_dir / "method_outputs/_segments" / chunk_id
        records: list[dict[str, Any]] = []
        attempted_usage: dict[str, int] = {}
        last_artifact: dict[str, Any] | None = None
        last_application_errors: list[str] = []
        last_model: str | None = None
        last_output_path: Path | None = None
        for model in self.active_models():
            model_slug = re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_")
            switch_for_capacity = False
            for attempt in range(1, self.max_attempts + 1):
                output_path = (
                    segment_root
                    / f"{segment_name}.{model_slug}.attempt{attempt:02d}.json"
                )
                record = run_structured_attempt(
                    request=request,
                    prompt_template=self.prompt_template,
                    schema_path=self.schema_path,
                    output_path=output_path,
                    model=model,
                    reasoning_effort=self.reasoning_effort,
                    timeout_seconds=self.timeout_seconds,
                    codex_bin=self.codex_bin,
                    attempt=attempt,
                )
                records.append(record)
                _add_usage(attempted_usage, record)
                if record.get("status") != "success" or not output_path.exists():
                    if model_capacity_failure(record):
                        self.disable_model(model, record)
                        switch_for_capacity = True
                        with PRINT_LOCK:
                            print(
                                f"style model unavailable {model}; trying next configured model",
                                flush=True,
                            )
                        break
                    continue
                artifact = read_json(output_path)
                result = artifact["result"]
                errors = application_validation_errors(request, result)
                if errors:
                    record["application_validation_errors"] = errors
                    last_artifact = artifact
                    last_application_errors = errors
                    last_model = model
                    last_output_path = output_path
                    continue
                return SegmentResult(
                    indexes=[int(row["index"]) for row in rows],
                    paragraphs=[str(row["zh"]) for row in result["paragraphs"]],
                    content_plan=[dict(row) for row in result["content_plan"]],
                    style_cues_applied=[
                        str(value) for value in result.get("style_cues_applied") or []
                    ],
                    style_cues_skipped=[
                        str(value) for value in result.get("style_cues_skipped") or []
                    ],
                    uncertainties=[
                        str(value) for value in result.get("uncertainties") or []
                    ],
                    artifacts=[
                        {
                            "segment": segment_name,
                            "depth": depth,
                            "source_indexes": [int(row["index"]) for row in rows],
                            "path": str(output_path.relative_to(self.style_run_dir)),
                            "file_sha256": file_sha256(output_path),
                            "request_sha256": sha256_json(request),
                            "response_id": record.get("response_id"),
                            "attempt": attempt,
                            "model": model,
                        }
                    ],
                    usage=attempted_usage,
                    models_used=[model],
                )
            if switch_for_capacity:
                continue
            break

        if len(rows) == 1:
            failures_path = segment_root / f"{segment_name}.failures.json"
            write_json(failures_path, records)
            fallback = neutral_leaf_fallback(
                request=request,
                artifact=last_artifact,
                validation_errors=last_application_errors,
            )
            if fallback is not None and last_model is not None and last_output_path is not None:
                fallback_result, rejected_errors = fallback
                return SegmentResult(
                    indexes=[int(rows[0]["index"])],
                    paragraphs=[str(fallback_result["paragraphs"][0]["zh"])],
                    content_plan=[dict(fallback_result["content_plan"][0])],
                    style_cues_applied=[
                        str(value)
                        for value in fallback_result.get("style_cues_applied") or []
                    ],
                    style_cues_skipped=[
                        str(value)
                        for value in fallback_result.get("style_cues_skipped") or []
                    ],
                    uncertainties=[
                        str(value)
                        for value in fallback_result.get("uncertainties") or []
                    ],
                    artifacts=[
                        {
                            "segment": segment_name,
                            "depth": depth,
                            "source_indexes": [int(rows[0]["index"])],
                            "path": str(last_output_path.relative_to(self.style_run_dir)),
                            "file_sha256": file_sha256(last_output_path),
                            "request_sha256": sha256_json(request),
                            "model": last_model,
                            "acceptance": "neutral_leaf_fallback",
                            "rejected_application_errors": rejected_errors,
                        }
                    ],
                    usage=attempted_usage,
                    models_used=[last_model],
                )
            errors = [
                error
                for record in records
                for error in (
                    list(record.get("validation_errors") or [])
                    + list(record.get("application_validation_errors") or [])
                )
            ]
            raise RuntimeError(
                f"{chunk_id} {segment_name} failed after {self.max_attempts} attempt(s): "
                + "; ".join(errors[-5:])
            )

        midpoint = len(rows) // 2
        left = self.invoke_segment(
            chunk_id=chunk_id,
            chapter_id=chapter_id,
            rows=rows[:midpoint],
            segment_name=f"{segment_name}a",
            depth=depth + 1,
        )
        right = self.invoke_segment(
            chunk_id=chunk_id,
            chapter_id=chapter_id,
            rows=rows[midpoint:],
            segment_name=f"{segment_name}b",
            depth=depth + 1,
        )
        usage = dict(left.usage)
        for source in (attempted_usage, right.usage):
            for key, value in source.items():
                usage[key] = usage.get(key, 0) + value
        return SegmentResult(
            indexes=[*left.indexes, *right.indexes],
            paragraphs=[*left.paragraphs, *right.paragraphs],
            content_plan=[*left.content_plan, *right.content_plan],
            style_cues_applied=list(
                dict.fromkeys([*left.style_cues_applied, *right.style_cues_applied])
            ),
            style_cues_skipped=list(
                dict.fromkeys([*left.style_cues_skipped, *right.style_cues_skipped])
            ),
            uncertainties=list(dict.fromkeys([*left.uncertainties, *right.uncertainties])),
            artifacts=[*left.artifacts, *right.artifacts],
            usage=usage,
            models_used=list(dict.fromkeys([*left.models_used, *right.models_used])),
        )

    def run_chunk(self, chunk: dict[str, Any]) -> dict[str, Any]:
        chunk_id = str(chunk["chunk_id"])
        request_path = self.style_run_dir / str(chunk["request_path"])
        output_path = self.style_run_dir / str(chunk["json_output_path"])
        raw_output_path = self.style_run_dir / str(chunk["raw_output_path"])
        method_output_path = self.style_run_dir / str(chunk["method_output_path"])
        request = read_json(request_path)
        request_sha256 = sha256_json(request)
        if request_sha256 != chunk["request_sha256"]:
            raise RuntimeError(f"request hash mismatch: {chunk_id}")
        if not self.overwrite and output_path.exists() and method_output_path.exists():
            existing = read_json(method_output_path)
            if existing.get("request_sha256") == request_sha256:
                expected = [int(value) for value in chunk["indexes"]]
                actual = [
                    int(row["index"])
                    for row in (read_json(output_path).get("translations") or [])
                ]
                if actual == expected:
                    return {
                        "chunk_id": chunk_id,
                        "status": "skipped",
                        "usage": {},
                        "models_used": list(existing.get("models_used") or []),
                    }
        rows = [
            {
                "index": int(index),
                "english": request["english_semantic_source"][position]["en"],
                "neutral_zh": request["neutral_zh"][position]["zh"],
            }
            for position, index in enumerate(chunk["indexes"])
        ]
        segment = self.invoke_segment(
            chunk_id=chunk_id,
            chapter_id=str(chunk["chapter_id"]),
            rows=rows,
            segment_name="root",
            depth=0,
        )
        if segment.indexes != [int(value) for value in chunk["indexes"]]:
            raise RuntimeError(f"merged segment index mismatch: {chunk_id}")
        final_ids = [f"p{index + 1:04d}" for index in range(len(rows))]
        merged_result = {
            "sample_id": request["sample_id"],
            "method_id": METHOD_ID,
            "intensity": INTENSITY,
            "content_plan": [
                {**plan, "id": final_ids[position]}
                for position, plan in enumerate(segment.content_plan)
            ],
            "paragraphs": [
                {"id": final_ids[position], "zh": text}
                for position, text in enumerate(segment.paragraphs)
            ],
            "style_cues_applied": segment.style_cues_applied,
            "style_cues_skipped": segment.style_cues_skipped,
            "uncertainties": segment.uncertainties,
        }
        errors = application_validation_errors(request, merged_result)
        if errors:
            raise RuntimeError(f"merged style-transfer validation failed for {chunk_id}: {errors}")
        method_artifact = {
            "schema_version": 1,
            "run_schema": RUN_SCHEMA,
            "chunk_id": chunk_id,
            "chapter_id": str(chunk["chapter_id"]),
            "completed_at": utc_now(),
            "requested_model_order": self.models,
            "models_used": segment.models_used,
            "reasoning_effort": self.reasoning_effort,
            "method_id": METHOD_ID,
            "intensity": INTENSITY,
            "request_sha256": request_sha256,
            "result_sha256": sha256_json(merged_result),
            "method_payload_sha256": sha256_json(request["method_payload"]),
            "reference_examples_sha256": sha256_json(request["reference_examples"]),
            "result": merged_result,
            "segment_artifacts": segment.artifacts,
            "response_usage": segment.usage,
        }
        write_json(method_output_path, method_artifact)
        standard_output = {
            "chapter_title": str(chunk.get("chapter_title") or ""),
            "translations": [
                {"index": index, "zh": text}
                for index, text in zip(segment.indexes, segment.paragraphs)
            ],
            "glossary_candidates": [],
            "style_transfer_provenance": {
                "run_schema": RUN_SCHEMA,
                "method_id": METHOD_ID,
                "intensity": INTENSITY,
                "request_sha256": request_sha256,
                "method_output_path": str(
                    method_output_path.relative_to(self.style_run_dir)
                ),
                "method_output_sha256": file_sha256(method_output_path),
                "models_used": segment.models_used,
            },
        }
        write_json(output_path, standard_output)
        raw_output_path.write_text(
            json.dumps(standard_output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return {
            "chunk_id": chunk_id,
            "status": "completed",
            "usage": segment.usage,
            "models_used": segment.models_used,
        }


def run_style_transfer(
    *,
    style_run_dir: Path,
    config: dict[str, Any],
    models: Sequence[str] | None = None,
    codex_bin: str = "codex",
    reasoning_effort: str | None = None,
    timeout_seconds: int | None = None,
    max_attempts: int | None = None,
    workers: int | None = None,
    chunk_ids: set[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    style_run_dir = style_run_dir.expanduser().resolve()
    manifest_path = style_run_dir / "run_manifest.json"
    manifest = read_json(manifest_path)
    transfer = manifest.get("style_transfer") or {}
    if manifest.get("pass") != "author_style_transfer" or transfer.get("schema") != RUN_SCHEMA:
        raise ValueError("run manifest is not an author style-transfer run")
    validate_style_transfer_runtime_assets(style_run_dir)
    validate_style_transfer_source_inputs(style_run_dir)
    style_config = config.get("style_transfer") or {}
    requested_models = configured_model_order(config, models)
    runner = StyleTransferRunner(
        style_run_dir=style_run_dir,
        asset_path=resolve_repo_path(str(transfer["asset_path"])),
        asset_file_sha256=str(transfer["asset_file_sha256"]),
        prompt_path=resolve_repo_path(str(transfer["prompt_path"])),
        schema_path=resolve_repo_path(str(transfer["schema_path"])),
        models=requested_models,
        reasoning_effort=str(
            reasoning_effort or style_config.get("reasoning_effort") or "high"
        ),
        timeout_seconds=int(
            timeout_seconds or style_config.get("timeout_seconds") or 900
        ),
        max_attempts=int(max_attempts or style_config.get("max_attempts") or 2),
        codex_bin=codex_bin,
        overwrite=overwrite,
    )
    chunks = [
        chunk
        for chunk in manifest.get("chunks") or []
        if chunk_ids is None or str(chunk["chunk_id"]) in chunk_ids
    ]
    worker_count = int(workers or style_config.get("workers") or 2)
    if worker_count < 1:
        raise ValueError("style-transfer workers must be positive")
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(runner.run_chunk, chunk): chunk for chunk in chunks}
        completed_count = 0
        for future in as_completed(futures):
            chunk = futures[future]
            chunk_id = str(chunk["chunk_id"])
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                failures.append({"chunk_id": chunk_id, "error": str(exc)})
                status = "failed"
            else:
                results.append(result)
                status = str(result["status"])
            completed_count += 1
            with PRINT_LOCK:
                print(
                    f"style transfer {status} {chunk_id} ({completed_count}/{len(chunks)})",
                    flush=True,
                )
    usage: dict[str, int] = {}
    model_counts: dict[str, int] = {}
    for result in results:
        for key, value in (result.get("usage") or {}).items():
            usage[key] = usage.get(key, 0) + int(value)
        for model in result.get("models_used") or []:
            model_counts[str(model)] = model_counts.get(str(model), 0) + 1
    summary = {
        "schema_version": 1,
        "run_schema": RUN_SCHEMA,
        "completed_at": utc_now(),
        "requested_model_order": requested_models,
        "disabled_models": runner.disabled_models(),
        "model_chunk_counts": model_counts,
        "reasoning_effort": runner.reasoning_effort,
        "workers": worker_count,
        "overwrite": overwrite,
        "selected_chunk_count": len(chunks),
        "completed_count": sum(row["status"] == "completed" for row in results),
        "skipped_count": sum(row["status"] == "skipped" for row in results),
        "failure_count": len(failures),
        "failures": failures,
        "response_usage": usage,
    }
    write_json(style_run_dir / "style_transfer_summary.json", summary)
    transfer["status"] = "complete" if not failures else "incomplete"
    transfer["last_run"] = summary
    manifest["style_transfer"] = transfer
    write_json(manifest_path, manifest)
    return summary


def is_author_style_transfer_run(run_dir: Path) -> bool:
    manifest_path = run_dir.expanduser() / "run_manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = read_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        manifest.get("pass") == "author_style_transfer"
        and (manifest.get("style_transfer") or {}).get("schema") == RUN_SCHEMA
        and bool(manifest.get("semantic_run_dir"))
    )


def validate_style_transfer_runtime_assets(run_dir: Path) -> dict[str, str]:
    run_dir = run_dir.expanduser().resolve()
    manifest = read_json(run_dir / "run_manifest.json")
    transfer = manifest.get("style_transfer") or {}
    checks = (
        ("asset", transfer.get("asset_path"), transfer.get("asset_file_sha256")),
        ("prompt", transfer.get("prompt_path"), transfer.get("prompt_sha256")),
        ("schema", transfer.get("schema_path"), transfer.get("schema_sha256")),
    )
    validated: dict[str, str] = {}
    for label, raw_path, expected_hash in checks:
        if not raw_path or not expected_hash:
            raise ValueError(f"style-transfer manifest is missing {label} provenance")
        path = Path(str(raw_path)).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"style-transfer {label} is missing: {path}")
        actual_hash = file_sha256(path)
        if actual_hash != str(expected_hash):
            raise ValueError(
                f"style-transfer {label} changed after preparation: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        validated[label] = actual_hash
    return validated


def validate_style_transfer_source_inputs(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    manifest = read_json(run_dir / "run_manifest.json")
    transfer = manifest.get("style_transfer") or {}
    semantic_run_dir = Path(str(manifest.get("semantic_run_dir") or "")).expanduser().resolve()
    snapshot_dir = Path(str(manifest.get("snapshot_dir") or "")).expanduser().resolve()
    semantic_manifest_path = semantic_run_dir / "run_manifest.json"
    snapshot_manifest_path = snapshot_dir / "manifest.json"
    expected_files = (
        (
            "semantic manifest",
            semantic_manifest_path,
            transfer.get("semantic_manifest_sha256"),
        ),
        (
            "snapshot manifest",
            snapshot_manifest_path,
            transfer.get("snapshot_manifest_sha256"),
        ),
    )
    state: dict[str, Any] = {"files": {}, "chunks": []}
    for label, path, expected_hash in expected_files:
        if not expected_hash:
            raise ValueError(f"style-transfer manifest is missing {label} provenance")
        if not path.exists():
            raise FileNotFoundError(f"style-transfer {label} is missing: {path}")
        actual_hash = file_sha256(path)
        if actual_hash != str(expected_hash):
            raise ValueError(
                f"style-transfer {label} changed after preparation: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        state["files"][label] = actual_hash

    snapshot_manifest = load_manifest(snapshot_dir)
    english_cache: dict[str, dict[int, str]] = {}
    neutral_cache: dict[str, dict[int, str]] = {}
    for chunk in manifest.get("chunks") or []:
        chunk_id = str(chunk["chunk_id"])
        chapter_id = str(chunk["chapter_id"])
        indexes = [int(value) for value in chunk.get("indexes") or []]
        if chapter_id not in english_cache:
            chapter = load_chapter(snapshot_dir, snapshot_manifest, chapter_id)
            english_cache[chapter_id] = {
                int(item["index"]): str(item.get("english") or "").strip()
                for item in chapter.get("paragraphs") or []
            }
            neutral_cache[chapter_id] = translation_map(
                semantic_run_dir / "translations" / f"{chapter_id}.json"
            )
        try:
            english_sha256 = sha256_json(
                [english_cache[chapter_id][index] for index in indexes]
            )
            neutral_sha256 = sha256_json(
                [neutral_cache[chapter_id][index] for index in indexes]
            )
        except KeyError as exc:
            raise ValueError(
                f"{chunk_id}: current source inputs are missing paragraph {exc.args[0]}"
            ) from exc
        if english_sha256 != str(chunk.get("english_sha256") or ""):
            raise ValueError(f"{chunk_id}: English source changed after preparation")
        if neutral_sha256 != str(chunk.get("neutral_zh_sha256") or ""):
            raise ValueError(f"{chunk_id}: neutral Chinese changed after preparation")
        state["chunks"].append(
            {
                "chunk_id": chunk_id,
                "english_sha256": english_sha256,
                "neutral_zh_sha256": neutral_sha256,
            }
        )
    return {
        "validated_chunk_count": len(state["chunks"]),
        "state_sha256": sha256_json(state),
    }


def validate_style_transfer_provenance(run_dir: Path) -> dict[str, int]:
    run_dir = run_dir.expanduser().resolve()
    if not is_author_style_transfer_run(run_dir):
        raise ValueError("run directory is not an author style-transfer run")
    validate_style_transfer_runtime_assets(run_dir)
    source_inputs = validate_style_transfer_source_inputs(run_dir)
    manifest = read_json(run_dir / "run_manifest.json")
    validated = 0
    for chunk in manifest.get("chunks") or []:
        chunk_id = str(chunk["chunk_id"])
        request_path = run_dir / str(chunk["request_path"])
        method_output_path = run_dir / str(chunk["method_output_path"])
        output_path = run_dir / str(chunk["json_output_path"])
        if not request_path.exists():
            raise FileNotFoundError(f"{chunk_id}: missing style request {request_path}")
        request = read_json(request_path)
        request_sha256 = sha256_json(request)
        if request_sha256 != str(chunk.get("request_sha256") or ""):
            raise ValueError(f"{chunk_id}: request hash does not match run manifest")
        if not method_output_path.exists():
            raise FileNotFoundError(
                f"{chunk_id}: missing style method artifact {method_output_path}"
            )
        method_output = read_json(method_output_path)
        if method_output.get("run_schema") != RUN_SCHEMA:
            raise ValueError(f"{chunk_id}: method artifact schema mismatch")
        if method_output.get("method_id") != METHOD_ID:
            raise ValueError(f"{chunk_id}: method artifact method mismatch")
        if method_output.get("request_sha256") != request_sha256:
            raise ValueError(f"{chunk_id}: method artifact is stale for this request")
        if not output_path.exists():
            raise FileNotFoundError(f"{chunk_id}: missing style output {output_path}")
        output = read_json(output_path)
        provenance = output.get("style_transfer_provenance") or {}
        if provenance.get("run_schema") != RUN_SCHEMA:
            raise ValueError(f"{chunk_id}: standard output provenance schema mismatch")
        if provenance.get("method_id") != METHOD_ID:
            raise ValueError(f"{chunk_id}: standard output provenance method mismatch")
        if provenance.get("request_sha256") != request_sha256:
            raise ValueError(f"{chunk_id}: standard output is stale for this request")
        if provenance.get("method_output_sha256") != file_sha256(method_output_path):
            raise ValueError(f"{chunk_id}: method artifact hash mismatch")
        validated += 1
    return {
        "validated_chunk_count": validated,
        "validated_source_chunk_count": int(source_inputs["validated_chunk_count"]),
    }
