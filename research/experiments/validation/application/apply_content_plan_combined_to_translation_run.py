#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence


from experiments.shared.paths import RESEARCH_ROOT


REPO_ROOT = RESEARCH_ROOT

from experiments.iteration1 import run_style_transfer_generation as generation
from experiments.iteration4 import style_transfer_payloads as payloads
from src.crawler.snapshot import (
    load_chapter,
    load_manifest,
    snapshot_chapter_ids,
    write_json,
)


APPLICATION_SCHEMA = "eternal_gate_content_plan_combined_application.v1"
METHOD_ID = "content_plan_combined_full_regeneration"
INTENSITY = "strong"
DEFAULT_BLOCK_SIZE = 12
DEFAULT_MODEL_ORDER = (
    "gpt-5.6-sol",
    "gpt-5.5",
    "gpt-5.3-codex-spark",
    "gpt-5.4",
)
DEFAULT_EXPERIMENT_ROOT = (
    REPO_ROOT
    / "generated/style_research/style_transfer_experiments/iterations/full_regeneration_v1"
)
HAN_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]{8,}")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
NUMBER_RE = re.compile(r"\d+(?:[.,:/-]\d+)*")
LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]*")
PLACEHOLDER_RE = re.compile(r"<(?:TERM|CONTENT|NUM|LATIN)>")
JSON_RESIDUE_RE = re.compile(r"(?:\}\s*,\s*\{|```|\[\s*\{|\}\s*\])")
MODEL_CAPACITY_RE = re.compile(
    r"(?:unsupported_with_chatgpt_account|unsupported model|not supported when using codex|"
    r"model[_ -]?not[_ -]?found|"
    r"model is not available|not available for this account|capacity|usage limit|"
    r"rate limit|rate_limit|quota|too many requests)",
    re.IGNORECASE,
)
PRINT_LOCK = threading.Lock()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def model_capacity_failure(record: dict[str, Any]) -> bool:
    evidence = canonical_json(
        {
            "validation_errors": record.get("validation_errors") or [],
            "response_errors": record.get("response_errors") or [],
            "stderr_tail": record.get("stderr_tail") or "",
        }
    )
    return bool(MODEL_CAPACITY_RE.search(evidence))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def verify_iteration4_sources(experiment_root: Path) -> dict[str, str]:
    protocol_path = experiment_root / "protocols/evaluation_protocol.v1.json"
    protocol = read_json(protocol_path)
    bindings = protocol.get("iteration4", {}).get("source_bindings", {})
    if not isinstance(bindings, dict) or not bindings:
        raise RuntimeError("Iteration-4 protocol lacks frozen source bindings")
    verified: dict[str, str] = {}
    for path_text, expected in sorted(bindings.items()):
        path = REPO_ROOT / path_text
        observed = file_sha256(path)
        if observed != expected:
            raise RuntimeError(f"Iteration-4 source binding mismatch: {path_text}")
        verified[path_text] = observed
    return verified


def translation_map(path: Path) -> dict[int, str]:
    data = read_json(path)
    return {
        int(item["index"]): str(item.get("zh") or "").strip()
        for item in data.get("translations") or []
    }


def request_for_rows(
    *,
    experiment_root: Path,
    chapter_id: str,
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    indexes = [int(row["index"]) for row in rows]
    english_texts = [str(row["english"]) for row in rows]
    neutral_texts = [str(row["neutral_zh"]) for row in rows]
    paragraph_ids = [f"p{index + 1:04d}" for index in range(len(rows))]
    return payloads.build_method_request(
        experiment_root=experiment_root,
        method_id=METHOD_ID,
        intensity=INTENSITY,
        sample_id=stable_sample_id(
            chapter_id,
            indexes,
            english_texts,
            neutral_texts,
        ),
        english_semantic_source=[
            {"id": paragraph_ids[index], "en": text}
            for index, text in enumerate(english_texts)
        ],
        neutral_zh=[
            {"id": paragraph_ids[index], "zh": text}
            for index, text in enumerate(neutral_texts)
        ],
    )


def prepare_application_run(
    *,
    semantic_run_dir: Path,
    style_run_dir: Path,
    experiment_root: Path,
    block_size: int,
) -> dict[str, Any]:
    semantic_manifest_path = semantic_run_dir / "run_manifest.json"
    semantic_manifest = read_json(semantic_manifest_path)
    semantic_translations_dir = semantic_run_dir / "translations"
    if not semantic_translations_dir.exists():
        raise FileNotFoundError(
            f"validated semantic translations not found: {semantic_translations_dir}"
        )

    snapshot_dir = Path(str(semantic_manifest["snapshot_dir"])).expanduser().resolve()
    snapshot_manifest = load_manifest(snapshot_dir)
    verified_sources = verify_iteration4_sources(experiment_root)
    frozen_assets = payloads.load_frozen_assets(experiment_root)
    prompt_template_path = experiment_root / "prompts/style_transfer_method.v1.md"
    schema_path = experiment_root / "schemas/style_transfer_output.v1.schema.json"
    prompt_template = prompt_template_path.read_text(encoding="utf-8")

    style_run_dir.mkdir(parents=True, exist_ok=True)
    for directory in ("requests", "prompts", "outputs", "method_outputs"):
        (style_run_dir / directory).mkdir(parents=True, exist_ok=True)

    semantic_chunks = list(semantic_manifest.get("chunks") or [])
    expected_by_chapter: dict[str, list[int]] = {}
    title_by_chapter: dict[str, str] = {}
    for chunk in semantic_chunks:
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
            chunk_id = f"{chapter_id}_m4_{block_number:03d}"
            request = request_for_rows(
                experiment_root=experiment_root,
                chapter_id=chapter_id,
                rows=rows,
            )
            request_path = style_run_dir / "requests" / f"{chunk_id}.json"
            prompt_path = style_run_dir / "prompts" / f"{chunk_id}.md"
            write_json(request_path, request)
            prompt_path.write_text(
                generation.build_prompt(prompt_template, request),
                encoding="utf-8",
            )
            indexes = [int(row["index"]) for row in rows]
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "chapter_id": chapter_id,
                    "chapter_title": title_by_chapter.get(chapter_id)
                    or str(chapter.get("title") or ""),
                    "indexes": indexes,
                    "prompt_path": str(prompt_path.relative_to(style_run_dir)),
                    "request_path": str(request_path.relative_to(style_run_dir)),
                    "raw_output_path": f"outputs/{chunk_id}.raw.txt",
                    "json_output_path": f"outputs/{chunk_id}.json",
                    "method_output_path": f"method_outputs/{chunk_id}.json",
                    "request_sha256": sha256_json(request),
                    "prompt_sha256": file_sha256(prompt_path),
                    "english_sha256": sha256_json(
                        [row["english"] for row in rows]
                    ),
                    "neutral_zh_sha256": sha256_json(
                        [row["neutral_zh"] for row in rows]
                    ),
                }
            )
            total_paragraphs += len(rows)

    manifest = {
        "schema_version": 2,
        "pass": "style_reconstruction",
        "snapshot_dir": str(snapshot_dir),
        "semantic_run_dir": str(semantic_run_dir),
        "config": semantic_manifest.get("config") or {},
        "chunks": chunks,
        "method4_application": {
            "schema": APPLICATION_SCHEMA,
            "method_id": METHOD_ID,
            "intensity": INTENSITY,
            "capacity_fallback_policy": {
                "requested_model_order": list(DEFAULT_MODEL_ORDER),
                "switch_only_on": "explicit_model_capacity_or_availability_failure",
                "content_or_contract_failure_action": "same_model_retry_then_recursive_bisection",
            },
            "status": "prepared",
            "prepared_at": utc_now(),
            "block_size": block_size,
            "chunk_count": len(chunks),
            "paragraph_count": total_paragraphs,
            "experiment_root": str(experiment_root),
            "prompt_path": str(prompt_template_path),
            "prompt_sha256": file_sha256(prompt_template_path),
            "schema_path": str(schema_path),
            "schema_sha256": file_sha256(schema_path),
            "semantic_manifest_sha256": file_sha256(semantic_manifest_path),
            "snapshot_manifest_sha256": file_sha256(snapshot_dir / "manifest.json"),
            "frozen_asset_lock": {
                key: frozen_assets[key]
                for key in (
                    "schema_version",
                    "content_sha256",
                    "file_sha256",
                    "asset_path",
                )
            },
            "frozen_component_hashes": frozen_assets["assets"]["component_hashes"],
            "verified_iteration4_sources": verified_sources,
            "runner_path": str(Path(__file__).resolve()),
            "runner_sha256": file_sha256(Path(__file__).resolve()),
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
        if sorted(NUMBER_RE.findall(source_text)) != sorted(
            NUMBER_RE.findall(output_text)
        ):
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
    source_total = sum(
        len(CJK_RE.findall(str(row.get("zh") or ""))) for row in neutral
    )
    output_total = sum(
        len(CJK_RE.findall(str(row.get("zh") or ""))) for row in candidate
    )
    aggregate_ratio = output_total / max(source_total, 1)
    if not 0.70 <= aggregate_ratio <= 1.35:
        failures.append("fidelity:aggregate_cjk_ratio_out_of_bounds")
    return failures


def application_validation_errors(
    request: dict[str, Any], result: dict[str, Any]
) -> list[str]:
    errors = generation.validate_result("style_transfer", request, result)
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


def add_response_usage(target: dict[str, int], record: dict[str, Any]) -> None:
    for key, value in (record.get("response_usage") or {}).items():
        if isinstance(value, int):
            target[key] = target.get(key, 0) + value


def neutral_leaf_fallback(
    *,
    request: dict[str, Any],
    artifact: dict[str, Any] | None,
    validation_errors: Sequence[str],
) -> tuple[dict[str, Any], list[str]] | None:
    if artifact is None:
        return None
    result = artifact.get("result")
    if not isinstance(result, dict):
        return None
    non_fidelity_errors = [
        error for error in validation_errors if not error.startswith("fidelity:")
    ]
    if non_fidelity_errors:
        return None
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
    fallback_errors = application_validation_errors(request, fallback)
    if fallback_errors:
        return None
    return fallback, list(validation_errors)


class ApplicationRunner:
    def __init__(
        self,
        *,
        style_run_dir: Path,
        experiment_root: Path,
        models: Sequence[str],
        reasoning_effort: str,
        timeout_seconds: int,
        max_attempts: int,
        overwrite: bool,
    ) -> None:
        self.style_run_dir = style_run_dir
        self.experiment_root = experiment_root
        self.models = list(dict.fromkeys(models))
        if not self.models:
            raise ValueError("at least one model is required")
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.overwrite = overwrite
        self.prompt_path = experiment_root / "prompts/style_transfer_method.v1.md"
        self.schema_path = experiment_root / "schemas/style_transfer_output.v1.schema.json"
        self.prompt_template = self.prompt_path.read_text(encoding="utf-8")
        self.prompt_sha256 = file_sha256(self.prompt_path)
        self.schema_sha256 = file_sha256(self.schema_path)
        self.codex_binary = generation.resolved_codex_binary()
        self.codex_cli_version = generation.codex_version()
        self.runner_sha256 = file_sha256(Path(__file__).resolve())
        self._disabled_models: dict[str, str] = {}
        self._model_lock = threading.Lock()
        environment = {
            "application_schema": APPLICATION_SCHEMA,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "codex_cli": self.codex_cli_version,
        }
        self.environment_sha256 = sha256_json(environment)
        self.sandbox_profile_sha256 = hashlib.sha256(
            generation.external_sandbox_profile().encode("utf-8")
        ).hexdigest()

    def active_models(self) -> list[str]:
        with self._model_lock:
            active = [model for model in self.models if model not in self._disabled_models]
            disabled = dict(self._disabled_models)
        if not active:
            raise RuntimeError(f"all requested models are at capacity or unavailable: {disabled}")
        return active

    def disable_model(self, model: str, record: dict[str, Any]) -> None:
        reason = canonical_json(
            {
                "validation_errors": record.get("validation_errors") or [],
                "response_errors": record.get("response_errors") or [],
                "stderr_tail": record.get("stderr_tail") or "",
            }
        )[-2000:]
        with self._model_lock:
            self._disabled_models.setdefault(model, reason)

    def disabled_models(self) -> dict[str, str]:
        with self._model_lock:
            return dict(self._disabled_models)

    def effective_command_sha256(self, model: str) -> str:
        return sha256_json(
            {
                "codex_binary": str(self.codex_binary),
                "model": model,
                "reasoning_effort": self.reasoning_effort,
                "timeout_seconds": self.timeout_seconds,
                "runner": "run_style_transfer_generation.run_one_attempt",
            }
        )

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
            experiment_root=self.experiment_root,
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
                execution_binding = {
                    "application_binding": {
                        "schema": APPLICATION_SCHEMA,
                        "chunk_id": chunk_id,
                        "chapter_id": chapter_id,
                        "segment": segment_name,
                        "recursive_depth": depth,
                        "source_indexes": [int(row["index"]) for row in rows],
                        "request_sha256": sha256_json(request),
                        "requested_model_order": self.models,
                        "effective_model": model,
                    }
                }
                record = generation.run_one_attempt(
                    stage="style_transfer",
                    sample_id=request["sample_id"],
                    request=request,
                    prompt_template=self.prompt_template,
                    prompt_sha256=self.prompt_sha256,
                    schema_path=self.schema_path,
                    schema_sha256=self.schema_sha256,
                    output_path=output_path,
                    run_id=f"method4-{chunk_id}",
                    attempt=attempt,
                    model_config={
                        "codex_model": model,
                        "provider": "openai",
                        "reasoning_effort": self.reasoning_effort,
                        "request_timeout_seconds": self.timeout_seconds,
                    },
                    codex_cli_version=self.codex_cli_version,
                    codex_binary=self.codex_binary,
                    runner_sha256=self.runner_sha256,
                    environment_sha256=self.environment_sha256,
                    sandbox_profile_sha256=self.sandbox_profile_sha256,
                    effective_command_sha256=self.effective_command_sha256(model),
                    execution_binding=execution_binding,
                )
                record["requested_model"] = model
                records.append(record)
                add_response_usage(attempted_usage, record)
                if record.get("status") != "success" or not output_path.exists():
                    if model_capacity_failure(record):
                        self.disable_model(model, record)
                        switch_for_capacity = True
                        with PRINT_LOCK:
                            print(
                                f"method4 model unavailable {model}; switching fallback",
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
                            "response_id": records[-1].get("response_id"),
                            "attempt": records[-1].get("attempt"),
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
        for key, value in attempted_usage.items():
            usage[key] = usage.get(key, 0) + value
        for key, value in right.usage.items():
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
        merged_plan = [
            {**plan, "id": final_ids[position]}
            for position, plan in enumerate(segment.content_plan)
        ]
        merged_result = {
            "sample_id": request["sample_id"],
            "method_id": METHOD_ID,
            "intensity": INTENSITY,
            "content_plan": merged_plan,
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
            raise RuntimeError(f"merged method4 validation failed for {chunk_id}: {errors}")

        method_artifact = {
            "schema_version": 1,
            "application_schema": APPLICATION_SCHEMA,
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
            "method4_provenance": {
                "application_schema": APPLICATION_SCHEMA,
                "method_id": METHOD_ID,
                "intensity": INTENSITY,
                "request_sha256": request_sha256,
                "method_output_path": str(method_output_path.relative_to(self.style_run_dir)),
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


def run_application(
    *,
    style_run_dir: Path,
    experiment_root: Path,
    models: Sequence[str],
    reasoning_effort: str,
    timeout_seconds: int,
    max_attempts: int,
    workers: int,
    chunk_ids: set[str] | None,
    overwrite: bool,
) -> dict[str, Any]:
    manifest_path = style_run_dir / "run_manifest.json"
    manifest = read_json(manifest_path)
    application = manifest.get("method4_application") or {}
    if application.get("schema") != APPLICATION_SCHEMA:
        raise ValueError("run manifest is not a method4 application run")
    runner = ApplicationRunner(
        style_run_dir=style_run_dir,
        experiment_root=experiment_root,
        models=models,
        reasoning_effort=reasoning_effort,
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
        overwrite=overwrite,
    )
    chunks = [
        chunk
        for chunk in manifest.get("chunks") or []
        if chunk_ids is None or str(chunk["chunk_id"]) in chunk_ids
    ]
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
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
                    f"method4 {status} {chunk_id} "
                    f"({completed_count}/{len(chunks)})",
                    flush=True,
                )

    usage: dict[str, int] = {}
    for result in results:
        for key, value in (result.get("usage") or {}).items():
            usage[key] = usage.get(key, 0) + int(value)
    model_counts: dict[str, int] = {}
    for result in results:
        for model in result.get("models_used") or []:
            model_counts[str(model)] = model_counts.get(str(model), 0) + 1
    summary = {
        "schema_version": 1,
        "application_schema": APPLICATION_SCHEMA,
        "completed_at": utc_now(),
        "requested_model_order": list(models),
        "disabled_models": runner.disabled_models(),
        "model_chunk_counts": model_counts,
        "reasoning_effort": reasoning_effort,
        "workers": workers,
        "overwrite": overwrite,
        "selected_chunk_count": len(chunks),
        "completed_count": sum(row["status"] == "completed" for row in results),
        "skipped_count": sum(row["status"] == "skipped" for row in results),
        "failure_count": len(failures),
        "failures": failures,
        "response_usage": usage,
    }
    write_json(style_run_dir / "method4_run_summary.json", summary)
    application["status"] = "complete" if not failures else "incomplete"
    application["last_run"] = summary
    manifest["method4_application"] = application
    write_json(manifest_path, manifest)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the frozen Iteration-4 content-plan combined method to a "
            "validated semantic translation run."
        )
    )
    parser.add_argument("--semantic-run-dir", type=Path, required=True)
    parser.add_argument("--style-run-dir", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help=(
            "Model in capacity-fallback order; repeat for multiple models. "
            "Defaults to gpt-5.6-sol, gpt-5.5, gpt-5.3-codex-spark, gpt-5.4."
        ),
    )
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--chunk-id", action="append", dest="chunk_ids")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    semantic_run_dir = args.semantic_run_dir.expanduser().resolve()
    style_run_dir = args.style_run_dir.expanduser().resolve()
    experiment_root = args.experiment_root.expanduser().resolve()
    if args.block_size < 1 or args.block_size > 12:
        raise ValueError("block size must be between 1 and 12")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if not args.skip_prepare:
        manifest = prepare_application_run(
            semantic_run_dir=semantic_run_dir,
            style_run_dir=style_run_dir,
            experiment_root=experiment_root,
            block_size=args.block_size,
        )
        application = manifest["method4_application"]
        print(
            f"prepared {application['chunk_count']} method4 block(s) / "
            f"{application['paragraph_count']} paragraph(s)",
            flush=True,
        )
    if args.prepare_only:
        return 0
    summary = run_application(
        style_run_dir=style_run_dir,
        experiment_root=experiment_root,
        models=args.models or DEFAULT_MODEL_ORDER,
        reasoning_effort=args.reasoning_effort,
        timeout_seconds=args.timeout_seconds,
        max_attempts=args.max_attempts,
        workers=args.workers,
        chunk_ids=set(args.chunk_ids) if args.chunk_ids else None,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 1 if summary["failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
