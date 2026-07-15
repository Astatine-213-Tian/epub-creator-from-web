#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

from experiments.validation.construct.construct_v2_protocol import han_ngram_overlap


ROOT = Path(
    "generated/style_research/style_transfer_experiments/iterations/"
    "content_resistant_v1/cr_fysm_v4/construct_validation_v2"
)
PREREGISTRATION = ROOT / "generation_preregistration.v2.json"
SELECTION = ROOT / "source_selection.private.jsonl"
STAGES = (
    "english_semantic_source",
    "english_adjudication",
    "neutral_translation",
    "style_sham_control",
    "style_positive_control",
)
DEPENDENCIES = {
    "english_semantic_source": (),
    "english_adjudication": ("english_semantic_source",),
    "neutral_translation": ("english_adjudication",),
    "style_sham_control": ("english_adjudication", "neutral_translation"),
    "style_positive_control": ("english_adjudication", "neutral_translation"),
}
PRINT_LOCK = threading.Lock()
HAN_SCRIPT_RANGES = (
    # Unicode 17.0 Scripts.txt entries whose Script property is Han.
    (0x2E80, 0x2E99),
    (0x2E9B, 0x2EF3),
    (0x2F00, 0x2FD5),
    (0x3005, 0x3005),
    (0x3007, 0x3007),
    (0x3021, 0x3029),
    (0x3038, 0x303A),
    (0x303B, 0x303B),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFA6D),
    (0xFA70, 0xFAD9),
    (0x16FE2, 0x16FE3),
    (0x16FF0, 0x16FF6),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B81D),
    (0x2B820, 0x2CEAD),
    (0x2CEB0, 0x2EBE0),
    (0x2EBF0, 0x2EE5D),
    (0x2F800, 0x2FA1D),
    (0x30000, 0x3134A),
    (0x31350, 0x33479),
)
CODEX_DISABLED_FEATURES = (
    "shell_tool",
    "unified_exec",
    "browser_use",
    "browser_use_external",
    "browser_use_full_cdp_access",
    "in_app_browser",
    "computer_use",
    "apps",
    "multi_agent",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frozen fresh-domain CR-FYSM-v4 construct generation."
    )
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sample-id", action="append", default=[])
    return parser.parse_args()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def lock_id(payload: dict[str, Any]) -> str:
    value = dict(payload)
    value.pop("lock_id", None)
    return sha256_text(canonical(value))


def is_han_character(char: str) -> bool:
    codepoint = ord(char)
    return any(lower <= codepoint <= upper for lower, upper in HAN_SCRIPT_RANGES)


def contains_han(text: str) -> bool:
    return any(is_han_character(char) for char in text)


def codex_command_template(codex_binary: str) -> list[str]:
    command = [
        codex_binary,
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--skip-git-repo-check",
        "--dangerously-bypass-approvals-and-sandbox",
    ]
    for feature in CODEX_DISABLED_FEATURES:
        command.extend(("--disable", feature))
    command.extend(
        (
            "--cd",
            "<isolated_working_directory>",
            "--model",
            "<stage_model_alias>",
            "--config",
            'model_reasoning_effort="<stage_reasoning_effort>"',
            "--output-schema",
            "<isolated_schema_path>",
            "--output-last-message",
            "<isolated_response_path>",
            "--json",
            "-",
        )
    )
    return command


def current_codex_runtime() -> dict[str, str | list[str]]:
    executable = shutil.which("codex")
    if not executable:
        raise FileNotFoundError("codex executable not found")
    wrapper = Path(executable).resolve()
    native_candidates = sorted(
        path
        for path in wrapper.parent.parent.parent.glob("codex-*/vendor/*/bin/codex*")
        if path.is_file() and path.name in {"codex", "codex.exe"}
    )
    if len(native_candidates) != 1:
        raise ValueError(f"expected one installed native Codex binary, found {native_candidates}")
    resolved = native_candidates[0].resolve()
    completed = subprocess.run(
        [str(resolved), "--version"],
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    template = codex_command_template(str(resolved))
    return {
        "codex_wrapper_resolved": str(wrapper),
        "codex_wrapper_sha256": sha256_file(wrapper),
        "codex_executable_resolved": str(resolved),
        "codex_executable_sha256": sha256_file(resolved),
        "codex_version": completed.stdout.strip() or completed.stderr.strip(),
        "command_template": template,
        "command_template_sha256": sha256_text(canonical(template)),
    }


def validate_preregistration() -> dict[str, Any]:
    payload = read_json(PREREGISTRATION)
    if payload.get("status") != "locked_before_any_construct_generation":
        raise ValueError("construct generation preregistration is not locked")
    if payload.get("lock_id") != lock_id(payload):
        raise ValueError("construct generation lock_id is invalid")
    actual = {path: sha256_file(Path(path)) for path in payload["input_hashes"]}
    if actual != payload["input_hashes"]:
        changed = sorted(path for path in actual if actual[path] != payload["input_hashes"][path])
        raise ValueError(f"construct generation inputs changed after lock: {changed}")
    if payload["selection"]["sha256"] != sha256_file(SELECTION):
        raise ValueError("construct source selection changed after lock")
    current_runtime = current_codex_runtime()
    for key, current_value in current_runtime.items():
        if payload["runtime"].get(key) != current_value:
            raise ValueError(f"Codex runtime changed after lock: {key}")
    return payload


def output_path(stage: str, sample_id: str) -> Path:
    return ROOT / "generation" / stage / f"{sample_id}.json"


def load_artifact(
    stage: str,
    sample: dict[str, Any],
    preregistration: dict[str, Any],
) -> dict[str, Any]:
    sample_id = sample["sample_id"]
    path = output_path(stage, sample_id)
    if not path.exists():
        raise FileNotFoundError(f"missing dependency {stage}/{sample_id}")
    artifact = read_json(path)
    request = build_request(stage, sample, preregistration)
    validate_artifact(stage, sample, preregistration, request, artifact)
    return artifact


def load_result(
    stage: str,
    sample: dict[str, Any],
    preregistration: dict[str, Any],
) -> dict[str, Any]:
    return load_artifact(stage, sample, preregistration)["result"]


def build_request(
    stage: str,
    sample: dict[str, Any],
    preregistration: dict[str, Any],
) -> dict[str, Any]:
    sample_id = sample["sample_id"]
    original = sample["paragraphs"]
    if stage == "english_semantic_source":
        return {
            "sample_id": sample_id,
            "paragraphs": original,
            "terminology_placeholders": {},
        }
    adjudicated = None
    if stage in {"neutral_translation", "style_sham_control", "style_positive_control"}:
        adjudicated = load_result("english_adjudication", sample, preregistration)["paragraphs"]
    if stage == "english_adjudication":
        draft = load_result("english_semantic_source", sample, preregistration)["paragraphs"]
        draft_by_id = {row["id"]: row["en"] for row in draft}
        return {
            "sample_id": sample_id,
            "paragraphs": [
                {"id": row["id"], "zh": row["zh"], "draft_en": draft_by_id[row["id"]]}
                for row in original
            ],
        }
    if stage == "neutral_translation":
        return {
            "sample_id": sample_id,
            "paragraphs": adjudicated,
            "glossary": {},
            "comments": [],
        }
    if stage == "style_sham_control":
        neutral = load_result("neutral_translation", sample, preregistration)["paragraphs"]
        return {
            "sample_id": sample_id,
            "english_paragraphs": adjudicated,
            "neutral_chinese": neutral,
        }
    if stage == "style_positive_control":
        neutral = load_result("neutral_translation", sample, preregistration)["paragraphs"]
        return {
            "sample_id": sample_id,
            "english_paragraphs": adjudicated,
            "neutral_chinese": neutral,
            "style_t_definition": preregistration["style_evidence"]["dimensions"],
            "reference_examples": preregistration["style_evidence"]["reference_examples"],
        }
    raise AssertionError(stage)


def expected_ids(stage: str, request: dict[str, Any]) -> list[str]:
    key = (
        "english_paragraphs"
        if stage in {"style_sham_control", "style_positive_control"}
        else "paragraphs"
    )
    return [row["id"] for row in request[key]]


def reference_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise TypeError(f"unsupported reference example value: {type(value).__name__}")
    return "\n".join(str(paragraph["zh"]) for paragraph in value)


def copied_reference_ids(
    candidate: str,
    examples: list[dict[str, Any]],
    *,
    window_size: int = 8,
) -> list[str]:
    candidate_cjk = "".join(char for char in candidate if is_han_character(char))
    hits: list[str] = []
    for example in examples:
        copied = False
        for field in ("neutral_zh", "target_style_zh"):
            reference = "".join(
                char
                for char in reference_text(example[field])
                if is_han_character(char)
            )
            windows = {
                reference[index : index + window_size]
                for index in range(max(0, len(reference) - window_size + 1))
            }
            if any(window and window in candidate_cjk for window in windows):
                hits.append(f"{example['pair_id']}:{field}")
                copied = True
                break
        if copied:
            continue
    return hits


def validate_schema_instance(
    instance: Any,
    schema: dict[str, Any],
    *,
    location: str = "$",
) -> list[str]:
    errors: list[str] = []
    expected_type = schema.get("type")
    type_checks = {
        "object": lambda value: isinstance(value, dict),
        "array": lambda value: isinstance(value, list),
        "string": lambda value: isinstance(value, str),
        "boolean": lambda value: isinstance(value, bool),
    }
    if expected_type in type_checks and not type_checks[expected_type](instance):
        return [f"{location}: expected {expected_type}, got {type(instance).__name__}"]
    if "const" in schema and instance != schema["const"]:
        errors.append(f"{location}: value does not match const")
    if isinstance(instance, str):
        if len(instance) < int(schema.get("minLength", 0)):
            errors.append(f"{location}: shorter than minLength")
        pattern = schema.get("pattern")
        if pattern and re.search(str(pattern), instance) is None:
            errors.append(f"{location}: does not match pattern {pattern}")
    if isinstance(instance, list):
        if len(instance) < int(schema.get("minItems", 0)):
            errors.append(f"{location}: fewer than minItems")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, value in enumerate(instance):
                errors.extend(
                    validate_schema_instance(value, item_schema, location=f"{location}[{index}]")
                )
    if isinstance(instance, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for key in required:
            if key not in instance:
                errors.append(f"{location}: missing required property {key}")
        if schema.get("additionalProperties") is False:
            for key in instance:
                if key not in properties:
                    errors.append(f"{location}: unexpected property {key}")
        for key, value in instance.items():
            child_schema = properties.get(key)
            if isinstance(child_schema, dict):
                errors.extend(
                    validate_schema_instance(value, child_schema, location=f"{location}.{key}")
                )
    return errors


def validate_result(
    stage: str,
    request: dict[str, Any],
    result: dict[str, Any],
    preregistration: dict[str, Any],
) -> list[str]:
    schema = read_json(Path(preregistration["generation_stages"][stage]["schema_path"]))
    errors = [
        f"JSON Schema: {error}"
        for error in validate_schema_instance(result, schema)
    ]
    if result.get("sample_id") != request["sample_id"]:
        errors.append("sample_id mismatch")
    paragraphs = result.get("paragraphs")
    if not isinstance(paragraphs, list):
        return [*errors, "paragraphs is not a list"]
    ids = [row.get("id") for row in paragraphs]
    if ids != expected_ids(stage, request):
        errors.append("paragraph IDs or order differ from input")
    text_key = "en" if stage in {"english_semantic_source", "english_adjudication"} else "zh"
    if any(not isinstance(row.get(text_key), str) or not row[text_key].strip() for row in paragraphs):
        errors.append(f"one or more {text_key} outputs are empty")
    if stage in {"english_semantic_source", "english_adjudication"} and any(
        contains_han(row.get("en", "")) for row in paragraphs
    ):
        errors.append("English output contains CJK characters")
    forbidden = ("I cannot", "I can't", "无法协助", "不能协助", "抱歉")
    if any(token in row.get(text_key, "") for row in paragraphs for token in forbidden):
        errors.append("refusal or placeholder text detected")
    if stage == "english_adjudication" and result.get("approved") is not True:
        errors.append("English adjudicator did not approve its corrected output")
    if stage == "style_positive_control":
        candidate = "\n".join(row["zh"] for row in paragraphs)
        examples = preregistration["style_evidence"]["reference_examples"]
        copied = copied_reference_ids(
            candidate,
            examples,
        )
        if copied:
            errors.append(f"eight-CJK reference copy detected: {copied[0]}")
        references = [
            reference_text(example[field])
            for example in examples
            for field in ("neutral_zh", "target_style_zh")
        ]
        overlap = han_ngram_overlap(candidate, references, n=4)
        maximum = float(
            preregistration["reference_overlap_control"]["maximum_overlap_rate"]
        )
        if overlap > maximum:
            errors.append(
                f"aggregate reference four-gram overlap {overlap:.6f} exceeds {maximum:.6f}"
            )
    return errors


def validate_artifact(
    stage: str,
    sample: dict[str, Any],
    preregistration: dict[str, Any],
    request: dict[str, Any],
    artifact: dict[str, Any],
) -> None:
    sample_id = sample["sample_id"]
    config = preregistration["generation_stages"][stage]
    prompt_path = Path(config["prompt_path"])
    schema_path = Path(config["schema_path"])
    expected_dependencies = {
        dependency: load_artifact(dependency, sample, preregistration)["result_sha256"]
        for dependency in DEPENDENCIES[stage]
    }
    expected = {
        "schema_version": 2,
        "status": "complete",
        "construct_id": preregistration["construct_id"],
        "generation_lock_id": preregistration["lock_id"],
        "stage": stage,
        "sample_id": sample_id,
        "model": config["model"],
        "reasoning_effort": config["reasoning_effort"],
        "prompt_sha256": sha256_file(prompt_path),
        "schema_sha256": sha256_file(schema_path),
        "request_sha256": sha256_text(canonical(request)),
        "dependency_result_sha256": expected_dependencies,
    }
    mismatches = [key for key, value in expected.items() if artifact.get(key) != value]
    if mismatches:
        raise ValueError(f"invalid or stale artifact {stage}/{sample_id}: {mismatches}")
    result = artifact.get("result")
    if not isinstance(result, dict):
        raise ValueError(f"artifact result is not an object: {stage}/{sample_id}")
    if artifact.get("result_sha256") != sha256_text(canonical(result)):
        raise ValueError(f"artifact result hash mismatch: {stage}/{sample_id}")
    errors = validate_result(stage, request, result, preregistration)
    if errors:
        raise ValueError(f"artifact result no longer validates: {stage}/{sample_id}: {errors}")


def codex_command(
    *,
    codex_binary: str,
    model: str,
    reasoning_effort: str,
    schema: Path,
    response: Path,
    cwd: Path,
) -> list[str]:
    command = codex_command_template(codex_binary)
    replacements = {
        "<isolated_working_directory>": str(cwd),
        "<stage_model_alias>": model,
        'model_reasoning_effort="<stage_reasoning_effort>"': (
            f'model_reasoning_effort="{reasoning_effort}"'
        ),
        "<isolated_schema_path>": str(schema),
        "<isolated_response_path>": str(response),
    }
    return [replacements.get(value, value) for value in command]


def run_sample(
    stage: str,
    sample: dict[str, Any],
    preregistration: dict[str, Any],
    *,
    resume: bool,
) -> tuple[str, str]:
    sample_id = sample["sample_id"]
    destination = output_path(stage, sample_id)
    if destination.exists():
        if resume:
            artifact = read_json(destination)
            request = build_request(stage, sample, preregistration)
            validate_artifact(stage, sample, preregistration, request, artifact)
            return sample_id, "skipped"
        raise FileExistsError(f"output already exists: {destination}")
    for dependency in DEPENDENCIES[stage]:
        load_result(dependency, sample, preregistration)

    config = preregistration["generation_stages"][stage]
    prompt_path = Path(config["prompt_path"])
    schema_path = Path(config["schema_path"])
    prompt = prompt_path.read_text(encoding="utf-8")
    request = build_request(stage, sample, preregistration)
    request_sha256 = sha256_text(canonical(request))
    codex_binary = str(preregistration["runtime"]["codex_executable_resolved"])
    dependency_hashes = {
        dependency: load_artifact(dependency, sample, preregistration)["result_sha256"]
        for dependency in DEPENDENCIES[stage]
    }

    errors: list[str] = []
    for attempt in range(1, int(config["max_attempts"]) + 1):
        with tempfile.TemporaryDirectory(prefix=f"cr-fysm-v4-{stage}-") as temp:
            temp_path = Path(temp)
            isolated_schema = temp_path / "schema.json"
            isolated_schema.write_bytes(schema_path.read_bytes())
            response_path = temp_path / "response.json"
            command = codex_command(
                codex_binary=codex_binary,
                model=config["model"],
                reasoning_effort=config["reasoning_effort"],
                schema=isolated_schema,
                response=response_path,
                cwd=temp_path,
            )
            completed = subprocess.run(
                command,
                input=prompt + "\n\n## Request JSON\n\n" + json.dumps(request, ensure_ascii=False),
                text=True,
                capture_output=True,
                timeout=int(config["timeout_seconds"]),
                env={
                    key: value
                    for key, value in os.environ.items()
                    if key in {"HOME", "LANG", "LC_ALL", "LOGNAME", "PATH", "TMPDIR", "USER"}
                },
            )
            if completed.returncode != 0:
                errors.append(f"attempt {attempt}: codex exit {completed.returncode}: {completed.stderr[-500:]}")
                continue
            try:
                result = read_json(response_path)
            except (FileNotFoundError, json.JSONDecodeError) as exc:
                errors.append(f"attempt {attempt}: invalid response: {exc}")
                continue
            validation_errors = validate_result(stage, request, result, preregistration)
            if validation_errors:
                errors.extend(f"attempt {attempt}: {error}" for error in validation_errors)
                continue
            usage = None
            response_id = None
            for line in completed.stdout.splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                response_id = response_id or event.get("thread_id")
                if event.get("type") == "turn.completed":
                    usage = event.get("usage")
            artifact = {
                "schema_version": 2,
                "status": "complete",
                "construct_id": preregistration["construct_id"],
                "generation_lock_id": preregistration["lock_id"],
                "stage": stage,
                "sample_id": sample_id,
                "attempt": attempt,
                "model": config["model"],
                "reasoning_effort": config["reasoning_effort"],
                "prompt_sha256": sha256_file(prompt_path),
                "schema_sha256": sha256_file(schema_path),
                "request_sha256": request_sha256,
                "dependency_result_sha256": dependency_hashes,
                "result_sha256": sha256_text(canonical(result)),
                "response_id": response_id,
                "usage": usage,
                "prior_attempt_errors": errors,
                "result": result,
            }
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp")
            temporary.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(destination)
            return sample_id, "complete"
    raise RuntimeError(f"{stage}/{sample_id} failed: {'; '.join(errors[-6:])}")


def main() -> None:
    args = parse_args()
    preregistration = validate_preregistration()
    samples = read_jsonl(SELECTION)
    if args.sample_id:
        requested = set(args.sample_id)
        samples = [row for row in samples if row["sample_id"] in requested]
        missing = requested - {row["sample_id"] for row in samples}
        if missing:
            raise ValueError(f"unknown sample IDs: {sorted(missing)}")
    failures: list[tuple[str, str]] = []
    counts = {"complete": 0, "skipped": 0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {
            pool.submit(
                run_sample,
                args.stage,
                sample,
                preregistration,
                resume=args.resume,
            ): sample["sample_id"]
            for sample in samples
        }
        for future in concurrent.futures.as_completed(futures):
            sample_id = futures[future]
            try:
                _, status = future.result()
                counts[status] += 1
                with PRINT_LOCK:
                    print(f"[{status}] {args.stage}/{sample_id}", flush=True)
            except Exception as exc:  # noqa: BLE001 - preserve independent failures
                failures.append((sample_id, str(exc)))
                with PRINT_LOCK:
                    print(f"[failed] {args.stage}/{sample_id}: {exc}", flush=True)
    summary = {
        "stage": args.stage,
        "expected": len(samples),
        **counts,
        "failed": len(failures),
        "failures": failures,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
