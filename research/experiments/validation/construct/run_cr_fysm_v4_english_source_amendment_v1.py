#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

from experiments.validation.construct import run_cr_fysm_v4_construct_generation as generation


AMENDMENT_ID = "CR-FYSM-v4-construct-v2-english-refusal-false-positive-amendment-v1"
SAMPLE_IDS = (
    "s_3e009c1ab42bafc0d8cb2acb",
    "s_8dcd648438fb31e8723260b8",
    "s_eefdbffe2ab5a722e2cef536",
    "s_ffb204cf626b39b6dd4dd916",
)
REPLACEMENTS = (
    ("I cannot", "I am unable to"),
    ("I can't", "I am unable to"),
)
AMENDMENT_ROOT = generation.ROOT / "protocol_amendment_english_source_v1"
DECLARATION = AMENDMENT_ROOT / "preregistration.json"
RAW_ATTEMPTS = AMENDMENT_ROOT / "raw_attempts"
PRINT_LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the predeclared English refusal false-positive amendment."
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def declaration_id(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("declaration_id", None)
    return sha256_text(canonical(payload))


def validate_declaration(preregistration: dict[str, Any]) -> dict[str, Any]:
    declaration = read_json(DECLARATION)
    if declaration.get("status") != "declared_before_amendment_rerun":
        raise ValueError("protocol amendment is not predeclared")
    if declaration.get("amendment_id") != AMENDMENT_ID:
        raise ValueError("protocol amendment ID mismatch")
    if declaration.get("declaration_id") != declaration_id(declaration):
        raise ValueError("protocol amendment declaration_id is invalid")
    if tuple(declaration.get("sample_ids", ())) != SAMPLE_IDS:
        raise ValueError("protocol amendment sample IDs changed")
    if declaration.get("generation_lock_id") != preregistration["lock_id"]:
        raise ValueError("protocol amendment does not bind the generation lock")
    if declaration.get("runner_sha256") != generation.sha256_file(Path(__file__)):
        raise ValueError("protocol amendment runner changed after declaration")
    if declaration.get("source_selection_sha256") != generation.sha256_file(
        generation.SELECTION
    ):
        raise ValueError("protocol amendment source selection changed")
    expected_replacements = [
        {"exact": source, "replacement": destination}
        for source, destination in REPLACEMENTS
    ]
    if declaration.get("deterministic_replacements") != expected_replacements:
        raise ValueError("protocol amendment replacement rule changed")
    return declaration


def response_metadata(stdout: str) -> tuple[str | None, dict[str, Any] | None]:
    response_id = None
    usage = None
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        response_id = response_id or event.get("thread_id")
        if event.get("type") == "turn.completed":
            usage = event.get("usage")
    return response_id, usage


def amend_result(result: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    amended = json.loads(json.dumps(result, ensure_ascii=False))
    locations: list[dict[str, Any]] = []
    for paragraph in amended["paragraphs"]:
        text = paragraph["en"]
        for source, destination in REPLACEMENTS:
            count = text.count(source)
            if count:
                locations.append(
                    {
                        "paragraph_id": paragraph["id"],
                        "exact": source,
                        "replacement": destination,
                        "count": count,
                    }
                )
                text = text.replace(source, destination)
        paragraph["en"] = text
    return amended, locations


def raw_attempt_path(sample_id: str, attempt: int) -> Path:
    return RAW_ATTEMPTS / sample_id / f"attempt_{attempt:02d}.json"


def run_sample(
    sample: dict[str, Any],
    preregistration: dict[str, Any],
    declaration: dict[str, Any],
    *,
    resume: bool,
) -> tuple[str, str]:
    stage = "english_semantic_source"
    sample_id = sample["sample_id"]
    destination = generation.output_path(stage, sample_id)
    if destination.exists():
        if not resume:
            raise FileExistsError(f"output already exists: {destination}")
        artifact = read_json(destination)
        request = generation.build_request(stage, sample, preregistration)
        generation.validate_artifact(stage, sample, preregistration, request, artifact)
        return sample_id, "skipped"

    config = preregistration["generation_stages"][stage]
    prompt_path = Path(config["prompt_path"])
    schema_path = Path(config["schema_path"])
    prompt = prompt_path.read_text(encoding="utf-8")
    request = generation.build_request(stage, sample, preregistration)
    errors: list[str] = []
    for attempt in range(1, int(declaration["max_attempts_per_sample"]) + 1):
        raw_path = raw_attempt_path(sample_id, attempt)
        if raw_path.exists():
            raise FileExistsError(f"raw amendment attempt already exists: {raw_path}")
        with tempfile.TemporaryDirectory(prefix=f"cr-fysm-v4-amend-{sample_id}-") as temp:
            temp_path = Path(temp)
            isolated_schema = temp_path / "schema.json"
            isolated_schema.write_bytes(schema_path.read_bytes())
            response_path = temp_path / "response.json"
            command = generation.codex_command(
                codex_binary=str(preregistration["runtime"]["codex_executable_resolved"]),
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
            response_id, usage = response_metadata(completed.stdout)
            raw_result: dict[str, Any] | None = None
            parse_error = None
            if completed.returncode == 0:
                try:
                    raw_result = read_json(response_path)
                except (FileNotFoundError, json.JSONDecodeError) as exc:
                    parse_error = str(exc)
            validation_errors = (
                generation.validate_result(stage, request, raw_result, preregistration)
                if raw_result is not None
                else []
            )
            raw_record = {
                "schema_version": 1,
                "amendment_id": AMENDMENT_ID,
                "declaration_id": declaration["declaration_id"],
                "generation_lock_id": preregistration["lock_id"],
                "sample_id": sample_id,
                "attempt": attempt,
                "returncode": completed.returncode,
                "response_id": response_id,
                "usage": usage,
                "stdout_sha256": sha256_text(completed.stdout),
                "stderr_sha256": sha256_text(completed.stderr),
                "stderr_tail": completed.stderr[-2000:],
                "parse_error": parse_error,
                "validation_errors": validation_errors,
                "raw_result_sha256": (
                    sha256_text(canonical(raw_result)) if raw_result is not None else None
                ),
                "raw_result": raw_result,
            }
            write_json(raw_path, raw_record)
            if completed.returncode != 0:
                errors.append(f"attempt {attempt}: codex exit {completed.returncode}")
                continue
            if raw_result is None:
                errors.append(f"attempt {attempt}: invalid response: {parse_error}")
                continue

            amendment_applied = False
            replacement_locations: list[dict[str, Any]] = []
            result = raw_result
            if validation_errors == ["refusal or placeholder text detected"]:
                result, replacement_locations = amend_result(raw_result)
                if not replacement_locations:
                    errors.append(
                        f"attempt {attempt}: refusal error had no exact predeclared replacement"
                    )
                    continue
                amendment_applied = True
            elif validation_errors:
                errors.append(
                    f"attempt {attempt}: non-amendable validation errors: {validation_errors}"
                )
                continue

            final_errors = generation.validate_result(stage, request, result, preregistration)
            if final_errors:
                errors.append(f"attempt {attempt}: amended result failed: {final_errors}")
                continue
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
                "prompt_sha256": generation.sha256_file(prompt_path),
                "schema_sha256": generation.sha256_file(schema_path),
                "request_sha256": generation.sha256_text(generation.canonical(request)),
                "dependency_result_sha256": {},
                "result_sha256": generation.sha256_text(generation.canonical(result)),
                "response_id": response_id,
                "usage": usage,
                "prior_attempt_errors": errors,
                "protocol_amendment": {
                    "amendment_id": AMENDMENT_ID,
                    "declaration_id": declaration["declaration_id"],
                    "applied": amendment_applied,
                    "original_result_sha256": generation.sha256_text(
                        generation.canonical(raw_result)
                    ),
                    "replacement_locations": replacement_locations,
                    "replacement_count": sum(
                        int(row["count"]) for row in replacement_locations
                    ),
                    "raw_attempt_path": str(raw_path),
                    "raw_attempt_sha256": generation.sha256_file(raw_path),
                },
                "result": result,
            }
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp")
            write_json(temporary, artifact)
            temporary.replace(destination)
            generation.validate_artifact(
                stage, sample, preregistration, request, read_json(destination)
            )
            return sample_id, "amended" if amendment_applied else "complete_unmodified"
    raise RuntimeError(f"{sample_id} failed amendment: {'; '.join(errors[-6:])}")


def main() -> None:
    args = parse_args()
    preregistration = generation.validate_preregistration()
    declaration = validate_declaration(preregistration)
    samples_by_id = {
        row["sample_id"]: row for row in generation.read_jsonl(generation.SELECTION)
    }
    if not set(SAMPLE_IDS).issubset(samples_by_id):
        raise ValueError("one or more amendment samples are absent from the selection")
    samples = [samples_by_id[sample_id] for sample_id in SAMPLE_IDS]
    failures: list[tuple[str, str]] = []
    counts = {"amended": 0, "complete_unmodified": 0, "skipped": 0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {
            pool.submit(
                run_sample,
                sample,
                preregistration,
                declaration,
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
                    print(f"[{status}] {sample_id}", flush=True)
            except Exception as exc:  # noqa: BLE001 - preserve independent failures
                failures.append((sample_id, str(exc)))
                with PRINT_LOCK:
                    print(f"[failed] {sample_id}: {exc}", flush=True)
    summary = {
        "amendment_id": AMENDMENT_ID,
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
