#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

from experiments.iteration5.construct import run_cr_fysm_v4_construct_generation as generation
from experiments.iteration5.construct import run_cr_fysm_v4_english_source_amendment_v1 as source_amendment


AMENDMENT_ID = "CR-FYSM-v4-construct-v2-english-adjudication-refusal-false-positive-amendment-v1"
SAMPLE_IDS = source_amendment.SAMPLE_IDS
REPLACEMENTS = source_amendment.REPLACEMENTS
AMENDMENT_ROOT = generation.ROOT / "protocol_amendment_english_adjudication_v1"
DECLARATION = AMENDMENT_ROOT / "preregistration.json"
RAW_ATTEMPTS = AMENDMENT_ROOT / "raw_attempts"
PRINT_LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the predeclared English-adjudication refusal amendment."
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip already completed artifacts only. A raw attempt without a completed "
            "artifact is never overwritten and requires a new amendment version."
        ),
    )
    return parser.parse_args()


def validate_declaration(preregistration: dict[str, Any]) -> dict[str, Any]:
    declaration = source_amendment.read_json(DECLARATION)
    if declaration.get("status") != "declared_before_amendment_rerun":
        raise ValueError("adjudication amendment is not predeclared")
    if declaration.get("amendment_id") != AMENDMENT_ID:
        raise ValueError("adjudication amendment ID mismatch")
    if declaration.get("declaration_id") != source_amendment.declaration_id(declaration):
        raise ValueError("adjudication amendment declaration_id is invalid")
    if tuple(declaration.get("sample_ids", ())) != SAMPLE_IDS:
        raise ValueError("adjudication amendment sample IDs changed")
    if declaration.get("generation_lock_id") != preregistration["lock_id"]:
        raise ValueError("adjudication amendment does not bind the generation lock")
    if declaration.get("runner_sha256") != generation.sha256_file(Path(__file__)):
        raise ValueError("adjudication amendment runner changed after declaration")
    if declaration.get("source_amendment_declaration_sha256") != generation.sha256_file(
        source_amendment.DECLARATION
    ):
        raise ValueError("source-stage protocol amendment declaration changed")
    expected_replacements = [
        {"exact": source, "replacement": destination}
        for source, destination in REPLACEMENTS
    ]
    if declaration.get("deterministic_replacements") != expected_replacements:
        raise ValueError("adjudication amendment replacement rule changed")
    source_hashes = {
        sample_id: generation.load_artifact(
            "english_semantic_source",
            next(
                row
                for row in generation.read_jsonl(generation.SELECTION)
                if row["sample_id"] == sample_id
            ),
            preregistration,
        )["result_sha256"]
        for sample_id in SAMPLE_IDS
    }
    if declaration.get("source_dependency_result_sha256") != source_hashes:
        raise ValueError("adjudication amendment source dependency changed")
    return declaration


def raw_attempt_path(sample_id: str, attempt: int) -> Path:
    return RAW_ATTEMPTS / sample_id / f"attempt_{attempt:02d}.json"


def run_sample(
    sample: dict[str, Any],
    preregistration: dict[str, Any],
    declaration: dict[str, Any],
    *,
    resume: bool,
) -> tuple[str, str]:
    stage = "english_adjudication"
    sample_id = sample["sample_id"]
    destination = generation.output_path(stage, sample_id)
    if destination.exists():
        if not resume:
            raise FileExistsError(f"output already exists: {destination}")
        request = generation.build_request(stage, sample, preregistration)
        generation.validate_artifact(
            stage, sample, preregistration, request, source_amendment.read_json(destination)
        )
        return sample_id, "skipped"

    config = preregistration["generation_stages"][stage]
    prompt_path = Path(config["prompt_path"])
    schema_path = Path(config["schema_path"])
    prompt = prompt_path.read_text(encoding="utf-8")
    request = generation.build_request(stage, sample, preregistration)
    dependency_hashes = {
        dependency: generation.load_artifact(dependency, sample, preregistration)[
            "result_sha256"
        ]
        for dependency in generation.DEPENDENCIES[stage]
    }
    errors: list[str] = []
    for attempt in range(1, int(declaration["max_attempts_per_sample"]) + 1):
        raw_path = raw_attempt_path(sample_id, attempt)
        if raw_path.exists():
            raise FileExistsError(f"raw amendment attempt already exists: {raw_path}")
        with tempfile.TemporaryDirectory(prefix=f"cr-fysm-v4-adjudicate-amend-{sample_id}-") as temp:
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
            response_id, usage = source_amendment.response_metadata(completed.stdout)
            raw_result: dict[str, Any] | None = None
            parse_error = None
            if completed.returncode == 0:
                try:
                    raw_result = source_amendment.read_json(response_path)
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
                "stdout_sha256": source_amendment.sha256_text(completed.stdout),
                "stderr_sha256": source_amendment.sha256_text(completed.stderr),
                "stderr_tail": completed.stderr[-2000:],
                "parse_error": parse_error,
                "validation_errors": validation_errors,
                "raw_result_sha256": (
                    source_amendment.sha256_text(source_amendment.canonical(raw_result))
                    if raw_result is not None
                    else None
                ),
                "raw_result": raw_result,
            }
            source_amendment.write_json(raw_path, raw_record)
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
                result, replacement_locations = source_amendment.amend_result(raw_result)
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
                "dependency_result_sha256": dependency_hashes,
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
            source_amendment.write_json(temporary, artifact)
            temporary.replace(destination)
            generation.validate_artifact(
                stage,
                sample,
                preregistration,
                request,
                source_amendment.read_json(destination),
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
