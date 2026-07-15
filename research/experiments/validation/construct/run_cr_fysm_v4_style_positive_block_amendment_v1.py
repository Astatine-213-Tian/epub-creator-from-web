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

from experiments.validation.construct import run_cr_fysm_v4_construct_generation as generation
from experiments.validation.construct import run_cr_fysm_v4_english_source_amendment_v1 as source_amendment
from experiments.validation.construct import prepare_cr_fysm_v4_style_positive_block_amendment_v1 as preparation


AMENDMENT_ID = preparation.AMENDMENT_ID
SAMPLE_IDS = preparation.SAMPLE_IDS
ROOT = preparation.ROOT
DECLARATION = preparation.DECLARATION
RAW_ATTEMPTS = ROOT / "raw_attempts"
PRINT_LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the predeclared blockwise style-positive completion amendment."
    )
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip canonical completed artifacts only. Existing raw attempts without a "
            "canonical artifact require a new amendment version."
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_declaration(preregistration: dict[str, Any]) -> dict[str, Any]:
    declaration = read_json(DECLARATION)
    if declaration.get("status") != (
        "revised_before_blockwise_amendment_generation_after_independent_no_go"
    ):
        raise ValueError("style-positive block amendment is not predeclared")
    if declaration.get("amendment_id") != AMENDMENT_ID:
        raise ValueError("style-positive amendment ID mismatch")
    if declaration.get("declaration_id") != preparation.declaration_id(declaration):
        raise ValueError("style-positive amendment declaration ID is invalid")
    if declaration.get("generation_lock_id") != preregistration["lock_id"]:
        raise ValueError("style-positive amendment does not bind the generation lock")
    if tuple(declaration.get("sample_ids", ())) != SAMPLE_IDS:
        raise ValueError("style-positive amendment sample IDs changed")
    if declaration.get("runner_sha256") != generation.sha256_file(Path(__file__)):
        raise ValueError("style-positive amendment runner changed after declaration")
    if declaration["method"]["block_size"] != preparation.BLOCK_SIZE:
        raise ValueError("style-positive amendment block size changed")
    samples = {
        row["sample_id"]: row for row in generation.read_jsonl(generation.SELECTION)
    }
    observed_dependencies = {
        sample_id: {
            dependency: generation.load_artifact(
                dependency, samples[sample_id], preregistration
            )["result_sha256"]
            for dependency in generation.DEPENDENCIES["style_positive_control"]
        }
        for sample_id in SAMPLE_IDS
    }
    if declaration.get("dependency_result_sha256") != observed_dependencies:
        raise ValueError("style-positive amendment dependencies changed after declaration")
    return declaration


def block_request(request: dict[str, Any], ids: list[str]) -> dict[str, Any]:
    selected = set(ids)
    result = dict(request)
    result["english_paragraphs"] = [
        row for row in request["english_paragraphs"] if row["id"] in selected
    ]
    result["neutral_chinese"] = [
        row for row in request["neutral_chinese"] if row["id"] in selected
    ]
    if [row["id"] for row in result["english_paragraphs"]] != ids:
        raise ValueError("English block partition no longer matches the declaration")
    if [row["id"] for row in result["neutral_chinese"]] != ids:
        raise ValueError("neutral-Chinese block partition no longer matches the declaration")
    return result


def raw_path(sample_id: str, block_number: int, attempt: int) -> Path:
    return (
        RAW_ATTEMPTS
        / sample_id
        / f"block_{block_number:02d}"
        / f"attempt_{attempt:02d}.json"
    )


def call_block(
    *,
    sample_id: str,
    block_number: int,
    request: dict[str, Any],
    preregistration: dict[str, Any],
    declaration: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    stage = "style_positive_control"
    config = preregistration["generation_stages"][stage]
    prompt_path = Path(config["prompt_path"])
    schema_path = Path(config["schema_path"])
    prompt = prompt_path.read_text(encoding="utf-8")
    errors: list[str] = []
    maximum = int(declaration["method"]["max_attempts_per_block"])
    for attempt in range(1, maximum + 1):
        destination = raw_path(sample_id, block_number, attempt)
        if destination.exists():
            raise FileExistsError(f"raw block attempt already exists: {destination}")
        with tempfile.TemporaryDirectory(
            prefix=f"cr-fysm-v4-positive-block-{sample_id}-{block_number:02d}-"
        ) as temp:
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
            timed_out = False
            try:
                completed = subprocess.run(
                    command,
                    input=prompt + "\n\n## Request JSON\n\n" + json.dumps(request, ensure_ascii=False),
                    text=True,
                    capture_output=True,
                    timeout=int(config["timeout_seconds"]),
                    env={
                        key: value
                        for key, value in os.environ.items()
                        if key
                        in {"HOME", "LANG", "LC_ALL", "LOGNAME", "PATH", "TMPDIR", "USER"}
                    },
                )
                returncode = completed.returncode
                stdout = completed.stdout
                stderr = completed.stderr
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                returncode = None
                stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            response_id, usage = source_amendment.response_metadata(stdout)
            result = None
            parse_error = None
            if returncode == 0:
                try:
                    result = read_json(response_path)
                except (FileNotFoundError, json.JSONDecodeError) as exc:
                    parse_error = str(exc)
            validation_errors = (
                generation.validate_result(stage, request, result, preregistration)
                if result is not None
                else []
            )
            raw_record = {
                "schema_version": 1,
                "amendment_id": AMENDMENT_ID,
                "declaration_id": declaration["declaration_id"],
                "generation_lock_id": preregistration["lock_id"],
                "sample_id": sample_id,
                "block_number": block_number,
                "paragraph_ids": generation.expected_ids(stage, request),
                "attempt": attempt,
                "timed_out": timed_out,
                "returncode": returncode,
                "response_id": response_id,
                "usage": usage,
                "stdout_sha256": source_amendment.sha256_text(stdout),
                "stderr_sha256": source_amendment.sha256_text(stderr),
                "stderr_tail": stderr[-2000:],
                "parse_error": parse_error,
                "validation_errors": validation_errors,
                "raw_result_sha256": (
                    source_amendment.sha256_text(source_amendment.canonical(result))
                    if result is not None
                    else None
                ),
                "raw_result": result,
            }
            write_json(destination, raw_record)
            if timed_out:
                errors.append(f"attempt {attempt}: timeout")
                continue
            if returncode != 0:
                errors.append(f"attempt {attempt}: Codex exit {returncode}")
                continue
            if result is None:
                errors.append(f"attempt {attempt}: invalid response: {parse_error}")
                continue
            if validation_errors:
                errors.append(f"attempt {attempt}: {validation_errors}")
                continue
            return result, {
                "block_number": block_number,
                "attempt": attempt,
                "response_id": response_id,
                "usage": usage,
                "raw_attempt_path": str(destination),
                "raw_attempt_sha256": generation.sha256_file(destination),
                "prior_attempt_errors": errors,
            }
    raise RuntimeError(
        f"{sample_id} block {block_number} failed amendment: {'; '.join(errors)}"
    )


def run_sample(
    sample: dict[str, Any],
    preregistration: dict[str, Any],
    declaration: dict[str, Any],
    *,
    resume: bool,
) -> tuple[str, str]:
    stage = "style_positive_control"
    sample_id = sample["sample_id"]
    destination = generation.output_path(stage, sample_id)
    request = generation.build_request(stage, sample, preregistration)
    if destination.exists():
        if not resume:
            raise FileExistsError(f"output already exists: {destination}")
        generation.validate_artifact(stage, sample, preregistration, request, read_json(destination))
        return sample_id, "skipped"

    partitions = declaration["method"]["partitions"][sample_id]
    results: list[dict[str, Any]] = []
    lineages: list[dict[str, Any]] = []
    for block_number, ids in enumerate(partitions, start=1):
        result, lineage = call_block(
            sample_id=sample_id,
            block_number=block_number,
            request=block_request(request, ids),
            preregistration=preregistration,
            declaration=declaration,
        )
        results.append(result)
        lineages.append(lineage)
    stitched = {
        "sample_id": sample_id,
        "paragraphs": [paragraph for result in results for paragraph in result["paragraphs"]],
    }
    final_errors = generation.validate_result(stage, request, stitched, preregistration)
    if final_errors:
        raise ValueError(f"stitched style-positive output failed: {final_errors}")

    config = preregistration["generation_stages"][stage]
    dependency_hashes = {
        dependency: generation.load_artifact(dependency, sample, preregistration)[
            "result_sha256"
        ]
        for dependency in generation.DEPENDENCIES[stage]
    }
    artifact = {
        "schema_version": 2,
        "status": "complete",
        "construct_id": preregistration["construct_id"],
        "generation_lock_id": preregistration["lock_id"],
        "stage": stage,
        "sample_id": sample_id,
        "attempt": "blockwise_amendment_v1",
        "model": config["model"],
        "reasoning_effort": config["reasoning_effort"],
        "prompt_sha256": generation.sha256_file(Path(config["prompt_path"])),
        "schema_sha256": generation.sha256_file(Path(config["schema_path"])),
        "request_sha256": generation.sha256_text(generation.canonical(request)),
        "dependency_result_sha256": dependency_hashes,
        "result_sha256": generation.sha256_text(generation.canonical(stitched)),
        "response_id": [row["response_id"] for row in lineages],
        "usage": [row["usage"] for row in lineages],
        "prior_attempt_errors": [
            "whole-passage generation repeatedly failed paragraph-ID/order validation"
        ],
        "protocol_amendment": {
            "amendment_id": AMENDMENT_ID,
            "declaration_id": declaration["declaration_id"],
            "method": "contiguous_block_generation_and_deterministic_stitching",
            "blocks": lineages,
        },
        "result": stitched,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    write_json(temporary, artifact)
    temporary.replace(destination)
    generation.validate_artifact(stage, sample, preregistration, request, read_json(destination))
    return sample_id, "amended"


def main() -> None:
    args = parse_args()
    preregistration = generation.validate_preregistration()
    declaration = validate_declaration(preregistration)
    samples = {
        row["sample_id"]: row for row in generation.read_jsonl(generation.SELECTION)
    }
    failures: list[tuple[str, str]] = []
    counts = {"amended": 0, "skipped": 0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {
            pool.submit(
                run_sample,
                samples[sample_id],
                preregistration,
                declaration,
                resume=args.resume,
            ): sample_id
            for sample_id in SAMPLE_IDS
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
        "expected": len(SAMPLE_IDS),
        **counts,
        "failed": len(failures),
        "failures": failures,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
