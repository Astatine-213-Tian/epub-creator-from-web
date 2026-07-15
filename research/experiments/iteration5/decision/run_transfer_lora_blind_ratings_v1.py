#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

from experiments.iteration5.decision import prepare_transfer_lora_blind_benchmark_v1 as prepare_v1
from experiments.iteration5.decision import prepare_transfer_lora_reference_benchmark_v2 as prepare_v2
from experiments.iteration5.construct import run_cr_fysm_v4_construct_generation as generation
from experiments.shared.hashing import canonical_json as canonical
from experiments.shared.hashing import sha256_text


MAX_ATTEMPTS = 2
TIMEOUT_SECONDS = 900
PRINT_LOCK = threading.Lock()
prepare = prepare_v1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen blind prompt/LoRA ratings.")
    parser.add_argument("task", choices=("style", "semantic"))
    parser.add_argument("--benchmark", choices=("v1", "v2"), default="v1")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_lock() -> dict[str, Any]:
    lock = read_json(prepare.PREREGISTRATION)
    if lock.get("status") not in {
        "locked_before_any_blind_rating",
        "locked_before_any_v2_rating_post_control_failure_amendment",
    }:
        raise ValueError("blind benchmark is not locked")
    if lock.get("lock_id") != prepare.lock_id(lock):
        raise ValueError("blind benchmark lock_id is invalid")
    actual_inputs = {path: prepare.sha256_file(Path(path)) for path in lock["input_hashes"]}
    if actual_inputs != lock["input_hashes"]:
        changed = sorted(path for path in actual_inputs if actual_inputs[path] != lock["input_hashes"][path])
        raise ValueError(f"blind benchmark inputs changed: {changed}")
    actual_private = {
        path: prepare.sha256_file(Path(path)) for path in lock["private_hashes"]
    }
    if actual_private != lock["private_hashes"]:
        raise ValueError("blind benchmark private mappings changed")
    actual_packets = {path: prepare.sha256_file(Path(path)) for path in lock["packet_hashes"]}
    if actual_packets != lock["packet_hashes"]:
        raise ValueError("blind benchmark packets changed")
    return lock


def expected_ids(packet: dict[str, Any]) -> list[str]:
    return [str(row["candidate_id"]) for row in packet["candidates"]]


def validate_result(result: dict[str, Any], packet: dict[str, Any], task: str) -> list[str]:
    errors: list[str] = []
    if result.get("packet_id") != packet["packet_id"]:
        errors.append("packet_id mismatch")
    judgments = result.get("judgments")
    if not isinstance(judgments, list):
        return errors + ["judgments is not a list"]
    actual_ids = [str(row.get("candidate_id", "")) for row in judgments if isinstance(row, dict)]
    if actual_ids != expected_ids(packet):
        errors.append("candidate IDs or order mismatch")
    for row in judgments:
        if not isinstance(row, dict):
            errors.append("judgment is not an object")
            continue
        required = (
            ("style_adherence", "naturalness", "rationale")
            if task == "style"
            else (
                "semantic_fidelity",
                "naturalness",
                "high_severity_semantic_error",
                "speaker_dialogue_topology_preserved",
                "rationale",
            )
        )
        missing = [key for key in required if key not in row]
        if missing:
            errors.append(f"{row.get('candidate_id')}: missing {missing}")
            continue
        score_keys = ("style_adherence", "naturalness") if task == "style" else ("semantic_fidelity", "naturalness")
        for key in score_keys:
            value = row[key]
            if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 5:
                errors.append(f"{row.get('candidate_id')}: invalid {key}")
        if task == "style" and packet.get("schema_version") == 2:
            dimensions = row.get("dimension_scores")
            expected_dimensions = tuple(packet.get("rubric_dimensions", ()))
            if not isinstance(dimensions, dict):
                errors.append(f"{row.get('candidate_id')}: dimension_scores is not an object")
            elif set(dimensions) != set(expected_dimensions):
                errors.append(f"{row.get('candidate_id')}: dimension score keys mismatch")
            else:
                for key in expected_dimensions:
                    value = dimensions[key]
                    if value is not None and (
                        not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 5
                    ):
                        errors.append(f"{row.get('candidate_id')}: invalid dimension {key}")
        if not isinstance(row["rationale"], str) or not row["rationale"].strip():
            errors.append(f"{row.get('candidate_id')}: empty rationale")
        if task == "semantic":
            for key in ("high_severity_semantic_error", "speaker_dialogue_topology_preserved"):
                if not isinstance(row[key], bool):
                    errors.append(f"{row.get('candidate_id')}: invalid {key}")
    return errors


def destination(identity: str, packet_id: str) -> Path:
    return prepare.ROOT / "ratings" / identity / f"{packet_id}.json"


def validate_artifact(path: Path, lock: dict[str, Any], packet_path: Path, task: str, identity: str) -> None:
    artifact = read_json(path)
    packet = read_json(packet_path)
    expected = {
        "schema_version": 1,
        "status": "complete",
        "lock_id": lock["lock_id"],
        "task": task,
        "identity": identity,
        "packet_id": packet["packet_id"],
        "packet_sha256": prepare.sha256_file(packet_path),
        "model": lock["models"]["blind_raters"],
        "reasoning_effort": lock["models"]["reasoning_effort"],
    }
    mismatches = [key for key, value in expected.items() if artifact.get(key) != value]
    if mismatches:
        raise ValueError(f"invalid artifact metadata {path}: {mismatches}")
    result = artifact.get("result")
    if not isinstance(result, dict) or artifact.get("result_sha256") != sha256_text(canonical(result)):
        raise ValueError(f"invalid result hash: {path}")
    errors = validate_result(result, packet, task)
    if errors:
        raise ValueError(f"invalid result: {path}: {errors}")


def run_packet(lock: dict[str, Any], task: str, identity: str, packet_path: Path, resume: bool) -> tuple[str, str]:
    packet = read_json(packet_path)
    packet_id = str(packet["packet_id"])
    output_path = destination(identity, packet_id)
    if output_path.exists():
        if not resume:
            raise FileExistsError(f"output exists: {output_path}")
        validate_artifact(output_path, lock, packet_path, task, identity)
        return packet_id, "skipped"
    prompt_path = Path(lock.get("prompts", {}).get(task, prepare.ASSETS / f"{task}_multicandidate.v1.md"))
    schema_path = Path(lock.get("schemas", {}).get(task, prepare.ASSETS / f"{task}_multicandidate.schema.json"))
    prompt = prompt_path.read_text(encoding="utf-8")
    runtime = generation.current_codex_runtime()
    failures: list[str] = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        with tempfile.TemporaryDirectory(prefix=f"blind-{task}-{identity}-") as temporary:
            temp = Path(temporary)
            isolated_schema = temp / "schema.json"
            shutil.copy2(schema_path, isolated_schema)
            response = temp / "response.json"
            command = generation.codex_command(
                codex_binary=str(runtime["codex_executable_resolved"]),
                model=lock["models"]["blind_raters"],
                reasoning_effort=lock["models"]["reasoning_effort"],
                schema=isolated_schema,
                response=response,
                cwd=temp,
            )
            try:
                completed = subprocess.run(
                    command,
                    input=prompt + "\n\n## Rating Packet JSON\n\n" + json.dumps(packet, ensure_ascii=False),
                    text=True,
                    capture_output=True,
                    timeout=TIMEOUT_SECONDS,
                )
                if completed.returncode != 0:
                    failures.append(f"attempt {attempt}: exit {completed.returncode}: {completed.stderr[-500:]}")
                    continue
                if not response.exists():
                    failures.append(f"attempt {attempt}: response file absent")
                    continue
                result = read_json(response)
                errors = validate_result(result, packet, task)
                if errors:
                    failures.append(f"attempt {attempt}: {errors}")
                    continue
                artifact = {
                    "schema_version": 1,
                    "status": "complete",
                    "lock_id": lock["lock_id"],
                    "task": task,
                    "identity": identity,
                    "packet_id": packet_id,
                    "packet_sha256": prepare.sha256_file(packet_path),
                    "prompt_sha256": prepare.sha256_file(prompt_path),
                    "schema_sha256": prepare.sha256_file(schema_path),
                    "model": lock["models"]["blind_raters"],
                    "reasoning_effort": lock["models"]["reasoning_effort"],
                    "attempt": attempt,
                    "result": result,
                    "result_sha256": sha256_text(canonical(result)),
                }
                write_json(output_path, artifact)
                return packet_id, "written"
            except subprocess.TimeoutExpired:
                failures.append(f"attempt {attempt}: timeout")
    raise RuntimeError(f"failed {packet_id}: {' | '.join(failures)}")


def main() -> None:
    global prepare
    args = parse_args()
    prepare = prepare_v2 if args.benchmark == "v2" else prepare_v1
    lock = validate_lock()
    identities = prepare.STYLE_IDENTITIES if args.task == "style" else prepare.SEMANTIC_IDENTITIES
    jobs: list[tuple[str, Path]] = []
    for identity in identities:
        jobs.extend((identity, path) for path in sorted((prepare.ROOT / "packets" / identity).glob("*.json")))
    written = skipped = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(run_packet, lock, args.task, identity, path, args.resume): (identity, path)
            for identity, path in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            packet_id, status = future.result()
            written += status == "written"
            skipped += status == "skipped"
            with PRINT_LOCK:
                print(f"{args.task} {packet_id}: {status}", flush=True)
    print(f"complete task={args.task} written={written} skipped={skipped} total={len(jobs)}")


if __name__ == "__main__":
    main()
