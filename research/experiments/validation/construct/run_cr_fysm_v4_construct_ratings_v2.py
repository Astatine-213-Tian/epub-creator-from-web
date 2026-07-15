#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterable

from experiments.validation.construct import prepare_cr_fysm_v4_construct_analysis_v2 as prepare
from experiments.validation.construct import run_cr_fysm_v4_construct_generation as generation


MAX_ATTEMPTS = 2
TIMEOUT_SECONDS = 900
PRINT_LOCK = threading.Lock()
RATINGS = prepare.ROOT / "ratings"
ADJUDICATION_MANIFEST = prepare.ROOT / "semantic_adjudication_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frozen construct-v2 semantic or blind-style ratings."
    )
    parser.add_argument("task", choices=("semantic", "style", "adjudicate"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--packet-id", action="append", default=[])
    return parser.parse_args()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_schema_instance(
    instance: Any,
    schema: dict[str, Any],
    *,
    location: str = "$",
) -> list[str]:
    errors: list[str] = []
    expected_type = schema.get("type")
    checks = {
        "object": lambda value: isinstance(value, dict),
        "array": lambda value: isinstance(value, list),
        "string": lambda value: isinstance(value, str),
        "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
        "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": lambda value: isinstance(value, bool),
    }
    if expected_type in checks and not checks[expected_type](instance):
        return [f"{location}: expected {expected_type}, got {type(instance).__name__}"]
    if "const" in schema and instance != schema["const"]:
        errors.append(f"{location}: does not match const")
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{location}: not in enum")
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{location}: below minimum")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{location}: above maximum")
    if isinstance(instance, str):
        if len(instance) < int(schema.get("minLength", 0)):
            errors.append(f"{location}: shorter than minLength")
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
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{location}: missing {key}")
        if schema.get("additionalProperties") is False:
            for key in instance:
                if key not in properties:
                    errors.append(f"{location}: unexpected {key}")
        for key, value in instance.items():
            child = properties.get(key)
            if isinstance(child, dict):
                errors.extend(
                    validate_schema_instance(value, child, location=f"{location}.{key}")
                )
    return errors


def validate_lock() -> dict[str, Any]:
    lock = read_json(prepare.PREREGISTRATION)
    if lock.get("status") != "locked_before_any_construct_rating_or_scoring":
        raise ValueError("construct analysis is not locked")
    if lock.get("lock_id") != prepare.lock_id(lock):
        raise ValueError("construct analysis lock_id is invalid")
    if lock.get("input_hashes") != prepare.current_input_hashes():
        raise ValueError("construct analysis inputs changed after lock")
    if lock.get("packet_hashes") != prepare.current_packet_hashes():
        raise ValueError("construct analysis packets changed after lock")
    expected_private = {
        str(prepare.PRIVATE_ITEMS): prepare.sha256_file(prepare.PRIVATE_ITEMS),
        str(prepare.PRIVATE_PAIRS): prepare.sha256_file(prepare.PRIVATE_PAIRS),
    }
    if lock.get("private_hashes") != expected_private:
        raise ValueError("construct private mappings changed after lock")
    if lock.get("runtime") != generation.current_codex_runtime():
        raise ValueError("Codex runtime changed after analysis lock")
    return lock


def semantic_pass(judgment: dict[str, Any]) -> bool:
    return (
        int(judgment["semantic_fidelity"]) >= 4
        and int(judgment["naturalness"]) >= 3
        and judgment["high_severity_semantic_error"] is False
        and judgment["speaker_dialogue_topology_preserved"] is True
    )


def packet_ids(packet: dict[str, Any], task: str) -> list[str]:
    if task == "style":
        return [str(row["pair_id"]) for row in packet["pairs"]]
    return [str(row["item_id"]) for row in packet["items"]]


def result_ids(result: dict[str, Any], task: str) -> list[str]:
    key = "pair_id" if task == "style" else "item_id"
    return [str(row.get(key, "")) for row in result.get("judgments", [])]


def validate_result(
    result: dict[str, Any],
    packet: dict[str, Any],
    task: str,
    schema: dict[str, Any],
) -> list[str]:
    errors = validate_schema_instance(result, schema)
    if result.get("packet_id") != packet["packet_id"]:
        errors.append("packet_id mismatch")
    if result_ids(result, task) != packet_ids(packet, task):
        errors.append("judgment IDs or order differ from packet")
    for judgment in result.get("judgments", []):
        rationale = judgment.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            errors.append("empty rationale")
    return errors


def rating_path(identity: str, packet_id: str) -> Path:
    return RATINGS / identity / f"{packet_id}.json"


def validate_artifact(
    artifact: dict[str, Any],
    *,
    lock: dict[str, Any],
    identity: str,
    task: str,
    packet_path: Path,
    prompt_path: Path,
    schema_path: Path,
) -> dict[str, Any]:
    packet = read_json(packet_path)
    expected = {
        "schema_version": 2,
        "status": "complete",
        "analysis_lock_id": lock["lock_id"],
        "task": task,
        "identity": identity,
        "packet_id": packet["packet_id"],
        "model": "gpt-5.5",
        "reasoning_effort": "high",
        "packet_sha256": prepare.sha256_file(packet_path),
        "prompt_sha256": prepare.sha256_file(prompt_path),
        "schema_sha256": prepare.sha256_file(schema_path),
    }
    mismatches = [key for key, value in expected.items() if artifact.get(key) != value]
    if mismatches:
        raise ValueError(f"invalid rating artifact metadata: {mismatches}")
    result = artifact.get("result")
    if not isinstance(result, dict):
        raise ValueError("rating artifact result is not an object")
    if artifact.get("result_sha256") != sha256_text(canonical(result)):
        raise ValueError("rating artifact result hash mismatch")
    errors = validate_result(result, packet, task, read_json(schema_path))
    if errors:
        raise ValueError(f"rating artifact no longer validates: {errors}")
    return result


def run_packet(
    *,
    lock: dict[str, Any],
    identity: str,
    task: str,
    packet_path: Path,
    resume: bool,
) -> tuple[str, str]:
    prompt_path = prepare.STYLE_PROMPT if task == "style" else prepare.SEMANTIC_PROMPT
    schema_path = prepare.STYLE_SCHEMA if task == "style" else prepare.SEMANTIC_SCHEMA
    packet = read_json(packet_path)
    packet_id = str(packet["packet_id"])
    destination = rating_path(identity, packet_id)
    if destination.exists():
        if not resume:
            raise FileExistsError(f"output already exists: {destination}")
        validate_artifact(
            read_json(destination),
            lock=lock,
            identity=identity,
            task=task,
            packet_path=packet_path,
            prompt_path=prompt_path,
            schema_path=schema_path,
        )
        return packet_id, "skipped"

    prompt = prompt_path.read_text(encoding="utf-8")
    schema = read_json(schema_path)
    errors: list[str] = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        with tempfile.TemporaryDirectory(prefix=f"cr-fysm-v4-rating-{identity}-") as temp:
            temp_path = Path(temp)
            isolated_schema = temp_path / "schema.json"
            isolated_schema.write_bytes(schema_path.read_bytes())
            response_path = temp_path / "response.json"
            command = generation.codex_command(
                codex_binary=str(lock["runtime"]["codex_executable_resolved"]),
                model="gpt-5.5",
                reasoning_effort="high",
                schema=isolated_schema,
                response=response_path,
                cwd=temp_path,
            )
            completed = subprocess.run(
                command,
                input=prompt + "\n\n## Rating Packet JSON\n\n" + json.dumps(packet, ensure_ascii=False),
                text=True,
                capture_output=True,
                timeout=TIMEOUT_SECONDS,
                env={
                    key: value
                    for key, value in os.environ.items()
                    if key in {"HOME", "LANG", "LC_ALL", "LOGNAME", "PATH", "TMPDIR", "USER"}
                },
            )
            if completed.returncode != 0:
                errors.append(
                    f"attempt {attempt}: codex exit {completed.returncode}: {completed.stderr[-500:]}"
                )
                continue
            try:
                result = read_json(response_path)
            except (FileNotFoundError, json.JSONDecodeError) as exc:
                errors.append(f"attempt {attempt}: invalid response: {exc}")
                continue
            validation_errors = validate_result(result, packet, task, schema)
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
                "analysis_lock_id": lock["lock_id"],
                "task": task,
                "identity": identity,
                "packet_id": packet_id,
                "attempt": attempt,
                "model": "gpt-5.5",
                "reasoning_effort": "high",
                "packet_sha256": prepare.sha256_file(packet_path),
                "prompt_sha256": prepare.sha256_file(prompt_path),
                "schema_sha256": prepare.sha256_file(schema_path),
                "result_sha256": sha256_text(canonical(result)),
                "response_id": response_id,
                "usage": usage,
                "prior_attempt_errors": errors,
                "result": result,
            }
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp")
            write_json(temporary, artifact)
            temporary.replace(destination)
            return packet_id, "complete"
    raise RuntimeError(f"{identity}/{packet_id} failed: {'; '.join(errors[-6:])}")


def identity_packets(task: str) -> list[tuple[str, str, Path]]:
    identities = prepare.RATER_IDS if task == "style" else prepare.VALIDATOR_IDS
    return [
        (identity, task, path)
        for identity in identities
        for path in sorted((prepare.PACKETS / identity).glob("*.json"))
    ]


def load_identity_judgments(
    lock: dict[str, Any], identity: str
) -> dict[str, dict[str, Any]]:
    judgments: dict[str, dict[str, Any]] = {}
    for packet_path in sorted((prepare.PACKETS / identity).glob("*.json")):
        packet = read_json(packet_path)
        artifact_path = rating_path(identity, packet["packet_id"])
        if not artifact_path.exists():
            raise FileNotFoundError(f"missing semantic rating: {artifact_path}")
        result = validate_artifact(
            read_json(artifact_path),
            lock=lock,
            identity=identity,
            task="semantic",
            packet_path=packet_path,
            prompt_path=prepare.SEMANTIC_PROMPT,
            schema_path=prepare.SEMANTIC_SCHEMA,
        )
        for judgment in result["judgments"]:
            item_id = str(judgment["item_id"])
            if item_id in judgments:
                raise ValueError(f"duplicate judgment for {identity}/{item_id}")
            judgments[item_id] = judgment
    return judgments


def chunks(values: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def prepare_adjudication(lock: dict[str, Any]) -> list[tuple[str, str, Path]]:
    validator_a = load_identity_judgments(lock, "validator_a")
    validator_b = load_identity_judgments(lock, "validator_b")
    if set(validator_a) != set(validator_b):
        raise ValueError("semantic validator coverage differs")
    private_items = {
        row["item_id"]: row for row in prepare.read_jsonl(prepare.PRIVATE_ITEMS)
    }
    disagreements = sorted(
        item_id
        for item_id in validator_a
        if semantic_pass(validator_a[item_id]) != semantic_pass(validator_b[item_id])
    )
    packet_dir = prepare.PACKETS / "validator_c"
    packet_dir.mkdir(parents=True, exist_ok=True)
    expected_paths: list[Path] = []
    for index, item_ids in enumerate(chunks(disagreements, 25), start=1):
        packet_id = f"validator_c.p{index:02d}"
        path = packet_dir / f"{packet_id}.json"
        payload = {
            "packet_id": packet_id,
            "items": [
                {
                    "item_id": item_id,
                    "english_source": private_items[item_id]["english"],
                    "chinese_candidate": private_items[item_id]["candidate_raw_zh"],
                }
                for item_id in item_ids
            ],
        }
        if path.exists() and read_json(path) != payload:
            raise ValueError(f"stale adjudication packet: {path}")
        if not path.exists():
            write_json(path, payload)
        expected_paths.append(path)
    extras = sorted(set(packet_dir.glob("*.json")) - set(expected_paths))
    if extras:
        raise ValueError(f"unexpected adjudication packets: {extras}")
    manifest = {
        "schema_version": 1,
        "status": "fixed_from_two_locked_semantic_validators",
        "analysis_lock_id": lock["lock_id"],
        "validator_a_rating_hashes": {
            str(path): prepare.sha256_file(path)
            for path in sorted((RATINGS / "validator_a").glob("*.json"))
        },
        "validator_b_rating_hashes": {
            str(path): prepare.sha256_file(path)
            for path in sorted((RATINGS / "validator_b").glob("*.json"))
        },
        "disagreement_item_ids": disagreements,
        "packet_hashes": {str(path): prepare.sha256_file(path) for path in expected_paths},
    }
    manifest["manifest_id"] = sha256_text(canonical(manifest))
    if ADJUDICATION_MANIFEST.exists() and read_json(ADJUDICATION_MANIFEST) != manifest:
        raise ValueError("semantic adjudication manifest changed")
    if not ADJUDICATION_MANIFEST.exists():
        write_json(ADJUDICATION_MANIFEST, manifest)
    return [("validator_c", "semantic", path) for path in expected_paths]


def selected_packets(
    rows: list[tuple[str, str, Path]], requested: list[str]
) -> list[tuple[str, str, Path]]:
    if not requested:
        return rows
    wanted = set(requested)
    selected = [row for row in rows if row[2].stem in wanted]
    missing = wanted - {row[2].stem for row in selected}
    if missing:
        raise ValueError(f"unknown packet IDs: {sorted(missing)}")
    return selected


def main() -> None:
    args = parse_args()
    lock = validate_lock()
    task = "semantic" if args.task == "adjudicate" else args.task
    rows = prepare_adjudication(lock) if args.task == "adjudicate" else identity_packets(task)
    rows = selected_packets(rows, args.packet_id)
    failures: list[tuple[str, str]] = []
    counts = {"complete": 0, "skipped": 0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {
            pool.submit(
                run_packet,
                lock=lock,
                identity=identity,
                task=row_task,
                packet_path=packet_path,
                resume=args.resume,
            ): f"{identity}/{packet_path.stem}"
            for identity, row_task, packet_path in rows
        }
        for future in concurrent.futures.as_completed(futures):
            label = futures[future]
            try:
                _, status = future.result()
                counts[status] += 1
                with PRINT_LOCK:
                    print(f"[{status}] {label}", flush=True)
            except Exception as exc:  # noqa: BLE001 - preserve independent failures
                failures.append((label, str(exc)))
                with PRINT_LOCK:
                    print(f"[failed] {label}: {exc}", flush=True)
    summary = {
        "task": args.task,
        "expected": len(rows),
        **counts,
        "failed": len(failures),
        "failures": failures,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
