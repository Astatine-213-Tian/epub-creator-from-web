#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


from experiments.shared.paths import RESEARCH_ROOT


REPO_ROOT = RESEARCH_ROOT
USAGE_LIMIT_MARKER = "You've hit your usage limit."
RESET_PATTERN = re.compile(r"try again at ([^.]+\.)")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"Expected an object at {path}:{line_number}")
        rows.append(value)
    return rows


def repo_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def usage_limit_messages(row: dict[str, Any]) -> list[str]:
    messages: list[str] = []
    for event in row.get("response_errors", []):
        if not isinstance(event, dict):
            continue
        message = event.get("message")
        if isinstance(message, str):
            messages.append(message)
        error = event.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            messages.append(error["message"])
    return [message for message in messages if USAGE_LIMIT_MARKER in message]


def extract_reset_at(messages: list[str]) -> str:
    values: set[str] = set()
    for message in messages:
        match = RESET_PATTERN.search(message)
        if match:
            values.add(match.group(1).rstrip("."))
    if len(values) != 1:
        raise RuntimeError(f"Expected one usage-limit reset time, found {sorted(values)}")
    return values.pop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a resumable LLM stage interrupted only by a provider usage limit."
    )
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--sample-set", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--selection-file", type=Path, required=True)
    parser.add_argument("--repair-round", type=int)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")
    if args.stage in {"english_source_repair", "english_source_repair_qa"}:
        if args.repair_round is None or args.repair_round < 1:
            raise SystemExit("Repair stages require --repair-round >= 1")

    root = args.experiment_root.expanduser().resolve()
    selection_path = args.selection_file.expanduser().resolve()
    selection = read_json(selection_path)
    sample_ids = [str(value) for value in selection.get("sample_ids", [])]
    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("Selection is empty or contains duplicate sample IDs")

    run_root = root / "runs" / args.sample_set / args.run_id
    stage_suffix = (
        f".round_{args.repair_round:02d}" if args.repair_round is not None else ""
    )
    stage_key = f"{args.stage}{stage_suffix}"
    output_dir = run_root / args.stage
    if args.repair_round is not None:
        output_dir = output_dir / f"round_{args.repair_round:02d}"
    ledger_path = run_root / "ledgers" / f"{stage_key}.jsonl"
    if not ledger_path.exists():
        raise RuntimeError(f"Missing stage ledger: {ledger_path}")

    rows = read_jsonl(ledger_path)
    rows_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if sample_id in sample_ids:
            rows_by_sample[sample_id].append(row)

    completed: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    reset_messages: list[str] = []
    models: set[str] = set()
    for sample_id in sorted(sample_ids):
        artifact_path = output_dir / f"{sample_id}.json"
        sample_rows = rows_by_sample.get(sample_id, [])
        models.update(
            str(row["model"]) for row in sample_rows if isinstance(row.get("model"), str)
        )
        if artifact_path.exists():
            artifact = read_json(artifact_path)
            if artifact.get("sample_id") != sample_id:
                raise RuntimeError(f"Artifact sample ID mismatch: {artifact_path}")
            if not any(row.get("status") == "success" for row in sample_rows):
                raise RuntimeError(f"Artifact lacks a successful ledger row: {sample_id}")
            completed.append(
                {
                    "sample_id": sample_id,
                    "artifact_path": repo_path(artifact_path),
                    "artifact_sha256": file_sha256(artifact_path),
                }
            )
            continue

        failed_rows = [row for row in sample_rows if row.get("status") == "failed"]
        if not failed_rows:
            raise RuntimeError(f"Pending sample lacks a failed ledger row: {sample_id}")
        messages_by_attempt = [usage_limit_messages(row) for row in failed_rows]
        if not all(messages_by_attempt):
            raise RuntimeError(f"Non-quota failure found for pending sample: {sample_id}")
        sample_messages = [message for messages in messages_by_attempt for message in messages]
        reset_messages.extend(sample_messages)
        pending.append(
            {
                "sample_id": sample_id,
                "failed_attempt_count": len(failed_rows),
                "latest_completed_at": failed_rows[-1].get("completed_at"),
            }
        )

    if not pending:
        raise RuntimeError("Stage is complete; no quota-resume manifest is needed")
    reset_at = extract_reset_at(reset_messages)
    if len(models) != 1:
        raise RuntimeError(f"Expected one model in the stage ledger, found {sorted(models)}")

    resume_argv = [
        "uv",
        "run",
        "python",
        "experiments/iteration1/run_style_transfer_generation.py",
        "run",
        "--experiment-root",
        repo_path(root),
        "--sample-set",
        args.sample_set,
        "--stage",
        args.stage,
        "--run-id",
        args.run_id,
        "--selection-file",
        repo_path(selection_path),
    ]
    if args.repair_round is not None:
        resume_argv.extend(["--repair-round", str(args.repair_round)])
    resume_argv.extend(["--jobs", str(args.jobs), "--resume"])

    payload = {
        "schema_version": 1,
        "status": "interrupted_provider_usage_limit",
        "interruption_is_research_outcome": False,
        "sample_set": args.sample_set,
        "run_id": args.run_id,
        "stage": args.stage,
        "repair_round": args.repair_round,
        "model": models.pop(),
        "selection_path": repo_path(selection_path),
        "selection_sha256": file_sha256(selection_path),
        "ledger_path": repo_path(ledger_path),
        "ledger_sha256": file_sha256(ledger_path),
        "selected_sample_count": len(sample_ids),
        "completed_sample_count": len(completed),
        "pending_sample_count": len(pending),
        "quota_reset_at_provider_text": reset_at,
        "completed_artifacts": completed,
        "pending_samples": pending,
        "resume_argv": resume_argv,
    }
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else run_root / "resume_manifests" / f"{stage_key}.usage_limit.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": repo_path(output_path),
                "completed_sample_count": len(completed),
                "pending_sample_count": len(pending),
                "quota_reset_at_provider_text": reset_at,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
