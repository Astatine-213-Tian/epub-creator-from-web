#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from experiments.iteration5.construct import run_cr_fysm_v4_construct_generation as runner


OUTPUT = runner.ROOT / "generation_completion_manifest.v2.json"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> None:
    preregistration = runner.validate_preregistration()
    if OUTPUT.exists():
        raise ValueError("construct generation closure already exists")
    samples = runner.read_jsonl(runner.SELECTION)
    expected_samples = int(
        preregistration["generation_completion_gates"]["rows_per_stage"]
    )
    if len(samples) != expected_samples:
        raise ValueError("source selection does not contain the registered sample count")
    sample_ids = {row["sample_id"] for row in samples}
    records: list[dict[str, Any]] = []
    for stage in runner.STAGES:
        stage_dir = runner.ROOT / "generation" / stage
        expected_paths = {stage_dir / f"{sample_id}.json" for sample_id in sample_ids}
        actual_paths = set(stage_dir.glob("*.json")) if stage_dir.is_dir() else set()
        if actual_paths != expected_paths:
            missing = sorted(str(path) for path in expected_paths - actual_paths)
            extra = sorted(str(path) for path in actual_paths - expected_paths)
            raise ValueError(f"incomplete {stage} artifact set: missing={missing}, extra={extra}")
        for sample in samples:
            sample_id = sample["sample_id"]
            request = runner.build_request(stage, sample, preregistration)
            artifact = runner.read_json(runner.output_path(stage, sample_id))
            runner.validate_artifact(stage, sample, preregistration, request, artifact)
            records.append(
                {
                    "stage": stage,
                    "sample_id": sample_id,
                    "path": str(runner.output_path(stage, sample_id)),
                    "artifact_sha256": runner.sha256_file(runner.output_path(stage, sample_id)),
                    "request_sha256": runner.sha256_text(canonical(request)),
                    "result_sha256": artifact["result_sha256"],
                }
            )
    expected_total = expected_samples * len(runner.STAGES)
    if len(records) != expected_total:
        raise ValueError("construct generation closure record count mismatch")
    payload: dict[str, Any] = {
        "schema_version": 2,
        "status": "complete_before_any_construct_rating_or_scoring",
        "generation_lock_id": preregistration["lock_id"],
        "selection_sha256": runner.sha256_file(runner.SELECTION),
        "stage_counts": {
            stage: sum(row["stage"] == stage for row in records) for stage in runner.STAGES
        },
        "artifact_count": len(records),
        "artifacts": sorted(records, key=lambda row: (row["stage"], row["sample_id"])),
    }
    payload["manifest_id"] = runner.sha256_text(canonical(payload))
    serialized = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    descriptor = os.open(OUTPUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    print(
        json.dumps(
            {
                "status": payload["status"],
                "manifest_id": payload["manifest_id"],
                "artifact_count": payload["artifact_count"],
                "output": str(OUTPUT),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
