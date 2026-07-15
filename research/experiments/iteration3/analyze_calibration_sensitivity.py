#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable


from experiments.shared.paths import RESEARCH_ROOT


REPO_ROOT = RESEARCH_ROOT

import experiments.iteration1.calibrate_style_meter as calibration


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure threshold sensitivity to the iteration-3 safety replacement."
    )
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--sample-set", required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--threshold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.experiment_root.expanduser().resolve()
    scores_path = args.scores.expanduser().resolve()
    threshold_path = args.threshold.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace sensitivity artifact: {output}")
    amendment_path = root / "protocols/iteration3_source_cohort_amendment.v1.json"
    protocol_path = root / "protocols/evaluation_protocol.v1.json"
    allocation_path = root / f"sample_sets/{args.sample_set}.evaluator_allocation.jsonl"
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    replacement_ids = {
        str(row["replacement_sample_id"]) for row in amendment["replacements"]
    }
    scores = [
        row for row in iter_jsonl(scores_path) if str(row["sample_id"]) not in replacement_ids
    ]
    allocation = [
        row
        for row in iter_jsonl(allocation_path)
        if str(row["sample_id"]) not in replacement_ids
    ]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    sensitivity_protocol = copy.deepcopy(protocol["calibration"])
    sensitivity_protocol["sample_count"] = len(scores)
    result = calibration.calibrate(scores, allocation, sensitivity_protocol)
    primary = json.loads(threshold_path.read_text(encoding="utf-8"))
    primary_threshold = float(primary["selected"]["threshold"])
    sensitivity_threshold = float(result["selected"]["threshold"])
    artifact = {
        "schema_version": 1,
        "analysis": "leave_one_safety_replacement_out_threshold_sensitivity",
        "sample_set": args.sample_set,
        "omitted_sample_ids": sorted(replacement_ids),
        "remaining_calibration_rows": len(scores),
        "scores_path": str(scores_path.relative_to(REPO_ROOT)),
        "scores_sha256": file_sha256(scores_path),
        "primary_threshold_path": str(threshold_path.relative_to(REPO_ROOT)),
        "primary_threshold_sha256": file_sha256(threshold_path),
        "amendment_sha256": file_sha256(amendment_path),
        "primary_threshold": primary_threshold,
        "sensitivity_threshold": sensitivity_threshold,
        "absolute_threshold_delta": abs(sensitivity_threshold - primary_threshold),
        "sensitivity_result": result,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
