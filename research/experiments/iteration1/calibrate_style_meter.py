from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


from experiments.shared.paths import RESEARCH_ROOT


REPO_ROOT = RESEARCH_ROOT
DEFAULT_EXPERIMENT_ROOT = (
    REPO_ROOT / "generated/style_research/style_transfer_experiments"
)
DEFAULT_SAMPLE_SET = "development_proxy_v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the provisional hinge-margin style threshold using calibration "
            "rows only."
        )
    )
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument(
        "--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT
    )
    parser.add_argument("--sample-set", default=DEFAULT_SAMPLE_SET)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def allocation_path(experiment_root: Path, sample_set: str) -> Path:
    return (
        experiment_root
        / "sample_sets"
        / f"{sample_set}.evaluator_allocation.jsonl"
    )


def threshold_metrics(
    threshold: float, positives: list[float], negatives: list[float]
) -> dict[str, float]:
    sensitivity = sum(value >= threshold for value in positives) / len(positives)
    false_positive_rate = sum(value >= threshold for value in negatives) / len(
        negatives
    )
    specificity = 1.0 - false_positive_rate
    return {
        "threshold": threshold,
        "sensitivity": sensitivity,
        "false_positive_rate": false_positive_rate,
        "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
    }


def calibrate(
    scores: list[dict[str, Any]],
    allocation: list[dict[str, Any]],
    calibration_protocol: dict[str, Any],
) -> dict[str, Any]:
    minimum_sensitivity = float(
        calibration_protocol["minimum_positive_sensitivity"]
    )
    maximum_false_positive_rate = float(
        calibration_protocol["maximum_neutral_false_positive_rate"]
    )
    if not 0.0 <= minimum_sensitivity <= 1.0:
        raise ValueError("--minimum-sensitivity must be between 0 and 1.")
    if not 0.0 <= maximum_false_positive_rate <= 1.0:
        raise ValueError(
            "--maximum-false-positive-rate must be between 0 and 1."
        )

    calibration_ids = {
        row["sample_id"]
        for row in allocation
        if row["research_role"] == "style_meter_calibration"
    }
    expected_count = int(calibration_protocol["sample_count"])
    if len(calibration_ids) != expected_count:
        raise ValueError(
            f"Frozen allocation has {len(calibration_ids)} calibration rows; "
            f"protocol requires {expected_count}."
        )
    score_ids: list[str] = []
    for row in scores:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str):
            raise ValueError("Every score row needs a string sample_id.")
        score_ids.append(sample_id)
    if len(score_ids) != len(set(score_ids)):
        raise ValueError("Score input contains duplicate sample IDs.")
    if set(score_ids) != calibration_ids:
        missing = sorted(calibration_ids - set(score_ids))
        extra = sorted(set(score_ids) - calibration_ids)
        raise ValueError(
            "Score input must contain exactly the frozen calibration rows; "
            f"missing={missing[:5]}, extra={extra[:5]}."
        )

    binding_fields = calibration_protocol["required_score_binding_fields"]
    stable_binding_fields = {
        "scorer_id",
        "classifier_artifact_sha256",
        "scorer_config_sha256",
        "masking_view",
        "masking_artifact_sha256",
    }
    for field in binding_fields:
        raw_values = [row.get(field) for row in scores]
        if any(not isinstance(value, str) or not value for value in raw_values):
            raise ValueError(f"Every score row needs a non-empty {field} binding.")
        values = [str(value) for value in raw_values]
        if field.endswith("_sha256") and any(
            not SHA256_RE.fullmatch(value) for value in values
        ):
            raise ValueError(f"Every {field} value must be a SHA-256 hex digest.")
        if field in stable_binding_fields and len(set(values)) != 1:
            raise ValueError(f"Calibration rows disagree on frozen {field}.")
    if {row["masking_view"] for row in scores} != {"entity_masked_v3"}:
        raise ValueError("Calibration scores must use entity_masked_v3.")

    positives: list[float] = []
    negatives: list[float] = []
    for row in scores:
        try:
            positives.append(float(row["original_target_margin"]))
            negatives.append(float(row["neutral_target_margin"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Each score row needs finite original_target_margin and "
                "neutral_target_margin values."
            ) from exc
    if not positives or any(
        value != value or value in {float("inf"), float("-inf")}
        for value in positives + negatives
    ):
        raise ValueError("Calibration margins must be finite.")

    thresholds = sorted(set(positives + negatives))
    candidates = [
        threshold_metrics(threshold, positives, negatives)
        for threshold in thresholds
    ]
    qualifying = [
        row
        for row in candidates
        if row["sensitivity"] >= minimum_sensitivity
        and row["false_positive_rate"] <= maximum_false_positive_rate
    ]
    if not qualifying:
        raise RuntimeError(
            "No threshold meets the preregistered sensitivity and false-positive "
            "constraints; the binary style-success endpoint remains undefined."
        )
    winner = max(
        qualifying,
        key=lambda row: (row["balanced_accuracy"], row["threshold"]),
    )
    return {
        "algorithm": "hinge_margin_threshold.v1",
        "calibration_rows": len(scores),
        "minimum_sensitivity": minimum_sensitivity,
        "maximum_false_positive_rate": maximum_false_positive_rate,
        "selected": winner,
        "qualifying_threshold_count": len(qualifying),
        "candidate_threshold_count": len(candidates),
        "score_binding": {
            field: scores[0][field]
            for field in sorted(stable_binding_fields)
        },
    }


def main() -> None:
    args = parse_args()
    allocation = allocation_path(args.experiment_root, args.sample_set)
    scores = list(iter_jsonl(args.scores))
    allocation_rows = list(iter_jsonl(allocation))
    protocol_path = args.experiment_root / "protocols/evaluation_protocol.v1.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    result = calibrate(
        scores,
        allocation_rows,
        protocol["calibration"],
    )
    output = args.output or (
        args.experiment_root / "calibration/style_meter_threshold.v1.json"
    )
    if output.exists():
        raise FileExistsError(
            f"Refusing to replace frozen calibration artifact: {output}"
        )
    artifact = {
        "schema_version": 1,
        "created_at": utc_now(),
        "sample_set": args.sample_set,
        "scores_path": str(args.scores),
        "scores_sha256": file_sha256(args.scores),
        "allocation_path": str(allocation),
        "allocation_sha256": file_sha256(allocation),
        "evaluation_protocol_path": str(protocol_path),
        "evaluation_protocol_sha256": file_sha256(protocol_path),
        **result,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
