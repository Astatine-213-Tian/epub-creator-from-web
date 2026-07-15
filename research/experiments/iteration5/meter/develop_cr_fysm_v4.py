#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from experiments.iteration5.meter.benchmark_content_resistant_meter import (
    TARGET_AUTHOR,
    author_book_weights,
    load_records,
)
from experiments.iteration5.meter.build_cr_fysm_v3 import (
    CANONICAL_PREREGISTRATION as V3_PREREGISTRATION,
    FamilyArtifacts,
    clustered_rate_lower_bound,
    dialogue_rate,
    ensemble,
    family_probabilities,
    load_preregistration,
    locked_family_contract,
    score_metrics,
    sha256_file,
)


DEFAULT_OUTPUT_DIR = V3_PREREGISTRATION.parent.parent / "cr_fysm_v4"
SPECIFICITY_FLOORS = (0.90, 0.95, 0.975, 0.99)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Develop CR-FYSM-v4 threshold policy from v3 fit/calibration roles only."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_family_artifacts(
    model_dir: Path, family_order: tuple[str, ...]
) -> dict[str, FamilyArtifacts]:
    artifacts: dict[str, FamilyArtifacts] = {}
    for name in family_order:
        vectorizer = joblib.load(model_dir / f"{name}.vectorizer.joblib")
        keep_mask = joblib.load(model_dir / f"{name}.keep_mask.joblib")
        feature_names = np.asarray(vectorizer.get_feature_names_out())[keep_mask].tolist()
        artifacts[name] = FamilyArtifacts(
            name=name,
            vectorizer=vectorizer,
            scaler=joblib.load(model_dir / f"{name}.scaler.joblib"),
            model=joblib.load(model_dir / f"{name}.model.joblib"),
            feature_names=feature_names,
            keep_mask=keep_mask,
        )
    return artifacts


def calibration_records(
    records: list[Any], partition: dict[str, Any]
) -> list[Any]:
    target_titles = set(partition["target"]["calibration"])
    comparison_authors = set(partition["comparison"]["calibration"])
    selected = [
        record
        for record in records
        if (
            record.author == TARGET_AUTHOR
            and record.title in target_titles
            and record.split == "dev"
        )
        or (
            record.author in comparison_authors
            and record.author != TARGET_AUTHOR
            and record.split == "dev"
        )
    ]
    selected.sort(key=lambda record: record.chunk_id)
    if {record.title for record in selected if record.author == TARGET_AUTHOR} != target_titles:
        raise ValueError("v3 target calibration books are incomplete")
    if {record.author for record in selected if record.author != TARGET_AUTHOR} != comparison_authors:
        raise ValueError("v3 comparison calibration authors are incomplete")
    return selected


def policy_row(
    records: list[Any],
    scores: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray,
    *,
    specificity_floor: float,
) -> dict[str, Any]:
    candidates: list[tuple[float, dict[str, float]]] = []
    for threshold in sorted(float(value) for value in np.unique(scores)):
        metrics = score_metrics(scores, truth, threshold, weights)
        if metrics["sensitivity"] >= 0.80 and metrics["specificity"] >= specificity_floor:
            candidates.append((threshold, metrics))
    if not candidates:
        raise ValueError(f"no threshold reaches specificity floor {specificity_floor}")
    threshold, metrics = min(candidates, key=lambda item: item[0])

    target_groups: dict[str, list[int]] = defaultdict(list)
    comparison_groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        destination = target_groups if record.author == TARGET_AUTHOR else comparison_groups
        key = record.title if record.author == TARGET_AUTHOR else record.author
        destination[key].append(index)
    book_sensitivities = [
        float(np.mean(scores[indices] >= threshold))
        for indices in target_groups.values()
    ]
    author_specificities = [
        float(np.mean(scores[indices] < threshold))
        for indices in comparison_groups.values()
    ]

    negative_indices = [
        index for index, record in enumerate(records) if record.author != TARGET_AUTHOR
    ]
    dialogue_values = np.asarray(
        [dialogue_rate(records[index].text) for index in negative_indices]
    )
    lower, upper = np.quantile(dialogue_values, [1.0 / 3.0, 2.0 / 3.0])
    stratum_fprs: dict[str, float] = {}
    for label in ("low", "mid", "high"):
        stratum: list[int] = []
        for index in negative_indices:
            value = dialogue_rate(records[index].text)
            if label == "low" and value <= lower:
                stratum.append(index)
            elif label == "mid" and lower < value <= upper:
                stratum.append(index)
            elif label == "high" and value > upper:
                stratum.append(index)
        stratum_fprs[label] = float(np.mean(scores[stratum] >= threshold))

    row = {
        "specificity_floor": specificity_floor,
        "threshold": threshold,
        **metrics,
        "clustered_sensitivity_lower": clustered_rate_lower_bound(
            records,
            scores,
            threshold,
            target_author=TARGET_AUTHOR,
            positive=True,
        ),
        "clustered_specificity_lower": clustered_rate_lower_bound(
            records,
            scores,
            threshold,
            target_author=TARGET_AUTHOR,
            positive=False,
        ),
        "minimum_target_book_sensitivity": min(book_sensitivities),
        "minimum_comparison_author_specificity": min(author_specificities),
        "low_dialogue_false_positive_rate": stratum_fprs["low"],
        "mid_dialogue_false_positive_rate": stratum_fprs["mid"],
        "high_dialogue_false_positive_rate": stratum_fprs["high"],
    }
    row["selection_eligible"] = bool(
        row["sensitivity"] >= 0.90
        and row["clustered_specificity_lower"] >= 0.90
        and row["minimum_target_book_sensitivity"] >= 0.90
        and row["minimum_comparison_author_specificity"] >= 0.90
        and max(
            row["low_dialogue_false_positive_rate"],
            row["mid_dialogue_false_positive_rate"],
            row["high_dialogue_false_positive_rate"],
        )
        <= 0.10
    )
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    v3 = load_preregistration(V3_PREREGISTRATION)
    v3_output = Path(v3["paths"]["output_dir"])
    v3_results = json.loads((v3_output / "results.json").read_text(encoding="utf-8"))
    if v3_results["status"] != "statistical_qualification_fail":
        raise ValueError("v4 development requires the frozen v3 NO-GO result")
    family_order, family_weights = locked_family_contract(v3["model_contract"])
    boundary = int(v3["model_contract"]["boundary_chunks_excluded"])
    records = load_records(Path(v3["paths"]["active_masked_dataset"]), boundary=boundary)
    development_records = calibration_records(records, v3["partition"])

    artifacts = load_family_artifacts(v3_output / "meter", family_order)
    family_scores = family_probabilities(
        artifacts, [record.text for record in development_records]
    )
    raw_scores = ensemble(family_scores, family_order, family_weights)
    truth = np.asarray(
        [record.author == TARGET_AUTHOR for record in development_records],
        dtype=np.int8,
    )
    scores = raw_scores
    weights = author_book_weights(development_records, binary_target=TARGET_AUTHOR)
    rows = [
        policy_row(
            development_records,
            scores,
            truth,
            weights,
            specificity_floor=floor,
        )
        for floor in SPECIFICITY_FLOORS
    ]
    eligible = [row for row in rows if row["selection_eligible"]]
    if not eligible:
        raise ValueError("no calibration-only policy passed the v4 selection contract")
    selected = max(
        eligible,
        key=lambda row: (
            float(row["balanced_accuracy"]),
            float(row["sensitivity"]),
            -float(row["threshold"]),
        ),
    )

    model_files = sorted((v3_output / "meter").glob("*.joblib"))
    payload = {
        "schema_version": 1,
        "development_id": "CR-FYSM-v4-calibration-only-risk-policy",
        "status": "development_complete_holdout_unscored",
        "source_meter": "CR-FYSM-v3",
        "source_lock_id": v3["lock_id"],
        "development_roles": {
            "target_books": sorted(
                {record.title for record in development_records if record.author == TARGET_AUTHOR}
            ),
            "comparison_authors": sorted(
                {record.author for record in development_records if record.author != TARGET_AUTHOR}
            ),
            "rows": len(development_records),
        },
        "forbidden_inputs": {
            "v3_target_qualification_books": v3["partition"]["target"]["qualification"],
            "v3_external_qualification_authors": v3["partition"]["comparison"]["qualification"],
            "v4_fresh_holdout_scored": False,
        },
        "candidate_specificity_floors": list(SPECIFICITY_FLOORS),
        "selection_rule": (
            "Among policies with sensitivity >=0.90, clustered specificity lower >=0.90, "
            "minimum target-book sensitivity >=0.90, minimum comparison-author specificity "
            ">=0.90, and every dialogue-stratum FPR <=0.10, select maximum balanced accuracy; "
            "then maximum sensitivity and lower threshold."
        ),
        "decision_score_space": "raw equal-weight family-probability ensemble",
        "platt_role": "probability calibration diagnostics only; never threshold selection",
        "candidates": rows,
        "selected": selected,
        "literature_basis": [
            {
                "title": "Neyman-Pearson classification algorithms and NP-ROC",
                "url": "https://arxiv.org/abs/1608.03109",
                "use": "empirical type-I error at the target ceiling is not sufficient control",
            },
            {
                "title": "Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control",
                "url": "https://arxiv.org/abs/2110.01052",
                "use": "select a low-dimensional threshold on calibration data before final testing",
            },
            {
                "title": "Unsupervised Calibration under Covariate Shift",
                "url": "https://arxiv.org/abs/2006.16405",
                "use": "calibration can be brittle under input-distribution shift",
            },
        ],
        "source_artifact_hashes": {
            "v3_preregistration": sha256_file(V3_PREREGISTRATION),
            "v3_results": sha256_file(v3_output / "results.json"),
            **{str(path): sha256_file(path) for path in model_files},
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "development_candidates.v4.csv", rows)
    (args.output_dir / "development.v4.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "selected": selected}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
