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

from experiments.validation.meter.benchmark_content_resistant_meter import (
    TARGET_AUTHOR,
    author_book_weights,
    load_records,
)
from experiments.validation.meter.build_cr_fysm_v3 import (
    CANONICAL_PREREGISTRATION,
    FamilyArtifacts,
    calibrated_scores,
    clustered_rate_lower_bound,
    cross_fitted_platt_scores,
    dialogue_rate,
    ensemble,
    family_probabilities,
    load_preregistration,
    locked_family_contract,
    role_records,
    score_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write exploratory post-hoc diagnostics for frozen CR-FYSM-v3."
    )
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=CANONICAL_PREREGISTRATION,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CANONICAL_PREREGISTRATION.parent / "posthoc_diagnostics",
    )
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    preregistration = load_preregistration(args.preregistration)
    paths = {key: Path(value) for key, value in preregistration["paths"].items()}
    output_dir = paths["output_dir"]
    results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    if results["status"] != "statistical_qualification_fail":
        raise ValueError("this diagnostic is locked to the failed CR-FYSM-v3 result")

    family_order, family_weights = locked_family_contract(
        preregistration["model_contract"]
    )
    boundary = int(preregistration["model_contract"]["boundary_chunks_excluded"])
    active_records = load_records(paths["active_masked_dataset"], boundary=boundary)
    external_records = load_records(paths["external_masked_dataset"], boundary=boundary)
    roles = role_records(
        active_records,
        external_records,
        preregistration["partition"],
        TARGET_AUTHOR,
    )
    qualification = roles["qualification"]
    calibration_records = roles["calibration"]
    artifacts = load_family_artifacts(output_dir / "meter", family_order)
    family_scores = family_probabilities(
        artifacts, [record.text for record in qualification]
    )
    raw_scores = ensemble(family_scores, family_order, family_weights)
    calibrator = joblib.load(output_dir / "meter/ensemble_platt_calibrator.joblib")
    scores = calibrated_scores(calibrator, raw_scores)
    threshold = float(results["ensemble"]["calibration"]["threshold"])

    calibration_family_scores = family_probabilities(
        artifacts, [record.text for record in calibration_records]
    )
    calibration_raw_scores = ensemble(
        calibration_family_scores, family_order, family_weights
    )
    calibration_truth = np.asarray(
        [record.author == TARGET_AUTHOR for record in calibration_records],
        dtype=np.int8,
    )
    calibration_scores, _development_calibrator, _fold_sizes = (
        cross_fitted_platt_scores(
            calibration_raw_scores,
            calibration_truth,
            calibration_records,
            target_author=TARGET_AUTHOR,
        )
    )
    calibration_weights = author_book_weights(
        calibration_records, binary_target=TARGET_AUTHOR
    )

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(qualification):
        if record.author != TARGET_AUTHOR:
            grouped[record.author].append(index)

    rows: list[dict[str, Any]] = []
    for author, indices in sorted(grouped.items()):
        author_scores = scores[indices]
        row: dict[str, Any] = {
            "author": author,
            "rows": len(indices),
            "false_positive_rate": float(np.mean(author_scores >= threshold)),
            "mean_calibrated_score": float(np.mean(author_scores)),
            "p95_calibrated_score": float(np.quantile(author_scores, 0.95)),
            "mean_dialogue_rate": float(
                np.mean([dialogue_rate(qualification[index].text) for index in indices])
            ),
        }
        for family in family_order:
            values = family_scores[family][indices]
            family_threshold = float(
                results["families"][family]["calibration_threshold"]["threshold"]
            )
            row[f"{family}_mean_probability"] = float(np.mean(values))
            row[f"{family}_false_positive_rate"] = float(
                np.mean(values >= family_threshold)
            )
        rows.append(row)
    rows.sort(key=lambda row: (-float(row["false_positive_rate"]), str(row["author"])))

    truth = np.asarray(
        [record.author == TARGET_AUTHOR for record in qualification], dtype=np.int8
    )
    weights = author_book_weights(qualification, binary_target=TARGET_AUTHOR)
    tradeoff_rows: list[dict[str, Any]] = []
    for sensitivity_floor in (0.99, 0.95, 0.90, 0.80):
        candidates: list[tuple[float, dict[str, float]]] = []
        for candidate in sorted(float(value) for value in np.unique(scores)):
            metrics = score_metrics(scores, truth, candidate, weights)
            if metrics["sensitivity"] >= sensitivity_floor:
                candidates.append((candidate, metrics))
        if not candidates:
            continue
        candidate, metrics = max(
            candidates,
            key=lambda item: (
                item[1]["specificity"],
                item[1]["balanced_accuracy"],
                item[0],
            ),
        )
        tradeoff_rows.append(
            {
                "target_sensitivity_floor": sensitivity_floor,
                "posthoc_threshold": candidate,
                **metrics,
            }
        )

    calibration_policy_rows: list[dict[str, Any]] = []
    for specificity_floor in (0.90, 0.95, 0.975, 0.99):
        candidates: list[tuple[float, dict[str, float]]] = []
        for candidate in sorted(float(value) for value in np.unique(calibration_scores)):
            metrics = score_metrics(
                calibration_scores,
                calibration_truth,
                candidate,
                calibration_weights,
            )
            if metrics["sensitivity"] >= 0.80 and metrics["specificity"] >= specificity_floor:
                candidates.append((candidate, metrics))
        if not candidates:
            continue
        candidate, metrics = min(candidates, key=lambda item: item[0])
        negative_groups: dict[str, list[int]] = defaultdict(list)
        target_groups: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(calibration_records):
            if record.author == TARGET_AUTHOR:
                target_groups[record.title].append(index)
            else:
                negative_groups[record.author].append(index)
        author_specificities = [
            float(np.mean(calibration_scores[indices] < candidate))
            for indices in negative_groups.values()
        ]
        book_sensitivities = [
            float(np.mean(calibration_scores[indices] >= candidate))
            for indices in target_groups.values()
        ]
        negative_indices = [
            index
            for index, record in enumerate(calibration_records)
            if record.author != TARGET_AUTHOR
        ]
        dialogue_values = np.asarray(
            [dialogue_rate(calibration_records[index].text) for index in negative_indices]
        )
        lower, upper = np.quantile(dialogue_values, [1.0 / 3.0, 2.0 / 3.0])
        stratum_fprs: dict[str, float] = {}
        for label in ("low", "mid", "high"):
            stratum = [
                index
                for index in negative_indices
                if (
                    (label == "low" and dialogue_rate(calibration_records[index].text) <= lower)
                    or (
                        label == "mid"
                        and lower < dialogue_rate(calibration_records[index].text) <= upper
                    )
                    or (
                        label == "high"
                        and dialogue_rate(calibration_records[index].text) > upper
                    )
                )
            ]
            stratum_fprs[label] = float(
                np.mean(calibration_scores[stratum] >= candidate)
            )
        calibration_policy_rows.append(
            {
                "specificity_floor": specificity_floor,
                "threshold": candidate,
                **metrics,
                "clustered_sensitivity_lower": clustered_rate_lower_bound(
                    calibration_records,
                    calibration_scores,
                    candidate,
                    target_author=TARGET_AUTHOR,
                    positive=True,
                ),
                "clustered_specificity_lower": clustered_rate_lower_bound(
                    calibration_records,
                    calibration_scores,
                    candidate,
                    target_author=TARGET_AUTHOR,
                    positive=False,
                ),
                "minimum_target_book_sensitivity": min(book_sensitivities),
                "minimum_comparison_author_specificity": min(author_specificities),
                "low_dialogue_false_positive_rate": stratum_fprs["low"],
                "mid_dialogue_false_positive_rate": stratum_fprs["mid"],
                "high_dialogue_false_positive_rate": stratum_fprs["high"],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "external_author_false_positives.csv", rows)
    write_csv(args.output_dir / "qualification_threshold_tradeoff.csv", tradeoff_rows)
    write_csv(
        args.output_dir / "calibration_threshold_policies.csv",
        calibration_policy_rows,
    )
    summary = {
        "schema_version": 1,
        "meter_id": "CR-FYSM-v3",
        "status": "exploratory_posthoc_only",
        "confirmatory": False,
        "may_change_v3_gates": False,
        "may_be_used_as_v3_result": False,
        "qualification_was_already_opened": True,
        "external_authors": len(rows),
        "locked_threshold": threshold,
        "highest_false_positive_authors": rows[:5],
        "threshold_tradeoff": tradeoff_rows,
        "calibration_only_threshold_policies": calibration_policy_rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
