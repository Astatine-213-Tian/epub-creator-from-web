#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import os
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
    calibrated_scores,
    canonical_lock_id,
    clustered_rate_lower_bound,
    dialogue_stratum_false_positive_rates,
    ensemble,
    expected_calibration_error,
    family_probabilities,
    locked_family_contract,
    runtime_versions,
    score_metrics,
    sha256_file,
)
from experiments.iteration5.meter.develop_cr_fysm_v4 import (
    calibration_records,
    load_family_artifacts,
)


CANONICAL_PREREGISTRATION = Path(
    "generated/style_research/style_transfer_experiments/iterations/"
    "content_resistant_v1/cr_fysm_v4/preregistration.v4.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute the one-shot preregistered CR-FYSM-v4 qualification."
    )
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=CANONICAL_PREREGISTRATION,
    )
    return parser.parse_args()


def load_preregistration(path: Path) -> dict[str, Any]:
    if path.resolve() != CANONICAL_PREREGISTRATION.resolve():
        raise ValueError("CR-FYSM-v4 only accepts the canonical preregistration path")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("meter_id") != "CR-FYSM-v4":
        raise ValueError("preregistration meter_id is not CR-FYSM-v4")
    if payload.get("status") != "locked_before_fresh_holdout_scoring":
        raise ValueError("v4 preregistration is not locked")
    if payload.get("lock_id") != canonical_lock_id(payload):
        raise ValueError("v4 lock_id does not match its canonical payload")
    paths = {key: Path(value) for key, value in payload["paths"].items()}
    artifact_paths = {
        key: Path(value) for key, value in payload["source_artifact_paths"].items()
    }
    actual = {
        f"path:{key}": sha256_file(value)
        for key, value in paths.items()
        if key not in {"source_model_dir", "output_dir"}
    }
    actual.update(
        {f"artifact:{key}": sha256_file(value) for key, value in artifact_paths.items()}
    )
    actual.update(
        {
            "script:builder": sha256_file(Path(__file__)),
            "script:preregister": sha256_file(
                Path(__file__).with_name("preregister_cr_fysm_v4.py")
            ),
            "script:development": sha256_file(
                Path(__file__).with_name("develop_cr_fysm_v4.py")
            ),
            "script:v3_builder": sha256_file(
                Path(__file__).with_name("build_cr_fysm_v3.py")
            ),
            "script:benchmark": sha256_file(
                Path(__file__).with_name("benchmark_content_resistant_meter.py")
            ),
            "script:shared_feature_definitions": sha256_file(
                Path("workflows/benchmark_author_style.py")
            ),
            "uv.lock": sha256_file(Path("uv.lock")),
        }
    )
    if actual != payload.get("input_hashes"):
        missing = sorted(set(actual) - set(payload.get("input_hashes", {})))
        extra = sorted(set(payload.get("input_hashes", {})) - set(actual))
        changed = sorted(
            key
            for key in set(actual) & set(payload.get("input_hashes", {}))
            if actual[key] != payload["input_hashes"][key]
        )
        raise ValueError(
            f"v4 input hashes differ: missing={missing}, extra={extra}, changed={changed}"
        )
    if payload.get("environment") != runtime_versions():
        raise ValueError("v4 runtime package versions differ from preregistration")
    return payload


def qualification_records(
    records: list[Any], partition: dict[str, Any]
) -> list[Any]:
    target_titles = set(partition["target_books"])
    comparison_authors = set(partition["comparison_authors"])
    selected = [
        record
        for record in records
        if (
            record.author == TARGET_AUTHOR
            and record.title in target_titles
            and record.split == "proxy_transfer"
        )
        or (record.author in comparison_authors and record.author != TARGET_AUTHOR)
    ]
    selected.sort(key=lambda record: record.chunk_id)
    if {record.title for record in selected if record.author == TARGET_AUTHOR} != target_titles:
        raise ValueError("fresh v4 target qualification books are incomplete")
    if {record.author for record in selected if record.author != TARGET_AUTHOR} != comparison_authors:
        raise ValueError("fresh v4 comparison qualification authors are incomplete")
    return selected


def per_comparison_author(
    records: list[Any], scores: np.ndarray, threshold: float
) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if record.author != TARGET_AUTHOR:
            grouped[record.author].append(index)
    return [
        {
            "author": author,
            "books": len({records[index].title for index in indices}),
            "rows": len(indices),
            "specificity": float(np.mean(scores[indices] < threshold)),
            "false_positive_rate": float(np.mean(scores[indices] >= threshold)),
            "mean_score": float(np.mean(scores[indices])),
            "p95_score": float(np.quantile(scores[indices], 0.95)),
        }
        for author, indices in sorted(grouped.items())
    ]


def per_target_book(
    records: list[Any], scores: np.ndarray, threshold: float
) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if record.author == TARGET_AUTHOR:
            grouped[record.title].append(index)
    return [
        {
            "book": title,
            "rows": len(indices),
            "sensitivity": float(np.mean(scores[indices] >= threshold)),
            "mean_score": float(np.mean(scores[indices])),
        }
        for title, indices in sorted(grouped.items())
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_chart(path: Path, rows: list[dict[str, Any]]) -> None:
    width, height = 920, 360
    left, right, top, bottom = 80, 24, 40, 105
    plot_w, plot_h = width - left - right, height - top - bottom
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="25" font-family="Arial" font-size="16" fill="#111827">CR-FYSM-v4 fresh-holdout qualification</text>',
    ]
    for tick in range(0, 101, 20):
        y = top + plot_h - tick / 100 * plot_h
        pieces.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>'
        )
        pieces.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#4b5563">{tick}%</text>'
        )
    bar_w = plot_w / max(len(rows), 1) * 0.55
    for index, row in enumerate(rows):
        value = float(row["value"]) * 100.0
        center = left + (index + 0.5) * plot_w / len(rows)
        y = top + plot_h - value / 100 * plot_h
        color = "#0f766e" if value >= float(row["gate"]) * 100 else "#b45309"
        label = html.escape(str(row["label"]))
        pieces.append(
            f'<rect x="{center - bar_w / 2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{top + plot_h - y:.1f}" fill="{color}"/>'
        )
        pieces.append(
            f'<text x="{center:.1f}" y="{y - 7:.1f}" text-anchor="middle" font-family="Arial" font-size="12">{value:.1f}%</text>'
        )
        pieces.append(
            f'<text x="{center:.1f}" y="{top + plot_h + 20}" text-anchor="end" transform="rotate(-25 {center:.1f} {top + plot_h + 20})" font-family="Arial" font-size="11" fill="#374151">{label}</text>'
        )
    pieces.append("</svg>")
    path.write_text("\n".join(pieces), encoding="utf-8")


def main() -> None:
    args = parse_args()
    preregistration = load_preregistration(args.preregistration)
    paths = {key: Path(value) for key, value in preregistration["paths"].items()}
    artifact_paths = {
        key: Path(value)
        for key, value in preregistration["source_artifact_paths"].items()
    }
    output_dir = paths["output_dir"]
    result_path = output_dir / "results.v4.json"
    opened_path = output_dir / "qualification_opened.v4.json"
    if result_path.exists() or opened_path.exists():
        raise ValueError("CR-FYSM-v4 fresh qualification was already opened")
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = (
        json.dumps(
            {
                "meter_id": "CR-FYSM-v4",
                "lock_id": preregistration["lock_id"],
                "preregistration_sha256": sha256_file(args.preregistration),
                "state": "fresh_holdout_opened_irreversible",
            },
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")
        + b"\n"
    )
    descriptor = os.open(opened_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(marker)
        handle.flush()
        os.fsync(handle.fileno())

    model_contract = preregistration["model_contract"]
    family_order, family_weights = locked_family_contract(model_contract)
    boundary = int(model_contract["boundary_chunks_excluded"])
    masked_records = load_records(paths["active_masked_dataset"], boundary=boundary)
    fresh_records = qualification_records(masked_records, preregistration["partition"])
    artifacts = load_family_artifacts(paths["source_model_dir"], family_order)
    family_scores = family_probabilities(
        artifacts, [record.text for record in fresh_records]
    )
    raw_scores = ensemble(family_scores, family_order, family_weights)
    calibrator = joblib.load(artifact_paths["ensemble_platt_calibrator"])
    calibrated = calibrated_scores(calibrator, raw_scores)
    scores = raw_scores
    threshold = float(preregistration["decision_policy"]["threshold"])
    truth = np.asarray(
        [record.author == TARGET_AUTHOR for record in fresh_records], dtype=np.int8
    )
    weights = author_book_weights(fresh_records, binary_target=TARGET_AUTHOR)
    metrics = score_metrics(scores, truth, threshold, weights)
    ece = expected_calibration_error(calibrated, truth, weights)
    brier = float(np.average((calibrated - truth) ** 2, weights=weights))

    target_books = per_target_book(fresh_records, scores, threshold)
    comparison_authors = per_comparison_author(fresh_records, scores, threshold)
    clustered_lower = {
        "sensitivity": clustered_rate_lower_bound(
            fresh_records,
            scores,
            threshold,
            target_author=TARGET_AUTHOR,
            positive=True,
        ),
        "specificity": clustered_rate_lower_bound(
            fresh_records,
            scores,
            threshold,
            target_author=TARGET_AUTHOR,
            positive=False,
        ),
    }
    v3_partition = json.loads(paths["v3_partition"].read_text(encoding="utf-8"))
    development_records = calibration_records(masked_records, v3_partition)
    dialogue_strata = dialogue_stratum_false_positive_rates(
        development_records,
        fresh_records,
        scores,
        threshold,
        target_author=TARGET_AUTHOR,
    )

    clean_records = load_records(paths["active_clean_dataset"], boundary=boundary)
    fresh_clean = qualification_records(clean_records, preregistration["partition"])
    clean_by_id = {record.chunk_id: record.text for record in fresh_clean}
    clean_family_scores = family_probabilities(
        artifacts, [clean_by_id[record.chunk_id] for record in fresh_records]
    )
    clean_scores = ensemble(clean_family_scores, family_order, family_weights)
    clean_delta = np.abs(clean_scores - scores)
    invariance = {
        "median_abs_delta": float(np.median(clean_delta)),
        "p95_abs_delta": float(np.quantile(clean_delta, 0.95)),
        "threshold_flip_rate": float(
            np.mean((clean_scores >= threshold) != (scores >= threshold))
        ),
    }

    gates = preregistration["qualification_gates"]
    gate_results = {
        "balanced_accuracy": metrics["balanced_accuracy"]
        >= float(gates["balanced_accuracy_min"]),
        "sensitivity": metrics["sensitivity"] >= float(gates["sensitivity_min"]),
        "specificity": metrics["specificity"] >= float(gates["specificity_min"]),
        "clustered_sensitivity_lower": clustered_lower["sensitivity"]
        >= float(gates["clustered_sensitivity_lower_min"]),
        "clustered_specificity_lower": clustered_lower["specificity"]
        >= float(gates["clustered_specificity_lower_min"]),
        "target_book_floor": min(row["sensitivity"] for row in target_books)
        >= float(gates["target_book_sensitivity_floor"]),
        "dialogue_stratum_false_positive_ceiling": max(
            float(row["false_positive_rate"]) for row in dialogue_strata.values()
        )
        <= float(gates["dialogue_stratum_false_positive_max"]),
        "calibration_ece": ece <= float(gates["ece_max"]),
        "brier": brier <= float(gates["brier_max"]),
        "clean_masked_flips": invariance["threshold_flip_rate"]
        <= float(gates["clean_masked_threshold_flip_max"]),
        "target_book_count": len(target_books) == int(gates["target_book_count"]),
        "comparison_author_count": len(comparison_authors)
        == int(gates["comparison_author_count"]),
        "source_artifact_hashes": all(
            sha256_file(path) == preregistration["input_hashes"][f"artifact:{key}"]
            for key, path in artifact_paths.items()
        ),
    }
    statistical_pass = all(gate_results.values())
    result = {
        "schema_version": 4,
        "meter_id": "CR-FYSM-v4",
        "status": (
            "statistical_qualification_pass_pending_construct_gates"
            if statistical_pass
            else "statistical_qualification_fail"
        ),
        "research_role": "risk-controlled threshold revision of frozen CR-FYSM-v3",
        "source_meter": "CR-FYSM-v3",
        "source_v3_status": "statistical_qualification_fail",
        "preregistration": {
            "path": str(args.preregistration),
            "sha256": sha256_file(args.preregistration),
            "lock_id": preregistration["lock_id"],
        },
        "decision_policy": preregistration["decision_policy"],
        "partition": preregistration["partition"],
        "rows": len(fresh_records),
        "metrics": metrics,
        "probability_calibration_diagnostics": {
            "method": "frozen final v3 Platt calibrator",
            "ece": ece,
            "brier": brier,
            "decision_threshold_uses_calibrated_probabilities": False,
        },
        "clustered_95pct_lower": clustered_lower,
        "ece": ece,
        "brier": brier,
        "target_books": target_books,
        "comparison_authors": comparison_authors,
        "dialogue_strata": dialogue_strata,
        "clean_masked_invariance": invariance,
        "gate_results": gate_results,
        "pending_construct_gates": preregistration["pending_after_statistical_qualification"],
        "freshness_limit": (
            "Fresh relative to CR-FYSM-v3 only; older broad author-meter research had "
            "previously scored the active corpus."
        ),
        "source_artifact_hashes": {
            key: preregistration["input_hashes"][f"artifact:{key}"]
            for key in artifact_paths
        },
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(output_dir / "qualification_target_books.v4.csv", target_books)
    write_csv(output_dir / "qualification_comparison_authors.v4.csv", comparison_authors)
    chart_rows = [
        {"label": "Balanced accuracy", "value": metrics["balanced_accuracy"], "gate": gates["balanced_accuracy_min"]},
        {"label": "Sensitivity", "value": metrics["sensitivity"], "gate": gates["sensitivity_min"]},
        {"label": "Specificity", "value": metrics["specificity"], "gate": gates["specificity_min"]},
        {"label": "Cluster sensitivity lower", "value": clustered_lower["sensitivity"], "gate": gates["clustered_sensitivity_lower_min"]},
        {"label": "Cluster specificity lower", "value": clustered_lower["specificity"], "gate": gates["clustered_specificity_lower_min"]},
    ]
    make_chart(output_dir / "qualification_summary.v4.svg", chart_rows)

    report = f"""# CR-FYSM-v4 Fresh-Holdout Qualification

- Status: **{result['status']}**
- Method: frozen CR-FYSM-v3 ranking model with calibration-only risk threshold
- Threshold: **{threshold:.6f}**
- Freshness: relative to v3; not globally pristine from older exploratory benchmarks

![Qualification summary](qualification_summary.v4.svg)

## Result

| Metric | Result | Gate |
| --- | ---: | ---: |
| Balanced accuracy | {metrics['balanced_accuracy']:.1%} | >= {float(gates['balanced_accuracy_min']):.1%} |
| Sensitivity | {metrics['sensitivity']:.1%} | >= {float(gates['sensitivity_min']):.1%} |
| Specificity | {metrics['specificity']:.1%} | >= {float(gates['specificity_min']):.1%} |
| Clustered sensitivity lower | {clustered_lower['sensitivity']:.1%} | >= {float(gates['clustered_sensitivity_lower_min']):.1%} |
| Clustered specificity lower | {clustered_lower['specificity']:.1%} | >= {float(gates['clustered_specificity_lower_min']):.1%} |
| ECE | {ece:.4f} | <= {float(gates['ece_max']):.2f} |
| Brier score | {brier:.4f} | <= {float(gates['brier_max']):.2f} |

## Data

- Target: {len(target_books)} untouched v3 `proxy_transfer` books
- Comparison: {len(comparison_authors)} authors explicitly unused by v3
- Rows: {len(fresh_records):,}
- V3 qualification books/authors were excluded from v4 development and scoring

## Gate Status

""" + "\n".join(
        f"- {'PASS' if passed else 'FAIL'} `{name}`" for name, passed in gate_results.items()
    ) + "\n"
    (output_dir / "report.v4.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": result["status"], "metrics": metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
