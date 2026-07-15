#!/usr/bin/env python3
from __future__ import annotations

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
from experiments.iteration5.meter.build_cr_fysm_v4 import (
    make_chart,
    per_comparison_author,
    per_target_book,
    qualification_records,
    write_csv,
)
from experiments.iteration5.meter.develop_cr_fysm_v4 import calibration_records, load_family_artifacts
from experiments.iteration5.meter.preregister_cr_fysm_v4_postclean_v2 import (
    OUTPUT,
    ROOT,
    current_input_hashes,
    read_json,
    resolved_evaluation_partition,
)


def validate_lock_and_partition() -> tuple[dict[str, Any], dict[str, Path]]:
    payload = read_json(OUTPUT)
    if payload.get("status") != "locked_before_any_postclean_score":
        raise ValueError("post-clean replication v2 is not locked")
    if payload.get("lock_id") != canonical_lock_id(payload):
        raise ValueError("post-clean replication v2 lock mismatch")
    paths = {
        key: Path(value) for key, value in payload["paths"].items() if key != "output_dir"
    }
    artifacts = {
        key: Path(value)
        for key, value in payload["historical_source"]["source_artifact_paths"].items()
    }
    dependencies = {
        key: Path(value) for key, value in payload["dependency_paths"].items()
    }
    actual_hashes = current_input_hashes(paths, artifacts, dependencies)
    if actual_hashes != payload.get("input_hashes"):
        expected = payload.get("input_hashes", {})
        changed = sorted(
            key
            for key in set(actual_hashes) | set(expected)
            if actual_hashes.get(key) != expected.get(key)
        )
        raise ValueError(f"post-clean replication v2 inputs changed: {changed}")
    if payload.get("environment") != runtime_versions():
        raise ValueError("post-clean replication v2 environment changed")
    historical = read_json(paths["old_preregistration"])
    resolved = resolved_evaluation_partition(historical)
    expected_resolved = payload["geometry"]["resolved_evaluation_partition"]
    if resolved != expected_resolved:
        raise ValueError("resolved repaired evaluation partition changed after lock")
    return payload, paths


def per_comparison_book(
    records: list[Any], scores: np.ndarray, threshold: float
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if record.author != TARGET_AUTHOR:
            grouped[(record.author, record.title)].append(index)
    return [
        {
            "author": author,
            "book": title,
            "rows": len(indices),
            "specificity": float(np.mean(scores[indices] < threshold)),
            "false_positive_rate": float(np.mean(scores[indices] >= threshold)),
            "mean_score": float(np.mean(scores[indices])),
            "p95_score": float(np.quantile(scores[indices], 0.95)),
        }
        for (author, title), indices in sorted(grouped.items())
    ]


def main() -> None:
    lock, paths = validate_lock_and_partition()
    result_path = ROOT / "results.json"
    opened_path = ROOT / "replication_opened.json"
    if result_path.exists() or opened_path.exists():
        raise ValueError("post-clean replication v2 was already opened")

    marker = json.dumps(
        {
            "experiment_id": lock["experiment_id"],
            "lock_id": lock["lock_id"],
            "state": "opened_irreversible_after_resolved_partition_validation",
        },
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8") + b"\n"
    descriptor = os.open(opened_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(marker)
        handle.flush()
        os.fsync(handle.fileno())

    source = lock["historical_source"]
    contract = source["model_contract"]
    family_order, family_weights = locked_family_contract(contract)
    boundary = int(contract["boundary_chunks_excluded"])
    masked_records = load_records(paths["active_masked_dataset"], boundary=boundary)
    records = qualification_records(masked_records, source["partition"])
    model_dir = Path(source["source_artifact_paths"]["function_grammar_vectorizer"]).parent
    artifacts = load_family_artifacts(model_dir, family_order)
    family_scores = family_probabilities(artifacts, [record.text for record in records])
    scores = ensemble(family_scores, family_order, family_weights)
    calibrator = joblib.load(Path(source["source_artifact_paths"]["ensemble_platt_calibrator"]))
    calibrated = calibrated_scores(calibrator, scores)
    threshold = float(source["decision_policy"]["threshold"])
    truth = np.asarray([record.author == TARGET_AUTHOR for record in records], dtype=np.int8)
    weights = author_book_weights(records, binary_target=TARGET_AUTHOR)
    metrics = score_metrics(scores, truth, threshold, weights)
    ece = expected_calibration_error(calibrated, truth, weights)
    brier = float(np.average((calibrated - truth) ** 2, weights=weights))
    target_books = per_target_book(records, scores, threshold)
    comparison_authors = per_comparison_author(records, scores, threshold)
    comparison_books = per_comparison_book(records, scores, threshold)
    clustered = {
        "sensitivity": clustered_rate_lower_bound(
            records, scores, threshold, target_author=TARGET_AUTHOR, positive=True
        ),
        "specificity": clustered_rate_lower_bound(
            records, scores, threshold, target_author=TARGET_AUTHOR, positive=False
        ),
    }

    v3_partition = read_json(paths["old_v3_partition"])
    development = calibration_records(masked_records, v3_partition)
    dialogue = dialogue_stratum_false_positive_rates(
        development, records, scores, threshold, target_author=TARGET_AUTHOR
    )
    clean_records = load_records(paths["active_clean_dataset"], boundary=boundary)
    clean_qualification = qualification_records(clean_records, source["partition"])
    clean_by_id = {record.chunk_id: record.text for record in clean_qualification}
    if set(clean_by_id) != {record.chunk_id for record in records}:
        raise ValueError("clean and masked resolved qualification IDs differ")
    clean_scores = ensemble(
        family_probabilities(
            artifacts, [clean_by_id[record.chunk_id] for record in records]
        ),
        family_order,
        family_weights,
    )
    clean_delta = np.abs(clean_scores - scores)
    invariance = {
        "median_abs_delta": float(np.median(clean_delta)),
        "p95_abs_delta": float(np.quantile(clean_delta, 0.95)),
        "threshold_flip_rate": float(
            np.mean((clean_scores >= threshold) != (scores >= threshold))
        ),
    }

    gates = source["qualification_gates"]
    gate_results = {
        "balanced_accuracy": metrics["balanced_accuracy"]
        >= float(gates["balanced_accuracy_min"]),
        "sensitivity": metrics["sensitivity"] >= float(gates["sensitivity_min"]),
        "specificity": metrics["specificity"] >= float(gates["specificity_min"]),
        "clustered_sensitivity_lower": clustered["sensitivity"]
        >= float(gates["clustered_sensitivity_lower_min"]),
        "clustered_specificity_lower": clustered["specificity"]
        >= float(gates["clustered_specificity_lower_min"]),
        "target_book_floor": min(row["sensitivity"] for row in target_books)
        >= float(gates["target_book_sensitivity_floor"]),
        "dialogue_stratum_false_positive_ceiling": max(
            float(row["false_positive_rate"]) for row in dialogue.values()
        )
        <= float(gates["dialogue_stratum_false_positive_max"]),
        "calibration_ece": ece <= float(gates["ece_max"]),
        "brier": brier <= float(gates["brier_max"]),
        "clean_masked_flips": invariance["threshold_flip_rate"]
        <= float(gates["clean_masked_threshold_flip_max"]),
        "target_book_count": len(target_books) == int(gates["target_book_count"]),
        "comparison_author_count": len(comparison_authors)
        == int(gates["comparison_author_count"]),
        "resolved_partition": True,
        "source_artifact_hashes": True,
    }

    old = read_json(paths["old_results"])
    old_authors = {row["author"]: row for row in old["comparison_authors"]}
    author_deltas = [
        {
            **row,
            "old_specificity": old_authors[row["author"]]["specificity"],
            "specificity_delta": row["specificity"]
            - old_authors[row["author"]]["specificity"],
            "old_rows": old_authors[row["author"]]["rows"],
            "delta_interpretation": "composition_change_not_paired_effect",
        }
        for row in comparison_authors
    ]
    wangya_books = [row for row in comparison_books if row["author"] == "妄鸦"]
    if {row["book"] for row in wangya_books} != {"日抛型人设", "无限练习生", "悬疑片导演"}:
        raise ValueError("active 妄鸦 per-book diagnostics are incomplete")
    passed = all(gate_results.values())
    result = {
        "schema_version": 2,
        "experiment_id": lock["experiment_id"],
        "status": "replication_pass_pending_construct_gates" if passed else "replication_fail",
        "claim_limit": lock["claim_limit"],
        "preregistration": {
            "path": str(OUTPUT),
            "lock_id": lock["lock_id"],
            "sha256": sha256_file(OUTPUT),
        },
        "decision_policy": source["decision_policy"],
        "resolved_evaluation_partition": lock["geometry"]["resolved_evaluation_partition"],
        "rows": len(records),
        "metrics": metrics,
        "clustered_95pct_lower": clustered,
        "ece": ece,
        "brier": brier,
        "target_books": target_books,
        "comparison_authors": comparison_authors,
        "comparison_books": comparison_books,
        "wangya_books": wangya_books,
        "comparison_author_deltas": author_deltas,
        "dialogue_strata": dialogue,
        "clean_masked_invariance": invariance,
        "gate_results": gate_results,
        "historical_comparison": {
            "old_metrics": old["metrics"],
            "old_clustered_95pct_lower": old["clustered_95pct_lower"],
            "balanced_accuracy_delta": metrics["balanced_accuracy"]
            - old["metrics"]["balanced_accuracy"],
            "specificity_delta": metrics["specificity"]
            - old["metrics"]["specificity"],
            "interpretation": "resolved-composition sensitivity; not a paired causal repair estimate",
        },
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(ROOT / "comparison_authors.csv", author_deltas)
    write_csv(ROOT / "comparison_books.csv", comparison_books)
    write_csv(ROOT / "target_books.csv", target_books)
    make_chart(
        ROOT / "replication_summary.svg",
        [
            {
                "label": "Balanced accuracy",
                "value": metrics["balanced_accuracy"],
                "gate": gates["balanced_accuracy_min"],
            },
            {
                "label": "Sensitivity",
                "value": metrics["sensitivity"],
                "gate": gates["sensitivity_min"],
            },
            {
                "label": "Specificity",
                "value": metrics["specificity"],
                "gate": gates["specificity_min"],
            },
            {
                "label": "Cluster sensitivity lower",
                "value": clustered["sensitivity"],
                "gate": gates["clustered_sensitivity_lower_min"],
            },
            {
                "label": "Cluster specificity lower",
                "value": clustered["specificity"],
                "gate": gates["clustered_specificity_lower_min"],
            },
        ],
    )

    lines = [
        "# CR-FYSM-v4 Post-Cleaning Replication v2",
        "",
        f"- Status: **{result['status']}**",
        f"- Frozen threshold: `{threshold:.12f}`",
        f"- Balanced accuracy: {metrics['balanced_accuracy']:.3%}",
        f"- Sensitivity: {metrics['sensitivity']:.3%}",
        f"- Specificity: {metrics['specificity']:.3%}",
        f"- Clustered specificity lower bound: {clustered['specificity']:.3%}",
        f"- Claim limit: {lock['claim_limit']}",
        "",
        "![Replication summary](replication_summary.svg)",
        "",
        "## Comparison Authors",
        "",
        "| Author | Books | Rows | Specificity | Old specificity | Delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in author_deltas:
        lines.append(
            f"| {row['author']} | {row['books']} | {row['rows']} | {row['specificity']:.3%} | "
            f"{row['old_specificity']:.3%} | {row['specificity_delta']:+.3%} |"
        )
    lines.extend(
        [
            "",
            "Author deltas reflect changed book composition and are not paired causal repair effects.",
            "",
            "## 妄鸦 Books",
            "",
            "| Book | Rows | Specificity | False-positive rate | Mean score |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in wangya_books:
        lines.append(
            f"| {row['book']} | {row['rows']} | {row['specificity']:.3%} | "
            f"{row['false_positive_rate']:.3%} | {row['mean_score']:.5f} |"
        )
    lines.extend(["", "## Gates", ""])
    lines.extend(
        f"- {'PASS' if value else 'FAIL'}: `{name}`"
        for name, value in gate_results.items()
    )
    (ROOT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "metrics": metrics,
                "clustered": clustered,
                "wangya_books": wangya_books,
                "gate_results": gate_results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
