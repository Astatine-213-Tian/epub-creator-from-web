#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.validation.meter.benchmark_content_resistant_meter import (
    TARGET_AUTHOR,
    load_records,
)
from experiments.validation.meter.build_cr_fysm_v3 import (
    canonical_lock_id,
    runtime_versions,
    sha256_file,
)
from experiments.validation.meter.build_cr_fysm_v4 import CANONICAL_PREREGISTRATION


OUTPUT_DIR = CANONICAL_PREREGISTRATION.parent
V3_DIR = CANONICAL_PREREGISTRATION.parent.parent / "cr_fysm_v3"
DEVELOPMENT_PATH = OUTPUT_DIR / "development.v4.json"
PATHS = {
    "active_masked_dataset": Path("datasets/masked/chunks.entity_masked_v3.jsonl"),
    "active_clean_dataset": Path("datasets/unmasked/chunks.clean.jsonl"),
    "active_manifest": Path("datasets/dataset_manifest.json"),
    "active_mask_plan": Path("datasets/masked/mask_terms.json"),
    "v3_preregistration": V3_DIR / "preregistration.v3.json",
    "v3_results": V3_DIR / "results.json",
    "v3_partition": V3_DIR / "partition.v3.json",
    "v4_development": DEVELOPMENT_PATH,
    "v4_development_candidates": OUTPUT_DIR / "development_candidates.v4.csv",
    "source_model_dir": V3_DIR / "meter",
    "output_dir": OUTPUT_DIR,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lock CR-FYSM-v4 before scoring the v3-untouched holdout."
    )
    parser.add_argument("--output", type=Path, default=CANONICAL_PREREGISTRATION)
    return parser.parse_args()


def artifact_paths(model_dir: Path) -> dict[str, Path]:
    paths = {
        path.name.removesuffix(".joblib").replace(".", "_"): path
        for path in sorted(model_dir.glob("*.joblib"))
    }
    expected = {
        f"{family}_{suffix}"
        for family in (
            "function_grammar",
            "punctuation_dialogue",
            "sentence_rhythm",
            "paragraph_discourse",
        )
        for suffix in ("vectorizer", "scaler", "keep_mask", "model")
    } | {"ensemble_platt_calibrator"}
    if set(paths) != expected:
        raise ValueError(
            f"v3 source artifact set mismatch: missing={sorted(expected - set(paths))}, "
            f"extra={sorted(set(paths) - expected)}"
        )
    return paths


def fresh_partition(
    masked_path: Path,
    v3_partition: dict[str, Any],
    *,
    boundary: int,
) -> dict[str, Any]:
    records = load_records(masked_path, boundary=boundary)
    target_titles = sorted(
        {
            record.title
            for record in records
            if record.author == TARGET_AUTHOR and record.split == "proxy_transfer"
        }
    )
    comparison_authors = sorted(v3_partition["comparison"]["internal_unused"])
    if len(target_titles) != 4:
        raise ValueError(f"expected four proxy_transfer target books, found {target_titles}")
    if len(comparison_authors) != 10:
        raise ValueError("expected ten v3-unused comparison authors")

    prior_target = set().union(
        *(set(v3_partition["target"][role]) for role in ("fit", "calibration", "qualification"))
    )
    prior_comparison = set().union(
        *(
            set(v3_partition["comparison"][role])
            for role in ("fit", "calibration", "qualification")
        )
    )
    if set(target_titles) & prior_target:
        raise ValueError("fresh v4 target books overlap v3 roles")
    if set(comparison_authors) & prior_comparison:
        raise ValueError("fresh v4 comparison authors overlap v3 roles")

    selected = [
        record
        for record in records
        if (
            record.author == TARGET_AUTHOR
            and record.title in target_titles
            and record.split == "proxy_transfer"
        )
        or record.author in comparison_authors
    ]
    comparison_books = sorted(
        {(record.author, record.title) for record in selected if record.author != TARGET_AUTHOR}
    )
    if {record.author for record in selected if record.author != TARGET_AUTHOR} != set(
        comparison_authors
    ):
        raise ValueError("one or more fresh comparison authors have no retained rows")
    return {
        "target_books": target_titles,
        "comparison_authors": comparison_authors,
        "target_rows": sum(record.author == TARGET_AUTHOR for record in selected),
        "comparison_rows": sum(record.author != TARGET_AUTHOR for record in selected),
        "comparison_books": [
            {"author": author, "title": title} for author, title in comparison_books
        ],
        "forbidden_v3_target_qualification_books": v3_partition["target"][
            "qualification"
        ],
        "forbidden_v3_external_qualification_authors": v3_partition["comparison"][
            "qualification"
        ],
        "fresh_relative_to": "CR-FYSM-v3",
        "globally_pristine": False,
    }


def main() -> None:
    args = parse_args()
    if args.output.resolve() != CANONICAL_PREREGISTRATION.resolve():
        raise ValueError("CR-FYSM-v4 only permits the canonical lock path")
    if args.output.exists():
        raise ValueError("CR-FYSM-v4 preregistration already exists")
    if (OUTPUT_DIR / "qualification_opened.v4.json").exists():
        raise ValueError("CR-FYSM-v4 qualification was already opened")
    if (OUTPUT_DIR / "results.v4.json").exists():
        raise ValueError("CR-FYSM-v4 results already exist")
    for key, path in PATHS.items():
        if key in {"source_model_dir", "output_dir"}:
            if not path.is_dir():
                raise ValueError(f"missing v4 directory input: {key}={path}")
        elif not path.is_file():
            raise ValueError(f"missing v4 file input: {key}={path}")

    development = json.loads(DEVELOPMENT_PATH.read_text(encoding="utf-8"))
    if development.get("status") != "development_complete_holdout_unscored":
        raise ValueError("v4 development did not preserve an unscored holdout")
    if development["forbidden_inputs"].get("v4_fresh_holdout_scored") is not False:
        raise ValueError("v4 development freshness assertion is missing")
    selected_policy = development["selected"]
    if not selected_policy.get("selection_eligible"):
        raise ValueError("selected v4 development policy was not eligible")

    v3_preregistration = json.loads(
        PATHS["v3_preregistration"].read_text(encoding="utf-8")
    )
    v3_results = json.loads(PATHS["v3_results"].read_text(encoding="utf-8"))
    if v3_results.get("status") != "statistical_qualification_fail":
        raise ValueError("v4 requires the frozen v3 NO-GO")
    v3_partition = json.loads(PATHS["v3_partition"].read_text(encoding="utf-8"))
    boundary = int(v3_preregistration["model_contract"]["boundary_chunks_excluded"])
    partition = fresh_partition(
        PATHS["active_masked_dataset"], v3_partition, boundary=boundary
    )
    artifacts = artifact_paths(PATHS["source_model_dir"])

    hash_values = {
        f"path:{key}": sha256_file(path)
        for key, path in PATHS.items()
        if key not in {"source_model_dir", "output_dir"}
    }
    hash_values.update(
        {f"artifact:{key}": sha256_file(path) for key, path in artifacts.items()}
    )
    hash_values.update(
        {
            "script:builder": sha256_file(
                Path(__file__).with_name("build_cr_fysm_v4.py")
            ),
            "script:preregister": sha256_file(Path(__file__)),
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
    payload: dict[str, Any] = {
        "schema_version": 4,
        "meter_id": "CR-FYSM-v4",
        "status": "locked_before_fresh_holdout_scoring",
        "target_author": TARGET_AUTHOR,
        "purpose": (
            "One-shot test of a calibration-only risk-threshold revision using "
            "groups untouched by CR-FYSM-v3."
        ),
        "paths": {key: str(path) for key, path in PATHS.items()},
        "source_artifact_paths": {
            key: str(path) for key, path in artifacts.items()
        },
        "input_hashes": hash_values,
        "source_meter": {
            "meter_id": "CR-FYSM-v3",
            "lock_id": v3_preregistration["lock_id"],
            "status": v3_results["status"],
            "model_refit": False,
            "family_selection_changed": False,
            "family_weights_changed": False,
        },
        "model_contract": v3_preregistration["model_contract"],
        "decision_policy": {
            "method": "calibration-only group-robust specificity-margin selection",
            "score_space": "raw equal-weight family-probability ensemble",
            "candidate_specificity_floors": development[
                "candidate_specificity_floors"
            ],
            "selection_rule": development["selection_rule"],
            "selected_specificity_floor": selected_policy["specificity_floor"],
            "threshold": selected_policy["threshold"],
            "development_metrics": selected_policy,
            "v3_qualification_labels_used": False,
            "v4_holdout_labels_used": False,
            "platt_calibrator_role": "ECE and Brier diagnostics only",
        },
        "partition": partition,
        "qualification_gates": {
            "balanced_accuracy_min": 0.90,
            "sensitivity_min": 0.90,
            "specificity_min": 0.90,
            "clustered_sensitivity_lower_min": 0.80,
            "clustered_specificity_lower_min": 0.90,
            "target_book_sensitivity_floor": 0.75,
            "dialogue_stratum_false_positive_max": 0.20,
            "ece_max": 0.10,
            "brier_max": 0.10,
            "clean_masked_threshold_flip_max": 0.05,
            "target_book_count": 4,
            "comparison_author_count": 10,
        },
        "pending_after_statistical_qualification": [
            "fresh_generated_domain_calibration",
            "three_rater_blind_convergence",
            "matched_negative_and_factorial_content_style_challenge",
            "independent_exact_artifact_audit",
            "new_external_multi_book_author_confirmation",
        ],
        "freshness": {
            "fresh_relative_to_v3": True,
            "globally_pristine_from_older_author_benchmarks": False,
            "publication_grade_external_confirmation_still_required": True,
        },
        "literature_basis": development["literature_basis"],
        "environment": runtime_versions(),
        "one_shot_policy": {
            "canonical_path": str(CANONICAL_PREREGISTRATION),
            "opened_marker": str(OUTPUT_DIR / "qualification_opened.v4.json"),
            "result_path": str(OUTPUT_DIR / "results.v4.json"),
            "builder_writes_O_EXCL_marker_before_loading_holdout": True,
            "failure_requires_new_meter_version": True,
        },
        "run_command": "uv run python -m experiments.validation.meter.build_cr_fysm_v4",
    }
    payload["lock_id"] = canonical_lock_id(payload)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "lock_id": payload["lock_id"],
                "threshold": payload["decision_policy"]["threshold"],
                "partition": {
                    "target_books": len(partition["target_books"]),
                    "comparison_authors": len(partition["comparison_authors"]),
                    "rows": partition["target_rows"] + partition["comparison_rows"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
