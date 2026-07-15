#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.iteration5.meter.benchmark_content_resistant_meter import TARGET_AUTHOR, read_jsonl
from experiments.iteration5.meter.build_cr_fysm_v2 import FAMILY_ORDER, SEED, stable_order


DEFAULT_ROOT = Path(
    "generated/style_research/style_transfer_experiments/iterations/"
    "content_resistant_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prospectively lock CR-FYSM-v2 before fitting or qualification."
    )
    parser.add_argument(
        "--masked-dataset",
        type=Path,
        default=Path("datasets/masked/chunks.entity_masked_v3.jsonl"),
    )
    parser.add_argument(
        "--clean-dataset",
        type=Path,
        default=Path("datasets/unmasked/chunks.clean.jsonl"),
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=Path("datasets/dataset_manifest.json"),
    )
    parser.add_argument(
        "--mask-plan",
        type=Path,
        default=Path("datasets/masked/mask_terms.json"),
    )
    parser.add_argument(
        "--v1-partition",
        type=Path,
        default=DEFAULT_ROOT / "cr_fysm_v1/partition.v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ROOT / "cr_fysm_v2/preregistration.v2.json",
    )
    parser.add_argument("--target-author", default=TARGET_AUTHOR)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def corpus_metadata(path: Path, target_author: str) -> tuple[dict[str, str], set[str]]:
    target_areas: dict[str, str] = {}
    comparison_authors: set[str] = set()
    for row in read_jsonl(path):
        if str(row["split"]) != "train":
            continue
        author = str(row["author"])
        if author == target_author:
            target_areas[str(row["title"])] = str(row.get("time_area") or "unknown")
        else:
            comparison_authors.add(author)
    return target_areas, comparison_authors


def rotated_partition(
    masked_dataset: Path,
    v1_partition_path: Path,
    target_author: str,
) -> dict[str, Any]:
    target_areas, comparison_authors = corpus_metadata(masked_dataset, target_author)
    if len(target_areas) != 28:
        raise ValueError(f"expected 28 corrected target train books, found {len(target_areas)}")
    if len(comparison_authors) != 49:
        raise ValueError(f"expected 49 comparison authors, found {len(comparison_authors)}")
    if "已枯之色" in target_areas:
        raise ValueError("quarantined 已枯之色 remains in the active target corpus")

    v1 = json.loads(v1_partition_path.read_text(encoding="utf-8"))
    old_target_fit = set(v1["target"]["fit"]) & set(target_areas)
    old_comparison_fit = set(v1["comparison"]["fit"]) & comparison_authors

    by_area: dict[str, list[str]] = defaultdict(list)
    for title in old_target_fit:
        by_area[target_areas[title]].append(title)
    calibration_titles: list[str] = []
    qualification_titles: list[str] = []
    for area in sorted(set(target_areas.values())):
        ordered = stable_order(
            by_area.get(area, []), salt=f"CR-FYSM-v2-rotated-target:{area}"
        )
        if len(ordered) < 2:
            raise ValueError(
                f"v1 fit role has fewer than two fresh v2 holdout candidates for {area}"
            )
        calibration_titles.append(ordered[0])
        qualification_titles.append(ordered[1])
    target_fit = sorted(
        set(target_areas) - set(calibration_titles) - set(qualification_titles)
    )
    if (len(target_fit), len(calibration_titles), len(qualification_titles)) != (20, 4, 4):
        raise ValueError("corrected target partition is not 20/4/4")

    ordered_old_fit_authors = stable_order(
        sorted(old_comparison_fit), salt="CR-FYSM-v2-rotated-comparison"
    )
    if len(ordered_old_fit_authors) != 29:
        raise ValueError("v1 comparison fit role is not available for rotation")
    calibration_authors = ordered_old_fit_authors[:10]
    qualification_authors = ordered_old_fit_authors[10:20]
    fit_authors = sorted(
        comparison_authors - set(calibration_authors) - set(qualification_authors)
    )
    if (len(fit_authors), len(calibration_authors), len(qualification_authors)) != (
        29,
        10,
        10,
    ):
        raise ValueError("rotated comparison partition is not 29/10/10")

    return {
        "schema_version": 2,
        "seed": SEED,
        "target_author": target_author,
        "target": {
            "fit": target_fit,
            "calibration": sorted(calibration_titles),
            "qualification": sorted(qualification_titles),
            "time_area": dict(sorted(target_areas.items())),
        },
        "comparison": {
            "fit": fit_authors,
            "calibration": sorted(calibration_authors),
            "qualification": sorted(qualification_authors),
        },
        "policy": {
            "target": (
                "v2 calibration and qualification each take one book per time-area "
                "from the former v1 fit role; all other corrected target books fit v2"
            ),
            "comparison": (
                "v2 calibration and qualification each take ten authors from the former "
                "v1 fit role; the remaining 29 authors fit v2"
            ),
            "boundary_chunks_excluded": 2,
            "qualification_was_not_individually_scored_in_v1": True,
        },
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise ValueError(f"preregistration already exists: {args.output}")
    result_path = args.output.parent / "results.json"
    if result_path.exists():
        raise ValueError("cannot preregister after a v2 result exists")

    build_script = Path(__file__).with_name("build_cr_fysm_v2.py")
    benchmark_script = Path(__file__).with_name("benchmark_content_resistant_meter.py")
    shared_feature_script = Path("workflows/benchmark_author_style.py")
    partition = rotated_partition(
        args.masked_dataset, args.v1_partition, args.target_author
    )
    payload: dict[str, Any] = {
        "schema_version": 2,
        "meter_id": "CR-FYSM-v2",
        "status": "locked_before_model_fit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_author": args.target_author,
        "purpose": (
            "Qualify a target-specific content-resistant style meter before any "
            "Iteration 5 style-transfer candidate generation."
        ),
        "v1_disposition": {
            "status": "invalidated_by_confirmed_target_label_error",
            "misattributed_title": "已枯之色",
            "former_manifest_author": "非天夜翔",
            "verified_author": "莫寻秋野",
            "verification_url": "https://www.jjwxc.net/onebook.php?novelid=4887203",
            "v1_outcomes_are_not_used_to_select_v2_qualification_members": True,
        },
        "partition": partition,
        "feature_contract": {
            "families": list(FAMILY_ORDER),
            "weights": {name: 0.25 for name in FAMILY_ORDER},
            "forbidden": [
                "raw lexical ngrams",
                "proper names",
                "lore or topic terms",
                "mask-token identity/count features",
                "Iteration 4 or Iteration 5 generated outputs as fitting data",
            ],
            "calibration": "Platt scaling on calibration books/authors only",
            "sparse_feature_dispersion": (
                "present in at least 80% of target fit books and at least 50% of "
                "comparison fit authors; aggregate continuous features are retained"
            ),
            "threshold": (
                "lowest calibrated score among solutions meeting at least 80% "
                "sensitivity and 90% specificity, tie-broken by balanced accuracy"
            ),
        },
        "qualification_gates": {
            "feature_schema_unregistered_feature_count_max": 0,
            "calibration_sensitivity_min": 0.80,
            "calibration_specificity_min": 0.90,
            "calibration_ece_max": 0.10,
            "qualification_sensitivity_min": 0.80,
            "qualification_specificity_min": 0.90,
            "clustered_sensitivity_lower_min": 0.70,
            "clustered_specificity_lower_min": 0.70,
            "target_book_sensitivity_floor": 0.60,
            "dialogue_stratum_false_positive_max": 0.20,
            "content_counterfactual_median_abs_delta_max": 0.05,
            "content_counterfactual_p95_abs_delta_max": 0.15,
            "content_counterfactual_threshold_flip_max": 0.05,
            "clean_masked_threshold_flip_max": 0.05,
            "legacy_original_over_neutral_min": 0.80,
            "legacy_three_of_four_family_positive_min": 0.80,
            "single_family_variance_share_max": 0.50,
        },
        "pending_after_statistical_qualification": [
            "fresh_generated_domain_calibration",
            "three_rater_blind_human_convergence",
            "matched_negative_and_factorial_content_style_challenge",
            "independent_exact_artifact_audit",
        ],
        "input_hashes": {
            "preregistration_script_sha256": sha256_file(Path(__file__)),
            "masked_dataset_sha256": sha256_file(args.masked_dataset),
            "clean_dataset_sha256": sha256_file(args.clean_dataset),
            "dataset_manifest_sha256": sha256_file(args.dataset_manifest),
            "mask_plan_sha256": sha256_file(args.mask_plan),
            "v1_partition_sha256": sha256_file(args.v1_partition),
            "build_script_sha256": sha256_file(build_script),
            "benchmark_script_sha256": sha256_file(benchmark_script),
            "shared_feature_script_sha256": sha256_file(shared_feature_script),
        },
        "one_shot_policy": {
            "result_path": str(result_path),
            "result_path_absent_at_lock": not result_path.exists(),
            "do_not_refit_or_repartition_after_opening_qualification": True,
            "failed_v2_is_preserved_and_any_revision_requires_v3": True,
        },
        "run_command": "uv run python -m experiments.iteration5.meter.build_cr_fysm_v2",
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    payload["lock_id"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "lock_id": payload["lock_id"],
                "output": str(args.output),
                "target_roles": {
                    role: len(partition["target"][role])
                    for role in ("fit", "calibration", "qualification")
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
