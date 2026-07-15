#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from experiments.iteration5.meter.benchmark_content_resistant_meter import (
    TARGET_AUTHOR,
    load_records,
    read_jsonl,
)
from experiments.iteration5.meter.build_cr_fysm_v3 import (
    CANONICAL_PREREGISTRATION,
    FAMILY_ORDER,
    SEED,
    canonical_lock_id,
    runtime_versions,
    sha256_file,
    stable_order,
)


OUTPUT_DIR = CANONICAL_PREREGISTRATION.parent
PATHS = {
    "active_masked_dataset": Path("datasets/masked/chunks.entity_masked_v3.jsonl"),
    "active_clean_dataset": Path("datasets/unmasked/chunks.clean.jsonl"),
    "active_manifest": Path("datasets/dataset_manifest.json"),
    "active_mask_plan": Path("datasets/masked/mask_terms.json"),
    "external_masked_dataset": Path(
        "generated/style_research/external_author_challenge_v1/"
        "masked/chunks.entity_masked_v3.jsonl"
    ),
    "external_clean_dataset": Path(
        "generated/style_research/external_author_challenge_v1/"
        "unmasked/chunks.clean.jsonl"
    ),
    "external_manifest": Path(
        "generated/style_research/external_author_challenge_v1/dataset_manifest.json"
    ),
    "external_mask_plan": Path(
        "generated/style_research/external_author_challenge_v1/masked/mask_terms.json"
    ),
    "external_audit": Path(
        "generated/style_research/external_author_challenge_v1/audit.md"
    ),
    "quarantine_manifest": Path(
        "datasets/raw/_quarantine/authorship_mismatch/quarantine_manifest.json"
    ),
    "prior_exposed_clean_text": Path(
        "generated/style_research/quarantine/authorship_mismatch/已枯之色.clean.txt"
    ),
    "preregistration_script": Path("experiments/iteration5/meter/preregister_cr_fysm_v3.py"),
    "output_dir": OUTPUT_DIR,
}

CONTENT_FINGERPRINT_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prospectively lock the one-shot CR-FYSM-v3 qualification."
    )
    parser.add_argument("--output", type=Path, default=CANONICAL_PREREGISTRATION)
    return parser.parse_args()


def content_fingerprint(text: str) -> str:
    normalized = "".join(CONTENT_FINGERPRINT_RE.findall(text))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def external_prior_exposure_violations(
    external_clean_path: Path,
    quarantine_manifest_path: Path,
    prior_exposed_clean_paths: list[Path],
) -> list[dict[str, str]]:
    quarantine_rows = json.loads(quarantine_manifest_path.read_text(encoding="utf-8"))
    rejected_titles = {str(row["title"]) for row in quarantine_rows}
    grouped: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for row in read_jsonl(external_clean_path):
        key = (str(row["author"]), str(row["title"]))
        grouped.setdefault(key, []).append((int(row["chunk_index"]), str(row["text"])))
    prior_hashes = {
        content_fingerprint(path.read_text(encoding="utf-8")): str(path)
        for path in prior_exposed_clean_paths
    }
    violations: list[dict[str, str]] = []
    for (author, title), chunks in sorted(grouped.items()):
        if title in rejected_titles:
            violations.append(
                {
                    "author": author,
                    "title": title,
                    "reason": "quarantined_title_was_observed_before_v3",
                }
            )
        joined = "\n".join(text for _, text in sorted(chunks))
        fingerprint = content_fingerprint(joined)
        if fingerprint in prior_hashes:
            violations.append(
                {
                    "author": author,
                    "title": title,
                    "reason": "content_fingerprint_matches_prior_exposed_text",
                    "prior_path": prior_hashes[fingerprint],
                }
            )
    return violations


def active_partition(
    masked_path: Path,
    external_path: Path,
    *,
    external_clean_path: Path,
    quarantine_manifest_path: Path,
    prior_exposed_clean_paths: list[Path],
) -> dict[str, Any]:
    target: dict[str, set[str]] = {
        "fit": set(),
        "calibration": set(),
        "qualification": set(),
    }
    active_comparison_authors: set[str] = set()
    comparison_splits: dict[str, set[str]] = {}
    for row in read_jsonl(masked_path):
        split = str(row["split"])
        author = str(row["author"])
        if author == TARGET_AUTHOR:
            role = {
                "train": "fit",
                "dev": "calibration",
                "test": "qualification",
            }.get(split)
            if role:
                target[role].add(str(row["title"]))
        else:
            comparison_splits.setdefault(author, set()).add(split)
            if split == "train":
                active_comparison_authors.add(author)

    if {role: len(values) for role, values in target.items()} != {
        "fit": 28,
        "calibration": 4,
        "qualification": 4,
    }:
        raise ValueError("target train/dev/test roles are not 28/4/4")
    if "已枯之色" in set().union(*target.values()):
        raise ValueError("quarantined title remains in active target roles")
    if len(active_comparison_authors) != 49:
        raise ValueError("expected 49 active comparison authors")
    incomplete_authors = {
        author: sorted(comparison_splits.get(author, set()))
        for author in active_comparison_authors
        if not {"train", "dev", "test"}.issubset(comparison_splits.get(author, set()))
    }
    if incomplete_authors:
        raise ValueError(f"comparison authors lack train/dev/test books: {incomplete_authors}")

    ordered = stable_order(
        sorted(active_comparison_authors), salt="CR-FYSM-v3-active-comparison-role"
    )
    fit_authors = ordered[:29]
    calibration_authors = ordered[29:39]
    internal_unused_authors = ordered[39:49]

    external_records = load_records(external_path, boundary=2)
    external_counts: dict[str, int] = {}
    for record in external_records:
        external_counts[record.author] = external_counts.get(record.author, 0) + 1
    external_authors = sorted(external_counts)
    if len(external_authors) != 10:
        raise ValueError(f"expected 10 external authors, found {len(external_authors)}")
    if set(external_authors) & active_comparison_authors:
        raise ValueError("external qualification authors overlap the active corpus")
    if min(external_counts.values()) < 3:
        raise ValueError("an external qualification author has fewer than three retained chunks")
    prior_exposure = external_prior_exposure_violations(
        external_clean_path,
        quarantine_manifest_path,
        prior_exposed_clean_paths,
    )
    if prior_exposure:
        raise ValueError(f"external qualification contains prior-exposed text: {prior_exposure}")

    return {
        "schema_version": 3,
        "seed": SEED,
        "target_author": TARGET_AUTHOR,
        "target": {role: sorted(values) for role, values in target.items()},
        "comparison": {
            "fit": sorted(fit_authors),
            "calibration": sorted(calibration_authors),
            "qualification": external_authors,
            "internal_unused": sorted(internal_unused_authors),
            "external_retained_rows": dict(sorted(external_counts.items())),
        },
        "policy": {
            "target_fit_source_split": "train",
            "target_calibration_source_split": "dev",
            "target_qualification_source_split": "test",
            "comparison_fit_source_split": "train",
            "comparison_calibration_source_split": "dev",
            "comparison_qualification_source": "external authors absent from active corpus",
            "target_proxy_transfer_books_untouched": True,
            "active_internal_unused_authors_untouched": True,
        },
    }


def input_hashes(paths: dict[str, Path]) -> dict[str, str]:
    hashes = {
        f"{key}_sha256": sha256_file(path)
        for key, path in paths.items()
        if key != "output_dir"
    }
    build_script = Path(__file__).with_name("build_cr_fysm_v3.py")
    hashes.update(
        {
            "build_script_sha256": sha256_file(build_script),
            "benchmark_script_sha256": sha256_file(
                Path(__file__).with_name("benchmark_content_resistant_meter.py")
            ),
            "shared_feature_script_sha256": sha256_file(
                Path("workflows/benchmark_author_style.py")
            ),
            "audit_script_sha256": sha256_file(
                Path("workflows/audit_style_dataset.py")
            ),
            "uv_lock_sha256": sha256_file(Path("uv.lock")),
        }
    )
    return hashes


def main() -> None:
    args = parse_args()
    if args.output.resolve() != CANONICAL_PREREGISTRATION.resolve():
        raise ValueError("CR-FYSM-v3 only permits the canonical lock path")
    if args.output.exists():
        raise ValueError(f"preregistration already exists: {args.output}")
    if (OUTPUT_DIR / "qualification_opened.v3.json").exists():
        raise ValueError("qualification was already opened")
    if (OUTPUT_DIR / "results.json").exists():
        raise ValueError("qualification result already exists")
    for key, path in PATHS.items():
        if key != "output_dir" and not path.is_file():
            raise ValueError(f"missing preregistration input: {key}={path}")

    partition = active_partition(
        PATHS["active_masked_dataset"],
        PATHS["external_masked_dataset"],
        external_clean_path=PATHS["external_clean_dataset"],
        quarantine_manifest_path=PATHS["quarantine_manifest"],
        prior_exposed_clean_paths=[PATHS["prior_exposed_clean_text"]],
    )
    payload: dict[str, Any] = {
        "schema_version": 3,
        "meter_id": "CR-FYSM-v3",
        "status": "locked_before_model_fit",
        "target_author": TARGET_AUTHOR,
        "purpose": (
            "One-shot qualification of a content-resistant target-style meter before "
            "Iteration 5 style-transfer generation."
        ),
        "prior_attempts": {
            "CR-FYSM-v1": "invalidated by confirmed target-book misattribution",
            "CR-FYSM-v2": "rejected before lock by independent methodology audit; never fit",
        },
        "freshness_scope": {
            "target_qualification": (
                "active test books never used by CR-FYSM-v1/v2 fitting, calibration, "
                "or qualification; they did appear in older general author-ID research"
            ),
            "comparison_qualification": (
                "ten authors absent from the active 50-author corpus and all CR-FYSM-v1/v2 roles"
            ),
        },
        "paths": {key: str(value) for key, value in PATHS.items()},
        "partition": partition,
        "model_contract": {
            "family_order": list(FAMILY_ORDER),
            "family_weights": {name: 0.25 for name in FAMILY_ORDER},
            "boundary_chunks_excluded": 2,
            "min_target_book_dispersion": 0.80,
            "min_comparison_author_dispersion": 0.50,
            "sparse_dispersion_operator": "AND",
            "classifier": "L2 logistic regression, C=1.0, liblinear, class/group weighted",
            "calibration": (
                "four-fold target-book/comparison-author group-cross-fitted Platt scores "
                "for ECE and threshold; final Platt fit on all calibration groups for qualification"
            ),
            "threshold": "lowest score meeting locked sensitivity and specificity minima",
            "function_channel": (
                "strict closed-class characters and curated multi-character function words; "
                "broad content-bearing single characters forbidden"
            ),
            "known_iteration4_pairs": "not loaded and not a gate",
        },
        "qualification_gates": {
            "calibration_sensitivity_min": 0.80,
            "calibration_specificity_min": 0.90,
            "calibration_ece_max": 0.10,
            "qualification_sensitivity_min": 0.80,
            "qualification_specificity_min": 0.90,
            "clustered_sensitivity_lower_min": 0.70,
            "clustered_specificity_lower_min": 0.70,
            "target_book_sensitivity_floor": 0.60,
            "dialogue_stratum_false_positive_max": 0.20,
            "clean_masked_threshold_flip_max": 0.05,
            "single_family_variance_share_max": 0.50,
            "external_qualification_author_count": 10,
            "unregistered_feature_count_max": 0,
        },
        "pending_after_statistical_qualification": [
            "fresh_generated_domain_calibration",
            "three_rater_blind_convergence",
            "matched_negative_and_factorial_content_style_challenge",
            "independent_exact_artifact_audit",
        ],
        "pending_protocols": {
            "generated_domain": (
                "balance human/generated status across positive and negative labels; "
                "require sensitivity >=0.80, specificity >=0.90, ECE <=0.10"
            ),
            "three_rater": (
                "three independently blinded expert LLM raters; Krippendorff alpha >=0.67; "
                "meter/rating Spearman rho >=0.50 with lower CI >=0.30; do not label as human"
            ),
            "factorial": (
                "at least 200 semantically validated matched items; style main effect positive, "
                "cluster CI excludes zero, effect >=3x absolute content effect, no material interaction"
            ),
            "artifact_audit": (
                "independent agent recomputes hashes, partitions, scores, gates, and leakage checks"
            ),
        },
        "input_hashes": input_hashes(PATHS),
        "environment": runtime_versions(),
        "one_shot_policy": {
            "canonical_preregistration": str(CANONICAL_PREREGISTRATION),
            "opened_marker": str(OUTPUT_DIR / "qualification_opened.v3.json"),
            "result_path": str(OUTPUT_DIR / "results.json"),
            "builder_rejects_noncanonical_lock_path": True,
            "builder_creates_opened_marker_with_O_EXCL_before_loading_qualification": True,
            "failure_or_crash_requires_a_new_meter_version": True,
        },
        "run_command": "uv run python -m experiments.iteration5.meter.build_cr_fysm_v3",
    }
    payload["lock_id"] = canonical_lock_id(payload)
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
                "comparison_roles": {
                    role: len(partition["comparison"][role])
                    for role in ("fit", "calibration", "qualification")
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
