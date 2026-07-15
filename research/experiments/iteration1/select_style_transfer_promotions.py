#!/usr/bin/env python3
from __future__ import annotations

"""Apply the preregistered style-transfer screening promotion rule."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.iteration1.style_analysis_lock import (
    require_matching_analysis_binding,
    validate_analysis_lock,
)
from experiments.iteration1.style_experiment_decisions import recompute_promotion


from experiments.shared.paths import RESEARCH_ROOT


REPO_ROOT = RESEARCH_ROOT
DEFAULT_EXPERIMENT_ROOT = (
    REPO_ROOT / "generated/style_research/style_transfer_experiments"
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze at most three methods using the screening promotion rule."
    )
    parser.add_argument("--screening-evaluation", type=Path, required=True)
    parser.add_argument("--analysis-lock", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--sample-set", default="development_proxy_v1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--provisional",
        action="store_true",
        help="Freeze a deterministic shortlist before blind screening critiques.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_root = args.experiment_root.resolve()
    analysis_lock_binding = validate_analysis_lock(
        args.analysis_lock.resolve(), experiment_root
    )
    evaluation_path = args.screening_evaluation.resolve()
    evaluation = load_json(evaluation_path)
    require_matching_analysis_binding(
        evaluation, analysis_lock_binding, label="screening evaluation"
    )
    sample_contract = evaluation.get("sample_contract", {})
    if sample_contract.get("selection_id") != "screening_v1":
        raise ValueError("Promotion requires an official screening_v1 evaluation")
    screening_path = (
        experiment_root / "sample_sets" / f"{args.sample_set}.screening_v1_ids.json"
    )
    if sample_contract.get("selection_sha256") != file_sha256(screening_path):
        raise ValueError("Screening evaluation is not bound to the frozen cohort")
    threshold = evaluation.get("threshold", {})
    if threshold.get("status") != "valid" or not threshold.get("sha256"):
        raise ValueError("Promotion requires the frozen valid style threshold")
    registry = evaluation.get("method_registry", {})
    registry_path = experiment_root / "method_registry/style_methods.v1.json"
    if registry.get("sha256") != file_sha256(registry_path):
        raise ValueError("Screening evaluation method registry binding is stale")
    combinations = evaluation.get("combinations")
    if not isinstance(combinations, list) or not combinations:
        raise ValueError("Screening evaluation contains no method combinations")
    roster = evaluation.get("screening_roster", {})
    expected_phase = "initial" if args.provisional else "refinement"
    if roster.get("status") != "valid" or roster.get("phase") != expected_phase:
        raise ValueError(
            f"Promotion requires a valid {expected_phase} screening roster binding"
        )

    if not args.provisional and evaluation.get("independent_judgments", {}).get(
        "status"
    ) != "loaded":
        raise ValueError("Final promotion requires frozen blind screening judgments")
    assessed, promoted = recompute_promotion(
        evaluation, require_judgments=not args.provisional
    )

    output = args.output or (
        experiment_root
        / "promotions"
        / (
            "screening_v1.provisional_shortlist.v1.json"
            if args.provisional
            else "screening_v1.promoted_methods.v1.json"
        )
    )
    report = args.report or output.with_suffix(".md")
    output = output.resolve()
    report = report.resolve()
    artifact = {
        "schema_version": 1,
        "promotion_id": (
            "screening_v1_deterministic_shortlist.v1"
            if args.provisional
            else "screening_v1_preregistered_judged_rule.v1"
        ),
        "sample_set": args.sample_set,
        "analysis_lock": analysis_lock_binding,
        "selection_id": "screening_v1",
        "screening_selection_path": display_path(screening_path),
        "screening_selection_sha256": file_sha256(screening_path),
        "screening_evaluation_path": display_path(evaluation_path),
        "screening_evaluation_sha256": file_sha256(evaluation_path),
        "threshold_sha256": threshold["sha256"],
        "scorer_binding": evaluation.get("scorer_binding"),
        "screening_roster": roster,
        "method_registry_path": display_path(registry_path),
        "method_registry_sha256": file_sha256(registry_path),
        "rule": {
            "maximum_promoted_methods": 3,
            "positive_paired_lift_required_in_each_arm": True,
            "maximum_hard_fidelity_failure_rate_each_arm": 0.10,
            "reference_copy_failures_allowed": 0,
            "tie_break": [
                "own_author_mean_paired_margin_lift_desc",
                "cross_author_mean_paired_margin_lift_desc",
                "method_id_asc",
                "intensity_asc",
            ],
            "maximum_one_intensity_per_method": True,
            "blind_judgments_required": not args.provisional,
            "semantic_noninferiority_required_each_arm": not args.provisional,
            "high_severity_readability_failures_allowed": 0
            if not args.provisional
            else None,
        },
        "promoted": [
            {
                "rank": index + 1,
                "method_id": row["method_id"],
                "intensity": row["intensity"],
                "own_author_mean_paired_margin_lift": row[
                    "own_author_reconstruction"
                ]["mean_paired_margin_lift"],
                "cross_author_mean_paired_margin_lift": row[
                    "cross_author_transfer"
                ]["mean_paired_margin_lift"],
            }
            for index, row in enumerate(promoted)
        ],
        "all_candidates": assessed,
        "status": (
            "provisional_shortlist_frozen"
            if args.provisional and promoted
            else "promotions_frozen"
            if promoted
            else "no_method_qualified"
        ),
    }
    content = json.dumps(artifact, ensure_ascii=False, indent=2) + "\n"
    if output.exists():
        if output.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Refusing to replace frozen promotion artifact: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
    lines = [
        "# Style-Transfer Screening Promotion",
        "",
        f"- Status: `{artifact['status']}`",
        f"- Screening evaluation SHA-256: `{artifact['screening_evaluation_sha256']}`",
        f"- Frozen threshold SHA-256: `{artifact['threshold_sha256']}`",
        f"- Pre-style analysis lock SHA-256: `{analysis_lock_binding['sha256']}`",
        "",
        "| Rank | Method | Intensity | Own lift | Cross lift |",
        "| ---: | --- | --- | ---: | ---: |",
    ]
    for row in artifact["promoted"]:
        lines.append(
            f"| {row['rank']} | `{row['method_id']}` | `{row['intensity']}` | "
            f"{row['own_author_mean_paired_margin_lift']:.4f} | "
            f"{row['cross_author_mean_paired_margin_lift']:.4f} |"
        )
    if not artifact["promoted"]:
        lines.append("| - | No method qualified | - | - | - |")
    lines.extend(("", "The full candidate decisions and exclusion reasons are in the JSON artifact.", ""))
    report_content = "\n".join(lines)
    if report.exists() and report.read_text(encoding="utf-8") != report_content:
        raise FileExistsError(f"Refusing to replace frozen promotion report: {report}")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(report_content, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "promoted_count": len(artifact["promoted"]),
                "output": display_path(output),
                "output_sha256": file_sha256(output),
                "report": display_path(report),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
