from __future__ import annotations

"""Pure deterministic promotion and winner decisions for the style study."""

from typing import Any, Mapping, Sequence


def require_complete_evaluation(
    evaluation: Mapping[str, Any], *, expected_selection_id: str
) -> None:
    """Reject outcome-dependent attrition before any decision is computed."""

    sample_contract = evaluation.get("sample_contract")
    if not isinstance(sample_contract, Mapping):
        raise ValueError("Evaluation has no sample contract")
    if sample_contract.get("selection_id") != expected_selection_id:
        raise ValueError(
            f"Decision requires {expected_selection_id}, got "
            f"{sample_contract.get('selection_id')!r}"
        )
    expected_total = int(sample_contract.get("method_evaluation_rows", 0))
    if expected_total <= 0:
        raise ValueError("Evaluation has no positive expected row count")
    if evaluation.get("status") != "complete":
        raise ValueError("Evaluation is incomplete or contains method failures")
    combinations = evaluation.get("combinations")
    if not isinstance(combinations, list) or not combinations:
        raise ValueError("Evaluation contains no method combinations")
    for row in combinations:
        if not isinstance(row, Mapping):
            raise ValueError("Evaluation contains an invalid method combination")
        key = (str(row.get("method_id", "")), str(row.get("intensity", "")))
        if not all(key):
            raise ValueError("Evaluation contains an unidentified method combination")
        if row.get("status") != "complete":
            raise ValueError(f"Incomplete registered combination: {key}")
        arm_summaries = row.get("arm_summaries")
        if not isinstance(arm_summaries, Mapping) or not arm_summaries:
            raise ValueError(f"Combination has no arm summaries: {key}")
        accounted = 0
        for arm, summary in arm_summaries.items():
            if not isinstance(summary, Mapping):
                raise ValueError(f"Invalid arm summary for {key}: {arm}")
            expected = int(summary.get("expected_rows", -1))
            scored = int(summary.get("scored_rows", -1))
            failed = int(summary.get("failed_or_missing_rows", -1))
            if expected < 0 or scored != expected or failed != 0:
                raise ValueError(
                    f"Incomplete rows or provenance failures for {key} / {arm}"
                )
            accounted += expected
        if accounted != expected_total:
            raise ValueError(
                f"Registered combination {key} accounts for {accounted} rows, "
                f"expected {expected_total}"
            )


def validate_stage_method_set(
    *,
    selection_id: str,
    selected: Sequence[tuple[str, str]] | set[tuple[str, str]],
    promoted: Sequence[tuple[str, str]] | set[tuple[str, str]],
    winner: tuple[str, str] | None = None,
) -> None:
    """Enforce full promotion at confirmation and singleton winner at final."""

    selected_set = set(selected)
    promoted_set = set(promoted)
    if not promoted_set or len(promoted_set) > 3:
        raise ValueError("Promoted method set must contain one to three combinations")
    if selection_id == "confirmation_v1":
        if selected_set != promoted_set:
            raise ValueError("Confirmation combinations must equal all frozen promotions")
        return
    if selection_id == "final_validation_v1":
        if winner is None or winner not in promoted_set:
            raise ValueError("Locked confirmation winner is absent from frozen promotions")
        if selected_set != {winner}:
            raise ValueError("Final combinations must equal the singleton locked winner")
        return
    raise ValueError(f"Unsupported decision-stage selection: {selection_id}")


def _arm_metrics(summary: Mapping[str, Any], arm: str) -> dict[str, Any]:
    row = summary.get(arm)
    if not isinstance(row, Mapping):
        return {
            "available": False,
            "expected_rows": 0,
            "mean_paired_margin_lift": None,
            "hard_fidelity_failure_rate": None,
            "no_copy_failures": None,
            "independent_judge_pending_rows": None,
            "candidate_high_severity_semantic_failures": None,
            "neutral_high_severity_semantic_failures": None,
            "high_severity_readability_failures": None,
            "independent_judgment_binding_failures": None,
            "deterministic_style_success": None,
            "deterministic_style_success_estimate": None,
        }
    expected = int(row.get("expected_rows", 0))
    gates = row.get("gate_counts", {})
    no_copy = gates.get("no_copy_gate", {}) if isinstance(gates, Mapping) else {}
    hard_failures = int(row.get("hard_fidelity_failure_rows", 0))
    style_success = row.get("deterministic_style_success")
    style_success_estimate = (
        style_success.get("estimate")
        if isinstance(style_success, Mapping)
        else None
    )
    return {
        "available": expected > 0,
        "expected_rows": expected,
        "mean_paired_margin_lift": row.get("mean_paired_margin_lift"),
        "hard_fidelity_failure_rate": hard_failures / expected if expected else None,
        "no_copy_failures": int(no_copy.get("fail", 0)),
        "independent_judge_pending_rows": int(
            row.get("independent_judge_pending_rows", 0)
        ),
        "candidate_high_severity_semantic_failures": int(
            row.get("independent_high_severity_semantic_failure_rows", 0)
        ),
        "neutral_high_severity_semantic_failures": int(
            row.get("independent_neutral_high_severity_semantic_failure_rows", 0)
        ),
        "high_severity_readability_failures": int(
            row.get("independent_high_severity_readability_failure_rows", 0)
        ),
        "independent_judgment_binding_failures": int(
            row.get("independent_judgment_binding_failure_rows", 0)
        ),
        "deterministic_style_success": row.get("deterministic_style_success"),
        "deterministic_style_success_estimate": style_success_estimate,
    }


def assess_promotion_candidate(
    row: Mapping[str, Any], *, require_judgments: bool
) -> dict[str, Any]:
    own = _arm_metrics(row.get("arm_summaries", {}), "own_author_reconstruction")
    cross = _arm_metrics(row.get("arm_summaries", {}), "cross_author_transfer")
    reasons: list[str] = []
    if row.get("method_id") == "neutral_only":
        reasons.append("control_not_promotable")
    if row.get("method_id") == "candidate_rerank":
        reasons.append("selection_conditioned_style_meter_diagnostic_not_promotable")
    if row.get("status") != "complete":
        reasons.append("incomplete_outputs")
    for label, metrics in (("own", own), ("cross", cross)):
        lift = metrics["mean_paired_margin_lift"]
        if not metrics["available"]:
            reasons.append(f"{label}_arm_missing")
        elif not isinstance(lift, (int, float)) or float(lift) <= 0.0:
            reasons.append(f"{label}_paired_lift_not_positive")
        failure_rate = metrics["hard_fidelity_failure_rate"]
        if failure_rate is None or float(failure_rate) > 0.10:
            reasons.append(f"{label}_hard_fidelity_failure_rate_above_0_10")
        if metrics["no_copy_failures"] not in {0}:
            reasons.append(f"{label}_reference_copy_failure")
        style_success_estimate = metrics["deterministic_style_success_estimate"]
        if not isinstance(style_success_estimate, (int, float)):
            reasons.append(f"{label}_deterministic_style_success_unavailable")
        elif float(style_success_estimate) < 0.80:
            reasons.append(f"{label}_deterministic_style_success_below_0_80")
        if require_judgments:
            if metrics["independent_judge_pending_rows"] != 0:
                reasons.append(f"{label}_independent_judgments_pending")
            if metrics["candidate_high_severity_semantic_failures"] > metrics[
                "neutral_high_severity_semantic_failures"
            ]:
                reasons.append(f"{label}_semantic_noninferiority_failed")
            if metrics["high_severity_readability_failures"] != 0:
                reasons.append(f"{label}_high_severity_readability_failure")
            if metrics["independent_judgment_binding_failures"] != 0:
                reasons.append(f"{label}_independent_judgment_binding_failure")
    return {
        "method_id": str(row.get("method_id", "")),
        "method_label": str(row.get("method_label", row.get("method_id", ""))),
        "intensity": str(row.get("intensity", "")),
        "eligible": not reasons,
        "exclusion_reasons": reasons,
        "own_author_reconstruction": own,
        "cross_author_transfer": cross,
    }


def recompute_promotion(
    evaluation: Mapping[str, Any], *, require_judgments: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    require_complete_evaluation(evaluation, expected_selection_id="screening_v1")
    combinations = evaluation.get("combinations")
    if not isinstance(combinations, list) or not combinations:
        raise ValueError("Screening evaluation contains no combinations")
    assessed = [
        assess_promotion_candidate(row, require_judgments=require_judgments)
        for row in combinations
        if isinstance(row, Mapping)
    ]
    eligible = [row for row in assessed if row["eligible"]]
    eligible.sort(
        key=lambda row: (
            -float(row["own_author_reconstruction"]["mean_paired_margin_lift"]),
            -float(row["cross_author_transfer"]["mean_paired_margin_lift"]),
            row["method_id"],
            row["intensity"],
        )
    )
    promoted: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in eligible:
        if row["method_id"] in seen:
            continue
        seen.add(row["method_id"])
        promoted.append(row)
        if len(promoted) == 3:
            break
    return assessed, promoted


def recompute_confirmation_winner(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    require_complete_evaluation(evaluation, expected_selection_id="confirmation_v1")
    if evaluation.get("independent_judgments", {}).get("status") != "loaded":
        raise ValueError("Winner selection requires independent judgments")
    candidates: list[dict[str, Any]] = []
    for row in evaluation.get("combinations", []):
        if not isinstance(row, Mapping) or row.get("method_id") == "neutral_only":
            continue
        endpoints = row.get("endpoints", {})
        own = endpoints.get("own_author_reconstruction") or {}
        cross = endpoints.get("cross_author_transfer") or {}
        if (
            endpoints.get("status") == "final_judgments_available"
            and own.get("final_pass") is True
            and cross.get("final_pass") is True
            and endpoints.get("final_selection") == "pass"
        ):
            own_summary = row.get("arm_summaries", {}).get(
                "own_author_reconstruction", {}
            )
            cross_summary = row.get("arm_summaries", {}).get(
                "cross_author_transfer", {}
            )
            candidates.append(
                {
                    "method_id": str(row["method_id"]),
                    "intensity": str(row["intensity"]),
                    "own_final_success": float(
                        own_summary["final_style_success"]["wilson_95"]["estimate"]
                    ),
                    "cross_final_success": float(
                        cross_summary["final_style_success"]["wilson_95"]["estimate"]
                    ),
                    "own_mean_lift": float(own_summary["mean_paired_margin_lift"]),
                    "cross_mean_lift": float(cross_summary["mean_paired_margin_lift"]),
                }
            )
    if not candidates:
        raise ValueError("No confirmation method passed all final endpoints")
    candidates.sort(
        key=lambda row: (
            -row["own_final_success"],
            -row["cross_final_success"],
            -row["own_mean_lift"],
            -row["cross_mean_lift"],
            row["method_id"],
            row["intensity"],
        )
    )
    return candidates[0]
