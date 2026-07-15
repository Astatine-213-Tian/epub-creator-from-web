#!/usr/bin/env python3
from __future__ import annotations

"""Run deterministic preregistration-gate regression checks."""

import argparse
import json
import tempfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from experiments.iteration1.style_analysis_lock import DEFAULT_EXPERIMENT_ROOT, validate_analysis_lock
from experiments.iteration1.style_experiment_decisions import (
    recompute_promotion,
    validate_stage_method_set,
)
from experiments.iteration1.style_experiment_provenance import (
    LEDGER_CONFIG_FIELDS,
    canonical_json,
    file_sha256,
    sha256_text,
    validate_artifact_and_ledger,
)
from experiments.iteration1.run_style_transfer_generation import validate_execution_admission


def complete_evaluation() -> dict[str, Any]:
    arm = {
        "expected_rows": 18,
        "scored_rows": 18,
        "failed_or_missing_rows": 0,
        "mean_paired_margin_lift": 1.0,
        "hard_fidelity_failure_rows": 0,
        "gate_counts": {"no_copy_gate": {"fail": 0}},
        "independent_judge_pending_rows": 0,
        "independent_high_severity_semantic_failure_rows": 0,
        "independent_neutral_high_severity_semantic_failure_rows": 0,
        "independent_high_severity_readability_failure_rows": 0,
        "independent_judgment_binding_failure_rows": 0,
        "deterministic_style_success": {"estimate": 1.0},
    }
    combinations = []
    for index in range(1, 4):
        combinations.append(
            {
                "method_id": f"method_{index}",
                "method_label": f"Method {index}",
                "intensity": "medium",
                "status": "complete",
                "arm_summaries": {
                    "own_author_reconstruction": deepcopy(arm),
                    "cross_author_transfer": {
                        **deepcopy(arm),
                        "mean_paired_margin_lift": 1.0 - index / 10,
                    },
                },
            }
        )
    return {
        "status": "complete",
        "sample_contract": {
            "selection_id": "screening_v1",
            "method_evaluation_rows": 36,
        },
        "combinations": combinations,
    }


def expect_failure(label: str, callback) -> str:
    try:
        callback()
    except ValueError:
        return label
    raise AssertionError(f"Expected deterministic rejection: {label}")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def analysis_lineage_fixture(active_binding: dict[str, Any]) -> list[str]:
    """Prove a self-consistent stale generation chain cannot be relabeled."""

    checks: list[str] = []
    stale_binding = dict(active_binding)
    stale_binding["sha256"] = "0" * 64
    stale_binding["content_sha256"] = "1" * 64
    with tempfile.TemporaryDirectory(prefix="style-analysis-lineage-") as temp:
        root = Path(temp)
        sample_id = "s_fixture"
        run_id = "lineage_fixture"
        stage = "style_transfer"
        resources: dict[str, Path] = {}
        for label in ("model_config", "prompt", "schema", "runner"):
            path = root / f"{label}.snapshot"
            path.write_text(label, encoding="utf-8")
            resources[label] = path
        config_path = root / "run_config.json"
        config = {
            "run_id": run_id,
            "sample_set": "fixture",
            "stage": stage,
            "analysis_lock": stale_binding,
            "model": "fixture-model",
            "reasoning_effort": "high",
            "provider": "fixture",
            "codex_cli_version": "fixture",
            "codex_binary_realpath": "/fixture/codex",
            "prompt_sha256": file_sha256(resources["prompt"]),
            "schema_sha256": file_sha256(resources["schema"]),
            "runner_sha256": file_sha256(resources["runner"]),
            "environment_sha256": "2" * 64,
            "sandbox_profile_sha256": "3" * 64,
            "effective_command_sha256": "4" * 64,
            "model_config_sha256": file_sha256(resources["model_config"]),
            "model_config_snapshot_path": str(resources["model_config"]),
            "prompt_snapshot_path": str(resources["prompt"]),
            "schema_snapshot_path": str(resources["schema"]),
            "runner_snapshot_path": str(resources["runner"]),
            "external_isolation_preflight": {
                "allowed_request_area": str(root),
                "denied_probes": [{"denied": True}],
            },
        }
        write_json(config_path, config)
        selection_path = root / "selection.json"
        write_json(
            selection_path,
            {
                "selection_id": "screening_v1",
                "sample_count": 1,
                "sample_ids": [sample_id],
            },
        )
        admission = {
            "admission_stage": "preregistered_screening",
            "analysis_lock": stale_binding,
        }
        selection_fields = {
            "execution_selection_id": "screening_v1",
            "execution_selection_sha256": file_sha256(selection_path),
            "execution_selection_sample_count": 1,
            "execution_selection_path": str(selection_path),
            "execution_selection_is_frozen_file": True,
            "active_sample_count": 1,
            "active_sample_ids_sha256": sha256_text(canonical_json([sample_id])),
        }
        batch_path = root / "batch.json"

        def write_batch(current_admission: dict[str, Any]) -> None:
            write_json(
                batch_path,
                {
                    "schema_version": 1,
                    "stage_run_config_path": str(config_path),
                    "stage_run_config_sha256": file_sha256(config_path),
                    **selection_fields,
                    "active_sample_ids": [sample_id],
                    "admission": current_admission,
                },
            )

        write_batch(admission)
        execution_fields = {
            **selection_fields,
            "execution_batch_config_path": str(batch_path),
            "execution_batch_config_sha256": file_sha256(batch_path),
            "execution_admission": admission,
        }
        result = {"sample_id": sample_id, "paragraphs": [{"id": "p0001", "zh": "测试"}]}
        output_sha = sha256_text(canonical_json(result))
        artifact_path = root / "artifact.json"

        def chain(current_execution: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            artifact = {
                "sample_id": sample_id,
                "sample_set": "fixture",
                "run_id": run_id,
                "stage": stage,
                "result": result,
                "output_sha256": output_sha,
                **current_execution,
            }
            ledger = {
                "sample_id": sample_id,
                "run_id": run_id,
                "stage": stage,
                "output_sha256": output_sha,
                "response_file": str(artifact_path),
                **{field: config[field] for field in LEDGER_CONFIG_FIELDS},
                **current_execution,
            }
            write_json(artifact_path, artifact)
            return artifact, ledger

        stale_artifact, stale_ledger = chain(execution_fields)
        stale_errors = validate_artifact_and_ledger(
            artifact_path=artifact_path,
            artifact=stale_artifact,
            ledger_row=stale_ledger,
            run_config_path=config_path,
            sample_id=sample_id,
            run_id=run_id,
            stage=stage,
            expected_analysis_lock=active_binding,
        )
        if not stale_errors or not any("analysis-lock" in value for value in stale_errors):
            raise AssertionError("Self-consistent stale analysis lineage was not rejected")
        checks.append("self_consistent_stale_analysis_lineage_rejected")

        config["analysis_lock"] = active_binding
        write_json(config_path, config)
        admission = {
            "admission_stage": "preregistered_screening",
            "analysis_lock": active_binding,
        }
        write_batch(admission)
        execution_fields.update(
            {
                "execution_batch_config_sha256": file_sha256(batch_path),
                "execution_admission": admission,
            }
        )
        active_artifact, active_ledger = chain(execution_fields)
        active_errors = validate_artifact_and_ledger(
            artifact_path=artifact_path,
            artifact=active_artifact,
            ledger_row=active_ledger,
            run_config_path=config_path,
            sample_id=sample_id,
            run_id=run_id,
            stage=stage,
            expected_analysis_lock=active_binding,
        )
        if active_errors:
            raise AssertionError(f"Current analysis lineage fixture failed: {active_errors}")
        checks.append("current_analysis_lineage_accepted")
    return checks


def admission_branch_fixtures(
    *,
    analysis_lock_path: Path,
    active_binding: dict[str, Any],
    experiment_root: Path,
) -> list[str]:
    """Exercise the real critique and confirmation admission branches."""

    checks: list[str] = []
    evaluation = complete_evaluation()
    preregistrations = sorted(
        (experiment_root / "protocols").glob("*preregistration*.json")
    )
    sample_sets = {
        str(payload["sample_set"])
        for path in preregistrations
        for payload in [json.loads(path.read_text(encoding="utf-8"))]
        if isinstance(payload, dict) and isinstance(payload.get("sample_set"), str)
    }
    if len(sample_sets) > 1:
        raise ValueError(f"Multiple preregistered sample sets: {sorted(sample_sets)}")
    sample_set = next(iter(sample_sets), "development_proxy_v1")
    registry = json.loads(
        (experiment_root / "method_registry/style_methods.v1.json").read_text(
            encoding="utf-8"
        )
    )
    real_methods = [
        (str(row["id"]), str(row["intensities"][0]))
        for row in registry.get("methods", [])
        if row.get("id") not in {"neutral_only", "candidate_rerank"}
        and isinstance(row.get("intensities"), list)
        and row["intensities"]
    ][:3]
    if len(real_methods) != 3:
        raise ValueError("Gate fixture requires three registered promotable methods")
    for row, (method_id, intensity) in zip(
        evaluation["combinations"], real_methods
    ):
        row["method_id"] = method_id
        row["method_label"] = method_id
        row["intensity"] = intensity
    _, promoted_rows = recompute_promotion(evaluation, require_judgments=True)
    promoted = [
        {"method_id": row["method_id"], "intensity": row["intensity"]}
        for row in promoted_rows
    ]
    candidate = promoted[0]
    screening_path = (
        experiment_root
        / f"sample_sets/{sample_set}.screening_v1_ids.json"
    )
    confirmation_path = (
        experiment_root
        / f"sample_sets/{sample_set}.confirmation_v1_ids.json"
    )

    def selection(path: Path, selection_id: str) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            "execution_selection_id": selection_id,
            "execution_selection_sha256": file_sha256(path),
            "execution_selection_sample_count": len(payload["sample_ids"]),
            "execution_selection_path": str(path),
            "execution_selection_is_frozen_file": True,
            "active_sample_count": len(payload["sample_ids"]),
            "active_sample_ids_sha256": sha256_text(
                canonical_json(sorted(payload["sample_ids"]))
            ),
            "_active_sample_ids": sorted(payload["sample_ids"]),
        }

    with tempfile.TemporaryDirectory(
        prefix="style-admission-branches-", dir=experiment_root
    ) as temp:
        root = Path(temp)
        evaluation_path = root / "screening_evaluation.json"
        write_json(evaluation_path, evaluation)
        shortlist_path = root / "shortlist.json"
        write_json(
            shortlist_path,
            {
                "status": "provisional_shortlist_frozen",
                "selection_id": "screening_v1",
                "analysis_lock": active_binding,
                "screening_evaluation_path": str(evaluation_path),
                "screening_evaluation_sha256": file_sha256(evaluation_path),
                "promoted": promoted,
            },
        )
        critique_args = SimpleNamespace(
            stage="style_critique",
            analysis_lock=analysis_lock_path,
            experiment_root=experiment_root,
            base_method_id=candidate["method_id"],
            base_intensity=candidate["intensity"],
            method_id=None,
            intensity=None,
            refinement_contract=None,
            provisional_shortlist=shortlist_path,
            promotion_file=None,
            final_lock=None,
            sample_set=sample_set,
        )
        critique_admission = validate_execution_admission(
            critique_args, selection(screening_path, "screening_v1")
        )
        if critique_admission.get("admission_stage") != (
            "screening_provisional_shortlist"
        ):
            raise AssertionError("Screening critique admission branch failed")
        checks.append("screening_critique_admission_branch_accepted")

        promotion_path = root / "promotion.json"
        write_json(
            promotion_path,
            {
                "status": "promotions_frozen",
                "selection_id": "screening_v1",
                "analysis_lock": active_binding,
                "screening_evaluation_path": str(evaluation_path),
                "screening_evaluation_sha256": file_sha256(evaluation_path),
                "promoted": promoted,
                "method_registry_sha256": file_sha256(
                    experiment_root / "method_registry/style_methods.v1.json"
                ),
                "threshold_sha256": file_sha256(
                    experiment_root / "calibration/style_meter_threshold.v1.json"
                ),
                "screening_selection_sha256": file_sha256(screening_path),
            },
        )
        confirmation_args = SimpleNamespace(
            stage="style_transfer",
            analysis_lock=analysis_lock_path,
            experiment_root=experiment_root,
            base_method_id=None,
            base_intensity=None,
            method_id=candidate["method_id"],
            intensity=candidate["intensity"],
            refinement_contract=None,
            provisional_shortlist=None,
            promotion_file=promotion_path,
            final_lock=None,
            sample_set=sample_set,
        )
        confirmation_admission = validate_execution_admission(
            confirmation_args, selection(confirmation_path, "confirmation_v1")
        )
        if confirmation_admission.get("admission_stage") != (
            "judged_screening_promotion"
        ):
            raise AssertionError("Confirmation admission branch failed")
        checks.append("confirmation_transfer_admission_branch_accepted")
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-lock", type=Path)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checks: list[str] = []
    evaluation = complete_evaluation()
    _, promoted_rows = recompute_promotion(evaluation, require_judgments=False)
    if len(promoted_rows) != 3:
        raise AssertionError("Complete screening evaluation did not promote three arms")
    checks.append("complete_screening_can_promote")

    incomplete = deepcopy(evaluation)
    incomplete["status"] = "complete_with_method_failures"
    incomplete["combinations"][0]["status"] = "completed_with_failures"
    incomplete["combinations"][0]["arm_summaries"][
        "own_author_reconstruction"
    ]["failed_or_missing_rows"] = 1
    checks.append(
        expect_failure(
            "incomplete_registered_arm_blocks_promotion",
            lambda: recompute_promotion(incomplete, require_judgments=False),
        )
    )

    for size in (1, 2, 3):
        promoted = [(f"method_{index}", "medium") for index in range(1, size + 1)]
        validate_stage_method_set(
            selection_id="confirmation_v1",
            selected=promoted,
            promoted=promoted,
        )
        validate_stage_method_set(
            selection_id="final_validation_v1",
            selected=[promoted[0]],
            promoted=promoted,
            winner=promoted[0],
        )
        checks.append(f"promotion_size_{size}_confirmation_and_final_admitted")

    promoted = [("method_1", "medium"), ("method_2", "medium")]
    checks.extend(
        (
            expect_failure(
                "partial_confirmation_roster_rejected",
                lambda: validate_stage_method_set(
                    selection_id="confirmation_v1",
                    selected=[promoted[0]],
                    promoted=promoted,
                ),
            ),
            expect_failure(
                "altered_winner_rejected",
                lambda: validate_stage_method_set(
                    selection_id="final_validation_v1",
                    selected=[("altered", "medium")],
                    promoted=promoted,
                    winner=("altered", "medium"),
                ),
            ),
            expect_failure(
                "nonwinner_final_method_rejected",
                lambda: validate_stage_method_set(
                    selection_id="final_validation_v1",
                    selected=[promoted[1]],
                    promoted=promoted,
                    winner=promoted[0],
                ),
            ),
        )
    )
    lock_binding = None
    if args.analysis_lock:
        lock_binding = validate_analysis_lock(
            args.analysis_lock.resolve(), args.experiment_root.resolve()
        )
        checks.append("analysis_lock_valid")
        checks.extend(analysis_lineage_fixture(lock_binding))
        checks.extend(
            admission_branch_fixtures(
                analysis_lock_path=args.analysis_lock.resolve(),
                active_binding=lock_binding,
                experiment_root=args.experiment_root.resolve(),
            )
        )
    print(
        json.dumps(
            {
                "status": "passed",
                "checks": checks,
                "analysis_lock": lock_binding,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
