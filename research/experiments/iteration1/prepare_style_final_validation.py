#!/usr/bin/env python3
from __future__ import annotations

"""Lock a confirmation winner, then reveal the reserved final-validation sample set."""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


from experiments.shared.paths import RESEARCH_ROOT


REPO_ROOT = RESEARCH_ROOT

from experiments.iteration1.style_transfer_research import (  # noqa: E402
    FINAL_VALIDATION_BOOKS,
    STRATIFICATION_FIELDS,
    base_metrics,
    enrich_strata,
    file_sha256,
    load_masked_targets,
    opaque_sample_id,
    relative_artifact_path,
    residue_hits,
    rows_sha256,
    select_diverse_rows,
    stable_seed,
    write_json,
    write_jsonl,
)
from experiments.iteration1.style_experiment_decisions import (  # noqa: E402
    recompute_confirmation_winner,
    recompute_promotion,
    validate_stage_method_set,
)
from experiments.iteration1.style_analysis_lock import (  # noqa: E402
    require_matching_analysis_binding,
    validate_analysis_lock,
)


DEFAULT_EXPERIMENT_ROOT = (
    REPO_ROOT / "generated/style_research/style_transfer_experiments"
)
DEFAULT_CLEAN = REPO_ROOT / "datasets/unmasked/chunks.clean.jsonl"
DEFAULT_MASKED = REPO_ROOT / "datasets/masked/chunks.entity_masked_v3.jsonl"
DEFAULT_SPLITS = REPO_ROOT / "generated/style_research/corpus/splits.json"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def recorded_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def iter_jsonl(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select and lock the reserved target-author test books only after a "
            "confirmation method passes all final gates."
        )
    )
    parser.add_argument("--confirmation-evaluation", type=Path, required=True)
    parser.add_argument("--promotion-file", type=Path, required=True)
    parser.add_argument("--analysis-lock", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--sample-set", default="final_validation_v1")
    parser.add_argument("--target-author", default="非天夜翔")
    parser.add_argument("--clean-chunks", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--masked-chunks", type=Path, default=DEFAULT_MASKED)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--chunks-per-book", type=int, default=20)
    parser.add_argument("--min-chunk-distance", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260711)
    return parser.parse_args()


def choose_winner(evaluation: dict[str, Any]) -> dict[str, Any]:
    return recompute_confirmation_winner(evaluation)


def main() -> None:
    args = parse_args()
    if args.chunks_per_book <= 0 or args.min_chunk_distance < 2:
        raise ValueError("Invalid chunk count or spacing")
    experiment_root = args.experiment_root.resolve()
    analysis_lock_binding = validate_analysis_lock(
        args.analysis_lock.resolve(), experiment_root
    )
    sample_root = experiment_root / "sample_sets"
    lock_path = experiment_root / "final_validation/final_validation_v1.lock.json"
    pre_reveal_lock_path = (
        experiment_root / "final_validation/final_validation_v1.pre_reveal.lock.json"
    )
    output_paths = {
        "runner": sample_root / f"{args.sample_set}.runner_manifest.jsonl",
        "allocation": sample_root / f"{args.sample_set}.evaluator_allocation.jsonl",
        "hidden": sample_root / f"{args.sample_set}.hidden_targets.jsonl",
        "method_ids": sample_root / f"{args.sample_set}.method_evaluation_ids.json",
        "final_ids": sample_root / f"{args.sample_set}.final_validation_v1_ids.json",
        "summary": sample_root / f"{args.sample_set}.summary.json",
        "lock": lock_path,
        "pre_reveal_lock": pre_reveal_lock_path,
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Final validation is one-time-only; existing artifacts: "
            + ", ".join(relative_artifact_path(path) for path in existing)
        )

    confirmation_path = args.confirmation_evaluation.resolve()
    promotion_path = args.promotion_file.resolve()
    confirmation = load_json(confirmation_path)
    require_matching_analysis_binding(
        confirmation, analysis_lock_binding, label="confirmation evaluation"
    )
    promotion = load_json(promotion_path)
    require_matching_analysis_binding(
        promotion, analysis_lock_binding, label="screening promotion"
    )
    if promotion.get("status") != "promotions_frozen":
        raise ValueError("Screening promotion artifact is not frozen")
    screening_evaluation_value = promotion.get("screening_evaluation_path")
    if not isinstance(screening_evaluation_value, str):
        raise ValueError("Promotion lacks its screening evaluation path")
    screening_evaluation_path = recorded_path(screening_evaluation_value)
    if not screening_evaluation_path.exists() or file_sha256(
        screening_evaluation_path
    ) != promotion.get("screening_evaluation_sha256"):
        raise ValueError("Promotion screening evaluation binding mismatch")
    screening_evaluation = load_json(screening_evaluation_path)
    require_matching_analysis_binding(
        screening_evaluation,
        analysis_lock_binding,
        label="promotion screening evaluation",
    )
    _, recomputed_promoted = recompute_promotion(
        screening_evaluation, require_judgments=True
    )
    recomputed_order = [
        (row["method_id"], row["intensity"]) for row in recomputed_promoted
    ]
    recorded_order = [
        (str(row.get("method_id", "")), str(row.get("intensity", "")))
        for row in promotion.get("promoted", [])
        if isinstance(row, dict)
    ]
    if recorded_order != recomputed_order:
        raise ValueError("Promotion decision does not reproduce from screening")
    promoted = {
        (str(row.get("method_id", "")), str(row.get("intensity", "")))
        for row in promotion.get("promoted", [])
    }
    confirmation_selected = {
        (str(row.get("method_id", "")), str(row.get("intensity", "")))
        for row in confirmation.get("combinations", [])
        if isinstance(row, dict) and row.get("method_id") != "neutral_only"
    }
    validate_stage_method_set(
        selection_id="confirmation_v1",
        selected=confirmation_selected,
        promoted=promoted,
    )
    confirmation_promotion = confirmation.get("promotion_contract")
    if not isinstance(confirmation_promotion, dict) or confirmation_promotion.get(
        "status"
    ) != "valid":
        raise ValueError("Confirmation evaluation has no valid promotion contract")
    promotion_binding_value = confirmation_promotion.get("path")
    if not isinstance(promotion_binding_value, str) or recorded_path(
        promotion_binding_value
    ) != promotion_path:
        raise ValueError("Confirmation evaluation promotion path mismatch")
    if confirmation_promotion.get("sha256") != file_sha256(promotion_path):
        raise ValueError("Confirmation evaluation promotion hash mismatch")
    confirmation_promoted = {
        (str(row.get("method_id", "")), str(row.get("intensity", "")))
        for row in confirmation_promotion.get("promoted", [])
        if isinstance(row, dict)
    }
    if confirmation_promoted != promoted:
        raise ValueError("Confirmation evaluation promotion roster mismatch")
    winner = choose_winner(confirmation)
    winner_key = (winner["method_id"], winner["intensity"])
    if winner_key not in promoted:
        raise ValueError("Confirmation winner is absent from frozen promotion artifact")

    prompt_paths = {
        "english_semantic_source": experiment_root
        / "prompts/english_semantic_source.v1.md",
        "english_source_qa": experiment_root / "prompts/english_source_qa.v1.md",
        "english_source_repair": experiment_root
        / "prompts/english_source_repair.v1.md",
        "neutral_translation": experiment_root / "prompts/neutral_translation.v1.md",
        "style_transfer": experiment_root / "prompts/style_transfer_method.v1.md",
        "style_critique": experiment_root
        / "prompts/style_transfer_critique.v1.md",
    }
    schema_paths = {
        "english_semantic_source": experiment_root
        / "schemas/english_semantic_source_output.v1.schema.json",
        "english_source_qa": experiment_root
        / "schemas/english_source_qa_output.v1.schema.json",
        "english_source_repair": experiment_root
        / "schemas/english_source_repair_output.v1.schema.json",
        "neutral_translation": experiment_root
        / "schemas/neutral_translation_output.v1.schema.json",
        "style_transfer": experiment_root
        / "schemas/style_transfer_output.v1.schema.json",
        "style_critique": experiment_root
        / "schemas/style_transfer_critique_output.v1.schema.json",
    }
    pre_reveal_lock = {
        "schema_version": 1,
        "lock_id": "final_validation_v1.pre_reveal",
        "analysis_lock": analysis_lock_binding,
        "winner": winner,
        "confirmation_evaluation_path": relative_artifact_path(confirmation_path),
        "confirmation_evaluation_sha256": file_sha256(confirmation_path),
        "promotion_path": relative_artifact_path(promotion_path),
        "promotion_sha256": file_sha256(promotion_path),
        "prompt_sha256": {
            key: file_sha256(path) for key, path in prompt_paths.items()
        },
        "schema_sha256": {
            key: file_sha256(path) for key, path in schema_paths.items()
        },
        "model_run_config_sha256": file_sha256(
            experiment_root / "protocols/model_run_config.v1.json"
        ),
        "evaluation_protocol_sha256": file_sha256(
            experiment_root / "protocols/evaluation_protocol.v1.json"
        ),
        "method_registry_sha256": file_sha256(
            experiment_root / "method_registry/style_methods.v1.json"
        ),
        "payload_lock_sha256": file_sha256(
            experiment_root / "method_assets/style_transfer_payloads.v1.lock.json"
        ),
        "runner_source_sha256": file_sha256(
            REPO_ROOT / "experiments/iteration1/run_style_transfer_generation.py"
        ),
        "evaluator_source_sha256": file_sha256(
            REPO_ROOT / "experiments/iteration1/evaluate_style_transfer_methods.py"
        ),
        "payload_builder_source_sha256": file_sha256(
            REPO_ROOT / "experiments/iteration1/style_transfer_payloads.py"
        ),
        "clean_chunks_sha256": file_sha256(args.clean_chunks.resolve()),
        "masked_chunks_sha256": file_sha256(args.masked_chunks.resolve()),
        "splits_sha256": file_sha256(args.splits.resolve()),
        "reserved_target_rows_not_selected_or_materialized_before_this_lock": True,
    }
    write_json(pre_reveal_lock_path, pre_reveal_lock)

    splits = load_json(args.splits.resolve())
    test_titles = {
        row["title"]
        for row in splits.get("test", [])
        if row.get("author") == args.target_author
    }
    if set(FINAL_VALIDATION_BOOKS) - test_titles:
        raise ValueError("Reserved final books are not all in the test split")
    rows_by_book: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in iter_jsonl(args.clean_chunks.resolve()):
        title = str(row.get("title", ""))
        if (
            row.get("author") != args.target_author
            or title not in FINAL_VALIDATION_BOOKS
            or row.get("split") != "test"
        ):
            continue
        text = str(row.get("text", ""))
        if not text.strip() or residue_hits(text):
            continue
        candidate = {
            "author": args.target_author,
            "book_title": title,
            "source_split": "test",
            "chunk_id": str(row["chunk_id"]),
            "chunk_index": int(row["chunk_index"]),
            "quality_flags": list(row.get("quality_flags", [])),
            "original_zh": text,
        }
        candidate.update(base_metrics(text))
        rows_by_book[title].append(candidate)
    missing = set(FINAL_VALIDATION_BOOKS) - set(rows_by_book)
    if missing:
        raise ValueError(f"No final-validation chunks for: {sorted(missing)}")
    enrich_strata(rows_by_book)
    selected: list[dict[str, Any]] = []
    for title in FINAL_VALIDATION_BOOKS:
        rows = select_diverse_rows(
            rows_by_book[title],
            args.chunks_per_book,
            stable_seed(args.seed, f"final:{title}"),
            min_chunk_distance=args.min_chunk_distance,
        )
        for row in rows:
            row["research_role"] = "final_validation"
            row["benchmark_arm"] = "own_author_reconstruction"
        selected.extend(rows)
    masked = load_masked_targets(
        args.masked_chunks.resolve(), {row["chunk_id"] for row in selected}
    )

    runner_rows: list[dict[str, Any]] = []
    allocation_rows: list[dict[str, Any]] = []
    hidden_rows: list[dict[str, Any]] = []
    for row in selected:
        sample_id = opaque_sample_id(args.seed, args.sample_set, row["chunk_id"])
        runner_rows.append(
            {
                "sample_id": sample_id,
                "sample_set": args.sample_set,
                "stage_status": {
                    "english_semantic_source": "pending",
                    "english_source_qa": "pending",
                    "neutral_translation": "pending",
                    "style_transfer": "pending",
                    "automatic_evaluation": "pending",
                    "independent_evaluation": "pending",
                },
            }
        )
        allocation_rows.append(
            {
                "sample_id": sample_id,
                "sample_set": args.sample_set,
                "benchmark_arm": row["benchmark_arm"],
                "author": row["author"],
                "book_title": row["book_title"],
                "source_split": row["source_split"],
                "research_role": row["research_role"],
                "chunk_id": row["chunk_id"],
                "chunk_index": row["chunk_index"],
                "quality_flags": row["quality_flags"],
                "selected_residue_hits": residue_hits(row["original_zh"]),
                "strata": {field: row[field] for field in STRATIFICATION_FIELDS},
                "access_policy": "frozen_output_evaluator_only",
            }
        )
        hidden_rows.append(
            {
                "sample_id": sample_id,
                "sample_set": args.sample_set,
                "original_zh": row["original_zh"],
                "entity_masked_v3_zh": masked[row["chunk_id"]],
                "access_policy": (
                    "one_way_semantic_source_generator_and_frozen_output_evaluator_only"
                ),
            }
        )
    runner_rows.sort(key=lambda row: row["sample_id"])
    allocation_rows.sort(key=lambda row: row["sample_id"])
    hidden_rows.sort(key=lambda row: row["sample_id"])
    sample_ids = [row["sample_id"] for row in runner_rows]
    write_jsonl(output_paths["runner"], runner_rows)
    write_jsonl(output_paths["allocation"], allocation_rows)
    write_jsonl(output_paths["hidden"], hidden_rows)
    method_ids = {
        "schema_version": 1,
        "sample_set": args.sample_set,
        "allowed_research_roles": ["final_validation"],
        "sample_count": len(sample_ids),
        "sample_ids": sample_ids,
    }
    final_ids = {
        "schema_version": 1,
        "selection_id": "final_validation_v1",
        "sample_set": args.sample_set,
        "sample_count": len(sample_ids),
        "sample_ids": sample_ids,
    }
    write_json(output_paths["method_ids"], method_ids)
    write_json(output_paths["final_ids"], final_ids)
    registry_path = experiment_root / "method_registry/style_methods.v1.json"
    method_config_path = (
        experiment_root
        / "method_registry/methods"
        / f"{winner['method_id']}.v1.json"
    )
    threshold_path = experiment_root / "calibration/style_meter_threshold.v1.json"
    payload_lock_path = experiment_root / "method_assets/style_transfer_payloads.v1.lock.json"
    summary = {
        "schema_version": 1,
        "sample_set": args.sample_set,
        "design": "one_time_reserved_final_validation",
        "target_author": args.target_author,
        "books": list(FINAL_VALIDATION_BOOKS),
        "chunks_per_book": args.chunks_per_book,
        "total_samples": len(sample_ids),
        "runner_manifest_sha256": rows_sha256(runner_rows),
        "evaluator_allocation_sha256": rows_sha256(allocation_rows),
        "hidden_targets_sha256": rows_sha256(hidden_rows),
        "method_evaluation_ids_sha256": file_sha256(output_paths["method_ids"]),
        "final_validation_ids_sha256": file_sha256(output_paths["final_ids"]),
        "winner": winner,
        "leakage_controls": {
            "created_only_after_confirmation_pass": True,
            "opaque_sample_ids": True,
            "hidden_targets_separate_from_runner_manifest": True,
            "test_split_only": True,
        },
    }
    write_json(output_paths["summary"], summary)
    lock = {
        "schema_version": 1,
        "lock_id": "final_validation_v1.one_time_lock",
        "analysis_lock": analysis_lock_binding,
        "sample_set": args.sample_set,
        "selection_id": "final_validation_v1",
        "winner": winner,
        "confirmation_evaluation_path": relative_artifact_path(confirmation_path),
        "confirmation_evaluation_sha256": file_sha256(confirmation_path),
        "promotion_path": relative_artifact_path(promotion_path),
        "promotion_sha256": file_sha256(promotion_path),
        "threshold_path": relative_artifact_path(threshold_path),
        "threshold_sha256": file_sha256(threshold_path),
        "method_registry_sha256": file_sha256(registry_path),
        "method_config_sha256": file_sha256(method_config_path),
        "payload_lock_sha256": file_sha256(payload_lock_path),
        "model_run_config_sha256": file_sha256(
            experiment_root / "protocols/model_run_config.v1.json"
        ),
        "style_prompt_sha256": file_sha256(
            experiment_root / "prompts/style_transfer_method.v1.md"
        ),
        "final_selection_sha256": file_sha256(output_paths["final_ids"]),
        "sample_summary_sha256": file_sha256(output_paths["summary"]),
        "pre_reveal_lock_path": relative_artifact_path(pre_reveal_lock_path),
        "pre_reveal_lock_sha256": file_sha256(pre_reveal_lock_path),
        "prompt_sha256": pre_reveal_lock["prompt_sha256"],
        "schema_sha256": pre_reveal_lock["schema_sha256"],
        "runner_source_sha256": pre_reveal_lock["runner_source_sha256"],
        "evaluator_source_sha256": pre_reveal_lock["evaluator_source_sha256"],
        "payload_builder_source_sha256": pre_reveal_lock[
            "payload_builder_source_sha256"
        ],
        "clean_chunks_sha256": pre_reveal_lock["clean_chunks_sha256"],
        "masked_chunks_sha256": pre_reveal_lock["masked_chunks_sha256"],
        "splits_sha256": pre_reveal_lock["splits_sha256"],
        "one_time_only": True,
    }
    write_json(lock_path, lock)
    output_paths["allocation"].chmod(0o600)
    output_paths["hidden"].chmod(0o600)
    print(
        json.dumps(
            {
                "status": "final_validation_locked",
                "winner": winner_key,
                "samples": len(sample_ids),
                "lock": relative_artifact_path(lock_path),
                "lock_sha256": file_sha256(lock_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
