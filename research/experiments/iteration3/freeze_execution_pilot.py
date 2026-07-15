#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


from experiments.shared.paths import RESEARCH_ROOT


REPO_ROOT = RESEARCH_ROOT


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def ledger_summary(path: Path) -> dict[str, Any]:
    rows = list(iter_jsonl(path))
    successes = [row for row in rows if row.get("status") == "success"]
    failures = [row for row in rows if row.get("status") != "success"]
    errors: Counter[str] = Counter()
    for row in failures:
        for value in row.get("validation_errors", []):
            category = str(value).split(":", 1)[0]
            errors[category] += 1
    return {
        "path": relative(path),
        "sha256": file_sha256(path),
        "rows": len(rows),
        "successful_rows": len(successes),
        "failed_rows": len(failures),
        "successful_unique_samples": len(
            {str(row.get("sample_id")) for row in successes}
        ),
        "failed_unique_samples": len(
            {str(row.get("sample_id")) for row in failures}
        ),
        "failure_categories": dict(sorted(errors.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze an unscored failed style-generation pilot as superseded."
    )
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--sample-set", required=True)
    parser.add_argument("--pilot-run-id", required=True)
    parser.add_argument("--source-run-id")
    parser.add_argument("--active-style-run-id", required=True)
    parser.add_argument(
        "--pilot-kind",
        choices=(
            "block30_schema_and_merge",
            "block12_reference_payload",
            "cross_run_evaluation_contract",
            "block12_source_id_adherence",
        ),
        required=True,
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.experiment_root.expanduser().resolve()
    run_root = root / "runs" / args.sample_set / args.pilot_run_id
    output = args.output or (
        root / "protocols/iteration3_execution_amendment.v1.json"
    )
    output = output.expanduser().resolve()
    evaluated = sorted(
        path
        for pattern in ("evaluation/**/*.json", "independent_evaluation/**/*.json")
        for path in run_root.glob(pattern)
        if path.is_file()
    )
    if evaluated:
        raise ValueError(
            "Pilot already has outcome-evaluation artifacts: "
            + ", ".join(relative(path) for path in evaluated[:5])
        )
    ledgers = sorted((run_root / "ledgers").glob("style_transfer.*.jsonl"))
    if args.pilot_kind in {
        "cross_run_evaluation_contract",
        "block12_source_id_adherence",
    }:
        outputs = sorted(path for path in run_root.rglob("*") if path.is_file())
    else:
        outputs = sorted(
            path
            for pattern in (
                "method_outputs/**/*.json",
                "ledgers/style_transfer.*.jsonl",
            )
            for path in run_root.glob(pattern)
            if path.is_file()
        )
    if not ledgers or not outputs:
        raise ValueError("Pilot has no style-generation evidence to freeze")
    inventory = [
        {"path": relative(path), "sha256": file_sha256(path)} for path in outputs
    ]
    decision_basis = "operational_schema_adherence_only"
    if args.pilot_kind == "block30_schema_and_merge":
        reason = (
            "The 30-paragraph execution pilot had systematic paragraph-ID/schema "
            "failures and a deterministic block-merge loader defect. No style-meter, "
            "semantic, readability, selection, or promotion evaluation was run."
        )
        superseded_contract = {
            "block_threshold": 36,
            "max_block_paragraphs": 30,
            "reference_representation": "four_full_aligned_chunks",
        }
        replacement_contract = {
            "block_threshold": 12,
            "max_block_paragraphs": 12,
            "reference_representation": "four_full_aligned_chunks",
            "all_fixed_arms_regenerated_from_scratch": True,
        }
    elif args.pilot_kind == "block12_reference_payload":
        reason = (
            "The 12-paragraph pilot showed that full-chunk aligned references still "
            "dominated the request: both aligned-only arms had zero successful rows "
            "before termination, while the no-aligned-reference microcard arm produced "
            "valid rows. The decision used schema/ID adherence only; no style outcome "
            "was scored or inspected."
        )
        superseded_contract = {
            "block_threshold": 12,
            "max_block_paragraphs": 12,
            "reference_representation": "four_full_aligned_chunks",
        }
        replacement_contract = {
            "block_threshold": 12,
            "max_block_paragraphs": 12,
            "reference_representation": (
                "same_four_pair_ids_each_as_deterministic_contiguous_three_paragraph_window"
            ),
            "all_fixed_arms_regenerated_from_scratch": True,
        }
    elif args.pilot_kind == "cross_run_evaluation_contract":
        reason = (
            "The compact-reference pilot produced schema-valid rows, but a static "
            "pre-evaluation provenance review found that the frozen evaluator, "
            "candidate reranker, and semantic-judgment freezer still resolved "
            "English and neutral prerequisites from the style run. Generation "
            "correctly used the separate source run. Execution was stopped before "
            "any style-meter, fidelity, semantic, readability, selection, or "
            "promotion evaluation."
        )
        decision_basis = "static_cross_run_provenance_preflight_only"
        superseded_contract = {
            "source_run_lookup": "generation_only",
            "evaluation_run_lookup": "implicitly_same_as_style_run",
            "source_run_id": args.source_run_id or args.pilot_run_id,
            "style_run_id": args.pilot_run_id,
        }
        replacement_contract = {
            "source_run_lookup": "explicit_end_to_end",
            "source_run_id": args.source_run_id or args.pilot_run_id,
            "style_run_id": args.active_style_run_id,
            "required_consumers": [
                "style_generation",
                "style_critique",
                "deterministic_evaluation",
                "candidate_rerank",
                "semantic_judgment_freeze",
            ],
            "all_fixed_arms_regenerated_from_scratch": True,
        }
    else:
        reason = (
            "The fixed 12-paragraph cross-run pilot produced schema-valid rows in "
            "every arm, but every arm had at least one sample exhaust the frozen "
            "two-attempt limit on paragraph-ID/order validation. The complete-roster "
            "promotion gate was therefore unreachable. Execution was stopped before "
            "any style-meter, fidelity, semantic, readability, selection, or "
            "promotion evaluation."
        )
        superseded_contract = {
            "block_threshold": 12,
            "max_block_paragraphs": 12,
            "model_visible_paragraph_ids": "source_global_ids",
            "failed_block_recovery": "repeat_full_sample_attempt",
            "max_sample_attempts": 2,
            "source_run_id": args.source_run_id or args.pilot_run_id,
            "style_run_id": args.pilot_run_id,
        }
        replacement_contract = {
            "block_threshold": 12,
            "max_block_paragraphs": 12,
            "model_visible_paragraph_ids": "block_local_ids",
            "source_id_restoration": "deterministic_position_preserving_remap",
            "failed_block_recovery": "deterministic_recursive_bisection",
            "minimum_block_paragraphs": 1,
            "max_sample_attempts": 2,
            "source_run_id": args.source_run_id or args.pilot_run_id,
            "style_run_id": args.active_style_run_id,
            "all_fixed_arms_regenerated_from_scratch": True,
        }
    artifact = {
        "schema_version": 1,
        "status": "frozen_before_outcome_evaluation",
        "amendment_type": "operational_style_generation_replacement",
        "sample_set": args.sample_set,
        "source_run_id": args.source_run_id or args.pilot_run_id,
        "superseded_pilot_run_id": args.pilot_run_id,
        "active_style_run_id": args.active_style_run_id,
        "pilot_kind": args.pilot_kind,
        "reason": reason,
        "decision_basis": decision_basis,
        "outcome_evaluation_artifacts_observed": [],
        "superseded_block_contract": superseded_contract,
        "replacement_block_contract": replacement_contract,
        "ledger_summaries": [ledger_summary(path) for path in ledgers],
        "superseded_output_inventory": inventory,
        "superseded_output_inventory_count": len(inventory),
        "exclusion_policy": (
            "Every inventoried pilot artifact is retained for provenance and excluded "
            "from generation resume, scoring, reranking, selection, promotion, and "
            "all reported style-transfer outcomes."
        ),
    }
    content = json.dumps(artifact, ensure_ascii=False, indent=2) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != content:
        raise FileExistsError(f"Refusing to replace different amendment: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "frozen",
                "output": relative(output),
                "sha256": file_sha256(output),
                "inventory_count": len(inventory),
                "ledger_count": len(ledgers),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
