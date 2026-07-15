#!/usr/bin/env python3
from __future__ import annotations

"""Freeze blind critique artifacts as independent evaluation judgments."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from experiments.iteration1.style_analysis_lock import (
    require_matching_analysis_binding,
    validate_analysis_lock,
)
from experiments.iteration1.style_experiment_provenance import (
    validate_artifact_and_ledger,
    validate_execution_binding,
    validate_run_config,
)


from experiments.shared.paths import RESEARCH_ROOT


REPO_ROOT = RESEARCH_ROOT
DEFAULT_EXPERIMENT_ROOT = (
    REPO_ROOT / "generated/style_research/style_transfer_experiments"
)
DEFAULT_SAMPLE_SET = "development_proxy_v1"


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def parse_method(value: str) -> tuple[str, str]:
    parts = value.split(":", 1)
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError("Use METHOD:INTENSITY")
    return parts[0], parts[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert separately generated blind style critiques into frozen "
            "semantic/readability judgments for the non-LLM evaluator."
        )
    )
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--sample-set", default=DEFAULT_SAMPLE_SET)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--input-run-id",
        help=(
            "Run containing the frozen English and neutral prerequisites. "
            "Defaults to --run-id."
        ),
    )
    parser.add_argument("--analysis-lock", type=Path, required=True)
    parser.add_argument(
        "--selection-file",
        type=Path,
        required=True,
        help="Frozen JSON file containing the sample_ids array to judge.",
    )
    parser.add_argument(
        "--method",
        type=parse_method,
        action="append",
        required=True,
        metavar="METHOD:INTENSITY",
        help="Candidate with an already completed style_critique stage; repeatable.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-pending",
        action="store_true",
        help="Emit pending rows for missing critique artifacts instead of failing.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Missing required artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def selected_ids(path: Path) -> list[str]:
    payload = load_json(path)
    values = payload.get("sample_ids")
    if not isinstance(values, list) or not values or not all(
        isinstance(value, str) and value for value in values
    ):
        raise ValueError(f"Invalid sample_ids array: {path}")
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate sample IDs: {path}")
    return sorted(values)


def success_ledger_row(path: Path, sample_id: str) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matches = [
        row
        for row in rows
        if row.get("sample_id") == sample_id and row.get("status") == "success"
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one successful ledger row for {sample_id}: {path}")
    return matches[0]


def effective_english_artifact(run_root: Path, sample_id: str) -> dict[str, Any]:
    source = load_json(run_root / "english_semantic_source" / f"{sample_id}.json")
    qa = load_json(run_root / "english_source_qa" / f"{sample_id}.json")
    approved = qa.get("approved_for_neutral_translation")
    if approved is not qa.get("result", {}).get("approved"):
        raise ValueError(f"Initial English QA approval mismatch: {sample_id}")
    if approved is True:
        return source
    found_round = False
    for repair_round in range(1, 21):
        round_name = f"round_{repair_round:02d}"
        repair_path = (
            run_root
            / "english_source_repair"
            / round_name
            / f"{sample_id}.json"
        )
        repair_qa_path = (
            run_root
            / "english_source_repair_qa"
            / round_name
            / f"{sample_id}.json"
        )
        if not repair_path.exists() and not repair_qa_path.exists():
            break
        found_round = True
        if not repair_path.exists() or not repair_qa_path.exists():
            raise ValueError(
                f"Incomplete English repair round {repair_round}: {sample_id}"
            )
        repair = load_json(repair_path)
        repair_qa = load_json(repair_qa_path)
        repair_approved = repair_qa.get("approved_for_neutral_translation")
        if repair_approved is not repair_qa.get("result", {}).get("approved"):
            raise ValueError(
                f"English repair QA approval mismatch in round {repair_round}: "
                f"{sample_id}"
            )
        if repair_approved is True:
            if repair.get("stage") == "english_source_adjudication":
                provenance = repair_qa.get("adjudication_provenance", {})
                audit_value = provenance.get("audit_path")
                if not isinstance(audit_value, str):
                    raise ValueError(f"English adjudication audit missing: {sample_id}")
                audit = load_json(REPO_ROOT / audit_value)
                expected = {
                    "status": "approved_by_independent_adjudication",
                    "sample_id": sample_id,
                    "frozen_repair_path": display_path(repair_path),
                    "frozen_repair_sha256": file_sha256(repair_path),
                    "frozen_qa_path": display_path(repair_qa_path),
                    "frozen_qa_sha256": file_sha256(repair_qa_path),
                }
                if any(audit.get(key) != value for key, value in expected.items()):
                    raise ValueError(
                        f"English adjudication audit mismatch: {sample_id}"
                    )
            return repair
    if found_round:
        raise ValueError(f"No English repair round was approved: {sample_id}")
    raise ValueError(f"Repaired English is missing: {sample_id}")


def validate_review(
    *,
    artifact: dict[str, Any],
    candidate: dict[str, Any],
    sample_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = artifact.get("result")
    if not isinstance(result, dict):
        raise ValueError(f"Critique has no result object: {sample_id}")
    if artifact.get("stage") != "style_critique":
        raise ValueError(f"Unexpected critique stage: {sample_id}")
    if artifact.get("sample_id") != sample_id or result.get("sample_id") != sample_id:
        raise ValueError(f"Critique sample ID mismatch: {sample_id}")
    if result.get("prompt_version") != "style_transfer_critique.v1":
        raise ValueError(f"Unexpected critique prompt version: {sample_id}")
    if artifact.get("output_sha256") != sha256_text(canonical_json(result)):
        raise ValueError(f"Critique output hash mismatch: {sample_id}")

    candidate_result = candidate.get("result")
    if not isinstance(candidate_result, dict):
        raise ValueError(f"Candidate has no result object: {sample_id}")
    candidate_paragraphs = candidate_result.get("paragraphs")
    reviews = result.get("paragraph_reviews")
    neutral_reviews = result.get("neutral_paragraph_reviews")
    if (
        not isinstance(candidate_paragraphs, list)
        or not isinstance(reviews, list)
        or not isinstance(neutral_reviews, list)
    ):
        raise ValueError(f"Missing candidate paragraphs or critique reviews: {sample_id}")
    expected_ids = [row.get("id") for row in candidate_paragraphs]
    observed_ids = [row.get("id") for row in reviews]
    neutral_ids = [row.get("id") for row in neutral_reviews]
    if observed_ids != expected_ids or neutral_ids != expected_ids:
        raise ValueError(f"Critique paragraph IDs/order mismatch: {sample_id}")

    neutral_major_fidelity: list[str] = []
    neutral_minor_fidelity: list[str] = []
    for review in neutral_reviews:
        paragraph_id = str(review.get("id", ""))
        status = review.get("status")
        fidelity = review.get("fidelity_issues")
        if status not in {"pass", "minor", "major"} or not isinstance(fidelity, list):
            raise ValueError(f"Invalid neutral critique row for {sample_id}/{paragraph_id}")
        if status == "pass" and fidelity:
            raise ValueError(f"Passing neutral row carries issues: {sample_id}/{paragraph_id}")
        if status == "major" and not fidelity:
            raise ValueError(f"Major neutral row has no issue: {sample_id}/{paragraph_id}")
        if status == "major" and fidelity:
            neutral_major_fidelity.append(paragraph_id)
        if status == "minor" and fidelity:
            neutral_minor_fidelity.append(paragraph_id)

    major_fidelity: list[str] = []
    major_readability: list[str] = []
    minor_fidelity: list[str] = []
    minor_readability: list[str] = []
    repair_needed = False
    for review in reviews:
        paragraph_id = str(review.get("id", ""))
        status = review.get("status")
        fidelity = review.get("fidelity_issues")
        readability = review.get("readability_issues")
        repairs = review.get("repair_instructions")
        if status not in {"pass", "minor", "major"}:
            raise ValueError(f"Invalid critique status for {sample_id}/{paragraph_id}")
        if not all(isinstance(value, list) for value in (fidelity, readability, repairs)):
            raise ValueError(f"Invalid critique issue arrays for {sample_id}/{paragraph_id}")
        if status == "pass" and (fidelity or readability or repairs):
            raise ValueError(f"Passing critique row carries issues: {sample_id}/{paragraph_id}")
        if status != "pass":
            repair_needed = True
        if status == "major" and not (fidelity or readability):
            raise ValueError(f"Major critique row has no stated issue: {sample_id}/{paragraph_id}")
        if status == "major" and fidelity:
            major_fidelity.append(paragraph_id)
        if status == "major" and readability:
            major_readability.append(paragraph_id)
        if status == "minor" and fidelity:
            minor_fidelity.append(paragraph_id)
        if status == "minor" and readability:
            minor_readability.append(paragraph_id)
    if bool(result.get("repair_required")) != repair_needed:
        raise ValueError(f"Critique repair_required mismatch: {sample_id}")

    verdict = {
        "high_severity_semantic_failure": bool(major_fidelity),
        "high_severity_readability_failure": bool(major_readability),
        "neutral_high_severity_semantic_failure": bool(neutral_major_fidelity),
        "major_fidelity_paragraph_ids": major_fidelity,
        "major_readability_paragraph_ids": major_readability,
        "minor_fidelity_paragraph_ids": minor_fidelity,
        "minor_readability_paragraph_ids": minor_readability,
        "neutral_major_fidelity_paragraph_ids": neutral_major_fidelity,
        "neutral_minor_fidelity_paragraph_ids": neutral_minor_fidelity,
        "repair_required": repair_needed,
    }
    return result, verdict


def render_jsonl(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(canonical_json(row) + "\n" for row in rows)


def main() -> None:
    args = parse_args()
    experiment_root = args.experiment_root.resolve()
    analysis_lock_binding = validate_analysis_lock(
        args.analysis_lock.resolve(), experiment_root
    )
    source_run_id = args.input_run_id or args.run_id
    protocol = load_json(experiment_root / "protocols/evaluation_protocol.v1.json")
    style_execution = protocol.get("iteration3", {}).get("style_execution", {})
    if style_execution:
        if args.run_id != style_execution.get("style_run_id"):
            raise ValueError("Judgment style run does not match the frozen protocol")
        if source_run_id != style_execution.get("source_run_id"):
            raise ValueError("Judgment source run does not match the frozen protocol")
    run_root = experiment_root / "runs" / args.sample_set / args.run_id
    source_root = experiment_root / "runs" / args.sample_set / source_run_id
    selection_file = args.selection_file.resolve()
    ids = selected_ids(selection_file)
    selection_payload = load_json(selection_file)
    selection_id = str(selection_payload.get("selection_id", ""))
    selection_sha256 = file_sha256(selection_file)
    if not selection_id:
        raise ValueError("Selection file has no selection_id")
    methods = list(dict.fromkeys(args.method))
    output = args.output
    if output is None:
        output = run_root / "independent_judgments" / "judgments.v1.jsonl"
    output = output.resolve()

    rows: list[dict[str, Any]] = []
    for method_id, intensity in methods:
        critique_config_path = (
            run_root
            / "run_configs"
            / f"style_critique.{method_id}.{intensity}.json"
        )
        critique_config = load_json(critique_config_path)
        _, critique_config_errors = validate_run_config(
            critique_config_path,
            expected={
                "sample_set": args.sample_set,
                "run_id": args.run_id,
                "stage": "style_critique",
                "base_method_id": method_id,
                "base_intensity": intensity,
                "input_run_id": source_run_id,
            },
            expected_analysis_lock=analysis_lock_binding,
        )
        if critique_config_errors:
            raise ValueError(
                "Critique run-config provenance failed: "
                + ", ".join(critique_config_errors)
            )
        if critique_config.get("stage") != "style_critique":
            raise ValueError(f"Unexpected critique run config: {critique_config_path}")
        if critique_config.get("base_method_id") != method_id:
            raise ValueError(f"Critique base method mismatch: {critique_config_path}")
        if critique_config.get("base_intensity") != intensity:
            raise ValueError(f"Critique base intensity mismatch: {critique_config_path}")

        for sample_id in ids:
            critique_path = (
                run_root / "style_critiques" / method_id / intensity / f"{sample_id}.json"
            )
            candidate_path = (
                run_root / "method_outputs" / method_id / intensity / f"{sample_id}.json"
            )
            if not critique_path.exists():
                if not args.allow_pending:
                    raise ValueError(f"Missing critique artifact: {critique_path}")
                rows.append(
                    {
                        "schema_version": 1,
                        "sample_id": sample_id,
                        "method_id": method_id,
                        "intensity": intensity,
                        "status": "pending",
                        "analysis_lock": analysis_lock_binding,
                        "high_severity_semantic_failure": None,
                        "high_severity_readability_failure": None,
                        "neutral_high_severity_semantic_failure": None,
                        "judgment_rule": "blind_critique_major_issue.v1",
                        "selection_sha256": selection_sha256,
                    }
                )
                continue
            artifact = load_json(critique_path)
            candidate = load_json(candidate_path)
            for label, generated in (
                ("critique artifact", artifact),
                ("candidate artifact", candidate),
            ):
                admission = generated.get("execution_admission")
                if not isinstance(admission, dict):
                    raise ValueError(f"{label} has no execution admission: {sample_id}")
                require_matching_analysis_binding(
                    admission, analysis_lock_binding, label=label
                )
            neutral_path = source_root / "neutral_translation" / f"{sample_id}.json"
            neutral = load_json(neutral_path)
            english = effective_english_artifact(source_root, sample_id)
            english_paragraphs = english.get("result", {}).get("paragraphs")
            neutral_paragraphs = neutral.get("result", {}).get("paragraphs")
            candidate_paragraphs = candidate.get("result", {}).get("paragraphs")
            if not all(
                isinstance(value, list)
                for value in (
                    english_paragraphs,
                    neutral_paragraphs,
                    candidate_paragraphs,
                )
            ):
                raise ValueError(f"Cannot reconstruct critique input: {sample_id}")
            critique_request = {
                "sample_id": sample_id,
                "english_semantic_source": english_paragraphs,
                "neutral_zh": neutral_paragraphs,
                "candidate_zh": candidate_paragraphs,
            }
            expected_input_sha256 = sha256_text(canonical_json(critique_request))
            expected_candidate_sha256 = sha256_text(
                canonical_json(candidate_paragraphs)
            )
            if artifact.get("input_sha256") != expected_input_sha256:
                raise ValueError(f"Critique input hash mismatch: {sample_id}")
            if artifact.get("candidate_sha256") != expected_candidate_sha256:
                raise ValueError(f"Critique candidate hash mismatch: {sample_id}")
            if artifact.get("base_method_id") != method_id:
                raise ValueError(f"Critique base method mismatch: {sample_id}")
            if artifact.get("base_intensity") != intensity:
                raise ValueError(f"Critique base intensity mismatch: {sample_id}")
            critique_ledger_path = (
                run_root
                / "ledgers"
                / f"style_critique.{method_id}.{intensity}.jsonl"
            )
            ledger_row = success_ledger_row(critique_ledger_path, sample_id)
            candidate_ledger_path = (
                run_root
                / "ledgers"
                / f"style_transfer.{method_id}.{intensity}.jsonl"
            )
            candidate_config_path = (
                run_root
                / "run_configs"
                / f"style_transfer.{method_id}.{intensity}.json"
            )
            candidate_config = load_json(candidate_config_path)
            if candidate_config.get("input_run_id") != source_run_id:
                raise ValueError(
                    f"Candidate source-run mismatch: {candidate_config_path}"
                )
            candidate_ledger_row = success_ledger_row(
                candidate_ledger_path, sample_id
            )
            neutral_ledger_path = source_root / "ledgers/neutral_translation.jsonl"
            neutral_config_path = source_root / "run_configs/neutral_translation.json"
            neutral_ledger_row = success_ledger_row(neutral_ledger_path, sample_id)
            provenance_checks = (
                (
                    "critique",
                    critique_path,
                    artifact,
                    ledger_row,
                    critique_config_path,
                    "style_critique",
                ),
                (
                    "candidate",
                    candidate_path,
                    candidate,
                    candidate_ledger_row,
                    candidate_config_path,
                    "style_transfer",
                ),
                (
                    "neutral",
                    neutral_path,
                    neutral,
                    neutral_ledger_row,
                    neutral_config_path,
                    "neutral_translation",
                ),
            )
            for (
                label,
                artifact_path,
                generated_artifact,
                generated_ledger,
                config_path,
                stage,
            ) in provenance_checks:
                provenance_errors = validate_artifact_and_ledger(
                    artifact_path=artifact_path,
                    artifact=generated_artifact,
                    ledger_row=generated_ledger,
                    run_config_path=config_path,
                    sample_id=sample_id,
                    run_id=source_run_id if label == "neutral" else args.run_id,
                    stage=stage,
                    expected_analysis_lock=(
                        analysis_lock_binding
                        if label in {"critique", "candidate"}
                        else None
                    ),
                )
                binding_kwargs: dict[str, Any] = {
                    "sample_id": sample_id,
                    "expected_run_config_path": config_path,
                }
                if label in {"critique", "candidate"}:
                    binding_kwargs.update(
                        {
                            "expected_selection_id": selection_id,
                            "expected_selection_sha256": selection_sha256,
                            "expected_analysis_lock": analysis_lock_binding,
                        }
                    )
                provenance_errors.extend(
                    validate_execution_binding(
                        generated_artifact,
                        **binding_kwargs,
                    )
                )
                if provenance_errors:
                    raise ValueError(
                        f"{label} provenance failed for {sample_id}: "
                        + ", ".join(sorted(set(provenance_errors)))
                    )
            ledger_expectations = {
                "run_id": args.run_id,
                "stage": "style_critique",
                "input_sha256": expected_input_sha256,
                "output_sha256": artifact.get("output_sha256"),
                "candidate_sha256": expected_candidate_sha256,
                "base_method_id": method_id,
                "base_intensity": intensity,
                "response_id": artifact.get("response_id"),
            }
            for field, expected in ledger_expectations.items():
                if ledger_row.get(field) != expected:
                    raise ValueError(
                        f"Critique ledger {field} mismatch: {sample_id}"
                    )
            if Path(str(ledger_row.get("response_file", ""))).resolve() != critique_path:
                recorded = REPO_ROOT / str(ledger_row.get("response_file", ""))
                if recorded.resolve() != critique_path:
                    raise ValueError(f"Critique ledger response path mismatch: {sample_id}")
            _, verdict = validate_review(
                artifact=artifact,
                candidate=candidate,
                sample_id=sample_id,
            )
            rows.append(
                {
                    "schema_version": 1,
                    "sample_id": sample_id,
                    "method_id": method_id,
                    "intensity": intensity,
                    "source_run_id": source_run_id,
                    "status": "complete",
                    "analysis_lock": analysis_lock_binding,
                    **verdict,
                    "judgment_rule": "blind_critique_major_issue.v1",
                    "selection_sha256": selection_sha256,
                    "critique_artifact_path": display_path(critique_path),
                    "critique_artifact_sha256": file_sha256(critique_path),
                    "critique_result_sha256": artifact["output_sha256"],
                    "candidate_artifact_sha256": file_sha256(candidate_path),
                    "candidate_output_sha256": candidate.get("output_sha256"),
                    "candidate_sha256": expected_candidate_sha256,
                    "candidate_ledger_path": display_path(candidate_ledger_path),
                    "candidate_ledger_row_sha256": sha256_text(
                        canonical_json(candidate_ledger_row)
                    ),
                    "candidate_run_config_path": display_path(
                        candidate_config_path
                    ),
                    "candidate_run_config_sha256": file_sha256(
                        candidate_config_path
                    ),
                    "neutral_artifact_sha256": file_sha256(neutral_path),
                    "neutral_output_sha256": neutral.get("output_sha256"),
                    "neutral_ledger_path": display_path(neutral_ledger_path),
                    "neutral_ledger_row_sha256": sha256_text(
                        canonical_json(neutral_ledger_row)
                    ),
                    "neutral_run_config_path": display_path(neutral_config_path),
                    "neutral_run_config_sha256": file_sha256(neutral_config_path),
                    "effective_english_artifact_canonical_sha256": sha256_text(
                        canonical_json(english)
                    ),
                    "effective_english_output_sha256": english.get("output_sha256"),
                    "critique_input_sha256": expected_input_sha256,
                    "critique_ledger_path": display_path(critique_ledger_path),
                    "critique_ledger_row_sha256": sha256_text(
                        canonical_json(ledger_row)
                    ),
                    "response_id": artifact.get("response_id"),
                    "judge_model": critique_config.get("model"),
                    "judge_reasoning_effort": critique_config.get("reasoning_effort"),
                    "judge_prompt_sha256": critique_config.get("prompt_sha256"),
                    "judge_run_config_path": display_path(critique_config_path),
                    "judge_run_config_sha256": file_sha256(critique_config_path),
                }
            )

    content = render_jsonl(rows)
    if output.exists():
        if output.read_text(encoding="utf-8") != content:
            raise ValueError(f"Refusing to replace different frozen judgments: {output}")
        status = "reused_identical"
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
        status = "created"
    print(
        json.dumps(
            {
                "status": status,
                "rows": len(rows),
                "methods": [f"{method}:{intensity}" for method, intensity in methods],
                "source_run_id": source_run_id,
                "selection_file": display_path(selection_file),
                "selection_sha256": selection_sha256,
                "analysis_lock": analysis_lock_binding,
                "output": display_path(output),
                "output_sha256": file_sha256(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
