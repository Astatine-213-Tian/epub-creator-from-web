#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


from experiments.shared.paths import RESEARCH_ROOT


REPO_ROOT = RESEARCH_ROOT


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to replace adjudication artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def failed_review_ids(qa_artifact: dict[str, Any]) -> list[str]:
    reviews = qa_artifact.get("result", {}).get("paragraph_reviews", [])
    if not isinstance(reviews, list):
        raise ValueError("QA paragraph_reviews must be a list")
    return sorted(
        str(row.get("id"))
        for row in reviews
        if isinstance(row, dict) and row.get("status") == "fail"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a one-paragraph independent adjudication after repeated "
            "English-source repair QA rejection."
        )
    )
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--sample-set", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--source-repair-round", type=int, required=True)
    parser.add_argument("--adjudication-round", type=int, required=True)
    parser.add_argument("--paragraph-id", required=True)
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--agent-model", required=True)
    parser.add_argument("--agent-reasoning-effort", default="medium")
    parser.add_argument("--verdict", choices=("PASS", "FAIL"), required=True)
    parser.add_argument("--recommended-rendering", required=True)
    parser.add_argument("--rationale", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source_repair_round < 2:
        raise ValueError("Adjudication requires at least two rejected repair-QA rounds")
    if args.adjudication_round != args.source_repair_round + 1:
        raise ValueError("Adjudication round must immediately follow the source repair round")
    if args.verdict != "PASS":
        raise ValueError("Only an independent PASS verdict can approve a source")
    rendering = args.recommended_rendering.strip()
    if not rendering:
        raise ValueError("Recommended rendering is empty")

    root = args.experiment_root.expanduser().resolve()
    run_root = root / "runs" / args.sample_set / args.run_id
    sample_id = args.sample_id
    source_round = f"round_{args.source_repair_round:02d}"
    prior_round = f"round_{args.source_repair_round - 1:02d}"
    adjudication_round = f"round_{args.adjudication_round:02d}"
    source_path = run_root / "english_source_repair" / source_round / f"{sample_id}.json"
    prior_qa_path = (
        run_root / "english_source_repair_qa" / prior_round / f"{sample_id}.json"
    )
    source_qa_path = (
        run_root / "english_source_repair_qa" / source_round / f"{sample_id}.json"
    )
    source = read_json(source_path)
    prior_qa = read_json(prior_qa_path)
    source_qa = read_json(source_qa_path)
    if prior_qa.get("approved_for_neutral_translation") is not False:
        raise ValueError("Prior QA must be rejected for adjudication")
    if source_qa.get("approved_for_neutral_translation") is not False:
        raise ValueError("Latest QA must be rejected for adjudication")
    if args.paragraph_id not in failed_review_ids(prior_qa):
        raise ValueError("Adjudicated paragraph was not rejected in the prior QA round")
    if failed_review_ids(source_qa) != [args.paragraph_id]:
        raise ValueError("Latest QA must reject only the adjudicated paragraph")

    source_result = source.get("result")
    if not isinstance(source_result, dict):
        raise ValueError("Source repair result is missing")
    paragraphs = source_result.get("paragraphs")
    if not isinstance(paragraphs, list):
        raise ValueError("Source repair paragraphs are missing")
    corrected_paragraphs: list[dict[str, Any]] = []
    replacement_count = 0
    for row in paragraphs:
        if not isinstance(row, dict):
            raise ValueError("Source repair paragraph is not an object")
        corrected = dict(row)
        if corrected.get("id") == args.paragraph_id:
            corrected["en"] = rendering
            replacement_count += 1
        corrected_paragraphs.append(corrected)
    if replacement_count != 1:
        raise ValueError("Adjudicated paragraph is missing or duplicated")
    corrected_result = {**source_result, "paragraphs": corrected_paragraphs}

    qa_reviews = [dict(row) for row in source_qa["result"]["paragraph_reviews"]]
    for row in qa_reviews:
        if row.get("id") == args.paragraph_id:
            row["status"] = "pass"
            row["issues"] = []
    qa_result = {
        **source_qa["result"],
        "approved": True,
        "paragraph_reviews": qa_reviews,
        "overall_issues": [],
    }

    repair_destination = (
        run_root / "english_source_repair" / adjudication_round / f"{sample_id}.json"
    )
    qa_destination = (
        run_root / "english_source_repair_qa" / adjudication_round / f"{sample_id}.json"
    )
    audit_path = root / "audits" / f"english_source_adjudication.{sample_id}.v1.json"
    provenance = {
        "agent_id": args.agent_id,
        "agent_model": args.agent_model,
        "agent_reasoning_effort": args.agent_reasoning_effort,
        "verdict": args.verdict,
        "source_repair_path": relative(source_path),
        "source_repair_sha256": file_sha256(source_path),
    }
    repair_artifact = {
        **source,
        "stage": "english_source_adjudication",
        "repair_round": args.adjudication_round,
        "model": args.agent_model,
        "reasoning_effort": args.agent_reasoning_effort,
        "result": corrected_result,
        "output_sha256": sha256_text(canonical_json(corrected_result)),
        "adjudication_provenance": provenance,
    }
    qa_artifact = {
        "schema_version": 1,
        "run_id": args.run_id,
        "stage": "english_source_adjudication_qa",
        "sample_id": sample_id,
        "repair_round": args.adjudication_round,
        "model": args.agent_model,
        "reasoning_effort": args.agent_reasoning_effort,
        "approved_for_neutral_translation": True,
        "output_sha256": sha256_text(canonical_json(qa_result)),
        "result": qa_result,
        "adjudication_provenance": {
            **provenance,
            "recommended_rendering": rendering,
            "rationale": args.rationale,
            "audit_path": relative(audit_path),
        },
    }
    write_json(repair_destination, repair_artifact)
    write_json(qa_destination, qa_artifact)
    audit = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "approved_by_independent_adjudication",
        "sample_id": sample_id,
        "paragraph_id": args.paragraph_id,
        "agent_id": args.agent_id,
        "agent_model": args.agent_model,
        "agent_reasoning_effort": args.agent_reasoning_effort,
        "verdict": args.verdict,
        "recommended_rendering": rendering,
        "rationale": args.rationale,
        "source_repair_path": relative(source_path),
        "source_repair_sha256": file_sha256(source_path),
        "contradictory_qa_paths": [relative(prior_qa_path), relative(source_qa_path)],
        "contradictory_qa_sha256": [
            file_sha256(prior_qa_path),
            file_sha256(source_qa_path),
        ],
        "frozen_repair_path": relative(repair_destination),
        "frozen_repair_sha256": file_sha256(repair_destination),
        "frozen_qa_path": relative(qa_destination),
        "frozen_qa_sha256": file_sha256(qa_destination),
        "claim_scope": (
            "This resolves one culturally specific slur rendering after repeated "
            "QA rejection; it does not relax any other English-source QA decision."
        ),
    }
    write_json(audit_path, audit)
    ledger_path = (
        run_root
        / "ledgers"
        / f"english_source_adjudication.round_{args.adjudication_round:02d}.jsonl"
    )
    if ledger_path.exists():
        raise FileExistsError(f"Refusing to replace adjudication ledger: {ledger_path}")
    ledger_path.write_text(canonical_json(audit) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
