#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.validation.construct import prepare_cr_fysm_v4_style_positive_block_amendment_v1 as block_preparation
from experiments.validation.construct import run_cr_fysm_v4_construct_generation as generation
from experiments.validation.construct import run_cr_fysm_v4_style_positive_block_amendment_v1 as block_runner


AMENDMENT_ID = "CR-FYSM-v4-construct-v2-style-positive-stitch-completion-v2"
SAMPLE_IDS = block_preparation.SAMPLE_IDS
ROOT = generation.ROOT / "protocol_amendment_style_positive_stitch_v2"
DECLARATION = ROOT / "preregistration.json"
RUNNER = Path("experiments/validation/construct/run_cr_fysm_v4_style_positive_stitch_amendment_v2.py")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def declaration_id(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("declaration_id", None)
    return generation.sha256_text(canonical(payload))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def first_valid_raw(
    *, sample_id: str, block_number: int, declared_ids: list[str]
) -> tuple[Path, dict[str, Any]]:
    directory = block_runner.RAW_ATTEMPTS / sample_id / f"block_{block_number:02d}"
    paths = sorted(directory.glob("attempt_*.json"))
    if not paths:
        raise FileNotFoundError(f"no raw attempts for {sample_id} block {block_number}")
    for path in paths:
        record = read_json(path)
        result = record.get("raw_result")
        if record.get("validation_errors") or not isinstance(result, dict):
            continue
        if [row.get("id") for row in result.get("paragraphs", [])] == declared_ids:
            return path, record
    raise ValueError(f"no valid raw attempt for {sample_id} block {block_number}")


def main() -> None:
    preregistration = generation.validate_preregistration()
    if DECLARATION.exists():
        raise FileExistsError(f"stitch amendment declaration already exists: {DECLARATION}")
    if not RUNNER.is_file():
        raise FileNotFoundError(f"stitch amendment runner is missing: {RUNNER}")
    parent = read_json(block_preparation.DECLARATION)
    if parent.get("status") != (
        "revised_before_blockwise_amendment_generation_after_independent_no_go"
    ):
        raise ValueError("parent block amendment is not the reviewed revision")
    if parent.get("declaration_id") != (
        "b962b772b9bde16d3494539ede2dd777b31ae93e8a438e75d6ecbd87482d01c1"
    ):
        raise ValueError("unexpected parent block amendment declaration")
    completed = [
        sample_id
        for sample_id in SAMPLE_IDS
        if generation.output_path("style_positive_control", sample_id).exists()
    ]
    if completed:
        raise ValueError(f"canonical style-positive outputs already exist: {completed}")

    accepted: dict[str, list[dict[str, Any]]] = {}
    all_raw_hashes: dict[str, str] = {}
    for path in sorted(block_runner.RAW_ATTEMPTS.glob("*/*/*.json"), key=str):
        all_raw_hashes[str(path)] = generation.sha256_file(path)
    for sample_id in SAMPLE_IDS:
        accepted[sample_id] = []
        partitions = parent["method"]["partitions"][sample_id]
        for block_number, ids in enumerate(partitions, start=1):
            path, record = first_valid_raw(
                sample_id=sample_id,
                block_number=block_number,
                declared_ids=ids,
            )
            accepted[sample_id].append(
                {
                    "block_number": block_number,
                    "paragraph_ids": ids,
                    "accepted_attempt": record["attempt"],
                    "raw_attempt_path": str(path),
                    "raw_attempt_sha256": generation.sha256_file(path),
                    "raw_result_sha256": record["raw_result_sha256"],
                    "selection_rule": "first attempt passing the locked block validator",
                }
            )
    declaration: dict[str, Any] = {
        "schema_version": 1,
        "amendment_id": AMENDMENT_ID,
        "status": "declared_before_deterministic_stitch_completion",
        "generation_lock_id": preregistration["lock_id"],
        "construct_id": preregistration["construct_id"],
        "stage": "style_positive_control",
        "sample_ids": list(SAMPLE_IDS),
        "trigger": {
            "block_generation_complete": True,
            "canonical_outputs_written": 0,
            "observed_stitch_validation_errors": [
                "JSON Schema: $: missing required property style_cues_applied",
                "JSON Schema: $: missing required property uncertainties",
            ],
            "failure_class": "deterministic_stitch_implementation_omitted_required_metadata",
        },
        "parent_amendment": {
            "amendment_id": parent["amendment_id"],
            "declaration_id": parent["declaration_id"],
            "declaration_path": str(block_preparation.DECLARATION),
            "declaration_sha256": generation.sha256_file(block_preparation.DECLARATION),
            "runner_path": str(Path(block_runner.__file__)),
            "runner_sha256": generation.sha256_file(Path(block_runner.__file__)),
        },
        "accepted_block_lineage": accepted,
        "all_parent_raw_attempt_hashes": all_raw_hashes,
        "method": {
            "new_model_calls": 0,
            "paragraphs": "concatenate accepted blocks in declared block order",
            "style_cues_applied": (
                "flatten accepted block lists in declared block order and retain the "
                "first occurrence of each exact string"
            ),
            "uncertainties": (
                "flatten accepted block lists in declared block order and retain the "
                "first occurrence of each exact string"
            ),
            "full_validation": "locked whole-request validator after aggregation",
        },
        "runner_sha256": generation.sha256_file(RUNNER),
        "preparer_sha256": generation.sha256_file(Path(__file__)),
        "claim_language": (
            "Two long oracle-control items use reviewed block generation plus a "
            "post-failure predeclared deterministic metadata stitch. No new model calls "
            "were made for the stitch correction."
        ),
    }
    declaration["declaration_id"] = declaration_id(declaration)
    write_json(DECLARATION, declaration)
    print(json.dumps(declaration, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
