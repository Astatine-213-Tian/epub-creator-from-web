#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.iteration5.construct import prepare_cr_fysm_v4_style_positive_block_amendment_v1 as block_preparation
from experiments.iteration5.construct import prepare_cr_fysm_v4_style_positive_stitch_amendment_v2 as preparation
from experiments.iteration5.construct import run_cr_fysm_v4_construct_generation as generation
from experiments.iteration5.construct import run_cr_fysm_v4_style_positive_block_amendment_v1 as block_runner


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stable_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def validate_declaration(preregistration: dict[str, Any]) -> dict[str, Any]:
    declaration = read_json(preparation.DECLARATION)
    if declaration.get("status") != "declared_before_deterministic_stitch_completion":
        raise ValueError("stitch amendment is not predeclared")
    if declaration.get("amendment_id") != preparation.AMENDMENT_ID:
        raise ValueError("stitch amendment ID mismatch")
    if declaration.get("declaration_id") != preparation.declaration_id(declaration):
        raise ValueError("stitch amendment declaration ID is invalid")
    if declaration.get("generation_lock_id") != preregistration["lock_id"]:
        raise ValueError("stitch amendment does not bind the generation lock")
    if declaration.get("runner_sha256") != generation.sha256_file(Path(__file__)):
        raise ValueError("stitch amendment runner changed after declaration")
    parent = read_json(block_preparation.DECLARATION)
    parent_meta = declaration["parent_amendment"]
    if parent_meta.get("declaration_id") != parent.get("declaration_id"):
        raise ValueError("parent block declaration changed")
    if parent_meta.get("declaration_sha256") != generation.sha256_file(
        block_preparation.DECLARATION
    ):
        raise ValueError("parent block declaration hash changed")
    if parent_meta.get("runner_sha256") != generation.sha256_file(Path(block_runner.__file__)):
        raise ValueError("parent block runner changed")
    observed_raw_hashes = {
        str(path): generation.sha256_file(path)
        for path in sorted(block_runner.RAW_ATTEMPTS.glob("*/*/*.json"), key=str)
    }
    if declaration.get("all_parent_raw_attempt_hashes") != observed_raw_hashes:
        raise ValueError("parent raw attempt set or content changed")
    return declaration


def aggregate_sample(
    *,
    sample: dict[str, Any],
    preregistration: dict[str, Any],
    declaration: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stage = "style_positive_control"
    sample_id = sample["sample_id"]
    request = generation.build_request(stage, sample, preregistration)
    paragraphs: list[dict[str, str]] = []
    cues: list[str] = []
    uncertainties: list[str] = []
    lineage: list[dict[str, Any]] = []
    for block in declaration["accepted_block_lineage"][sample_id]:
        path = Path(block["raw_attempt_path"])
        if block["raw_attempt_sha256"] != generation.sha256_file(path):
            raise ValueError(f"accepted block hash changed: {path}")
        record = read_json(path)
        result = record.get("raw_result")
        if not isinstance(result, dict) or record.get("validation_errors"):
            raise ValueError(f"accepted block no longer validates: {path}")
        if block["raw_result_sha256"] != generation.sha256_text(
            generation.canonical(result)
        ):
            raise ValueError(f"accepted block result hash changed: {path}")
        block_request = block_runner.block_request(request, block["paragraph_ids"])
        errors = generation.validate_result(stage, block_request, result, preregistration)
        if errors:
            raise ValueError(f"accepted block failed locked validation: {path}: {errors}")
        paragraphs.extend(result["paragraphs"])
        cues.extend(result["style_cues_applied"])
        uncertainties.extend(result["uncertainties"])
        lineage.append(block)
    aggregated = {
        "sample_id": sample_id,
        "paragraphs": paragraphs,
        "style_cues_applied": stable_unique(cues),
        "uncertainties": stable_unique(uncertainties),
    }
    errors = generation.validate_result(stage, request, aggregated, preregistration)
    if errors:
        raise ValueError(f"aggregated output failed locked validation: {sample_id}: {errors}")
    return aggregated, lineage


def main() -> None:
    preregistration = generation.validate_preregistration()
    declaration = validate_declaration(preregistration)
    samples = {
        row["sample_id"]: row for row in generation.read_jsonl(generation.SELECTION)
    }
    for sample_id in preparation.SAMPLE_IDS:
        destination = generation.output_path("style_positive_control", sample_id)
        if destination.exists():
            raise FileExistsError(f"canonical style-positive output already exists: {destination}")
    for sample_id in preparation.SAMPLE_IDS:
        sample = samples[sample_id]
        result, lineage = aggregate_sample(
            sample=sample,
            preregistration=preregistration,
            declaration=declaration,
        )
        stage = "style_positive_control"
        request = generation.build_request(stage, sample, preregistration)
        config = preregistration["generation_stages"][stage]
        artifact = {
            "schema_version": 2,
            "status": "complete",
            "construct_id": preregistration["construct_id"],
            "generation_lock_id": preregistration["lock_id"],
            "stage": stage,
            "sample_id": sample_id,
            "attempt": "deterministic_stitch_amendment_v2",
            "model": config["model"],
            "reasoning_effort": config["reasoning_effort"],
            "prompt_sha256": generation.sha256_file(Path(config["prompt_path"])),
            "schema_sha256": generation.sha256_file(Path(config["schema_path"])),
            "request_sha256": generation.sha256_text(generation.canonical(request)),
            "dependency_result_sha256": {
                dependency: generation.load_artifact(
                    dependency, sample, preregistration
                )["result_sha256"]
                for dependency in generation.DEPENDENCIES[stage]
            },
            "result_sha256": generation.sha256_text(generation.canonical(result)),
            "response_id": [read_json(Path(row["raw_attempt_path"]))["response_id"] for row in lineage],
            "usage": [read_json(Path(row["raw_attempt_path"]))["usage"] for row in lineage],
            "prior_attempt_errors": [
                "whole-passage generation failed paragraph-ID/order validation",
                "reviewed block runner omitted required top-level metadata during stitching",
            ],
            "protocol_amendment": {
                "amendment_id": preparation.AMENDMENT_ID,
                "declaration_id": declaration["declaration_id"],
                "parent_amendment_id": declaration["parent_amendment"]["amendment_id"],
                "parent_declaration_id": declaration["parent_amendment"]["declaration_id"],
                "method": "deterministic_block_and_metadata_stitch_without_new_model_calls",
                "accepted_blocks": lineage,
            },
            "result": result,
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        write_json(temporary, artifact)
        temporary.replace(destination)
        generation.validate_artifact(stage, sample, preregistration, request, read_json(destination))
        print(f"[amended] {sample_id}", flush=True)
    print(
        json.dumps(
            {
                "amendment_id": preparation.AMENDMENT_ID,
                "expected": len(preparation.SAMPLE_IDS),
                "amended": len(preparation.SAMPLE_IDS),
                "new_model_calls": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
