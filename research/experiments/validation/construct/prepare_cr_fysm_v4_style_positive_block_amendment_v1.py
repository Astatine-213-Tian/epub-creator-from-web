#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.validation.construct import run_cr_fysm_v4_construct_generation as generation


AMENDMENT_ID = "CR-FYSM-v4-construct-v2-style-positive-block-completion-v1"
SAMPLE_IDS = (
    "s_730db1b91d5f81a56b5d348d",
    "s_9f3f4ac45c4fd09c2ce5f77a",
)
BLOCK_SIZE = 18
ROOT = generation.ROOT / "protocol_amendment_style_positive_block_v1"
DECLARATION = ROOT / "preregistration.json"
RUNNER = Path("experiments/validation/construct/run_cr_fysm_v4_style_positive_block_amendment_v1.py")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def declaration_id(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("declaration_id", None)
    return generation.sha256_text(generation.canonical(payload))


def block_ids(ids: list[str]) -> list[list[str]]:
    return [ids[start : start + BLOCK_SIZE] for start in range(0, len(ids), BLOCK_SIZE)]


def main() -> None:
    preregistration = generation.validate_preregistration()
    if DECLARATION.exists():
        raise FileExistsError(f"amendment declaration already exists: {DECLARATION}")
    if not RUNNER.exists():
        raise FileNotFoundError(f"amendment runner is missing: {RUNNER}")

    samples = {
        row["sample_id"]: row for row in generation.read_jsonl(generation.SELECTION)
    }
    if not set(SAMPLE_IDS).issubset(samples):
        raise ValueError("one or more amendment samples are absent from the locked selection")
    completed = {
        path.stem
        for path in (generation.ROOT / "generation" / "style_positive_control").glob("*.json")
    }
    expected = set(samples)
    if expected - completed != set(SAMPLE_IDS):
        raise ValueError(
            "style-positive missing set differs from the two predeclared long samples: "
            f"{sorted(expected - completed)}"
        )

    partitions: dict[str, list[list[str]]] = {}
    dependencies: dict[str, dict[str, str]] = {}
    geometry: dict[str, dict[str, int]] = {}
    for sample_id in SAMPLE_IDS:
        sample = samples[sample_id]
        request = generation.build_request("style_positive_control", sample, preregistration)
        ids = generation.expected_ids("style_positive_control", request)
        partitions[sample_id] = block_ids(ids)
        geometry[sample_id] = {
            "paragraphs": len(ids),
            "blocks": len(partitions[sample_id]),
            "maximum_block_paragraphs": max(map(len, partitions[sample_id])),
        }
        dependencies[sample_id] = {
            dependency: generation.load_artifact(dependency, sample, preregistration)[
                "result_sha256"
            ]
            for dependency in generation.DEPENDENCIES["style_positive_control"]
        }

    config = preregistration["generation_stages"]["style_positive_control"]
    declaration: dict[str, Any] = {
        "schema_version": 1,
        "amendment_id": AMENDMENT_ID,
        "status": "declared_before_blockwise_amendment_generation",
        "generation_lock_id": preregistration["lock_id"],
        "construct_id": preregistration["construct_id"],
        "stage": "style_positive_control",
        "sample_ids": list(SAMPLE_IDS),
        "trigger": {
            "canonical_stage_completed": len(completed),
            "canonical_stage_expected": len(expected),
            "failure_class": "paragraph_id_or_order_mismatch_after_repeated_whole-passage_attempts",
            "selection_rule": (
                "all and only locked style-positive items still lacking canonical artifacts"
            ),
            "no_selection_by_style_score": True,
        },
        "method": {
            "partition": "contiguous paragraph-ID blocks, preserving original order",
            "block_size": BLOCK_SIZE,
            "partitions": partitions,
            "stitching": "concatenate validated block outputs in locked paragraph order",
            "prompt": "unchanged locked style-positive prompt",
            "schema": "unchanged locked style-positive schema",
            "model": config["model"],
            "reasoning_effort": config["reasoning_effort"],
            "max_attempts_per_block": 2,
            "post_stitch_validation": "full original request through locked validator",
        },
        "geometry": geometry,
        "dependency_result_sha256": dependencies,
        "prompt_sha256": generation.sha256_file(Path(config["prompt_path"])),
        "schema_sha256": generation.sha256_file(Path(config["schema_path"])),
        "runner_sha256": generation.sha256_file(RUNNER),
        "preparer_sha256": generation.sha256_file(Path(__file__)),
        "claim_language": (
            "The oracle control used a predeclared blockwise structural-completion "
            "amendment for two long items. It is not an unamended whole-passage result."
        ),
    }
    declaration["declaration_id"] = declaration_id(declaration)
    write_json(DECLARATION, declaration)
    print(json.dumps(declaration, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
