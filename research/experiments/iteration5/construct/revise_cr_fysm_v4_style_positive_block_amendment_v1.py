#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.iteration5.construct import prepare_cr_fysm_v4_style_positive_block_amendment_v1 as preparation
from experiments.iteration5.construct import run_cr_fysm_v4_construct_generation as generation


OLD_DECLARATION_ID = "ab20eb77fe5a191925b459862ba81c611b9b15d94774fee95ff9a5182073f526"
OLD_RUNNER_SHA256 = "977000d3d948eaad89f7519e82bd3d570c38b87d9d626ac03536093066b5755d"
AUDITOR_AGENT_ID = "019f5fcd-d5d2-7cd3-91be-b580f6602fc0"
AUDIT = preparation.ROOT / "independent_audit_pre_run.md"
RUNNER = Path("experiments/iteration5/construct/run_cr_fysm_v4_style_positive_block_amendment_v1.py")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    declaration = read_json(preparation.DECLARATION)
    if declaration.get("status") != "declared_before_blockwise_amendment_generation":
        raise ValueError("original pre-run declaration status changed")
    if declaration.get("declaration_id") != OLD_DECLARATION_ID:
        raise ValueError("original pre-run declaration ID changed")
    if declaration.get("runner_sha256") != OLD_RUNNER_SHA256:
        raise ValueError("original pre-run runner hash changed in the declaration")
    if not AUDIT.is_file():
        raise FileNotFoundError("independent pre-run audit record is missing")
    raw_paths = list((preparation.ROOT / "raw_attempts").glob("*/*/*.json"))
    if raw_paths:
        raise ValueError("cannot revise after block-amendment raw attempts exist")
    completed = [
        sample_id
        for sample_id in preparation.SAMPLE_IDS
        if generation.output_path("style_positive_control", sample_id).exists()
    ]
    if completed:
        raise ValueError(f"cannot revise after canonical amendment outputs exist: {completed}")

    declaration["status"] = (
        "revised_before_blockwise_amendment_generation_after_independent_no_go"
    )
    declaration["runner_sha256"] = generation.sha256_file(RUNNER)
    declaration["revision_script_sha256"] = generation.sha256_file(Path(__file__))
    declaration["pre_run_revision"] = {
        "supersedes_declaration_id": OLD_DECLARATION_ID,
        "supersedes_runner_sha256": OLD_RUNNER_SHA256,
        "auditor_agent_id": AUDITOR_AGENT_ID,
        "auditor_verdict": "NO-GO",
        "audit_path": str(AUDIT),
        "audit_sha256": generation.sha256_file(AUDIT),
        "blocking_finding": "declared dependency hashes were not enforced at runtime",
        "correction": (
            "validate_declaration now recomputes every selected sample's declared "
            "upstream result hashes and requires exact equality before generation"
        ),
        "raw_attempts_before_revision": 0,
        "canonical_outputs_before_revision": 0,
        "method_selection_and_partitions_changed": False,
    }
    declaration["declaration_id"] = preparation.declaration_id(declaration)
    write_json(preparation.DECLARATION, declaration)
    print(
        json.dumps(
            {
                "status": declaration["status"],
                "declaration_id": declaration["declaration_id"],
                "runner_sha256": declaration["runner_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
