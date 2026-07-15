#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.validation.construct import prepare_cr_fysm_v4_style_positive_block_amendment_v1 as block_v1
from experiments.validation.construct import prepare_cr_fysm_v4_style_positive_stitch_amendment_v2 as stitch_v2
from experiments.validation.construct import run_cr_fysm_v4_construct_generation as generation


ROOT = generation.ROOT / "protocol_incident_style_positive_completion_v1"
OUTPUT = ROOT / "incident.json"
WHOLE_PASSAGE_ID = "s_730db1b91d5f81a56b5d348d"
STITCHED_ID = "s_9f3f4ac45c4fd09c2ce5f77a"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    preregistration = generation.validate_preregistration()
    if OUTPUT.exists():
        raise FileExistsError(f"incident record already exists: {OUTPUT}")
    samples = {
        row["sample_id"]: row for row in generation.read_jsonl(generation.SELECTION)
    }
    whole_path = generation.output_path("style_positive_control", WHOLE_PASSAGE_ID)
    stitched_path = generation.output_path("style_positive_control", STITCHED_ID)
    whole = read_json(whole_path)
    stitched = read_json(stitched_path)
    for sample_id, artifact in (
        (WHOLE_PASSAGE_ID, whole),
        (STITCHED_ID, stitched),
    ):
        request = generation.build_request(
            "style_positive_control", samples[sample_id], preregistration
        )
        generation.validate_artifact(
            "style_positive_control",
            samples[sample_id],
            preregistration,
            request,
            artifact,
        )
    if whole.get("protocol_amendment") is not None or whole.get("attempt") != 1:
        raise ValueError("unexpected final whole-passage lineage")
    stitched_meta = stitched.get("protocol_amendment")
    if not isinstance(stitched_meta, dict):
        raise ValueError("stitched final artifact lacks protocol amendment metadata")
    stitch_declaration = read_json(stitch_v2.DECLARATION)
    if stitched_meta.get("declaration_id") != stitch_declaration["declaration_id"]:
        raise ValueError("stitched final declaration lineage mismatch")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "incident_id_text": "CR-FYSM-v4-style-positive-completion-destination-reuse-v1",
        "status": "recorded_before_generation_closure_and_before_any_rating_or_scoring",
        "generation_lock_id": preregistration["lock_id"],
        "affected_sample_ids": [WHOLE_PASSAGE_ID, STITCHED_ID],
        "affected_source_roles": {
            sample_id: samples[sample_id]["source_role"]
            for sample_id in (WHOLE_PASSAGE_ID, STITCHED_ID)
        },
        "observed_sequence": [
            "The reviewed block runner generated and retained all eight raw attempts.",
            "Its stitch omitted two required metadata fields, so no canonical outputs were written.",
            "The reviewed v2 deterministic stitch was run with zero model calls.",
            "A destination-variable reuse defect wrote both loop iterations to the second sample path; the second iteration replaced the transient first-sample content.",
            "The subsequent full-stage --resume validation found the first sample absent and made one additional whole-passage call under the original locked prompt/model/schema.",
            "That call passed the locked validator; the second sample retained its v2 deterministic stitched artifact.",
        ],
        "root_cause": {
            "file": "experiments/validation/construct/run_cr_fysm_v4_style_positive_stitch_amendment_v2.py",
            "file_sha256": generation.sha256_file(
                Path("experiments/validation/construct/run_cr_fysm_v4_style_positive_stitch_amendment_v2.py")
            ),
            "defect": (
                "destination was assigned in the preflight loop and not reassigned in "
                "the write loop, so both writes targeted the final sample path"
            ),
        },
        "verification_command": (
            "uv run python -m experiments.validation.construct.run_cr_fysm_v4_construct_generation "
            "style_positive_control --jobs 8 --resume"
        ),
        "selection_integrity": {
            "style_scores_observed_before_final_lineage": False,
            "semantic_scores_observed_before_final_lineage": False,
            "alternative_output_selection_performed": False,
            "final_rule": "retain the first canonical artifact written at each sample path",
        },
        "final_lineage": {
            WHOLE_PASSAGE_ID: {
                "kind": "additional_whole_passage_completion_call",
                "artifact_path": str(whole_path),
                "artifact_sha256": generation.sha256_file(whole_path),
                "result_sha256": whole["result_sha256"],
                "response_id": whole["response_id"],
                "original_stage_prompt_model_schema_unchanged": True,
                "claim_limit": "not within the original per-run two-attempt budget",
            },
            STITCHED_ID: {
                "kind": "deterministic_stitch_amendment_v2",
                "artifact_path": str(stitched_path),
                "artifact_sha256": generation.sha256_file(stitched_path),
                "result_sha256": stitched["result_sha256"],
                "amendment_id": stitched_meta["amendment_id"],
                "declaration_id": stitched_meta["declaration_id"],
                "new_model_calls_for_stitch": 0,
            },
        },
        "bound_protocols": {
            "block_v1_declaration": {
                "path": str(block_v1.DECLARATION),
                "sha256": generation.sha256_file(block_v1.DECLARATION),
            },
            "stitch_v2_declaration": {
                "path": str(stitch_v2.DECLARATION),
                "sha256": generation.sha256_file(stitch_v2.DECLARATION),
            },
        },
        "required_analysis": {
            "primary": "retain all 50 source clusters with transparent lineage labels",
            "sensitivity": (
                "repeat all construct-validity conclusions after excluding both affected "
                "source clusters across neutral, sham, positive, and human-anchor variants"
            ),
            "claim_language": (
                "The construct is completed with documented source/adjudication amendments "
                "and a style-positive completion incident; it is not a pristine unamended run."
            ),
        },
    }
    payload["incident_id"] = generation.sha256_text(canonical(payload))
    ROOT.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
