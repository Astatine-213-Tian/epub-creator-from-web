#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


from experiments.shared.paths import RESEARCH_ROOT


REPO_ROOT = RESEARCH_ROOT
BASE_PATH = REPO_ROOT / "experiments/iteration1/run_style_transfer_generation.py"
PAYLOAD_PATH = Path(__file__).with_name("style_transfer_payloads.py")
BLOCK_THRESHOLD = 12
MAX_BLOCK_PARAGRAPHS = 12


def argument(name: str) -> str | None:
    for index, value in enumerate(sys.argv):
        if value == name and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
    return None


def verify_sources(root: Path) -> None:
    protocol_path = root / "protocols/evaluation_protocol.v1.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    bindings = protocol.get("iteration3", {}).get("source_bindings", {})
    if not isinstance(bindings, dict) or not bindings:
        raise RuntimeError("Iteration-3 protocol lacks frozen source bindings")
    for path_text, expected in bindings.items():
        observed = hashlib.sha256((REPO_ROOT / path_text).read_bytes()).hexdigest()
        if observed != expected:
            raise RuntimeError(f"Iteration-3 source binding mismatch: {path_text}")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def install_block_generation(base) -> None:
    original = base.run_one_attempt

    def run_one_attempt(**kwargs):
        request = kwargs["request"]
        if kwargs["stage"] != "style_transfer":
            return original(**kwargs)
        english = list(request["english_semantic_source"])
        neutral = list(request["neutral_zh"])
        if len(english) != len(neutral):
            raise ValueError("English and neutral paragraph counts differ")
        if [row["id"] for row in english] != [row["id"] for row in neutral]:
            raise ValueError("English and neutral paragraph IDs differ")

        segment_records: list[dict[str, Any]] = []
        leaf_segments: list[dict[str, Any]] = []
        output_path: Path = kwargs["output_path"]
        block_root = (
            output_path.parent
            / "_blocks"
            / kwargs["sample_id"]
            / f"attempt_{kwargs['attempt']:02d}"
        )

        def run_segment(
            start: int,
            end: int,
            depth: int,
            parent: str | None,
        ) -> dict[str, Any] | None:
            segment_name = f"s{start + 1:04d}_e{end:04d}_d{depth:02d}"
            source_english = english[start:end]
            source_neutral = neutral[start:end]
            local_ids = [f"p{index + 1:04d}" for index in range(end - start)]
            local_english = [
                {**row, "id": local_ids[index]}
                for index, row in enumerate(source_english)
            ]
            local_neutral = [
                {**row, "id": local_ids[index]}
                for index, row in enumerate(source_neutral)
            ]
            block_request = {
                **request,
                "english_semantic_source": local_english,
                "neutral_zh": local_neutral,
            }
            payload = dict(block_request.get("method_payload", {}))
            payload["block_execution"] = {
                "segment": segment_name,
                "recursive_depth": depth,
                "paragraph_offset": start,
                "full_paragraph_count": len(neutral),
                "paragraph_ids": local_ids,
            }
            block_request["method_payload"] = payload
            block_path = block_root / f"segment_{segment_name}.json"
            record = original(
                **{
                    **kwargs,
                    "request": block_request,
                    "output_path": block_path,
                }
            )
            trace = {
                "segment": segment_name,
                "parent": parent,
                "start": start,
                "end": end,
                "depth": depth,
                "local_ids": local_ids,
                "source_ids": [row["id"] for row in source_neutral],
                "path": block_path,
                "record": record,
            }
            segment_records.append(trace)
            if record.get("status") == "success" and block_path.exists():
                artifact = json.loads(block_path.read_text(encoding="utf-8"))
                paragraphs = artifact["result"]["paragraphs"]
                remapped = [
                    {**paragraph, "id": source_neutral[index]["id"]}
                    for index, paragraph in enumerate(paragraphs)
                ]
                leaf_segments.append(
                    {
                        **trace,
                        "artifact": artifact,
                        "paragraphs": remapped,
                    }
                )
                return None
            if end - start <= 1:
                return trace
            midpoint = start + (end - start) // 2
            failed = run_segment(start, midpoint, depth + 1, segment_name)
            if failed is not None:
                return failed
            return run_segment(midpoint, end, depth + 1, segment_name)

        for start in range(0, len(neutral), MAX_BLOCK_PARAGRAPHS):
            end = min(start + MAX_BLOCK_PARAGRAPHS, len(neutral))
            failed_trace = run_segment(start, end, 0, None)
            if failed_trace is not None:
                failed_record = dict(failed_trace["record"])
                failed_record["input_sha256"] = base.sha256_text(
                    base.canonical_json(request)
                )
                failed_record["response_file"] = str(
                    failed_trace["path"].relative_to(REPO_ROOT)
                )
                failed_record["validation_errors"] = [
                    *failed_record.get("validation_errors", []),
                    f"block_generation_failed:{failed_trace['segment']}",
                ]
                failed_record["block_execution"] = {
                    "schema_version": 2,
                    "status": "failed",
                    "failed_segment": failed_trace["segment"],
                    "attempted_segments": len(segment_records),
                    "completed_leaf_segments": len(leaf_segments),
                }
                return failed_record

        paragraphs: list[dict[str, Any]] = []
        applied: list[str] = []
        skipped: list[str] = []
        uncertainties: list[str] = []
        usage: dict[str, int] = {}
        for leaf in sorted(leaf_segments, key=lambda value: value["start"]):
            artifact = leaf["artifact"]
            result = artifact["result"]
            paragraphs.extend(leaf["paragraphs"])
            applied.extend(str(value) for value in result.get("style_cues_applied", []))
            skipped.extend(str(value) for value in result.get("style_cues_skipped", []))
            uncertainties.extend(str(value) for value in result.get("uncertainties", []))
        for trace in segment_records:
            record = trace["record"]
            for key, value in (record.get("response_usage") or {}).items():
                if isinstance(value, int):
                    usage[key] = usage.get(key, 0) + value
        result = {
            "sample_id": request["sample_id"],
            "method_id": request["method_id"],
            "intensity": request["intensity"],
            "paragraphs": paragraphs,
            "style_cues_applied": list(dict.fromkeys(applied)),
            "style_cues_skipped": list(dict.fromkeys(skipped)),
            "uncertainties": list(dict.fromkeys(uncertainties)),
        }
        validation_errors = base.validate_result("style_transfer", request, result)
        output_sha = base.sha256_text(base.canonical_json(result))
        input_sha = base.sha256_text(base.canonical_json(request))
        method_sha = base.sha256_text(
            base.canonical_json(request.get("method_payload", {}))
        )
        reference_sha = base.sha256_text(
            base.canonical_json(request.get("reference_examples", []))
        )
        block_provenance = [
            {
                "segment": trace["segment"],
                "parent": trace["parent"],
                "source_start": trace["start"],
                "source_end": trace["end"],
                "recursive_depth": trace["depth"],
                "local_ids": trace["local_ids"],
                "source_ids_sha256": base.sha256_text(
                    base.canonical_json(trace["source_ids"])
                ),
                "status": trace["record"].get("status"),
                "path": str(trace["path"].relative_to(REPO_ROOT)),
                "file_sha256": (
                    hashlib.sha256(trace["path"].read_bytes()).hexdigest()
                    if trace["path"].exists()
                    else None
                ),
                "response_id": trace["record"].get("response_id"),
                "input_sha256": trace["record"].get("input_sha256"),
                "output_sha256": trace["record"].get("output_sha256"),
            }
            for trace in segment_records
        ]
        if not validation_errors:
            artifact = {
                "schema_version": 1,
                "run_id": kwargs["run_id"],
                "stage": "style_transfer",
                "sample_id": kwargs["sample_id"],
                "attempt": kwargs["attempt"],
                "model": kwargs["model_config"]["codex_model"],
                "reasoning_effort": kwargs["model_config"]["reasoning_effort"],
                "prompt_sha256": kwargs["prompt_sha256"],
                "input_sha256": input_sha,
                "output_sha256": output_sha,
                "response_id": "block_merge:" + output_sha[:24],
                "response_usage": usage,
                "result": result,
                **kwargs["execution_binding"],
                "method_id": request["method_id"],
                "intensity": request["intensity"],
                "method_payload_sha256": method_sha,
                "reference_examples_sha256": reference_sha,
                "block_execution": {
                    "schema_version": 2,
                    "threshold": BLOCK_THRESHOLD,
                    "max_block_paragraphs": MAX_BLOCK_PARAGRAPHS,
                    "model_visible_paragraph_ids": "block_local_ids",
                    "source_id_restoration": "deterministic_position_preserving_remap",
                    "failed_block_recovery": "deterministic_recursive_bisection",
                    "attempted_segment_count": len(segment_records),
                    "leaf_segment_count": len(leaf_segments),
                    "segments": block_provenance,
                },
            }
            base.write_json(output_path, artifact)
        template = dict(segment_records[-1]["record"])
        template.update(
            {
                "started_at": segment_records[0]["record"]["started_at"],
                "completed_at": utc_now(),
                "status": "success" if not validation_errors else "failed",
                "input_sha256": input_sha,
                "output_sha256": output_sha,
                "response_id": "block_merge:" + output_sha[:24],
                "response_usage": usage,
                "response_file": str(output_path.relative_to(REPO_ROOT)),
                "validation_errors": validation_errors,
                "stderr_tail": "",
                "method_payload_sha256": method_sha,
                "reference_examples_sha256": reference_sha,
                "block_execution": {
                    "schema_version": 2,
                    "status": "success" if not validation_errors else "failed",
                    "attempted_segment_count": len(segment_records),
                    "leaf_segment_count": len(leaf_segments),
                    "recovered_split_count": sum(
                        1
                        for trace in segment_records
                        if trace["record"].get("status") != "success"
                        and trace["end"] - trace["start"] > 1
                    ),
                    "segments": block_provenance,
                },
            }
        )
        return template

    base.run_one_attempt = run_one_attempt


def main() -> None:
    root_text = argument("--experiment-root")
    if root_text is None:
        raise SystemExit("Iteration-3 generation requires --experiment-root")
    root = Path(root_text).expanduser().resolve()
    stage = argument("--stage")
    method_id = argument("--method-id")
    if stage == "style_transfer":
        verify_sources(root)
        if method_id == "candidate_rerank":
            raise SystemExit(
                "candidate_rerank is deterministic; run build_candidate_rerank.py"
            )
    payload = load_module("iteration3_style_transfer_payloads", PAYLOAD_PATH)
    base = load_module("iteration3_generation_base", BASE_PATH)
    base.build_method_request = payload.build_method_request
    install_block_generation(base)
    base.__file__ = str(Path(__file__).resolve())
    base.main()


if __name__ == "__main__":
    main()
