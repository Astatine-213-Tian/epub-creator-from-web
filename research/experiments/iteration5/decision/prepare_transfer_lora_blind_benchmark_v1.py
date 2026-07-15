#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any

from experiments.shared.hashing import canonical_json as canonical
from experiments.shared.hashing import sha256_file, sha256_text


ITERATIONS = Path("generated/style_research/style_transfer_experiments/iterations")
FULL = ITERATIONS / "full_regeneration_v1"
STYLE_RUN = FULL / "runs/iteration4_proxy_v1/iteration4_style_gpt55_v1"
SOURCE_RUN = FULL / "runs/iteration4_proxy_v1/iteration4_source_gpt54_official"
SAMPLES = FULL / "sample_sets"
LORA_RUN = ITERATIONS / "lora_paired_reconstruction_v1/runs/qwen3_4b_mlx_smoke_v1"
CONSTRUCT = ITERATIONS / "content_resistant_v1/cr_fysm_v4/construct_validation_v2"
ROOT = ITERATIONS / "paired_reconstruction_decision_v1/blind_benchmark_v1"
ASSETS = Path(__file__).with_name("assets")
PREREGISTRATION = ROOT / "preregistration.json"
PRIVATE_BLOCKS = ROOT / "blocks.private.jsonl"
PRIVATE_MAPPING = ROOT / "candidate_mapping.private.jsonl"
ELIGIBILITY = ROOT / "eligibility.private.jsonl"
SEED = 20260719
MAX_PROMPT_BLOCK_PARAGRAPHS = 20
STYLE_IDENTITIES = ("style_a", "style_b", "style_c")
SEMANTIC_IDENTITIES = ("semantic_a", "semantic_b")
PROMPT_METHODS = (
    "neutral_only",
    "generic_full_regeneration",
    "aligned_pairs_full_regeneration",
    "style_definition_examples_full_regeneration",
    "aligned_pairs_style_definition_full_regeneration",
    "content_plan_combined_full_regeneration",
    "independent_candidate_selector",
    "hidden_original",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def paragraphs(path: Path, language: str) -> list[dict[str, str]]:
    rows = read_json(path)["result"]["paragraphs"]
    return [{"id": str(row["id"]), language: str(row[language])} for row in rows]


def prompt_blocks() -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[Path]]:
    development = set(read_json(SAMPLES / "iteration4_proxy_v1.development_v1_ids.json")["sample_ids"])
    allocation_path = SAMPLES / "iteration4_proxy_v1.evaluator_allocation.jsonl"
    allocation = read_jsonl(allocation_path)
    own = sorted(
        (
            row
            for row in allocation
            if row["sample_id"] in development
            and row["benchmark_arm"] == "own_author_reconstruction"
        ),
        key=lambda row: row["sample_id"],
    )
    hidden_path = SAMPLES / "iteration4_proxy_v1.hidden_targets.jsonl"
    hidden = {row["sample_id"]: row for row in read_jsonl(hidden_path)}
    blocks: list[dict[str, Any]] = []
    eligibility: list[dict[str, Any]] = []
    inputs: set[Path] = {allocation_path, hidden_path, SAMPLES / "iteration4_proxy_v1.development_v1_ids.json"}
    for source_index, meta in enumerate(own, start=1):
        source_id = str(meta["sample_id"])
        english_path = SOURCE_RUN / "english_semantic_source" / f"{source_id}.json"
        neutral_path = SOURCE_RUN / "neutral_translation" / f"{source_id}.json"
        english = paragraphs(english_path, "en")
        candidates: dict[str, list[dict[str, str]]] = {
            "neutral_only": paragraphs(neutral_path, "zh"),
            "hidden_original": [
                {"id": english[index]["id"], "zh": line}
                for index, line in enumerate(hidden[source_id]["original_zh"].splitlines())
            ],
        }
        inputs.update((english_path, neutral_path))
        for method in PROMPT_METHODS[1:-1]:
            method_path = STYLE_RUN / "method_outputs" / method / "strong" / f"{source_id}.json"
            candidates[method] = paragraphs(method_path, "zh")
            inputs.add(method_path)
        expected_ids = [row["id"] for row in english]
        for method, rows in candidates.items():
            if [row["id"] for row in rows] != expected_ids:
                raise ValueError(f"paragraph topology mismatch: {source_id} {method}")
        for block_index, start in enumerate(range(0, len(english), MAX_PROMPT_BLOCK_PARAGRAPHS), start=1):
            end = min(start + MAX_PROMPT_BLOCK_PARAGRAPHS, len(english))
            block_id = f"prompt.s{source_index:02d}.b{block_index:02d}"
            blocks.append(
                {
                    "arm": "prompt_development",
                    "block_id": block_id,
                    "source_id": source_id,
                    "book": str(meta["book_title"]),
                    "english_source": english[start:end],
                    "candidates": {method: rows[start:end] for method, rows in candidates.items()},
                }
            )
            for method in candidates:
                eligibility.append(
                    {"arm": "prompt_development", "block_id": block_id, "source_id": source_id,
                     "method_id": method, "contract_valid": True, "errors": []}
                )
    return blocks, eligibility, inputs


def lora_blocks() -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[Path]]:
    data_path = LORA_RUN / "mlx_data/test.jsonl"
    inference_root = LORA_RUN / "inference_v1"
    rows = read_jsonl(data_path)
    blocks: list[dict[str, Any]] = []
    eligibility: list[dict[str, Any]] = []
    inputs: set[Path] = {data_path, inference_root / "preregistration.json"}
    for row_index, row in enumerate(rows, start=1):
        user = json.loads(row["messages"][1]["content"])
        target = json.loads(row["messages"][2]["content"])
        block_id = f"lora.b{row_index:02d}"
        block_sample_id = str(user["sample_id"])
        source_id = block_sample_id.split(".block", 1)[0]
        candidates: dict[str, list[dict[str, str]]] = {
            "neutral_only": user["neutral_chinese"],
            "hidden_original": target["paragraphs"],
        }
        for method, arm in (("frozen_base_4b", "base"), ("lora_smoke_4b", "lora")):
            artifact_path = inference_root / arm / f"{block_sample_id}.json"
            artifact = read_json(artifact_path)
            inputs.add(artifact_path)
            contract_valid = bool(artifact["valid"])
            if contract_valid:
                candidates[method] = artifact["result"]["paragraphs"]
            eligibility.append(
                {"arm": "lora_internal_test", "block_id": block_id, "source_id": source_id,
                 "method_id": method, "contract_valid": contract_valid,
                 "errors": artifact["validation_errors"]}
            )
        for method in ("neutral_only", "hidden_original"):
            eligibility.append(
                {"arm": "lora_internal_test", "block_id": block_id, "source_id": source_id,
                 "method_id": method, "contract_valid": True, "errors": []}
            )
        blocks.append(
            {
                "arm": "lora_internal_test",
                "block_id": block_id,
                "source_id": source_id,
                "book": source_id,
                "english_source": user["english_source"],
                "candidates": candidates,
            }
        )
    return blocks, eligibility, inputs


def shuffled_methods(methods: list[str], identity: str, block_id: str) -> list[str]:
    seed = int(sha256_bytes(f"{SEED}|{identity}|{block_id}".encode())[:16], 16)
    result = methods[:]
    random.Random(seed).shuffle(result)
    return result


def lock_id(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "lock_id"}
    return sha256_text(canonical(body))


def main() -> None:
    if PREREGISTRATION.exists() or (ROOT / "ratings").exists():
        raise FileExistsError("blind benchmark already prepared or rated")
    if ROOT.exists():
        shutil.rmtree(ROOT)
    prompt, prompt_eligibility, prompt_inputs = prompt_blocks()
    lora, lora_eligibility, lora_inputs = lora_blocks()
    blocks = prompt + lora
    eligibility = prompt_eligibility + lora_eligibility
    write_jsonl(PRIVATE_BLOCKS, blocks)
    write_jsonl(ELIGIBILITY, eligibility)

    style_definition = read_json(CONSTRUCT / "generation_preregistration.v2.json")["style_evidence"]["dimensions"]
    mappings: list[dict[str, Any]] = []
    packet_hashes: dict[str, str] = {}
    for task, identities in (("style", STYLE_IDENTITIES), ("semantic", SEMANTIC_IDENTITIES)):
        for identity in identities:
            for block in blocks:
                methods = shuffled_methods(sorted(block["candidates"]), identity, block["block_id"])
                aliases = {method: f"c{index:02d}" for index, method in enumerate(methods, start=1)}
                packet = {
                    "schema_version": 1,
                    "packet_id": f"{identity}.{block['block_id']}",
                    "arm": block["arm"],
                    "english_source": block["english_source"],
                    "candidates": [
                        {"candidate_id": aliases[method], "paragraphs": block["candidates"][method]}
                        for method in methods
                    ],
                }
                if task == "style":
                    packet["style_t_definition"] = style_definition
                packet_path = ROOT / "packets" / identity / f"{packet['packet_id']}.json"
                write_json(packet_path, packet)
                packet_hashes[str(packet_path)] = sha256_file(packet_path)
                mappings.append(
                    {"task": task, "identity": identity, "packet_id": packet["packet_id"],
                     "arm": block["arm"], "block_id": block["block_id"],
                     "source_id": block["source_id"], "book": block["book"],
                     "alias_to_method": {alias: method for method, alias in aliases.items()}}
                )
    write_jsonl(PRIVATE_MAPPING, mappings)
    prompt_paths = {
        "style": ASSETS / "style_multicandidate.v1.md",
        "semantic": ASSETS / "semantic_multicandidate.v1.md",
    }
    schema_paths = {
        "style": ASSETS / "style_multicandidate.schema.json",
        "semantic": ASSETS / "semantic_multicandidate.schema.json",
    }
    input_paths = prompt_inputs | lora_inputs | {
        CONSTRUCT / "generation_preregistration.v2.json",
        Path(__file__).resolve(),
        *prompt_paths.values(),
        *schema_paths.values(),
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": "transfer-prompt-lora-blind-benchmark-v1",
        "status": "locked_before_any_blind_rating",
        "seed": SEED,
        "geometry": {
            "prompt_sources": 8,
            "prompt_methods": len(PROMPT_METHODS),
            "prompt_blocks": len(prompt),
            "lora_books": 3,
            "lora_blocks": len(lora),
            "style_raters": len(STYLE_IDENTITIES),
            "semantic_raters": len(SEMANTIC_IDENTITIES),
        },
        "methods": {
            "prompt_development": list(PROMPT_METHODS),
            "lora_internal_test": ["neutral_only", "frozen_base_4b", "lora_smoke_4b", "hidden_original"],
        },
        "primary_rules": {
            "semantic_block_pass": "both validators: fidelity >=4, naturalness >=3, no high-severity error, topology preserved",
            "source_semantic_pass": ">=80% blocks pass and no block has majority high-severity error",
            "source_style_improvement": "mean blind Style-T score above neutral and >50% of rater-block comparisons above neutral",
            "source_success": "all source blocks contract-valid plus semantic pass plus style improvement plus median style-rater naturalness >=3",
            "prompt_80pct_gate": "at least 7 of 8 target-author development sources succeed",
            "lora_gate": "descriptive only; all 3 internal-test books must be contract-valid before any efficacy claim",
        },
        "decision_scope": "development method diagnosis and LoRA feasibility; not official screening or production efficacy",
        "classifier_policy": "CR-FYSM-v4 excluded from all primary decisions after failed fresh construct validation",
        "models": {"blind_raters": "gpt-5.5", "reasoning_effort": "high"},
        "input_hashes": {str(path): sha256_file(path) for path in sorted(input_paths, key=str)},
        "private_hashes": {
            str(PRIVATE_BLOCKS): sha256_file(PRIVATE_BLOCKS),
            str(PRIVATE_MAPPING): sha256_file(PRIVATE_MAPPING),
            str(ELIGIBILITY): sha256_file(ELIGIBILITY),
        },
        "packet_hashes": packet_hashes,
        "forbidden_claims": [
            "official screening success",
            "production readiness",
            "human-equivalent author imitation",
            "generalization beyond the enumerated books",
        ],
    }
    payload["lock_id"] = lock_id(payload)
    write_json(PREREGISTRATION, payload)
    print(f"wrote {PREREGISTRATION}")
    print(json.dumps(payload["geometry"], indent=2))
    print(f"lock_id={payload['lock_id']}")


if __name__ == "__main__":
    main()
