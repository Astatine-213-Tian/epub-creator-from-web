#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from experiments.iteration5.decision import prepare_transfer_lora_blind_benchmark_v1 as v1
from experiments.shared.hashing import canonical_json as canonical
from experiments.shared.hashing import sha256_file, sha256_text


ROOT = v1.ITERATIONS / "paired_reconstruction_decision_v1/blind_benchmark_v2_reference_anchored"
ASSETS = v1.ASSETS
PREREGISTRATION = ROOT / "preregistration.json"
PRIVATE_BLOCKS = ROOT / "blocks.private.jsonl"
CALIBRATION_BLOCKS = ROOT / "calibration_blocks.private.jsonl"
PRIVATE_MAPPING = ROOT / "candidate_mapping.private.jsonl"
ELIGIBILITY = ROOT / "eligibility.private.jsonl"
SEED = 20260720
STYLE_IDENTITIES = v1.STYLE_IDENTITIES
SEMANTIC_IDENTITIES = v1.SEMANTIC_IDENTITIES
RUNNER = Path(__file__).with_name("run_transfer_lora_blind_ratings_v1.py")
ANALYZER = Path(__file__).with_name("analyze_transfer_lora_reference_benchmark_v2.py")
CLEAN_CHUNKS = Path("datasets/unmasked/chunks.clean.jsonl")
MASKED_CHUNKS = Path("datasets/masked/chunks.entity_masked_v3.jsonl")
LORA_PAIRS = (
    v1.ITERATIONS
    / "lora_paired_reconstruction_v1/smoke_data_v1/paired_records.private.jsonl"
)
HIDDEN_TARGETS = v1.SAMPLES / "iteration4_proxy_v1.hidden_targets.jsonl"
STYLE_PROMPT = ASSETS / "style_reference_anchored.v2.md"
STYLE_SCHEMA = ASSETS / "style_reference_anchored.schema.v2.json"
SEMANTIC_PROMPT = ASSETS / "semantic_multicandidate.v1.md"
SEMANTIC_SCHEMA = ASSETS / "semantic_multicandidate.schema.json"
CALIBRATION_SOURCE_BLOCKS = (
    "prompt.s01.b01",
    "prompt.s05.b01",
    "lora.b01",
    "lora.b07",
)
RUBRIC_DIMENSIONS = (
    "clause_packaging_and_syntax",
    "sentence_rhythm",
    "paragraph_discourse_progression",
    "dialogue_turn_and_attribution_architecture",
    "narrator_stance_and_reaction_timing",
    "punctuation_and_emphasis",
    "function_word_and_register_texture",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def lock_id(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "lock_id"}
    return sha256_text(canonical(body))


def shuffled_methods(methods: list[str], identity: str, block_id: str) -> list[str]:
    seed = int(sha256_text(f"{SEED}|{identity}|{block_id}")[:16], 16)
    result = methods[:]
    random.Random(seed).shuffle(result)
    return result


def paragraph_index(paragraph_id: str) -> int:
    match = re.fullmatch(r"p(\d+)", paragraph_id)
    if not match:
        raise ValueError(f"invalid paragraph id: {paragraph_id}")
    return int(match.group(1)) - 1


def prompt_reference_views(
    block: dict[str, Any], hidden: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    source = hidden[block["source_id"]]
    clean_lines = str(source["original_zh"]).splitlines()
    masked_lines = str(source["entity_masked_v3_zh"]).splitlines()
    if len(clean_lines) != len(masked_lines):
        raise ValueError(f"prompt clean/masked line mismatch: {block['source_id']}")
    clean_reference = block["candidates"]["hidden_original"]
    masked_reference: list[dict[str, str]] = []
    for row in clean_reference:
        index = paragraph_index(str(row["id"]))
        if index >= len(clean_lines) or clean_lines[index] != row["zh"]:
            raise ValueError(f"prompt clean reference mismatch: {block['block_id']} {row['id']}")
        masked_reference.append({"id": str(row["id"]), "zh": masked_lines[index]})
    return clean_reference, masked_reference


def load_chunk_rows(path: Path, chunk_ids: set[str]) -> dict[str, dict[str, Any]]:
    rows = {row["chunk_id"]: row for row in read_jsonl(path) if row.get("chunk_id") in chunk_ids}
    if set(rows) != chunk_ids:
        raise ValueError(f"missing chunks in {path}: {sorted(chunk_ids - set(rows))}")
    return rows


def recover_clean_lora_targets() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, Any]]]:
    pairs = {
        str(row["sample_id"]): row
        for row in read_jsonl(LORA_PAIRS)
        if row.get("role") == "internal_test"
    }
    chunk_ids = {str(row["chunk_id"]) for row in pairs.values()}
    clean_chunks = load_chunk_rows(CLEAN_CHUNKS, chunk_ids)
    masked_chunks = load_chunk_rows(MASKED_CHUNKS, chunk_ids)
    clean_by_source_and_id: dict[str, dict[str, str]] = {}
    lineage: dict[str, dict[str, Any]] = {}
    for source_id, pair in sorted(pairs.items()):
        chunk_id = str(pair["chunk_id"])
        clean_lines = str(clean_chunks[chunk_id]["text"]).splitlines()
        masked_lines = str(masked_chunks[chunk_id]["text"]).splitlines()
        if len(clean_lines) != len(masked_lines):
            raise ValueError(f"LoRA clean/masked line mismatch: {chunk_id}")
        mapped: dict[str, str] = {}
        indices: list[int] = []
        previous = -1
        for target in pair["target"]:
            matches = [
                index
                for index, line in enumerate(masked_lines)
                if index > previous and line == target["zh"]
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"LoRA target does not map uniquely: {source_id} {target['id']} {matches}"
                )
            previous = matches[0]
            indices.append(previous)
            mapped[str(target["id"])] = clean_lines[previous]
        clean_by_source_and_id[source_id] = mapped
        lineage[source_id] = {
            "book": str(pair["book"]),
            "chunk_id": chunk_id,
            "target_paragraphs": len(pair["target"]),
            "mapped_line_indices": indices,
            "clean_chunk_sha256": sha256_text(str(clean_chunks[chunk_id]["text"])),
            "masked_chunk_sha256": sha256_text(str(masked_chunks[chunk_id]["text"])),
        }
    return clean_by_source_and_id, lineage


def lora_reference_views(
    block: dict[str, Any], clean_targets: dict[str, dict[str, str]]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    masked_reference = block["candidates"]["hidden_original"]
    source_clean = clean_targets[block["source_id"]]
    clean_reference = [
        {"id": str(row["id"]), "zh": source_clean[str(row["id"])]}
        for row in masked_reference
    ]
    return clean_reference, masked_reference


def structure_disrupted_decoy(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    texts = [str(row["zh"]) for row in rows]
    reordered = list(reversed(texts))
    result: list[dict[str, str]] = []
    for row, text in zip(rows, reordered):
        units = [unit for unit in re.split(r"(?<=[。！？；：])", text) if unit]
        disrupted = "".join(reversed(units)) if len(units) > 1 else text
        result.append({"id": str(row["id"]), "zh": disrupted})
    if [row["zh"] for row in result] == texts:
        raise ValueError("calibration decoy did not change structure")
    if sorted("".join(row["zh"] for row in result)) != sorted("".join(texts)):
        raise ValueError("calibration decoy changed lexical inventory")
    return result


def build_blocks() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    inherited = read_jsonl(v1.PRIVATE_BLOCKS)
    hidden = {row["sample_id"]: row for row in read_jsonl(HIDDEN_TARGETS)}
    clean_lora_targets, lora_lineage = recover_clean_lora_targets()
    blocks: list[dict[str, Any]] = []
    reference_lineage: list[dict[str, Any]] = []
    for block in inherited:
        if block["arm"] == "prompt_development":
            clean_reference, masked_reference = prompt_reference_views(block, hidden)
            lineage = {
                "arm": block["arm"],
                "block_id": block["block_id"],
                "source_id": block["source_id"],
                "book": block["book"],
                "reference_source": str(HIDDEN_TARGETS),
                "reference_view": "original_zh_clean",
            }
        else:
            clean_reference, masked_reference = lora_reference_views(block, clean_lora_targets)
            source_lineage = lora_lineage[block["source_id"]]
            lineage = {
                "arm": block["arm"],
                "block_id": block["block_id"],
                "source_id": block["source_id"],
                "book": source_lineage["book"],
                "reference_source": str(CLEAN_CHUNKS),
                "reference_view": "clean",
                **{key: value for key, value in source_lineage.items() if key != "book"},
            }
        candidates = {
            method: paragraphs
            for method, paragraphs in block["candidates"].items()
            if method != "hidden_original"
        }
        blocks.append(
            {
                "arm": block["arm"],
                "block_id": block["block_id"],
                "source_id": block["source_id"],
                "book": lineage["book"],
                "english_source": block["english_source"],
                "style_reference": clean_reference,
                "masked_structure_control": masked_reference,
                "candidates": candidates,
            }
        )
        reference_lineage.append(lineage)
    inherited_eligibility = read_jsonl(v1.ELIGIBILITY)
    eligibility = [row for row in inherited_eligibility if row["method_id"] != "hidden_original"]
    return blocks, eligibility, reference_lineage


def build_calibration_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {block["block_id"]: block for block in blocks}
    controls: list[dict[str, Any]] = []
    for index, source_block_id in enumerate(CALIBRATION_SOURCE_BLOCKS, start=1):
        block = by_id[source_block_id]
        positive = block["masked_structure_control"]
        controls.append(
            {
                "arm": block["arm"],
                "block_id": f"calibration.c{index:02d}",
                "source_block_id": source_block_id,
                "source_id": block["source_id"],
                "book": block["book"],
                "style_reference": block["style_reference"],
                "candidates": {
                    "calibration_positive": positive,
                    "calibration_decoy": structure_disrupted_decoy(positive),
                },
            }
        )
    return controls


def packet(
    *,
    block: dict[str, Any],
    task: str,
    identity: str,
    methods: list[str],
) -> tuple[dict[str, Any], dict[str, str]]:
    ordered = shuffled_methods(methods, identity, block["block_id"])
    aliases = {method: f"c{index:02d}" for index, method in enumerate(ordered, start=1)}
    result: dict[str, Any] = {
        "schema_version": 2,
        "packet_id": f"{identity}.{block['block_id']}",
        "arm": block["arm"],
        "candidates": [
            {"candidate_id": aliases[method], "paragraphs": block["candidates"][method]}
            for method in ordered
        ],
    }
    if task == "style":
        result["estimand"] = "passage-specific structural reconstruction"
        result["rubric_dimensions"] = list(RUBRIC_DIMENSIONS)
        result["style_reference"] = block["style_reference"]
    else:
        result["english_source"] = block["english_source"]
    return result, aliases


def main() -> None:
    if ROOT.exists():
        raise FileExistsError(f"reference-anchored benchmark path already exists: {ROOT}")
    if not ANALYZER.exists():
        raise FileNotFoundError(f"v2 analyzer must exist before preregistration: {ANALYZER}")

    blocks, eligibility, reference_lineage = build_blocks()
    calibration_blocks = build_calibration_blocks(blocks)
    write_jsonl(PRIVATE_BLOCKS, blocks)
    write_jsonl(CALIBRATION_BLOCKS, calibration_blocks)
    write_jsonl(ELIGIBILITY, eligibility)
    write_jsonl(ROOT / "reference_lineage.private.jsonl", reference_lineage)

    mappings: list[dict[str, Any]] = []
    packet_hashes: dict[str, str] = {}
    for task, identities in (("style", STYLE_IDENTITIES), ("semantic", SEMANTIC_IDENTITIES)):
        for identity in identities:
            task_blocks = blocks + calibration_blocks if task == "style" else blocks
            for block in task_blocks:
                is_calibration = block["block_id"].startswith("calibration.")
                methods = sorted(block["candidates"])
                rating_packet, aliases = packet(
                    block=block,
                    task=task,
                    identity=identity,
                    methods=methods,
                )
                packet_path = ROOT / "packets" / identity / f"{rating_packet['packet_id']}.json"
                write_json(packet_path, rating_packet)
                packet_hashes[str(packet_path)] = sha256_file(packet_path)
                mappings.append(
                    {
                        "task": task,
                        "identity": identity,
                        "packet_id": rating_packet["packet_id"],
                        "arm": block["arm"],
                        "block_id": block["block_id"],
                        "source_id": block["source_id"],
                        "book": block["book"],
                        "is_calibration": is_calibration,
                        "source_block_id": block.get("source_block_id"),
                        "alias_to_method": {alias: method for method, alias in aliases.items()},
                    }
                )
    write_jsonl(PRIVATE_MAPPING, mappings)

    prompt_paths = {"style": STYLE_PROMPT, "semantic": SEMANTIC_PROMPT}
    schema_paths = {"style": STYLE_SCHEMA, "semantic": SEMANTIC_SCHEMA}
    private_paths = {
        PRIVATE_BLOCKS,
        CALIBRATION_BLOCKS,
        PRIVATE_MAPPING,
        ELIGIBILITY,
        ROOT / "reference_lineage.private.jsonl",
    }
    input_paths = {
        v1.PREREGISTRATION,
        v1.ROOT / "results.json",
        v1.PRIVATE_BLOCKS,
        v1.ELIGIBILITY,
        HIDDEN_TARGETS,
        CLEAN_CHUNKS,
        MASKED_CHUNKS,
        LORA_PAIRS,
        Path(__file__).resolve(),
        RUNNER,
        ANALYZER,
        *prompt_paths.values(),
        *schema_paths.values(),
    }
    payload: dict[str, Any] = {
        "schema_version": 2,
        "experiment_id": "transfer-prompt-lora-reference-anchored-benchmark-v2",
        "status": "locked_before_any_v2_rating_post_control_failure_amendment",
        "epistemic_status": "post-outcome exploratory amendment; not confirmatory",
        "amendment_reason": (
            "v1 style and semantic panels penalized clean original Chinese against imperfect "
            "generated English, invalidating the hidden-original positive control"
        ),
        "parent_v1": {
            "lock_id": read_json(v1.PREREGISTRATION)["lock_id"],
            "preregistration_sha256": sha256_file(v1.PREREGISTRATION),
            "results_sha256": sha256_file(v1.ROOT / "results.json"),
            "role": "invalid-control benchmark retained as semantic sensitivity evidence only",
        },
        "seed": SEED,
        "estimand": (
            "passage-specific structural reconstruction relative to an evaluation-only clean "
            "Chinese reference; not generic author identification or production efficacy"
        ),
        "geometry": {
            "prompt_sources": 8,
            "prompt_blocks": sum(block["arm"] == "prompt_development" for block in blocks),
            "lora_books": 3,
            "lora_blocks": sum(block["arm"] == "lora_internal_test" for block in blocks),
            "style_raters": len(STYLE_IDENTITIES),
            "semantic_raters": len(SEMANTIC_IDENTITIES),
            "style_calibration_blocks_per_rater": len(calibration_blocks),
        },
        "separation": {
            "style_panel": "clean original Chinese reference plus v2 structural rubric; no English",
            "semantic_panel": "English semantic source plus anonymous candidates; no Chinese reference",
            "hidden_original_candidate": "removed",
            "candidate_generation": "unchanged from v1 and completed before v2",
            "lora_reference_recovery": "unique monotonic masked-to-clean line mapping, hash-bound",
        },
        "calibration": {
            "scored_with_methods": False,
            "positive": (
                "entity-masked version of the same clean reference, preserving paragraph and "
                "sentence structure while changing entity-bearing lexical material"
            ),
            "decoy": (
                "the same masked characters and punctuation with paragraph order reversed and "
                "multi-unit sentence order reversed"
            ),
            "rater_pass": (
                "mean positive-minus-decoy style score > 0 and positive > decoy on at least "
                "3 of 4 calibration blocks"
            ),
            "benchmark_interpretation_gate": "all three style raters must pass calibration",
            "limitation": (
                "the positive is deterministic entity masking, not a broad human lexical paraphrase"
            ),
        },
        "primary_rules": {
            "semantic_block_pass": (
                "both validators: fidelity >=4, naturalness >=3, no high-severity error, "
                "speaker/dialogue topology preserved"
            ),
            "source_semantic_pass": ">=80% blocks pass and no block has majority high-severity error",
            "source_style_improvement": (
                "mean reference-anchored structural score above neutral and >50% of paired "
                "rater-block comparisons above neutral"
            ),
            "source_success": (
                "all blocks contract-valid plus semantic pass plus structural-style improvement "
                "plus median style-panel naturalness >=3"
            ),
            "prompt_80pct_gate": "at least 7 of 8 target-author development sources succeed",
            "lora_gate": (
                "descriptive only; all 3 internal-test books must be contract-valid before any "
                "style-efficacy interpretation"
            ),
            "uncertainty_unit": "source/book cluster, never individual paragraph or rating",
        },
        "semantic_policy": {
            "v2": "rerate unchanged candidates after removing hidden original",
            "sensitivity": "report v1 per-method semantic source outcomes beside v2",
        },
        "models": {"blind_raters": "gpt-5.5", "reasoning_effort": "high"},
        "independence_limit": (
            "all automated raters are repeated calls to one model family; results remain "
            "developmental until an independent Chinese human panel or different model family confirms them"
        ),
        "classifier_policy": "CR-FYSM-v4 excluded from primary decisions",
        "decision_scope": (
            "exploratory diagnosis of frozen prompt methods and whether clean paired-data "
            "construction plus a later powered LoRA pilot is warranted"
        ),
        "input_hashes": {str(path): sha256_file(path) for path in sorted(input_paths, key=str)},
        "private_hashes": {str(path): sha256_file(path) for path in sorted(private_paths, key=str)},
        "packet_hashes": packet_hashes,
        "prompts": {task: str(path) for task, path in prompt_paths.items()},
        "schemas": {task: str(path) for task, path in schema_paths.items()},
        "forbidden_claims": [
            "confirmatory method efficacy",
            "official screening success",
            "production readiness",
            "human-equivalent author imitation",
            "generic author-style identification from this reference-anchored endpoint",
        ],
    }
    payload["lock_id"] = lock_id(payload)
    write_json(PREREGISTRATION, payload)
    print(f"wrote {PREREGISTRATION}")
    print(json.dumps(payload["geometry"], indent=2))
    print(f"lock_id={payload['lock_id']}")


if __name__ == "__main__":
    main()
