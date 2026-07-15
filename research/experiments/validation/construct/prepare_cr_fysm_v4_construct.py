#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from experiments.validation.meter.build_cr_fysm_v3 import canonical_lock_id, sha256_file


SEED = 20260716
TARGET_AUTHOR = "非天夜翔"
ITER4_ROOT = Path(
    "generated/style_research/style_transfer_experiments/iterations/"
    "full_regeneration_v1"
)
V4_ROOT = Path(
    "generated/style_research/style_transfer_experiments/iterations/"
    "content_resistant_v1/cr_fysm_v4"
)
OUTPUT_DIR = V4_ROOT / "construct_validation_v1"
PREREGISTRATION = OUTPUT_DIR / "preregistration.construct.v1.json"
PRIVATE_ITEMS = OUTPUT_DIR / "construct_items.private.jsonl"
RUBRIC = OUTPUT_DIR / "blind_rater_rubric.json"
RATER_NAMES = ("rater_a", "rater_b", "rater_c")
TARGET_INFORMED_METHODS = (
    "aligned_pairs_full_regeneration",
    "style_definition_examples_full_regeneration",
    "aligned_pairs_style_definition_full_regeneration",
    "content_plan_combined_full_regeneration",
)
GENERIC_CONTROL = "generic_full_regeneration"
ALL_METHODS = (GENERIC_CONTROL, *TARGET_INFORMED_METHODS)
STYLE_ASSET = (
    ITER4_ROOT
    / "method_assets/style_transfer_payloads.v1/"
    "assets.497a0db8919ccc9cfd97f6a729ff0528953d2d97477e83e07e55c42e4bb994d8.json"
)
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and lock the blind CR-FYSM-v4 construct challenge."
    )
    parser.add_argument(
        "--mode",
        choices=("draft", "lock", "validate"),
        default="draft",
    )
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def cjk_count(text: str) -> int:
    return len(CJK_RE.findall(text))


def load_inputs() -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    list[str],
]:
    allocation_path = ITER4_ROOT / "sample_sets/iteration4_proxy_v1.evaluator_allocation.jsonl"
    hidden_path = ITER4_ROOT / "sample_sets/iteration4_proxy_v1.hidden_targets.jsonl"
    development_path = ITER4_ROOT / "sample_sets/iteration4_proxy_v1.development_v1_ids.json"
    allocation = {row["sample_id"]: row for row in read_jsonl(allocation_path)}
    hidden = {row["sample_id"]: row for row in read_jsonl(hidden_path)}
    sample_ids = read_json(development_path)["sample_ids"]
    if len(sample_ids) != 16 or len(set(sample_ids)) != 16:
        raise ValueError("construct challenge requires the frozen 16-row development set")
    if set(sample_ids) - allocation.keys() or set(sample_ids) - hidden.keys():
        raise ValueError("development IDs are missing from Iteration 4 source artifacts")
    arm_counts = Counter(allocation[sample_id]["benchmark_arm"] for sample_id in sample_ids)
    if arm_counts != Counter({"own_author_reconstruction": 8, "cross_author_transfer": 8}):
        raise ValueError(f"unexpected development arm geometry: {arm_counts}")
    return allocation, hidden, sample_ids


def paragraph_maps(
    sample_id: str,
    hidden: dict[str, Any],
) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, dict[str, str]]]:
    source_root = ITER4_ROOT / "runs/iteration4_proxy_v1/iteration4_source_gpt54_official"
    style_root = ITER4_ROOT / "runs/iteration4_proxy_v1/iteration4_style_gpt55_v1"
    english_rows = read_json(source_root / "english_semantic_source" / f"{sample_id}.json")[
        "result"
    ]["paragraphs"]
    neutral_rows = read_json(source_root / "neutral_translation" / f"{sample_id}.json")[
        "result"
    ]["paragraphs"]
    original_lines = hidden["original_zh"].splitlines()
    english = {row["id"]: row["en"] for row in english_rows}
    neutral = {row["id"]: row["zh"] for row in neutral_rows}
    ids = list(english)
    original = {paragraph_id: text for paragraph_id, text in zip(ids, original_lines, strict=True)}
    if list(neutral) != ids or len(original_lines) != len(ids):
        raise ValueError(f"paragraph alignment mismatch for {sample_id}")
    methods: dict[str, dict[str, str]] = {}
    for method in ALL_METHODS:
        path = style_root / "method_outputs" / method / "strong" / f"{sample_id}.json"
        rows = read_json(path)["result"]["paragraphs"]
        methods[method] = {row["id"]: row["zh"] for row in rows}
        if list(methods[method]) != ids:
            raise ValueError(f"method paragraph alignment mismatch: {method}/{sample_id}")
    return english, original, neutral, methods


def select_quartile_blocks(paragraph_ids: list[str], original: dict[str, str]) -> list[list[str]]:
    blocks: list[list[str]] = []
    count = len(paragraph_ids)
    if count < 8:
        raise ValueError("construct source has too few paragraphs for quartile blocks")
    for quartile in range(4):
        lower = quartile * count // 4
        upper = (quartile + 1) * count // 4
        center = min(upper - 1, lower + max(0, upper - lower - 1) // 2)
        left = center
        right = center + 1
        while cjk_count("\n".join(original[key] for key in paragraph_ids[left:right])) < 180:
            can_left = left > lower
            can_right = right < upper
            if not can_left and not can_right:
                break
            if can_right:
                right += 1
            if cjk_count("\n".join(original[key] for key in paragraph_ids[left:right])) >= 180:
                break
            if can_left:
                left -= 1
        selected = paragraph_ids[left:right]
        if not selected:
            raise ValueError("empty construct block")
        blocks.append(selected)
    if len({tuple(block) for block in blocks}) != 4:
        raise ValueError("quartile block selection produced duplicates")
    return blocks


def item_record(
    *,
    sample_id: str,
    block_index: int,
    paragraph_ids: list[str],
    variant: str,
    method_id: str | None,
    chinese: str,
    english: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    item_id = f"{sample_id}.b{block_index:02d}.{variant}"
    return {
        "item_id": item_id,
        "blind_id": "cv_" + stable_hash("CR-FYSM-v4-construct:" + item_id)[:20],
        "sample_id": sample_id,
        "block_index": block_index,
        "paragraph_ids": paragraph_ids,
        "variant": variant,
        "method_id": method_id,
        "benchmark_arm": metadata["benchmark_arm"],
        "source_author": metadata["author"],
        "source_book": metadata["book_title"],
        "source_chunk_id": metadata["chunk_id"],
        "scene_type": metadata["strata"]["scene_type"],
        "surface_origin": "human" if variant == "original" else "generated",
        "target_style_label_by_provenance": (
            metadata["author"] == TARGET_AUTHOR and variant == "original"
        ),
        "english_source": english,
        "chinese_candidate": chinese,
        "english_sha256": stable_hash(english),
        "chinese_sha256": stable_hash(chinese),
        "chinese_cjk": cjk_count(chinese),
    }


def build_items() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    allocation, hidden, sample_ids = load_inputs()
    all_items: list[dict[str, Any]] = []
    by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    base_blocks: list[tuple[str, int]] = []
    for sample_id in sorted(sample_ids):
        english, original, neutral, methods = paragraph_maps(sample_id, hidden[sample_id])
        paragraph_ids = list(english)
        for block_index, block_ids in enumerate(
            select_quartile_blocks(paragraph_ids, original), start=1
        ):
            base_blocks.append((sample_id, block_index))
            english_text = "\n".join(english[key] for key in block_ids)
            variants = {
                "original": (None, "\n".join(original[key] for key in block_ids)),
                "neutral": (None, "\n".join(neutral[key] for key in block_ids)),
                **{
                    f"method:{method}": (
                        method,
                        "\n".join(methods[method][key] for key in block_ids),
                    )
                    for method in ALL_METHODS
                },
            }
            for variant, (method_id, chinese_text) in variants.items():
                row = item_record(
                    sample_id=sample_id,
                    block_index=block_index,
                    paragraph_ids=block_ids,
                    variant=variant,
                    method_id=method_id,
                    chinese=chinese_text,
                    english=english_text,
                    metadata=allocation[sample_id],
                )
                all_items.append(row)
                by_key[(sample_id, block_index, variant)] = row

    if len(base_blocks) != 64 or len(all_items) != 448:
        raise ValueError(
            f"unexpected construct geometry: blocks={len(base_blocks)} items={len(all_items)}"
        )

    rater_items: list[dict[str, Any]] = []
    for sample_id, block_index in base_blocks:
        rater_items.append(by_key[(sample_id, block_index, "original")])
        rater_items.append(by_key[(sample_id, block_index, "neutral")])

    # Each method contributes 16 items: eight per arm, selected without score access.
    for method in ALL_METHODS:
        for arm in ("own_author_reconstruction", "cross_author_transfer"):
            eligible = [
                by_key[(sample_id, block_index, f"method:{method}")]
                for sample_id, block_index in base_blocks
                if allocation[sample_id]["benchmark_arm"] == arm
            ]
            eligible.sort(key=lambda row: stable_hash(f"{SEED}:{method}:{row['item_id']}"))
            rater_items.extend(eligible[:8])

    if len(rater_items) != 208 or len({row["item_id"] for row in rater_items}) != 208:
        raise ValueError("blind rater selection must contain exactly 208 unique items")
    method_counts = Counter(row["method_id"] for row in rater_items if row["method_id"])
    if method_counts != Counter({method: 16 for method in ALL_METHODS}):
        raise ValueError(f"unbalanced blind method allocation: {method_counts}")

    summary = {
        "source_samples": len(sample_ids),
        "source_arm_counts": dict(
            sorted(Counter(allocation[sample_id]["benchmark_arm"] for sample_id in sample_ids).items())
        ),
        "base_blocks": len(base_blocks),
        "all_construct_items": len(all_items),
        "factorial_target_informed_pairs": len(base_blocks) * len(TARGET_INFORMED_METHODS),
        "generic_control_pairs": len(base_blocks),
        "blind_rater_items": len(rater_items),
        "blind_rater_variant_counts": dict(
            sorted(Counter(row["variant"] for row in rater_items).items())
        ),
        "blind_rater_method_counts": dict(sorted(method_counts.items())),
    }
    return all_items, rater_items, summary


def build_rubric() -> dict[str, Any]:
    style_definition = read_json(STYLE_ASSET)["assets"]["style_definition"]["definition"]
    dimensions = [
        {
            "dimension_id": row["dimension_id"],
            "label": row["label"],
            "description": row["description"],
            "trigger": row["trigger"],
            "guardrail": row["guardrail"],
            "failure_mode": row["failure_mode"],
        }
        for row in style_definition["dimensions"]
    ]
    return {
        "schema_version": 1,
        "task": "blind_target_style_and_semantic_rating",
        "blinding": [
            "Do not inspect private mappings, meter scores, other rater outputs, or method names.",
            "The target is identified only as Style T; do not infer provenance from names or lore.",
            "Rate structural prose behavior, not topic, character identity, or perceived genre.",
        ],
        "style_dimensions": dimensions,
        "ratings": {
            "style_score": {
                "scale": "1-5 integer",
                "anchors": {
                    "1": "strongly inconsistent with Style T",
                    "2": "mostly inconsistent",
                    "3": "mixed or insufficient evidence",
                    "4": "mostly consistent",
                    "5": "strongly and repeatedly consistent",
                },
            },
            "semantic_fidelity": {
                "scale": "1-5 integer",
                "anchor_5": "all English propositions, speakers, relations, polarity, and modality preserved",
                "anchor_1": "major meaning loss or invention",
            },
            "naturalness": {
                "scale": "1-5 integer",
                "anchor_5": "publication-ready idiomatic Chinese",
                "anchor_1": "severely broken or translation-like Chinese",
            },
            "high_severity_semantic_error": (
                "true only for changed speaker/action/relation, reversed polarity or modality, "
                "invented consequential event, or material omission"
            ),
        },
        "required_output_fields": [
            "blind_id",
            "style_score",
            "semantic_fidelity",
            "naturalness",
            "high_severity_semantic_error",
            "rationale",
        ],
        "rationale_limit": "one concise sentence; do not name an author or method",
    }


def input_paths() -> list[Path]:
    paths = [
        Path(__file__),
        Path("experiments/validation/meter/build_cr_fysm_v4.py"),
        Path("experiments/validation/meter/develop_cr_fysm_v4.py"),
        Path("experiments/validation/meter/build_cr_fysm_v3.py"),
        Path("workflows/benchmark_author_style.py"),
        V4_ROOT / "preregistration.v4.json",
        V4_ROOT / "results.v4.json",
        V4_ROOT / "development.v4.json",
        ITER4_ROOT / "sample_sets/iteration4_proxy_v1.summary.json",
        ITER4_ROOT / "sample_sets/iteration4_proxy_v1.development_v1_ids.json",
        ITER4_ROOT / "sample_sets/iteration4_proxy_v1.evaluator_allocation.jsonl",
        ITER4_ROOT / "sample_sets/iteration4_proxy_v1.hidden_targets.jsonl",
        STYLE_ASSET,
    ]
    source_root = ITER4_ROOT / "runs/iteration4_proxy_v1/iteration4_source_gpt54_official"
    style_root = ITER4_ROOT / "runs/iteration4_proxy_v1/iteration4_style_gpt55_v1"
    _, _, sample_ids = load_inputs()
    for sample_id in sorted(sample_ids):
        paths.extend(
            [
                source_root / "english_semantic_source" / f"{sample_id}.json",
                source_root / "neutral_translation" / f"{sample_id}.json",
            ]
        )
        paths.extend(
            style_root / "method_outputs" / method / "strong" / f"{sample_id}.json"
            for method in ALL_METHODS
        )
    return paths


def artifact_hashes() -> dict[str, str]:
    return {
        str(path): sha256_file(path)
        for path in sorted(input_paths(), key=lambda value: str(value))
    }


def write_packets(rater_items: list[dict[str, Any]]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for offset, rater_name in enumerate(RATER_NAMES):
        rows = [
            {
                "blind_id": row["blind_id"],
                "english_source": row["english_source"],
                "chinese_candidate": row["chinese_candidate"],
            }
            for row in rater_items
        ]
        random.Random(SEED + offset).shuffle(rows)
        path = OUTPUT_DIR / f"{rater_name}.packet.jsonl"
        write_jsonl(path, rows)
        hashes[str(path)] = sha256_file(path)
    return hashes


def preregistration_payload(
    *,
    status: str,
    summary: dict[str, Any],
    packet_hashes: dict[str, str],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "construct_id": "CR-FYSM-v4-construct-v1",
        "status": status,
        "seed": SEED,
        "source_meter": {
            "meter_id": "CR-FYSM-v4",
            "lock_id": read_json(V4_ROOT / "preregistration.v4.json")["lock_id"],
            "threshold": read_json(V4_ROOT / "development.v4.json")["selected"]["threshold"],
            "statistical_result": read_json(V4_ROOT / "results.v4.json")["status"],
            "model_refit_for_construct": False,
            "threshold_retuned_for_construct": False,
        },
        "data_role": {
            "source": "Iteration 4 development texts generated before CR-FYSM-v4 construction",
            "fresh_to_v4_scoring": True,
            "production_compatible_neutral_prompt": True,
            "paragraph_alignment": "exact IDs; four disjoint quartile blocks per source",
            "effective_cluster": "source sample, never paragraph block",
            "known_limitation": (
                "The source texts are not globally pristine because Iteration 4 used them; "
                "they are untouched by CR-FYSM-v4 fitting and threshold selection."
            ),
        },
        "geometry": summary,
        "methods": {
            "generic_negative_control": GENERIC_CONTROL,
            "target_informed": list(TARGET_INFORMED_METHODS),
        },
        "blind_rating_protocol": {
            "raters": 3,
            "rater_type": "independent expert LLM agents; never described as human",
            "same_items": True,
            "independently_shuffled_packets": True,
            "forbidden_inputs": [
                "private item mapping",
                "meter scores",
                "source author and book",
                "variant and method labels",
                "other rater outputs",
            ],
            "outputs": [str(OUTPUT_DIR / f"{name}.ratings.jsonl") for name in RATER_NAMES],
        },
        "gates": {
            "three_rater_convergence": {
                "krippendorff_alpha_ordinal_min": 0.67,
                "meter_consensus_spearman_min": 0.50,
                "cluster_bootstrap_spearman_lower_min": 0.30,
                "bootstrap_resamples": 5000,
            },
            "generated_domain": {
                "consensus_positive": "median style_score >= 4",
                "consensus_negative": "median style_score <= 2",
                "ambiguous_excluded_from_binary_only": "median style_score == 3",
                "minimum_items_per_origin_label_cell": 20,
                "human_sensitivity_min": 0.80,
                "human_specificity_min": 0.90,
                "generated_sensitivity_min": 0.80,
                "generated_specificity_min": 0.90,
                "overall_ece_max": 0.10,
            },
            "matched_original_neutral": {
                "target_original_minus_neutral_positive": True,
                "cluster_bootstrap_lower_gt": 0.0,
                "cross_original_false_positive_rate_max": 0.10,
            },
            "factorial": {
                "minimum_semantically_valid_target_informed_pairs": 200,
                "semantic_validity": (
                    "median semantic_fidelity >=4, no rater high-severity error, and exact artifact alignment"
                ),
                "style_main_effect_positive": True,
                "cluster_bootstrap_style_lower_gt": 0.0,
                "style_effect_to_absolute_content_effect_ratio_min": 3.0,
                "material_interaction_max": 0.10,
                "interaction_interval_must_include_zero": True,
                "unit": "paired raw CR-FYSM-v4 score lift over same-content neutral block",
            },
        },
        "one_shot_policy": {
            "score_output": str(OUTPUT_DIR / "construct_scores.v1.jsonl"),
            "opened_marker": str(OUTPUT_DIR / "construct_scoring_opened.v1.json"),
            "result_output": str(OUTPUT_DIR / "construct_results.v1.json"),
            "private_mapping_never_given_to_raters": True,
            "failure_requires_new_construct_version": True,
        },
        "input_hashes": artifact_hashes(),
        "prepared_artifact_hashes": {
            str(PRIVATE_ITEMS): sha256_file(PRIVATE_ITEMS),
            str(RUBRIC): sha256_file(RUBRIC),
            **packet_hashes,
        },
    }
    payload["lock_id"] = canonical_lock_id(payload)
    return payload


def prepare(mode: str) -> None:
    if mode == "validate":
        payload = read_json(PREREGISTRATION)
        if payload.get("status") != "locked_before_construct_scoring_and_rating":
            raise ValueError("construct preregistration is not locked")
        if payload.get("lock_id") != canonical_lock_id(payload):
            raise ValueError("construct lock_id is invalid")
        if payload.get("input_hashes") != artifact_hashes():
            raise ValueError("construct input hashes changed")
        actual_prepared = {
            path: sha256_file(Path(path))
            for path in payload["prepared_artifact_hashes"]
        }
        if actual_prepared != payload["prepared_artifact_hashes"]:
            raise ValueError("construct prepared artifacts changed")
        print(json.dumps({"status": "valid", "lock_id": payload["lock_id"]}, indent=2))
        return

    if mode == "lock" and not PREREGISTRATION.exists():
        raise ValueError("create and audit the draft before locking")
    if mode == "lock":
        existing = read_json(PREREGISTRATION)
        if existing.get("status") != "draft_before_construct_scoring_and_rating":
            raise ValueError("only an audited construct draft can be locked")
        if existing.get("input_hashes") != artifact_hashes():
            raise ValueError("inputs changed since construct draft preparation")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_items, rater_items, summary = build_items()
    write_jsonl(PRIVATE_ITEMS, all_items)
    write_json(RUBRIC, build_rubric())
    packet_hashes = write_packets(rater_items)
    status = (
        "locked_before_construct_scoring_and_rating"
        if mode == "lock"
        else "draft_before_construct_scoring_and_rating"
    )
    payload = preregistration_payload(
        status=status,
        summary=summary,
        packet_hashes=packet_hashes,
    )
    write_json(PREREGISTRATION, payload)
    print(
        json.dumps(
            {
                "status": status,
                "lock_id": payload["lock_id"],
                "geometry": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    prepare(parse_args().mode)


if __name__ == "__main__":
    main()
