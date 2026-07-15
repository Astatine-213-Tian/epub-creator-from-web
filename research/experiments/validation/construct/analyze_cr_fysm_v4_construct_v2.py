#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from experiments.validation.construct import prepare_cr_fysm_v4_construct_analysis_v2 as prepare
from experiments.validation.construct import run_cr_fysm_v4_construct_ratings_v2 as ratings
from experiments.validation.meter.build_cr_fysm_v3 import (
    ensemble,
    family_probabilities,
    locked_family_contract,
)
from experiments.validation.meter.develop_cr_fysm_v4 import load_family_artifacts


RESULTS = prepare.ROOT / "results.json"
OPENED = prepare.ROOT / "analysis_opened.json"
REPORT = prepare.ROOT / "report.md"
BOOTSTRAP_REPETITIONS = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the locked CR-FYSM-v4 fresh construct experiment."
    )
    parser.add_argument(
        "--resume-opened",
        action="store_true",
        help=(
            "Resume only an implementation-failed open analysis with the same lock "
            "and byte-identical rating artifacts; never rerun a completed result."
        ),
    )
    return parser.parse_args()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_analysis_lock() -> dict[str, Any]:
    return ratings.validate_lock()


def load_rating_judgments(
    lock: dict[str, Any],
    *,
    identity: str,
    task: str,
    packet_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    prompt = prepare.STYLE_PROMPT if task == "style" else prepare.SEMANTIC_PROMPT
    schema = prepare.STYLE_SCHEMA if task == "style" else prepare.SEMANTIC_SCHEMA
    key = "pair_id" if task == "style" else "item_id"
    judgments: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for packet_path in sorted(packet_dir.glob("*.json")):
        packet = read_json(packet_path)
        artifact_path = ratings.rating_path(identity, packet["packet_id"])
        if not artifact_path.exists():
            raise FileNotFoundError(f"missing rating artifact: {artifact_path}")
        result = ratings.validate_artifact(
            read_json(artifact_path),
            lock=lock,
            identity=identity,
            task=task,
            packet_path=packet_path,
            prompt_path=prompt,
            schema_path=schema,
        )
        hashes[str(artifact_path)] = prepare.sha256_file(artifact_path)
        for judgment in result["judgments"]:
            value = str(judgment[key])
            if value in judgments:
                raise ValueError(f"duplicate {identity} judgment: {value}")
            judgments[value] = judgment
    expected_artifacts = {
        ratings.rating_path(identity, path.stem)
        for path in packet_dir.glob("*.json")
    }
    extra_artifacts = set((ratings.RATINGS / identity).glob("*.json")) - expected_artifacts
    if extra_artifacts:
        raise ValueError(f"unexpected rating artifacts for {identity}: {sorted(extra_artifacts)}")
    return judgments, hashes


def load_all_ratings(
    lock: dict[str, Any],
    item_ids: set[str],
    pair_ids: set[str],
) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, dict[str, Any]]],
    dict[str, str],
]:
    semantic: dict[str, dict[str, dict[str, Any]]] = {}
    style: dict[str, dict[str, dict[str, Any]]] = {}
    hashes: dict[str, str] = {}
    for identity in prepare.VALIDATOR_IDS:
        values, value_hashes = load_rating_judgments(
            lock,
            identity=identity,
            task="semantic",
            packet_dir=prepare.PACKETS / identity,
        )
        if set(values) != item_ids:
            raise ValueError(f"semantic coverage mismatch for {identity}")
        semantic[identity] = values
        hashes.update(value_hashes)

    manifest = read_json(ratings.ADJUDICATION_MANIFEST)
    expected_manifest_id = dict(manifest)
    observed_manifest_id = expected_manifest_id.pop("manifest_id", None)
    if observed_manifest_id != sha256_text(canonical(expected_manifest_id)):
        raise ValueError("semantic adjudication manifest_id is invalid")
    if manifest.get("analysis_lock_id") != lock["lock_id"]:
        raise ValueError("semantic adjudication manifest is stale")
    for identity in prepare.VALIDATOR_IDS:
        key = f"{identity}_rating_hashes"
        expected_hashes = {
            str(path): prepare.sha256_file(path)
            for path in sorted((ratings.RATINGS / identity).glob("*.json"))
        }
        if manifest.get(key) != expected_hashes:
            raise ValueError(f"semantic adjudication manifest does not bind {identity}")
    disagreement_ids = set(manifest["disagreement_item_ids"])
    observed_disagreements = {
        item_id
        for item_id in item_ids
        if ratings.semantic_pass(semantic["validator_a"][item_id])
        != ratings.semantic_pass(semantic["validator_b"][item_id])
    }
    if disagreement_ids != observed_disagreements:
        raise ValueError("semantic adjudication selection differs from validator disagreements")
    if manifest.get("packet_hashes") != {
        str(path): prepare.sha256_file(path)
        for path in sorted((prepare.PACKETS / "validator_c").glob("*.json"))
    }:
        raise ValueError("semantic adjudication packets changed")
    if disagreement_ids:
        values, value_hashes = load_rating_judgments(
            lock,
            identity="validator_c",
            task="semantic",
            packet_dir=prepare.PACKETS / "validator_c",
        )
        if set(values) != disagreement_ids:
            raise ValueError("semantic adjudicator coverage mismatch")
        semantic["validator_c"] = values
        hashes.update(value_hashes)
    else:
        semantic["validator_c"] = {}
        if any((ratings.RATINGS / "validator_c").glob("*.json")):
            raise ValueError("adjudicator ratings exist without disagreements")
    hashes[str(ratings.ADJUDICATION_MANIFEST)] = prepare.sha256_file(
        ratings.ADJUDICATION_MANIFEST
    )

    for identity in prepare.RATER_IDS:
        values, value_hashes = load_rating_judgments(
            lock,
            identity=identity,
            task="style",
            packet_dir=prepare.PACKETS / identity,
        )
        if set(values) != pair_ids:
            raise ValueError(f"style coverage mismatch for {identity}")
        style[identity] = values
        hashes.update(value_hashes)
    return semantic, style, dict(sorted(hashes.items()))


def final_semantic_decision(
    item_id: str,
    semantic: dict[str, dict[str, dict[str, Any]]],
) -> tuple[bool, str]:
    first = ratings.semantic_pass(semantic["validator_a"][item_id])
    second = ratings.semantic_pass(semantic["validator_b"][item_id])
    if first == second:
        return first, "two_validator_agreement"
    adjudicator = semantic["validator_c"].get(item_id)
    if adjudicator is None:
        raise ValueError(f"missing adjudication for {item_id}")
    return ratings.semantic_pass(adjudicator), "third_validator_adjudication"


def meter_scores(lock: dict[str, Any], texts: list[str]) -> tuple[np.ndarray, float]:
    postclean = read_json(prepare.POSTCLEAN_PREREGISTRATION)
    source = postclean["historical_source"]
    family_order, family_weights = locked_family_contract(source["model_contract"])
    model_dir = Path(source["source_artifact_paths"]["function_grammar_vectorizer"]).parent
    artifacts = load_family_artifacts(model_dir, family_order)
    probabilities = family_probabilities(artifacts, texts)
    scores = ensemble(probabilities, family_order, family_weights)
    threshold = float(source["decision_policy"]["threshold"])
    if lock["input_hashes"][str(prepare.POSTCLEAN_RESULTS)] != prepare.sha256_file(
        prepare.POSTCLEAN_RESULTS
    ):
        raise ValueError("post-clean meter result binding changed")
    return scores, threshold


def wilson_lower(successes: int, total: int, *, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = proportion + z * z / (2.0 * total)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    )
    return (center - radius) / denominator


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def spearman(values_a: Iterable[float], values_b: Iterable[float]) -> float:
    first = np.asarray(list(values_a), dtype=np.float64)
    second = np.asarray(list(values_b), dtype=np.float64)
    if len(first) < 2 or len(first) != len(second):
        return float("nan")
    rank_a = average_ranks(first)
    rank_b = average_ranks(second)
    if np.ptp(rank_a) == 0.0 or np.ptp(rank_b) == 0.0:
        return float("nan")
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def source_bootstrap_lower(
    values: dict[str, float],
    *,
    seed: int,
    transform: str = "mean",
) -> float:
    source_ids = sorted(values)
    observed = np.asarray([values[source_id] for source_id in source_ids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(source_ids), size=(BOOTSTRAP_REPETITIONS, len(source_ids)))
    sampled = observed[draws]
    if transform == "positive_proportion":
        estimates = np.mean(sampled > 0.0, axis=1)
    elif transform == "mean":
        estimates = np.mean(sampled, axis=1)
    else:
        raise ValueError(transform)
    return float(np.quantile(estimates, 0.025))


def correlation_bootstrap_lower(
    values_a: dict[str, float],
    values_b: dict[str, float],
    *,
    seed: int,
) -> float:
    source_ids = sorted(values_a)
    if source_ids != sorted(values_b):
        raise ValueError("correlation source sets differ")
    first = np.asarray([values_a[source_id] for source_id in source_ids], dtype=np.float64)
    second = np.asarray([values_b[source_id] for source_id in source_ids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
    for index in range(BOOTSTRAP_REPETITIONS):
        drawn = rng.integers(0, len(source_ids), size=len(source_ids))
        value = spearman(first[drawn], second[drawn])
        estimates[index] = -1.0 if math.isnan(value) else value
    return float(np.quantile(estimates, 0.025))


def decode_style_vote(
    judgment: dict[str, Any],
    *,
    side_order: str,
) -> int:
    choice = judgment["choice"]
    if choice == "tie":
        return 0
    style_side = "A" if side_order == "style_first" else "B"
    return 1 if choice == style_side else -1


def make_summary_chart(path: Path, rows: list[dict[str, Any]]) -> None:
    width = 940
    left = 330
    bar_width = 500
    row_height = 48
    height = 70 + row_height * len(rows)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#202124;letter-spacing:0}.label{font-size:14px}.value{font-size:13px}.axis{stroke:#9aa0a6;stroke-width:1}.gate{stroke:#b3261e;stroke-width:2;stroke-dasharray:5 4}</style>',
        f'<line x1="{left}" y1="35" x2="{left + bar_width}" y2="35" class="axis"/>',
    ]
    for tick in range(0, 101, 20):
        x = left + bar_width * tick / 100
        parts.append(f'<text x="{x}" y="24" text-anchor="middle" class="value">{tick}%</text>')
    for index, row in enumerate(rows):
        y = 55 + index * row_height
        value = max(0.0, min(1.0, float(row["value"])))
        gate = max(0.0, min(1.0, float(row["gate"])))
        color = "#146c43" if row["passed"] else "#b3261e"
        parts.extend(
            [
                f'<text x="12" y="{y + 18}" class="label">{row["label"]}</text>',
                f'<rect x="{left}" y="{y}" width="{bar_width}" height="24" fill="#eceff1"/>',
                f'<rect x="{left}" y="{y}" width="{bar_width * value:.2f}" height="24" fill="{color}"/>',
                f'<line x1="{left + bar_width * gate:.2f}" y1="{y - 4}" x2="{left + bar_width * gate:.2f}" y2="{y + 28}" class="gate"/>',
                f'<text x="{left + bar_width + 12}" y="{y + 18}" class="value">{value:.1%}</text>',
            ]
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    lock = validate_analysis_lock()
    if RESULTS.exists():
        raise ValueError("construct-v2 analysis already has a completed result")
    if OPENED.exists() and not args.resume_opened:
        raise ValueError(
            "construct-v2 analysis was opened; use --resume-opened only for an "
            "implementation-failed run with no completed result"
        )
    if not OPENED.exists() and args.resume_opened:
        raise ValueError("--resume-opened requires an existing analysis opening marker")

    items = prepare.read_jsonl(prepare.PRIVATE_ITEMS)
    pairs = prepare.read_jsonl(prepare.PRIVATE_PAIRS)
    item_by_id = {row["item_id"]: row for row in items}
    pair_by_id = {row["pair_id"]: row for row in pairs}
    if len(item_by_id) != 750 or len(pair_by_id) != 500:
        raise ValueError("private construct geometry changed")
    semantic, style, rating_hashes = load_all_ratings(
        lock, set(item_by_id), set(pair_by_id)
    )
    marker = {
        "analysis_id": lock["analysis_id"],
        "analysis_lock_id": lock["lock_id"],
        "status": "opened_irreversibly_after_complete_rating_validation",
        "rating_artifact_hashes": rating_hashes,
        "recovery_rule": (
            "Rerun only with --resume-opened when results.json is absent and the "
            "analysis lock plus every rating artifact hash remains byte-identical."
        ),
    }
    if OPENED.exists():
        if read_json(OPENED) != marker:
            raise ValueError("opened analysis cannot resume because its bindings changed")
    else:
        descriptor = os.open(OPENED, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(marker, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    item_scores, threshold = meter_scores(
        lock, [str(row["candidate_masked_v3_zh"]) for row in items]
    )
    score_by_item = {
        row["item_id"]: float(item_scores[index]) for index, row in enumerate(items)
    }
    semantic_rows: list[dict[str, Any]] = []
    semantic_valid_by_item: dict[str, bool] = {}
    for row in items:
        valid, decision_source = final_semantic_decision(row["item_id"], semantic)
        semantic_valid_by_item[row["item_id"]] = valid
        semantic_rows.append(
            {
                "item_id": row["item_id"],
                "sample_id": row["sample_id"],
                "block_id": row["block_id"],
                "variant": row["variant"],
                "valid": valid,
                "decision_source": decision_source,
                "validator_a_pass": ratings.semantic_pass(
                    semantic["validator_a"][row["item_id"]]
                ),
                "validator_b_pass": ratings.semantic_pass(
                    semantic["validator_b"][row["item_id"]]
                ),
                "adjudicator_pass": (
                    ratings.semantic_pass(semantic["validator_c"][row["item_id"]])
                    if row["item_id"] in semantic["validator_c"]
                    else ""
                ),
                "meter_score": score_by_item[row["item_id"]],
                "reference_4gram_overlap": (
                    row["reference_4gram_overlap"]
                    if row["reference_4gram_overlap"] is not None
                    else ""
                ),
            }
        )

    items_by_block: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in items:
        items_by_block[(row["sample_id"], row["block_id"])][row["variant"]] = row
    expected_variants = set(prepare.VARIANT_STAGES)
    valid_block: dict[tuple[str, str], bool] = {}
    for key, variants in items_by_block.items():
        if set(variants) != expected_variants:
            raise ValueError(f"incomplete block variants: {key}")
        valid_block[key] = all(
            semantic_valid_by_item[variants[variant]["item_id"]]
            for variant in expected_variants
        )
    valid_blocks_by_source: dict[str, int] = defaultdict(int)
    for (sample_id, _), valid in valid_block.items():
        valid_blocks_by_source[sample_id] += int(valid)
    valid_block_total = sum(valid_block.values())

    selections = prepare.generation.read_jsonl(prepare.generation.SELECTION)
    original_scores, _ = meter_scores(
        lock, [str(row["entity_masked_v3_zh"]) for row in selections]
    )
    known_rows: list[dict[str, Any]] = []
    target_successes = 0
    comparison_successes = 0
    target_total = 0
    comparison_total = 0
    for index, sample in enumerate(selections):
        is_target = str(sample["source_role"]).startswith("target_")
        predicted_target = float(original_scores[index]) >= threshold
        if is_target:
            target_total += 1
            target_successes += int(predicted_target)
        else:
            comparison_total += 1
            comparison_successes += int(not predicted_target)
        known_rows.append(
            {
                "sample_id": sample["sample_id"],
                "source_role": sample["source_role"],
                "source_author": sample["source_author"],
                "source_book": sample["source_book"],
                "meter_score": float(original_scores[index]),
                "predicted_target": predicted_target,
                "correct": predicted_target if is_target else not predicted_target,
            }
        )
    sensitivity = target_successes / target_total
    specificity = comparison_successes / comparison_total
    if (target_total, comparison_total) != (25, 25):
        raise ValueError(
            f"fresh known-group geometry changed: target={target_total}, comparison={comparison_total}"
        )
    balanced_accuracy = (sensitivity + specificity) / 2.0
    sensitivity_lower = wilson_lower(target_successes, target_total)
    specificity_lower = wilson_lower(comparison_successes, comparison_total)

    style_rows: list[dict[str, Any]] = []
    majority_by_pair: dict[str, int] = {}
    margin_by_pair: dict[str, float] = {}
    for pair in pairs:
        votes = [
            decode_style_vote(
                style[identity][pair["pair_id"]],
                side_order=pair["side_order"][identity],
            )
            for identity in prepare.RATER_IDS
        ]
        vote_sum = sum(votes)
        majority = 1 if vote_sum >= 2 else -1 if vote_sum <= -2 else 0
        majority_by_pair[pair["pair_id"]] = majority
        margin_by_pair[pair["pair_id"]] = float(np.mean(votes))
        style_rows.append(
            {
                "pair_id": pair["pair_id"],
                "pair_type": pair["pair_type"],
                "sample_id": pair["sample_id"],
                "block_id": pair["block_id"],
                "block_semantically_valid": valid_block[(pair["sample_id"], pair["block_id"])],
                "rater_a_vote": votes[0],
                "rater_b_vote": votes[1],
                "rater_c_vote": votes[2],
                "vote_margin": margin_by_pair[pair["pair_id"]],
                "majority": "style" if majority == 1 else "control" if majority == -1 else "tie",
            }
        )

    pair_for_block = {
        (row["sample_id"], row["block_id"], row["pair_type"]): row
        for row in pairs
    }
    block_rows: list[dict[str, Any]] = []
    d_blocks: dict[tuple[str, str], float] = {}
    g_blocks: dict[tuple[str, str], float] = {}
    overlap_by_block: dict[tuple[str, str], float] = {}
    for key, variants in sorted(items_by_block.items()):
        sample_id, block_id = key
        neutral = score_by_item[variants["neutral_translation"]["item_id"]]
        sham = score_by_item[variants["style_sham_control"]["item_id"]]
        style_score = score_by_item[variants["style_positive_control"]["item_id"]]
        delta = style_score - sham
        discriminant = (style_score - sham) - (sham - neutral)
        overlap = float(variants["style_positive_control"]["reference_4gram_overlap"])
        primary = pair_for_block[(sample_id, block_id, "primary_style_vs_sham")]
        secondary = pair_for_block[(sample_id, block_id, "secondary_style_vs_neutral")]
        d_blocks[key] = delta
        g_blocks[key] = discriminant
        overlap_by_block[key] = overlap
        block_rows.append(
            {
                "sample_id": sample_id,
                "block_id": block_id,
                "valid": valid_block[key],
                "neutral_score": neutral,
                "sham_score": sham,
                "style_positive_score": style_score,
                "style_minus_sham": delta,
                "generic_rewrite_discriminant": discriminant,
                "reference_4gram_overlap": overlap,
                "primary_majority": majority_by_pair[primary["pair_id"]],
                "primary_vote_margin": margin_by_pair[primary["pair_id"]],
                "secondary_majority": majority_by_pair[secondary["pair_id"]],
                "secondary_vote_margin": margin_by_pair[secondary["pair_id"]],
            }
        )

    source_ids = sorted({row["sample_id"] for row in items})
    d_source: dict[str, float] = {}
    g_source: dict[str, float] = {}
    primary_preference_source: dict[str, float] = {}
    primary_margin_source: dict[str, float] = {}
    secondary_preference_source: dict[str, float] = {}
    source_rows: list[dict[str, Any]] = []
    for source_id in source_ids:
        keys = [
            key for key in sorted(valid_block) if key[0] == source_id and valid_block[key]
        ]
        if not keys:
            d_source[source_id] = float("nan")
            g_source[source_id] = float("nan")
            primary_preference_source[source_id] = 0.0
            primary_margin_source[source_id] = 0.0
            secondary_preference_source[source_id] = 0.0
            continue
        primary_pairs = [
            pair_for_block[(source_id, key[1], "primary_style_vs_sham")] for key in keys
        ]
        secondary_pairs = [
            pair_for_block[(source_id, key[1], "secondary_style_vs_neutral")] for key in keys
        ]
        d_source[source_id] = float(np.mean([d_blocks[key] for key in keys]))
        g_source[source_id] = float(np.mean([g_blocks[key] for key in keys]))
        primary_preference_source[source_id] = float(
            np.mean([majority_by_pair[pair["pair_id"]] == 1 for pair in primary_pairs])
        )
        primary_margin_source[source_id] = float(
            np.mean([margin_by_pair[pair["pair_id"]] for pair in primary_pairs])
        )
        secondary_preference_source[source_id] = float(
            np.mean([majority_by_pair[pair["pair_id"]] == 1 for pair in secondary_pairs])
        )
        source_rows.append(
            {
                "sample_id": source_id,
                "valid_blocks": len(keys),
                "D_style_minus_sham": d_source[source_id],
                "G_generic_rewrite_discriminant": g_source[source_id],
                "primary_majority_style_proportion": primary_preference_source[source_id],
                "primary_vote_margin": primary_margin_source[source_id],
                "secondary_majority_style_proportion": secondary_preference_source[source_id],
            }
        )

    complete_sources = len(d_source) == 50 and all(
        not math.isnan(value) for value in d_source.values()
    )
    d_mean = float(np.mean(list(d_source.values()))) if complete_sources else float("nan")
    d_lower = (
        source_bootstrap_lower(d_source, seed=prepare.SEED) if complete_sources else float("nan")
    )
    positive_proportion = (
        float(np.mean([value > 0.0 for value in d_source.values()]))
        if complete_sources
        else 0.0
    )
    positive_lower = (
        source_bootstrap_lower(
            d_source, seed=prepare.SEED, transform="positive_proportion"
        )
        if complete_sources
        else float("nan")
    )
    g_mean = float(np.mean(list(g_source.values()))) if complete_sources else float("nan")
    g_lower = (
        source_bootstrap_lower(g_source, seed=prepare.SEED) if complete_sources else float("nan")
    )
    primary_preference = float(np.mean(list(primary_preference_source.values())))
    primary_preference_lower = source_bootstrap_lower(
        primary_preference_source, seed=prepare.SEED
    )
    secondary_preference = float(np.mean(list(secondary_preference_source.values())))
    convergence = spearman(d_source.values(), primary_margin_source.values())
    convergence_lower = (
        correlation_bootstrap_lower(
            d_source, primary_margin_source, seed=prepare.SEED
        )
        if complete_sources
        else float("nan")
    )

    valid_keys = [key for key, valid in valid_block.items() if valid]
    removal_count = math.ceil(len(valid_keys) * 0.10)
    ranked_for_removal = sorted(
        valid_keys,
        key=lambda key: (
            -overlap_by_block[key],
            pair_for_block[(key[0], key[1], "primary_style_vs_sham")]["pair_id"],
        ),
    )
    removed = set(ranked_for_removal[:removal_count])
    retained_by_source: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in valid_keys:
        if key not in removed:
            retained_by_source[key[0]].append(key)
    overlap_sources_complete = set(retained_by_source) == set(source_ids) and all(
        retained_by_source[source_id] for source_id in source_ids
    )
    overlap_removed_delta = (
        float(
            np.mean(
                [
                    np.mean([d_blocks[key] for key in retained_by_source[source_id]])
                    for source_id in source_ids
                ]
            )
        )
        if overlap_sources_complete
        else float("nan")
    )
    maximum_overlap = max(overlap_by_block.values())
    overlap_limit = float(lock["reference_overlap_control"]["maximum_overlap_rate"])

    source_amendment = read_json(
        Path(lock["generation_protocol_amendment"]["path"])
    )
    adjudication_amendment = read_json(
        Path(lock["english_adjudication_protocol_amendment"]["path"])
    )
    source_amendment_ids = set(source_amendment["sample_ids"])
    adjudication_amendment_ids = set(adjudication_amendment["sample_ids"])
    if source_amendment_ids != adjudication_amendment_ids:
        raise ValueError("source and adjudication amendment sample sets differ")
    style_positive_incident = read_json(
        Path(lock["style_positive_protocol_amendment"]["incident"]["path"])
    )
    style_positive_incident_ids = set(style_positive_incident["affected_sample_ids"])
    if style_positive_incident_ids != set(
        lock["style_positive_protocol_amendment"]["incident"]["affected_source_ids"]
    ):
        raise ValueError("style-positive incident source set differs from analysis lock")
    amended_source_ids = (
        source_amendment_ids
        | adjudication_amendment_ids
        | style_positive_incident_ids
    )
    unamended_source_ids = sorted(set(source_ids) - amended_source_ids)
    d_unamended = {source_id: d_source[source_id] for source_id in unamended_source_ids}
    preference_unamended = {
        source_id: primary_preference_source[source_id]
        for source_id in unamended_source_ids
    }
    margin_unamended = {
        source_id: primary_margin_source[source_id] for source_id in unamended_source_ids
    }
    known_unamended = [
        row for row in known_rows if row["sample_id"] not in amended_source_ids
    ]
    unamended_target = [
        row
        for row in known_unamended
        if str(row["source_role"]).startswith("target_")
    ]
    unamended_comparison = [
        row
        for row in known_unamended
        if not str(row["source_role"]).startswith("target_")
    ]
    unamended_sensitivity = float(np.mean([row["correct"] for row in unamended_target]))
    unamended_specificity = float(
        np.mean([row["correct"] for row in unamended_comparison])
    )
    unamended_balanced_accuracy = (unamended_sensitivity + unamended_specificity) / 2.0
    unamended_convergence = spearman(
        d_unamended.values(), margin_unamended.values()
    )
    amendment_exclusion_sensitivity = {
        "source_amendment_ids": sorted(source_amendment_ids),
        "adjudication_amendment_ids": sorted(adjudication_amendment_ids),
        "style_positive_incident_ids": sorted(style_positive_incident_ids),
        "excluded_source_ids": sorted(amended_source_ids),
        "remaining_sources": len(unamended_source_ids),
        "known_group_balanced_accuracy": unamended_balanced_accuracy,
        "known_group_target_sensitivity": unamended_sensitivity,
        "known_group_comparison_specificity": unamended_specificity,
        "mean_source_style_minus_sham": float(np.mean(list(d_unamended.values()))),
        "mean_source_style_minus_sham_bootstrap_95pct_lower": source_bootstrap_lower(
            d_unamended, seed=prepare.SEED
        ),
        "positive_source_proportion": float(
            np.mean([value > 0.0 for value in d_unamended.values()])
        ),
        "blind_style_over_sham_majority_proportion": float(
            np.mean(list(preference_unamended.values()))
        ),
        "meter_rater_spearman": unamended_convergence,
    }

    gates = lock["outcome_identification_protocol"]["primary_gates"]
    gate_results = {
        "completion_and_semantics": (
            len(items) == 750
            and len(pairs) == 500
            and valid_block_total >= 200
            and len(valid_blocks_by_source) == 50
            and min(valid_blocks_by_source.values()) >= 4
        ),
        "fresh_human_known_groups": (
            balanced_accuracy >= float(gates["fresh_human_known_groups"]["balanced_accuracy_min"])
            and sensitivity >= float(gates["fresh_human_known_groups"]["target_sensitivity_min"])
            and specificity >= float(gates["fresh_human_known_groups"]["comparison_specificity_min"])
            and sensitivity_lower >= float(gates["fresh_human_known_groups"]["wilson_95pct_lower_each_min"])
            and specificity_lower >= float(gates["fresh_human_known_groups"]["wilson_95pct_lower_each_min"])
        ),
        "oracle_meter_sensitivity": (
            complete_sources
            and d_lower > float(gates["oracle_meter_sensitivity"]["mean_delta_source_cluster_bootstrap_95pct_lower_min"])
            and positive_proportion >= float(gates["oracle_meter_sensitivity"]["positive_source_proportion_min"])
            and positive_lower >= float(gates["oracle_meter_sensitivity"]["positive_source_proportion_bootstrap_95pct_lower_min"])
        ),
        "generic_rewrite_discriminant": (
            complete_sources
            and g_lower > float(gates["generic_rewrite_discriminant"]["source_cluster_bootstrap_95pct_lower_min"])
        ),
        "blind_close_reading_manipulation_check": (
            primary_preference >= float(gates["blind_close_reading_manipulation_check"]["majority_style_preference_min"])
            and primary_preference_lower >= float(gates["blind_close_reading_manipulation_check"]["source_cluster_bootstrap_95pct_lower_min"])
        ),
        "meter_rater_convergence": (
            not math.isnan(convergence)
            and convergence >= float(gates["meter_rater_convergence"]["point_min"])
            and convergence_lower > float(gates["meter_rater_convergence"]["source_cluster_bootstrap_95pct_lower_min"])
        ),
        "reference_leakage": (
            maximum_overlap <= overlap_limit
            and overlap_sources_complete
            and overlap_removed_delta > 0.0
        ),
    }
    passed = all(gate_results.values())
    result = {
        "schema_version": 2,
        "analysis_id": lock["analysis_id"],
        "status": (
            "construct_pass_secondary_domain_bounded_meter"
            if passed
            else "construct_fail_meter_exploratory_only"
        ),
        "claim_limit": lock["outcome_identification_protocol"]["decision"],
        "analysis_lock": {
            "path": str(prepare.PREREGISTRATION),
            "lock_id": lock["lock_id"],
            "sha256": prepare.sha256_file(prepare.PREREGISTRATION),
        },
        "rating_artifact_hashes": rating_hashes,
        "geometry": {
            "sources": len(source_ids),
            "blocks": len(valid_block),
            "semantic_items": len(items),
            "style_pairs": len(pairs),
            "valid_blocks": valid_block_total,
            "minimum_valid_blocks_per_source": min(valid_blocks_by_source.values()),
            "semantic_disagreements": len(semantic["validator_c"]),
        },
        "frozen_meter": {
            "threshold": threshold,
            "known_groups": {
                "balanced_accuracy": balanced_accuracy,
                "target_sensitivity": sensitivity,
                "comparison_specificity": specificity,
                "target_wilson_95pct_lower": sensitivity_lower,
                "comparison_wilson_95pct_lower": specificity_lower,
                "target_n": target_total,
                "comparison_n": comparison_total,
            },
            "oracle_style_vs_sham": {
                "mean_source_delta": d_mean,
                "bootstrap_95pct_lower": d_lower,
                "positive_source_proportion": positive_proportion,
                "positive_source_bootstrap_95pct_lower": positive_lower,
            },
            "generic_rewrite_discriminant": {
                "mean_source_contrast": g_mean,
                "bootstrap_95pct_lower": g_lower,
            },
        },
        "blind_style_rating": {
            "primary_style_vs_sham_majority_proportion": primary_preference,
            "primary_source_bootstrap_95pct_lower": primary_preference_lower,
            "secondary_style_vs_neutral_majority_proportion": secondary_preference,
            "meter_rater_spearman": convergence,
            "meter_rater_spearman_bootstrap_95pct_lower": convergence_lower,
        },
        "reference_leakage": {
            "maximum_4gram_overlap": maximum_overlap,
            "frozen_limit": overlap_limit,
            "top_decile_blocks_removed": removal_count,
            "all_sources_retained": overlap_sources_complete,
            "oracle_delta_after_top_decile_removal": overlap_removed_delta,
        },
        "protocol_amendment_exclusion_sensitivity": amendment_exclusion_sensitivity,
        "gate_results": gate_results,
    }
    write_csv(prepare.ROOT / "semantic_items.csv", semantic_rows)
    write_csv(prepare.ROOT / "known_groups.csv", known_rows)
    write_csv(prepare.ROOT / "style_votes.csv", style_rows)
    write_csv(prepare.ROOT / "block_scores.csv", block_rows)
    write_csv(prepare.ROOT / "source_estimands.csv", source_rows)

    chart_rows = [
        {
            "label": "Known-group balanced accuracy",
            "value": balanced_accuracy,
            "gate": 0.80,
            "passed": gate_results["fresh_human_known_groups"],
        },
        {
            "label": "Positive sources: style > sham",
            "value": positive_proportion,
            "gate": 0.70,
            "passed": gate_results["oracle_meter_sensitivity"],
        },
        {
            "label": "Blind majority: style > sham",
            "value": primary_preference,
            "gate": 0.65,
            "passed": gate_results["blind_close_reading_manipulation_check"],
        },
        {
            "label": "Meter-rater Spearman",
            "value": max(0.0, convergence) if not math.isnan(convergence) else 0.0,
            "gate": 0.20,
            "passed": gate_results["meter_rater_convergence"],
        },
    ]
    make_summary_chart(prepare.ROOT / "construct_gate_summary.svg", chart_rows)

    lines = [
        "# CR-FYSM-v4 Fresh Construct Validation v2",
        "",
        f"- Status: **{result['status']}**",
        f"- Analysis lock: `{lock['lock_id']}`",
        f"- Valid semantic blocks: {valid_block_total}/250; minimum per source: {min(valid_blocks_by_source.values())}/5",
        f"- Frozen known-group balanced accuracy: {balanced_accuracy:.1%}",
        f"- Oracle style-minus-sham source delta: {d_mean:+.4f} (bootstrap lower {d_lower:+.4f})",
        f"- Blind style-over-sham majority: {primary_preference:.1%} (bootstrap lower {primary_preference_lower:.1%})",
        f"- Meter/rater Spearman: {convergence:+.3f} (bootstrap lower {convergence_lower:+.3f})",
        "",
        "![Construct gate summary](construct_gate_summary.svg)",
        "",
        "## Frozen Known Groups",
        "",
        "| Statistic | Result | Gate |",
        "| --- | ---: | ---: |",
        f"| Balanced accuracy | {balanced_accuracy:.1%} | >=80% |",
        f"| Target sensitivity | {sensitivity:.1%} | >=80% |",
        f"| Comparison specificity | {specificity:.1%} | >=80% |",
        f"| Target Wilson lower | {sensitivity_lower:.1%} | >=65% |",
        f"| Comparison Wilson lower | {specificity_lower:.1%} | >=65% |",
        "",
        "## Manipulation And Convergence",
        "",
        "| Estimand | Result | Gate |",
        "| --- | ---: | ---: |",
        f"| Mean source D: style-positive minus sham | {d_mean:+.4f} | bootstrap lower >0 |",
        f"| Positive-source proportion | {positive_proportion:.1%} | >=70%; lower >=55% |",
        f"| Mean source G: D minus sham-neutral | {g_mean:+.4f} | bootstrap lower >0 |",
        f"| Blind majority style over sham | {primary_preference:.1%} | >=65%; lower >=55% |",
        f"| Blind majority style over neutral | {secondary_preference:.1%} | descriptive |",
        f"| Meter/rater Spearman | {convergence:+.3f} | >=.20; lower >0 |",
        f"| D after top-overlap decile removal | {overlap_removed_delta:+.4f} | >0 |",
        "",
        "## Protocol-Amendment Sensitivity",
        "",
        "The union of source-stage, adjudication-stage, and style-positive incident "
        "clusters is excluded below. The first two declarations cover the same four "
        "clusters; the completion incident adds two target clusters. These values are "
        "descriptive sensitivity diagnostics and do not replace or alter the "
        "preregistered gate denominators.",
        "",
        "| Estimand | Excluding amended sources |",
        "| --- | ---: |",
        f"| Remaining sources | {len(unamended_source_ids)} |",
        f"| Known-group balanced accuracy | {unamended_balanced_accuracy:.1%} |",
        f"| Mean source D | {amendment_exclusion_sensitivity['mean_source_style_minus_sham']:+.4f} |",
        f"| Positive-source proportion | {amendment_exclusion_sensitivity['positive_source_proportion']:.1%} |",
        f"| Blind style-over-sham majority | {amendment_exclusion_sensitivity['blind_style_over_sham_majority_proportion']:.1%} |",
        f"| Meter/rater Spearman | {unamended_convergence:+.3f} |",
        "",
        "## Gates",
        "",
    ]
    lines.extend(
        f"- {'PASS' if value else 'FAIL'}: `{name}`" for name, value in gate_results.items()
    )
    lines.extend(
        [
            "",
            "This experiment tests whether the frozen meter responds to a blind-rater-confirmed, "
            "semantically valid oracle manipulation. It does not establish production transfer "
            "quality, causal style dimensions, unseen-author generalization, or training efficacy.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(RESULTS, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
