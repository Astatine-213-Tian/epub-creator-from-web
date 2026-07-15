#!/usr/bin/env python3
from __future__ import annotations

import csv
import difflib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from experiments.iteration5.decision import prepare_transfer_lora_reference_benchmark_v2 as prepare
from experiments.iteration5.decision import run_transfer_lora_blind_ratings_v1 as ratings
from experiments.shared.hashing import canonical_json as canonical
from experiments.shared.hashing import sha256_text


RESULTS = prepare.ROOT / "results.json"
SUMMARY = prepare.ROOT / "analysis_summary.md"
METHOD_CSV = prepare.ROOT / "method_summary.csv"
SOURCE_CSV = prepare.ROOT / "source_summary.csv"
CALIBRATION_CSV = prepare.ROOT / "calibration_summary.csv"
FIGURE = prepare.ROOT / "method_success.svg"
BOOTSTRAPS = 10_000


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def mean(values: Iterable[float]) -> float | None:
    rows = list(values)
    return statistics.fmean(rows) if rows else None


def median(values: Iterable[float]) -> float | None:
    rows = list(values)
    return statistics.median(rows) if rows else None


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    probability = successes / total
    denominator = 1 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(probability * (1 - probability) / total + z * z / (4 * total * total))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        rank = (index + 1 + end) / 2
        for offset in range(index, end):
            ranks[indexed[offset][0]] = rank
        index = end
    return ranks


def correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left)
        * sum((b - right_mean) ** 2 for b in right)
    )
    return numerator / denominator if denominator else None


def spearman(left: list[float], right: list[float]) -> float | None:
    return correlation(average_ranks(left), average_ranks(right))


def cluster_bootstrap_ci(values: list[float], seed: str) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    rng = random.Random(int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16))
    draws = sorted(
        statistics.fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(BOOTSTRAPS)
    )
    return draws[int(0.025 * (BOOTSTRAPS - 1))], draws[int(0.975 * (BOOTSTRAPS - 1))]


def semantic_judgment_pass(row: dict[str, Any]) -> bool:
    return (
        int(row["semantic_fidelity"]) >= 4
        and int(row["naturalness"]) >= 3
        and row["high_severity_semantic_error"] is False
        and row["speaker_dialogue_topology_preserved"] is True
    )


def load_ratings(lock: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    records: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for private in read_jsonl(prepare.PRIVATE_MAPPING):
        packet_id = private["packet_id"]
        path = prepare.ROOT / "ratings" / private["identity"] / f"{packet_id}.json"
        packet_path = prepare.ROOT / "packets" / private["identity"] / f"{packet_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing rating: {path}")
        ratings.validate_artifact(path, lock, packet_path, private["task"], private["identity"])
        artifact = read_json(path)
        hashes[str(path)] = prepare.sha256_file(path)
        for judgment in artifact["result"]["judgments"]:
            method = private["alias_to_method"][judgment["candidate_id"]]
            records.append(
                {
                    "task": private["task"],
                    "identity": private["identity"],
                    "arm": private["arm"],
                    "block_id": private["block_id"],
                    "source_id": private["source_id"],
                    "book": private["book"],
                    "is_calibration": bool(private["is_calibration"]),
                    "method_id": method,
                    **{
                        key: value
                        for key, value in judgment.items()
                        if key not in {"candidate_id", "rationale"}
                    },
                }
            )
    return records, hashes


def calibration_summary(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    controls = [row for row in records if row["is_calibration"] and row["task"] == "style"]
    by_identity_block: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in controls:
        by_identity_block[(row["identity"], row["block_id"])][row["method_id"]] = float(
            row["style_adherence"]
        )
    rows: list[dict[str, Any]] = []
    for identity in prepare.STYLE_IDENTITIES:
        deltas: list[float] = []
        for (row_identity, _), methods in sorted(by_identity_block.items()):
            if row_identity != identity:
                continue
            if set(methods) != {"calibration_positive", "calibration_decoy"}:
                raise ValueError(f"incomplete calibration pair for {identity}")
            deltas.append(methods["calibration_positive"] - methods["calibration_decoy"])
        if len(deltas) != 4:
            raise ValueError(f"expected four calibration blocks for {identity}, got {len(deltas)}")
        positive_orderings = sum(delta > 0 for delta in deltas)
        delta_mean = statistics.fmean(deltas)
        passed = delta_mean > 0 and positive_orderings >= 3
        rows.append(
            {
                "identity": identity,
                "blocks": len(deltas),
                "positive_above_decoy": positive_orderings,
                "positive_rate": positive_orderings / len(deltas),
                "mean_style_delta": delta_mean,
                "passed": passed,
            }
        )
    return rows, all(row["passed"] for row in rows)


def han_text(rows: list[dict[str, str]]) -> str:
    return "".join(character for row in rows for character in row["zh"] if "\u4e00" <= character <= "\u9fff")


def longest_reference_copy(reference: list[dict[str, str]], candidate: list[dict[str, str]]) -> int:
    left = han_text(reference)
    right = han_text(candidate)
    return difflib.SequenceMatcher(None, left, right, autojunk=False).find_longest_match().size


def copy_diagnostics() -> dict[tuple[str, str], int]:
    result: dict[tuple[str, str], int] = {}
    for block in read_jsonl(prepare.PRIVATE_BLOCKS):
        for method, candidate in block["candidates"].items():
            result[(block["block_id"], method)] = longest_reference_copy(
                block["style_reference"], candidate
            )
    return result


def summarize_sources(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    main_records = [row for row in records if not row["is_calibration"]]
    eligibility_rows = read_jsonl(prepare.ELIGIBILITY)
    eligibility = {
        (row["arm"], row["block_id"], row["source_id"], row["method_id"]): row
        for row in eligibility_rows
    }
    by_key: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in main_records:
        by_key[(row["task"], row["arm"], row["block_id"], row["source_id"], row["method_id"])].append(row)
    source_blocks: dict[tuple[str, str], set[str]] = defaultdict(set)
    source_methods: dict[tuple[str, str], set[str]] = defaultdict(set)
    books: dict[tuple[str, str], str] = {}
    for row in eligibility_rows:
        source_blocks[(row["arm"], row["source_id"])].add(row["block_id"])
        source_methods[(row["arm"], row["source_id"])].add(row["method_id"])
    for row in read_jsonl(prepare.PRIVATE_MAPPING):
        if not row["is_calibration"]:
            books[(row["arm"], row["source_id"])] = row["book"]
    copies = copy_diagnostics()

    source_summary: list[dict[str, Any]] = []
    for (arm, source_id), blocks in sorted(source_blocks.items()):
        ordered_blocks = sorted(blocks)
        for method in sorted(source_methods[(arm, source_id)]):
            contract = [
                bool(
                    eligibility.get(
                        (arm, block, source_id, method), {"contract_valid": False}
                    )["contract_valid"]
                )
                for block in ordered_blocks
            ]
            semantic_block_passes: list[bool] = []
            severe_majority_blocks = 0
            style_scores: list[float] = []
            style_naturalness: list[float] = []
            style_deltas: list[float] = []
            dimension_deltas: dict[str, list[float]] = defaultdict(list)
            copied_spans: list[int] = []
            for block in ordered_blocks:
                semantic_rows = by_key.get(("semantic", arm, block, source_id, method), [])
                semantic_block_passes.append(
                    len(semantic_rows) == len(prepare.SEMANTIC_IDENTITIES)
                    and all(semantic_judgment_pass(row) for row in semantic_rows)
                )
                if sum(bool(row["high_severity_semantic_error"]) for row in semantic_rows) >= 2:
                    severe_majority_blocks += 1
                style_rows = by_key.get(("style", arm, block, source_id, method), [])
                neutral_rows = {
                    row["identity"]: row
                    for row in by_key.get(("style", arm, block, source_id, "neutral_only"), [])
                }
                for row in style_rows:
                    style_scores.append(float(row["style_adherence"]))
                    style_naturalness.append(float(row["naturalness"]))
                    neutral = neutral_rows.get(row["identity"])
                    if neutral is None:
                        continue
                    style_deltas.append(float(row["style_adherence"] - neutral["style_adherence"]))
                    for dimension in prepare.RUBRIC_DIMENSIONS:
                        value = row["dimension_scores"][dimension]
                        neutral_value = neutral["dimension_scores"][dimension]
                        if value is not None and neutral_value is not None:
                            dimension_deltas[dimension].append(float(value - neutral_value))
                if (block, method) in copies:
                    copied_spans.append(copies[(block, method)])
            semantic_required = math.ceil(0.8 * len(ordered_blocks))
            semantic_source_pass = (
                sum(semantic_block_passes) >= semantic_required and severe_majority_blocks == 0
            )
            style_delta = mean(style_deltas)
            positive_rate = (
                sum(delta > 0 for delta in style_deltas) / len(style_deltas)
                if style_deltas
                else None
            )
            style_improvement = bool(
                style_delta is not None
                and style_delta > 0
                and positive_rate is not None
                and positive_rate > 0.5
            )
            naturalness_median = median(style_naturalness)
            naturalness_pass = bool(naturalness_median is not None and naturalness_median >= 3)
            contract_all = all(contract)
            source_summary.append(
                {
                    "arm": arm,
                    "source_id": source_id,
                    "book": books[(arm, source_id)],
                    "method_id": method,
                    "blocks": len(ordered_blocks),
                    "contract_valid_blocks": sum(contract),
                    "contract_all": contract_all,
                    "semantic_pass_blocks": sum(semantic_block_passes),
                    "semantic_required_blocks": semantic_required,
                    "semantic_source_pass": semantic_source_pass,
                    "severe_majority_blocks": severe_majority_blocks,
                    "mean_structural_style_score_eligible": mean(style_scores),
                    "mean_structural_style_delta_vs_neutral_eligible": style_delta,
                    "style_positive_comparison_rate_eligible": positive_rate,
                    "style_improvement": style_improvement,
                    "median_style_panel_naturalness_eligible": naturalness_median,
                    "naturalness_pass": naturalness_pass,
                    "source_success": (
                        contract_all
                        and semantic_source_pass
                        and style_improvement
                        and naturalness_pass
                    ),
                    "max_exact_reference_copy_han": max(copied_spans) if copied_spans else None,
                    "dimension_deltas_vs_neutral": {
                        dimension: mean(dimension_deltas[dimension])
                        for dimension in prepare.RUBRIC_DIMENSIONS
                    },
                }
            )
    return source_summary


def summarize_methods(source_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in source_summary:
        grouped[(row["arm"], row["method_id"])].append(row)
    result: list[dict[str, Any]] = []
    for (arm, method), rows in sorted(grouped.items()):
        successes = sum(row["source_success"] for row in rows)
        lower, upper = wilson(successes, len(rows))
        source_deltas = [
            float(row["mean_structural_style_delta_vs_neutral_eligible"])
            for row in rows
            if row["mean_structural_style_delta_vs_neutral_eligible"] is not None
        ]
        bootstrap_lower, bootstrap_upper = cluster_bootstrap_ci(
            source_deltas, f"v2|{arm}|{method}"
        )
        result.append(
            {
                "arm": arm,
                "method_id": method,
                "sources": len(rows),
                "successes": successes,
                "success_rate": successes / len(rows),
                "success_wilson_95pct_lower": lower,
                "success_wilson_95pct_upper": upper,
                "contract_complete_sources": sum(row["contract_all"] for row in rows),
                "semantic_pass_sources": sum(row["semantic_source_pass"] for row in rows),
                "style_improved_sources": sum(row["style_improvement"] for row in rows),
                "naturalness_pass_sources": sum(row["naturalness_pass"] for row in rows),
                "mean_source_structural_delta_vs_neutral": mean(source_deltas),
                "source_cluster_bootstrap_95pct_lower": bootstrap_lower,
                "source_cluster_bootstrap_95pct_upper": bootstrap_upper,
                "max_exact_reference_copy_han": max(
                    (
                        int(row["max_exact_reference_copy_han"])
                        for row in rows
                        if row["max_exact_reference_copy_han"] is not None
                    ),
                    default=None,
                ),
                "passes_prompt_80pct_gate": arm == "prompt_development" and successes >= 7,
            }
        )
    return result


def reliability(records: list[dict[str, Any]]) -> dict[str, Any]:
    main = [row for row in records if not row["is_calibration"]]
    style_by_identity: dict[str, dict[tuple[str, str, str], float]] = defaultdict(dict)
    semantic_by_identity: dict[str, dict[tuple[str, str, str], bool]] = defaultdict(dict)
    for row in main:
        key = (row["arm"], row["block_id"], row["method_id"])
        if row["task"] == "style":
            style_by_identity[row["identity"]][key] = float(row["style_adherence"])
        else:
            semantic_by_identity[row["identity"]][key] = semantic_judgment_pass(row)
    correlations: dict[str, float | None] = {}
    identities = list(prepare.STYLE_IDENTITIES)
    for index, left in enumerate(identities):
        for right in identities[index + 1 :]:
            shared = sorted(set(style_by_identity[left]) & set(style_by_identity[right]))
            correlations[f"{left}__{right}"] = spearman(
                [style_by_identity[left][key] for key in shared],
                [style_by_identity[right][key] for key in shared],
            )
    left, right = prepare.SEMANTIC_IDENTITIES
    shared = sorted(set(semantic_by_identity[left]) & set(semantic_by_identity[right]))
    agreement = (
        sum(semantic_by_identity[left][key] == semantic_by_identity[right][key] for key in shared)
        / len(shared)
        if shared
        else None
    )
    return {
        "style_pairwise_spearman": correlations,
        "semantic_binary_pass_agreement": agreement,
        "semantic_shared_candidate_blocks": len(shared),
        "independence_limit": (
            "all automated raters used gpt-5.5; calls are repeated measurements, not "
            "independent human or cross-family validation"
        ),
    }


def semantic_sensitivity(source_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parent = read_json(prepare.v1.ROOT / "results.json")
    v1_rows = parent["source_summary"]
    current: dict[tuple[str, str], list[bool]] = defaultdict(list)
    previous: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for row in source_summary:
        current[(row["arm"], row["method_id"])].append(bool(row["semantic_source_pass"]))
    for row in v1_rows:
        key = (row["arm"], row["method_id"])
        if key in current:
            previous[key].append(bool(row["semantic_source_pass"]))
    return [
        semantic_sensitivity_row(
            arm=arm,
            method=method,
            v2_values=values,
            v1_values=previous[(arm, method)],
        )
        for (arm, method), values in sorted(current.items())
    ]


def semantic_sensitivity_row(
    *, arm: str, method: str, v2_values: list[bool], v1_values: list[bool]
) -> dict[str, Any]:
    if len(v1_values) != len(v2_values):
        raise ValueError(
            f"v1/v2 semantic sensitivity denominator mismatch: {arm} {method} "
            f"v1={len(v1_values)} v2={len(v2_values)}"
        )
    return {
        "arm": arm,
        "method_id": method,
        "v2_sources": len(v2_values),
        "v1_sources": len(v1_values),
        "denominators_match": True,
        "v2_semantic_pass_sources": sum(v2_values),
        "v1_semantic_pass_sources": sum(v1_values),
        "change_sources": sum(v2_values) - sum(v1_values),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    normalized = []
    for row in rows:
        normalized.append(
            {
                key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list))
                else value
                for key, value in row.items()
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(normalized[0]))
        writer.writeheader()
        writer.writerows(normalized)


def pct(value: float | None) -> str:
    return "NA" if value is None else f"{value * 100:.1f}%"


def number(value: float | None) -> str:
    return "NA" if value is None else f"{value:+.3f}"


def write_figure(rows: list[dict[str, Any]]) -> None:
    prompt = [row for row in rows if row["arm"] == "prompt_development"]
    width = 1050
    height = 100 + 48 * len(prompt)
    left = 390
    chart_width = 540
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#20242a;letter-spacing:0}.label{font-size:14px}.value{font-size:13px;font-weight:700}.axis{stroke:#8a9199;stroke-width:1}</style>',
        '<text x="24" y="30" font-size="20" font-weight="700">Reference-anchored source success on eight development sources</text>',
        f'<line class="axis" x1="{left}" y1="50" x2="{left}" y2="{height-28}"/>',
        f'<line x1="{left + chart_width * .8:.1f}" y1="50" x2="{left + chart_width * .8:.1f}" y2="{height-28}" stroke="#b3272d" stroke-dasharray="5 5"/>',
        f'<text x="{left + chart_width * .8 + 5:.1f}" y="64" font-size="12" fill="#b3272d">80% target</text>',
    ]
    for index, row in enumerate(prompt):
        y = 82 + index * 48
        rate = float(row["success_rate"])
        color = "#2c7a4b" if row["passes_prompt_80pct_gate"] else "#446d8c"
        lines.extend(
            [
                f'<text class="label" x="24" y="{y+16}">{row["method_id"]}</text>',
                f'<rect x="{left}" y="{y}" width="{chart_width}" height="22" fill="#edf0f2"/>',
                f'<rect x="{left}" y="{y}" width="{chart_width * rate:.1f}" height="22" fill="{color}"/>',
                f'<text class="value" x="{left + chart_width + 10}" y="{y+16}">{row["successes"]}/{row["sources"]} ({rate*100:.1f}%)</text>',
            ]
        )
    lines.append("</svg>")
    FIGURE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_summary(results: dict[str, Any]) -> None:
    prompt = [row for row in results["method_summary"] if row["arm"] == "prompt_development"]
    lora = [row for row in results["method_summary"] if row["arm"] == "lora_internal_test"]
    lines = [
        "# Reference-Anchored Prompt and LoRA Benchmark v2",
        "",
        f"- Status: **{results['status']}**",
        f"- Analysis lock: `{results['lock_id']}`",
        "- Estimand: passage-specific structural reconstruction, not author identification",
        "- Style panel: clean Chinese reference, no English",
        "- Semantic panel: English source, no Chinese reference",
        "- CR-FYSM-v4: excluded from primary decisions",
        "",
        "![Reference-anchored source success](method_success.svg)",
        "",
        "## Calibration",
        "",
        "| Rater | Positive above decoy | Mean delta | Gate |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in results["calibration_summary"]:
        lines.append(
            f"| `{row['identity']}` | {row['positive_above_decoy']}/{row['blocks']} "
            f"| {number(row['mean_style_delta'])} | {'PASS' if row['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "Method outcomes are interpretable only when all three calibration raters pass. The positive control preserves structure through deterministic entity masking; the decoy has identical characters and punctuation but disrupted paragraph and sentence-unit order.",
            "",
            "## Prompt Methods",
            "",
            "| Method | Full success | Contract | Semantic | Structure over neutral | Naturalness | Mean delta [cluster 95% CI] | 80% |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in prompt:
        lines.append(
            f"| `{row['method_id']}` | {row['successes']}/{row['sources']} ({pct(row['success_rate'])}) "
            f"| {row['contract_complete_sources']}/{row['sources']} "
            f"| {row['semantic_pass_sources']}/{row['sources']} "
            f"| {row['style_improved_sources']}/{row['sources']} "
            f"| {row['naturalness_pass_sources']}/{row['sources']} "
            f"| {number(row['mean_source_structural_delta_vs_neutral'])} "
            f"[{number(row['source_cluster_bootstrap_95pct_lower'])}, {number(row['source_cluster_bootstrap_95pct_upper'])}] "
            f"| {'PASS' if row['passes_prompt_80pct_gate'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## LoRA Internal Test",
            "",
            "| Method | Full success | Contract-complete books | Semantic | Structure over neutral | Mean delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in lora:
        lines.append(
            f"| `{row['method_id']}` | {row['successes']}/{row['sources']} "
            f"| {row['contract_complete_sources']}/{row['sources']} "
            f"| {row['semantic_pass_sources']}/{row['sources']} "
            f"| {row['style_improved_sources']}/{row['sources']} "
            f"| {number(row['mean_source_structural_delta_vs_neutral'])} |"
        )
    lines.extend(
        [
            "",
            "## Reliability And Limits",
            "",
            f"Semantic binary-pass agreement: {pct(results['reliability']['semantic_binary_pass_agreement'])}.",
            "",
            "All automated raters use one model family. This benchmark is developmental and requires independent Chinese human or cross-family confirmation before an efficacy claim.",
            "",
            "## Decision",
            "",
            results["decision"],
            "",
            "Exact source outcomes, calibration values, v1 semantic sensitivity, copy diagnostics, and hash-bound ratings are recorded beside this summary.",
        ]
    )
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if RESULTS.exists() or SUMMARY.exists():
        raise FileExistsError("v2 analysis outputs already exist")
    ratings.prepare = prepare
    lock = ratings.validate_lock()
    records, rating_hashes = load_ratings(lock)
    calibrations, calibration_passed = calibration_summary(records)
    source_summary = summarize_sources(records)
    method_summary = summarize_methods(source_summary)
    reliability_result = reliability(records)
    sensitivity = semantic_sensitivity(source_summary)
    prompt_promotions = [
        row["method_id"]
        for row in method_summary
        if row["arm"] == "prompt_development"
        and row["passes_prompt_80pct_gate"]
        and row["method_id"] != "neutral_only"
    ]
    lora_row = next(
        row
        for row in method_summary
        if row["arm"] == "lora_internal_test" and row["method_id"] == "lora_smoke_4b"
    )
    base_row = next(
        row
        for row in method_summary
        if row["arm"] == "lora_internal_test" and row["method_id"] == "frozen_base_4b"
    )
    if not calibration_passed:
        status = "calibration_fail_no_method_interpretation"
        decision = (
            "The style panel failed its locked structural calibration. Treat every prompt and "
            "LoRA style outcome as uninterpretable; do not select a method or train a production LoRA."
        )
    elif prompt_promotions:
        status = "developmental_prompt_candidate_no_production_lora"
        decision = (
            "At least one frozen prompt method reaches the exploratory 80% source gate, but the "
            "single-family automated panel cannot authorize selection. Confirm the candidate with "
            "an independent Chinese panel. Do not train or deploy a production LoRA from the smoke corpus."
        )
    elif lora_row["contract_complete_sources"] == 3 and lora_row["successes"] > base_row["successes"]:
        status = "conditional_go_clean_corpus_and_multiseed_lora_pilot"
        decision = (
            "No prompt method reaches 80%, while the smoke adapter clears all book contracts and "
            "outperforms the base directionally. Build a clean 1,500-2,000-pair corpus and run a "
            "separately preregistered multi-seed LoRA pilot; production use remains prohibited."
        )
    else:
        status = "no_go_smoke_lora_clean_pair_corpus_only"
        decision = (
            "No frozen prompt method reaches the 80% exploratory gate, and the smoke LoRA does not "
            "clear all three book contracts. The smoke demonstrates training and output-contract "
            "plumbing, not style efficacy. Do not train or deploy a production LoRA. Proceed only "
            "with clean paired-corpus construction; authorize a multi-seed pilot only under a new "
            "preregistration after the corpus and contract baseline pass their own readiness gates."
        )
    payload: dict[str, Any] = {
        "schema_version": 2,
        "analysis_id": "transfer-prompt-lora-reference-anchored-benchmark-v2-analysis",
        "lock_id": lock["lock_id"],
        "status": status,
        "decision": decision,
        "calibration_gate_passed": calibration_passed,
        "calibration_summary": calibrations,
        "prompt_promotions": prompt_promotions,
        "method_summary": method_summary,
        "source_summary": source_summary,
        "semantic_sensitivity_vs_v1": sensitivity,
        "reliability": reliability_result,
        "rating_hashes": rating_hashes,
        "claim_limit": lock["decision_scope"],
        "forbidden_claims": lock["forbidden_claims"],
    }
    payload["analysis_sha256"] = sha256_text(canonical(payload))
    write_json(RESULTS, payload)
    write_csv(METHOD_CSV, method_summary)
    write_csv(SOURCE_CSV, source_summary)
    write_csv(CALIBRATION_CSV, calibrations)
    write_figure(method_summary)
    render_summary(payload)
    print(f"wrote {RESULTS}")
    print(f"status={status}")


if __name__ == "__main__":
    main()
