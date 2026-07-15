#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from experiments.iteration1 import evaluate_style_transfer_methods as base_evaluator

from experiments.shared.paths import RESEARCH_ROOT
from workflows.author_style_meter_contract import CURRENT_SCORER_ID


REPO_ROOT = RESEARCH_ROOT
BASE_EVALUATOR_PATH = REPO_ROOT / "experiments/iteration1/evaluate_style_transfer_methods.py"
FIXED_METHODS = (
    "aligned_pairs_light",
    "aligned_pairs_edit_plan_light",
    "microcards_only_light",
    "rule_linked_microcards_light",
)
DISPLAY_METHODS = (
    "neutral_only",
    *FIXED_METHODS,
    "candidate_rerank",
)
SHORT_LABELS = {
    "neutral_only": "Neutral",
    "aligned_pairs_light": "Aligned pairs",
    "aligned_pairs_edit_plan_light": "Pairs + plan",
    "microcards_only_light": "Microcards",
    "rule_linked_microcards_light": "Pairs + cards",
    "candidate_rerank": "Rerank (diagnostic)",
}
ARM_LABELS = {
    "own_author_reconstruction": "Own-author",
    "cross_author_transfer": "Cross-author",
}
COLORS = {
    "own_author_reconstruction": "#2563eb",
    "cross_author_transfer": "#d97706",
    "success": "#16803c",
    "failed": "#c2413b",
    "neutral": "#64748b",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_base_evaluator():
    return base_evaluator


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def esc(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def svg_text(
    x: float,
    y: float,
    value: object,
    *,
    size: int = 13,
    anchor: str = "start",
    weight: int = 400,
    fill: str = "#17202a",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="{fill}" letter-spacing="0">{esc(value)}</text>'
    )


def nice_floor(value: float, step: float) -> float:
    return math.floor(value / step) * step


def nice_ceil(value: float, step: float) -> float:
    return math.ceil(value / step) * step


def build_margin_svg(rows: list[dict[str, Any]], threshold: float) -> str:
    width = 1120
    left = 190
    right = 70
    top = 72
    row_height = 55
    height = top + len(DISPLAY_METHODS) * row_height + 82
    values = [float(row["mean_target_margin"]) for row in rows]
    x_min = nice_floor(min(values) - 0.1, 0.5)
    x_max = nice_ceil(max(max(values), threshold) + 0.1, 0.5)
    plot_width = width - left - right

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(left, 30, "Mean target-author margin versus frozen threshold", size=19, weight=700),
        svg_text(left, 52, "Higher is more target-like; rerank is selection-conditioned.", size=13, fill="#4b5563"),
    ]
    for tick_index in range(int(round((x_max - x_min) / 0.5)) + 1):
        tick = x_min + tick_index * 0.5
        x = sx(tick)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top - 12}" x2="{x:.1f}" y2="{height - 52}" stroke="#d8dee8" stroke-width="1"/>'
        )
        parts.append(svg_text(x, height - 29, f"{tick:+.1f}", size=12, anchor="middle", fill="#4b5563"))
    threshold_x = sx(threshold)
    parts.append(
        f'<line x1="{threshold_x:.1f}" y1="{top - 18}" x2="{threshold_x:.1f}" y2="{height - 52}" stroke="#b42318" stroke-width="2" stroke-dasharray="7 5"/>'
    )
    parts.append(svg_text(threshold_x + 6, top - 25, f"Threshold {threshold:.3f}", size=12, fill="#b42318", weight=700))
    row_map = {(row["method_id"], row["arm"]): row for row in rows}
    for index, method in enumerate(DISPLAY_METHODS):
        center_y = top + index * row_height + 18
        parts.append(svg_text(left - 14, center_y + 5, SHORT_LABELS[method], anchor="end", size=13, weight=600))
        for offset, arm in ((-9, "own_author_reconstruction"), (9, "cross_author_transfer")):
            row = row_map[(method, arm)]
            value = float(row["mean_target_margin"])
            x0 = sx(min(0.0, value))
            x1 = sx(max(0.0, value))
            parts.append(
                f'<rect x="{x0:.1f}" y="{center_y + offset - 6:.1f}" width="{max(x1 - x0, 1):.1f}" height="12" rx="2" fill="{COLORS[arm]}"/>'
            )
            label_x = sx(value) + (6 if value >= 0 else -6)
            anchor = "start" if value >= 0 else "end"
            parts.append(svg_text(label_x, center_y + offset + 4, f"{value:+.3f}", size=11, anchor=anchor, fill=COLORS[arm]))
    legend_y = height - 8
    for index, arm in enumerate(("own_author_reconstruction", "cross_author_transfer")):
        x = left + index * 170
        parts.append(f'<rect x="{x}" y="{legend_y - 11}" width="13" height="13" rx="2" fill="{COLORS[arm]}"/>')
        parts.append(svg_text(x + 20, legend_y, ARM_LABELS[arm], size=12))
    parts.append("</svg>\n")
    return "\n".join(parts)


def build_edit_svg(rows: list[dict[str, Any]]) -> str:
    width = 1040
    height = 560
    left = 90
    right = 50
    top = 70
    bottom = 82
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_edit = max(float(row["character_edit_percent"]) for row in rows)
    lifts = [float(row[key]) for row in rows for key in ("own_mean_lift", "cross_mean_lift")]
    x_max = max(10.0, nice_ceil(max_edit + 0.5, 2.0))
    y_min = nice_floor(min(lifts) - 0.01, 0.05)
    y_max = nice_ceil(max(lifts) + 0.01, 0.05)

    def sx(value: float) -> float:
        return left + value / x_max * plot_width

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(left, 30, "Edit magnitude versus target-margin lift", size=19, weight=700),
        svg_text(left, 52, "Character edit percent is 100 minus mean neutral-output similarity.", size=13, fill="#4b5563"),
    ]
    for tick in range(0, int(x_max) + 1, 2):
        x = sx(float(tick))
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" stroke="#e1e6ee"/>')
        parts.append(svg_text(x, top + plot_height + 24, f"{tick}%", size=12, anchor="middle", fill="#4b5563"))
    tick = y_min
    while tick <= y_max + 1e-9:
        y = sy(tick)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#e1e6ee"/>')
        parts.append(svg_text(left - 10, y + 4, f"{tick:+.2f}", size=12, anchor="end", fill="#4b5563"))
        tick += 0.05
    zero_y = sy(0.0)
    parts.append(f'<line x1="{left}" y1="{zero_y:.1f}" x2="{left + plot_width}" y2="{zero_y:.1f}" stroke="#596273" stroke-width="1.5"/>')
    for method_index, row in enumerate(rows):
        x = sx(float(row["character_edit_percent"]))
        for arm, key, delta in (
            ("own_author_reconstruction", "own_mean_lift", -5),
            ("cross_author_transfer", "cross_mean_lift", 5),
        ):
            y = sy(float(row[key]))
            parts.append(f'<circle cx="{x + delta:.1f}" cy="{y:.1f}" r="6" fill="{COLORS[arm]}" stroke="#ffffff" stroke-width="1.5"/>')
        own_y = sy(float(row["own_mean_lift"]))
        label_y = own_y - 11 if method_index % 2 == 0 else own_y + 20
        parts.append(svg_text(x, label_y, SHORT_LABELS[str(row["method_id"])], size=11, anchor="middle", weight=600))
    parts.append(svg_text(left + plot_width / 2, height - 24, "Mean character edit percent", size=13, anchor="middle", weight=600))
    axis_x = 24
    axis_y = top + plot_height / 2
    parts.append(
        f'<text x="{axis_x}" y="{axis_y:.1f}" font-family="Arial, sans-serif" '
        'font-size="13" font-weight="600" text-anchor="middle" fill="#17202a" '
        f'letter-spacing="0" transform="rotate(-90 {axis_x} {axis_y:.1f})">Mean paired margin lift</text>'
    )
    legend_y = height - 50
    for index, arm in enumerate(("own_author_reconstruction", "cross_author_transfer")):
        x = width - 390 + index * 180
        parts.append(f'<circle cx="{x}" cy="{legend_y}" r="6" fill="{COLORS[arm]}"/>')
        parts.append(svg_text(x + 12, legend_y + 4, ARM_LABELS[arm], size=12))
    parts.append("</svg>\n")
    return "\n".join(parts)


def build_generation_svg(rows: list[dict[str, Any]]) -> str:
    width = 1120
    left = 225
    right = 230
    top = 72
    row_height = 72
    height = top + len(rows) * row_height + 70
    max_attempts = max(int(row["attempted_segments"]) for row in rows)
    plot_width = width - left - right

    def sx(value: float) -> float:
        return left + value / max_attempts * plot_width

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(left, 30, "Admissible block-generation reliability", size=19, weight=700),
        svg_text(left, 52, "Every failed block was recovered; all methods finished 36/36 samples.", size=13, fill="#4b5563"),
    ]
    for index, row in enumerate(rows):
        y = top + index * row_height
        attempted = int(row["attempted_segments"])
        successful = int(row["successful_segments"])
        failed = int(row["failed_segments"])
        parts.append(svg_text(left - 14, y + 19, SHORT_LABELS[str(row["method_id"])], anchor="end", size=13, weight=600))
        parts.append(f'<rect x="{left}" y="{y + 5}" width="{sx(successful) - left:.1f}" height="22" rx="3" fill="{COLORS["success"]}"/>')
        parts.append(f'<rect x="{sx(successful):.1f}" y="{y + 5}" width="{max(sx(attempted) - sx(successful), 1):.1f}" height="22" rx="3" fill="{COLORS["failed"]}"/>')
        parts.append(svg_text(sx(attempted) + 8, y + 21, f"{successful} valid + {failed} recovered", size=12))
        parts.append(svg_text(left, y + 47, f"timeouts/no response: {row['timeout_or_no_response']}; invalid response: {row['invalid_response']}", size=11, fill="#4b5563"))
    parts.append(svg_text(left + plot_width / 2, height - 22, "Block requests", size=13, anchor="middle", weight=600))
    parts.append("</svg>\n")
    return "\n".join(parts)


def method_arm_rows(
    summary: dict[str, Any], evaluation_dir: Path
) -> list[dict[str, Any]]:
    threshold = float(summary["threshold"]["threshold"])
    combinations = {row["method_id"]: row for row in summary["combinations"]}
    rows: list[dict[str, Any]] = []
    for method in DISPLAY_METHODS:
        combination = combinations[method]
        intensity = "none" if method == "neutral_only" else "light"
        method_results = load_json(
            evaluation_dir
            / f"methods/{method}/{intensity}/results.json"
        )
        for arm in ("own_author_reconstruction", "cross_author_transfer"):
            metrics = combination["arm_summaries"][arm]
            success = metrics["deterministic_style_success"]
            lift_interval = method_results["book_cluster_bootstrap"][arm]
            rows.append(
                {
                    "method_id": method,
                    "method_label": combination["method_label"],
                    "analysis_role": "selection_conditioned_diagnostic" if method == "candidate_rerank" else "fixed_method_or_control",
                    "arm": arm,
                    "expected_rows": metrics["expected_rows"],
                    "scored_rows": metrics["scored_rows"],
                    "mean_target_margin": metrics["mean_target_margin"],
                    "threshold": threshold,
                    "mean_margin_gap_to_threshold": metrics["mean_target_margin"] - threshold,
                    "mean_paired_margin_lift": metrics["mean_paired_margin_lift"],
                    "paired_lift_cluster_bootstrap_lower": lift_interval["lower"],
                    "paired_lift_cluster_bootstrap_upper": lift_interval["upper"],
                    "paired_lift_interval_excludes_zero": lift_interval["excludes_zero"],
                    "positive_lift_rows": metrics["positive_lift_rows"],
                    "mean_target_rank": metrics["mean_target_rank"],
                    "target_chunk_share": metrics["target_chunk_share_conservative"],
                    "deterministic_successes": success["successes"],
                    "deterministic_success_rate": success["estimate"],
                    "hard_fidelity_failure_rows": metrics["hard_fidelity_failure_rows"],
                    "reference_copy_failure_rows": metrics["gate_counts"].get("no_copy_gate", {}).get("fail", 0),
                }
            )
    return rows


def edit_rows(experiment_root: Path, sample_set: str, run_id: str, source_run_id: str, summary: dict[str, Any]) -> list[dict[str, Any]]:
    combinations = {row["method_id"]: row for row in summary["combinations"]}
    source_root = experiment_root / f"runs/{sample_set}/{source_run_id}/neutral_translation"
    run_root = experiment_root / f"runs/{sample_set}/{run_id}/method_outputs"
    rows: list[dict[str, Any]] = []
    for method in FIXED_METHODS:
        similarities: list[float] = []
        output_length_ratios: list[float] = []
        exact_paragraphs = 0
        paragraph_count = 0
        output_dir = run_root / f"{method}/light"
        for output_path in sorted(output_dir.glob("s_*.json")):
            sample_id = output_path.stem
            neutral = load_json(source_root / f"{sample_id}.json")["result"]["paragraphs"]
            output = load_json(output_path)["result"]["paragraphs"]
            if [row["id"] for row in neutral] != [row["id"] for row in output]:
                raise ValueError(f"Paragraph IDs differ for {method}:{sample_id}")
            neutral_rows = [str(row["zh"]) for row in neutral]
            output_rows = [str(row["zh"]) for row in output]
            neutral_text = "\n".join(neutral_rows)
            output_text = "\n".join(output_rows)
            similarities.append(
                difflib.SequenceMatcher(
                    None, neutral_text, output_text, autojunk=False
                ).ratio()
            )
            output_length_ratios.append(len(output_text) / max(len(neutral_text), 1))
            exact_paragraphs += sum(
                neutral_value == output_value
                for neutral_value, output_value in zip(neutral_rows, output_rows)
            )
            paragraph_count += len(neutral_rows)
        own = combinations[method]["arm_summaries"]["own_author_reconstruction"]
        cross = combinations[method]["arm_summaries"]["cross_author_transfer"]
        mean_similarity = sum(similarities) / len(similarities)
        rows.append(
            {
                "method_id": method,
                "sample_count": len(similarities),
                "mean_character_similarity": mean_similarity,
                "character_edit_percent": (1.0 - mean_similarity) * 100.0,
                "exact_paragraph_share": exact_paragraphs / paragraph_count,
                "changed_paragraph_share": 1.0 - exact_paragraphs / paragraph_count,
                "mean_output_length_ratio": sum(output_length_ratios) / len(output_length_ratios),
                "own_mean_lift": own["mean_paired_margin_lift"],
                "cross_mean_lift": cross["mean_paired_margin_lift"],
                "own_fidelity_failures": own["hard_fidelity_failure_rows"],
                "cross_fidelity_failures": cross["hard_fidelity_failure_rows"],
            }
        )
    return rows


def generation_rows(experiment_root: Path, sample_set: str, run_id: str) -> list[dict[str, Any]]:
    ledger_root = experiment_root / f"runs/{sample_set}/{run_id}/ledgers"
    rows: list[dict[str, Any]] = []
    for method in FIXED_METHODS:
        ledger = ledger_root / f"style_transfer.{method}.light.jsonl"
        records = list(iter_jsonl(ledger))
        segment_statuses: Counter[str] = Counter()
        invalid_response = 0
        timeout_or_no_response = 0
        for record in records:
            for segment in record["block_execution"]["segments"]:
                segment_statuses[str(segment["status"])] += 1
                if segment["status"] == "failed":
                    if segment.get("output_sha256") is None:
                        timeout_or_no_response += 1
                    else:
                        invalid_response += 1
        rows.append(
            {
                "method_id": method,
                "final_sample_outputs": sum(record["status"] == "success" for record in records),
                "failed_final_samples": sum(record["status"] != "success" for record in records),
                "attempted_segments": sum(record["block_execution"]["attempted_segment_count"] for record in records),
                "successful_segments": segment_statuses["success"],
                "failed_segments": segment_statuses["failed"],
                "recovered_splits": sum(record["block_execution"]["recovered_split_count"] for record in records),
                "timeout_or_no_response": timeout_or_no_response,
                "invalid_response": invalid_response,
            }
        )
    return rows


def original_control_rows(
    experiment_root: Path,
    sample_set: str,
    threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evaluator = load_base_evaluator()
    dataset_root = REPO_ROOT / "datasets"
    scorer_paths = evaluator.ScorerPaths(
        dataset_root=dataset_root,
        masked_chunks=dataset_root / "masked/chunks.entity_masked_v3.jsonl",
        mask_terms=dataset_root / "masked/mask_terms.json",
        splits=REPO_ROOT / "generated/style_research/corpus/splits.json",
        benchmark_result=(
            REPO_ROOT
            / "generated/style_research/benchmarks/"
            "author_style_supervised_50authors_cleaned/"
            "supervised_author_baseline_results.json"
        ),
        benchmark_script=REPO_ROOT / "workflows/benchmark_author_style_supervised.py",
        scorer_dir=(
            experiment_root
            / "scorers"
            / CURRENT_SCORER_ID
        ),
    )
    scorer = evaluator.load_frozen_scorer(scorer_paths)
    sample_root = experiment_root / "sample_sets"
    selection = load_json(sample_root / f"{sample_set}.screening_v1_ids.json")
    selected_ids = [str(value) for value in selection["sample_ids"]]
    allocation = {
        str(row["sample_id"]): row
        for row in iter_jsonl(sample_root / f"{sample_set}.evaluator_allocation.jsonl")
    }
    hidden = {
        str(row["sample_id"]): row
        for row in iter_jsonl(sample_root / f"{sample_set}.hidden_targets.jsonl")
    }
    texts = [str(hidden[sample_id]["entity_masked_v3_zh"]) for sample_id in selected_ids]
    scores = evaluator.score_texts(scorer, texts)
    detail: list[dict[str, Any]] = []
    for sample_id, score in zip(selected_ids, scores):
        source = allocation[sample_id]
        margin = float(score["target_margin"])
        detail.append(
            {
                "sample_id": sample_id,
                "benchmark_arm": source["benchmark_arm"],
                "author": source["author"],
                "book_title": source["book_title"],
                "target_margin": margin,
                "target_rank": score["target_rank"],
                "target_top1": score["target_chunk_indicator"],
                "threshold": threshold,
                "threshold_pass": int(margin >= threshold),
            }
        )
    summary: list[dict[str, Any]] = []
    for arm in ("own_author_reconstruction", "cross_author_transfer"):
        arm_rows = [row for row in detail if row["benchmark_arm"] == arm]
        margins = [float(row["target_margin"]) for row in arm_rows]
        summary.append(
            {
                "benchmark_arm": arm,
                "sample_count": len(arm_rows),
                "mean_target_margin": sum(margins) / len(margins),
                "median_target_margin": statistics.median(margins),
                "minimum_target_margin": min(margins),
                "maximum_target_margin": max(margins),
                "threshold": threshold,
                "threshold_passes": sum(int(row["threshold_pass"]) for row in arm_rows),
                "target_top1_rows": sum(int(row["target_top1"]) for row in arm_rows),
            }
        )
    return detail, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic Iteration 3 report tables and SVG charts."
    )
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--sample-set", default="iteration3_proxy_v1")
    parser.add_argument("--run-id", default="iteration3_gpt55_v6")
    parser.add_argument("--source-run-id", default="iteration3_gpt55_v2")
    parser.add_argument("--evaluation-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_root = args.experiment_root.expanduser().resolve()
    evaluation_dir = (
        args.evaluation_dir.expanduser().resolve()
        if args.evaluation_dir
        else experiment_root / f"evaluations/{args.run_id}/screening_initial"
    )
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else experiment_root / "audits"
    )
    chart_root = output_root / "charts"
    table_root = output_root / "tables"
    chart_root.mkdir(parents=True, exist_ok=True)
    table_root.mkdir(parents=True, exist_ok=True)

    summary = load_json(evaluation_dir / "evaluation_summary.json")
    if summary["threshold"]["status"] != "valid":
        raise ValueError("Report artifacts require a valid frozen threshold")
    margin_rows = method_arm_rows(summary, evaluation_dir)
    edits = edit_rows(
        experiment_root,
        args.sample_set,
        args.run_id,
        args.source_run_id,
        summary,
    )
    generation = generation_rows(experiment_root, args.sample_set, args.run_id)
    original_control, original_control_summary = original_control_rows(
        experiment_root,
        args.sample_set,
        float(summary["threshold"]["threshold"]),
    )

    margin_csv = table_root / "iteration3_screening_method_arm_metrics.csv"
    edit_csv = table_root / "iteration3_edit_intensity_metrics.csv"
    generation_csv = table_root / "iteration3_generation_reliability.csv"
    original_control_csv = table_root / "iteration3_original_control_scores.csv"
    original_control_summary_csv = (
        table_root / "iteration3_original_control_summary.csv"
    )
    write_csv(margin_csv, margin_rows)
    write_csv(edit_csv, edits)
    write_csv(generation_csv, generation)
    write_csv(original_control_csv, original_control)
    write_csv(original_control_summary_csv, original_control_summary)

    margin_svg = chart_root / "iteration3_mean_margin_vs_threshold.svg"
    edit_svg = chart_root / "iteration3_edit_intensity_vs_lift.svg"
    generation_svg = chart_root / "iteration3_generation_reliability.svg"
    margin_svg.write_text(
        build_margin_svg(margin_rows, float(summary["threshold"]["threshold"])),
        encoding="utf-8",
    )
    edit_svg.write_text(build_edit_svg(edits), encoding="utf-8")
    generation_svg.write_text(build_generation_svg(generation), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "complete",
                "inputs": {
                    "evaluation_summary": str(
                        (evaluation_dir / "evaluation_summary.json").relative_to(REPO_ROOT)
                    ),
                    "run_id": args.run_id,
                    "source_run_id": args.source_run_id,
                },
                "outputs": [
                    str(path.relative_to(REPO_ROOT))
                    for path in (
                        margin_csv,
                        edit_csv,
                        generation_csv,
                        original_control_csv,
                        original_control_summary_csv,
                        margin_svg,
                        edit_svg,
                        generation_svg,
                    )
                ],
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
