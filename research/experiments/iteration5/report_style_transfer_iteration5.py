#!/usr/bin/env python3
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


ITERATIONS = Path("generated/style_research/style_transfer_experiments/iterations")
BENCHMARK = (
    ITERATIONS
    / "paired_reconstruction_decision_v1/blind_benchmark_v2_reference_anchored"
)
LORA = ITERATIONS / "lora_paired_reconstruction_v1/runs/qwen3_4b_mlx_smoke_v1"
OUTPUT = ITERATIONS / "paired_reconstruction_decision_v1/iteration5_report_assets"
RESULTS = BENCHMARK / "results.json"
LORA_RESULTS = LORA / "results.json"

METHOD_LABELS = {
    "content_plan_combined_full_regeneration": "Content plan + combined",
    "aligned_pairs_style_definition_full_regeneration": "Aligned + definition",
    "style_definition_examples_full_regeneration": "Definition + examples",
    "aligned_pairs_full_regeneration": "Aligned pairs",
    "independent_candidate_selector": "Independent selector",
    "generic_full_regeneration": "Generic regeneration",
    "neutral_only": "Neutral control",
    "frozen_base_4b": "Frozen Qwen3-4B",
    "lora_smoke_4b": "Qwen3-4B LoRA smoke",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def svg_start(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#20242a;letter-spacing:0}.title{font-size:20px;font-weight:700}.label{font-size:13px}.small{font-size:11px}.value{font-size:12px;font-weight:700}.axis{stroke:#8a9199;stroke-width:1}</style>',
        f'<text class="title" x="24" y="30">{html.escape(title)}</text>',
    ]


def write_prompt_components(results: dict[str, Any]) -> None:
    rows = sorted(
        (row for row in results["method_summary"] if row["arm"] == "prompt_development"),
        key=lambda row: (row["successes"], row["style_improved_sources"]),
        reverse=True,
    )
    metrics = (
        ("Contract", "contract_complete_sources", "#4a6878"),
        ("Semantic", "semantic_pass_sources", "#cf8a2e"),
        ("Structure", "style_improved_sources", "#6c5c9b"),
        ("Joint success", "successes", "#2d7b57"),
    )
    width, left, chart_width = 1180, 360, 680
    height = 105 + len(rows) * 70
    lines = svg_start(width, height, "Prompt method component success by source/book")
    for index, (label, _, color) in enumerate(metrics):
        x = left + index * 150
        lines.extend(
            [
                f'<rect x="{x}" y="46" width="12" height="12" fill="{color}"/>',
                f'<text class="small" x="{x + 18}" y="57">{label}</text>',
            ]
        )
    lines.append(
        f'<line x1="{left + chart_width * .8:.1f}" y1="68" x2="{left + chart_width * .8:.1f}" y2="{height - 24}" stroke="#b3272d" stroke-dasharray="5 5"/>'
    )
    for row_index, row in enumerate(rows):
        y = 82 + row_index * 70
        lines.append(
            f'<text class="label" x="24" y="{y + 22}">{html.escape(METHOD_LABELS[row["method_id"]])}</text>'
        )
        for metric_index, (_, key, color) in enumerate(metrics):
            bar_y = y + metric_index * 12
            rate = row[key] / row["sources"]
            lines.extend(
                [
                    f'<rect x="{left}" y="{bar_y}" width="{chart_width}" height="8" fill="#edf0f2"/>',
                    f'<rect x="{left}" y="{bar_y}" width="{chart_width * rate:.1f}" height="8" fill="{color}"/>',
                    f'<text class="small" x="{left + chart_width + 10}" y="{bar_y + 8}">{row[key]}/{row["sources"]}</text>',
                ]
            )
    lines.append("</svg>")
    (OUTPUT / "prompt_component_success.svg").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_source_matrix(results: dict[str, Any]) -> None:
    rows = [row for row in results["source_summary"] if row["arm"] == "prompt_development"]
    books = [
        "天宝伏妖录",
        "山有木兮",
        "清平梦华录",
        "相见欢",
        "万物风华录",
        "骑士之歌",
        "图灵密码",
        "乱世为王",
    ]
    method_rows = sorted(
        (row for row in results["method_summary"] if row["arm"] == "prompt_development"),
        key=lambda row: row["successes"],
        reverse=True,
    )
    lookup = {(row["method_id"], row["book"]): row for row in rows}
    width, height, left, top, cell_w, cell_h = 1250, 570, 360, 115, 100, 48
    lines = svg_start(width, height, "Joint source-level success matrix")
    for index, book in enumerate(books):
        x = left + index * cell_w + cell_w / 2
        lines.append(
            f'<text class="small" x="{x:.1f}" y="78" text-anchor="end" transform="rotate(-35 {x:.1f} 78)">{html.escape(book)}</text>'
        )
    for row_index, method in enumerate(method_rows):
        y = top + row_index * cell_h
        lines.append(
            f'<text class="label" x="24" y="{y + 29}">{html.escape(METHOD_LABELS[method["method_id"]])}</text>'
        )
        for column, book in enumerate(books):
            result = lookup[(method["method_id"], book)]["source_success"]
            color, marker = ("#dcefe4", "PASS") if result else ("#f5dddd", "FAIL")
            x = left + column * cell_w
            lines.extend(
                [
                    f'<rect x="{x + 2}" y="{y + 2}" width="{cell_w - 4}" height="{cell_h - 4}" fill="{color}" stroke="#ffffff"/>',
                    f'<text class="value" x="{x + cell_w / 2:.1f}" y="{y + 30}" text-anchor="middle">{marker}</text>',
                ]
            )
    lines.append("</svg>")
    (OUTPUT / "prompt_source_matrix.svg").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_lora_components(results: dict[str, Any]) -> None:
    rows = [row for row in results["method_summary"] if row["arm"] == "lora_internal_test"]
    rows.sort(key=lambda row: (row["method_id"] != "lora_smoke_4b", row["method_id"]))
    metrics = (
        ("Contract", "contract_complete_sources", "#4a6878"),
        ("Semantic", "semantic_pass_sources", "#cf8a2e"),
        ("Structure", "style_improved_sources", "#6c5c9b"),
        ("Joint success", "successes", "#2d7b57"),
    )
    width, height, left, chart_width = 1050, 330, 310, 600
    lines = svg_start(width, height, "LoRA smoke: book-level component success")
    for row_index, row in enumerate(rows):
        y = 72 + row_index * 78
        lines.append(
            f'<text class="label" x="24" y="{y + 24}">{html.escape(METHOD_LABELS[row["method_id"]])}</text>'
        )
        for metric_index, (label, key, color) in enumerate(metrics):
            bar_y = y + metric_index * 14
            rate = row[key] / row["sources"]
            lines.extend(
                [
                    f'<rect x="{left}" y="{bar_y}" width="{chart_width}" height="9" fill="#edf0f2"/>',
                    f'<rect x="{left}" y="{bar_y}" width="{chart_width * rate:.1f}" height="9" fill="{color}"/>',
                    f'<text class="small" x="{left - 8}" y="{bar_y + 9}" text-anchor="end">{label}</text>',
                    f'<text class="value" x="{left + chart_width + 10}" y="{bar_y + 9}">{row[key]}/{row["sources"]}</text>',
                ]
            )
    lines.append("</svg>")
    (OUTPUT / "lora_component_success.svg").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_lora_loss(results: dict[str, Any]) -> None:
    metrics = results["metrics"]
    train = [(row["iteration"], row["loss"]) for row in metrics["train_loss"]]
    valid = [(row["iteration"], row["loss"]) for row in metrics["validation_loss"]]
    width, height, left, top, chart_width, chart_height = 1000, 500, 90, 60, 820, 350
    max_x, max_y = 120, 2.3
    x = lambda value: left + chart_width * value / max_x
    y = lambda value: top + chart_height * (1 - value / max_y)
    lines = svg_start(width, height, "Qwen3-4B MLX LoRA smoke loss")
    lines.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}"/>',
            f'<line class="axis" x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}"/>',
        ]
    )
    for value in (0, 0.5, 1.0, 1.5, 2.0):
        py = y(value)
        lines.extend(
            [
                f'<line x1="{left}" y1="{py:.1f}" x2="{left + chart_width}" y2="{py:.1f}" stroke="#e6e8ea"/>',
                f'<text class="small" x="{left - 12}" y="{py + 4:.1f}" text-anchor="end">{value:.1f}</text>',
            ]
        )
    for value in (0, 20, 40, 60, 80, 100, 120):
        px = x(value)
        lines.append(
            f'<text class="small" x="{px:.1f}" y="{top + chart_height + 24}" text-anchor="middle">{value}</text>'
        )
    train_points = " ".join(f"{x(a):.1f},{y(b):.1f}" for a, b in train)
    valid_points = " ".join(f"{x(a):.1f},{y(b):.1f}" for a, b in valid)
    lines.extend(
        [
            f'<polyline points="{train_points}" fill="none" stroke="#2d6f9f" stroke-width="3"/>',
            f'<polyline points="{valid_points}" fill="none" stroke="#c15b36" stroke-width="3"/>',
            '<rect x="660" y="64" width="14" height="4" fill="#2d6f9f"/><text class="small" x="682" y="70">Train loss</text>',
            '<rect x="780" y="64" width="14" height="4" fill="#c15b36"/><text class="small" x="802" y="70">Validation loss</text>',
            f'<text class="label" x="{left + chart_width / 2}" y="{height - 24}" text-anchor="middle">Iteration</text>',
            f'<text class="small" x="{left}" y="{height - 6}">Test loss {metrics["test_loss"]:.3f}; perplexity {metrics["test_perplexity"]:.3f}; infrastructure smoke only</text>',
        ]
    )
    lines.append("</svg>")
    (OUTPUT / "lora_loss.svg").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    benchmark = read_json(RESULTS)
    lora = read_json(LORA_RESULTS)
    if benchmark["status"] != "no_go_smoke_lora_clean_pair_corpus_only":
        raise ValueError("unexpected Iteration 5 benchmark status")
    if lora["status"] != "pass_infrastructure_smoke_only":
        raise ValueError("unexpected LoRA smoke status")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_prompt_components(benchmark)
    write_source_matrix(benchmark)
    write_lora_components(benchmark)
    write_lora_loss(lora)
    print(f"wrote report figures to {OUTPUT}")


if __name__ == "__main__":
    main()
