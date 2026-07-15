#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


METHODS = (
    "aligned_pairs_only",
    "aligned_pairs_plus_cards",
    "aligned_pairs_edit_plan",
)
ARMS = ("own_author_reconstruction", "cross_author_transfer")


def read_scores(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def select_rows(evaluation_root: Path) -> list[dict[str, Any]]:
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for method_id in METHODS:
        score_path = evaluation_root / "methods" / method_id / "medium" / "scores.csv"
        for row in read_scores(score_path):
            candidate = {
                "sample_id": row["sample_id"],
                "benchmark_arm": row["benchmark_arm"],
                "author": row["author"],
                "book_title": row["book_title"],
                "method_id": method_id,
                "target_margin": float(row["target_margin"]),
                "neutral_target_margin": float(row["neutral_target_margin"]),
                "paired_margin_lift": float(row["paired_margin_lift"]),
                "hard_fidelity_failure": as_bool(row["hard_fidelity_failure"]),
                "no_copy_pass": '"status":"pass"' in row["no_copy_gate"].replace(" ", ""),
            }
            by_sample[row["sample_id"]].append(candidate)

    selections: list[dict[str, Any]] = []
    for sample_id in sorted(by_sample):
        candidates = by_sample[sample_id]
        eligible = [
            row
            for row in candidates
            if not row["hard_fidelity_failure"] and row["no_copy_pass"]
        ]
        eligible.sort(
            key=lambda row: (
                -row["target_margin"],
                -row["paired_margin_lift"],
                row["method_id"],
            )
        )
        base = candidates[0]
        if eligible:
            winner = eligible[0]
            selections.append({**winner, "status": "selected"})
        else:
            selections.append(
                {
                    **base,
                    "method_id": None,
                    "target_margin": base["neutral_target_margin"],
                    "paired_margin_lift": 0.0,
                    "status": "unresolved_no_eligible_method",
                }
            )
    return selections


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["benchmark_arm"] == arm]
        selected = [row for row in arm_rows if row["status"] == "selected"]
        result[arm] = {
            "rows": len(arm_rows),
            "selected_rows": len(selected),
            "unresolved_rows": len(arm_rows) - len(selected),
            "conservative_mean_paired_margin_lift": sum(
                row["paired_margin_lift"] for row in arm_rows
            )
            / len(arm_rows),
            "selected_only_mean_paired_margin_lift": (
                sum(row["paired_margin_lift"] for row in selected) / len(selected)
                if selected
                else None
            ),
            "positive_lift_rows": sum(row["paired_margin_lift"] > 0 for row in arm_rows),
            "method_selection_counts": dict(
                sorted(Counter(row["method_id"] for row in selected).items())
            ),
        }
    return result


def write_outputs(output_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "family_rerank_summary.json"
    json_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "diagnostic_only",
                "selection_order": [
                    "exclude_hard_fidelity_or_copy_failures",
                    "maximize_target_margin",
                    "maximize_paired_margin_lift",
                    "method_id_lexical_tiebreak",
                ],
                "methods": list(METHODS),
                "arms": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    csv_path = output_dir / "family_rerank_selections.csv"
    fieldnames = [
        "sample_id",
        "benchmark_arm",
        "author",
        "book_title",
        "status",
        "method_id",
        "neutral_target_margin",
        "target_margin",
        "paired_margin_lift",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fieldnames} for row in rows)

    lines = [
        "# Registered Family Rerank Diagnostic",
        "",
        "This is a deterministic diagnostic over frozen base-method outputs, not a generated method arm.",
        "Unresolved rows contribute zero lift in the conservative estimate and are not silently dropped.",
        "",
        "| Arm | Rows | Selected | Unresolved | Conservative lift | Selected-only lift | Positive lift |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        row = summary[arm]
        selected_lift = row["selected_only_mean_paired_margin_lift"]
        lines.append(
            f"| `{arm}` | {row['rows']} | {row['selected_rows']} | {row['unresolved_rows']} | "
            f"{row['conservative_mean_paired_margin_lift']:.4f} | "
            f"{selected_lift:.4f} | {row['positive_lift_rows']} |"
        )
    lines.extend(("", "## Selection Counts", ""))
    for arm in ARMS:
        counts = summary[arm]["method_selection_counts"]
        rendered = ", ".join(f"`{key}`: {value}" for key, value in counts.items()) or "none"
        lines.append(f"- `{arm}`: {rendered}")
    lines.append("")
    (output_dir / "family_rerank_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the preregistered deterministic family-rerank diagnostic."
    )
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluation_root = args.evaluation_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else evaluation_root / "diagnostics" / "family_rerank"
    )
    rows = select_rows(evaluation_root)
    summary = summarize(rows)
    write_outputs(output_dir, rows, summary)
    print(
        json.dumps(
            {"status": "complete", "output_dir": str(output_dir), "arms": summary},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
