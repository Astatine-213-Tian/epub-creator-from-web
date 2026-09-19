"""Persist normalization findings and cumulative repair history."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from src.content.normalization import NormalizationReport


def automatic_report_path(path: Path) -> Path:
    path = Path(path).resolve()
    root = Path(__file__).resolve().parents[2]
    books = (root / "books").resolve()
    try:
        relative = path.relative_to(books)
    except ValueError:
        return path.with_suffix(".normalization.json")
    return root / "reports" / "normalization" / relative.with_suffix(".json")


def write_reports(path: Path, reports: Iterable[NormalizationReport]) -> None:
    current_reports = [report.to_dict() for report in reports]

    def has_applied_fixes(snapshot: object) -> bool:
        return isinstance(snapshot, list) and any(
            isinstance(item, dict)
            and item.get("applied") is True
            and int(item.get("total_changes", 0)) > 0
            for item in snapshot
        )

    fix_history: list[dict[str, object]] = []
    if path.exists():
        try:
            previous_payload = json.loads(path.read_text(encoding="utf-8"))
            previous_history = previous_payload.get("fix_history", [])
            if isinstance(previous_history, list):
                fix_history.extend(
                    dict(item)
                    for item in previous_history
                    if isinstance(item, dict) and isinstance(item.get("reports"), list)
                )
            previous_reports = previous_payload.get("reports", [])
            if (
                isinstance(previous_reports, list)
                and previous_reports
                and has_applied_fixes(previous_reports)
                and (
                    not fix_history
                    or fix_history[-1].get("reports") != previous_reports
                )
            ):
                fix_history.append({"reports": previous_reports})
        except (OSError, ValueError, TypeError):
            fix_history = []

    material_runs = fix_history + (
        [{"reports": current_reports}]
        if has_applied_fixes(current_reports)
        and (not fix_history or fix_history[-1].get("reports") != current_reports)
        else []
    )
    cumulative_counts: Counter[str] = Counter()
    for entry in material_runs:
        entry_reports = entry.get("reports", [])
        if not isinstance(entry_reports, list):
            continue
        for item in entry_reports:
            if not isinstance(item, dict):
                continue
            counts = item.get("change_counts", {})
            if isinstance(counts, dict):
                cumulative_counts.update(
                    {
                        str(kind): int(count)
                        for kind, count in counts.items()
                        if isinstance(count, int)
                    }
                )
    payload = {
        "schema_version": 1,
        "reports": current_reports,
        "fix_history": fix_history,
        "cumulative_fixes": {
            "material_runs": len(material_runs),
            "total_changes": sum(cumulative_counts.values()),
            "change_counts": dict(sorted(cumulative_counts.items())),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
