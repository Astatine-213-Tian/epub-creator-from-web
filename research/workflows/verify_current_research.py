from __future__ import annotations

"""Fail closed when maintained research artifacts do not share one corpus contract."""

import json
from collections import Counter
from pathlib import Path

from .author_style_meter_contract import (
    CROSS_BOOK_DECONTAMINATION_VERSION,
    CURRENT_BENCHMARK_RESULT,
    CURRENT_BENCHMARK_RUN_KEY,
    CURRENT_SCORER_VALIDATION_RESULT,
    CURRENT_SCORER_VALIDATION_RUN_KEY,
    MASKING_POLICY_VERSION,
    RESEARCH_ROOT,
)
from .benchmark_author_style import chunk_paths
from .benchmark_author_style_supervised import load_records, validate_view_alignment
from .report_author_style_meter import (
    validate_payload_bindings,
    validate_scorer_validation_bindings,
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    dataset_root = RESEARCH_ROOT / "datasets"
    manifest = read_json(dataset_root / "dataset_manifest.json")
    for row in manifest:
        txt_path = Path(str(row.get("txt_path", "")))
        if txt_path.is_absolute() or not str(txt_path).startswith("raw/"):
            raise SystemExit("active manifest contains a noncanonical TXT path")
        if not (dataset_root / txt_path).is_file():
            raise SystemExit(f"active manifest TXT is missing: {txt_path}")
        source_epub = str(row.get("source_epub") or "")
        if source_epub and Path(source_epub).is_absolute():
            raise SystemExit("active manifest contains an absolute EPUB path")
    cleaned = read_json(RESEARCH_ROOT / "generated/style_research/corpus/cleaned_manifest.json")
    manifest_books = {(row["author"], row["title"]) for row in manifest}
    cleaned_books = {(row["author"], row["title"]) for row in cleaned if row.get("exists")}
    if cleaned_books != manifest_books:
        raise SystemExit("cleaned manifest does not exactly project the active dataset manifest")
    author_counts = Counter(row["author"] for row in manifest)
    if len(author_counts) != 50 or min(author_counts.values(), default=0) < 3:
        raise SystemExit("active corpus is not 50 authors with at least three books each")
    if {
        row.get("cross_book_decontamination") for row in cleaned if row.get("exists")
    } != {CROSS_BOOK_DECONTAMINATION_VERSION}:
        raise SystemExit("cleaned books do not share the current decontamination policy")
    overlap_report = read_json(
        RESEARCH_ROOT / "generated/style_research/corpus/cross_book_passage_report.json"
    )
    if (
        overlap_report.get("policy") != CROSS_BOOK_DECONTAMINATION_VERSION
        or overlap_report.get("remaining_fingerprint_count") != 0
    ):
        raise SystemExit("cross-book overlap audit is stale or nonzero")
    mask_plan = read_json(dataset_root / "masked/mask_terms.json")
    provenance = mask_plan.get("provenance", {})
    if mask_plan.get("masking_policy") != MASKING_POLICY_VERSION:
        raise SystemExit("mask plan uses a stale policy")
    if provenance.get("fit_split") != "train":
        raise SystemExit("mask plan was not fit on train books")
    if provenance.get("transform_uses_author_label") is not False:
        raise SystemExit("mask transform is author-label-aware")
    if provenance.get("held_out_corpus_statistics_used_for_global_terms") is not False:
        raise SystemExit("mask plan uses held-out corpus statistics")

    paths = chunk_paths(dataset_root)
    records_by_view = {
        view: load_records(paths[view])
        for view in ("clean", "train_global_masked", "entity_masked_v3")
    }
    validate_view_alignment(records_by_view)

    payload = read_json(CURRENT_BENCHMARK_RESULT)
    validate_payload_bindings(CURRENT_BENCHMARK_RESULT, payload)
    if CURRENT_BENCHMARK_RUN_KEY not in payload.get("results", {}):
        raise SystemExit(f"current benchmark is missing {CURRENT_BENCHMARK_RUN_KEY}")
    scorer_validation = read_json(CURRENT_SCORER_VALIDATION_RESULT)
    validate_scorer_validation_bindings(
        CURRENT_SCORER_VALIDATION_RESULT, scorer_validation
    )
    if CURRENT_SCORER_VALIDATION_RUN_KEY not in scorer_validation.get("results", {}):
        raise SystemExit(
            f"scorer validation is missing {CURRENT_SCORER_VALIDATION_RUN_KEY}"
        )

    print(
        json.dumps(
            {
                "status": "pass",
                "books": len(manifest),
                "authors": len(author_counts),
                "chunks": len(records_by_view["clean"]),
                "benchmark": str(CURRENT_BENCHMARK_RESULT.relative_to(RESEARCH_ROOT)),
                "scorer_validation": str(
                    CURRENT_SCORER_VALIDATION_RESULT.relative_to(RESEARCH_ROOT)
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
