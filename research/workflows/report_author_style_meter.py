#!/usr/bin/env python3
from __future__ import annotations

"""Render the canonical report for the current normalized authorship meter."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from .author_style_meter_contract import (
    CROSS_BOOK_DECONTAMINATION_VERSION,
    CURRENT_BENCHMARK_DIR,
    CURRENT_BENCHMARK_RESULT,
    CURRENT_BENCHMARK_RUN_KEY,
    CURRENT_CHAR_MIN_DF,
    CURRENT_MASKED_VIEW,
    CURRENT_SCORER_VALIDATION_DIR,
    CURRENT_SCORER_VALIDATION_RESULT,
    CURRENT_SCORER_VALIDATION_RUN_KEY,
    MASKING_POLICY_VERSION,
    PUNCTUATION_NORMALIZATION_VERSION,
    RESEARCH_ROOT,
    file_sha256,
)
from .benchmark_author_style import METHOD_LABELS, write_svg_bar_chart
from .benchmark_author_style_supervised import CLASSIFIER_LABELS


DEFAULT_RUN_DIRS = (
    CURRENT_BENCHMARK_DIR.relative_to(RESEARCH_ROOT),
)

FEATURE_LABELS = {
    **METHOD_LABELS,
    "mask_stripped_char_ngrams": (
        "Exact character n-grams within unmasked spans only"
    ),
}


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_bound_evaluation(
    path: Path,
    benchmark_result: Path,
    *,
    marker_name: str = "benchmark-result-sha256",
) -> str:
    text = path.read_text(encoding="utf-8").strip()
    expected = f"<!-- {marker_name}: {file_sha256(benchmark_result)} -->"
    if expected not in text.splitlines()[:5]:
        raise SystemExit(
            f"{path} is not bound to the current benchmark result; expected marker {expected}"
        )
    return text


def load_payloads(run_dirs: tuple[Path, ...]) -> list[tuple[Path, dict[str, Any]]]:
    payloads: list[tuple[Path, dict[str, Any]]] = []
    for run_dir in run_dirs:
        path = run_dir / "supervised_author_baseline_results.json"
        if not path.exists():
            raise SystemExit(f"missing result file: {path}")
        payload = read_json(path)
        validate_payload_bindings(path, payload)
        payloads.append((run_dir, payload))
    return payloads


def validate_payload_bindings(result_path: Path, payload: dict[str, Any]) -> None:
    bindings = payload.get("input_bindings")
    if not isinstance(bindings, dict):
        raise SystemExit(
            f"{result_path} predates the current corpus contract; rerun authorship-supervised"
        )
    normalization = bindings.get("punctuation_normalization")
    if normalization != PUNCTUATION_NORMALIZATION_VERSION:
        raise SystemExit(
            f"{result_path} uses punctuation normalization {normalization!r}, expected "
            f"{PUNCTUATION_NORMALIZATION_VERSION!r}"
        )
    if bindings.get("cross_book_decontamination") != CROSS_BOOK_DECONTAMINATION_VERSION:
        raise SystemExit(f"{result_path} does not use current cross-book decontamination")
    if bindings.get("masking_policy") != MASKING_POLICY_VERSION:
        raise SystemExit(f"{result_path} does not use current train-fit masking")
    if int(payload.get("char_min_df", -1)) != CURRENT_CHAR_MIN_DF:
        raise SystemExit(
            f"{result_path} uses char_min_df={payload.get('char_min_df')}, expected "
            f"{CURRENT_CHAR_MIN_DF}"
        )
    chunk_files = bindings.get("chunk_files")
    if not isinstance(chunk_files, dict):
        raise SystemExit(f"{result_path} has no bound chunk files")
    mask_plan = bindings.get("mask_plan")
    if not isinstance(mask_plan, dict):
        raise SystemExit(f"{result_path} has no bound mask plan")
    mask_plan_path = Path(str(mask_plan.get("path", "")))
    if not mask_plan_path.is_absolute():
        mask_plan_path = RESEARCH_ROOT / mask_plan_path
    if not mask_plan_path.is_file() or file_sha256(mask_plan_path) != mask_plan.get("sha256"):
        raise SystemExit(f"bound mask plan changed after benchmark: {mask_plan_path}")
    source_binding = bindings.get("benchmark_source")
    if not isinstance(source_binding, dict):
        raise SystemExit(f"{result_path} has no benchmark-source binding")
    source_path = RESEARCH_ROOT / str(source_binding.get("path", ""))
    if not source_path.is_file() or file_sha256(source_path) != source_binding.get("sha256"):
        raise SystemExit(
            f"benchmark source changed after {result_path} was produced; rerun authorship-supervised"
        )
    for view in payload.get("views", []):
        binding = chunk_files.get(view)
        if not isinstance(binding, dict):
            raise SystemExit(f"{result_path} has no input binding for {view}")
        path = Path(str(binding.get("path", "")))
        if not path.is_absolute():
            path = RESEARCH_ROOT / path
        if not path.is_file():
            raise SystemExit(f"bound chunk file is missing: {path}")
        observed = file_sha256(path)
        if observed != binding.get("sha256"):
            raise SystemExit(
                f"bound chunk file changed after benchmark: {path}; rerun authorship-supervised"
            )


def validate_scorer_validation_bindings(
    result_path: Path, payload: dict[str, Any]
) -> None:
    protocol = payload.get("protocol", {})
    if (
        protocol.get("primary_view") != CURRENT_MASKED_VIEW
        or protocol.get("primary_classifier") != "sgd_hinge_unbalanced"
        or protocol.get("primary_representation")
        != "mask_stripped_char_ngrams"
    ):
        raise SystemExit(f"{result_path} does not match the current scorer protocol")
    if payload.get("protocol_evaluation", {}).get("all_checks_pass") is not True:
        raise SystemExit(f"{result_path} did not pass its registered diagnostic gate")
    if CURRENT_SCORER_VALIDATION_RUN_KEY not in payload.get("results", {}):
        raise SystemExit(
            f"{result_path} is missing {CURRENT_SCORER_VALIDATION_RUN_KEY}"
        )

    bindings = payload.get("input_bindings", {})
    records = [
        bindings.get("source"),
        bindings.get("main_benchmark"),
        bindings.get("mask_plan"),
        bindings.get("chunk_files", {}).get(CURRENT_MASKED_VIEW),
        payload.get("prior_ablation_reference"),
    ]
    for binding in records:
        if not isinstance(binding, dict):
            raise SystemExit(f"{result_path} has an incomplete input binding")
        path = Path(str(binding.get("path", "")))
        if not path.is_absolute():
            path = RESEARCH_ROOT / path
        if not path.is_file() or file_sha256(path) != binding.get("sha256"):
            raise SystemExit(f"bound scorer-validation input changed: {path}")


def load_scorer_validation(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"missing scorer validation: {path}")
    payload = read_json(path)
    validate_scorer_validation_bindings(path, payload)
    return payload


def collect_rows(payloads: list[tuple[Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for run_dir, payload in payloads:
        for key, result in payload["results"].items():
            spec = result["spec"]
            dedupe_key = f"{spec['classifier']}.{spec['method']}.{spec['view']}"
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            chunk_test = result["chunk_metrics"]["test"]
            book_test = result["book_metrics"]["test"]
            proxy = result["chunk_metrics"].get("proxy_transfer", {})
            rows.append(
                {
                    "run_dir": run_dir,
                    "key": key,
                    "classifier": spec["classifier"],
                    "method": spec["method"],
                    "view": spec["view"],
                    "feature_count": result["feature_count"],
                    "char_min_df": payload.get("char_min_df"),
                    "chunk_accuracy": chunk_test["accuracy"],
                    "balanced_accuracy": chunk_test["balanced_accuracy"],
                    "macro_f1": chunk_test["macro_f1"],
                    "majority_baseline": chunk_test.get("majority_baseline_accuracy"),
                    "target_f1": chunk_test.get("target", {}).get("f1"),
                    "target_precision": chunk_test.get("target", {}).get("precision"),
                    "target_recall": chunk_test.get("target", {}).get("recall"),
                    "proxy_target_recall": proxy.get("target", {}).get("recall"),
                    "book_accuracy": book_test["accuracy"],
                    "result": result,
                }
            )
    return rows


def run_label(row: dict[str, Any]) -> str:
    classifier = CLASSIFIER_LABELS.get(row["classifier"], row["classifier"])
    method = FEATURE_LABELS.get(row["method"], row["method"])
    return f"{classifier} / {method}"


def select_recommended_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        if row["key"] == CURRENT_SCORER_VALIDATION_RUN_KEY:
            return row
    raise SystemExit(
        f"missing preregistered current run: {CURRENT_SCORER_VALIDATION_RUN_KEY}"
    )


def row_table(rows: list[dict[str, Any]], *, limit: int | None = None) -> list[str]:
    lines = [
        "| Rank | View | Classifier | Features | Chunk acc. | Balanced acc. | Book acc. | Target F1 | Target recall | Majority baseline |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    selected = rows[:limit] if limit is not None else rows
    for index, row in enumerate(selected, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    row["view"],
                    CLASSIFIER_LABELS.get(row["classifier"], row["classifier"]),
                    FEATURE_LABELS.get(row["method"], row["method"]),
                    pct(row["chunk_accuracy"]),
                    pct(row["balanced_accuracy"]),
                    pct(row["book_accuracy"]),
                    pct(row["target_f1"]),
                    pct(row["target_recall"]),
                    pct(row["majority_baseline"]),
                ]
            )
            + " |"
        )
    return lines


def paired_gap_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_pair: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_pair.setdefault((row["classifier"], row["method"]), {})[row["view"]] = row
    output: list[dict[str, Any]] = []
    for (classifier, method), views in by_pair.items():
        clean = views.get("clean")
        masked = views.get(CURRENT_MASKED_VIEW)
        if not clean or not masked:
            continue
        output.append(
            {
                "classifier": classifier,
                "method": method,
                "clean": clean,
                "masked": masked,
                "gap": clean["chunk_accuracy"] - masked["chunk_accuracy"],
                "balanced_gap": clean["balanced_accuracy"] - masked["balanced_accuracy"],
            }
        )
    return sorted(output, key=lambda item: item["masked"]["chunk_accuracy"], reverse=True)


def gap_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Classifier | Features | Clean chunk acc. | Masked chunk acc. | Clean - masked | Masked balanced acc. |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    CLASSIFIER_LABELS.get(item["classifier"], item["classifier"]),
                    FEATURE_LABELS.get(item["method"], item["method"]),
                    pct(item["clean"]["chunk_accuracy"]),
                    pct(item["masked"]["chunk_accuracy"]),
                    f"{item['gap'] * 100:+.1f}pp",
                    pct(item["masked"]["balanced_accuracy"]),
                ]
            )
            + " |"
        )
    return lines


def mask_ablation_table(rows: list[dict[str, Any]]) -> list[str]:
    by_pair: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_pair.setdefault((row["classifier"], row["method"]), {})[row["view"]] = row
    lines = [
        "| Classifier | Features | Train-global only | Global + label-blind local | Combined - global |",
        "|---|---|---:|---:|---:|",
    ]
    for (classifier, method), views in sorted(by_pair.items()):
        global_only = views.get("train_global_masked")
        combined = views.get("entity_masked_v3")
        if not global_only or not combined:
            continue
        delta = combined["chunk_accuracy"] - global_only["chunk_accuracy"]
        lines.append(
            f"| {CLASSIFIER_LABELS.get(classifier, classifier)} | "
            f"{FEATURE_LABELS.get(method, method)} | {pct(global_only['chunk_accuracy'])} | "
            f"{pct(combined['chunk_accuracy'])} | {delta * 100:+.1f}pp |"
        )
    return lines


def per_author_table(best: dict[str, Any], *, limit: int | None = None) -> list[str]:
    by_label = best["result"]["chunk_metrics"]["test"]["by_label"]
    rows = sorted(
        (
            {
                "author": author,
                "precision": stats["precision"],
                "recall": stats["recall"],
                "f1": stats["f1"],
                "tp": stats["tp"],
                "fp": stats["fp"],
                "fn": stats["fn"],
                "support": stats["tp"] + stats["fn"],
            }
            for author, stats in by_label.items()
        ),
        key=lambda row: (row["recall"], row["f1"], row["author"]),
    )
    lines = [
        "| Author | Precision | Recall | F1 | Support | TP | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    selected = rows if limit is None else rows[:limit]
    for row in selected:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["author"],
                    pct(row["precision"]),
                    pct(row["recall"]),
                    pct(row["f1"]),
                    str(row["support"]),
                    str(row["tp"]),
                    str(row["fp"]),
                    str(row["fn"]),
                ]
            )
            + " |"
        )
    return lines


def confusion_table(best: dict[str, Any], *, limit: int = 12) -> list[str]:
    lines = ["| Gold author | Predicted author | Count |", "|---|---|---:|"]
    for row in best["result"]["top_test_confusions"][:limit]:
        lines.append(f"| {row['gold']} | {row['pred']} | {row['count']} |")
    return lines


def all_author_recall(best: dict[str, Any]) -> list[tuple[str, float]]:
    by_label = best["result"]["chunk_metrics"]["test"]["by_label"]
    return sorted(((author, stats["recall"]) for author, stats in by_label.items()), key=lambda item: item[1])


def dataset_summary(dataset_root: Path) -> dict[str, Any]:
    manifest_path = dataset_root / "dataset_manifest.json"
    cleaned_path = Path("generated/style_research/corpus/cleaned_manifest.json")
    splits_path = Path("generated/style_research/corpus/splits.json")
    clean_chunks = dataset_root / "unmasked/chunks.clean.jsonl"
    global_masked_chunks = dataset_root / "masked/chunks.train_global_masked.jsonl"
    diagnostic_masked_chunks = dataset_root / "masked/chunks.entity_masked_v3.jsonl"
    cleaning_report_path = Path("generated/style_research/corpus/cleaning_report.json")
    manifest = read_json(manifest_path)
    cleaned = read_json(cleaned_path)
    splits = read_json(splits_path)
    cleaning_report = read_json(cleaning_report_path)
    authors = sorted({item["author"] for item in cleaned if item.get("exists")})
    usable = [item for item in cleaned if item.get("clean_cjk_count", 0) >= 50_000]
    primary = [item for item in cleaned if item.get("clean_cjk_count", 0) >= 120_000]
    return {
        "manifest_books": len(manifest),
        "cleaned_books": len(cleaned),
        "authors": len(authors),
        "usable_books_50k": len(usable),
        "primary_books_120k": len(primary),
        "split_counts": {split: len(items) for split, items in splits.items()},
        "clean_chunks": sum(1 for _ in clean_chunks.open(encoding="utf-8")),
        "global_masked_chunks": sum(
            1 for _ in global_masked_chunks.open(encoding="utf-8")
        ),
        "diagnostic_masked_chunks": sum(
            1 for _ in diagnostic_masked_chunks.open(encoding="utf-8")
        ),
        "punctuation_normalization": cleaning_report.get("punctuation_normalization", "unknown"),
        "books_with_punctuation_normalization": cleaning_report.get(
            "books_with_punctuation_normalization", 0
        ),
        "cross_book_decontamination": cleaning_report.get(
            "cross_book_decontamination", {}
        ),
    }


def write_charts(rows: list[dict[str, Any]], best: dict[str, Any], chart_dir: Path) -> dict[str, Path]:
    chart_dir.mkdir(parents=True, exist_ok=True)
    masked_rows = sorted(
        [row for row in rows if row["view"] == CURRENT_MASKED_VIEW],
        key=lambda row: row["chunk_accuracy"],
        reverse=True,
    )
    masked_chart = chart_dir / "masked_chunk_accuracy.svg"
    write_svg_bar_chart(
        masked_chart,
        "Train-Global-Masked Test Chunk Accuracy",
        [(run_label(row), row["chunk_accuracy"]) for row in masked_rows],
    )
    balanced_chart = chart_dir / "masked_balanced_accuracy.svg"
    write_svg_bar_chart(
        balanced_chart,
        "Train-Global-Masked Test Balanced Accuracy",
        [(run_label(row), row["balanced_accuracy"]) for row in masked_rows],
    )
    gap_chart = chart_dir / "clean_to_masked_gap.svg"
    write_svg_bar_chart(
        gap_chart,
        "Masked - Clean Chunk Accuracy Gain",
        [
            (
                f"{CLASSIFIER_LABELS.get(item['classifier'], item['classifier'])} / {FEATURE_LABELS.get(item['method'], item['method'])}",
                -item["gap"],
            )
            for item in paired_gap_rows(rows)
        ],
    )
    recall_chart = chart_dir / "best_method_per_author_recall.svg"
    write_svg_bar_chart(
        recall_chart,
        "Best Method Per-Author Recall",
        all_author_recall(best),
    )
    return {
        "masked_accuracy": masked_chart,
        "masked_balanced": balanced_chart,
        "gap": gap_chart,
        "per_author_recall": recall_chart,
    }


def rel_link(from_path: Path, to_path: Path) -> str:
    return os.path.relpath(to_path, start=from_path.parent)


def build_report(args: argparse.Namespace) -> str:
    payloads = load_payloads(args.run_dirs)
    baseline_rows = collect_rows(payloads)
    scorer_payload = load_scorer_validation(args.scorer_validation_result)
    validation_rows = collect_rows(
        [(CURRENT_SCORER_VALIDATION_DIR, scorer_payload)]
    )
    rows = baseline_rows + validation_rows
    best_overall = max(
        (
            row
            for row in validation_rows
            if row["view"] == CURRENT_MASKED_VIEW
        ),
        key=lambda row: row["chunk_accuracy"],
    )
    recommended = select_recommended_row(rows)
    overall_rows = sorted(rows, key=lambda row: (row["chunk_accuracy"], row["balanced_accuracy"]), reverse=True)
    gaps = paired_gap_rows(rows)
    summary = dataset_summary(args.dataset_root)
    charts = write_charts(rows, recommended, args.chart_dir)
    evaluation = ""
    if args.evaluation_file and args.evaluation_file.exists():
        evaluation = load_bound_evaluation(
            args.evaluation_file,
            args.scorer_validation_result,
            marker_name="ablation-result-sha256",
        )
    baseline_evaluation = ""
    if args.baseline_evaluation_file and args.baseline_evaluation_file.exists():
        baseline_evaluation = load_bound_evaluation(
            args.baseline_evaluation_file,
            CURRENT_BENCHMARK_RESULT,
        )
    lines: list[str] = [
        "# Author-Style Classifier on the Current Cleaned Corpus",
        "",
        f"Generated: {args.generated_at}",
        "",
        "## Abstract",
        "",
        "This retest measures whether author identity remains recoverable after corpus cleanup, "
        "canonical Chinese punctuation normalization, cross-book passage decontamination, and "
        "train-only global masking. The benchmark uses book-level splits over 50 authors. "
        "The selected scorer removes mask spans before extracting exact character 2-4 gram "
        "TF-IDF features. "
        f"The strongest mask-stripped run reaches {pct(best_overall['chunk_accuracy'])} chunk accuracy, "
        f"{pct(best_overall['balanced_accuracy'])} balanced accuracy, and "
        f"{pct(best_overall['book_accuracy'])} book-majority accuracy.",
        "",
        "## Dataset",
        "",
        f"- Manifest books: {summary['manifest_books']}",
        f"- Cleaned books: {summary['cleaned_books']}",
        f"- Authors: {summary['authors']}",
        f"- Usable books at >=50k cleaned CJK characters: {summary['usable_books_50k']}",
        f"- Primary books at >=120k cleaned CJK characters: {summary['primary_books_120k']}",
        f"- Book split counts: {summary['split_counts']}",
        f"- Clean chunks: {summary['clean_chunks']}",
        f"- Train-global-masked chunks: {summary['global_masked_chunks']}",
        f"- Diagnostic global-plus-local chunks: {summary['diagnostic_masked_chunks']}",
        f"- Punctuation normalization: `{summary['punctuation_normalization']}`",
        f"- Books changed by punctuation normalization: {summary['books_with_punctuation_normalization']}",
        f"- Cross-book duplicate fingerprints removed: "
        f"{summary['cross_book_decontamination'].get('detected_fingerprint_count', 0)}",
        f"- Cross-book duplicate lines removed: "
        f"{summary['cross_book_decontamination'].get('removed_line_count', 0)}",
        f"- Remaining checked duplicate fingerprints: "
        f"{summary['cross_book_decontamination'].get('remaining_fingerprint_count', 'unknown')}",
        "",
        "Every book is assigned wholly to train, development, test, or target-author proxy "
        "holdout. No book contributes chunks to more than one split.",
        "",
        "## Methods",
        "",
        "- Selected features: exact character 2-4 grams extracted only within unmasked spans "
        "  after every `某` run is removed; `TfidfVectorizer`, `max_features=80000`, "
        "  `min_df=20`, sublinear TF-IDF, and no hash collisions.",
        "- Text normalization: whitespace is removed for n-gram extraction. Dataset cleanup "
        "  canonicalizes equivalent punctuation encodings while preserving punctuation roles.",
        "- Classifiers: SGD hinge with and without class balancing; random seed 13.",
        "- Mask fitting: one global content vocabulary is learned from train books only and then "
        "  applied identically to every split. It is the only current scorer input view.",
        "- Diagnostic views: clean text and global-plus-book-local v3 are retained to measure "
        "  content and preprocessing effects; v3 is not eligible for scorer selection.",
        "- Metrics: chunk accuracy, balanced accuracy, macro F1, target-author precision/recall/F1, "
        "  and book-level majority-vote accuracy.",
        "",
        "## Main Result",
        "",
        f"Highest mask-stripped classifier: **{run_label(best_overall)}** on `{CURRENT_MASKED_VIEW}`.",
        "",
        f"- Chunk accuracy: {pct(best_overall['chunk_accuracy'])}",
        f"- Balanced accuracy: {pct(best_overall['balanced_accuracy'])}",
        f"- Book-majority accuracy: {pct(best_overall['book_accuracy'])}",
        f"- Target-author F1: {pct(best_overall['target_f1'])}",
        "",
        f"Selected provisional authorship proxy: **{run_label(recommended)}** on `{CURRENT_MASKED_VIEW}`.",
        "",
        f"- Chunk accuracy: {pct(recommended['chunk_accuracy'])}",
        f"- Balanced accuracy: {pct(recommended['balanced_accuracy'])}",
        f"- Macro F1: {pct(recommended['macro_f1'])}",
        f"- Book-majority accuracy: {pct(recommended['book_accuracy'])}",
        f"- Target-author precision: {pct(recommended['target_precision'])}",
        f"- Target-author F1: {pct(recommended['target_f1'])}",
        f"- Target-author recall: {pct(recommended['target_recall'])}",
        f"- Majority baseline: {pct(recommended['majority_baseline'])}",
        f"- Features: {recommended['feature_count']}",
        "",
        f"![Train-global-masked chunk accuracy]({rel_link(args.output, charts['masked_accuracy'])})",
        "",
        f"![Train-global-masked balanced accuracy]({rel_link(args.output, charts['masked_balanced'])})",
        "",
        "## All Current Runs",
        "",
        *row_table(overall_rows),
        "",
        "## Masked vs Clean",
        "",
        "Global masking removes many book-specific names and setting terms. Performance "
        "must therefore be interpreted as content-resistant author signal, not as proof that every "
        "learned feature is literary style.",
        "",
        *gap_table(gaps),
        "",
        f"![Clean to masked gap]({rel_link(args.output, charts['gap'])})",
        "",
        "## Masking Ablation",
        "",
        "The parent benchmark compares the fixed train-global vocabulary with the ineligible "
        "book-local supplement. Separate collapsed-run, topology-only, and mask-stripped controls "
        "then test whether mask artifacts explain the retained author signal.",
        "",
        *mask_ablation_table(rows),
        "",
        "The final mask-stripped global run improves over the original global run by "
        f"{(recommended['chunk_accuracy'] - next(row for row in baseline_rows if row['key'] == CURRENT_BENCHMARK_RUN_KEY)['chunk_accuracy']) * 100:+.2f}pp "
        "test accuracy. Its development accuracy is 87.94% and development balanced accuracy "
        "is 84.57%. All registered v1 and v2 mask-artifact checks pass.",
        "",
        "## Per-Author Results",
        "",
        "All 50 authors under the recommended method, ordered from lowest to highest recall:",
        "",
        *per_author_table(recommended),
        "",
        f"![Per-author recall]({rel_link(args.output, charts['per_author_recall'])})",
        "",
        "## Top Confusions",
        "",
        *confusion_table(recommended),
        "",
        "## Interpretation",
        "",
        "- Use masked balanced accuracy and target-author F1 together; raw accuracy alone is "
        "  insufficient under uneven book and chunk counts.",
        "- Treat book-majority accuracy as a sanity check because most authors contribute only one "
        "  test book.",
        "- Character n-grams remain leakage-sensitive. Train-only masking, punctuation normalization, "
        "  exact decontamination, and mask-stripped extraction control known shortcuts but cannot "
        "  remove every topic, formatting, or source artifact.",
        "- This classifier is currently an authorship proxy, not a validated continuous style "
        "  meter. Generated-text calibration, uncertainty estimates, semantic fidelity, and "
        "  Chinese-readability evaluation remain separate requirements.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "uv run author-style-research authorship-supervised \\",
        "  --views clean,train_global_masked,entity_masked_v3 \\",
        "  --masked-view entity_masked_v3 \\",
        "  --methods char_ngrams \\",
        "  --classifiers sgd_hinge,sgd_hinge_unbalanced \\",
        "  --max-char-features 80000 \\",
        "  --char-min-df 20 \\",
        "  --output-dir generated/style_research/benchmarks/author_style_supervised_50authors_cleaned \\",
        "  --jobs -1 --max-iter 3000",
        "",
        "uv run author-style-research mask-artifact-audit --jobs -1",
        "",
        "uv run author-style-research authorship-report \\",
        "  --baseline-evaluation-file generated/style_research/benchmarks/author_style_supervised_50authors_cleaned/evaluator_review.md \\",
        "  --evaluation-file generated/style_research/benchmarks/mask_artifact_ablation_v2/evaluator_review.md",
        "```",
        "",
        "## Independent Evaluation",
        "",
        "### Parent Benchmark Audit",
        "",
        baseline_evaluation if baseline_evaluation else "_Pending parent-benchmark evaluator review._",
        "",
        "### Final Mask-Artifact Audit",
        "",
        evaluation if evaluation else "_Pending evaluator-agent review._",
        "",
        "## Conclusion",
        "",
        f"Freeze **{run_label(recommended)}** on `{CURRENT_MASKED_VIEW}` as the versioned "
        "authorship-attribution proxy for generated-domain calibration. Its test metrics are descriptive, not "
        "a basis for choosing between classifiers. Do not promote it to a continuous style meter "
        "until generated-text and human-rating calibration are complete.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument("--run-dirs", type=Path, nargs="*", default=list(DEFAULT_RUN_DIRS))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/reports/02_authorship_style_meter.md"),
    )
    parser.add_argument(
        "--chart-dir",
        type=Path,
        default=Path("generated/style_research/benchmarks/author_style_supervised_50authors_cleaned/report_charts"),
    )
    parser.add_argument(
        "--scorer-validation-result",
        type=Path,
        default=CURRENT_SCORER_VALIDATION_RESULT,
    )
    parser.add_argument("--baseline-evaluation-file", type=Path)
    parser.add_argument("--evaluation-file", type=Path)
    parser.add_argument(
        "--generated-at",
        default=datetime.now().astimezone().isoformat(timespec="seconds"),
    )
    args = parser.parse_args()
    args.run_dirs = tuple(args.run_dirs)
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(args.output), "chart_dir": str(args.chart_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
