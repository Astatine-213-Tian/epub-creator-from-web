#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from benchmark_author_style import METHOD_LABELS, fmt, safe_filename, write_svg_bar_chart
from benchmark_author_style_supervised import CLASSIFIER_LABELS


DEFAULT_RUN_DIRS = (
    Path("generated/style_research/benchmarks/author_style_supervised_50authors_iter1_sgd_nb"),
    Path("generated/style_research/benchmarks/author_style_supervised_50authors_iter2_hash_classifier_sweep"),
    Path("generated/style_research/benchmarks/author_style_supervised_50authors_iter3_exact_hinge_mindf20"),
    Path("generated/style_research/benchmarks/author_style_supervised_50authors_iter4_function_words"),
)


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_payloads(run_dirs: tuple[Path, ...]) -> list[tuple[Path, dict[str, Any]]]:
    payloads: list[tuple[Path, dict[str, Any]]] = []
    for run_dir in run_dirs:
        path = run_dir / "supervised_author_baseline_results.json"
        if not path.exists():
            raise SystemExit(f"missing result file: {path}")
        payloads.append((run_dir, read_json(path)))
    return payloads


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
                    "target_recall": chunk_test.get("target", {}).get("recall"),
                    "proxy_target_recall": proxy.get("target", {}).get("recall"),
                    "book_accuracy": book_test["accuracy"],
                    "result": result,
                }
            )
    return rows


def run_label(row: dict[str, Any]) -> str:
    classifier = CLASSIFIER_LABELS.get(row["classifier"], row["classifier"])
    method = METHOD_LABELS.get(row["method"], row["method"])
    return f"{classifier} / {method}"


def select_recommended_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    preferred_key = "sgd_hinge.char_ngrams.entity_masked_v3"
    for row in rows:
        if row["key"] == preferred_key:
            return row
    masked_rows = [row for row in rows if row["view"] == "entity_masked_v3"]
    return max(masked_rows, key=lambda row: ((row["target_f1"] or 0.0), row["balanced_accuracy"], row["chunk_accuracy"]))


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
                    METHOD_LABELS.get(row["method"], row["method"]),
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
        masked = views.get("entity_masked_v3")
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
                    METHOD_LABELS.get(item["method"], item["method"]),
                    pct(item["clean"]["chunk_accuracy"]),
                    pct(item["masked"]["chunk_accuracy"]),
                    f"{item['gap'] * 100:+.1f}pp",
                    pct(item["masked"]["balanced_accuracy"]),
                ]
            )
            + " |"
        )
    return lines


def per_author_table(best: dict[str, Any], *, limit: int = 20) -> list[str]:
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
    for row in rows[:limit]:
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
    masked_chunks = dataset_root / "masked/chunks.entity_masked_v3.jsonl"
    manifest = read_json(manifest_path)
    cleaned = read_json(cleaned_path)
    splits = read_json(splits_path)
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
        "masked_chunks": sum(1 for _ in masked_chunks.open(encoding="utf-8")),
    }


def write_charts(rows: list[dict[str, Any]], best: dict[str, Any], chart_dir: Path) -> dict[str, Path]:
    chart_dir.mkdir(parents=True, exist_ok=True)
    masked_rows = sorted(
        [row for row in rows if row["view"] == "entity_masked_v3"],
        key=lambda row: row["chunk_accuracy"],
        reverse=True,
    )
    masked_chart = chart_dir / "masked_chunk_accuracy.svg"
    write_svg_bar_chart(
        masked_chart,
        "Masked Test Chunk Accuracy",
        [(run_label(row), row["chunk_accuracy"]) for row in masked_rows],
    )
    balanced_chart = chart_dir / "masked_balanced_accuracy.svg"
    write_svg_bar_chart(
        balanced_chart,
        "Masked Test Balanced Accuracy",
        [(run_label(row), row["balanced_accuracy"]) for row in masked_rows],
    )
    gap_chart = chart_dir / "clean_to_masked_gap.svg"
    write_svg_bar_chart(
        gap_chart,
        "Masked - Clean Chunk Accuracy Gain",
        [
            (
                f"{CLASSIFIER_LABELS.get(item['classifier'], item['classifier'])} / {METHOD_LABELS.get(item['method'], item['method'])}",
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
    rows = collect_rows(payloads)
    best_overall = max((row for row in rows if row["view"] == "entity_masked_v3"), key=lambda row: row["chunk_accuracy"])
    recommended = select_recommended_row(rows)
    masked_rows = sorted([row for row in rows if row["view"] == "entity_masked_v3"], key=lambda row: row["chunk_accuracy"], reverse=True)
    overall_rows = sorted(rows, key=lambda row: (row["chunk_accuracy"], row["balanced_accuracy"]), reverse=True)
    gaps = paired_gap_rows(rows)
    summary = dataset_summary(args.dataset_root)
    charts = write_charts(rows, recommended, args.chart_dir)
    evaluation = ""
    if args.evaluation_file and args.evaluation_file.exists():
        evaluation = args.evaluation_file.read_text(encoding="utf-8").strip()
    function_word_evaluation = ""
    if args.function_word_evaluation_file and args.function_word_evaluation_file.exists():
        function_word_evaluation = args.function_word_evaluation_file.read_text(encoding="utf-8").strip()

    lines: list[str] = [
        "# Author-Style Classifier Retest on 50-Author BL Corpus",
        "",
        f"Generated: {args.generated_at}",
        "",
        "## Abstract",
        "",
        "This retest evaluates whether author identity remains recoverable after entity masking and whether the classifier is strong enough to serve as a proxy style meter for later author-style transfer experiments. The strongest run is an exact character n-gram TF-IDF model with SGD hinge loss on `entity_masked_v3`, reaching "
        f"{pct(best_overall['chunk_accuracy'])} chunk accuracy, {pct(best_overall['balanced_accuracy'])} balanced accuracy, and {pct(best_overall['book_accuracy'])} book-majority accuracy over 50 authors. For target-author style-transfer scoring, the recommended meter is the class-balanced exact n-gram hinge model, which trades a small amount of overall accuracy for higher target-author F1.",
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
        f"- `entity_masked_v3` chunks: {summary['masked_chunks']}",
        "",
        "The split is book-level: chunks from a given book remain in one split. This is stricter than random chunk splitting and is intended to reduce within-book leakage.",
        "",
        "## Methods",
        "",
        "- Exact character n-grams: sklearn `TfidfVectorizer`, character 2-4 grams, `max_features=80000`, `min_df=20`, sublinear TF-IDF, real vocabulary, no hash collisions.",
        "- Hashed character n-grams: sklearn `HashingVectorizer`, character 2-4 grams, 262144 hash buckets, `alternate_sign=False`, TF-IDF normalization.",
        "- Interpretable baselines: punctuation/dialogue, sentence/paragraph length, function-character, richer Chinese function-word lexicon, function words plus characters, and combined interpretable feature families.",
        "- Classifiers: SGD hinge, unweighted SGD hinge, SGD logistic, unweighted SGD logistic, passive-aggressive, and ComplementNB where applicable.",
        "- Metrics: chunk-level accuracy, balanced accuracy, macro F1, target-author F1/recall, majority-class baseline, and book-level majority-vote accuracy.",
        "",
        "## Main Result",
        "",
        f"Highest overall classifier: **{run_label(best_overall)}** on `entity_masked_v3`.",
        "",
        f"- Chunk accuracy: {pct(best_overall['chunk_accuracy'])}",
        f"- Balanced accuracy: {pct(best_overall['balanced_accuracy'])}",
        f"- Book-majority accuracy: {pct(best_overall['book_accuracy'])}",
        f"- Target-author F1: {pct(best_overall['target_f1'])}",
        "",
        f"Recommended target-author proxy meter: **{run_label(recommended)}** on `entity_masked_v3`.",
        "",
        f"- Chunk accuracy: {pct(recommended['chunk_accuracy'])}",
        f"- Balanced accuracy: {pct(recommended['balanced_accuracy'])}",
        f"- Macro F1: {pct(recommended['macro_f1'])}",
        f"- Book-majority accuracy: {pct(recommended['book_accuracy'])}",
        f"- Target-author F1: {pct(recommended['target_f1'])}",
        f"- Target-author recall: {pct(recommended['target_recall'])}",
        f"- Majority baseline: {pct(recommended['majority_baseline'])}",
        f"- Features: {recommended['feature_count']}",
        "",
        f"![Masked chunk accuracy]({rel_link(args.output, charts['masked_accuracy'])})",
        "",
        f"![Masked balanced accuracy]({rel_link(args.output, charts['masked_balanced'])})",
        "",
        "## Masked vs Clean",
        "",
        "The strongest n-gram models score higher on masked text than clean text. This is surprising but plausible: entity masking removes book-specific names and topical anchors that can make train/test books from the same author less distributionally consistent. It does not by itself prove the model is purely stylistic, but it does show author signal survives content masking strongly.",
        "",
        *gap_table(gaps),
        "",
        f"![Clean to masked gap]({rel_link(args.output, charts['gap'])})",
        "",
        "## Method Comparison",
        "",
        *row_table(masked_rows),
        "",
        "## Overall Ranking",
        "",
        *row_table(overall_rows, limit=15),
        "",
        "## Per-Author Weak Spots",
        "",
        "Lowest-recall authors under the recommended method:",
        "",
        *per_author_table(recommended, limit=20),
        "",
        f"![Per-author recall]({rel_link(args.output, charts['per_author_recall'])})",
        "",
        "## Top Confusions",
        "",
        *confusion_table(recommended),
        "",
        "## Interpretation",
        "",
        "- The previous concern that the classifier was too weak is addressed for this dataset and split: the best masked chunk-level result is above 80%, and the balanced accuracy is also above 80%.",
        "- N-grams are essential. Interpretable-only features are useful diagnostics but do not reach the target chunk-level accuracy.",
        "- Richer Chinese function-word features materially improve the interpretable baselines: the best combined-rich function-word run reaches 69.5% masked chunk accuracy / 63.9% balanced accuracy, and function words plus function characters reaches 84.3% target-author F1. This is useful as a guardrail, but still below the exact n-gram meter.",
        "- Exact n-grams outperform the hashed approximation in the final hinge run, so the exact model should be treated as the preferred proxy meter when runtime allows.",
        "- The clean-to-masked direction needs caution. Masked > clean suggests the masking process may improve cross-book consistency, but it also means we should keep monitoring whether masks introduce regular artifacts that classifiers exploit.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "uv run author-style-research authorship-supervised \\",
        "  --views clean,entity_masked_v3 \\",
        "  --masked-view entity_masked_v3 \\",
        "  --methods char_hashing,punctuation_dialogue,length_shape,function_chars,combined_interpretable \\",
        "  --classifiers sgd_logistic,complement_nb \\",
        "  --max-char-features 262144 \\",
        "  --char-min-df 5 \\",
        "  --output-dir generated/style_research/benchmarks/author_style_supervised_50authors_iter1_sgd_nb \\",
        "  --jobs -1 --max-iter 3000",
        "",
        "uv run author-style-research authorship-supervised \\",
        "  --views clean,entity_masked_v3 \\",
        "  --masked-view entity_masked_v3 \\",
        "  --methods char_hashing \\",
        "  --classifiers sgd_logistic,sgd_logistic_unbalanced,sgd_hinge,sgd_hinge_unbalanced,passive_aggressive,complement_nb \\",
        "  --max-char-features 262144 \\",
        "  --char-min-df 5 \\",
        "  --output-dir generated/style_research/benchmarks/author_style_supervised_50authors_iter2_hash_classifier_sweep \\",
        "  --jobs -1 --max-iter 3000",
        "",
        "uv run author-style-research authorship-supervised \\",
        "  --views clean,entity_masked_v3 \\",
        "  --masked-view entity_masked_v3 \\",
        "  --methods char_ngrams \\",
        "  --classifiers sgd_hinge,sgd_hinge_unbalanced \\",
        "  --max-char-features 80000 \\",
        "  --char-min-df 20 \\",
        "  --output-dir generated/style_research/benchmarks/author_style_supervised_50authors_iter3_exact_hinge_mindf20 \\",
        "  --jobs -1 --max-iter 3000",
        "",
        "uv run author-style-research authorship-supervised \\",
        "  --views clean,entity_masked_v3 \\",
        "  --masked-view entity_masked_v3 \\",
        "  --methods function_chars,function_words,function_words_plus_chars,combined_interpretable,combined_rich_function_words \\",
        "  --classifiers sgd_logistic,sgd_hinge,sgd_hinge_unbalanced,passive_aggressive,complement_nb \\",
        "  --output-dir generated/style_research/benchmarks/author_style_supervised_50authors_iter4_function_words \\",
        "  --jobs -1 --max-iter 3000",
        "",
        "uv run author-style-research authorship-report \\",
        "  --evaluation-file generated/style_research/benchmarks/author_classifier_retest_50authors_report/evaluator_review.md \\",
        "  --function-word-evaluation-file generated/style_research/benchmarks/author_classifier_retest_50authors_report/function_word_evaluator_review.md",
        "```",
        "",
        "## Independent Evaluation",
        "",
        evaluation if evaluation else "_Pending evaluator-agent review._",
        "",
        "## Independent Evaluation: Function-Word Extension",
        "",
        function_word_evaluation if function_word_evaluation else "_Pending function-word evaluator-agent review._",
        "",
        "## Conclusion",
        "",
        "Use the exact character n-gram + class-balanced SGD hinge classifier on `entity_masked_v3` as the next target-author proxy style meter. Keep the unweighted exact hinge model as the best overall 50-way classifier, and keep the hashed n-gram model as a faster iteration/debugging approximation. Do not proceed with interpretable-only scoring for style-transfer method selection because it is below the required chunk-level accuracy.",
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
        default=Path("generated/style_research/benchmarks/author_classifier_retest_50authors_report/charts"),
    )
    parser.add_argument("--evaluation-file", type=Path)
    parser.add_argument("--function-word-evaluation-file", type=Path)
    parser.add_argument("--generated-at", default="2026-07-10")
    args = parser.parse_args()
    args.run_dirs = tuple(args.run_dirs)
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(args.output), "chart_dir": str(args.chart_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
