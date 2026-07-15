#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from .author_style_meter_contract import (
    CROSS_BOOK_DECONTAMINATION_VERSION,
    CURRENT_BENCHMARK_DIR,
    CURRENT_CHAR_MIN_DF,
    MASKING_POLICY_VERSION,
    PUNCTUATION_NORMALIZATION_VERSION,
    RESEARCH_ROOT,
    file_sha256,
    normalize_char_text,
)
from .benchmark_author_style import (
    FEATURE_FNS,
    METHOD_LABELS,
    TARGET_AUTHOR_DEFAULT,
    chunk_paths,
    confusion_matrix,
    evaluate_predictions,
    fmt,
    jsonl_records,
    metric,
    safe_filename,
    top_confusion_pairs,
    write_confusion_csv,
    write_svg_bar_chart,
)


CLASSIFIER_LABELS = {
    "linear_svm": "Linear SVM",
    "sgd_logistic": "SGD logistic regression",
    "sgd_logistic_unbalanced": "SGD logistic regression (unweighted)",
    "sgd_hinge": "SGD linear SVM",
    "sgd_hinge_unbalanced": "SGD linear SVM (unweighted)",
    "passive_aggressive": "Passive-aggressive linear classifier",
    "complement_nb": "Complement Naive Bayes",
}


@dataclass(frozen=True)
class ChunkRecord:
    split: str
    author: str
    title: str
    chunk_id: str
    text: str


@dataclass(frozen=True)
class RunSpec:
    key: str
    classifier: str
    method: str
    view: str


@dataclass
class MatrixBundle:
    labels: list[str]
    train_author_counts: dict[str, int]
    train_rows: int
    feature_count: int
    records_by_split: dict[str, list[ChunkRecord]]
    matrix_by_split: dict[str, sparse.csr_matrix]


def parse_csv_arg(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def load_records(path: Path) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    seen_chunks: set[str] = set()
    book_splits: dict[tuple[str, str], str] = {}
    for item in jsonl_records(path):
        normalization = str(item.get("punctuation_normalization", ""))
        if normalization != PUNCTUATION_NORMALIZATION_VERSION:
            raise ValueError(
                f"{path} contains chunks built with punctuation normalization "
                f"{normalization or '<missing>'}; rebuild with "
                f"`author-style-research corpus-build --stage all`"
            )
        decontamination = str(item.get("cross_book_decontamination", ""))
        if decontamination != CROSS_BOOK_DECONTAMINATION_VERSION:
            raise ValueError(
                f"{path} contains chunks built with cross-book decontamination "
                f"{decontamination or '<missing>'}; rebuild the corpus"
            )
        masking_policy = str(item.get("masking_policy", ""))
        if masking_policy != MASKING_POLICY_VERSION:
            raise ValueError(
                f"{path} contains chunks built with masking policy "
                f"{masking_policy or '<missing>'}; rebuild the corpus"
            )
        record = ChunkRecord(
            split=str(item["split"]),
            author=str(item["author"]),
            title=str(item["title"]),
            chunk_id=str(item["chunk_id"]),
            text=str(item["text"]),
        )
        if record.chunk_id in seen_chunks:
            raise ValueError(f"duplicate chunk_id in {path}: {record.chunk_id}")
        seen_chunks.add(record.chunk_id)
        book_key = (record.author, record.title)
        previous_split = book_splits.setdefault(book_key, record.split)
        if previous_split != record.split:
            raise ValueError(f"book appears in multiple splits in {path}: {book_key}")
        records.append(record)
    if not records:
        raise ValueError(f"no chunk records found in {path}")
    return records


def validate_view_alignment(records_by_view: dict[str, list[ChunkRecord]]) -> None:
    reference_view = next(iter(records_by_view))
    reference = {
        record.chunk_id: (record.split, record.author, record.title)
        for record in records_by_view[reference_view]
    }
    for view, records in records_by_view.items():
        observed = {
            record.chunk_id: (record.split, record.author, record.title)
            for record in records
        }
        if observed != reference:
            raise ValueError(
                f"chunk identity or split mismatch between {reference_view} and {view}; "
                "rebuild all corpus views together"
            )


def validate_mask_plan(path: Path, records: list[ChunkRecord]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("masking_policy") != MASKING_POLICY_VERSION:
        raise ValueError(f"{path} does not use the current masking policy")
    provenance = payload.get("provenance", {})
    if provenance.get("fit_split") != "train":
        raise ValueError(f"{path} was not fit on the train split")
    if provenance.get("transform_uses_author_label") is not False:
        raise ValueError(f"{path} permits author-label-aware transforms")
    if provenance.get("held_out_corpus_statistics_used_for_global_terms") is not False:
        raise ValueError(f"{path} permits held-out statistics in global mask selection")
    expected_fit_books = sorted({
        f"{record.author}::{record.title}"
        for record in records
        if record.split == "train"
    })
    if provenance.get("fit_book_ids") != expected_fit_books:
        raise ValueError(f"{path} fit-book provenance differs from current train books")
    if int(provenance.get("fit_book_count", -1)) != len(expected_fit_books):
        raise ValueError(f"{path} has an invalid fit-book count")
    return payload


def group_records_by_split(records: list[ChunkRecord]) -> dict[str, list[ChunkRecord]]:
    grouped: dict[str, list[ChunkRecord]] = defaultdict(list)
    for record in records:
        grouped[record.split].append(record)
    return dict(grouped)


def fit_feature_matrix(
    records: list[ChunkRecord],
    *,
    method: str,
    max_char_features: int,
    char_min_df: int,
) -> MatrixBundle:
    records_by_split = group_records_by_split(records)
    train_records = records_by_split.get("train", [])
    if not train_records:
        raise ValueError("no train records found")

    train_author_counts = Counter(record.author for record in train_records)
    labels = sorted(train_author_counts)

    if method == "char_ngrams":
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            max_features=max_char_features,
            min_df=char_min_df,
            sublinear_tf=True,
            norm="l2",
            lowercase=False,
            preprocessor=normalize_char_text,
            dtype=np.float32,
        )
        x_train = vectorizer.fit_transform(record.text for record in train_records)
        matrix_by_split = {
            "train": x_train,
            **{
                split: vectorizer.transform(record.text for record in split_records)
                for split, split_records in records_by_split.items()
                if split != "train"
            },
        }
        feature_count = len(vectorizer.get_feature_names_out())
    elif method == "char_hashing":
        vectorizer = HashingVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            n_features=max_char_features,
            alternate_sign=False,
            norm=None,
            lowercase=False,
            preprocessor=normalize_char_text,
            dtype=np.float32,
        )
        transformer = TfidfTransformer(sublinear_tf=True, norm="l2")
        train_counts = vectorizer.transform(record.text for record in train_records)
        x_train = transformer.fit_transform(train_counts)
        matrix_by_split = {"train": x_train}
        for split, split_records in records_by_split.items():
            if split == "train":
                continue
            counts = vectorizer.transform(record.text for record in split_records)
            matrix_by_split[split] = transformer.transform(counts)
        feature_count = max_char_features
    else:
        feature_fn = FEATURE_FNS[method]
        vectorizer = DictVectorizer()
        transformer = TfidfTransformer(sublinear_tf=True, norm="l2")
        train_counts = vectorizer.fit_transform(feature_fn(record.text) for record in train_records)
        x_train = transformer.fit_transform(train_counts)
        matrix_by_split = {"train": x_train}
        for split, split_records in records_by_split.items():
            if split == "train":
                continue
            counts = vectorizer.transform(feature_fn(record.text) for record in split_records)
            matrix_by_split[split] = transformer.transform(counts)
        feature_count = len(vectorizer.feature_names_)

    return MatrixBundle(
        labels=labels,
        train_author_counts=dict(train_author_counts),
        train_rows=len(train_records),
        feature_count=feature_count,
        records_by_split=records_by_split,
        matrix_by_split=matrix_by_split,
    )


def make_classifier(name: str, *, random_state: int, max_iter: int, n_jobs: int | None):
    if name == "linear_svm":
        return LinearSVC(
            C=1.0,
            class_weight="balanced",
            dual="auto",
            max_iter=max_iter,
            random_state=random_state,
        )
    if name == "sgd_logistic":
        return SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-5,
            class_weight="balanced",
            max_iter=max_iter,
            tol=1e-3,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    if name == "sgd_logistic_unbalanced":
        return SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-5,
            class_weight=None,
            max_iter=max_iter,
            tol=1e-3,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    if name == "sgd_hinge":
        return SGDClassifier(
            loss="hinge",
            penalty="l2",
            alpha=1e-5,
            class_weight="balanced",
            max_iter=max_iter,
            tol=1e-3,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    if name == "sgd_hinge_unbalanced":
        return SGDClassifier(
            loss="hinge",
            penalty="l2",
            alpha=1e-5,
            class_weight=None,
            max_iter=max_iter,
            tol=1e-3,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    if name == "passive_aggressive":
        return PassiveAggressiveClassifier(
            C=1.0,
            class_weight="balanced",
            max_iter=max_iter,
            tol=1e-3,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    if name == "complement_nb":
        return ComplementNB(alpha=0.5)
    raise ValueError(f"unknown classifier: {name}")


def evaluate_supervised_run(
    bundle: MatrixBundle,
    spec: RunSpec,
    *,
    target_author: str,
    random_state: int,
    max_iter: int,
    n_jobs: int | None,
) -> dict[str, Any]:
    classifier = make_classifier(spec.classifier, random_state=random_state, max_iter=max_iter, n_jobs=n_jobs)
    train_records = bundle.records_by_split["train"]
    classifier.fit(bundle.matrix_by_split["train"], [record.author for record in train_records])

    chunk_rows_by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    book_votes: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    for split, matrix in bundle.matrix_by_split.items():
        if split == "train":
            continue
        split_records = bundle.records_by_split.get(split, [])
        if not split_records:
            continue
        predictions = classifier.predict(matrix)
        for record, pred in zip(split_records, predictions):
            row = {
                "split": split,
                "gold": record.author,
                "pred": str(pred),
                "title": record.title,
                "chunk_id": record.chunk_id,
            }
            chunk_rows_by_split[split].append(row)
            book_votes[(split, record.author, record.title)][str(pred)] += 1

    majority_author = max(bundle.train_author_counts.items(), key=lambda item: (item[1], item[0]))[0]
    metrics_by_split = {
        split: evaluate_predictions(rows, bundle.labels, target_author=target_author, majority_author=majority_author)
        for split, rows in chunk_rows_by_split.items()
    }
    book_rows_by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    for (split, gold, title), votes in book_votes.items():
        pred = max(votes.items(), key=lambda item: (item[1], item[0]))[0]
        book_rows_by_split[split].append({"split": split, "gold": gold, "pred": pred, "title": title})
    book_metrics_by_split = {
        split: evaluate_predictions(rows, bundle.labels, target_author=target_author, majority_author=majority_author)
        for split, rows in book_rows_by_split.items()
    }
    test_rows = chunk_rows_by_split.get("test", [])
    matrix = confusion_matrix(test_rows, bundle.labels)
    return {
        "spec": spec.__dict__,
        "labels": bundle.labels,
        "train_chunks": bundle.train_rows,
        "train_chunks_by_author": bundle.train_author_counts,
        "feature_count": bundle.feature_count,
        "chunk_metrics": metrics_by_split,
        "book_metrics": book_metrics_by_split,
        "test_confusion_matrix": matrix,
        "top_test_confusions": top_confusion_pairs(matrix, bundle.labels, limit=10),
    }


def result_label(result: dict[str, Any]) -> str:
    spec = result["spec"]
    classifier = CLASSIFIER_LABELS.get(spec["classifier"], spec["classifier"])
    method = METHOD_LABELS.get(spec["method"], spec["method"])
    return f"{classifier} / {method} / {spec['view']}"


def build_gap_rows(
    results: dict[str, dict[str, Any]],
    *,
    classifiers: tuple[str, ...],
    methods: tuple[str, ...],
    masked_view: str,
) -> list[dict[str, Any]]:
    rows = []
    for classifier in classifiers:
        for method in methods:
            clean = results.get(f"{classifier}.{method}.clean")
            masked = results.get(f"{classifier}.{method}.{masked_view}")
            if not clean or not masked:
                continue
            rows.append(
                {
                    "classifier": classifier,
                    "method": method,
                    "clean_accuracy": metric(clean, "test", "accuracy"),
                    "masked_accuracy": metric(masked, "test", "accuracy"),
                    "accuracy_gap": metric(clean, "test", "accuracy") - metric(masked, "test", "accuracy"),
                    "clean_target_recall": metric(clean, "test", "target.recall"),
                    "masked_target_recall": metric(masked, "test", "target.recall"),
                    "target_recall_gap": metric(clean, "test", "target.recall") - metric(masked, "test", "target.recall"),
                    "masked_target_f1": metric(masked, "test", "target.f1"),
                    "masked_proxy_target_recall": metric(masked, "proxy_transfer", "target.recall"),
                }
            )
    return rows


def format_confusions(confusions: list[dict[str, Any]]) -> str:
    if not confusions:
        return "none"
    return "; ".join(f"{item['gold']} -> {item['pred']} ({item['count']})" for item in confusions[:5])


def write_gap_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "classifier",
        "method",
        "clean_accuracy",
        "masked_accuracy",
        "accuracy_gap",
        "clean_target_recall",
        "masked_target_recall",
        "target_recall_gap",
        "masked_target_f1",
        "masked_proxy_target_recall",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_supervised_markdown_report(
    path: Path,
    results: dict[str, dict[str, Any]],
    gap_rows: list[dict[str, Any]],
    chart_paths: dict[str, Path],
    *,
    target_author: str,
    masked_view: str,
) -> None:
    lines = [
        "# Supervised Author-Style Baseline Benchmark",
        "",
        f"Generated: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %z')}",
        "",
        "## Purpose",
        "",
        "This report tests stronger supervised classifiers on the same book-level splits and content-control views as the nearest-profile baseline.",
        f"The current masked view is `{masked_view}` and the target author is `{target_author}`.",
        "",
        "## Systems",
        "",
        "- `linear_svm`: class-balanced linear support vector classifier.",
        "- `sgd_logistic`: class-balanced linear logistic regression trained with stochastic gradient descent.",
        "- `complement_nb`: Complement Naive Bayes, a strong sparse text baseline.",
        "",
        "## Test Metrics By System",
        "",
        "| Classifier | Features | View | Features Used | Test Acc | Balanced Acc | Macro F1 | Target Precision | Target Recall | Target F1 | Proxy Target Recall | Book Test Acc | Majority Baseline |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, result in sorted(results.items()):
        spec = result["spec"]
        test_metrics = result["chunk_metrics"].get("test", {})
        target = test_metrics.get("target", {})
        lines.append(
            f"| {CLASSIFIER_LABELS.get(spec['classifier'], spec['classifier'])} | "
            f"{METHOD_LABELS.get(spec['method'], spec['method'])} | {spec['view']} | "
            f"{int(result.get('feature_count', 0)):,} | {fmt(test_metrics.get('accuracy', 0.0))} | "
            f"{fmt(test_metrics.get('balanced_accuracy', 0.0))} | {fmt(test_metrics.get('macro_f1', 0.0))} | "
            f"{fmt(target.get('precision', 0.0))} | {fmt(target.get('recall', 0.0))} | {fmt(target.get('f1', 0.0))} | "
            f"{fmt(metric(result, 'proxy_transfer', 'target.recall'))} | "
            f"{fmt(metric(result, 'test', 'accuracy', level='book'))} | "
            f"{fmt(test_metrics.get('majority_baseline_accuracy', 0.0))} |"
        )

    lines.extend(
        [
            "",
            f"## Clean vs {masked_view} Gap",
            "",
            "| Classifier | Features | Clean Acc | Masked Acc | Acc Gap | Clean Target Recall | Masked Target Recall | Target Recall Gap | Masked Target F1 | Masked Proxy Recall |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in gap_rows:
        lines.append(
            f"| {CLASSIFIER_LABELS.get(row['classifier'], row['classifier'])} | "
            f"{METHOD_LABELS.get(row['method'], row['method'])} | "
            f"{fmt(row['clean_accuracy'])} | {fmt(row['masked_accuracy'])} | {fmt(row['accuracy_gap'])} | "
            f"{fmt(row['clean_target_recall'])} | {fmt(row['masked_target_recall'])} | {fmt(row['target_recall_gap'])} | "
            f"{fmt(row['masked_target_f1'])} | {fmt(row['masked_proxy_target_recall'])} |"
        )

    masked_results = [result for result in results.values() if result["spec"]["view"] == masked_view]
    masked_results.sort(
        key=lambda result: (
            metric(result, "test", "balanced_accuracy"),
            metric(result, "test", "target.f1"),
            metric(result, "proxy_transfer", "target.recall"),
        ),
        reverse=True,
    )
    lines.extend(
        [
            "",
            "## Masked Ranking",
            "",
            "| Rank | System | Balanced Acc | Target F1 | Proxy Target Recall | Acc Gap | Target Recall Gap |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    gap_index = {(row["classifier"], row["method"]): row for row in gap_rows}
    for rank, result in enumerate(masked_results[:10], start=1):
        spec = result["spec"]
        row = gap_index.get((spec["classifier"], spec["method"]), {})
        lines.append(
            f"| {rank} | {CLASSIFIER_LABELS.get(spec['classifier'], spec['classifier'])} / "
            f"{METHOD_LABELS.get(spec['method'], spec['method'])} | "
            f"{fmt(metric(result, 'test', 'balanced_accuracy'))} | {fmt(metric(result, 'test', 'target.f1'))} | "
            f"{fmt(metric(result, 'proxy_transfer', 'target.recall'))} | "
            f"{fmt(float(row.get('accuracy_gap', math.nan)) if row else 0.0)} | "
            f"{fmt(float(row.get('target_recall_gap', math.nan)) if row else 0.0)} |"
        )

    lines.extend(["", "## Graphs", ""])
    for name, chart_path in chart_paths.items():
        rel = chart_path.relative_to(path.parent).as_posix()
        lines.append(f"![{html.escape(name)}]({rel})")
        lines.append("")

    lines.extend(["## Per-System Notes", ""])
    for row in gap_rows:
        clean_key = f"{row['classifier']}.{row['method']}.clean"
        masked_key = f"{row['classifier']}.{row['method']}.{masked_view}"
        clean = results.get(clean_key)
        masked = results.get(masked_key)
        if not clean or not masked:
            continue
        lines.extend(
            [
                f"### {CLASSIFIER_LABELS.get(row['classifier'], row['classifier'])} / {METHOD_LABELS.get(row['method'], row['method'])}",
                "",
                f"- Clean test accuracy: {fmt(row['clean_accuracy'])}; `{masked_view}` test accuracy: {fmt(row['masked_accuracy'])}; gap: {fmt(row['accuracy_gap'])}.",
                f"- Clean `{target_author}` recall: {fmt(row['clean_target_recall'])}; `{masked_view}` `{target_author}` recall: {fmt(row['masked_target_recall'])}; gap: {fmt(row['target_recall_gap'])}.",
                f"- Clean top test confusions: {format_confusions(clean['top_test_confusions'])}",
                f"- `{masked_view}` top test confusions: {format_confusions(masked['top_test_confusions'])}",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation Guardrails",
            "",
            "- Prefer masked balanced accuracy and target F1 over raw clean accuracy.",
            "- Treat character n-grams as leakage-sensitive even when they perform well.",
            "- A classifier is a style-meter candidate only if masked target performance remains useful and the clean-to-masked gap is small.",
            "- Final style-transfer evaluation still requires semantic fidelity and readability gates.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(
    output_dir: Path,
    results: dict[str, dict[str, Any]],
    *,
    classifiers: tuple[str, ...],
    methods: tuple[str, ...],
    views: tuple[str, ...],
    masked_view: str,
    target_author: str,
    dataset_root: Path,
    char_min_df: int,
    input_bindings: dict[str, Any],
) -> None:
    gap_rows = build_gap_rows(results, classifiers=classifiers, methods=methods, masked_view=masked_view)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "supervised_author_baseline_results.json").write_text(
        json.dumps(
            {
                "target_author": target_author,
                "dataset_root": str(dataset_root),
                "views": views,
                "masked_view": masked_view,
                "classifiers": classifiers,
                "methods": methods,
                "char_min_df": char_min_df,
                "input_bindings": input_bindings,
                "results": results,
                "masked_gap": gap_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_gap_csv(output_dir / "supervised_masked_gap.csv", gap_rows)

    charts_dir = output_dir / "charts"
    chart_paths = {
        "Masked Balanced Accuracy": charts_dir / "masked_balanced_accuracy.svg",
        "Masked Target F1": charts_dir / "masked_target_f1.svg",
        "Clean To Masked Accuracy Gap": charts_dir / "clean_masked_accuracy_gap.svg",
    }
    masked_results = [result for result in results.values() if result["spec"]["view"] == masked_view]
    write_svg_bar_chart(
        chart_paths["Masked Balanced Accuracy"],
        f"{masked_view} Balanced Accuracy",
        [(result_label(result), metric(result, "test", "balanced_accuracy")) for result in sorted(masked_results, key=result_label)],
    )
    write_svg_bar_chart(
        chart_paths["Masked Target F1"],
        f"{masked_view} {target_author} Target F1",
        [(result_label(result), metric(result, "test", "target.f1")) for result in sorted(masked_results, key=result_label)],
    )
    write_svg_bar_chart(
        chart_paths["Clean To Masked Accuracy Gap"],
        f"Clean To {masked_view} Accuracy Gap",
        [
            (f"{CLASSIFIER_LABELS.get(row['classifier'], row['classifier'])} / {METHOD_LABELS.get(row['method'], row['method'])}", row["accuracy_gap"])
            for row in gap_rows
        ],
    )
    write_supervised_markdown_report(
        output_dir / "supervised_author_baseline_results.md",
        results,
        gap_rows,
        chart_paths,
        target_author=target_author,
        masked_view=masked_view,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark supervised author-style classifiers on clean and masked chunks.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CURRENT_BENCHMARK_DIR.relative_to(RESEARCH_ROOT),
    )
    parser.add_argument("--target-author", default=TARGET_AUTHOR_DEFAULT)
    parser.add_argument(
        "--views",
        default="clean,train_global_masked,entity_masked_v3",
        help="Comma-separated views to benchmark.",
    )
    parser.add_argument("--masked-view", default="entity_masked_v3", help="Masked view to compare against clean.")
    parser.add_argument(
        "--methods",
        default="char_ngrams",
        help="Comma-separated feature families to benchmark.",
    )
    parser.add_argument("--classifiers", default="sgd_hinge,sgd_hinge_unbalanced")
    parser.add_argument("--max-char-features", type=int, default=80_000)
    parser.add_argument("--char-min-df", type=int, default=CURRENT_CHAR_MIN_DF)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--report-only", action="store_true", help="Regenerate Markdown/charts from completed JSON results.")
    args = parser.parse_args()

    views = parse_csv_arg(args.views)
    methods = parse_csv_arg(args.methods)
    classifiers = parse_csv_arg(args.classifiers)
    paths = chunk_paths(args.dataset_root)

    missing_views = [view for view in views if view not in paths or not paths[view].exists()]
    if missing_views:
        raise SystemExit("missing chunk files for views: " + ", ".join(missing_views))
    matrix_methods = {"char_ngrams", "char_hashing"}
    unknown_methods = [method for method in methods if method not in FEATURE_FNS and method not in matrix_methods]
    if unknown_methods:
        raise SystemExit("unknown methods: " + ", ".join(unknown_methods))
    unknown_classifiers = [classifier for classifier in classifiers if classifier not in CLASSIFIER_LABELS]
    if unknown_classifiers:
        raise SystemExit("unknown classifiers: " + ", ".join(unknown_classifiers))

    result_path = args.output_dir / "supervised_author_baseline_results.json"
    if args.report_only:
        if not result_path.exists():
            raise SystemExit(f"missing result file for --report-only: {result_path}")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        write_outputs(
            args.output_dir,
            payload["results"],
            classifiers=tuple(payload.get("classifiers", classifiers)),
            methods=tuple(payload.get("methods", methods)),
            views=tuple(payload.get("views", views)),
            masked_view=str(payload.get("masked_view", args.masked_view)),
            target_author=str(payload.get("target_author", args.target_author)),
            dataset_root=Path(str(payload.get("dataset_root", args.dataset_root))),
            char_min_df=int(payload.get("char_min_df", args.char_min_df)),
            input_bindings=dict(payload.get("input_bindings", {})),
        )
        print(json.dumps({"report": str(args.output_dir / "supervised_author_baseline_results.md"), "report_only": True}, indent=2))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_by_view = {view: load_records(paths[view]) for view in views}
    validate_view_alignment(records_by_view)
    mask_plan_path = args.dataset_root / "masked/mask_terms.json"
    mask_plan = validate_mask_plan(mask_plan_path, records_by_view[views[0]])
    input_bindings = {
        "punctuation_normalization": PUNCTUATION_NORMALIZATION_VERSION,
        "cross_book_decontamination": CROSS_BOOK_DECONTAMINATION_VERSION,
        "masking_policy": MASKING_POLICY_VERSION,
        "mask_plan": {
            "path": str(mask_plan_path),
            "sha256": file_sha256(mask_plan_path),
            "fit_book_count": int(mask_plan["provenance"]["fit_book_count"]),
            "fit_book_ids_sha256": str(
                mask_plan["provenance"]["fit_book_ids_sha256"]
            ),
        },
        "benchmark_source": {
            "path": str(Path(__file__).resolve().relative_to(RESEARCH_ROOT)),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
        "chunk_files": {
            view: {
                "path": str(paths[view]),
                "sha256": file_sha256(paths[view]),
                "rows": len(records_by_view[view]),
            }
            for view in views
        },
    }
    matrix_cache: dict[tuple[str, str], MatrixBundle] = {}
    results: dict[str, dict[str, Any]] = {}
    for view in views:
        for method in methods:
            print(f"building features {method}.{view}", flush=True)
            matrix_cache[(method, view)] = fit_feature_matrix(
                records_by_view[view],
                method=method,
                max_char_features=args.max_char_features,
                char_min_df=args.char_min_df,
            )
            for classifier in classifiers:
                spec = RunSpec(key=f"{classifier}.{method}.{view}", classifier=classifier, method=method, view=view)
                print(f"running {spec.key}", flush=True)
                result = evaluate_supervised_run(
                    matrix_cache[(method, view)],
                    spec,
                    target_author=args.target_author,
                    random_state=args.random_state,
                    max_iter=args.max_iter,
                    n_jobs=args.jobs,
                )
                results[spec.key] = result
                write_confusion_csv(
                    args.output_dir / "confusion_matrices" / f"{safe_filename(spec.key)}.test.csv",
                    result["labels"],
                    result["test_confusion_matrix"],
                )
                (args.output_dir / "supervised_author_baseline_results.partial.json").write_text(
                    json.dumps(
                        {
                            "target_author": args.target_author,
                            "dataset_root": str(args.dataset_root),
                            "views": views,
                            "masked_view": args.masked_view,
                            "classifiers": classifiers,
                            "methods": methods,
                            "char_min_df": args.char_min_df,
                            "input_bindings": input_bindings,
                            "completed_runs": sorted(results),
                            "results": results,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )

    write_outputs(
        args.output_dir,
        results,
        classifiers=classifiers,
        methods=methods,
        views=views,
        masked_view=args.masked_view,
        target_author=args.target_author,
        dataset_root=args.dataset_root,
        char_min_df=args.char_min_df,
        input_bindings=input_bindings,
    )
    print(
        json.dumps(
            {
                "report": str(args.output_dir / "supervised_author_baseline_results.md"),
                "results": str(result_path),
                "runs": len(results),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
