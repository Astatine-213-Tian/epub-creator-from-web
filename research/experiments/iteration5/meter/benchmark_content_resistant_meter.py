#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from workflows.benchmark_author_style import FUNCTION_CHARS, FUNCTION_WORDS


TARGET_AUTHOR = "非天夜翔"
SEED = 20260713
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
PLACEHOLDER_RE = re.compile(r"<(?:NUM|LATIN|TERM)>|某+")
SENTENCE_RE = re.compile(r"[^。！？!?]+[。！？!?]?")
PUNCTUATION = "，。！？；：、“”‘’（）《》【】…—,.!?;:\"'()[]"
DIALOGUE_OPEN = ("“", "‘", "「", "『", '"')
SPEECH_WORDS = (
    "说道", "问道", "答道", "笑道", "喝道", "叫道", "喊道", "骂道", "叹道",
    "低声道", "沉声道", "冷冷道", "心想", "暗想", "只得", "继而", "旋即",
    "随即", "片刻后", "半晌", "良久", "忽然", "突然", "然而", "于是",
)
SINGLE_SPEECH = frozenset("说道问答笑喝喊骂叹想")
STYLE_WORDS = tuple(
    sorted(
        {word for word in (*FUNCTION_WORDS, *SPEECH_WORDS) if len(word) >= 2},
        key=lambda value: (-len(value), value),
    )
)
STYLE_WORD_RE = re.compile("|".join(re.escape(word) for word in STYLE_WORDS))
TOPIC_TERMS = (
    "小悦", "魔法", "克里", "杜景", "大陆", "魔法师", "骑士", "吕布", "曹天裁",
    "阿加斯", "佣兵",
)


@dataclass(frozen=True)
class Record:
    chunk_id: str
    split: str
    author: str
    title: str
    chunk_index: int
    text: str


@dataclass
class FeatureBundle:
    name: str
    train: sparse.csr_matrix
    dev: sparse.csr_matrix
    test: sparse.csr_matrix
    proxy: sparse.csr_matrix
    feature_names: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark content-resistant author-style meters for Iteration 5."
    )
    parser.add_argument(
        "--masked-dataset",
        type=Path,
        default=Path("datasets/masked/chunks.entity_masked_v3.jsonl"),
    )
    parser.add_argument(
        "--clean-dataset",
        type=Path,
        default=Path("datasets/unmasked/chunks.clean.jsonl"),
    )
    parser.add_argument(
        "--iteration4-root",
        type=Path,
        default=Path(
            "generated/style_research/style_transfer_experiments/iterations/"
            "full_regeneration_v1"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "generated/style_research/style_transfer_experiments/iterations/"
            "content_resistant_v1/meter_benchmark"
        ),
    )
    parser.add_argument("--target-author", default=TARGET_AUTHOR)
    parser.add_argument("--min-df", type=int, default=20)
    parser.add_argument("--min-target-books", type=int, default=8)
    parser.add_argument("--min-comparison-authors", type=int, default=15)
    parser.add_argument("--exclude-boundary-chunks", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=60000)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc


def load_records(path: Path, *, boundary: int) -> list[Record]:
    grouped: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for item in read_jsonl(path):
        record = Record(
            chunk_id=str(item["chunk_id"]),
            split=str(item["split"]),
            author=str(item["author"]),
            title=str(item["title"]),
            chunk_index=int(item["chunk_index"]),
            text=str(item["text"]),
        )
        grouped[(record.author, record.title)].append(record)

    retained: list[Record] = []
    for records in grouped.values():
        records.sort(key=lambda item: item.chunk_index)
        if len(records) <= boundary * 2:
            continue
        retained.extend(records[boundary : len(records) - boundary])
    retained.sort(key=lambda item: item.chunk_id)
    return retained


def cjk_len(text: str) -> int:
    return len(CJK_RE.findall(text))


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def matched_style_word(text: str, index: int) -> str | None:
    match = STYLE_WORD_RE.match(text, index)
    return match.group(0) if match else None


def structuralize(text: str) -> str:
    """Erase lexical content while retaining low-content style operators."""

    result: list[str] = []
    index = 0
    while index < len(text):
        placeholder = PLACEHOLDER_RE.match(text, index)
        if placeholder:
            token = "C"
            index = placeholder.end()
        else:
            char = text[index]
            if char == "\n":
                token = "¶"
                index += 1
            elif char in PUNCTUATION:
                token = char
                index += 1
            else:
                word = matched_style_word(text, index)
                if word is not None:
                    token = word
                    index += len(word)
                elif CJK_RE.fullmatch(char):
                    previous = text[index - 1] if index else ""
                    following = text[index + 1] if index + 1 < len(text) else ""
                    if char in SINGLE_SPEECH and (
                        previous in PUNCTUATION or following in PUNCTUATION
                    ):
                        token = char
                    else:
                        token = "C"
                    index += 1
                elif char.isalnum():
                    token = "C"
                    index += 1
                else:
                    index += 1
                    continue

        if token in {"C", "¶"} and result and result[-1] == token:
            continue
        result.append(token)
    return "".join(result)


def content_counterfactual(text: str) -> str:
    """Replace lexical surfaces while retaining registered style operators."""

    result: list[str] = []
    index = 0
    while index < len(text):
        placeholder = PLACEHOLDER_RE.match(text, index)
        if placeholder:
            result.extend("龘" * len(CJK_RE.findall(placeholder.group(0))))
            index = placeholder.end()
            continue
        char = text[index]
        word = matched_style_word(text, index)
        if word is not None:
            result.append(word)
            index += len(word)
        elif CJK_RE.fullmatch(char):
            previous = text[index - 1] if index else ""
            following = text[index + 1] if index + 1 < len(text) else ""
            if char in FUNCTION_CHARS:
                result.append(char)
            elif char in SINGLE_SPEECH and (
                previous in PUNCTUATION or following in PUNCTUATION
            ):
                result.append(char)
            else:
                result.append("龘")
            index += 1
        elif char.isalnum():
            result.append("X")
            index += 1
        else:
            result.append(char)
            index += 1
    return "".join(result)


def topic_swap(text: str) -> str:
    for term in TOPIC_TERMS:
        text = text.replace(term, "龘" * len(term))
    return text


def style_ablation(text: str) -> str:
    text = STYLE_WORD_RE.sub("", text)
    text = "".join(char for char in text if char not in PUNCTUATION)
    return "".join("龘" if char in FUNCTION_CHARS else char for char in text)


def continuous_features(text: str) -> dict[str, float]:
    total_cjk = max(cjk_len(text), 1)
    paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
    sentences = [item.group(0) for item in SENTENCE_RE.finditer(text) if cjk_len(item.group(0))]
    sentence_lengths = [cjk_len(item) for item in sentences]
    paragraph_lengths = [cjk_len(item) for item in paragraphs]

    features: dict[str, float] = {}
    for char in PUNCTUATION:
        features[f"punc_rate:{char}"] = text.count(char) * 100.0 / total_cjk
    for word in STYLE_WORDS:
        features[f"style_word_rate:{word}"] = text.count(word) * 1000.0 / total_cjk
    for char in FUNCTION_CHARS:
        features[f"function_char_rate:{char}"] = text.count(char) * 1000.0 / total_cjk

    dialogue_lines = sum(1 for line in paragraphs if line.startswith(DIALOGUE_OPEN))
    speech_tags = sum(text.count(word) for word in SPEECH_WORDS)
    features.update(
        {
            "dialogue_line_share": dialogue_lines / max(len(paragraphs), 1),
            "speech_tags_per_1000": speech_tags * 1000.0 / total_cjk,
            "sentences_per_paragraph": len(sentences) / max(len(paragraphs), 1),
            "commas_per_sentence": (text.count("，") + text.count(","))
            / max(len(sentences), 1),
            "mean_sentence_cjk": float(np.mean(sentence_lengths)) if sentence_lengths else 0.0,
            "std_sentence_cjk": float(np.std(sentence_lengths)) if sentence_lengths else 0.0,
            "p25_sentence_cjk": percentile(sentence_lengths, 0.25),
            "p50_sentence_cjk": percentile(sentence_lengths, 0.50),
            "p75_sentence_cjk": percentile(sentence_lengths, 0.75),
            "p90_sentence_cjk": percentile(sentence_lengths, 0.90),
            "short_sentence_share": sum(value <= 12 for value in sentence_lengths)
            / max(len(sentence_lengths), 1),
            "long_sentence_share": sum(value >= 40 for value in sentence_lengths)
            / max(len(sentence_lengths), 1),
            "mean_paragraph_cjk": float(np.mean(paragraph_lengths)) if paragraph_lengths else 0.0,
            "std_paragraph_cjk": float(np.std(paragraph_lengths)) if paragraph_lengths else 0.0,
            "p50_paragraph_cjk": percentile(paragraph_lengths, 0.50),
            "p90_paragraph_cjk": percentile(paragraph_lengths, 0.90),
        }
    )
    return features


def group_by_split(records: list[Record]) -> dict[str, list[Record]]:
    grouped: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        grouped[record.split].append(record)
    return grouped


def dispersion_mask(
    matrix: sparse.csr_matrix,
    records: list[Record],
    *,
    target_author: str,
    min_target_books: int,
    min_comparison_authors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_books: list[set[str]] = [set() for _ in range(matrix.shape[1])]
    comparison_authors: list[set[str]] = [set() for _ in range(matrix.shape[1])]
    for row_index, record in enumerate(records):
        feature_indices = matrix.indices[matrix.indptr[row_index] : matrix.indptr[row_index + 1]]
        if record.author == target_author:
            for feature_index in feature_indices:
                target_books[int(feature_index)].add(record.title)
        else:
            for feature_index in feature_indices:
                comparison_authors[int(feature_index)].add(record.author)
    target_counts = np.asarray([len(value) for value in target_books], dtype=np.int32)
    comparison_counts = np.asarray([len(value) for value in comparison_authors], dtype=np.int32)
    keep = (target_counts >= min_target_books) | (
        comparison_counts >= min_comparison_authors
    )
    return keep, target_counts, comparison_counts


def build_feature_bundles(
    records_by_split: dict[str, list[Record]],
    *,
    target_author: str,
    min_df: int,
    min_target_books: int,
    min_comparison_authors: int,
    max_features: int,
) -> tuple[dict[str, FeatureBundle], dict[str, Any]]:
    train = records_by_split["train"]
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
        norm="l2",
        lowercase=False,
        dtype=np.float32,
        preprocessor=structuralize,
        token_pattern=None,
    )
    skeleton_matrices: dict[str, sparse.csr_matrix] = {
        "train": vectorizer.fit_transform(record.text for record in train).tocsr()
    }
    for split in ("dev", "test", "proxy_transfer"):
        skeleton_matrices[split] = vectorizer.transform(
            record.text for record in records_by_split.get(split, [])
        ).tocsr()

    keep, target_books, comparison_authors = dispersion_mask(
        skeleton_matrices["train"],
        train,
        target_author=target_author,
        min_target_books=min_target_books,
        min_comparison_authors=min_comparison_authors,
    )
    skeleton_names = np.asarray(vectorizer.get_feature_names_out())[keep].tolist()
    for split in skeleton_matrices:
        skeleton_matrices[split] = skeleton_matrices[split][:, keep].tocsr()

    dict_vectorizer = DictVectorizer(sparse=True, sort=True)
    continuous_matrices: dict[str, sparse.csr_matrix] = {
        "train": dict_vectorizer.fit_transform(
            continuous_features(record.text) for record in train
        ).tocsr()
    }
    for split in ("dev", "test", "proxy_transfer"):
        continuous_matrices[split] = dict_vectorizer.transform(
            continuous_features(record.text)
            for record in records_by_split.get(split, [])
        ).tocsr()
    scaler = StandardScaler(with_mean=False)
    continuous_matrices["train"] = scaler.fit_transform(
        continuous_matrices["train"]
    ).tocsr()
    for split in ("dev", "test", "proxy_transfer"):
        continuous_matrices[split] = scaler.transform(
            continuous_matrices[split]
        ).tocsr()

    continuous_names = dict_vectorizer.get_feature_names_out().tolist()
    bundles = {
        "continuous_only": FeatureBundle(
            name="continuous_only",
            train=continuous_matrices["train"],
            dev=continuous_matrices["dev"],
            test=continuous_matrices["test"],
            proxy=continuous_matrices["proxy_transfer"],
            feature_names=continuous_names,
        ),
        "structural_ngrams": FeatureBundle(
            name="structural_ngrams",
            train=skeleton_matrices["train"],
            dev=skeleton_matrices["dev"],
            test=skeleton_matrices["test"],
            proxy=skeleton_matrices["proxy_transfer"],
            feature_names=skeleton_names,
        ),
        "combined_content_resistant": FeatureBundle(
            name="combined_content_resistant",
            train=sparse.hstack(
                [skeleton_matrices["train"], continuous_matrices["train"]],
                format="csr",
            ),
            dev=sparse.hstack(
                [skeleton_matrices["dev"], continuous_matrices["dev"]],
                format="csr",
            ),
            test=sparse.hstack(
                [skeleton_matrices["test"], continuous_matrices["test"]],
                format="csr",
            ),
            proxy=sparse.hstack(
                [skeleton_matrices["proxy_transfer"], continuous_matrices["proxy_transfer"]],
                format="csr",
            ),
            feature_names=[f"skeleton:{name}" for name in skeleton_names]
            + [f"continuous:{name}" for name in continuous_names],
        ),
    }
    artifacts = {
        "skeleton_vectorizer": vectorizer,
        "skeleton_keep_mask": keep,
        "skeleton_target_book_support": target_books,
        "skeleton_comparison_author_support": comparison_authors,
        "continuous_vectorizer": dict_vectorizer,
        "continuous_scaler": scaler,
    }
    return bundles, artifacts


def author_book_weights(records: list[Record], *, binary_target: str | None) -> np.ndarray:
    author_books: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, record in enumerate(records):
        author_books[record.author][record.title].append(index)
    weights = np.zeros(len(records), dtype=np.float64)
    if binary_target is None:
        author_mass = 1.0 / len(author_books)
        for books in author_books.values():
            book_mass = author_mass / len(books)
            for indices in books.values():
                for index in indices:
                    weights[index] = book_mass / len(indices)
        return weights * len(records)

    target_books = author_books[binary_target]
    for indices in target_books.values():
        for index in indices:
            weights[index] = 0.5 / len(target_books) / len(indices)
    comparison = {author: books for author, books in author_books.items() if author != binary_target}
    for books in comparison.values():
        for indices in books.values():
            for index in indices:
                weights[index] = 0.5 / len(comparison) / len(books) / len(indices)
    return weights * len(records)


def select_threshold(scores: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    thresholds = np.unique(scores)
    best: tuple[float, float, float, float] | None = None
    for threshold in thresholds:
        predicted = scores >= threshold
        sensitivity = recall_score(truth, predicted, zero_division=0)
        negatives = truth == 0
        false_positive_rate = float(np.mean(predicted[negatives])) if negatives.any() else 0.0
        balanced = balanced_accuracy_score(truth, predicted)
        eligible = sensitivity >= 0.80 and false_positive_rate <= 0.10
        rank = (1.0 if eligible else 0.0, balanced, -false_positive_rate, threshold)
        if best is None or rank > best:
            best = rank
    if best is None:
        raise ValueError("cannot select threshold from empty scores")
    threshold = best[3]
    predicted = scores >= threshold
    negatives = truth == 0
    return {
        "threshold": float(threshold),
        "sensitivity": float(recall_score(truth, predicted, zero_division=0)),
        "false_positive_rate": float(np.mean(predicted[negatives])) if negatives.any() else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "eligible": bool(best[0]),
    }


def binary_metrics(truth: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = scores >= threshold
    negatives = truth == 0
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "target_precision": float(precision_score(truth, predicted, zero_division=0)),
        "target_recall": float(recall_score(truth, predicted, zero_division=0)),
        "target_f1": float(f1_score(truth, predicted, zero_division=0)),
        "false_positive_rate": float(np.mean(predicted[negatives])) if negatives.any() else 0.0,
        "roc_auc": float(roc_auc_score(truth, scores)),
    }


def multiclass_metrics(truth: np.ndarray, predicted: np.ndarray, target_author: str) -> dict[str, float]:
    target_truth = truth == target_author
    target_predicted = predicted == target_author
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "target_precision": float(precision_score(target_truth, target_predicted, zero_division=0)),
        "target_recall": float(recall_score(target_truth, target_predicted, zero_division=0)),
        "target_f1": float(f1_score(target_truth, target_predicted, zero_division=0)),
    }


def transform_arbitrary(
    texts: list[str],
    *,
    bundle_name: str,
    artifacts: dict[str, Any],
) -> sparse.csr_matrix:
    skeleton = artifacts["skeleton_vectorizer"].transform(texts).tocsr()
    skeleton = skeleton[:, artifacts["skeleton_keep_mask"]].tocsr()
    continuous = artifacts["continuous_vectorizer"].transform(
        continuous_features(text) for text in texts
    ).tocsr()
    continuous = artifacts["continuous_scaler"].transform(continuous).tocsr()
    if bundle_name == "structural_ngrams":
        return skeleton
    if bundle_name == "continuous_only":
        return continuous
    return sparse.hstack([skeleton, continuous], format="csr")


def load_legacy_pairs(root: Path) -> tuple[list[str], list[str], list[str]]:
    selection = json.loads(
        (root / "sample_sets/iteration4_proxy_v1.calibration_v1_ids.json").read_text(
            encoding="utf-8"
        )
    )
    sample_ids = [str(value) for value in selection["sample_ids"]]
    wanted = set(sample_ids)
    originals = {
        str(item["sample_id"]): str(item["original_zh"])
        for item in read_jsonl(root / "sample_sets/iteration4_proxy_v1.hidden_targets.jsonl")
        if str(item["sample_id"]) in wanted
    }
    neutral_dir = (
        root
        / "runs/iteration4_proxy_v1/iteration4_source_gpt54_official/neutral_translation"
    )
    neutral: dict[str, str] = {}
    for sample_id in sample_ids:
        payload = json.loads((neutral_dir / f"{sample_id}.json").read_text(encoding="utf-8"))
        neutral[sample_id] = "\n".join(
            str(item["zh"]) for item in payload["result"]["paragraphs"]
        )
    return sample_ids, [originals[value] for value in sample_ids], [neutral[value] for value in sample_ids]


def target_book_rows(
    records: list[Record], scores: np.ndarray, threshold: float, target_author: str
) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if record.author == target_author:
            grouped[record.title].append(index)
    rows: list[dict[str, Any]] = []
    for title, indices in sorted(grouped.items()):
        values = scores[indices]
        rows.append(
            {
                "book": title,
                "rows": len(indices),
                "recall": float(np.mean(values >= threshold)),
                "mean_margin": float(np.mean(values)),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_bar_chart(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = [str(row["method"]) for row in rows]
    values = [float(row["test_balanced_accuracy"]) * 100.0 for row in rows]
    width, height = 920, 360
    left, right, top, bottom = 90, 24, 36, 120
    plot_w, plot_h = width - left - right, height - top - bottom
    bar_w = plot_w / max(len(values), 1) * 0.58
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="24" font-family="Arial" font-size="16" fill="#1f2937">Book-disjoint binary target-style balanced accuracy</text>',
    ]
    for tick in range(0, 101, 20):
        y = top + plot_h - tick / 100 * plot_h
        pieces.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        pieces.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#4b5563">{tick}%</text>')
    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        center = left + (index + 0.5) * plot_w / len(values)
        x = center - bar_w / 2
        y = top + plot_h - value / 100 * plot_h
        color = "#0f766e" if value >= 80 else "#b45309"
        pieces.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{top + plot_h - y:.1f}" fill="{color}"/>')
        pieces.append(f'<text x="{center:.1f}" y="{y - 7:.1f}" text-anchor="middle" font-family="Arial" font-size="12" fill="#111827">{value:.1f}%</text>')
        pieces.append(f'<text x="{center:.1f}" y="{top + plot_h + 18}" text-anchor="end" transform="rotate(-28 {center:.1f} {top + plot_h + 18})" font-family="Arial" font-size="11" fill="#374151">{html.escape(label)}</text>')
    pieces.append("</svg>")
    path.write_text("\n".join(pieces), encoding="utf-8")


def main() -> None:
    args = parse_args()
    records = load_records(args.masked_dataset, boundary=args.exclude_boundary_chunks)
    records_by_split = group_by_split(records)
    required = {"train", "dev", "test", "proxy_transfer"}
    missing = required - records_by_split.keys()
    if missing:
        raise ValueError(f"missing required splits: {sorted(missing)}")

    bundles, artifacts = build_feature_bundles(
        records_by_split,
        target_author=args.target_author,
        min_df=args.min_df,
        min_target_books=args.min_target_books,
        min_comparison_authors=args.min_comparison_authors,
        max_features=args.max_features,
    )
    train_records = records_by_split["train"]
    dev_records = records_by_split["dev"]
    test_records = records_by_split["test"]
    proxy_records = records_by_split["proxy_transfer"]
    binary_train = np.asarray(
        [record.author == args.target_author for record in train_records], dtype=np.int8
    )
    binary_dev = np.asarray(
        [record.author == args.target_author for record in dev_records], dtype=np.int8
    )
    binary_test = np.asarray(
        [record.author == args.target_author for record in test_records], dtype=np.int8
    )
    binary_proxy = np.asarray(
        [record.author == args.target_author for record in proxy_records], dtype=np.int8
    )
    author_train = np.asarray([record.author for record in train_records])
    author_dev = np.asarray([record.author for record in dev_records])
    author_test = np.asarray([record.author for record in test_records])
    author_proxy = np.asarray([record.author for record in proxy_records])

    binary_weights = author_book_weights(train_records, binary_target=args.target_author)
    multiclass_weights = author_book_weights(train_records, binary_target=None)
    _pair_ids, pair_originals, pair_neutral = load_legacy_pairs(args.iteration4_root)

    clean_records = load_records(args.clean_dataset, boundary=args.exclude_boundary_chunks)
    clean_by_id = {record.chunk_id: record.text for record in clean_records}
    aligned_test_indices = [
        index for index, record in enumerate(test_records) if record.chunk_id in clean_by_id
    ]

    all_results: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    models: dict[str, tuple[LinearSVC, LinearSVC]] = {}
    for name, bundle in bundles.items():
        binary_model = LinearSVC(C=0.5, dual="auto", max_iter=20000, random_state=SEED)
        binary_model.fit(bundle.train, binary_train, sample_weight=binary_weights)
        dev_scores = binary_model.decision_function(bundle.dev)
        threshold = select_threshold(dev_scores, binary_dev)
        split_scores = {
            "dev": dev_scores,
            "test": binary_model.decision_function(bundle.test),
            "proxy_transfer": binary_model.decision_function(bundle.proxy),
        }
        binary_result = {
            split: binary_metrics(
                truth,
                split_scores[split],
                float(threshold["threshold"]),
            )
            for split, truth in (
                ("dev", binary_dev),
                ("test", binary_test),
                ("proxy_transfer", binary_proxy),
            )
        }

        multiclass_model = LinearSVC(C=0.5, dual="auto", max_iter=20000, random_state=SEED)
        multiclass_model.fit(bundle.train, author_train, sample_weight=multiclass_weights)
        multiclass_result = {
            split: multiclass_metrics(truth, multiclass_model.predict(matrix), args.target_author)
            for split, truth, matrix in (
                ("dev", author_dev, bundle.dev),
                ("test", author_test, bundle.test),
                ("proxy_transfer", author_proxy, bundle.proxy),
            )
        }

        original_matrix = transform_arbitrary(
            pair_originals, bundle_name=name, artifacts=artifacts
        )
        neutral_matrix = transform_arbitrary(
            pair_neutral, bundle_name=name, artifacts=artifacts
        )
        original_scores = binary_model.decision_function(original_matrix)
        neutral_scores = binary_model.decision_function(neutral_matrix)
        paired = {
            "rows": len(pair_originals),
            "original_above_dev_threshold": float(
                np.mean(original_scores >= threshold["threshold"])
            ),
            "neutral_above_dev_threshold": float(
                np.mean(neutral_scores >= threshold["threshold"])
            ),
            "original_preferred": float(np.mean(original_scores > neutral_scores)),
            "mean_original_minus_neutral": float(np.mean(original_scores - neutral_scores)),
        }

        masked_texts = [test_records[index].text for index in aligned_test_indices]
        clean_texts = [clean_by_id[test_records[index].chunk_id] for index in aligned_test_indices]
        clean_scores = binary_model.decision_function(
            transform_arbitrary(clean_texts, bundle_name=name, artifacts=artifacts)
        )
        masked_scores = split_scores["test"][aligned_test_indices]
        score_scale = float(np.std(masked_scores)) or 1.0
        counterfactual_texts = [content_counterfactual(text) for text in masked_texts]
        counterfactual_scores = binary_model.decision_function(
            transform_arbitrary(counterfactual_texts, bundle_name=name, artifacts=artifacts)
        )
        topic_scores = binary_model.decision_function(
            transform_arbitrary(
                [topic_swap(text) for text in masked_texts],
                bundle_name=name,
                artifacts=artifacts,
            )
        )
        target_test_indices = [
            index for index, record in enumerate(test_records) if record.author == args.target_author
        ]
        target_texts = [test_records[index].text for index in target_test_indices]
        ablated_scores = binary_model.decision_function(
            transform_arbitrary(
                [style_ablation(text) for text in target_texts],
                bundle_name=name,
                artifacts=artifacts,
            )
        )
        target_scores = split_scores["test"][target_test_indices]
        invariance = {
            "aligned_clean_masked_rows": len(aligned_test_indices),
            "clean_masked_mean_abs_delta": float(np.mean(np.abs(clean_scores - masked_scores))),
            "clean_masked_normalized_mean_abs_delta": float(
                np.mean(np.abs(clean_scores - masked_scores)) / score_scale
            ),
            "clean_masked_threshold_flip_rate": float(
                np.mean(
                    (clean_scores >= threshold["threshold"])
                    != (masked_scores >= threshold["threshold"])
                )
            ),
            "content_counterfactual_normalized_mean_abs_delta": float(
                np.mean(np.abs(counterfactual_scores - masked_scores)) / score_scale
            ),
            "topic_swap_normalized_mean_abs_delta": float(
                np.mean(np.abs(topic_scores - masked_scores)) / score_scale
            ),
            "style_ablation_mean_margin_drop_target_test": float(
                np.mean(target_scores - ablated_scores)
            ),
        }

        all_results[name] = {
            "feature_count": bundle.train.shape[1],
            "threshold_selection": threshold,
            "binary": binary_result,
            "multiclass": multiclass_result,
            "legacy_generated_pair_diagnostic": paired,
            "content_invariance": invariance,
            "target_test_by_book": target_book_rows(
                test_records,
                split_scores["test"],
                float(threshold["threshold"]),
                args.target_author,
            ),
        }
        summary_rows.append(
            {
                "method": name,
                "features": bundle.train.shape[1],
                "dev_balanced_accuracy": binary_result["dev"]["balanced_accuracy"],
                "dev_target_recall": binary_result["dev"]["target_recall"],
                "dev_false_positive_rate": binary_result["dev"]["false_positive_rate"],
                "test_balanced_accuracy": binary_result["test"]["balanced_accuracy"],
                "test_target_recall": binary_result["test"]["target_recall"],
                "test_false_positive_rate": binary_result["test"]["false_positive_rate"],
                "test_target_f1": binary_result["test"]["target_f1"],
                "multiclass_test_accuracy": multiclass_result["test"]["accuracy"],
                "paired_original_preferred": paired["original_preferred"],
                "paired_mean_gap": paired["mean_original_minus_neutral"],
                "clean_masked_normalized_delta": invariance[
                    "clean_masked_normalized_mean_abs_delta"
                ],
                "counterfactual_normalized_delta": invariance[
                    "content_counterfactual_normalized_mean_abs_delta"
                ],
                "style_ablation_margin_drop": invariance[
                    "style_ablation_mean_margin_drop_target_test"
                ],
            }
        )
        models[name] = (binary_model, multiclass_model)

    eligible = [
        row
        for row in summary_rows
        if row["dev_target_recall"] >= 0.80
        and row["dev_false_positive_rate"] <= 0.10
        and row["clean_masked_normalized_delta"] <= 0.10
        and row["counterfactual_normalized_delta"] <= 0.10
    ]
    candidates = eligible or summary_rows
    selected_row = max(
        candidates,
        key=lambda row: (
            row["paired_original_preferred"],
            row["dev_balanced_accuracy"],
            row["style_ablation_margin_drop"],
            -row["clean_masked_normalized_delta"],
        ),
    )
    selected = str(selected_row["method"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "method_summary.csv", summary_rows)
    for name, result in all_results.items():
        write_csv(args.output_dir / f"target_test_by_book.{name}.csv", result["target_test_by_book"])
    write_bar_chart(args.output_dir / "binary_balanced_accuracy.svg", summary_rows)

    selected_bundle = bundles[selected]
    binary_model, multiclass_model = models[selected]
    top_indices = np.argsort(binary_model.coef_[0])[::-1][:200]
    top_rows = [
        {
            "rank": rank,
            "feature": selected_bundle.feature_names[int(index)],
            "weight": float(binary_model.coef_[0, int(index)]),
        }
        for rank, index in enumerate(top_indices, start=1)
    ]
    write_csv(args.output_dir / "selected_top_positive_features.csv", top_rows)

    model_dir = args.output_dir / "selected_meter"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(binary_model, model_dir / "binary_model.joblib")
    joblib.dump(multiclass_model, model_dir / "multiclass_model.joblib")
    joblib.dump(artifacts["skeleton_vectorizer"], model_dir / "skeleton_vectorizer.joblib")
    joblib.dump(artifacts["skeleton_keep_mask"], model_dir / "skeleton_keep_mask.joblib")
    joblib.dump(artifacts["continuous_vectorizer"], model_dir / "continuous_vectorizer.joblib")
    joblib.dump(artifacts["continuous_scaler"], model_dir / "continuous_scaler.joblib")

    result_payload = {
        "schema_version": 1,
        "status": "complete",
        "target_author": args.target_author,
        "seed": SEED,
        "dataset": str(args.masked_dataset),
        "clean_dataset": str(args.clean_dataset),
        "boundary_chunks_excluded": args.exclude_boundary_chunks,
        "records_by_split": {key: len(value) for key, value in records_by_split.items()},
        "selection_rule": {
            "eligibility": {
                "dev_target_recall_min": 0.80,
                "dev_false_positive_rate_max": 0.10,
                "clean_masked_normalized_delta_max": 0.10,
                "content_counterfactual_normalized_delta_max": 0.10,
            },
            "ranking": [
                "legacy_paired_original_preference",
                "dev_balanced_accuracy",
                "style_ablation_sensitivity",
                "clean_masked_invariance",
            ],
            "note": "Legacy generated pairs are method-development diagnostics, not the Iteration 5 efficacy cohort.",
        },
        "selected_method": selected,
        "methods": all_results,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (model_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "meter_id": "content_resistant_target_style.v1",
                "selected_method": selected,
                "target_author": args.target_author,
                "feature_count": selected_bundle.train.shape[1],
                "dev_threshold": all_results[selected]["threshold_selection"],
                "research_role": "candidate_primary_meter_pending_independent_audit_and_fresh_generated_domain_calibration",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Iteration 5 Content-Resistant Meter Benchmark",
        "",
        f"- Selected candidate: **`{selected}`**",
        f"- Target author: `{args.target_author}`",
        f"- Boundary chunks excluded per book: {args.exclude_boundary_chunks}",
        "- Primary classification task: target author versus all comparison authors",
        "- Secondary task: 50-author attribution (diagnostic only)",
        "",
        "The representation erases lexical content before character n-gram extraction,",
        "retaining only registered function/discourse words, speech operators,",
        "punctuation, paragraph boundaries, and continuous flow features. Feature",
        "columns must recur across at least the configured number of target books or",
        "comparison authors. The old exact lexical n-gram model is not used for",
        "selection.",
        "",
        "![Binary balanced accuracy](binary_balanced_accuracy.svg)",
        "",
        "## Overall Results",
        "",
        "| Method | Features | Dev bal acc | Dev recall | Dev FPR | Test bal acc | Test recall | Test FPR | Test target F1 | 50-author test acc | Original > neutral | Clean/masked delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {method} | {features} | {dev_balanced_accuracy:.1%} | {dev_target_recall:.1%} | "
            "{dev_false_positive_rate:.1%} | {test_balanced_accuracy:.1%} | "
            "{test_target_recall:.1%} | {test_false_positive_rate:.1%} | "
            "{test_target_f1:.1%} | {multiclass_test_accuracy:.1%} | "
            "{paired_original_preferred:.1%} | {clean_masked_normalized_delta:.3f} |".format(
                **row
            )
        )
    chosen = all_results[selected]
    lines.extend(
        [
            "",
            "## Selected Candidate Diagnostics",
            "",
            f"- Dev threshold: `{chosen['threshold_selection']['threshold']:.6f}`",
            f"- Legacy 32-pair original-preferred rate: **{chosen['legacy_generated_pair_diagnostic']['original_preferred']:.1%}**",
            f"- Legacy mean original-minus-neutral margin: **{chosen['legacy_generated_pair_diagnostic']['mean_original_minus_neutral']:+.4f}**",
            f"- Clean-to-masked normalized mean absolute delta: **{chosen['content_invariance']['clean_masked_normalized_mean_abs_delta']:.4f}**",
            f"- Content-counterfactual normalized delta: **{chosen['content_invariance']['content_counterfactual_normalized_mean_abs_delta']:.4f}**",
            f"- Topic-term swap normalized delta: **{chosen['content_invariance']['topic_swap_normalized_mean_abs_delta']:.4f}**",
            f"- Target-test style-ablation margin drop: **{chosen['content_invariance']['style_ablation_mean_margin_drop_target_test']:+.4f}**",
            "",
            "The legacy 32 generated pairs are used only to test whether the meter",
            "prefers original target prose over its neutral back-translation. Iteration 5",
            "requires a fresh generated-domain calibration cohort before any method output.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "uv run python -m experiments.iteration5.meter.benchmark_content_resistant_meter",
            "```",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "selected_method": selected, "output_dir": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
