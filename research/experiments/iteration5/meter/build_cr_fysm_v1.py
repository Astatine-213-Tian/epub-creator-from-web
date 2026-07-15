#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from workflows.benchmark_author_style import (
    FUNCTION_CHARS,
    FUNCTION_WORDS,
    FUNCTION_WORD_TO_GROUP,
)
from experiments.iteration5.meter.benchmark_content_resistant_meter import (
    CJK_RE,
    DIALOGUE_OPEN,
    PLACEHOLDER_RE,
    PUNCTUATION,
    SEED,
    SINGLE_SPEECH,
    SPEECH_WORDS,
    STYLE_WORD_RE,
    TARGET_AUTHOR,
    TOPIC_TERMS,
    Record,
    author_book_weights,
    content_counterfactual,
    cjk_len,
    load_legacy_pairs,
    load_records,
    percentile,
    read_jsonl,
    topic_swap,
    write_csv,
)


SENTENCE_RE = re.compile(r"[^。！？!?]+[。！？!?]?")
SENTENCE_END = "。！？!?"
QUESTION_END = "？?"
EXCLAMATION_END = "！!"
FAMILY_ORDER = (
    "function_grammar",
    "punctuation_dialogue",
    "sentence_rhythm",
    "paragraph_discourse",
)


@dataclass
class FamilyArtifacts:
    name: str
    vectorizer: DictVectorizer
    scaler: StandardScaler
    model: LogisticRegression
    feature_names: list[str]
    keep_mask: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the four-family content-resistant Feitian style meter."
    )
    parser.add_argument(
        "--dataset",
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
            "content_resistant_v1/cr_fysm_v1"
        ),
    )
    parser.add_argument("--target-author", default=TARGET_AUTHOR)
    parser.add_argument("--exclude-boundary-chunks", type=int, default=2)
    parser.add_argument("--min-target-book-dispersion", type=float, default=0.80)
    parser.add_argument("--min-comparison-author-dispersion", type=float, default=0.50)
    return parser.parse_args()


def stable_order(values: list[str], *, salt: str) -> list[str]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(f"{SEED}:{salt}:{value}".encode()).hexdigest(),
    )


def corpus_metadata(path: Path, target_author: str) -> tuple[dict[str, str], list[str]]:
    title_area: dict[str, str] = {}
    comparison_authors: set[str] = set()
    for item in read_jsonl(path):
        if str(item["split"]) != "train":
            continue
        author = str(item["author"])
        if author == target_author:
            title_area[str(item["title"])] = str(item.get("time_area") or "unknown")
        else:
            comparison_authors.add(author)
    return title_area, sorted(comparison_authors)


def build_partition(path: Path, target_author: str) -> dict[str, Any]:
    title_area, comparison_authors = corpus_metadata(path, target_author)
    by_area: dict[str, list[str]] = defaultdict(list)
    for title, area in title_area.items():
        by_area[area].append(title)
    if len(title_area) != 29:
        raise ValueError(f"expected 29 target train books, found {len(title_area)}")
    if len(comparison_authors) != 49:
        raise ValueError(f"expected 49 comparison authors, found {len(comparison_authors)}")

    calibration_titles: list[str] = []
    qualification_titles: list[str] = []
    fit_titles: list[str] = []
    for area, titles in sorted(by_area.items()):
        ordered = stable_order(titles, salt=f"target:{area}")
        if len(ordered) < 2:
            raise ValueError(f"target stratum {area!r} has fewer than two books")
        calibration_titles.append(ordered[0])
        qualification_titles.append(ordered[1])
        fit_titles.extend(ordered[2:])
    if (len(fit_titles), len(calibration_titles), len(qualification_titles)) != (21, 4, 4):
        raise ValueError("target partition is not 21/4/4")

    ordered_authors = stable_order(comparison_authors, salt="comparison-author-role")
    fit_authors = ordered_authors[:29]
    calibration_authors = ordered_authors[29:39]
    qualification_authors = ordered_authors[39:49]
    return {
        "schema_version": 1,
        "seed": SEED,
        "target_author": target_author,
        "target": {
            "fit": sorted(fit_titles),
            "calibration": sorted(calibration_titles),
            "qualification": sorted(qualification_titles),
            "time_area": dict(sorted(title_area.items())),
        },
        "comparison": {
            "fit": sorted(fit_authors),
            "calibration": sorted(calibration_authors),
            "qualification": sorted(qualification_authors),
        },
        "policy": {
            "target": "one calibration and one qualification book per time-area stratum; remaining books fit",
            "comparison": "29/10/10 author-disjoint deterministic hash split",
            "boundary_chunks_excluded": 2,
        },
    }


def role_records(
    records: list[Record], partition: dict[str, Any], target_author: str
) -> dict[str, list[Record]]:
    target_roles = {
        title: role
        for role in ("fit", "calibration", "qualification")
        for title in partition["target"][role]
    }
    comparison_roles = {
        author: role
        for role in ("fit", "calibration", "qualification")
        for author in partition["comparison"][role]
    }
    result: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        if record.split != "train":
            continue
        role = (
            target_roles.get(record.title)
            if record.author == target_author
            else comparison_roles.get(record.author)
        )
        if role:
            result[role].append(record)
    for values in result.values():
        values.sort(key=lambda item: item.chunk_id)
    return result


def rate(count: float, total_cjk: int, scale: float = 1000.0) -> float:
    return count * scale / max(total_cjk, 1)


def function_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    index = 0
    while index < len(text):
        match = STYLE_WORD_RE.match(text, index)
        if match and match.group(0) in FUNCTION_WORDS:
            tokens.append(match.group(0))
            index = match.end()
            continue
        char = text[index]
        if char in FUNCTION_CHARS:
            tokens.append(char)
        if char in SENTENCE_END:
            tokens.append("<SENT>")
        index += 1
    return tokens


def function_grammar_features(text: str) -> dict[str, float]:
    total_cjk = max(cjk_len(text), 1)
    features: dict[str, float] = {}
    tokens = function_tokens(text)
    counts = Counter(token for token in tokens if token != "<SENT>")
    for token, count in counts.items():
        prefix = "word" if len(token) > 1 else "char"
        features[f"function_{prefix}:{token}"] = rate(count, total_cjk)
        if token in FUNCTION_WORD_TO_GROUP:
            group = FUNCTION_WORD_TO_GROUP[token]
            features[f"group:{group}"] = features.get(f"group:{group}", 0.0) + rate(
                count, total_cjk
            )
    for left, right in zip(tokens, tokens[1:], strict=False):
        if "<SENT>" not in {left, right}:
            features[f"transition:{left}>{right}"] = features.get(
                f"transition:{left}>{right}", 0.0
            ) + rate(1, total_cjk)
    features["function_token_rate"] = rate(sum(counts.values()), total_cjk)
    features["function_diversity"] = len(counts) / max(len(FUNCTION_CHARS) + len(FUNCTION_WORDS), 1)
    return features


def dialogue_shape(text: str) -> str:
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
                speech = next(
                    (word for word in SPEECH_WORDS if text.startswith(word, index)), None
                )
                if speech:
                    token = "S"
                    index += len(speech)
                elif char in SINGLE_SPEECH:
                    previous = text[index - 1] if index else ""
                    following = text[index + 1] if index + 1 < len(text) else ""
                    token = "S" if previous in PUNCTUATION or following in PUNCTUATION else "C"
                    index += 1
                elif CJK_RE.fullmatch(char) or char.isalnum():
                    token = "C"
                    index += 1
                else:
                    index += 1
                    continue
        if token in {"C", "¶"} and result and result[-1] == token:
            continue
        result.append(token)
    return "".join(result)


def punctuation_dialogue_features(text: str) -> dict[str, float]:
    total_cjk = max(cjk_len(text), 1)
    paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
    features: dict[str, float] = {
        f"punc:{char}": rate(text.count(char), total_cjk, 100.0) for char in PUNCTUATION
    }
    shape = dialogue_shape(text)
    for size in range(2, 6):
        for index in range(len(shape) - size + 1):
            value = shape[index : index + size]
            features[f"shape{size}:{value}"] = features.get(f"shape{size}:{value}", 0.0) + 1.0
    dialogue_lines = sum(1 for line in paragraphs if line.startswith(DIALOGUE_OPEN))
    features["dialogue_line_share"] = dialogue_lines / max(len(paragraphs), 1)
    features["speech_operator_rate"] = rate(shape.count("S"), total_cjk)
    features["quote_colon_rate"] = rate(text.count("：“") + text.count("：‘"), total_cjk)
    return features


def sentence_rhythm_features(text: str) -> dict[str, float]:
    sentences = [item.group(0) for item in SENTENCE_RE.finditer(text) if cjk_len(item.group(0))]
    lengths = [cjk_len(item) for item in sentences]
    comma_counts = [item.count("，") + item.count(",") for item in sentences]
    short_runs: list[int] = []
    current = 0
    for value in lengths:
        if value <= 12:
            current += 1
        elif current:
            short_runs.append(current)
            current = 0
    if current:
        short_runs.append(current)
    return {
        "mean_sentence_cjk": float(np.mean(lengths)) if lengths else 0.0,
        "std_sentence_cjk": float(np.std(lengths)) if lengths else 0.0,
        "cv_sentence_cjk": float(np.std(lengths) / max(np.mean(lengths), 1.0)) if lengths else 0.0,
        "p10_sentence_cjk": percentile(lengths, 0.10),
        "p25_sentence_cjk": percentile(lengths, 0.25),
        "p50_sentence_cjk": percentile(lengths, 0.50),
        "p75_sentence_cjk": percentile(lengths, 0.75),
        "p90_sentence_cjk": percentile(lengths, 0.90),
        "short_sentence_share": sum(value <= 12 for value in lengths) / max(len(lengths), 1),
        "long_sentence_share": sum(value >= 40 for value in lengths) / max(len(lengths), 1),
        "mean_short_run": float(np.mean(short_runs)) if short_runs else 0.0,
        "max_short_run": max(short_runs, default=0),
        "mean_commas_per_sentence": float(np.mean(comma_counts)) if comma_counts else 0.0,
        "p90_commas_per_sentence": percentile(comma_counts, 0.90),
        "question_sentence_share": sum(item.rstrip().endswith(tuple(QUESTION_END)) for item in sentences)
        / max(len(sentences), 1),
        "exclamation_sentence_share": sum(item.rstrip().endswith(tuple(EXCLAMATION_END)) for item in sentences)
        / max(len(sentences), 1),
    }


def paragraph_type(paragraph: str) -> str:
    stripped = paragraph.strip()
    if not stripped:
        return "empty"
    if stripped.startswith(DIALOGUE_OPEN):
        return "dialogue"
    if stripped.endswith(tuple(QUESTION_END)):
        return "question"
    if any(word in stripped[:18] for word in SPEECH_WORDS):
        return "attribution_open"
    return "narration"


def paragraph_discourse_features(text: str) -> dict[str, float]:
    paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
    lengths = [cjk_len(value) for value in paragraphs]
    types = [paragraph_type(value) for value in paragraphs]
    transitions = Counter(zip(types, types[1:], strict=False))
    features = {
        "mean_paragraph_cjk": float(np.mean(lengths)) if lengths else 0.0,
        "std_paragraph_cjk": float(np.std(lengths)) if lengths else 0.0,
        "p25_paragraph_cjk": percentile(lengths, 0.25),
        "p50_paragraph_cjk": percentile(lengths, 0.50),
        "p75_paragraph_cjk": percentile(lengths, 0.75),
        "p90_paragraph_cjk": percentile(lengths, 0.90),
        "short_paragraph_share": sum(value <= 15 for value in lengths) / max(len(lengths), 1),
        "long_paragraph_share": sum(value >= 80 for value in lengths) / max(len(lengths), 1),
        "paragraph_length_cv": float(np.std(lengths) / max(np.mean(lengths), 1.0)) if lengths else 0.0,
    }
    type_counts = Counter(types)
    for value, count in type_counts.items():
        features[f"paragraph_type_share:{value}"] = count / max(len(types), 1)
    for (left, right), count in transitions.items():
        features[f"paragraph_transition:{left}>{right}"] = count / max(len(types) - 1, 1)
    question_response = sum(
        left == "question" and right in {"dialogue", "attribution_open"}
        for left, right in zip(types, types[1:], strict=False)
    )
    features["question_response_share"] = question_response / max(len(types) - 1, 1)
    return features


FEATURE_FNS: dict[str, Callable[[str], dict[str, float]]] = {
    "function_grammar": function_grammar_features,
    "punctuation_dialogue": punctuation_dialogue_features,
    "sentence_rhythm": sentence_rhythm_features,
    "paragraph_discourse": paragraph_discourse_features,
}


def feature_dispersion_mask(
    matrix: sparse.csr_matrix,
    records: list[Record],
    feature_names: list[str],
    *,
    target_author: str,
    min_target_fraction: float,
    min_comparison_fraction: float,
) -> np.ndarray:
    target_titles = sorted({item.title for item in records if item.author == target_author})
    comparison_authors = sorted({item.author for item in records if item.author != target_author})
    target_support = [set() for _ in feature_names]
    comparison_support = [set() for _ in feature_names]
    for row_index, record in enumerate(records):
        indices = matrix.indices[matrix.indptr[row_index] : matrix.indptr[row_index + 1]]
        destination = target_support if record.author == target_author else comparison_support
        key = record.title if record.author == target_author else record.author
        for index in indices:
            destination[int(index)].add(key)
    keep = np.asarray(
        [
            len(target_support[index]) / len(target_titles) >= min_target_fraction
            or len(comparison_support[index]) / len(comparison_authors)
            >= min_comparison_fraction
            for index in range(len(feature_names))
        ],
        dtype=bool,
    )
    # Aggregate continuous features are defined for every row and always survive.
    for index, name in enumerate(feature_names):
        if not name.startswith(("shape", "transition:")):
            keep[index] = True
    return keep


def fit_family(
    name: str,
    fit_records: list[Record],
    *,
    target_author: str,
    min_target_fraction: float,
    min_comparison_fraction: float,
) -> FamilyArtifacts:
    fn = FEATURE_FNS[name]
    vectorizer = DictVectorizer(sparse=True, sort=True)
    raw = vectorizer.fit_transform(fn(item.text) for item in fit_records).tocsr()
    names = vectorizer.get_feature_names_out().tolist()
    keep = feature_dispersion_mask(
        raw,
        fit_records,
        names,
        target_author=target_author,
        min_target_fraction=min_target_fraction,
        min_comparison_fraction=min_comparison_fraction,
    )
    raw = raw[:, keep].tocsr()
    names = np.asarray(names)[keep].tolist()
    scaler = StandardScaler(with_mean=False)
    matrix = scaler.fit_transform(raw).tocsr()
    truth = np.asarray([item.author == target_author for item in fit_records], dtype=np.int8)
    weights = author_book_weights(fit_records, binary_target=target_author)
    model = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="liblinear",
        max_iter=2000,
        random_state=SEED,
    )
    model.fit(matrix, truth, sample_weight=weights)
    return FamilyArtifacts(name, vectorizer, scaler, model, names, keep)


def transform_family(artifacts: FamilyArtifacts, texts: list[str]) -> sparse.csr_matrix:
    fn = FEATURE_FNS[artifacts.name]
    raw = artifacts.vectorizer.transform(fn(text) for text in texts).tocsr()
    raw = raw[:, artifacts.keep_mask].tocsr()
    return artifacts.scaler.transform(raw).tocsr()


def family_probabilities(
    artifacts: dict[str, FamilyArtifacts], texts: list[str]
) -> dict[str, np.ndarray]:
    return {
        name: family.model.predict_proba(transform_family(family, texts))[:, 1]
        for name, family in artifacts.items()
    }


def ensemble(probabilities: dict[str, np.ndarray], excluded: str | None = None) -> np.ndarray:
    values = [value for name, value in probabilities.items() if name != excluded]
    return np.mean(np.vstack(values), axis=0)


def choose_threshold(scores: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    candidates = np.unique(scores)
    selected: tuple[float, float, float, float] | None = None
    for threshold in candidates:
        predicted = scores >= threshold
        sensitivity = recall_score(truth, predicted, zero_division=0)
        specificity = recall_score(1 - truth, 1 - predicted, zero_division=0)
        balanced = balanced_accuracy_score(truth, predicted)
        eligible = sensitivity >= 0.80 and specificity >= 0.90
        rank = (1.0 if eligible else 0.0, balanced, specificity, -float(threshold))
        if selected is None or rank > selected:
            selected = rank
    if selected is None:
        raise ValueError("cannot calibrate threshold")
    threshold = -selected[3]
    predicted = scores >= threshold
    return {
        "threshold": threshold,
        "sensitivity": float(recall_score(truth, predicted, zero_division=0)),
        "specificity": float(recall_score(1 - truth, 1 - predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "eligible": bool(selected[0]),
    }


def expected_calibration_error(scores: np.ndarray, truth: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    positives = max(int(truth.sum()), 1)
    negatives = max(int((1 - truth).sum()), 1)
    weights = np.where(truth == 1, 0.5 / positives, 0.5 / negatives)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (scores >= lower) & (scores < upper if upper < 1.0 else scores <= upper)
        if not mask.any():
            continue
        bin_weight = float(weights[mask].sum())
        confidence = float(np.average(scores[mask], weights=weights[mask]))
        accuracy = float(np.average(truth[mask], weights=weights[mask]))
        value += bin_weight * abs(confidence - accuracy)
    return value


def score_metrics(scores: np.ndarray, truth: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = scores >= threshold
    return {
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "sensitivity": float(recall_score(truth, predicted, zero_division=0)),
        "specificity": float(recall_score(1 - truth, 1 - predicted, zero_division=0)),
        "target_precision": float(precision_score(truth, predicted, zero_division=0)),
        "target_f1": float(f1_score(truth, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, scores)),
    }


def per_target_book(
    records: list[Record], scores: np.ndarray, threshold: float, target_author: str
) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if record.author == target_author:
            grouped[record.title].append(index)
    return [
        {
            "book": title,
            "rows": len(indices),
            "sensitivity": float(np.mean(scores[indices] >= threshold)),
            "mean_score": float(np.mean(scores[indices])),
        }
        for title, indices in sorted(grouped.items())
    ]


def make_chart(path: Path, rows: list[dict[str, Any]]) -> None:
    width, height = 920, 360
    left, right, top, bottom = 80, 24, 40, 105
    plot_w, plot_h = width - left - right, height - top - bottom
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="25" font-family="Arial" font-size="16" fill="#111827">CR-FYSM-v1 calibration and one-shot qualification</text>',
    ]
    for tick in range(0, 101, 20):
        y = top + plot_h - tick / 100 * plot_h
        pieces.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        pieces.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#4b5563">{tick}%</text>')
    labels = [str(row["label"]) for row in rows]
    bar_w = plot_w / max(len(rows), 1) * 0.55
    for index, row in enumerate(rows):
        value = float(row["value"]) * 100.0
        center = left + (index + 0.5) * plot_w / len(rows)
        y = top + plot_h - value / 100 * plot_h
        color = "#0f766e" if value >= float(row["gate"]) * 100 else "#b45309"
        pieces.append(f'<rect x="{center - bar_w / 2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{top + plot_h - y:.1f}" fill="{color}"/>')
        pieces.append(f'<text x="{center:.1f}" y="{y - 7:.1f}" text-anchor="middle" font-family="Arial" font-size="12">{value:.1f}%</text>')
        pieces.append(f'<text x="{center:.1f}" y="{top + plot_h + 20}" text-anchor="end" transform="rotate(-25 {center:.1f} {top + plot_h + 20})" font-family="Arial" font-size="11" fill="#374151">{html.escape(labels[index])}</text>')
    pieces.append("</svg>")
    path.write_text("\n".join(pieces), encoding="utf-8")


def main() -> None:
    args = parse_args()
    partition = build_partition(args.dataset, args.target_author)
    records = load_records(args.dataset, boundary=args.exclude_boundary_chunks)
    roles = role_records(records, partition, args.target_author)
    if set(roles) != {"fit", "calibration", "qualification"}:
        raise ValueError(f"incomplete roles: {sorted(roles)}")

    artifacts = {
        name: fit_family(
            name,
            roles["fit"],
            target_author=args.target_author,
            min_target_fraction=args.min_target_book_dispersion,
            min_comparison_fraction=args.min_comparison_author_dispersion,
        )
        for name in FAMILY_ORDER
    }
    role_probabilities = {
        role: family_probabilities(artifacts, [item.text for item in role_records])
        for role, role_records in roles.items()
    }
    role_truth = {
        role: np.asarray([item.author == args.target_author for item in role_records], dtype=np.int8)
        for role, role_records in roles.items()
    }
    role_scores = {
        role: ensemble(probabilities) for role, probabilities in role_probabilities.items()
    }
    calibration = choose_threshold(role_scores["calibration"], role_truth["calibration"])
    threshold = float(calibration["threshold"])
    metrics = {
        role: score_metrics(role_scores[role], role_truth[role], threshold)
        for role in ("calibration", "qualification")
    }
    ece = expected_calibration_error(role_scores["calibration"], role_truth["calibration"])

    family_metrics: dict[str, Any] = {}
    leave_one_out: dict[str, Any] = {}
    for name in FAMILY_ORDER:
        family_threshold = choose_threshold(
            role_probabilities["calibration"][name], role_truth["calibration"]
        )
        family_metrics[name] = {
            "feature_count": len(artifacts[name].feature_names),
            "calibration_threshold": family_threshold,
            "qualification": score_metrics(
                role_probabilities["qualification"][name],
                role_truth["qualification"],
                float(family_threshold["threshold"]),
            ),
        }
        ablated_cal = ensemble(role_probabilities["calibration"], excluded=name)
        ablated_qual = ensemble(role_probabilities["qualification"], excluded=name)
        ablated_threshold = choose_threshold(ablated_cal, role_truth["calibration"])
        leave_one_out[name] = {
            "threshold": ablated_threshold,
            "qualification": score_metrics(
                ablated_qual,
                role_truth["qualification"],
                float(ablated_threshold["threshold"]),
            ),
        }

    qual_texts = [item.text for item in roles["qualification"]]
    qual_counterfactual = [content_counterfactual(text) for text in qual_texts]
    qual_topic = [topic_swap(text) for text in qual_texts]
    counterfactual_scores = ensemble(family_probabilities(artifacts, qual_counterfactual))
    topic_scores = ensemble(family_probabilities(artifacts, qual_topic))
    counterfactual_delta = np.abs(counterfactual_scores - role_scores["qualification"])
    topic_delta = np.abs(topic_scores - role_scores["qualification"])

    clean_records = load_records(args.clean_dataset, boundary=args.exclude_boundary_chunks)
    clean_by_id = {item.chunk_id: item.text for item in clean_records}
    clean_texts = [clean_by_id[item.chunk_id] for item in roles["qualification"]]
    clean_scores = ensemble(family_probabilities(artifacts, clean_texts))
    clean_delta = np.abs(clean_scores - role_scores["qualification"])

    pair_ids, pair_originals, pair_neutral = load_legacy_pairs(args.iteration4_root)
    original_probabilities = family_probabilities(artifacts, pair_originals)
    neutral_probabilities = family_probabilities(artifacts, pair_neutral)
    original_scores = ensemble(original_probabilities)
    neutral_scores = ensemble(neutral_probabilities)
    family_positive_lifts = np.vstack(
        [original_probabilities[name] > neutral_probabilities[name] for name in FAMILY_ORDER]
    ).sum(axis=0)
    legacy_pairs = {
        "rows": len(pair_ids),
        "original_preferred": float(np.mean(original_scores > neutral_scores)),
        "mean_style_effect": float(np.mean(original_scores - neutral_scores)),
        "at_least_three_family_positive": float(np.mean(family_positive_lifts >= 3)),
        "original_above_corpus_threshold": float(np.mean(original_scores >= threshold)),
        "neutral_above_corpus_threshold": float(np.mean(neutral_scores >= threshold)),
    }

    variances = {
        name: float(np.var(role_probabilities["qualification"][name])) for name in FAMILY_ORDER
    }
    variance_total = sum(variances.values()) or 1.0
    variance_shares = {name: value / variance_total for name, value in variances.items()}
    target_books = per_target_book(
        roles["qualification"],
        role_scores["qualification"],
        threshold,
        args.target_author,
    )
    invariance = {
        "clean_masked_median_abs_delta": float(np.median(clean_delta)),
        "clean_masked_p95_abs_delta": float(np.quantile(clean_delta, 0.95)),
        "clean_masked_threshold_flip_rate": float(
            np.mean((clean_scores >= threshold) != (role_scores["qualification"] >= threshold))
        ),
        "content_counterfactual_median_abs_delta": float(np.median(counterfactual_delta)),
        "content_counterfactual_p95_abs_delta": float(np.quantile(counterfactual_delta, 0.95)),
        "content_counterfactual_threshold_flip_rate": float(
            np.mean(
                (counterfactual_scores >= threshold)
                != (role_scores["qualification"] >= threshold)
            )
        ),
        "topic_swap_median_abs_delta": float(np.median(topic_delta)),
        "topic_swap_p95_abs_delta": float(np.quantile(topic_delta, 0.95)),
    }
    statistical_gates = {
        "calibration_sensitivity": metrics["calibration"]["sensitivity"] >= 0.80,
        "calibration_specificity": metrics["calibration"]["specificity"] >= 0.90,
        "calibration_ece": ece <= 0.10,
        "qualification_sensitivity": metrics["qualification"]["sensitivity"] >= 0.80,
        "qualification_specificity": metrics["qualification"]["specificity"] >= 0.90,
        "qualification_book_floor": min(row["sensitivity"] for row in target_books) >= 0.60,
        "content_counterfactual_median": invariance[
            "content_counterfactual_median_abs_delta"
        ]
        <= 0.05,
        "content_counterfactual_p95": invariance["content_counterfactual_p95_abs_delta"]
        <= 0.15,
        "content_counterfactual_flips": invariance[
            "content_counterfactual_threshold_flip_rate"
        ]
        <= 0.05,
        "clean_masked_flips": invariance["clean_masked_threshold_flip_rate"] <= 0.05,
        "legacy_pair_preference": legacy_pairs["original_preferred"] >= 0.80,
        "legacy_three_family_agreement": legacy_pairs["at_least_three_family_positive"]
        >= 0.80,
        "family_variance_share": max(variance_shares.values()) <= 0.50,
    }
    statistical_pass = all(statistical_gates.values())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    partition_path = args.output_dir / "partition.v1.json"
    partition_path.write_text(
        json.dumps(partition, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "qualification_target_books.csv", target_books)
    family_rows = []
    for name in FAMILY_ORDER:
        result = family_metrics[name]
        family_rows.append(
            {
                "family": name,
                "features": result["feature_count"],
                "qualification_balanced_accuracy": result["qualification"][
                    "balanced_accuracy"
                ],
                "qualification_sensitivity": result["qualification"]["sensitivity"],
                "qualification_specificity": result["qualification"]["specificity"],
                "qualification_f1": result["qualification"]["target_f1"],
                "variance_share": variance_shares[name],
                "leave_one_out_balanced_accuracy": leave_one_out[name]["qualification"][
                    "balanced_accuracy"
                ],
            }
        )
    write_csv(args.output_dir / "family_results.csv", family_rows)

    model_dir = args.output_dir / "meter"
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, family in artifacts.items():
        joblib.dump(family.vectorizer, model_dir / f"{name}.vectorizer.joblib")
        joblib.dump(family.scaler, model_dir / f"{name}.scaler.joblib")
        joblib.dump(family.keep_mask, model_dir / f"{name}.keep_mask.joblib")
        joblib.dump(family.model, model_dir / f"{name}.model.joblib")
        coefficient_indices = np.argsort(family.model.coef_[0])[::-1][:100]
        write_csv(
            args.output_dir / f"top_positive_features.{name}.csv",
            [
                {
                    "rank": rank,
                    "feature": family.feature_names[int(index)],
                    "weight": float(family.model.coef_[0, int(index)]),
                }
                for rank, index in enumerate(coefficient_indices, start=1)
            ],
        )

    result = {
        "schema_version": 1,
        "meter_id": "CR-FYSM-v1",
        "status": (
            "statistical_qualification_pass_pending_human_convergence_and_fresh_generated_calibration"
            if statistical_pass
            else "statistical_qualification_fail"
        ),
        "target_author": args.target_author,
        "partition": partition,
        "role_rows": {role: len(values) for role, values in roles.items()},
        "families": family_metrics,
        "leave_one_family_out": leave_one_out,
        "ensemble": {
            "family_order": list(FAMILY_ORDER),
            "weights": {name: 0.25 for name in FAMILY_ORDER},
            "calibration": calibration,
            "calibration_metrics": metrics["calibration"],
            "calibration_ece": ece,
            "qualification_metrics": metrics["qualification"],
            "qualification_target_books": target_books,
            "family_variance_share": variance_shares,
        },
        "content_invariance": invariance,
        "legacy_generated_pair_diagnostic": legacy_pairs,
        "statistical_gates": statistical_gates,
        "pending_fatal_gates": [
            "fresh_generated_domain_calibration",
            "three_rater_blind_human_convergence",
            "matched_negative_and_factorial_content_style_challenge",
            "independent_exact_artifact_audit",
        ],
        "old_exact_ngram_role": "secondary_historical_only_hidden_from_generation_and_selection",
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "meter_id": "CR-FYSM-v1",
        "status": result["status"],
        "family_order": list(FAMILY_ORDER),
        "family_weights": {name: 0.25 for name in FAMILY_ORDER},
        "corpus_threshold": threshold,
        "partition_path": str(partition_path),
        "research_role": "candidate_primary_meter_not_authorized_for_generation_until_pending_fatal_gates_pass",
    }
    (model_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    chart_rows = [
        {"label": "Calibration sensitivity", "value": metrics["calibration"]["sensitivity"], "gate": 0.80},
        {"label": "Calibration specificity", "value": metrics["calibration"]["specificity"], "gate": 0.90},
        {"label": "Qualification sensitivity", "value": metrics["qualification"]["sensitivity"], "gate": 0.80},
        {"label": "Qualification specificity", "value": metrics["qualification"]["specificity"], "gate": 0.90},
        {"label": "Original > neutral", "value": legacy_pairs["original_preferred"], "gate": 0.80},
        {"label": "3/4 family agreement", "value": legacy_pairs["at_least_three_family_positive"], "gate": 0.80},
    ]
    make_chart(args.output_dir / "qualification_summary.svg", chart_rows)

    lines = [
        "# CR-FYSM-v1 Statistical Qualification",
        "",
        f"- Status: **{result['status']}**",
        "- Primary estimand: target-style form, not 50-author attribution",
        "- Families: function grammar, punctuation/dialogue, sentence rhythm, paragraph/discourse",
        "- Family weights: 25% each",
        "- Old exact lexical ngram meter: secondary historical evidence only",
        "",
        "![Qualification summary](qualification_summary.svg)",
        "",
        "## Data Roles",
        "",
        f"- Fit rows: {len(roles['fit']):,}",
        f"- Calibration rows: {len(roles['calibration']):,}",
        f"- One-shot qualification rows: {len(roles['qualification']):,}",
        "- Target books: 21 fit / 4 calibration / 4 qualification, stratified by time area",
        "- Comparison authors: 29 fit / 10 calibration / 10 qualification, author-disjoint",
        "- Existing target dev, proxy-transfer, and test books are outside meter construction",
        "",
        "## Ensemble Result",
        "",
        "| Stage | Balanced accuracy | Sensitivity | Specificity | Target F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Calibration | {metrics['calibration']['balanced_accuracy']:.1%} | {metrics['calibration']['sensitivity']:.1%} | {metrics['calibration']['specificity']:.1%} | {metrics['calibration']['target_f1']:.1%} |",
        f"| Qualification | {metrics['qualification']['balanced_accuracy']:.1%} | {metrics['qualification']['sensitivity']:.1%} | {metrics['qualification']['specificity']:.1%} | {metrics['qualification']['target_f1']:.1%} |",
        "",
        f"Calibration ECE: **{ece:.4f}** (gate <= 0.10).",
        "",
        "## Family Ablations",
        "",
        "| Family | Features | Qualification balanced acc | Sensitivity | Specificity | Target F1 | Variance share | Leave-family-out balanced acc |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in family_rows:
        lines.append(
            f"| `{row['family']}` | {row['features']} | "
            f"{row['qualification_balanced_accuracy']:.1%} | "
            f"{row['qualification_sensitivity']:.1%} | "
            f"{row['qualification_specificity']:.1%} | "
            f"{row['qualification_f1']:.1%} | "
            f"{row['variance_share']:.1%} | "
            f"{row['leave_one_out_balanced_accuracy']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Content Invariance",
            "",
            "| Test | Median absolute score delta | 95th percentile | Threshold flips |",
            "| --- | ---: | ---: | ---: |",
            f"| Clean versus entity-masked | {invariance['clean_masked_median_abs_delta']:.4f} | {invariance['clean_masked_p95_abs_delta']:.4f} | {invariance['clean_masked_threshold_flip_rate']:.2%} |",
            f"| Lexical counterfactual | {invariance['content_counterfactual_median_abs_delta']:.4f} | {invariance['content_counterfactual_p95_abs_delta']:.4f} | {invariance['content_counterfactual_threshold_flip_rate']:.2%} |",
            f"| Audited topic-term swap | {invariance['topic_swap_median_abs_delta']:.4f} | {invariance['topic_swap_p95_abs_delta']:.4f} | n/a |",
            "",
            "## Same-Content Diagnostic",
            "",
            f"- Legacy paired rows: {legacy_pairs['rows']}",
            f"- Original target passage scored above neutral translation: **{legacy_pairs['original_preferred']:.1%}**",
            f"- At least three of four family lifts positive: **{legacy_pairs['at_least_three_family_positive']:.1%}**",
            f"- Mean original-minus-neutral score: **{legacy_pairs['mean_style_effect']:+.4f}**",
            "",
            "These 32 pairs are legacy diagnostics and cannot calibrate or evaluate an",
            "Iteration 5 transfer method.",
            "",
            "## Remaining Fatal Gates",
            "",
        ]
    )
    lines.extend(f"- `{item}`" for item in result["pending_fatal_gates"])
    lines.extend(
        [
            "",
            "Generation remains unauthorized until those gates are prospectively",
            "materialized, rehearsed, and independently audited.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "uv run python -m experiments.iteration5.meter.build_cr_fysm_v1",
            "```",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
