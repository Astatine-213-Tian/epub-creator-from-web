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
from typing import Any, Callable, Iterable


CJK_RE = re.compile(r"[\u4e00-\u9fff]")
SENTENCE_SPLIT_RE = re.compile(r"[。！？!?]+")
PUNCT_CHARS = "，。！？；：、“”‘’（）《》【】…—,.!?;:\"'()[]"
DIALOGUE_CHARS = "“”‘’「」『』"
FUNCTION_CHARS = set(
    "的一是在不了有和人这中为上个我以要他时来用们到地于出就对成会可也能下过"
    "而后定行得经之着等里如自起把性好应开还因由其些然前那与关各并已又但"
    "只没给被很最才让吗呢啊吧呀么着了过"
)
FUNCTION_WORD_GROUPS = {
    "connective": (
        "因为", "所以", "因此", "但是", "不过", "然而", "而且", "并且", "或者", "还是",
        "如果", "假如", "虽然", "尽管", "既然", "于是", "然后", "接着", "随后", "况且",
        "此外", "另外", "以及", "甚至", "尤其", "总之", "至少", "反正", "其实", "当然",
    ),
    "modal_aspect": (
        "已经", "正在", "曾经", "将要", "快要", "就要", "仍然", "依然", "终于", "立刻",
        "马上", "忽然", "突然", "大概", "可能", "也许", "应该", "必须", "需要", "可以",
        "能够", "愿意", "不会", "不是", "没有", "不能", "不用", "不要", "不得不", "未必",
    ),
    "deictic_pronoun": (
        "这个", "那个", "这些", "那些", "这里", "那里", "这样", "那样", "这么", "那么",
        "怎么", "什么", "为什么", "哪里", "哪个", "自己", "别人", "大家", "彼此", "对方",
    ),
    "particle_phrase": (
        "而已", "罢了", "似的", "一样", "起来", "下去", "出来", "进去", "过去", "过来",
        "一下", "一会儿", "是不是", "对不对", "有没有", "好不好", "怎么办", "怎么样", "什么样", "没什么",
    ),
    "preposition_frame": (
        "关于", "对于", "由于", "为了", "按照", "根据", "通过", "经过", "除了", "作为",
        "并非", "并不", "无法", "无论", "不管", "只要", "只有", "除非", "无非", "至于",
    ),
}
FUNCTION_WORDS = tuple(
    sorted({word for words in FUNCTION_WORD_GROUPS.values() for word in words}, key=lambda word: (-len(word), word))
)
FUNCTION_WORD_TO_GROUP = {word: group for group, words in FUNCTION_WORD_GROUPS.items() for word in words}
FUNCTION_WORD_RE = re.compile("|".join(re.escape(word) for word in FUNCTION_WORDS))
TARGET_AUTHOR_DEFAULT = "非天夜翔"


@dataclass(frozen=True)
class RunSpec:
    key: str
    method: str
    view: str
    feature_family: str
    max_profile_features: int


@dataclass
class AuthorProfileModel:
    labels: list[str]
    idf: dict[str, float]
    inverted_profiles: dict[str, list[tuple[int, float]]]
    train_docs_by_author: dict[str, int]
    train_chunks: int
    profile_feature_counts: dict[str, int]


FeatureFn = Callable[[str], Counter[str]]


def chunk_paths(dataset_root: Path) -> dict[str, Path]:
    return {
        "clean": dataset_root / "unmasked" / "chunks.clean.jsonl",
        "entity_masked": dataset_root / "masked" / "chunks.entity_masked.jsonl",
        "entity_masked_v2": dataset_root / "masked" / "chunks.entity_masked_v2.jsonl",
        "entity_masked_v3": dataset_root / "masked" / "chunks.entity_masked_v3.jsonl",
        "topic_distorted": dataset_root / "masked" / "chunks.topic_distorted.jsonl",
        "structure_only": dataset_root / "masked" / "chunks.structure_only.jsonl",
    }


def jsonl_records(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc


def cjk_len(text: str) -> int:
    return len(CJK_RE.findall(text))


def bucket(value: float, edges: tuple[float, ...]) -> str:
    for edge in edges:
        if value <= edge:
            return f"le_{int(edge)}"
    return f"gt_{int(edges[-1])}"


def char_ngram_features(text: str, *, min_n: int = 2, max_n: int = 4) -> Counter[str]:
    normalized = re.sub(r"\s+", "", text)
    counts: Counter[str] = Counter()
    for size in range(min_n, max_n + 1):
        if len(normalized) < size:
            continue
        for index in range(0, len(normalized) - size + 1):
            counts[f"char{size}:{normalized[index:index + size]}"] += 1
    return counts


def punctuation_dialogue_features(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for char in text:
        if char in PUNCT_CHARS:
            counts[f"punc:{char}"] += 1
        if char in DIALOGUE_CHARS:
            counts[f"dialogue_mark:{char}"] += 1
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    dialogue_lines = sum(1 for line in lines if line.startswith(("“", '"', "「", "『")))
    counts[f"dialogue_line_ratio:{bucket(dialogue_lines / max(len(lines), 1) * 100, (5, 15, 30, 50, 75))}"] += 1
    counts[f"quote_count:{bucket(sum(text.count(char) for char in DIALOGUE_CHARS), (2, 6, 12, 24, 48))}"] += 1
    counts[f"colon_count:{bucket(text.count('：') + text.count(':'), (1, 3, 6, 12, 24))}"] += 1
    counts[f"ellipsis_count:{bucket(text.count('…'), (1, 2, 4, 8, 16))}"] += 1
    return counts


def length_shape_features(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
    sentences = [part for part in SENTENCE_SPLIT_RE.split(text) if cjk_len(part) > 0]
    sentence_lengths = [cjk_len(sentence) for sentence in sentences]
    paragraph_lengths = [cjk_len(paragraph) for paragraph in paragraphs]

    counts[f"paragraph_count:{bucket(len(paragraphs), (1, 2, 4, 8, 16, 32))}"] += 1
    counts[f"sentence_count:{bucket(len(sentences), (3, 6, 10, 16, 24, 40))}"] += 1
    if sentence_lengths:
        avg_sentence = sum(sentence_lengths) / len(sentence_lengths)
        counts[f"avg_sentence_cjk:{bucket(avg_sentence, (12, 20, 32, 48, 72, 110))}"] += 1
        for length in sentence_lengths:
            counts[f"sentence_cjk:{bucket(length, (8, 16, 28, 45, 70, 110, 180))}"] += 1
    if paragraph_lengths:
        avg_paragraph = sum(paragraph_lengths) / len(paragraph_lengths)
        counts[f"avg_paragraph_cjk:{bucket(avg_paragraph, (40, 80, 140, 220, 360, 600))}"] += 1
        for length in paragraph_lengths:
            counts[f"paragraph_cjk:{bucket(length, (30, 70, 120, 200, 340, 560, 900))}"] += 1
    return counts


def function_char_features(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for char in text:
        if char in FUNCTION_CHARS:
            counts[f"function_char:{char}"] += 1
    total_cjk = max(cjk_len(text), 1)
    function_total = sum(counts.values())
    counts[f"function_ratio:{bucket(function_total / total_cjk * 100, (20, 30, 40, 50, 60, 70))}"] += 1
    return counts


def function_word_features(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    matched_words: set[str] = set()
    for match in FUNCTION_WORD_RE.finditer(text):
        word = match.group(0)
        group = FUNCTION_WORD_TO_GROUP[word]
        counts[f"function_word:{word}"] += 1
        counts[f"function_word_group:{group}"] += 1
        matched_words.add(word)
    total_words = sum(value for key, value in counts.items() if key.startswith("function_word:"))
    total_cjk = max(cjk_len(text), 1)
    counts[f"function_word_ratio:{bucket(total_words / total_cjk * 1000, (2, 5, 10, 20, 35, 55, 80, 120))}"] += 1
    counts[f"function_word_diversity:{bucket(len(matched_words), (2, 4, 8, 12, 18, 26, 36, 50))}"] += 1
    return counts


def function_words_plus_chars_features(text: str) -> Counter[str]:
    counts = function_char_features(text)
    counts.update(function_word_features(text))
    return counts


def combined_interpretable_features(text: str) -> Counter[str]:
    counts = punctuation_dialogue_features(text)
    counts.update(length_shape_features(text))
    counts.update(function_char_features(text))
    return counts


def combined_rich_function_words_features(text: str) -> Counter[str]:
    counts = combined_interpretable_features(text)
    counts.update(function_word_features(text))
    return counts


FEATURE_FNS: dict[str, FeatureFn] = {
    "char_ngrams": char_ngram_features,
    "punctuation_dialogue": punctuation_dialogue_features,
    "length_shape": length_shape_features,
    "function_chars": function_char_features,
    "function_words": function_word_features,
    "function_words_plus_chars": function_words_plus_chars_features,
    "combined_interpretable": combined_interpretable_features,
    "combined_rich_function_words": combined_rich_function_words_features,
}


METHOD_LABELS = {
    "char_ngrams": "Character n-grams",
    "char_hashing": "Character n-grams (hashed)",
    "punctuation_dialogue": "Punctuation/dialogue",
    "length_shape": "Sentence/paragraph length",
    "function_chars": "Function-character",
    "function_words": "Chinese function words",
    "function_words_plus_chars": "Function words + characters",
    "combined_interpretable": "Combined interpretable",
    "combined_rich_function_words": "Combined + function words",
}


def default_run_specs(views: tuple[str, ...], *, max_char_features: int, max_interpretable_features: int) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for view in views:
        specs.append(
            RunSpec(
                key=f"char_ngrams.{view}",
                method="char_ngrams",
                view=view,
                feature_family="char_ngrams",
                max_profile_features=max_char_features,
            )
        )
        for family in (
            "punctuation_dialogue",
            "length_shape",
            "function_chars",
            "function_words",
            "function_words_plus_chars",
            "combined_interpretable",
            "combined_rich_function_words",
        ):
            specs.append(
                RunSpec(
                    key=f"{family}.{view}",
                    method=family,
                    view=view,
                    feature_family=family,
                    max_profile_features=max_interpretable_features,
                )
            )
    return specs


def train_author_profile_model(path: Path, feature_fn: FeatureFn, *, max_profile_features: int) -> AuthorProfileModel:
    class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    train_docs_by_author: Counter[str] = Counter()
    for item in jsonl_records(path):
        if item["split"] != "train":
            continue
        author = str(item["author"])
        train_docs_by_author[author] += 1
        class_counts[author].update(feature_fn(str(item["text"])))

    labels = sorted(class_counts)
    author_feature_presence: Counter[str] = Counter()
    for counts in class_counts.values():
        author_feature_presence.update(counts.keys())
    idf = {
        feature: math.log((1 + len(labels)) / (1 + df)) + 1.0
        for feature, df in author_feature_presence.items()
    }

    inverted_profiles: dict[str, list[tuple[int, float]]] = defaultdict(list)
    profile_feature_counts: dict[str, int] = {}
    for class_index, author in enumerate(labels):
        counts = class_counts[author]
        total = max(sum(counts.values()), 1)
        weighted = [
            (feature, count / total * idf.get(feature, 1.0))
            for feature, count in counts.items()
        ]
        weighted.sort(key=lambda item: (-item[1], item[0]))
        selected = weighted[:max_profile_features]
        norm = math.sqrt(sum(weight * weight for _feature, weight in selected)) or 1.0
        profile_feature_counts[author] = len(selected)
        for feature, weight in selected:
            inverted_profiles[feature].append((class_index, weight / norm))

    return AuthorProfileModel(
        labels=labels,
        idf=idf,
        inverted_profiles=dict(inverted_profiles),
        train_docs_by_author=dict(train_docs_by_author),
        train_chunks=sum(train_docs_by_author.values()),
        profile_feature_counts=profile_feature_counts,
    )


def score_features(model: AuthorProfileModel, features: Counter[str]) -> list[float]:
    total = max(sum(features.values()), 1)
    weighted: list[tuple[str, float]] = []
    for feature, count in features.items():
        if feature not in model.inverted_profiles:
            continue
        weighted.append((feature, count / total * model.idf.get(feature, 1.0)))
    norm = math.sqrt(sum(weight * weight for _feature, weight in weighted)) or 1.0
    scores = [0.0 for _label in model.labels]
    for feature, weight in weighted:
        doc_weight = weight / norm
        for class_index, profile_weight in model.inverted_profiles.get(feature, ()):
            scores[class_index] += doc_weight * profile_weight
    return scores


def predict(model: AuthorProfileModel, feature_fn: FeatureFn, text: str) -> tuple[str, list[float]]:
    scores = score_features(model, feature_fn(text))
    if not scores:
        return model.labels[0], scores
    best_index = max(range(len(model.labels)), key=lambda index: (scores[index], model.labels[index]))
    return model.labels[best_index], scores


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_predictions(
    rows: list[dict[str, str]],
    labels: list[str],
    *,
    target_author: str,
    majority_author: str,
) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if row["gold"] == row["pred"])
    by_label: dict[str, dict[str, int]] = {}
    for label in labels:
        tp = sum(1 for row in rows if row["gold"] == label and row["pred"] == label)
        fp = sum(1 for row in rows if row["gold"] != label and row["pred"] == label)
        fn = sum(1 for row in rows if row["gold"] == label and row["pred"] != label)
        by_label[label] = {"tp": tp, "fp": fp, "fn": fn, **prf(tp, fp, fn)}
    recalls = [by_label[label]["recall"] for label in labels if any(row["gold"] == label for row in rows)]
    f1s = [by_label[label]["f1"] for label in labels if any(row["gold"] == label for row in rows)]
    majority_correct = sum(1 for row in rows if row["gold"] == majority_author)
    return {
        "rows": total,
        "accuracy": correct / total if total else 0.0,
        "balanced_accuracy": sum(recalls) / len(recalls) if recalls else 0.0,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "majority_baseline_accuracy": majority_correct / total if total else 0.0,
        "by_label": by_label,
        "target": by_label.get(target_author, {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0}),
    }


def evaluate_run(path: Path, spec: RunSpec, model: AuthorProfileModel, *, target_author: str) -> dict[str, Any]:
    feature_fn = FEATURE_FNS[spec.feature_family]
    chunk_rows_by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    book_votes: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)

    for item in jsonl_records(path):
        split = str(item["split"])
        if split == "train":
            continue
        pred, _scores = predict(model, feature_fn, str(item["text"]))
        gold = str(item["author"])
        title = str(item["title"])
        row = {
            "split": split,
            "gold": gold,
            "pred": pred,
            "title": title,
            "chunk_id": str(item["chunk_id"]),
        }
        chunk_rows_by_split[split].append(row)
        book_votes[(split, gold, title)][pred] += 1

    majority_author = max(model.train_docs_by_author.items(), key=lambda item: (item[1], item[0]))[0]
    metrics_by_split = {
        split: evaluate_predictions(rows, model.labels, target_author=target_author, majority_author=majority_author)
        for split, rows in chunk_rows_by_split.items()
    }
    book_rows_by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    for (split, gold, title), votes in book_votes.items():
        pred = max(votes.items(), key=lambda item: (item[1], item[0]))[0]
        book_rows_by_split[split].append({"split": split, "gold": gold, "pred": pred, "title": title})
    book_metrics_by_split = {
        split: evaluate_predictions(rows, model.labels, target_author=target_author, majority_author=majority_author)
        for split, rows in book_rows_by_split.items()
    }
    test_rows = chunk_rows_by_split.get("test", [])
    confusion = confusion_matrix(test_rows, model.labels)
    top_confusions = top_confusion_pairs(confusion, model.labels, limit=10)
    return {
        "spec": spec.__dict__,
        "labels": model.labels,
        "train_chunks": model.train_chunks,
        "train_docs_by_author": model.train_docs_by_author,
        "profile_feature_counts": model.profile_feature_counts,
        "chunk_metrics": metrics_by_split,
        "book_metrics": book_metrics_by_split,
        "test_confusion_matrix": confusion,
        "top_test_confusions": top_confusions,
    }


def confusion_matrix(rows: list[dict[str, str]], labels: list[str]) -> list[list[int]]:
    index = {label: pos for pos, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for row in rows:
        if row["gold"] in index and row["pred"] in index:
            matrix[index[row["gold"]]][index[row["pred"]]] += 1
    return matrix


def top_confusion_pairs(matrix: list[list[int]], labels: list[str], *, limit: int) -> list[dict[str, Any]]:
    pairs = []
    for gold_index, row in enumerate(matrix):
        for pred_index, count in enumerate(row):
            if gold_index == pred_index or not count:
                continue
            pairs.append({"gold": labels[gold_index], "pred": labels[pred_index], "count": count})
    pairs.sort(key=lambda item: (-item["count"], item["gold"], item["pred"]))
    return pairs[:limit]


def write_confusion_csv(path: Path, labels: list[str], matrix: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gold\\pred", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row])


def fmt(value: float) -> str:
    return f"{value * 100:.1f}%"


def metric(result: dict[str, Any], split: str, key: str, *, level: str = "chunk") -> float:
    metrics = result[f"{level}_metrics"].get(split, {})
    current: Any = metrics
    for part in key.split("."):
        current = current.get(part, 0.0) if isinstance(current, dict) else 0.0
    return float(current or 0.0)


def safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "item"


def write_svg_bar_chart(path: Path, title: str, rows: list[tuple[str, float]], *, width: int = 960) -> None:
    bar_height = 24
    gap = 10
    left = 260
    right = 90
    top = 52
    height = top + len(rows) * (bar_height + gap) + 35
    chart_width = width - left - right
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="30" font-family="Arial, sans-serif" font-size="20" font-weight="700">{html.escape(title)}</text>',
    ]
    for index, (label, value) in enumerate(rows):
        y = top + index * (bar_height + gap)
        bar_width = max(1, int(chart_width * max(0.0, min(1.0, value))))
        lines.append(f'<text x="24" y="{y + 17}" font-family="Arial, sans-serif" font-size="13">{html.escape(label)}</text>')
        lines.append(f'<rect x="{left}" y="{y}" width="{bar_width}" height="{bar_height}" fill="#2f6f73"/>')
        lines.append(f'<text x="{left + bar_width + 8}" y="{y + 17}" font-family="Arial, sans-serif" font-size="13">{fmt(value)}</text>')
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_gap_rows(results: dict[str, dict[str, Any]], methods: Iterable[str], *, masked_view: str) -> list[dict[str, Any]]:
    rows = []
    for method in methods:
        clean = results.get(f"{method}.clean")
        masked = results.get(f"{method}.{masked_view}")
        if not clean or not masked:
            continue
        clean_acc = metric(clean, "test", "accuracy")
        masked_acc = metric(masked, "test", "accuracy")
        clean_target = metric(clean, "test", "target.recall")
        masked_target = metric(masked, "test", "target.recall")
        rows.append(
            {
                "method": method,
                "clean_accuracy": clean_acc,
                "masked_accuracy": masked_acc,
                "accuracy_gap": clean_acc - masked_acc,
                "clean_target_recall": clean_target,
                "masked_target_recall": masked_target,
                "target_recall_gap": clean_target - masked_target,
            }
        )
    return rows


def write_markdown_report(
    path: Path,
    results: dict[str, dict[str, Any]],
    gap_rows: list[dict[str, Any]],
    chart_paths: dict[str, Path],
    *,
    target_author: str,
    masked_view: str,
) -> None:
    lines = [
        "# Author Style Baseline Benchmark",
        "",
        f"Generated: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %z')}",
        "",
        "## Purpose",
        "",
        f"This benchmark asks whether author signal survives after content masking. The `clean` view is an upper-bound diagnostic; `{masked_view}` is the current content-controlled candidate style meter.",
        "",
        "Classifier: nearest author profile with normalized sparse feature profiles trained only on the `train` split.",
        "",
        "## Key Tables",
        "",
        "### Test Metrics By System",
        "",
        "| System | View | Test Acc | Test Balanced Acc | Test Macro F1 | Target Precision | Target Recall | Target F1 | Proxy Target Recall | Book Test Acc | Majority Baseline |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, result in sorted(results.items()):
        spec = result["spec"]
        label = METHOD_LABELS.get(spec["method"], spec["method"])
        test_metrics = result["chunk_metrics"].get("test", {})
        target = test_metrics.get("target", {})
        proxy_target_recall = metric(result, "proxy_transfer", "target.recall")
        book_test_acc = metric(result, "test", "accuracy", level="book")
        lines.append(
            f"| {label} | {spec['view']} | {fmt(test_metrics.get('accuracy', 0.0))} | "
            f"{fmt(test_metrics.get('balanced_accuracy', 0.0))} | {fmt(test_metrics.get('macro_f1', 0.0))} | "
            f"{fmt(target.get('precision', 0.0))} | {fmt(target.get('recall', 0.0))} | {fmt(target.get('f1', 0.0))} | "
            f"{fmt(proxy_target_recall)} | {fmt(book_test_acc)} | {fmt(test_metrics.get('majority_baseline_accuracy', 0.0))} |"
        )

    lines.extend(
        [
            "",
        f"### Clean vs {masked_view} Gap",
            "",
        f"| Method | Clean Acc | {masked_view} Acc | Acc Gap | Clean Target Recall | {masked_view} Target Recall | Target Recall Gap |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in gap_rows:
        lines.append(
            f"| {METHOD_LABELS.get(row['method'], row['method'])} | {fmt(row['clean_accuracy'])} | "
            f"{fmt(row['masked_accuracy'])} | {fmt(row['accuracy_gap'])} | "
            f"{fmt(row['clean_target_recall'])} | {fmt(row['masked_target_recall'])} | {fmt(row['target_recall_gap'])} |"
        )

    lines.extend(["", "## Graphs", ""])
    for name, chart_path in chart_paths.items():
        rel = chart_path.relative_to(path.parent).as_posix()
        lines.append(f"![{name}]({rel})")
        lines.append("")

    lines.extend(["## Per-Method Notes", ""])
    for row in gap_rows:
        method = row["method"]
        clean = results[f"{method}.clean"]
        masked = results[f"{method}.{masked_view}"]
        lines.extend(
            [
                f"### {METHOD_LABELS.get(method, method)}",
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
            "- High `clean` scores can reflect theme, setting, character names, or book identity.",
            "- The style-meter candidate is the method whose target-author recall remains high after entity masking and whose accuracy does not collapse relative to clean.",
            "- `proxy_transfer` contains held-out target-author books, so proxy target recall is a stricter sanity check for whether the model still recognizes the target author outside train/dev/test books.",
            "- Book-level accuracy is reported because chunk-level results can overstate confidence when many chunks come from the same book.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def format_confusions(confusions: list[dict[str, Any]]) -> str:
    if not confusions:
        return "none"
    return "; ".join(f"{item['gold']} -> {item['pred']} ({item['count']})" for item in confusions[:5])


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark baseline author-style classifiers on clean and masked chunks.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("generated/style_research/benchmarks/author_style_baselines"))
    parser.add_argument("--target-author", default=TARGET_AUTHOR_DEFAULT)
    parser.add_argument("--views", default="clean,entity_masked", help="Comma-separated views to benchmark.")
    parser.add_argument("--masked-view", default="entity_masked", help="Masked view to compare against clean in the gap table.")
    parser.add_argument("--report-only", action="store_true", help="Regenerate Markdown and charts from an existing author_baseline_results.json without rerunning classifiers.")
    parser.add_argument("--max-char-profile-features", type=int, default=2500)
    parser.add_argument("--max-interpretable-profile-features", type=int, default=800)
    args = parser.parse_args()

    paths = chunk_paths(args.dataset_root)
    views = tuple(view.strip() for view in args.views.split(",") if view.strip())
    missing = [str(paths[view]) for view in views if view not in paths or not paths[view].exists()]
    if missing:
        raise SystemExit("missing chunk files: " + ", ".join(missing))

    specs = default_run_specs(
        views,
        max_char_features=args.max_char_profile_features,
        max_interpretable_features=args.max_interpretable_profile_features,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods = ("char_ngrams", "punctuation_dialogue", "length_shape", "function_chars", "combined_interpretable")

    if args.report_only:
        result_path = args.output_dir / "author_baseline_results.json"
        if not result_path.exists():
            raise SystemExit(f"missing result file for --report-only: {result_path}")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        results = payload["results"]
        gap_rows = payload.get("masked_gap") or build_gap_rows(results, methods, masked_view=args.masked_view)
        charts_dir = args.output_dir / "charts"
        chart_paths = {
            "Test Accuracy": charts_dir / "test_accuracy.svg",
            "Target Recall": charts_dir / "target_recall.svg",
            "Clean To Entity-Masked Accuracy Gap": charts_dir / "clean_entity_accuracy_gap.svg",
        }
        write_svg_bar_chart(
            chart_paths["Test Accuracy"],
            "Test Accuracy By System",
            [
                (f"{METHOD_LABELS.get(result['spec']['method'], result['spec']['method'])} / {result['spec']['view']}", metric(result, "test", "accuracy"))
                for _key, result in sorted(results.items())
            ],
        )
        write_svg_bar_chart(
            chart_paths["Target Recall"],
            f"{args.target_author} Test Recall By System",
            [
                (f"{METHOD_LABELS.get(result['spec']['method'], result['spec']['method'])} / {result['spec']['view']}", metric(result, "test", "target.recall"))
                for _key, result in sorted(results.items())
            ],
        )
        write_svg_bar_chart(
            chart_paths["Clean To Entity-Masked Accuracy Gap"],
            f"Clean To {args.masked_view} Accuracy Gap",
            [(METHOD_LABELS.get(row["method"], row["method"]), row["accuracy_gap"]) for row in gap_rows],
        )
        write_markdown_report(
            args.output_dir / "author_baseline_results.md",
            results,
            gap_rows,
            chart_paths,
            target_author=args.target_author,
            masked_view=args.masked_view,
        )
        print(
            json.dumps(
                {
                    "report": str(args.output_dir / "author_baseline_results.md"),
                    "results": str(result_path),
                    "runs": len(results),
                    "report_only": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    results: dict[str, dict[str, Any]] = {}
    for spec in specs:
        print(f"running {spec.key}", flush=True)
        path = paths[spec.view]
        feature_fn = FEATURE_FNS[spec.feature_family]
        model = train_author_profile_model(path, feature_fn, max_profile_features=spec.max_profile_features)
        result = evaluate_run(path, spec, model, target_author=args.target_author)
        results[spec.key] = result
        (args.output_dir / "author_baseline_results.partial.json").write_text(
            json.dumps(
                {
                    "target_author": args.target_author,
                    "dataset_root": str(args.dataset_root),
                    "views": views,
                    "completed_runs": sorted(results),
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        write_confusion_csv(
            args.output_dir / "confusion_matrices" / f"{safe_filename(spec.key)}.test.csv",
            result["labels"],
            result["test_confusion_matrix"],
        )

    gap_rows = build_gap_rows(results, methods, masked_view=args.masked_view)
    (args.output_dir / "author_baseline_results.json").write_text(
        json.dumps(
            {
                "target_author": args.target_author,
                "dataset_root": str(args.dataset_root),
                "views": views,
                "masked_view": args.masked_view,
                "results": results,
                "masked_gap": gap_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    charts_dir = args.output_dir / "charts"
    chart_paths = {
        "Test Accuracy": charts_dir / "test_accuracy.svg",
        "Target Recall": charts_dir / "target_recall.svg",
        "Clean To Entity-Masked Accuracy Gap": charts_dir / "clean_entity_accuracy_gap.svg",
    }
    write_svg_bar_chart(
        chart_paths["Test Accuracy"],
        "Test Accuracy By System",
        [
            (f"{METHOD_LABELS.get(result['spec']['method'], result['spec']['method'])} / {result['spec']['view']}", metric(result, "test", "accuracy"))
            for _key, result in sorted(results.items())
        ],
    )
    write_svg_bar_chart(
        chart_paths["Target Recall"],
        f"{args.target_author} Test Recall By System",
        [
            (f"{METHOD_LABELS.get(result['spec']['method'], result['spec']['method'])} / {result['spec']['view']}", metric(result, "test", "target.recall"))
            for _key, result in sorted(results.items())
        ],
    )
    write_svg_bar_chart(
        chart_paths["Clean To Entity-Masked Accuracy Gap"],
        f"Clean To {args.masked_view} Accuracy Gap",
        [(METHOD_LABELS.get(row["method"], row["method"]), row["accuracy_gap"]) for row in gap_rows],
    )
    write_markdown_report(
        args.output_dir / "author_baseline_results.md",
        results,
        gap_rows,
        chart_paths,
        target_author=args.target_author,
        masked_view=args.masked_view,
    )
    print(
        json.dumps(
            {
                "report": str(args.output_dir / "author_baseline_results.md"),
                "results": str(args.output_dir / "author_baseline_results.json"),
                "runs": len(results),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
