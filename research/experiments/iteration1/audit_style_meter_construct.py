#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import normalize


from experiments.shared.paths import RESEARCH_ROOT


REPO_ROOT = RESEARCH_ROOT

from workflows.benchmark_author_style import (  # noqa: E402
    FUNCTION_CHARS,
    FUNCTION_WORDS,
)


WHITESPACE_RE = re.compile(r"\s+")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
PLACEHOLDER_RE = re.compile(r"某|<(?:NUM|LATIN|TERM)>")
SPEECH_OR_BEAT_TERMS = (
    "说",
    "道",
    "问",
    "答",
    "喊",
    "叫",
    "心想",
    "点头",
    "摇头",
    "转头",
)
GENERAL_DISCOURSE_TERMS = set(FUNCTION_WORDS) | {
    "只得",
    "继而",
    "犹如",
    "就像",
    "以后",
    "前去",
    "有点",
    "一会",
    "则是",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether a frozen exact-character authorship scorer contains "
            "content/theme shortcuts that weaken its validity as a style meter."
        )
    )
    parser.add_argument("--scorer-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mask-terms", type=Path, required=True)
    parser.add_argument("--target-author", default="非天夜翔")
    parser.add_argument("--top-k", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluation-summary", type=Path)
    parser.add_argument(
        "--ablation-splits",
        default="",
        help=(
            "Optional comma-separated splits for frozen-model grouped-feature "
            "ablation, for example dev,test."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc


def target_mask_terms(path: Path, target_author: str) -> set[str]:
    payload = read_json(path)
    terms: set[str] = {
        str(value)
        for value in payload.get("global_terms", {}).get("entity_terms_v2", [])
        if str(value)
    }
    books = payload.get("books", [])
    records = books.values() if isinstance(books, dict) else books
    for record in records:
        if not isinstance(record, dict):
            continue
        if record.get("author") != target_author:
            continue
        for key in ("entity_terms", "entity_terms_v2", "topic_terms"):
            values = record.get(key, [])
            if isinstance(values, list):
                terms.update(str(value) for value in values if str(value))
    return terms


def matches_known_mask_term(feature: str, terms: set[str]) -> bool:
    cjk = "".join(CJK_RE.findall(feature))
    if len(cjk) < 2:
        return False
    return any(cjk in term or term in cjk for term in terms if len(term) >= 2)


def surface_category(
    feature: str,
    *,
    target_book_count: int,
    target_book_total: int,
    known_terms: set[str],
) -> str:
    cjk = "".join(CJK_RE.findall(feature))
    if PLACEHOLDER_RE.search(feature):
        return "mask_artifact"
    if matches_known_mask_term(feature, known_terms):
        return "known_entity_or_topic_fragment"
    if any(term in feature for term in SPEECH_OR_BEAT_TERMS):
        return "dialogue_structure"
    if not cjk and is_punctuation_only(feature):
        return "punctuation_only"
    if not cjk and is_punctuation_or_symbol_only(feature):
        return "source_format_symbol"
    if cjk and all(char in FUNCTION_CHARS for char in cjk):
        return "function_grammar"
    if any(term == cjk or term in feature for term in GENERAL_DISCOURSE_TERMS):
        return "general_discourse"
    # Recurring characters can span several books in a series; several target books
    # is still too concentrated to treat a lexical n-gram as author-general.
    low_dispersion_limit = max(4, round(target_book_total * 0.15))
    if target_book_count <= low_dispersion_limit:
        return "low_book_dispersion_lexical"
    return "recurrent_lexical_ambiguous"


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def cjk_core(feature: str) -> str:
    return "".join(CJK_RE.findall(feature))


def is_punctuation_only(feature: str) -> bool:
    return bool(feature) and all(
        unicodedata.category(char).startswith("P") for char in feature
    )


def is_punctuation_or_symbol_only(feature: str) -> bool:
    return bool(feature) and all(
        unicodedata.category(char).startswith(("P", "S")) for char in feature
    )


def is_cjk_core_with_punctuation(feature: str, core: str) -> bool:
    return cjk_core(feature) == core and all(
        CJK_RE.fullmatch(char) or unicodedata.category(char).startswith("P")
        for char in feature
    )


def interpretation_group_key(feature: str) -> str:
    if is_punctuation_only(feature):
        return "pure_punctuation"
    if is_punctuation_or_symbol_only(feature) and not cjk_core(feature):
        return "source_format_symbols"
    core = cjk_core(feature)
    if core and is_cjk_core_with_punctuation(feature, core):
        return f"cjk_core:{core}"
    return f"exact:{feature}"


def build_interpretation_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[interpretation_group_key(str(row["feature"]))].append(row)

    output: list[dict[str, Any]] = []
    for key, members in grouped.items():
        ordered = sorted(
            members,
            key=lambda row: (-float(row["coefficient"]), int(row["rank"])),
        )
        strongest = ordered[0]
        core = key.removeprefix("cjk_core:") if key.startswith("cjk_core:") else ""
        if key == "pure_punctuation":
            label = "纯标点（合并）"
        elif key == "source_format_symbols":
            label = "格式符号（合并）"
        elif core:
            label = core
        else:
            label = str(strongest["feature"])
        category_mass: Counter[str] = Counter()
        for row in ordered:
            category_mass[str(row["category"])] += float(row["coefficient"])
        output.append(
            {
                "group_key": key,
                "label": label,
                "member_count": len(ordered),
                "strongest_feature": str(strongest["feature"]),
                "strongest_coefficient": float(strongest["coefficient"]),
                "positive_coefficient_mass": sum(
                    float(row["coefficient"]) for row in ordered
                ),
                "examples": " / ".join(str(row["feature"]) for row in ordered[:5]),
                "dominant_category": category_mass.most_common(1)[0][0],
                "categories": ";".join(sorted(category_mass)),
            }
        )
    output.sort(
        key=lambda row: (
            -float(row["strongest_coefficient"]),
            -float(row["positive_coefficient_mass"]),
            str(row["group_key"]),
        )
    )
    for rank, row in enumerate(output, start=1):
        row["group_rank"] = rank
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_svg(path: Path, category_rows: list[dict[str, Any]]) -> None:
    width = 980
    left = 300
    right = 110
    top = 58
    row_height = 42
    height = top + row_height * len(category_rows) + 72
    max_mass = max((float(row["positive_weight_mass"]) for row in category_rows), default=1.0)
    colors = {
        "dialogue_structure": "#2878b5",
        "punctuation_only": "#55a868",
        "function_grammar": "#4c956c",
        "general_discourse": "#76b041",
        "mask_artifact": "#c44e52",
        "known_entity_or_topic_fragment": "#dd8452",
        "low_book_dispersion_lexical": "#e6a23c",
        "recurrent_lexical_ambiguous": "#8172b2",
    }
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="32" font-family="sans-serif" font-size="20" font-weight="700" fill="#202124">Top positive target-feature weight by diagnostic category</text>',
    ]
    chart_width = width - left - right
    for index, row in enumerate(category_rows):
        y = top + index * row_height
        category = str(row["category"])
        mass = float(row["positive_weight_mass"])
        bar_width = chart_width * safe_div(mass, max_mass)
        color = colors.get(category, "#7f8c8d")
        elements.extend(
            [
                f'<text x="24" y="{y + 20}" font-family="sans-serif" font-size="14" fill="#30343b">{html.escape(category)}</text>',
                f'<rect x="{left}" y="{y + 4}" width="{bar_width:.2f}" height="22" fill="{color}" rx="2"/>',
                f'<text x="{left + bar_width + 8:.2f}" y="{y + 20}" font-family="sans-serif" font-size="13" fill="#30343b">{mass:.2f} ({float(row["weight_share"]):.1%})</text>',
            ]
        )
    elements.append(
        f'<text x="24" y="{height - 24}" font-family="sans-serif" font-size="12" fill="#5f6368">Categories are deterministic diagnostics, not gold linguistic annotations.</text>'
    )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def write_top_features_svg(
    path: Path, rows: list[dict[str, Any]], *, limit: int = 20
) -> None:
    selected = rows[:limit]
    width = 1180
    left = 220
    right = 250
    top = 126
    row_height = 34
    height = top + row_height * len(selected) + 72
    chart_width = width - left - right
    max_weight = max((float(row["coefficient"]) for row in selected), default=1.0)
    colors = {
        "dialogue_structure": "#0072b2",
        "punctuation_only": "#56b4e9",
        "function_grammar": "#009e73",
        "general_discourse": "#6a994e",
        "mask_artifact": "#c44e52",
        "known_entity_or_topic_fragment": "#d55e00",
        "low_book_dispersion_lexical": "#e69f00",
        "recurrent_lexical_ambiguous": "#8172b2",
    }
    labels = {
        "dialogue_structure": "对话结构",
        "punctuation_only": "标点",
        "function_grammar": "功能语法",
        "general_discourse": "一般语篇",
        "mask_artifact": "遮蔽伪影",
        "known_entity_or_topic_fragment": "实体／题材风险",
        "low_book_dispersion_lexical": "低跨书词汇",
        "recurrent_lexical_ambiguous": "待解释词汇",
    }
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<g font-family="Noto Sans CJK SC, PingFang SC, Microsoft YaHei, sans-serif">',
        '<text x="32" y="40" font-size="25" font-weight="700" fill="#17202a">非天夜翔：作者识别模型的前 20 个正向相邻字符片段</text>',
        '<text x="32" y="70" font-size="15" fill="#4d5966">横条为目标类别的线性权重；权重表示判别贡献，不等于出现频率或完整风格定义</text>',
    ]
    legend = [
        ("#0072b2", "对话结构"),
        ("#009e73", "功能语法／一般语篇"),
        ("#e69f00", "内容风险"),
        ("#8172b2", "待解释词汇"),
    ]
    legend_x = 32
    for color, label in legend:
        elements.extend(
            [
                f'<rect x="{legend_x}" y="88" width="14" height="14" rx="2" fill="{color}"/>',
                f'<text x="{legend_x + 21}" y="100" font-size="13" fill="#4d5966">{label}</text>',
            ]
        )
        legend_x += 166

    for index, row in enumerate(selected):
        y = top + index * row_height
        category = str(row["category"])
        color = colors.get(category, "#7f8c8d")
        coefficient = float(row["coefficient"])
        bar_width = chart_width * safe_div(coefficient, max_weight)
        feature = html.escape(str(row["feature"]))
        category_label = labels.get(category, category)
        if index % 2:
            elements.append(
                f'<rect x="24" y="{y - 4}" width="1132" height="31" fill="#f7f8f9"/>'
            )
        elements.extend(
            [
                f'<text x="45" y="{y + 18}" font-size="12" fill="#7a858f">{index + 1}</text>',
                f'<text x="198" y="{y + 18}" text-anchor="end" font-size="16" font-weight="600" fill="#26323d">{feature}</text>',
                f'<rect x="{left}" y="{y + 2}" width="{bar_width:.2f}" height="21" rx="2" fill="{color}" opacity="0.88"/>',
                f'<text x="{left + bar_width + 8:.2f}" y="{y + 18}" font-size="13" font-weight="600" fill="#26323d">{coefficient:.2f}</text>',
                f'<text x="1140" y="{y + 18}" text-anchor="end" font-size="12" fill="#5f6b75">{category_label}</text>',
            ]
        )
    elements.extend(
        [
            f'<text x="32" y="{height - 25}" font-size="12" fill="#697784">诊断类别由确定性规则生成，并非人工金标准；相邻的二、三、四字片段会重叠。</text>',
            "</g>",
            "</svg>",
        ]
    )
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def write_grouped_features_svg(
    path: Path, rows: list[dict[str, Any]], *, limit: int = 18
) -> None:
    selected = rows[:limit]
    width = 1280
    left = 250
    right = 360
    top = 126
    row_height = 38
    height = top + row_height * len(selected) + 76
    chart_width = width - left - right
    max_weight = max(
        (float(row["strongest_coefficient"]) for row in selected), default=1.0
    )
    colors = {
        "dialogue_structure": "#0072b2",
        "punctuation_only": "#c44e52",
        "source_format_symbol": "#8c564b",
        "function_grammar": "#009e73",
        "general_discourse": "#6a994e",
        "mask_artifact": "#c44e52",
        "known_entity_or_topic_fragment": "#d55e00",
        "low_book_dispersion_lexical": "#e69f00",
        "recurrent_lexical_ambiguous": "#8172b2",
    }
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<g font-family="Noto Sans CJK SC, PingFang SC, Microsoft YaHei, sans-serif">',
        '<text x="32" y="40" font-size="25" font-weight="700" fill="#17202a">非天夜翔：权重最高的相邻字符片段家族</text>',
        '<text x="32" y="70" font-size="15" fill="#4d5966">同一汉字核心的二至四字窗口合为一行；横条取家族中最强单项权重，避免重复相加夸大影响</text>',
        '<text x="32" y="99" font-size="13" fill="#697784">纯标点单列为来源／排版敏感诊断，不解释为作者写作规则</text>',
    ]
    for index, row in enumerate(selected):
        y = top + index * row_height
        category = str(row["dominant_category"])
        if row["group_key"] == "pure_punctuation":
            color = "#c44e52"
        elif row["group_key"] == "source_format_symbols":
            color = "#8c564b"
        else:
            color = colors.get(category, "#7f8c8d")
        coefficient = float(row["strongest_coefficient"])
        bar_width = chart_width * safe_div(coefficient, max_weight)
        label = html.escape(str(row["label"]))
        examples = str(row["examples"])
        if len(examples) > 34:
            examples = examples[:33] + "…"
        detail = html.escape(f"{row['member_count']} 项：{examples}")
        if index % 2:
            elements.append(
                f'<rect x="24" y="{y - 4}" width="1232" height="34" fill="#f7f8f9"/>'
            )
        elements.extend(
            [
                f'<text x="45" y="{y + 20}" font-size="12" fill="#7a858f">{index + 1}</text>',
                f'<text x="225" y="{y + 20}" text-anchor="end" font-size="15" font-weight="600" fill="#26323d">{label}</text>',
                f'<rect x="{left}" y="{y + 3}" width="{bar_width:.2f}" height="22" rx="2" fill="{color}" opacity="0.88"/>',
                f'<text x="{left + bar_width + 8:.2f}" y="{y + 20}" font-size="13" font-weight="600" fill="#26323d">{coefficient:.2f}</text>',
                f'<text x="1248" y="{y + 20}" text-anchor="end" font-size="11" fill="#5f6b75">{detail}</text>',
            ]
        )
    elements.extend(
        [
            f'<text x="32" y="{height - 27}" font-size="12" fill="#697784">这是一张解释图，不改变冻结分类器；完整原始系数仍保留在审计 CSV 中。</text>',
            "</g>",
            "</svg>",
        ]
    )
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def evaluation_context(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = read_json(path)
    rows: list[dict[str, Any]] = []
    for combination in payload.get("combinations", []):
        rows.append(
            {
                "method_id": combination["method_id"],
                "own_success": combination["arm_summaries"]["own_author_reconstruction"]["deterministic_style_success"]["estimate"],
                "cross_success": combination["arm_summaries"]["cross_author_transfer"]["deterministic_style_success"]["estimate"],
                "own_lift": combination["arm_summaries"]["own_author_reconstruction"]["mean_paired_margin_lift"],
                "cross_lift": combination["arm_summaries"]["cross_author_transfer"]["mean_paired_margin_lift"],
            }
        )
    return {
        "path": str(path.resolve().relative_to(REPO_ROOT)),
        "sha256": sha256_file(path),
        "rows": rows,
    }


def parse_ablation_splits(value: str) -> tuple[str, ...]:
    splits = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    if "train" in splits:
        raise ValueError("train is not allowed as a grouped-feature ablation split")
    return splits


def prediction_metrics(
    classifier: Any,
    matrix: Any,
    labels: list[str],
    *,
    target_author: str,
) -> dict[str, float | int]:
    truth = np.asarray(labels)
    predictions = classifier.predict(matrix)
    target_truth = truth == target_author
    target_predictions = predictions == target_author
    true_positive = int(np.sum(target_truth & target_predictions))
    false_positive = int(np.sum(~target_truth & target_predictions))
    false_negative = int(np.sum(target_truth & ~target_predictions))
    precision = safe_div(true_positive, true_positive + false_positive)
    recall = safe_div(true_positive, true_positive + false_negative)
    f1 = safe_div(2.0 * precision * recall, precision + recall)

    class_labels = [str(value) for value in classifier.classes_]
    target_index = class_labels.index(target_author)
    scores = classifier.decision_function(matrix)
    target_scores = scores[:, target_index]
    other_scores = np.delete(scores, target_index, axis=1)
    target_margin = target_scores - np.max(other_scores, axis=1)
    return {
        "rows": len(labels),
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "target_precision": precision,
        "target_recall": recall,
        "target_f1": f1,
        "target_true_positive": true_positive,
        "target_false_positive": false_positive,
        "target_false_negative": false_negative,
        "target_mean_margin": float(np.mean(target_margin[target_truth])),
        "comparison_mean_target_margin": float(np.mean(target_margin[~target_truth])),
    }


def zero_columns_and_renormalize(matrix: Any, columns: np.ndarray) -> Any:
    ablated = matrix.copy().tocsr()
    if len(columns):
        remove = np.isin(ablated.indices, columns)
        ablated.data[remove] = 0.0
        ablated.eliminate_zeros()
        normalize(ablated, norm="l2", copy=False)
    return ablated


def run_group_ablation(
    vectorizer: Any,
    classifier: Any,
    rows_by_split: dict[str, list[tuple[str, str]]],
    *,
    target_author: str,
) -> dict[str, Any]:
    features = [str(value) for value in vectorizer.get_feature_names_out()]
    shuo_columns = np.asarray(
        [
            index
            for index, feature in enumerate(features)
            if is_cjk_core_with_punctuation(feature, "说")
        ],
        dtype=np.int64,
    )
    punctuation_columns = np.asarray(
        [
            index
            for index, feature in enumerate(features)
            if is_punctuation_only(feature)
        ],
        dtype=np.int64,
    )
    source_format_columns = np.asarray(
        [
            index
            for index, feature in enumerate(features)
            if is_punctuation_or_symbol_only(feature)
            and not is_punctuation_only(feature)
            and not cjk_core(feature)
        ],
        dtype=np.int64,
    )
    groups = {
        "shuo_punctuation_family": shuo_columns,
        "pure_punctuation_family": punctuation_columns,
        "source_format_symbol_family": source_format_columns,
        "shuo_and_pure_punctuation": np.union1d(
            shuo_columns, punctuation_columns
        ),
        "all_audited_surface_families": np.union1d(
            np.union1d(shuo_columns, punctuation_columns), source_format_columns
        ),
    }
    output: dict[str, Any] = {
        "schema_version": 1,
        "method": (
            "zero every frozen TF-IDF column in the registered family, then "
            "L2-renormalize without refitting the classifier"
        ),
        "scope": "post_hoc_frozen_model_sensitivity_not_new_model_selection",
        "feature_groups": {
            name: {
                "feature_count": int(len(columns)),
                "examples": [features[index] for index in columns[:20]],
            }
            for name, columns in groups.items()
        },
        "splits": {},
    }

    for split, records in rows_by_split.items():
        texts = [text for text, _ in records]
        labels = [label for _, label in records]
        matrix = vectorizer.transform(texts).tocsr()
        baseline = prediction_metrics(
            classifier, matrix, labels, target_author=target_author
        )
        ablations: dict[str, Any] = {}
        for name, columns in groups.items():
            ablated_matrix = zero_columns_and_renormalize(matrix, columns)
            metrics = prediction_metrics(
                classifier,
                ablated_matrix,
                labels,
                target_author=target_author,
            )
            metrics["accuracy_delta"] = float(metrics["accuracy"]) - float(
                baseline["accuracy"]
            )
            metrics["balanced_accuracy_delta"] = float(
                metrics["balanced_accuracy"]
            ) - float(baseline["balanced_accuracy"])
            metrics["target_f1_delta"] = float(metrics["target_f1"]) - float(
                baseline["target_f1"]
            )
            metrics["target_mean_margin_delta"] = float(
                metrics["target_mean_margin"]
            ) - float(baseline["target_mean_margin"])
            ablations[name] = metrics
        output["splits"][split] = {
            "baseline": baseline,
            "ablations": ablations,
        }
    return output


def main() -> None:
    args = parse_args()
    scorer_dir = args.scorer_dir.expanduser().resolve()
    dataset_path = args.dataset.expanduser().resolve()
    mask_terms_path = args.mask_terms.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    ablation_splits = parse_ablation_splits(args.ablation_splits)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = read_json(scorer_dir / "scorer_config.json")
    expected_dataset_hash = config["training_contract"]["dataset_sha256"]
    observed_dataset_hash = sha256_file(dataset_path)
    if observed_dataset_hash != expected_dataset_hash:
        raise ValueError("Dataset does not match the scorer's frozen training binding")
    if config.get("target_author") != args.target_author:
        raise ValueError("Target author does not match the frozen scorer")

    vectorizer = joblib.load(scorer_dir / "vectorizer.joblib")
    classifier = joblib.load(scorer_dir / "classifier.joblib")
    labels = [str(value) for value in classifier.classes_]
    target_index = labels.index(args.target_author)
    features = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[target_index]
    positive_indices = np.flatnonzero(coefficients > 0)
    ranked_indices = positive_indices[np.argsort(coefficients[positive_indices])[::-1]]
    selected_indices = ranked_indices[: args.top_k]
    selected_features = [str(features[index]) for index in selected_indices]
    selected_set = set(selected_features)

    target_chunk_hits: Counter[str] = Counter()
    comparison_chunk_hits: Counter[str] = Counter()
    target_books: set[str] = set()
    comparison_books: set[str] = set()
    comparison_authors: set[str] = set()
    target_feature_books: dict[str, set[str]] = defaultdict(set)
    comparison_feature_books: dict[str, set[str]] = defaultdict(set)
    comparison_feature_authors: dict[str, set[str]] = defaultdict(set)
    ablation_rows: dict[str, list[tuple[str, str]]] = {
        split: [] for split in ablation_splits
    }
    target_chunks = 0
    comparison_chunks = 0

    for row in iter_jsonl(dataset_path):
        split = str(row.get("split", ""))
        author = str(row["author"])
        raw_text = str(row["text"])
        if split in ablation_rows:
            ablation_rows[split].append((raw_text, author))
        if split != "train":
            continue
        title = str(row["title"])
        text = WHITESPACE_RE.sub("", raw_text)
        hits = {feature for feature in selected_set if feature in text}
        if author == args.target_author:
            target_chunks += 1
            target_books.add(title)
            for feature in hits:
                target_chunk_hits[feature] += 1
                target_feature_books[feature].add(title)
        else:
            comparison_chunks += 1
            comparison_authors.add(author)
            comparison_books.add(f"{author}\u241f{title}")
            for feature in hits:
                comparison_chunk_hits[feature] += 1
                comparison_feature_books[feature].add(f"{author}\u241f{title}")
                comparison_feature_authors[feature].add(author)

    known_terms = target_mask_terms(mask_terms_path, args.target_author)
    rows: list[dict[str, Any]] = []
    for rank, index in enumerate(selected_indices, start=1):
        feature = str(features[index])
        target_book_count = len(target_feature_books[feature])
        category = surface_category(
            feature,
            target_book_count=target_book_count,
            target_book_total=len(target_books),
            known_terms=known_terms,
        )
        rows.append(
            {
                "rank": rank,
                "feature": feature,
                "coefficient": float(coefficients[index]),
                "category": category,
                "target_chunk_hits": target_chunk_hits[feature],
                "target_chunk_rate": safe_div(target_chunk_hits[feature], target_chunks),
                "target_book_hits": target_book_count,
                "target_book_rate": safe_div(target_book_count, len(target_books)),
                "comparison_chunk_hits": comparison_chunk_hits[feature],
                "comparison_chunk_rate": safe_div(comparison_chunk_hits[feature], comparison_chunks),
                "comparison_book_hits": len(comparison_feature_books[feature]),
                "comparison_author_hits": len(comparison_feature_authors[feature]),
            }
        )
    interpretation_groups = build_interpretation_groups(rows)

    missing_ablation_splits = [
        split for split, records in ablation_rows.items() if not records
    ]
    if missing_ablation_splits:
        raise ValueError(
            "requested ablation splits are absent from the dataset: "
            + ", ".join(missing_ablation_splits)
        )
    group_ablation = (
        run_group_ablation(
            vectorizer,
            classifier,
            ablation_rows,
            target_author=args.target_author,
        )
        if ablation_rows
        else None
    )

    category_counts: Counter[str] = Counter()
    category_mass: Counter[str] = Counter()
    for row in rows:
        category_counts[str(row["category"])] += 1
        category_mass[str(row["category"])] += float(row["coefficient"])
    total_mass = sum(category_mass.values())
    category_rows = [
        {
            "category": category,
            "feature_count": category_counts[category],
            "feature_share": safe_div(category_counts[category], len(rows)),
            "positive_weight_mass": category_mass[category],
            "weight_share": safe_div(category_mass[category], total_mass),
        }
        for category in sorted(category_mass, key=category_mass.get, reverse=True)
    ]

    risky_categories = {
        "mask_artifact",
        "source_format_symbol",
        "known_entity_or_topic_fragment",
        "low_book_dispersion_lexical",
    }
    risky_top_20 = sum(row["category"] in risky_categories for row in rows[:20])
    risky_top_50 = sum(row["category"] in risky_categories for row in rows[:50])
    risky_mass = sum(
        float(row["positive_weight_mass"])
        for row in category_rows
        if row["category"] in risky_categories
    )
    evaluation = evaluation_context(
        args.evaluation_summary.expanduser().resolve()
        if args.evaluation_summary
        else None
    )
    audit = {
        "schema_version": 1,
        "status": "construct_validity_concern",
        "scope": "descriptive_post_outcome_diagnostic_not_a_new_style_meter",
        "target_author": args.target_author,
        "top_k": len(rows),
        "scorer": {
            "directory": str(scorer_dir.relative_to(REPO_ROOT)),
            "classifier_sha256": sha256_file(scorer_dir / "classifier.joblib"),
            "vectorizer_sha256": sha256_file(scorer_dir / "vectorizer.joblib"),
            "config_sha256": sha256_file(scorer_dir / "scorer_config.json"),
        },
        "dataset": {
            "path": str(dataset_path.relative_to(REPO_ROOT)),
            "sha256": observed_dataset_hash,
            "target_train_chunks": target_chunks,
            "target_train_books": len(target_books),
            "comparison_train_chunks": comparison_chunks,
            "comparison_train_books": len(comparison_books),
            "comparison_train_authors": len(comparison_authors),
        },
        "diagnostics": {
            "risky_features_top_20": risky_top_20,
            "risky_features_top_50": risky_top_50,
            "risky_positive_weight_share_top_k": safe_div(risky_mass, total_mass),
            "category_summary": category_rows,
            "interpretation_groups": interpretation_groups,
            "group_ablation": group_ablation,
        },
        "interpretation": (
            "Book-disjoint attribution accuracy demonstrates predictive signal, "
            "but the target coefficient vector still contains residual content "
            "shortcuts and lexical features with low target-book dispersion. The "
            "scorer therefore cannot by itself distinguish authorial style from "
            "residual content."
        ),
        "evaluation_context": evaluation,
    }

    write_csv(output_dir / "target_positive_features.csv", rows)
    write_csv(
        output_dir / "target_positive_feature_groups.csv", interpretation_groups
    )
    write_csv(output_dir / "category_summary.csv", category_rows)
    write_svg(output_dir / "category_weight_mass.svg", category_rows)
    write_top_features_svg(output_dir / "top_positive_features.svg", rows)
    write_grouped_features_svg(
        output_dir / "top_positive_feature_groups.svg", interpretation_groups
    )
    if group_ablation is not None:
        (output_dir / "group_ablation.json").write_text(
            json.dumps(group_ablation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        ablation_csv_rows: list[dict[str, Any]] = []
        for split, split_payload in group_ablation["splits"].items():
            baseline = split_payload["baseline"]
            ablation_csv_rows.append(
                {
                    "split": split,
                    "variant": "baseline",
                    "feature_count": 0,
                    **baseline,
                    "accuracy_delta": 0.0,
                    "balanced_accuracy_delta": 0.0,
                    "target_f1_delta": 0.0,
                    "target_mean_margin_delta": 0.0,
                }
            )
            for name, metrics in split_payload["ablations"].items():
                ablation_csv_rows.append(
                    {
                        "split": split,
                        "variant": name,
                        "feature_count": group_ablation["feature_groups"][name][
                            "feature_count"
                        ],
                        **metrics,
                    }
                )
        write_csv(output_dir / "group_ablation.csv", ablation_csv_rows)
    (output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    report_lines = [
        "# Frozen Style-Meter Construct-Validity Audit",
        "",
        f"- Status: **{audit['status']}**",
        f"- Scope: `{audit['scope']}`",
        f"- Target author: `{args.target_author}`",
        f"- Frozen scorer: `{config['scorer_id']}`",
        f"- Audited positive features: top {len(rows):,}",
        f"- Target train evidence: {target_chunks:,} chunks / {len(target_books)} books",
        f"- Comparison train evidence: {comparison_chunks:,} chunks / {len(comparison_authors)} authors",
        "",
        "This audit asks whether a high-accuracy authorship classifier is a valid",
        "style construct. It does not refit the scorer, change the threshold, or",
        "retroactively rescore any transfer method.",
        "",
        "## Diagnostic Summary",
        "",
        f"- Risk-flagged features among top 20: **{risky_top_20}/20**",
        f"- Risk-flagged features among top 50: **{risky_top_50}/50**",
        f"- Risk-flagged positive-weight share in top {len(rows)}: **{safe_div(risky_mass, total_mass):.1%}**",
        "",
        "## Punctuation-Grouped Interpretation",
        "",
        "Exact 2-4 character windows overlap. For interpretation, rows sharing the",
        "same Chinese-character core and differing only in punctuation are grouped.",
        "Punctuation-only and symbol-only rows are kept in separate source/formatting-",
        "sensitive families. The chart ranks each group by its strongest constituent",
        "coefficient rather than summing correlated weights. This grouping does not",
        "alter the frozen classifier.",
        "",
        "![Grouped highest-weight target features](top_positive_feature_groups.svg)",
        "",
        "| Group | Members in top-k | Strongest raw feature | Strongest weight | Examples |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for row in interpretation_groups[:20]:
        examples = str(row["examples"]).replace("|", "\\|")
        report_lines.append(
            f"| {row['label']} | {row['member_count']} | "
            f"`{row['strongest_feature']}` | "
            f"{float(row['strongest_coefficient']):.3f} | `{examples}` |"
        )
    report_lines.extend(
        [
        "",
        "### Raw exact features",
        "",
        "![Ungrouped highest-weight target features](top_positive_features.svg)",
        "",
        "![Positive feature weight by category](category_weight_mass.svg)",
        "",
        "| Diagnostic category | Features | Feature share | Positive-weight share |",
        "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in category_rows:
        report_lines.append(
            f"| `{row['category']}` | {row['feature_count']} | "
            f"{float(row['feature_share']):.1%} | {float(row['weight_share']):.1%} |"
        )
    report_lines.extend(
        [
            "",
            "## Highest-Weight Features",
            "",
            "| Rank | Feature | Weight | Category | Target books | Comparison authors |",
            "| ---: | --- | ---: | --- | ---: | ---: |",
        ]
    )
    for row in rows[:40]:
        feature = str(row["feature"]).replace("|", "\\|")
        report_lines.append(
            f"| {row['rank']} | `{feature}` | {float(row['coefficient']):.3f} | "
            f"`{row['category']}` | {row['target_book_hits']}/{len(target_books)} | "
            f"{row['comparison_author_hits']}/{len(comparison_authors)} |"
        )
    if group_ablation is not None:
        report_lines.extend(
            [
                "",
                "## Frozen-Model Group Ablation",
                "",
                "This post-hoc sensitivity check removes every column in a registered",
                "feature family, L2-renormalizes the remaining frozen TF-IDF vector, and",
                "scores it with the unchanged classifier. It is not a refitted model or a",
                "new untouched benchmark.",
                "",
                "| Split | Variant | Removed features | Accuracy | Delta | Balanced | Target F1 | Target-margin delta |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for split, split_payload in group_ablation["splits"].items():
            baseline = split_payload["baseline"]
            report_lines.append(
                f"| {split} | baseline | 0 | {float(baseline['accuracy']):.1%} | "
                f"+0.0pp | {float(baseline['balanced_accuracy']):.1%} | "
                f"{float(baseline['target_f1']):.1%} | +0.000 |"
            )
            for name, metrics in split_payload["ablations"].items():
                feature_count = group_ablation["feature_groups"][name]["feature_count"]
                report_lines.append(
                    f"| {split} | `{name}` | {feature_count} | "
                    f"{float(metrics['accuracy']):.1%} | "
                    f"{float(metrics['accuracy_delta']) * 100:+.1f}pp | "
                    f"{float(metrics['balanced_accuracy']):.1%} | "
                    f"{float(metrics['target_f1']):.1%} | "
                    f"{float(metrics['target_mean_margin_delta']):+.3f} |"
                )
        test_payload = group_ablation["splits"].get("test")
        if test_payload:
            baseline = test_payload["baseline"]
            shuo = test_payload["ablations"]["shuo_punctuation_family"]
            punctuation_only = test_payload["ablations"]["pure_punctuation_family"]
            source_symbols = test_payload["ablations"]["source_format_symbol_family"]
            report_lines.extend(
                [
                    "",
                    "### What the ablation says",
                    "",
                    f"- The operationally defined `说 + punctuation` family contains "
                    f"{group_ablation['feature_groups']['shuo_punctuation_family']['feature_count']} "
                    "fitted columns. Removing it changes test accuracy by "
                    f"{float(shuo['accuracy_delta']) * 100:+.2f}pp and changes target "
                    f"false positives from {baseline['target_false_positive']} to "
                    f"{shuo['target_false_positive']}. Its repeated raw rows are overlapping "
                    "same-core variants and should not be read as independent cues.",
                    f"- Removing {group_ablation['feature_groups']['pure_punctuation_family']['feature_count']} "
                    "punctuation-only columns changes test accuracy by "
                    f"{float(punctuation_only['accuracy_delta']) * 100:+.2f}pp. This shows "
                    "predictive sensitivity, but does not distinguish author punctuation habits "
                    "from edition/provider formatting.",
                    f"- Removing {group_ablation['feature_groups']['source_format_symbol_family']['feature_count']} "
                    "symbol-only formatting columns changes test accuracy by "
                    f"{float(source_symbols['accuracy_delta']) * 100:+.2f}pp. These columns are "
                    "treated as source-format risk, not author-style evidence.",
                    "- This is a frozen-model sensitivity test. A separately trained "
                    "punctuation-normalized model would be required to estimate recoverable "
                    "accuracy without those features.",
                ]
            )
    report_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The frozen model contains useful surface-style signal, especially",
            "dialogue punctuation, speech attribution, function grammar, and discourse",
            "markers. It can also assign positive weight to known entity/topic fragments",
            "and lexical sequences concentrated in few target books. Placeholder artifacts",
            "are reported as their own category when present. Book-disjoint accuracy",
            "therefore does not establish content-independent style measurement.",
            "",
            "The Iteration-4 cross-author result is consistent with this diagnosis: methods",
            "can improve target margin while remaining far below threshold because source",
            "content continues to anchor the classifier toward the comparison author. It",
            "would be invalid to solve that gap by inserting names, settings, or theme words.",
            "",
            "## Research Consequence",
            "",
            "1. Keep this classifier as an attribution diagnostic and paired-margin meter.",
            "2. Do not treat its absolute cross-author threshold as a standalone style endpoint.",
            "3. Before the next efficacy claim, add a content-resistant style endpoint built",
            "   from function/discourse/punctuation/syntax features or a contrastive",
            "   content-controlled representation.",
            "4. Any classifier-guided generator must exclude entity/topic features and must",
            "   pass independent semantic/readability evaluation on untouched rows.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "uv run python experiments/iteration1/audit_style_meter_construct.py \\",
            f"  --scorer-dir {scorer_dir.relative_to(REPO_ROOT)} \\",
            f"  --dataset {dataset_path.relative_to(REPO_ROOT)} \\",
            f"  --mask-terms {mask_terms_path.relative_to(REPO_ROOT)} \\",
            f"  --target-author {args.target_author} \\",
            f"  --top-k {len(rows)} \\",
            *(
                [f"  --ablation-splits {','.join(ablation_splits)} \\"]
                if ablation_splits
                else []
            ),
            f"  --output-dir {output_dir.relative_to(REPO_ROOT)}",
            "```",
            "",
            "The category rules are deterministic diagnostics rather than gold linguistic",
            "annotations. Raw rows are in `target_positive_features.csv`; the grouped",
            "interpretation is in `target_positive_feature_groups.csv`.",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
