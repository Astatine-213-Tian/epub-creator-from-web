#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np


from experiments.shared.paths import RESEARCH_ROOT


REPO_ROOT = RESEARCH_ROOT

from workflows.benchmark_author_style import (  # noqa: E402
    FUNCTION_CHARS,
    FUNCTION_WORDS,
    PUNCT_CHARS,
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
    punctuation = any(char in PUNCT_CHARS for char in feature)
    if PLACEHOLDER_RE.search(feature):
        return "mask_artifact"
    if matches_known_mask_term(feature, known_terms):
        return "known_entity_or_topic_fragment"
    if any(term in feature for term in SPEECH_OR_BEAT_TERMS):
        return "dialogue_structure"
    if not cjk and punctuation:
        return "punctuation_only"
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


def main() -> None:
    args = parse_args()
    scorer_dir = args.scorer_dir.expanduser().resolve()
    dataset_path = args.dataset.expanduser().resolve()
    mask_terms_path = args.mask_terms.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
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
    target_chunks = 0
    comparison_chunks = 0

    for row in iter_jsonl(dataset_path):
        if row.get("split") != "train":
            continue
        author = str(row["author"])
        title = str(row["title"])
        text = WHITESPACE_RE.sub("", str(row["text"]))
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
    write_csv(output_dir / "category_summary.csv", category_rows)
    write_svg(output_dir / "category_weight_mass.svg", category_rows)
    write_top_features_svg(output_dir / "top_positive_features.svg", rows)
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
        "![Highest-weight target features](top_positive_features.svg)",
        "",
        "![Positive feature weight by category](category_weight_mass.svg)",
        "",
        "| Diagnostic category | Features | Feature share | Positive-weight share |",
        "| --- | ---: | ---: | ---: |",
    ]
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
            f"  --output-dir {output_dir.relative_to(REPO_ROOT)}",
            "```",
            "",
            "The category rules are deterministic diagnostics rather than gold linguistic",
            "annotations. The full feature table is `target_positive_features.csv`.",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
