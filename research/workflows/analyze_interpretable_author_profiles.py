#!/usr/bin/env python3
from __future__ import annotations

"""Build interpretable, book-weighted author-profile evidence for publication."""

import argparse
import html
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.iteration4 import build_style_definition as style_definition


RESEARCH_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = RESEARCH_ROOT / "generated/style_research/corpus/cleaned_manifest.json"
DEFAULT_MASKED_CHUNKS = RESEARCH_ROOT / "datasets/masked/chunks.entity_masked_v3.jsonl"
DEFAULT_OUTPUT_DIR = RESEARCH_ROOT / "docs/synthesis/assets"
DEFAULT_AUTHORS = ("非天夜翔", "priest", "巫哲", "木苏里")
AUTHOR_DISPLAY_NAMES = {"priest": "Priest"}

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
SENTENCE_SPLIT_RE = re.compile(r"[。！？!?]+")
DIALOGUE_PREFIX_RE = re.compile(r"^[-—]\s*[\u3400-\u9fff]")
DIALOGUE_MARKS = frozenset("“”「」『』")
CONNECTIVES = (
    "因为", "所以", "因此", "但是", "不过", "然而", "而且", "并且", "如果",
    "虽然", "尽管", "既然", "于是", "然后", "接着", "随后", "此外", "另外",
    "甚至", "其实", "当然", "只是", "仍然", "依然", "无论", "不管", "只要",
    "除非", "至于", "总之", "毕竟",
)
CONNECTIVE_RE = re.compile("|".join(sorted(map(re.escape, CONNECTIVES), key=len, reverse=True)))

METRIC_LABELS = {
    "median_sentence_cjk": "句长中位数",
    "short_sentence_pct": "短句占比",
    "median_paragraph_cjk": "段长中位数",
    "dialogue_paragraph_pct": "含对话段落",
    "questions_per_10k": "问号 / 万字",
    "connectives_per_10k": "显性连接词 / 万字",
}

METRIC_DEFINITIONS = {
    "median_sentence_cjk": "每本书按句末标点切分后，句内汉字数的中位数。",
    "short_sentence_pct": "每本书中不超过 12 个汉字的句子占比。",
    "median_paragraph_cjk": "以清理文本的非空行作为段落，计算段内汉字数中位数。",
    "dialogue_paragraph_pct": "含中文引号，或以破折号式对白开头的段落占比。",
    "questions_per_10k": "每一万汉字中的全角或半角问号数。",
    "connectives_per_10k": "每一万汉字中的显性连接词数；词表记录在脚本 CONNECTIVES 中。",
}

TARGET_FEATURE_LABELS = {
    "dialogue.quote_marks_per_kcjk": "对话标点",
    "dialogue.simple_speech_tags": "朴素说话标签",
    "dialogue.micro_reactions": "话语旁局部反应",
    "dialogue.laughter_tags": "紧凑笑意标签",
    "flow.short_sentence_ratio": "短句占比",
    "flow.mean_sentence_cjk": "平均句长",
    "punctuation.comma": "逗号",
    "function_word.connective": "连接词",
    "function_word.modal_aspect": "情态与体貌词",
    "function_word.preposition_frame": "介词框架",
    "function_word.particle_phrase": "助词短语",
    "function_word.deictic_pronoun": "指示与代词",
    "dialogue.question_tags": "问句标签",
    "punctuation.colon": "冒号",
    "punctuation.exclamation": "感叹号",
}

TARGET_FEATURE_ORDER = tuple(TARGET_FEATURE_LABELS)


def cjk_len(text: str) -> int:
    return sum(1 for _ in CJK_RE.finditer(text))


def extract_book_metrics(text: str) -> dict[str, float]:
    total_cjk = cjk_len(text)
    if total_cjk == 0:
        raise ValueError("cannot profile a text with no CJK characters")

    paragraphs = [line.strip() for line in text.splitlines() if CJK_RE.search(line)]
    sentence_lengths = [
        cjk_len(part)
        for part in SENTENCE_SPLIT_RE.split(text)
        if CJK_RE.search(part)
    ]
    if not paragraphs or not sentence_lengths:
        raise ValueError("cannot profile a text without paragraphs and sentences")

    dialogue_paragraphs = sum(
        1
        for paragraph in paragraphs
        if any(mark in paragraph for mark in DIALOGUE_MARKS)
        or DIALOGUE_PREFIX_RE.match(paragraph)
    )
    connective_count = sum(1 for _ in CONNECTIVE_RE.finditer(text))

    return {
        "median_sentence_cjk": float(statistics.median(sentence_lengths)),
        "short_sentence_pct": 100.0
        * sum(length <= 12 for length in sentence_lengths)
        / len(sentence_lengths),
        "median_paragraph_cjk": float(
            statistics.median(cjk_len(paragraph) for paragraph in paragraphs)
        ),
        "dialogue_paragraph_pct": 100.0 * dialogue_paragraphs / len(paragraphs),
        "questions_per_10k": 10000.0
        * (text.count("？") + text.count("?"))
        / total_cjk,
        "connectives_per_10k": 10000.0 * connective_count / total_cjk,
    }


def load_book_profiles(manifest_path: Path, *, min_cjk: int) -> dict[str, list[dict[str, Any]]]:
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    profiles: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if not record.get("exists", False) or int(record.get("clean_cjk_count", 0)) < min_cjk:
            continue
        text_path = RESEARCH_ROOT / record["clean_txt_path"]
        metrics = extract_book_metrics(text_path.read_text(encoding="utf-8"))
        profiles[record["author"]].append(
            {
                "title": record["title"],
                "clean_cjk_count": int(record["clean_cjk_count"]),
                "metrics": metrics,
            }
        )
    return dict(profiles)


def summarize_authors(
    book_profiles: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for author, books in sorted(book_profiles.items()):
        author_metrics = {
            metric: statistics.median(book["metrics"][metric] for book in books)
            for metric in METRIC_LABELS
        }
        summaries[author] = {
            "book_count": len(books),
            "metrics": author_metrics,
            "books": books,
        }

    for metric in METRIC_LABELS:
        values = [summary["metrics"][metric] for summary in summaries.values()]
        mean = statistics.mean(values)
        std = statistics.stdev(values)
        for summary in summaries.values():
            summary.setdefault("standard_scores", {})[metric] = (
                (summary["metrics"][metric] - mean) / std if std else 0.0
            )
    return summaries


def build_target_features(masked_chunks: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = list(style_definition.iter_jsonl(masked_chunks))
    target_titles = sorted(
        {
            str(row["title"])
            for row in rows
            if row.get("author") == style_definition.TARGET_AUTHOR
            and row.get("split") == "train"
        },
        key=style_definition.stable_order,
    )
    discovery_books = target_titles[: style_definition.DISCOVERY_BOOKS]
    validation_books = target_titles[style_definition.DISCOVERY_BOOKS :]
    discovery_profile, target_books, comparison_authors = style_definition.build_profiles(
        rows, set(discovery_books)
    )
    contrasts = style_definition.validated_contrasts(
        discovery_profile,
        target_books,
        comparison_authors,
        discovery_books,
        validation_books,
    )
    by_id = {
        item["feature_id"]: item
        for item in contrasts
        if item.get("retained")
    }
    features = []
    for feature_id in TARGET_FEATURE_ORDER:
        item = by_id.get(feature_id)
        if item is None:
            continue
        features.append(
            {
                "feature_id": feature_id,
                "label": TARGET_FEATURE_LABELS[feature_id],
                "target_value": item["target_value"],
                "comparison_author_mean": item["comparison_author_mean"],
                "z_score": item["z_score"],
                "discovery_recurrence": item["discovery_recurrence"]["rate"],
                "validation_recurrence": item["validation_recurrence"]["rate"],
            }
        )
    return features, {
        "target_train_books": len(target_titles),
        "discovery_books": len(discovery_books),
        "validation_books": len(validation_books),
        "comparison_authors": len(comparison_authors),
    }


def mix_color(base: tuple[int, int, int], amount: float) -> str:
    amount = min(max(amount, 0.0), 1.0)
    rgb = tuple(round(255 + (channel - 255) * amount) for channel in base)
    return "#" + "".join(f"{channel:02x}" for channel in rgb)


def heat_color(value: float, *, cap: float = 2.5) -> tuple[str, str]:
    intensity = min(abs(value) / cap, 1.0)
    base = (213, 94, 0) if value >= 0 else (0, 114, 178)
    return mix_color(base, intensity), ("#ffffff" if intensity >= 0.68 else "#17202a")


def raw_metric_label(metric: str, value: float) -> str:
    if metric in {"median_sentence_cjk", "median_paragraph_cjk"}:
        return f"{value:.1f} 字"
    if metric in {"short_sentence_pct", "dialogue_paragraph_pct"}:
        return f"{value:.1f}%"
    return f"{value:.1f}"


def render_profile_heatmap(
    summaries: dict[str, dict[str, Any]], authors: tuple[str, ...]
) -> str:
    missing = [author for author in authors if author not in summaries]
    if missing:
        raise ValueError(f"authors missing from corpus: {', '.join(missing)}")

    width = 1260
    left = 190
    top = 150
    cell_w = 168
    cell_h = 94
    height = top + len(authors) * cell_h + 135
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<g font-family="Noto Sans CJK SC, PingFang SC, Microsoft YaHei, sans-serif">',
        '<text x="32" y="42" font-size="25" font-weight="700" fill="#17202a">四位作者的可解释文体坐标</text>',
        '<text x="32" y="73" font-size="15" fill="#4d5966">每本书先独立计算，再取作者的书级中位数；颜色表示相对 50 位作者的标准位置</text>',
    ]

    for column, (metric, label) in enumerate(METRIC_LABELS.items()):
        x = left + column * cell_w + cell_w / 2
        elements.append(
            f'<text x="{x:.1f}" y="122" text-anchor="middle" font-size="15" '
            f'font-weight="600" fill="#26323d">{html.escape(label)}</text>'
        )

    for row, author in enumerate(authors):
        y = top + row * cell_h
        book_count = summaries[author]["book_count"]
        display_name = AUTHOR_DISPLAY_NAMES.get(author, author)
        elements.append(
            f'<text x="{left - 18}" y="{y + 36}" text-anchor="end" font-size="18" '
            f'font-weight="700" fill="#17202a">{html.escape(display_name)}</text>'
        )
        elements.append(
            f'<text x="{left - 18}" y="{y + 59}" text-anchor="end" font-size="13" '
            f'fill="#697784">{book_count} 本书</text>'
        )
        for column, metric in enumerate(METRIC_LABELS):
            x = left + column * cell_w
            z_score = summaries[author]["standard_scores"][metric]
            raw_value = summaries[author]["metrics"][metric]
            fill, text_fill = heat_color(z_score)
            elements.extend(
                [
                    f'<rect x="{x + 3}" y="{y + 3}" width="{cell_w - 6}" '
                    f'height="{cell_h - 6}" rx="3" fill="{fill}" stroke="#d4d9de"/>',
                    f'<text x="{x + cell_w / 2:.1f}" y="{y + 38}" text-anchor="middle" '
                    f'font-size="19" font-weight="700" fill="{text_fill}">{z_score:+.2f}σ</text>',
                    f'<text x="{x + cell_w / 2:.1f}" y="{y + 63}" text-anchor="middle" '
                    f'font-size="13" fill="{text_fill}">{html.escape(raw_metric_label(metric, raw_value))}</text>',
                ]
            )

    legend_y = top + len(authors) * cell_h + 38
    legend_items = (("低于平均", -2.0), ("接近平均", 0.0), ("高于平均", 2.0))
    for index, (label, value) in enumerate(legend_items):
        fill, _ = heat_color(value)
        x = left + index * 210
        elements.append(f'<rect x="{x}" y="{legend_y}" width="30" height="18" fill="{fill}" stroke="#c8ced4"/>')
        elements.append(
            f'<text x="{x + 40}" y="{legend_y + 14}" font-size="13" fill="#4d5966">{label}</text>'
        )
    elements.append(
        f'<text x="{left}" y="{legend_y + 48}" font-size="12" fill="#697784">'
        "注：高低只表示相对位置，不代表文学质量；这些指标会受题材与场景构成影响。</text>"
    )
    elements.extend(("</g>", "</svg>"))
    return "\n".join(elements) + "\n"


def render_target_feature_chart(
    features: list[dict[str, Any]], partition: dict[str, int]
) -> str:
    width = 1180
    left = 300
    right = 100
    top = 128
    row_h = 43
    plot_width = width - left - right
    x_min = min(-1.5, math.floor(min(item["z_score"] for item in features) * 2) / 2)
    x_max = max(4.0, math.ceil(max(item["z_score"] for item in features) * 2) / 2)
    height = top + len(features) * row_h + 112

    def x_position(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    zero_x = x_position(0.0)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<g font-family="Noto Sans CJK SC, PingFang SC, Microsoft YaHei, sans-serif">',
        f'<text x="32" y="42" font-size="25" font-weight="700" fill="#17202a">非天夜翔：跨书验证后保留的 {len(features)} 个信号</text>',
        f'<text x="32" y="73" font-size="15" fill="#4d5966">{partition["discovery_books"]} 本发现书 + {partition["validation_books"]} 本验证书，对照其余 {partition["comparison_authors"]} 位作者；横轴为相对对照作者的标准位置</text>',
    ]

    first_tick = math.ceil(x_min)
    last_tick = math.floor(x_max)
    for tick in range(first_tick, last_tick + 1):
        x = x_position(float(tick))
        elements.append(
            f'<line x1="{x:.1f}" y1="{top - 16}" x2="{x:.1f}" '
            f'y2="{top + len(features) * row_h}" stroke="{("#66727d" if tick == 0 else "#e4e7ea")}" '
            f'stroke-width="{(1.5 if tick == 0 else 1)}"/>'
        )
        elements.append(
            f'<text x="{x:.1f}" y="{top - 25}" text-anchor="middle" font-size="12" '
            f'fill="#697784">{tick:+d}σ</text>'
        )

    for row, item in enumerate(features):
        y = top + row * row_h
        score = float(item["z_score"])
        end_x = x_position(score)
        bar_x = min(zero_x, end_x)
        bar_width = max(abs(end_x - zero_x), 2.0)
        color = "#d55e00" if score >= 0 else "#0072b2"
        elements.extend(
            [
                f'<text x="{left - 18}" y="{y + 26}" text-anchor="end" font-size="14" '
                f'fill="#26323d">{html.escape(item["label"])}</text>',
                f'<rect x="{bar_x:.1f}" y="{y + 9}" width="{bar_width:.1f}" height="22" '
                f'rx="2" fill="{color}" opacity="0.88"/>',
                f'<text x="{(end_x + (8 if score >= 0 else -8)):.1f}" y="{y + 26}" '
                f'text-anchor="{("start" if score >= 0 else "end")}" font-size="13" '
                f'font-weight="600" fill="#17202a">{score:+.2f}σ</text>',
            ]
        )

    note_y = top + len(features) * row_h + 48
    elements.append(
        f'<text x="32" y="{note_y}" font-size="12" fill="#697784">'
        "注：橙色表示高于对照作者平均，蓝色表示低于；统计倾向必须由当前场景触发，不能当成写作配额。</text>"
    )
    elements.extend(("</g>", "</svg>"))
    return "\n".join(elements) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build book-weighted, interpretable author profile evidence and SVG figures."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--masked-chunks", type=Path, default=DEFAULT_MASKED_CHUNKS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-cjk", type=int, default=50_000)
    parser.add_argument("--authors", nargs="+", default=list(DEFAULT_AUTHORS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    authors = tuple(args.authors)
    book_profiles = load_book_profiles(args.manifest, min_cjk=args.min_cjk)
    summaries = summarize_authors(book_profiles)
    target_features, target_partition = build_target_features(args.masked_chunks)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence = {
        "schema_version": "interpretable_author_profiles.v1",
        "method": {
            "observational_unit": "book",
            "author_aggregation": "median across eligible books",
            "author_standardization": "z-score across author-level medians",
            "minimum_clean_cjk": args.min_cjk,
            "metric_definitions": METRIC_DEFINITIONS,
        },
        "corpus": {
            "author_count": len(summaries),
            "book_count": sum(summary["book_count"] for summary in summaries.values()),
        },
        "selected_authors": {
            author: {
                "book_count": summaries[author]["book_count"],
                "metrics": summaries[author]["metrics"],
                "standard_scores": summaries[author]["standard_scores"],
            }
            for author in authors
        },
        "target_validated_features": target_features,
        "target_partition": target_partition,
        "limitations": [
            "Descriptive coordinates are not a complete definition of style.",
            "Genre and scene composition can affect every displayed metric.",
            "The target-feature chart comes from a separate masked, discovery-validation analysis.",
            "No source text or quotation is included in this derived evidence file.",
        ],
    }
    (args.output_dir / "author_style_evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "author_profile_heatmap.svg").write_text(
        render_profile_heatmap(summaries, authors), encoding="utf-8"
    )
    (args.output_dir / "feitianyexiang_validated_features.svg").write_text(
        render_target_feature_chart(target_features, target_partition), encoding="utf-8"
    )
    print(f"authors={len(summaries)} books={evidence['corpus']['book_count']}")
    print(args.output_dir / "author_style_evidence.json")
    print(args.output_dir / "author_profile_heatmap.svg")
    print(args.output_dir / "feitianyexiang_validated_features.svg")


if __name__ == "__main__":
    main()
