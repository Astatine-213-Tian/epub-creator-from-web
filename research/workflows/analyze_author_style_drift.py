#!/usr/bin/env python3
from __future__ import annotations

"""Measure within-author style drift across publication years and broad settings."""

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from workflows.analyze_interpretable_author_profiles import (
    CJK_RE,
    METRIC_DEFINITIONS,
    METRIC_LABELS,
    SENTENCE_SPLIT_RE,
    cjk_len,
    extract_book_metrics,
)


RESEARCH_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_MANIFEST = RESEARCH_ROOT / "datasets/dataset_manifest.json"
DEFAULT_CLEANED_MANIFEST = (
    RESEARCH_ROOT / "generated/style_research/corpus/cleaned_manifest.json"
)
DEFAULT_OUTPUT = (
    RESEARCH_ROOT
    / "generated/style_research/author_drift/feitianyexiang_style_drift.json"
)
TARGET_AUTHOR = "非天夜翔"

WESTERN_FANTASY_TITLES = frozenset(
    {"天之战记", "逆世界之书", "银河咏叹曲", "骑士之歌"}
)
STYLE_FAMILY_ORDER = (
    "西方架空／玄幻",
    "东方古代／武侠",
    "现代都市",
    "科幻／未来",
)
PERIOD_ORDER = ("2008–2012", "2013–2017", "2018–2025")

CLAUSE_SPLIT_RE = re.compile(r"[，；：]")
ELLIPSIS_RUN_RE = re.compile(r"…+")
SPEECH_TAG_PRE_RE = re.compile(r"([说道])\s*[：:]?\s*[“「『]")
SPEECH_TAG_POST_RE = re.compile(
    r"[”」』][^“”「」『』\n]{0,8}?([说道])(?:[。，！？!?]|\n|$)"
)
THIRD_PERSON_LEXICAL_RE = re.compile(r"其[他她它]|吉他")
QUOTE_PAIRS = {"“": "”", "「": "」", "『": "』"}
QUOTE_CLOSES = frozenset(QUOTE_PAIRS.values())

CORE_METRICS = tuple(METRIC_LABELS)
GENRE_DIAGNOSTIC_METRICS = (
    "commas_per_1k",
    "exact_four_cjk_clause_pct",
    "third_person_pronouns_per_1k",
    "mean_clause_cjk",
    "sentence_p90_cjk",
)
ADOPTED_METRICS = (
    "sentence_p90_cjk",
    "long_sentence_pct",
    "clauses_per_sentence",
    "mean_clause_cjk",
    "commas_per_1k",
    "quoted_cjk_pct",
    "dao_speech_tag_share",
    "speech_tags_per_quoted_span",
    "third_person_pronouns_per_1k",
    "exact_four_cjk_clause_pct",
)
SPEC_AUDIT_METRICS = (
    "semicolons_per_1k",
    "ellipses_per_1k",
    "exclamations_per_1k",
    "colons_per_1k",
    "double_dashes_per_1k",
    "aspect_le_per_1k",
    "passive_bei_per_1k",
    "experiential_guo_per_1k",
)
ALL_METRICS = CORE_METRICS + ADOPTED_METRICS + SPEC_AUDIT_METRICS

METRIC_LABELS_EXTENDED = {
    **METRIC_LABELS,
    "sentence_p90_cjk": "句长 P90",
    "long_sentence_pct": "长句占比（至少 46 字）",
    "clauses_per_sentence": "每句分句数",
    "mean_clause_cjk": "平均分句长度",
    "commas_per_1k": "逗号／千字",
    "quoted_cjk_pct": "引号内文字占比",
    "dao_speech_tag_share": "说话标签中“道”占比",
    "speech_tags_per_quoted_span": "每段引语的显式说话标签",
    "third_person_pronouns_per_1k": "第三人称代词／千字",
    "exact_four_cjk_clause_pct": "四字短分句占比",
    "semicolons_per_1k": "分号／千字",
    "ellipses_per_1k": "省略号／千字",
    "exclamations_per_1k": "感叹号／千字",
    "colons_per_1k": "冒号／千字",
    "double_dashes_per_1k": "双破折号／千字",
    "aspect_le_per_1k": "“了”／千字",
    "passive_bei_per_1k": "“被”／千字",
    "experiential_guo_per_1k": "“过”／千字",
}

METRIC_DEFINITIONS_EXTENDED = {
    **METRIC_DEFINITIONS,
    "sentence_p90_cjk": "每本书句长分布的第九十分位数，保留长句尾部信息。",
    "long_sentence_pct": "每本书中至少含 46 个汉字的句子占比。",
    "clauses_per_sentence": "按逗号、分号和冒号切分后，每个句子的非空分句数。",
    "mean_clause_cjk": "按逗号、分号和冒号切分后，非空分句的平均汉字数。",
    "commas_per_1k": "每一千汉字中的中文逗号数。",
    "quoted_cjk_pct": "位于配对中文引号内的汉字占全书汉字比例。",
    "dao_speech_tag_share": "检测到的“说／道”说话标签中，“道”所占比例。",
    "speech_tags_per_quoted_span": "每个中文引号起始标记对应的显式“说／道”标签数。",
    "third_person_pronouns_per_1k": (
        "每一千汉字中的“他／她／它”数；先排除“其他／其它／吉他”等词内命中。"
    ),
    "exact_four_cjk_clause_pct": (
        "按逗号、分号和冒号切分后，恰好四个汉字的短分句占比；不等同于成语率。"
    ),
    "semicolons_per_1k": "每一千汉字中的中文分号数。",
    "ellipses_per_1k": "每一千汉字中的省略号组数；连续省略点按两点一组。",
    "exclamations_per_1k": "每一千汉字中的全角或半角感叹号数。",
    "colons_per_1k": "每一千汉字中的中文冒号数。",
    "double_dashes_per_1k": "每一千汉字中的双破折号数。",
    "aspect_le_per_1k": "每一千汉字中的“了”字数。",
    "passive_bei_per_1k": "每一千汉字中的“被”字数。",
    "experiential_guo_per_1k": "每一千汉字中的“过”字数。",
}


def percentile_higher(values: Iterable[int], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a percentile for an empty sequence")
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return float(ordered[index])


def count_quoted_cjk(text: str) -> int:
    stack: list[str] = []
    count = 0
    for character in text:
        expected_close = QUOTE_PAIRS.get(character)
        if expected_close is not None:
            stack.append(expected_close)
            continue
        if character in QUOTE_CLOSES:
            if stack:
                if character in stack:
                    while stack and stack[-1] != character:
                        stack.pop()
                    if stack:
                        stack.pop()
                else:
                    stack.pop()
            continue
        if stack and CJK_RE.fullmatch(character):
            count += 1
    return count


def count_third_person_pronouns(text: str) -> int:
    without_lexical_false_positives = THIRD_PERSON_LEXICAL_RE.sub(
        "", text
    )
    return sum(without_lexical_false_positives.count(char) for char in "他她它")


def extract_extended_metrics(text: str) -> dict[str, float]:
    total_cjk = cjk_len(text)
    if total_cjk == 0:
        raise ValueError("cannot profile a text with no CJK characters")

    sentence_parts = [
        part for part in SENTENCE_SPLIT_RE.split(text) if CJK_RE.search(part)
    ]
    sentence_lengths = [cjk_len(part) for part in sentence_parts]
    clause_lengths: list[int] = []
    clause_counts: list[int] = []
    for sentence in sentence_parts:
        lengths = [
            cjk_len(clause)
            for clause in CLAUSE_SPLIT_RE.split(sentence)
            if CJK_RE.search(clause)
        ]
        clause_lengths.extend(lengths)
        clause_counts.append(len(lengths))

    opening_quote_count = sum(text.count(mark) for mark in QUOTE_PAIRS)
    tags = SPEECH_TAG_PRE_RE.findall(text) + SPEECH_TAG_POST_RE.findall(text)
    dao_count = sum(tag == "道" for tag in tags)
    per_1k = 1000.0 / total_cjk

    return {
        "sentence_p90_cjk": percentile_higher(sentence_lengths, 0.90),
        "long_sentence_pct": 100.0
        * sum(length >= 46 for length in sentence_lengths)
        / len(sentence_lengths),
        "clauses_per_sentence": float(statistics.mean(clause_counts)),
        "mean_clause_cjk": float(statistics.mean(clause_lengths)),
        "commas_per_1k": text.count("，") * per_1k,
        "quoted_cjk_pct": 100.0 * count_quoted_cjk(text) / total_cjk,
        "dao_speech_tag_share": dao_count / len(tags) if tags else 0.0,
        "speech_tags_per_quoted_span": (
            len(tags) / opening_quote_count if opening_quote_count else 0.0
        ),
        "third_person_pronouns_per_1k": (
            count_third_person_pronouns(text) * per_1k
        ),
        "exact_four_cjk_clause_pct": 100.0
        * sum(length == 4 for length in clause_lengths)
        / len(clause_lengths),
        "semicolons_per_1k": text.count("；") * per_1k,
        "ellipses_per_1k": sum(
            max(1, len(run) // 2) for run in ELLIPSIS_RUN_RE.findall(text)
        )
        * per_1k,
        "exclamations_per_1k": (
            text.count("！") + text.count("!")
        )
        * per_1k,
        "colons_per_1k": text.count("：") * per_1k,
        "double_dashes_per_1k": text.count("——") * per_1k,
        "aspect_le_per_1k": text.count("了") * per_1k,
        "passive_bei_per_1k": text.count("被") * per_1k,
        "experiential_guo_per_1k": text.count("过") * per_1k,
    }


def assign_style_family(title: str, time_area: str) -> str:
    if title in WESTERN_FANTASY_TITLES:
        return "西方架空／玄幻"
    if time_area == "近代现代":
        return "现代都市"
    if time_area == "幻想未来":
        return "科幻／未来"
    return "东方古代／武侠"


def assign_period(publication_year: int) -> str:
    if publication_year <= 2012:
        return "2008–2012"
    if publication_year <= 2017:
        return "2013–2017"
    return "2018–2025"


def load_target_books(
    dataset_manifest_path: Path,
    cleaned_manifest_path: Path,
    *,
    author: str = TARGET_AUTHOR,
) -> list[dict[str, Any]]:
    dataset_records = json.loads(
        dataset_manifest_path.read_text(encoding="utf-8")
    )
    cleaned_records = json.loads(
        cleaned_manifest_path.read_text(encoding="utf-8")
    )
    metadata = {
        record["title"]: record
        for record in dataset_records
        if record.get("author") == author
    }
    cleaned = {
        record["title"]: record
        for record in cleaned_records
        if record.get("author") == author
    }
    if metadata.keys() != cleaned.keys():
        missing_cleaned = sorted(metadata.keys() - cleaned.keys())
        missing_metadata = sorted(cleaned.keys() - metadata.keys())
        raise ValueError(
            "target manifests disagree: "
            f"missing_cleaned={missing_cleaned}, "
            f"missing_metadata={missing_metadata}"
        )

    books: list[dict[str, Any]] = []
    for title in sorted(metadata):
        record = metadata[title]
        year = record.get("publication_year")
        if year is None:
            raise ValueError(f"missing publication_year for {title}")
        clean_record = cleaned[title]
        clean_path = RESEARCH_ROOT / clean_record["clean_txt_path"]
        if not clean_path.exists():
            raise FileNotFoundError(clean_path)
        text = clean_path.read_text(encoding="utf-8")
        metrics = {
            **extract_book_metrics(text),
            **extract_extended_metrics(text),
        }
        books.append(
            {
                "title": title,
                "publication_year": int(year),
                "period": assign_period(int(year)),
                "genre": record["genre"],
                "time_area": record["time_area"],
                "style_family": assign_style_family(
                    title, record["time_area"]
                ),
                "clean_cjk_count": int(clean_record["clean_cjk_count"]),
                "metrics": metrics,
            }
        )
    return books


def design_categorical(
    labels: list[str], levels: tuple[str, ...]
) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(len(labels)),
            *(
                np.array([float(value == level) for value in labels])
                for level in levels[1:]
            ),
        ]
    )


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 1.0
    for rank, index in reversed(list(enumerate(order, start=1))):
        running = min(running, values[index] * len(values) / rank)
        adjusted[index] = running
    return adjusted.tolist()


def _projection_matrix(design: np.ndarray) -> np.ndarray:
    return design @ np.linalg.pinv(design)


def permutation_effect(
    outcomes: np.ndarray,
    reduced_design: np.ndarray,
    added_design: np.ndarray,
    metric_names: tuple[str, ...],
    *,
    panels: dict[str, tuple[str, ...]],
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    full_design = np.column_stack([reduced_design, added_design])
    identity = np.eye(len(outcomes))
    reduced_residualizer = identity - _projection_matrix(reduced_design)
    full_residualizer = identity - _projection_matrix(full_design)

    means = outcomes.mean(axis=0)
    standard_deviations = outcomes.std(axis=0, ddof=1)
    if np.any(standard_deviations == 0):
        constant_metrics = [
            metric_names[index]
            for index in np.flatnonzero(standard_deviations == 0)
        ]
        raise ValueError(f"constant metrics cannot be tested: {constant_metrics}")
    standardized = (outcomes - means) / standard_deviations

    fitted_reduced = (
        _projection_matrix(reduced_design) @ standardized
    )
    residual_reduced = reduced_residualizer @ standardized
    sse_reduced = np.sum(residual_reduced**2, axis=0)
    sse_full = np.sum((full_residualizer @ standardized) ** 2, axis=0)
    observed_improvement = sse_reduced - sse_full

    panel_indexes = {
        name: np.array([metric_names.index(metric) for metric in metrics])
        for name, metrics in panels.items()
    }
    metric_exceedances = np.zeros(len(metric_names), dtype=int)
    panel_exceedances = {name: 0 for name in panels}
    observed_panel_improvements = {
        name: float(np.sum(observed_improvement[indexes]))
        for name, indexes in panel_indexes.items()
    }

    random = np.random.default_rng(seed)
    for _ in range(permutations):
        permuted = fitted_reduced + residual_reduced[
            random.permutation(len(outcomes))
        ]
        permuted_reduced_sse = np.sum(
            (reduced_residualizer @ permuted) ** 2, axis=0
        )
        permuted_full_sse = np.sum(
            (full_residualizer @ permuted) ** 2, axis=0
        )
        improvement = permuted_reduced_sse - permuted_full_sse
        metric_exceedances += improvement >= observed_improvement - 1e-12
        for name, indexes in panel_indexes.items():
            panel_exceedances[name] += (
                float(np.sum(improvement[indexes]))
                >= observed_panel_improvements[name] - 1e-12
            )

    p_values = (
        (metric_exceedances + 1) / (permutations + 1)
    ).tolist()
    q_values = benjamini_hochberg(p_values)
    metrics = {}
    for index, metric in enumerate(metric_names):
        metrics[metric] = {
            "partial_r2": float(
                1.0 - sse_full[index] / sse_reduced[index]
            ),
            "p_value": float(p_values[index]),
            "q_value_bh": float(q_values[index]),
        }

    panel_results = {}
    for name, indexes in panel_indexes.items():
        panel_results[name] = {
            "partial_r2": float(
                1.0
                - np.sum(sse_full[indexes]) / np.sum(sse_reduced[indexes])
            ),
            "p_value": float(
                (panel_exceedances[name] + 1) / (permutations + 1)
            ),
            "metrics": list(panels[name]),
        }
    return {"metrics": metrics, "panels": panel_results}


def median_table(
    books: list[dict[str, Any]],
    group_key: str,
    group_order: tuple[str, ...],
) -> dict[str, Any]:
    table = {}
    for group in group_order:
        members = [book for book in books if book[group_key] == group]
        table[group] = {
            "book_count": len(members),
            "metrics": {
                metric: float(
                    statistics.median(
                        book["metrics"][metric] for book in members
                    )
                )
                for metric in ALL_METRICS
            },
        }
    return table


def build_analysis(
    books: list[dict[str, Any]], *, permutations: int, seed: int
) -> dict[str, Any]:
    metric_names = ALL_METRICS
    outcomes = np.asarray(
        [
            [book["metrics"][metric] for metric in metric_names]
            for book in books
        ],
        dtype=float,
    )
    years = np.asarray(
        [book["publication_year"] for book in books], dtype=float
    )
    standardized_year = (
        (years - years.mean()) / years.std(ddof=1)
    )[:, None]
    family_labels = [book["style_family"] for book in books]
    family_design = design_categorical(
        family_labels, STYLE_FAMILY_ORDER
    )
    intercept = np.ones((len(books), 1))
    family_dummies = family_design[:, 1:]
    panels = {
        "core_six": CORE_METRICS,
        "adopted_diagnostics": ADOPTED_METRICS,
        "genre_diagnostic_five": GENRE_DIAGNOSTIC_METRICS,
        "all_metrics": ALL_METRICS,
    }

    year_effect = permutation_effect(
        outcomes,
        family_design,
        standardized_year,
        metric_names,
        panels=panels,
        permutations=permutations,
        seed=seed,
    )
    full_year_design = np.column_stack(
        [family_design, standardized_year]
    )
    coefficients = np.linalg.lstsq(
        full_year_design,
        (outcomes - outcomes.mean(axis=0))
        / outcomes.std(axis=0, ddof=1),
        rcond=None,
    )[0]
    for index, metric in enumerate(metric_names):
        year_effect["metrics"][metric]["standardized_year_beta"] = float(
            coefficients[-1, index]
        )

    family_effect = permutation_effect(
        outcomes,
        np.column_stack([intercept, standardized_year]),
        family_dummies,
        metric_names,
        panels=panels,
        permutations=permutations,
        seed=seed,
    )

    return {
        "metadata": {
            "author": TARGET_AUTHOR,
            "book_count": len(books),
            "publication_year_min": int(years.min()),
            "publication_year_max": int(years.max()),
            "permutations": permutations,
            "seed": seed,
            "analysis_unit": "book",
            "style_family_order": list(STYLE_FAMILY_ORDER),
            "western_fantasy_titles": sorted(WESTERN_FANTASY_TITLES),
        },
        "metric_labels": {
            metric: METRIC_LABELS_EXTENDED[metric]
            for metric in ALL_METRICS
        },
        "metric_definitions": {
            metric: METRIC_DEFINITIONS_EXTENDED[metric]
            for metric in ALL_METRICS
        },
        "panels": {name: list(metrics) for name, metrics in panels.items()},
        "period_medians": median_table(
            books, "period", PERIOD_ORDER
        ),
        "family_medians": median_table(
            books, "style_family", STYLE_FAMILY_ORDER
        ),
        "effects": {
            "publication_year_controlling_style_family": year_effect,
            "style_family_controlling_publication_year": family_effect,
        },
        "books": books,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure publication-year and broad-setting style drift for "
            "the target author's complete cleaned corpus."
        )
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=DEFAULT_DATASET_MANIFEST,
    )
    parser.add_argument(
        "--cleaned-manifest",
        type=Path,
        default=DEFAULT_CLEANED_MANIFEST,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--permutations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20_260_718)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    books = load_target_books(
        args.dataset_manifest, args.cleaned_manifest
    )
    if len(books) != 40:
        raise ValueError(
            f"expected 40 target-author books, found {len(books)}"
        )
    analysis = build_analysis(
        books, permutations=args.permutations, seed=args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    year_core = analysis["effects"][
        "publication_year_controlling_style_family"
    ]["panels"]["core_six"]
    family_core = analysis["effects"][
        "style_family_controlling_publication_year"
    ]["panels"]["core_six"]
    family_diagnostics = analysis["effects"][
        "style_family_controlling_publication_year"
    ]["panels"]["genre_diagnostic_five"]
    print(
        f"OK: {len(books)} books; "
        f"core year partial R²={year_core['partial_r2']:.4f}; "
        f"core family partial R²={family_core['partial_r2']:.4f}; "
        "genre diagnostics family partial "
        f"R²={family_diagnostics['partial_r2']:.4f}; "
        f"wrote {args.output}"
    )


if __name__ == "__main__":
    main()
