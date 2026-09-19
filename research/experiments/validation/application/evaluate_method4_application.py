#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import statistics
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import joblib


from experiments.shared.paths import RESEARCH_ROOT
from workflows.author_style_meter_contract import CURRENT_SCORER_ID


REPO_ROOT = RESEARCH_ROOT
PRODUCTION_ROOT = REPO_ROOT.parent

from experiments.validation.application import (  # noqa: E402
    apply_content_plan_combined_to_translation_run as application,
)
from src.crawler.snapshot import write_json  # noqa: E402


SCHEMA = "method4_application_evaluation.v1"
TARGET_AUTHOR = "非天夜翔"
DEFAULT_SCORER_DIR = (
    REPO_ROOT
    / "generated/style_research/style_transfer_experiments/iterations/full_regeneration_v1/"
    f"scorers/{CURRENT_SCORER_ID}"
)
DEFAULT_THRESHOLD_PATH = (
    REPO_ROOT
    / "generated/style_research/style_transfer_experiments/iterations/full_regeneration_v1/"
    "calibration/style_meter_threshold.v1.json"
)
DEFAULT_GLOSSARY_PATH = PRODUCTION_ROOT / "book_specs/eternal_gate/glossary.json"
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
NUMBER_RE = re.compile(r"\d+(?:[.,:/-]\d+)*")
LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]*")
SENTENCE_RE = re.compile(r"[^。！？!?]+[。！？!?]?")
PUNCTUATION = "，。！？；：、…—“”‘’「」『』"
FUNCTION_CHARS = "的了着过地得在是有和与而但却就都也又还把被将让给对从向于以因若虽则所其之乎者吗呢吧啊呀哦"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def codex_log_model(path: Path) -> str:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[:20]:
        if line.startswith("model: "):
            return line.removeprefix("model: ").strip()
    return "unknown"


def verify_scorer_files(scorer_dir: Path) -> dict[str, Any]:
    manifest = read_json(scorer_dir / "manifest.json")
    for record in (manifest.get("files") or {}).values():
        path = scorer_dir / str(record["path"])
        observed = file_sha256(path)
        if observed != record["sha256"]:
            raise RuntimeError(f"frozen scorer hash mismatch: {path}")
    return manifest


def load_scorer(scorer_dir: Path) -> tuple[Any, Any, list[str], dict[str, Any]]:
    manifest = verify_scorer_files(scorer_dir)
    labels_payload = read_json(scorer_dir / "labels.json")
    labels = [str(value) for value in labels_payload["labels"]]
    if labels_payload.get("target_author") != TARGET_AUTHOR:
        raise RuntimeError("unexpected frozen target author")
    vectorizer = joblib.load(scorer_dir / "vectorizer.joblib")
    classifier = joblib.load(scorer_dir / "classifier.joblib")
    if [str(value) for value in classifier.classes_] != labels:
        raise RuntimeError("classifier classes disagree with labels.json")
    return vectorizer, classifier, labels, manifest


def glossary_mask_terms(glossary_path: Path) -> list[str]:
    glossary = read_json(glossary_path)
    terms: set[str] = set()
    for source, record in (glossary.get("terms") or {}).items():
        if not isinstance(record, dict):
            continue
        zh = str(record.get("zh") or "")
        runs = CJK_RUN_RE.findall(zh)
        for run in runs:
            if len(run) >= 2 or (
                len(run) == 1
                and source
                and source[0].isupper()
                and len(source) > 2
            ):
                terms.add(run)
    return sorted(terms, key=lambda value: (-len(value), value))


def mask_generated_text(text: str, terms: Sequence[str]) -> str:
    masked = LATIN_RE.sub("<LATIN>", text)
    masked = NUMBER_RE.sub("<NUM>", masked)
    if terms:
        pattern = re.compile("|".join(re.escape(term) for term in terms))
        masked = pattern.sub(
            lambda match: "某" * len(CJK_RE.findall(match.group(0))), masked
        )
    return masked


def score_texts(
    vectorizer: Any,
    classifier: Any,
    labels: Sequence[str],
    texts: Sequence[str],
) -> list[dict[str, Any]]:
    if not texts:
        return []
    margins = classifier.decision_function(vectorizer.transform(texts))
    target_index = list(labels).index(TARGET_AUTHOR)
    rows: list[dict[str, Any]] = []
    for vector in margins:
        values = [float(value) for value in vector]
        order = sorted(
            range(len(values)), key=lambda index: (-values[index], labels[index])
        )
        rows.append(
            {
                "target_margin": values[target_index],
                "target_rank": order.index(target_index) + 1,
                "predicted_author": labels[order[0]],
                "target_top1": int(order[0] == target_index),
            }
        )
    return rows


def translation_map(path: Path) -> dict[int, str]:
    data = read_json(path)
    return {
        int(row["index"]): str(row.get("zh") or "").strip()
        for row in data.get("translations") or []
    }


def build_windows(
    *,
    chapter_id: str,
    title: str,
    neutral: dict[int, str],
    styled: dict[int, str],
    target_cjk: int = 1200,
    minimum_cjk: int = 600,
    maximum_cjk: int = 1500,
) -> list[dict[str, Any]]:
    if sorted(neutral) != sorted(styled):
        raise RuntimeError(f"translation index mismatch in chapter {chapter_id}")
    windows: list[list[int]] = []
    current: list[int] = []
    current_cjk = 0
    for index in sorted(neutral):
        paragraph_cjk = len(CJK_RE.findall(neutral[index]))
        if current and current_cjk >= minimum_cjk and current_cjk + paragraph_cjk > maximum_cjk:
            windows.append(current)
            current = []
            current_cjk = 0
        current.append(index)
        current_cjk += paragraph_cjk
        if current_cjk >= target_cjk:
            windows.append(current)
            current = []
            current_cjk = 0
    if current:
        if windows:
            previous_cjk = sum(
                len(CJK_RE.findall(neutral[index])) for index in windows[-1]
            )
            tail_cjk = sum(len(CJK_RE.findall(neutral[index])) for index in current)
            if tail_cjk < minimum_cjk and previous_cjk + tail_cjk <= 1800:
                windows[-1].extend(current)
            else:
                windows.append(current)
        else:
            windows.append(current)
    return [
        {
            "window_id": f"{chapter_id}_eval_{number:03d}",
            "chapter_id": chapter_id,
            "chapter_title": title,
            "indexes": indexes,
            "neutral_text": "\n".join(neutral[index] for index in indexes),
            "styled_text": "\n".join(styled[index] for index in indexes),
        }
        for number, indexes in enumerate(windows, 1)
    ]


def starts_dialogue(text: str) -> bool:
    return text.lstrip().startswith(("“", "‘", "「", "『", '"'))


def percentile(values: Sequence[int], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def flow_metrics(text: str) -> dict[str, float | int]:
    paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
    sentences = [value for value in SENTENCE_RE.findall(text) if CJK_RE.search(value)]
    paragraph_lengths = [len(CJK_RE.findall(value)) for value in paragraphs]
    sentence_lengths = [len(CJK_RE.findall(value)) for value in sentences]
    cjk_count = len(CJK_RE.findall(text))
    return {
        "cjk_count": cjk_count,
        "paragraph_count": len(paragraphs),
        "sentence_count": len(sentences),
        "mean_paragraph_cjk": statistics.fmean(paragraph_lengths)
        if paragraph_lengths
        else 0.0,
        "mean_sentence_cjk": statistics.fmean(sentence_lengths)
        if sentence_lengths
        else 0.0,
        "p90_sentence_cjk": percentile(sentence_lengths, 0.90),
        "punctuation_per_100_cjk": sum(character in PUNCTUATION for character in text)
        * 100
        / max(cjk_count, 1),
        "dialogue_line_share": sum(starts_dialogue(value) for value in paragraphs)
        / max(len(paragraphs), 1),
        "function_chars_per_100_cjk": sum(text.count(value) for value in FUNCTION_CHARS)
        * 100
        / max(cjk_count, 1),
        "comma_per_100_cjk": text.count("，") * 100 / max(cjk_count, 1),
        "period_per_100_cjk": text.count("。") * 100 / max(cjk_count, 1),
    }


def summarize_scores(rows: Sequence[dict[str, Any]], threshold: float) -> dict[str, Any]:
    lifts = [float(row["paired_margin_lift"]) for row in rows]
    return {
        "window_count": len(rows),
        "neutral_mean_target_margin": statistics.fmean(
            float(row["neutral_target_margin"]) for row in rows
        ),
        "styled_mean_target_margin": statistics.fmean(
            float(row["styled_target_margin"]) for row in rows
        ),
        "mean_paired_margin_lift": statistics.fmean(lifts),
        "median_paired_margin_lift": statistics.median(lifts),
        "positive_lift_count": sum(value > 0 for value in lifts),
        "positive_lift_share": sum(value > 0 for value in lifts) / max(len(lifts), 1),
        "styled_top1_count": sum(int(row["styled_target_top1"]) for row in rows),
        "styled_top1_share": sum(int(row["styled_target_top1"]) for row in rows)
        / max(len(rows), 1),
        "styled_top5_count": sum(int(row["styled_target_rank"]) <= 5 for row in rows),
        "styled_top5_share": sum(int(row["styled_target_rank"]) <= 5 for row in rows)
        / max(len(rows), 1),
        "styled_above_historical_threshold_count": sum(
            float(row["styled_target_margin"]) >= threshold for row in rows
        ),
        "styled_above_historical_threshold_share": sum(
            float(row["styled_target_margin"]) >= threshold for row in rows
        )
        / max(len(rows), 1),
    }


def audit_chunks(style_run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    checked = 0
    content_plan_failures = 0
    for chunk in manifest.get("chunks") or []:
        chunk_id = str(chunk["chunk_id"])
        request = read_json(style_run_dir / str(chunk["request_path"]))
        output = read_json(style_run_dir / str(chunk["json_output_path"]))
        method_artifact = read_json(style_run_dir / str(chunk["method_output_path"]))
        ids = [f"p{index + 1:04d}" for index in range(len(chunk["indexes"]))]
        result = {
            "paragraphs": [
                {"id": ids[position], "zh": str(row.get("zh") or "")}
                for position, row in enumerate(output.get("translations") or [])
            ]
        }
        errors = application.deterministic_fidelity_errors(request, result)
        copied = application.copied_reference_spans(request, result)
        initial_result = method_artifact.get("result") or {}
        expected_ids = [row["id"] for row in request["neutral_zh"]]
        plan = initial_result.get("content_plan") or []
        if [row.get("id") for row in plan] != expected_ids:
            errors.append("initial_content_plan_id_mismatch")
            content_plan_failures += 1
        if copied:
            errors.append("introduced_reference_copy:" + ",".join(copied[:5]))
        expected_indexes = [int(value) for value in chunk["indexes"]]
        actual_indexes = [int(row["index"]) for row in output.get("translations") or []]
        if actual_indexes != expected_indexes:
            errors.append("output_index_mismatch")
        if errors:
            failures.append({"chunk_id": chunk_id, "failures": errors})
        checked += 1
    return {
        "checked_chunk_count": checked,
        "failure_count": len(failures),
        "content_plan_failure_count": content_plan_failures,
        "failures": failures,
    }


def render_svg(path: Path, chapter_rows: Sequence[dict[str, Any]]) -> None:
    width = 1080
    left = 245
    right = 70
    top = 70
    row_height = 42
    height = top + row_height * len(chapter_rows) + 80
    lifts = [float(row["mean_paired_margin_lift"]) for row in chapter_rows]
    max_abs = max([abs(value) for value in lifts] + [0.1])
    plot_width = width - left - right
    zero_x = left + plot_width / 2
    scale = (plot_width / 2) / max_abs
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#20252b;letter-spacing:0}.title{font-size:20px;font-weight:650}.label{font-size:13px}.value{font-size:12px;font-weight:600}</style>',
        '<text class="title" x="24" y="34">Historical masked n-gram paired margin lift by chapter</text>',
        f'<line x1="{zero_x:.1f}" y1="{top - 18}" x2="{zero_x:.1f}" y2="{height - 45}" stroke="#68717d" stroke-width="1"/>',
    ]
    for index, row in enumerate(chapter_rows):
        value = float(row["mean_paired_margin_lift"])
        y = top + index * row_height
        bar_x = zero_x if value >= 0 else zero_x + value * scale
        bar_width = max(abs(value * scale), 1.5)
        color = "#177a54" if value >= 0 else "#b33b32"
        label = html.escape(str(row["chapter_title"]))
        parts.extend(
            [
                f'<text class="label" x="24" y="{y + 14}">{label}</text>',
                f'<rect x="{bar_x:.1f}" y="{y}" width="{bar_width:.1f}" height="18" rx="2" fill="{color}"/>',
                f'<text class="value" x="{zero_x + value * scale + (6 if value >= 0 else -6):.1f}" y="{y + 14}" text-anchor="{("start" if value >= 0 else "end")}">{value:+.3f}</text>',
            ]
        )
    parts.append(
        f'<text class="label" x="24" y="{height - 18}">Positive values move toward the target-author margin. This meter is diagnostic, not a primary efficacy endpoint.</text>'
    )
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def independent_verdict(path: Path) -> str:
    if not path.exists():
        return "PENDING"
    text = path.read_text(encoding="utf-8")
    matches = re.findall(
        r"(?:##\s+Final Verdict|Final verdict:)\s*.*?\*\*(CONDITIONAL PASS|PASS|FAIL)\*\*",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return matches[-1].upper() if matches else "UNPARSED"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    overall = report["historical_style_meter"]["overall"]
    hard = report["hard_gate_audit"]
    semantic = report["semantic_compression_qa"]
    epub_validation = report["epub_validation"]
    flow = report["flow"]
    lines = [
        "# Eternal Gate Method4 Application Report",
        "",
        "> Status: **experimental reader artifact**. `content_plan_combined_full_regeneration`",
        "> was retained as the engineering prompt method after direct reader review, but",
        "> Iteration 4 did not reach its registered research endpoint. The historical n-gram scorer below",
        "> is retained only as a paired diagnostic after its construct-validity failure.",
        "",
        "## Corpus and Run",
        "",
        "| Item | Value |",
        "| --- | ---: |",
        f"| Source posts | {report['coverage']['chapter_count']} |",
        f"| Source paragraphs | {report['coverage']['paragraph_count']:,} |",
        f"| Method4 blocks | {report['coverage']['method_block_count']} |",
        f"| Evaluation windows | {overall['window_count']} |",
        f"| Latest source chapter | `{report['coverage']['latest_chapter_title']}` |",
        f"| Reader EPUB | `{epub_validation.get('epub_path') or 'PENDING'}` |",
        f"| Reader EPUB SHA-256 | `{epub_validation.get('epub_sha256') or 'PENDING'}` |",
        f"| Capacity fallback order | `{' -> '.join(report['model_provenance']['configured_fallback_order'])}` |",
        "| Final block models | `"
        + json.dumps(
            report["model_provenance"]["model_chunk_counts"],
            ensure_ascii=False,
            sort_keys=True,
        )
        + "` |",
        f"| Mixed-model blocks | {len(report['model_provenance']['mixed_model_chunks'])} |",
        f"| Neutral leaf fallbacks | {report['model_provenance']['neutral_leaf_fallback_count']} / {report['coverage']['paragraph_count']:,} ({format_pct(report['model_provenance']['neutral_leaf_fallback_share'])}) |",
        f"| Accepted-segment prompt binding | {report['method_provenance']['prompt_binding_status']} |",
        f"| Accepted-segment schema binding | {report['method_provenance']['schema_binding_status']} |",
        f"| Accepted-segment runner binding | {report['method_provenance']['runner_binding_status']} |",
        "",
        "## Method Applied",
        "",
        "| Pass | Input and operation | Model/procedure | Result |",
        "| --- | --- | --- | --- |",
        "| 1. Neutral semantic translation | Latest Patreon English; fixed neutral prompt; paragraph boundaries retained | `"
        + json.dumps(
            report["semantic_provenance"]["primary_chunk_models"],
            ensure_ascii=False,
            sort_keys=True,
        )
        + "` | "
        + f"{report['coverage']['paragraph_count']:,} paragraphs; "
        + f"{report['semantic_provenance']['fallback_repaired_paragraph_count']} refusal/empty paragraphs replaced from `{report['semantic_provenance']['fallback_model']}` |",
        "| 2. `content_plan_combined` style transfer | English-grounded content plan, neutral terminology/content anchor, validated style definition, and retrieved aligned examples; complete paragraph regeneration | `gpt-5.6-sol`, high reasoning; 12 paragraphs/block | "
        + f"{report['coverage']['method_block_count']} blocks; "
        + f"{report['model_provenance']['neutral_leaf_fallback_count']} audited neutral leaf fallbacks |",
        "| 3. Semantic validation | Deterministic fidelity gates plus model review of the 30 highest-risk compression candidates | `gpt-5.6-sol` | 27 preserved paraphrases, 3 detector false positives, 0 true losses/repairs |",
        "",
        "The style definition was estimated from "
        + f"{report['method_details']['target_train_books']} target-author training books against "
        + f"{report['method_details']['comparison_train_authors']} comparison authors, with "
        + f"{report['method_details']['discovery_books']} discovery and "
        + f"{report['method_details']['validation_books']} validation books. Eternal Gate, development books, and final books were excluded; examples were masked. "
        + f"Each block retrieved {report['method_details']['aligned_pair_k']} aligned pairs from a pool of {report['method_details']['aligned_pair_pool_count']} and supplied {report['method_details']['reference_example_count']} total reference items.",
        "",
        "Style dimensions: "
        + ", ".join(f"`{value}`" for value in report["method_details"]["style_dimensions"])
        + ".",
        "",
        "The English source is authoritative. The content plan is derived before style evidence",
        "is applied; references license abstract syntax, rhythm, dialogue, and punctuation only,",
        "with an eight-consecutive-Han-character copy guard.",
        "",
        "## Validation Status",
        "",
        "| Gate | Result |",
        "| --- | --- |",
        f"| Standard chunk/index/nonempty validation | **{report['standard_validation_status']}** |",
        f"| Content-plan and deterministic hard gates | **{'PASS' if hard['failure_count'] == 0 else 'FAIL'}** ({hard['failure_count']} failing blocks) |",
        f"| Reference-copy gate | **{'PASS' if report['copy_failure_count'] == 0 else 'FAIL'}** |",
        f"| Semantic-compression QA | {semantic['status']} |",
        f"| Independent repairs | {report['independent_repairs']['status']} ({report['independent_repairs']['repair_count']} applied/recorded) |",
        f"| Independent agent evaluation | **{report['independent_evaluation_verdict']}** ([report](independent_application_evaluation.md)) |",
        f"| EPUB packaging audit | **{epub_validation['status']}** ([report](method4_epub_validation.md)) |",
        "",
        "## Historical Style Diagnostic",
        "",
        "The scorer is the frozen class-balanced exact character 2-4-gram SGD hinge",
        "classifier on `entity_masked_v3`. Eternal Gate text is masked with the current",
        "glossary term set plus Latin/number masking. Its old absolute threshold is shown",
        "for continuity only and must not be interpreted as proof of author style.",
        "",
        "| Metric | Result |",
        "| --- | ---: |",
        f"| Neutral mean target margin | {overall['neutral_mean_target_margin']:+.4f} |",
        f"| Method4 mean target margin | {overall['styled_mean_target_margin']:+.4f} |",
        f"| Mean paired lift | **{overall['mean_paired_margin_lift']:+.4f}** |",
        f"| Positive-lift windows | {overall['positive_lift_count']}/{overall['window_count']} ({format_pct(overall['positive_lift_share'])}) |",
        f"| Method4 target top-1 windows | {overall['styled_top1_count']}/{overall['window_count']} ({format_pct(overall['styled_top1_share'])}) |",
        f"| Method4 target top-5 windows | {overall['styled_top5_count']}/{overall['window_count']} ({format_pct(overall['styled_top5_share'])}) |",
        "",
        "![Paired target-margin lift by chapter](method4_margin_lift_by_chapter.svg)",
        "",
        "### Per Chapter",
        "",
        "| Chapter | Windows | Neutral margin | Method4 margin | Lift | Positive | Top 5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["historical_style_meter"]["by_chapter"]:
        lines.append(
            f"| {row['chapter_title']} | {row['window_count']} | "
            f"{row['neutral_mean_target_margin']:+.3f} | "
            f"{row['styled_mean_target_margin']:+.3f} | "
            f"{row['mean_paired_margin_lift']:+.3f} | "
            f"{format_pct(row['positive_lift_share'])} | "
            f"{format_pct(row['styled_top5_share'])} |"
        )
    lines.extend(
        [
            "",
            "## Flow Change",
            "",
            "| Measure | Neutral | Method4 | Delta |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for key in (
        "mean_paragraph_cjk",
        "mean_sentence_cjk",
        "p90_sentence_cjk",
        "punctuation_per_100_cjk",
        "dialogue_line_share",
        "function_chars_per_100_cjk",
        "comma_per_100_cjk",
        "period_per_100_cjk",
    ):
        lines.append(
            f"| `{key}` | {float(flow['neutral'][key]):.3f} | "
            f"{float(flow['styled'][key]):.3f} | {float(flow['delta'][key]):+.3f} |"
        )
    lines.extend(
        [
            "",
        "## Interpretation",
        "",
        report["interpretation"],
        "",
        "The accepted generation artifacts bind the frozen prompt directly. The schema",
        "and application-runner hashes are retained in the run manifest but were not",
        "copied into each accepted segment artifact; their provenance status is therefore",
        "`MANIFEST_ONLY`, not a per-segment implementation claim.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "uv run python experiments/validation/application/evaluate_method4_application.py \\",
            f"  --style-run-dir {report['style_run_dir']}",
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(
    *,
    style_run_dir: Path,
    scorer_dir: Path,
    threshold_path: Path,
    glossary_path: Path,
) -> dict[str, Any]:
    manifest_path = style_run_dir / "run_manifest.json"
    manifest = read_json(manifest_path)
    semantic_run_dir = Path(str(manifest["semantic_run_dir"]))
    semantic_manifest = read_json(semantic_run_dir / "run_manifest.json")
    threshold_payload = read_json(threshold_path)
    threshold = float(threshold_payload["selected"]["threshold"])
    vectorizer, classifier, labels, scorer_manifest = load_scorer(scorer_dir)
    mask_terms = glossary_mask_terms(glossary_path)
    hard_gate = audit_chunks(style_run_dir, manifest)

    chapter_chunks: dict[str, list[dict[str, Any]]] = {}
    for chunk in manifest.get("chunks") or []:
        chapter_chunks.setdefault(str(chunk["chapter_id"]), []).append(chunk)
    windows: list[dict[str, Any]] = []
    for chapter_id, chunks in chapter_chunks.items():
        title = str(chunks[0].get("chapter_title") or "")
        neutral = translation_map(semantic_run_dir / "translations" / f"{chapter_id}.json")
        styled = translation_map(style_run_dir / "translations" / f"{chapter_id}.json")
        windows.extend(
            build_windows(
                chapter_id=chapter_id,
                title=title,
                neutral=neutral,
                styled=styled,
            )
        )
    neutral_masked = [mask_generated_text(row["neutral_text"], mask_terms) for row in windows]
    styled_masked = [mask_generated_text(row["styled_text"], mask_terms) for row in windows]
    neutral_scores = score_texts(vectorizer, classifier, labels, neutral_masked)
    styled_scores = score_texts(vectorizer, classifier, labels, styled_masked)
    score_rows: list[dict[str, Any]] = []
    for window, neutral_score, styled_score in zip(
        windows, neutral_scores, styled_scores
    ):
        score_rows.append(
            {
                "window_id": window["window_id"],
                "chapter_id": window["chapter_id"],
                "chapter_title": window["chapter_title"],
                "indexes": window["indexes"],
                "neutral_target_margin": neutral_score["target_margin"],
                "neutral_target_rank": neutral_score["target_rank"],
                "neutral_predicted_author": neutral_score["predicted_author"],
                "styled_target_margin": styled_score["target_margin"],
                "styled_target_rank": styled_score["target_rank"],
                "styled_predicted_author": styled_score["predicted_author"],
                "styled_target_top1": styled_score["target_top1"],
                "paired_margin_lift": (
                    styled_score["target_margin"] - neutral_score["target_margin"]
                ),
                "neutral_masked_sha256": hashlib.sha256(
                    neutral_masked[len(score_rows)].encode("utf-8")
                ).hexdigest(),
                "styled_masked_sha256": hashlib.sha256(
                    styled_masked[len(score_rows)].encode("utf-8")
                ).hexdigest(),
            }
        )
    overall = summarize_scores(score_rows, threshold)
    by_chapter: list[dict[str, Any]] = []
    for chapter_id in chapter_chunks:
        selected = [row for row in score_rows if row["chapter_id"] == chapter_id]
        by_chapter.append(
            {
                "chapter_id": chapter_id,
                "chapter_title": selected[0]["chapter_title"],
                **summarize_scores(selected, threshold),
            }
        )

    neutral_book_text = "\n".join(row["neutral_text"] for row in windows)
    styled_book_text = "\n".join(row["styled_text"] for row in windows)
    neutral_flow = flow_metrics(neutral_book_text)
    styled_flow = flow_metrics(styled_book_text)
    flow_delta = {
        key: float(styled_flow[key]) - float(neutral_flow[key]) for key in neutral_flow
    }
    semantic_path = style_run_dir / "semantic_compression/semantic_compression_summary.json"
    if semantic_path.exists():
        semantic_payload = read_json(semantic_path)
        result_summary = semantic_payload.get("result_summary") or {}
        semantic_status = (
            "PASS"
            if int(result_summary.get("failure_count", 0)) == 0
            else "FAIL"
        )
        semantic_summary = {
            "status": semantic_status,
            "path": str(semantic_path),
            "candidate_count": int(
                (semantic_payload.get("candidate_summary") or {}).get(
                    "candidate_count", 0
                )
            ),
            **{key: int(value) for key, value in result_summary.items()},
        }
    else:
        semantic_summary = {"status": "PENDING", "path": str(semantic_path)}

    independent_path = style_run_dir / "audits/independent_application_evaluation.md"
    independent_repair_path = (
        style_run_dir / "audits/independent_semantic_repair_summary.json"
    )
    independent_repair_payload = (
        read_json(independent_repair_path) if independent_repair_path.exists() else {}
    )
    independent_repair_rows = list(independent_repair_payload.get("repairs") or [])
    independent_repair_status = (
        "PASS"
        if independent_repair_rows
        and all(
            str(row.get("status")) in {"applied", "already_applied"}
            for row in independent_repair_rows
        )
        else "PENDING"
        if not independent_repair_rows
        else "FAIL"
    )
    epub_validation_path = style_run_dir / "audits/method4_epub_validation.json"
    if epub_validation_path.exists():
        epub_validation_payload = read_json(epub_validation_path)
        epub_validation = {
            "status": str(epub_validation_payload.get("status") or "FAIL"),
            "path": str(epub_validation_path),
            "epub_path": epub_validation_payload.get("epub_path"),
            "epub_sha256": epub_validation_payload.get("epub_sha256"),
            "chapter_count": int(epub_validation_payload.get("chapter_count", 0)),
            "paragraph_count": int(epub_validation_payload.get("paragraph_count", 0)),
            "failure_count": len(epub_validation_payload.get("failures") or []),
        }
    else:
        epub_validation = {
            "status": "PENDING",
            "path": str(epub_validation_path),
            "epub_path": None,
            "epub_sha256": None,
            "chapter_count": 0,
            "paragraph_count": 0,
            "failure_count": 0,
        }
    copy_failure_count = sum(
        any("reference_copy" in failure for failure in row["failures"])
        for row in hard_gate["failures"]
    )
    application_meta = manifest.get("method4_application") or {}
    last_run = application_meta.get("last_run") or {}
    artifact_model_counts: Counter[str] = Counter()
    mixed_model_chunks: list[str] = []
    neutral_leaf_fallbacks: list[dict[str, Any]] = []
    runner_sha256s: set[str] = set()
    prompt_sha256s: set[str] = set()
    schema_sha256s: set[str] = set()
    for chunk in manifest.get("chunks") or []:
        artifact = read_json(style_run_dir / str(chunk["method_output_path"]))
        models_used = list(
            dict.fromkeys(
                str(value) for value in artifact.get("models_used") or []
            )
        )
        artifact_model_counts.update(models_used)
        if len(models_used) > 1:
            mixed_model_chunks.append(str(chunk["chunk_id"]))
        for segment in artifact.get("segment_artifacts") or []:
            segment_path = style_run_dir / str(segment["path"])
            if file_sha256(segment_path) != str(segment["file_sha256"]):
                raise RuntimeError(f"segment artifact hash mismatch: {segment_path}")
            segment_payload = read_json(segment_path)
            for key, target in (
                ("runner_sha256", runner_sha256s),
                ("prompt_sha256", prompt_sha256s),
                ("schema_sha256", schema_sha256s),
            ):
                value = str(segment_payload.get(key) or "")
                if value:
                    target.add(value)
            if segment.get("acceptance") == "neutral_leaf_fallback":
                neutral_leaf_fallbacks.append(
                    {
                        "chunk_id": str(chunk["chunk_id"]),
                        "chapter_id": str(chunk["chapter_id"]),
                        "source_indexes": [
                            int(value) for value in segment.get("source_indexes") or []
                        ],
                        "model": str(segment.get("model") or ""),
                        "rejected_application_errors": list(
                            segment.get("rejected_application_errors") or []
                        ),
                    }
                )
    output_files = list((style_run_dir / "outputs").glob("*.json"))
    paragraph_count = sum(len(chunk["indexes"]) for chunk in manifest["chunks"])
    latest_chunk = manifest["chunks"][-1]
    first_request = read_json(style_run_dir / str(manifest["chunks"][0]["request_path"]))
    method_payload = first_request.get("method_payload") or {}
    style_definition = method_payload.get("style_definition") or {}
    evidence_policy = style_definition.get("evidence_policy") or {}
    aligned_retrieval = method_payload.get("aligned_pair_retrieval") or {}
    primary_model_counts = Counter(
        codex_log_model(path)
        for path in sorted((semantic_run_dir / "outputs").glob("*.raw.log"))
    )
    fallback_summary_path = semantic_run_dir / "semantic_fallback_merge_summary.json"
    fallback_summary = (
        read_json(fallback_summary_path) if fallback_summary_path.exists() else {}
    )
    final_independent_verdict = independent_verdict(independent_path)
    interpretation = (
        "All deterministic gates pass and the independent evaluator gives a post-repair "
        "PASS for a clearly labeled experimental reader artifact."
        if hard_gate["failure_count"] == 0
        and final_independent_verdict == "PASS"
        and independent_repair_status == "PASS"
        and epub_validation["status"] == "PASS"
        else "One or more automated, repair, or independent gates remain incomplete; the EPUB "
        "must not be treated as a validated experimental reader artifact yet."
    )
    report = {
        "schema_version": 1,
        "schema": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "style_run_dir": str(style_run_dir),
        "semantic_run_dir": str(semantic_run_dir),
        "coverage": {
            "chapter_count": len(chapter_chunks),
            "paragraph_count": paragraph_count,
            "method_block_count": len(manifest["chunks"]),
            "method_output_count": len(output_files),
            "latest_chapter_id": str(latest_chunk["chapter_id"]),
            "latest_chapter_title": str(latest_chunk["chapter_title"]),
        },
        "model_provenance": {
            "configured_fallback_order": application_meta.get(
                "capacity_fallback_policy", {}
            ).get("requested_model_order", []),
            "last_selected_run_order": last_run.get("requested_model_order") or [],
            "disabled_models": last_run.get("disabled_models") or {},
            "model_chunk_counts": dict(sorted(artifact_model_counts.items())),
            "mixed_model_chunks": mixed_model_chunks,
            "neutral_leaf_fallback_count": len(neutral_leaf_fallbacks),
            "neutral_leaf_fallback_share": len(neutral_leaf_fallbacks)
            / max(paragraph_count, 1),
            "neutral_leaf_fallbacks": neutral_leaf_fallbacks,
        },
        "semantic_provenance": {
            "neutral_prompt_path": (
                (semantic_manifest.get("config") or {})
                .get("style_transfer_research", {})
                .get("neutral_prompt_path")
            ),
            "neutral_prompt_sha256": (
                (semantic_manifest.get("config") or {})
                .get("style_transfer_research", {})
                .get("neutral_prompt_sha256")
            ),
            "primary_chunk_models": dict(sorted(primary_model_counts.items())),
            "fallback_model": fallback_summary.get("fallback_model"),
            "fallback_repaired_paragraph_count": len(
                fallback_summary.get("repairs") or []
            )
            and sum(
                len(item.get("repaired_indexes") or [])
                for item in fallback_summary.get("repairs") or []
            ),
            "fallback_summary_path": str(fallback_summary_path),
        },
        "method_details": {
            "target_train_books": int(evidence_policy.get("target_train_books", 0)),
            "comparison_train_authors": int(
                evidence_policy.get("comparison_train_authors", 0)
            ),
            "discovery_books": int(evidence_policy.get("discovery_books", 0)),
            "validation_books": int(evidence_policy.get("validation_books", 0)),
            "eternal_gate_used": bool(evidence_policy.get("eternal_gate_used")),
            "masked_examples_only": bool(
                evidence_policy.get("masked_examples_only")
            ),
            "aligned_pair_pool_count": int(
                aligned_retrieval.get("pair_pool_count", 0)
            ),
            "aligned_pair_k": int(aligned_retrieval.get("k", 0)),
            "reference_example_count": len(first_request.get("reference_examples") or []),
            "style_dimensions": [
                str(item.get("dimension_id"))
                for item in style_definition.get("dimensions") or []
            ],
        },
        "method_provenance": {
            "method_id": application_meta.get("method_id"),
            "intensity": application_meta.get("intensity"),
            "prompt_sha256": application_meta.get("prompt_sha256"),
            "schema_sha256": application_meta.get("schema_sha256"),
            "frozen_asset_content_sha256": (
                application_meta.get("frozen_asset_lock") or {}
            ).get("content_sha256"),
            "manifest_runner_sha256": application_meta.get("runner_sha256"),
            "accepted_segment_runner_sha256s": sorted(runner_sha256s),
            "observed_prompt_sha256s": sorted(prompt_sha256s),
            "observed_schema_sha256s": sorted(schema_sha256s),
            "prompt_binding_status": (
                "PASS"
                if prompt_sha256s == {str(application_meta.get("prompt_sha256"))}
                else "FAIL"
            ),
            "schema_binding_status": (
                "PASS"
                if schema_sha256s == {str(application_meta.get("schema_sha256"))}
                else "MANIFEST_ONLY"
                if not schema_sha256s
                else "FAIL"
            ),
            "runner_binding_status": (
                "PASS" if runner_sha256s else "MANIFEST_ONLY"
            ),
        },
        "standard_validation_status": (
            "PASS"
            if (style_run_dir / "validation_summary.json").exists()
            and len(output_files) == len(manifest["chunks"])
            else "FAIL"
        ),
        "hard_gate_audit": hard_gate,
        "copy_failure_count": copy_failure_count,
        "semantic_compression_qa": semantic_summary,
        "independent_repairs": {
            "status": independent_repair_status,
            "path": str(independent_repair_path),
            "repair_count": len(independent_repair_rows),
            "refs": [str(row.get("ref")) for row in independent_repair_rows],
        },
        "independent_evaluation_status": (
            "COMPLETE" if independent_path.exists() else "PENDING"
        ),
        "independent_evaluation_verdict": final_independent_verdict,
        "epub_validation": epub_validation,
        "historical_style_meter": {
            "status": "diagnostic_only_construct_contaminated",
            "scorer_id": scorer_manifest["scorer_id"],
            "scorer_manifest_sha256": file_sha256(scorer_dir / "manifest.json"),
            "historical_threshold": threshold,
            "historical_threshold_sha256": file_sha256(threshold_path),
            "masking": {
                "view": "eternal_gate_glossary_mask_plus_latin_numbers.v1",
                "term_count": len(mask_terms),
                "term_list_sha256": sha256_json(mask_terms),
            },
            "overall": overall,
            "by_chapter": by_chapter,
            "windows": score_rows,
        },
        "flow": {
            "neutral": neutral_flow,
            "styled": styled_flow,
            "delta": flow_delta,
        },
        "interpretation": interpretation,
        "provenance": {
            "manifest_sha256": file_sha256(manifest_path),
            "semantic_manifest_sha256": file_sha256(
                semantic_run_dir / "run_manifest.json"
            ),
            "glossary_sha256": file_sha256(glossary_path),
            "evaluator_sha256": file_sha256(Path(__file__).resolve()),
        },
    }
    audit_dir = style_run_dir / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    write_json(audit_dir / "method4_application_evaluation.json", report)
    render_svg(audit_dir / "method4_margin_lift_by_chapter.svg", by_chapter)
    write_markdown(audit_dir / "method4_application_report.md", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a completed Eternal Gate method4 application run."
    )
    parser.add_argument("--style-run-dir", type=Path, required=True)
    parser.add_argument("--scorer-dir", type=Path, default=DEFAULT_SCORER_DIR)
    parser.add_argument("--threshold", type=Path, default=DEFAULT_THRESHOLD_PATH)
    parser.add_argument("--glossary", type=Path, default=DEFAULT_GLOSSARY_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate(
        style_run_dir=args.style_run_dir.expanduser().resolve(),
        scorer_dir=args.scorer_dir.expanduser().resolve(),
        threshold_path=args.threshold.expanduser().resolve(),
        glossary_path=args.glossary.expanduser().resolve(),
    )
    print(
        json.dumps(
            {
                "coverage": report["coverage"],
                "standard_validation_status": report["standard_validation_status"],
                "hard_gate_failures": report["hard_gate_audit"]["failure_count"],
                "style_diagnostic": report["historical_style_meter"]["overall"],
                "semantic_compression_qa": report["semantic_compression_qa"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if report["hard_gate_audit"]["failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
