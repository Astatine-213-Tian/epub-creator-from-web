from __future__ import annotations

import argparse
import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


from experiments.shared.paths import RESEARCH_ROOT


PROJECT_ROOT = RESEARCH_ROOT
DEFAULT_EXPERIMENT_ROOT = Path(
    "generated/style_research/style_transfer_experiments"
)
DEFAULT_EVALUATION = Path(
    "generated/style_research/style_transfer_experiments/evaluations/"
    "proxy_v1_gpt54_20260711_v2/screening_initial/evaluation_summary.json"
)
DEFAULT_OUTPUT = Path(
    "docs/reports/03_transfer_iteration1_prompt_methods.md"
)
DEFAULT_REPORT_ARTIFACTS = Path(
    "generated/style_research/style_transfer_experiments/evaluations/"
    "proxy_v1_gpt54_20260711_v2/research_report"
)


METHOD_EXPLANATIONS: dict[str, dict[str, str]] = {
    "neutral_only": {
        "short": "No style rewrite",
        "detail": (
            "The production-compatible neutral Chinese draft is scored unchanged. "
            "This is the negative control and the paired baseline for every lift value."
        ),
        "evidence": "English semantic source plus neutral Chinese only.",
    },
    "generic_author_style_light": {
        "short": "Author-name prompt, minimal rewrite",
        "detail": (
            "The model sees the target author name but no style cards or examples. "
            "It is told to make only a light, source-supported recast and to skip weak opportunities."
        ),
        "evidence": "Target author name only; no corpus excerpts are supplied.",
    },
    "generic_author_style_strong": {
        "short": "Author-name prompt, stress-test rewrite",
        "detail": (
            "The same author-name-only cue is applied assertively as a stress test. "
            "This tests whether stronger generic prompting raises style strength at the cost of fidelity."
        ),
        "evidence": "Target author name only; stronger rewrite-scope instruction.",
    },
    "global_style_cards": {
        "short": "Corpus-derived global style rules",
        "detail": (
            "The model receives compact executable rules derived from 29 target-author training books "
            "against 49 comparison authors. Rules cover dialogue tags, punctuation, function words, "
            "and sentence flow, each with a semantic guardrail."
        ),
        "evidence": "Aggregate statistics from target train books and the 49-author train corpus.",
    },
    "scene_routed_style_cards": {
        "short": "Scene-specific corpus rules",
        "detail": (
            "The neutral draft is routed to scene types such as dialogue, action, reflection, care, "
            "travel, or exposition. Only cards registered for the inferred scene are supplied. "
            "Routing uses neutral Chinese, never hidden target metadata."
        ),
        "evidence": "Scene-conditioned cards derived from target and comparison train books.",
    },
    "sentence_flow_cards": {
        "short": "Sentence and paragraph rhythm rules",
        "detail": (
            "The intervention isolates flow: short-sentence share, mean sentence length, "
            "comma-linked clauses, and paragraph progression. It tests whether cadence alone carries "
            "enough author signal beyond vocabulary."
        ),
        "evidence": "Target-versus-comparison sentence-flow statistics from train books.",
    },
    "function_word_punctuation_dialogue_cards": {
        "short": "Interpretable grammar and dialogue rules",
        "detail": (
            "The intervention isolates lower-content cues: Chinese function-word families, "
            "punctuation, plain speech tags, laughter tags, and micro-reactions. Each rule prohibits "
            "adding unsupported emotion, motive, causality, or action."
        ),
        "evidence": "Interpretable feature statistics from target and comparison train books.",
    },
    "retrieved_examples_only": {
        "short": "Nearest target-author examples",
        "detail": (
            "For each neutral draft, retrieval returns three entity-masked target-author train chunks "
            "of at most 450 CJK characters. The model gets examples but no explicit style definition."
        ),
        "evidence": "Three entity_masked_v3 examples; train split only; near duplicates excluded.",
    },
    "scene_cards_plus_examples": {
        "short": "Scene rules plus target examples",
        "detail": (
            "This hybrid combines scene-routed executable cards with three retrieved masked examples. "
            "It tests whether explicit constraints and tacit demonstrations are complementary."
        ),
        "evidence": "Scene cards plus three masked target-author train examples.",
    },
    "contrastive_examples": {
        "short": "Target examples versus hard negatives",
        "detail": (
            "The model receives three target-author examples and matched hard-negative examples from "
            "other authors. It is told to infer only structural differences shared by target examples "
            "and absent from negatives, reducing genre/topic imitation."
        ),
        "evidence": "Masked target and hard-negative train examples from the 50-author corpus.",
    },
    "llm_close_reading_style_definition": {
        "short": "LLM close-reading definition",
        "detail": (
            "A separate close-reading pass synthesizes discourse, grammar, narrator stance, and "
            "scene-mechanics observations from masked target and comparison passages. The transfer "
            "model receives that qualitative definition without aggregate statistical gates."
        ),
        "evidence": "Close reading of 78 masked train passages: 29 target and 49 comparison.",
    },
    "llm_close_reading_cards_statistical_gates": {
        "short": "Close reading constrained by statistics",
        "detail": (
            "This hybrid combines the LLM close-reading definition with corpus-derived cards and "
            "requires qualitative claims to agree with measured target-versus-comparison tendencies."
        ),
        "evidence": "Masked close reading plus statistically supported train-corpus cards.",
    },
}


EXCLUSION_LABELS = {
    "control_not_promotable": "control, not promotable",
    "own_paired_lift_not_positive": "own-author lift was not positive",
    "cross_paired_lift_not_positive": "cross-author lift was not positive",
    "own_hard_fidelity_failure_rate_above_0_10": "own-author fidelity failures exceeded 10%",
    "cross_hard_fidelity_failure_rate_above_0_10": "cross-author fidelity failures exceeded 10%",
    "reference_copy_failures_present": "reference-copy failure detected",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the detailed Eternal Gate style-transfer research report."
    )
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_REPORT_ARTIFACTS)
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def percentage(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def number(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}"


def split_target_books(splits: dict[str, list[dict[str, Any]]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for split, books in splits.items():
        result[split] = [
            row["title"] for row in books if row.get("author") == "非天夜翔"
        ]
    return result


def promotion_decisions(promotion: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (row["method_id"], row["intensity"]): row
        for row in promotion.get("all_candidates", [])
    }


def chart_label(method_id: str) -> str:
    labels = {
        "neutral_only": "Neutral control",
        "generic_author_style_light": "Generic light",
        "generic_author_style_strong": "Generic strong",
        "global_style_cards": "Global cards",
        "scene_routed_style_cards": "Scene cards",
        "sentence_flow_cards": "Flow cards",
        "function_word_punctuation_dialogue_cards": "Function/dialogue",
        "retrieved_examples_only": "Examples only",
        "scene_cards_plus_examples": "Scene + examples",
        "contrastive_examples": "Contrastive",
        "llm_close_reading_style_definition": "Close reading",
        "llm_close_reading_cards_statistical_gates": "Close + stats",
    }
    return labels.get(method_id, method_id)


def write_dual_bar_chart(
    path: Path,
    *,
    title: str,
    rows: list[tuple[str, float, float]],
    minimum: float,
    maximum: float,
    own_label: str,
    cross_label: str,
    tick_format: str,
    reference: float | None = None,
) -> None:
    width = 1120
    label_width = 210
    plot_width = 830
    top = 92
    row_height = 42
    height = top + row_height * len(rows) + 70
    left = label_width

    def x(value: float) -> float:
        return left + (value - minimum) / (maximum - minimum) * plot_width

    def fmt(value: float) -> str:
        if tick_format == "percent":
            return f"{value * 100:.0f}%"
        return f"{value:+.2f}"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="30" font-family="Arial,sans-serif" font-size="20" font-weight="700" fill="#1f2933">{html.escape(title)}</text>',
        f'<rect x="{left}" y="48" width="14" height="14" fill="#176b87"/><text x="{left + 20}" y="60" font-family="Arial,sans-serif" font-size="13" fill="#334e68">{html.escape(own_label)}</text>',
        f'<rect x="{left + 180}" y="48" width="14" height="14" fill="#d1495b"/><text x="{left + 200}" y="60" font-family="Arial,sans-serif" font-size="13" fill="#334e68">{html.escape(cross_label)}</text>',
    ]
    for index in range(6):
        value = minimum + (maximum - minimum) * index / 5
        xpos = x(value)
        parts.append(
            f'<line x1="{xpos:.1f}" y1="72" x2="{xpos:.1f}" y2="{height - 48}" stroke="#d9e2ec" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{xpos:.1f}" y="{height - 25}" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="#627d98">{html.escape(fmt(value))}</text>'
        )
    if minimum < 0 < maximum:
        zero_x = x(0)
        parts.append(
            f'<line x1="{zero_x:.1f}" y1="72" x2="{zero_x:.1f}" y2="{height - 48}" stroke="#52616b" stroke-width="1.5"/>'
        )
    if reference is not None and minimum <= reference <= maximum:
        ref_x = x(reference)
        parts.append(
            f'<line x1="{ref_x:.1f}" y1="72" x2="{ref_x:.1f}" y2="{height - 48}" stroke="#7c3aed" stroke-width="2" stroke-dasharray="5 4"/>'
        )
        parts.append(
            f'<text x="{ref_x + 5:.1f}" y="82" font-family="Arial,sans-serif" font-size="11" fill="#6d28d9">threshold {reference:+.3f}</text>'
        )
    baseline_x = x(0 if minimum < 0 < maximum else minimum)
    for index, (label, own, cross) in enumerate(rows):
        y = top + index * row_height
        parts.append(
            f'<text x="{left - 12}" y="{y + 16}" text-anchor="end" font-family="Arial,sans-serif" font-size="13" fill="#243b53">{html.escape(label)}</text>'
        )
        for offset, value, color in ((3, own, "#176b87"), (20, cross, "#d1495b")):
            value_x = x(value)
            start = min(value_x, baseline_x)
            bar_width = max(abs(value_x - baseline_x), 1.0)
            parts.append(
                f'<rect x="{start:.1f}" y="{y + offset}" width="{bar_width:.1f}" height="13" fill="{color}" rx="1"/>'
            )
            anchor = "start" if value >= (0 if minimum < 0 < maximum else minimum) else "end"
            text_x = value_x + (5 if anchor == "start" else -5)
            parts.append(
                f'<text x="{text_x:.1f}" y="{y + offset + 11}" text-anchor="{anchor}" font-family="Arial,sans-serif" font-size="11" fill="#243b53">{html.escape(fmt(value))}</text>'
            )
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def build_report(
    *,
    experiment_root: Path,
    evaluation_path: Path,
    output_path: Path,
    artifact_dir: Path,
) -> None:
    evaluation = read_json(evaluation_path)
    protocol = read_json(experiment_root / "protocols/evaluation_protocol.v1.json")
    model_config = read_json(experiment_root / "protocols/model_run_config.v1.json")
    method_registry = read_json(
        experiment_root / "method_registry/style_methods.v1.json"
    )
    threshold = read_json(experiment_root / "calibration/style_meter_threshold.v1.json")
    sample_summary = read_json(
        experiment_root / "sample_sets/development_proxy_v1.summary.json"
    )
    promotion = read_json(
        experiment_root / "promotions/screening_v1.provisional_shortlist.v1.json"
    )
    scorer_config = read_json(
        experiment_root
        / "scorers"
        / evaluation["scorer_binding"]["scorer_id"]
        / "scorer_config.json"
    )
    corpus_splits = read_json(PROJECT_ROOT / "generated/style_research/corpus/splits.json")
    chunk_report = read_json(PROJECT_ROOT / "generated/style_research/corpus/chunk_report.json")
    target_books = split_target_books(corpus_splits)
    decisions = promotion_decisions(promotion)
    method_bindings = {row["id"]: row for row in method_registry["methods"]}
    combinations = evaluation["combinations"]

    lift_rows: list[tuple[str, float, float]] = []
    fidelity_rows: list[tuple[str, float, float]] = []
    margin_rows: list[tuple[str, float, float]] = []
    for row in combinations:
        own = row["arm_summaries"]["own_author_reconstruction"]
        cross = row["arm_summaries"]["cross_author_transfer"]
        label = chart_label(row["method_id"])
        lift_rows.append(
            (label, own["mean_paired_margin_lift"], cross["mean_paired_margin_lift"])
        )
        fidelity_rows.append(
            (
                label,
                own["hard_fidelity_failure_rows"] / own["expected_rows"],
                cross["hard_fidelity_failure_rows"] / cross["expected_rows"],
            )
        )
        margin_rows.append(
            (label, own["mean_target_margin"], cross["mean_target_margin"])
        )

    charts = artifact_dir / "charts"
    write_dual_bar_chart(
        charts / "iteration1_paired_lift.svg",
        title="Iteration 1: mean target-author margin lift over neutral",
        rows=lift_rows,
        minimum=-0.08,
        maximum=0.09,
        own_label="Own-author reconstruction",
        cross_label="Cross-author transfer",
        tick_format="number",
    )
    write_dual_bar_chart(
        charts / "iteration1_fidelity_failures.svg",
        title="Iteration 1: deterministic hard-fidelity failure rate",
        rows=fidelity_rows,
        minimum=0.0,
        maximum=0.75,
        own_label="Own-author reconstruction",
        cross_label="Cross-author transfer",
        tick_format="percent",
        reference=0.10,
    )
    write_dual_bar_chart(
        charts / "iteration1_target_margin.svg",
        title="Iteration 1: mean absolute target-author decision margin",
        rows=margin_rows,
        minimum=-1.4,
        maximum=0.3,
        own_label="Own-author reconstruction",
        cross_label="Cross-author transfer",
        tick_format="number",
        reference=threshold["selected"]["threshold"],
    )

    chart_prefix = (
        "../../generated/style_research/style_transfer_experiments/evaluations/"
        "proxy_v1_gpt54_20260711_v2/research_report/charts"
    )
    now = datetime.now(UTC).isoformat(timespec="seconds")
    split_counts = {name: len(rows) for name, rows in corpus_splits.items()}
    split_author_counts = {
        name: len({row["author"] for row in rows})
        for name, rows in corpus_splits.items()
    }
    selected_threshold = threshold["selected"]
    target_train_chunks = scorer_config["training_contract"][
        "expected_train_chunks_by_author"
    ][scorer_config["target_author"]]
    comparison_train_chunks = chunk_report["chunks_by_split"]["train"] - target_train_chunks
    prompt_versions = model_config["prompt_versions"]

    lines: list[str] = [
        "# Eternal Gate Author-Style Transfer Research Report",
        "",
        f"Generated from frozen artifacts at `{now}`.",
        "",
        "## Current Status: Iteration 2",
        "",
        "Iteration 2 has now tested three aligned pseudo-parallel transfer methods with `gpt-5.5` "
        "on a fresh 36-row screen. All three methods increased mean target-author margin in both "
        "benchmark arms, but none passed the preregistered joint gate because own-author hard-"
        "fidelity failures remained above 10%. The reused binary threshold also has stale "
        "iteration-1 allocation/protocol bindings, so no threshold-based accuracy or 80% result "
        "can be claimed. The 116-row confirmation set remains unopened.",
        "",
        "Detailed procedures, method definitions, per-arm results, graphs, retry statistics, and "
        "the independent evaluator's findings are in "
        "[the iteration-2 aligned-pairs report](04_transfer_iteration2_aligned_pairs.md).",
        "",
        "## Iteration 1 Executive Result",
        "",
        "The author-identification prerequisite passed, but the first registered style-transfer "
        "screen did not. The exact masked character n-gram meter reaches **87.8%** 50-author "
        "test accuracy and **83.6%** balanced accuracy, yet none of the 11 non-control transfer "
        "methods met the joint screening gate. The frozen promotion decision is "
        "**`no_method_qualified`**, so confirmation and final validation have not started.",
        "",
        "| Stage | Data | Procedure | Result | Status |",
        "| --- | --- | --- | --- | --- |",
        "| 0. Corpus cleanup | 200 books, 50 authors | Clean, deduplicate, book-level split, six chunk views | 199 usable books chunked; 87,174 chunks; zero checked residue hits | complete |",
        "| 1. Author-style meter | 24,957 train / 23,683 dev / 35,812 test chunks | Compare n-grams and interpretable classifiers on clean and entity-masked text | Selected exact 2-4 character n-gram class-balanced hinge model: 87.8% test accuracy | complete |",
        "| 2. Generated-domain calibration | 32 held-out target passages | Original target Chinese versus production-compatible neutral Chinese | Threshold `0.218175`; sensitivity 100%; neutral FPR 0% | complete |",
        "| 3. Proxy screening | 24 own-author + 12 cross-author chunks | Neutral control plus 11 transfer interventions, one frozen output per method/sample | 396/396 non-control outputs valid; all methods 0% deterministic style success | complete, failed gate |",
        "| 4. Independent outcome audit | Frozen evaluation and promotion artifacts | Separate evaluator checks rule compliance and interpretation | NO-GO; empty shortlist is correct | complete |",
        "| 5. Refinement / repair | Would use shortlisted methods only | Lighter intensity and one critique repair | Not run because shortlist is empty | stopped by protocol |",
        "| 6. Confirmation | 104 own-author + 48 cross-author rows | Exact promoted methods only | Not opened | locked |",
        "| 7. Final validation | Four transfer-held-out target books; 80 rows planned but not yet sampled | One-time transfer replication after confirmation passes | Row IDs do not yet exist | locked |",
        "",
        "## Research Question",
        "",
        "Can a controlled second-pass rewrite move production-compatible neutral Chinese toward "
        "the prose style of 非天夜翔 while preserving the English source meaning, natural Chinese "
        "readability, paragraph structure, and no-copy constraints?",
        "",
        "The intended production pipeline remains:",
        "",
        "```text",
        "Eternal Gate English",
        "  -> pass 1: neutral Simplified Chinese (English is semantic authority)",
        "  -> pass 2: tested author-style transfer",
        "  -> pass 3: English-grounded semantic/readability validation and repair",
        "```",
        "",
        "No pass-2 method is approved for production yet.",
        "",
        "## Experimental Design and Data Lineage",
        "",
        "The **experimental unit is one contiguous prose chunk**, normally about 1,500 cleaned "
        "CJK characters. Book is the primary independence boundary: chunks from the same book "
        "never cross corpus train/dev/test splits. Confirmation inference is registered at the "
        "book or author cluster level rather than treating thousands of correlated chunks as "
        "independent observations.",
        "",
        "| Data role | Authors / books | Units | May influence | Access status |",
        "| --- | --- | ---: | --- | --- |",
        f"| Classifier fit | {split_author_counts['train']} authors / {split_counts['train']} train books | {chunk_report['chunks_by_split']['train']:,} masked chunks | Vectorizer vocabulary and classifier weights | used |",
        f"| Classifier development | {split_author_counts['dev']} authors / {split_counts['dev']} dev books | {chunk_report['chunks_by_split']['dev']:,} masked chunks | Diagnostic model comparison | used |",
        f"| Classifier book-disjoint benchmark | {split_author_counts['test']} authors / {split_counts['test']} test books | {chunk_report['chunks_by_split']['test']:,} masked chunks | Reported author-classifier metrics only | used; not available to transfer prompts |",
        f"| Transfer evidence pool | 1 target author / 29 target train books, contrasted with 49 authors | {target_train_chunks:,} target + {comparison_train_chunks:,} comparison train chunks | Cards, retrieval, close reading, and style-meter weights | used |",
        "| Proxy benchmark pool | 13 authors / 20 books | 220 sampled chunks | Calibration, screening, or confirmation according to frozen IDs | allocated |",
        "| Generated-domain calibration | 1 target author / 8 proxy books | 32 chunks, 4 per book | One frozen target-margin threshold | used once |",
        "| Iteration-1 screening | 13 authors / 20 books | 24 own-author + 12 cross-author chunks | Method-family promotion decision | used |",
        "| Iteration-1 confirmation (historical) | same 13 authors / 20 books | 104 own-author + 48 cross-author chunks | Registered iteration-1 endpoint; superseded by the 116-row iteration-2 allocation | unopened in iteration 1 |",
        "| Final validation | 1 target author / 4 books held out from transfer development | Planned 80 chunks, 20 per book; exact row IDs do not yet exist | One-time transfer replication; books contributed to Stage 1 classifier benchmark metrics | book reservation only |",
        "| Eternal Gate production input | English novel; no Chinese target exists | 0 research rows | Nothing in method selection | excluded from all training and benchmark sets |",
        "",
        "No style-transfer LLM was fine-tuned in iteration 1. Here, **training evidence** means "
        "corpus material used to fit the style meter or construct prompts; **development data** "
        "means the frozen proxy books used for calibration/screening/confirmation; and **final "
        "test data** means the four target-author books reserved for one-time validation. These "
        "roles are kept separate throughout the report.",
        "",
        "### Information available at each generation step",
        "",
        "| Step | Inputs visible to that step | Inputs deliberately hidden | Output |",
        "| --- | --- | --- | --- |",
        "| Benchmark English construction | Held-out original Chinese, paragraph IDs | Method label and later score | Synthetic plain-English semantic source |",
        "| English QA/repair | Original Chinese plus synthetic English | Style-transfer output | Approved or selectively repaired English |",
        "| Neutral Chinese generation | Approved English plus paragraph IDs | Original Chinese, author/book identity, style evidence | Production-compatible neutral Chinese |",
        "| Style-transfer generation | Approved English, byte-identical neutral Chinese, and registered method evidence | Original Chinese, book title, sampling stratum, evaluation role, scores | One candidate Chinese rewrite |",
        "| Deterministic evaluation | Candidate, neutral control, frozen scorer, hidden allocation, registered references | No adaptive prompt changes | Style scores and hard surface/copy flags |",
        "| Independent semantic/readability judgment | English source and blinded neutral/candidate pair | Method identity and target original | Paired semantic/readability findings; only for deterministic survivors |",
        "",
        "## Stage 0: Corpus Construction",
        "",
        "### Source corpus",
        "",
        "The current corpus contains **200 books by 50 authors**. Every retained author has at "
        "least three books. Cleaning operates on copies under `generated/style_research/corpus/`; "
        "raw TXT files under `datasets/raw/` are not modified. The cleaner removes chapter headings, "
        "author-note blocks, URL lines, and known source boilerplate. Exact cleaned-text duplicates: 0.",
        "",
        "| Split | Authors | Books | Chunks | Purpose |",
        "| --- | ---: | ---: | ---: | --- |",
        f"| train | {split_author_counts['train']} | {split_counts['train']} | {chunk_report['chunks_by_split']['train']:,} | Fit classifiers and derive transfer evidence; only this split may supply target examples/cards |",
        f"| dev | {split_author_counts['dev']} | {split_counts['dev']} | {chunk_report['chunks_by_split']['dev']:,} | Classifier development diagnostics and four target proxy books |",
        f"| test | {split_author_counts['test']} | {split_counts['test']} | {chunk_report['chunks_by_split']['test']:,} | Book-disjoint 50-author classifier benchmark; four target books reserved from transfer development |",
        f"| proxy_transfer | {split_author_counts['proxy_transfer']} | {split_counts['proxy_transfer']} | {chunk_report['chunks_by_split']['proxy_transfer']:,} | Additional target-author proxy reconstruction books |",
        f"| excluded | {split_author_counts['excluded']} | {split_counts['excluded']} | 0 | Too short for primary chunk benchmark |",
        "",
        "Chunks target 1,500 cleaned CJK characters with an 800-character minimum. Chunks from one "
        "book never cross train/dev/test boundaries. This prevents random chapter mixing from "
        "leaking book-specific language into evaluation.",
        "",
        "### Target-author book allocation",
        "",
        f"- **Style/card training, 29 books:** {', '.join(target_books['train'])}.",
        f"- **Development proxy, 4 books:** {', '.join(target_books['dev'])}.",
        f"- **Additional proxy-transfer, 4 books:** {', '.join(target_books['proxy_transfer'])}.",
        f"- **Transfer-held-out final-validation book reservation, 4 books:** {', '.join(target_books['test'])}. The exact 80 rows are deliberately created only after a confirmation method passes and is locked.",
        f"- **Excluded for length:** {', '.join(target_books['excluded'])}.",
        "",
        "### Content masking",
        "",
        "`entity_masked_v3` replaces names and author-concentrated content terms with length-preserving "
        "`某` spans. It retains 100% CJK length, has row parity with clean chunks, and has no malformed "
        "placeholder chunks. The masked view is used for the classifier and retrieval evidence to "
        "reduce theme, character, and book-identity learning. Masking is a control, not proof that all "
        "content signal has been removed.",
        "",
        "## Stage 1: Author-Style Meter",
        "",
        "All classifier variants fit feature transforms and model weights on the same 24,957 "
        "`train` chunks. The 23,683 `dev` chunks and 35,812 `test` chunks are book-disjoint from "
        "training. Because several classifier families were iteratively compared using their "
        "reported test results, `test` is best interpreted as a **book-disjoint benchmark**, not "
        "a pristine selection-blind estimate. The four final books remain unopened to transfer "
        "generation, threshold calibration, and transfer-method selection, but their chunks did "
        "contribute to Stage 1 classifier benchmark metrics. Final validation is therefore a "
        "transfer-held-out replication, not a fully end-to-end selection-blind test.",
        "",
        "### Methods compared",
        "",
        "| Feature family | What it measures | Best masked chunk accuracy | Research role |",
        "| --- | --- | ---: | --- |",
        "| Exact character n-grams | Auditable TF-IDF over exact Chinese 2-4 character sequences | **88.2%** unweighted; **87.8%** class-balanced | Primary style meter |",
        "| Hashed character n-grams | Approximate 2-4 character n-grams in 262,144 hash buckets | 85.0% | Faster approximation; rejected as final meter |",
        "| Combined interpretable + richer function words | Punctuation, dialogue, length, function characters, multi-character function words | 69.5% | Diagnostic guardrail |",
        "| Function words + characters | Chinese function phrases plus high-frequency function characters | 60.0% | Grammar/function-word diagnostic |",
        "| Function characters only | Single-character low-semantic inventory | 45.0% | Weak interpretable baseline |",
        "| Chinese function words only | Multi-character connective, aspect, particle, deixis, and frame terms | 39.4% | Weak interpretable baseline |",
        "| Punctuation/dialogue only | Quote, dialogue, punctuation, and related rates | 21.5% | Diagnostic only |",
        "| Sentence/paragraph length only | Sentence and paragraph shape statistics | 16.7% | Diagnostic only |",
        "",
        "### Selected meter",
        "",
        "The selected proxy is the **class-balanced SGD hinge classifier with exact character 2-4 "
        "grams on `entity_masked_v3`**, `min_df=20`, maximum 80,000 features. It was fit on train "
        "books and evaluated on book-disjoint benchmark books. The fit contains 5,650 target-author "
        "chunks and 19,307 comparison-author chunks; class balancing reduces the effect of that "
        "unequal training support.",
        "",
        "| Metric | Result |",
        "| --- | ---: |",
        "| 50-author test chunk accuracy | 87.8% |",
        "| Balanced accuracy | 83.6% |",
        "| Macro F1 | 82.3% |",
        "| Book-majority accuracy | 90.6% |",
        "| Target-author F1 | 85.9% |",
        "| Target-author recall | 100.0% |",
        "| Majority-class baseline | 6.0% |",
        "",
        "This is a **proxy meter**, not a human style judgment and not a probability model. The raw "
        "hinge margin is reported as a decision margin. Interpretable features remain secondary "
        "guardrails because they are easier to understand but materially less accurate.",
        "",
        "## Stage 2: Iteration-1 Proxy Benchmark Data",
        "",
        "### Why the source passes through English",
        "",
        "The actual Eternal Gate input is English, so the benchmark reproduces that information "
        "bottleneck instead of merely neutralizing Chinese directly:",
        "",
        "```text",
        "held-out original Chinese (hidden from generator)",
        "  -> LLM-generated plain-English benchmark source",
        "  -> independent semantic QA and selective repair",
        "  -> production-compatible neutral Chinese",
        "  -> style-transfer candidate",
        "  -> evaluation against English, neutral, and hidden original",
        "```",
        "",
        "The benchmark English is **synthetic English reconstructed from held-out Chinese**, not "
        "the published Eternal Gate translation. This design gives every proxy row a recoverable "
        "English semantic source while matching the production direction. It does not test quirks "
        "unique to Eternal Gate's actual English translator, so a later external validation sample "
        "must use real Eternal Gate English before production adoption.",
        "",
        "The neutral prompt is version `neutral_translation.v1` and is pinned as the core "
        "`eternal_gate_pass_1` contract. It preserves paragraph IDs and details, uses natural neutral "
        "Simplified Chinese, and forbids author imitation, style cards, examples, and embellishment. "
        "Production may add glossary context, but changing these core neutrality rules requires a "
        "new benchmark version.",
        "",
        "### Frozen 220-row development allocation",
        "",
        "| Arm | Books/authors | Rows | Purpose |",
        "| --- | --- | ---: | --- |",
        "| Own-author reconstruction | 8 target books x 20 chunks | 160 | Can the pipeline recover author signal lost through English and neutral Chinese? |",
        "| Cross-author transfer | 12 other authors x 1 book x 5 chunks | 60 | Can target style transfer to different content instead of reconstructing residual target content? |",
        "",
        "The 160 own-author rows are divided into 32 calibration rows and 128 method-evaluation "
        "rows. The first registered screen uses 24 method rows: three from each target book. The "
        "cross-author screen uses one row from each of the 12 authors below.",
        "",
        "| Own-author screening books (3 rows each) | Cross-author screening books (1 row each) |",
        "| --- | --- |",
        "| 骑士之歌, 天宝伏妖录, 万物风华录, 清平梦华录, 相见欢, 山有木兮, 乱世为王, 图灵密码 | priest/杀破狼; 木苏里/全球高考; 巫哲/轻狂; 唐酒卿/南禅; 墨香铜臭/魔道祖师; 淮上/破云; 梦溪石/无双; 西子绪/我五行缺你; 酱子贝/放学等我; 稚楚/营业悖论[娱乐圈]; 莫晨欢/第四视角; 漫漫何其多/当年万里觅封侯 |",
        "",
        "The frozen sample selector balances book position, six heuristic scene types, length, "
        "punctuation, sentence flow, and dialogue density, with a minimum within-book chunk-index "
        "distance of three. The generator sees opaque sample IDs, English, and neutral Chinese. It "
        "does not see book title, author, sampling stratum, original Chinese, or evaluation role.",
        "",
        "### Frozen sample artifact chain",
        "",
        "| Artifact | Contents and access role |",
        "| --- | --- |",
        "| `sample_sets/development_proxy_v1.runner_manifest.jsonl` | Opaque sample IDs and generated-artifact paths; this is the only sample manifest available to neutral/style runners |",
        "| `sample_sets/development_proxy_v1.evaluator_allocation.jsonl` | Author, book, arm, research role, and sampling strata; evaluator-only |",
        "| `sample_sets/development_proxy_v1.hidden_targets.jsonl` | Original Chinese target text keyed by opaque ID; evaluator-only |",
        "| `sample_sets/development_proxy_v1.screening_v1_ids.json` | Exact 36-row iteration-1 screening cohort |",
        "| `sample_sets/development_proxy_v1.confirmation_v1_ids.json` | Exact 152-row iteration-1 confirmation cohort; historical and not the current 116-row iteration-2 allocation |",
        "| `sample_sets/development_proxy_v1.summary.json` | Counts, book/author roster, canonical row hashes, and leakage-control declarations |",
        "| `protocols/pre_style_analysis_lock.v1.fee40c3846764c1264b23f960909345956567257370cf1cecfec355e07721ce4.json` | Raw file-byte SHA-256 bindings for the active frozen analysis |",
        "",
        "The summary's `runner_manifest_sha256`, `evaluator_allocation_sha256`, and "
        "`hidden_targets_sha256` are hashes of normalized, key-sorted JSON rows. The active analysis "
        "lock stores raw file-byte hashes, so those values are expected to differ. Running "
        "`style_transfer_research.py validate` recomputes both the row integrity and leakage checks; "
        "the current 220-row sample set passes with zero target-metadata, hidden-target, reserved-book, "
        "residue, or spacing leakage.",
        "",
        "### English-source QA and repair",
        "",
        "English source artifacts exist for all 220 rows, but the calibrated screening cohort uses "
        "68 rows: 32 calibration plus 36 screening. Every selected English source is checked by a "
        "separate stateless Chinese-to-English semantic QA request. A repair round regenerates only "
        "the still-failing English sources from the prior QA findings, then sends them through a new "
        "QA request. It does not change Chinese style and does not repeatedly edit already approved rows.",
        "",
        "| QA point | Rows entering repair | Rows still unresolved after QA |",
        "| --- | ---: | ---: |",
        "| Initial source QA | 68 | 41 |",
        "| Repair round 1 | 41 | 19 |",
        "| Repair round 2 | 19 | 13 |",
        "| Repair round 3 | 13 | 8 |",
        "| Repair round 4 | 8 | 4 |",
        "| Repair round 5 | 4 | 3 |",
        "| Repair round 6 | 3 | **0** |",
        "",
        "After source approval, neutral generation completed 68/68 rows. Calibration consumes 32; "
        "the style-transfer screen consumes the remaining 36. The initial QA and six repair/QA "
        "rounds are recorded under `runs/development_proxy_v1/proxy_v1_gpt54_20260711_v2/ledgers/` "
        "as `english_source_qa.jsonl` and `english_source_repair_qa.round_01.jsonl` through "
        "`round_06.jsonl`.",
        "",
        "## Stage 3: Style-Meter Calibration",
        "",
        "Calibration compares the hidden original target passage (positive) with its neutral "
        "translation (negative) on the same 32 rows. It enumerates observed margins, keeps thresholds "
        "with at least 80% original sensitivity and at most 10% neutral false positives, maximizes "
        "balanced accuracy, and chooses the higher threshold on ties.",
        "",
        "| Calibration field | Frozen value |",
        "| --- | ---: |",
        f"| Rows | {threshold['calibration_rows']} |",
        f"| Margin threshold | `{selected_threshold['threshold']:.9f}` |",
        f"| Original sensitivity | {percentage(selected_threshold['sensitivity'])} |",
        f"| Neutral false-positive rate | {percentage(selected_threshold['false_positive_rate'])} |",
        f"| Balanced accuracy | {percentage(selected_threshold['balanced_accuracy'])} |",
        "",
        "Method-evaluation rows are forbidden from recalibrating this threshold.",
        "",
        "## Stage 4: Registered Transfer Methods",
        "",
        f"All requests used `{model_config['codex_model']}` with `{model_config['reasoning_effort']}` "
        "reasoning, one ephemeral session per sample, a macOS Seatbelt profile denying repository "
        "reads/writes, and at most two schema-generation attempts. English and neutral inputs are "
        "byte-identical across methods. Every non-control method generated 36/36 valid outputs, for "
        "396 method outputs total.",
        "",
        "### Generation provenance",
        "",
        "| Component | Frozen version | SHA-256 |",
        "| --- | --- | --- |",
        f"| English semantic source | `{prompt_versions['english_semantic_source']['version']}` | `{prompt_versions['english_semantic_source']['sha256']}` |",
        f"| English source QA | `{prompt_versions['english_source_qa']['version']}` | `{prompt_versions['english_source_qa']['sha256']}` |",
        f"| English source repair | `{prompt_versions['english_source_repair']['version']}` | `{prompt_versions['english_source_repair']['sha256']}` |",
        f"| Neutral translation | `{prompt_versions['neutral_translation']['version']}` / `{prompt_versions['neutral_translation']['production_contract']}` | `{prompt_versions['neutral_translation']['sha256']}` |",
        f"| Style transfer | `{prompt_versions['style_transfer']['version']}` | `{prompt_versions['style_transfer']['sha256']}` |",
        "",
        "The run ledger records prompt/input/output hashes, response IDs, model name, Codex CLI "
        "version, runner hash, and sandbox profile. The provider exposes no temperature and no "
        "replayable backend snapshot; therefore artifacts provide provenance and deterministic "
        "reevaluation of frozen outputs, but not guaranteed byte-identical future LLM regeneration.",
        "",
        "### Tested interventions",
        "",
        "| Method / tested intensity | What was tested | Training evidence exposed to the method |",
        "| --- | --- | --- |",
    ]
    for row in combinations:
        info = METHOD_EXPLANATIONS[row["method_id"]]
        lines.append(
            f"| `{row['method_id']}:{row['intensity']}` | {info['short']}. {info['detail']} | {info['evidence']} |"
        )

    lines.extend(
        [
            "",
            "### Exact method and payload bindings",
            "",
            "The family descriptions above are explanatory. The table below identifies the exact "
            "machine-readable config and per-intensity payload hash used to build each request.",
            "",
            "| Method / intensity | Family | Config path | Config SHA-256 | Payload SHA-256 |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in combinations:
        binding = method_bindings[row["method_id"]]
        payload_sha = binding["asset_manifest"]["method_intensity_asset_sha256"][
            row["intensity"]
        ]
        lines.append(
            f"| `{row['method_id']}:{row['intensity']}` | `{binding['family']}` | "
            f"`{binding['config_path']}` | `{binding['config_sha256']}` | `{payload_sha}` |"
        )

    lines.extend(
        [
            "",
            "All per-method payloads are contained in and bound by "
            "`method_assets/style_transfer_payloads.v1.lock.json`; the active immutable asset is "
            "`method_assets/style_transfer_payloads.v1/assets.2be41701eabfecced860e3b8eeea2d48f68cffdc4d23a70ed215d3a0fb0589a3.json`. "
            "The close-reading methods additionally bind source packet "
            "`source_packet.v1.d91119be7bdcb84e84b1c5023622e56e3a6a33570525ef675458e34b768daa59.json` "
            "and result `result.v1.d91119be7bdcb84e84b1c5023622e56e3a6a33570525ef675458e34b768daa59.json`.",
            "",
            "`self_critique_repair` was registered as a conditional refinement method but was **not "
            "tested**. The protocol permits it only after a non-empty provisional shortlist and "
            "independent critique; iteration 1 produced no shortlist.",
            "",
            "## Stage 5: Evaluation Rules",
            "",
            "For each candidate and arm, the evaluator reports:",
            "",
            "- **Target margin:** raw target-author hinge decision margin; not a probability.",
            "- **Paired margin lift:** candidate target margin minus its byte-matched neutral margin.",
            "- **Target rank:** rank of 非天夜翔 among 50 authors; lower is better.",
            "- **Target chunk share:** fraction predicted as 非天夜翔 at rank 1.",
            "- **Deterministic hard fidelity:** paragraph order, CJK-length envelope, Latin token "
            "preservation, dialogue-turn surface, quote balance, and reference-copy checks.",
            "- **Independent semantic/readability judgment:** English-grounded blind critique, required "
            "only after a method survives deterministic screening.",
            "",
            "The deterministic `surface_fidelity.v1` gate is intentionally conservative and fully "
            "reproducible. It requires exact paragraph IDs/order and nonempty output; exact multisets "
            "of numbers, placeholders, and Latin tokens; a CJK-length ratio of 0.35-2.50 per paragraph "
            "for source paragraphs of at least 20 CJK characters; a total CJK ratio of 0.55-1.75; "
            "preserved dialogue-start status; balanced Chinese quotation marks; and no copied "
            "reference sequence of eight or more CJK characters or reference-name leakage. These "
            "checks do **not** prove semantic equivalence.",
            "",
            "| Fidelity layer | Iteration-1 status | Failure examples |",
            "| --- | --- | --- |",
            "| Deterministic surface gate | executed for all 432 control/method rows | Missing/duplicate output, paragraph mismatch, number/placeholder/Latin mismatch, length envelope, dialogue-start mismatch, unbalanced quotes |",
            "| Deterministic reference-copy gate | executed for all methods with registered evidence | Introduced eight-CJK reference sequence or reference name/place/lore leakage |",
            "| Blind semantic judgment | not executed because no method survived screening | Added/removed event, role/speaker, negation, causality, chronology, modality, or unsupported detail |",
            "| Blind readability judgment | not executed because no method survived screening | High-severity unnaturalness or readability regression against neutral |",
            "",
            "The protocol's top-level `hard_fidelity_failures` list is the union used by the complete "
            "multi-stage study. It should not be read as saying that semantic/readability judgments "
            "were executed during iteration-1 deterministic screening.",
            "",
            "A chunk counts as deterministic style success only if all four conditions hold:",
            "",
            "```text",
            "target margin >= 0.218174636",
            "target rank <= 5",
            "paired lift > 0",
            "no hard-fidelity failure",
            "```",
            "",
            "Screening promotion requires positive mean lift in both arms, deterministic hard-fidelity "
            "failure rate no greater than 10% in either arm, no reference-copy failure, and at most "
            "three promoted methods. The 36-row screen is descriptive; it cannot itself satisfy the "
            "final 80% endpoint.",
            "",
            "### How to read the metrics",
            "",
            "- **Rank 1 is weaker than calibrated success.** `Target share` only asks whether the "
            "target author has the largest of 50 classifier scores. All 50 scores can still be "
            "low, so a rank-1 prediction can have a negative margin and fail the `+0.218` gate.",
            "- **Lift is relative; margin is absolute.** Positive lift means a rewrite moved in the "
            "desired direction relative to its own neutral control. It is insufficient when the "
            "resulting margin remains below the generated-domain threshold.",
            "- **Own- and cross-author arms answer different questions.** Own-author reconstruction "
            "can retain author signal through content or translation; cross-author transfer is the "
            "stronger test that the method can impose target style on unfamiliar content.",
            "- **Screening results are descriptive.** With 24 and 12 rows per arm, iteration 1 is a "
            "method-family filter, not an 80% success claim. Registered Wilson and cluster-bootstrap "
            "inference begins only on the unopened 104/48 confirmation rows.",
            "",
            "### Registered confirmation and final endpoints",
            "",
            "| Stage / arm | Required evidence for success |",
            "| --- | --- |",
            "| Confirmation, own-author (104 rows) | Style-success point estimate >=80%; >=70% in every one of the eight development books; Wilson 95% lower bound >=70%; 10,000-resample book-cluster bootstrap lower bound >=70%; paired book-cluster bootstrap 95% CI for margin lift excludes zero |",
            "| Confirmation, cross-author (48 rows) | Reported separately; style-success point estimate >=80%; Wilson 95% lower bound >=70%; positive paired lift in at least 10 of 12 authors; author-cluster bootstrap lower bound >=70% |",
            "| Semantic noninferiority | No increase in high-severity candidate failures versus neutral; paired discordant counts reported; cluster-bootstrap 95% upper bound for candidate-minus-neutral failure rate <=0 |",
            "| Readability noninferiority | No high-severity readability regression versus neutral under blinded independent judgment |",
            "| Final validation (80 rows) | Same frozen prompt/model/method/threshold; one-time own-author point estimate >=80%; book-cluster bootstrap lower bound >=70% |",
            "",
            "The registered final validation is still a **proxy reconstruction study** using "
            "synthetic English. A separate, preregistered pilot on actual Eternal Gate English is "
            "required before production adoption; it is not part of evaluation protocol v1 and must "
            "not be implied by a future proxy pass.",
            "",
            "## Stage 6: Results by Method",
            "",
            f"![Paired margin lift]({chart_prefix}/iteration1_paired_lift.svg)",
            "",
            f"![Fidelity failure rate]({chart_prefix}/iteration1_fidelity_failures.svg)",
            "",
            f"![Absolute target margin]({chart_prefix}/iteration1_target_margin.svg)",
            "",
            "### Own-author reconstruction (24 rows)",
            "",
            "| Method | Mean target margin | Mean lift | Positive-lift rows | Mean target rank | Target share | Fidelity failures | Style success |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in combinations:
        own = row["arm_summaries"]["own_author_reconstruction"]
        lines.append(
            f"| `{row['method_id']}:{row['intensity']}` | {number(own['mean_target_margin'])} | "
            f"{number(own['mean_paired_margin_lift'])} | {own['positive_lift_rows']}/24 | "
            f"{own['mean_target_rank']:.2f} | {percentage(own['target_chunk_share_scored'])} | "
            f"{own['hard_fidelity_failure_rows']}/24 ({percentage(own['hard_fidelity_failure_rows'] / 24)}) | "
            f"{own['deterministic_style_success']['successes']}/24 |"
        )
    lines.extend(
        [
            "",
            "### Cross-author transfer (12 rows)",
            "",
            "| Method | Mean target margin | Mean lift | Positive-lift rows | Mean target rank | Target share | Fidelity failures | Style success |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in combinations:
        cross = row["arm_summaries"]["cross_author_transfer"]
        lines.append(
            f"| `{row['method_id']}:{row['intensity']}` | {number(cross['mean_target_margin'])} | "
            f"{number(cross['mean_paired_margin_lift'])} | {cross['positive_lift_rows']}/12 | "
            f"{cross['mean_target_rank']:.2f} | {percentage(cross['target_chunk_share_scored'])} | "
            f"{cross['hard_fidelity_failure_rows']}/12 ({percentage(cross['hard_fidelity_failure_rows'] / 12)}) | "
            f"{cross['deterministic_style_success']['successes']}/12 |"
        )

    lines.extend(
        [
            "",
            "### Screening gate matrix",
            "",
            "`PASS` below means the method met that one registered screening condition; it does "
            "not mean the method achieved chunk-level style success.",
            "",
            "| Method | Positive mean lift in both arms | Own fidelity <=10% | Cross fidelity <=10% | Copy-free | Joint promotion gate |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in combinations:
        own = row["arm_summaries"]["own_author_reconstruction"]
        cross = row["arm_summaries"]["cross_author_transfer"]
        decision = decisions.get((row["method_id"], row["intensity"]), {})
        own_failure_rate = own["hard_fidelity_failure_rows"] / own["expected_rows"]
        cross_failure_rate = cross["hard_fidelity_failure_rows"] / cross["expected_rows"]
        lift_pass = (
            own["mean_paired_margin_lift"] > 0
            and cross["mean_paired_margin_lift"] > 0
        )
        copy_pass = (
            decision.get("own_author_reconstruction", {}).get("no_copy_failures", 0) == 0
            and decision.get("cross_author_transfer", {}).get("no_copy_failures", 0) == 0
        )
        lines.append(
            f"| `{row['method_id']}:{row['intensity']}` | "
            f"{'PASS' if lift_pass else 'FAIL'} | "
            f"{'PASS' if own_failure_rate <= 0.10 else 'FAIL'} | "
            f"{'PASS' if cross_failure_rate <= 0.10 else 'FAIL'} | "
            f"{'PASS' if copy_pass else 'FAIL'} | "
            f"{'PASS' if decision.get('eligible') else 'FAIL'} |"
        )

    lines.extend(
        [
            "",
            "### Method-by-method interpretation",
            "",
        ]
    )
    for row in combinations:
        own = row["arm_summaries"]["own_author_reconstruction"]
        cross = row["arm_summaries"]["cross_author_transfer"]
        decision = decisions.get((row["method_id"], row["intensity"]), {})
        reasons = [
            EXCLUSION_LABELS.get(reason, reason)
            for reason in decision.get("exclusion_reasons", [])
        ]
        reason_text = "; ".join(reasons) if reasons else "not a promotion candidate"
        lines.extend(
            [
                f"#### `{row['method_id']}:{row['intensity']}`",
                "",
                METHOD_EXPLANATIONS[row["method_id"]]["detail"],
                "",
                f"- Own-author: lift {number(own['mean_paired_margin_lift'])}, mean margin "
                f"{number(own['mean_target_margin'])}, target share "
                f"{percentage(own['target_chunk_share_scored'])}, fidelity failures "
                f"{own['hard_fidelity_failure_rows']}/24.",
                f"- Cross-author: lift {number(cross['mean_paired_margin_lift'])}, mean margin "
                f"{number(cross['mean_target_margin'])}, target share "
                f"{percentage(cross['target_chunk_share_scored'])}, fidelity failures "
                f"{cross['hard_fidelity_failure_rows']}/12.",
                f"- Decision: **not promoted**; {reason_text}. Deterministic style success was "
                f"0/{own['expected_rows']} own-author and 0/{cross['expected_rows']} cross-author.",
                "",
            ]
        )

    lines.extend(
        [
            "## Why Every Method Failed",
            "",
            "1. **The absolute margin gap remained large.** Neutral mean margins were `-0.416` "
            "own-author and `-1.238` cross-author, while success required `+0.218`. The best "
            "method means reached only `-0.340` own-author and `-1.167` cross-author.",
            "2. **Positive relative movement was too small.** Six methods had positive mean lift in "
            "both arms, but the maximum mean lifts were only `+0.076` own-author and `+0.070` "
            "cross-author.",
            "3. **Fidelity failures increased with many structured interventions.** The dominant "
            "deterministic flags were dialogue-turn surface mismatches, changed/missing Latin "
            "tokens, and unbalanced dialogue quotes. These are surface hard gates; semantic "
            "equivalence was not yet judged.",
            "4. **Raw examples did not teach a transformation.** Retrieved-example and contrastive "
            "methods supplied target prose, but not aligned neutral-to-author rewrites. Both moved "
            "mean style margin backward in both arms.",
            "5. **Stronger generic prompting was not better.** The strong generic prompt changed "
            "own-author lift from `+0.066` to `-0.004` and raised own fidelity failures from "
            "12.5% to 29.2%.",
            "",
            "An oracle diagnostic selected, for each row, the highest-margin candidate among all 11 "
            "methods. Even this post-hoc upper-bound exercise produced 0/24 own-author and 0/12 "
            "cross-author threshold successes. Therefore simple reranking of the existing outputs "
            "cannot rescue iteration 1.",
            "",
            "## Independent Outcome Evaluation",
            "",
            "A separate evaluator reviewed the frozen summary, protocol, calibration, and promotion "
            "artifacts. Its verdict was **NO-GO for confirmation**.",
            "",
            "| Registered condition | Non-control methods passing |",
            "| --- | ---: |",
            "| Positive mean lift in both arms | 6/11 |",
            "| Own-author fidelity failure rate <=10% | 0/11 |",
            "| Cross-author fidelity failure rate <=10% | 3/11 |",
            "| Zero detected copying | 11/11 |",
            "| All conditions jointly | **0/11** |",
            "",
            "The closest gate miss was `generic_author_style_light`: lift `+0.0657` own and "
            "`+0.0128` cross, with 3/24 own and 1/12 cross fidelity failures. The 10% rule allows "
            "at most 2/24 and 1/12; advancing it because it missed by one row would be a post-hoc "
            "relaxation. The complete independent audit is in "
            "`audits/screening_iteration1_outcome_audit.md`.",
            "",
            "Because the shortlist is empty, independent per-output semantic/readability critiques, "
            "lighter-intensity refinement, self-critique repair, confirmation, and final validation "
            "were correctly not run. This means iteration 1 establishes deterministic screening "
            "failure, not a human judgment that every output is semantically bad or stylistically "
            "worthless.",
            "",
            "## Threats to Validity",
            "",
            "1. **Classifier selection exposure.** Multiple classifier families were compared using "
            "the book-disjoint `test` report. Those metrics are valid evidence of out-of-book "
            "performance, but they are not a selection-blind estimate after iteration.",
            "2. **Final books are only transfer-held-out.** Their future 80 transfer rows have not "
            "been sampled or generated, but the four source books contributed chunks to the "
            "Stage 1 classifier benchmark. A stronger academic replication should add target "
            "books never used in scorer development or evaluation.",
            "3. **Synthetic English domain.** Proxy English is reconstructed from Chinese and QA'd. "
            "It approximates the production direction but cannot reproduce translator-specific "
            "omissions, errors, or phrasing in the actual Eternal Gate English source.",
            "4. **Small screening sample.** The 24/12 screening arms support method rejection and "
            "direction-of-effect diagnostics, not precise success-rate estimation.",
            "5. **Masking is imperfect content control.** `entity_masked_v3` removes concentrated "
            "names and topic terms but may leave content cues or introduce regular mask artifacts.",
            "6. **Single model and one sample per cell.** Iteration 1 measures one frozen `gpt-5.4` "
            "output per method/sample. It does not estimate generation variance or portability "
            "to another model snapshot.",
            "7. **No human-equivalent outcome judgment was reached.** The deterministic screen "
            "blocked all methods before blind semantic/readability judging. Surface fidelity "
            "failures are not synonymous with semantic failure, and zero recorded judge failures "
            "would not be evidence of equivalence.",
            "",
            "## Independent Report Methodology Review",
            "",
            "A separate Codex reviewer (`019f57d9-2134-7613-96c8-f8ffe8af10d8`) audited "
            "researcher-facing data lineage, exact evidence exposure, metric interpretation, leakage "
            "controls, and reproducibility. Its initial review requested clearer final-row "
            "materialization status, direct sample and method-asset bindings, QA ledger citations, "
            "and separation of deterministic versus judged fidelity layers.",
            "",
            "After those revisions, the same reviewer reported **no remaining blocking findings** "
            "and rated this document **ADEQUATE for an iteration-1 academic screening report**. "
            "The review, finding dispositions, hash-scheme verification, and re-review verdict are "
            "recorded in `audits/style_transfer_report_methodology_review.md`.",
            "",
            "## Iteration 2 Design (Historical Preregistration)",
            "",
            "The following design was frozen after iteration 1 and has now been executed. See the "
            "[iteration-2 aligned-pairs report](04_transfer_iteration2_aligned_pairs.md) for "
            "current results. The main missing method family was **pseudo-parallel target-author "
            "transformation evidence**:",
            "",
            "```text",
            "target-author train passage",
            "  -> English semantic source",
            "  -> same production neutral Chinese",
            "  -> aligned pair: neutral Chinese -> original target-author Chinese",
            "  -> retrieve aligned pairs for a new neutral input",
            "  -> generate and fidelity-filter candidate rewrites",
            "```",
            "",
            "This differs materially from iteration-1 retrieval: the model sees how neutral prose was "
            "transformed into target prose, not only unrelated target examples. The design is "
            "supported by Styll/STRAP-style neutral-to-target pseudo-parallel training and by "
            "Prompt-and-Rerank's explicit separation of style strength, meaning preservation, and "
            "fluency.",
            "",
            "Preregistered ablations and diagnostics were:",
            "",
            "1. aligned pseudo-parallel demonstrations only;",
            "2. aligned demonstrations plus compact corpus cards;",
            "3. aligned demonstrations with multiple candidate generation and registered reranking;",
            "4. if prompting remains far below threshold, a separately registered learned-adapter or "
            "policy-optimization study rather than further prompt wording changes.",
            "",
            "Iteration 2 used a fresh 36-row screen and retained 116 confirmation rows. Those "
            "confirmation rows and the four final target books remain unopened to transfer "
            "generation and selection. No method may "
            "enter production until a new method passes screening, independent semantic/readability "
            "judgment, confirmation with at least 80% success, and one-time final validation.",
            "",
            "## Reproduction",
            "",
            "Generate this report and its charts:",
            "",
            "```bash",
            "uv run python experiments/iteration1/report_style_transfer_experiment.py",
            "```",
            "",
            "Validate sample counts, canonical row hashes, isolation, and leakage controls:",
            "",
            "```bash",
            "uv run python experiments/iteration1/style_transfer_research.py validate --sample-id development_proxy_v1",
            "```",
            "",
            "Re-run the deterministic 12-arm evaluation from frozen outputs:",
            "",
            "```bash",
            "uv run python experiments/iteration1/evaluate_style_transfer_methods.py evaluate \\",
            "  --run-id proxy_v1_gpt54_20260711_v2 \\",
            "  --selection-file generated/style_research/style_transfer_experiments/sample_sets/development_proxy_v1.screening_v1_ids.json \\",
            "  --analysis-lock generated/style_research/style_transfer_experiments/protocols/pre_style_analysis_lock.v1.fee40c3846764c1264b23f960909345956567257370cf1cecfec355e07721ce4.json \\",
            "  --scorer-mode load \\",
            "  --method neutral_only:none \\",
            "  --method generic_author_style_light:light \\",
            "  --method generic_author_style_strong:strong \\",
            "  --method global_style_cards:medium \\",
            "  --method scene_routed_style_cards:medium \\",
            "  --method sentence_flow_cards:medium \\",
            "  --method function_word_punctuation_dialogue_cards:medium \\",
            "  --method retrieved_examples_only:medium \\",
            "  --method scene_cards_plus_examples:medium \\",
            "  --method contrastive_examples:medium \\",
            "  --method llm_close_reading_style_definition:medium \\",
            "  --method llm_close_reading_cards_statistical_gates:medium \\",
            "  --output-dir generated/style_research/style_transfer_experiments/evaluations/proxy_v1_gpt54_20260711_v2/screening_initial",
            "```",
            "",
            "## Primary Artifacts",
            "",
            "- Corpus split: `generated/style_research/corpus/splits.json`",
            "- Chunk report: `generated/style_research/corpus/chunk_report.json`",
            "- Classifier report: `docs/reports/02_authorship_style_meter.md`",
            "- Proxy sample summary: `generated/style_research/style_transfer_experiments/sample_sets/development_proxy_v1.summary.json`",
            "- Runner-visible sample manifest: `generated/style_research/style_transfer_experiments/sample_sets/development_proxy_v1.runner_manifest.jsonl`",
            "- Evaluator-only allocation: `generated/style_research/style_transfer_experiments/sample_sets/development_proxy_v1.evaluator_allocation.jsonl`",
            "- Evaluator-only hidden targets: `generated/style_research/style_transfer_experiments/sample_sets/development_proxy_v1.hidden_targets.jsonl`",
            "- Screening IDs: `generated/style_research/style_transfer_experiments/sample_sets/development_proxy_v1.screening_v1_ids.json`",
            "- Confirmation IDs: `generated/style_research/style_transfer_experiments/sample_sets/development_proxy_v1.confirmation_v1_ids.json`",
            "- Evaluation protocol: `generated/style_research/style_transfer_experiments/protocols/evaluation_protocol.v1.json`",
            "- Active iteration-1 analysis lock: `generated/style_research/style_transfer_experiments/protocols/pre_style_analysis_lock.v1.fee40c3846764c1264b23f960909345956567257370cf1cecfec355e07721ce4.json`",
            "- Model and prompt provenance: `generated/style_research/style_transfer_experiments/protocols/model_run_config.v1.json`",
            "- Method registry and config bindings: `generated/style_research/style_transfer_experiments/method_registry/style_methods.v1.json` and `method_registry/methods/`",
            "- Active method payload lock and asset: `generated/style_research/style_transfer_experiments/method_assets/style_transfer_payloads.v1.lock.json` and `method_assets/style_transfer_payloads.v1/assets.2be41701eabfecced860e3b8eeea2d48f68cffdc4d23a70ed215d3a0fb0589a3.json`",
            "- Close-reading source/result: `generated/style_research/style_transfer_experiments/method_assets/close_reading_source_packets/source_packet.v1.d91119be7bdcb84e84b1c5023622e56e3a6a33570525ef675458e34b768daa59.json` and `method_assets/close_reading_results/result.v1.d91119be7bdcb84e84b1c5023622e56e3a6a33570525ef675458e34b768daa59.json`",
            "- English QA/repair ledgers: `generated/style_research/style_transfer_experiments/runs/development_proxy_v1/proxy_v1_gpt54_20260711_v2/ledgers/english_source_qa.jsonl` and `english_source_repair_qa.round_01.jsonl` through `round_06.jsonl`",
            "- Calibration: `generated/style_research/style_transfer_experiments/calibration/style_meter_threshold.v1.json`",
            "- Screening summary: `generated/style_research/style_transfer_experiments/evaluations/proxy_v1_gpt54_20260711_v2/screening_initial/evaluation_summary.json`",
            "- Per-method CSV/JSON/graphs: `generated/style_research/style_transfer_experiments/evaluations/proxy_v1_gpt54_20260711_v2/screening_initial/methods/`",
            "- Frozen shortlist decision: `generated/style_research/style_transfer_experiments/promotions/screening_v1.provisional_shortlist.v1.json`",
            "- Independent outcome audit: `generated/style_research/style_transfer_experiments/audits/screening_iteration1_outcome_audit.md`",
            "- Independent report-methodology review: `generated/style_research/style_transfer_experiments/audits/style_transfer_report_methodology_review.md`",
            "",
            "## Method Sources",
            "",
            "- Krishna, Wieting, and Iyyer (2020), STRAP / style transfer as paraphrase generation: https://aclanthology.org/2020.emnlp-main.55/",
            "- Patel, Andrews, and Callison-Burch (2024 revision), Styll low-resource authorship transfer with neutral-to-target in-context pairs: https://arxiv.org/abs/2212.08986",
            "- Suzgun, Melas-Kyriazi, and Jurafsky (2022), Prompt-and-Rerank: https://aclanthology.org/2022.emnlp-main.141/",
            "- Zhu et al. (2023), StoryTrans for Chinese and English long-story author transfer: https://aclanthology.org/2023.acl-long.827/",
            "- Tao et al. (2024/2025), CAT-LLM Chinese style definitions: https://arxiv.org/abs/2401.05707",
            "- Horvitz et al. (2024), TinyStyler authorship embeddings and reranking: https://aclanthology.org/2024.findings-emnlp.781/",
            "- Liu and May (2025), multi-iteration preference optimization with pseudo-parallel data: https://aclanthology.org/2025.naacl-long.135/",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    build_report(
        experiment_root=absolute(args.experiment_root),
        evaluation_path=absolute(args.evaluation),
        output_path=absolute(args.output),
        artifact_dir=absolute(args.artifact_dir),
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "report": str(absolute(args.output).relative_to(PROJECT_ROOT)),
                "artifact_dir": str(absolute(args.artifact_dir).relative_to(PROJECT_ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
