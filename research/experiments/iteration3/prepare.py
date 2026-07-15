#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


from experiments.shared.paths import RESEARCH_ROOT
from workflows.author_style_meter_contract import CURRENT_SCORER_ID


REPO_ROOT = RESEARCH_ROOT

import experiments.iteration1.style_transfer_research as research


BASE_ROOT = REPO_ROOT / "generated/style_research/style_transfer_experiments"
ITERATION2_ROOT = BASE_ROOT / "iterations/aligned_pairs_v1"
ITERATION_ROOT = BASE_ROOT / "iterations/constrained_rerank_v1"
SAMPLE_SET = "iteration3_proxy_v1"
RUN_ID = "iteration3_gpt55_v2"
STYLE_RUN_ID = "iteration3_gpt55_v6"
SUPERSEDED_SOURCE_RUN_ID = "iteration3_gpt55_v1"
TARGET_AUTHOR = "非天夜翔"
SEED = 20260714
MIN_PRIOR_DISTANCE = 5
CALIBRATION_PER_BOOK = 4
SCREEN_PER_BOOK = 3
SCREEN_PER_AUTHOR = 1
SAFETY_EXCLUDED_CHUNK_IDS = {
    "非天夜翔__相见欢__0286",
}
METHODS = (
    "neutral_only",
    "aligned_pairs_light",
    "aligned_pairs_edit_plan_light",
    "microcards_only_light",
    "rule_linked_microcards_light",
    "candidate_rerank",
)
GENERATED_METHODS = (
    "aligned_pairs_light",
    "aligned_pairs_edit_plan_light",
    "microcards_only_light",
    "rule_linked_microcards_light",
)
INTENSITY = "light"
PAIR_K = 4
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
SENTENCE_RE = re.compile(r"[^。！？!?]+[。！？!?]?")
CONNECTIVES = (
    "但是",
    "不过",
    "然而",
    "因此",
    "所以",
    "于是",
    "然后",
    "接着",
    "同时",
    "其实",
    "而且",
    "并且",
)
LAUGHTER_TERMS = ("笑道", "笑着", "笑了", "一笑", "微笑", "失笑", "笑问")
REACTION_TERMS = ("点头", "摇头", "一怔", "愣住", "沉默", "皱眉", "抬头", "低头", "看了一眼")
SPEECH_TAG_RE = re.compile(
    r"(?:[”」』][^。！？\n]{0,12}(?:说(?:道)?|问(?:道)?|答(?:道)?|喊(?:道)?|道)[，。！？：]"
    r"|(?:说(?:道)?|问(?:道)?|答(?:道)?|喊(?:道)?|道)：[“「『])"
)
JSON_RESIDUE_RE = re.compile(r"(?:\}\s*,\s*\{|```|\[\s*\{|\}\s*\])")
NUMBER_RE = re.compile(r"\d+(?:[.,:/-]\d+)*")
LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]*")
PLACEHOLDER_RE = re.compile(r"<(?:TERM|NUM|LATIN)>")
SAFE_MICROCARD_IDS = {
    "global.flow.short_sentence_ratio",
    "global.function_word.connective",
    "global.dialogue.simple_speech_tags",
}
MICROCARD_CONFOUND_IDS = {
    "global.function_word.connective",
    "global.dialogue.simple_speech_tags",
}


STYLE_PROMPT = """# Style-Transfer Method Prompt v2

## Task

Recast the neutral Chinese draft using the supplied frozen method payload. The
English semantic source is authoritative. Apply only local style operations that
are supported by the payload and by an opportunity in the current source.

## Non-negotiable constraints

1. Preserve every paragraph ID and its order.
2. Preserve facts, event order, causality, negation, modality, intensity,
   entities, numbers, Latin tokens, and speaker attribution.
3. Preserve whether each paragraph begins with direct dialogue. Do not move a
   speech tag from outside a quotation to inside it or vice versa.
4. Do not import names, places, lore, imagery, phrases, or plot facts from any
   reference passage. Never copy eight or more consecutive Chinese characters.
5. Make a light local edit only when the source provides a clear trigger. Leave
   all other wording unchanged. It is valid to skip every cue.
6. Return valid JSON only. Paragraph text must never contain JSON delimiters,
   Markdown fences, commentary, card labels, or analysis.

## Input

```json
{
  "sample_id": "...",
  "method_id": "...",
  "intensity": "light",
  "english_semantic_source": [{"id": "p0001", "en": "..."}],
  "neutral_zh": [{"id": "p0001", "zh": "..."}],
  "method_payload": {},
  "reference_examples": []
}
```

## Output

Return JSON only:

```json
{
  "sample_id": "...",
  "method_id": "...",
  "intensity": "light",
  "paragraphs": [{"id": "p0001", "zh": "..."}],
  "style_cues_applied": [],
  "style_cues_skipped": [],
  "uncertainties": []
}
```
"""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows_sha256(rows: Sequence[dict[str, Any]]) -> str:
    return sha256_text("".join(canonical_json(row) + "\n" for row in rows))


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if not path.exists() or path.read_text(encoding="utf-8") != content:
        path.write_text(content, encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    if not path.exists() or path.read_text(encoding="utf-8") != content:
        path.write_text(content, encoding="utf-8")


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or destination.read_bytes() != source.read_bytes():
        shutil.copy2(source, destination)


def artifact_paths(sample_id: str) -> dict[str, str]:
    root = ITERATION_ROOT / "runs" / SAMPLE_SET / "<run_id>"
    return {
        "english_semantic_source": relative(root / "english_semantic_source" / f"{sample_id}.json"),
        "english_source_qa": relative(root / "english_source_qa" / f"{sample_id}.json"),
        "neutral_translation": relative(root / "neutral_translation" / f"{sample_id}.json"),
        "method_outputs": relative(root / "method_outputs/<method_id>/<intensity>" / f"{sample_id}.json"),
        "evaluation": relative(root / "evaluation/<method_id>/<intensity>" / f"{sample_id}.json"),
    }


def runner_row(sample_id: str) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "sample_set": SAMPLE_SET,
        "stage_status": {
            "english_semantic_source": "pending",
            "english_source_qa": "pending",
            "neutral_translation": "pending",
            "style_transfer": "pending",
            "automatic_evaluation": "pending",
            "independent_evaluation": "pending",
        },
        "artifact_paths": artifact_paths(sample_id),
    }


def prior_rows() -> list[dict[str, Any]]:
    return list(
        iter_jsonl(
            ITERATION2_ROOT / "sample_sets/development_proxy_v1.evaluator_allocation.jsonl"
        )
    )


def filter_prior_neighbours(
    rows: Sequence[dict[str, Any]], prior: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    used: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in prior:
        used[(str(row["author"]), str(row["book_title"]))].append(
            int(row["chunk_index"])
        )
    return [
        row
        for row in rows
        if all(
            abs(int(row["chunk_index"]) - index) >= MIN_PRIOR_DISTANCE
            for index in used[(str(row["author"]), str(row["book_title"]))]
        )
    ]


def replace_safety_exclusions(
    selected: Sequence[dict[str, Any]],
    eligible: Sequence[dict[str, Any]],
    *,
    selection_key: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Replace blocked rows one-for-one without resampling the retained cohort."""
    result = list(selected)
    replacements: dict[str, str] = {}
    for index, row in enumerate(result):
        excluded_id = str(row["chunk_id"])
        if excluded_id not in SAFETY_EXCLUDED_CHUNK_IDS:
            continue
        retained = [candidate for offset, candidate in enumerate(result) if offset != index]
        unavailable = {str(candidate["chunk_id"]) for candidate in result}
        candidates = [
            candidate
            for candidate in eligible
            if str(candidate["chunk_id"]) not in unavailable
            and str(candidate["chunk_id"]) not in SAFETY_EXCLUDED_CHUNK_IDS
            and all(
                abs(int(candidate["chunk_index"]) - int(kept["chunk_index"]))
                >= MIN_PRIOR_DISTANCE
                for kept in retained
            )
        ]
        replacement = research.select_diverse_rows(
            candidates,
            1,
            research.stable_seed(
                SEED, f"iteration3:safety-replacement:{selection_key}:{excluded_id}"
            ),
            min_chunk_distance=MIN_PRIOR_DISTANCE,
        )
        if len(replacement) != 1:
            raise ValueError(f"No deterministic safety replacement for {excluded_id}")
        result[index] = replacement[0]
        replacements[excluded_id] = str(replacement[0]["chunk_id"])
    return result, replacements


def make_fresh_rows() -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], dict[str, str]
]:
    prior = prior_rows()
    own_by_book = research.load_development_candidates(
        REPO_ROOT / "datasets/unmasked/chunks.clean.jsonl", TARGET_AUTHOR
    )
    research.enrich_strata(own_by_book)
    cross_by_author = research.load_comparison_candidates(
        REPO_ROOT / "datasets/unmasked/chunks.clean.jsonl", TARGET_AUTHOR
    )
    research.enrich_strata(cross_by_author)

    calibration: list[dict[str, Any]] = []
    screening: list[dict[str, Any]] = []
    safety_replacements: dict[str, str] = {}
    for split_name, titles in research.DEVELOPMENT_BOOKS.items():
        for title in titles:
            eligible = filter_prior_neighbours(own_by_book[title], prior)
            selected = research.select_diverse_rows(
                eligible,
                CALIBRATION_PER_BOOK + SCREEN_PER_BOOK,
                research.stable_seed(SEED, f"iteration3:{split_name}:{title}"),
                min_chunk_distance=MIN_PRIOR_DISTANCE,
            )
            original_calibration_ids = {
                row["chunk_id"]
                for row in research.select_diverse_rows(
                    selected,
                    CALIBRATION_PER_BOOK,
                    research.stable_seed(SEED, f"iteration3:calibration:{title}"),
                    min_chunk_distance=0,
                )
            }
            selected, replacements = replace_safety_exclusions(
                selected,
                eligible,
                selection_key=f"{split_name}:{title}",
            )
            safety_replacements.update(replacements)
            calibration_ids = {
                replacements.get(str(chunk_id), str(chunk_id))
                for chunk_id in original_calibration_ids
            }
            for row in selected:
                row = dict(row)
                row["benchmark_arm"] = "own_author_reconstruction"
                if row["chunk_id"] in calibration_ids:
                    row["research_role"] = "style_meter_calibration"
                    calibration.append(row)
                else:
                    row["research_role"] = "method_evaluation"
                    screening.append(row)

    for author in research.COMPARISON_AUTHORS:
        eligible = filter_prior_neighbours(cross_by_author[author], prior)
        selected = research.select_diverse_rows(
            eligible,
            SCREEN_PER_AUTHOR,
            research.stable_seed(SEED, f"iteration3:cross:{author}"),
            min_chunk_distance=MIN_PRIOR_DISTANCE,
        )
        for row in selected:
            row = dict(row)
            row["research_role"] = "cross_author_method_evaluation"
            row["benchmark_arm"] = "cross_author_transfer"
            screening.append(row)
    if (len(calibration), len(screening)) != (32, 36):
        raise ValueError("Fresh iteration-3 calibration/screen counts are incorrect")
    return calibration, screening, safety_replacements


def materialize_fresh_rows(
    rows: Sequence[dict[str, Any]], masked_by_id: Mapping[str, str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    runner: list[dict[str, Any]] = []
    allocation: list[dict[str, Any]] = []
    hidden: list[dict[str, Any]] = []
    for row in rows:
        sample_id = research.opaque_sample_id(SEED, SAMPLE_SET, str(row["chunk_id"]))
        runner.append(runner_row(sample_id))
        allocation.append(
            {
                "sample_id": sample_id,
                "sample_set": SAMPLE_SET,
                "benchmark_arm": row["benchmark_arm"],
                "author": row["author"],
                "book_title": row["book_title"],
                "source_split": row["source_split"],
                "research_role": row["research_role"],
                "chunk_id": row["chunk_id"],
                "chunk_index": row["chunk_index"],
                "quality_flags": row["quality_flags"],
                "selected_residue_hits": research.residue_hits(row["original_zh"]),
                "strata": {
                    field: row[field] for field in research.STRATIFICATION_FIELDS
                },
                "measurements": {
                    key: round(row[key], 6) if isinstance(row[key], float) else row[key]
                    for key in (
                        "cjk_count",
                        "paragraph_count",
                        "sentence_count",
                        "mean_paragraph_cjk",
                        "mean_sentence_cjk",
                        "punctuation_density",
                        "dialogue_density",
                    )
                },
                "access_policy": "frozen_output_evaluator_only",
            }
        )
        hidden.append(
            {
                "sample_id": sample_id,
                "sample_set": SAMPLE_SET,
                "original_zh": row["original_zh"],
                "entity_masked_v3_zh": masked_by_id[row["chunk_id"]],
                "access_policy": "one_way_semantic_source_generator_and_frozen_output_evaluator_only",
            }
        )
    return runner, allocation, hidden


def copied_confirmation() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    sample_root = ITERATION2_ROOT / "sample_sets"
    ids = set(
        read_json(sample_root / "development_proxy_v1.confirmation_v1_ids.json")[
            "sample_ids"
        ]
    )
    allocation_by_id = {
        row["sample_id"]: row
        for row in iter_jsonl(sample_root / "development_proxy_v1.evaluator_allocation.jsonl")
    }
    hidden_by_id = {
        row["sample_id"]: row
        for row in iter_jsonl(sample_root / "development_proxy_v1.hidden_targets.jsonl")
    }
    runner = [runner_row(sample_id) for sample_id in sorted(ids)]
    allocation = [
        {**allocation_by_id[sample_id], "sample_set": SAMPLE_SET}
        for sample_id in sorted(ids)
    ]
    hidden = [
        {**hidden_by_id[sample_id], "sample_set": SAMPLE_SET}
        for sample_id in sorted(ids)
    ]
    return runner, allocation, hidden


def build_sample_set() -> dict[str, Any]:
    calibration_rows, screening_rows, safety_replacements = make_fresh_rows()
    fresh = calibration_rows + screening_rows
    masked = research.load_masked_targets(
        REPO_ROOT / "datasets/masked/chunks.entity_masked_v3.jsonl",
        {str(row["chunk_id"]) for row in fresh},
    )
    if len(masked) != len(fresh):
        raise ValueError("Masked view is missing fresh iteration-3 chunks")
    fresh_runner, fresh_allocation, fresh_hidden = materialize_fresh_rows(
        fresh, masked
    )
    confirm_runner, confirm_allocation, confirm_hidden = copied_confirmation()
    runner = sorted(fresh_runner + confirm_runner, key=lambda row: row["sample_id"])
    allocation = sorted(
        fresh_allocation + confirm_allocation, key=lambda row: row["sample_id"]
    )
    hidden = sorted(fresh_hidden + confirm_hidden, key=lambda row: row["sample_id"])
    calibration_ids = sorted(
        row["sample_id"]
        for row in fresh_allocation
        if row["research_role"] == "style_meter_calibration"
    )
    screen_ids = sorted(
        row["sample_id"]
        for row in fresh_allocation
        if row["research_role"] in {"method_evaluation", "cross_author_method_evaluation"}
    )
    confirmation_ids = sorted(row["sample_id"] for row in confirm_allocation)
    method_ids = sorted(screen_ids + confirmation_ids)
    root = ITERATION_ROOT / "sample_sets"
    write_jsonl(root / f"{SAMPLE_SET}.runner_manifest.jsonl", runner)
    write_jsonl(root / f"{SAMPLE_SET}.evaluator_allocation.jsonl", allocation)
    write_jsonl(root / f"{SAMPLE_SET}.hidden_targets.jsonl", hidden)
    selections = {
        "method_evaluation_ids": {
            "schema_version": 1,
            "sample_set": SAMPLE_SET,
            "allowed_research_roles": ["method_evaluation", "cross_author_method_evaluation"],
            "sample_count": len(method_ids),
            "sample_ids": method_ids,
        },
        "screening_v1_ids": {
            "schema_version": 1,
            "selection_id": "screening_v1",
            "sample_set": SAMPLE_SET,
            "seed": SEED,
            "sample_count": len(screen_ids),
            "own_author_rows": 24,
            "cross_author_rows": 12,
            "design": {
                "fresh_chunks_not_in_iterations_1_or_2": True,
                "minimum_prior_chunk_distance": MIN_PRIOR_DISTANCE,
                "own_author_chunks_per_book": SCREEN_PER_BOOK,
                "cross_author_chunks_per_author": SCREEN_PER_AUTHOR,
            },
            "sample_ids": screen_ids,
        },
        "confirmation_v1_ids": {
            "schema_version": 1,
            "selection_id": "confirmation_v1",
            "sample_set": SAMPLE_SET,
            "sample_count": len(confirmation_ids),
            "own_author_rows": 80,
            "cross_author_rows": 36,
            "source": "iteration2_untouched_confirmation_v1",
            "sample_ids": confirmation_ids,
        },
        "screening_calibration_v1_ids": {
            "schema_version": 1,
            "selection_id": "screening_calibration_v1",
            "sample_set": SAMPLE_SET,
            "sample_count": len(calibration_ids) + len(screen_ids),
            "composition": {
                "style_meter_calibration": len(calibration_ids),
                "method_screening": len(screen_ids),
            },
            "sample_ids": sorted(calibration_ids + screen_ids),
        },
    }
    for suffix, payload in selections.items():
        write_json(root / f"{SAMPLE_SET}.{suffix}.json", payload)
    prior_chunk_ids = {row["chunk_id"] for row in prior_rows()}
    fresh_chunk_ids = {row["chunk_id"] for row in fresh_allocation}
    summary = {
        "schema_version": 1,
        "iteration": 3,
        "sample_set": SAMPLE_SET,
        "seed": SEED,
        "target_author": TARGET_AUTHOR,
        "total_samples": len(allocation),
        "style_meter_calibration_samples": len(calibration_ids),
        "screening_samples": len(screen_ids),
        "confirmation_samples": len(confirmation_ids),
        "screening_own": 24,
        "screening_cross": 12,
        "confirmation_own": 80,
        "confirmation_cross": 36,
        "fresh_prior_chunk_overlap": len(fresh_chunk_ids & prior_chunk_ids),
        "minimum_prior_chunk_distance": MIN_PRIOR_DISTANCE,
        "confirmation_ids_sha256_from_iteration2": file_sha256(
            ITERATION2_ROOT / "sample_sets/development_proxy_v1.confirmation_v1_ids.json"
        ),
        "runner_manifest_sha256": rows_sha256(runner),
        "evaluator_allocation_sha256": rows_sha256(allocation),
        "hidden_targets_sha256": rows_sha256(hidden),
        "method_evaluation_ids_sha256": file_sha256(
            root / f"{SAMPLE_SET}.method_evaluation_ids.json"
        ),
        "screening_ids_sha256": file_sha256(root / f"{SAMPLE_SET}.screening_v1_ids.json"),
        "confirmation_ids_sha256": file_sha256(
            root / f"{SAMPLE_SET}.confirmation_v1_ids.json"
        ),
        "screening_calibration_ids_sha256": file_sha256(
            root / f"{SAMPLE_SET}.screening_calibration_v1_ids.json"
        ),
        "counts": {
            "by_role": dict(sorted(Counter(row["research_role"] for row in allocation).items())),
            "by_arm": dict(sorted(Counter(row["benchmark_arm"] for row in allocation).items())),
            "screen_by_scene": dict(
                sorted(
                    Counter(
                        row["strata"]["scene_type"]
                        for row in fresh_allocation
                        if row["sample_id"] in set(screen_ids)
                    ).items()
                )
            ),
        },
        "leakage_controls": {
            "opaque_sample_ids": True,
            "runner_contains_target_metadata": False,
            "hidden_targets_separate": True,
            "eternal_gate_excluded": True,
            "final_target_books_excluded": True,
            "confirmation_untouched": True,
        },
        "safety_exclusions": {
            "policy": "one_for_one_same_book_replacement_before_calibration_or_style_generation",
            "excluded_chunk_ids": sorted(SAFETY_EXCLUDED_CHUNK_IDS),
            "replacements": dict(sorted(safety_replacements.items())),
        },
    }
    write_json(root / f"{SAMPLE_SET}.summary.json", summary)
    return summary


def copy_runtime_artifacts() -> None:
    for directory in ("prompts", "schemas"):
        for source in (ITERATION2_ROOT / directory).glob("*"):
            if source.is_file():
                copy_file(source, ITERATION_ROOT / directory / source.name)
    (ITERATION_ROOT / "prompts/style_transfer_method.v1.md").write_text(
        STYLE_PROMPT, encoding="utf-8"
    )
    scorer = CURRENT_SCORER_ID
    for source in (ITERATION2_ROOT / "scorers" / scorer).iterdir():
        if source.is_file():
            copy_file(source, ITERATION_ROOT / "scorers" / scorer / source.name)


def write_model_config(model: str, phase: str) -> dict[str, Any]:
    config = read_json(ITERATION2_ROOT / "protocols/model_run_config.v1.json")
    prompt_files = {
        "english_semantic_source": "english_semantic_source.v1.md",
        "english_source_qa": "english_source_qa.v1.md",
        "english_source_repair": "english_source_repair.v1.md",
        "neutral_translation": "neutral_translation.v1.md",
        "style_transfer": "style_transfer_method.v1.md",
        "style_transfer_critique": "style_transfer_critique.v1.md",
    }
    prompt_versions = {
        key: dict(value)
        for key, value in config.get("prompt_versions", {}).items()
        if isinstance(value, Mapping)
    }
    for key, filename in prompt_files.items():
        if key not in prompt_versions:
            raise ValueError(f"Model config lacks prompt version metadata: {key}")
        prompt_versions[key]["sha256"] = file_sha256(
            ITERATION_ROOT / "prompts" / filename
        )
    config.update(
        {
            "codex_model": model,
            "reasoning_effort": "high",
            "default_parallel_jobs": 4,
            "experiment_iteration": 3,
            "experiment_phase": phase,
            "prompt_versions": prompt_versions,
            "model_selection": {
                "neutral_reconstruction_model": "gpt-5.4",
                "style_transfer_model": "gpt-5.5",
                "no_model_mixing_within_style_arms": True,
            },
        }
    )
    write_json(ITERATION_ROOT / "protocols/model_run_config.v1.json", config)
    return config


def pair_assets() -> dict[str, Any]:
    lock = read_json(ITERATION2_ROOT / "method_assets/style_transfer_payloads.v1.lock.json")
    bundle = read_json(ITERATION2_ROOT / lock["asset_path"])
    return bundle["assets"]


def card_value(card_id: str, text: str) -> float:
    cjk = max(len(CJK_RE.findall(text)), 1)
    if card_id.endswith("laughter_tags"):
        return sum(text.count(term) for term in LAUGHTER_TERMS) * 1000 / cjk
    if card_id.endswith("short_sentence_ratio"):
        sentences = [value for value in SENTENCE_RE.findall(text) if CJK_RE.search(value)]
        return sum(len(CJK_RE.findall(value)) <= 10 for value in sentences) / max(len(sentences), 1)
    if card_id.endswith("exclamation"):
        return (text.count("！") + text.count("!")) * 1000 / cjk
    if card_id.endswith("connective"):
        return sum(text.count(term) for term in CONNECTIVES) * 1000 / cjk
    if card_id.endswith("simple_speech_tags"):
        return len(SPEECH_TAG_RE.findall(text)) * 1000 / cjk
    if card_id.endswith("micro_reactions"):
        return sum(text.count(term) for term in REACTION_TERMS) * 1000 / cjk
    if card_id.endswith("colon"):
        return (text.count("：") + text.count(":")) * 1000 / cjk
    if card_id.endswith("quote_marks_per_kcjk"):
        return sum(text.count(mark) for mark in "“”‘’「」『』") * 1000 / cjk
    return 0.0


def starts_dialogue(text: str) -> bool:
    return text.lstrip().startswith(("“", "‘", "「", "『", '"'))


def quotes_balanced(text: str) -> bool:
    return (
        text.count("“") == text.count("”")
        and text.count("‘") == text.count("’")
        and text.count("「") == text.count("」")
        and text.count("『") == text.count("』")
    )


def safe_microcard_example(card_id: str, before: str, after: str) -> bool:
    if starts_dialogue(before) != starts_dialogue(after):
        return False
    if not quotes_balanced(before) or not quotes_balanced(after):
        return False
    if JSON_RESIDUE_RE.search(before) or JSON_RESIDUE_RE.search(after):
        return False
    if Counter(NUMBER_RE.findall(before)) != Counter(NUMBER_RE.findall(after)):
        return False
    if Counter(LATIN_RE.findall(before)) != Counter(LATIN_RE.findall(after)):
        return False
    if Counter(PLACEHOLDER_RE.findall(before)) != Counter(PLACEHOLDER_RE.findall(after)):
        return False
    before_cjk = len(CJK_RE.findall(before))
    after_cjk = len(CJK_RE.findall(after))
    if not 0.70 <= after_cjk / max(before_cjk, 1) <= 1.30:
        return False
    if difflib.SequenceMatcher(a=before, b=after, autojunk=False).ratio() < 0.70:
        return False
    if card_id.endswith("short_sentence_ratio") and any(
        mark in before + after for mark in "“”‘’「」『』"
    ):
        return False
    if card_id.endswith("simple_speech_tags") and (
        before.count("：") + before.count(":")
        != after.count("：") + after.count(":")
    ):
        return False
    if card_id.endswith(
        ("laughter_tags", "connective", "simple_speech_tags", "micro_reactions")
    ) and card_value(card_id, before) <= 0:
        return False
    return True


def minimal_microcard_after(card_id: str, before: str, observed_target: str) -> str | None:
    if card_id.endswith("connective"):
        for term in CONNECTIVES:
            if term in before and observed_target.count(term) < before.count(term):
                return before.replace(term, "", 1)
        return None
    if card_id.endswith("simple_speech_tags"):
        mappings = (
            ("说道", ("道", "说")),
            ("问道", ("问",)),
            ("答道", ("答",)),
            ("喊道", ("喊",)),
        )
        for source, targets in mappings:
            match = re.search(re.escape(source) + r"(?=：[“「『])", before)
            if match is None:
                match = re.search(
                    r"(?<=[”」』])([^。！？\n]{0,10})" + re.escape(source) + r"(?=[，。！？：])",
                    before,
                )
            if match is None:
                continue
            replacement = next(
                (target for target in targets if target in observed_target), targets[0]
            )
            start = match.start() if match.lastindex is None else match.end(1)
            return before[:start] + replacement + before[start + len(source) :]
        return None
    return None


def paragraph_pairs(assets: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in assets["aligned_pairs"]["pairs"]:
        neutral = {row["id"]: row["zh"] for row in pair["neutral_zh"]}
        target = {row["id"]: row["zh"] for row in pair["target_style_zh"]}
        for paragraph_id in sorted(set(neutral) & set(target)):
            before = neutral[paragraph_id]
            after = target[paragraph_id]
            if before == after or not (20 <= len(CJK_RE.findall(before)) <= 180):
                continue
            rows.append(
                {
                    "pair_id": pair["pair_id"],
                    "source_book_hash": pair["source_book_hash"],
                    "paragraph_id": paragraph_id,
                    "before": before,
                    "after": after,
                }
            )
    return rows


def build_microcards(assets: Mapping[str, Any]) -> list[dict[str, Any]]:
    examples = paragraph_pairs(assets)
    result: list[dict[str, Any]] = []
    for card in assets["style_cards"]["global"]:
        card_id = str(card["card_id"])
        if card_id not in SAFE_MICROCARD_IDS:
            continue
        direction = str(card["direction"])
        ranked: list[tuple[float, str, dict[str, Any]]] = []
        for row in examples:
            if not safe_microcard_example(card_id, row["before"], row["after"]):
                continue
            if any(
                abs(
                    card_value(other_id, row["after"])
                    - card_value(other_id, row["before"])
                )
                > 1e-9
                for other_id in MICROCARD_CONFOUND_IDS
                if other_id != card_id
            ):
                continue
            before = card_value(card_id, row["before"])
            after = card_value(card_id, row["after"])
            delta = after - before
            signed = delta if direction == "higher" else -delta
            if signed <= 0:
                continue
            minimal_after = minimal_microcard_after(card_id, row["before"], row["after"])
            if minimal_after is None or minimal_after == row["before"]:
                continue
            minimal_delta = card_value(card_id, minimal_after) - card_value(
                card_id, row["before"]
            )
            minimal_signed = minimal_delta if direction == "higher" else -minimal_delta
            if minimal_signed <= 0:
                continue
            ranked.append(
                (
                    signed,
                    str(row["pair_id"]),
                    {
                        **row,
                        "after": minimal_after,
                        "feature_delta": round(minimal_delta, 6),
                        "observed_target_sha256": sha256_text(row["after"]),
                        "example_scope": "single_rule_isolated_from_observed_aligned_pair",
                    },
                )
            )
        ranked.sort(key=lambda value: (-value[0], value[1], value[2]["paragraph_id"]))
        selected: list[dict[str, Any]] = []
        books: set[str] = set()
        for _score, _pair_id, row in ranked:
            if row["source_book_hash"] in books:
                continue
            books.add(row["source_book_hash"])
            selected.append(row)
            if len(selected) == 2:
                break
        if len(selected) < 2:
            continue
        counter_candidates = [
                {
                    **row,
                    "feature_delta": round(
                        card_value(card_id, row["after"])
                        - card_value(card_id, row["before"]),
                        6,
                    ),
                }
                for row in examples
                if row["source_book_hash"] not in books
                and safe_microcard_example(card_id, row["before"], row["after"])
                and card_value(card_id, row["before"]) > 0
            ]
        if not counter_candidates:
            continue
        counter = min(
            counter_candidates,
            key=lambda row: (abs(row["feature_delta"]), row["pair_id"], row["paragraph_id"]),
        )
        result.append(
            {
                **card,
                "card_schema": "rule_linked_microcard.v1",
                "trigger": "Apply only when the current neutral paragraph contains the same source-supported opportunity.",
                "semantic_invariants": [
                    "speaker_and_attribution",
                    "facts_and_event_order",
                    "negation_modality_and_intensity",
                    "entities_numbers_and_latin_tokens",
                    "dialogue_start_surface",
                ],
                "positive_before_after_examples": selected,
                "counterexample": {
                    "pair_id": counter["pair_id"],
                    "source_book_hash": counter["source_book_hash"],
                    "paragraph_id": counter["paragraph_id"],
                    "neutral_text": counter["before"],
                    "recommended_action": "leave_unchanged",
                },
                "counterexample_instruction": "A stylistic opportunity is not mandatory; leave the text unchanged when the rule's trigger is weak.",
            }
        )
    if len(result) < 2:
        raise ValueError(f"Only {len(result)} rule-linked microcards have two-book support")
    return result


def source_bindings() -> dict[str, str]:
    paths = [
        REPO_ROOT / "experiments/iteration3/prepare.py",
        REPO_ROOT / "experiments/iteration3/style_transfer_payloads.py",
        REPO_ROOT / "experiments/iteration3/run_style_transfer_generation.py",
        REPO_ROOT / "experiments/iteration3/run_style_transfer_block_generation.py",
        REPO_ROOT / "experiments/iteration3/evaluate_style_transfer_methods.py",
        REPO_ROOT / "experiments/iteration3/build_candidate_rerank.py",
        REPO_ROOT / "experiments/iteration3/analyze_calibration_sensitivity.py",
        REPO_ROOT / "experiments/iteration3/freeze_english_source_adjudication.py",
        REPO_ROOT / "experiments/iteration3/freeze_execution_pilot.py",
        REPO_ROOT / "experiments/iteration1/run_style_transfer_generation.py",
        REPO_ROOT / "experiments/iteration1/evaluate_style_transfer_methods.py",
        REPO_ROOT / "experiments/iteration1/style_analysis_lock.py",
        REPO_ROOT / "experiments/iteration1/build_style_analysis_lock.py",
        REPO_ROOT / "experiments/iteration1/calibrate_style_meter.py",
        REPO_ROOT / "experiments/iteration1/style_experiment_decisions.py",
        REPO_ROOT / "experiments/iteration1/build_style_semantic_judgments.py",
        REPO_ROOT / "experiments/iteration1/select_style_transfer_promotions.py",
        REPO_ROOT / "datasets/unmasked/chunks.clean.jsonl",
        REPO_ROOT / "datasets/masked/chunks.entity_masked_v3.jsonl",
        REPO_ROOT / "datasets/masked/mask_terms.json",
    ]
    paths.extend(sorted((ITERATION_ROOT / "prompts").glob("*.md")))
    paths.extend(sorted((ITERATION_ROOT / "schemas").glob("*.json")))
    paths.extend(sorted((ITERATION_ROOT / "sample_sets").glob(f"{SAMPLE_SET}.*")))
    paths.extend(sorted((ITERATION_ROOT / "scorers").glob("**/*")))
    pair_lock_path = ITERATION2_ROOT / "method_assets/style_transfer_payloads.v1.lock.json"
    paths.append(pair_lock_path)
    pair_lock = read_json(pair_lock_path)
    paths.append(ITERATION2_ROOT / str(pair_lock["asset_path"]))
    iteration3_lock_path = ITERATION_ROOT / "method_assets/style_transfer_payloads.v1.lock.json"
    if iteration3_lock_path.exists():
        paths.append(iteration3_lock_path)
        iteration3_lock = read_json(iteration3_lock_path)
        paths.append(ITERATION_ROOT / str(iteration3_lock["asset_path"]))
    paths = [path for path in paths if path.is_file()]
    return {relative(path): file_sha256(path) for path in paths}


def method_assets(assets: Mapping[str, Any], microcards: list[dict[str, Any]]) -> dict[str, Any]:
    common = {
        "availability": "ready",
        "intensity_contract": {
            "label": INTENSITY,
            "rewrite_scope": "Light local edits only; unchanged paragraphs are preferred when support is weak.",
            "semantic_priority": "English is authoritative. Preserve dialogue-start surface, quote topology, numbers, and Latin tokens.",
            "paragraph_contract": "Preserve paragraph IDs and order exactly; output no JSON residue inside paragraph strings.",
        },
        "evidence_source": {
            "pair_pool": (
                f"{len(pair_rows)} entity-masked train excerpts from "
                f"{len({row['title'] for row in pair_rows})} target-author books"
            ),
            "retrieval_query": "semantic_and_structural_signature_v1",
            "k": PAIR_K,
            "book_diversity": "at_most_one_pair_per_book",
            "reference_representation": (
                "same retrieved pair ID represented by a deterministic contiguous "
                "three-paragraph before/after window"
            ),
        },
    }
    return {
        "neutral_only": {
            "intensities": {
                "none": {
                    "availability": "ready",
                    "intensity_contract": {"label": "none"},
                    "evidence_source": {"kind": "control"},
                    "instructions": ["Return the neutral draft unchanged."],
                }
            }
        },
        "aligned_pairs_light": {
            "intensities": {
                INTENSITY: {
                    **common,
                    "instructions": [
                        "Infer recurring transformations from four aligned pairs.",
                        "Apply only one clearly supported local operation per changed paragraph.",
                        "Preserve dialogue-tag placement and all Latin tokens exactly.",
                    ],
                }
            }
        },
        "aligned_pairs_edit_plan_light": {
            "intensities": {
                INTENSITY: {
                    **common,
                    "instructions": [
                        "Infer a private edit plan from repeated pair transformations, then apply only low-risk operations.",
                        "Do not output the plan. Prefer no edit over a weakly supported edit.",
                    ],
                    "edit_plan_contract": {
                        "minimum_pair_support": 2,
                        "max_operations_per_paragraph": 1,
                        "allowed_dimensions": [
                            "clause_compression",
                            "sentence_boundary",
                            "plain_speech_tag_without_relocation",
                            "function_word_reduction",
                            "reaction_beat_placement_without_new_action",
                        ],
                        "forbidden": [
                            "dialogue_tag_relocation",
                            "quote_topology_change",
                            "latin_token_change",
                            "new_event_emotion_motive_lore_or_imagery",
                            "reference_phrase_copying",
                        ],
                    },
                }
            }
        },
        "rule_linked_microcards_light": {
            "intensities": {
                INTENSITY: {
                    **common,
                    "instructions": [
                        "Use aligned pairs as the baseline transformation evidence.",
                        "Use at most two activated microcards. Each card's examples demonstrate only that card's operation.",
                        "Treat the counterexample as evidence that no edit is often correct.",
                    ],
                    "rule_linked_microcards": microcards,
                    "max_active_microcards": 2,
                }
            }
        },
        "microcards_only_light": {
            "intensities": {
                INTENSITY: {
                    "availability": "ready",
                    "intensity_contract": common["intensity_contract"],
                    "evidence_source": {
                        "kind": "rule_linked_microcards_only",
                        "card_source": "isolated rules induced from target-author train pairs",
                        "aligned_pair_retrieval_exposed": False,
                    },
                    "instructions": [
                        "Use only activated rule descriptions, isolated before/after examples, and counterexamples.",
                        "Use at most two activated microcards and at most one supported local operation per changed paragraph.",
                        "Treat the counterexample as evidence that no edit is often correct.",
                    ],
                    "rule_linked_microcards": microcards,
                    "max_active_microcards": 2,
                }
            }
        },
        "candidate_rerank": {
            "intensities": {
                INTENSITY: {
                    "availability": "derived_after_frozen_candidate_generation",
                    "analysis_role": "selection_conditioned_style_meter_assisted_diagnostic",
                    "independent_style_success_claim_allowed": False,
                    "intensity_contract": {"label": INTENSITY},
                    "evidence_source": {"kind": "frozen_candidate_outputs_only"},
                    "candidate_methods": list(GENERATED_METHODS),
                    "neutral_candidate_enabled": True,
                    "selection_order": [
                        "reject_structure_token_quote_and_json_residue_failures",
                        "require_positive_target_margin_lift_over_neutral",
                        "maximize_frozen_target_author_margin",
                        "minimize_character_edit_ratio_within_margin_tolerance_0_03",
                        "method_id_lexical_tiebreak",
                    ],
                }
            }
        },
    }


def build_asset_bundle() -> tuple[dict[str, Any], dict[str, Any]]:
    source = pair_assets()
    microcards = build_microcards(source)
    methods = method_assets(source, microcards)
    content: dict[str, Any] = {
        "schema_version": 1,
        "target_author": TARGET_AUTHOR,
        "generator": {
            "id": "experiments.iteration3.style_transfer_payloads",
            "version": 1,
            "source_sha256": file_sha256(REPO_ROOT / "experiments/iteration3/style_transfer_payloads.py"),
        },
        "registry": {"iteration_id": "constrained_rerank_v1", "method_ids": list(METHODS)},
        "source_projections": {
            "style_prompt_sha256": file_sha256(ITERATION_ROOT / "prompts/style_transfer_method.v1.md")
        },
        "evidence_policy": {
            "target_train_book_count": 29,
            "pair_source_split": "train",
            "pair_source_view": "entity_masked_v2",
            "development_proxy_and_test_books_excluded": True,
        },
        "statistics": source.get("statistics", {}),
        "style_cards": {"rule_linked_microcards": microcards},
        "retrieval": {"algorithm": "semantic_and_structural_signature_v1"},
        "close_reading": {"asset": {}, "source_packet": {"sha256": None}},
        "aligned_pairs": source["aligned_pairs"],
        "methods": methods,
    }
    content["component_hashes"] = {
        "train_statistics_sha256": sha256_json(content["statistics"]),
        "style_cards_sha256": sha256_json(content["style_cards"]),
        "target_retrieval_index_sha256": sha256_json(
            content["retrieval"].get("target_index")
        ),
        "hard_negative_retrieval_index_sha256": sha256_json(
            content["retrieval"].get("hard_negative_index")
        ),
        "close_reading_asset_sha256": sha256_json(
            content["close_reading"]["asset"]
        ),
        "close_reading_source_packet_sha256": content["close_reading"][
            "source_packet"
        ]["sha256"],
        "aligned_pairs_sha256": sha256_json(content["aligned_pairs"]),
        "microcards_sha256": sha256_json(microcards),
        "methods_sha256": sha256_json(methods),
        "method_intensity_asset_sha256": {
            f"{method_id}:{intensity}": sha256_json(asset)
            for method_id, method in methods.items()
            for intensity, asset in method["intensities"].items()
        },
    }
    content_sha = sha256_json(content)
    bundle = {
        "schema_version": "style_transfer_payload_assets.v1",
        "content_sha256": content_sha,
        "assets": content,
    }
    asset_path = (
        ITERATION_ROOT
        / "method_assets/style_transfer_payloads.v1"
        / f"assets.{content_sha}.json"
    )
    write_json(asset_path, bundle)
    lock = {
        "schema_version": 1,
        "asset_schema": "style_transfer_payload_assets.v1",
        "asset_path": str(asset_path.relative_to(ITERATION_ROOT)),
        "content_sha256": content_sha,
        "file_sha256": file_sha256(asset_path),
    }
    write_json(ITERATION_ROOT / "method_assets/style_transfer_payloads.v1.lock.json", lock)
    return bundle, lock


def write_method_config(
    method_id: str,
    family: str,
    label: str,
    hypothesis: str,
    inputs: list[str],
    intensities: list[str],
    bundle: Mapping[str, Any],
    lock: Mapping[str, Any],
) -> Path:
    hashes = bundle["assets"]["component_hashes"]["method_intensity_asset_sha256"]
    generated = method_id in GENERATED_METHODS
    config = {
        "schema_version": 1,
        "method_id": method_id,
        "label": label,
        "family": family,
        "status": "ready",
        "hypothesis": hypothesis,
        "intensities": intensities,
        "prompt_version": "style_transfer_method.v2",
        "model_config": "protocols/model_run_config.v1.json",
        "input_contract": {
            "english_and_neutral_inputs_must_be_byte_identical_across_methods": True,
            "target_derived_allocation_metadata_available_to_runner": False,
            "original_evaluation_chinese_available_to_runner": False,
        },
        "payload_builder": {
            "id": f"{method_id}.iteration3_payload.v1",
            "source_corpus": "target_author_train_books_only",
            "development_proxy_and_test_books_excluded": True,
        },
        "generation": {
            "one_output_per_sample_and_intensity": True,
            "generated_by_llm": generated,
            "derived_from_frozen_candidates": method_id == "candidate_rerank",
            "selection_conditioned_diagnostic": method_id == "candidate_rerank",
            "max_generation_attempts": 2 if generated else 0,
            "output_schema": "schemas/style_transfer_output.v1.schema.json",
        },
        "inputs": inputs,
        "asset_manifest": {
            "path": relative(ITERATION_ROOT / str(lock["asset_path"])),
            "sha256": lock["content_sha256"],
            "file_sha256": lock["file_sha256"],
            "method_intensity_asset_sha256": {
                intensity: hashes[f"{method_id}:{intensity}"] for intensity in intensities
            },
            "ready": True,
        },
    }
    path = ITERATION_ROOT / "method_registry/methods" / f"{method_id}.v1.json"
    write_json(path, config)
    return path


def write_protocol_and_registry(bundle: dict[str, Any], lock: dict[str, Any]) -> None:
    protocol = read_json(ITERATION2_ROOT / "protocols/evaluation_protocol.v1.json")
    protocol.pop("iteration2", None)
    protocol["schema_version"] = 3
    protocol["style_meter_status"] = "requires_fresh_iteration3_calibration_before_style_generation"
    protocol["calibration"]["sample_count"] = 32
    protocol["calibration"]["freeze_artifact"] = "calibration/style_meter_threshold.v1.json"
    protocol["iteration3"] = {
        "iteration_id": "constrained_rerank_v1",
        "prior_iteration_status": "no_method_qualified",
        "fresh_screen_prior_chunk_overlap": 0,
        "pair_count": bundle["assets"]["aligned_pairs"]["count"],
        "microcard_count": len(bundle["assets"]["style_cards"]["rule_linked_microcards"]),
        "source_bindings": source_bindings(),
        "style_model": "gpt-5.5",
        "neutral_model": "gpt-5.4",
        "style_execution": {
            "source_run_id": RUN_ID,
            "style_run_id": STYLE_RUN_ID,
            "superseded_pilot_run_ids": [
                RUN_ID,
                "iteration3_gpt55_v3",
                "iteration3_gpt55_v4",
                "iteration3_gpt55_v5",
            ],
            "block_threshold": 12,
            "max_block_paragraphs": 12,
            "model_visible_paragraph_ids": "block_local_ids",
            "source_id_restoration": "deterministic_position_preserving_remap",
            "failed_block_recovery": "deterministic_recursive_bisection",
            "minimum_block_paragraphs": 1,
            "reference_representation": (
                "same_four_pair_ids_each_as_deterministic_contiguous_three_paragraph_window"
            ),
            "execution_amendments": [
                "protocols/iteration3_execution_amendment.v1.json",
                "protocols/iteration3_execution_amendment.block12_pilot.v1.json",
                "protocols/iteration3_execution_amendment.cross_run_pilot.v1.json",
                "protocols/iteration3_execution_amendment.id_safe_blocks_pilot.v1.json",
            ],
        },
        "candidate_rerank": bundle["assets"]["methods"]["candidate_rerank"]["intensities"][INTENSITY],
    }
    protocol["staged_execution"]["screening"].update(
        {
            "selection_file": f"sample_sets/{SAMPLE_SET}.screening_v1_ids.json",
            "sample_count": 36,
            "composition": "Fresh iteration-3 rows: three per eight target proxy books and one per 12 comparison authors; every row is at least five chunks from all prior allocations.",
            "initial_method_intensities": {
                "neutral_only": "none",
                "aligned_pairs_light": INTENSITY,
                "aligned_pairs_edit_plan_light": INTENSITY,
                "microcards_only_light": INTENSITY,
                "rule_linked_microcards_light": INTENSITY,
                "candidate_rerank": INTENSITY,
            },
            "promotion_rule": "Retain at most three fixed non-control generators with valid threshold-based style success, positive mean target-margin lift in both arms, hard-fidelity failure rate <=0.10 in each arm, no reference-copy failure, and completed independent semantic/readability evaluation. Candidate rerank is selection-conditioned on this style meter and is diagnostic only; it cannot satisfy the independent 80% endpoint.",
            "prior_iteration_chunk_overlap": 0,
        }
    )
    protocol["staged_execution"]["confirmation"].update(
        {
            "selection_file": f"sample_sets/{SAMPLE_SET}.confirmation_v1_ids.json",
            "sample_count": 116,
            "own_author_rows": 80,
            "cross_author_rows": 36,
            "screening_overlap": 0,
        }
    )
    protocol_path = ITERATION_ROOT / "protocols/evaluation_protocol.v1.json"
    write_json(protocol_path, protocol)

    specs = (
        ("neutral_only", "control", "Neutral only", "No-edit negative control.", ["english_semantic_source", "neutral_zh"], ["none"]),
        ("aligned_pairs_light", "pseudo_parallel", "Aligned pairs, light", "Light aligned-pair transformations improve style without iteration-2 over-editing.", ["english_semantic_source", "neutral_zh", "aligned_target_pairs"], [INTENSITY]),
        ("aligned_pairs_edit_plan_light", "pseudo_parallel_reasoned", "Aligned pairs plus constrained edit plan", "A one-operation local edit plan preserves fidelity while retaining aligned-pair lift.", ["english_semantic_source", "neutral_zh", "aligned_target_pairs", "edit_plan_contract"], [INTENSITY]),
        ("microcards_only_light", "rule_linked_microcards_ablation", "Rule-linked microcards only", "Explicit descriptions, isolated examples, and counterexamples can transfer local style without aligned-pair retrieval.", ["english_semantic_source", "neutral_zh", "activated_rule_linked_microcards"], [INTENSITY]),
        ("rule_linked_microcards_light", "pseudo_parallel_plus_rule_linked_microcards", "Aligned pairs plus rule-linked microcards", "Adding explicit rule descriptions and isolated examples to aligned-pair evidence improves transfer beyond aligned pairs alone.", ["english_semantic_source", "neutral_zh", "aligned_target_pairs", "activated_rule_linked_microcards"], [INTENSITY]),
        ("candidate_rerank", "style_meter_assisted_diagnostic", "Style-meter-assisted candidate rerank", "A prospectively frozen passage-level selector measures candidate complementarity, but its meter score is selection-conditioned and descriptive.", ["neutral_zh", "frozen_candidate_outputs", "frozen_style_meter", "deterministic_fidelity_gates"], [INTENSITY]),
    )
    rows: list[dict[str, Any]] = []
    for priority, spec in enumerate(specs):
        method_id, family, label, hypothesis, inputs, intensities = spec
        path = write_method_config(
            method_id, family, label, hypothesis, inputs, intensities, bundle, lock
        )
        rows.append(
            {
                "id": method_id,
                "family": family,
                "label": label,
                "status": "ready",
                "priority": priority,
                "intensities": intensities,
                "hypothesis": hypothesis,
                "inputs": inputs,
                "config_path": relative(path),
                "config_sha256": file_sha256(path),
                "asset_manifest": read_json(path)["asset_manifest"],
            }
        )
    model_path = ITERATION_ROOT / "protocols/model_run_config.v1.json"
    registry = {
        "schema_version": 4,
        "iteration_id": "constrained_rerank_v1",
        "target_author": TARGET_AUTHOR,
        "model_run_config": {"path": relative(model_path), "sha256": file_sha256(model_path)},
        "evaluation_protocol": {"path": relative(protocol_path), "sha256": file_sha256(protocol_path)},
        "methods": rows,
        "asset_manifest": {
            "path": relative(ITERATION_ROOT / str(lock["asset_path"])),
            "sha256": lock["content_sha256"],
            "file_sha256": lock["file_sha256"],
            "all_method_assets_verified": True,
        },
        "iteration_rule": {
            "confirmation_requires": "At least 80% deterministic and independently judged style success in both screening arms with all frozen gates passing.",
            "stop_condition": "At least 80% confirmation success with registered uncertainty bounds and final replication.",
        },
    }
    write_json(ITERATION_ROOT / "method_registry/style_methods.v1.json", registry)


def cohort_row_snapshot(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "author": str(row["author"]),
        "book_title": str(row["book_title"]),
        "source_split": str(row["source_split"]),
        "chunk_index": int(row["chunk_index"]),
        "quality_flags": list(row.get("quality_flags", [])),
        "strata": {field: row[field] for field in research.STRATIFICATION_FIELDS},
        "measurements": {
            key: round(float(row[key]), 6)
            for key in (
                "cjk_count",
                "paragraph_count",
                "sentence_count",
                "mean_paragraph_cjk",
                "mean_sentence_cjk",
                "punctuation_density",
                "dialogue_density",
            )
        },
    }


def build_safety_amendment(summary: Mapping[str, Any]) -> dict[str, Any]:
    rows_by_book = research.load_development_candidates(
        REPO_ROOT / "datasets/unmasked/chunks.clean.jsonl", TARGET_AUTHOR
    )
    research.enrich_strata(rows_by_book)
    row_by_id = {
        str(row["chunk_id"]): row
        for rows in rows_by_book.values()
        for row in rows
    }
    prior = prior_rows()
    replacements = summary["safety_exclusions"]["replacements"]
    details: list[dict[str, Any]] = []
    for excluded_id, replacement_id in sorted(replacements.items()):
        excluded = row_by_id[excluded_id]
        replacement = row_by_id[replacement_id]
        split_name = str(excluded["source_split"])
        selection_key = f"{split_name}:{excluded['book_title']}"
        eligible = filter_prior_neighbours(
            rows_by_book[str(excluded["book_title"])], prior
        )
        originally_selected = research.select_diverse_rows(
            eligible,
            CALIBRATION_PER_BOOK + SCREEN_PER_BOOK,
            research.stable_seed(
                SEED, f"iteration3:{split_name}:{excluded['book_title']}"
            ),
            min_chunk_distance=MIN_PRIOR_DISTANCE,
        )
        retained = [
            row
            for row in originally_selected
            if str(row["chunk_id"]) != excluded_id
        ]
        unavailable = {str(row["chunk_id"]) for row in originally_selected}
        replacement_pool = [
            row
            for row in eligible
            if str(row["chunk_id"]) not in unavailable
            and str(row["chunk_id"]) not in SAFETY_EXCLUDED_CHUNK_IDS
            and all(
                abs(int(row["chunk_index"]) - int(kept["chunk_index"]))
                >= MIN_PRIOR_DISTANCE
                for kept in retained
            )
        ]
        reproduced = research.select_diverse_rows(
            replacement_pool,
            1,
            research.stable_seed(
                SEED,
                f"iteration3:safety-replacement:{selection_key}:{excluded_id}",
            ),
            min_chunk_distance=MIN_PRIOR_DISTANCE,
        )
        if len(reproduced) != 1 or str(reproduced[0]["chunk_id"]) != replacement_id:
            raise ValueError(f"Safety replacement no longer reproduces: {excluded_id}")
        excluded_sample_id = research.opaque_sample_id(SEED, SAMPLE_SET, excluded_id)
        replacement_sample_id = research.opaque_sample_id(
            SEED, SAMPLE_SET, replacement_id
        )
        refusal_paths = sorted(
            (
                ITERATION_ROOT
                / "runs"
                / SAMPLE_SET
                / SUPERSEDED_SOURCE_RUN_ID
                / "english_semantic_source/_quarantine"
            ).glob(f"{excluded_sample_id}.attempt*.json")
        )
        refusal_path = refusal_paths[-1] if refusal_paths else None
        before = cohort_row_snapshot(excluded)
        after = cohort_row_snapshot(replacement)
        details.append(
            {
                "excluded_chunk_id": excluded_id,
                "excluded_sample_id": excluded_sample_id,
                "replacement_chunk_id": replacement_id,
                "replacement_sample_id": replacement_sample_id,
                "role_preserved": "style_meter_calibration",
                "author_and_book_preserved": True,
                "selection_key": selection_key,
                "selection_seed": research.stable_seed(
                    SEED,
                    f"iteration3:safety-replacement:{selection_key}:{excluded_id}",
                ),
                "eligible_pool_count": len(replacement_pool),
                "eligible_pool_chunk_ids_sha256": sha256_json(
                    sorted(str(row["chunk_id"]) for row in replacement_pool)
                ),
                "excluded_row": before,
                "replacement_row": after,
                "measurement_delta_replacement_minus_excluded": {
                    key: round(after["measurements"][key] - before["measurements"][key], 6)
                    for key in before["measurements"]
                },
                "refusal_artifact_path": relative(refusal_path) if refusal_path else None,
                "refusal_artifact_sha256": file_sha256(refusal_path) if refusal_path else None,
            }
        )
    evidence_ledger = (
        ITERATION_ROOT
        / "runs"
        / SAMPLE_SET
        / SUPERSEDED_SOURCE_RUN_ID
        / "ledgers/english_semantic_source.jsonl"
    )
    return {
        "schema_version": 2,
        "status": "frozen_before_style_generation_and_style_meter_calibration",
        "reason": "Source construction returned a safety refusal for one calibration chunk; the content is outside the permitted generation scope.",
        "policy": summary["safety_exclusions"]["policy"],
        "replacements": details,
        "calibration_sensitivity_required": "recompute the threshold with the replacement row omitted",
        "provenance_note": "The superseded source run remains immutable. The amended cohort is regenerated under a new run ID; no retained source output is reused across run IDs.",
        "superseded_evidence_ledger": relative(evidence_ledger),
        "superseded_evidence_ledger_sha256": file_sha256(evidence_ledger),
        "active_replacement_generation": None,
    }


def finalize_safety_amendment() -> Path:
    path = ITERATION_ROOT / "protocols/iteration3_source_cohort_amendment.v1.json"
    amendment = read_json(path)
    run_root = ITERATION_ROOT / "runs" / SAMPLE_SET / RUN_ID
    ledger_path = run_root / "ledgers/english_semantic_source.jsonl"
    active: list[dict[str, Any]] = []
    ledger_rows = list(iter_jsonl(ledger_path))
    for replacement in amendment["replacements"]:
        sample_id = replacement["replacement_sample_id"]
        source_path = run_root / "english_semantic_source" / f"{sample_id}.json"
        rows = [
            row
            for row in ledger_rows
            if row.get("sample_id") == sample_id and row.get("status") == "success"
        ]
        if not source_path.exists() or len(rows) != 1:
            raise ValueError(f"Replacement source provenance incomplete: {sample_id}")
        active.append(
            {
                "sample_id": sample_id,
                "source_artifact_path": relative(source_path),
                "source_artifact_sha256": file_sha256(source_path),
                "successful_ledger_row_sha256": rows_sha256(rows),
                "response_id": rows[0].get("response_id"),
                "execution_selection_sha256": rows[0].get(
                    "execution_selection_sha256"
                ),
            }
        )
    amendment["active_replacement_generation"] = active
    amendment["status"] = "complete_before_style_generation_and_style_meter_calibration"
    write_json(path, amendment)
    return path


def write_confirmation_independence_ledger() -> Path:
    confirmation = read_json(
        ITERATION_ROOT / f"sample_sets/{SAMPLE_SET}.confirmation_v1_ids.json"
    )["sample_ids"]
    observed_outputs = [
        str(path.relative_to(REPO_ROOT))
        for sample_id in confirmation
        for path in (ITERATION_ROOT / "runs").glob(
            f"**/method_outputs/**/{sample_id}.json"
        )
    ]
    path = ITERATION_ROOT / "protocols/iteration3_confirmation_independence.v1.json"
    write_json(
        path,
        {
            "schema_version": 1,
            "status": "unopened" if not observed_outputs else "contaminated",
            "sample_count": len(confirmation),
            "confirmation_ids_sha256": file_sha256(
                ITERATION_ROOT
                / f"sample_sets/{SAMPLE_SET}.confirmation_v1_ids.json"
            ),
            "method_outputs_present_before_style_freeze": observed_outputs,
            "design_access": [
                "opaque IDs and aggregate allocation geometry were copied from iteration 2",
                "no confirmation method output, style score, or diagnostic was available",
                "original target text remains evaluator-only and unavailable to generation",
            ],
            "claim_scope": "Operational holdout independence is established by the absence of generated outputs and diagnostics; this is not a claim that the encrypted/local target file does not exist.",
        },
    )
    if observed_outputs:
        raise ValueError("Confirmation outputs existed before iteration-3 style freeze")
    return path


def init_iteration() -> dict[str, Any]:
    if list((ITERATION_ROOT / "runs").glob("**/method_outputs/**/*.json")):
        raise ValueError("Cannot reinitialize iteration 3 after style outputs exist")
    copy_runtime_artifacts()
    summary = build_sample_set()
    amendment = build_safety_amendment(summary)
    amendment_path = ITERATION_ROOT / "protocols/iteration3_source_cohort_amendment.v1.json"
    write_json(amendment_path, amendment)
    model = write_model_config("gpt-5.4", "iteration3_source_construction")
    preregistration = {
        "schema_version": 1,
        "status": "source_inputs_frozen_before_generation",
        "iteration_id": "constrained_rerank_v1",
        "research_question": "Do rule-linked microcards alone or as an aligned-pair add-on improve fixed-generator target-author style success while preserving English meaning, and how much candidate complementarity is visible in a selection-conditioned rerank diagnostic?",
        "sample_set": SAMPLE_SET,
        "run_id": RUN_ID,
        "source_run_id": RUN_ID,
        "style_run_id": STYLE_RUN_ID,
        "calibration_rows": 32,
        "screening_rows": 36,
        "confirmation_rows": 116,
        "planned_methods": list(METHODS),
        "source_bindings": source_bindings(),
        "model_config_sha256": file_sha256(ITERATION_ROOT / "protocols/model_run_config.v1.json"),
        "sample_summary_sha256": file_sha256(ITERATION_ROOT / f"sample_sets/{SAMPLE_SET}.summary.json"),
        "source_cohort_amendment_sha256": file_sha256(amendment_path),
    }
    write_json(ITERATION_ROOT / "protocols/iteration3_preregistration.v1.json", preregistration)
    return {
        "status": "initialized",
        "experiment_root": relative(ITERATION_ROOT),
        "sample_summary": summary,
        "source_model": model["codex_model"],
    }


def qa_approved(sample_id: str) -> bool:
    root = ITERATION_ROOT / "runs" / SAMPLE_SET / RUN_ID
    initial = root / "english_source_qa" / f"{sample_id}.json"
    if initial.exists() and read_json(initial).get("approved_for_neutral_translation"):
        return True
    for round_number in range(1, 21):
        qa = root / "english_source_repair_qa" / f"round_{round_number:02d}" / f"{sample_id}.json"
        repair = root / "english_source_repair" / f"round_{round_number:02d}" / f"{sample_id}.json"
        if not qa.exists() and not repair.exists():
            break
        if qa.exists() and repair.exists() and read_json(qa).get("approved_for_neutral_translation"):
            return True
    return False


def freeze_style() -> dict[str, Any]:
    import experiments.iteration1.style_analysis_lock as analysis_locking

    observed_outputs = set(analysis_locking.style_output_inventory(ITERATION_ROOT))
    superseded_outputs = {
        entry["path"]
        for entry in analysis_locking.superseded_style_output_inventory(
            ITERATION_ROOT
        )
    }
    active_outputs = sorted(observed_outputs - superseded_outputs)
    if active_outputs:
        raise ValueError(
            "Cannot freeze iteration 3 with active style outputs: "
            + ", ".join(active_outputs[:5])
        )
    source_ids = read_json(
        ITERATION_ROOT / f"sample_sets/{SAMPLE_SET}.screening_calibration_v1_ids.json"
    )["sample_ids"]
    run_root = ITERATION_ROOT / "runs" / SAMPLE_SET / RUN_ID
    missing = [
        sample_id
        for sample_id in source_ids
        if not (run_root / "neutral_translation" / f"{sample_id}.json").exists()
        or not qa_approved(sample_id)
    ]
    if missing:
        raise ValueError(f"Source/neutral prerequisites incomplete: {len(missing)} rows")
    amendment_path = finalize_safety_amendment()
    confirmation_path = write_confirmation_independence_ledger()
    write_model_config("gpt-5.5", "iteration3_style_transfer")
    bundle, lock = build_asset_bundle()
    write_protocol_and_registry(bundle, lock)
    prereg_path = ITERATION_ROOT / "protocols/iteration3_preregistration.v1.json"
    prereg = read_json(prereg_path)
    prereg.update(
        {
            "status": "awaiting_fresh_calibration_and_pre_generation_audit",
            "style_model_config_sha256": file_sha256(ITERATION_ROOT / "protocols/model_run_config.v1.json"),
            "evaluation_protocol_sha256": file_sha256(ITERATION_ROOT / "protocols/evaluation_protocol.v1.json"),
            "method_registry_sha256": file_sha256(ITERATION_ROOT / "method_registry/style_methods.v1.json"),
            "method_asset_content_sha256": lock["content_sha256"],
            "microcard_count": len(bundle["assets"]["style_cards"]["rule_linked_microcards"]),
            "source_bindings": source_bindings(),
            "source_run_id": RUN_ID,
            "style_run_id": STYLE_RUN_ID,
            "execution_amendment_sha256": file_sha256(
                ITERATION_ROOT
                / "protocols/iteration3_execution_amendment.id_safe_blocks_pilot.v1.json"
            ),
            "execution_amendment_sha256s": {
                path.name: file_sha256(path)
                for path in sorted(
                    (ITERATION_ROOT / "protocols").glob(
                        "*execution_amendment*.json"
                    )
                )
            },
            "source_cohort_amendment_sha256": file_sha256(amendment_path),
            "confirmation_independence_sha256": file_sha256(confirmation_path),
        }
    )
    write_json(prereg_path, prereg)
    return {
        "status": "style_assets_frozen",
        "methods": list(METHODS),
        "microcards": prereg["microcard_count"],
        "asset_content_sha256": lock["content_sha256"],
        "next": "score fresh calibration, freeze threshold, then build the analysis lock",
    }


def validate_iteration() -> dict[str, Any]:
    errors: list[str] = []
    root = ITERATION_ROOT / "sample_sets"
    summary = read_json(root / f"{SAMPLE_SET}.summary.json")
    allocation = list(iter_jsonl(root / f"{SAMPLE_SET}.evaluator_allocation.jsonl"))
    runner = list(iter_jsonl(root / f"{SAMPLE_SET}.runner_manifest.jsonl"))
    hidden = list(iter_jsonl(root / f"{SAMPLE_SET}.hidden_targets.jsonl"))
    screen = read_json(root / f"{SAMPLE_SET}.screening_v1_ids.json")
    confirmation = read_json(root / f"{SAMPLE_SET}.confirmation_v1_ids.json")
    method = read_json(root / f"{SAMPLE_SET}.method_evaluation_ids.json")
    if (len(allocation), len(runner), len(hidden)) != (184, 184, 184):
        errors.append("sample_set_count_not_184")
    if len({row["sample_id"] for row in allocation}) != 184:
        errors.append("duplicate_sample_ids")
    if screen.get("sample_count") != 36 or confirmation.get("sample_count") != 116:
        errors.append("selection_count_mismatch")
    if set(screen["sample_ids"]) & set(confirmation["sample_ids"]):
        errors.append("screen_confirmation_overlap")
    if set(screen["sample_ids"]) | set(confirmation["sample_ids"]) != set(method["sample_ids"]):
        errors.append("method_partition_mismatch")
    if summary.get("fresh_prior_chunk_overlap") != 0:
        errors.append("fresh_prior_chunk_overlap")
    bindings = {
        "runner_manifest_sha256": rows_sha256(runner),
        "evaluator_allocation_sha256": rows_sha256(allocation),
        "hidden_targets_sha256": rows_sha256(hidden),
        "method_evaluation_ids_sha256": file_sha256(root / f"{SAMPLE_SET}.method_evaluation_ids.json"),
        "screening_ids_sha256": file_sha256(root / f"{SAMPLE_SET}.screening_v1_ids.json"),
        "confirmation_ids_sha256": file_sha256(root / f"{SAMPLE_SET}.confirmation_v1_ids.json"),
        "screening_calibration_ids_sha256": file_sha256(root / f"{SAMPLE_SET}.screening_calibration_v1_ids.json"),
    }
    for key, observed in bindings.items():
        if summary.get(key) != observed:
            errors.append(f"summary_binding_mismatch:{key}")
    if any(any(key in row for key in ("author", "book_title", "original_zh")) for row in runner):
        errors.append("runner_metadata_leak")
    threshold_path = ITERATION_ROOT / "calibration/style_meter_threshold.v1.json"
    if threshold_path.exists():
        threshold = read_json(threshold_path)
        protocol_path = ITERATION_ROOT / "protocols/evaluation_protocol.v1.json"
        if threshold.get("allocation_sha256") != file_sha256(root / f"{SAMPLE_SET}.evaluator_allocation.jsonl"):
            errors.append("threshold_allocation_binding_mismatch")
        if threshold.get("evaluation_protocol_sha256") != file_sha256(protocol_path):
            errors.append("threshold_protocol_binding_mismatch")
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "experiment_root": relative(ITERATION_ROOT),
        "samples": len(allocation),
        "screening": screen["sample_count"],
        "confirmation": confirmation["sample_count"],
        "threshold_present": threshold_path.exists(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare iteration-3 constrained rerank research artifacts.")
    parser.add_argument("command", choices=("init", "freeze-style", "validate"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "init":
        result = init_iteration()
    elif args.command == "freeze-style":
        result = freeze_style()
    else:
        result = validate_iteration()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
