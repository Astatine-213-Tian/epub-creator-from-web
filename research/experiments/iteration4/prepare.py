#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


from experiments.shared.paths import RESEARCH_ROOT
from workflows.audit_style_dataset import compile_term_matcher, mask_terms
from workflows.author_style_meter_contract import CURRENT_SCORER_ID


REPO_ROOT = RESEARCH_ROOT

import experiments.iteration1.style_transfer_research as research  # noqa: E402
from experiments.iteration4.build_style_definition import build as build_style_definition


BASE_ROOT = REPO_ROOT / "generated/style_research/style_transfer_experiments"
ITERATION3_ROOT = BASE_ROOT / "iterations/constrained_rerank_v1"
ITERATION_ROOT = BASE_ROOT / "iterations/full_regeneration_v1"
SAMPLE_SET = "iteration4_proxy_v1"
SOURCE_RUN_ID = "iteration4_source_gpt54_official"
STYLE_RUN_ID = "iteration4_style_gpt55_v1"
TARGET_AUTHOR = "非天夜翔"
SEED = 20260715
MIN_DISTANCE = 5
PILOT_CROSS_AUTHORS = 8
SCREEN_CROSS_AUTHORS = 20
CONFIRM_CROSS_AUTHORS = 21
SAFETY_EXCLUDED_CHUNK_IDS = {"非天夜翔__相见欢__0286"}
MASK_TERMS_PATH = REPO_ROOT / "datasets/masked/mask_terms.json"
MASKED_V3_PATH = REPO_ROOT / "datasets/masked/chunks.entity_masked_v3.jsonl"
ALIGNED_PAIR_ALLOCATION_PATH = (
    BASE_ROOT
    / "iterations/aligned_pairs_v1/sample_sets/iteration2_aligned_pair_pool_v1.evaluator_allocation.jsonl"
)
TERM_SENTINEL = "\ufff0"
PLACEHOLDER_SENTINELS = {
    "<CONTENT>": "\ue000",
    "<NUM>": "\ue001",
    "<LATIN>": "\ue002",
}
LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]*")
NUMBER_RE = re.compile(r"\d+(?:[.,:/-]\d+)*")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")

METHODS = (
    "neutral_only",
    "generic_full_regeneration",
    "aligned_pairs_full_regeneration",
    "style_definition_examples_full_regeneration",
    "aligned_pairs_style_definition_full_regeneration",
    "content_plan_combined_full_regeneration",
    "independent_candidate_selector",
)
GENERATED_METHODS = METHODS[1:6]
PROMOTABLE_METHODS = METHODS[2:]
INTENSITY = "strong"

STYLE_PROMPT = """# Style-Transfer Full-Regeneration Prompt v1

## Task

Regenerate every Chinese paragraph using the supplied frozen method payload.
The English semantic source is authoritative. The neutral Chinese is a
terminology and content-preservation anchor, not a syntax template. Apply only
target-style tendencies licensed by both the method evidence and the current
English paragraph.

## Non-negotiable constraints

1. Preserve every paragraph ID and its order. Return one output paragraph for
   every input paragraph.
2. Preserve facts, events, event order, causality, negation, modality,
   intensity, quantities, Latin tokens, entities, and speaker attribution.
3. Preserve whether each paragraph contains dialogue, its speaker sequence, and
   whether it begins with direct dialogue. Do not invent dialogue, gestures,
   thoughts, emotions, imagery, lore, or transitions.
4. Regenerate the complete paragraph. Do not merely patch punctuation or swap a
   few words, but do not expand, summarize, or embellish the content.
5. Reference passages teach only abstract rhythm, syntax, discourse, dialogue,
   and punctuation patterns. Never copy eight or more consecutive Chinese
   characters or import their wording, events, entities, or imagery.
6. Corpus rates are tendencies, not quotas. Skip a cue when its semantic trigger
   is absent.
7. For `content_plan_combined_full_regeneration`, first derive a compact
   paragraph-level content plan from English and return it. For every other
   method, return an empty `content_plan` array.
8. Return valid JSON only. Paragraph text must not contain JSON wrappers,
   Markdown fences, commentary, labels, or analysis.

## Input

```json
{
  "sample_id": "...",
  "method_id": "...",
  "intensity": "strong",
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
  "intensity": "strong",
  "content_plan": [
    {"id": "p0001", "facts": ["..."], "constraints": ["..."]}
  ],
  "paragraphs": [{"id": "p0001", "zh": "..."}],
  "style_cues_applied": [],
  "style_cues_skipped": [],
  "uncertainties": []
}
```
"""

STYLE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "sample_id",
        "method_id",
        "intensity",
        "content_plan",
        "paragraphs",
        "style_cues_applied",
        "style_cues_skipped",
        "uncertainties",
    ],
    "properties": {
        "sample_id": {"type": "string", "pattern": "^s_[0-9a-f]{24}$"},
        "method_id": {"type": "string"},
        "intensity": {"type": "string", "const": INTENSITY},
        "content_plan": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "facts", "constraints"],
                "properties": {
                    "id": {"type": "string", "pattern": "^p[0-9]{4}$"},
                    "facts": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "constraints": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        },
        "paragraphs": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "zh"],
                "properties": {
                    "id": {"type": "string", "pattern": "^p[0-9]{4}$"},
                    "zh": {"type": "string", "minLength": 1},
                },
            },
        },
        "style_cues_applied": {"type": "array", "items": {"type": "string"}},
        "style_cues_skipped": {"type": "array", "items": {"type": "string"}},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
}

SELECTOR_PROMPT = """# Independent Candidate Selector v1

## Role

Select the best complete Chinese realization of the supplied English semantic
source. Candidate identities are randomized. You are not shown method names,
the frozen author classifier, or the original Chinese target.

## Decision rule

1. Treat English as authoritative. Reject invented, omitted, reordered, or
   weakened facts; changed causality, negation, modality, intensity, quantity,
   entity, or speaker attribution; and altered dialogue topology.
2. Among semantically faithful candidates, prefer natural, coherent literary
   Chinese with good sentence and paragraph flow.
3. Use only the supplied train-corpus style definition to assess style. Reward
   a tendency only where the English content licenses it. Do not reward cue
   counting, maximal rewriting, or conspicuous punctuation by itself.
4. Score every candidate independently from 1 to 5 on semantic fidelity,
   naturalness, and style adherence. Rank every candidate exactly once.
5. Select one candidate. A neutral candidate may be best when styled candidates
   introduce semantic or language defects.
6. Return JSON only. Do not mention or infer hidden method identities.
"""

SELECTOR_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "selected_candidate_id",
        "rankings",
        "selection_reason",
        "uncertainties",
    ],
    "properties": {
        "selected_candidate_id": {"type": "string", "pattern": "^candidate_[0-9]{2}$"},
        "rankings": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "candidate_id",
                    "rank",
                    "semantic_fidelity",
                    "naturalness",
                    "style_adherence",
                    "hard_semantic_error",
                    "reason",
                ],
                "properties": {
                    "candidate_id": {
                        "type": "string",
                        "pattern": "^candidate_[0-9]{2}$",
                    },
                    "rank": {"type": "integer", "minimum": 1},
                    "semantic_fidelity": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                    },
                    "naturalness": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                    },
                    "style_adherence": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                    },
                    "hard_semantic_error": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
            },
        },
        "selection_reason": {"type": "string"},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    value = "".join(canonical_json(row) + "\n" for row in rows)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def stable_order(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode("utf-8")).hexdigest()


def comparison_authors() -> tuple[str, ...]:
    splits = read_json(REPO_ROOT / "generated/style_research/corpus/splits.json")
    authors = sorted(
        {
            str(row["author"])
            for split_rows in splits.values()
            if isinstance(split_rows, list)
            for row in split_rows
            if isinstance(row, dict) and row.get("author") != TARGET_AUTHOR
        }
    )
    if len(authors) != 49:
        raise ValueError(f"Expected 49 comparison authors, found {len(authors)}")
    return tuple(authors)


def artifact_paths(sample_id: str) -> dict[str, str]:
    root = ITERATION_ROOT / "runs" / SAMPLE_SET / "<run_id>"
    return {
        "english_semantic_source": relative(
            root / "english_semantic_source" / f"{sample_id}.json"
        ),
        "english_source_qa": relative(
            root / "english_source_qa" / f"{sample_id}.json"
        ),
        "neutral_translation": relative(
            root / "neutral_translation" / f"{sample_id}.json"
        ),
        "style_transfer": relative(
            root / "method_outputs/<method_id>/<intensity>" / f"{sample_id}.json"
        ),
        "automatic_evaluation": relative(
            root / "evaluation/<method_id>/<intensity>" / f"{sample_id}.json"
        ),
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


def prior_allocation_paths() -> list[Path]:
    return [
        path
        for path in sorted(BASE_ROOT.glob("**/*.evaluator_allocation.jsonl"))
        if ITERATION_ROOT not in path.parents
    ]


def prior_allocations() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in prior_allocation_paths():
        rows.extend(iter_jsonl(path))
    return rows


def used_indices(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], list[int]]:
    used: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        used[(str(row["author"]), str(row["book_title"]))].append(
            int(row["chunk_index"])
        )
    return used


def filter_distance(
    rows: Sequence[Mapping[str, Any]],
    unavailable: Mapping[tuple[str, str], Sequence[int]],
) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in rows
        if str(row["chunk_id"]) not in SAFETY_EXCLUDED_CHUNK_IDS
        and all(
            abs(int(row["chunk_index"]) - index) >= MIN_DISTANCE
            for index in unavailable.get(
                (str(row["author"]), str(row["book_title"])), ()
            )
        )
    ]


def reserve_rows(
    candidates: Sequence[Mapping[str, Any]],
    count: int,
    *,
    seed_key: str,
    unavailable: dict[tuple[str, str], list[int]],
) -> list[dict[str, Any]]:
    eligible = filter_distance(candidates, unavailable)
    selected = research.select_diverse_rows(
        eligible,
        count,
        research.stable_seed(SEED, seed_key),
        min_chunk_distance=MIN_DISTANCE,
    )
    for row in selected:
        unavailable[(str(row["author"]), str(row["book_title"]))].append(
            int(row["chunk_index"])
        )
    return [dict(row) for row in selected]


def load_comparison_candidates(split: str) -> dict[str, list[dict[str, Any]]]:
    by_author: dict[str, list[dict[str, Any]]] = defaultdict(list)
    allowed = set(comparison_authors())
    for row in iter_jsonl(REPO_ROOT / "datasets/unmasked/chunks.clean.jsonl"):
        author = str(row.get("author", ""))
        if author not in allowed or row.get("split") != split:
            continue
        text = str(row.get("text", ""))
        if not text.strip() or research.residue_hits(text):
            continue
        candidate = {
            "author": author,
            "book_title": str(row["title"]),
            "source_split": split,
            "chunk_id": str(row["chunk_id"]),
            "chunk_index": int(row["chunk_index"]),
            "quality_flags": list(row.get("quality_flags", [])),
            "original_zh": text,
        }
        candidate.update(research.base_metrics(text))
        by_author[author].append(candidate)
    missing = sorted(allowed - set(by_author))
    if missing:
        raise ValueError(f"Missing {split} comparison authors: {missing}")
    research.enrich_strata(by_author)
    return dict(by_author)


def mark_rows(
    rows: Sequence[Mapping[str, Any]], *, role: str, arm: str
) -> list[dict[str, Any]]:
    return [
        {**row, "research_role": role, "benchmark_arm": arm} for row in rows
    ]


def select_rows() -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    unavailable = used_indices(prior_allocations())
    own_by_book = research.load_development_candidates(
        REPO_ROOT / "datasets/unmasked/chunks.clean.jsonl", TARGET_AUTHOR
    )
    research.enrich_strata(own_by_book)
    selected: list[dict[str, Any]] = []
    selections: dict[str, list[str]] = defaultdict(list)

    for split_name, titles in research.DEVELOPMENT_BOOKS.items():
        for title in titles:
            candidates = own_by_book[title]
            for role, count in (
                ("development_pilot", 1),
                ("style_meter_calibration", 4),
                ("method_screening", 3),
                ("method_confirmation", 10),
            ):
                rows = reserve_rows(
                    candidates,
                    count,
                    seed_key=f"iteration4:{split_name}:{title}:{role}",
                    unavailable=unavailable,
                )
                rows = mark_rows(
                    rows, role=role, arm="own_author_reconstruction"
                )
                selected.extend(rows)
                selections[role].extend(str(row["chunk_id"]) for row in rows)

    ordered_authors = sorted(comparison_authors(), key=stable_order)
    pilot_authors = ordered_authors[:PILOT_CROSS_AUTHORS]
    screen_authors = ordered_authors[
        PILOT_CROSS_AUTHORS : PILOT_CROSS_AUTHORS + SCREEN_CROSS_AUTHORS
    ]
    confirm_authors = ordered_authors[
        PILOT_CROSS_AUTHORS + SCREEN_CROSS_AUTHORS :
    ]
    if len(confirm_authors) != CONFIRM_CROSS_AUTHORS:
        raise ValueError("Comparison-author partition geometry is incorrect")
    train_by_author = load_comparison_candidates("train")
    dev_by_author = load_comparison_candidates("dev")
    for author in pilot_authors:
        rows = reserve_rows(
            train_by_author[author],
            1,
            seed_key=f"iteration4:cross:pilot:{author}",
            unavailable=unavailable,
        )
        rows = mark_rows(rows, role="development_pilot", arm="cross_author_transfer")
        selected.extend(rows)
        selections["development_pilot"].extend(
            str(row["chunk_id"]) for row in rows
        )
    for author in screen_authors:
        rows = reserve_rows(
            dev_by_author[author],
            1,
            seed_key=f"iteration4:cross:screen:{author}",
            unavailable=unavailable,
        )
        rows = mark_rows(rows, role="method_screening", arm="cross_author_transfer")
        selected.extend(rows)
        selections["method_screening"].extend(
            str(row["chunk_id"]) for row in rows
        )
    for author in confirm_authors:
        rows = reserve_rows(
            dev_by_author[author],
            2,
            seed_key=f"iteration4:cross:confirmation:{author}",
            unavailable=unavailable,
        )
        rows = mark_rows(
            rows, role="method_confirmation", arm="cross_author_transfer"
        )
        selected.extend(rows)
        selections["method_confirmation"].extend(
            str(row["chunk_id"]) for row in rows
        )

    counts = Counter((row["research_role"], row["benchmark_arm"]) for row in selected)
    expected = {
        ("development_pilot", "own_author_reconstruction"): 8,
        ("development_pilot", "cross_author_transfer"): 8,
        ("style_meter_calibration", "own_author_reconstruction"): 32,
        ("method_screening", "own_author_reconstruction"): 24,
        ("method_screening", "cross_author_transfer"): 20,
        ("method_confirmation", "own_author_reconstruction"): 80,
        ("method_confirmation", "cross_author_transfer"): 42,
    }
    if counts != Counter(expected):
        raise ValueError(f"Iteration-4 cohort mismatch: {counts}")
    author_partitions = {
        "pilot_cross_authors": pilot_authors,
        "screen_cross_authors": screen_authors,
        "confirmation_cross_authors": confirm_authors,
    }
    return selected, author_partitions


def materialize_samples() -> dict[str, Any]:
    selected, author_partitions = select_rows()
    masked_by_id = {
        str(row["chunk_id"]): str(row["text"])
        for row in iter_jsonl(REPO_ROOT / "datasets/masked/chunks.entity_masked_v3.jsonl")
    }
    runner: list[dict[str, Any]] = []
    allocation: list[dict[str, Any]] = []
    hidden: list[dict[str, Any]] = []
    ids_by_role: dict[str, list[str]] = defaultdict(list)
    for row in selected:
        chunk_id = str(row["chunk_id"])
        sample_id = research.opaque_sample_id(SEED, SAMPLE_SET, chunk_id)
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
                "chunk_id": chunk_id,
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
        if chunk_id not in masked_by_id:
            raise ValueError(f"Missing entity-masked target: {chunk_id}")
        hidden.append(
            {
                "sample_id": sample_id,
                "sample_set": SAMPLE_SET,
                "original_zh": row["original_zh"],
                "entity_masked_v3_zh": masked_by_id[chunk_id],
                "access_policy": (
                    "one_way_semantic_source_generator_and_frozen_output_evaluator_only"
                ),
            }
        )
        ids_by_role[str(row["research_role"])].append(sample_id)

    runner.sort(key=lambda row: row["sample_id"])
    allocation.sort(key=lambda row: row["sample_id"])
    hidden.sort(key=lambda row: row["sample_id"])
    sample_root = ITERATION_ROOT / "sample_sets"
    write_jsonl(sample_root / f"{SAMPLE_SET}.runner_manifest.jsonl", runner)
    write_jsonl(
        sample_root / f"{SAMPLE_SET}.evaluator_allocation.jsonl", allocation
    )
    write_jsonl(sample_root / f"{SAMPLE_SET}.hidden_targets.jsonl", hidden)

    selection_specs = {
        "development_v1": sorted(ids_by_role["development_pilot"]),
        "calibration_v1": sorted(ids_by_role["style_meter_calibration"]),
        "screening_v1": sorted(ids_by_role["method_screening"]),
        "confirmation_v1": sorted(ids_by_role["method_confirmation"]),
    }
    selection_specs["source_construction_v1"] = sorted(
        selection_specs["development_v1"]
        + selection_specs["calibration_v1"]
        + selection_specs["screening_v1"]
    )
    selection_specs["method_evaluation"] = sorted(
        selection_specs["development_v1"]
        + selection_specs["screening_v1"]
        + selection_specs["confirmation_v1"]
    )
    for selection_id, sample_ids in selection_specs.items():
        suffix = (
            "method_evaluation_ids"
            if selection_id == "method_evaluation"
            else f"{selection_id}_ids"
        )
        write_json(
            sample_root / f"{SAMPLE_SET}.{suffix}.json",
            {
                "schema_version": 1,
                "selection_id": selection_id,
                "sample_set": SAMPLE_SET,
                "sample_count": len(sample_ids),
                "sample_ids": sample_ids,
            },
        )

    prior = prior_allocations()
    current_chunks = {row["chunk_id"] for row in allocation}
    prior_chunks = {row["chunk_id"] for row in prior}
    summary = {
        "schema_version": 1,
        "sample_set": SAMPLE_SET,
        "seed": SEED,
        "minimum_chunk_distance": MIN_DISTANCE,
        "target_author": TARGET_AUTHOR,
        "total_samples": len(runner),
        "development_samples": len(selection_specs["development_v1"]),
        "calibration_samples": len(selection_specs["calibration_v1"]),
        "screening_samples": len(selection_specs["screening_v1"]),
        "confirmation_samples": len(selection_specs["confirmation_v1"]),
        "source_construction_samples": len(selection_specs["source_construction_v1"]),
        "geometry": {
            "development_own": 8,
            "development_cross": 8,
            "calibration_own": 32,
            "screening_own": 24,
            "screening_cross": 20,
            "confirmation_own": 80,
            "confirmation_cross": 42,
        },
        "development_books": research.DEVELOPMENT_BOOKS,
        "final_validation_books_reserved": list(research.FINAL_VALIDATION_BOOKS),
        "comparison_author_partitions": author_partitions,
        "cross_author_partitions_disjoint": (
            not set(author_partitions["pilot_cross_authors"])
            & set(author_partitions["screen_cross_authors"])
            and not set(author_partitions["pilot_cross_authors"])
            & set(author_partitions["confirmation_cross_authors"])
            and not set(author_partitions["screen_cross_authors"])
            & set(author_partitions["confirmation_cross_authors"])
        ),
        "prior_allocation_rows": len(prior),
        "prior_allocation_bindings": [
            {"path": relative(path), "sha256": file_sha256(path)}
            for path in prior_allocation_paths()
        ],
        "prior_chunk_overlap": len(current_chunks & prior_chunks),
        "eternal_gate_excluded": True,
        "runner_manifest_sha256": rows_sha256(runner),
        "evaluator_allocation_sha256": rows_sha256(allocation),
        "hidden_targets_sha256": rows_sha256(hidden),
        "selection_hashes": {
            selection_id: file_sha256(
                sample_root
                / f"{SAMPLE_SET}.{'method_evaluation_ids' if selection_id == 'method_evaluation' else selection_id + '_ids'}.json"
            )
            for selection_id in selection_specs
        },
        "scene_counts": dict(
            sorted(Counter(row["strata"]["scene_type"] for row in allocation).items())
        ),
    }
    if summary["prior_chunk_overlap"] != 0:
        raise ValueError("Iteration-4 allocation overlaps an earlier chunk")
    if not summary["cross_author_partitions_disjoint"]:
        raise ValueError("Cross-author partitions overlap")
    write_json(sample_root / f"{SAMPLE_SET}.summary.json", summary)
    return summary


def copy_runtime_artifacts() -> None:
    source_root = ITERATION3_ROOT
    copies = {
        "prompts/english_semantic_source.v1.md": "prompts/english_semantic_source.v1.md",
        "prompts/english_source_qa.v1.md": "prompts/english_source_qa.v1.md",
        "prompts/english_source_repair.v1.md": "prompts/english_source_repair.v1.md",
        "prompts/neutral_translation.v1.md": "prompts/neutral_translation.v1.md",
        "prompts/style_transfer_critique.v1.md": "prompts/style_transfer_critique.v1.md",
        "schemas/english_semantic_source_output.v1.schema.json": "schemas/english_semantic_source_output.v1.schema.json",
        "schemas/english_source_qa_output.v1.schema.json": "schemas/english_source_qa_output.v1.schema.json",
        "schemas/english_source_repair_output.v1.schema.json": "schemas/english_source_repair_output.v1.schema.json",
        "schemas/neutral_translation_output.v1.schema.json": "schemas/neutral_translation_output.v1.schema.json",
        "schemas/style_transfer_critique_output.v1.schema.json": "schemas/style_transfer_critique_output.v1.schema.json",
    }
    for source_name, target_name in copies.items():
        source = source_root / source_name
        target = ITERATION_ROOT / target_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (ITERATION_ROOT / "prompts").mkdir(parents=True, exist_ok=True)
    (ITERATION_ROOT / "prompts/style_transfer_method.v1.md").write_text(
        STYLE_PROMPT, encoding="utf-8"
    )
    (ITERATION_ROOT / "prompts/independent_candidate_selector.v1.md").write_text(
        SELECTOR_PROMPT, encoding="utf-8"
    )
    write_json(
        ITERATION_ROOT / "schemas/style_transfer_output.v1.schema.json", STYLE_SCHEMA
    )
    write_json(
        ITERATION_ROOT / "schemas/independent_candidate_selector_output.v1.schema.json",
        SELECTOR_SCHEMA,
    )
    scorer_name = CURRENT_SCORER_ID
    shutil.copytree(
        ITERATION3_ROOT / "scorers" / scorer_name,
        ITERATION_ROOT / "scorers" / scorer_name,
        dirs_exist_ok=True,
    )


def mask_term_plans() -> dict[tuple[str, str], tuple[str, ...]]:
    payload = read_json(MASK_TERMS_PATH)
    global_terms = [
        str(value)
        for value in payload.get("global_terms", {}).get("entity_terms_v2", [])
    ]
    return {
        (str(row["author"]), str(row["title"])): tuple(
            sorted(
                {
                    *global_terms,
                    *(str(value) for value in row.get("entity_terms_v2", [])),
                },
                key=lambda value: (-len(value), value),
            )
        )
        for row in payload["books"]
    }


@lru_cache(maxsize=128)
def cached_term_matcher(terms: tuple[str, ...]):
    return compile_term_matcher(list(terms))


def mask_v3_text(text: str, terms: Sequence[str]) -> str:
    masked = text.replace("<TERM>", TERM_SENTINEL)
    for placeholder, sentinel in PLACEHOLDER_SENTINELS.items():
        masked = masked.replace(placeholder, sentinel)
    masked = LATIN_RE.sub("<LATIN>", masked)
    masked = NUMBER_RE.sub("<NUM>", masked)
    masked = mask_terms(
        masked,
        cached_term_matcher(tuple(terms)),
        placeholder="某",
        preserve_length=True,
    )
    for placeholder, sentinel in PLACEHOLDER_SENTINELS.items():
        masked = masked.replace(sentinel, placeholder)
    masked = re.sub(r"<+(CONTENT|NUM|LATIN)>+", r"<\1>", masked)
    return masked


def remask_pair_paragraphs(
    *,
    pair: Mapping[str, Any],
    author: str,
    title: str,
    chunk_text: str,
    terms: Sequence[str],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    chunk_lines = [line for line in chunk_text.splitlines() if line]
    target_rows = list(pair["target_style_zh"])
    neutral_rows = list(pair["neutral_zh"])
    if [row["id"] for row in target_rows] != [row["id"] for row in neutral_rows]:
        raise ValueError(f"Aligned pair paragraph IDs disagree: {pair['pair_id']}")
    start = 0
    remasked_target: list[dict[str, str]] = []
    paragraph_captures: list[tuple[str, ...]] = []
    for target_row in target_rows:
        template = mask_v3_text(str(target_row["zh"]), terms)
        pieces = [re.escape(value) for value in template.split(TERM_SENTINEL)]
        pattern = re.compile("(某+)".join(pieces))
        matched_line: str | None = None
        captures: tuple[str, ...] = ()
        matched_index = -1
        for index in range(start, len(chunk_lines)):
            match = pattern.fullmatch(chunk_lines[index])
            if match is not None:
                matched_line = chunk_lines[index]
                captures = match.groups()
                matched_index = index
                break
        if matched_line is None:
            raise ValueError(
                "Cannot align pair paragraph to entity_masked_v3: "
                f"{pair['pair_id']} / {target_row['id']} / {author} / {title}"
            )
        start = matched_index + 1
        paragraph_captures.append(captures)
        remasked_target.append({"id": str(target_row["id"]), "zh": matched_line})
    all_captures = [value for values in paragraph_captures for value in values]
    fallback = Counter(all_captures).most_common(1)[0][0] if all_captures else "某某"
    remasked_neutral: list[dict[str, str]] = []
    for neutral_row, captures in zip(neutral_rows, paragraph_captures):
        neutral_masked = mask_v3_text(str(neutral_row["zh"]), terms)
        placeholder_count = neutral_masked.count(TERM_SENTINEL)
        replacements = list(captures)
        if len(replacements) < placeholder_count:
            replacements.extend(
                [replacements[-1] if replacements else fallback]
                * (placeholder_count - len(replacements))
            )
        for replacement in replacements[:placeholder_count]:
            neutral_masked = neutral_masked.replace(TERM_SENTINEL, replacement, 1)
        if TERM_SENTINEL in neutral_masked or "<TERM>" in neutral_masked:
            raise ValueError(f"Unresolved pair mask placeholder: {pair['pair_id']}")
        remasked_neutral.append(
            {"id": str(neutral_row["id"]), "zh": neutral_masked}
        )
    return remasked_neutral, remasked_target


def load_iteration3_pairs() -> dict[str, Any]:
    lock = read_json(ITERATION3_ROOT / "method_assets/style_transfer_payloads.v1.lock.json")
    source = ITERATION3_ROOT / str(lock["asset_path"])
    if file_sha256(source) != lock["file_sha256"]:
        raise ValueError("Iteration-3 aligned-pair source is corrupt")
    bundle = read_json(source)
    pairs = bundle["assets"]["aligned_pairs"]
    allocation = {
        str(row["sample_id"]): row
        for row in iter_jsonl(ALIGNED_PAIR_ALLOCATION_PATH)
    }
    chunks = {
        str(row["chunk_id"]): row
        for row in iter_jsonl(MASKED_V3_PATH)
        if row.get("author") == TARGET_AUTHOR and row.get("split") == "train"
    }
    plans = mask_term_plans()
    remasked_pairs: list[dict[str, Any]] = []
    for source_pair in pairs["pairs"]:
        pair = dict(source_pair)
        metadata = allocation[str(pair["source_sample_id"])]
        author = str(metadata["author"])
        title = str(metadata["book_title"])
        chunk_id = str(metadata["chunk_id"])
        if chunk_id not in chunks:
            continue
        chunk = chunks[chunk_id]
        neutral, target = remask_pair_paragraphs(
            pair=pair,
            author=author,
            title=title,
            chunk_text=str(chunk["text"]),
            terms=plans[(author, title)],
        )
        remasked = {
            **pair,
            "source_view": "entity_masked_v3",
            "neutral_zh": neutral,
            "target_style_zh": target,
            "remasking": {
                "algorithm": (
                    "book_term_plan_exact_target_alignment_with_target_mask_fallback.v1"
                ),
                "source_chunk_id": str(metadata["chunk_id"]),
                "mask_terms_sha256": file_sha256(MASK_TERMS_PATH),
                "masked_corpus_sha256": file_sha256(MASKED_V3_PATH),
            },
        }
        remasked_without_hash = {
            key: value for key, value in remasked.items() if key != "pair_sha256"
        }
        remasked["pair_sha256"] = sha256_json(remasked_without_hash)
        remasked_pairs.append(remasked)
    current_train_books = {
        (str(row["author"]), str(row["title"])) for row in chunks.values()
    }
    pair_books = {
        (
            str(allocation[str(pair["source_sample_id"])]["author"]),
            str(allocation[str(pair["source_sample_id"])]["book_title"]),
        )
        for pair in remasked_pairs
    }
    if pair_books != current_train_books or len(remasked_pairs) != len(current_train_books):
        raise ValueError(
            "Aligned pairs do not provide exactly one source for every current "
            "target-author train book"
        )
    return {
        **pairs,
        "count": len(remasked_pairs),
        "pairs": remasked_pairs,
        "source_view": "entity_masked_v3",
        "remasking_algorithm": (
            "book_term_plan_exact_target_alignment_with_target_mask_fallback.v1"
        ),
        "source_iteration": 3,
        "source_asset_path": relative(source),
        "source_asset_file_sha256": file_sha256(source),
    }


def method_static_assets() -> dict[str, Any]:
    full_contract = {
        "mode": "full_paragraph_regeneration",
        "english_authoritative": True,
        "neutral_role": "terminology_and_content_anchor_not_syntax_template",
        "rewrite_every_paragraph": True,
        "preserve_paragraph_ids_and_order": True,
        "preserve_facts_events_causality_negation_modality_intensity": True,
        "preserve_entities_numbers_latin_tokens_speakers": True,
        "preserve_dialogue_topology": True,
        "no_expansion_summary_or_embellishment": True,
        "reference_copy_limit_consecutive_cjk": 7,
        "style_tendencies_are_source_triggered_not_quotas": True,
    }
    return {
        "neutral_only": {
            "intensities": {
                "none": {
                    "generation_contract": {"mode": "no_generation_control"},
                    "evidence_source": {"kind": "none"},
                    "instructions": [],
                }
            }
        },
        "generic_full_regeneration": {
            "intensities": {
                INTENSITY: {
                    "generation_contract": full_contract,
                    "evidence_source": {
                        "kind": "none",
                        "target_style_evidence_visible": False,
                    },
                    "instructions": [
                        "Write natural contemporary literary Chinese without imitating any named author.",
                        "Use English for meaning and neutral Chinese for terminology only.",
                    ],
                }
            }
        },
        "aligned_pairs_full_regeneration": {
            "intensities": {
                INTENSITY: {
                    "generation_contract": full_contract,
                    "evidence_source": {
                        "kind": "retrieved_aligned_neutral_to_target_pairs",
                        "aligned_pair_k": 6,
                        "one_pair_per_train_book": True,
                    },
                    "instructions": [
                        "Infer recurring structural transformations from the aligned examples.",
                        "Apply only transformations licensed by the current English source.",
                    ],
                }
            }
        },
        "style_definition_examples_full_regeneration": {
            "intensities": {
                INTENSITY: {
                    "generation_contract": full_contract,
                    "evidence_source": {
                        "kind": "validated_style_definition_and_masked_scene_examples",
                        "aligned_pair_k": 0,
                    },
                    "instructions": [
                        "Use the validated dimensions as conditional guidance, not frequency quotas.",
                        "Use examples only to understand rhythm and structure; never copy content or wording.",
                    ],
                }
            }
        },
        "aligned_pairs_style_definition_full_regeneration": {
            "intensities": {
                INTENSITY: {
                    "generation_contract": full_contract,
                    "evidence_source": {
                        "kind": "aligned_pairs_plus_validated_style_definition",
                        "aligned_pair_k": 4,
                    },
                    "instructions": [
                        "Use the definition to interpret which aligned transformations generalize.",
                        "When pair evidence and a semantic guardrail conflict, preserve English meaning.",
                    ],
                }
            }
        },
        "content_plan_combined_full_regeneration": {
            "intensities": {
                INTENSITY: {
                    "generation_contract": full_contract,
                    "evidence_source": {
                        "kind": "english_content_plan_plus_aligned_pairs_plus_validated_style_definition",
                        "aligned_pair_k": 4,
                    },
                    "instructions": [
                        "Build and return the English-grounded content plan before regenerating Chinese.",
                        "After drafting, verify every output paragraph against its plan and English source.",
                    ],
                    "content_plan_contract": {
                        "one_entry_per_paragraph": True,
                        "source": "english_semantic_source_only",
                        "facts": "events_states_participants_speakers",
                        "constraints": "negation_modality_intensity_temporal_causal_and_protected_tokens",
                        "style_evidence_forbidden_during_plan": True,
                    },
                }
            }
        },
        "independent_candidate_selector": {
            "intensities": {
                INTENSITY: {
                    "generation_contract": {"mode": "derived_blind_candidate_selection"},
                    "evidence_source": {
                        "kind": "english_neutral_anonymized_candidates_and_style_definition",
                        "frozen_style_meter_visible": False,
                        "method_labels_visible": False,
                    },
                    "instructions": [
                        "Select for English fidelity, naturalness, and source-licensed style-definition adherence.",
                    ],
                }
            }
        },
    }


def build_method_bundle() -> dict[str, Any]:
    style_lock = build_style_definition(ITERATION_ROOT)
    style_wrapper = read_json(ITERATION_ROOT / str(style_lock["asset_path"]))
    style_definition = style_wrapper["asset"]
    aligned_pairs = load_iteration3_pairs()
    methods = method_static_assets()
    payload_source = REPO_ROOT / "experiments/iteration4/style_transfer_payloads.py"
    if not payload_source.exists():
        raise ValueError(f"Missing payload builder: {payload_source}")
    assets: dict[str, Any] = {
        "schema_version": 1,
        "target_author": TARGET_AUTHOR,
        "generator": {
            "id": "experiments.iteration4.style_transfer_payloads",
            "version": 1,
            "source_sha256": file_sha256(payload_source),
        },
        "source_projections": {
            "style_prompt_sha256": file_sha256(
                ITERATION_ROOT / "prompts/style_transfer_method.v1.md"
            ),
            "style_output_schema_sha256": file_sha256(
                ITERATION_ROOT / "schemas/style_transfer_output.v1.schema.json"
            ),
        },
        "evidence_policy": {
            "masked_train_only": True,
            "development_final_and_eternal_gate_excluded": True,
            "target_train_books": 29,
            "comparison_train_authors": 49,
        },
        "style_definition": style_definition,
        "aligned_pairs": aligned_pairs,
        "methods": methods,
    }
    assets["component_hashes"] = {
        "source_projections_sha256": sha256_json(assets["source_projections"]),
        "style_definition_sha256": sha256_json(style_definition),
        "aligned_pairs_sha256": sha256_json(aligned_pairs),
        "method_intensity_asset_sha256": {
            f"{method_id}:{intensity}": sha256_json(asset)
            for method_id, method in methods.items()
            for intensity, asset in method["intensities"].items()
        },
    }
    content_sha = sha256_json(assets)
    wrapper = {
        "schema_version": 1,
        "content_sha256": content_sha,
        "assets": assets,
    }
    asset_relative = Path("method_assets/style_transfer_payloads.v1") / f"assets.{content_sha}.json"
    asset_path = ITERATION_ROOT / asset_relative
    write_json(asset_path, wrapper)
    lock = {
        "schema_version": 1,
        "asset_schema": "style_transfer_payload_assets.v4.full_regeneration",
        "asset_path": asset_relative.as_posix(),
        "content_sha256": content_sha,
        "file_sha256": file_sha256(asset_path),
    }
    write_json(ITERATION_ROOT / "method_assets/style_transfer_payloads.v1.lock.json", lock)
    return {"lock": lock, "assets": assets}


def write_model_config(model: str, phase: str) -> dict[str, Any]:
    prompt_files = {
        "english_semantic_source": "prompts/english_semantic_source.v1.md",
        "english_source_qa": "prompts/english_source_qa.v1.md",
        "english_source_repair": "prompts/english_source_repair.v1.md",
        "neutral_translation": "prompts/neutral_translation.v1.md",
        "style_transfer": "prompts/style_transfer_method.v1.md",
        "style_transfer_critique": "prompts/style_transfer_critique.v1.md",
    }
    versions = {
        "english_semantic_source": "english_semantic_source.v1",
        "english_source_qa": "english_source_qa.v1",
        "english_source_repair": "english_source_repair.v1",
        "neutral_translation": "neutral_translation.v1",
        "style_transfer": "style_transfer_full_regeneration.v1",
        "style_transfer_critique": "style_transfer_critique.v1",
    }
    config = {
        "schema_version": 1,
        "runner": "codex_exec_external_seatbelt_v2",
        "codex_model": model,
        "reasoning_effort": "high",
        "provider": "openai_codex_cli_subscription_backend",
        "backend_snapshot_available": False,
        "provenance_claim": "frozen_output_provenance_not_exact_rerun_identity",
        "external_sandbox": "macos_seatbelt_deny_repository_read_write",
        "codex_internal_sandbox": "disabled_inside_external_sandbox",
        "ephemeral_session_per_request": True,
        "ignore_user_config": True,
        "ignore_project_rules": True,
        "temperature": "not_exposed_by_codex_cli",
        "max_attempts": 2,
        "request_timeout_seconds": 900,
        "default_parallel_jobs": 4,
        "prompt_versions": {
            key: {
                "version": versions[key],
                "sha256": file_sha256(ITERATION_ROOT / path),
                **(
                    {"production_contract": "eternal_gate_pass_1"}
                    if key == "neutral_translation"
                    else {}
                ),
            }
            for key, path in prompt_files.items()
        },
        "required_ledger_fields": [
            "run_id",
            "stage",
            "sample_id",
            "attempt",
            "started_at",
            "completed_at",
            "status",
            "model",
            "reasoning_effort",
            "prompt_sha256",
            "input_sha256",
            "output_sha256",
            "schema_sha256",
            "response_file",
            "response_id",
            "response_usage",
            "runner_sha256",
            "environment_sha256",
            "sandbox_profile_sha256",
            "validation_errors",
        ],
        "experiment_iteration": 4,
        "experiment_phase": phase,
        "model_selection": {
            "neutral_reconstruction_model": "gpt-5.4",
            "style_transfer_model": "gpt-5.5",
            "no_model_mixing_within_style_arms": True,
        },
    }
    write_json(ITERATION_ROOT / "protocols/model_run_config.v1.json", config)
    return config


def write_protocols_and_registry(
    summary: Mapping[str, Any], bundle: Mapping[str, Any], model_config: Mapping[str, Any]
) -> None:
    protocol = {
        "schema_version": 1,
        "iteration_id": "full_regeneration_v1",
        "target_author": TARGET_AUTHOR,
        "sample_set": SAMPLE_SET,
        "status": "development_source_construction",
        "source_run_id": SOURCE_RUN_ID,
        "style_run_id": STYLE_RUN_ID,
        "neutral_production_contract": "eternal_gate_pass_1",
        "style_meter_status": "requires_fresh_iteration4_calibration_before_style_generation",
        "style_meter": {
            "model": "class_balanced_sgd_hinge_exact_char_ngrams_min_df_20",
            "input_view": "entity_masked_v3",
            "reported_outputs": [
                "target_decision_margin",
                "target_rank",
                "target_chunk_share",
                "paired_margin_lift_over_neutral",
            ],
        },
        "calibration": {
            "allowed_role": "style_meter_calibration",
            "sample_count": 32,
            "minimum_positive_sensitivity": 0.80,
            "maximum_neutral_false_positive_rate": 0.10,
            "required_score_binding_fields": [
                "scorer_id",
                "classifier_artifact_sha256",
                "scorer_config_sha256",
                "masking_view",
                "masking_artifact_sha256",
                "original_input_sha256",
                "neutral_input_sha256",
            ],
            "positive_distribution": "held_out_original_entity_masked_v3",
            "negative_distribution": "neutral_translation_entity_masked_v3",
            "threshold_algorithm": (
                "Enumerate unique target-author decision margins. Keep thresholds "
                "with original-positive sensitivity >=0.80 and neutral false-positive "
                "rate <=0.10; choose highest balanced accuracy, then the higher "
                "threshold on ties. If none qualifies, binary style success is undefined."
            ),
            "freeze_artifact": "calibration/style_meter_threshold.v1.json",
            "forbidden": [
                "recalibration_on_method_evaluation_rows",
                "recalibration_after_viewing_method_labels",
                "probability_interpretation_of_raw_hinge_margins",
            ],
        },
        "chunk_style_success": {
            "all_required": [
                "target_decision_margin_at_or_above_frozen_threshold",
                "target_rank_at_most_5",
                "paired_target_margin_lift_over_neutral_strictly_positive",
                "no_hard_fidelity_failure",
            ]
        },
        "hard_fidelity_failures": [
            "missing_or_duplicate_output",
            "paragraph_id_or_order_mismatch",
            "entity_number_role_or_speaker_mismatch",
            "causality_chronology_negation_modality_or_intensity_change",
            "unsupported_event_motive_lore_joke_intimacy_injury_or_setting_detail",
            "copied_reference_sequence_of_8_or_more_cjk_characters",
            "reference_book_name_place_or_lore_leakage",
            "stale_or_mispaired_independent_judgment_binding",
            "high_severity_blind_llm_semantic_judge_failure",
            "high_severity_blind_llm_readability_judge_failure",
        ],
        "primary_endpoints": {
            "own_author_reconstruction": (
                "At least 0.80 style-success over 80 confirmation rows, Wilson 95% "
                "lower bound >=0.70, book-cluster bootstrap lower bound >=0.70, "
                "and positive paired margin lift."
            ),
            "cross_author_transfer": (
                "Report separately over 42 confirmation rows from 21 authors; at "
                "least 0.80 style-success, Wilson 95% lower bound >=0.70, author-"
                "cluster bootstrap lower bound >=0.70, and no pooling with own-author rows."
            ),
            "semantic_noninferiority": (
                "No increase in paired high-severity semantic failure versus neutral; "
                "missing outputs count as failures."
            ),
            "readability_noninferiority": (
                "No high-severity readability regression versus neutral under blind "
                "independent judgment; missing outputs count as failures."
            ),
        },
        "final_validation": {
            "books": list(research.FINAL_VALIDATION_BOOKS),
            "sample_count": 80,
            "chunks_per_book": 20,
            "one_time_only": True,
            "creation_gate": (
                "Create reserved artifacts only after a confirmation method passes "
                "all judged endpoints and is locked."
            ),
            "lock_artifact": "final_validation/final_validation_v1.lock.json",
            "pre_reveal_lock_artifact": (
                "final_validation/final_validation_v1.pre_reveal.lock.json"
            ),
            "same_frozen_threshold_prompt_model_and_method_payload": True,
            "required_own_author_style_success_point_estimate": 0.80,
            "required_cluster_bootstrap_lower_bound": 0.70,
        },
        "cohort": {
            "summary_path": f"sample_sets/{SAMPLE_SET}.summary.json",
            "summary_sha256": file_sha256(
                ITERATION_ROOT / f"sample_sets/{SAMPLE_SET}.summary.json"
            ),
            "geometry": summary["geometry"],
            "minimum_chunk_distance": MIN_DISTANCE,
            "prior_chunk_overlap": 0,
            "cross_author_partitions_disjoint": True,
        },
        "staged_execution": {
            "development": {
                "selection_file": f"sample_sets/{SAMPLE_SET}.development_v1_ids.json",
                "sample_count": 16,
                "official_efficacy_evidence": False,
                "generation_method_intensities": {
                    method_id: INTENSITY for method_id in GENERATED_METHODS
                },
                "adaptation_after_pilot": "forbidden_within_iteration4",
            },
            "calibration": {
                "selection_file": f"sample_sets/{SAMPLE_SET}.calibration_v1_ids.json",
                "sample_count": 32,
                "allowed_role": "style_meter_calibration",
                "style_generation_forbidden": True,
            },
            "screening": {
                "selection_file": f"sample_sets/{SAMPLE_SET}.screening_v1_ids.json",
                "sample_count": 44,
                "own": 24,
                "cross": 20,
                "initial_method_intensities": {
                    "neutral_only": "none",
                    "generic_full_regeneration": INTENSITY,
                    "aligned_pairs_full_regeneration": INTENSITY,
                    "style_definition_examples_full_regeneration": INTENSITY,
                    "aligned_pairs_style_definition_full_regeneration": INTENSITY,
                    "content_plan_combined_full_regeneration": INTENSITY,
                    "independent_candidate_selector": INTENSITY,
                },
                "promotion_rule": (
                    "Retain at most three promotable methods with valid threshold-based "
                    "style success, positive mean target-margin lift in both arms, "
                    "hard-fidelity failure <=0.10 in each arm, no reference-copy "
                    "failure, and completed independent semantic/readability evaluation."
                ),
                "interpretation": (
                    "Method-family screening only; these 44 rows cannot satisfy the "
                    "confirmation or final 80% endpoint."
                ),
                "selection_enforcement": (
                    "Bind the frozen path, hash, 24/20 arm counts, eight-book geometry, "
                    "and 20 distinct comparison authors."
                ),
                "roster_enforcement": (
                    "Initial screening contains every registered method/intensity "
                    "combination exactly once."
                ),
                "promotion_artifact": "promotions/screening_v1.promoted_methods.v1.json",
            },
            "confirmation": {
                "selection_file": f"sample_sets/{SAMPLE_SET}.confirmation_v1_ids.json",
                "sample_count": 122,
                "own": 80,
                "cross": 42,
                "own_author_rows": 80,
                "cross_author_rows": 42,
                "locked_until_promotion": True,
                "screening_overlap": 0,
                "selection_bias_control": (
                    "Only methods frozen by screening promotion enter confirmation."
                ),
                "admission_control": (
                    "The frozen promotion artifact and confirmation selection hash are "
                    "mandatory; arbitrary subsets are descriptive only."
                ),
            },
            "final_validation": {
                "sample_count": 80,
                "books": list(research.FINAL_VALIDATION_BOOKS),
                "artifacts_created": False,
                "locked_until_confirmation": True,
            },
        },
        "methods": {
            "all": list(METHODS),
            "generated": list(GENERATED_METHODS),
            "promotable": list(PROMOTABLE_METHODS),
            "intensity": INTENSITY,
            "asset_lock": "method_assets/style_transfer_payloads.v1.lock.json",
            "asset_content_sha256": bundle["lock"]["content_sha256"],
        },
        "development_policy": {
            "style_meter_raw_margin_allowed": True,
            "official_screening_unopened": True,
            "prompt_changes_require_documentation_before_official_lock": True,
            "high_severity_semantic_error_rejects_prompt": True,
        },
        "screening_gate": {
            "minimum_success_rate_each_arm": 0.80,
            "maximum_hard_fidelity_failure_rate_each_arm": 0.10,
            "positive_mean_lift_each_arm": True,
            "zero_reference_copy_failures": True,
            "independent_semantic_readability_required": True,
            "missing_outputs_fail": True,
        },
        "confirmation_gate": {
            "minimum_success_rate_each_arm": 0.80,
            "minimum_wilson_lower_bound": 0.70,
            "minimum_cluster_bootstrap_lower_bound": 0.70,
            "semantic_readability_noninferiority": True,
        },
        "final_gate": {
            "minimum_success_rate": 0.80,
            "minimum_book_cluster_bootstrap_lower_bound": 0.70,
        },
        "selector_independence": {
            "selection_model": "gpt-5.4",
            "reasoning_effort": "high",
            "frozen_style_meter_visible": False,
            "method_labels_visible": False,
            "original_target_visible": False,
            "candidate_methods": [
                "neutral_only",
                "aligned_pairs_full_regeneration",
                "style_definition_examples_full_regeneration",
                "aligned_pairs_style_definition_full_regeneration",
                "content_plan_combined_full_regeneration",
            ],
            "criteria": [
                "english_semantic_fidelity",
                "naturalness",
                "source_licensed_style_definition_adherence",
            ],
        },
        "iteration4": {
            "source_bindings": {
                relative(REPO_ROOT / "experiments/iteration4/prepare.py"): file_sha256(
                    REPO_ROOT / "experiments/iteration4/prepare.py"
                ),
                relative(
                    REPO_ROOT / "experiments/iteration4/build_style_definition.py"
                ): file_sha256(
                    REPO_ROOT / "experiments/iteration4/build_style_definition.py"
                ),
                relative(
                    REPO_ROOT / "experiments/iteration4/style_transfer_payloads.py"
                ): file_sha256(
                    REPO_ROOT / "experiments/iteration4/style_transfer_payloads.py"
                ),
                relative(
                    REPO_ROOT
                    / "experiments/iteration4/run_style_transfer_block_generation.py"
                ): file_sha256(
                    REPO_ROOT
                    / "experiments/iteration4/run_style_transfer_block_generation.py"
                ),
                relative(
                    REPO_ROOT
                    / "experiments/iteration4/evaluate_style_transfer_methods.py"
                ): file_sha256(
                    REPO_ROOT
                    / "experiments/iteration4/evaluate_style_transfer_methods.py"
                ),
                relative(
                    REPO_ROOT
                    / "experiments/iteration4/build_independent_candidate_selector.py"
                ): file_sha256(
                    REPO_ROOT
                    / "experiments/iteration4/build_independent_candidate_selector.py"
                ),
                relative(
                    REPO_ROOT
                    / "experiments/iteration4/build_source_repair_selection.py"
                ): file_sha256(
                    REPO_ROOT
                    / "experiments/iteration4/build_source_repair_selection.py"
                ),
            }
        },
    }
    write_json(ITERATION_ROOT / "protocols/evaluation_protocol.v1.json", protocol)
    preregistration = {
        "schema_version": 1,
        "iteration_id": "full_regeneration_v1",
        "sample_set": SAMPLE_SET,
        "research_question": (
            "Can full regeneration reach at least 80% frozen style success without "
            "semantic or readability regression?"
        ),
        "prior_iteration_results_visible": True,
        "iteration4_outcomes_visible_at_registration": False,
        "method_ids": list(METHODS),
        "hypotheses": {
            "H1": "Generic regeneration estimates model-default movement and is not promotable.",
            "H2": "Aligned pairs produce positive lift in both arms at full regeneration strength.",
            "H3": "A validated style definition plus examples produces positive lift in both arms.",
            "H4": "Pairs plus definition outperform either evidence source alone without exceeding fidelity limits.",
            "H5": "An explicit English content plan lowers semantic failures while retaining style movement.",
            "H6": "A meter-blind selector exploits candidate complementarity and generalizes to untouched confirmation.",
        },
        "cohort_summary_sha256": protocol["cohort"]["summary_sha256"],
        "method_asset_content_sha256": bundle["lock"]["content_sha256"],
        "model_config_sha256": sha256_json(model_config),
        "evaluation_protocol_sha256": sha256_json(protocol),
        "report_policy": "one_canonical_human_readable_report_for_iteration4",
    }
    write_json(
        ITERATION_ROOT / "protocols/iteration4_preregistration.v1.json",
        preregistration,
    )

    method_definitions = {
        "neutral_only": (
            "control",
            "Neutral only",
            "No-generation paired baseline.",
            False,
        ),
        "generic_full_regeneration": (
            "generic_full_regeneration_control",
            "Generic full regeneration",
            "Measure movement caused by regeneration without target evidence.",
            False,
        ),
        "aligned_pairs_full_regeneration": (
            "pseudo_parallel_full_regeneration",
            "Aligned pairs, full regeneration",
            "Direct transformations move style at adequate strength.",
            True,
        ),
        "style_definition_examples_full_regeneration": (
            "style_definition_examples_full_regeneration",
            "Validated style definition plus examples",
            "Comprehensive descriptions and scene examples transfer recurring style.",
            True,
        ),
        "aligned_pairs_style_definition_full_regeneration": (
            "combined_evidence_full_regeneration",
            "Aligned pairs plus style definition",
            "Explicit interpretation improves pair generalization.",
            True,
        ),
        "content_plan_combined_full_regeneration": (
            "content_planned_combined_full_regeneration",
            "Content plan plus combined evidence",
            "English planning permits strong style with lower semantic risk.",
            True,
        ),
        "independent_candidate_selector": (
            "meter_blind_candidate_selection",
            "Independent candidate selector",
            "Blind multidimensional selection exploits complementary candidates.",
            True,
        ),
    }
    method_rows: list[dict[str, Any]] = []
    for priority, method_id in enumerate(METHODS):
        family, label, hypothesis, promotable = method_definitions[method_id]
        intensities = ["none"] if method_id == "neutral_only" else [INTENSITY]
        config = {
            "schema_version": 1,
            "method_id": method_id,
            "family": family,
            "label": label,
            "status": "ready",
            "promotable": promotable,
            "intensities": intensities,
            "hypothesis": hypothesis,
            "asset_content_sha256": bundle["lock"]["content_sha256"],
        }
        config_path = (
            ITERATION_ROOT / f"method_registry/methods/{method_id}.v1.json"
        )
        write_json(config_path, config)
        method_rows.append(
            {
                "id": method_id,
                **config,
                "priority": priority,
                "config_path": relative(config_path),
                "config_sha256": file_sha256(config_path),
            }
        )
    registry = {
        "schema_version": 1,
        "iteration_id": "full_regeneration_v1",
        "target_author": TARGET_AUTHOR,
        "model_run_config": {
            "path": relative(
                ITERATION_ROOT / "protocols/model_run_config.v1.json"
            ),
            "sha256": file_sha256(
                ITERATION_ROOT / "protocols/model_run_config.v1.json"
            ),
        },
        "evaluation_protocol": {
            "path": relative(
                ITERATION_ROOT / "protocols/evaluation_protocol.v1.json"
            ),
            "sha256": file_sha256(
                ITERATION_ROOT / "protocols/evaluation_protocol.v1.json"
            ),
        },
        "methods": method_rows,
        "asset_manifest": bundle["lock"],
    }
    write_json(ITERATION_ROOT / "method_registry/style_methods.v1.json", registry)


def init(model: str) -> dict[str, Any]:
    if list((ITERATION_ROOT / "runs").glob("**/method_outputs/**/*.json")):
        raise ValueError("Cannot reinitialize Iteration 4 after style outputs exist")
    ITERATION_ROOT.mkdir(parents=True, exist_ok=True)
    copy_runtime_artifacts()
    summary = materialize_samples()
    bundle = build_method_bundle()
    model_config = write_model_config(model, "iteration4_source_construction")
    write_protocols_and_registry(summary, bundle, model_config)
    return {
        "status": "initialized",
        "experiment_root": relative(ITERATION_ROOT),
        "sample_set": SAMPLE_SET,
        "source_run_id": SOURCE_RUN_ID,
        "style_run_id": STYLE_RUN_ID,
        "source_model": model,
        "total_rows": summary["total_samples"],
        "source_construction_rows": summary["source_construction_samples"],
        "methods": list(METHODS),
        "method_asset_content_sha256": bundle["lock"]["content_sha256"],
    }


def set_style_model(model: str) -> dict[str, Any]:
    if not (ITERATION_ROOT / "protocols/evaluation_protocol.v1.json").exists():
        raise ValueError("Run init before switching to the style model")
    config = write_model_config(model, "iteration4_style_transfer")
    config_path = ITERATION_ROOT / "protocols/model_run_config.v1.json"
    registry_path = ITERATION_ROOT / "method_registry/style_methods.v1.json"
    registry = read_json(registry_path)
    registry["model_run_config"] = {
        "path": relative(config_path),
        "sha256": file_sha256(config_path),
    }
    write_json(registry_path, registry)
    preregistration_path = (
        ITERATION_ROOT / "protocols/iteration4_preregistration.v1.json"
    )
    preregistration = read_json(preregistration_path)
    preregistration.update(
        {
            "status": "style_model_frozen_before_style_generation",
            "style_model": model,
            "style_model_config_sha256": file_sha256(config_path),
            "method_registry_sha256": file_sha256(registry_path),
        }
    )
    write_json(preregistration_path, preregistration)
    return {
        "status": "style_model_configured",
        "model": model,
        "model_config_sha256": file_sha256(
            config_path
        ),
        "config": config,
    }


def validate() -> dict[str, Any]:
    sample_root = ITERATION_ROOT / "sample_sets"
    summary = read_json(sample_root / f"{SAMPLE_SET}.summary.json")
    allocation = list(
        iter_jsonl(sample_root / f"{SAMPLE_SET}.evaluator_allocation.jsonl")
    )
    errors: list[str] = []
    preregistration = read_json(
        ITERATION_ROOT / "protocols/iteration4_preregistration.v1.json"
    )
    if preregistration.get("sample_set") != SAMPLE_SET:
        errors.append("preregistration_sample_set")
    if len(allocation) != 214:
        errors.append("allocation_count")
    if summary.get("prior_chunk_overlap") != 0:
        errors.append("prior_chunk_overlap")
    if not summary.get("cross_author_partitions_disjoint"):
        errors.append("cross_author_partition_overlap")
    roles = Counter((row["research_role"], row["benchmark_arm"]) for row in allocation)
    if roles[("development_pilot", "own_author_reconstruction")] != 8:
        errors.append("development_own_count")
    if roles[("development_pilot", "cross_author_transfer")] != 8:
        errors.append("development_cross_count")
    if roles[("style_meter_calibration", "own_author_reconstruction")] != 32:
        errors.append("calibration_count")
    if roles[("method_screening", "own_author_reconstruction")] != 24:
        errors.append("screen_own_count")
    if roles[("method_screening", "cross_author_transfer")] != 20:
        errors.append("screen_cross_count")
    if roles[("method_confirmation", "own_author_reconstruction")] != 80:
        errors.append("confirmation_own_count")
    if roles[("method_confirmation", "cross_author_transfer")] != 42:
        errors.append("confirmation_cross_count")
    lock = read_json(ITERATION_ROOT / "method_assets/style_transfer_payloads.v1.lock.json")
    asset = ITERATION_ROOT / str(lock["asset_path"])
    if file_sha256(asset) != lock["file_sha256"]:
        errors.append("method_asset_hash")
    registry = read_json(ITERATION_ROOT / "method_registry/style_methods.v1.json")
    registry_methods = registry.get("methods", [])
    if not isinstance(registry_methods, list) or len(registry_methods) != len(METHODS):
        errors.append("method_registry_count")
    else:
        for row in registry_methods:
            method_id = str(row.get("id", ""))
            config_path = REPO_ROOT / str(row.get("config_path", ""))
            if method_id not in METHODS or not config_path.is_file():
                errors.append(f"method_registry_entry:{method_id}")
                continue
            config = read_json(config_path)
            if config.get("method_id") != method_id:
                errors.append(f"method_config_identity:{method_id}")
            if file_sha256(config_path) != row.get("config_sha256"):
                errors.append(f"method_config_hash:{method_id}")
    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "allocation_rows": len(allocation),
        "role_arm_counts": {
            f"{role}:{arm}": count for (role, arm), count in sorted(roles.items())
        },
        "method_asset_content_sha256": lock["content_sha256"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Iteration 4 artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--model", default="gpt-5.4")
    style_parser = subparsers.add_parser("set-style-model")
    style_parser.add_argument("--model", default="gpt-5.5")
    subparsers.add_parser("validate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "init":
        result = init(args.model)
    elif args.command == "set-style-model":
        result = set_style_model(args.model)
    else:
        result = validate()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("status") == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
