#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


from experiments.shared.paths import RESEARCH_ROOT
from workflows.author_style_meter_contract import CURRENT_SCORER_ID


REPO_ROOT = RESEARCH_ROOT
BASE_ROOT = REPO_ROOT / "generated/style_research/style_transfer_experiments"
ITERATION_ROOT = BASE_ROOT / "iterations/aligned_pairs_v1"
PAIR_SAMPLE_SET = "iteration2_aligned_pair_pool_v1"
PROXY_SAMPLE_SET = "development_proxy_v1"
PAIR_RUN_ID = "aligned_pair_pool_gpt54_v1"
SCREEN_RUN_ID = "iteration2_aligned_gpt55_v1"
TARGET_AUTHOR = "非天夜翔"
PAIR_SEED = 20260712
SCREEN_SEED = 20260713
PAIR_VIEW = "entity_masked_v2"
PAIR_METHODS = (
    "neutral_only",
    "aligned_pairs_only",
    "aligned_pairs_plus_cards",
    "aligned_pairs_edit_plan",
)
GENERATED_METHODS = PAIR_METHODS[1:]
INTENSITY = "medium"
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
SENTENCE_RE = re.compile(r"(?<=[。！？!?])")
CHAPTER_RE = re.compile(r"^(?:第[〇零一二三四五六七八九十百千万两\d]+[章节回卷部]|chapter\b)", re.I)
SCENE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "action_conflict": ("冲", "打", "剑", "刀", "杀", "追", "挡", "攻", "躲", "战"),
    "dialogue": ("问", "答", "说", "道", "喊", "笑道", "低声"),
    "internal_reflection": ("想", "意识", "明白", "记得", "觉得", "心里", "忽然"),
    "interpersonal_care": ("握", "抱", "扶", "看着", "安慰", "担心", "照顾"),
    "travel_transition": ("出发", "前往", "抵达", "离开", "返回", "路上", "走进"),
    "worldbuilding_exposition": ("传说", "历史", "规则", "制度", "王国", "教会", "意味着"),
}


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
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_bytes() == source.read_bytes():
        return
    shutil.copy2(source, destination)


def cjk_count(text: str) -> int:
    return len(CJK_RE.findall(text))


def stable_value(seed: int, value: str) -> str:
    return sha256_text(f"{seed}:{value}")


def position_bucket(index: int, maximum: int) -> str:
    ratio = index / max(maximum, 1)
    if ratio <= 0.10:
        return "opening"
    if ratio <= 0.33:
        return "early"
    if ratio <= 0.67:
        return "middle"
    if ratio <= 0.90:
        return "late"
    return "ending"


def scene_label(text: str) -> str:
    dialogue_lines = sum(
        line.lstrip().startswith(("“", "‘", "「", "『", '"'))
        for line in text.splitlines()
        if line.strip()
    )
    scores = {
        label: sum(text.count(term) for term in terms)
        for label, terms in SCENE_KEYWORDS.items()
    }
    scores["dialogue"] += dialogue_lines * 3 + text.count("“")
    return max(scores, key=lambda label: (scores[label], label))


def clean_paragraphs(text: str) -> list[str]:
    paragraphs: list[str] = []
    for line in text.splitlines():
        value = line.strip()
        if not value or CHAPTER_RE.match(value):
            continue
        if cjk_count(value) == 0 and value not in {"<NUM>", "<TERM>", "<LATIN>"}:
            continue
        paragraphs.append(value)
    return paragraphs


def excerpt_windows(text: str) -> list[str]:
    paragraphs = clean_paragraphs(text)
    windows: list[str] = []
    for start in range(len(paragraphs)):
        selected: list[str] = []
        total = 0
        for paragraph in paragraphs[start:]:
            selected.append(paragraph)
            total += cjk_count(paragraph)
            if total >= 330:
                windows.append("\n".join(selected))
            if total >= 520:
                break
    return [value for value in windows if 280 <= cjk_count(value) <= 620]


def build_pair_candidates() -> dict[str, list[dict[str, Any]]]:
    path = REPO_ROOT / "datasets/masked/chunks.entity_masked_v2.jsonl"
    rows = [
        row
        for row in iter_jsonl(path)
        if row.get("author") == TARGET_AUTHOR and row.get("split") == "train"
    ]
    maximum_by_book: dict[str, int] = defaultdict(int)
    for row in rows:
        maximum_by_book[str(row["title"])] = max(
            maximum_by_book[str(row["title"])], int(row["chunk_index"])
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        windows = excerpt_windows(str(row["text"]))
        if not windows:
            continue
        excerpt = min(
            windows,
            key=lambda value: (
                abs(cjk_count(value) - 450),
                stable_value(PAIR_SEED, f"{row['chunk_id']}:{value[:40]}"),
            ),
        )
        title = str(row["title"])
        grouped[title].append(
            {
                "author": TARGET_AUTHOR,
                "title": title,
                "chunk_id": str(row["chunk_id"]),
                "chunk_index": int(row["chunk_index"]),
                "source_split": "train",
                "view": PAIR_VIEW,
                "text": excerpt,
                "cjk_count": cjk_count(excerpt),
                "scene_type": scene_label(excerpt),
                "position_bucket": position_bucket(
                    int(row["chunk_index"]), maximum_by_book[title]
                ),
                "quality_flags": list(row.get("quality_flags", [])),
            }
        )
    splits = read_json(REPO_ROOT / "generated/style_research/corpus/splits.json")
    expected_titles = {
        str(row["title"])
        for row in splits["train"]
        if row.get("author") == TARGET_AUTHOR
    }
    if set(grouped) != expected_titles:
        raise ValueError(
            "Target train books differ between current splits and masked chunks: "
            f"missing={sorted(expected_titles - set(grouped))}, "
            f"extra={sorted(set(grouped) - expected_titles)}"
        )
    return grouped


def select_pair_rows() -> list[dict[str, Any]]:
    grouped = build_pair_candidates()
    scene_counts: Counter[str] = Counter()
    position_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for title in sorted(grouped, key=lambda value: stable_value(PAIR_SEED, value)):
        candidate = min(
            grouped[title],
            key=lambda row: (
                scene_counts[row["scene_type"]],
                position_counts[row["position_bucket"]],
                bool(row["quality_flags"]),
                abs(row["cjk_count"] - 450),
                stable_value(PAIR_SEED, row["chunk_id"]),
            ),
        )
        scene_counts[candidate["scene_type"]] += 1
        position_counts[candidate["position_bucket"]] += 1
        selected.append(candidate)
    return sorted(selected, key=lambda row: row["title"])


def artifact_paths(sample_set: str, sample_id: str) -> dict[str, str]:
    root = ITERATION_ROOT / "runs" / sample_set / "<run_id>"
    return {
        "english_semantic_source": relative(root / "english_semantic_source" / f"{sample_id}.json"),
        "english_source_qa": relative(root / "english_source_qa" / f"{sample_id}.json"),
        "neutral_translation": relative(root / "neutral_translation" / f"{sample_id}.json"),
        "method_outputs": relative(root / "method_outputs/<method_id>/<intensity>" / f"{sample_id}.json"),
        "evaluation": relative(root / "evaluation/<method_id>/<intensity>" / f"{sample_id}.json"),
    }


def build_pair_sample_set() -> dict[str, Any]:
    selected = select_pair_rows()
    runner: list[dict[str, Any]] = []
    allocation: list[dict[str, Any]] = []
    hidden: list[dict[str, Any]] = []
    for row in selected:
        sample_id = "s_" + sha256_text(
            f"{PAIR_SEED}:{row['chunk_id']}:{row['text']}"
        )[:24]
        runner.append(
            {
                "artifact_paths": artifact_paths(PAIR_SAMPLE_SET, sample_id),
                "sample_id": sample_id,
                "sample_set": PAIR_SAMPLE_SET,
                "stage_status": {
                    "english_semantic_source": "pending",
                    "english_source_qa": "pending",
                    "neutral_translation": "pending",
                    "style_transfer": "not_applicable",
                    "automatic_evaluation": "pending",
                    "independent_evaluation": "pending",
                },
            }
        )
        allocation.append(
            {
                "access_policy": "iteration2_pair_builder_and_evaluator_only",
                "author": TARGET_AUTHOR,
                "book_title": row["title"],
                "chunk_id": row["chunk_id"],
                "chunk_index": row["chunk_index"],
                "research_role": "aligned_pair_training_evidence",
                "sample_id": sample_id,
                "sample_set": PAIR_SAMPLE_SET,
                "source_split": "train",
                "source_view": PAIR_VIEW,
                "strata": {
                    "scene_type": row["scene_type"],
                    "position_bucket": row["position_bucket"],
                },
                "measurements": {"cjk_count": row["cjk_count"]},
                "quality_flags": row["quality_flags"],
            }
        )
        hidden.append(
            {
                "access_policy": "one_way_pair_source_generator_and_evaluator_only",
                "original_zh": row["text"],
                "sample_id": sample_id,
                "sample_set": PAIR_SAMPLE_SET,
                "source_view": PAIR_VIEW,
            }
        )
    paths = ITERATION_ROOT / "sample_sets"
    write_jsonl(paths / f"{PAIR_SAMPLE_SET}.runner_manifest.jsonl", runner)
    write_jsonl(paths / f"{PAIR_SAMPLE_SET}.evaluator_allocation.jsonl", allocation)
    write_jsonl(paths / f"{PAIR_SAMPLE_SET}.hidden_targets.jsonl", hidden)
    ids = sorted(row["sample_id"] for row in runner)
    for suffix, selection_id in (
        ("method_evaluation_ids", "aligned_pair_pool_v1"),
        ("screening_v1_ids", "aligned_pair_pool_v1"),
        ("confirmation_v1_ids", "aligned_pair_pool_v1"),
    ):
        write_json(
            paths / f"{PAIR_SAMPLE_SET}.{suffix}.json",
            {
                "schema_version": 1,
                "selection_id": selection_id,
                "sample_set": PAIR_SAMPLE_SET,
                "sample_count": len(ids),
                "sample_ids": ids,
            },
        )
    summary = {
        "schema_version": 1,
        "sample_set": PAIR_SAMPLE_SET,
        "seed": PAIR_SEED,
        "source_split": "train",
        "source_view": PAIR_VIEW,
        "target_author": TARGET_AUTHOR,
        "total_samples": len(ids),
        "books": len({row["book_title"] for row in allocation}),
        "one_excerpt_per_target_train_book": True,
        "scene_counts": dict(sorted(Counter(row["strata"]["scene_type"] for row in allocation).items())),
        "position_counts": dict(sorted(Counter(row["strata"]["position_bucket"] for row in allocation).items())),
        "runner_manifest_sha256": rows_sha256(runner),
        "evaluator_allocation_sha256": rows_sha256(allocation),
        "hidden_targets_sha256": rows_sha256(hidden),
        "content_control": {
            "names_numbers_and_concentrated_terms_masked": True,
            "clean_target_text_exposed_to_pair_generator": False,
            "development_proxy_and_test_books_excluded": True,
        },
    }
    write_json(paths / f"{PAIR_SAMPLE_SET}.summary.json", summary)
    return summary


def choose_group_rows(
    rows: Sequence[dict[str, Any]],
    count: int,
    scene_counts: Counter[str],
    position_counts: Counter[str],
) -> list[dict[str, Any]]:
    remaining = list(rows)
    chosen: list[dict[str, Any]] = []
    while remaining and len(chosen) < count:
        row = min(
            remaining,
            key=lambda value: (
                scene_counts[value["strata"]["scene_type"]],
                position_counts[value["strata"]["position_bucket"]],
                stable_value(SCREEN_SEED, value["sample_id"]),
            ),
        )
        remaining.remove(row)
        chosen.append(row)
        scene_counts[row["strata"]["scene_type"]] += 1
        position_counts[row["strata"]["position_bucket"]] += 1
    if len(chosen) != count:
        raise ValueError(f"Cannot select {count} rows from group of {len(rows)}")
    return chosen


def derive_iteration2_proxy_selections() -> dict[str, Any]:
    base_sample = BASE_ROOT / "sample_sets"
    runner = list(iter_jsonl(base_sample / f"{PROXY_SAMPLE_SET}.runner_manifest.jsonl"))
    allocation = list(iter_jsonl(base_sample / f"{PROXY_SAMPLE_SET}.evaluator_allocation.jsonl"))
    hidden = list(iter_jsonl(base_sample / f"{PROXY_SAMPLE_SET}.hidden_targets.jsonl"))
    prior_screen_ids = set(
        read_json(base_sample / f"{PROXY_SAMPLE_SET}.screening_v1_ids.json")["sample_ids"]
    )
    allocation = [
        {
            **row,
            "research_role": (
                "prior_iteration_screening"
                if row["sample_id"] in prior_screen_ids
                else row["research_role"]
            ),
        }
        for row in allocation
    ]
    original_confirmation = read_json(
        base_sample / f"{PROXY_SAMPLE_SET}.confirmation_v1_ids.json"
    )["sample_ids"]
    by_id = {row["sample_id"]: row for row in allocation}
    candidates = [by_id[sample_id] for sample_id in original_confirmation]
    own_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cross_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        if row["benchmark_arm"] == "own_author_reconstruction":
            own_groups[row["book_title"]].append(row)
        else:
            cross_groups[row["author"]].append(row)
    if len(own_groups) != 8 or len(cross_groups) != 12:
        raise ValueError("Fresh screen requires eight target books and 12 comparison authors")
    scene_counts: Counter[str] = Counter()
    position_counts: Counter[str] = Counter()
    selected_rows: list[dict[str, Any]] = []
    for title in sorted(own_groups, key=lambda value: stable_value(SCREEN_SEED, value)):
        selected_rows.extend(
            choose_group_rows(own_groups[title], 3, scene_counts, position_counts)
        )
    for author in sorted(cross_groups, key=lambda value: stable_value(SCREEN_SEED, value)):
        selected_rows.extend(
            choose_group_rows(cross_groups[author], 1, scene_counts, position_counts)
        )
    screen_ids = sorted(row["sample_id"] for row in selected_rows)
    confirmation_ids = sorted(set(original_confirmation) - set(screen_ids))
    own_screen = sum(by_id[value]["benchmark_arm"] == "own_author_reconstruction" for value in screen_ids)
    own_confirmation = sum(by_id[value]["benchmark_arm"] == "own_author_reconstruction" for value in confirmation_ids)
    if (len(screen_ids), own_screen, len(confirmation_ids), own_confirmation) != (36, 24, 116, 80):
        raise ValueError("Iteration-2 split counts do not match 24/12 screen and 80/36 confirmation")

    paths = ITERATION_ROOT / "sample_sets"
    rewritten_runner: list[dict[str, Any]] = []
    for row in runner:
        sample_id = row["sample_id"]
        rewritten_runner.append(
            {
                **row,
                "artifact_paths": artifact_paths(PROXY_SAMPLE_SET, sample_id),
            }
        )
    write_jsonl(paths / f"{PROXY_SAMPLE_SET}.runner_manifest.jsonl", rewritten_runner)
    write_jsonl(paths / f"{PROXY_SAMPLE_SET}.evaluator_allocation.jsonl", allocation)
    write_jsonl(paths / f"{PROXY_SAMPLE_SET}.hidden_targets.jsonl", hidden)
    write_json(
        paths / f"{PROXY_SAMPLE_SET}.screening_v1_ids.json",
        {
            "schema_version": 1,
            "selection_id": "screening_v1",
            "sample_set": PROXY_SAMPLE_SET,
            "seed": SCREEN_SEED,
            "design": {
                "source": "iteration1_confirmation_v1",
                "iteration1_screening_overlap": 0,
                "own_author_chunks_per_book": 3,
                "cross_author_chunks_per_author": 1,
                "purpose": "iteration2_aligned_pair_method_screening",
            },
            "sample_count": len(screen_ids),
            "sample_ids": screen_ids,
        },
    )
    write_json(
        paths / f"{PROXY_SAMPLE_SET}.confirmation_v1_ids.json",
        {
            "schema_version": 1,
            "selection_id": "confirmation_v1",
            "sample_set": PROXY_SAMPLE_SET,
            "excluded_selection_id": "screening_v1",
            "purpose": "iteration2_post_screen_confirmation",
            "sample_count": len(confirmation_ids),
            "own_author_rows": own_confirmation,
            "cross_author_rows": len(confirmation_ids) - own_confirmation,
            "sample_ids": confirmation_ids,
        },
    )
    write_json(
        paths / f"{PROXY_SAMPLE_SET}.method_evaluation_ids.json",
        {
            "schema_version": 1,
            "sample_set": PROXY_SAMPLE_SET,
            "allowed_research_roles": ["method_evaluation", "cross_author_method_evaluation"],
            "sample_count": len(original_confirmation),
            "sample_ids": sorted(original_confirmation),
        },
    )
    calibration_ids = sorted(
        row["sample_id"] for row in allocation if row["research_role"] == "style_meter_calibration"
    )
    write_json(
        paths / f"{PROXY_SAMPLE_SET}.screening_calibration_v1_ids.json",
        {
            "schema_version": 1,
            "selection_id": "screening_calibration_v1",
            "sample_set": PROXY_SAMPLE_SET,
            "purpose": "reuse_frozen_calibration_and_generate_fresh_iteration2_screen",
            "sample_count": len(calibration_ids) + len(screen_ids),
            "composition": {"style_meter_calibration": len(calibration_ids), "method_screening": len(screen_ids)},
            "sample_ids": sorted(calibration_ids + screen_ids),
        },
    )
    base_summary = read_json(base_sample / f"{PROXY_SAMPLE_SET}.summary.json")
    output_paths = {
        "runner_manifest": paths / f"{PROXY_SAMPLE_SET}.runner_manifest.jsonl",
        "evaluator_allocation": paths / f"{PROXY_SAMPLE_SET}.evaluator_allocation.jsonl",
        "hidden_targets": paths / f"{PROXY_SAMPLE_SET}.hidden_targets.jsonl",
        "method_evaluation_ids": paths / f"{PROXY_SAMPLE_SET}.method_evaluation_ids.json",
        "screening_ids": paths / f"{PROXY_SAMPLE_SET}.screening_v1_ids.json",
        "confirmation_ids": paths / f"{PROXY_SAMPLE_SET}.confirmation_v1_ids.json",
        "summary": paths / f"{PROXY_SAMPLE_SET}.summary.json",
    }
    base_summary.update(
        {
            "schema_version": 3,
            "iteration": 2,
            "iteration2_seed": SCREEN_SEED,
            "method_evaluation_samples": len(original_confirmation),
            "cross_author_method_evaluation_samples": 48,
            "style_meter_calibration_samples": len(calibration_ids),
            "prior_iteration_screening_samples": len(prior_screen_ids),
            "runner_manifest_sha256": rows_sha256(rewritten_runner),
            "evaluator_allocation_sha256": rows_sha256(allocation),
            "hidden_targets_sha256": rows_sha256(hidden),
            "method_evaluation_ids_sha256": file_sha256(output_paths["method_evaluation_ids"]),
            "screening_ids_sha256": file_sha256(output_paths["screening_ids"]),
            "confirmation_ids_sha256": file_sha256(output_paths["confirmation_ids"]),
            "outputs": {key: relative(path) for key, path in output_paths.items()},
            "counts": {
                **base_summary["counts"],
                "by_research_role": dict(
                    sorted(Counter(row["research_role"] for row in allocation).items())
                ),
            },
            "iteration2_screening_samples": len(screen_ids),
            "iteration2_confirmation_samples": len(confirmation_ids),
            "iteration1_screening_overlap": 0,
        }
    )
    write_json(paths / f"{PROXY_SAMPLE_SET}.summary.json", base_summary)
    return {
        "screening_rows": len(screen_ids),
        "screening_own": own_screen,
        "confirmation_rows": len(confirmation_ids),
        "confirmation_own": own_confirmation,
        "screening_ids_sha256": file_sha256(paths / f"{PROXY_SAMPLE_SET}.screening_v1_ids.json"),
        "confirmation_ids_sha256": file_sha256(paths / f"{PROXY_SAMPLE_SET}.confirmation_v1_ids.json"),
    }


def copy_base_runtime_artifacts() -> None:
    for directory in ("prompts", "schemas"):
        for source in (BASE_ROOT / directory).glob("*.v1.*"):
            copy_file(source, ITERATION_ROOT / directory / source.name)
    scorer = CURRENT_SCORER_ID
    source_scorer = BASE_ROOT / "scorers" / scorer
    destination_scorer = ITERATION_ROOT / "scorers" / scorer
    destination_scorer.mkdir(parents=True, exist_ok=True)
    for source in source_scorer.iterdir():
        if source.is_file():
            copy_file(source, destination_scorer / source.name)
    for filename in ("style_meter_scores.v1.jsonl", "style_meter_threshold.v1.json"):
        copy_file(BASE_ROOT / "calibration" / filename, ITERATION_ROOT / "calibration" / filename)


def write_model_config(model: str, phase: str) -> dict[str, Any]:
    config = read_json(BASE_ROOT / "protocols/model_run_config.v1.json")
    config.update(
        {
            "codex_model": model,
            "reasoning_effort": "high",
            "default_parallel_jobs": 4,
            "experiment_iteration": 2,
            "experiment_phase": phase,
            "model_selection": {
                "requested_order": ["gpt-5.6", "gpt-5.5", "gpt-5.4"],
                "gpt-5.6_probe": "unsupported_with_chatgpt_account",
                "gpt-5.5_probe": "available",
                "neutral_reconstruction_model": "gpt-5.4",
                "style_transfer_model": "gpt-5.5",
                "no_model_mixing_within_style_arms": True,
            },
        }
    )
    write_json(ITERATION_ROOT / "protocols/model_run_config.v1.json", config)
    return config


def source_bindings() -> dict[str, str]:
    paths = (
        REPO_ROOT / "experiments/iteration2/prepare.py",
        REPO_ROOT / "experiments/iteration2/style_transfer_payloads.py",
        REPO_ROOT / "experiments/iteration2/run_style_transfer_generation.py",
        REPO_ROOT / "experiments/iteration2/evaluate_style_transfer_methods.py",
    )
    return {relative(path): file_sha256(path) for path in paths}


def init_iteration() -> dict[str, Any]:
    if list((ITERATION_ROOT / "runs").glob("**/method_outputs/**/*.json")):
        raise ValueError("Cannot reinitialize iteration 2 after style outputs exist")
    copy_base_runtime_artifacts()
    pair_summary = build_pair_sample_set()
    proxy_summary = derive_iteration2_proxy_selections()
    model_config = write_model_config("gpt-5.4", "neutral_and_pair_reconstruction")
    preregistration = {
        "schema_version": 1,
        "status": "pair_and_proxy_inputs_frozen_before_generation",
        "iteration_id": "aligned_pairs_v1",
        "research_question": "Do aligned neutral-to-target demonstrations create materially stronger author-style transfer than unaligned examples while preserving English meaning?",
        "prior_iteration": {
            "status": "no_method_qualified",
            "screening_overlap": 0,
        },
        "pair_pool": {
            "sample_set": PAIR_SAMPLE_SET,
            "run_id": PAIR_RUN_ID,
            "rows": pair_summary["total_samples"],
            "books": pair_summary["books"],
            "source_split": "train",
            "source_view": PAIR_VIEW,
            "one_excerpt_per_book": True,
        },
        "proxy_screen": {
            "sample_set": PROXY_SAMPLE_SET,
            "run_id": SCREEN_RUN_ID,
            **proxy_summary,
        },
        "planned_methods": list(PAIR_METHODS),
        "style_model": "gpt-5.5",
        "neutral_model": "gpt-5.4",
        "screening_gates": {
            "positive_mean_lift_in_both_arms": True,
            "hard_fidelity_failure_rate_max": 0.10,
            "reference_copy_failures_max": 0,
            "style_success_threshold_reused": True,
        },
        "source_bindings": source_bindings(),
        "model_config_sha256": file_sha256(ITERATION_ROOT / "protocols/model_run_config.v1.json"),
    }
    write_json(ITERATION_ROOT / "protocols/iteration2_preregistration.v1.json", preregistration)
    return {
        "status": "initialized",
        "experiment_root": relative(ITERATION_ROOT),
        "pair_summary": pair_summary,
        "proxy_selection": proxy_summary,
        "model": model_config["codex_model"],
    }


def split_paragraphs(text: str, field: str) -> list[dict[str, str]]:
    return [
        {"id": f"p{index:04d}", field: line.strip()}
        for index, line in enumerate((line for line in text.splitlines() if line.strip()), start=1)
    ]


def qa_approved(sample_set: str, run_id: str, sample_id: str) -> bool:
    run_root = ITERATION_ROOT / "runs" / sample_set / run_id
    initial = run_root / "english_source_qa" / f"{sample_id}.json"
    if initial.exists() and read_json(initial).get("approved_for_neutral_translation"):
        return True
    for round_number in range(1, 21):
        path = run_root / "english_source_repair_qa" / f"round_{round_number:02d}" / f"{sample_id}.json"
        repair = run_root / "english_source_repair" / f"round_{round_number:02d}" / f"{sample_id}.json"
        if not path.exists() and not repair.exists():
            break
        if path.exists() and repair.exists() and read_json(path).get("approved_for_neutral_translation"):
            return True
    return False


def require_neutral_outputs(sample_set: str, run_id: str, sample_ids: Sequence[str]) -> None:
    root = ITERATION_ROOT / "runs" / sample_set / run_id
    missing: list[str] = []
    unapproved: list[str] = []
    for sample_id in sample_ids:
        if not qa_approved(sample_set, run_id, sample_id):
            unapproved.append(sample_id)
        if not (root / "neutral_translation" / f"{sample_id}.json").exists():
            missing.append(sample_id)
    if missing or unapproved:
        raise ValueError(
            f"Neutral prerequisites incomplete for {sample_set}: missing={len(missing)}, unapproved={len(unapproved)}"
        )


def load_base_cards() -> list[dict[str, Any]]:
    lock = read_json(BASE_ROOT / "method_assets/style_transfer_payloads.v1.lock.json")
    bundle = read_json(BASE_ROOT / lock["asset_path"])
    cards = bundle["assets"]["style_cards"].get("global_style_cards")
    if not isinstance(cards, list) or not cards:
        raise ValueError("Base global style cards are missing")
    return cards


def build_aligned_pairs() -> list[dict[str, Any]]:
    sample_root = ITERATION_ROOT / "sample_sets"
    hidden = {row["sample_id"]: row for row in iter_jsonl(sample_root / f"{PAIR_SAMPLE_SET}.hidden_targets.jsonl")}
    allocation = {row["sample_id"]: row for row in iter_jsonl(sample_root / f"{PAIR_SAMPLE_SET}.evaluator_allocation.jsonl")}
    ids = sorted(hidden)
    require_neutral_outputs(PAIR_SAMPLE_SET, PAIR_RUN_ID, ids)
    run_root = ITERATION_ROOT / "runs" / PAIR_SAMPLE_SET / PAIR_RUN_ID
    pairs: list[dict[str, Any]] = []
    for sample_id in ids:
        neutral_artifact = read_json(run_root / "neutral_translation" / f"{sample_id}.json")
        target = split_paragraphs(hidden[sample_id]["original_zh"], "zh")
        neutral = neutral_artifact["result"]["paragraphs"]
        if [row["id"] for row in target] != [row["id"] for row in neutral]:
            raise ValueError(f"Pair paragraph mismatch: {sample_id}")
        metadata = allocation[sample_id]
        pair = {
            "pair_id": "pair_" + sha256_text(sample_id)[:20],
            "source_sample_id": sample_id,
            "source_book_hash": sha256_text(metadata["book_title"])[:16],
            "source_chunk_hash": sha256_text(metadata["chunk_id"])[:16],
            "source_split": "train",
            "source_view": PAIR_VIEW,
            "scene_type": metadata["strata"]["scene_type"],
            "position_bucket": metadata["strata"]["position_bucket"],
            "neutral_zh": neutral,
            "target_style_zh": target,
        }
        pair["pair_sha256"] = sha256_json(pair)
        pairs.append(pair)
    return pairs


def intensity_contract(label: str) -> dict[str, Any]:
    return {
        "label": label,
        "rewrite_scope": "Apply only recurring neutral-to-target transformations demonstrated by at least two retrieved pairs.",
        "semantic_priority": "English remains authoritative; skip any transformation that changes facts, roles, chronology, negation, modality, or intensity.",
        "paragraph_contract": "Preserve paragraph IDs and order exactly.",
    }


def method_assets(pairs: list[dict[str, Any]], cards: list[dict[str, Any]]) -> dict[str, Any]:
    common = {
        "availability": "ready",
        "intensity_contract": intensity_contract("medium"),
        "evidence_source": {
            "pair_pool": "29 entity-masked train excerpts, one per target-author train book",
            "retrieval_query": "neutral_zh_only",
            "k": 3,
            "book_diversity": "at_most_one_pair_per_book",
        },
    }
    methods = {
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
        "aligned_pairs_only": {
            "intensities": {
                INTENSITY: {
                    **common,
                    "instructions": [
                        "Infer only the recurring structural transformation shown by the aligned neutral and target-style versions.",
                        "Apply those transformations where the current neutral draft provides a source-supported opportunity.",
                        "Do not copy content or wording from demonstrations.",
                    ],
                }
            }
        },
        "aligned_pairs_plus_cards": {
            "intensities": {
                INTENSITY: {
                    **common,
                    "instructions": [
                        "Use the aligned pairs as primary transformation evidence.",
                        "Use compact corpus cards only when they agree with the transformations visible in the pairs.",
                    ],
                    "global_style_cards": cards,
                }
            }
        },
        "aligned_pairs_edit_plan": {
            "intensities": {
                INTENSITY: {
                    **common,
                    "instructions": [
                        "Internally compare each aligned pair and identify repeated edits in clause order, dialogue timing, sentence boundaries, function words, and reaction-beat placement.",
                        "Apply only edits supported by at least two pairs; output prose only, not the plan.",
                    ],
                    "edit_plan_contract": {
                        "minimum_pair_support": 2,
                        "allowed_dimensions": [
                            "clause_order",
                            "sentence_boundaries",
                            "dialogue_timing",
                            "function_words",
                            "punctuation",
                            "reaction_beat_placement",
                        ],
                        "forbidden": [
                            "new_event",
                            "new_emotion",
                            "new_motive",
                            "new_lore",
                            "new_imagery",
                            "reference_phrase_copying",
                        ],
                    },
                }
            }
        },
    }
    return methods


def build_asset_bundle(pairs: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    cards = load_base_cards()
    methods = method_assets(pairs, cards)
    prompt_hash = file_sha256(ITERATION_ROOT / "prompts/style_transfer_method.v1.md")
    content: dict[str, Any] = {
        "schema_version": 1,
        "target_author": TARGET_AUTHOR,
        "generator": {
            "id": "experiments.iteration2.style_transfer_payloads",
            "version": 1,
            "source_sha256": file_sha256(REPO_ROOT / "experiments/iteration2/style_transfer_payloads.py"),
        },
        "registry": {"iteration_id": "aligned_pairs_v1", "method_ids": list(PAIR_METHODS)},
        "source_projections": {"style_prompt_sha256": prompt_hash},
        "evidence_policy": {
            "target_train_book_count": 29,
            "comparison_author_count": 49,
            "pair_source_split": "train",
            "pair_source_view": PAIR_VIEW,
            "development_proxy_and_test_books_excluded": True,
        },
        "statistics": {},
        "style_cards": {"global": cards},
        "retrieval": {"target_index": [], "hard_negative_index": []},
        "close_reading": {"asset": {}, "source_packet": {"sha256": None}},
        "aligned_pairs": {
            "schema_version": 1,
            "count": len(pairs),
            "books": len({pair["source_book_hash"] for pair in pairs}),
            "pairs": pairs,
        },
        "methods": methods,
    }
    component_hashes = {
        "train_statistics_sha256": sha256_json(content["statistics"]),
        "style_cards_sha256": sha256_json(content["style_cards"]),
        "target_retrieval_index_sha256": sha256_json(content["retrieval"]["target_index"]),
        "hard_negative_retrieval_index_sha256": sha256_json(content["retrieval"]["hard_negative_index"]),
        "close_reading_asset_sha256": sha256_json(content["close_reading"]["asset"]),
        "close_reading_source_packet_sha256": None,
        "methods_sha256": sha256_json(content["methods"]),
        "aligned_pairs_sha256": sha256_json(content["aligned_pairs"]),
        "method_intensity_asset_sha256": {
            f"{method_id}:{intensity}": sha256_json(asset)
            for method_id, method in methods.items()
            for intensity, asset in method["intensities"].items()
        },
    }
    content["component_hashes"] = component_hashes
    content_sha = sha256_json(content)
    bundle = {
        "schema_version": "style_transfer_payload_assets.v1",
        "content_sha256": content_sha,
        "assets": content,
    }
    asset_directory = ITERATION_ROOT / "method_assets/style_transfer_payloads.v1"
    asset_path = asset_directory / f"assets.{content_sha}.json"
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
    lock: dict[str, Any],
    bundle: dict[str, Any],
) -> Path:
    hashes = bundle["assets"]["component_hashes"]["method_intensity_asset_sha256"]
    config = {
        "schema_version": 1,
        "method_id": method_id,
        "label": label,
        "family": family,
        "status": "ready",
        "hypothesis": hypothesis,
        "intensities": intensities,
        "prompt_version": "style_transfer_method.v1",
        "model_config": "protocols/model_run_config.v1.json",
        "input_contract": {
            "english_and_neutral_inputs_must_be_byte_identical_across_methods": True,
            "target_derived_allocation_metadata_available_to_runner": False,
            "original_evaluation_chinese_available_to_runner": False,
        },
        "payload_builder": {
            "id": f"{method_id}.iteration2_payload.v1",
            "source_corpus": "target_author_train_books_only",
            "development_proxy_and_test_books_excluded": True,
        },
        "aligned_pair_retrieval": {
            "enabled": method_id != "neutral_only",
            "query": "neutral_zh",
            "source_split": "train",
            "source_view": PAIR_VIEW,
            "k": 3 if method_id != "neutral_only" else 0,
            "max_pairs_per_book": 1,
        },
        "generation": {
            "one_output_per_sample_and_intensity": True,
            "max_generation_attempts": 2,
            "output_schema": "schemas/style_transfer_output.v1.schema.json",
        },
        "inputs": inputs,
        "asset_manifest": {
            "path": relative(ITERATION_ROOT / lock["asset_path"]),
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


def write_protocol_and_registry(bundle: dict[str, Any], lock: dict[str, Any]) -> dict[str, Any]:
    screening = read_json(ITERATION_ROOT / "sample_sets/development_proxy_v1.screening_v1_ids.json")
    confirmation = read_json(ITERATION_ROOT / "sample_sets/development_proxy_v1.confirmation_v1_ids.json")
    protocol = read_json(BASE_ROOT / "protocols/evaluation_protocol.v1.json")
    protocol["schema_version"] = 2
    protocol["iteration2"] = {
        "iteration_id": "aligned_pairs_v1",
        "prior_iteration_status": "no_method_qualified",
        "prior_screening_overlap": 0,
        "pair_asset_content_sha256": lock["content_sha256"],
        "pair_count": bundle["assets"]["aligned_pairs"]["count"],
        "pair_book_count": bundle["assets"]["aligned_pairs"]["books"],
        "source_bindings": source_bindings(),
        "style_model": "gpt-5.5",
        "neutral_model": "gpt-5.4",
        "registered_family_rerank": {
            "status": "diagnostic_only_until_base_outputs_exist",
            "candidate_methods": list(GENERATED_METHODS),
            "selection_order": [
                "exclude_hard_fidelity_failures",
                "maximize_frozen_target_margin",
                "maximize_paired_lift",
                "method_id_lexical_tiebreak",
            ],
        },
    }
    protocol["staged_execution"]["screening"].update(
        {
            "selection_file": "sample_sets/development_proxy_v1.screening_v1_ids.json",
            "sample_count": screening["sample_count"],
            "composition": "Fresh iteration-2 screen: three rows from each of eight target books and one row from each of 12 cross-author books; zero overlap with iteration-1 screening.",
            "initial_method_intensities": {
                "neutral_only": "none",
                "aligned_pairs_only": INTENSITY,
                "aligned_pairs_plus_cards": INTENSITY,
                "aligned_pairs_edit_plan": INTENSITY,
            },
            "promotion_rule": "Retain at most three non-control methods with positive paired target-margin lift in both arms, deterministic hard-fidelity failure rate <=0.10 in each arm, and no reference-copy failure. Break ties by own-author mean lift, then cross-author mean lift.",
            "iteration1_overlap": 0,
        }
    )
    protocol["staged_execution"]["confirmation"].update(
        {
            "selection_file": "sample_sets/development_proxy_v1.confirmation_v1_ids.json",
            "sample_count": confirmation["sample_count"],
            "own_author_rows": confirmation["own_author_rows"],
            "cross_author_rows": confirmation["cross_author_rows"],
            "screening_overlap": 0,
        }
    )
    protocol["primary_endpoints"]["own_author_reconstruction"] = (
        "At least 0.80 style-success point estimate over 80 confirmation rows, at least 0.70 in every development book, Wilson 95% lower bound >=0.70, book-cluster bootstrap lower bound >=0.70, and paired book-cluster bootstrap 95% confidence interval for margin lift excluding zero."
    )
    protocol["primary_endpoints"]["cross_author_transfer"] = (
        "Report separately over 36 confirmation rows; at least 0.80 style-success point estimate, Wilson 95% lower bound >=0.70, positive paired lift in at least 10 of 12 authors, author-cluster bootstrap lower bound >=0.70, and no pooling with own-author rows."
    )
    protocol_path = ITERATION_ROOT / "protocols/evaluation_protocol.v1.json"
    write_json(protocol_path, protocol)

    method_specs = (
        ("neutral_only", "control", "Neutral only", "Negative control for paired margin lift.", ["english_semantic_source", "neutral_zh"], ["none"]),
        ("aligned_pairs_only", "pseudo_parallel", "Aligned pairs only", "Aligned neutral-to-target examples teach transformations that unaligned prose examples do not.", ["english_semantic_source", "neutral_zh", "aligned_target_pairs"], [INTENSITY]),
        ("aligned_pairs_plus_cards", "pseudo_parallel_hybrid", "Aligned pairs plus cards", "Corpus cards improve pair generalization when both evidence sources agree.", ["english_semantic_source", "neutral_zh", "aligned_target_pairs", "global_style_cards"], [INTENSITY]),
        ("aligned_pairs_edit_plan", "pseudo_parallel_reasoned", "Aligned-pair edit plan", "An explicit repeated-edit inference contract improves application of paired demonstrations.", ["english_semantic_source", "neutral_zh", "aligned_target_pairs", "edit_plan_contract"], [INTENSITY]),
    )
    registry_rows: list[dict[str, Any]] = []
    for priority, spec in enumerate(method_specs):
        method_id, family, label, hypothesis, inputs, intensities = spec
        config_path = write_method_config(
            method_id, family, label, hypothesis, inputs, intensities, lock, bundle
        )
        registry_rows.append(
            {
                "id": method_id,
                "family": family,
                "label": label,
                "status": "ready",
                "priority": priority,
                "intensities": intensities,
                "hypothesis": hypothesis,
                "inputs": inputs,
                "config_path": relative(config_path),
                "config_sha256": file_sha256(config_path),
                "asset_manifest": read_json(config_path)["asset_manifest"],
            }
        )
    model_path = ITERATION_ROOT / "protocols/model_run_config.v1.json"
    registry = {
        "schema_version": 3,
        "iteration_id": "aligned_pairs_v1",
        "target_author": TARGET_AUTHOR,
        "model_run_config": {"path": relative(model_path), "sha256": file_sha256(model_path)},
        "evaluation_protocol": {"path": relative(protocol_path), "sha256": file_sha256(protocol_path)},
        "methods": registry_rows,
        "asset_manifest": {
            "path": relative(ITERATION_ROOT / lock["asset_path"]),
            "sha256": lock["content_sha256"],
            "file_sha256": lock["file_sha256"],
            "all_method_assets_verified": True,
        },
        "iteration_rule": {
            "shortlist_requires": [
                "positive_style_lift_both_arms",
                "hard_fidelity_rate_at_most_0_10_both_arms",
                "no_reference_copying",
                "independent_semantic_and_readability_evaluation",
            ],
            "stop_condition": "At least 80% judged style success with registered uncertainty bounds and final replication.",
        },
    }
    write_json(ITERATION_ROOT / "method_registry/style_methods.v1.json", registry)
    return protocol


def freeze_style_assets() -> dict[str, Any]:
    screen_ids = read_json(
        ITERATION_ROOT / "sample_sets/development_proxy_v1.screening_v1_ids.json"
    )["sample_ids"]
    require_neutral_outputs(PROXY_SAMPLE_SET, SCREEN_RUN_ID, screen_ids)
    pairs = build_aligned_pairs()
    model_config = write_model_config("gpt-5.5", "style_transfer_iteration2")
    bundle, lock = build_asset_bundle(pairs)
    protocol = write_protocol_and_registry(bundle, lock)
    preregistration_path = ITERATION_ROOT / "protocols/iteration2_preregistration.v1.json"
    preregistration = read_json(preregistration_path)
    preregistration.update(
        {
            "status": "analysis_ready_for_pre_generation_audit",
            "pair_asset_content_sha256": lock["content_sha256"],
            "style_model_config_sha256": file_sha256(ITERATION_ROOT / "protocols/model_run_config.v1.json"),
            "evaluation_protocol_sha256": file_sha256(ITERATION_ROOT / "protocols/evaluation_protocol.v1.json"),
            "method_registry_sha256": file_sha256(ITERATION_ROOT / "method_registry/style_methods.v1.json"),
            "source_bindings": source_bindings(),
        }
    )
    write_json(preregistration_path, preregistration)
    return {
        "status": "style_assets_frozen",
        "pair_count": len(pairs),
        "pair_books": len({pair["source_book_hash"] for pair in pairs}),
        "method_count": len(protocol["staged_execution"]["screening"]["initial_method_intensities"]),
        "style_model": model_config["codex_model"],
        "asset_content_sha256": lock["content_sha256"],
        "next_command": f"uv run python experiments/iteration1/build_style_analysis_lock.py --experiment-root {relative(ITERATION_ROOT)}",
    }


def validate_iteration() -> dict[str, Any]:
    errors: list[str] = []
    sample_root = ITERATION_ROOT / "sample_sets"
    pair_summary = read_json(sample_root / f"{PAIR_SAMPLE_SET}.summary.json")
    screen = read_json(sample_root / f"{PROXY_SAMPLE_SET}.screening_v1_ids.json")
    confirmation = read_json(sample_root / f"{PROXY_SAMPLE_SET}.confirmation_v1_ids.json")
    if pair_summary.get("total_samples") != 29 or pair_summary.get("books") != 29:
        errors.append("pair_pool_not_29_books")
    if screen.get("sample_count") != 36:
        errors.append("screening_count_not_36")
    if confirmation.get("sample_count") != 116:
        errors.append("confirmation_count_not_116")
    if set(screen.get("sample_ids", [])) & set(confirmation.get("sample_ids", [])):
        errors.append("screening_confirmation_overlap")
    proxy_summary = read_json(sample_root / f"{PROXY_SAMPLE_SET}.summary.json")
    bound_files = {
        "runner_manifest_sha256": sample_root / f"{PROXY_SAMPLE_SET}.runner_manifest.jsonl",
        "evaluator_allocation_sha256": sample_root / f"{PROXY_SAMPLE_SET}.evaluator_allocation.jsonl",
        "hidden_targets_sha256": sample_root / f"{PROXY_SAMPLE_SET}.hidden_targets.jsonl",
        "method_evaluation_ids_sha256": sample_root / f"{PROXY_SAMPLE_SET}.method_evaluation_ids.json",
        "screening_ids_sha256": sample_root / f"{PROXY_SAMPLE_SET}.screening_v1_ids.json",
        "confirmation_ids_sha256": sample_root / f"{PROXY_SAMPLE_SET}.confirmation_v1_ids.json",
    }
    for key, path in bound_files.items():
        digest = rows_sha256(list(iter_jsonl(path))) if path.suffix == ".jsonl" else file_sha256(path)
        if proxy_summary.get(key) != digest:
            errors.append(f"proxy_summary_binding_mismatch:{key}")
    allocation = {
        row["sample_id"]: row
        for row in iter_jsonl(sample_root / f"{PROXY_SAMPLE_SET}.evaluator_allocation.jsonl")
    }
    method_ids_payload = read_json(sample_root / f"{PROXY_SAMPLE_SET}.method_evaluation_ids.json")
    method_ids = set(method_ids_payload.get("sample_ids", []))
    allowed_roles = set(method_ids_payload.get("allowed_research_roles", []))
    expected_method_ids = {
        sample_id
        for sample_id, row in allocation.items()
        if row.get("research_role") in allowed_roles
    }
    prior_screen_ids = {
        sample_id
        for sample_id, row in allocation.items()
        if row.get("research_role") == "prior_iteration_screening"
    }
    calibration_ids = {
        sample_id
        for sample_id, row in allocation.items()
        if row.get("research_role") == "style_meter_calibration"
    }
    if method_ids != expected_method_ids:
        errors.append("method_ids_role_partition_mismatch")
    if len(prior_screen_ids) != 36 or prior_screen_ids & method_ids:
        errors.append("prior_iteration_screening_partition_mismatch")
    if method_ids | prior_screen_ids | calibration_ids != set(allocation):
        errors.append("proxy_role_partition_incomplete")
    if set(screen.get("sample_ids", [])) | set(confirmation.get("sample_ids", [])) != method_ids:
        errors.append("iteration2_selections_do_not_equal_method_ids")
    if (ITERATION_ROOT / "method_registry/style_methods.v1.json").exists():
        protocol = read_json(ITERATION_ROOT / "protocols/evaluation_protocol.v1.json")
        for path_text, expected in protocol.get("iteration2", {}).get("source_bindings", {}).items():
            path = REPO_ROOT / path_text
            if not path.exists() or file_sha256(path) != expected:
                errors.append(f"source_binding_mismatch:{path_text}")
        registry = read_json(ITERATION_ROOT / "method_registry/style_methods.v1.json")
        if len(registry.get("methods", [])) != 4:
            errors.append("method_registry_count_not_4")
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "experiment_root": relative(ITERATION_ROOT),
        "pair_samples": pair_summary.get("total_samples"),
        "screening_samples": screen.get("sample_count"),
        "confirmation_samples": confirmation.get("sample_count"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and freeze iteration-2 aligned-pair style-transfer artifacts."
    )
    parser.add_argument("command", choices=("init", "freeze", "validate"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "init":
        result = init_iteration()
    elif args.command == "freeze":
        result = freeze_style_assets()
    else:
        result = validate_iteration()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
