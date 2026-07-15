#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy
import sklearn

from experiments.iteration5.construct.construct_v2_protocol import (
    deterministic_blocks,
    han_ngram_overlap,
    han_text,
    outcome_protocol,
)


SEED = 20260716
TARGET_AUTHOR = "非天夜翔"
ROOT = Path(
    "generated/style_research/style_transfer_experiments/iterations/"
    "content_resistant_v1/cr_fysm_v4/construct_validation_v2"
)
PREREGISTRATION = ROOT / "generation_preregistration.v2.json"
SELECTION = ROOT / "source_selection.private.jsonl"
ACTIVE_CLEAN = Path("datasets/unmasked/chunks.clean.jsonl")
ACTIVE_MASKED = Path("datasets/masked/chunks.entity_masked_v3.jsonl")
STYLE_ASSET = Path(
    "generated/style_research/style_transfer_experiments/iterations/"
    "full_regeneration_v1/method_assets/style_transfer_payloads.v1/"
    "assets.497a0db8919ccc9cfd97f6a729ff0528953d2d97477e83e07e55c42e4bb994d8.json"
)
ENGLISH_PROMPT = Path(
    "generated/style_research/style_transfer_experiments/iterations/"
    "full_regeneration_v1/prompts/english_semantic_source.v1.md"
)
NEUTRAL_PROMPT = Path(
    "generated/style_research/style_transfer_experiments/iterations/"
    "full_regeneration_v1/prompts/neutral_translation.v1.md"
)
ENGLISH_SCHEMA = Path(
    "generated/style_research/style_transfer_experiments/iterations/"
    "full_regeneration_v1/schemas/english_semantic_source_output.v1.schema.json"
)
NEUTRAL_SCHEMA = Path(
    "generated/style_research/style_transfer_experiments/iterations/"
    "full_regeneration_v1/schemas/neutral_translation_output.v1.schema.json"
)
ASSET_DIR = Path("experiments/iteration5/construct/assets")
ADJUDICATION_PROMPT = ASSET_DIR / "english_semantic_adjudication.v1.md"
ADJUDICATION_SCHEMA = ASSET_DIR / "english_adjudication.schema.json"
STYLE_PROMPT = ASSET_DIR / "oracle_style_positive_control.v1.md"
STYLE_SCHEMA = ASSET_DIR / "style_positive_control.schema.json"
SHAM_PROMPT = ASSET_DIR / "style_sham_control.v1.md"
RUNNER = Path("experiments/iteration5/construct/run_cr_fysm_v4_construct_generation.py")
CLOSURE = Path("experiments/iteration5/construct/close_cr_fysm_v4_construct_generation.py")
REGRESSION_TEST = Path(
    "tests/experiments/iteration5/test_cr_fysm_v4_construct_v2.py"
)
POSTCLEAN_ROOT = Path(
    "generated/style_research/style_transfer_experiments/iterations/"
    "content_resistant_v1/cr_fysm_v4_postclean_replication_v2"
)
POSTCLEAN_PREREGISTRATION = POSTCLEAN_ROOT / "preregistration.json"
POSTCLEAN_RESULTS = POSTCLEAN_ROOT / "results.json"
EXPLICIT_RE = re.compile(
    "勃起|男根|龟头|乳头|裸体|赤裸|做爱|性交|插入|射精|精液|肛交|下身|胯下|"
    "情欲|吸吮|穴口|套弄|男性部位|阴茎|肉棒|后穴|龙根|阳物|往里挺进|"
    "淫唐传|高[Hh]|[Gg][Aa][Nn]死|AV片|色情片|毛片|黄片|肉文|春宫|"
    "床板.{0,12}吱呀|办完事|进入.{0,24}强袭|侵略自己|"
    "潮红.{0,80}喘|喘.{0,80}(?:潮红|黏黏糊糊|腰.{0,12}掐红)"
)
SOURCE_CORRUPTION_RE = re.compile(
    r"€{2,}|�|[（(][ｃcＣ][ｏoＯ](?:[ｍmＭ])?[）)]|"
    r"[（(][A-Za-zＡ-Ｚａ-ｚ0-9０-９._-]{1,40}[）)]"
    r"[（(][ｃcＣ][ｏoＯ](?:[ｍmＭ])?[）)]",
    re.I,
)
KNOWN_UNSAFE_SOURCE_CHUNK_IDS = frozenset(
    {
        "唐酒卿__将进酒__0616",
        "非天夜翔__北城天街__0002",
        "非天夜翔__锦衣卫__0001",
    }
)
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
CHAPTER_HEADING_RE = re.compile(
    r"^\s*(?:正文\s*)?(?:第[〇零一二三四五六七八九十百千万两0-9]+[章节卷回部集篇]|"
    r"番外(?:[〇零一二三四五六七八九十百千万两0-9]+)?|"
    r"楔子|序章|序言|前言|后记|尾声)(?:\s|[:：、.．-]|$)"
)
BOOK_TITLE_LINE_RE = re.compile(r"^\s*[《〈][^》〉]{1,100}[》〉]\s*$")
AUTHOR_HEADER_RE = re.compile(r"^\s*作者\s*[:：]")
AUTHOR_NOTE_RE = re.compile(r"(?:作者有话(?:要)?说|作者的话|作话)\s*[:：]?")
MARKUP_META_RE = re.compile(r"^\s*\*{2,}")
FRONT_MATTER_RE = re.compile(
    r"^\s*(?:内容标签|搜索关键字|一句话简介|立意|主角|配角|其它)\s*[:：]"
)
FANWAI_END_RE = re.compile(r"番外.{0,40}(?:·|[：:])?.{0,40}(?:完|终)")
DECORATED_END_RE = re.compile(r"^[—-]{2,}.{0,100}(?:End|END|终|完)[—-]{2,}$")
TERMINAL_MARKERS = (
    "正文完",
    "全文完",
    "本章完",
    "全书完",
    "本文到此",
    "彻底完结",
    "最后的番外",
    "不会再增加",
    "爱大家",
    "身为作者的我",
    "成书至今",
    "本书中也",
    "各位对本书",
    "书里书外",
    "朝各位道谢",
    "写序",
    "剧透太多",
    "文案已于",
    "这是我写过的",
    "契合正文",
    "直接拉末尾",
)
AUTHOR_META_TERMS = (
    "我的文",
    "其他作者",
    "读者",
    "画手",
    "微博",
    "安利",
    "专栏",
    "文案",
    "公告",
    "写文",
)
CODEX_DISABLED_FEATURES = (
    "shell_tool",
    "unified_exec",
    "browser_use",
    "browser_use_external",
    "browser_use_full_cdp_access",
    "in_app_browser",
    "computer_use",
    "apps",
    "multi_agent",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the fresh-domain CR-FYSM-v4 construct generation lock."
    )
    parser.add_argument("--mode", choices=("draft", "lock", "validate"), default="draft")
    return parser.parse_args()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def lock_id(payload: dict[str, Any]) -> str:
    value = dict(payload)
    value.pop("lock_id", None)
    return stable_hash(canonical(value))


def cjk_count(text: str) -> int:
    return len(CJK_RE.findall(text))


def codex_runtime_provenance() -> dict[str, Any]:
    executable = shutil.which("codex")
    if not executable:
        raise FileNotFoundError("codex executable not found")
    wrapper = Path(executable).resolve()
    native_candidates = sorted(
        path
        for path in wrapper.parent.parent.parent.glob("codex-*/vendor/*/bin/codex*")
        if path.is_file() and path.name in {"codex", "codex.exe"}
    )
    if len(native_candidates) != 1:
        raise ValueError(f"expected one installed native Codex binary, found {native_candidates}")
    resolved = native_candidates[0].resolve()
    completed = subprocess.run(
        [str(resolved), "--version"],
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    command_template = [
        str(resolved),
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--skip-git-repo-check",
        "--dangerously-bypass-approvals-and-sandbox",
    ]
    for feature in CODEX_DISABLED_FEATURES:
        command_template.extend(("--disable", feature))
    command_template.extend(
        (
            "--cd",
            "<isolated_working_directory>",
            "--model",
            "<stage_model_alias>",
            "--config",
            'model_reasoning_effort="<stage_reasoning_effort>"',
            "--output-schema",
            "<isolated_schema_path>",
            "--output-last-message",
            "<isolated_response_path>",
            "--json",
            "-",
        )
    )
    return {
        "codex_wrapper_resolved": str(wrapper),
        "codex_wrapper_sha256": sha256_file(wrapper),
        "codex_executable_resolved": str(resolved),
        "codex_executable_sha256": sha256_file(resolved),
        "codex_version": completed.stdout.strip() or completed.stderr.strip(),
        "command_template": command_template,
        "command_template_sha256": stable_hash(canonical(command_template)),
        "model_snapshot_identity": None,
        "model_snapshot_limitation": (
            "The hosted API exposes requested model aliases but no immutable model snapshot "
            "identifier; exact provider-side rerun identity cannot be guaranteed."
        ),
    }


def prior_allocation_paths() -> list[Path]:
    base = Path("generated/style_research/style_transfer_experiments")
    return sorted(base.glob("**/sample_sets/*.evaluator_allocation.jsonl"))


def prior_chunk_ids(paths: list[Path]) -> set[str]:
    values: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            chunk_id = row.get("chunk_id")
            if chunk_id:
                values.add(str(chunk_id))
    return values


def masked_map(path: Path) -> dict[str, str]:
    return {row["chunk_id"]: row["text"] for row in read_jsonl(path)}


def max_chunk_indices(rows: list[dict[str, Any]]) -> dict[tuple[str, str], int]:
    result: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        key = (row["author"], row["title"])
        result[key] = max(result[key], int(row["chunk_index"]))
    return dict(result)


def safe_source(row: dict[str, Any]) -> bool:
    text = row["text"]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    normalized_lines = [re.sub(r"\s+", "", line) for line in lines]
    return (
        900 <= int(row["chunk_clean_cjk_count"]) <= 2200
        and row.get("chunk_id") not in KNOWN_UNSAFE_SOURCE_CHUNK_IDS
        and not EXPLICIT_RE.search(text)
        and not any(marker in text for marker in TERMINAL_MARKERS)
        and sum(term in text for term in AUTHOR_META_TERMS) < 3
        and not SOURCE_CORRUPTION_RE.search(text)
        and "□" not in text
        and "?" not in text
        and not any(CHAPTER_HEADING_RE.match(line) for line in lines)
        and not any(BOOK_TITLE_LINE_RE.match(line) for line in lines)
        and not any(AUTHOR_HEADER_RE.match(line) for line in lines)
        and not AUTHOR_NOTE_RE.search(text)
        and not any(MARKUP_META_RE.match(line) for line in lines)
        and not any(FRONT_MATTER_RE.match(line) for line in lines)
        and not FANWAI_END_RE.search(text)
        and not any(DECORATED_END_RE.match(line) for line in lines)
        and not any(
            left == right
            for left, right in zip(normalized_lines, normalized_lines[1:], strict=False)
        )
        and "jjwxc" not in text.lower()
        and "http://" not in text.lower()
        and "https://" not in text.lower()
    )


def select_target_rows(
    rows: list[dict[str, Any]],
    *,
    prior_ids: set[str],
) -> list[dict[str, Any]]:
    target = [row for row in rows if row["author"] == TARGET_AUTHOR]
    maxima = max_chunk_indices(target)
    by_book: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in target:
        maximum = maxima[(row["author"], row["title"])]
        index = int(row["chunk_index"])
        if index not in {1, 2, maximum - 1, maximum}:
            continue
        if row["chunk_id"] in prior_ids or not safe_source(row):
            continue
        by_book[row["title"]].append(row)
    chosen_per_book = []
    for title, candidates in by_book.items():
        candidates.sort(key=lambda row: stable_hash(f"{SEED}:target:{row['chunk_id']}"))
        chosen_per_book.append(candidates[0])
    chosen_per_book.sort(key=lambda row: stable_hash(f"{SEED}:target-book:{row['title']}"))
    selected = chosen_per_book[:25]
    if len(selected) != 25 or len({row["title"] for row in selected}) != 25:
        raise ValueError("fresh target selection requires 25 boundary chunks from 25 books")
    return selected


def select_comparison_rows(
    rows: list[dict[str, Any]],
    *,
    prior_ids: set[str],
    target_time_area_counts: Counter[str],
) -> list[dict[str, Any]]:
    maxima = max_chunk_indices(rows)
    by_author_book: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        author = row["author"]
        if author == TARGET_AUTHOR or row["chunk_id"] in prior_ids or not safe_source(row):
            continue
        maximum = maxima[(author, row["title"])]
        if int(row["chunk_index"]) not in {1, 2, maximum - 1, maximum}:
            continue
        by_author_book[author][row["title"]].append(row)
    categories = tuple(sorted(target_time_area_counts))
    target_profile = tuple(target_time_area_counts[category] for category in categories)
    options: dict[str, dict[tuple[int, ...], tuple[dict[str, Any], ...]]] = {}
    for author, books in by_author_book.items():
        candidates = sorted(
            (row for book_rows in books.values() for row in book_rows),
            key=lambda row: stable_hash(f"{SEED}:comparison-row:{author}:{row['chunk_id']}"),
        )
        if len(candidates) < 5 or len({row["title"] for row in candidates}) < 3:
            continue
        best_by_profile: dict[tuple[int, ...], tuple[dict[str, Any], ...]] = {}
        best_hashes: dict[tuple[int, ...], str] = {}
        for candidate_rows in combinations(candidates, 5):
            if len({row["title"] for row in candidate_rows}) < 3:
                continue
            counts = Counter(str(row["time_area"]) for row in candidate_rows)
            if any(category not in target_time_area_counts for category in counts):
                continue
            profile = tuple(counts[category] for category in categories)
            if any(value > limit for value, limit in zip(profile, target_profile, strict=True)):
                continue
            signature = stable_hash(
                f"{SEED}:comparison-option:{author}:"
                + ":".join(row["chunk_id"] for row in candidate_rows)
            )
            if profile not in best_hashes or signature < best_hashes[profile]:
                best_hashes[profile] = signature
                best_by_profile[profile] = candidate_rows
        if best_by_profile:
            options[author] = best_by_profile

    authors = sorted(options, key=lambda author: stable_hash(f"{SEED}:comparison-author:{author}"))
    zero_profile = tuple(0 for _ in categories)
    states: dict[
        tuple[int, tuple[int, ...]],
        tuple[tuple[str, tuple[dict[str, Any], ...]], ...],
    ] = {(0, zero_profile): ()}
    state_hashes: dict[tuple[int, tuple[int, ...]], str] = {(0, zero_profile): ""}
    for author in authors:
        updated = dict(states)
        updated_hashes = dict(state_hashes)
        author_options = sorted(
            options[author].items(),
            key=lambda item: stable_hash(
                f"{SEED}:comparison-profile:{author}:{item[0]}:"
                + ":".join(row["chunk_id"] for row in item[1])
            ),
        )
        for (author_count, current_profile), current_selection in states.items():
            if author_count >= 5:
                continue
            for profile, candidate_rows in author_options:
                combined = tuple(
                    left + right
                    for left, right in zip(current_profile, profile, strict=True)
                )
                if any(
                    value > limit
                    for value, limit in zip(combined, target_profile, strict=True)
                ):
                    continue
                key = (author_count + 1, combined)
                selection = (*current_selection, (author, candidate_rows))
                signature = stable_hash(
                    f"{SEED}:comparison-state:"
                    + ":".join(
                        f"{selected_author}[{','.join(row['chunk_id'] for row in selected_rows)}]"
                        for selected_author, selected_rows in selection
                    )
                )
                if key not in updated_hashes or signature < updated_hashes[key]:
                    updated[key] = selection
                    updated_hashes[key] = signature
        states = updated
        state_hashes = updated_hashes

    final = states.get((5, target_profile))
    if final is None:
        raise ValueError(
            f"no five-author comparison selection matches time-area profile {dict(target_time_area_counts)}"
        )
    selected = [row for _, selected_rows in final for row in selected_rows]
    authors = [author for author, _ in final]
    if len(selected) != 25 or Counter(row["author"] for row in selected) != Counter(
        {author: 5 for author in authors}
    ):
        raise ValueError("comparison selection is not author balanced")
    return selected


def paragraph_rows(text: str) -> list[dict[str, str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 8:
        raise ValueError("construct source has fewer than eight nonempty paragraphs")
    return [
        {"id": f"p{index:04d}", "zh": line}
        for index, line in enumerate(lines, start=1)
    ]


def construct_blocks(paragraphs: list[dict[str, str]]) -> list[dict[str, Any]]:
    blocks = deterministic_blocks(paragraphs)
    by_id = {row["id"]: row["zh"] for row in paragraphs}
    result = [
        {
            "block_id": f"b{index:02d}",
            "paragraph_ids": paragraph_ids,
            "original_han": len(han_text("\n".join(by_id[value] for value in paragraph_ids))),
        }
        for index, paragraph_ids in enumerate(blocks, start=1)
    ]
    minimum = int(outcome_protocol()["units"]["minimum_original_han_per_block"])
    if any(row["original_han"] < minimum for row in result):
        raise ValueError(f"construct source has a block below {minimum} Han characters")
    return result


def selection_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    active = read_jsonl(ACTIVE_CLEAN)
    allocation_paths = prior_allocation_paths()
    allocated = prior_chunk_ids(allocation_paths)
    target = select_target_rows(active, prior_ids=allocated)
    target_time_area_counts = Counter(str(row["time_area"]) for row in target)
    comparison = select_comparison_rows(
        active,
        prior_ids=allocated,
        target_time_area_counts=target_time_area_counts,
    )
    active_masked = masked_map(ACTIVE_MASKED)
    rows: list[dict[str, Any]] = []
    for role, source_rows, mask_lookup in (
        ("target_boundary_human_positive", target, active_masked),
        ("comparison_boundary_human_negative", comparison, active_masked),
    ):
        for source in source_rows:
            sample_id = "s_" + stable_hash(f"construct-v2:{source['chunk_id']}")[:24]
            paragraphs = paragraph_rows(source["text"])
            rows.append(
                {
                    "sample_id": sample_id,
                    "source_role": role,
                    "source_author": source["author"],
                    "source_book": source["title"],
                    "source_chunk_id": source["chunk_id"],
                    "source_chunk_index": source["chunk_index"],
                    "source_split": source["split"],
                    "time_area": source["time_area"],
                    "genre": source["genre"],
                    "original_sha256": stable_hash(source["text"]),
                    "masked_sha256": stable_hash(mask_lookup[source["chunk_id"]]),
                    "paragraphs": paragraphs,
                    "construct_blocks": construct_blocks(paragraphs),
                    "entity_masked_v3_zh": mask_lookup[source["chunk_id"]],
                }
            )
    rows.sort(key=lambda row: row["sample_id"])
    summary = {
        "samples": len(rows),
        "target_samples": sum(row["source_author"] == TARGET_AUTHOR for row in rows),
        "comparison_samples": sum(row["source_author"] != TARGET_AUTHOR for row in rows),
        "target_books": len({row["source_book"] for row in rows if row["source_author"] == TARGET_AUTHOR}),
        "comparison_authors": len({row["source_author"] for row in rows if row["source_author"] != TARGET_AUTHOR}),
        "comparison_books": len({row["source_book"] for row in rows if row["source_author"] != TARGET_AUTHOR}),
        "prior_allocation_files": len(allocation_paths),
        "prior_chunk_overlap": len({row["source_chunk_id"] for row in rows} & allocated),
        "target_time_area_counts": dict(sorted(target_time_area_counts.items())),
        "comparison_time_area_counts": dict(
            sorted(Counter(str(row["time_area"]) for row in comparison).items())
        ),
        "source_role_counts": dict(sorted(Counter(row["source_role"] for row in rows).items())),
    }
    if summary != {
        "samples": 50,
        "target_samples": 25,
        "comparison_samples": 25,
        "target_books": 25,
        "comparison_authors": 5,
        "comparison_books": summary["comparison_books"],
        "prior_allocation_files": len(allocation_paths),
        "prior_chunk_overlap": 0,
        "target_time_area_counts": summary["target_time_area_counts"],
        "comparison_time_area_counts": summary["target_time_area_counts"],
        "source_role_counts": {
            "comparison_boundary_human_negative": 25,
            "target_boundary_human_positive": 25,
        },
    }:
        raise ValueError(f"unexpected fresh construct geometry: {summary}")
    if summary["comparison_books"] < 15:
        raise ValueError(f"comparison selection uses too few books: {summary}")
    return rows, summary


def style_evidence() -> dict[str, Any]:
    asset = read_json(STYLE_ASSET)["assets"]
    dimensions = [
        {
            key: row[key]
            for key in ("dimension_id", "label", "description", "trigger", "guardrail", "failure_mode")
        }
        for row in asset["style_definition"]["definition"]["dimensions"]
    ]
    def example_text(value: Any) -> str:
        if isinstance(value, str):
            return value
        return "\n".join(str(paragraph["zh"]) for paragraph in value)

    pairs = [
        row
        for row in asset["aligned_pairs"]["pairs"]
        if not EXPLICIT_RE.search(
            example_text(row["neutral_zh"]) + "\n" + example_text(row["target_style_zh"])
        )
    ]
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        by_scene[row["scene_type"]].append(row)
    selected: list[dict[str, Any]] = []
    for scene_type in sorted(by_scene):
        by_scene[scene_type].sort(key=lambda row: stable_hash(f"{SEED}:evidence:{row['pair_id']}"))
        selected.append(by_scene[scene_type][0])
    remaining = [row for row in pairs if row not in selected]
    remaining.sort(key=lambda row: stable_hash(f"{SEED}:evidence-extra:{row['pair_id']}"))
    selected.extend(remaining[: max(0, 8 - len(selected))])
    selected = selected[:8]
    if len(selected) != 8:
        raise ValueError("style positive control requires eight safe reference pairs")
    examples = [
        {
            key: row[key]
            for key in ("pair_id", "scene_type", "neutral_zh", "target_style_zh")
        }
        for row in selected
    ]
    return {
        "dimensions": dimensions,
        "reference_examples": examples,
        "source_asset_sha256": sha256_file(STYLE_ASSET),
    }


def flatten_example(value: Any) -> str:
    if isinstance(value, str):
        return value
    return "\n".join(str(paragraph["zh"]) for paragraph in value)


def reference_overlap_control(
    rows: list[dict[str, Any]], evidence: dict[str, Any]
) -> dict[str, Any]:
    references = [
        flatten_example(example[side])
        for example in evidence["reference_examples"]
        for side in ("neutral_zh", "target_style_zh")
    ]
    baseline = [
        han_ngram_overlap(
            "\n".join(paragraph["zh"] for paragraph in row["paragraphs"]),
            references,
            n=4,
        )
        for row in rows
    ]
    q99 = float(numpy.quantile(numpy.asarray(baseline, dtype=float), 0.99))
    threshold = min(0.12, max(0.03, q99 + 0.02))
    return {
        "ngram_n": 4,
        "reference_sides": ["neutral_zh", "target_style_zh"],
        "calibration_population": "50 selected pre-generation human originals",
        "baseline_mean": float(numpy.mean(baseline)),
        "baseline_p99": q99,
        "threshold_rule": "min(0.12, max(0.03, baseline_p99 + 0.02))",
        "maximum_overlap_rate": threshold,
        "exact_consecutive_han_max": 7,
    }


def fixed_paths() -> list[Path]:
    return [
        Path(__file__),
        RUNNER,
        ACTIVE_CLEAN,
        ACTIVE_MASKED,
        STYLE_ASSET,
        ENGLISH_PROMPT,
        NEUTRAL_PROMPT,
        ENGLISH_SCHEMA,
        NEUTRAL_SCHEMA,
        ADJUDICATION_PROMPT,
        ADJUDICATION_SCHEMA,
        STYLE_PROMPT,
        SHAM_PROMPT,
        STYLE_SCHEMA,
        REGRESSION_TEST,
        CLOSURE,
        Path("experiments/iteration5/construct/construct_v2_protocol.py"),
        POSTCLEAN_PREREGISTRATION,
        POSTCLEAN_RESULTS,
        POSTCLEAN_ROOT / "report.md",
        Path("experiments/iteration5/meter/preregister_cr_fysm_v4_postclean_v2.py"),
        Path("experiments/iteration5/meter/rescore_cr_fysm_v4_postclean_v2.py"),
        Path("experiments/iteration5/meter/benchmark_content_resistant_meter.py"),
        Path("experiments/iteration5/meter/build_cr_fysm_v3.py"),
        Path("experiments/iteration5/meter/build_cr_fysm_v4.py"),
        Path("uv.lock"),
        Path("pyproject.toml"),
        *prior_allocation_paths(),
    ]


def input_hashes() -> dict[str, str]:
    paths = sorted(set(fixed_paths()), key=lambda value: str(value))
    return {str(path): sha256_file(path) for path in paths}


def meter_replication() -> dict[str, Any]:
    preregistration = read_json(POSTCLEAN_PREREGISTRATION)
    result = read_json(POSTCLEAN_RESULTS)
    if preregistration.get("status") != "locked_before_any_postclean_score":
        raise ValueError("post-clean meter replication preregistration is not locked")
    if preregistration.get("lock_id") != lock_id(preregistration):
        raise ValueError("post-clean meter replication lock is invalid")
    if result.get("status") != "replication_pass_pending_construct_gates":
        raise ValueError("post-clean meter replication did not pass its limited gates")
    if result.get("preregistration", {}).get("lock_id") != preregistration["lock_id"]:
        raise ValueError("post-clean meter result does not bind to its preregistration")
    return {
        "experiment_id": result["experiment_id"],
        "preregistration_path": str(POSTCLEAN_PREREGISTRATION),
        "preregistration_lock_id": preregistration["lock_id"],
        "result_path": str(POSTCLEAN_RESULTS),
        "result_status": result["status"],
        "claim_limit": result["claim_limit"],
        "threshold": result["decision_policy"]["threshold"],
        "construct_gate_pending": True,
    }


def payload(
    status: str,
    summary: dict[str, Any],
    evidence: dict[str, Any],
    replication: dict[str, Any],
    overlap_control: dict[str, Any],
) -> dict[str, Any]:
    runtime = codex_runtime_provenance()
    result: dict[str, Any] = {
        "schema_version": 2,
        "construct_id": "CR-FYSM-v4-fresh-generated-construct-v2",
        "status": status,
        "seed": SEED,
        "selection": {
            "path": str(SELECTION),
            "sha256": sha256_file(SELECTION),
            "geometry": summary,
            "target_freshness": (
                "exact first/last-two chunks excluded by boundary=2 from every CR-FYSM-v3/v4 "
                "fit, calibration, and qualification load; books themselves are not unseen"
            ),
            "comparison_freshness": (
                "exact first/last-two chunks excluded by boundary=2 from every CR-FYSM-v3/v4 "
                "fit, calibration, and qualification load; authors and books themselves are not unseen"
            ),
            "prior_allocation_overlap": 0,
            "score_blind_selection": True,
        },
        "source_meter_replication": replication,
        "style_evidence": evidence,
        "reference_overlap_control": overlap_control,
        "generation_stages": {
            "english_semantic_source": {
                "model": "gpt-5.4",
                "reasoning_effort": "high",
                "prompt_path": str(ENGLISH_PROMPT),
                "schema_path": str(ENGLISH_SCHEMA),
                "max_attempts": 2,
                "timeout_seconds": 900,
            },
            "english_adjudication": {
                "model": "gpt-5.5",
                "reasoning_effort": "high",
                "prompt_path": str(ADJUDICATION_PROMPT),
                "schema_path": str(ADJUDICATION_SCHEMA),
                "max_attempts": 2,
                "timeout_seconds": 900,
            },
            "neutral_translation": {
                "model": "gpt-5.4",
                "reasoning_effort": "high",
                "prompt_path": str(NEUTRAL_PROMPT),
                "schema_path": str(NEUTRAL_SCHEMA),
                "max_attempts": 2,
                "timeout_seconds": 900,
                "production_prompt_identity": True,
            },
            "style_positive_control": {
                "model": "gpt-5.5",
                "reasoning_effort": "high",
                "prompt_path": str(STYLE_PROMPT),
                "schema_path": str(STYLE_SCHEMA),
                "max_attempts": 2,
                "timeout_seconds": 900,
                "research_role": "high-access positive control, not a production method candidate",
            },
            "style_sham_control": {
                "model": "gpt-5.5",
                "reasoning_effort": "high",
                "prompt_path": str(SHAM_PROMPT),
                "schema_path": str(STYLE_SCHEMA),
                "max_attempts": 2,
                "timeout_seconds": 900,
                "research_role": "same-model reference-free full-regeneration control",
            },
        },
        "generation_completion_gates": {
            "rows_per_stage": 50,
            "missing_or_invalid_outputs": 0,
            "paragraph_id_and_order_match": True,
            "english_adjudication_approved": True,
            "reference_copy_max_consecutive_cjk": 7,
            "reference_4gram_overlap_max": overlap_control["maximum_overlap_rate"],
            "study_closure_manifest_required": True,
            "later_semantic_validation_required": True,
        },
        "future_rating_geometry": {
            "blocks_per_source": 5,
            "candidate_matched_generated_style_neutral_pairs": 250,
            "candidate_matched_generated_style_sham_pairs": 250,
            "generated_variants_for_semantic_validation": 750,
            "minimum_semantically_valid_matched_pairs": 200,
            "minimum_valid_blocks_per_source": 4,
            "required_sources_meeting_per_source_minimum": 50,
            "minimum_effective_clusters": 50,
            "three_independent_style_raters": True,
            "two_independent_semantic_validators_for_all_750_generated_variants": True,
        },
        "outcome_identification_protocol": outcome_protocol(),
        "forbidden_claims_before_later_lock": [
            "no meter score or construct result",
            "no style-transfer efficacy claim",
            "no human-rater claim",
        ],
        "runtime": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "scikit_learn": sklearn.__version__,
            **runtime,
        },
        "input_hashes": input_hashes(),
    }
    result["lock_id"] = lock_id(result)
    return result


def main() -> None:
    args = parse_args()
    if args.mode == "validate":
        existing = read_json(PREREGISTRATION)
        if existing.get("status") != "locked_before_any_construct_generation":
            raise ValueError("construct v2 generation preregistration is not locked")
        if existing.get("lock_id") != lock_id(existing):
            raise ValueError("construct v2 generation lock_id is invalid")
        if existing.get("input_hashes") != input_hashes():
            raise ValueError("construct v2 generation inputs changed")
        if existing["selection"]["sha256"] != sha256_file(SELECTION):
            raise ValueError("construct v2 source selection changed")
        print(json.dumps({"status": "valid", "lock_id": existing["lock_id"]}, indent=2))
        return

    if args.mode == "lock":
        if not PREREGISTRATION.exists():
            raise ValueError("prepare and audit a draft before lock")
        existing = read_json(PREREGISTRATION)
        if existing.get("status") != "draft_before_any_construct_generation":
            raise ValueError("only a draft can be locked")
        if any((ROOT / "generation" / stage).exists() for stage in (
            "english_semantic_source",
            "english_adjudication",
            "neutral_translation",
            "style_sham_control",
            "style_positive_control",
        )):
            raise ValueError("generation outputs exist before the v2 lock")

    ROOT.mkdir(parents=True, exist_ok=True)
    rows, summary = selection_rows()
    write_jsonl(SELECTION, rows)
    evidence = style_evidence()
    replication = meter_replication()
    overlap_control = reference_overlap_control(rows, evidence)
    status = (
        "locked_before_any_construct_generation"
        if args.mode == "lock"
        else "draft_before_any_construct_generation"
    )
    preregistration = payload(status, summary, evidence, replication, overlap_control)
    write_json(PREREGISTRATION, preregistration)
    print(
        json.dumps(
            {
                "status": status,
                "lock_id": preregistration["lock_id"],
                "selection": summary,
                "reference_examples": len(evidence["reference_examples"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
