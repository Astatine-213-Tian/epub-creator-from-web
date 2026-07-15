from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


from experiments.shared.paths import RESEARCH_ROOT
from workflows.author_style_meter_contract import (
    CURRENT_MASKED_VIEW,
    CURRENT_SCORER_ID,
    CURRENT_SCORER_VALIDATION_RESULT,
    CURRENT_SCORER_VALIDATION_RUN_KEY,
)


REPO_ROOT = RESEARCH_ROOT
DEFAULT_CLEAN_CHUNKS = REPO_ROOT / "datasets/unmasked/chunks.clean.jsonl"
DEFAULT_MASKED_CHUNKS = REPO_ROOT / "datasets/masked/chunks.entity_masked_v3.jsonl"
DEFAULT_SPLITS = REPO_ROOT / "generated/style_research/corpus/splits.json"
DEFAULT_EXPERIMENT_ROOT = (
    REPO_ROOT / "generated/style_research/style_transfer_experiments"
)
DEFAULT_REPORT = (
    REPO_ROOT
    / "docs/reports/03_transfer_iteration1_prompt_methods.md"
)
DEFAULT_SAMPLE_ID = "development_proxy_v1"
DEFAULT_SEED = 20260710
DEFAULT_TARGET_AUTHOR = "非天夜翔"
DEFAULT_CHUNKS_PER_BOOK = 20
DEFAULT_CALIBRATION_PER_BOOK = 4
DEFAULT_COMPARISON_CHUNKS_PER_AUTHOR = 5
DEFAULT_MIN_CHUNK_DISTANCE = 3

DEVELOPMENT_BOOKS: dict[str, tuple[str, ...]] = {
    "proxy_transfer": (
        "骑士之歌",
        "天宝伏妖录",
        "万物风华录",
        "清平梦华录",
    ),
    "dev": (
        "相见欢",
        "山有木兮",
        "乱世为王",
        "图灵密码",
    ),
}

FINAL_VALIDATION_BOOKS: tuple[str, ...] = (
    "星辰骑士",
    "夺梦",
    "定海浮生录",
    "国家一级注册驱魔师上岗培训通知",
)

COMPARISON_AUTHORS: tuple[str, ...] = (
    "priest",
    "木苏里",
    "巫哲",
    "唐酒卿",
    "墨香铜臭",
    "淮上",
    "梦溪石",
    "西子绪",
    "酱子贝",
    "稚楚",
    "莫晨欢",
    "漫漫何其多",
)

RESIDUE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("author_note", re.compile(r"作者有话要说")),
    ("jjwxc", re.compile(r"晋江文学城|jjwxc", re.IGNORECASE)),
    ("subscription_prompt", re.compile(r"请收藏|霸王票|营养液加更")),
    ("navigation_boilerplate", re.compile(r"最新网址|返回目录|手机阅读")),
    ("url", re.compile(r"https?://|www\.", re.IGNORECASE)),
)

CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
SENTENCE_BOUNDARY_RE = re.compile(r"[。！？!?…]+")
SAFE_NAME_RE = re.compile(r"[^0-9A-Za-z\u3400-\u9fff]+")
PUNCTUATION = set("，。！？；：、,.!?;:…—（）()【】[]《》“”‘’「」『』")
DIALOGUE_MARKS = set("“”‘’「」『』")

SCENE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "action_conflict": (
        "打",
        "杀",
        "冲",
        "追",
        "逃",
        "剑",
        "刀",
        "血",
        "战",
        "挡",
        "抓",
        "撞",
        "攻",
    ),
    "internal_reflection": (
        "想",
        "觉得",
        "意识到",
        "记得",
        "回忆",
        "心想",
        "明白",
        "仿佛",
        "似乎",
        "为什么",
    ),
    "worldbuilding_exposition": (
        "天下",
        "朝廷",
        "皇帝",
        "王",
        "城",
        "国",
        "族",
        "军",
        "历史",
        "传说",
        "制度",
        "规矩",
    ),
    "interpersonal_care": (
        "笑",
        "抱",
        "吻",
        "牵",
        "握住",
        "看着",
        "担心",
        "喜欢",
        "爱",
        "照顾",
        "安慰",
    ),
    "travel_transition": (
        "路上",
        "出发",
        "抵达",
        "离开",
        "回去",
        "回来",
        "走进",
        "穿过",
        "翌日",
        "次日",
        "片刻后",
    ),
    "dialogue": (
        "说道",
        "问道",
        "答道",
        "笑道",
        "喊道",
        "开口",
        "沉声",
        "低声",
        "道：",
        "问：",
    ),
}

METHOD_REGISTRY: tuple[dict[str, Any], ...] = (
    {
        "id": "neutral_only",
        "family": "control",
        "label": "Neutral only",
        "status": "not_run",
        "priority": 0,
        "intensities": ["none"],
        "hypothesis": "Negative control; establishes the pass-1 style-meter floor.",
        "inputs": ["english_semantic_source", "neutral_zh"],
    },
    {
        "id": "generic_author_style_light",
        "family": "generic_prompt",
        "label": "Generic author prompt, light",
        "status": "not_run",
        "priority": 1,
        "intensities": ["light"],
        "hypothesis": "An author name alone provides weak and unstable control.",
        "inputs": ["english_semantic_source", "neutral_zh", "target_author_name"],
    },
    {
        "id": "generic_author_style_strong",
        "family": "generic_prompt",
        "label": "Generic author prompt, strong",
        "status": "not_run",
        "priority": 1,
        "intensities": ["strong"],
        "hypothesis": "A strong generic instruction may raise style score but increase drift.",
        "inputs": ["english_semantic_source", "neutral_zh", "target_author_name"],
    },
    {
        "id": "global_style_cards",
        "family": "structured_definition",
        "label": "Global evidence cards",
        "status": "not_run",
        "priority": 2,
        "intensities": ["light", "medium"],
        "hypothesis": "Compact corpus-derived rules outperform an author-name prompt.",
        "inputs": ["english_semantic_source", "neutral_zh", "global_style_cards"],
    },
    {
        "id": "scene_routed_style_cards",
        "family": "structured_definition",
        "label": "Scene-routed evidence cards",
        "status": "not_run",
        "priority": 2,
        "intensities": ["light", "medium"],
        "hypothesis": "Routing rules by scene type improves opportunity fit.",
        "inputs": ["english_semantic_source", "neutral_zh", "scene_style_cards"],
    },
    {
        "id": "sentence_flow_cards",
        "family": "structured_definition",
        "label": "Sentence-flow cards",
        "status": "not_run",
        "priority": 2,
        "intensities": ["light", "medium"],
        "hypothesis": "Explicit rhythm, clause, and paragraph-flow rules add signal beyond diction.",
        "inputs": ["english_semantic_source", "neutral_zh", "flow_style_cards"],
    },
    {
        "id": "function_word_punctuation_dialogue_cards",
        "family": "structured_definition",
        "label": "Function-word and dialogue cards",
        "status": "not_run",
        "priority": 2,
        "intensities": ["light", "medium"],
        "hypothesis": "Interpretable function-word, punctuation, and dialogue cues provide a robust guardrail.",
        "inputs": ["english_semantic_source", "neutral_zh", "interpretable_style_cards"],
    },
    {
        "id": "retrieved_examples_only",
        "family": "retrieval",
        "label": "Retrieved examples only",
        "status": "not_run",
        "priority": 3,
        "intensities": ["light", "medium"],
        "hypothesis": "Local examples may convey tacit flow better than explicit rules.",
        "inputs": ["english_semantic_source", "neutral_zh", "retrieved_target_examples"],
    },
    {
        "id": "scene_cards_plus_examples",
        "family": "hybrid",
        "label": "Scene cards plus examples",
        "status": "not_run",
        "priority": 3,
        "intensities": ["light", "medium"],
        "hypothesis": "Structured controls and local demonstrations are complementary.",
        "inputs": [
            "english_semantic_source",
            "neutral_zh",
            "scene_style_cards",
            "retrieved_target_examples",
        ],
    },
    {
        "id": "contrastive_examples",
        "family": "retrieval",
        "label": "Contrastive examples",
        "status": "not_run",
        "priority": 3,
        "intensities": ["light", "medium"],
        "hypothesis": "Positive and hard-negative examples help isolate author style from genre.",
        "inputs": [
            "english_semantic_source",
            "neutral_zh",
            "target_examples",
            "hard_negative_examples",
        ],
    },
    {
        "id": "llm_close_reading_style_definition",
        "family": "llm_close_reading",
        "label": "LLM close-reading definition",
        "status": "not_run",
        "priority": 4,
        "intensities": ["light", "medium"],
        "hypothesis": "Close reading captures discourse and grammar cues missed by aggregate statistics.",
        "inputs": ["english_semantic_source", "neutral_zh", "close_reading_definition"],
    },
    {
        "id": "llm_close_reading_cards_statistical_gates",
        "family": "hybrid",
        "label": "Close reading plus statistical gates",
        "status": "not_run",
        "priority": 4,
        "intensities": ["light", "medium"],
        "hypothesis": "LLM synthesis is more reliable when every rule is checked against corpus statistics.",
        "inputs": [
            "english_semantic_source",
            "neutral_zh",
            "validated_close_reading_cards",
        ],
    },
    {
        "id": "self_critique_repair",
        "family": "verifier_loop",
        "label": "Generate, critique, and repair",
        "status": "not_run",
        "priority": 5,
        "intensities": ["light", "medium"],
        "hypothesis": "A separate fidelity/style critique can recover gains without accepting semantic drift.",
        "inputs": [
            "english_semantic_source",
            "neutral_zh",
            "best_prior_method_payload",
            "independent_critique",
        ],
    },
)


ENGLISH_SEMANTIC_SOURCE_PROMPT = """# English Semantic Source Prompt v1

## Purpose

Create a controlled English semantic source from a held-out Chinese passage. This
is benchmark construction, not literary translation. The Chinese target remains
hidden from every later neutral-translation and style-transfer run.

## Rules

1. Preserve events, causality, negation, modality, chronology, speaker identity,
   dialogue attribution, entities, numbers, and paragraph boundaries.
2. Use plain, natural English. Remove source-language rhythm, ornamental diction,
   idioms, and author-specific verbal mannerisms when they are not semantic facts.
3. Do not summarize, embellish, explain, censor, or infer unstated motives.
4. Keep each supplied paragraph ID unchanged and return exactly one output item
   per input item in the same order.
5. Use supplied terminology placeholders verbatim. Do not add notes.

## Input

```json
{
  "sample_id": "...",
  "paragraphs": [{"id": "p0001", "zh": "..."}],
  "terminology_placeholders": {}
}
```

## Output

Return JSON only:

```json
{
  "sample_id": "...",
  "prompt_version": "english_semantic_source.v1",
  "paragraphs": [{"id": "p0001", "en": "..."}],
  "uncertainties": []
}
```
"""


NEUTRAL_TRANSLATION_PROMPT = """# Neutral English-to-Chinese Translation Prompt v1

## Contract

This prompt family is the pass-1 contract for both the proxy benchmark and the
later Eternal Gate production translation. A production run may add its glossary
and comments, but it must not change these neutrality and fidelity rules without
creating a new prompt version and rerunning the benchmark.

## Rules

1. Treat the English text as the semantic authority.
2. Translate into natural, readable Simplified Chinese web-fiction prose with a
   neutral contemporary register.
3. Preserve events, sequence, causality, negation, modality, intensity, entities,
   numbers, speaker turns, dialogue attribution, and paragraph IDs.
4. Do not imitate any named author. Do not use style cards, author examples, or
   remembered passages. Do not deliberately make the prose archaic, lyrical,
   terse, ornate, comic, or genre-coded.
5. Do not add imagery, motives, lore, jokes, intimacy, injuries, or setting detail.
   Do not remove awkward semantic detail merely to improve elegance.
6. Use glossary terms and placeholders exactly. Return one output item for each
   input item in the same order.
7. Return the translation only in the required JSON structure; do not discuss the
   translation process.

## Input

```json
{
  "sample_id": "...",
  "paragraphs": [{"id": "p0001", "en": "..."}],
  "glossary": {},
  "comments": []
}
```

## Output

Return JSON only:

```json
{
  "sample_id": "...",
  "prompt_version": "neutral_translation.v1",
  "paragraphs": [{"id": "p0001", "zh": "..."}],
  "uncertainties": []
}
```

Recommended decoding for comparable research runs: temperature 0 to 0.2, fixed
model snapshot, and recorded model/provider metadata.
"""


ENGLISH_SOURCE_QA_PROMPT = """# English Semantic Source QA Prompt v1

## Role

Independently verify that a generated English semantic source is faithful to its
Chinese benchmark input. You are an evaluator, not a translator or editor.

## Rules

1. Check every paragraph for events, entities, numbers, roles, speaker identity,
   dialogue attribution, causality, chronology, negation, modality, intensity,
   omissions, and inventions.
2. Mark a paragraph `fail` for any semantic change that could affect a later
   English-to-Chinese reconstruction. Stylistic flattening is allowed and expected.
3. Mark the whole sample approved only when every paragraph passes.
4. Do not repair the text. Do not use tools or external context.
5. Return only the required JSON object.

## Input

```json
{
  "sample_id": "s_...",
  "paragraphs": [{"id": "p0001", "zh": "...", "en": "..."}]
}
```

## Output

```json
{
  "sample_id": "s_...",
  "prompt_version": "english_source_qa.v1",
  "approved": true,
  "paragraph_reviews": [
    {"id": "p0001", "status": "pass", "issues": []}
  ],
  "overall_issues": []
}
```
"""


ENGLISH_SOURCE_REPAIR_PROMPT = """# English Semantic Source Repair Prompt v1

## Role

Repair a quarantined plain-English semantic source using an independent QA report.
This is benchmark-source correction, not literary translation.

## Rules

1. Treat the supplied Chinese paragraphs as semantic authority.
2. Correct every QA-listed meaning error while preserving events, entities,
   numbers, roles, speaker identity, causality, chronology, negation, modality,
   intensity, and paragraph boundaries.
3. Keep paragraphs whose `issues` list is empty byte-for-byte unchanged unless a
   change is strictly required to keep cross-paragraph reference coherent.
4. Use plain, natural English and remove author-specific rhythm or ornament that
   is not semantic content.
5. Do not summarize, embellish, explain, or add notes. Preserve every paragraph ID
   and return exactly one output item per input item in the same order.

## Input

```json
{
  "sample_id": "s_...",
  "paragraphs": [
    {"id": "p0001", "zh": "...", "en": "...", "issues": []}
  ]
}
```

## Output

```json
{
  "sample_id": "s_...",
  "prompt_version": "english_source_repair.v1",
  "paragraphs": [{"id": "p0001", "en": "..."}],
  "uncertainties": []
}
```
"""


STYLE_TRANSFER_PROMPT = """# Style-Transfer Method Prompt v1

## Task

Recast the neutral Chinese draft using the supplied method payload. The English
semantic source remains authoritative. Apply only style operations supported by
the payload and by an opportunity actually present in the source.

## Non-negotiable constraints

1. Preserve paragraph IDs, facts, event order, causality, negation, modality,
   intensity, entities, numbers, and speaker attribution.
2. Do not import names, places, lore, imagery, phrases, or plot facts from any
   reference passage.
3. Do not copy a sequence of eight or more Chinese characters from a reference,
   except common fixed expressions or required terminology.
4. Prefer the lightest rewrite that realizes the requested style cue. If a cue
   conflicts with meaning or naturalness, skip it and record the reason.
5. Do not mention the author, style analysis, rules, cards, or references in the
   prose.

## Input

```json
{
  "sample_id": "...",
  "method_id": "...",
  "intensity": "light|medium|strong",
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
  "intensity": "...",
  "paragraphs": [{"id": "p0001", "zh": "..."}],
  "style_cues_applied": [],
  "style_cues_skipped": [],
  "uncertainties": []
}
```
"""


STYLE_TRANSFER_CRITIQUE_PROMPT = """# Independent Style-Transfer Critique Prompt v1

## Role

Evaluate one anonymized styled Chinese candidate against its English semantic
source and neutral Chinese draft. You did not generate the candidate. Do not infer
or discuss which method produced it.

## Rules

1. Treat English as semantic authority. Independently check both the neutral draft
   and styled candidate for events, entities, numbers, roles, speaker turns,
   causality, chronology, negation, modality, intensity, omissions, and inventions.
2. Check candidate readability relative to the neutral draft for translationese,
   clipped pseudo-literary prose, repetitive mannerisms, register instability,
   and analysis/prompt leakage.
3. For either Chinese version, mark semantic status `major` for a material meaning
   change or invention, `minor` for a local ambiguity that preserves the main
   event, and otherwise `pass`. For the candidate, also mark `major` when it is
   materially less readable than neutral or cannot be repaired locally.
4. Candidate `repair_required` and repair instructions concern the candidate only.
   Give concrete instructions, but do not write replacement prose.
5. Evaluate each version independently; do not excuse a candidate error because
   the neutral draft has the same error.
6. Return JSON only and preserve every paragraph ID and order in both review lists.

## Input

```json
{
  "sample_id": "s_...",
  "english_semantic_source": [{"id": "p0001", "en": "..."}],
  "neutral_zh": [{"id": "p0001", "zh": "..."}],
  "candidate_zh": [{"id": "p0001", "zh": "..."}]
}
```

## Output

```json
{
  "sample_id": "s_...",
  "prompt_version": "style_transfer_critique.v1",
  "repair_required": false,
  "neutral_paragraph_reviews": [
    {
      "id": "p0001",
      "status": "pass|minor|major",
      "fidelity_issues": []
    }
  ],
  "paragraph_reviews": [
    {
      "id": "p0001",
      "status": "pass|minor|major",
      "fidelity_issues": [],
      "readability_issues": [],
      "repair_instructions": []
    }
  ],
  "overall_risks": []
}
```
"""


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and report controlled author-style transfer experiments."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_paths(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT
        )
        command.add_argument("--report", type=Path, default=DEFAULT_REPORT)

    prepare = subparsers.add_parser(
        "prepare-samples", help="Build a deterministic, stratified development set."
    )
    add_common_paths(prepare)
    prepare.add_argument("--clean-chunks", type=Path, default=DEFAULT_CLEAN_CHUNKS)
    prepare.add_argument("--masked-chunks", type=Path, default=DEFAULT_MASKED_CHUNKS)
    prepare.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    prepare.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    prepare.add_argument("--target-author", default=DEFAULT_TARGET_AUTHOR)
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare.add_argument(
        "--chunks-per-book", type=int, default=DEFAULT_CHUNKS_PER_BOOK
    )
    prepare.add_argument(
        "--calibration-per-book", type=int, default=DEFAULT_CALIBRATION_PER_BOOK
    )
    prepare.add_argument(
        "--comparison-chunks-per-author",
        type=int,
        default=DEFAULT_COMPARISON_CHUNKS_PER_AUTHOR,
    )
    prepare.add_argument(
        "--min-chunk-distance", type=int, default=DEFAULT_MIN_CHUNK_DISTANCE
    )

    prompts = subparsers.add_parser(
        "init-prompts", help="Write versioned prompts and the method registry."
    )
    add_common_paths(prompts)

    report = subparsers.add_parser(
        "report", help="Regenerate the readable research report from artifacts."
    )
    add_common_paths(report)
    report.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    report.add_argument("--evaluation", type=Path)

    validate = subparsers.add_parser(
        "validate", help="Validate artifact isolation, counts, and reproducibility metadata."
    )
    add_common_paths(validate)
    validate.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)

    initialize = subparsers.add_parser(
        "init", help="Prepare samples, initialize prompts, validate, and write the report."
    )
    add_common_paths(initialize)
    initialize.add_argument("--clean-chunks", type=Path, default=DEFAULT_CLEAN_CHUNKS)
    initialize.add_argument("--masked-chunks", type=Path, default=DEFAULT_MASKED_CHUNKS)
    initialize.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    initialize.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    initialize.add_argument("--target-author", default=DEFAULT_TARGET_AUTHOR)
    initialize.add_argument("--seed", type=int, default=DEFAULT_SEED)
    initialize.add_argument(
        "--chunks-per-book", type=int, default=DEFAULT_CHUNKS_PER_BOOK
    )
    initialize.add_argument(
        "--calibration-per-book", type=int, default=DEFAULT_CALIBRATION_PER_BOOK
    )
    initialize.add_argument(
        "--comparison-chunks-per-author",
        type=int,
        default=DEFAULT_COMPARISON_CHUNKS_PER_AUTHOR,
    )
    initialize.add_argument(
        "--min-chunk-distance", type=int, default=DEFAULT_MIN_CHUNK_DISTANCE
    )
    initialize.add_argument("--evaluation", type=Path)
    return parser.parse_args()


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
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def cjk_count(text: str) -> int:
    return len(CJK_RE.findall(text))


def safe_name(text: str) -> str:
    return SAFE_NAME_RE.sub("_", text).strip("_")


def stable_seed(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def tercile_bucket(value: float, first: float, second: float) -> str:
    if value <= first:
        return "low"
    if value <= second:
        return "medium"
    return "high"


def relative_position_bucket(chunk_index: int, max_chunk_index: int) -> str:
    ratio = chunk_index / max(max_chunk_index, 1)
    if ratio <= 0.10:
        return "opening"
    if ratio <= 0.33:
        return "early"
    if ratio <= 0.67:
        return "middle"
    if ratio <= 0.90:
        return "late"
    return "ending"


def scene_scores(text: str, dialogue_density: float) -> dict[str, int]:
    scores: dict[str, int] = {
        label: sum(text.count(keyword) for keyword in keywords)
        for label, keywords in SCENE_KEYWORDS.items()
    }
    scores["dialogue"] += int(dialogue_density * 300)
    return scores


def scene_type(
    scores: dict[str, int], distributions: dict[str, list[int]], row_count: int
) -> str:
    """Choose the scene whose cue count is most unusual within this corpus."""
    return max(
        scores,
        key=lambda label: (
            bisect_right(distributions[label], scores[label]) / max(row_count, 1),
            scores[label],
            label,
        ),
    )


def base_metrics(text: str) -> dict[str, Any]:
    paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
    sentences = [
        sentence.strip()
        for sentence in SENTENCE_BOUNDARY_RE.split(text)
        if sentence.strip()
    ]
    text_cjk = cjk_count(text)
    punctuation_count = sum(character in PUNCTUATION for character in text)
    dialogue_mark_count = sum(character in DIALOGUE_MARKS for character in text)
    paragraph_lengths = [cjk_count(paragraph) for paragraph in paragraphs]
    sentence_lengths = [cjk_count(sentence) for sentence in sentences]
    return {
        "cjk_count": text_cjk,
        "paragraph_count": len(paragraphs),
        "sentence_count": len(sentences),
        "mean_paragraph_cjk": (
            sum(paragraph_lengths) / len(paragraph_lengths) if paragraph_lengths else 0.0
        ),
        "mean_sentence_cjk": (
            sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0.0
        ),
        "punctuation_density": punctuation_count / max(len(text), 1),
        "dialogue_density": dialogue_mark_count / max(len(text), 1),
    }


def residue_hits(text: str) -> list[str]:
    return [label for label, pattern in RESIDUE_PATTERNS if pattern.search(text)]


def validate_split_contract(splits_path: Path, target_author: str) -> None:
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    for split_name, expected_titles in DEVELOPMENT_BOOKS.items():
        actual_titles = {
            row["title"]
            for row in splits.get(split_name, [])
            if row.get("author") == target_author
        }
        missing = set(expected_titles) - actual_titles
        if missing:
            raise ValueError(
                f"Split contract mismatch: {split_name} is missing {sorted(missing)}"
            )
    test_titles = {
        row["title"]
        for row in splits.get("test", [])
        if row.get("author") == target_author
    }
    missing_test = set(FINAL_VALIDATION_BOOKS) - test_titles
    if missing_test:
        raise ValueError(
            f"Final-validation contract mismatch: test is missing {sorted(missing_test)}"
        )


def load_development_candidates(
    clean_path: Path, target_author: str
) -> dict[str, list[dict[str, Any]]]:
    expected_split_by_title = {
        title: split_name
        for split_name, titles in DEVELOPMENT_BOOKS.items()
        for title in titles
    }
    rows_by_book: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in iter_jsonl(clean_path):
        title = row.get("title")
        if row.get("author") != target_author or title not in expected_split_by_title:
            continue
        expected_split = expected_split_by_title[title]
        if row.get("split") != expected_split:
            raise ValueError(
                f"Chunk {row.get('chunk_id')} has split {row.get('split')!r}; "
                f"expected {expected_split!r}."
            )
        text = row.get("text", "")
        if not text.strip() or residue_hits(text):
            continue
        candidate = {
            "author": row["author"],
            "book_title": title,
            "source_split": row["split"],
            "chunk_id": row["chunk_id"],
            "chunk_index": int(row["chunk_index"]),
            "quality_flags": row.get("quality_flags", []),
            "original_zh": text,
        }
        candidate.update(base_metrics(text))
        rows_by_book[title].append(candidate)

    missing = sorted(set(expected_split_by_title) - set(rows_by_book))
    if missing:
        raise ValueError(f"No clean chunks found for development books: {missing}")
    return rows_by_book


def load_comparison_candidates(
    clean_path: Path, target_author: str
) -> dict[str, list[dict[str, Any]]]:
    rows_by_author: dict[str, list[dict[str, Any]]] = defaultdict(list)
    comparison_authors = set(COMPARISON_AUTHORS)
    for row in iter_jsonl(clean_path):
        author = row.get("author")
        if (
            author == target_author
            or author not in comparison_authors
            or row.get("split") != "dev"
        ):
            continue
        text = row.get("text", "")
        if not text.strip() or residue_hits(text):
            continue
        candidate = {
            "author": author,
            "book_title": row["title"],
            "source_split": row["split"],
            "chunk_id": row["chunk_id"],
            "chunk_index": int(row["chunk_index"]),
            "quality_flags": row.get("quality_flags", []),
            "original_zh": text,
        }
        candidate.update(base_metrics(text))
        rows_by_author[author].append(candidate)

    missing = sorted(comparison_authors - set(rows_by_author))
    if missing:
        raise ValueError(f"No clean dev chunks found for comparison authors: {missing}")
    return rows_by_author


def enrich_strata(rows_by_book: dict[str, list[dict[str, Any]]]) -> None:
    all_rows = [row for rows in rows_by_book.values() for row in rows]
    raw_scene_scores = {
        row["chunk_id"]: scene_scores(row["original_zh"], row["dialogue_density"])
        for row in all_rows
    }
    scene_distributions = {
        label: sorted(scores[label] for scores in raw_scene_scores.values())
        for label in SCENE_KEYWORDS
    }
    length_breaks = (
        percentile([row["cjk_count"] for row in all_rows], 1 / 3),
        percentile([row["cjk_count"] for row in all_rows], 2 / 3),
    )
    punctuation_breaks = (
        percentile([row["punctuation_density"] for row in all_rows], 1 / 3),
        percentile([row["punctuation_density"] for row in all_rows], 2 / 3),
    )
    flow_breaks = (
        percentile([row["mean_sentence_cjk"] for row in all_rows], 1 / 3),
        percentile([row["mean_sentence_cjk"] for row in all_rows], 2 / 3),
    )
    dialogue_breaks = (
        percentile([row["dialogue_density"] for row in all_rows], 1 / 3),
        percentile([row["dialogue_density"] for row in all_rows], 2 / 3),
    )

    for rows in rows_by_book.values():
        max_chunk_index = max(row["chunk_index"] for row in rows)
        for row in rows:
            row["position_bucket"] = relative_position_bucket(
                row["chunk_index"], max_chunk_index
            )
            row["length_bucket"] = tercile_bucket(row["cjk_count"], *length_breaks)
            row["punctuation_bucket"] = tercile_bucket(
                row["punctuation_density"], *punctuation_breaks
            )
            row["flow_bucket"] = tercile_bucket(
                row["mean_sentence_cjk"], *flow_breaks
            )
            row["dialogue_bucket"] = tercile_bucket(
                row["dialogue_density"], *dialogue_breaks
            )
            row["scene_type"] = scene_type(
                raw_scene_scores[row["chunk_id"]],
                scene_distributions,
                len(all_rows),
            )


STRATIFICATION_FIELDS: tuple[str, ...] = (
    "position_bucket",
    "scene_type",
    "length_bucket",
    "punctuation_bucket",
    "flow_bucket",
    "dialogue_bucket",
)


def select_diverse_rows(
    rows: Sequence[dict[str, Any]],
    count: int,
    seed: int,
    min_chunk_distance: int = DEFAULT_MIN_CHUNK_DISTANCE,
) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise ValueError(f"Need {count} rows but only {len(rows)} are available.")
    rng = random.Random(seed)
    ordered = sorted(rows, key=lambda row: row["chunk_id"])
    tie_breakers = {row["chunk_id"]: rng.random() for row in ordered}
    selected: list[dict[str, Any]] = []
    remaining = list(ordered)
    coverage = {field: Counter() for field in STRATIFICATION_FIELDS}
    combo_coverage: Counter[tuple[str, ...]] = Counter()

    while len(selected) < count:
        if not remaining:
            raise ValueError(
                f"Could not select {count} spaced rows; selected {len(selected)}."
            )

        def diversity_score(row: dict[str, Any]) -> tuple[float, float]:
            score = sum(
                1.0 / (1.0 + coverage[field][row[field]])
                for field in STRATIFICATION_FIELDS
            )
            combo = tuple(row[field] for field in STRATIFICATION_FIELDS[:4])
            score += 1.5 / (1.0 + combo_coverage[combo])
            return score, tie_breakers[row["chunk_id"]]

        chosen = max(remaining, key=diversity_score)
        selected.append(chosen)
        remaining = [
            row
            for row in remaining
            if row["chunk_id"] != chosen["chunk_id"]
            and not (
                row["book_title"] == chosen["book_title"]
                and abs(row["chunk_index"] - chosen["chunk_index"])
                < min_chunk_distance
            )
        ]
        for field in STRATIFICATION_FIELDS:
            coverage[field][chosen[field]] += 1
        combo_coverage[
            tuple(chosen[field] for field in STRATIFICATION_FIELDS[:4])
        ] += 1
    return sorted(selected, key=lambda row: row["chunk_index"])


def assign_research_roles(
    rows: Sequence[dict[str, Any]], calibration_count: int, seed: int
) -> None:
    calibration = {
        row["chunk_id"]
        for row in select_diverse_rows(
            rows, calibration_count, seed, min_chunk_distance=0
        )
    }
    for row in rows:
        row["research_role"] = (
            "style_meter_calibration"
            if row["chunk_id"] in calibration
            else "method_evaluation"
        )


def load_masked_targets(masked_path: Path, selected_ids: set[str]) -> dict[str, str]:
    masked_by_id: dict[str, str] = {}
    for row in iter_jsonl(masked_path):
        chunk_id = row.get("chunk_id")
        if chunk_id in selected_ids:
            masked_by_id[chunk_id] = row.get("text", "")
            if len(masked_by_id) == len(selected_ids):
                break
    return masked_by_id


def sample_paths(experiment_root: Path, sample_id: str) -> dict[str, Path]:
    sample_dir = experiment_root / "sample_sets"
    return {
        "runner_manifest": sample_dir / f"{sample_id}.runner_manifest.jsonl",
        "evaluator_allocation": sample_dir
        / f"{sample_id}.evaluator_allocation.jsonl",
        "hidden_targets": sample_dir / f"{sample_id}.hidden_targets.jsonl",
        "method_evaluation_ids": sample_dir
        / f"{sample_id}.method_evaluation_ids.json",
        "screening_ids": sample_dir / f"{sample_id}.screening_v1_ids.json",
        "confirmation_ids": sample_dir / f"{sample_id}.confirmation_v1_ids.json",
        "summary": sample_dir / f"{sample_id}.summary.json",
    }


def relative_artifact_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def rows_sha256(rows: Sequence[dict[str, Any]]) -> str:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def opaque_sample_id(seed: int, sample_set: str, chunk_id: str) -> str:
    digest = hashlib.sha256(f"{seed}:{sample_set}:{chunk_id}".encode()).hexdigest()
    return f"s_{digest[:24]}"


def prepare_samples(args: argparse.Namespace) -> dict[str, Any]:
    if args.chunks_per_book <= 0:
        raise ValueError("--chunks-per-book must be positive.")
    if args.comparison_chunks_per_author <= 0:
        raise ValueError("--comparison-chunks-per-author must be positive.")
    if args.min_chunk_distance < 2:
        raise ValueError("--min-chunk-distance must be at least 2.")
    if not 0 < args.calibration_per_book < args.chunks_per_book:
        raise ValueError(
            "--calibration-per-book must be positive and smaller than --chunks-per-book."
        )
    validate_split_contract(args.splits, args.target_author)
    rows_by_book = load_development_candidates(args.clean_chunks, args.target_author)
    enrich_strata(rows_by_book)
    comparison_by_author = load_comparison_candidates(
        args.clean_chunks, args.target_author
    )
    enrich_strata(comparison_by_author)

    selected: list[dict[str, Any]] = []
    for split_name, titles in DEVELOPMENT_BOOKS.items():
        for title in titles:
            book_rows = select_diverse_rows(
                rows_by_book[title],
                args.chunks_per_book,
                stable_seed(args.seed, f"sample:{split_name}:{title}"),
                min_chunk_distance=args.min_chunk_distance,
            )
            assign_research_roles(
                book_rows,
                args.calibration_per_book,
                stable_seed(args.seed, f"role:{split_name}:{title}"),
            )
            for row in book_rows:
                row["benchmark_arm"] = "own_author_reconstruction"
            selected.extend(book_rows)

    for author in COMPARISON_AUTHORS:
        author_rows = select_diverse_rows(
            comparison_by_author[author],
            args.comparison_chunks_per_author,
            stable_seed(args.seed, f"comparison:{author}"),
            min_chunk_distance=args.min_chunk_distance,
        )
        for row in author_rows:
            row["research_role"] = "cross_author_method_evaluation"
            row["benchmark_arm"] = "cross_author_transfer"
        selected.extend(author_rows)

    selected_ids = {row["chunk_id"] for row in selected}
    masked_by_id = load_masked_targets(args.masked_chunks, selected_ids)
    missing_masked = sorted(selected_ids - set(masked_by_id))
    if missing_masked:
        raise ValueError(
            f"Masked view is missing {len(missing_masked)} selected chunks: "
            f"{missing_masked[:5]}"
        )

    runner_rows: list[dict[str, Any]] = []
    allocation_rows: list[dict[str, Any]] = []
    hidden_rows: list[dict[str, Any]] = []
    for row in selected:
        sample_id = opaque_sample_id(args.seed, args.sample_id, row["chunk_id"])
        base_output = args.experiment_root / "runs" / args.sample_id
        run_output = base_output / "<run_id>"
        runner_rows.append(
            {
                "sample_id": sample_id,
                "sample_set": args.sample_id,
                "stage_status": {
                    "english_semantic_source": "pending",
                    "english_source_qa": "pending",
                    "neutral_translation": "pending",
                    "style_transfer": "pending",
                    "automatic_evaluation": "pending",
                    "independent_evaluation": "pending",
                },
                "artifact_paths": {
                    "english_semantic_source": relative_artifact_path(
                        run_output
                        / "english_semantic_source"
                        / f"{sample_id}.json"
                    ),
                    "neutral_translation": relative_artifact_path(
                        run_output / "neutral_translation" / f"{sample_id}.json"
                    ),
                    "english_source_qa": relative_artifact_path(
                        run_output / "english_source_qa" / f"{sample_id}.json"
                    ),
                    "method_outputs": relative_artifact_path(
                        run_output
                        / "method_outputs"
                        / "<method_id>"
                        / "<intensity>"
                        / f"{sample_id}.json"
                    ),
                    "evaluation": relative_artifact_path(
                        run_output
                        / "evaluation"
                        / "<method_id>"
                        / "<intensity>"
                        / f"{sample_id}.json"
                    ),
                },
            }
        )
        allocation_rows.append(
            {
                "sample_id": sample_id,
                "sample_set": args.sample_id,
                "benchmark_arm": row["benchmark_arm"],
                "author": row["author"],
                "book_title": row["book_title"],
                "source_split": row["source_split"],
                "research_role": row["research_role"],
                "chunk_id": row["chunk_id"],
                "chunk_index": row["chunk_index"],
                "quality_flags": row["quality_flags"],
                "selected_residue_hits": residue_hits(row["original_zh"]),
                "strata": {field: row[field] for field in STRATIFICATION_FIELDS},
                "measurements": {
                    key: round(row[key], 6)
                    if isinstance(row[key], float)
                    else row[key]
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
        hidden_rows.append(
            {
                "sample_id": sample_id,
                "sample_set": args.sample_id,
                "original_zh": row["original_zh"],
                "entity_masked_v3_zh": masked_by_id[row["chunk_id"]],
                "access_policy": (
                    "one_way_semantic_source_generator_and_frozen_output_evaluator_only"
                ),
            }
        )

    runner_rows.sort(key=lambda row: row["sample_id"])
    allocation_rows.sort(key=lambda row: row["sample_id"])
    hidden_rows.sort(key=lambda row: row["sample_id"])
    paths = sample_paths(args.experiment_root, args.sample_id)
    write_jsonl(paths["runner_manifest"], runner_rows)
    write_jsonl(paths["evaluator_allocation"], allocation_rows)
    write_jsonl(paths["hidden_targets"], hidden_rows)
    method_evaluation_ids = {
        "schema_version": 1,
        "sample_set": args.sample_id,
        "allowed_research_roles": [
            "method_evaluation",
            "cross_author_method_evaluation",
        ],
        "sample_ids": sorted(
            row["sample_id"]
            for row in allocation_rows
            if row["research_role"]
            in {"method_evaluation", "cross_author_method_evaluation"}
        ),
    }
    method_evaluation_ids["sample_count"] = len(
        method_evaluation_ids["sample_ids"]
    )
    write_json(paths["method_evaluation_ids"], method_evaluation_ids)

    screening_rows: list[dict[str, Any]] = []
    own_books = sorted(
        {
            row["book_title"]
            for row in allocation_rows
            if row["research_role"] == "method_evaluation"
        }
    )
    for title in own_books:
        candidates = [
            row
            for row in allocation_rows
            if row["research_role"] == "method_evaluation"
            and row["book_title"] == title
        ]
        candidates.sort(
            key=lambda row: stable_seed(
                args.seed, f"screening:{title}:{row['sample_id']}"
            )
        )
        chosen: list[dict[str, Any]] = []
        while candidates and len(chosen) < 3:
            best = max(
                candidates,
                key=lambda row: (
                    sum(
                        row["strata"][field]
                        not in {picked["strata"][field] for picked in chosen}
                        for field in (
                            "scene_type",
                            "flow_bucket",
                            "dialogue_bucket",
                            "position_bucket",
                        )
                    ),
                    -stable_seed(
                        args.seed,
                        f"screening:{title}:{row['sample_id']}",
                    ),
                ),
            )
            chosen.append(best)
            candidates.remove(best)
        screening_rows.extend(chosen)
    for author in COMPARISON_AUTHORS:
        candidates = [
            row
            for row in allocation_rows
            if row["research_role"] == "cross_author_method_evaluation"
            and row["author"] == author
        ]
        candidates.sort(
            key=lambda row: stable_seed(
                args.seed, f"screening:{author}:{row['sample_id']}"
            )
        )
        screening_rows.append(candidates[0])
    screening_ids = {
        "schema_version": 1,
        "selection_id": "screening_v1",
        "sample_set": args.sample_id,
        "seed": args.seed,
        "design": {
            "own_author_chunks_per_book": 3,
            "cross_author_chunks_per_author": 1,
            "purpose": "method_family_and_intensity_screening_only",
        },
        "sample_count": len(screening_rows),
        "sample_ids": sorted(row["sample_id"] for row in screening_rows),
    }
    write_json(paths["screening_ids"], screening_ids)
    confirmation_ids = {
        "schema_version": 1,
        "selection_id": "confirmation_v1",
        "sample_set": args.sample_id,
        "excluded_selection_id": "screening_v1",
        "purpose": "post_screen_method_confirmation",
        "sample_ids": sorted(
            set(method_evaluation_ids["sample_ids"])
            - set(screening_ids["sample_ids"])
        ),
    }
    confirmation_ids["sample_count"] = len(confirmation_ids["sample_ids"])
    write_json(paths["confirmation_ids"], confirmation_ids)
    paths["evaluator_allocation"].chmod(0o600)
    paths["hidden_targets"].chmod(0o600)

    def count_field(field: str) -> dict[str, int]:
        if field in STRATIFICATION_FIELDS:
            values = (row["strata"][field] for row in allocation_rows)
        else:
            values = (row[field] for row in allocation_rows)
        return dict(sorted(Counter(values).items()))

    quality_flags = Counter(
        flag for row in allocation_rows for flag in row.get("quality_flags", [])
    )
    comparison_books_by_author = {
        author: sorted(
            {
                row["book_title"]
                for row in allocation_rows
                if row["author"] == author
            }
        )
        for author in COMPARISON_AUTHORS
    }
    spacing_violations = 0
    by_book: dict[str, list[int]] = defaultdict(list)
    for row in allocation_rows:
        by_book[f"{row['author']}::{row['book_title']}"] .append(row["chunk_index"])
    for indices in by_book.values():
        ordered = sorted(indices)
        spacing_violations += sum(
            right - left < args.min_chunk_distance
            for left, right in zip(ordered, ordered[1:])
        )

    summary = {
        "schema_version": 2,
        "sample_set": args.sample_id,
        "seed": args.seed,
        "target_author": args.target_author,
        "chunks_per_book": args.chunks_per_book,
        "calibration_per_book": args.calibration_per_book,
        "comparison_chunks_per_author": args.comparison_chunks_per_author,
        "min_chunk_distance": args.min_chunk_distance,
        "total_samples": len(runner_rows),
        "own_author_samples": sum(
            row["benchmark_arm"] == "own_author_reconstruction"
            for row in allocation_rows
        ),
        "cross_author_samples": sum(
            row["benchmark_arm"] == "cross_author_transfer"
            for row in allocation_rows
        ),
        "method_evaluation_samples": sum(
            row["research_role"] == "method_evaluation"
            for row in allocation_rows
        ),
        "cross_author_method_evaluation_samples": sum(
            row["research_role"] == "cross_author_method_evaluation"
            for row in allocation_rows
        ),
        "style_meter_calibration_samples": sum(
            row["research_role"] == "style_meter_calibration"
            for row in allocation_rows
        ),
        "development_books": {
            split_name: list(titles)
            for split_name, titles in DEVELOPMENT_BOOKS.items()
        },
        "comparison_authors": list(COMPARISON_AUTHORS),
        "comparison_books_by_author": comparison_books_by_author,
        "final_validation_books_reserved": list(FINAL_VALIDATION_BOOKS),
        "counts": {
            "by_arm": count_field("benchmark_arm"),
            "by_book": count_field("book_title"),
            "by_author": count_field("author"),
            "by_source_split": count_field("source_split"),
            "by_research_role": count_field("research_role"),
            "by_position": count_field("position_bucket"),
            "by_scene": count_field("scene_type"),
            "by_length": count_field("length_bucket"),
            "by_punctuation": count_field("punctuation_bucket"),
            "by_flow": count_field("flow_bucket"),
            "by_dialogue": count_field("dialogue_bucket"),
            "quality_flag_provenance": dict(sorted(quality_flags.items())),
        },
        "selected_sample_residue_hits": sum(
            bool(row["selected_residue_hits"]) for row in allocation_rows
        ),
        "spacing_violations": spacing_violations,
        "inputs": {
            "clean_chunks": relative_artifact_path(args.clean_chunks),
            "masked_chunks": relative_artifact_path(args.masked_chunks),
            "splits": relative_artifact_path(args.splits),
        },
        "outputs": {
            key: relative_artifact_path(path) for key, path in paths.items()
        },
        "runner_manifest_sha256": rows_sha256(runner_rows),
        "evaluator_allocation_sha256": rows_sha256(allocation_rows),
        "hidden_targets_sha256": rows_sha256(hidden_rows),
        "method_evaluation_ids_sha256": file_sha256(
            paths["method_evaluation_ids"]
        ),
        "screening_ids_sha256": file_sha256(paths["screening_ids"]),
        "confirmation_ids_sha256": file_sha256(paths["confirmation_ids"]),
        "leakage_controls": {
            "opaque_sample_ids": True,
            "runner_manifest_contains_target_metadata": False,
            "target_derived_strata_evaluator_only": True,
            "eternal_gate_excluded": True,
            "final_test_books_excluded": True,
            "hidden_targets_separate_from_runner_manifest": True,
            "neutral_and_style_runners_may_read_hidden_targets": False,
        },
    }
    write_json(paths["summary"], summary)
    return summary


def initialize_prompts(experiment_root: Path) -> dict[str, str]:
    prompts_dir = experiment_root / "prompts"
    registry_dir = experiment_root / "method_registry"
    protocol_dir = experiment_root / "protocols"
    schemas_dir = experiment_root / "schemas"
    method_config_dir = registry_dir / "methods"
    prompt_paths = {
        "english_semantic_source": prompts_dir / "english_semantic_source.v1.md",
        "english_source_qa": prompts_dir / "english_source_qa.v1.md",
        "english_source_repair": prompts_dir / "english_source_repair.v1.md",
        "neutral_translation": prompts_dir / "neutral_translation.v1.md",
        "style_transfer": prompts_dir / "style_transfer_method.v1.md",
        "style_transfer_critique": prompts_dir
        / "style_transfer_critique.v1.md",
    }
    prompt_paths["english_semantic_source"].parent.mkdir(parents=True, exist_ok=True)
    prompt_paths["english_semantic_source"].write_text(
        ENGLISH_SEMANTIC_SOURCE_PROMPT, encoding="utf-8"
    )
    prompt_paths["english_source_qa"].write_text(
        ENGLISH_SOURCE_QA_PROMPT, encoding="utf-8"
    )
    prompt_paths["english_source_repair"].write_text(
        ENGLISH_SOURCE_REPAIR_PROMPT, encoding="utf-8"
    )
    prompt_paths["neutral_translation"].write_text(
        NEUTRAL_TRANSLATION_PROMPT, encoding="utf-8"
    )
    prompt_paths["style_transfer"].write_text(
        STYLE_TRANSFER_PROMPT, encoding="utf-8"
    )
    prompt_paths["style_transfer_critique"].write_text(
        STYLE_TRANSFER_CRITIQUE_PROMPT, encoding="utf-8"
    )
    prompt_hashes = {
        key: file_sha256(path) for key, path in prompt_paths.items()
    }

    def translation_schema(content_field: str, prompt_version: str) -> dict[str, Any]:
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "required": [
                "sample_id",
                "prompt_version",
                "paragraphs",
                "uncertainties",
            ],
            "properties": {
                "sample_id": {"type": "string", "pattern": "^s_[0-9a-f]{24}$"},
                "prompt_version": {"type": "string", "const": prompt_version},
                "paragraphs": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["id", content_field],
                        "properties": {
                            "id": {"type": "string", "pattern": "^p[0-9]{4}$"},
                            content_field: {"type": "string", "minLength": 1},
                        },
                    },
                },
                "uncertainties": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        }

    english_schema_path = (
        schemas_dir / "english_semantic_source_output.v1.schema.json"
    )
    neutral_schema_path = (
        schemas_dir / "neutral_translation_output.v1.schema.json"
    )
    english_qa_schema_path = schemas_dir / "english_source_qa_output.v1.schema.json"
    english_repair_schema_path = (
        schemas_dir / "english_source_repair_output.v1.schema.json"
    )
    style_schema_path = schemas_dir / "style_transfer_output.v1.schema.json"
    style_critique_schema_path = (
        schemas_dir / "style_transfer_critique_output.v1.schema.json"
    )
    write_json(
        english_schema_path,
        translation_schema("en", "english_semantic_source.v1"),
    )
    write_json(
        neutral_schema_path,
        translation_schema("zh", "neutral_translation.v1"),
    )
    write_json(
        english_repair_schema_path,
        translation_schema("en", "english_source_repair.v1"),
    )
    write_json(
        english_qa_schema_path,
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "required": [
                "sample_id",
                "prompt_version",
                "approved",
                "paragraph_reviews",
                "overall_issues",
            ],
            "properties": {
                "sample_id": {"type": "string", "pattern": "^s_[0-9a-f]{24}$"},
                "prompt_version": {
                    "type": "string",
                    "const": "english_source_qa.v1",
                },
                "approved": {"type": "boolean"},
                "paragraph_reviews": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["id", "status", "issues"],
                        "properties": {
                            "id": {"type": "string", "pattern": "^p[0-9]{4}$"},
                            "status": {
                                "type": "string",
                                "enum": ["pass", "fail"],
                            },
                            "issues": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    },
                },
                "overall_issues": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
    )
    write_json(
        style_schema_path,
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "required": [
                "sample_id",
                "method_id",
                "intensity",
                "paragraphs",
                "style_cues_applied",
                "style_cues_skipped",
                "uncertainties",
            ],
            "properties": {
                "sample_id": {"type": "string", "pattern": "^s_[0-9a-f]{24}$"},
                "method_id": {"type": "string"},
                "intensity": {
                    "type": "string",
                    "enum": ["none", "light", "medium", "strong"],
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
                "style_cues_applied": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "style_cues_skipped": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "uncertainties": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
    )
    write_json(
        style_critique_schema_path,
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "required": [
                "sample_id",
                "prompt_version",
                "repair_required",
                "neutral_paragraph_reviews",
                "paragraph_reviews",
                "overall_risks",
            ],
            "properties": {
                "sample_id": {"type": "string", "pattern": "^s_[0-9a-f]{24}$"},
                "prompt_version": {
                    "type": "string",
                    "const": "style_transfer_critique.v1",
                },
                "repair_required": {"type": "boolean"},
                "neutral_paragraph_reviews": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["id", "status", "fidelity_issues"],
                        "properties": {
                            "id": {"type": "string", "pattern": "^p[0-9]{4}$"},
                            "status": {
                                "type": "string",
                                "enum": ["pass", "minor", "major"],
                            },
                            "fidelity_issues": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    },
                },
                "paragraph_reviews": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "id",
                            "status",
                            "fidelity_issues",
                            "readability_issues",
                            "repair_instructions",
                        ],
                        "properties": {
                            "id": {"type": "string", "pattern": "^p[0-9]{4}$"},
                            "status": {
                                "type": "string",
                                "enum": ["pass", "minor", "major"],
                            },
                            "fidelity_issues": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "readability_issues": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "repair_instructions": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    },
                },
                "overall_risks": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
    )
    model_run_config = {
        "schema_version": 1,
        "runner": "codex_exec_external_seatbelt_v2",
        "codex_model": "gpt-5.4",
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
            "english_semantic_source": {
                "version": "english_semantic_source.v1",
                "sha256": prompt_hashes["english_semantic_source"],
            },
            "english_source_qa": {
                "version": "english_source_qa.v1",
                "sha256": prompt_hashes["english_source_qa"],
            },
            "english_source_repair": {
                "version": "english_source_repair.v1",
                "sha256": prompt_hashes["english_source_repair"],
            },
            "neutral_translation": {
                "version": "neutral_translation.v1",
                "sha256": prompt_hashes["neutral_translation"],
                "production_contract": "eternal_gate_pass_1",
            },
            "style_transfer": {
                "version": "style_transfer_method.v1",
                "sha256": prompt_hashes["style_transfer"],
            },
            "style_transfer_critique": {
                "version": "style_transfer_critique.v1",
                "sha256": prompt_hashes["style_transfer_critique"],
            },
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
            "codex_cli_version",
            "codex_binary_realpath",
            "provider",
            "prompt_sha256",
            "input_sha256",
            "glossary_sha256",
            "output_sha256",
            "schema_sha256",
            "response_file",
            "response_id",
            "response_usage",
            "response_errors",
            "runner_sha256",
            "environment_sha256",
            "sandbox_profile_sha256",
            "effective_command_sha256",
            "validation_errors",
        ],
    }
    run_config_path = protocol_dir / "model_run_config.v1.json"
    write_json(run_config_path, model_run_config)

    evaluation_protocol = {
        "schema_version": 1,
        "style_meter_status": "provisional_pending_generated_domain_calibration",
        "style_meter": {
            "model": CURRENT_SCORER_ID,
            "input_view": CURRENT_MASKED_VIEW,
            "reported_outputs": [
                "target_decision_margin",
                "target_rank",
                "target_chunk_share",
                "paired_margin_lift_over_neutral",
            ],
        },
        "staged_execution": {
            "screening": {
                "selection_file": (
                    "sample_sets/development_proxy_v1.screening_v1_ids.json"
                ),
                "sample_count": 36,
                "composition": (
                    "Three method-evaluation chunks from each of eight target "
                    "books and one chunk from each of 12 cross-author books."
                ),
                "initial_method_intensities": {
                    "neutral_only": "none",
                    "generic_author_style_light": "light",
                    "generic_author_style_strong": "strong",
                    "global_style_cards": "medium",
                    "scene_routed_style_cards": "medium",
                    "sentence_flow_cards": "medium",
                    "function_word_punctuation_dialogue_cards": "medium",
                    "retrieved_examples_only": "medium",
                    "scene_cards_plus_examples": "medium",
                    "contrastive_examples": "medium",
                    "llm_close_reading_style_definition": "medium",
                    "llm_close_reading_cards_statistical_gates": "medium",
                },
                "promotion_rule": (
                    "Retain at most three non-control methods with positive paired "
                    "target-margin lift in both arms, no deterministic hard-fidelity "
                    "failure rate above 0.10, and no evidence of reference copying. "
                    "Break ties by own-author mean lift, then cross-author mean lift."
                ),
                "interpretation": (
                    "Method-family screening only; confidence intervals from these "
                    "36 rows are descriptive and cannot satisfy the 80% endpoint."
                ),
                "selection_enforcement": (
                    "The evaluator must bind the exact frozen path, SHA-256, 24/12 "
                    "arm counts, eight-book composition, and 12-author composition."
                ),
                "roster_enforcement": (
                    "Initial screening must contain exactly every listed "
                    "method/intensity combination; omissions and additions hard-fail."
                ),
                "promotion_artifact": (
                    "promotions/screening_v1.promoted_methods.v1.json"
                ),
            },
            "intensity_and_repair": {
                "rule": (
                    "Freeze a provisional deterministic shortlist of at most three "
                    "methods, compare registered lighter intensities on the same "
                    "screening set, then produce independent blind English-grounded "
                    "critiques. Final promotion requires semantic noninferiority and "
                    "zero high-severity readability failures in each arm. Run "
                    "self_critique_repair only after freezing the best prior method."
                ),
                "provisional_shortlist_artifact": (
                    "promotions/screening_v1.provisional_shortlist.v1.json"
                ),
                "refinement_contract_artifact": (
                    "promotions/screening_v1.refinement_contract.v1.json"
                ),
                "derived_arm_identity": (
                    "self_critique_repair binds one rank-1 base method, base "
                    "intensity, frozen critique, repair intensity, and selection."
                ),
            },
            "confirmation": {
                "selection_file": (
                    "sample_sets/development_proxy_v1.confirmation_v1_ids.json"
                ),
                "sample_count": 152,
                "own_author_rows": 104,
                "cross_author_rows": 48,
                "screening_overlap": 0,
                "selection_bias_control": (
                    "Only frozen promoted method/intensity combinations enter "
                    "confirmation; screening rows are reported separately."
                ),
                "admission_control": (
                    "The exact frozen promotion artifact and confirmation selection "
                    "hash are mandatory; arbitrary or partial subsets are descriptive only."
                ),
            },
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
                "with positive sensitivity >=0.80 and neutral false-positive rate "
                "<=0.10; choose the threshold with highest balanced accuracy, then "
                "the higher threshold on ties. If no threshold qualifies, binary "
                "style success is undefined and method selection is blocked."
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
                "At least 0.80 style-success point estimate over the 104 confirmation rows, "
                "at least 0.70 in every development book, Wilson 95% lower bound "
                ">=0.70, a 10,000-resample book-cluster bootstrap lower bound >=0.70, "
                "and a paired book-cluster bootstrap 95% confidence interval for "
                "margin lift excluding zero."
            ),
            "cross_author_transfer": (
                "Report separately; at least 0.80 style-success point estimate over "
                "48 confirmation rows, Wilson 95% lower bound >=0.70, positive paired lift in at "
                "least 10 of 12 authors, author-cluster bootstrap lower bound >=0.70, "
                "and no pooled result with the own-author arm."
            ),
            "semantic_noninferiority": (
                "No increase in paired high-severity semantic failure versus neutral: "
                "report candidate-only and neutral-only discordant counts and require "
                "the 95% cluster-bootstrap upper bound of candidate-minus-neutral "
                "failure rate to be <=0. Missing outputs count as failures."
            ),
            "readability_noninferiority": (
                "No high-severity readability regression versus the neutral draft "
                "under the blind independent critique; missing outputs count as failures."
            ),
        },
        "final_validation": {
            "books": list(FINAL_VALIDATION_BOOKS),
            "sample_count": 80,
            "chunks_per_book": 20,
            "one_time_only": True,
            "creation_gate": (
                "Reserved sample artifacts are created only after a confirmation "
                "method passes final judged endpoints and is locked."
            ),
            "lock_artifact": "final_validation/final_validation_v1.lock.json",
            "pre_reveal_lock_artifact": (
                "final_validation/final_validation_v1.pre_reveal.lock.json"
            ),
            "pre_reveal_binding_scope": (
                "All upstream prompts, schemas, model config, runner, evaluator, "
                "payload builder, corpus views, splits, promotion, and confirmation."
            ),
            "same_frozen_threshold_prompt_model_and_method_payload": True,
            "required_own_author_style_success_point_estimate": 0.80,
            "required_cluster_bootstrap_lower_bound": 0.70,
        },
    }
    evaluation_protocol_path = protocol_dir / "evaluation_protocol.v1.json"
    write_json(evaluation_protocol_path, evaluation_protocol)

    frozen_methods: list[dict[str, Any]] = []
    method_config_dir.mkdir(parents=True, exist_ok=True)
    for method in METHOD_REGISTRY:
        uses_retrieval = method["id"] in {
            "retrieved_examples_only",
            "scene_cards_plus_examples",
            "contrastive_examples",
        }
        config = {
            "schema_version": 1,
            "method_id": method["id"],
            "label": method["label"],
            "family": method["family"],
            "status": method["status"],
            "hypothesis": method["hypothesis"],
            "intensities": method["intensities"],
            "prompt_version": (
                "neutral_translation.v1"
                if method["id"] == "neutral_only"
                else "style_transfer_method.v1"
            ),
            "model_config": "protocols/model_run_config.v1.json",
            "input_contract": {
                "english_and_neutral_inputs_must_be_byte_identical_across_methods": True,
                "scene_routing_source": "neutral_zh_only",
                "target_derived_allocation_metadata_available_to_runner": False,
            },
            "payload_builder": {
                "id": f"{method['id']}.payload.v1",
                "source_corpus": "target_author_train_books_only",
                "development_proxy_and_test_books_excluded": True,
                "asset_hash_required_before_status_can_change_to_ready": True,
            },
            "retrieval": (
                {
                    "enabled": True,
                    "query": "neutral_zh",
                    "index_view": "entity_masked_v3",
                    "return_view": "entity_masked_v3",
                    "source_split": "train",
                    "k": 3,
                    "max_cjk_per_example": 450,
                    "near_duplicate_8gram_jaccard_max": 0.20,
                    "evaluation_books_excluded": True,
                }
                if uses_retrieval
                else {"enabled": False}
            ),
            "generation": {
                "one_output_per_sample_and_intensity": True,
                "max_generation_attempts": 2,
                "max_repair_attempts": 1
                if method["id"] == "self_critique_repair"
                else 0,
                "output_schema": "schemas/style_transfer_output.v1.schema.json",
            },
            "planned_comparisons": [
                "paired_vs_neutral_only",
                "paired_vs_generic_author_style_light",
                "own_author_and_cross_author_arms_reported_separately",
            ],
        }
        config_path = method_config_dir / f"{method['id']}.v1.json"
        write_json(config_path, config)
        frozen_methods.append(
            {
                **method,
                "config_path": relative_artifact_path(config_path),
                "config_sha256": file_sha256(config_path),
            }
        )
    registry = {
        "schema_version": 2,
        "target_author": DEFAULT_TARGET_AUTHOR,
        "prompt_hashes": prompt_hashes,
        "model_run_config": {
            "path": relative_artifact_path(run_config_path),
            "sha256": file_sha256(run_config_path),
        },
        "evaluation_protocol": {
            "path": relative_artifact_path(evaluation_protocol_path),
            "sha256": file_sha256(evaluation_protocol_path),
        },
        "methods": frozen_methods,
        "iteration_rule": {
            "shortlist_requires": [
                "positive_style_lift_over_neutral",
                "no_increase_in_high_severity_semantic_errors",
                "no_high_severity_readability_regression",
                "no_copy_gate_pass",
                "independent_agent_evaluation",
            ],
            "stop_condition": (
                "At least 80% of method-evaluation chunks pass the calibrated "
                "style-success gate and all hard fidelity gates, with the result "
                "replicated on final-validation books."
            ),
            "failure_action": (
                "If registered methods fail, use academic literature research to "
                "add a preregistered method family, then run a new numbered iteration."
            ),
        },
    }
    registry_path = registry_dir / "style_methods.v1.json"
    write_json(registry_path, registry)
    return {
        **{
            key: relative_artifact_path(path) for key, path in prompt_paths.items()
        },
        "method_registry": relative_artifact_path(registry_path),
        "model_run_config": relative_artifact_path(run_config_path),
        "evaluation_protocol": relative_artifact_path(evaluation_protocol_path),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def validate_artifacts(experiment_root: Path, sample_id: str) -> dict[str, Any]:
    paths = sample_paths(experiment_root, sample_id)
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    runner = read_jsonl(paths["runner_manifest"])
    allocation = read_jsonl(paths["evaluator_allocation"])
    hidden = read_jsonl(paths["hidden_targets"])
    method_evaluation_ids = json.loads(
        paths["method_evaluation_ids"].read_text(encoding="utf-8")
    )
    screening_ids = json.loads(
        paths["screening_ids"].read_text(encoding="utf-8")
    )
    confirmation_ids = json.loads(
        paths["confirmation_ids"].read_text(encoding="utf-8")
    )
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))

    runner_ids = [row["sample_id"] for row in runner]
    allocation_ids = [row["sample_id"] for row in allocation]
    hidden_ids = [row["sample_id"] for row in hidden]
    errors: list[str] = []
    if len(runner_ids) != len(set(runner_ids)):
        errors.append("runner manifest has duplicate sample IDs")
    if len(allocation_ids) != len(set(allocation_ids)):
        errors.append("evaluator allocation has duplicate sample IDs")
    if len(hidden_ids) != len(set(hidden_ids)):
        errors.append("hidden targets have duplicate sample IDs")
    if not set(runner_ids) == set(allocation_ids) == set(hidden_ids):
        errors.append("runner, allocation, and hidden-target sample IDs differ")
    allowed_runner_keys = {
        "sample_id",
        "sample_set",
        "stage_status",
        "artifact_paths",
    }
    for row in runner:
        if set(row) != allowed_runner_keys:
            errors.append("runner manifest contains target-derived metadata")
            break
        if not re.fullmatch(r"s_[0-9a-f]{24}", row["sample_id"]):
            errors.append("runner manifest contains a non-opaque sample ID")
            break
    if any(row["book_title"] in FINAL_VALIDATION_BOOKS for row in allocation):
        errors.append("allocation includes a reserved final-validation book")
    if any(row["selected_residue_hits"] for row in allocation):
        errors.append("selected clean chunks contain residue-pattern hits")
    if summary["spacing_violations"]:
        errors.append("selected chunks violate the minimum within-book distance")
    if summary["total_samples"] != len(runner):
        errors.append("summary total does not match runner manifest")
    if rows_sha256(runner) != summary["runner_manifest_sha256"]:
        errors.append("runner-manifest SHA-256 does not match summary")
    if (
        rows_sha256(allocation)
        != summary["evaluator_allocation_sha256"]
    ):
        errors.append("evaluator-allocation SHA-256 does not match summary")
    if rows_sha256(hidden) != summary["hidden_targets_sha256"]:
        errors.append("hidden-target SHA-256 does not match summary")
    expected_method_ids = sorted(
        row["sample_id"]
        for row in allocation
        if row["research_role"]
        in {"method_evaluation", "cross_author_method_evaluation"}
    )
    if method_evaluation_ids.get("sample_ids") != expected_method_ids:
        errors.append("method-evaluation ID set disagrees with allocation")
    if method_evaluation_ids.get("sample_count") != len(expected_method_ids):
        errors.append("method-evaluation ID count is incorrect")
    if file_sha256(paths["method_evaluation_ids"]) != summary.get(
        "method_evaluation_ids_sha256"
    ):
        errors.append("method-evaluation ID SHA-256 does not match summary")
    screening_set = screening_ids.get("sample_ids", [])
    if len(screening_set) != 36 or len(screening_set) != len(set(screening_set)):
        errors.append("screening set must contain 36 unique samples")
    if not set(screening_set).issubset(expected_method_ids):
        errors.append("screening set contains non-evaluation samples")
    if file_sha256(paths["screening_ids"]) != summary.get(
        "screening_ids_sha256"
    ):
        errors.append("screening ID SHA-256 does not match summary")
    confirmation_set = confirmation_ids.get("sample_ids", [])
    if len(confirmation_set) != 152 or len(confirmation_set) != len(
        set(confirmation_set)
    ):
        errors.append("confirmation set must contain 152 unique samples")
    if set(screening_set) & set(confirmation_set):
        errors.append("screening and confirmation sets overlap")
    if set(screening_set) | set(confirmation_set) != set(expected_method_ids):
        errors.append("screening and confirmation sets do not partition evaluation")
    if file_sha256(paths["confirmation_ids"]) != summary.get(
        "confirmation_ids_sha256"
    ):
        errors.append("confirmation ID SHA-256 does not match summary")
    expected = sum(len(titles) for titles in DEVELOPMENT_BOOKS.values())
    expected *= int(summary["chunks_per_book"])
    expected += len(COMPARISON_AUTHORS) * int(
        summary["comparison_chunks_per_author"]
    )
    if len(runner) != expected:
        errors.append(f"expected {expected} samples, found {len(runner)}")
    if paths["evaluator_allocation"].stat().st_mode & 0o077:
        errors.append("evaluator allocation permissions are broader than 0600")
    if paths["hidden_targets"].stat().st_mode & 0o077:
        errors.append("hidden-target permissions are broader than 0600")

    required_files = (
        experiment_root / "prompts/english_semantic_source.v1.md",
        experiment_root / "prompts/english_source_qa.v1.md",
        experiment_root / "prompts/english_source_repair.v1.md",
        experiment_root / "prompts/neutral_translation.v1.md",
        experiment_root / "prompts/style_transfer_method.v1.md",
        experiment_root / "prompts/style_transfer_critique.v1.md",
        experiment_root / "method_registry/style_methods.v1.json",
        experiment_root / "protocols/model_run_config.v1.json",
        experiment_root / "protocols/evaluation_protocol.v1.json",
        experiment_root
        / "schemas/english_semantic_source_output.v1.schema.json",
        experiment_root / "schemas/english_source_qa_output.v1.schema.json",
        experiment_root / "schemas/english_source_repair_output.v1.schema.json",
        experiment_root / "schemas/neutral_translation_output.v1.schema.json",
        experiment_root / "schemas/style_transfer_output.v1.schema.json",
        experiment_root / "schemas/style_transfer_critique_output.v1.schema.json",
    )
    missing_files = [relative_artifact_path(path) for path in required_files if not path.exists()]
    if missing_files:
        errors.append(f"missing prompt or registry artifacts: {missing_files}")
    if errors:
        raise ValueError("Artifact validation failed: " + "; ".join(errors))
    return {
        "status": "passed",
        "sample_set": sample_id,
        "samples": len(runner),
        "runner_manifest_sha256": summary["runner_manifest_sha256"],
        "evaluator_allocation_sha256": summary[
            "evaluator_allocation_sha256"
        ],
        "hidden_targets_sha256": summary["hidden_targets_sha256"],
        "method_evaluation_ids_sha256": summary[
            "method_evaluation_ids_sha256"
        ],
        "screening_ids_sha256": summary["screening_ids_sha256"],
        "confirmation_ids_sha256": summary["confirmation_ids_sha256"],
        "target_metadata_in_runner_manifest": 0,
        "hidden_target_leakage": 0,
        "reserved_test_book_leakage": 0,
        "selected_residue_hits": 0,
        "spacing_violations": 0,
    }


def markdown_table(rows: Sequence[Sequence[Any]], headers: Sequence[str]) -> str:
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    output.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(output)


def bar(value: int, maximum: int, width: int = 24) -> str:
    filled = round(width * value / max(maximum, 1))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def distribution_table(values: dict[str, int]) -> str:
    maximum = max(values.values(), default=1)
    rows = [(key, value, bar(value, maximum)) for key, value in values.items()]
    return markdown_table(rows, ("Category", "Chunks", "Relative distribution"))


def load_evaluation(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def render_report(
    experiment_root: Path,
    sample_id: str,
    report_path: Path,
    evaluation_path: Path | None = None,
) -> None:
    paths = sample_paths(experiment_root, sample_id)
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    registry_path = experiment_root / "method_registry/style_methods.v1.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    run_config_path = experiment_root / "protocols/model_run_config.v1.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    evaluation_protocol_path = experiment_root / "protocols/evaluation_protocol.v1.json"
    evaluation = load_evaluation(evaluation_path)
    authorship_payload = json.loads(
        CURRENT_SCORER_VALIDATION_RESULT.read_text(encoding="utf-8")
    )
    authorship_result = authorship_payload["results"][
        CURRENT_SCORER_VALIDATION_RUN_KEY
    ]
    authorship_test = authorship_result["chunk_metrics"]["test"]
    authorship_accuracy = authorship_test["accuracy"] * 100
    authorship_balanced_accuracy = authorship_test["balanced_accuracy"] * 100

    if evaluation and "reviews" in evaluation:
        reviews = evaluation["reviews"]
    elif evaluation:
        reviews = [evaluation]
    else:
        reviews = []
    current_review = reviews[-1] if reviews else None
    current_verdict = current_review.get("verdict") if current_review else "pending"
    generation_status = (
        "ready"
        if current_verdict in {"approved", "approved_with_changes"}
        else "blocked"
    )

    status_rows = [
        ("Corpus cleaning and book-level splits", "complete", "Existing reproducible corpus"),
        (
            "Masked author-style classifier",
            "complete",
            f"{authorship_accuracy:.1f}% accuracy / {authorship_balanced_accuracy:.1f}% balanced",
        ),
        ("Generated-domain style-meter calibration", "pending", "32 rows reserved"),
        ("Development sample freeze", "complete", f"{summary['total_samples']} chunks, two arms"),
        ("Runner/evaluator target isolation", "complete", "Opaque IDs and evaluator-only files"),
        ("Child-process read isolation", "complete", "Seatbelt deny-repository canary passes"),
        ("Versioned prompt contracts", "complete", "Four v1 prompt templates"),
        ("Generation runner and ledger", "complete", "Strict provenance validation and quarantine"),
        ("One-sample isolated pipeline pilot", "complete", "English, source QA, and neutral pass"),
        ("Method registry", "complete", f"{len(registry['methods'])} versioned method configs"),
        ("Independent setup review", current_verdict, "Latest external evaluation below"),
        ("English semantic source generation", generation_status, "Pilot only; full batch not started"),
        ("Neutral translation generation", "pending", "Pilot only; batch awaits English QA"),
        ("Style-transfer experiments", "pending", "No method may be selected yet"),
        ("Final held-out validation", "locked", "Four test books reserved"),
    ]
    method_rows = [
        (
            method["priority"],
            f"`{method['id']}`",
            method["family"],
            ", ".join(method["intensities"]),
            method["status"],
        )
        for method in registry["methods"]
    ]
    book_rows = [
        (book, summary["counts"]["by_book"][book], split_name)
        for split_name, books in DEVELOPMENT_BOOKS.items()
        for book in books
    ]
    comparison_rows = [
        (author, ", ".join(summary["comparison_books_by_author"][author]), summary["comparison_chunks_per_author"])
        for author in COMPARISON_AUTHORS
    ]
    final_rows = [(book, "test", "untouched") for book in FINAL_VALIDATION_BOOKS]

    if reviews:
        review_sections: list[str] = []
        for index, review in enumerate(reviews, start=1):
            findings = review.get("findings", [])
            review_sections.extend(
                [
                    f"### Review {index}: {review.get('verdict', 'not stated')}",
                    "",
                    f"Evaluator: {review.get('evaluator', 'independent subagent')}  ",
                    f"Reviewed at: {review.get('reviewed_at', 'not stated')}",
                    "",
                    *[
                        f"- **{finding.get('severity', 'note').upper()}**: "
                        f"{finding.get('finding', '')}"
                        for finding in findings
                    ],
                    "",
                    f"Recommendation: {review.get('recommendation', 'not stated')}",
                    "",
                ]
            )
        evaluation_text = "\n".join(review_sections).rstrip()
    else:
        evaluation_text = (
            "Pending. A separate agent must review the frozen artifacts before any "
            "model-generated batch. Every later method iteration must receive another "
            "independent evaluation recorded here."
        )

    prompt_hash_rows = [
        (name, details["version"], details["sha256"])
        for name, details in run_config["prompt_versions"].items()
    ]

    report = f"""# Eternal Gate Textless Back-Translation: Style-Transfer Method Research

Generated from reproducible artifacts on {utc_now()}.

## Executive Status

The author-style identification prerequisite passes the 80% classifier requirement:
the current exact character n-gram classifier reaches **{authorship_accuracy:.1f}% masked test accuracy**
and **{authorship_balanced_accuracy:.1f}% balanced accuracy** across 50 authors. The unweighted exact n-gram
SGD hinge model is a **provisional proxy meter**, not yet a calibrated style-transfer
endpoint. Its generated-domain threshold must be frozen on the 32 calibration cases.

The style-transfer study itself is initialized but has **no transfer result yet**.
No method is selected, and model generation remains `{generation_status}` under the
latest independent-review verdict.

{markdown_table(status_rows, ("Research stage", "Status", "Evidence or next action"))}

## Research Questions and Hypotheses

- **RQ1:** Can a controlled style-transfer method move neutral Chinese toward the
  target author's style while preserving the English-source meaning?
- **RQ2:** Which representation carries the useful signal: generic author prompting,
  corpus-derived cards, sentence-flow controls, retrieved examples, LLM close
  reading, or a verifier loop?
- **RQ3:** Does a method generalize across books and scene types rather than recover
  book-specific content?

The primary hypothesis is that validated close-reading or scene-routed cards plus
retrieved examples will produce greater style lift than generic prompting, without
the semantic drift expected from a strong unconstrained rewrite.

## Controlled Benchmark Construction

The own-author arm starts from held-out original Chinese, generates a plain English
semantic source, translates that English with the production-compatible neutral
prompt, and freezes every candidate before the evaluator opens the target:

```text
held-out original Chinese (hidden)
  -> plain English semantic source
  -> neutral Chinese using the production pass-1 prompt family
  -> candidate style-transfer method
  -> freeze output
  -> style, semantic, readability, and no-copy evaluation
```

The separate cross-author arm applies the same target style to content from 12
other authors. It tests whether a method transfers style instead of merely
reconstructing residual content from the target author's own books. The two arms
must never be pooled into one success rate.

This design reproduces the English-to-Chinese information bottleneck that will
exist for *Eternal Gate*. Random generated prose and online fan translations are
not primary benchmark sources. *Eternal Gate* itself is excluded.

The sample seed is `{summary['seed']}`. The opaque runner-manifest hash is
`{summary['runner_manifest_sha256']}`; the evaluator-allocation hash is
`{summary['evaluator_allocation_sha256']}`; and the hidden-target hash is
`{summary['hidden_targets_sha256']}`.

The runner manifest contains only opaque IDs, stage state, and output paths. Book,
author, chunk position, sampling strata, and quality metadata are held in a `0600`
evaluator allocation file. Original and entity-masked targets are in a second
`0600` file. For each request, the parent streams only that request into a fresh
ephemeral Codex process started outside the repository. A macOS Seatbelt profile
denies repository reads and writes at the OS boundary. The preflight proves that
the child can read its temporary request area but cannot read hidden targets,
evaluator allocation, or corpus files. The neutral process has no parent code path
that opens target files. Scene routing is recomputed from neutral Chinese, never
copied from the target-derived sampling label.

### Own-author reconstruction arm

{markdown_table(book_rows, ("Book", "Chunks", "Source split"))}

These {summary['own_author_samples']} chunks contain
{summary['style_meter_calibration_samples']} style-meter calibration chunks and
{summary['method_evaluation_samples']} method-evaluation chunks. The runner cannot
see those roles; only the evaluator can. Calibration cases cannot enter method-level
hypothesis testing.

### Cross-author transfer arm

{markdown_table(comparison_rows, ("Author", "Development book", "Chunks"))}

These {summary['cross_author_samples']} cases are a separate endpoint. References
for target-style methods may come only from target-author `train` books, never from
either development arm or the final test books.

### Reserved final validation

{markdown_table(final_rows, ("Book", "Split", "Status"))}

These books remain untouched until a method and intensity are shortlisted. Final
validation should sample 20 to 30 chunks per book using the same stratification
algorithm and a new preregistered sample ID.

## Sample Composition

The selector enforces a minimum index distance of {summary['min_chunk_distance']}
within each book; observed spacing violations: {summary['spacing_violations']}.
The selected-text residue scan found {summary['selected_sample_residue_hits']} hits.
Raw-book provenance flags remain recorded for audit but do not indicate residue in
the cleaned selected chunks:

{distribution_table(summary['counts']['quality_flag_provenance'])}

### Relative book position

{distribution_table(summary['counts']['by_position'])}

### Heuristic scene type

{distribution_table(summary['counts']['by_scene'])}

Scene labels are sampling strata, not ground-truth literary annotations. Their
purpose is coverage and they stay evaluator-only. Method results must still be
reported per book and by measured flow/dialogue buckets.

### Sentence-flow bucket

{distribution_table(summary['counts']['by_flow'])}

### Dialogue-density bucket

{distribution_table(summary['counts']['by_dialogue'])}

## Neutral Translation Contract

Proxy experiments and later *Eternal Gate* pass 1 use the same versioned prompt
family: `neutral_translation.v1`. Both treat English as semantic authority, produce
natural neutral Simplified Chinese, preserve paragraph IDs and semantic details,
and prohibit author imitation, style cards, retrieved examples, and literary
embellishment. Production may add a glossary and comments, but changing the core
neutrality rules requires a new prompt version and benchmark rerun.

Runs record the declared provider, model slug, Codex CLI build and binary identity,
effective command, environment allowlist, prompt/glossary/input/output/schema hashes,
response ID and usage, retries, runner hash, dependency lock, and isolation profile.
The backend exposes neither an immutable service snapshot nor temperature through
Codex CLI, so this is explicitly **frozen-output provenance**, not a claim that a
future rerun will produce identical prose. Compare methods only when their frozen
English and neutral inputs are byte-identical.

{markdown_table(prompt_hash_rows, ("Stage", "Prompt version", "SHA-256"))}

The research neutral prompt and hash remain frozen inside this experiment root.
Production configuration is intentionally decoupled from historical experiment
validation, so an operational method change cannot rewrite or invalidate prior
research artifacts. Any research prompt rule change requires a new version and a
complete proxy rerun.

## English Source Quarantine

Every generated English source must first pass paragraph identity/order checks,
English-script and Chinese-copy checks, broad length diagnostics, and deterministic
number-surface warnings. It then enters a separate, stateless Chinese-to-English
semantic QA request that checks every paragraph. Neutral translation refuses any
sample without a successful QA artifact whose paragraph verdicts all pass. Failed
or malformed model responses are retained under `_quarantine/` with their response
ID, output hash, validation errors, and deterministic diagnostics.

The one-sample run `isolated_pipeline_pilot_v1` completed English generation,
independent source QA, and neutral translation. All three strict validators report
one valid and approved output with no provenance errors. This pilot proves the
execution path; it is not a style-transfer result and is not included in method
statistics.

## Preregistered Transfer Methods

{markdown_table(method_rows, ("Priority", "Method ID", "Family", "Intensity", "Status"))}

Each row links to a hashed method config. Retrieval is fixed at `k=3`, queries with
neutral Chinese against `entity_masked_v3`, returns bounded clean examples from
target-author `train` books only, and excludes evaluation books and near duplicates.
An asset hash must be frozen before a method changes from `not_run` to `ready`.
Failed methods stay in the report.

## Evaluation Protocol

Evaluate each candidate separately on:

1. **Style movement:** target-author rank, raw hinge decision margin, target-chunk
   share, and paired lift over neutral. Raw margins are never called probabilities.
2. **Interpretable guardrails:** sentence/paragraph flow, function-character,
   punctuation, and dialogue features. These diagnose how the score moved; they do
   not replace the stronger style meter.
3. **Semantic fidelity:** English-grounded checks for entities, numbers, roles,
   speaker turns, causality, chronology, negation, modality, intensity, omissions,
   and inventions.
4. **Readability:** natural Chinese fiction cadence, no translationese, no
   style-guide leakage, and no repetitive signature phrase stuffing.
5. **Contamination:** no unsupported reference-book concepts and no copied Chinese
   sequence of eight or more characters, except required terminology and common
   fixed expressions.

Use a 10,000-resample paired cluster bootstrap over books, Wilson intervals for
success proportions, and per-book outcomes. Eight development books, not 128
chunks, are the inferential clusters for the own-author arm. A blind LLM close-
reading judge receives randomized output order and no method ID. Deterministic
semantic and overlap failures override style gains.

### Selection gate

The calibration algorithm enumerates raw-margin thresholds on the 32 calibration
rows, requires at least 80% sensitivity on held-out originals and at most 10%
false positives on neutral drafts, maximizes balanced accuracy, and freezes the
highest threshold on ties. If no threshold qualifies, binary style success is
undefined and method selection stops.

A chunk succeeds only when its margin reaches the frozen threshold, target rank is
at most five, paired lift over neutral is positive, and no hard fidelity failure is
present. Missing output, paragraph mismatch, semantic invention/change, reference
leakage, or copied eight-character sequence is a hard failure.

Both the own-author and cross-author arms require at least 80% point success and a
Wilson 95% lower bound of at least 70%; the own-author result must also reach at
least 70% in every book. Cross-author transfer must improve at least 10 of 12
authors. The four final books receive a one-time replication with frozen settings.

If every preregistered method fails, the next iteration must use the academic
research workflow to identify another evidence-backed method family, preregister
its hypothesis and ablation, and rerun without changing earlier outputs.

## Reproduction

```bash
uv run python experiments/iteration1/style_transfer_research.py init
uv run python experiments/iteration1/style_transfer_research.py validate
uv run python experiments/iteration1/run_style_transfer_generation.py plan \\
  --stage english_semantic_source
# Run only after independent setup approval:
uv run python experiments/iteration1/run_style_transfer_generation.py run \\
  --stage english_semantic_source \\
  --run-id proxy_v1_gpt54 --jobs 4
uv run python experiments/iteration1/run_style_transfer_generation.py validate \\
  --stage english_semantic_source \\
  --run-id proxy_v1_gpt54
uv run python experiments/iteration1/run_style_transfer_generation.py run \\
  --stage english_source_qa \\
  --run-id proxy_v1_gpt54 --jobs 4
uv run python experiments/iteration1/run_style_transfer_generation.py validate \\
  --stage english_source_qa \\
  --run-id proxy_v1_gpt54
uv run python experiments/iteration1/run_style_transfer_generation.py run \\
  --stage neutral_translation \\
  --run-id proxy_v1_gpt54 --jobs 4
# After producing calibration-only style-meter score rows:
uv run python experiments/iteration1/calibrate_style_meter.py \\
  --scores generated/style_research/style_transfer_experiments/calibration/style_meter_scores.v1.jsonl
uv run python experiments/iteration1/style_transfer_research.py report \\
  --evaluation generated/style_research/style_transfer_experiments/evaluations/setup_review_history.v1.json
```

Generated artifact layout:

```text
generated/style_research/style_transfer_experiments/
  sample_sets/{sample_id}.runner_manifest.jsonl
  sample_sets/{sample_id}.evaluator_allocation.jsonl
  sample_sets/{sample_id}.hidden_targets.jsonl
  sample_sets/{sample_id}.summary.json
  prompts/english_semantic_source.v1.md
  prompts/english_source_qa.v1.md
  prompts/neutral_translation.v1.md
  prompts/style_transfer_method.v1.md
  schemas/*.schema.json
  protocols/model_run_config.v1.json
  protocols/evaluation_protocol.v1.json
  calibration/style_meter_threshold.v1.json
  method_registry/style_methods.v1.json
  method_registry/methods/*.v1.json
  runs/{sample_id}/{{run_id}}/...
  evaluations/isolated_pipeline_pilot_v1.summary.json
  evaluations/...
```

## Independent Evaluator Review

{evaluation_text}

## Current Conclusion

The setup remains **{generation_status}** for controlled English-source generation
under the latest independent verdict (`{current_verdict}`). The experiment samples
eight target-author and 12 cross-author development books while reserving four
target-author books for replication; it does not translate whole books. No style-
transfer method is currently justified for production.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.command in {"prepare-samples", "init"}:
        prepare_samples(args)
    if args.command in {"init-prompts", "init"}:
        initialize_prompts(args.experiment_root)
    if args.command in {"validate", "init"}:
        result = validate_artifacts(args.experiment_root, args.sample_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command in {"report", "init"}:
        render_report(
            args.experiment_root,
            args.sample_id,
            args.report,
            getattr(args, "evaluation", None),
        )
        print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
