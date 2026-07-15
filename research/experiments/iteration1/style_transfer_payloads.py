#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import heapq
import json
import math
import os
import re
import struct
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable, Iterator, Mapping, Sequence


from experiments.shared.paths import RESEARCH_ROOT


REPO_ROOT = RESEARCH_ROOT
TARGET_AUTHOR = "非天夜翔"
ASSET_SCHEMA = "style_transfer_payload_assets.v1"
PAYLOAD_SCHEMA = "style_transfer_method_payload.v1"
GENERATOR_ID = "experiments.iteration1.style_transfer_payloads"
GENERATOR_VERSION = 1
LOCK_NAME = "style_transfer_payloads.v1.lock.json"
ASSET_DIRECTORY = "style_transfer_payloads.v1"
CLOSE_READING_PACKET_SCHEMA = "style_transfer_close_reading_source_packet.v1"
CLOSE_READING_RESULT_SCHEMA = "style_transfer_close_reading_result.v1"
CLOSE_READING_PACKET_DIRECTORY = "close_reading_source_packets"
CLOSE_READING_RESULT_DIRECTORY = "close_reading_results"
REGISTRY_RELATIVE_PATH = Path("method_registry/style_methods.v1.json")
STYLE_PROMPT_RELATIVE_PATH = Path("prompts/style_transfer_method.v1.md")
SPLITS_RELATIVE_PATH = Path("generated/style_research/corpus/splits.json")
CLEAN_CHUNKS_RELATIVE_PATH = Path("datasets/unmasked/chunks.clean.jsonl")
MASKED_CHUNKS_RELATIVE_PATH = Path(
    "datasets/masked/chunks.entity_masked_v3.jsonl"
)

RETRIEVAL_K = 3
MAX_EXAMPLE_CJK = 450
NEAR_DUPLICATE_NGRAM = 8
NEAR_DUPLICATE_JACCARD_MAX = 0.2
HASH_VECTOR_BUCKETS = 256
HARD_NEGATIVE_CHUNKS_PER_AUTHOR = 64
EXPECTED_COMPARISON_AUTHOR_COUNT = 49

CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
SENTENCE_SPLIT_RE = re.compile(r"[。！？!?]+")
LATIN_RE = re.compile(r"[A-Za-z]+(?:[A-Za-z0-9_.:/+-]*[A-Za-z0-9])?")
NUMBER_RE = re.compile(r"\d+(?:[.,:/-]\d+)*")
WHITESPACE_RE = re.compile(r"\s+")

PUNCTUATION = {
    "punctuation.comma": "，",
    "punctuation.period": "。",
    "punctuation.semicolon": "；",
    "punctuation.colon": "：",
    "punctuation.question": "？",
    "punctuation.exclamation": "！",
    "punctuation.ellipsis": "…",
    "punctuation.dash": "—",
}
DIALOGUE_OPENERS = ("“", '"', "「", "『")
DIALOGUE_MARKS = "“”‘’「」『』"

FUNCTION_WORD_GROUPS: dict[str, tuple[str, ...]] = {
    "connective": (
        "因为",
        "所以",
        "因此",
        "但是",
        "不过",
        "然而",
        "而且",
        "并且",
        "如果",
        "虽然",
        "尽管",
        "于是",
        "然后",
        "接着",
        "随后",
        "此外",
        "另外",
        "甚至",
        "其实",
        "当然",
    ),
    "modal_aspect": (
        "已经",
        "正在",
        "曾经",
        "仍然",
        "依然",
        "终于",
        "忽然",
        "突然",
        "大概",
        "可能",
        "也许",
        "应该",
        "必须",
        "可以",
        "不会",
        "不是",
        "没有",
        "不能",
        "不要",
        "未必",
    ),
    "deictic_pronoun": (
        "这个",
        "那个",
        "这些",
        "那些",
        "这里",
        "那里",
        "这样",
        "那样",
        "这么",
        "那么",
        "怎么",
        "什么",
        "为什么",
        "哪里",
        "自己",
        "别人",
        "大家",
        "彼此",
        "对方",
    ),
    "particle_phrase": (
        "而已",
        "罢了",
        "似的",
        "一样",
        "起来",
        "下去",
        "出来",
        "进去",
        "过去",
        "过来",
        "一下",
        "一会儿",
        "是不是",
        "怎么办",
        "怎么样",
        "没什么",
    ),
    "preposition_frame": (
        "关于",
        "对于",
        "由于",
        "为了",
        "按照",
        "根据",
        "通过",
        "经过",
        "除了",
        "作为",
        "并非",
        "并不",
        "无法",
        "无论",
        "不管",
        "只要",
        "只有",
        "除非",
        "至于",
    ),
}

DIALOGUE_TERMS: dict[str, tuple[str, ...]] = {
    "dialogue.simple_speech_tags": ("说道", "说：", "道："),
    "dialogue.question_tags": ("问道", "问："),
    "dialogue.laughter_tags": ("笑道", "笑着说", "笑了起来"),
    "dialogue.silence_beats": ("没有说话", "没有回答", "沉默", "一时无言"),
    "dialogue.micro_reactions": ("点了点头", "摇了摇头", "顿了顿", "怔了怔"),
}

SCENE_CUES: dict[str, tuple[str, ...]] = {
    "action_conflict": (
        "攻击",
        "战斗",
        "冲上",
        "扑向",
        "击中",
        "撞上",
        "躲开",
        "挥剑",
        "拔刀",
        "爆炸",
        "鲜血",
        "追上",
    ),
    "dialogue": ("说道", "问道", "回答", "开口", "没有说话", "沉默"),
    "internal_reflection": (
        "心想",
        "心里",
        "想到",
        "觉得",
        "意识到",
        "明白",
        "记得",
        "仿佛",
        "似乎",
    ),
    "interpersonal_care": (
        "扶住",
        "抱住",
        "握住",
        "伤口",
        "受伤",
        "流血",
        "没事吧",
        "照顾",
        "安慰",
    ),
    "travel_transition": (
        "出发",
        "前往",
        "抵达",
        "路上",
        "离开",
        "返回",
        "穿过",
        "走进",
        "走出",
        "上车",
        "下车",
    ),
    "worldbuilding_exposition": (
        "传说",
        "据说",
        "历史",
        "规则",
        "制度",
        "意味着",
        "力量",
        "魔法",
        "世界",
        "教会",
        "王国",
    ),
}

SUPPORTED_METHOD_IDS = {
    "neutral_only",
    "generic_author_style_light",
    "generic_author_style_strong",
    "global_style_cards",
    "scene_routed_style_cards",
    "sentence_flow_cards",
    "function_word_punctuation_dialogue_cards",
    "retrieved_examples_only",
    "scene_cards_plus_examples",
    "contrastive_examples",
    "llm_close_reading_style_definition",
    "llm_close_reading_cards_statistical_gates",
    "self_critique_repair",
}


@dataclass(frozen=True)
class FeatureSpec:
    feature_id: str
    family: str
    label: str
    scale: float
    higher_instruction: str
    lower_instruction: str
    guardrail: str


FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        "flow.mean_sentence_cjk",
        "sentence_flow",
        "mean sentence length",
        1.0,
        "Where meaning permits, keep related clauses in one controlled sentence instead of fragmenting every beat.",
        "Prefer compact sentences and split only at source-preserving clause boundaries.",
        "Never change event order, causality, or emphasis merely to hit a length target.",
    ),
    FeatureSpec(
        "flow.short_sentence_ratio",
        "sentence_flow",
        "short-sentence share",
        100.0,
        "Use occasional short sentences for source-supported decisions, impacts, or replies.",
        "Avoid turning ordinary clauses into repeated clipped fragments.",
        "Shortening must not remove modality, negation, attribution, or concrete detail.",
    ),
    FeatureSpec(
        "flow.long_sentence_ratio",
        "sentence_flow",
        "long-sentence share",
        100.0,
        "Allow a longer sentence when the source presents one continuous chain of perception or action.",
        "Break overloaded sentences at natural semantic boundaries.",
        "Do not merge unrelated events or split a sentence in a way that changes scope.",
    ),
    FeatureSpec(
        "flow.mean_paragraph_cjk",
        "sentence_flow",
        "mean paragraph length",
        1.0,
        "Preserve sustained paragraphs when their clauses form one source-supported unit.",
        "Keep paragraphs focused and avoid unnecessary analytical expansion.",
        "Paragraph IDs and paragraph count remain fixed; apply this only within each paragraph.",
    ),
    FeatureSpec(
        "flow.sentences_per_paragraph",
        "sentence_flow",
        "sentences per paragraph",
        1.0,
        "Use more than one sentence inside a paragraph when this clarifies source-supported progression.",
        "Favor a tighter paragraph cadence without deleting information.",
        "Do not add sentence boundaries inside names, quotations, or fixed terminology.",
    ),
    FeatureSpec(
        "flow.commas_per_sentence",
        "sentence_flow",
        "commas per sentence",
        1.0,
        "Use comma-linked clauses for genuinely continuous action, perception, or speech beats.",
        "Reduce loosely chained commas and prefer clear sentence boundaries.",
        "Punctuation must follow semantic structure rather than imitate a numeric target.",
    ),
    FeatureSpec(
        "dialogue.line_ratio",
        "dialogue",
        "standalone dialogue-line share",
        100.0,
        "Keep direct speech turns visually direct when the source contains an actual speaker turn.",
        "Integrate brief speech with its action beat when that is clearer and source-faithful.",
        "Do not invent dialogue, change speaker attribution, or split one turn across speakers.",
    ),
    FeatureSpec(
        "dialogue.quote_marks_per_kcjk",
        "dialogue",
        "dialogue punctuation rate",
        1000.0,
        "Retain direct quotation where the source presents direct speech.",
        "Avoid redundant quotation framing around reported thought or exposition.",
        "Direct versus indirect speech is semantic and must not be changed for style alone.",
    ),
    FeatureSpec(
        "dialogue.simple_speech_tags",
        "dialogue",
        "simple speech-tag rate",
        1000.0,
        "Prefer a plain speech tag when attribution is needed and the source supplies no stronger manner.",
        "Omit repetitive speech tags when speaker identity remains unambiguous.",
        "Never add tone, motive, or emotion absent from the source.",
    ),
    FeatureSpec(
        "dialogue.question_tags",
        "dialogue",
        "question-tag rate",
        1000.0,
        "Use a concise question tag for an actual source question when attribution is needed.",
        "Let clear quoted questions stand without repetitive attribution where natural.",
        "Do not turn statements into questions or alter who asks.",
    ),
    FeatureSpec(
        "dialogue.laughter_tags",
        "dialogue",
        "laughter-tag rate",
        1000.0,
        "Use a compact laugh or smile tag only when the source already signals amusement.",
        "Keep amusement implicit when the source does not require an extra tag.",
        "Do not add warmth, flirtation, mockery, or humor.",
    ),
    FeatureSpec(
        "dialogue.silence_beats",
        "dialogue",
        "silence-beat rate",
        1000.0,
        "Render a stated pause or non-answer with a concise visible silence beat.",
        "Avoid inserting silence as interpretation when the source simply moves on.",
        "Do not infer motives or emotions for silence.",
    ),
    FeatureSpec(
        "dialogue.micro_reactions",
        "dialogue",
        "micro-reaction rate",
        1000.0,
        "Use a brief nod, pause, or visible reaction only when the source supplies it.",
        "Avoid padding dialogue with repeated nods and pauses.",
        "Do not convert internal states into visible actions.",
    ),
    FeatureSpec(
        "punctuation.comma",
        "punctuation",
        "Chinese comma rate",
        1000.0,
        "Use Chinese commas to preserve continuous clause rhythm where the semantics are linked.",
        "Prefer firmer sentence boundaries over loosely accumulated clauses.",
        "Punctuation may clarify structure but may not change scope or event relations.",
    ),
    FeatureSpec(
        "punctuation.semicolon",
        "punctuation",
        "semicolon rate",
        1000.0,
        "Use a semicolon sparingly for genuinely parallel source-supported clauses.",
        "Prefer ordinary commas or full stops unless parallel structure is explicit.",
        "Do not create rhetorical parallelism absent from the source.",
    ),
    FeatureSpec(
        "punctuation.colon",
        "punctuation",
        "colon rate",
        1000.0,
        "Use a colon for a source-supported quotation, list, or direct explanation.",
        "Avoid explanatory colons when ordinary narrative syntax is clearer.",
        "Do not turn narrative into report-like exposition.",
    ),
    FeatureSpec(
        "punctuation.question",
        "punctuation",
        "question-mark rate",
        1000.0,
        "Keep source questions direct and concise.",
        "Avoid rhetorical questions not present in the source.",
        "Question force and modality must remain unchanged.",
    ),
    FeatureSpec(
        "punctuation.exclamation",
        "punctuation",
        "exclamation-mark rate",
        1000.0,
        "Retain an exclamation only for source-supported intensity.",
        "Prefer restrained punctuation when intensity is not explicit.",
        "Never amplify emotion to satisfy a punctuation tendency.",
    ),
    FeatureSpec(
        "punctuation.ellipsis",
        "punctuation",
        "ellipsis rate",
        1000.0,
        "Use an ellipsis for an actual trailing-off, interruption, or unresolved pause.",
        "Avoid decorative or repetitive ellipses.",
        "Do not add hesitation or uncertainty absent from the source.",
    ),
    FeatureSpec(
        "punctuation.dash",
        "punctuation",
        "dash rate",
        1000.0,
        "Use a dash only for a source-supported interruption or abrupt turn.",
        "Prefer ordinary syntax when no interruption is present.",
        "Do not increase drama or speed through punctuation alone.",
    ),
    *tuple(
        FeatureSpec(
            f"function_word.{group}",
            "function_word",
            f"{group.replace('_', ' ')} function-word rate",
            1000.0,
            "Use this function-word family naturally where the source relation is explicit.",
            "Avoid leaning on this function-word family when the relation is already clear.",
            "Never insert causality, contrast, modality, sequence, or deixis absent from the source.",
        )
        for group in FUNCTION_WORD_GROUPS
    ),
)

FEATURE_SPEC_BY_ID = {spec.feature_id: spec for spec in FEATURE_SPECS}


@dataclass
class FeatureProfile:
    numerators: Counter[str] = field(default_factory=Counter)
    denominators: Counter[str] = field(default_factory=Counter)
    supports: Counter[str] = field(default_factory=Counter)
    books_by_feature: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    chunk_count: int = 0
    cjk_count: int = 0
    books: set[str] = field(default_factory=set)

    def add(
        self,
        observations: Mapping[str, tuple[float, float, int]],
        *,
        title: str,
        cjk_count: int,
    ) -> None:
        self.chunk_count += 1
        self.cjk_count += cjk_count
        self.books.add(title)
        for feature_id, (numerator, denominator, support) in observations.items():
            self.numerators[feature_id] += numerator
            self.denominators[feature_id] += denominator
            self.supports[feature_id] += support
            if support > 0:
                self.books_by_feature[feature_id].add(title)

    def value(self, spec: FeatureSpec) -> float | None:
        denominator = self.denominators.get(spec.feature_id, 0.0)
        if denominator <= 0:
            return None
        return self.numerators[spec.feature_id] / denominator * spec.scale


class MethodAssetUnavailableError(ValueError):
    """Raised when a registered method lacks leakage-safe frozen evidence."""


def _without_mutable_freeze_fields(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): child
        for key, child in value.items()
        if key not in {"status", "asset_manifest", "config_sha256"}
    }


def _registry_definition_projection(registry: Mapping[str, Any]) -> dict[str, Any]:
    projection = _without_mutable_freeze_fields(registry)
    methods = registry.get("methods", [])
    if not isinstance(methods, list):
        raise ValueError("Method registry must contain a methods list")
    projection["methods"] = [
        _without_mutable_freeze_fields(method)
        for method in methods
        if isinstance(method, dict)
    ]
    return projection


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Required input is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    try:
        handle = path.open("r", encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"Required input is missing: {path}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield value


def _find_repo_root(experiment_root: Path) -> Path:
    resolved = experiment_root.expanduser().resolve()
    for candidate in (resolved, *resolved.parents):
        if (
            (candidate / CLEAN_CHUNKS_RELATIVE_PATH).is_file()
            and (candidate / MASKED_CHUNKS_RELATIVE_PATH).is_file()
            and (candidate / SPLITS_RELATIVE_PATH).is_file()
        ):
            return candidate
    raise ValueError(
        f"Could not locate repository root above experiment root: {experiment_root}"
    )


def _cjk_len(text: str) -> int:
    return len(CJK_RE.findall(text))


def _stable_priority(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest(), "big")


def _normalized_search_text(text: str) -> str:
    sentinels = {
        "<CONTENT>": "\ue000",
        "<NUM>": "\ue001",
        "<LATIN>": "\ue002",
    }
    for placeholder, sentinel in sentinels.items():
        text = text.replace(placeholder, sentinel)
    text = LATIN_RE.sub("<LATIN>", text)
    text = NUMBER_RE.sub("<NUM>", text)
    for placeholder, sentinel in sentinels.items():
        text = text.replace(sentinel, placeholder)
    text = re.sub(r"<+(LATIN|NUM)>+", r"<\1>", text)
    return WHITESPACE_RE.sub("", text)


def _hashed_search_vector(text: str) -> tuple[str, int]:
    normalized = _normalized_search_text(text)
    counts = [0] * HASH_VECTOR_BUCKETS
    for size in (2, 3, 4):
        for index in range(max(0, len(normalized) - size + 1)):
            ngram = normalized[index : index + size].encode("utf-8")
            bucket = int.from_bytes(hashlib.blake2s(ngram, digest_size=2).digest(), "big")
            counts[bucket % HASH_VECTOR_BUCKETS] += 1
    packed = struct.pack(f">{HASH_VECTOR_BUCKETS}H", *counts)
    return base64.b64encode(packed).decode("ascii"), sum(value * value for value in counts)


def _choose_excerpt_start(text: str, key: str, max_cjk: int) -> int:
    total_cjk = _cjk_len(text)
    if total_cjk <= max_cjk:
        return 0
    starts = [0]
    cjk_seen = 0
    for line in text.splitlines(keepends=True):
        cjk_seen += _cjk_len(line)
        if cjk_seen < total_cjk:
            starts.append(cjk_seen)
    usable = [start for start in starts if start <= total_cjk - min(80, max_cjk)]
    if not usable:
        usable = [0]
    return usable[_stable_priority(key) % len(usable)]


def _slice_cjk_window(text: str, start_cjk: int, max_cjk: int) -> str:
    cjk_seen = 0
    output: list[str] = []
    started = start_cjk == 0
    for char in text:
        is_cjk = CJK_RE.fullmatch(char) is not None
        if not started:
            if is_cjk and cjk_seen >= start_cjk:
                started = True
            else:
                if is_cjk:
                    cjk_seen += 1
                continue
        if is_cjk:
            if cjk_seen >= start_cjk + max_cjk:
                break
            cjk_seen += 1
        output.append(char)
    excerpt = "".join(output).strip()
    if _cjk_len(excerpt) > max_cjk:
        raise AssertionError("CJK window exceeded its declared bound")
    return excerpt


def _feature_observations(text: str) -> dict[str, tuple[float, float, int]]:
    cjk_count = max(_cjk_len(text), 1)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sentences = [
        part for part in SENTENCE_SPLIT_RE.split(text) if _cjk_len(part) > 0
    ]
    sentence_lengths = [_cjk_len(sentence) for sentence in sentences]
    paragraph_lengths = [_cjk_len(line) for line in lines if _cjk_len(line) > 0]
    sentence_count = max(len(sentence_lengths), 1)
    paragraph_count = max(len(paragraph_lengths), 1)

    observations: dict[str, tuple[float, float, int]] = {
        "flow.mean_sentence_cjk": (
            float(sum(sentence_lengths)),
            float(sentence_count),
            len(sentence_lengths),
        ),
        "flow.short_sentence_ratio": (
            float(sum(length <= 12 for length in sentence_lengths)),
            float(sentence_count),
            len(sentence_lengths),
        ),
        "flow.long_sentence_ratio": (
            float(sum(length >= 40 for length in sentence_lengths)),
            float(sentence_count),
            len(sentence_lengths),
        ),
        "flow.mean_paragraph_cjk": (
            float(sum(paragraph_lengths)),
            float(paragraph_count),
            len(paragraph_lengths),
        ),
        "flow.sentences_per_paragraph": (
            float(len(sentence_lengths)),
            float(paragraph_count),
            len(sentence_lengths),
        ),
        "flow.commas_per_sentence": (
            float(text.count("，")),
            float(sentence_count),
            text.count("，"),
        ),
    }
    dialogue_lines = sum(line.startswith(DIALOGUE_OPENERS) for line in lines)
    observations["dialogue.line_ratio"] = (
        float(dialogue_lines),
        float(max(len(lines), 1)),
        dialogue_lines,
    )
    quote_count = sum(text.count(mark) for mark in DIALOGUE_MARKS)
    observations["dialogue.quote_marks_per_kcjk"] = (
        float(quote_count),
        float(cjk_count),
        quote_count,
    )
    for feature_id, terms in DIALOGUE_TERMS.items():
        count = sum(text.count(term) for term in terms)
        observations[feature_id] = (float(count), float(cjk_count), count)
    for feature_id, character in PUNCTUATION.items():
        count = text.count(character)
        observations[feature_id] = (float(count), float(cjk_count), count)
    for group, words in FUNCTION_WORD_GROUPS.items():
        count = sum(text.count(word) for word in words)
        observations[f"function_word.{group}"] = (
            float(count),
            float(cjk_count),
            count,
        )
    return observations


def _scene_scores(text: str) -> dict[str, int]:
    scores = {
        scene: sum(text.count(cue) for cue in cues)
        for scene, cues in SCENE_CUES.items()
    }
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    dialogue_lines = sum(line.startswith(DIALOGUE_OPENERS) for line in lines)
    if dialogue_lines:
        scores["dialogue"] += dialogue_lines * 2
    return scores


def _route_scenes_from_neutral(
    neutral_zh: Sequence[Mapping[str, str]], *, limit: int
) -> list[dict[str, Any]]:
    text = "\n".join(paragraph["zh"] for paragraph in neutral_zh)
    scores = _scene_scores(text)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    selected = [item for item in ranked if item[1] > 0][:limit]
    if not selected:
        return [{"scene": "general_narration", "score": 0}]
    return [{"scene": scene, "score": score} for scene, score in selected]


def _profile_contrasts(
    target: FeatureProfile,
    comparisons: Mapping[str, FeatureProfile],
    *,
    minimum_comparison_authors: int = 40,
) -> list[dict[str, Any]]:
    contrasts: list[dict[str, Any]] = []
    for spec in FEATURE_SPECS:
        target_value = target.value(spec)
        if target_value is None:
            continue
        other_values = [
            value
            for profile in comparisons.values()
            if (value := profile.value(spec)) is not None
        ]
        if len(other_values) < minimum_comparison_authors:
            continue
        comparison_mean = fmean(other_values)
        comparison_std = pstdev(other_values)
        delta = target_value - comparison_mean
        z_score = delta / comparison_std if comparison_std > 1e-12 else 0.0
        support = int(target.supports.get(spec.feature_id, 0))
        target_book_count = len(target.books_by_feature.get(spec.feature_id, set()))
        contrasts.append(
            {
                "feature_id": spec.feature_id,
                "family": spec.family,
                "label": spec.label,
                "target_value": round(target_value, 6),
                "comparison_author_mean": round(comparison_mean, 6),
                "comparison_author_std": round(comparison_std, 6),
                "delta": round(delta, 6),
                "z_score": round(z_score, 6),
                "direction": "higher" if delta >= 0 else "lower",
                "target_support": support,
                "target_book_count": target_book_count,
                "target_book_coverage": round(
                    target_book_count / max(len(target.books), 1), 6
                ),
                "comparison_author_count": len(other_values),
            }
        )
    return sorted(
        contrasts,
        key=lambda row: (-abs(float(row["z_score"])), row["feature_id"]),
    )


def _contrast_to_card(contrast: Mapping[str, Any], *, scope: str) -> dict[str, Any]:
    spec = FEATURE_SPEC_BY_ID[str(contrast["feature_id"])]
    direction = str(contrast["direction"])
    instruction = (
        spec.higher_instruction if direction == "higher" else spec.lower_instruction
    )
    return {
        "card_id": f"{scope}.{spec.feature_id}",
        "scope": scope,
        "feature_id": spec.feature_id,
        "family": spec.family,
        "direction": direction,
        "instruction": instruction,
        "guardrail": spec.guardrail,
        "evidence": dict(contrast),
    }


def _select_diverse_cards(
    contrasts: Sequence[Mapping[str, Any]],
    *,
    families: set[str] | None,
    limit: int,
    minimum_abs_z: float = 0.0,
) -> list[dict[str, Any]]:
    candidates = [
        contrast
        for contrast in contrasts
        if (families is None or str(contrast["family"]) in families)
        and abs(float(contrast["z_score"])) >= minimum_abs_z
        and int(contrast["target_support"]) >= 20
    ]
    selected: list[Mapping[str, Any]] = []
    seen_families: set[str] = set()
    for contrast in candidates:
        family = str(contrast["family"])
        if family not in seen_families:
            selected.append(contrast)
            seen_families.add(family)
        if len(selected) == limit:
            break
    for contrast in candidates:
        if contrast in selected:
            continue
        selected.append(contrast)
        if len(selected) == limit:
            break
    return [_contrast_to_card(item, scope="global") for item in selected]


def _summary_exclusions(experiment_root: Path) -> tuple[set[str], list[dict[str, str]]]:
    summary_paths = sorted((experiment_root / "sample_sets").glob("*.summary.json"))
    if not summary_paths:
        raise ValueError("No sample-set summary exists for development/final exclusions")
    titles: set[str] = {"永恒之门"}
    sources: list[dict[str, str]] = []
    for path in summary_paths:
        summary = _load_json(path)
        development = summary.get("development_books", {})
        if isinstance(development, dict):
            for values in development.values():
                if isinstance(values, list):
                    titles.update(str(value) for value in values)
        final_books = summary.get("final_validation_books_reserved", [])
        if isinstance(final_books, list):
            titles.update(str(value) for value in final_books)
        comparison = summary.get("comparison_books_by_author", {})
        if isinstance(comparison, dict):
            for values in comparison.values():
                if isinstance(values, list):
                    titles.update(str(value) for value in values)
        safe_projection = {
            "sample_set": str(summary.get("sample_set", path.stem)),
            "development_books": development,
            "comparison_books_by_author": comparison,
            "final_validation_books_reserved": final_books,
        }
        sources.append(
            {
                "path": path.relative_to(experiment_root).as_posix(),
                "safe_projection_sha256": _sha256_json(safe_projection),
            }
        )
    return titles, sources


def _train_pairs_and_titles(
    splits: Mapping[str, Any], excluded_titles: set[str]
) -> tuple[set[tuple[str, str]], set[str], set[str]]:
    train_rows = splits.get("train")
    if not isinstance(train_rows, list):
        raise ValueError("splits.json must contain a train list")
    pairs: set[tuple[str, str]] = set()
    target_titles: set[str] = set()
    train_authors: set[str] = set()
    for row in train_rows:
        if not isinstance(row, dict):
            raise ValueError("splits.json train rows must be objects")
        author = str(row.get("author", ""))
        title = str(row.get("title", ""))
        if not author or not title or title in excluded_titles:
            continue
        pairs.add((author, title))
        train_authors.add(author)
        if author == TARGET_AUTHOR:
            target_titles.add(title)
    if not target_titles:
        raise ValueError(f"No {TARGET_AUTHOR} train books remain after exclusions")
    comparison_count = len(train_authors - {TARGET_AUTHOR})
    if comparison_count != EXPECTED_COMPARISON_AUTHOR_COUNT:
        raise ValueError(
            "Expected the target author plus 49 comparison authors in train, "
            f"found {comparison_count} comparison authors"
        )
    return pairs, target_titles, train_authors


def _validate_registry(
    experiment_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    registry_path = experiment_root / REGISTRY_RELATIVE_PATH
    registry = _load_json(registry_path)
    methods = registry.get("methods")
    if not isinstance(methods, list):
        raise ValueError("Method registry must contain a methods list")
    method_ids = {str(method.get("id", "")) for method in methods}
    unknown = sorted(method_ids - SUPPORTED_METHOD_IDS)
    missing = sorted(SUPPORTED_METHOD_IDS - method_ids)
    if unknown or missing:
        raise ValueError(
            f"Payload implementation/registry mismatch; unknown={unknown}, missing={missing}"
        )
    config_definition_hashes: dict[str, str] = {}
    validated: list[dict[str, Any]] = []
    for method in methods:
        method_id = str(method["id"])
        intensities = method.get("intensities")
        if not isinstance(intensities, list) or not intensities:
            raise ValueError(f"No intensities registered for {method_id}")
        config_relative = Path(str(method.get("config_path", "")))
        try:
            config_path = (experiment_root.parent.parent.parent / config_relative).resolve()
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"Invalid config path for {method_id}") from exc
        expected_root = experiment_root.resolve()
        if expected_root not in config_path.parents:
            raise ValueError(f"Method config escapes experiment root: {config_relative}")
        config = _load_json(config_path)
        actual_hash = _file_sha256(config_path)
        registered_hash = str(method.get("config_sha256", ""))
        if actual_hash != registered_hash:
            raise ValueError(
                f"Method config hash mismatch for {method_id}: "
                f"registry={registered_hash}, actual={actual_hash}"
            )
        if config.get("method_id") != method_id:
            raise ValueError(f"Method config ID mismatch for {method_id}")
        if config.get("intensities") != intensities:
            raise ValueError(f"Method intensity mismatch for {method_id}")
        config_definition_sha256 = _sha256_json(
            _without_mutable_freeze_fields(config)
        )
        contract = config.get("input_contract", {})
        payload_builder = config.get("payload_builder", {})
        if contract.get("target_derived_allocation_metadata_available_to_runner") is not False:
            raise ValueError(f"Unsafe allocation contract for {method_id}")
        if payload_builder.get("source_corpus") != "target_author_train_books_only":
            raise ValueError(f"Unsafe payload source corpus for {method_id}")
        if payload_builder.get("development_proxy_and_test_books_excluded") is not True:
            raise ValueError(f"Held-out exclusion is not enabled for {method_id}")
        retrieval = config.get("retrieval", {})
        if retrieval.get("enabled"):
            required = {
                "query": "neutral_zh",
                "index_view": "entity_masked_v3",
                "return_view": "entity_masked_v3",
                "source_split": "train",
                "k": RETRIEVAL_K,
                "max_cjk_per_example": MAX_EXAMPLE_CJK,
                "near_duplicate_8gram_jaccard_max": NEAR_DUPLICATE_JACCARD_MAX,
                "evaluation_books_excluded": True,
            }
            mismatches = {
                key: (retrieval.get(key), expected)
                for key, expected in required.items()
                if retrieval.get(key) != expected
            }
            if mismatches:
                raise ValueError(f"Retrieval contract mismatch for {method_id}: {mismatches}")
        config_definition_hashes[method_id] = config_definition_sha256
        validated.append(
            {
                "id": method_id,
                "family": str(method.get("family", "")),
                "intensities": [str(value) for value in intensities],
                "inputs": [str(value) for value in method.get("inputs", [])],
                "config_definition_sha256": config_definition_sha256,
                "payload_builder_id": str(payload_builder.get("id", "")),
                "retrieval_enabled": bool(retrieval.get("enabled")),
            }
        )
    return registry, validated, config_definition_hashes


def _compact_masked_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    text = str(row.get("text", ""))
    chunk_id = str(row.get("chunk_id", ""))
    start_cjk = _choose_excerpt_start(text, chunk_id, MAX_EXAMPLE_CJK)
    masked_excerpt = _slice_cjk_window(text, start_cjk, MAX_EXAMPLE_CJK)
    vector, norm_sq = _hashed_search_vector(masked_excerpt)
    return {
        "chunk_id": chunk_id,
        "author": str(row.get("author", "")),
        "title": str(row.get("title", "")),
        "split": str(row.get("split", "")),
        "chunk_index": int(row.get("chunk_index", 0)),
        "genre": str(row.get("genre", "")),
        "time_area": str(row.get("time_area", "")),
        "article_type": str(row.get("article_type", "")),
        "excerpt_start_cjk": start_cjk,
        "search_vector_b64": vector,
        "search_norm_sq": norm_sq,
        "masked_text": masked_excerpt,
        "masked_excerpt_sha256": _sha256_bytes(masked_excerpt.encode("utf-8")),
        "masked_cjk_count": _cjk_len(masked_excerpt),
    }


def _build_retrieval_indexes(
    *,
    masked_path: Path,
    clean_path: Path,
    allowed_train_pairs: set[tuple[str, str]],
    target_train_titles: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    target_candidates: dict[str, dict[str, Any]] = {}
    hard_heaps: dict[str, list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    target_masked_hashes: list[str] = []
    for row in _iter_jsonl(masked_path):
        if row.get("view") != "entity_masked_v3" or row.get("split") != "train":
            continue
        author = str(row.get("author", ""))
        title = str(row.get("title", ""))
        if (author, title) not in allowed_train_pairs:
            continue
        chunk_id = str(row.get("chunk_id", ""))
        if not chunk_id:
            raise ValueError("Masked train row is missing chunk_id")
        if author == TARGET_AUTHOR:
            if title not in target_train_titles:
                raise ValueError(f"Target retrieval row is not in target train titles: {title}")
            candidate = _compact_masked_candidate(row)
            target_candidates[chunk_id] = candidate
            target_masked_hashes.append(
                _sha256_json(
                    {
                        "chunk_id": chunk_id,
                        "author": author,
                        "title": title,
                        "split": "train",
                        "view": "entity_masked_v3",
                        "text": row.get("text", ""),
                    }
                )
            )
            continue
        candidate = _compact_masked_candidate(row)
        priority = _stable_priority(chunk_id)
        heap = hard_heaps[author]
        item = (-priority, chunk_id, candidate)
        if len(heap) < HARD_NEGATIVE_CHUNKS_PER_AUTHOR:
            heapq.heappush(heap, item)
        elif priority < -heap[0][0]:
            heapq.heapreplace(heap, item)

    hard_candidates = {
        candidate["chunk_id"]: candidate
        for heap in hard_heaps.values()
        for _priority, _chunk_id, candidate in heap
    }
    selected = {**target_candidates, **hard_candidates}
    found_clean: set[str] = set()
    author_profiles: dict[str, FeatureProfile] = defaultdict(FeatureProfile)
    book_profiles: dict[tuple[str, str], FeatureProfile] = defaultdict(FeatureProfile)
    scene_author_profiles: dict[tuple[str, str], FeatureProfile] = defaultdict(
        FeatureProfile
    )
    clean_projection_hashes: list[str] = []
    for row in _iter_jsonl(clean_path):
        if row.get("view") != "clean" or row.get("split") != "train":
            continue
        author = str(row.get("author", ""))
        title = str(row.get("title", ""))
        if (author, title) not in allowed_train_pairs:
            continue
        text = str(row.get("text", ""))
        cjk_count = _cjk_len(text)
        observations = _feature_observations(text)
        author_profiles[author].add(
            observations, title=title, cjk_count=cjk_count
        )
        book_profiles[(author, title)].add(
            observations, title=title, cjk_count=cjk_count
        )
        for scene, score in _scene_scores(text).items():
            if score > 0:
                scene_author_profiles[(scene, author)].add(
                    observations, title=title, cjk_count=cjk_count
                )
        clean_projection_hashes.append(
            _sha256_json(
                {
                    "chunk_id": row.get("chunk_id"),
                    "author": author,
                    "title": title,
                    "split": "train",
                    "view": "clean",
                    "text": text,
                }
            )
        )
        chunk_id = str(row.get("chunk_id", ""))
        candidate = selected.get(chunk_id)
        if candidate is None:
            continue
        if candidate["author"] != author or candidate["title"] != title:
            raise ValueError(f"Clean/masked metadata mismatch for {chunk_id}")
        clean_excerpt = _slice_cjk_window(
            text,
            int(candidate["excerpt_start_cjk"]),
            MAX_EXAMPLE_CJK,
        )
        if not clean_excerpt:
            raise ValueError(f"Empty clean retrieval excerpt for {chunk_id}")
        candidate["clean_text"] = clean_excerpt
        candidate["clean_text_sha256"] = _sha256_bytes(clean_excerpt.encode("utf-8"))
        candidate["clean_cjk_count"] = _cjk_len(clean_excerpt)
        candidate["reference_id"] = "ref_" + _sha256_json(
            {
                "chunk_id": chunk_id,
                "clean_text_sha256": candidate["clean_text_sha256"],
            }
        )[:24]
        found_clean.add(chunk_id)

    missing_clean = sorted(selected.keys() - found_clean)
    if missing_clean:
        raise ValueError(
            f"Clean view is missing {len(missing_clean)} selected masked chunks; "
            f"first={missing_clean[:3]}"
        )
    if set(author_profiles) != {author for author, _title in allowed_train_pairs}:
        raise ValueError("Train split authors and clean dataset authors disagree")
    target_titles_found = {
        candidate["title"] for candidate in target_candidates.values()
    }
    if target_titles_found != target_train_titles:
        raise ValueError(
            "Target retrieval index does not cover every target train book: "
            f"missing={sorted(target_train_titles - target_titles_found)}"
        )

    target_index = sorted(target_candidates.values(), key=lambda item: item["chunk_id"])
    hard_index = sorted(hard_candidates.values(), key=lambda item: item["chunk_id"])
    for entry in (*target_index, *hard_index):
        entry.pop("excerpt_start_cjk", None)

    comparison_profiles = {
        author: profile
        for author, profile in author_profiles.items()
        if author != TARGET_AUTHOR
    }
    global_contrasts = _profile_contrasts(
        author_profiles[TARGET_AUTHOR], comparison_profiles
    )
    scene_contrasts: dict[str, list[dict[str, Any]]] = {}
    for scene in SCENE_CUES:
        target_profile = scene_author_profiles.get((scene, TARGET_AUTHOR))
        if target_profile is None:
            scene_contrasts[scene] = []
            continue
        comparisons = {
            author: profile
            for (candidate_scene, author), profile in scene_author_profiles.items()
            if candidate_scene == scene and author != TARGET_AUTHOR
        }
        scene_contrasts[scene] = _profile_contrasts(
            target_profile,
            comparisons,
            minimum_comparison_authors=30,
        )
    evidence = {
        "target_profile": author_profiles[TARGET_AUTHOR],
        "comparison_profiles": comparison_profiles,
        "book_profiles": book_profiles,
        "global_contrasts": global_contrasts,
        "scene_contrasts": scene_contrasts,
        "train_clean_projection_sha256": _sha256_json(sorted(clean_projection_hashes)),
        "target_masked_projection_sha256": _sha256_json(sorted(target_masked_hashes)),
    }
    return target_index, hard_index, evidence


def _profile_summary(profile: FeatureProfile) -> dict[str, Any]:
    return {
        "chunk_count": profile.chunk_count,
        "cjk_count": profile.cjk_count,
        "book_count": len(profile.books),
    }


def _build_card_assets(evidence: Mapping[str, Any]) -> dict[str, Any]:
    global_contrasts = evidence["global_contrasts"]
    global_cards = _select_diverse_cards(
        global_contrasts,
        families=None,
        limit=8,
        minimum_abs_z=0.35,
    )
    flow_cards = _select_diverse_cards(
        global_contrasts,
        families={"sentence_flow"},
        limit=5,
        minimum_abs_z=0.25,
    )
    interpretable_cards = _select_diverse_cards(
        global_contrasts,
        families={"function_word", "punctuation", "dialogue"},
        limit=7,
        minimum_abs_z=0.25,
    )
    if not global_cards or not flow_cards or not interpretable_cards:
        raise ValueError("Train-only contrastive evidence produced an empty card family")

    scene_cards: dict[str, dict[str, Any]] = {}
    for scene, contrasts in evidence["scene_contrasts"].items():
        selected = _select_diverse_cards(
            contrasts,
            families={"sentence_flow", "function_word", "punctuation", "dialogue"},
            limit=3,
            minimum_abs_z=0.2,
        )
        if not selected:
            selected = global_cards[:2]
        scene_cards[scene] = {
            "scene": scene,
            "routing_source": "neutral_zh_only",
            "route_cues_zh": list(SCENE_CUES[scene]),
            "cards": [
                {**card, "scope": f"scene.{scene}"} for card in selected
            ],
        }
    scene_cards["general_narration"] = {
        "scene": "general_narration",
        "routing_source": "neutral_zh_only",
        "route_cues_zh": [],
        "cards": global_cards[:3],
    }
    return {
        "global_style_cards": global_cards,
        "sentence_flow_cards": flow_cards,
        "interpretable_style_cards": interpretable_cards,
        "scene_style_cards": scene_cards,
    }


def _one_passage_per_group(
    entries: Sequence[Mapping[str, Any]], *, group_field: str, prefix: str
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[str(entry[group_field])].append(entry)
    passages: list[dict[str, Any]] = []
    for index, group in enumerate(sorted(grouped), start=1):
        chosen = min(
            grouped[group],
            key=lambda entry: (
                _stable_priority(str(entry["reference_id"])),
                str(entry["reference_id"]),
            ),
        )
        passages.append(
            {
                "passage_id": f"{prefix}{index:03d}",
                "text": str(chosen["masked_text"]),
                "text_sha256": str(chosen["masked_excerpt_sha256"]),
                "view": "entity_masked_v3",
                "source_split": "train",
            }
        )
    return passages


def _close_reading_prompt_contract() -> dict[str, Any]:
    return {
        "task": (
            "Compare anonymized target and comparison passages. Describe only "
            "reusable discourse, syntax, function-word, punctuation, dialogue, "
            "and sentence-flow tendencies supported by the supplied statistical "
            "features. Do not infer topic, genre, identity, or book-specific diction."
        ),
        "constraints": [
            "Every claim and card must cite one or more supplied feature_id values.",
            "Do not quote or closely paraphrase any evidence passage.",
            "Do not mention authors, books, characters, places, or source identities.",
            "Phrase cards as optional source-triggered operations with fidelity guardrails.",
            "Return JSON only using the declared output contract.",
        ],
        "output_contract": {
            "schema_version": CLOSE_READING_RESULT_SCHEMA,
            "source_packet_sha256": "<packet hash supplied by runner>",
            "prompt_sha256": "<prompt hash supplied by packet>",
            "generation_provenance": {
                "model": "<model identifier>",
                "run_id": "<immutable run identifier>",
            },
            "claims": [
                {
                    "claim_id": "cr_claim_001",
                    "text": "<concise close-reading claim>",
                    "feature_ids": ["<supplied feature_id>"],
                }
            ],
            "cards": [
                {
                    "card_id": "cr_card_001",
                    "when": "<source opportunity>",
                    "instruction": "<optional style operation>",
                    "guardrail": "<fidelity constraint>",
                    "feature_ids": ["<supplied feature_id>"],
                }
            ],
            "limitations": ["<evidence limitation>"],
        },
    }


def _build_close_reading_source_packet(
    *,
    experiment_root: Path,
    target_index: Sequence[Mapping[str, Any]],
    hard_index: Sequence[Mapping[str, Any]],
    global_contrasts: Sequence[Mapping[str, Any]],
    target_train_book_count: int,
) -> dict[str, Any]:
    target_passages = _one_passage_per_group(
        target_index, group_field="title", prefix="target_"
    )
    comparison_passages = _one_passage_per_group(
        hard_index, group_field="author", prefix="comparison_"
    )
    if len(target_passages) != target_train_book_count:
        raise ValueError("Close-reading packet does not cover every target train book")
    if len(comparison_passages) != EXPECTED_COMPARISON_AUTHOR_COUNT:
        raise ValueError("Close-reading packet does not cover all 49 comparison authors")
    prompt = _close_reading_prompt_contract()
    prompt_sha256 = _sha256_json(prompt)
    eligible_contrasts = [
        dict(contrast)
        for contrast in global_contrasts
        if abs(float(contrast["z_score"])) >= 0.35
        and int(contrast["target_support"]) >= 20
        and float(contrast["target_book_coverage"]) >= 0.35
    ]
    if len(eligible_contrasts) < 5:
        raise ValueError("Insufficient gated train-only contrasts for close reading")
    packet_content = {
        "schema_version": CLOSE_READING_PACKET_SCHEMA,
        "target_label": "target_corpus",
        "comparison_label": "comparison_corpus",
        "evidence_policy": {
            "target_author_source_split": "train",
            "target_train_book_count": target_train_book_count,
            "comparison_author_count": EXPECTED_COMPARISON_AUTHOR_COUNT,
            "passages_anonymized": True,
            "single_author_production_assets_used": False,
            "development_final_and_application_text_used": False,
        },
        "target_passages": target_passages,
        "comparison_passages": comparison_passages,
        "eligible_statistical_contrasts": eligible_contrasts,
        "prompt": prompt,
        "prompt_sha256": prompt_sha256,
    }
    packet_sha256 = _sha256_json(packet_content)
    packet = {
        "schema_version": CLOSE_READING_PACKET_SCHEMA,
        "content_sha256": packet_sha256,
        "packet": packet_content,
    }
    packet_bytes = _canonical_json_bytes(packet) + b"\n"
    relative_path = (
        Path("method_assets")
        / CLOSE_READING_PACKET_DIRECTORY
        / f"source_packet.v1.{packet_sha256}.json"
    )
    packet_path = experiment_root / relative_path
    if packet_path.exists():
        if packet_path.read_bytes() != packet_bytes:
            raise ValueError(f"Immutable close-reading packet collision at {packet_path}")
    else:
        _atomic_write(packet_path, packet_bytes)
    return {
        "schema_version": CLOSE_READING_PACKET_SCHEMA,
        "path": relative_path.as_posix(),
        "sha256": packet_sha256,
        "file_sha256": _sha256_bytes(packet_bytes),
        "prompt_sha256": prompt_sha256,
        "packet": packet_content,
    }


def _validate_close_reading_text(
    text: Any,
    *,
    label: str,
    evidence_ngrams: set[str],
) -> str:
    if not isinstance(text, str) or not text.strip() or len(text) > 1200:
        raise ValueError(f"Invalid close-reading {label}")
    overlap = _cjk_ngrams(text, NEAR_DUPLICATE_NGRAM) & evidence_ngrams
    if overlap:
        raise ValueError(
            f"Close-reading {label} copies an {NEAR_DUPLICATE_NGRAM}-character "
            "sequence from its source packet"
        )
    return text.strip()


def _load_close_reading_result(
    *,
    experiment_root: Path,
    source_packet: Mapping[str, Any],
) -> dict[str, Any]:
    packet_sha256 = str(source_packet["sha256"])
    relative_path = (
        Path("method_assets")
        / CLOSE_READING_RESULT_DIRECTORY
        / f"result.v1.{packet_sha256}.json"
    )
    path = experiment_root / relative_path
    if not path.exists():
        return {
            "availability": "unavailable",
            "reason_code": "fresh_train_only_llm_close_reading_not_frozen",
            "source_packet_path": str(source_packet["path"]),
            "source_packet_sha256": packet_sha256,
            "expected_result_path": relative_path.as_posix(),
        }
    result = _load_json(path)
    ledger_path = path.with_suffix(".ledger.json")
    if not ledger_path.exists():
        raise ValueError("Close-reading result lacks its immutable generation ledger")
    ledger = _load_json(ledger_path)
    if ledger.get("schema_version") != 2:
        raise ValueError("Close-reading generation ledger must use schema version 2")
    expected_ledger = {
        "source_packet_path": str(
            (experiment_root / str(source_packet["path"])).relative_to(REPO_ROOT)
        ),
        "source_packet_sha256": packet_sha256,
        "source_packet_file_sha256": _file_sha256(
            experiment_root / str(source_packet["path"])
        ),
        "prompt_sha256": source_packet.get("prompt_sha256"),
        "result_path": str(path.relative_to(REPO_ROOT)),
        "result_file_sha256": _file_sha256(path),
        "result_output_sha256": _sha256_json(result),
    }
    for field, expected in expected_ledger.items():
        if ledger.get(field) != expected:
            raise ValueError(f"Close-reading ledger {field} mismatch")
    for path_field, hash_field in (
        ("schema_snapshot_path", "schema_snapshot_sha256"),
        ("generator_snapshot_path", "generator_snapshot_sha256"),
    ):
        recorded = ledger.get(path_field)
        if not isinstance(recorded, str):
            raise ValueError(f"Close-reading ledger lacks {path_field}")
        recorded_path = REPO_ROOT / recorded
        if not recorded_path.exists() or _file_sha256(recorded_path) != ledger.get(
            hash_field
        ):
            raise ValueError(f"Close-reading ledger {path_field} binding mismatch")
    for hash_field in (
        "concrete_request_sha256",
        "schema_snapshot_sha256",
        "generator_snapshot_sha256",
        "external_isolation_profile_sha256",
    ):
        value = ledger.get(hash_field)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"Close-reading ledger lacks valid {hash_field}")
    if not ledger.get("response_id") or not isinstance(
        ledger.get("response_usage"), dict
    ):
        raise ValueError("Close-reading ledger lacks model response provenance")
    for passage_group in ("target_passages", "comparison_passages"):
        passages = source_packet["packet"].get(passage_group)
        if not isinstance(passages, list) or not passages or any(
            not isinstance(passage, dict)
            or passage.get("view") != "entity_masked_v3"
            for passage in passages
        ):
            raise ValueError("Close-reading result is bound to unmasked evidence")
    if result.get("schema_version") != CLOSE_READING_RESULT_SCHEMA:
        raise ValueError(f"Invalid close-reading result schema in {path}")
    if result.get("source_packet_sha256") != packet_sha256:
        raise ValueError("Close-reading result does not match the current source packet")
    if result.get("prompt_sha256") != source_packet.get("prompt_sha256"):
        raise ValueError("Close-reading result prompt hash does not match its packet")
    provenance = result.get("generation_provenance")
    if not isinstance(provenance, dict) or not provenance.get("model") or not provenance.get(
        "run_id"
    ):
        raise ValueError("Close-reading result lacks immutable generation provenance")
    if provenance.get("model") != ledger.get("model") or provenance.get(
        "run_id"
    ) != ledger.get("run_id"):
        raise ValueError("Close-reading result and generation ledger disagree")
    eligible = {
        str(contrast["feature_id"]): contrast
        for contrast in source_packet["packet"]["eligible_statistical_contrasts"]
    }
    evidence_ngrams: set[str] = set()
    for arm in ("target_passages", "comparison_passages"):
        for passage in source_packet["packet"][arm]:
            evidence_ngrams.update(
                _cjk_ngrams(str(passage["text"]), NEAR_DUPLICATE_NGRAM)
            )

    def validated_feature_ids(value: Any, label: str) -> list[str]:
        if not isinstance(value, list) or not value:
            raise ValueError(f"Close-reading {label} must cite feature IDs")
        feature_ids = [str(item) for item in value]
        invalid = sorted(set(feature_ids) - set(eligible))
        if invalid:
            raise ValueError(
                f"Close-reading {label} cites ungated feature IDs: {invalid}"
            )
        return feature_ids

    claims = result.get("claims")
    cards = result.get("cards")
    if not isinstance(claims, list) or not 3 <= len(claims) <= 16:
        raise ValueError("Close-reading result must contain 3-16 claims")
    if not isinstance(cards, list) or not 3 <= len(cards) <= 16:
        raise ValueError("Close-reading result must contain 3-16 cards")
    validated_claims: list[dict[str, Any]] = []
    for index, claim in enumerate(claims):
        if not isinstance(claim, dict):
            raise ValueError("Close-reading claims must be objects")
        feature_ids = validated_feature_ids(claim.get("feature_ids"), f"claim {index}")
        validated_claims.append(
            {
                "claim_id": str(claim.get("claim_id", f"cr_claim_{index + 1:03d}")),
                "text": _validate_close_reading_text(
                    claim.get("text"),
                    label=f"claim {index}",
                    evidence_ngrams=evidence_ngrams,
                ),
                "feature_ids": feature_ids,
                "statistical_evidence": [eligible[item] for item in feature_ids],
            }
        )
    validated_cards: list[dict[str, Any]] = []
    for index, card in enumerate(cards):
        if not isinstance(card, dict):
            raise ValueError("Close-reading cards must be objects")
        feature_ids = validated_feature_ids(card.get("feature_ids"), f"card {index}")
        validated_cards.append(
            {
                "card_id": str(card.get("card_id", f"cr_card_{index + 1:03d}")),
                "when": _validate_close_reading_text(
                    card.get("when"),
                    label=f"card {index} when",
                    evidence_ngrams=evidence_ngrams,
                ),
                "instruction": _validate_close_reading_text(
                    card.get("instruction"),
                    label=f"card {index} instruction",
                    evidence_ngrams=evidence_ngrams,
                ),
                "guardrail": _validate_close_reading_text(
                    card.get("guardrail"),
                    label=f"card {index} guardrail",
                    evidence_ngrams=evidence_ngrams,
                ),
                "feature_ids": feature_ids,
                "statistical_gates": [eligible[item] for item in feature_ids],
            }
        )
    limitations = result.get("limitations", [])
    if not isinstance(limitations, list) or not all(
        isinstance(item, str) for item in limitations
    ):
        raise ValueError("Close-reading limitations must be a string list")
    return {
        "availability": "ready",
        "source_packet_path": str(source_packet["path"]),
        "source_packet_sha256": packet_sha256,
        "result_path": relative_path.as_posix(),
        "result_file_sha256": _file_sha256(path),
        "ledger_path": str(ledger_path.relative_to(experiment_root)),
        "ledger_file_sha256": _file_sha256(ledger_path),
        "generation_provenance": provenance,
        "definition": {
            "claims": validated_claims,
            "limitations": limitations,
        },
        "validated_cards": validated_cards,
    }


def _intensity_contract(intensity: str) -> dict[str, Any]:
    contracts = {
        "none": {
            "rewrite_scope": "none",
            "max_active_cards": 0,
            "instruction": "Return the neutral Chinese paragraphs unchanged.",
        },
        "light": {
            "rewrite_scope": "minimal",
            "max_active_cards": 3,
            "instruction": "Use the lightest source-supported recast and skip weak opportunities.",
        },
        "medium": {
            "rewrite_scope": "moderate",
            "max_active_cards": 6,
            "instruction": "Apply several compatible cues while preserving naturalness and every semantic fact.",
        },
        "strong": {
            "rewrite_scope": "stress_test",
            "max_active_cards": 8,
            "instruction": "Recast assertively as a stress test, but never trade fidelity for style strength.",
        },
    }
    try:
        return contracts[intensity]
    except KeyError as exc:
        raise ValueError(f"Unsupported intensity: {intensity}") from exc


def _method_intensity_asset(
    method_id: str,
    intensity: str,
    card_assets: Mapping[str, Any],
    close_reading: Mapping[str, Any],
) -> dict[str, Any]:
    contract = _intensity_contract(intensity)
    base: dict[str, Any] = {
        "availability": "ready",
        "method_id": method_id,
        "intensity": intensity,
        "intensity_contract": contract,
        "evidence_source": "multi_author_train_dataset",
        "target_author": TARGET_AUTHOR,
    }
    card_limit = int(contract["max_active_cards"])
    if method_id == "neutral_only":
        base["instructions"] = [
            "Return neutral_zh byte-for-byte by paragraph.",
            "Apply no author-style operation.",
        ]
    elif method_id.startswith("generic_author_style_"):
        base["target_author_name"] = TARGET_AUTHOR
        base["instructions"] = [
            "Recast toward the named author's general prose manner without importing content.",
            "Treat the author name as a weak generic cue, not permission to quote or invent.",
        ]
    elif method_id == "global_style_cards":
        base["global_style_cards"] = card_assets["global_style_cards"][:card_limit]
    elif method_id == "scene_routed_style_cards":
        base["scene_style_cards"] = card_assets["scene_style_cards"]
        base["scene_limit"] = 1 if intensity == "light" else 2
    elif method_id == "sentence_flow_cards":
        base["flow_style_cards"] = card_assets["sentence_flow_cards"][:card_limit]
    elif method_id == "function_word_punctuation_dialogue_cards":
        base["interpretable_style_cards"] = card_assets[
            "interpretable_style_cards"
        ][:card_limit]
    elif method_id == "retrieved_examples_only":
        base["retrieval"] = _retrieval_policy()
    elif method_id == "scene_cards_plus_examples":
        base["scene_style_cards"] = card_assets["scene_style_cards"]
        base["scene_limit"] = 1 if intensity == "light" else 2
        base["retrieval"] = _retrieval_policy()
    elif method_id == "contrastive_examples":
        base["retrieval"] = {
            **_retrieval_policy(),
            "hard_negative_authors": EXPECTED_COMPARISON_AUTHOR_COUNT,
        }
        base["instructions"] = [
            "Infer only structural differences shared by target examples and absent from hard negatives.",
            "Never import names, setting, facts, or eight-character sequences from either arm.",
        ]
    elif method_id in {
        "llm_close_reading_style_definition",
        "llm_close_reading_cards_statistical_gates",
    }:
        if close_reading.get("availability") == "ready":
            base["source_packet_path"] = close_reading["source_packet_path"]
            base["source_packet_sha256"] = close_reading["source_packet_sha256"]
            base["result_path"] = close_reading["result_path"]
            base["result_file_sha256"] = close_reading["result_file_sha256"]
            base["generation_provenance"] = close_reading[
                "generation_provenance"
            ]
            if method_id == "llm_close_reading_style_definition":
                base["close_reading_definition"] = close_reading["definition"]
            else:
                base["validated_close_reading_cards"] = close_reading[
                    "validated_cards"
                ][:card_limit]
        else:
            base.update(
                {
                    "availability": "unavailable",
                    "reason_code": close_reading["reason_code"],
                    "reason": (
                        "A fresh train-only source packet exists, but no validated "
                        "packet-bound LLM close-reading result is frozen. Legacy "
                        "single-author outputs are intentionally excluded."
                    ),
                    "source_packet_path": close_reading["source_packet_path"],
                    "source_packet_sha256": close_reading[
                        "source_packet_sha256"
                    ],
                    "expected_result_path": close_reading[
                        "expected_result_path"
                    ],
                    "required_before_ready": [
                        "run the source packet through a separately versioned LLM generation step",
                        "preserve packet, prompt, model, and run hashes",
                        "pass feature-ID statistical gates and no-copy validation",
                        "rerun freeze to bind the validated result",
                    ],
                }
            )
    elif method_id == "self_critique_repair":
        base["repair_contract"] = {
            "max_repair_attempts": 1,
            "critique_basis": [
                "english_semantic_source",
                "neutral_zh",
                "prior_output",
            ],
            "required_checks": [
                "paragraph identity and order",
                "entities, numbers, roles, and speaker attribution",
                "causality, chronology, negation, modality, and intensity",
                "unsupported invention or compression",
                "readability and style-guide leakage",
                "reference copying of eight or more Chinese characters",
            ],
            "repair_rule": "Change only critique-supported spans and prefer fidelity over style.",
        }
    else:
        raise ValueError(f"No payload implementation for {method_id}")
    return base


def _retrieval_policy() -> dict[str, Any]:
    return {
        "query_source": "neutral_zh_only",
        "index_view": "entity_masked_v3",
        "return_view": "entity_masked_v3",
        "source_split": "train",
        "target_author": TARGET_AUTHOR,
        "k": RETRIEVAL_K,
        "max_cjk_per_example": MAX_EXAMPLE_CJK,
        "near_duplicate": {
            "ngram": NEAR_DUPLICATE_NGRAM,
            "jaccard_max": NEAR_DUPLICATE_JACCARD_MAX,
        },
        "development_and_final_books_excluded": True,
    }


def _build_methods_asset(
    registry_methods: Sequence[Mapping[str, Any]],
    card_assets: Mapping[str, Any],
    close_reading: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    methods: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for method in registry_methods:
        method_id = str(method["id"])
        intensities: dict[str, Any] = {}
        for intensity in method["intensities"]:
            asset = _method_intensity_asset(
                method_id, str(intensity), card_assets, close_reading
            )
            intensities[str(intensity)] = asset
            hashes[f"{method_id}:{intensity}"] = _sha256_json(asset)
        methods[method_id] = {
            "family": method["family"],
            "inputs": method["inputs"],
            "config_definition_sha256": method["config_definition_sha256"],
            "payload_builder_id": method["payload_builder_id"],
            "intensities": intensities,
        }
    return methods, hashes


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _pretty_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _update_method_readiness(
    *,
    experiment_root: Path,
    repo_root: Path,
    frozen: Mapping[str, Any],
) -> None:
    assets = frozen["assets"]
    methods = assets["methods"]
    method_hashes = assets["component_hashes"][
        "method_intensity_asset_sha256"
    ]
    registry_path = experiment_root / REGISTRY_RELATIVE_PATH
    registry = _load_json(registry_path)
    registry_rows = registry.get("methods")
    if not isinstance(registry_rows, list):
        raise ValueError("Method registry must contain a methods list")
    asset_path = (experiment_root / str(frozen["asset_path"])).resolve()
    manifest_path = asset_path.relative_to(repo_root).as_posix()
    config_file_hashes: dict[str, str] = {}
    registry_by_id = {
        str(row.get("id", "")): row
        for row in registry_rows
        if isinstance(row, dict)
    }
    if set(registry_by_id) != set(methods):
        raise ValueError("Frozen methods and registry methods disagree before readiness update")

    for method_id, method in methods.items():
        intensity_assets = method.get("intensities", {})
        registered_intensities = registry_by_id[method_id].get("intensities", [])
        if set(intensity_assets) != set(registered_intensities):
            raise ValueError(f"Frozen intensities disagree for {method_id}")
        availability = {
            intensity: str(asset.get("availability", "unavailable"))
            for intensity, asset in intensity_assets.items()
        }
        ready = all(value == "ready" for value in availability.values())
        blocked_reasons = sorted(
            {
                str(asset.get("reason_code", "required_asset_unavailable"))
                for asset in intensity_assets.values()
                if asset.get("availability") != "ready"
            }
        )
        manifest = {
            "path": manifest_path,
            "sha256": str(frozen["content_sha256"]),
            "file_sha256": str(frozen["file_sha256"]),
            "availability_by_intensity": availability,
            "method_intensity_asset_sha256": {
                intensity: str(method_hashes[f"{method_id}:{intensity}"])
                for intensity in sorted(intensity_assets)
            },
            "ready": ready,
        }
        if blocked_reasons:
            manifest["blocked_reasons"] = blocked_reasons
            manifest["close_reading_source_packet"] = assets["close_reading"][
                "source_packet"
            ]
        registry_row = registry_by_id[method_id]
        config_relative = Path(str(registry_row["config_path"]))
        config_path = (repo_root / config_relative).resolve()
        if experiment_root not in config_path.parents:
            raise ValueError(f"Method config escapes experiment root: {config_relative}")
        config = _load_json(config_path)
        config["status"] = "ready" if ready else "not_run"
        config["asset_manifest"] = manifest
        config_bytes = _pretty_json_bytes(config)
        _atomic_write(config_path, config_bytes)
        config_file_hashes[method_id] = _sha256_bytes(config_bytes)
        registry_row["status"] = config["status"]
        registry_row["asset_manifest"] = manifest

    for method_id, registry_row in registry_by_id.items():
        registry_row["config_sha256"] = config_file_hashes[method_id]
    ready_methods = sorted(
        method_id
        for method_id, row in registry_by_id.items()
        if row["status"] == "ready"
    )
    blocked_methods = sorted(set(methods) - set(ready_methods))
    registry["asset_manifest"] = {
        "path": manifest_path,
        "sha256": str(frozen["content_sha256"]),
        "file_sha256": str(frozen["file_sha256"]),
        "ready_methods": ready_methods,
        "blocked_methods": blocked_methods,
        "all_method_assets_verified": True,
    }
    _atomic_write(registry_path, _pretty_json_bytes(registry))


def freeze_method_assets(experiment_root: Path) -> dict[str, Any]:
    """Build and atomically freeze deterministic assets for the method registry.

    Evidence is restricted to train rows in the clean and entity_masked_v3 dataset
    views. Sample-set summaries are read only for public development/final title
    exclusions; hidden targets and evaluator allocations are never opened.
    """

    experiment_root = experiment_root.expanduser().resolve()
    repo_root = _find_repo_root(experiment_root)
    registry, registry_methods, config_definition_hashes = _validate_registry(
        experiment_root
    )
    excluded_titles, summary_sources = _summary_exclusions(experiment_root)
    splits_path = repo_root / SPLITS_RELATIVE_PATH
    splits = _load_json(splits_path)
    allowed_pairs, target_train_titles, train_authors = _train_pairs_and_titles(
        splits, excluded_titles
    )
    nontrain_target_titles = {
        str(row.get("title", ""))
        for split, rows in splits.items()
        if split != "train" and isinstance(rows, list)
        for row in rows
        if isinstance(row, dict) and row.get("author") == TARGET_AUTHOR
    }
    if target_train_titles & (excluded_titles | nontrain_target_titles):
        raise ValueError("Target train titles overlap held-out exclusions")

    target_index, hard_index, evidence = _build_retrieval_indexes(
        masked_path=repo_root / MASKED_CHUNKS_RELATIVE_PATH,
        clean_path=repo_root / CLEAN_CHUNKS_RELATIVE_PATH,
        allowed_train_pairs=allowed_pairs,
        target_train_titles=target_train_titles,
    )
    comparison_profiles: Mapping[str, FeatureProfile] = evidence["comparison_profiles"]
    if len(comparison_profiles) != EXPECTED_COMPARISON_AUTHOR_COUNT:
        raise ValueError("Contrastive statistics do not contain all 49 comparison authors")
    card_assets = _build_card_assets(evidence)
    close_reading_source_packet = _build_close_reading_source_packet(
        experiment_root=experiment_root,
        target_index=target_index,
        hard_index=hard_index,
        global_contrasts=evidence["global_contrasts"],
        target_train_book_count=len(target_train_titles),
    )
    close_reading = _load_close_reading_result(
        experiment_root=experiment_root,
        source_packet=close_reading_source_packet,
    )
    methods, method_asset_hashes = _build_methods_asset(
        registry_methods, card_assets, close_reading
    )

    statistics = {
        "target": _profile_summary(evidence["target_profile"]),
        "comparison_author_count": len(comparison_profiles),
        "comparison_authors_sha256": _sha256_json(sorted(comparison_profiles)),
        "global_feature_contrasts": evidence["global_contrasts"],
        "scene_feature_contrasts": evidence["scene_contrasts"],
    }
    source_projections = {
        "registry_definition_sha256": _sha256_json(
            _registry_definition_projection(registry)
        ),
        "method_config_definition_sha256": dict(
            sorted(config_definition_hashes.items())
        ),
        "style_prompt_sha256": _file_sha256(
            experiment_root / STYLE_PROMPT_RELATIVE_PATH
        ),
        "train_split_projection_sha256": _sha256_json(
            sorted([list(pair) for pair in allowed_pairs])
        ),
        "sample_exclusion_sources": summary_sources,
        "sample_excluded_titles_sha256": _sha256_json(sorted(excluded_titles)),
        "train_clean_projection_sha256": evidence[
            "train_clean_projection_sha256"
        ],
        "target_masked_v3_projection_sha256": evidence[
            "target_masked_projection_sha256"
        ],
    }
    component_hashes = {
        "train_statistics_sha256": _sha256_json(statistics),
        "style_cards_sha256": _sha256_json(card_assets),
        "target_retrieval_index_sha256": _sha256_json(target_index),
        "hard_negative_retrieval_index_sha256": _sha256_json(hard_index),
        "close_reading_source_packet_sha256": str(
            close_reading_source_packet["sha256"]
        ),
        "close_reading_asset_sha256": _sha256_json(close_reading),
        "methods_sha256": _sha256_json(methods),
        "method_intensity_asset_sha256": method_asset_hashes,
    }
    content = {
        "schema_version": 1,
        "generator": {"id": GENERATOR_ID, "version": GENERATOR_VERSION},
        "target_author": TARGET_AUTHOR,
        "registry": {
            "schema_version": registry.get("schema_version"),
            "target_author": registry.get("target_author"),
            "definition_sha256": source_projections[
                "registry_definition_sha256"
            ],
            "methods": registry_methods,
        },
        "evidence_policy": {
            "allowed_inputs": [
                CLEAN_CHUNKS_RELATIVE_PATH.as_posix(),
                MASKED_CHUNKS_RELATIVE_PATH.as_posix(),
                SPLITS_RELATIVE_PATH.as_posix(),
                "sample_sets/*.summary.json safe exclusion projection",
                REGISTRY_RELATIVE_PATH.as_posix(),
                "method_registry/methods/*.json",
                STYLE_PROMPT_RELATIVE_PATH.as_posix(),
            ],
            "style_evidence_split": "train",
            "target_train_books": sorted(target_train_titles),
            "target_train_book_count": len(target_train_titles),
            "train_author_count": len(train_authors),
            "comparison_author_count": len(train_authors - {TARGET_AUTHOR}),
            "development_final_and_application_titles_excluded": sorted(
                excluded_titles | nontrain_target_titles
            ),
            "single_author_production_assets_used": False,
            "hidden_targets_accessed": False,
            "evaluator_allocation_accessed": False,
        },
        "source_projections": source_projections,
        "statistics": statistics,
        "style_cards": card_assets,
        "close_reading": {
            "source_packet": {
                key: close_reading_source_packet[key]
                for key in (
                    "schema_version",
                    "path",
                    "sha256",
                    "file_sha256",
                    "prompt_sha256",
                )
            },
            "asset": close_reading,
        },
        "retrieval": {
            "policy": _retrieval_policy(),
            "target_index": target_index,
            "hard_negative_index": hard_index,
            "target_entry_count": len(target_index),
            "hard_negative_entry_count": len(hard_index),
            "hard_negative_author_count": len(
                {entry["author"] for entry in hard_index}
            ),
        },
        "methods": methods,
        "component_hashes": component_hashes,
    }
    content_sha256 = _sha256_json(content)
    bundle = {
        "schema_version": ASSET_SCHEMA,
        "content_sha256": content_sha256,
        "assets": content,
    }
    bundle_bytes = _canonical_json_bytes(bundle) + b"\n"
    bundle_file_sha256 = _sha256_bytes(bundle_bytes)
    relative_asset_path = (
        Path("method_assets")
        / ASSET_DIRECTORY
        / f"assets.{content_sha256}.json"
    )
    asset_path = experiment_root / relative_asset_path
    if asset_path.exists():
        if asset_path.read_bytes() != bundle_bytes:
            raise ValueError(f"Immutable asset collision at {asset_path}")
    else:
        _atomic_write(asset_path, bundle_bytes)
    lock = {
        "schema_version": 1,
        "asset_schema": ASSET_SCHEMA,
        "content_sha256": content_sha256,
        "file_sha256": bundle_file_sha256,
        "asset_path": relative_asset_path.as_posix(),
    }
    _atomic_write(
        experiment_root / "method_assets" / LOCK_NAME,
        _canonical_json_bytes(lock) + b"\n",
    )
    _load_frozen_cached.cache_clear()
    _decoded_retrieval_index.cache_clear()
    frozen = load_frozen_assets(experiment_root)
    _update_method_readiness(
        experiment_root=experiment_root,
        repo_root=repo_root,
        frozen=frozen,
    )
    return frozen


def _verify_frozen_components(content: Mapping[str, Any]) -> None:
    component_hashes = content.get("component_hashes")
    if not isinstance(component_hashes, dict):
        raise ValueError("Frozen assets lack component hashes")
    checks = {
        "train_statistics_sha256": content.get("statistics"),
        "style_cards_sha256": content.get("style_cards"),
        "target_retrieval_index_sha256": content.get("retrieval", {}).get(
            "target_index"
        ),
        "hard_negative_retrieval_index_sha256": content.get("retrieval", {}).get(
            "hard_negative_index"
        ),
        "close_reading_asset_sha256": content.get("close_reading", {}).get(
            "asset"
        ),
        "methods_sha256": content.get("methods"),
    }
    for key, value in checks.items():
        if _sha256_json(value) != component_hashes.get(key):
            raise ValueError(f"Frozen component hash mismatch: {key}")
    packet_sha256 = (
        content.get("close_reading", {}).get("source_packet", {}).get("sha256")
    )
    if packet_sha256 != component_hashes.get(
        "close_reading_source_packet_sha256"
    ):
        raise ValueError("Frozen close-reading source packet hash mismatch")
    method_hashes = component_hashes.get("method_intensity_asset_sha256")
    if not isinstance(method_hashes, dict):
        raise ValueError("Frozen assets lack method/intensity hashes")
    for method_id, method in content.get("methods", {}).items():
        for intensity, asset in method.get("intensities", {}).items():
            key = f"{method_id}:{intensity}"
            if _sha256_json(asset) != method_hashes.get(key):
                raise ValueError(f"Frozen method asset hash mismatch: {key}")


@lru_cache(maxsize=4)
def _load_frozen_cached(
    experiment_root_text: str, lock_sha256: str
) -> dict[str, Any]:
    del lock_sha256
    experiment_root = Path(experiment_root_text)
    lock_path = experiment_root / "method_assets" / LOCK_NAME
    lock = _load_json(lock_path)
    if lock.get("asset_schema") != ASSET_SCHEMA:
        raise ValueError(f"Unsupported frozen asset schema: {lock.get('asset_schema')}")
    relative_path = Path(str(lock.get("asset_path", "")))
    asset_path = (experiment_root / relative_path).resolve()
    method_assets_root = (experiment_root / "method_assets").resolve()
    if method_assets_root not in asset_path.parents:
        raise ValueError("Frozen asset path escapes method_assets")
    try:
        data = asset_path.read_bytes()
    except FileNotFoundError as exc:
        raise ValueError(f"Frozen asset bundle is missing: {asset_path}") from exc
    if _sha256_bytes(data) != lock.get("file_sha256"):
        raise ValueError("Frozen asset file hash does not match lock")
    try:
        bundle = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid frozen asset JSON: {asset_path}") from exc
    if not isinstance(bundle, dict) or bundle.get("schema_version") != ASSET_SCHEMA:
        raise ValueError("Invalid frozen asset bundle schema")
    content = bundle.get("assets")
    if not isinstance(content, dict):
        raise ValueError("Frozen asset bundle lacks assets object")
    content_sha256 = _sha256_json(content)
    if content_sha256 != bundle.get("content_sha256"):
        raise ValueError("Frozen asset content hash does not match bundle")
    if content_sha256 != lock.get("content_sha256"):
        raise ValueError("Frozen asset content hash does not match lock")
    _verify_frozen_components(content)
    return {
        "schema_version": ASSET_SCHEMA,
        "content_sha256": content_sha256,
        "file_sha256": str(lock["file_sha256"]),
        "asset_path": relative_path.as_posix(),
        "assets": content,
    }


def load_frozen_assets(experiment_root: Path) -> dict[str, Any]:
    """Load and verify the currently locked immutable method-asset bundle."""

    experiment_root = experiment_root.expanduser().resolve()
    lock_path = experiment_root / "method_assets" / LOCK_NAME
    try:
        lock_bytes = lock_path.read_bytes()
    except FileNotFoundError as exc:
        raise ValueError(
            f"Frozen method assets are missing; run freeze_method_assets first: {lock_path}"
        ) from exc
    return _load_frozen_cached(str(experiment_root), _sha256_bytes(lock_bytes))


def _unpack_vector(encoded: str) -> bytes:
    try:
        packed = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid frozen retrieval vector") from exc
    if len(packed) != HASH_VECTOR_BUCKETS * 2:
        raise ValueError("Frozen retrieval vector has the wrong size")
    return packed


@lru_cache(maxsize=8)
def _decoded_retrieval_index(
    content_sha256: str,
    kind: str,
    experiment_root_text: str,
    lock_sha256: str,
) -> tuple[tuple[dict[str, Any], bytes], ...]:
    del content_sha256
    frozen = _load_frozen_cached(experiment_root_text, lock_sha256)
    entries = frozen["assets"]["retrieval"][f"{kind}_index"]
    return tuple(
        (entry, _unpack_vector(str(entry["search_vector_b64"])))
        for entry in entries
    )


def _query_vector(text: str) -> tuple[list[int], list[int], int]:
    encoded, norm_sq = _hashed_search_vector(text)
    packed = _unpack_vector(encoded)
    values = list(struct.unpack(f">{HASH_VECTOR_BUCKETS}H", packed))
    nonzero = [index for index, value in enumerate(values) if value]
    return values, nonzero, norm_sq


def _vector_score(
    query_values: Sequence[int],
    query_nonzero: Sequence[int],
    query_norm_sq: int,
    packed_candidate: bytes,
    candidate_norm_sq: int,
) -> float:
    if query_norm_sq <= 0 or candidate_norm_sq <= 0:
        return 0.0
    dot = sum(
        query_values[index]
        * struct.unpack_from(">H", packed_candidate, index * 2)[0]
        for index in query_nonzero
    )
    return dot / math.sqrt(query_norm_sq * candidate_norm_sq)


def _cjk_ngrams(text: str, size: int) -> set[str]:
    normalized = "".join(CJK_RE.findall(text))
    if len(normalized) < size:
        return {normalized} if normalized else set()
    return {
        normalized[index : index + size]
        for index in range(len(normalized) - size + 1)
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _retrieve_examples(
    *,
    frozen: Mapping[str, Any],
    experiment_root: Path,
    neutral_zh: Sequence[Mapping[str, str]],
    kind: str,
    role: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query_text = "\n".join(paragraph["zh"] for paragraph in neutral_zh)
    normalized_query = _normalized_search_text(query_text)
    query_values, query_nonzero, query_norm_sq = _query_vector(normalized_query)
    lock_path = experiment_root / "method_assets" / LOCK_NAME
    lock_sha256 = _sha256_bytes(lock_path.read_bytes())
    decoded = _decoded_retrieval_index(
        str(frozen["content_sha256"]),
        kind,
        str(experiment_root),
        lock_sha256,
    )
    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for entry, packed in decoded:
        score = _vector_score(
            query_values,
            query_nonzero,
            query_norm_sq,
            packed,
            int(entry["search_norm_sq"]),
        )
        ranked.append((-score, str(entry["reference_id"]), entry))
    ranked.sort(key=lambda item: (item[0], item[1]))

    query_ngrams = _cjk_ngrams(query_text, NEAR_DUPLICATE_NGRAM)
    selected: list[dict[str, Any]] = []
    selected_ngrams: list[set[str]] = []
    seen_origins: set[str] = set()
    origin_field = "title" if kind == "target" else "author"
    for negative_score, _reference_id, entry in ranked:
        candidate_ngrams = _cjk_ngrams(
            str(entry["masked_text"]), NEAR_DUPLICATE_NGRAM
        )
        if _jaccard(query_ngrams, candidate_ngrams) > NEAR_DUPLICATE_JACCARD_MAX:
            continue
        if any(
            _jaccard(previous, candidate_ngrams) > NEAR_DUPLICATE_JACCARD_MAX
            for previous in selected_ngrams
        ):
            continue
        origin = str(entry[origin_field])
        if origin in seen_origins:
            continue
        text = str(entry["masked_text"])
        if _cjk_len(text) > MAX_EXAMPLE_CJK:
            raise ValueError("Frozen masked example exceeds retrieval bound")
        selected.append(
            {
                "reference_id": str(entry["reference_id"]),
                "role": role,
                "rank": len(selected) + 1,
                "text": text,
                "text_sha256": str(entry["masked_excerpt_sha256"]),
                "view": "entity_masked_v3",
                "source_split": "train",
                "retrieval_score": round(-negative_score, 8),
                "cjk_count": int(entry["masked_cjk_count"]),
            }
        )
        selected_ngrams.append(candidate_ngrams)
        seen_origins.add(origin)
        if len(selected) == RETRIEVAL_K:
            break
    if len(selected) != RETRIEVAL_K:
        raise ValueError(
            f"Could not retrieve {RETRIEVAL_K} non-duplicate {kind} examples; "
            f"found {len(selected)}"
        )
    metadata = {
        "query_source": "neutral_zh_only",
        "query_view": "entity_masked_v3",
        "query_sha256": _sha256_bytes(normalized_query.encode("utf-8")),
        "return_view": "entity_masked_v3",
        "k": RETRIEVAL_K,
        "selected_reference_ids": [item["reference_id"] for item in selected],
        "near_duplicate_8gram_jaccard_max": NEAR_DUPLICATE_JACCARD_MAX,
    }
    return selected, metadata


def _validate_paragraphs(
    paragraphs: Sequence[Mapping[str, str]],
    *,
    text_field: str,
    label: str,
) -> list[dict[str, str]]:
    if not isinstance(paragraphs, list) or not paragraphs:
        raise ValueError(f"{label} must be a non-empty list")
    validated: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    expected_keys = {"id", text_field}
    for index, paragraph in enumerate(paragraphs):
        if not isinstance(paragraph, dict) or set(paragraph) != expected_keys:
            raise ValueError(
                f"{label}[{index}] must contain exactly {sorted(expected_keys)}"
            )
        paragraph_id = paragraph["id"]
        text = paragraph[text_field]
        if not isinstance(paragraph_id, str) or not paragraph_id:
            raise ValueError(f"{label}[{index}] has an invalid paragraph ID")
        if paragraph_id in seen_ids:
            raise ValueError(f"{label} contains duplicate paragraph ID {paragraph_id}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{label}[{index}] has empty {text_field}")
        seen_ids.add(paragraph_id)
        validated.append({"id": paragraph_id, text_field: text})
    return validated


def _find_forbidden_critique_key(value: Any) -> str | None:
    forbidden_fragments = (
        "hidden_target",
        "hidden_targets",
        "evaluator_allocation",
        "held_out_original",
        "gold_zh",
    )
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(fragment in normalized for fragment in forbidden_fragments):
                return str(key)
            nested = _find_forbidden_critique_key(child)
            if nested:
                return nested
    elif isinstance(value, list):
        for child in value:
            nested = _find_forbidden_critique_key(child)
            if nested:
                return nested
    return None


def _method_asset_hashes(
    frozen: Mapping[str, Any], method_id: str, intensity: str
) -> dict[str, str]:
    assets = frozen["assets"]
    components = assets["component_hashes"]
    hashes = {
        "bundle_sha256": str(frozen["content_sha256"]),
        "bundle_file_sha256": str(frozen["file_sha256"]),
        "registry_definition_sha256": str(
            assets["registry"]["definition_sha256"]
        ),
        "style_prompt_sha256": str(
            assets["source_projections"]["style_prompt_sha256"]
        ),
        "method_config_definition_sha256": str(
            assets["methods"][method_id]["config_definition_sha256"]
        ),
        "method_intensity_asset_sha256": str(
            components["method_intensity_asset_sha256"][
                f"{method_id}:{intensity}"
            ]
        ),
        "train_statistics_sha256": str(components["train_statistics_sha256"]),
        "close_reading_source_packet_sha256": str(
            components["close_reading_source_packet_sha256"]
        ),
    }
    if assets["methods"][method_id]["intensities"][intensity].get("retrieval"):
        hashes["target_retrieval_index_sha256"] = str(
            components["target_retrieval_index_sha256"]
        )
        if method_id == "contrastive_examples":
            hashes["hard_negative_retrieval_index_sha256"] = str(
                components["hard_negative_retrieval_index_sha256"]
            )
    if method_id.startswith("llm_close_reading_"):
        hashes["close_reading_asset_sha256"] = str(
            components["close_reading_asset_sha256"]
        )
    return hashes


def build_method_request(
    *,
    experiment_root: Path,
    method_id: str,
    intensity: str,
    sample_id: str,
    english_semantic_source: list[dict[str, str]],
    neutral_zh: list[dict[str, str]],
    prior_output: list[dict[str, str]] | None = None,
    critique: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one deterministic style_transfer_method.v1 input request."""

    experiment_root = experiment_root.expanduser().resolve()
    frozen = load_frozen_assets(experiment_root)
    assets = frozen["assets"]
    methods = assets.get("methods", {})
    if method_id not in methods:
        raise ValueError(f"Unknown frozen method ID: {method_id}")
    intensities = methods[method_id].get("intensities", {})
    if intensity not in intensities:
        raise ValueError(
            f"Intensity {intensity!r} is not frozen for {method_id}: "
            f"{sorted(intensities)}"
        )
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be a non-empty string")
    english = _validate_paragraphs(
        english_semantic_source,
        text_field="en",
        label="english_semantic_source",
    )
    neutral = _validate_paragraphs(
        neutral_zh,
        text_field="zh",
        label="neutral_zh",
    )
    if [item["id"] for item in english] != [item["id"] for item in neutral]:
        raise ValueError("English and neutral paragraph IDs/order must match")
    static_asset = intensities[intensity]
    if static_asset.get("availability") != "ready":
        raise MethodAssetUnavailableError(
            f"{method_id}:{intensity} is unavailable: "
            f"{static_asset.get('reason_code', 'no frozen evidence')}"
        )
    if method_id != "self_critique_repair" and (
        prior_output is not None or critique is not None
    ):
        raise ValueError("prior_output and critique apply only to self_critique_repair")

    method_payload: dict[str, Any] = {
        "schema_version": PAYLOAD_SCHEMA,
        "builder": {"id": GENERATOR_ID, "version": GENERATOR_VERSION},
        "asset_hashes": _method_asset_hashes(frozen, method_id, intensity),
        "input_hashes": {
            "english_semantic_source_sha256": _sha256_json(english),
            "neutral_zh_sha256": _sha256_json(neutral),
        },
        "intensity_contract": static_asset["intensity_contract"],
        "evidence_source": static_asset["evidence_source"],
    }
    reference_examples: list[dict[str, Any]] = []

    if method_id == "neutral_only":
        method_payload["instructions"] = static_asset["instructions"]
    elif method_id.startswith("generic_author_style_"):
        method_payload["target_author_name"] = static_asset["target_author_name"]
        method_payload["instructions"] = static_asset["instructions"]
    elif method_id == "global_style_cards":
        method_payload["global_style_cards"] = static_asset[
            "global_style_cards"
        ]
    elif method_id == "scene_routed_style_cards":
        routes = _route_scenes_from_neutral(
            neutral, limit=int(static_asset["scene_limit"])
        )
        method_payload["scene_routing"] = {
            "source": "neutral_zh_only",
            "routes": routes,
        }
        method_payload["scene_style_cards"] = [
            static_asset["scene_style_cards"][route["scene"]] for route in routes
        ]
    elif method_id == "sentence_flow_cards":
        method_payload["flow_style_cards"] = static_asset["flow_style_cards"]
    elif method_id == "function_word_punctuation_dialogue_cards":
        method_payload["interpretable_style_cards"] = static_asset[
            "interpretable_style_cards"
        ]
    elif method_id == "retrieved_examples_only":
        target_examples, retrieval = _retrieve_examples(
            frozen=frozen,
            experiment_root=experiment_root,
            neutral_zh=neutral,
            kind="target",
            role="target_style_example",
        )
        reference_examples.extend(target_examples)
        method_payload["retrieved_target_examples"] = [
            {"reference_id": item["reference_id"], "rank": item["rank"]}
            for item in target_examples
        ]
        method_payload["retrieval"] = retrieval
    elif method_id == "scene_cards_plus_examples":
        routes = _route_scenes_from_neutral(
            neutral, limit=int(static_asset["scene_limit"])
        )
        target_examples, retrieval = _retrieve_examples(
            frozen=frozen,
            experiment_root=experiment_root,
            neutral_zh=neutral,
            kind="target",
            role="target_style_example",
        )
        reference_examples.extend(target_examples)
        method_payload["scene_routing"] = {
            "source": "neutral_zh_only",
            "routes": routes,
        }
        method_payload["scene_style_cards"] = [
            static_asset["scene_style_cards"][route["scene"]] for route in routes
        ]
        method_payload["retrieved_target_examples"] = [
            {"reference_id": item["reference_id"], "rank": item["rank"]}
            for item in target_examples
        ]
        method_payload["retrieval"] = retrieval
    elif method_id == "contrastive_examples":
        target_examples, target_retrieval = _retrieve_examples(
            frozen=frozen,
            experiment_root=experiment_root,
            neutral_zh=neutral,
            kind="target",
            role="target_style_example",
        )
        hard_examples, hard_retrieval = _retrieve_examples(
            frozen=frozen,
            experiment_root=experiment_root,
            neutral_zh=neutral,
            kind="hard_negative",
            role="hard_negative_example",
        )
        reference_examples.extend(target_examples)
        reference_examples.extend(hard_examples)
        method_payload["target_examples"] = [
            {"reference_id": item["reference_id"], "rank": item["rank"]}
            for item in target_examples
        ]
        method_payload["hard_negative_examples"] = [
            {"reference_id": item["reference_id"], "rank": item["rank"]}
            for item in hard_examples
        ]
        method_payload["retrieval"] = {
            "target": target_retrieval,
            "hard_negative": hard_retrieval,
        }
        method_payload["instructions"] = static_asset["instructions"]
    elif method_id == "llm_close_reading_style_definition":
        method_payload["close_reading_definition"] = static_asset[
            "close_reading_definition"
        ]
        method_payload["close_reading_provenance"] = {
            key: static_asset[key]
            for key in (
                "source_packet_path",
                "source_packet_sha256",
                "result_path",
                "result_file_sha256",
                "generation_provenance",
            )
        }
    elif method_id == "llm_close_reading_cards_statistical_gates":
        method_payload["validated_close_reading_cards"] = static_asset[
            "validated_close_reading_cards"
        ]
        method_payload["close_reading_provenance"] = {
            key: static_asset[key]
            for key in (
                "source_packet_path",
                "source_packet_sha256",
                "result_path",
                "result_file_sha256",
                "generation_provenance",
            )
        }
    elif method_id == "self_critique_repair":
        if prior_output is None or critique is None:
            raise ValueError(
                "self_critique_repair requires both prior_output and an independent critique"
            )
        prior = _validate_paragraphs(
            prior_output, text_field="zh", label="prior_output"
        )
        if [item["id"] for item in prior] != [item["id"] for item in neutral]:
            raise ValueError("prior_output paragraph IDs/order must match neutral_zh")
        forbidden_key = _find_forbidden_critique_key(critique)
        if forbidden_key:
            raise ValueError(
                f"Critique contains forbidden hidden/evaluator-derived key: {forbidden_key}"
            )
        try:
            _canonical_json_bytes(critique)
        except (TypeError, ValueError) as exc:
            raise ValueError("critique must be finite JSON-compatible data") from exc
        method_payload["best_prior_method_payload"] = {
            "paragraphs": prior,
            "sha256": _sha256_json(prior),
        }
        method_payload["independent_critique"] = critique
        method_payload["repair_contract"] = static_asset["repair_contract"]
    else:
        raise ValueError(f"No request builder for {method_id}")

    return {
        "sample_id": sample_id,
        "method_id": method_id,
        "intensity": intensity,
        "english_semantic_source": english,
        "neutral_zh": neutral,
        "method_payload": method_payload,
        "reference_examples": reference_examples,
    }


def _summary_for_cli(frozen: Mapping[str, Any]) -> dict[str, Any]:
    assets = frozen["assets"]
    methods = assets["methods"]
    unavailable = sorted(
        f"{method_id}:{intensity}"
        for method_id, method in methods.items()
        for intensity, asset in method["intensities"].items()
        if asset.get("availability") != "ready"
    )
    return {
        "schema_version": frozen["schema_version"],
        "content_sha256": frozen["content_sha256"],
        "file_sha256": frozen["file_sha256"],
        "asset_path": frozen["asset_path"],
        "method_count": len(methods),
        "target_train_book_count": assets["evidence_policy"][
            "target_train_book_count"
        ],
        "comparison_author_count": assets["evidence_policy"][
            "comparison_author_count"
        ],
        "target_retrieval_entries": assets["retrieval"]["target_entry_count"],
        "hard_negative_retrieval_entries": assets["retrieval"][
            "hard_negative_entry_count"
        ],
        "close_reading_source_packet": assets["close_reading"]["source_packet"],
        "close_reading_result_available": (
            assets["close_reading"]["asset"].get("availability") == "ready"
        ),
        "unavailable_method_intensities": unavailable,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze and verify deterministic style-transfer method assets."
    )
    parser.add_argument(
        "command", choices=("freeze", "verify"), help="Operation to perform."
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=(
            RESEARCH_ROOT
            / "generated/style_research/style_transfer_experiments"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    frozen = (
        freeze_method_assets(args.experiment_root)
        if args.command == "freeze"
        else load_frozen_assets(args.experiment_root)
    )
    print(json.dumps(_summary_for_cli(frozen), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
