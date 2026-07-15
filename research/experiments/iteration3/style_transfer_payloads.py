#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence


PAYLOAD_SCHEMA = "style_transfer_method_payload.v3.constrained_rerank"
BUILDER_ID = "experiments.iteration3.style_transfer_payloads"
BUILDER_VERSION = 1
REFERENCE_WINDOW_PARAGRAPHS = 3
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
PLACEHOLDER_RE = re.compile(r"<(?:TERM|NUM|LATIN)>")
SENTENCE_RE = re.compile(r"[^。！？!?]+[。！？!?]?")
SPEECH_TAG_RE = re.compile(
    r"(?:[”」』][^。！？\n]{0,12}(?:说(?:道)?|问(?:道)?|答(?:道)?|喊(?:道)?|道)[，。！？：]"
    r"|(?:说(?:道)?|问(?:道)?|答(?:道)?|喊(?:道)?|道)：[“「『])"
)
CONNECTIVES = ("但是", "不过", "然而", "因此", "所以", "于是", "然后", "接着", "同时", "其实", "而且", "并且")
LAUGHTER_TERMS = ("笑道", "笑着", "笑了", "一笑", "微笑", "失笑", "笑问")
REACTION_TERMS = ("点头", "摇头", "一怔", "愣住", "沉默", "皱眉", "抬头", "低头", "看了一眼")
SUPPORTED_METHODS = {
    "neutral_only",
    "aligned_pairs_light",
    "aligned_pairs_edit_plan_light",
    "microcards_only_light",
    "rule_linked_microcards_light",
    "candidate_rerank",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_paragraphs(
    value: Sequence[Mapping[str, Any]], *, field: str, label: str
) -> list[dict[str, str]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a paragraph array")
    rows: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError(f"{label} contains a non-object")
        paragraph_id = item.get("id")
        text = item.get(field)
        if not isinstance(paragraph_id, str) or not paragraph_id:
            raise ValueError(f"{label} has an invalid paragraph ID")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{label} has empty {field} text")
        rows.append({"id": paragraph_id, field: text})
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"{label} has duplicate paragraph IDs")
    return rows


@lru_cache(maxsize=8)
def load_assets(root_text: str) -> dict[str, Any]:
    root = Path(root_text).expanduser().resolve()
    lock_path = root / "method_assets/style_transfer_payloads.v1.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    asset_path = root / str(lock["asset_path"])
    if file_sha256(asset_path) != lock["file_sha256"]:
        raise ValueError("Iteration-3 method asset file hash mismatch")
    bundle = json.loads(asset_path.read_text(encoding="utf-8"))
    assets = bundle.get("assets")
    if not isinstance(assets, dict) or sha256_json(assets) != lock["content_sha256"]:
        raise ValueError("Iteration-3 method asset content mismatch")
    hashes = assets.get("component_hashes", {})
    if sha256_json(assets.get("aligned_pairs")) != hashes.get("aligned_pairs_sha256"):
        raise ValueError("Iteration-3 aligned-pair hash mismatch")
    if sha256_json(assets.get("style_cards", {}).get("rule_linked_microcards")) != hashes.get("microcards_sha256"):
        raise ValueError("Iteration-3 microcard hash mismatch")
    # Current code is allowed to diverge from the source hash recorded by the
    # completed snapshot; component and bundle hashes remain authoritative.
    return {"lock": lock, "assets": assets}


def normalized_text(paragraphs: Sequence[Mapping[str, str]]) -> str:
    text = "\n".join(str(row.get("zh", "")) for row in paragraphs)
    return "".join(CJK_RE.findall(PLACEHOLDER_RE.sub("", text)))


def ngrams(text: str) -> Counter[str]:
    result: Counter[str] = Counter()
    for size in (2, 3, 4):
        result.update(text[index : index + size] for index in range(max(0, len(text) - size + 1)))
    return result


def cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(left[key] * right[key] for key in left.keys() & right.keys())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / max(left_norm * right_norm, 1e-12)


def structural_signature(paragraphs: Sequence[Mapping[str, str]]) -> tuple[float, ...]:
    texts = [str(row.get("zh", "")) for row in paragraphs]
    text = "\n".join(texts)
    cjk = max(len(CJK_RE.findall(text)), 1)
    sentences = [value for value in SENTENCE_RE.findall(text) if CJK_RE.search(value)]
    dialogue = sum(value.lstrip().startswith(("“", "‘", "「", "『", '"')) for value in texts)
    punctuation = sum(character in "，。！？；：、…—“”‘’「」『』" for character in text)
    return (
        dialogue / max(len(texts), 1),
        punctuation / cjk,
        sum(len(CJK_RE.findall(value)) for value in sentences) / max(len(sentences), 1) / 50.0,
        len(texts) / 60.0,
    )


def structural_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))
    return 1.0 / (1.0 + distance)


def retrieve_pairs(
    assets: Mapping[str, Any], neutral: Sequence[Mapping[str, str]], k: int = 4
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query_text = normalized_text(neutral)
    query_ngrams = ngrams(query_text)
    query_structure = structural_signature(neutral)
    ranked: list[tuple[float, str, dict[str, Any], float, float]] = []
    for pair in assets["aligned_pairs"]["pairs"]:
        candidate_text = normalized_text(pair["neutral_zh"])
        semantic = cosine(query_ngrams, ngrams(candidate_text))
        structural = structural_similarity(query_structure, structural_signature(pair["neutral_zh"]))
        score = 0.70 * semantic + 0.30 * structural
        ranked.append((score, str(pair["pair_id"]), pair, semantic, structural))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected: list[dict[str, Any]] = []
    books: set[str] = set()
    for score, _pair_id, pair, semantic, structural in ranked:
        book_hash = str(pair["source_book_hash"])
        if book_hash in books:
            continue
        books.add(book_hash)
        selected.append(
            {
                "reference_id": pair["pair_id"],
                "role": "aligned_pseudo_parallel_demonstration",
                "rank": len(selected) + 1,
                "combined_similarity": round(score, 8),
                "semantic_similarity": round(semantic, 8),
                "structural_similarity": round(structural, 8),
                "scene_type": pair["scene_type"],
                "neutral_version": pair["neutral_zh"],
                "target_style_version": pair["target_style_zh"],
                "source_view": pair["source_view"],
            }
        )
        if len(selected) == k:
            break
    if len(selected) != k:
        raise ValueError(f"Iteration-3 retrieval returned {len(selected)} pairs")
    return selected, {
        "query_source": "neutral_zh_only",
        "algorithm": "semantic_2_4gram_0_70_plus_structural_signature_0_30_v1",
        "pair_pool_count": assets["aligned_pairs"]["count"],
        "k": k,
        "book_diversity": "at_most_one_pair_per_source_book",
        "selected": [
            {
                "reference_id": row["reference_id"],
                "rank": row["rank"],
                "combined_similarity": row["combined_similarity"],
            }
            for row in selected
        ],
    }


def compact_reference(reference: Mapping[str, Any]) -> dict[str, Any]:
    neutral = validate_paragraphs(
        reference["neutral_version"], field="zh", label="reference neutral version"
    )
    target = validate_paragraphs(
        reference["target_style_version"],
        field="zh",
        label="reference target-style version",
    )
    if [row["id"] for row in neutral] != [row["id"] for row in target]:
        raise ValueError("Aligned reference paragraph IDs/order differ")
    if len(neutral) <= REFERENCE_WINDOW_PARAGRAPHS:
        start = 0
    else:
        paragraph_scores = [
            1.0 - cosine(ngrams(normalized_text([before])), ngrams(normalized_text([after])))
            for before, after in zip(neutral, target)
        ]
        candidates = [
            (sum(paragraph_scores[index : index + REFERENCE_WINDOW_PARAGRAPHS]), index)
            for index in range(len(neutral) - REFERENCE_WINDOW_PARAGRAPHS + 1)
        ]
        start = min(candidates, key=lambda item: (-item[0], item[1]))[1]
    stop = min(start + REFERENCE_WINDOW_PARAGRAPHS, len(neutral))
    return {
        **{
            key: value
            for key, value in reference.items()
            if key not in {"neutral_version", "target_style_version"}
        },
        "source_paragraph_count": len(neutral),
        "reference_representation": "max_change_contiguous_window_v1",
        "selected_window_start": start,
        "selected_window_count": stop - start,
        "neutral_version": neutral[start:stop],
        "target_style_version": target[start:stop],
    }


def opportunity_score(card_id: str, text: str) -> float:
    if card_id.endswith("laughter_tags"):
        return float(sum(text.count(term) for term in LAUGHTER_TERMS))
    if card_id.endswith("short_sentence_ratio"):
        sentences = [value for value in SENTENCE_RE.findall(text) if CJK_RE.search(value)]
        return float(sum(len(CJK_RE.findall(value)) >= 25 for value in sentences))
    if card_id.endswith("exclamation"):
        return float(text.count("！") + text.count("!"))
    if card_id.endswith("connective"):
        return float(sum(text.count(term) for term in CONNECTIVES))
    if card_id.endswith("simple_speech_tags"):
        return float(len(SPEECH_TAG_RE.findall(text)))
    if card_id.endswith("micro_reactions"):
        return float(sum(text.count(term) for term in REACTION_TERMS))
    if card_id.endswith("colon"):
        return float(text.count("：") + text.count(":"))
    if card_id.endswith("quote_marks_per_kcjk"):
        return float(sum(text.count(mark) for mark in "“”‘’「」『』"))
    return 0.0


def activate_microcards(
    cards: Sequence[Mapping[str, Any]], neutral: Sequence[Mapping[str, str]], limit: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    text = "\n".join(str(row["zh"]) for row in neutral)
    ranked = sorted(
        (
            (opportunity_score(str(card["card_id"]), text), str(card["card_id"]), dict(card))
            for card in cards
        ),
        key=lambda item: (-item[0], item[1]),
    )
    selected = [card for score, _card_id, card in ranked if score > 0][:limit]
    return selected, {
        "algorithm": "neutral_source_opportunity_count_v1",
        "max_active_microcards": limit,
        "selected_card_ids": [card["card_id"] for card in selected],
        "zero_card_allowed": True,
    }


def build_method_request(
    *,
    experiment_root: Path,
    method_id: str,
    intensity: str,
    sample_id: str,
    english_semantic_source: Sequence[Mapping[str, Any]],
    neutral_zh: Sequence[Mapping[str, Any]],
    prior_output: Sequence[Mapping[str, Any]] | None = None,
    critique: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if prior_output is not None or critique is not None:
        raise ValueError("Iteration-3 initial methods do not accept prior output or critique")
    if method_id not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown iteration-3 method: {method_id}")
    english = validate_paragraphs(english_semantic_source, field="en", label="english_semantic_source")
    neutral = validate_paragraphs(neutral_zh, field="zh", label="neutral_zh")
    if [row["id"] for row in english] != [row["id"] for row in neutral]:
        raise ValueError("English and neutral paragraph IDs/order differ")
    frozen = load_assets(str(Path(experiment_root).resolve()))
    assets = frozen["assets"]
    static = assets["methods"][method_id]["intensities"].get(intensity)
    if not isinstance(static, dict):
        raise ValueError(f"Method/intensity is not frozen: {method_id}:{intensity}")
    if method_id in {"candidate_rerank"}:
        return {
            "sample_id": sample_id,
            "method_id": method_id,
            "intensity": intensity,
            "english_semantic_source": english,
            "neutral_zh": neutral,
            "method_payload": static,
            "reference_examples": [],
        }
    payload: dict[str, Any] = {
        "schema_version": PAYLOAD_SCHEMA,
        "builder": {
            "id": BUILDER_ID,
            "version": BUILDER_VERSION,
            "source_sha256": file_sha256(Path(__file__)),
        },
        "asset_hashes": {
            "bundle_content_sha256": frozen["lock"]["content_sha256"],
            "aligned_pairs_sha256": assets["component_hashes"]["aligned_pairs_sha256"],
            "microcards_sha256": assets["component_hashes"]["microcards_sha256"],
            "method_intensity_sha256": assets["component_hashes"]["method_intensity_asset_sha256"][f"{method_id}:{intensity}"],
        },
        "input_hashes": {
            "english_semantic_source_sha256": sha256_json(english),
            "neutral_zh_sha256": sha256_json(neutral),
        },
        "intensity_contract": static["intensity_contract"],
        "evidence_source": static["evidence_source"],
        "instructions": static.get("instructions", []),
    }
    references: list[dict[str, Any]] = []
    if method_id not in {"neutral_only", "microcards_only_light"}:
        references, retrieval = retrieve_pairs(assets, neutral, k=int(static["evidence_source"]["k"]))
        references = [compact_reference(reference) for reference in references]
        retrieval["reference_representation"] = {
            "algorithm": "max_change_contiguous_window_v1",
            "paragraphs_per_pair": REFERENCE_WINDOW_PARAGRAPHS,
            "pair_ids_and_ranks_preserved": True,
        }
        payload["aligned_pair_retrieval"] = retrieval
        payload["aligned_pair_references"] = [
            {key: row[key] for key in ("reference_id", "rank", "scene_type")}
            for row in references
        ]
    if method_id == "aligned_pairs_edit_plan_light":
        payload["edit_plan_contract"] = static["edit_plan_contract"]
    if method_id in {"microcards_only_light", "rule_linked_microcards_light"}:
        active, activation = activate_microcards(
            static["rule_linked_microcards"], neutral, int(static["max_active_microcards"])
        )
        payload["activated_rule_linked_microcards"] = active
        payload["microcard_activation"] = activation
    return {
        "sample_id": sample_id,
        "method_id": method_id,
        "intensity": intensity,
        "english_semantic_source": english,
        "neutral_zh": neutral,
        "method_payload": payload,
        "reference_examples": references,
    }
