from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence


ASSET_SCHEMA = "author_style_transfer_assets.v1"
PAYLOAD_SCHEMA = "style_transfer_method_payload.v4.full_regeneration"
METHOD_ID = "content_plan_combined_full_regeneration"
INTENSITY = "strong"
REFERENCE_WINDOW_PARAGRAPHS = 3
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
PLACEHOLDER_RE = re.compile(r"<(?:TERM|NUM|LATIN)>")
SENTENCE_RE = re.compile(r"[^。！？!?]+[。！？!?]?")


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
def _load_assets(path_text: str, expected_file_sha256: str) -> dict[str, Any]:
    path = Path(path_text).expanduser().resolve()
    observed_file_sha256 = file_sha256(path)
    if expected_file_sha256 and observed_file_sha256 != expected_file_sha256:
        raise ValueError(f"style-transfer asset file hash mismatch: {path}")
    bundle = json.loads(path.read_text(encoding="utf-8"))
    if bundle.get("schema_version") != 1 or bundle.get("asset_schema") != ASSET_SCHEMA:
        raise ValueError(f"unsupported style-transfer asset schema: {path}")
    assets = bundle.get("assets")
    if not isinstance(assets, dict):
        raise ValueError(f"style-transfer asset payload is missing: {path}")
    content_sha256 = sha256_json(assets)
    if content_sha256 != bundle.get("content_sha256"):
        raise ValueError(f"style-transfer asset content hash mismatch: {path}")
    component_hashes = assets.get("component_hashes") or {}
    expected_components = {
        "aligned_pairs_sha256": assets.get("aligned_pairs"),
        "style_definition_sha256": assets.get("style_definition"),
        "method_sha256": assets.get("method"),
    }
    for key, value in expected_components.items():
        if sha256_json(value) != component_hashes.get(key):
            raise ValueError(f"style-transfer component hash mismatch: {key}")
    return {
        "path": str(path),
        "file_sha256": observed_file_sha256,
        "content_sha256": content_sha256,
        "source": bundle.get("source") or {},
        "assets": assets,
    }


def load_assets(path: Path, *, expected_file_sha256: str = "") -> dict[str, Any]:
    return _load_assets(str(path.expanduser().resolve()), expected_file_sha256)


def normalized_text(paragraphs: Sequence[Mapping[str, str]]) -> str:
    text = "\n".join(str(row.get("zh", "")) for row in paragraphs)
    return "".join(CJK_RE.findall(PLACEHOLDER_RE.sub("", text)))


def ngrams(text: str) -> Counter[str]:
    result: Counter[str] = Counter()
    for size in (2, 3, 4):
        result.update(
            text[index : index + size]
            for index in range(max(0, len(text) - size + 1))
        )
    return result


def cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(left[key] * right[key] for key in left.keys() & right.keys())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / max(left_norm * right_norm, 1e-12)


def structural_signature(
    paragraphs: Sequence[Mapping[str, str]],
) -> tuple[float, ...]:
    texts = [str(row.get("zh", "")) for row in paragraphs]
    text = "\n".join(texts)
    cjk = max(len(CJK_RE.findall(text)), 1)
    sentences = [value for value in SENTENCE_RE.findall(text) if CJK_RE.search(value)]
    dialogue = sum(
        value.lstrip().startswith(("“", "‘", "「", "『", '"')) for value in texts
    )
    punctuation = sum(character in "，。！？；：、…—“”‘’「」『』" for character in text)
    return (
        dialogue / max(len(texts), 1),
        punctuation / cjk,
        sum(len(CJK_RE.findall(value)) for value in sentences)
        / max(len(sentences), 1)
        / 50.0,
        len(texts) / 60.0,
    )


def structural_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))
    return 1.0 / (1.0 + distance)


def compact_reference(reference: Mapping[str, Any]) -> dict[str, Any]:
    neutral = validate_paragraphs(
        reference["neutral_zh"], field="zh", label="reference neutral version"
    )
    target = validate_paragraphs(
        reference["target_style_zh"],
        field="zh",
        label="reference target-style version",
    )
    if [row["id"] for row in neutral] != [row["id"] for row in target]:
        raise ValueError("aligned reference paragraph IDs/order differ")
    if len(neutral) <= REFERENCE_WINDOW_PARAGRAPHS:
        start = 0
    else:
        paragraph_scores = [
            1.0
            - cosine(
                ngrams(normalized_text([before])),
                ngrams(normalized_text([after])),
            )
            for before, after in zip(neutral, target)
        ]
        candidates = [
            (
                sum(paragraph_scores[index : index + REFERENCE_WINDOW_PARAGRAPHS]),
                index,
            )
            for index in range(len(neutral) - REFERENCE_WINDOW_PARAGRAPHS + 1)
        ]
        start = min(candidates, key=lambda item: (-item[0], item[1]))[1]
    stop = min(start + REFERENCE_WINDOW_PARAGRAPHS, len(neutral))
    return {
        "reference_id": reference["pair_id"],
        "role": "aligned_neutral_to_target_transformation",
        "scene_type": reference["scene_type"],
        "source_view": reference["source_view"],
        "neutral_version": neutral[start:stop],
        "target_style_version": target[start:stop],
        "instruction": (
            "Infer reusable structural transformations only. Never copy wording, "
            "events, entities, imagery, or lore."
        ),
    }


def retrieve_pairs(
    assets: Mapping[str, Any], neutral: Sequence[Mapping[str, str]], k: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query_ngrams = ngrams(normalized_text(neutral))
    query_structure = structural_signature(neutral)
    ranked: list[tuple[float, str, Mapping[str, Any], float, float]] = []
    for pair in assets["aligned_pairs"]["pairs"]:
        semantic = cosine(query_ngrams, ngrams(normalized_text(pair["neutral_zh"])))
        structural = structural_similarity(
            query_structure, structural_signature(pair["neutral_zh"])
        )
        score = 0.70 * semantic + 0.30 * structural
        ranked.append((score, str(pair["pair_id"]), pair, semantic, structural))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected: list[dict[str, Any]] = []
    used_books: set[str] = set()
    retrieval_rows: list[dict[str, Any]] = []
    for score, pair_id, pair, semantic, structural in ranked:
        book_hash = str(pair["source_book_hash"])
        if book_hash in used_books:
            continue
        used_books.add(book_hash)
        compact = compact_reference(pair)
        compact["rank"] = len(selected) + 1
        compact["combined_similarity"] = round(score, 8)
        selected.append(compact)
        retrieval_rows.append(
            {
                "reference_id": pair_id,
                "rank": len(selected),
                "combined_similarity": round(score, 8),
                "semantic_similarity": round(semantic, 8),
                "structural_similarity": round(structural, 8),
            }
        )
        if len(selected) == k:
            break
    if len(selected) != k:
        raise ValueError(f"style-transfer retrieval returned {len(selected)} pairs")
    return selected, {
        "algorithm": "semantic_2_4gram_0_70_plus_structural_signature_0_30_v1",
        "query_source": "neutral_zh_only",
        "pair_pool_count": assets["aligned_pairs"]["count"],
        "k": k,
        "book_diversity": "at_most_one_pair_per_source_book",
        "selected": retrieval_rows,
    }


def compact_statistic(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "feature_id": row["feature_id"],
        "family": row["family"],
        "direction": row["direction"],
        "target_value": row["target_value"],
        "comparison_author_mean": row["comparison_author_mean"],
        "z_score": row["z_score"],
        "discovery_recurrence": row["discovery_recurrence"]["rate"],
        "validation_recurrence": row["validation_recurrence"]["rate"],
    }


def style_definition_projection(assets: Mapping[str, Any]) -> dict[str, Any]:
    source = assets["style_definition"]
    definition = source["definition"]
    return {
        "schema_version": "iteration4_prompt_style_definition.v1",
        "evidence_policy": source["evidence_policy"],
        "global_guardrails": definition["global_guardrails"],
        "dimensions": [
            {
                **{
                    key: value
                    for key, value in dimension.items()
                    if key not in {"validated_statistics", "close_reading_claim_ids"}
                },
                "validated_statistics": [
                    compact_statistic(row)
                    for row in dimension["validated_statistics"]
                ],
            }
            for dimension in definition["dimensions"]
        ],
        "counterexample_policy": definition["counterexample_policy"],
    }


def style_examples(assets: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "reference_id": row["example_id"],
            "role": "masked_target_style_exemplar",
            "scene_type": row["scene"],
            "target_style_text": row["text"],
            "source_view": row["source_view"],
            "instruction": row["instruction"],
        }
        for row in assets["style_definition"]["definition"]["masked_scene_examples"]
    ]


def build_method_request(
    *,
    asset_path: Path,
    asset_file_sha256: str,
    sample_id: str,
    english_semantic_source: Sequence[Mapping[str, Any]],
    neutral_zh: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    english = validate_paragraphs(
        english_semantic_source,
        field="en",
        label="english_semantic_source",
    )
    neutral = validate_paragraphs(neutral_zh, field="zh", label="neutral_zh")
    if [row["id"] for row in english] != [row["id"] for row in neutral]:
        raise ValueError("English and neutral paragraph IDs/order differ")
    frozen = load_assets(asset_path, expected_file_sha256=asset_file_sha256)
    assets = frozen["assets"]
    static = assets["method"]
    pair_rows, retrieval = retrieve_pairs(
        assets, neutral, int(static["evidence_source"]["aligned_pair_k"])
    )
    payload = {
        "schema_version": PAYLOAD_SCHEMA,
        "asset_hashes": {
            "bundle_content_sha256": frozen["content_sha256"],
            **assets["component_hashes"],
        },
        "input_hashes": {
            "english_semantic_source_sha256": sha256_json(english),
            "neutral_zh_sha256": sha256_json(neutral),
        },
        "generation_contract": static["generation_contract"],
        "evidence_source": static["evidence_source"],
        "instructions": static.get("instructions", []),
        "aligned_pair_retrieval": retrieval,
        "style_definition": style_definition_projection(assets),
        "content_plan_contract": static["content_plan_contract"],
    }
    return {
        "sample_id": sample_id,
        "method_id": METHOD_ID,
        "intensity": INTENSITY,
        "english_semantic_source": english,
        "neutral_zh": neutral,
        "method_payload": payload,
        "reference_examples": [*pair_rows, *style_examples(assets)],
    }
