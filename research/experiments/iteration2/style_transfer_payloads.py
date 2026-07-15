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


from experiments.shared.paths import RESEARCH_ROOT


REPO_ROOT = RESEARCH_ROOT
PAYLOAD_SCHEMA = "style_transfer_method_payload.v2.aligned_pairs"
BUILDER_ID = "experiments.iteration2.style_transfer_payloads"
BUILDER_VERSION = 1
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
PLACEHOLDER_RE = re.compile(r"<(?:TERM|NUM|LATIN)>")
SUPPORTED_METHODS = {
    "neutral_only",
    "aligned_pairs_only",
    "aligned_pairs_plus_cards",
    "aligned_pairs_edit_plan",
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
def load_assets(experiment_root_text: str) -> dict[str, Any]:
    root = Path(experiment_root_text).expanduser().resolve()
    lock_path = root / "method_assets/style_transfer_payloads.v1.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    asset_path = root / str(lock["asset_path"])
    if file_sha256(asset_path) != lock["file_sha256"]:
        raise ValueError("Iteration-2 method asset file hash mismatch")
    bundle = json.loads(asset_path.read_text(encoding="utf-8"))
    assets = bundle.get("assets")
    if not isinstance(assets, dict):
        raise ValueError("Iteration-2 method asset bundle has no assets object")
    if sha256_json(assets) != lock["content_sha256"]:
        raise ValueError("Iteration-2 method asset content hash mismatch")
    component_hashes = assets.get("component_hashes", {})
    if sha256_json(assets.get("aligned_pairs")) != component_hashes.get(
        "aligned_pairs_sha256"
    ):
        raise ValueError("Aligned-pair component hash mismatch")
    # The frozen bundle records the original source hash as historical evidence.
    # Current reproduction code may move or be cleaned without changing the asset.
    return {"lock": lock, "bundle": bundle, "assets": assets}


def normalized_search_text(paragraphs: Sequence[Mapping[str, str]]) -> str:
    text = "\n".join(str(row.get("zh", "")) for row in paragraphs)
    text = PLACEHOLDER_RE.sub("", text)
    return "".join(CJK_RE.findall(text))


def ngram_counter(text: str) -> Counter[str]:
    values: Counter[str] = Counter()
    for size in (2, 3, 4):
        values.update(text[index : index + size] for index in range(max(0, len(text) - size + 1)))
    return values


def cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    common = left.keys() & right.keys()
    dot = sum(left[key] * right[key] for key in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / max(left_norm * right_norm, 1e-12)


def retrieve_pairs(
    assets: Mapping[str, Any], neutral_zh: Sequence[Mapping[str, str]], k: int = 3
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query_text = normalized_search_text(neutral_zh)
    query = ngram_counter(query_text)
    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for pair in assets["aligned_pairs"]["pairs"]:
        candidate_text = normalized_search_text(pair["neutral_zh"])
        score = cosine(query, ngram_counter(candidate_text))
        ranked.append((score, str(pair["pair_id"]), pair))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected: list[dict[str, Any]] = []
    seen_books: set[str] = set()
    for score, _pair_id, pair in ranked:
        book_hash = str(pair["source_book_hash"])
        if book_hash in seen_books:
            continue
        seen_books.add(book_hash)
        selected.append(
            {
                "reference_id": pair["pair_id"],
                "role": "aligned_pseudo_parallel_demonstration",
                "rank": len(selected) + 1,
                "similarity": round(score, 8),
                "scene_type": pair["scene_type"],
                "neutral_version": pair["neutral_zh"],
                "target_style_version": pair["target_style_zh"],
                "source_view": pair["source_view"],
            }
        )
        if len(selected) == k:
            break
    if len(selected) != k:
        raise ValueError(f"Aligned retrieval returned {len(selected)} pairs, expected {k}")
    return selected, {
        "query_source": "neutral_zh_only",
        "algorithm": "exact_character_2_4gram_cosine_v1",
        "pair_pool_count": assets["aligned_pairs"]["count"],
        "k": k,
        "book_diversity": "at_most_one_pair_per_source_book",
        "selected": [
            {
                "reference_id": row["reference_id"],
                "rank": row["rank"],
                "similarity": row["similarity"],
            }
            for row in selected
        ],
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
        raise ValueError("Iteration-2 initial methods do not accept prior output or critique")
    if method_id not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown iteration-2 method: {method_id}")
    english = validate_paragraphs(
        english_semantic_source, field="en", label="english_semantic_source"
    )
    neutral = validate_paragraphs(neutral_zh, field="zh", label="neutral_zh")
    if [row["id"] for row in english] != [row["id"] for row in neutral]:
        raise ValueError("English and neutral paragraph IDs/order differ")
    frozen = load_assets(str(Path(experiment_root).resolve()))
    assets = frozen["assets"]
    methods = assets.get("methods", {})
    if method_id not in methods or intensity not in methods[method_id]["intensities"]:
        raise ValueError(f"Method/intensity is not frozen: {method_id}:{intensity}")
    static = methods[method_id]["intensities"][intensity]
    if static.get("availability") != "ready":
        raise ValueError(f"Method asset is unavailable: {method_id}:{intensity}")

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
            "method_intensity_sha256": assets["component_hashes"][
                "method_intensity_asset_sha256"
            ][f"{method_id}:{intensity}"],
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
    if method_id != "neutral_only":
        references, retrieval = retrieve_pairs(assets, neutral, k=3)
        payload["aligned_pair_retrieval"] = retrieval
        payload["aligned_pair_references"] = [
            {
                "reference_id": row["reference_id"],
                "rank": row["rank"],
                "scene_type": row["scene_type"],
            }
            for row in references
        ]
    if method_id == "aligned_pairs_plus_cards":
        payload["global_style_cards"] = static["global_style_cards"]
        payload["evidence_precedence"] = "aligned_pairs_then_agreeing_cards"
    if method_id == "aligned_pairs_edit_plan":
        payload["edit_plan_contract"] = static["edit_plan_contract"]
    return {
        "sample_id": sample_id,
        "method_id": method_id,
        "intensity": intensity,
        "english_semantic_source": english,
        "neutral_zh": neutral,
        "method_payload": payload,
        "reference_examples": references,
    }
