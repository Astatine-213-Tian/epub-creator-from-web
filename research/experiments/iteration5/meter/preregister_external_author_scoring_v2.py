#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy
import sklearn


ROOT = Path(
    "generated/style_research/style_transfer_experiments/iterations/"
    "content_resistant_v1/cr_fysm_v4/external_confirmation_v2"
)
OUTPUT = ROOT / "scoring_preregistration.external.v2.json"
COLLECTION_LOCK = ROOT / "preregistration.external.v2.json"
SCORER = Path("experiments/iteration5/meter/score_external_author_v2.py")
EXTERNAL_MANIFEST = Path("datasets/external_author_challenge_v2/dataset_manifest.json")
EXTERNAL_CLEAN = Path("datasets/external_author_challenge_v2/unmasked/chunks.clean.jsonl")
EXTERNAL_MASKED = Path(
    "datasets/external_author_challenge_v2/masked/chunks.entity_masked_v3.jsonl"
)
ACTIVE_MANIFEST = Path("datasets/dataset_manifest.json")
ACTIVE_CLEAN = Path("datasets/unmasked/chunks.clean.jsonl")
V4_PREREGISTRATION = Path(
    "generated/style_research/style_transfer_experiments/iterations/"
    "content_resistant_v1/cr_fysm_v4/preregistration.v4.json"
)
V4_DEVELOPMENT = V4_PREREGISTRATION.parent / "development.v4.json"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lock_id(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("lock_id", None)
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def validate_roster() -> dict[str, Any]:
    collection = read_json(COLLECTION_LOCK)
    manifest = read_json(EXTERNAL_MANIFEST)
    active = read_json(ACTIVE_MANIFEST)
    expected = {
        (entry["author"], selected[0])
        for entry in collection["roster"]
        for selected in entry["selected"]
    }
    actual = {(row["author"], row["title"]) for row in manifest}
    if actual != expected or len(actual) != 10:
        raise ValueError("external manifest differs from the locked ten-book roster")
    authors = {author for author, _ in actual}
    if authors & {row["author"] for row in active}:
        raise ValueError("external confirmation authors overlap the active corpus")
    clean_rows = read_jsonl(EXTERNAL_CLEAN)
    masked_rows = read_jsonl(EXTERNAL_MASKED)
    if len(clean_rows) != len(masked_rows) or len(clean_rows) != 5119:
        raise ValueError("external clean/masked row geometry is incomplete")
    clean_ids = [row["chunk_id"] for row in clean_rows]
    masked_ids = [row["chunk_id"] for row in masked_rows]
    if clean_ids != masked_ids:
        raise ValueError("external clean and masked rows are not aligned")
    active_fingerprints = {
        hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
        for row in read_jsonl(ACTIVE_CLEAN)
    }
    overlap = sum(
        hashlib.sha256(row["text"].encode("utf-8")).hexdigest() in active_fingerprints
        for row in clean_rows
    )
    if overlap:
        raise ValueError(f"external clean text overlaps active text: {overlap}")
    return {
        "authors": len(authors),
        "books": len(actual),
        "rows_per_view_before_boundary": len(clean_rows),
        "active_clean_text_fingerprint_overlap": overlap,
    }


def main() -> None:
    if OUTPUT.exists():
        raise ValueError("external v2 scoring preregistration already exists")
    roster = validate_roster()
    v4 = read_json(V4_PREREGISTRATION)
    artifact_paths = [Path(path) for path in v4["source_artifact_paths"].values()]
    input_paths = [
        Path(__file__),
        SCORER,
        COLLECTION_LOCK,
        EXTERNAL_MANIFEST,
        EXTERNAL_CLEAN,
        EXTERNAL_MASKED,
        ACTIVE_MANIFEST,
        ACTIVE_CLEAN,
        V4_PREREGISTRATION,
        V4_DEVELOPMENT,
        Path(v4["paths"]["v3_preregistration"]),
        Path("experiments/iteration5/meter/build_cr_fysm_v3.py"),
        Path("experiments/iteration5/meter/develop_cr_fysm_v4.py"),
        Path("experiments/iteration5/meter/benchmark_content_resistant_meter.py"),
        Path("workflows/benchmark_author_style.py"),
        Path("uv.lock"),
        *artifact_paths,
    ]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "challenge_id": "CR-FYSM-v4-external-author-confirmation-v2",
        "status": "locked_before_external_scores",
        "roster_geometry": roster,
        "meter": {
            "meter_id": "CR-FYSM-v4",
            "v4_lock_id": v4["lock_id"],
            "threshold": v4["decision_policy"]["threshold"],
            "score_space": v4["decision_policy"]["score_space"],
            "refit": False,
            "retune": False,
        },
        "preprocessing": {
            "view": "entity_masked_v3",
            "boundary_chunks_excluded_per_book": 2,
            "clean_view_scored_for_invariance_only": True,
        },
        "estimand": {
            "primary": "equal-author, equal-book-within-author specificity",
            "row_weighting": "equal within each book",
            "missing_rows": "fail closed",
            "author_cluster_bootstrap_resamples": 5000,
            "bootstrap_seed": 20260716,
            "interval": "2.5th percentile lower bound",
        },
        "gates": {
            "overall_specificity_min": 0.90,
            "each_author_specificity_min": 0.90,
            "each_book_specificity_min": 0.80,
            "author_cluster_bootstrap_lower_min": 0.90,
            "clean_masked_threshold_flip_max": 0.05,
            "authors_exact": 5,
            "books_exact": 10,
        },
        "one_shot": {
            "opened_marker": str(ROOT / "external_scoring_opened.v2.json"),
            "result": str(ROOT / "external_results.v2.json"),
            "failure_requires_new_external_version": True,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "input_hashes": {
            str(path): sha256_file(path)
            for path in sorted(set(input_paths), key=lambda value: str(value))
        },
    }
    payload["lock_id"] = lock_id(payload)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "lock_id": payload["lock_id"], **roster}, indent=2))


if __name__ == "__main__":
    main()
