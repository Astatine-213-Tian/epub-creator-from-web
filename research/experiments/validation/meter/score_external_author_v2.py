#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from experiments.validation.meter.build_cr_fysm_v3 import (
    ensemble,
    family_probabilities,
    locked_family_contract,
)
from experiments.validation.meter.develop_cr_fysm_v4 import load_family_artifacts
from experiments.validation.meter.preregister_external_author_scoring_v2 import (
    EXTERNAL_CLEAN,
    EXTERNAL_MASKED,
    OUTPUT as PREREGISTRATION,
    V4_PREREGISTRATION,
    canonical,
    lock_id,
    read_json,
    read_jsonl,
    sha256_file,
)


ROOT = PREREGISTRATION.parent
MARKER = ROOT / "external_scoring_opened.v2.json"
RESULT = ROOT / "external_results.v2.json"
AUTHOR_CSV = ROOT / "external_authors.v2.csv"
BOOK_CSV = ROOT / "external_books.v2.csv"
SEED = 20260716


def validate_lock() -> dict[str, Any]:
    payload = read_json(PREREGISTRATION)
    if payload.get("status") != "locked_before_external_scores":
        raise ValueError("external scoring preregistration is not locked")
    if payload.get("lock_id") != lock_id(payload):
        raise ValueError("external scoring lock_id is invalid")
    actual = {path: sha256_file(Path(path)) for path in payload["input_hashes"]}
    if actual != payload["input_hashes"]:
        changed = sorted(path for path in actual if actual[path] != payload["input_hashes"][path])
        raise ValueError(f"external scoring inputs changed: {changed}")
    return payload


def retained(rows: list[dict[str, Any]], boundary: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["author"], row["title"])].append(row)
    result: list[dict[str, Any]] = []
    for key in sorted(grouped):
        values = sorted(grouped[key], key=lambda row: int(row["chunk_index"]))
        if len(values) <= boundary * 2:
            raise ValueError(f"book has too few chunks after boundary exclusion: {key}")
        result.extend(values[boundary:-boundary])
    return result


def score_texts(
    rows: list[dict[str, Any]],
    *,
    preregistration: dict[str, Any],
) -> np.ndarray:
    v4 = read_json(V4_PREREGISTRATION)
    v3 = read_json(Path(v4["paths"]["v3_preregistration"]))
    family_order, family_weights = locked_family_contract(v3["model_contract"])
    model_dir = Path(v4["paths"]["source_model_dir"])
    artifacts = load_family_artifacts(model_dir, family_order)
    probabilities = family_probabilities(artifacts, [row["text"] for row in rows])
    return ensemble(probabilities, family_order, family_weights)


def group_rates(
    rows: list[dict[str, Any]], scores: np.ndarray, threshold: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_book: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_book[(row["author"], row["title"])].append(index)
    books = []
    for (author, title), indices in sorted(by_book.items()):
        values = scores[indices]
        books.append(
            {
                "author": author,
                "book": title,
                "rows": len(indices),
                "specificity": float(np.mean(values < threshold)),
                "false_positive_rate": float(np.mean(values >= threshold)),
                "mean_score": float(np.mean(values)),
                "p95_score": float(np.quantile(values, 0.95)),
            }
        )
    by_author: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in books:
        by_author[row["author"]].append(row)
    authors = [
        {
            "author": author,
            "books": len(values),
            "rows": sum(int(value["rows"]) for value in values),
            "specificity": float(np.mean([value["specificity"] for value in values])),
            "false_positive_rate": float(1.0 - np.mean([value["specificity"] for value in values])),
        }
        for author, values in sorted(by_author.items())
    ]
    return authors, books


def bootstrap_lower(author_rates: list[float], resamples: int = 5000) -> float:
    rng = np.random.default_rng(SEED)
    values = np.asarray(author_rates, dtype=np.float64)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        draws[index] = float(np.mean(rng.choice(values, size=len(values), replace=True)))
    return float(np.quantile(draws, 0.025))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    preregistration = validate_lock()
    if RESULT.exists() or MARKER.exists():
        raise ValueError("external scoring was already opened")
    descriptor = os.open(MARKER, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "challenge_id": preregistration["challenge_id"],
                "lock_id": preregistration["lock_id"],
                "status": "opened_before_external_text_loading",
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    boundary = int(preregistration["preprocessing"]["boundary_chunks_excluded_per_book"])
    clean = retained(read_jsonl(EXTERNAL_CLEAN), boundary)
    masked = retained(read_jsonl(EXTERNAL_MASKED), boundary)
    if [row["chunk_id"] for row in clean] != [row["chunk_id"] for row in masked]:
        raise ValueError("retained clean/masked external rows differ")
    masked_scores = score_texts(masked, preregistration=preregistration)
    clean_scores = score_texts(clean, preregistration=preregistration)
    threshold = float(preregistration["meter"]["threshold"])
    authors, books = group_rates(masked, masked_scores, threshold)
    overall = float(np.mean([row["specificity"] for row in authors]))
    clustered_lower = bootstrap_lower([row["specificity"] for row in authors])
    flips = float(np.mean((masked_scores >= threshold) != (clean_scores >= threshold)))
    gates = preregistration["gates"]
    gate_results = {
        "overall_specificity": overall >= float(gates["overall_specificity_min"]),
        "each_author_specificity": min(row["specificity"] for row in authors)
        >= float(gates["each_author_specificity_min"]),
        "each_book_specificity": min(row["specificity"] for row in books)
        >= float(gates["each_book_specificity_min"]),
        "author_cluster_bootstrap_lower": clustered_lower
        >= float(gates["author_cluster_bootstrap_lower_min"]),
        "clean_masked_threshold_flip": flips <= float(gates["clean_masked_threshold_flip_max"]),
        "authors_exact": len(authors) == int(gates["authors_exact"]),
        "books_exact": len(books) == int(gates["books_exact"]),
    }
    payload = {
        "schema_version": 1,
        "challenge_id": preregistration["challenge_id"],
        "lock_id": preregistration["lock_id"],
        "status": "pass" if all(gate_results.values()) else "fail",
        "rows_after_boundary": len(masked),
        "threshold": threshold,
        "overall_author_book_weighted_specificity": overall,
        "author_cluster_bootstrap_95pct_lower": clustered_lower,
        "clean_masked_threshold_flip_rate": flips,
        "minimum_author_specificity": min(row["specificity"] for row in authors),
        "minimum_book_specificity": min(row["specificity"] for row in books),
        "authors": authors,
        "books": books,
        "gate_results": gate_results,
        "result_inputs": {
            "masked_sha256": sha256_file(EXTERNAL_MASKED),
            "clean_sha256": sha256_file(EXTERNAL_CLEAN),
            "preregistration_sha256": sha256_file(PREREGISTRATION),
        },
    }
    write_csv(AUTHOR_CSV, authors)
    write_csv(BOOK_CSV, books)
    RESULT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
