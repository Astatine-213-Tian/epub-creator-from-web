#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib

from experiments.validation.meter.build_cr_fysm_v3 import (
    calibrated_scores,
    canonical_lock_id,
    ensemble,
    family_probabilities,
    locked_family_contract,
    sha256_file,
)
from experiments.validation.meter.develop_cr_fysm_v4 import load_family_artifacts
from experiments.validation.construct.prepare_cr_fysm_v4_construct import (
    OUTPUT_DIR,
    PREREGISTRATION,
    PRIVATE_ITEMS,
    artifact_hashes,
)


SCORE_OUTPUT = OUTPUT_DIR / "construct_scores.v1.jsonl"
OPENED_MARKER = OUTPUT_DIR / "construct_scoring_opened.v1.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def validate_lock() -> dict[str, Any]:
    payload = read_json(PREREGISTRATION)
    if payload.get("status") != "locked_before_construct_scoring_and_rating":
        raise ValueError("construct scoring requires the locked preregistration")
    if payload.get("lock_id") != canonical_lock_id(payload):
        raise ValueError("construct preregistration lock_id is invalid")
    if payload.get("input_hashes") != artifact_hashes():
        raise ValueError("construct input hashes changed after lock")
    actual_prepared = {
        path: sha256_file(Path(path)) for path in payload["prepared_artifact_hashes"]
    }
    if actual_prepared != payload["prepared_artifact_hashes"]:
        raise ValueError("construct prepared artifacts changed after lock")
    return payload


def main() -> None:
    construct = validate_lock()
    if SCORE_OUTPUT.exists() or OPENED_MARKER.exists():
        raise ValueError("construct scoring has already been opened")
    marker_fd = os.open(OPENED_MARKER, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(marker_fd, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "construct_id": construct["construct_id"],
                "lock_id": construct["lock_id"],
                "state": "opened_before_private_text_loading",
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")

    v4_prereg = read_json(
        Path(
            "generated/style_research/style_transfer_experiments/iterations/"
            "content_resistant_v1/cr_fysm_v4/preregistration.v4.json"
        )
    )
    v3_prereg = read_json(Path(v4_prereg["paths"]["v3_preregistration"]))
    family_order, family_weights = locked_family_contract(v3_prereg["model_contract"])
    model_dir = Path(v4_prereg["paths"]["source_model_dir"])
    artifacts = load_family_artifacts(model_dir, family_order)
    calibrator = joblib.load(model_dir / "ensemble_platt_calibrator.joblib")

    items = read_jsonl(PRIVATE_ITEMS)
    if len(items) != construct["geometry"]["all_construct_items"]:
        raise ValueError("private construct item count differs from the lock")
    texts = [row["chinese_candidate"] for row in items]
    family_scores = family_probabilities(artifacts, texts)
    raw_scores = ensemble(family_scores, family_order, family_weights)
    probability_scores = calibrated_scores(calibrator, raw_scores)
    threshold = float(construct["source_meter"]["threshold"])

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        rows.append(
            {
                "item_id": item["item_id"],
                "blind_id": item["blind_id"],
                "sample_id": item["sample_id"],
                "chinese_sha256": item["chinese_sha256"],
                "raw_score": float(raw_scores[index]),
                "calibrated_probability": float(probability_scores[index]),
                "threshold": threshold,
                "predicted_target_style": bool(raw_scores[index] >= threshold),
                "family_scores": {
                    family: float(family_scores[family][index]) for family in family_order
                },
            }
        )
    SCORE_OUTPUT.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "construct_id": construct["construct_id"],
                "rows": len(rows),
                "threshold": threshold,
                "score_output": str(SCORE_OUTPUT),
                "score_output_sha256": sha256_file(SCORE_OUTPUT),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
