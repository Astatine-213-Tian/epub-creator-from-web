#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


from experiments.shared.paths import RESEARCH_ROOT


REPO_ROOT = RESEARCH_ROOT
BASE_PATH = REPO_ROOT / "experiments/iteration1/evaluate_style_transfer_methods.py"
PAYLOAD_PATH = Path(__file__).with_name("style_transfer_payloads.py")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def requested_experiment_root() -> Path | None:
    for index, value in enumerate(sys.argv):
        if value == "--experiment-root" and index + 1 < len(sys.argv):
            return Path(sys.argv[index + 1]).expanduser().resolve()
        if value.startswith("--experiment-root="):
            return Path(value.split("=", 1)[1]).expanduser().resolve()
    return None


def verify_iteration2_sources(root: Path) -> None:
    protocol_path = root / "protocols/evaluation_protocol.v1.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    bindings = protocol.get("iteration2", {}).get("source_bindings", {})
    if not isinstance(bindings, dict) or not bindings:
        raise RuntimeError("Iteration-2 evaluation protocol lacks source bindings")
    import hashlib

    for path_text, expected in bindings.items():
        path = REPO_ROOT / path_text
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(f"Iteration-2 source binding mismatch: {path_text}")


def load_iteration2_sample_contract(base, experiment_root: Path, sample_set: str) -> dict[str, Any]:
    if sample_set != "development_proxy_v1":
        return base._iteration2_original_load_sample_contract(experiment_root, sample_set)

    paths = base.sample_paths(experiment_root, sample_set)
    for label in ("runner", "allocation", "hidden", "summary", "method_ids"):
        base.require_file(paths[label], f"sample-set {label}")
    summary = base.load_json(paths["summary"])
    allocation = base.unique_by_id(base.iter_jsonl(paths["allocation"]), paths["allocation"])
    hidden = base.unique_by_id(base.iter_jsonl(paths["hidden"]), paths["hidden"])
    runner = base.unique_by_id(base.iter_jsonl(paths["runner"]), paths["runner"])
    method_ids_payload = base.load_json(paths["method_ids"])
    method_ids = tuple(str(value) for value in method_ids_payload.get("sample_ids", []))
    if len(method_ids) != len(set(method_ids)):
        raise base.EvaluationError("Frozen method-evaluation ID artifact contains duplicates")
    if set(allocation) != set(hidden) or set(allocation) != set(runner):
        raise base.EvaluationError("Runner, allocation, and hidden-target sample IDs disagree")

    allowed_roles = {
        str(value) for value in method_ids_payload.get("allowed_research_roles", [])
    }
    expected_method_ids = {
        sample_id
        for sample_id, row in allocation.items()
        if row.get("research_role") in allowed_roles
    }
    if not allowed_roles or set(method_ids) != expected_method_ids:
        raise base.EvaluationError("Frozen method IDs disagree with evaluator allocation")
    if int(method_ids_payload.get("sample_count", -1)) != len(method_ids):
        raise base.EvaluationError("Frozen method ID count is incorrect")
    prior_screen_ids = {
        sample_id
        for sample_id, row in allocation.items()
        if row.get("research_role") == "prior_iteration_screening"
    }
    if len(prior_screen_ids) != 36 or prior_screen_ids & set(method_ids):
        raise base.EvaluationError("Prior-iteration screening role contract mismatch")
    for row in runner.values():
        if any(key in row for key in ("author", "book_title", "original_zh")):
            raise base.EvaluationError("Runner manifest exposes evaluator-only target metadata")

    if summary.get("sample_set") != sample_set:
        raise base.EvaluationError("Sample summary has a different sample_set")
    if int(summary.get("total_samples", -1)) != len(allocation):
        raise base.EvaluationError("Sample summary total does not match allocation")
    for summary_key, path_key in (
        ("runner_manifest_sha256", "runner"),
        ("evaluator_allocation_sha256", "allocation"),
        ("hidden_targets_sha256", "hidden"),
    ):
        expected = summary.get(summary_key)
        if expected and base.canonical_jsonl_sha256(paths[path_key]) != expected:
            raise base.EvaluationError(f"Sample summary binding mismatch: {summary_key}")
    if base.file_sha256(paths["method_ids"]) != summary.get(
        "method_evaluation_ids_sha256"
    ):
        raise base.EvaluationError(
            "Sample summary binding mismatch: method_evaluation_ids_sha256"
        )

    selections: dict[str, dict[str, Any]] = {}
    selection_payloads: dict[str, dict[str, Any]] = {}
    for selection_id, path_key, hash_key in (
        ("screening_v1", "screening_ids", "screening_ids_sha256"),
        ("confirmation_v1", "confirmation_ids", "confirmation_ids_sha256"),
    ):
        selection_path = base.require_file(
            paths[path_key], f"frozen {selection_id} selection"
        )
        payload = base.load_json(selection_path)
        ids = tuple(sorted(str(value) for value in payload.get("sample_ids", [])))
        if payload.get("selection_id") != selection_id:
            raise base.EvaluationError(f"Frozen {selection_id} artifact has the wrong ID")
        if not ids or len(ids) != len(set(ids)):
            raise base.EvaluationError(f"Frozen {selection_id} IDs are empty or duplicated")
        if not set(ids).issubset(set(method_ids)):
            raise base.EvaluationError(f"Frozen {selection_id} contains non-method IDs")
        if int(payload.get("sample_count", -1)) != len(ids):
            raise base.EvaluationError(f"Frozen {selection_id} sample count is incorrect")
        expected_hash = summary.get(hash_key)
        if not isinstance(expected_hash, str) or base.file_sha256(selection_path) != expected_hash:
            raise base.EvaluationError(f"Sample summary binding mismatch: {hash_key}")
        selections[selection_id] = {
            "selection_id": selection_id,
            "path": selection_path,
            "sha256": expected_hash,
            "sample_ids": ids,
        }
        selection_payloads[selection_id] = payload

    screen_ids = set(selections["screening_v1"]["sample_ids"])
    confirmation_ids = set(selections["confirmation_v1"]["sample_ids"])
    if screen_ids & confirmation_ids:
        raise base.EvaluationError("Frozen screening and confirmation cohorts overlap")
    if screen_ids | confirmation_ids != set(method_ids):
        raise base.EvaluationError("Screening plus confirmation does not equal method cohort")
    if screen_ids & prior_screen_ids:
        raise base.EvaluationError("Iteration-2 screening overlaps iteration-1 screening")

    screen_design = selection_payloads["screening_v1"].get("design", {})
    expected = {
        "screening_v1": {
            "own": 24,
            "cross": 12,
            "per_book": int(screen_design.get("own_author_chunks_per_book", -1)),
            "per_author": int(screen_design.get("cross_author_chunks_per_author", -1)),
        },
        "confirmation_v1": {
            "own": int(selection_payloads["confirmation_v1"].get("own_author_rows", -1)),
            "cross": int(selection_payloads["confirmation_v1"].get("cross_author_rows", -1)),
            "per_book": 10,
            "per_author": 3,
        },
    }
    for selection_id, contract in expected.items():
        rows = [allocation[sample_id] for sample_id in selections[selection_id]["sample_ids"]]
        arms = Counter(str(row["benchmark_arm"]) for row in rows)
        if arms != Counter(
            {
                "own_author_reconstruction": contract["own"],
                "cross_author_transfer": contract["cross"],
            }
        ):
            raise base.EvaluationError(f"Frozen {selection_id} arm composition mismatch")
        own_books = Counter(
            str(row["book_title"])
            for row in rows
            if row["benchmark_arm"] == "own_author_reconstruction"
        )
        cross_authors = Counter(
            str(row["author"])
            for row in rows
            if row["benchmark_arm"] == "cross_author_transfer"
        )
        if len(own_books) != 8 or set(own_books.values()) != {contract["per_book"]}:
            raise base.EvaluationError(f"Frozen {selection_id} per-book composition mismatch")
        if len(cross_authors) != 12 or set(cross_authors.values()) != {
            contract["per_author"]
        }:
            raise base.EvaluationError(f"Frozen {selection_id} per-author composition mismatch")

    protocol_path = experiment_root / "protocols/evaluation_protocol.v1.json"
    protocol = base.load_json(base.require_file(protocol_path, "evaluation protocol"))
    calibration_ids = tuple(
        sorted(
            sample_id
            for sample_id, row in allocation.items()
            if row.get("research_role") == protocol["calibration"]["allowed_role"]
        )
    )
    if len(calibration_ids) != int(protocol["calibration"]["sample_count"]):
        raise base.EvaluationError("Calibration allocation count differs from protocol")
    if set(method_ids) | prior_screen_ids | set(calibration_ids) != set(allocation):
        raise base.EvaluationError("Iteration-2 role partition does not cover the sample set")
    return {
        "paths": paths,
        "summary": summary,
        "allocation": allocation,
        "hidden": hidden,
        "runner": runner,
        "method_ids": tuple(sorted(method_ids)),
        "frozen_selections": selections,
        "calibration_ids": calibration_ids,
        "protocol": protocol,
        "protocol_path": protocol_path,
    }


def main() -> None:
    root = requested_experiment_root()
    if root is None:
        raise SystemExit("Iteration-2 evaluation requires --experiment-root")
    verify_iteration2_sources(root)
    payload = load_module("iteration2_style_transfer_payloads", PAYLOAD_PATH)
    base = load_module("iteration2_evaluation_base", BASE_PATH)
    base._iteration2_original_load_sample_contract = base.load_sample_contract
    base.load_sample_contract = lambda experiment_root, sample_set: load_iteration2_sample_contract(
        base, experiment_root, sample_set
    )
    base._PAYLOAD_BUILDER = payload
    base.main()


if __name__ == "__main__":
    main()
