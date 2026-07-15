#!/usr/bin/env python3
from __future__ import annotations

import hashlib
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
DERIVED_METHOD = "candidate_rerank"


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


def verify_sources(root: Path) -> None:
    protocol = json.loads(
        (root / "protocols/evaluation_protocol.v1.json").read_text(encoding="utf-8")
    )
    bindings = protocol.get("iteration3", {}).get("source_bindings", {})
    if not isinstance(bindings, dict) or not bindings:
        raise RuntimeError("Iteration-3 protocol lacks source bindings")
    for path_text, expected in bindings.items():
        observed = hashlib.sha256((REPO_ROOT / path_text).read_bytes()).hexdigest()
        if observed != expected:
            raise RuntimeError(f"Iteration-3 source binding mismatch: {path_text}")


def load_iteration3_sample_contract(base, experiment_root: Path, sample_set: str) -> dict[str, Any]:
    if sample_set != "iteration3_proxy_v1":
        return base._iteration3_original_load_sample_contract(experiment_root, sample_set)
    paths = base.sample_paths(experiment_root, sample_set)
    for label in ("runner", "allocation", "hidden", "summary", "method_ids"):
        base.require_file(paths[label], f"sample-set {label}")
    summary = base.load_json(paths["summary"])
    allocation = base.unique_by_id(base.iter_jsonl(paths["allocation"]), paths["allocation"])
    hidden = base.unique_by_id(base.iter_jsonl(paths["hidden"]), paths["hidden"])
    runner = base.unique_by_id(base.iter_jsonl(paths["runner"]), paths["runner"])
    method_payload = base.load_json(paths["method_ids"])
    method_ids = tuple(str(value) for value in method_payload.get("sample_ids", []))
    if len(method_ids) != len(set(method_ids)):
        raise base.EvaluationError("Iteration-3 method IDs contain duplicates")
    if set(allocation) != set(hidden) or set(allocation) != set(runner):
        raise base.EvaluationError("Iteration-3 runner/allocation/hidden IDs disagree")
    allowed_roles = {str(value) for value in method_payload.get("allowed_research_roles", [])}
    expected_method_ids = {
        sample_id
        for sample_id, row in allocation.items()
        if row.get("research_role") in allowed_roles
    }
    if not allowed_roles or set(method_ids) != expected_method_ids:
        raise base.EvaluationError("Iteration-3 method IDs disagree with roles")
    if int(method_payload.get("sample_count", -1)) != len(method_ids):
        raise base.EvaluationError("Iteration-3 method count is incorrect")
    for row in runner.values():
        if any(key in row for key in ("author", "book_title", "original_zh")):
            raise base.EvaluationError("Iteration-3 runner exposes target metadata")
    if summary.get("sample_set") != sample_set or int(summary.get("total_samples", -1)) != len(allocation):
        raise base.EvaluationError("Iteration-3 summary identity/count mismatch")
    for summary_key, path_key in (
        ("runner_manifest_sha256", "runner"),
        ("evaluator_allocation_sha256", "allocation"),
        ("hidden_targets_sha256", "hidden"),
    ):
        if base.canonical_jsonl_sha256(paths[path_key]) != summary.get(summary_key):
            raise base.EvaluationError(f"Iteration-3 summary mismatch: {summary_key}")
    if base.file_sha256(paths["method_ids"]) != summary.get("method_evaluation_ids_sha256"):
        raise base.EvaluationError("Iteration-3 method ID hash mismatch")

    selections: dict[str, dict[str, Any]] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for selection_id, path_key, hash_key in (
        ("screening_v1", "screening_ids", "screening_ids_sha256"),
        ("confirmation_v1", "confirmation_ids", "confirmation_ids_sha256"),
    ):
        path = base.require_file(paths[path_key], f"iteration-3 {selection_id}")
        payload = base.load_json(path)
        ids = tuple(sorted(str(value) for value in payload.get("sample_ids", [])))
        if payload.get("selection_id") != selection_id or not ids or len(ids) != len(set(ids)):
            raise base.EvaluationError(f"Iteration-3 {selection_id} artifact is invalid")
        if not set(ids).issubset(set(method_ids)) or int(payload.get("sample_count", -1)) != len(ids):
            raise base.EvaluationError(f"Iteration-3 {selection_id} membership/count mismatch")
        if base.file_sha256(path) != summary.get(hash_key):
            raise base.EvaluationError(f"Iteration-3 {selection_id} hash mismatch")
        selections[selection_id] = {
            "selection_id": selection_id,
            "path": path,
            "sha256": summary[hash_key],
            "sample_ids": ids,
        }
        payloads[selection_id] = payload
    screen = set(selections["screening_v1"]["sample_ids"])
    confirmation = set(selections["confirmation_v1"]["sample_ids"])
    if screen & confirmation or screen | confirmation != set(method_ids):
        raise base.EvaluationError("Iteration-3 screen/confirmation partition mismatch")

    expected = {
        "screening_v1": (24, 12, 3, 1),
        "confirmation_v1": (80, 36, 10, 3),
    }
    for selection_id, (own, cross, per_book, per_author) in expected.items():
        rows = [allocation[sample_id] for sample_id in selections[selection_id]["sample_ids"]]
        arms = Counter(str(row["benchmark_arm"]) for row in rows)
        if arms != Counter({"own_author_reconstruction": own, "cross_author_transfer": cross}):
            raise base.EvaluationError(f"Iteration-3 {selection_id} arm mismatch")
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
        if len(own_books) != 8 or set(own_books.values()) != {per_book}:
            raise base.EvaluationError(f"Iteration-3 {selection_id} own-book geometry mismatch")
        if len(cross_authors) != 12 or set(cross_authors.values()) != {per_author}:
            raise base.EvaluationError(f"Iteration-3 {selection_id} cross-author geometry mismatch")

    protocol_path = experiment_root / "protocols/evaluation_protocol.v1.json"
    protocol = base.load_json(base.require_file(protocol_path, "iteration-3 protocol"))
    calibration_ids = tuple(
        sorted(
            sample_id
            for sample_id, row in allocation.items()
            if row.get("research_role") == protocol["calibration"]["allowed_role"]
        )
    )
    if len(calibration_ids) != int(protocol["calibration"]["sample_count"]):
        raise base.EvaluationError("Iteration-3 calibration count mismatch")
    if set(method_ids) | set(calibration_ids) != set(allocation):
        raise base.EvaluationError("Iteration-3 role partition is incomplete")
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


def install_derived_method_support(base, payload) -> None:
    original_provenance = base.combination_provenance
    original_context = base.load_style_context

    def combination_provenance(**kwargs):
        method = kwargs["method"]
        if str(method["method_id"]) != DERIVED_METHOD:
            return original_provenance(**kwargs)
        root = base.run_root(
            kwargs["experiment_root"], kwargs["sample_set"], kwargs["run_id"]
        )
        errors: list[str] = []
        for sample_id in kwargs["contract"]["method_ids"]:
            path = root / f"method_outputs/{DERIVED_METHOD}/light/{sample_id}.json"
            if not path.exists():
                errors.append(f"{sample_id}:missing_derived_rerank_output")
        return {}, errors, {"derived_method": DERIVED_METHOD}

    def load_style_context(**kwargs):
        if kwargs["method_id"] != DERIVED_METHOD:
            return original_context(**kwargs)
        experiment_root = kwargs["experiment_root"]
        sample_set = kwargs["sample_set"]
        run_id = kwargs["run_id"]
        sample_id = kwargs["sample_id"]
        neutral = kwargs["neutral"]
        run_root = base.run_root(experiment_root, sample_set, run_id)
        protocol = base.load_json(
            experiment_root / "protocols/evaluation_protocol.v1.json"
        )
        source_run_id = str(
            protocol["iteration3"]["style_execution"]["source_run_id"]
        )
        source_root = base.run_root(
            experiment_root, sample_set, source_run_id
        )
        path = run_root / f"method_outputs/{DERIVED_METHOD}/light/{sample_id}.json"
        errors = list(kwargs.get("shared_errors", []))
        artifact: dict[str, Any] = {}
        result: dict[str, Any] = {}
        input_request: dict[str, Any] = {"reference_examples": [], "method_payload": {}}
        try:
            artifact = base.load_json(path)
            result = artifact.get("result", {})
            provenance = artifact.get("derived_provenance", {})
            decision_path = base.resolve_recorded_path(str(provenance["decision_path"]))
            if base.file_sha256(decision_path) != provenance.get("decision_file_sha256"):
                errors.append("derived_decision_file_hash_mismatch")
            decision = base.load_json(decision_path)
            decision_copy = dict(decision)
            decision_sha = decision_copy.pop("decision_sha256", None)
            if base.sha256_text(base.canonical_json(decision_copy)) != decision_sha:
                errors.append("derived_decision_content_hash_mismatch")
            rule_sha = base.sha256_text(
                base.canonical_json(protocol["iteration3"]["candidate_rerank"])
            )
            if provenance.get("selection_rule_sha256") != rule_sha:
                errors.append("derived_selection_rule_hash_mismatch")
            lock_content = kwargs["analysis_lock_binding"].get("content_sha256")
            if provenance.get("analysis_lock_content_sha256") != lock_content:
                errors.append("derived_analysis_lock_mismatch")
            selected_method = str(provenance.get("selected_method_id"))
            if provenance.get("source_run_id") != source_run_id:
                errors.append("derived_source_run_mismatch")
            if selected_method == "neutral_only":
                selected_path = source_root / f"neutral_translation/{sample_id}.json"
            else:
                if selected_method not in {
                    "aligned_pairs_light",
                    "aligned_pairs_edit_plan_light",
                    "microcards_only_light",
                    "rule_linked_microcards_light",
                }:
                    errors.append("derived_selected_method_invalid")
                selected_path = run_root / f"method_outputs/{selected_method}/light/{sample_id}.json"
                english = (
                    neutral.input_request.get("paragraphs", [])
                    if neutral.input_request
                    else []
                )
                if not english:
                    errors.append("derived_effective_english_missing")
                input_request = payload.build_method_request(
                    experiment_root=experiment_root,
                    method_id=selected_method,
                    intensity="light",
                    sample_id=sample_id,
                    english_semantic_source=english,
                    neutral_zh=neutral.paragraphs,
                )
            if base.file_sha256(selected_path) != provenance.get("selected_output_file_sha256"):
                errors.append("derived_selected_output_hash_mismatch")
        except Exception as exc:
            errors.append(f"derived_context_error:{exc}")
        rows = result.get("paragraphs", []) if isinstance(result, dict) else []
        paragraph_rows = [dict(row) for row in rows if isinstance(row, dict)]
        return base.GeneratedContext(
            sample_id=sample_id,
            result=result,
            paragraphs=paragraph_rows,
            text=base.paragraphs_to_text(paragraph_rows) if paragraph_rows else None,
            artifact=artifact,
            input_request=input_request,
            errors=sorted(set(errors)),
        )

    base.combination_provenance = combination_provenance
    base.load_style_context = load_style_context


def main() -> None:
    root = requested_experiment_root()
    if root is None:
        raise SystemExit("Iteration-3 evaluation requires --experiment-root")
    verify_sources(root)
    payload = load_module("iteration3_style_transfer_payloads", PAYLOAD_PATH)
    base = load_module("iteration3_evaluation_base", BASE_PATH)
    base._iteration3_original_load_sample_contract = base.load_sample_contract
    base.load_sample_contract = lambda experiment_root, sample_set: load_iteration3_sample_contract(
        base, experiment_root, sample_set
    )
    base._PAYLOAD_BUILDER = payload
    install_derived_method_support(base, payload)
    base.main()


if __name__ == "__main__":
    main()
