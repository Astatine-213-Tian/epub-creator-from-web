#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib


from experiments.shared.paths import RESEARCH_ROOT
from workflows.author_style_meter_contract import CURRENT_SCORER_ID


REPO_ROOT = RESEARCH_ROOT

from experiments.iteration1.style_analysis_lock import validate_analysis_lock


PAYLOAD_PATH = Path(__file__).with_name("style_transfer_payloads.py")
TARGET_AUTHOR = "非天夜翔"
CANDIDATE_METHODS = (
    "aligned_pairs_light",
    "aligned_pairs_edit_plan_light",
    "microcards_only_light",
    "rule_linked_microcards_light",
)
INTENSITY = "light"
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
NUMBER_RE = re.compile(r"\d+(?:[.,:/-]\d+)*")
LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]*")
PLACEHOLDER_RE = re.compile(r"<(?:TERM|NUM|LATIN)>")
JSON_RESIDUE_RE = re.compile(r"(?:\}\s*,\s*\{|```|\[\s*\{|\}\s*\])")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def effective_english_artifact(source_root: Path, sample_id: str) -> dict[str, Any]:
    source = read_json(source_root / "english_semantic_source" / f"{sample_id}.json")
    qa = read_json(source_root / "english_source_qa" / f"{sample_id}.json")
    approved = qa.get("approved_for_neutral_translation")
    if approved is not qa.get("result", {}).get("approved"):
        raise RuntimeError(f"English QA approval mismatch: {sample_id}")
    if approved is True:
        return source
    for repair_round in range(1, 21):
        round_name = f"round_{repair_round:02d}"
        repair_path = (
            source_root / "english_source_repair" / round_name / f"{sample_id}.json"
        )
        qa_path = (
            source_root
            / "english_source_repair_qa"
            / round_name
            / f"{sample_id}.json"
        )
        if not repair_path.exists() and not qa_path.exists():
            break
        if not repair_path.exists() or not qa_path.exists():
            raise RuntimeError(f"Incomplete English repair round: {sample_id}")
        repair = read_json(repair_path)
        repair_qa = read_json(qa_path)
        repair_approved = repair_qa.get("approved_for_neutral_translation")
        if repair_approved is not repair_qa.get("result", {}).get("approved"):
            raise RuntimeError(f"English repair QA mismatch: {sample_id}")
        if repair_approved is True:
            if repair.get("stage") == "english_source_adjudication":
                provenance = repair_qa.get("adjudication_provenance", {})
                audit_value = provenance.get("audit_path")
                if not isinstance(audit_value, str):
                    raise RuntimeError(f"English adjudication audit missing: {sample_id}")
                audit_path = REPO_ROOT / audit_value
                audit = read_json(audit_path)
                expected = {
                    "status": "approved_by_independent_adjudication",
                    "sample_id": sample_id,
                    "frozen_repair_path": str(repair_path.relative_to(REPO_ROOT)),
                    "frozen_repair_sha256": file_sha256(repair_path),
                    "frozen_qa_path": str(qa_path.relative_to(REPO_ROOT)),
                    "frozen_qa_sha256": file_sha256(qa_path),
                }
                if any(audit.get(key) != value for key, value in expected.items()):
                    raise RuntimeError(
                        f"English adjudication audit mismatch: {sample_id}"
                    )
            return repair
    raise RuntimeError(f"No approved English source: {sample_id}")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_payload_module():
    spec = importlib.util.spec_from_file_location("iteration3_rerank_payloads", PAYLOAD_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load iteration-3 payload builder")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def paragraphs(artifact: Mapping[str, Any]) -> list[dict[str, str]]:
    result = artifact.get("result", {})
    values = result.get("paragraphs", []) if isinstance(result, Mapping) else []
    return [
        {"id": str(row.get("id", "")), "zh": str(row.get("zh", ""))}
        for row in values
        if isinstance(row, Mapping)
    ]


def text_of(rows: Sequence[Mapping[str, str]]) -> str:
    return "\n".join(str(row["zh"]) for row in rows)


def starts_dialogue(text: str) -> bool:
    return text.lstrip().startswith(("“", "‘", "「", "『", '"'))


def quotes_balanced(text: str) -> bool:
    return (
        text.count("“") == text.count("”")
        and text.count("‘") == text.count("’")
        and text.count("「") == text.count("」")
        and text.count("『") == text.count("』")
    )


def cjk_only(text: str) -> str:
    return "".join(CJK_RE.findall(text))


def introduced_copy_failure(
    neutral_text: str, candidate_text: str, request: Mapping[str, Any]
) -> bool:
    neutral = cjk_only(neutral_text)
    candidate = cjk_only(candidate_text)
    references: list[str] = []
    for row in request.get("reference_examples", []):
        if not isinstance(row, Mapping):
            continue
        for paragraph in row.get("target_style_version", []):
            if isinstance(paragraph, Mapping):
                references.append(cjk_only(str(paragraph.get("zh", ""))))
    neutral_ngrams = {neutral[index : index + 8] for index in range(max(0, len(neutral) - 7))}
    candidate_ngrams = {candidate[index : index + 8] for index in range(max(0, len(candidate) - 7))}
    introduced = candidate_ngrams - neutral_ngrams
    return any(
        gram in reference
        for gram in introduced
        for reference in references
        if len(reference) >= 8
    )


def hard_gate(
    neutral: Sequence[Mapping[str, str]],
    candidate: Sequence[Mapping[str, str]],
    request: Mapping[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    if [row["id"] for row in candidate] != [row["id"] for row in neutral]:
        failures.append("paragraph_id_or_order_mismatch")
        return {"status": "fail", "failures": failures}
    for source, output in zip(neutral, candidate):
        source_text = str(source["zh"])
        output_text = str(output["zh"])
        prefix = str(output["id"])
        if Counter(NUMBER_RE.findall(source_text)) != Counter(NUMBER_RE.findall(output_text)):
            failures.append(f"{prefix}:number_surface_mismatch")
        if Counter(PLACEHOLDER_RE.findall(source_text)) != Counter(PLACEHOLDER_RE.findall(output_text)):
            failures.append(f"{prefix}:placeholder_mismatch")
        if Counter(LATIN_RE.findall(source_text)) != Counter(LATIN_RE.findall(output_text)):
            failures.append(f"{prefix}:latin_token_mismatch")
        if starts_dialogue(source_text) != starts_dialogue(output_text):
            failures.append(f"{prefix}:dialogue_turn_surface_mismatch")
        if not quotes_balanced(output_text):
            failures.append(f"{prefix}:unbalanced_dialogue_quotes")
        if JSON_RESIDUE_RE.search(output_text):
            failures.append(f"{prefix}:json_residue")
        source_cjk = len(CJK_RE.findall(source_text))
        output_cjk = len(CJK_RE.findall(output_text))
        ratio = output_cjk / max(source_cjk, 1)
        if source_cjk >= 20 and not 0.50 <= ratio <= 1.80:
            failures.append(f"{prefix}:paragraph_cjk_ratio_out_of_bounds")
    neutral_text = text_of(neutral)
    candidate_text = text_of(candidate)
    ratio = len(CJK_RE.findall(candidate_text)) / max(len(CJK_RE.findall(neutral_text)), 1)
    if not 0.70 <= ratio <= 1.35:
        failures.append("aggregate_cjk_ratio_out_of_bounds")
    if introduced_copy_failure(neutral_text, candidate_text, request):
        failures.append("introduced_reference_8gram")
    return {"status": "pass" if not failures else "fail", "failures": failures}


def edit_ratio(neutral: str, candidate: str) -> float:
    return 1.0 - difflib.SequenceMatcher(a=neutral, b=candidate, autojunk=False).ratio()


class StyleScorer:
    def __init__(self, root: Path):
        scorer = root / "scorers" / CURRENT_SCORER_ID
        self.vectorizer = joblib.load(scorer / "vectorizer.joblib")
        self.classifier = joblib.load(scorer / "classifier.joblib")
        labels = read_json(scorer / "labels.json")
        self.labels = list(labels["labels"])
        self.target_index = self.labels.index(TARGET_AUTHOR)
        evaluator_path = REPO_ROOT / "experiments/iteration1/evaluate_style_transfer_methods.py"
        spec = importlib.util.spec_from_file_location("iteration3_rerank_evaluator", evaluator_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot load evaluator masker")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.masker = module.EntityMasker(REPO_ROOT / "datasets/masked/mask_terms.json")

    def score(self, text: str, author: str, title: str) -> dict[str, Any]:
        masked = self.masker.mask(text, author=author, title=title)
        matrix = self.vectorizer.transform([masked])
        values = [float(value) for value in self.classifier.decision_function(matrix)[0]]
        order = sorted(range(len(values)), key=lambda index: (-values[index], self.labels[index]))
        return {
            "target_margin": values[self.target_index],
            "target_rank": order.index(self.target_index) + 1,
            "predicted_author": self.labels[order[0]],
            "masked_input_sha256": sha256_text(masked),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the frozen iteration-3 candidate-rerank outputs.")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--sample-set", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--input-run-id",
        help=(
            "Run containing the frozen English and neutral prerequisites. "
            "Defaults to --run-id."
        ),
    )
    parser.add_argument("--selection-file", type=Path, required=True)
    parser.add_argument("--analysis-lock", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.experiment_root.expanduser().resolve()
    selection_path = args.selection_file.expanduser().resolve()
    lock = validate_analysis_lock(args.analysis_lock.expanduser().resolve(), root)
    protocol = read_json(root / "protocols/evaluation_protocol.v1.json")
    source_run_id = args.input_run_id or args.run_id
    style_execution = protocol["iteration3"]["style_execution"]
    if args.run_id != style_execution["style_run_id"]:
        raise RuntimeError("Candidate rerank style run does not match the frozen protocol")
    if source_run_id != style_execution["source_run_id"]:
        raise RuntimeError("Candidate rerank source run does not match the frozen protocol")
    rule = protocol["iteration3"]["candidate_rerank"]
    rule_sha = sha256_text(canonical_json(rule))
    bindings = protocol["iteration3"]["source_bindings"]
    for path_text, expected in bindings.items():
        if file_sha256(REPO_ROOT / path_text) != expected:
            raise RuntimeError(f"Iteration-3 source binding mismatch: {path_text}")
    selection = read_json(selection_path)
    ids = [str(value) for value in selection["sample_ids"]]
    allocation = {
        row["sample_id"]: row
        for row in iter_jsonl(root / f"sample_sets/{args.sample_set}.evaluator_allocation.jsonl")
    }
    payloads = load_payload_module()
    scorer = StyleScorer(root)
    run_root = root / "runs" / args.sample_set / args.run_id
    source_root = root / "runs" / args.sample_set / source_run_id
    for method_id in CANDIDATE_METHODS:
        config_path = (
            run_root
            / "run_configs"
            / f"style_transfer.{method_id}.{INTENSITY}.json"
        )
        config = read_json(config_path)
        expected = {
            "run_id": args.run_id,
            "input_run_id": source_run_id,
            "stage": "style_transfer",
            "method_id": method_id,
            "intensity": INTENSITY,
        }
        mismatches = [
            key for key, value in expected.items() if config.get(key) != value
        ]
        if mismatches:
            raise RuntimeError(
                f"Candidate run-config mismatch for {method_id}: {mismatches}"
            )
        if config.get("analysis_lock", {}).get("content_sha256") != lock.get(
            "content_sha256"
        ):
            raise RuntimeError(f"Candidate analysis-lock mismatch for {method_id}")
    output_root = run_root / "method_outputs/candidate_rerank/light"
    decision_root = run_root / "rerank_decisions/candidate_rerank/light"
    output_root.mkdir(parents=True, exist_ok=True)
    decision_root.mkdir(parents=True, exist_ok=True)
    selected_counts: Counter[str] = Counter()
    unresolved = 0
    for sample_id in ids:
        english_artifact = effective_english_artifact(source_root, sample_id)
        neutral_artifact = read_json(
            source_root / "neutral_translation" / f"{sample_id}.json"
        )
        if english_artifact.get("run_id") != source_run_id:
            raise RuntimeError(f"English source-run mismatch: {sample_id}")
        if neutral_artifact.get("run_id") != source_run_id:
            raise RuntimeError(f"Neutral source-run mismatch: {sample_id}")
        english = english_artifact["result"]["paragraphs"]
        neutral = neutral_artifact["result"]["paragraphs"]
        neutral_text = text_of(neutral)
        metadata = allocation[sample_id]
        neutral_score = scorer.score(neutral_text, metadata["author"], metadata["book_title"])
        candidates: list[dict[str, Any]] = []
        for method_id in CANDIDATE_METHODS:
            path = run_root / f"method_outputs/{method_id}/light/{sample_id}.json"
            artifact = read_json(path)
            if (
                artifact.get("run_id") != args.run_id
                or artifact.get("method_id") != method_id
                or artifact.get("intensity") != INTENSITY
            ):
                raise RuntimeError(
                    f"Candidate output provenance mismatch: {method_id}/{sample_id}"
                )
            output = paragraphs(artifact)
            request = payloads.build_method_request(
                experiment_root=root,
                method_id=method_id,
                intensity=INTENSITY,
                sample_id=sample_id,
                english_semantic_source=english,
                neutral_zh=neutral,
            )
            gate = hard_gate(neutral, output, request)
            text = text_of(output)
            score = scorer.score(text, metadata["author"], metadata["book_title"])
            candidates.append(
                {
                    "method_id": method_id,
                    "output_path": str(path.relative_to(REPO_ROOT)),
                    "output_file_sha256": file_sha256(path),
                    "output_sha256": artifact.get("output_sha256"),
                    "hard_gate": gate,
                    "target_margin": score["target_margin"],
                    "target_rank": score["target_rank"],
                    "paired_margin_lift": score["target_margin"] - neutral_score["target_margin"],
                    "edit_ratio": edit_ratio(neutral_text, text),
                    "paragraphs": output,
                }
            )
        eligible = [
            row
            for row in candidates
            if row["hard_gate"]["status"] == "pass" and row["paired_margin_lift"] > 0
        ]
        if eligible:
            max_margin = max(row["target_margin"] for row in eligible)
            near = [row for row in eligible if row["target_margin"] >= max_margin - 0.03]
            winner = min(near, key=lambda row: (row["edit_ratio"], row["method_id"]))
            selected_method = str(winner["method_id"])
            selected_paragraphs = winner["paragraphs"]
            selection_reason = "highest_margin_band_then_minimum_edit"
        else:
            winner = None
            selected_method = "neutral_only"
            selected_paragraphs = neutral
            selection_reason = "no_candidate_passed_and_improved"
            unresolved += 1
        selected_counts[selected_method] += 1
        decision = {
            "schema_version": 1,
            "created_at": utc_now(),
            "sample_id": sample_id,
            "sample_set": args.sample_set,
            "run_id": args.run_id,
            "source_run_id": source_run_id,
            "analysis_lock": lock,
            "selection_path": str(selection_path.relative_to(REPO_ROOT)),
            "selection_sha256": file_sha256(selection_path),
            "selection_rule_sha256": rule_sha,
            "neutral_target_margin": neutral_score["target_margin"],
            "candidates": [{key: value for key, value in row.items() if key != "paragraphs"} for row in candidates],
            "selected_method_id": selected_method,
            "selection_reason": selection_reason,
            "selected_output_file_sha256": winner["output_file_sha256"] if winner else file_sha256(source_root / "neutral_translation" / f"{sample_id}.json"),
            "original_target_used": False,
            "source_metadata_used_for_entity_masking": True,
            "target_author_margin_used_for_selection": True,
            "original_target_text_used_for_selection": False,
            "selection_conditioned_style_meter_diagnostic": True,
        }
        decision_sha = sha256_text(canonical_json(decision))
        decision_path = decision_root / f"{sample_id}.json"
        write_json(decision_path, {**decision, "decision_sha256": decision_sha})
        result = {
            "sample_id": sample_id,
            "method_id": "candidate_rerank",
            "intensity": INTENSITY,
            "paragraphs": selected_paragraphs,
            "style_cues_applied": [f"selected:{selected_method}"],
            "style_cues_skipped": [],
            "uncertainties": [] if winner else ["No generated candidate passed the frozen rerank gate."],
        }
        artifact = {
            "schema_version": 1,
            "run_id": args.run_id,
            "source_run_id": source_run_id,
            "stage": "derived_candidate_rerank",
            "sample_id": sample_id,
            "model": "deterministic_candidate_rerank_v1",
            "reasoning_effort": "none",
            "result": result,
            "output_sha256": sha256_text(canonical_json(result)),
            "derived_provenance": {
                "decision_path": str(decision_path.relative_to(REPO_ROOT)),
                "decision_file_sha256": file_sha256(decision_path),
                "decision_sha256": decision_sha,
                "selection_rule_sha256": rule_sha,
                "analysis_lock_content_sha256": lock["content_sha256"],
                "selected_method_id": selected_method,
                "selected_output_file_sha256": decision["selected_output_file_sha256"],
                "source_run_id": source_run_id,
            },
        }
        write_json(output_root / f"{sample_id}.json", artifact)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "sample_count": len(ids),
        "source_run_id": source_run_id,
        "selected_counts": dict(sorted(selected_counts.items())),
        "neutral_fallback_rows": unresolved,
        "analysis_role": "selection_conditioned_style_meter_assisted_diagnostic",
        "independent_style_success_claim_allowed": False,
        "selection_rule_sha256": rule_sha,
        "analysis_lock_content_sha256": lock["content_sha256"],
    }
    write_json(run_root / "rerank_decisions/candidate_rerank/light.summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
