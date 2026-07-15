#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


from experiments.shared.paths import RESEARCH_ROOT


REPO_ROOT = RESEARCH_ROOT

from experiments.iteration1.style_analysis_lock import validate_analysis_lock  # noqa: E402


PAYLOAD_PATH = Path(__file__).with_name("style_transfer_payloads.py")
GENERATION_PATH = REPO_ROOT / "experiments/iteration1/run_style_transfer_generation.py"
DERIVED_METHOD = "independent_candidate_selector"
INTENSITY = "strong"
CANDIDATE_METHODS = (
    "aligned_pairs_full_regeneration",
    "style_definition_examples_full_regeneration",
    "aligned_pairs_style_definition_full_regeneration",
    "content_plan_combined_full_regeneration",
)
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
NUMBER_RE = re.compile(r"\d+(?:[.,:/-]\d+)*")
LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]*")
PLACEHOLDER_RE = re.compile(r"<(?:TERM|CONTENT|NUM|LATIN)>")
JSON_RESIDUE_RE = re.compile(r"(?:\}\s*,\s*\{|```|\[\s*\{|\}\s*\])")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
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


def english_paragraphs(artifact: Mapping[str, Any]) -> list[dict[str, str]]:
    result = artifact.get("result", {})
    values = result.get("paragraphs", []) if isinstance(result, Mapping) else []
    return [
        {"id": str(row.get("id", "")), "en": str(row.get("en", ""))}
        for row in values
        if isinstance(row, Mapping)
    ]


def text_of(rows: Sequence[Mapping[str, str]]) -> str:
    return "\n".join(str(row["zh"]) for row in rows)


def cjk_only(text: str) -> str:
    return "".join(CJK_RE.findall(text))


def starts_dialogue(text: str) -> bool:
    return text.lstrip().startswith(("“", "‘", "「", "『", '"'))


def quotes_balanced(text: str) -> bool:
    return (
        text.count("“") == text.count("”")
        and text.count("‘") == text.count("’")
        and text.count("「") == text.count("」")
        and text.count("『") == text.count("』")
    )


def introduced_reference_copy(
    neutral_text: str,
    candidate_text: str,
    request: Mapping[str, Any],
) -> bool:
    neutral = cjk_only(neutral_text)
    candidate = cjk_only(candidate_text)
    neutral_ngrams = {
        neutral[index : index + 8] for index in range(max(0, len(neutral) - 7))
    }
    candidate_ngrams = {
        candidate[index : index + 8]
        for index in range(max(0, len(candidate) - 7))
    }
    introduced = candidate_ngrams - neutral_ngrams
    references: list[str] = []
    for row in request.get("reference_examples", []):
        if not isinstance(row, Mapping):
            continue
        target_style_text = row.get("target_style_text")
        if isinstance(target_style_text, str):
            references.append(cjk_only(target_style_text))
        for field in ("target_style_version", "paragraphs"):
            values = row.get(field, [])
            if not isinstance(values, list):
                continue
            for paragraph in values:
                if isinstance(paragraph, Mapping):
                    references.append(cjk_only(str(paragraph.get("zh", ""))))
                elif isinstance(paragraph, str):
                    references.append(cjk_only(paragraph))
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
        return {"status": "fail", "failures": ["paragraph_id_or_order_mismatch"]}
    for source, output in zip(neutral, candidate):
        source_text = str(source["zh"])
        output_text = str(output["zh"])
        prefix = str(output["id"])
        if Counter(NUMBER_RE.findall(source_text)) != Counter(
            NUMBER_RE.findall(output_text)
        ):
            failures.append(f"{prefix}:number_surface_mismatch")
        if Counter(PLACEHOLDER_RE.findall(source_text)) != Counter(
            PLACEHOLDER_RE.findall(output_text)
        ):
            failures.append(f"{prefix}:placeholder_mismatch")
        if Counter(LATIN_RE.findall(source_text)) != Counter(
            LATIN_RE.findall(output_text)
        ):
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
    ratio = len(CJK_RE.findall(candidate_text)) / max(
        len(CJK_RE.findall(neutral_text)), 1
    )
    if not 0.70 <= ratio <= 1.35:
        failures.append("aggregate_cjk_ratio_out_of_bounds")
    if introduced_reference_copy(neutral_text, candidate_text, request):
        failures.append("introduced_reference_8gram")
    return {"status": "pass" if not failures else "fail", "failures": failures}


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
            return repair
    raise RuntimeError(f"No approved English source: {sample_id}")


def validate_selector_result(
    result: Mapping[str, Any], candidate_ids: Sequence[str]
) -> list[str]:
    errors: list[str] = []
    expected = set(candidate_ids)
    selected = result.get("selected_candidate_id")
    if selected not in expected:
        errors.append("selected_candidate_id_invalid")
    rankings = result.get("rankings")
    if not isinstance(rankings, list):
        return [*errors, "rankings_missing"]
    observed = [row.get("candidate_id") for row in rankings if isinstance(row, Mapping)]
    if len(observed) != len(candidate_ids) or set(observed) != expected:
        errors.append("rankings_candidate_set_mismatch")
    observed_ranks = [row.get("rank") for row in rankings if isinstance(row, Mapping)]
    if set(observed_ranks) != set(range(1, len(candidate_ids) + 1)):
        errors.append("rankings_rank_set_mismatch")
    for row in rankings:
        if not isinstance(row, Mapping):
            errors.append("ranking_row_invalid")
            continue
        for key in ("semantic_fidelity", "naturalness", "style_adherence"):
            value = row.get(key)
            if not isinstance(value, int) or not 1 <= value <= 5:
                errors.append(f"ranking_{key}_invalid")
        if not isinstance(row.get("hard_semantic_error"), bool):
            errors.append("ranking_hard_semantic_error_invalid")
    selected_rows = [
        row
        for row in rankings
        if isinstance(row, Mapping) and row.get("candidate_id") == selected
    ]
    if len(selected_rows) != 1 or selected_rows[0].get("rank") != 1:
        errors.append("selected_candidate_not_rank_one")
    elif selected_rows[0].get("hard_semantic_error") is True:
        errors.append("selected_candidate_hard_semantic_error")
    return sorted(set(errors))


def build_blind_request(
    *,
    sample_id: str,
    english: Sequence[Mapping[str, str]],
    style_definition: Mapping[str, Any],
    eligible: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "english_semantic_source": [dict(row) for row in english],
        "style_definition": dict(style_definition),
        "candidates": [
            {
                "candidate_id": str(row["candidate_id"]),
                "paragraphs": [dict(paragraph) for paragraph in row["paragraphs"]],
            }
            for row in eligible
        ],
    }


def validate_selector_admission(
    generation: Any,
    args: argparse.Namespace,
    analysis_lock: Mapping[str, Any],
    sample_ids: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    selection_args = argparse.Namespace(
        experiment_root=args.experiment_root.expanduser().resolve(),
        sample_set=args.sample_set,
        stage="style_transfer",
        sample_id=[],
        selection_file=args.selection_file.expanduser().resolve(),
        limit=None,
    )
    selection = generation.execution_selection(selection_args, sample_ids)
    registered = generation.validate_registered_execution_selection(
        selection_args, selection
    )
    selection_id = selection["execution_selection_id"]
    if registered is None and selection_id != "final_validation_v1":
        raise RuntimeError(f"Selector selection is not registered: {selection_id}")
    if selection_id == "development_v1":
        if args.promotion_file or args.final_lock:
            raise RuntimeError(
                "Development selector does not accept outcome-dependent admission files"
            )
        return selection, {
            "admission_stage": "preregistered_development_selector",
            "selection_id": selection_id,
            "official_efficacy_evidence": False,
            "analysis_lock": dict(analysis_lock),
        }

    admission_args = argparse.Namespace(
        experiment_root=selection_args.experiment_root,
        sample_set=args.sample_set,
        stage="style_transfer",
        analysis_lock=args.analysis_lock.expanduser().resolve(),
        method_id=DERIVED_METHOD,
        intensity=INTENSITY,
        base_method_id=None,
        base_intensity=None,
        provisional_shortlist=None,
        refinement_contract=None,
        promotion_file=(
            args.promotion_file.expanduser().resolve()
            if args.promotion_file
            else None
        ),
        final_lock=(args.final_lock.expanduser().resolve() if args.final_lock else None),
    )
    admission = generation.validate_execution_admission(admission_args, selection)
    if not isinstance(admission, dict):
        raise RuntimeError("Selector execution admission is missing")
    return selection, admission


def selector_prompt(template: str, request: Mapping[str, Any]) -> str:
    return template.rstrip() + "\n\n## Frozen input\n\n```json\n" + json.dumps(
        request, ensure_ascii=False, indent=2
    ) + "\n```\n"


def invoke_selector(
    *,
    generation: Any,
    prompt_template: str,
    schema_path: Path,
    request: Mapping[str, Any],
    model: str,
    reasoning_effort: str,
    timeout_seconds: int,
    attempts: int,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    codex_binary = generation.resolved_codex_binary()
    for attempt in range(1, attempts + 1):
        started_at = utc_now()
        with tempfile.TemporaryDirectory(prefix="iteration4-selector-") as temp:
            sandbox_dir = Path(temp)
            response_path = sandbox_dir / "response.json"
            profile_path = sandbox_dir / "isolation.sb"
            profile_path.write_text(
                generation.external_sandbox_profile(), encoding="utf-8"
            )
            isolated_schema = sandbox_dir / "output.schema.json"
            isolated_schema.write_bytes(schema_path.read_bytes())
            command = [
                "/usr/bin/sandbox-exec",
                "-f",
                str(profile_path),
                str(codex_binary),
                "exec",
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--skip-git-repo-check",
                "--dangerously-bypass-approvals-and-sandbox",
                "--disable",
                "shell_tool",
                "--disable",
                "unified_exec",
                "--disable",
                "browser_use",
                "--disable",
                "browser_use_external",
                "--disable",
                "browser_use_full_cdp_access",
                "--disable",
                "in_app_browser",
                "--disable",
                "computer_use",
                "--disable",
                "apps",
                "--disable",
                "multi_agent",
                "--cd",
                str(sandbox_dir),
                "--model",
                model,
                "--config",
                f'model_reasoning_effort="{reasoning_effort}"',
                "--output-schema",
                str(isolated_schema),
                "--output-last-message",
                str(response_path),
                "--json",
                "-",
            ]
            try:
                completed = subprocess.run(
                    command,
                    input=selector_prompt(prompt_template, request),
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    env=generation.minimal_child_environment(sandbox_dir),
                    cwd=sandbox_dir,
                )
            except subprocess.TimeoutExpired:
                failures.append({"attempt": attempt, "error": "timeout"})
                continue
            result: dict[str, Any] | None = None
            errors: list[str] = []
            if completed.returncode != 0:
                errors.append(f"codex_exit_{completed.returncode}")
            elif not response_path.exists():
                errors.append("response_file_missing")
            else:
                try:
                    result = read_json(response_path)
                    errors.extend(
                        validate_selector_result(
                            result, [row["candidate_id"] for row in request["candidates"]]
                        )
                    )
                except Exception as exc:
                    errors.append(f"response_parse_error:{exc}")
            response_id = generation.parse_thread_id(completed.stdout)
            usage = None
            for line in completed.stdout.splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict) and isinstance(event.get("usage"), dict):
                    usage = event["usage"]
            if not errors and result is not None:
                return {
                    "status": "success",
                    "attempt": attempt,
                    "started_at": started_at,
                    "completed_at": utc_now(),
                    "response_id": response_id,
                    "response_usage": usage,
                    "result": result,
                    "prior_failures": failures,
                }
            failures.append(
                {
                    "attempt": attempt,
                    "errors": errors,
                    "stderr_tail": completed.stderr[-2000:],
                    "response_id": response_id,
                }
            )
    return {"status": "failed", "failures": failures}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the meter-blind Iteration-4 independent selector outputs."
    )
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--sample-set", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-run-id", required=True)
    parser.add_argument("--selection-file", type=Path, required=True)
    parser.add_argument("--analysis-lock", type=Path, required=True)
    parser.add_argument("--promotion-file", type=Path)
    parser.add_argument("--final-lock", type=Path)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.experiment_root.expanduser().resolve()
    selection_path = args.selection_file.expanduser().resolve()
    lock = validate_analysis_lock(args.analysis_lock.expanduser().resolve(), root)
    protocol = read_json(root / "protocols/evaluation_protocol.v1.json")
    if args.run_id != protocol["style_run_id"]:
        raise RuntimeError("Selector run does not match the frozen style run")
    if args.input_run_id != protocol["source_run_id"]:
        raise RuntimeError("Selector input does not match the frozen source run")
    selector_contract = protocol["selector_independence"]
    if args.model != selector_contract["selection_model"]:
        raise RuntimeError("Selector model does not match the frozen protocol")
    if args.reasoning_effort != selector_contract["reasoning_effort"]:
        raise RuntimeError("Selector reasoning effort does not match the frozen protocol")
    expected_candidates = ["neutral_only", *CANDIDATE_METHODS]
    if selector_contract["candidate_methods"] != expected_candidates:
        raise RuntimeError("Selector candidate roster does not match the implementation")
    for path_text, expected in protocol["iteration4"]["source_bindings"].items():
        if file_sha256(REPO_ROOT / path_text) != expected:
            raise RuntimeError(f"Iteration-4 source binding mismatch: {path_text}")
    rule_sha = sha256_text(canonical_json(selector_contract))
    selection = read_json(selection_path)
    sample_ids = [str(value) for value in selection["sample_ids"]]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("Selector selection contains duplicate sample IDs")

    method_lock = read_json(root / "method_assets/style_transfer_payloads.v1.lock.json")
    method_asset_path = root / str(method_lock["asset_path"])
    if file_sha256(method_asset_path) != method_lock["file_sha256"]:
        raise RuntimeError("Style-transfer method asset hash mismatch")
    method_asset = read_json(method_asset_path)
    style_definition = method_asset["assets"]["style_definition"]["definition"]
    prompt_path = root / "prompts/independent_candidate_selector.v1.md"
    schema_path = root / "schemas/independent_candidate_selector_output.v1.schema.json"
    prompt_template = prompt_path.read_text(encoding="utf-8")
    generation = load_module("iteration4_selector_generation", GENERATION_PATH)
    execution_selection, execution_admission = validate_selector_admission(
        generation, args, lock, sample_ids
    )
    selection_id = str(execution_selection["execution_selection_id"])
    payloads = load_module("iteration4_selector_payloads", PAYLOAD_PATH)

    run_root = root / "runs" / args.sample_set / args.run_id
    source_root = root / "runs" / args.sample_set / args.input_run_id
    output_root = run_root / f"method_outputs/{DERIVED_METHOD}/{INTENSITY}"
    decision_root = run_root / f"selector_decisions/{DERIVED_METHOD}/{INTENSITY}"
    output_root.mkdir(parents=True, exist_ok=True)
    decision_root.mkdir(parents=True, exist_ok=True)

    config = {
        "schema_version": 1,
        "stage": "derived_independent_candidate_selector",
        "run_id": args.run_id,
        "input_run_id": args.input_run_id,
        "sample_set": args.sample_set,
        "selection_path": str(selection_path.relative_to(REPO_ROOT)),
        "selection_sha256": file_sha256(selection_path),
        "selection_id": selection_id,
        "execution_selection": {
            key: value
            for key, value in execution_selection.items()
            if not key.startswith("_")
        },
        "execution_admission": execution_admission,
        "analysis_lock": lock,
        "method_id": DERIVED_METHOD,
        "intensity": INTENSITY,
        "candidate_methods": list(CANDIDATE_METHODS),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "prompt_path": str(prompt_path.relative_to(REPO_ROOT)),
        "prompt_sha256": file_sha256(prompt_path),
        "schema_path": str(schema_path.relative_to(REPO_ROOT)),
        "schema_sha256": file_sha256(schema_path),
        "style_definition_asset_content_sha256": method_lock["content_sha256"],
        "selection_rule_sha256": rule_sha,
        "frozen_style_meter_visible": False,
        "method_labels_visible": False,
        "original_target_visible": False,
    }
    write_json(
        run_root
        / f"run_configs/{DERIVED_METHOD}.{INTENSITY}.{selection_id}.json",
        config,
    )

    def build_one(sample_id: str) -> dict[str, Any]:
        output_path = output_root / f"{sample_id}.json"
        if output_path.exists():
            return {"sample_id": sample_id, "status": "existing"}
        english_artifact = effective_english_artifact(source_root, sample_id)
        neutral_path = source_root / "neutral_translation" / f"{sample_id}.json"
        neutral_artifact = read_json(neutral_path)
        english = english_paragraphs(english_artifact)
        neutral = paragraphs(neutral_artifact)
        if [row["id"] for row in english] != [row["id"] for row in neutral]:
            raise RuntimeError(f"English/neutral IDs differ: {sample_id}")

        candidate_records: list[dict[str, Any]] = [
            {
                "method_id": "neutral_only",
                "path": neutral_path,
                "artifact": neutral_artifact,
                "paragraphs": neutral,
                "hard_gate": {"status": "pass", "failures": []},
            }
        ]
        for method_id in CANDIDATE_METHODS:
            path = run_root / f"method_outputs/{method_id}/{INTENSITY}/{sample_id}.json"
            artifact = read_json(path)
            if (
                artifact.get("run_id") != args.run_id
                or artifact.get("method_id") != method_id
                or artifact.get("intensity") != INTENSITY
            ):
                raise RuntimeError(
                    f"Candidate provenance mismatch: {method_id}/{sample_id}"
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
            candidate_records.append(
                {
                    "method_id": method_id,
                    "path": path,
                    "artifact": artifact,
                    "paragraphs": output,
                    "hard_gate": hard_gate(neutral, output, request),
                }
            )

        eligible = [
            row for row in candidate_records if row["hard_gate"]["status"] == "pass"
        ]
        rng = random.Random(
            int(sha256_text(f"{sample_id}:{rule_sha}")[:16], 16)
        )
        rng.shuffle(eligible)
        for index, row in enumerate(eligible, start=1):
            row["candidate_id"] = f"candidate_{index:02d}"
        request = build_blind_request(
            sample_id=sample_id,
            english=english,
            style_definition=style_definition,
            eligible=eligible,
        )
        selection_run = invoke_selector(
            generation=generation,
            prompt_template=prompt_template,
            schema_path=schema_path,
            request=request,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            timeout_seconds=args.timeout_seconds,
            attempts=args.attempts,
        )
        if selection_run["status"] != "success":
            return {
                "sample_id": sample_id,
                "status": "failed",
                "failures": selection_run["failures"],
            }
        selector_result = selection_run["result"]
        selected = next(
            row
            for row in eligible
            if row["candidate_id"] == selector_result["selected_candidate_id"]
        )
        selected_method = str(selected["method_id"])
        selected_path = selected["path"]
        decision = {
            "schema_version": 1,
            "created_at": utc_now(),
            "sample_id": sample_id,
            "sample_set": args.sample_set,
            "run_id": args.run_id,
            "source_run_id": args.input_run_id,
            "analysis_lock": lock,
            "selection_path": str(selection_path.relative_to(REPO_ROOT)),
            "selection_sha256": file_sha256(selection_path),
            "selection_id": selection_id,
            "execution_selection": {
                key: value
                for key, value in execution_selection.items()
                if not key.startswith("_")
            },
            "execution_admission": execution_admission,
            "selection_rule_sha256": rule_sha,
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "model_request_sha256": sha256_text(canonical_json(request)),
            "model_response": selector_result,
            "response_id": selection_run.get("response_id"),
            "response_usage": selection_run.get("response_usage"),
            "candidate_mapping": [
                {
                    "candidate_id": row.get("candidate_id"),
                    "method_id": row["method_id"],
                    "output_path": str(row["path"].relative_to(REPO_ROOT)),
                    "output_file_sha256": file_sha256(row["path"]),
                    "hard_gate": row["hard_gate"],
                    "eligible": row in eligible,
                }
                for row in candidate_records
            ],
            "selected_method_id": selected_method,
            "selected_output_file_sha256": file_sha256(selected_path),
            "frozen_style_meter_visible_to_selector": False,
            "method_labels_visible_to_selector": False,
            "original_target_visible_to_selector": False,
        }
        decision_sha = sha256_text(canonical_json(decision))
        decision_path = decision_root / f"{sample_id}.json"
        write_json(decision_path, {**decision, "decision_sha256": decision_sha})
        result = {
            "sample_id": sample_id,
            "method_id": DERIVED_METHOD,
            "intensity": INTENSITY,
            "content_plan": [],
            "paragraphs": selected["paragraphs"],
            "style_cues_applied": [f"blind_selector:{selected['candidate_id']}"],
            "style_cues_skipped": [],
            "uncertainties": [],
        }
        artifact = {
            "schema_version": 1,
            "run_id": args.run_id,
            "source_run_id": args.input_run_id,
            "stage": "derived_independent_candidate_selector",
            "sample_id": sample_id,
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "result": result,
            "output_sha256": sha256_text(canonical_json(result)),
            "derived_provenance": {
                "decision_path": str(decision_path.relative_to(REPO_ROOT)),
                "decision_file_sha256": file_sha256(decision_path),
                "decision_sha256": decision_sha,
                "selection_rule_sha256": rule_sha,
                "analysis_lock_content_sha256": lock["content_sha256"],
                "selected_method_id": selected_method,
                "selected_output_file_sha256": file_sha256(selected_path),
                "source_run_id": args.input_run_id,
                "selection_id": selection_id,
                "execution_admission": execution_admission,
            },
        }
        write_json(output_path, artifact)
        return {
            "sample_id": sample_id,
            "status": "success",
            "selected_method_id": selected_method,
        }

    outcomes: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = {executor.submit(build_one, sample_id): sample_id for sample_id in sample_ids}
        for future in as_completed(futures):
            sample_id = futures[future]
            try:
                outcomes.append(future.result())
            except Exception as exc:
                outcomes.append(
                    {"sample_id": sample_id, "status": "failed", "error": str(exc)}
                )
    counts = Counter(str(row["status"]) for row in outcomes)
    selected_counts = Counter(
        str(row["selected_method_id"])
        for row in outcomes
        if row.get("status") == "success"
    )
    summary = {
        "schema_version": 1,
        "status": "complete" if counts.get("failed", 0) == 0 else "incomplete",
        "sample_count": len(sample_ids),
        "outcome_counts": dict(sorted(counts.items())),
        "selected_counts": dict(sorted(selected_counts.items())),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "selection_rule_sha256": rule_sha,
        "selection_id": selection_id,
        "selection_path": str(selection_path.relative_to(REPO_ROOT)),
        "selection_sha256": file_sha256(selection_path),
        "execution_admission": execution_admission,
        "analysis_lock_content_sha256": lock["content_sha256"],
        "frozen_style_meter_visible": False,
        "method_labels_visible": False,
        "original_target_visible": False,
        "failures": [row for row in outcomes if row.get("status") == "failed"],
    }
    write_json(
        run_root
        / f"selector_decisions/{DERIVED_METHOD}/{INTENSITY}.{selection_id}.summary.json",
        summary,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
