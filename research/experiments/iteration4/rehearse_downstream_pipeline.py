#!/usr/bin/env python3
from __future__ import annotations

"""Exercise the Iteration-4 pipeline without producing research outcomes."""

import argparse
import contextlib
import importlib.util
import io
import json
import math
import re
import shutil
import sys
import tempfile
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator
from unittest.mock import patch


from experiments.shared.paths import RESEARCH_ROOT
from workflows.author_style_meter_contract import CURRENT_SCORER_ID


REPO_ROOT = RESEARCH_ROOT

SOURCE_ROOT = (
    REPO_ROOT
    / "generated/style_research/style_transfer_experiments/iterations/full_regeneration_v1"
)
SAMPLE_SET = "iteration4_proxy_v1"
SOURCE_RUN_ID = "iteration4_source_gpt54_official"
STYLE_RUN_ID = "iteration4_style_gpt55_v1"
GENERATED_METHODS = (
    "generic_full_regeneration",
    "aligned_pairs_full_regeneration",
    "style_definition_examples_full_regeneration",
    "aligned_pairs_style_definition_full_regeneration",
    "content_plan_combined_full_regeneration",
)
SCORER_ID = CURRENT_SCORER_ID
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
NUMBER_RE = re.compile(r"\d+(?:[.,:/-]\d+)*")
CHINESE_NUMERALS = set("\u96f6\u3007\u4e00\u4e8c\u4e24\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u5343\u4e07\u4ebf\u5146\u51e0\u5eff\u5345\u534c")


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def parse_request(prompt: str) -> dict[str, Any]:
    marker = "```json\n"
    if marker not in prompt:
        raise RuntimeError("Synthetic backend received a prompt without JSON")
    payload = prompt.rsplit(marker, 1)[1].split("\n```", 1)[0]
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise RuntimeError("Synthetic request is not an object")
    return value


def synthetic_english(chinese: str) -> str:
    cjk = CJK_RE.findall(chinese)
    numbers = NUMBER_RE.findall(chinese)
    if cjk and all(character in CHINESE_NUMERALS for character in cjk):
        return " ".join(numbers or ["1"])
    word_count = max(1, math.ceil(len(cjk) * 0.25))
    words = ["event"] * word_count
    return " ".join([*words, *numbers]).strip() or "scene break"


def synthetic_result(stage: str, request: dict[str, Any]) -> dict[str, Any]:
    sample_id = str(request["sample_id"])
    if stage == "english_semantic_source":
        return {
            "sample_id": sample_id,
            "prompt_version": "english_semantic_source.v1",
            "paragraphs": [
                {"id": row["id"], "en": synthetic_english(str(row["zh"]))}
                for row in request["paragraphs"]
            ],
            "uncertainties": [],
        }
    if stage in {"english_source_qa", "english_source_repair_qa"}:
        return {
            "sample_id": sample_id,
            "prompt_version": "english_source_qa.v1",
            "approved": True,
            "paragraph_reviews": [
                {"id": row["id"], "status": "pass", "issues": []}
                for row in request["paragraphs"]
            ],
            "overall_issues": [],
        }
    if stage == "english_source_repair":
        return {
            "sample_id": sample_id,
            "prompt_version": "english_source_repair.v1",
            "paragraphs": [
                {"id": row["id"], "en": str(row["en"])}
                for row in request["paragraphs"]
            ],
            "uncertainties": [],
        }
    if stage == "neutral_translation":
        neutral = "\u8fd9\u662f\u4e2d\u6027\u6d4b\u8bd5\u6587\u672c\uff0c\u4fdd\u6301\u539f\u6709\u5185\u5bb9\u987a\u5e8f\u3002"
        return {
            "sample_id": sample_id,
            "prompt_version": "neutral_translation.v1",
            "paragraphs": [
                {"id": row["id"], "zh": neutral} for row in request["paragraphs"]
            ],
            "uncertainties": [],
        }
    if stage == "style_transfer":
        return {
            "sample_id": sample_id,
            "method_id": request["method_id"],
            "intensity": request["intensity"],
            "content_plan": [],
            "paragraphs": [dict(row) for row in request["neutral_zh"]],
            "style_cues_applied": [],
            "style_cues_skipped": [],
            "uncertainties": [],
        }
    raise RuntimeError(f"Unsupported synthetic stage: {stage}")


def fake_completed_process(
    stage: str, command: list[str], prompt: str
) -> SimpleNamespace:
    request = parse_request(prompt)
    result = synthetic_result(stage, request)
    output_flag = command.index("--output-last-message")
    Path(command[output_flag + 1]).write_text(
        json.dumps(result, ensure_ascii=False), encoding="utf-8"
    )
    stdout = "\n".join(
        (
            json.dumps(
                {"type": "thread.started", "thread_id": f"rehearsal-{stage}"}
            ),
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            ),
        )
    )
    return SimpleNamespace(returncode=0, stdout=stdout, stderr="")


def stage_args(
    *,
    root: Path,
    stage: str,
    run_id: str,
    selection_file: Path,
    method_id: str | None = None,
    analysis_lock: Path | None = None,
    input_run_id: str | None = None,
    resume: bool = False,
) -> Namespace:
    return Namespace(
        experiment_root=root,
        sample_set=SAMPLE_SET,
        stage=stage,
        sample_id=[],
        selection_file=selection_file,
        limit=None,
        method_id=method_id,
        intensity="strong" if method_id else None,
        base_method_id=None,
        base_intensity=None,
        repair_round=1,
        analysis_lock=analysis_lock,
        input_run_id=input_run_id,
        provisional_shortlist=None,
        refinement_contract=None,
        promotion_file=None,
        final_lock=None,
        run_id=run_id,
        jobs=1,
        resume=resume,
    )


def run_synthetic_stage(
    generation: Any, args: Namespace, stage: str
) -> dict[str, Any]:
    def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        return fake_completed_process(stage, command, str(kwargs.get("input", "")))

    isolation = {
        "profile_version": "rehearsal",
        "profile_sha256": "0" * 64,
        "allowed_request_area": True,
        "denied_probes": [
            {"path_class": label, "denied": True, "returncode": 1}
            for label in ("hidden_targets", "evaluator_allocation", "corpus")
        ],
    }
    with (
        patch.object(generation, "resolved_codex_binary", return_value=Path("/usr/bin/true")),
        patch.object(generation, "codex_version", return_value="codex-cli rehearsal-v1"),
        patch.object(generation, "verify_external_isolation", return_value=isolation),
        patch.object(generation.subprocess, "run", side_effect=fake_run),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        generation.run_stage(args)
    validation = generation.validate_stage(args)
    if validation["status"] != "passed":
        raise RuntimeError(
            f"Synthetic {stage} validation failed: {validation['errors'][:10]}"
        )
    return validation


@contextlib.contextmanager
def temporary_experiment() -> Iterator[Path]:
    parent = (
        REPO_ROOT
        / "generated/style_research/style_transfer_experiments/.rehearsals"
    )
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="iteration4-", dir=parent) as temp:
        root = Path(temp) / "experiment"
        shutil.copytree(
            SOURCE_ROOT,
            root,
            ignore=shutil.ignore_patterns("runs", "audits", "__pycache__"),
        )
        yield root


def invoke_main(module: Any, argv: list[str]) -> str:
    output = io.StringIO()
    with patch.object(sys, "argv", argv), contextlib.redirect_stdout(output):
        module.main()
    return output.getvalue()


def fake_selector_run(**kwargs: Any) -> dict[str, Any]:
    request = kwargs["request"]
    rankings = [
        {
            "candidate_id": row["candidate_id"],
            "rank": index,
            "semantic_fidelity": 5,
            "naturalness": 5,
            "style_adherence": 3,
            "hard_semantic_error": False,
            "reason": "deterministic rehearsal",
        }
        for index, row in enumerate(request["candidates"], start=1)
    ]
    return {
        "status": "success",
        "attempt": 1,
        "response_id": "rehearsal-selector",
        "response_usage": {"input_tokens": 1, "output_tokens": 1},
        "result": {
            "selected_candidate_id": rankings[0]["candidate_id"],
            "rankings": rankings,
            "selection_reason": "deterministic rehearsal",
            "uncertainties": [],
        },
        "prior_failures": [],
    }


def run_rehearsal(bootstrap_resamples: int) -> dict[str, Any]:
    generation = load_module(
        "iteration4_rehearsal_generation",
        REPO_ROOT / "experiments/iteration1/run_style_transfer_generation.py",
    )
    prepare = load_module(
        "iteration4_rehearsal_prepare", REPO_ROOT / "experiments/iteration4/prepare.py"
    )
    evaluator = load_module(
        "iteration4_rehearsal_evaluator",
        REPO_ROOT / "experiments/iteration4/evaluate_style_transfer_methods.py",
    )
    calibrator = load_module(
        "iteration4_rehearsal_calibrator",
        REPO_ROOT / "experiments/iteration1/calibrate_style_meter.py",
    )
    lock_module = load_module(
        "iteration4_rehearsal_lock", REPO_ROOT / "experiments/iteration1/style_analysis_lock.py"
    )
    selector = load_module(
        "iteration4_rehearsal_selector",
        REPO_ROOT / "experiments/iteration4/build_independent_candidate_selector.py",
    )
    payload = load_module(
        "iteration4_rehearsal_payload",
        REPO_ROOT / "experiments/iteration4/style_transfer_payloads.py",
    )

    with temporary_experiment() as root:
        sample_root = root / "sample_sets"
        source_selection = sample_root / f"{SAMPLE_SET}.source_construction_v1_ids.json"
        development_selection = sample_root / f"{SAMPLE_SET}.development_v1_ids.json"
        screening_selection = sample_root / f"{SAMPLE_SET}.screening_v1_ids.json"
        stage_validations: dict[str, int] = {}
        for stage in (
            "english_semantic_source",
            "english_source_qa",
            "neutral_translation",
        ):
            validation = run_synthetic_stage(
                generation,
                stage_args(
                    root=root,
                    stage=stage,
                    run_id=SOURCE_RUN_ID,
                    selection_file=source_selection,
                ),
                stage,
            )
            stage_validations[stage] = int(validation["valid_outputs"])

        prepare.ITERATION_ROOT = root
        model_config = prepare.write_model_config(
            "gpt-5.5", "iteration4_style_transfer_rehearsal"
        )
        cohort_summary = prepare.read_json(
            sample_root / f"{SAMPLE_SET}.summary.json"
        )
        method_lock = prepare.read_json(
            root / "method_assets/style_transfer_payloads.v1.lock.json"
        )
        method_bundle = prepare.read_json(root / str(method_lock["asset_path"]))
        prepare.write_protocols_and_registry(
            cohort_summary,
            {"lock": method_lock, "assets": method_bundle["assets"]},
            model_config,
        )

        calibration_scores = root / "calibration/style_meter_scores.v1.jsonl"
        evaluation_stdout = invoke_main(
            evaluator,
            [
                str(REPO_ROOT / "experiments/iteration4/evaluate_style_transfer_methods.py"),
                "score-calibration",
                "--experiment-root",
                str(root),
                "--sample-set",
                SAMPLE_SET,
                "--run-id",
                SOURCE_RUN_ID,
                "--scorer-dir",
                str(root / "scorers" / SCORER_ID),
                "--scorer-mode",
                "load",
                "--output",
                str(calibration_scores),
            ],
        )
        threshold_path = root / "calibration/style_meter_threshold.v1.json"
        invoke_main(
            calibrator,
            [
                str(REPO_ROOT / "experiments/iteration1/calibrate_style_meter.py"),
                "--scores",
                str(calibration_scores),
                "--experiment-root",
                str(root),
                "--sample-set",
                SAMPLE_SET,
                "--output",
                str(threshold_path),
            ],
        )

        lock_payload = lock_module.build_lock_payload(root)
        analysis_lock = root / "protocols/style_analysis.pre_style.lock.json"
        write_json(analysis_lock, lock_payload)
        lock_binding = lock_module.validate_analysis_lock(analysis_lock, root)

        generation.build_method_request = payload.build_method_request
        generation.load_frozen_assets = payload.load_frozen_assets
        generation.__file__ = str(
            REPO_ROOT / "experiments/iteration4/run_style_transfer_block_generation.py"
        )
        for method_id in GENERATED_METHODS:
            validation = run_synthetic_stage(
                generation,
                stage_args(
                    root=root,
                    stage="style_transfer",
                    run_id=STYLE_RUN_ID,
                    selection_file=development_selection,
                    method_id=method_id,
                    analysis_lock=analysis_lock,
                    input_run_id=SOURCE_RUN_ID,
                ),
                "style_transfer",
            )
            stage_validations[f"development:{method_id}"] = int(
                validation["valid_outputs"]
            )

        with (
            patch.object(
                sys,
                "argv",
                [
                    str(
                        REPO_ROOT
                        / "experiments/iteration4/build_independent_candidate_selector.py"
                    ),
                    "--experiment-root",
                    str(root),
                    "--sample-set",
                    SAMPLE_SET,
                    "--run-id",
                    STYLE_RUN_ID,
                    "--input-run-id",
                    SOURCE_RUN_ID,
                    "--selection-file",
                    str(development_selection),
                    "--analysis-lock",
                    str(analysis_lock),
                    "--jobs",
                    "1",
                ],
            ),
            patch.object(selector, "invoke_selector", side_effect=fake_selector_run),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            selector.main()

        evaluation_dir = root / "evaluations/development_rehearsal"
        invoke_main(
            evaluator,
            [
                str(REPO_ROOT / "experiments/iteration4/evaluate_style_transfer_methods.py"),
                "evaluate",
                "--experiment-root",
                str(root),
                "--sample-set",
                SAMPLE_SET,
                "--run-id",
                STYLE_RUN_ID,
                "--input-run-id",
                SOURCE_RUN_ID,
                "--analysis-lock",
                str(analysis_lock),
                "--selection-file",
                str(development_selection),
                "--threshold",
                str(threshold_path),
                "--scorer-dir",
                str(root / "scorers" / SCORER_ID),
                "--scorer-mode",
                "load",
                "--output-dir",
                str(evaluation_dir),
                "--bootstrap-resamples",
                str(bootstrap_resamples),
            ],
        )
        summary_paths = sorted(evaluation_dir.glob("evaluation_summary.json"))
        if not summary_paths:
            raise RuntimeError(
                "Rehearsal evaluation did not produce a summary: "
                + evaluation_stdout[-2000:]
            )
        development_selector_summary = json.loads(
            (
                root
                / f"runs/{SAMPLE_SET}/{STYLE_RUN_ID}/selector_decisions/"
                "independent_candidate_selector/strong.development_v1.summary.json"
            ).read_text(encoding="utf-8")
        )

        for method_id in GENERATED_METHODS:
            validation = run_synthetic_stage(
                generation,
                stage_args(
                    root=root,
                    stage="style_transfer",
                    run_id=STYLE_RUN_ID,
                    selection_file=screening_selection,
                    method_id=method_id,
                    analysis_lock=analysis_lock,
                    input_run_id=SOURCE_RUN_ID,
                    resume=True,
                ),
                "style_transfer",
            )
            stage_validations[f"screening:{method_id}"] = int(
                validation["valid_outputs"]
            )

        with (
            patch.object(
                sys,
                "argv",
                [
                    str(
                        REPO_ROOT
                        / "experiments/iteration4/build_independent_candidate_selector.py"
                    ),
                    "--experiment-root",
                    str(root),
                    "--sample-set",
                    SAMPLE_SET,
                    "--run-id",
                    STYLE_RUN_ID,
                    "--input-run-id",
                    SOURCE_RUN_ID,
                    "--selection-file",
                    str(screening_selection),
                    "--analysis-lock",
                    str(analysis_lock),
                    "--jobs",
                    "1",
                ],
            ),
            patch.object(selector, "invoke_selector", side_effect=fake_selector_run),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            selector.main()
        screening_selector_summary = json.loads(
            (
                root
                / f"runs/{SAMPLE_SET}/{STYLE_RUN_ID}/selector_decisions/"
                "independent_candidate_selector/strong.screening_v1.summary.json"
            ).read_text(encoding="utf-8")
        )

        screening_evaluation_dir = root / "evaluations/screening_rehearsal"
        screening_stdout = invoke_main(
            evaluator,
            [
                str(REPO_ROOT / "experiments/iteration4/evaluate_style_transfer_methods.py"),
                "evaluate",
                "--experiment-root",
                str(root),
                "--sample-set",
                SAMPLE_SET,
                "--run-id",
                STYLE_RUN_ID,
                "--input-run-id",
                SOURCE_RUN_ID,
                "--analysis-lock",
                str(analysis_lock),
                "--selection-file",
                str(screening_selection),
                "--threshold",
                str(threshold_path),
                "--scorer-dir",
                str(root / "scorers" / SCORER_ID),
                "--scorer-mode",
                "load",
                "--screening-phase",
                "initial",
                "--output-dir",
                str(screening_evaluation_dir),
                "--bootstrap-resamples",
                str(bootstrap_resamples),
            ],
        )
        screening_summary_paths = sorted(
            screening_evaluation_dir.glob("evaluation_summary.json")
        )
        if not screening_summary_paths:
            raise RuntimeError(
                "Screening rehearsal did not produce a summary: "
                + screening_stdout[-2000:]
            )
        return {
            "schema_version": 1,
            "status": "passed",
            "evidence_class": "synthetic_nonresearch_rehearsal",
            "real_iteration4_outputs_written": False,
            "temporary_workspace_deleted_on_exit": True,
            "stage_valid_outputs": stage_validations,
            "calibration_rows": sum(1 for _ in calibration_scores.open(encoding="utf-8")),
            "analysis_lock_status": lock_binding["status"],
            "analysis_lock_source_count": len(lock_payload["source_bindings"]),
            "analysis_lock_artifact_count": len(lock_payload["artifact_bindings"]),
            "development_selector_status": development_selector_summary["status"],
            "development_selector_sample_count": development_selector_summary[
                "sample_count"
            ],
            "development_evaluation_summary_count": len(summary_paths),
            "screening_selector_status": screening_selector_summary["status"],
            "screening_selector_sample_count": screening_selector_summary[
                "sample_count"
            ],
            "screening_evaluation_summary_count": len(screening_summary_paths),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a disposable, deterministic, nonresearch Iteration-4 downstream "
            "pipeline rehearsal."
        )
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_resamples < 20:
        raise SystemExit("--bootstrap-resamples must be at least 20")
    result = run_rehearsal(args.bootstrap_resamples)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
