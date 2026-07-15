from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from experiments.iteration1.style_analysis_lock import (
    require_matching_analysis_binding,
    validate_analysis_lock,
)
from experiments.iteration1.style_transfer_payloads import build_method_request, load_frozen_assets
from experiments.iteration1.style_experiment_provenance import (
    validate_execution_binding as validate_shared_execution_binding,
)
from experiments.iteration1.style_experiment_decisions import (
    recompute_confirmation_winner,
    recompute_promotion,
)


from experiments.shared.paths import RESEARCH_ROOT


REPO_ROOT = RESEARCH_ROOT
DEFAULT_EXPERIMENT_ROOT = (
    REPO_ROOT / "generated/style_research/style_transfer_experiments"
)
DEFAULT_SAMPLE_SET = "development_proxy_v1"
SUPPORTED_STAGES = (
    "english_semantic_source",
    "english_source_qa",
    "english_source_repair",
    "english_source_repair_qa",
    "neutral_translation",
    "style_transfer",
    "style_critique",
)
REPAIR_STAGES = {"english_source_repair", "english_source_repair_qa"}
CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
ENGLISH_LETTER_RE = re.compile(r"[A-Za-z]")
ARABIC_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
CHINESE_NUMERAL_CHARS = set("零〇一二三四五六七八九十百千万亿两")
EXTERNAL_SANDBOX_VERSION = "macos_seatbelt_deny_repo.v1"


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_source_snapshot(source: Path, run_root: Path) -> Path:
    source_hash = file_sha256(source)
    target = run_root / "code_snapshots" / f"{source.stem}.{source_hash}.py"
    if target.exists():
        if file_sha256(target) != source_hash:
            raise ValueError(f"Frozen code snapshot is corrupt: {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    return target


def freeze_input_snapshot(source: Path, target_dir: Path) -> Path:
    source_hash = file_sha256(source)
    target = target_dir / f"{source.stem}.{source_hash}{source.suffix}"
    if target.exists():
        if file_sha256(target) != source_hash:
            raise ValueError(f"Frozen input snapshot is corrupt: {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    return target


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run stateless, isolated generation jobs for the author-style "
            "benchmark."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT
        )
        command.add_argument("--sample-set", default=DEFAULT_SAMPLE_SET)
        command.add_argument("--stage", choices=SUPPORTED_STAGES, required=True)
        command.add_argument(
            "--sample-id",
            action="append",
            default=[],
            help="Restrict to an opaque sample ID; repeat for multiple samples.",
        )
        command.add_argument(
            "--selection-file",
            type=Path,
            help=(
                "JSON file containing a sample_ids array. This slices execution "
                "without changing the frozen intervention."
            ),
        )
        command.add_argument(
            "--limit", type=int, help="Deterministic smoke-test limit after ID sorting."
        )
        command.add_argument(
            "--method-id",
            help="Required for the style_transfer stage.",
        )
        command.add_argument(
            "--intensity",
            choices=("none", "light", "medium", "strong"),
            help="Required for the style_transfer stage.",
        )
        command.add_argument(
            "--base-method-id",
            help=(
                "Candidate method to critique, or the prior method repaired by "
                "self_critique_repair."
            ),
        )
        command.add_argument(
            "--base-intensity",
            choices=("light", "medium", "strong"),
            help="Intensity of --base-method-id.",
        )
        command.add_argument(
            "--repair-round",
            type=int,
            default=1,
            help="One-based immutable repair round for English repair stages.",
        )
        command.add_argument(
            "--analysis-lock",
            type=Path,
            help="Content-addressed pre-style analysis lock required for style stages.",
        )
        command.add_argument(
            "--input-run-id",
            help=(
                "Run containing frozen English and neutral prerequisites. "
                "Defaults to --run-id and is only valid for style stages."
            ),
        )
        command.add_argument(
            "--provisional-shortlist",
            type=Path,
            help="Frozen screening shortlist required for blind screening critiques.",
        )
        command.add_argument(
            "--refinement-contract",
            type=Path,
            help="Frozen exact post-screening intensity/repair arm contract.",
        )
        command.add_argument(
            "--promotion-file",
            type=Path,
            help="Frozen judged promotion required for confirmation execution.",
        )
        command.add_argument(
            "--final-lock",
            type=Path,
            help="One-time final-validation lock required for reserved-test execution.",
        )

    plan = subparsers.add_parser(
        "plan", help="Validate frozen inputs and print the pending run plan."
    )
    add_common(plan)
    plan.add_argument(
        "--run-id",
        help="Required for neutral planning so it can locate English outputs.",
    )

    run = subparsers.add_parser(
        "run", help="Execute one independent ephemeral Codex process per sample."
    )
    add_common(run)
    run.add_argument("--run-id", required=True)
    run.add_argument("--jobs", type=int)
    run.add_argument("--resume", action="store_true")

    validate = subparsers.add_parser(
        "validate", help="Validate frozen outputs and their append-only ledger."
    )
    add_common(validate)
    validate.add_argument("--run-id", required=True)
    return parser.parse_args()


def sample_paths(experiment_root: Path, sample_set: str) -> dict[str, Path]:
    sample_dir = experiment_root / "sample_sets"
    return {
        "runner_manifest": sample_dir / f"{sample_set}.runner_manifest.jsonl",
        "hidden_targets": sample_dir / f"{sample_set}.hidden_targets.jsonl",
        "method_evaluation_ids": sample_dir
        / f"{sample_set}.method_evaluation_ids.json",
        "summary": sample_dir / f"{sample_set}.summary.json",
    }


def stage_paths(
    experiment_root: Path,
    sample_set: str,
    run_id: str,
    stage: str,
    method_id: str | None = None,
    intensity: str | None = None,
    repair_round: int = 1,
) -> dict[str, Path]:
    run_root = experiment_root / "runs" / sample_set / run_id
    if stage in REPAIR_STAGES:
        if repair_round <= 0:
            raise ValueError("--repair-round must be a positive integer")
        round_name = f"round_{repair_round:02d}"
        stem = f"{stage}.{round_name}"
        return {
            "run_root": run_root,
            "outputs": run_root / stage / round_name,
            "ledger": run_root / "ledgers" / f"{stem}.jsonl",
            "run_config": run_root / "run_configs" / f"{stem}.json",
            "stage_stem": stem,
        }
    if stage == "style_transfer":
        if not method_id or not intensity:
            raise ValueError("style_transfer requires method_id and intensity")
        if not re.fullmatch(r"[a-z0-9_]+", method_id):
            raise ValueError(f"Unsafe method ID: {method_id!r}")
        stem = f"style_transfer.{method_id}.{intensity}"
        return {
            "run_root": run_root,
            "outputs": run_root / "method_outputs" / method_id / intensity,
            "ledger": run_root / "ledgers" / f"{stem}.jsonl",
            "run_config": run_root / "run_configs" / f"{stem}.json",
            "stage_stem": stem,
        }
    if stage == "style_critique":
        if not method_id or not intensity:
            raise ValueError("style_critique requires a base method and intensity")
        if not re.fullmatch(r"[a-z0-9_]+", method_id):
            raise ValueError(f"Unsafe base method ID: {method_id!r}")
        stem = f"style_critique.{method_id}.{intensity}"
        return {
            "run_root": run_root,
            "outputs": run_root / "style_critiques" / method_id / intensity,
            "ledger": run_root / "ledgers" / f"{stem}.jsonl",
            "run_config": run_root / "run_configs" / f"{stem}.json",
            "stage_stem": stem,
        }
    return {
        "run_root": run_root,
        "outputs": run_root / stage,
        "ledger": run_root / "ledgers" / f"{stage}.jsonl",
        "run_config": run_root / "run_configs" / f"{stage}.json",
        "stage_stem": stage,
    }


def load_model_config(experiment_root: Path) -> tuple[Path, dict[str, Any]]:
    path = experiment_root / "protocols/model_run_config.v1.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def validate_method_args(
    experiment_root: Path,
    stage: str,
    method_id: str | None,
    intensity: str | None,
) -> tuple[Path | None, dict[str, Any] | None]:
    if stage != "style_transfer":
        if method_id or intensity:
            raise ValueError("--method-id/--intensity apply only to style_transfer")
        return None, None
    if not method_id or not intensity:
        raise ValueError("style_transfer requires --method-id and --intensity")
    config_path = experiment_root / "method_registry" / "methods" / f"{method_id}.v1.json"
    if not config_path.exists():
        raise ValueError(f"Unknown style-transfer method: {method_id}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if intensity not in config.get("intensities", []):
        raise ValueError(
            f"Intensity {intensity!r} is not registered for {method_id}: "
            f"{config.get('intensities', [])}"
        )
    if method_id == "neutral_only":
        raise ValueError(
            "neutral_only is a deterministic evaluation control and is scored "
            "directly from neutral_translation outputs"
        )
    load_frozen_assets(experiment_root)
    return config_path, config


def validate_base_args(
    experiment_root: Path,
    stage: str,
    method_id: str | None,
    base_method_id: str | None,
    base_intensity: str | None,
) -> None:
    requires_base = stage == "style_critique" or (
        stage == "style_transfer" and method_id == "self_critique_repair"
    )
    if requires_base and (not base_method_id or not base_intensity):
        raise ValueError(
            f"{stage} requires --base-method-id and --base-intensity"
        )
    if not requires_base and (base_method_id or base_intensity):
        raise ValueError(
            "--base-method-id/--base-intensity apply only to style_critique "
            "or self_critique_repair"
        )
    if not requires_base:
        return
    base_config = (
        experiment_root
        / "method_registry"
        / "methods"
        / f"{base_method_id}.v1.json"
    )
    invalid_base = base_method_id == "neutral_only" or (
        stage == "style_transfer" and base_method_id == "self_critique_repair"
    )
    if not base_config.exists() or invalid_base:
        raise ValueError(f"Invalid base style method: {base_method_id}")
    config = json.loads(base_config.read_text(encoding="utf-8"))
    if base_intensity not in config.get("intensities", []):
        raise ValueError(
            f"Base intensity {base_intensity!r} is not registered for "
            f"{base_method_id}"
        )


def filter_sample_ids(
    rows: Sequence[dict[str, Any]], requested: Sequence[str], limit: int | None
) -> list[dict[str, Any]]:
    selected = sorted(rows, key=lambda row: row["sample_id"])
    if requested:
        requested_set = set(requested)
        known = {row["sample_id"] for row in selected}
        missing = sorted(requested_set - known)
        if missing:
            raise ValueError(f"Unknown sample IDs: {missing}")
        selected = [row for row in selected if row["sample_id"] in requested_set]
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive.")
        selected = selected[:limit]
    return selected


def requested_sample_ids(args: argparse.Namespace) -> list[str]:
    requested = list(args.sample_id)
    if args.selection_file:
        payload = json.loads(args.selection_file.read_text(encoding="utf-8"))
        file_ids = payload if isinstance(payload, list) else payload.get("sample_ids")
        if not isinstance(file_ids, list) or not all(
            isinstance(sample_id, str) for sample_id in file_ids
        ):
            raise ValueError("--selection-file must contain a sample_ids string array")
        requested.extend(file_ids)
    if len(requested) != len(set(requested)):
        raise ValueError("Requested sample IDs contain duplicates")
    return requested


def validate_stage_options(args: argparse.Namespace) -> None:
    if args.repair_round <= 0:
        raise ValueError("--repair-round must be a positive integer")
    if args.stage not in REPAIR_STAGES and args.repair_round != 1:
        raise ValueError("--repair-round applies only to English repair stages")
    if args.selection_file and (args.sample_id or args.limit is not None):
        raise ValueError(
            "--selection-file cannot be combined with --sample-id or --limit"
        )
    if args.stage not in {"style_transfer", "style_critique"} and any(
        (
            args.analysis_lock,
            args.provisional_shortlist,
            args.refinement_contract,
            args.promotion_file,
            args.final_lock,
        )
    ):
        raise ValueError(
            "Admission artifacts apply only to style_transfer/style_critique"
        )
    if args.stage in {"style_transfer", "style_critique"} and not args.analysis_lock:
        raise ValueError("Style generation requires --analysis-lock")
    if args.input_run_id and args.stage not in {"style_transfer", "style_critique"}:
        raise ValueError("--input-run-id applies only to style stages")
    if args.stage in {"style_transfer", "style_critique"}:
        validate_analysis_lock(args.analysis_lock.resolve(), args.experiment_root)


def execution_selection(
    args: argparse.Namespace, active_sample_ids: Sequence[str]
) -> dict[str, Any]:
    active_ids = sorted(set(active_sample_ids))
    if len(active_ids) != len(active_sample_ids):
        raise ValueError("Active execution samples contain duplicates")
    if args.selection_file:
        path = args.selection_file.resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Official --selection-file must contain a JSON object")
        sample_ids = payload.get("sample_ids")
        if not isinstance(sample_ids, list) or not all(
            isinstance(sample_id, str) for sample_id in sample_ids
        ):
            raise ValueError("Official selection has no valid sample_ids array")
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("Official selection contains duplicate sample IDs")
        if payload.get("sample_set") != args.sample_set:
            raise ValueError("Official selection sample_set does not match --sample-set")
        if int(payload.get("sample_count", -1)) != len(sample_ids):
            raise ValueError("Official selection sample_count is incorrect")
        selection_id = payload.get("selection_id")
        if not isinstance(selection_id, str) or not re.fullmatch(
            r"[a-z0-9_]+", selection_id
        ):
            raise ValueError("Official selection_id is missing or unsafe")
        if not set(active_ids).issubset(set(sample_ids)):
            raise ValueError("Active samples are not a subset of the frozen selection")
        return {
            "execution_selection_id": selection_id,
            "execution_selection_sha256": file_sha256(path),
            "execution_selection_sample_count": len(sample_ids),
            "execution_selection_path": str(path.relative_to(REPO_ROOT)),
            "execution_selection_is_frozen_file": True,
            "active_sample_count": len(active_ids),
            "active_sample_ids_sha256": sha256_text(canonical_json(active_ids)),
            "_active_sample_ids": active_ids,
        }

    synthetic = {
        "sample_set": args.sample_set,
        "stage": args.stage,
        "sample_ids": active_ids,
        "limit": args.limit,
        "source": "all_active_rows" if not args.sample_id else "explicit_sample_ids",
    }
    digest = sha256_text(canonical_json(synthetic))
    return {
        "execution_selection_id": f"descriptive_{digest[:16]}",
        "execution_selection_sha256": digest,
        "execution_selection_sample_count": len(active_ids),
        "execution_selection_path": None,
        "execution_selection_is_frozen_file": False,
        "active_sample_count": len(active_ids),
        "active_sample_ids_sha256": sha256_text(canonical_json(active_ids)),
        "_active_sample_ids": active_ids,
    }


def execution_batch_path(
    paths: dict[str, Path], selection: dict[str, Any]
) -> Path:
    selection_id = selection["execution_selection_id"]
    selection_hash = selection["execution_selection_sha256"]
    return (
        paths["run_root"]
        / "run_configs"
        / "execution_batches"
        / f"{paths['stage_stem']}.{selection_id}.{selection_hash[:16]}.json"
    )


def freeze_execution_batch(
    paths: dict[str, Path],
    selection: dict[str, Any],
    admission: dict[str, Any] | None = None,
) -> dict[str, Any]:
    public_selection = {
        key: value for key, value in selection.items() if not key.startswith("_")
    }
    record = {
        "schema_version": 1,
        "stage_run_config_path": str(paths["run_config"].relative_to(REPO_ROOT)),
        "stage_run_config_sha256": file_sha256(paths["run_config"]),
        **public_selection,
        "active_sample_ids": selection["_active_sample_ids"],
        "admission": admission,
    }
    path = execution_batch_path(paths, selection)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != record:
            raise ValueError(f"Execution batch is frozen with different values: {path}")
    else:
        write_json(path, record)
    return {
        **public_selection,
        "execution_batch_config_path": str(path.relative_to(REPO_ROOT)),
        "execution_batch_config_sha256": file_sha256(path),
        "execution_admission": admission,
    }


def load_admission_artifact(path: Path, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = path.resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return resolved, value


def validate_registered_execution_selection(
    args: argparse.Namespace, selection: Mapping[str, Any]
) -> tuple[Path, dict[str, Any], str] | None:
    selection_id = selection["execution_selection_id"]
    registered_stage = {
        "development_v1": "development",
        "screening_v1": "screening",
        "confirmation_v1": "confirmation",
    }.get(selection_id)
    if registered_stage is None:
        return None
    protocol_path = (
        args.experiment_root / "protocols/evaluation_protocol.v1.json"
    ).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    stage_contract = protocol.get("staged_execution", {}).get(
        registered_stage, {}
    )
    expected_relative = stage_contract.get("selection_file")
    expected_count = int(stage_contract.get("sample_count", -1))
    if not isinstance(expected_relative, str):
        raise ValueError(
            f"Evaluation protocol lacks the {registered_stage} selection file"
        )
    expected_path = (args.experiment_root / expected_relative).resolve()
    recorded_value = selection.get("execution_selection_path")
    recorded_selection = (
        recorded_path(str(recorded_value)).resolve()
        if isinstance(recorded_value, str)
        else None
    )
    if recorded_selection != expected_path:
        raise ValueError(
            f"{registered_stage.capitalize()} execution does not use the "
            "registered selection file"
        )
    if selection.get("execution_selection_sha256") != file_sha256(expected_path):
        raise ValueError(f"{registered_stage.capitalize()} selection hash mismatch")
    if (
        selection.get("execution_selection_sample_count") != expected_count
        or selection.get("active_sample_count") != expected_count
    ):
        raise ValueError(
            f"{registered_stage.capitalize()} execution must contain the full "
            f"registered {expected_count}-sample cohort"
        )
    return protocol_path, protocol, registered_stage


def validate_execution_admission(
    args: argparse.Namespace, selection: dict[str, Any]
) -> dict[str, Any] | None:
    if args.stage not in {"style_transfer", "style_critique"}:
        return None
    if not selection["execution_selection_is_frozen_file"]:
        raise ValueError(
            "Style generation requires an official frozen --selection-file"
        )
    analysis_lock_binding = validate_analysis_lock(
        args.analysis_lock.resolve(), args.experiment_root
    )

    def bind_analysis_lock(record: dict[str, Any]) -> dict[str, Any]:
        return {**record, "analysis_lock": analysis_lock_binding}

    selection_id = selection["execution_selection_id"]
    registered = validate_registered_execution_selection(args, selection)
    if registered is not None:
        protocol_path, protocol, _ = registered
    method_id = (
        args.base_method_id if args.stage == "style_critique" else args.method_id
    )
    intensity = (
        args.base_intensity if args.stage == "style_critique" else args.intensity
    )
    if args.stage == "style_transfer" and args.method_id == "self_critique_repair":
        method_id = args.base_method_id
        intensity = args.base_intensity
    candidate = (str(method_id or ""), str(intensity or ""))

    if selection_id == "development_v1":
        if args.stage != "style_transfer" or args.method_id == "self_critique_repair":
            raise ValueError(
                "The frozen development pilot admits fixed style-transfer methods only"
            )
        if any(
            (
                args.provisional_shortlist,
                args.refinement_contract,
                args.promotion_file,
                args.final_lock,
            )
        ):
            raise ValueError(
                "Development-pilot generation does not accept outcome-dependent "
                "admission artifacts"
            )
        roster = (
            protocol.get("staged_execution", {})
            .get("development", {})
            .get("generation_method_intensities")
        )
        if not isinstance(roster, dict) or not roster:
            raise ValueError("Evaluation protocol has no development-pilot roster")
        if roster.get(args.method_id) != args.intensity:
            raise ValueError(
                "Method combination is outside the exact development-pilot roster: "
                f"{candidate}"
            )
        return bind_analysis_lock(
            {
                "admission_stage": "preregistered_development_pilot",
                "selection_id": selection_id,
                "development_roster_path": str(protocol_path.relative_to(REPO_ROOT)),
                "development_roster_sha256": file_sha256(protocol_path),
                "official_efficacy_evidence": False,
            }
        )

    if selection_id == "screening_v1":
        if args.refinement_contract:
            path, artifact = load_admission_artifact(
                args.refinement_contract, "screening refinement contract"
            )
            require_matching_analysis_binding(
                artifact, analysis_lock_binding, label="screening refinement contract"
            )
            if artifact.get("status") != "refinement_contract_frozen" or artifact.get(
                "selection_id"
            ) != "screening_v1":
                raise ValueError("Invalid frozen screening refinement contract")
            if artifact.get("selection_sha256") != selection[
                "execution_selection_sha256"
            ]:
                raise ValueError("Refinement contract selection binding mismatch")
            if artifact.get("method_registry_sha256") != file_sha256(
                args.experiment_root / "method_registry/style_methods.v1.json"
            ):
                raise ValueError("Refinement contract method registry is stale")
            shortlist_value = artifact.get("provisional_shortlist_path")
            if not isinstance(shortlist_value, str):
                raise ValueError("Refinement contract lacks its provisional shortlist")
            shortlist_path = recorded_path(shortlist_value)
            if not shortlist_path.exists() or file_sha256(
                shortlist_path
            ) != artifact.get("provisional_shortlist_sha256"):
                raise ValueError("Refinement shortlist binding mismatch")
            shortlist = json.loads(shortlist_path.read_text(encoding="utf-8"))
            require_matching_analysis_binding(
                shortlist, analysis_lock_binding, label="provisional shortlist"
            )
            shortlist_evaluation_value = shortlist.get("screening_evaluation_path")
            if not isinstance(shortlist_evaluation_value, str):
                raise ValueError("Provisional shortlist lacks its evaluation")
            shortlist_evaluation_path = recorded_path(shortlist_evaluation_value)
            if not shortlist_evaluation_path.exists() or file_sha256(
                shortlist_evaluation_path
            ) != shortlist.get("screening_evaluation_sha256"):
                raise ValueError("Provisional shortlist evaluation binding mismatch")
            _, recomputed_shortlist = recompute_promotion(
                json.loads(shortlist_evaluation_path.read_text(encoding="utf-8")),
                require_judgments=False,
            )
            if [
                (row["method_id"], row["intensity"])
                for row in recomputed_shortlist
            ] != [
                (str(row.get("method_id", "")), str(row.get("intensity", "")))
                for row in shortlist.get("promoted", [])
                if isinstance(row, dict)
            ]:
                raise ValueError("Provisional shortlist decision does not reproduce")
            actual_method_id = (
                args.base_method_id
                if args.stage == "style_critique"
                else args.method_id
            )
            actual_intensity = (
                args.base_intensity
                if args.stage == "style_critique"
                else args.intensity
            )
            matches = [
                row
                for row in artifact.get("evaluation_combinations", [])
                if isinstance(row, dict)
                and row.get("method_id") == actual_method_id
                and row.get("intensity") == actual_intensity
            ]
            if len(matches) != 1:
                raise ValueError(
                    "Method combination is absent or ambiguous in the frozen "
                    "refinement contract"
                )
            arm = matches[0]
            if args.stage == "style_transfer" and args.method_id == "self_critique_repair":
                if arm.get("base_method_id") != args.base_method_id or arm.get(
                    "base_intensity"
                ) != args.base_intensity:
                    raise ValueError("Derived repair base binding mismatch")
            return bind_analysis_lock({
                "admission_stage": "screening_refinement_contract",
                "selection_id": selection_id,
                "admission_path": str(path.relative_to(REPO_ROOT)),
                "admission_sha256": file_sha256(path),
                "admitted_method_id": str(actual_method_id),
                "admitted_intensity": str(actual_intensity),
                "derived_base_method_id": arm.get("base_method_id"),
                "derived_base_intensity": arm.get("base_intensity"),
            })
        if args.stage == "style_transfer" and args.method_id != "self_critique_repair":
            if any(
                (
                    args.provisional_shortlist,
                    args.refinement_contract,
                    args.promotion_file,
                    args.final_lock,
                )
            ):
                raise ValueError(
                    "Initial screening transfer does not accept admission artifacts"
                )
            protocol_path = (
                args.experiment_root / "protocols/evaluation_protocol.v1.json"
            ).resolve()
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            roster = (
                protocol.get("staged_execution", {})
                .get("screening", {})
                .get("initial_method_intensities")
            )
            if not isinstance(roster, dict) or not roster:
                raise ValueError("Evaluation protocol has no initial screening roster")
            if roster.get(args.method_id) != args.intensity:
                raise ValueError(
                    "Method combination is outside the exact initial screening "
                    f"roster: {(args.method_id, args.intensity)}"
                )
            return bind_analysis_lock({
                "admission_stage": "preregistered_screening",
                "selection_id": selection_id,
                "screening_roster_path": str(protocol_path.relative_to(REPO_ROOT)),
                "screening_roster_sha256": file_sha256(protocol_path),
            })
        if args.stage == "style_transfer" and args.method_id == "self_critique_repair":
            raise ValueError(
                "self_critique_repair requires --refinement-contract"
            )
        if not args.provisional_shortlist:
            raise ValueError(
                "Screening critiques and self-critique repairs require "
                "--provisional-shortlist"
            )
        path, artifact = load_admission_artifact(
            args.provisional_shortlist, "provisional shortlist"
        )
        require_matching_analysis_binding(
            artifact, analysis_lock_binding, label="provisional shortlist"
        )
        if artifact.get("selection_id") != "screening_v1" or artifact.get(
            "status"
        ) != "provisional_shortlist_frozen":
            raise ValueError("Invalid frozen screening shortlist")
        evaluation_value = artifact.get("screening_evaluation_path")
        if not isinstance(evaluation_value, str):
            raise ValueError("Provisional shortlist lacks its screening evaluation")
        evaluation_path = recorded_path(evaluation_value)
        if not evaluation_path.exists() or file_sha256(evaluation_path) != artifact.get(
            "screening_evaluation_sha256"
        ):
            raise ValueError("Provisional shortlist evaluation binding mismatch")
        _, recomputed = recompute_promotion(
            json.loads(evaluation_path.read_text(encoding="utf-8")),
            require_judgments=False,
        )
        if [
            (row["method_id"], row["intensity"]) for row in recomputed
        ] != [
            (str(row.get("method_id", "")), str(row.get("intensity", "")))
            for row in artifact.get("promoted", [])
            if isinstance(row, dict)
        ]:
            raise ValueError("Provisional shortlist decision does not reproduce")
        admitted_methods = {
            (str(row.get("method_id", "")), str(row.get("intensity", "")))
            for row in artifact.get("promoted", [])
        }
        if candidate not in admitted_methods:
            raise ValueError(f"Method combination is not shortlisted: {candidate}")
        return bind_analysis_lock({
            "admission_stage": "screening_provisional_shortlist",
            "selection_id": selection_id,
            "admission_path": str(path.relative_to(REPO_ROOT)),
            "admission_sha256": file_sha256(path),
            "admitted_method_id": candidate[0],
            "admitted_intensity": candidate[1],
        })

    if selection_id == "confirmation_v1":
        if not args.promotion_file:
            raise ValueError("Confirmation execution requires --promotion-file")
        path, artifact = load_admission_artifact(args.promotion_file, "promotion")
        require_matching_analysis_binding(
            artifact, analysis_lock_binding, label="judged promotion"
        )
        if artifact.get("selection_id") != "screening_v1" or artifact.get(
            "status"
        ) != "promotions_frozen":
            raise ValueError("Invalid judged promotion artifact")
        evaluation_value = artifact.get("screening_evaluation_path")
        if not isinstance(evaluation_value, str):
            raise ValueError("Promotion lacks its screening evaluation path")
        evaluation_path = recorded_path(evaluation_value)
        if not evaluation_path.exists() or file_sha256(evaluation_path) != artifact.get(
            "screening_evaluation_sha256"
        ):
            raise ValueError("Promotion screening evaluation binding mismatch")
        screening_evaluation = json.loads(
            evaluation_path.read_text(encoding="utf-8")
        )
        _, recomputed = recompute_promotion(
            screening_evaluation, require_judgments=True
        )
        recomputed_keys = [
            (row["method_id"], row["intensity"]) for row in recomputed
        ]
        recorded_keys = [
            (str(row.get("method_id", "")), str(row.get("intensity", "")))
            for row in artifact.get("promoted", [])
            if isinstance(row, dict)
        ]
        if recorded_keys != recomputed_keys:
            raise ValueError("Promotion decision does not reproduce from evaluation")
        current_bindings = {
            "method_registry_sha256": file_sha256(
                args.experiment_root / "method_registry/style_methods.v1.json"
            ),
            "threshold_sha256": file_sha256(
                args.experiment_root / "calibration/style_meter_threshold.v1.json"
            ),
            "screening_selection_sha256": file_sha256(
                args.experiment_root
                / "sample_sets"
                / f"{args.sample_set}.screening_v1_ids.json"
            ),
        }
        for field, expected in current_bindings.items():
            if artifact.get(field) != expected:
                raise ValueError(f"Promotion {field} binding mismatch")
        admitted_methods = {
            (str(row.get("method_id", "")), str(row.get("intensity", "")))
            for row in artifact.get("promoted", [])
        }
        if candidate not in admitted_methods:
            raise ValueError(f"Method combination was not promoted: {candidate}")
        return bind_analysis_lock({
            "admission_stage": "judged_screening_promotion",
            "selection_id": selection_id,
            "admission_path": str(path.relative_to(REPO_ROOT)),
            "admission_sha256": file_sha256(path),
            "admitted_method_id": candidate[0],
            "admitted_intensity": candidate[1],
        })

    if selection_id == "final_validation_v1":
        if not args.final_lock:
            raise ValueError("Reserved final execution requires --final-lock")
        path, artifact = load_admission_artifact(args.final_lock, "final lock")
        require_matching_analysis_binding(
            artifact, analysis_lock_binding, label="final-validation lock"
        )
        if artifact.get("selection_id") != "final_validation_v1" or artifact.get(
            "one_time_only"
        ) is not True:
            raise ValueError("Invalid one-time final-validation lock")
        winner = artifact.get("winner", {})
        locked = (
            str(winner.get("method_id", "")),
            str(winner.get("intensity", "")),
        )
        if candidate != locked:
            raise ValueError(f"Method combination is not the locked winner: {candidate}")
        confirmation_value = artifact.get("confirmation_evaluation_path")
        if not isinstance(confirmation_value, str):
            raise ValueError("Final lock lacks confirmation evaluation path")
        confirmation_path = recorded_path(confirmation_value)
        if not confirmation_path.exists() or file_sha256(
            confirmation_path
        ) != artifact.get("confirmation_evaluation_sha256"):
            raise ValueError("Final lock confirmation binding mismatch")
        recomputed_winner = recompute_confirmation_winner(
            json.loads(confirmation_path.read_text(encoding="utf-8"))
        )
        if recomputed_winner != winner:
            raise ValueError("Final winner does not reproduce from confirmation")
        pre_reveal_value = artifact.get("pre_reveal_lock_path")
        if not isinstance(pre_reveal_value, str):
            raise ValueError("Final lock lacks the pre-reveal lock")
        pre_reveal_path = recorded_path(pre_reveal_value)
        if not pre_reveal_path.exists() or file_sha256(
            pre_reveal_path
        ) != artifact.get("pre_reveal_lock_sha256"):
            raise ValueError("Final pre-reveal lock binding mismatch")
        pre_reveal = json.loads(pre_reveal_path.read_text(encoding="utf-8"))
        require_matching_analysis_binding(
            pre_reveal, analysis_lock_binding, label="final pre-reveal lock"
        )
        prompt_files = {
            "english_semantic_source": "english_semantic_source.v1.md",
            "english_source_qa": "english_source_qa.v1.md",
            "english_source_repair": "english_source_repair.v1.md",
            "neutral_translation": "neutral_translation.v1.md",
            "style_transfer": "style_transfer_method.v1.md",
            "style_critique": "style_transfer_critique.v1.md",
        }
        schema_files = {
            "english_semantic_source": "english_semantic_source_output.v1.schema.json",
            "english_source_qa": "english_source_qa_output.v1.schema.json",
            "english_source_repair": "english_source_repair_output.v1.schema.json",
            "neutral_translation": "neutral_translation_output.v1.schema.json",
            "style_transfer": "style_transfer_output.v1.schema.json",
            "style_critique": "style_transfer_critique_output.v1.schema.json",
        }
        expected_prompt_hashes = {
            key: file_sha256(args.experiment_root / "prompts" / filename)
            for key, filename in prompt_files.items()
        }
        expected_schema_hashes = {
            key: file_sha256(args.experiment_root / "schemas" / filename)
            for key, filename in schema_files.items()
        }
        if pre_reveal.get("prompt_sha256") != expected_prompt_hashes:
            raise ValueError("Final pre-reveal prompt bindings are stale")
        if pre_reveal.get("schema_sha256") != expected_schema_hashes:
            raise ValueError("Final pre-reveal schema bindings are stale")
        expected_pre_reveal = {
            "model_run_config_sha256": file_sha256(
                args.experiment_root / "protocols/model_run_config.v1.json"
            ),
            "evaluation_protocol_sha256": file_sha256(
                args.experiment_root / "protocols/evaluation_protocol.v1.json"
            ),
            "method_registry_sha256": file_sha256(
                args.experiment_root / "method_registry/style_methods.v1.json"
            ),
            "payload_lock_sha256": file_sha256(
                args.experiment_root
                / "method_assets/style_transfer_payloads.v1.lock.json"
            ),
            "runner_source_sha256": file_sha256(Path(__file__)),
            "evaluator_source_sha256": file_sha256(
                Path(__file__).with_name("evaluate_style_transfer_methods.py")
            ),
            "payload_builder_source_sha256": file_sha256(
                Path(__file__).with_name("style_transfer_payloads.py")
            ),
            "clean_chunks_sha256": file_sha256(
                REPO_ROOT / "datasets/unmasked/chunks.clean.jsonl"
            ),
            "masked_chunks_sha256": file_sha256(
                REPO_ROOT / "datasets/masked/chunks.entity_masked_v3.jsonl"
            ),
            "splits_sha256": file_sha256(
                REPO_ROOT / "generated/style_research/corpus/splits.json"
            ),
        }
        for field, expected in expected_pre_reveal.items():
            if pre_reveal.get(field) != expected or artifact.get(field) != expected:
                raise ValueError(f"Final lock {field} binding mismatch")
        if artifact.get("final_selection_sha256") != selection[
            "execution_selection_sha256"
        ]:
            raise ValueError("Final lock does not bind this frozen selection")
        return bind_analysis_lock({
            "admission_stage": "one_time_final_validation_lock",
            "selection_id": selection_id,
            "admission_path": str(path.relative_to(REPO_ROOT)),
            "admission_sha256": file_sha256(path),
            "admitted_method_id": candidate[0],
            "admitted_intensity": candidate[1],
        })

    raise ValueError(
        f"Style generation is not admitted for selection {selection_id!r}"
    )


def split_paragraphs(text: str, field: str) -> list[dict[str, str]]:
    paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
    if not paragraphs:
        raise ValueError("Input contains no non-empty paragraphs.")
    return [
        {"id": f"p{index:04d}", field: paragraph}
        for index, paragraph in enumerate(paragraphs, start=1)
    ]


def load_stage_inputs(
    experiment_root: Path,
    sample_set: str,
    stage: str,
    run_id: str,
    requested_ids: Sequence[str],
    limit: int | None,
    method_id: str | None = None,
    intensity: str | None = None,
    base_method_id: str | None = None,
    base_intensity: str | None = None,
    repair_round: int = 1,
    input_run_id: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    paths = sample_paths(experiment_root, sample_set)
    validate_method_args(experiment_root, stage, method_id, intensity)
    validate_base_args(
        experiment_root,
        stage,
        method_id,
        base_method_id,
        base_intensity,
    )
    restricted_stage = stage in {"style_transfer", "style_critique"}
    runner_rows = filter_sample_ids(
        list(iter_jsonl(paths["runner_manifest"])),
        requested_ids,
        None if restricted_stage else limit,
    )
    if restricted_stage:
        ids_path = paths["method_evaluation_ids"]
        if not ids_path.exists():
            raise ValueError(
                "Frozen method-evaluation ID set is missing; regenerate the "
                "experiment artifacts before style transfer."
            )
        method_set = json.loads(ids_path.read_text(encoding="utf-8"))
        eligible_ids = set(method_set.get("sample_ids", []))
        requested_ineligible = sorted(set(requested_ids) - eligible_ids)
        if requested_ineligible:
            raise ValueError(
                f"{stage} cannot use calibration or unregistered IDs: "
                f"{requested_ineligible}"
            )
        runner_rows = [
            row for row in runner_rows if row["sample_id"] in eligible_ids
        ]
        if len(runner_rows) != len(eligible_ids) and not requested_ids:
            raise ValueError("Method-evaluation ID set disagrees with runner manifest")
        if limit is not None:
            if limit <= 0:
                raise ValueError("--limit must be positive.")
            runner_rows = runner_rows[:limit]
    if stage == "english_semantic_source":
        hidden = {
            row["sample_id"]: row for row in iter_jsonl(paths["hidden_targets"])
        }
        inputs = {
            row["sample_id"]: {
                "sample_id": row["sample_id"],
                "paragraphs": split_paragraphs(
                    hidden[row["sample_id"]]["original_zh"], "zh"
                ),
                "terminology_placeholders": {},
            }
            for row in runner_rows
        }
        return runner_rows, inputs

    prerequisite_run_id = input_run_id or run_id
    english_paths = stage_paths(
        experiment_root, sample_set, prerequisite_run_id, "english_semantic_source"
    )
    hidden = (
        {
            row["sample_id"]: row
            for row in iter_jsonl(paths["hidden_targets"])
        }
        if stage
        in {
            "english_source_qa",
            "english_source_repair",
            "english_source_repair_qa",
        }
        else {}
    )
    qa_paths = stage_paths(
        experiment_root, sample_set, prerequisite_run_id, "english_source_qa"
    )
    repair_paths = stage_paths(
        experiment_root,
        sample_set,
        prerequisite_run_id,
        "english_source_repair",
        repair_round=repair_round,
    )
    repair_qa_paths = stage_paths(
        experiment_root,
        sample_set,
        prerequisite_run_id,
        "english_source_repair_qa",
        repair_round=repair_round,
    )
    neutral_paths = stage_paths(
        experiment_root, sample_set, prerequisite_run_id, "neutral_translation"
    )
    inputs: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    unapproved: list[str] = []
    deterministic_failures: list[str] = []
    active_rows: list[dict[str, Any]] = []
    for row in runner_rows:
        sample_id = row["sample_id"]
        english_path = english_paths["outputs"] / f"{sample_id}.json"
        if not english_path.exists():
            missing.append(sample_id)
            continue
        english_artifact = json.loads(english_path.read_text(encoding="utf-8"))
        english_result = english_artifact["result"]
        if stage == "english_source_qa":
            original = split_paragraphs(
                hidden[sample_id]["original_zh"], "zh"
            )
            source_request = {
                "sample_id": sample_id,
                "paragraphs": original,
            }
            deterministic = deterministic_english_qa(
                source_request, english_result
            )
            if not deterministic["approved"]:
                deterministic_failures.append(sample_id)
                continue
            english_by_id = {
                paragraph["id"]: paragraph["en"]
                for paragraph in english_result["paragraphs"]
            }
            inputs[sample_id] = {
                "sample_id": sample_id,
                "paragraphs": [
                    {
                        "id": paragraph["id"],
                        "zh": paragraph["zh"],
                        "en": english_by_id[paragraph["id"]],
                    }
                    for paragraph in original
                ],
            }
            active_rows.append(row)
            continue

        qa_path = qa_paths["outputs"] / f"{sample_id}.json"
        if not qa_path.exists():
            missing.append(sample_id)
            continue
        qa_artifact = json.loads(qa_path.read_text(encoding="utf-8"))
        qa_approved = bool(qa_artifact.get("approved_for_neutral_translation"))
        if stage in REPAIR_STAGES and qa_approved:
            continue
        prior_english_result = english_result
        prior_qa_artifact = qa_artifact
        if stage in REPAIR_STAGES and repair_round > 1:
            approved_in_prior_round = False
            last_completed_round = 0
            for prior_round in range(1, repair_round):
                previous_repair_paths = stage_paths(
                    experiment_root,
                    sample_set,
                    prerequisite_run_id,
                    "english_source_repair",
                    repair_round=prior_round,
                )
                previous_qa_paths = stage_paths(
                    experiment_root,
                    sample_set,
                    prerequisite_run_id,
                    "english_source_repair_qa",
                    repair_round=prior_round,
                )
                previous_repair_path = (
                    previous_repair_paths["outputs"] / f"{sample_id}.json"
                )
                previous_qa_path = (
                    previous_qa_paths["outputs"] / f"{sample_id}.json"
                )
                if not previous_repair_path.exists() or not previous_qa_path.exists():
                    break
                previous_repair = json.loads(
                    previous_repair_path.read_text(encoding="utf-8")
                )
                previous_qa = json.loads(
                    previous_qa_path.read_text(encoding="utf-8")
                )
                last_completed_round = prior_round
                if previous_qa.get("approved_for_neutral_translation"):
                    approved_in_prior_round = True
                    break
                prior_english_result = previous_repair["result"]
                prior_qa_artifact = previous_qa
            if approved_in_prior_round:
                continue
            if last_completed_round != repair_round - 1:
                missing.append(sample_id)
                continue

        if stage == "english_source_repair":
            original = split_paragraphs(hidden[sample_id]["original_zh"], "zh")
            english_by_id = {
                paragraph["id"]: paragraph["en"]
                for paragraph in prior_english_result["paragraphs"]
            }
            issues_by_id = {
                review["id"]: list(review.get("issues", []))
                for review in prior_qa_artifact["result"]["paragraph_reviews"]
            }
            inputs[sample_id] = {
                "sample_id": sample_id,
                "repair_round": repair_round,
                "paragraphs": [
                    {
                        "id": paragraph["id"],
                        "zh": paragraph["zh"],
                        "en": english_by_id[paragraph["id"]],
                        "issues": issues_by_id.get(paragraph["id"], []),
                    }
                    for paragraph in original
                ],
            }
            active_rows.append(row)
            continue
        if stage == "english_source_repair_qa":
            repair_path = repair_paths["outputs"] / f"{sample_id}.json"
            if not repair_path.exists():
                missing.append(sample_id)
                continue
            repair_artifact = json.loads(repair_path.read_text(encoding="utf-8"))
            original = split_paragraphs(hidden[sample_id]["original_zh"], "zh")
            repaired_by_id = {
                paragraph["id"]: paragraph["en"]
                for paragraph in repair_artifact["result"]["paragraphs"]
            }
            inputs[sample_id] = {
                "sample_id": sample_id,
                "repair_round": repair_round,
                "paragraphs": [
                    {
                        "id": paragraph["id"],
                        "zh": paragraph["zh"],
                        "en": repaired_by_id[paragraph["id"]],
                    }
                    for paragraph in original
                ],
            }
            active_rows.append(row)
            continue

        if qa_approved:
            downstream_english_result = english_result
        else:
            downstream_english_result = None
            found_any_round = False
            for candidate_round in range(1, 21):
                candidate_repair_paths = stage_paths(
                    experiment_root,
                    sample_set,
                    prerequisite_run_id,
                    "english_source_repair",
                    repair_round=candidate_round,
                )
                candidate_qa_paths = stage_paths(
                    experiment_root,
                    sample_set,
                    prerequisite_run_id,
                    "english_source_repair_qa",
                    repair_round=candidate_round,
                )
                repair_path = candidate_repair_paths["outputs"] / f"{sample_id}.json"
                repair_qa_path = candidate_qa_paths["outputs"] / f"{sample_id}.json"
                if not repair_path.exists() and not repair_qa_path.exists():
                    break
                if not repair_path.exists() or not repair_qa_path.exists():
                    missing.append(sample_id)
                    break
                found_any_round = True
                repair_artifact = json.loads(repair_path.read_text(encoding="utf-8"))
                repair_qa_artifact = json.loads(
                    repair_qa_path.read_text(encoding="utf-8")
                )
                if repair_qa_artifact.get("approved_for_neutral_translation"):
                    downstream_english_result = repair_artifact["result"]
                    break
            if downstream_english_result is None:
                if found_any_round:
                    unapproved.append(sample_id)
                elif sample_id not in missing:
                    missing.append(sample_id)
                continue
        if restricted_stage:
            neutral_path = neutral_paths["outputs"] / f"{sample_id}.json"
            if not neutral_path.exists():
                missing.append(sample_id)
                continue
            neutral_artifact = json.loads(
                neutral_path.read_text(encoding="utf-8")
            )
        if stage == "style_critique":
            base_paths = stage_paths(
                experiment_root,
                sample_set,
                run_id,
                "style_transfer",
                base_method_id,
                base_intensity,
            )
            base_path = base_paths["outputs"] / f"{sample_id}.json"
            if not base_path.exists():
                missing.append(sample_id)
                continue
            base_artifact = json.loads(base_path.read_text(encoding="utf-8"))
            inputs[sample_id] = {
                "sample_id": sample_id,
                "english_semantic_source": downstream_english_result["paragraphs"],
                "neutral_zh": neutral_artifact["result"]["paragraphs"],
                "candidate_zh": base_artifact["result"]["paragraphs"],
            }
        elif stage == "style_transfer":
            prior_output: list[dict[str, str]] | None = None
            critique: dict[str, Any] | None = None
            if method_id == "self_critique_repair":
                base_paths = stage_paths(
                    experiment_root,
                    sample_set,
                    run_id,
                    "style_transfer",
                    base_method_id,
                    base_intensity,
                )
                base_path = base_paths["outputs"] / f"{sample_id}.json"
                critique_paths = stage_paths(
                    experiment_root,
                    sample_set,
                    run_id,
                    "style_critique",
                    base_method_id,
                    base_intensity,
                )
                critique_path = critique_paths["outputs"] / f"{sample_id}.json"
                if not base_path.exists() or not critique_path.exists():
                    missing.append(sample_id)
                    continue
                base_artifact = json.loads(base_path.read_text(encoding="utf-8"))
                critique_artifact = json.loads(
                    critique_path.read_text(encoding="utf-8")
                )
                prior_output = base_artifact["result"]["paragraphs"]
                critique = critique_artifact["result"]
            inputs[sample_id] = build_method_request(
                experiment_root=experiment_root,
                method_id=method_id or "",
                intensity=intensity or "",
                sample_id=sample_id,
                english_semantic_source=downstream_english_result["paragraphs"],
                neutral_zh=neutral_artifact["result"]["paragraphs"],
                prior_output=prior_output,
                critique=critique,
            )
        else:
            inputs[sample_id] = {
                "sample_id": sample_id,
                "paragraphs": downstream_english_result["paragraphs"],
                "glossary": {},
                "comments": [],
            }
        active_rows.append(row)
    if missing:
        raise ValueError(
            f"{stage} requires successful prerequisite outputs from input run "
            f"{prerequisite_run_id!r}; missing {len(missing)} samples."
        )
    if unapproved:
        raise ValueError(
            f"{stage} is quarantined until English source QA passes; "
            f"{len(unapproved)} samples are unapproved."
        )
    if deterministic_failures:
        raise ValueError(
            "English source QA cannot start until deterministic source checks "
            f"pass; {len(deterministic_failures)} samples failed."
        )
    return active_rows, inputs


def expected_paragraph_ids(stage: str, request: dict[str, Any]) -> list[str]:
    paragraphs = (
        request["neutral_zh"]
        if stage == "style_transfer"
        else request["candidate_zh"]
        if stage == "style_critique"
        else request["paragraphs"]
    )
    return [paragraph["id"] for paragraph in paragraphs]


def deterministic_english_qa(
    request: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    source_by_id = {
        paragraph["id"]: paragraph["zh"] for paragraph in request["paragraphs"]
    }
    diagnostics: list[dict[str, Any]] = []
    for paragraph in result.get("paragraphs", []):
        paragraph_id = paragraph.get("id", "")
        english = paragraph.get("en", "")
        chinese = source_by_id.get(paragraph_id, "")
        english_letters = len(ENGLISH_LETTER_RE.findall(english))
        output_cjk = len(CJK_RE.findall(english))
        source_cjk_chars = CJK_RE.findall(chinese)
        source_cjk = len(source_cjk_chars)
        numeric_marker = bool(source_cjk_chars) and all(
            character in CHINESE_NUMERAL_CHARS for character in source_cjk_chars
        ) and bool(ARABIC_NUMBER_RE.search(english))
        word_count = len(re.findall(r"\b[A-Za-z]+(?:['-][A-Za-z]+)*\b", english))
        ratio = word_count / max(source_cjk, 1)
        issues: list[str] = []
        warnings: list[str] = []
        minimum_letters = 0 if source_cjk == 0 or numeric_marker else (
            1 if source_cjk <= 4 else max(4, source_cjk // 5)
        )
        if english_letters < minimum_letters:
            issues.append("insufficient_english_script")
        if output_cjk > 2:
            issues.append("suspicious_chinese_copy")
        if source_cjk >= 10 and not 0.20 <= ratio <= 3.00:
            issues.append("implausible_length_ratio")
        if Counter(ARABIC_NUMBER_RE.findall(chinese)) != Counter(
            ARABIC_NUMBER_RE.findall(english)
        ):
            warnings.append("arabic_number_surface_mismatch_requires_semantic_qa")
        diagnostics.append(
            {
                "id": paragraph_id,
                "status": "pass" if not issues else "fail",
                "issues": issues,
                "warnings": warnings,
                "source_cjk": source_cjk,
                "source_is_numeric_marker": numeric_marker,
                "english_words": word_count,
                "output_cjk": output_cjk,
                "word_to_source_cjk_ratio": round(ratio, 6),
            }
        )
    return {
        "protocol": "english_source_deterministic_qa.v2",
        "approved": bool(diagnostics)
        and len(diagnostics) == len(request["paragraphs"])
        and all(row["status"] == "pass" for row in diagnostics),
        "paragraphs": diagnostics,
    }


def validate_result(
    stage: str, request: dict[str, Any], result: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    if result.get("sample_id") != request["sample_id"]:
        errors.append("sample_id mismatch")
    if stage == "style_transfer":
        if result.get("method_id") != request.get("method_id"):
            errors.append("method_id mismatch")
        if result.get("intensity") != request.get("intensity"):
            errors.append("intensity mismatch")
        for field in (
            "style_cues_applied",
            "style_cues_skipped",
            "uncertainties",
        ):
            if not isinstance(result.get(field), list):
                errors.append(f"{field} must be a list")
    elif stage == "style_critique":
        if result.get("prompt_version") != "style_transfer_critique.v1":
            errors.append("prompt_version mismatch")
        if not isinstance(result.get("repair_required"), bool):
            errors.append("repair_required must be boolean")
        output_reviews = result.get("paragraph_reviews")
        if not isinstance(output_reviews, list):
            return errors + ["paragraph_reviews must be a list"]
        output_ids = [paragraph.get("id") for paragraph in output_reviews]
        if output_ids != expected_paragraph_ids(stage, request):
            errors.append("paragraph IDs or order do not match input")
        neutral_reviews = result.get("neutral_paragraph_reviews")
        if not isinstance(neutral_reviews, list):
            return errors + ["neutral_paragraph_reviews must be a list"]
        neutral_ids = [paragraph.get("id") for paragraph in neutral_reviews]
        if neutral_ids != expected_paragraph_ids(stage, request):
            errors.append("neutral paragraph IDs or order do not match input")
        statuses = [paragraph.get("status") for paragraph in output_reviews]
        if any(status not in {"pass", "minor", "major"} for status in statuses):
            errors.append("one or more critique statuses are invalid")
        if result.get("repair_required") != any(
            status != "pass" for status in statuses
        ):
            errors.append("repair_required disagrees with paragraph statuses")
        if len(output_ids) != len(set(output_ids)):
            errors.append("duplicate paragraph IDs")
        if len(neutral_ids) != len(set(neutral_ids)):
            errors.append("duplicate neutral paragraph IDs")
        return errors
    else:
        expected_version = {
            "english_semantic_source": "english_semantic_source.v1",
            "english_source_qa": "english_source_qa.v1",
            "english_source_repair": "english_source_repair.v1",
            "english_source_repair_qa": "english_source_qa.v1",
            "neutral_translation": "neutral_translation.v1",
        }[stage]
        if result.get("prompt_version") != expected_version:
            errors.append("prompt_version mismatch")
    output_paragraphs = (
        result.get("paragraph_reviews")
        if stage in {"english_source_qa", "english_source_repair_qa"}
        else result.get("paragraphs")
    )
    if not isinstance(output_paragraphs, list):
        return errors + ["paragraphs must be a list"]
    expected_ids = expected_paragraph_ids(stage, request)
    output_ids = [paragraph.get("id") for paragraph in output_paragraphs]
    if output_ids != expected_ids:
        errors.append("paragraph IDs or order do not match input")
    if stage in {"english_source_qa", "english_source_repair_qa"}:
        statuses = [paragraph.get("status") for paragraph in output_paragraphs]
        if any(status not in {"pass", "fail"} for status in statuses):
            errors.append("one or more QA statuses are invalid")
        approved = result.get("approved")
        if approved != all(status == "pass" for status in statuses):
            errors.append("QA approved flag disagrees with paragraph statuses")
    else:
        content_field = (
            "en"
            if stage in {"english_semantic_source", "english_source_repair"}
            else "zh"
        )
        if any(
            not isinstance(paragraph.get(content_field), str)
            or not paragraph[content_field].strip()
            for paragraph in output_paragraphs
        ):
            errors.append(f"one or more {content_field} outputs are empty")
    if len(output_ids) != len(set(output_ids)):
        errors.append("duplicate paragraph IDs")
    return errors


def parse_thread_id(stdout: str) -> str | None:
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "thread.started":
            return event.get("thread_id")
    return None


def codex_version() -> str:
    result = subprocess.run(
        ["codex", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def output_schema_path(experiment_root: Path, stage: str) -> Path:
    filename = {
        "english_semantic_source": "english_semantic_source_output.v1.schema.json",
        "english_source_qa": "english_source_qa_output.v1.schema.json",
        "english_source_repair": "english_source_repair_output.v1.schema.json",
        "english_source_repair_qa": "english_source_qa_output.v1.schema.json",
        "neutral_translation": "neutral_translation_output.v1.schema.json",
        "style_transfer": "style_transfer_output.v1.schema.json",
        "style_critique": "style_transfer_critique_output.v1.schema.json",
    }[stage]
    return experiment_root / "schemas" / filename


def prompt_path(experiment_root: Path, stage: str) -> Path:
    filename = {
        "english_semantic_source": "english_semantic_source.v1.md",
        "english_source_qa": "english_source_qa.v1.md",
        "english_source_repair": "english_source_repair.v1.md",
        "english_source_repair_qa": "english_source_qa.v1.md",
        "neutral_translation": "neutral_translation.v1.md",
        "style_transfer": "style_transfer_method.v1.md",
        "style_critique": "style_transfer_critique.v1.md",
    }[stage]
    return experiment_root / "prompts" / filename


def build_prompt(template: str, request: dict[str, Any]) -> str:
    return (
        template.rstrip()
        + "\n\n## Concrete Request\n\n"
        + "Do not use tools or inspect the filesystem. Work only from the JSON "
        + "request below.\n\n```json\n"
        + json.dumps(request, ensure_ascii=False, indent=2)
        + "\n```\n"
    )


def external_sandbox_profile() -> str:
    repo = str(REPO_ROOT)
    return "\n".join(
        (
            "(version 1)",
            "(allow default)",
            f'(deny file-read* (subpath "{repo}"))',
            f'(deny file-write* (subpath "{repo}"))',
            "",
        )
    )


def minimal_child_environment(temp_dir: Path) -> dict[str, str]:
    allowed_names = (
        "HOME",
        "USER",
        "LOGNAME",
        "LANG",
        "LC_ALL",
        "CODEX_HOME",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
    )
    environment = {
        name: os.environ[name] for name in allowed_names if name in os.environ
    }
    environment["PATH"] = os.environ.get("PATH", "/usr/bin:/bin")
    environment["TMPDIR"] = str(temp_dir)
    return environment


def resolved_codex_binary() -> Path:
    located = shutil.which("codex")
    if not located:
        raise FileNotFoundError("codex executable is not available on PATH")
    resolved = Path(located).resolve()
    version = codex_version()
    if not version.startswith("codex-cli "):
        raise RuntimeError(
            f"Refusing non-production Codex executable {resolved}: {version}"
        )
    return resolved


def verify_external_isolation(
    profile_path: Path,
    sandbox_dir: Path,
    experiment_root: Path,
    sample_set: str,
) -> dict[str, Any]:
    sandbox_exec = Path("/usr/bin/sandbox-exec")
    if platform.system() != "Darwin" or not sandbox_exec.exists():
        raise RuntimeError(
            "The v2 runner requires macOS sandbox-exec for repository read isolation."
        )
    allowed_canary = sandbox_dir / "allowed-canary.txt"
    allowed_canary.write_text("allowed-canary-v1", encoding="utf-8")
    base_command = [str(sandbox_exec), "-f", str(profile_path), "/bin/cat"]
    allowed = subprocess.run(
        [*base_command, str(allowed_canary)],
        capture_output=True,
        text=True,
        env=minimal_child_environment(sandbox_dir),
    )
    if allowed.returncode != 0 or allowed.stdout != "allowed-canary-v1":
        raise RuntimeError("External sandbox cannot read its isolated request area.")

    sample_dir = experiment_root / "sample_sets"
    denied_paths = (
        sample_dir / f"{sample_set}.hidden_targets.jsonl",
        sample_dir / f"{sample_set}.evaluator_allocation.jsonl",
        REPO_ROOT / "datasets/unmasked/chunks.clean.jsonl",
    )
    denied_results: list[dict[str, Any]] = []
    for path in denied_paths:
        probe = subprocess.run(
            [*base_command, str(path)],
            capture_output=True,
            text=True,
            env=minimal_child_environment(sandbox_dir),
        )
        denied = probe.returncode != 0 and not probe.stdout
        denied_results.append(
            {
                "path_class": (
                    "hidden_targets"
                    if "hidden_targets" in path.name
                    else "evaluator_allocation"
                    if "evaluator_allocation" in path.name
                    else "corpus"
                ),
                "denied": denied,
                "returncode": probe.returncode,
            }
        )
        if not denied:
            raise RuntimeError(f"External sandbox read-isolation probe failed: {path}")
    return {
        "profile_version": EXTERNAL_SANDBOX_VERSION,
        "profile_sha256": sha256_text(profile_path.read_text(encoding="utf-8")),
        "allowed_request_area": True,
        "denied_probes": denied_results,
    }


def run_one_attempt(
    *,
    stage: str,
    sample_id: str,
    request: dict[str, Any],
    prompt_template: str,
    prompt_sha256: str,
    schema_path: Path,
    schema_sha256: str,
    output_path: Path,
    run_id: str,
    attempt: int,
    model_config: dict[str, Any],
    codex_cli_version: str,
    codex_binary: Path,
    runner_sha256: str,
    environment_sha256: str,
    sandbox_profile_sha256: str,
    effective_command_sha256: str,
    execution_binding: dict[str, Any],
    repair_round: int = 1,
    base_method_id: str | None = None,
    base_intensity: str | None = None,
) -> dict[str, Any]:
    started_at = utc_now()
    input_sha256 = sha256_text(canonical_json(request))
    glossary_payload = request.get(
        "glossary", request.get("terminology_placeholders", {})
    )
    glossary_sha256 = sha256_text(canonical_json(glossary_payload))
    method_payload_sha256 = (
        sha256_text(canonical_json(request.get("method_payload", {})))
        if stage == "style_transfer"
        else None
    )
    reference_examples_sha256 = (
        sha256_text(canonical_json(request.get("reference_examples", [])))
        if stage == "style_transfer"
        else None
    )
    candidate_sha256 = (
        sha256_text(canonical_json(request.get("candidate_zh", [])))
        if stage == "style_critique"
        else None
    )
    validation_errors: list[str] = []
    response_id: str | None = None
    output_sha256: str | None = None
    response_usage: dict[str, Any] | None = None
    response_errors: list[dict[str, Any]] = []
    deterministic_qa: dict[str, Any] | None = None
    recorded_response_path = output_path
    status = "failed"
    stderr_tail = ""

    with tempfile.TemporaryDirectory(prefix=f"style-{stage}-{sample_id}-") as temp:
        sandbox_dir = Path(temp)
        response_path = sandbox_dir / "response.json"
        profile_path = sandbox_dir / "isolation.sb"
        profile_path.write_text(external_sandbox_profile(), encoding="utf-8")
        isolated_schema_path = sandbox_dir / "output.schema.json"
        isolated_schema_path.write_bytes(schema_path.read_bytes())
        codex_command = [
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
            model_config["codex_model"],
            "--config",
            f'model_reasoning_effort="{model_config["reasoning_effort"]}"',
            "--output-schema",
            str(isolated_schema_path),
            "--output-last-message",
            str(response_path),
            "--json",
            "-",
        ]
        command = [
            "/usr/bin/sandbox-exec",
            "-f",
            str(profile_path),
            *codex_command,
        ]
        try:
            completed = subprocess.run(
                command,
                input=build_prompt(prompt_template, request),
                capture_output=True,
                text=True,
                timeout=int(model_config["request_timeout_seconds"]),
                env=minimal_child_environment(sandbox_dir),
                cwd=sandbox_dir,
            )
            response_id = parse_thread_id(completed.stdout)
            for line in completed.stdout.splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "turn.completed":
                    response_usage = event.get("usage")
                if event.get("type") in {"turn.failed", "error"}:
                    response_errors.append(event)
            stderr_tail = completed.stderr[-2000:]
            if completed.returncode != 0:
                validation_errors.append(
                    f"codex exited with status {completed.returncode}"
                )
            elif not response_path.exists():
                validation_errors.append("codex did not write a final response")
            else:
                try:
                    result = json.loads(response_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    validation_errors.append(f"response is not valid JSON: {exc}")
                else:
                    output_sha256 = sha256_text(canonical_json(result))
                    validation_errors.extend(validate_result(stage, request, result))
                    if stage in {
                        "english_semantic_source",
                        "english_source_repair",
                    }:
                        deterministic_qa = deterministic_english_qa(request, result)
                        if not deterministic_qa["approved"]:
                            validation_errors.append(
                                "deterministic English-source QA failed"
                            )
                    if not validation_errors:
                        artifact = {
                            "schema_version": 1,
                            "run_id": run_id,
                            "stage": stage,
                            "sample_id": sample_id,
                            "attempt": attempt,
                            "model": model_config["codex_model"],
                            "reasoning_effort": model_config["reasoning_effort"],
                            "prompt_sha256": prompt_sha256,
                            "input_sha256": input_sha256,
                            "output_sha256": output_sha256,
                            "response_id": response_id,
                            "response_usage": response_usage,
                            "result": result,
                            **execution_binding,
                        }
                        if stage in REPAIR_STAGES:
                            artifact["repair_round"] = repair_round
                        if deterministic_qa is not None:
                            artifact["deterministic_qa"] = deterministic_qa
                        if stage in {
                            "english_source_qa",
                            "english_source_repair_qa",
                        }:
                            artifact["approved_for_neutral_translation"] = bool(
                                result["approved"]
                            )
                        if stage == "style_transfer":
                            artifact.update(
                                {
                                    "method_id": request["method_id"],
                                    "intensity": request["intensity"],
                                    "method_payload_sha256": method_payload_sha256,
                                    "reference_examples_sha256": (
                                        reference_examples_sha256
                                    ),
                                }
                            )
                        if stage == "style_critique":
                            artifact.update(
                                {
                                    "base_method_id": base_method_id,
                                    "base_intensity": base_intensity,
                                    "candidate_sha256": candidate_sha256,
                                }
                            )
                        elif stage == "style_transfer" and base_method_id:
                            artifact.update(
                                {
                                    "base_method_id": base_method_id,
                                    "base_intensity": base_intensity,
                                }
                            )
                        write_json(output_path, artifact)
                        status = "success"
                    else:
                        recorded_response_path = (
                            output_path.parent
                            / "_quarantine"
                            / f"{sample_id}.attempt{attempt}.json"
                        )
                        write_json(
                            recorded_response_path,
                            {
                                "schema_version": 1,
                                "run_id": run_id,
                                "stage": stage,
                                "sample_id": sample_id,
                                "attempt": attempt,
                                "response_id": response_id,
                                "output_sha256": output_sha256,
                                "validation_errors": validation_errors,
                                "deterministic_qa": deterministic_qa,
                                "result": result,
                            },
                        )
        except subprocess.TimeoutExpired:
            validation_errors.append("codex request timed out")

    ledger_row = {
        "run_id": run_id,
        "stage": stage,
        "sample_id": sample_id,
        "attempt": attempt,
        "started_at": started_at,
        "completed_at": utc_now(),
        "status": status,
        "model": model_config["codex_model"],
        "reasoning_effort": model_config["reasoning_effort"],
        "codex_cli_version": codex_cli_version,
        "codex_binary_realpath": str(codex_binary.resolve()),
        "provider": model_config["provider"],
        "prompt_sha256": prompt_sha256,
        "input_sha256": input_sha256,
        "glossary_sha256": glossary_sha256,
        "output_sha256": output_sha256,
        "schema_sha256": schema_sha256,
        "response_id": response_id,
        "response_usage": response_usage,
        "response_errors": response_errors,
        "response_file": str(recorded_response_path.relative_to(REPO_ROOT)),
        "runner_sha256": runner_sha256,
        "environment_sha256": environment_sha256,
        "sandbox_profile_sha256": sandbox_profile_sha256,
        "effective_command_sha256": effective_command_sha256,
        "deterministic_qa": deterministic_qa,
        "validation_errors": validation_errors,
        "stderr_tail": stderr_tail,
        **execution_binding,
    }
    if stage in REPAIR_STAGES:
        ledger_row["repair_round"] = repair_round
    if stage == "style_transfer":
        ledger_row.update(
            {
                "method_id": request["method_id"],
                "intensity": request["intensity"],
                "method_payload_sha256": method_payload_sha256,
                "reference_examples_sha256": reference_examples_sha256,
            }
        )
    if stage == "style_critique":
        ledger_row.update(
            {
                "base_method_id": base_method_id,
                "base_intensity": base_intensity,
                "candidate_sha256": candidate_sha256,
            }
        )
    elif stage == "style_transfer" and base_method_id:
        ledger_row.update(
            {
                "base_method_id": base_method_id,
                "base_intensity": base_intensity,
            }
        )
    return ledger_row


def load_successful_ledger_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    successful: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        if row.get("status") == "success":
            successful[row["sample_id"]] = row
    return successful


def append_ledger_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def frozen_run_record(
    *,
    args: argparse.Namespace,
    model_config_path: Path,
    model_config: dict[str, Any],
    prompt: Path,
    schema: Path,
    sample_summary: Path,
    codex_cli_version: str,
    codex_binary: Path,
    isolation_preflight: dict[str, Any],
) -> dict[str, Any]:
    path_method_id = (
        args.base_method_id if args.stage == "style_critique" else args.method_id
    )
    path_intensity = (
        args.base_intensity if args.stage == "style_critique" else args.intensity
    )
    run_root = stage_paths(
        args.experiment_root,
        args.sample_set,
        args.run_id,
        args.stage,
        path_method_id,
        path_intensity,
        args.repair_round,
    )["run_root"]
    runner_snapshot = freeze_source_snapshot(Path(__file__), run_root)
    model_config_snapshot = freeze_input_snapshot(
        model_config_path, run_root / "config_snapshots"
    )
    prompt_snapshot = freeze_input_snapshot(
        prompt, run_root / "config_snapshots" / "prompts"
    )
    schema_snapshot = freeze_input_snapshot(
        schema, run_root / "config_snapshots" / "schemas"
    )
    environment_keys = sorted(minimal_child_environment(Path("<isolated_tmp>")).keys())
    effective_command = {
        "wrapper": "/usr/bin/sandbox-exec -f <profile>",
        "codex_binary_realpath": str(codex_binary.resolve()),
        "arguments": [
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
            "--disable shell_tool",
            "--disable unified_exec",
            "--disable browser_use*",
            "--disable in_app_browser",
            "--disable computer_use",
            "--disable apps",
            "--disable multi_agent",
            "--cd <isolated_tmp>",
            f"--model {model_config['codex_model']}",
            f"model_reasoning_effort={model_config['reasoning_effort']}",
            "--output-schema <isolated_schema>",
            "--output-last-message <isolated_response>",
            "--json -",
        ],
    }
    record = {
        "schema_version": 2,
        "run_id": args.run_id,
        "input_run_id": args.input_run_id or args.run_id,
        "sample_set": args.sample_set,
        "stage": args.stage,
        "created_at": utc_now(),
        "model_config_path": str(model_config_path.relative_to(REPO_ROOT)),
        "model_config_sha256": file_sha256(model_config_path),
        "model_config_snapshot_path": str(
            model_config_snapshot.relative_to(REPO_ROOT)
        ),
        "model": model_config["codex_model"],
        "reasoning_effort": model_config["reasoning_effort"],
        "provider": model_config["provider"],
        "backend_snapshot_available": model_config["backend_snapshot_available"],
        "provenance_claim": model_config["provenance_claim"],
        "codex_cli_version": codex_cli_version,
        "codex_binary_realpath": str(codex_binary.resolve()),
        "effective_command": effective_command,
        "effective_command_sha256": sha256_text(canonical_json(effective_command)),
        "environment_allowlist": environment_keys,
        "prompt_path": str(prompt.relative_to(REPO_ROOT)),
        "prompt_sha256": file_sha256(prompt),
        "prompt_snapshot_path": str(prompt_snapshot.relative_to(REPO_ROOT)),
        "schema_path": str(schema.relative_to(REPO_ROOT)),
        "schema_sha256": file_sha256(schema),
        "schema_snapshot_path": str(schema_snapshot.relative_to(REPO_ROOT)),
        "sample_summary_path": str(sample_summary.relative_to(REPO_ROOT)),
        "sample_summary_sha256": file_sha256(sample_summary),
        "runner_sha256": file_sha256(Path(__file__)),
        "runner_snapshot_path": str(runner_snapshot.relative_to(REPO_ROOT)),
        "dependency_lock_sha256": file_sha256(REPO_ROOT / "uv.lock"),
        "environment_sha256": sha256_text(canonical_json(environment_keys)),
        "external_isolation_preflight": isolation_preflight,
        "sandbox_profile_sha256": isolation_preflight["profile_sha256"],
        "independent_session_per_request": True,
        "execution_batch_provenance_required": True,
        "hidden_target_access": (
            "single_sample_parent_stream_only"
            if args.stage
            in {
                "english_semantic_source",
                "english_source_qa",
                "english_source_repair",
                "english_source_repair_qa",
            }
            else "forbidden"
        ),
    }
    if args.stage in REPAIR_STAGES:
        record["repair_round"] = args.repair_round
    if args.stage in {"style_transfer", "style_critique"}:
        record["analysis_lock"] = validate_analysis_lock(
            args.analysis_lock.resolve(), args.experiment_root
        )
    if args.stage == "style_transfer":
        method_config_path, _ = validate_method_args(
            args.experiment_root,
            args.stage,
            args.method_id,
            args.intensity,
        )
        assert method_config_path is not None
        assets = load_frozen_assets(args.experiment_root)
        asset_style_prompt_sha256 = str(
            assets["assets"]["source_projections"]["style_prompt_sha256"]
        )
        if asset_style_prompt_sha256 != file_sha256(prompt):
            raise ValueError(
                "Current style-transfer prompt differs from the frozen method assets"
            )
        ids_path = sample_paths(args.experiment_root, args.sample_set)[
            "method_evaluation_ids"
        ]
        record.update(
            {
                "method_id": args.method_id,
                "intensity": args.intensity,
                "method_config_path": str(
                    method_config_path.relative_to(REPO_ROOT)
                ),
                "method_config_sha256": file_sha256(method_config_path),
                "payload_builder_sha256": file_sha256(
                    Path(__file__).with_name("style_transfer_payloads.py")
                ),
                "method_assets_sha256": sha256_text(canonical_json(assets)),
                "asset_style_prompt_sha256": asset_style_prompt_sha256,
                "method_evaluation_ids_sha256": file_sha256(ids_path),
            }
        )
    elif args.stage == "style_critique":
        ids_path = sample_paths(args.experiment_root, args.sample_set)[
            "method_evaluation_ids"
        ]
        record["method_evaluation_ids_sha256"] = file_sha256(ids_path)
    if args.base_method_id:
        base_config_path = (
            args.experiment_root
            / "method_registry"
            / "methods"
            / f"{args.base_method_id}.v1.json"
        )
        record.update(
            {
                "base_method_id": args.base_method_id,
                "base_intensity": args.base_intensity,
                "base_method_config_sha256": file_sha256(base_config_path),
            }
        )
    return record


def ensure_frozen_run_config(path: Path, record: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable_existing = {key: value for key, value in existing.items() if key != "created_at"}
        comparable_record = {key: value for key, value in record.items() if key != "created_at"}
        if comparable_existing != comparable_record:
            raise ValueError(f"Run config already exists with different values: {path}")
        return
    write_json(path, record)


def recorded_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def frozen_resource_path(
    *,
    run_config: dict[str, Any],
    snapshot_field: str,
    source_field: str,
    hash_field: str,
    fallback: Path,
) -> Path:
    candidates: list[Path] = []
    snapshot_value = run_config.get(snapshot_field)
    if isinstance(snapshot_value, str):
        candidates.append(recorded_path(snapshot_value))
    source_value = run_config.get(source_field)
    if isinstance(source_value, str):
        candidates.append(recorded_path(source_value))
    candidates.append(fallback)
    expected_hash = run_config.get(hash_field)
    runner_snapshot_value = run_config.get("runner_snapshot_path")
    if isinstance(runner_snapshot_value, str) and isinstance(expected_hash, str):
        run_root = recorded_path(runner_snapshot_value).parent.parent
        for candidate in sorted(run_root.glob(f"config_snapshots/**/*{expected_hash}*")):
            if candidate.is_file():
                candidates.append(candidate)
    for candidate in candidates:
        if candidate.exists() and file_sha256(candidate) == expected_hash:
            return candidate
    raise ValueError(
        f"No intact frozen resource for {snapshot_field}; expected {expected_hash}"
    )


def existing_stage_resources(
    args: argparse.Namespace, paths: dict[str, Path]
) -> tuple[dict[str, Any], Path, dict[str, Any], Path, Path]:
    run_config = json.loads(paths["run_config"].read_text(encoding="utf-8"))
    if file_sha256(Path(__file__)) != run_config.get("runner_sha256"):
        raise ValueError(
            "Current runner differs from the frozen stage runner; use a new run ID "
            "or complete runner development before extending this stage."
        )
    current_model_path, _ = load_model_config(args.experiment_root)
    model_config_path = current_model_path
    model_path = frozen_resource_path(
        run_config=run_config,
        snapshot_field="model_config_snapshot_path",
        source_field="model_config_path",
        hash_field="model_config_sha256",
        fallback=current_model_path,
    )
    prompt = frozen_resource_path(
        run_config=run_config,
        snapshot_field="prompt_snapshot_path",
        source_field="prompt_path",
        hash_field="prompt_sha256",
        fallback=prompt_path(args.experiment_root, args.stage),
    )
    schema = frozen_resource_path(
        run_config=run_config,
        snapshot_field="schema_snapshot_path",
        source_field="schema_path",
        hash_field="schema_sha256",
        fallback=output_schema_path(args.experiment_root, args.stage),
    )
    return (
        run_config,
        model_path,
        json.loads(model_path.read_text(encoding="utf-8")),
        prompt,
        schema,
    )


def prepare_plan(args: argparse.Namespace) -> dict[str, Any]:
    validate_stage_options(args)
    if args.stage != "english_semantic_source" and not args.run_id:
        raise ValueError(f"--run-id is required when planning {args.stage}.")
    model_config_path, model_config = load_model_config(args.experiment_root)
    runner_rows, inputs = load_stage_inputs(
        args.experiment_root,
        args.sample_set,
        args.stage,
        args.run_id or "",
        requested_sample_ids(args),
        args.limit,
        args.method_id,
        args.intensity,
        args.base_method_id,
        args.base_intensity,
        args.repair_round,
        args.input_run_id,
    )
    selected = runner_rows
    selection = execution_selection(
        args, [row["sample_id"] for row in selected]
    )
    admission = validate_execution_admission(args, selection)
    prompt = prompt_path(args.experiment_root, args.stage)
    schema = output_schema_path(args.experiment_root, args.stage)
    summary = sample_paths(args.experiment_root, args.sample_set)["summary"]
    plan = {
        "sample_set": args.sample_set,
        "stage": args.stage,
        "input_run_id": args.input_run_id or args.run_id,
        "samples": len(selected),
        "sample_ids": [row["sample_id"] for row in selected],
        "input_sha256": {
            row["sample_id"]: sha256_text(canonical_json(inputs[row["sample_id"]]))
            for row in selected
        },
        "model": model_config["codex_model"],
        "reasoning_effort": model_config["reasoning_effort"],
        "prompt_sha256": file_sha256(prompt),
        "schema_sha256": file_sha256(schema),
        "sample_summary_sha256": file_sha256(summary),
        "execution_selection_id": selection["execution_selection_id"],
        "execution_selection_sha256": selection["execution_selection_sha256"],
        "execution_admission": admission,
    }
    if args.stage in REPAIR_STAGES:
        plan["repair_round"] = args.repair_round
    if args.stage == "style_transfer":
        plan.update(
            {
                "method_id": args.method_id,
                "intensity": args.intensity,
            }
        )
    elif args.stage == "style_critique":
        plan.update(
            {
                "base_method_id": args.base_method_id,
                "base_intensity": args.base_intensity,
            }
        )
    return plan


def run_stage(args: argparse.Namespace) -> None:
    validate_stage_options(args)
    runner_rows, inputs = load_stage_inputs(
        args.experiment_root,
        args.sample_set,
        args.stage,
        args.run_id,
        requested_sample_ids(args),
        args.limit,
        args.method_id,
        args.intensity,
        args.base_method_id,
        args.base_intensity,
        args.repair_round,
        args.input_run_id,
    )
    selected = runner_rows
    paths = stage_paths(
        args.experiment_root,
        args.sample_set,
        args.run_id,
        args.stage,
        args.base_method_id if args.stage == "style_critique" else args.method_id,
        args.base_intensity if args.stage == "style_critique" else args.intensity,
        args.repair_round,
    )
    summary = sample_paths(args.experiment_root, args.sample_set)["summary"]
    codex_binary = resolved_codex_binary()
    cli_version = codex_version()
    if paths["run_config"].exists():
        (
            run_record,
            model_config_path,
            model_config,
            prompt,
            schema,
        ) = existing_stage_resources(args, paths)
        if run_record.get("codex_cli_version") != cli_version:
            raise ValueError("Codex CLI version differs from the frozen stage run")
        if run_record.get("codex_binary_realpath") != str(codex_binary.resolve()):
            raise ValueError("Codex binary differs from the frozen stage run")
    else:
        model_config_path, model_config = load_model_config(args.experiment_root)
        prompt = prompt_path(args.experiment_root, args.stage)
        schema = output_schema_path(args.experiment_root, args.stage)
        with tempfile.TemporaryDirectory(prefix="style-isolation-preflight-") as temp:
            preflight_dir = Path(temp)
            profile_path = preflight_dir / "isolation.sb"
            profile_path.write_text(external_sandbox_profile(), encoding="utf-8")
            isolation_preflight = verify_external_isolation(
                profile_path,
                preflight_dir,
                args.experiment_root,
                args.sample_set,
            )
        run_record = frozen_run_record(
            args=args,
            model_config_path=model_config_path,
            model_config=model_config,
            prompt=prompt,
            schema=schema,
            sample_summary=summary,
            codex_cli_version=cli_version,
            codex_binary=codex_binary,
            isolation_preflight=isolation_preflight,
        )
        ensure_frozen_run_config(paths["run_config"], run_record)
    selection = execution_selection(
        args, [row["sample_id"] for row in selected]
    )
    admission = validate_execution_admission(args, selection)
    execution_binding = freeze_execution_batch(paths, selection, admission)
    successful = load_successful_ledger_rows(paths["ledger"])
    if successful:
        provenance = validate_stage(args, require_complete=False)
        if provenance["status"] != "passed":
            raise ValueError(
                "Existing outputs failed strict provenance validation: "
                f"{provenance['errors'][:5]}"
            )
    if not args.resume and successful:
        raise ValueError(
            "Successful ledger rows already exist; pass --resume or use a new run ID."
        )

    pending = [row for row in selected if row["sample_id"] not in successful]
    prompt_template = prompt.read_text(encoding="utf-8")
    prompt_sha256 = file_sha256(prompt)
    schema_sha256 = file_sha256(schema)
    max_attempts = int(model_config["max_attempts"])
    jobs = args.jobs or int(model_config["default_parallel_jobs"])
    if jobs <= 0:
        raise ValueError("--jobs must be positive.")
    paths["outputs"].mkdir(parents=True, exist_ok=True)
    append_lock = threading.Lock()

    def run_sample(row: dict[str, Any]) -> list[dict[str, Any]]:
        sample_id = row["sample_id"]
        output_path = paths["outputs"] / f"{sample_id}.json"
        if output_path.exists() and sample_id not in successful:
            raise FileExistsError(
                f"Refusing to replace unledgered output: {output_path}"
            )
        attempts: list[dict[str, Any]] = []
        for attempt in range(1, max_attempts + 1):
            ledger_row = run_one_attempt(
                stage=args.stage,
                sample_id=sample_id,
                request=inputs[sample_id],
                prompt_template=prompt_template,
                prompt_sha256=prompt_sha256,
                schema_path=schema,
                schema_sha256=schema_sha256,
                output_path=output_path,
                run_id=args.run_id,
                attempt=attempt,
                model_config=model_config,
                codex_cli_version=cli_version,
                codex_binary=codex_binary,
                runner_sha256=run_record["runner_sha256"],
                environment_sha256=run_record["environment_sha256"],
                sandbox_profile_sha256=run_record[
                    "sandbox_profile_sha256"
                ],
                effective_command_sha256=run_record[
                    "effective_command_sha256"
                ],
                execution_binding=execution_binding,
                repair_round=args.repair_round,
                base_method_id=args.base_method_id,
                base_intensity=args.base_intensity,
            )
            attempts.append(ledger_row)
            with append_lock:
                append_ledger_rows(paths["ledger"], [ledger_row])
            if ledger_row["status"] == "success":
                break
        return attempts

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(run_sample, row): row for row in pending}
        for future in as_completed(futures):
            sample_id = futures[future]["sample_id"]
            try:
                rows = future.result()
            except Exception as exc:
                failures.append(f"{sample_id}: {exc}")
                continue
            if rows[-1]["status"] != "success":
                failures.append(sample_id)

    if failures:
        raise RuntimeError(
            f"{len(failures)} samples failed; see {paths['ledger']}: {failures[:5]}"
        )
    print(
        json.dumps(
            {
                "status": "complete",
                "run_id": args.run_id,
                "stage": args.stage,
                "new_outputs": len(pending),
                "resumed_outputs": len(selected) - len(pending),
                "ledger": str(paths["ledger"]),
                "execution_selection_id": selection["execution_selection_id"],
                "execution_selection_sha256": selection[
                    "execution_selection_sha256"
                ],
            },
            indent=2,
        )
    )


def validate_stage(
    args: argparse.Namespace, *, require_complete: bool = True
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        validate_stage_options(args)
        runner_rows, inputs = load_stage_inputs(
            args.experiment_root,
            args.sample_set,
            args.stage,
            args.run_id,
            requested_sample_ids(args),
            args.limit,
            args.method_id,
            args.intensity,
            args.base_method_id,
            args.base_intensity,
            args.repair_round,
            args.input_run_id,
        )
    except Exception as exc:
        return {
            "status": "failed",
            "run_id": args.run_id,
            "stage": args.stage,
            "expected_outputs": 0,
            "valid_outputs": 0,
            "approved_outputs": 0,
            "errors": [f"input preflight failed: {exc}"],
        }
    selected = runner_rows
    selected_ids = {row["sample_id"] for row in selected}
    try:
        expected_execution_selection = execution_selection(
            args, [row["sample_id"] for row in selected]
        )
        expected_execution_admission = validate_execution_admission(
            args, expected_execution_selection
        )
    except Exception as exc:
        expected_execution_selection = {}
        expected_execution_admission = None
        errors.append(f"execution admission validation failed: {exc}")
    paths = stage_paths(
        args.experiment_root,
        args.sample_set,
        args.run_id,
        args.stage,
        args.base_method_id if args.stage == "style_critique" else args.method_id,
        args.base_intensity if args.stage == "style_critique" else args.intensity,
        args.repair_round,
    )
    if not paths["run_config"].exists():
        errors.append("frozen run configuration is missing")
        run_config: dict[str, Any] = {}
    else:
        run_config = json.loads(paths["run_config"].read_text(encoding="utf-8"))

    current_model_path, _ = load_model_config(args.experiment_root)
    try:
        model_config_path = frozen_resource_path(
            run_config=run_config,
            snapshot_field="model_config_snapshot_path",
            source_field="model_config_path",
            hash_field="model_config_sha256",
            fallback=current_model_path,
        )
        prompt = frozen_resource_path(
            run_config=run_config,
            snapshot_field="prompt_snapshot_path",
            source_field="prompt_path",
            hash_field="prompt_sha256",
            fallback=prompt_path(args.experiment_root, args.stage),
        )
        schema = frozen_resource_path(
            run_config=run_config,
            snapshot_field="schema_snapshot_path",
            source_field="schema_path",
            hash_field="schema_sha256",
            fallback=output_schema_path(args.experiment_root, args.stage),
        )
        model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"frozen run resource validation failed: {exc}")
        model_config = {}
        prompt = prompt_path(args.experiment_root, args.stage)
        schema = output_schema_path(args.experiment_root, args.stage)
    summary = sample_paths(args.experiment_root, args.sample_set)["summary"]
    static_expectations = {
        "run_id": args.run_id,
        "input_run_id": args.input_run_id or args.run_id,
        "sample_set": args.sample_set,
        "stage": args.stage,
        "model_config_sha256": file_sha256(model_config_path),
        "model": model_config.get("codex_model"),
        "reasoning_effort": model_config.get("reasoning_effort"),
        "provider": model_config.get("provider"),
        "prompt_sha256": file_sha256(prompt),
        "schema_sha256": file_sha256(schema),
        "sample_summary_sha256": file_sha256(summary),
    }
    if args.stage in REPAIR_STAGES:
        static_expectations["repair_round"] = args.repair_round
    if args.stage in {"style_transfer", "style_critique"}:
        static_expectations["analysis_lock"] = validate_analysis_lock(
            args.analysis_lock.resolve(), args.experiment_root
        )
    if args.stage == "style_transfer":
        method_config_path, _ = validate_method_args(
            args.experiment_root,
            args.stage,
            args.method_id,
            args.intensity,
        )
        assert method_config_path is not None
        assets = load_frozen_assets(args.experiment_root)
        asset_style_prompt_sha256 = str(
            assets["assets"]["source_projections"]["style_prompt_sha256"]
        )
        if asset_style_prompt_sha256 != file_sha256(prompt):
            errors.append(
                "current style-transfer prompt differs from frozen method assets"
            )
        ids_path = sample_paths(args.experiment_root, args.sample_set)[
            "method_evaluation_ids"
        ]
        static_expectations.update(
            {
                "method_id": args.method_id,
                "intensity": args.intensity,
                "method_config_sha256": file_sha256(method_config_path),
                "payload_builder_sha256": file_sha256(
                    Path(__file__).with_name("style_transfer_payloads.py")
                ),
                "method_assets_sha256": sha256_text(canonical_json(assets)),
                "asset_style_prompt_sha256": asset_style_prompt_sha256,
                "method_evaluation_ids_sha256": file_sha256(ids_path),
            }
        )
    elif args.stage == "style_critique":
        ids_path = sample_paths(args.experiment_root, args.sample_set)[
            "method_evaluation_ids"
        ]
        static_expectations["method_evaluation_ids_sha256"] = file_sha256(
            ids_path
        )
    if args.base_method_id:
        base_config_path = (
            args.experiment_root
            / "method_registry"
            / "methods"
            / f"{args.base_method_id}.v1.json"
        )
        static_expectations.update(
            {
                "base_method_id": args.base_method_id,
                "base_intensity": args.base_intensity,
                "base_method_config_sha256": file_sha256(base_config_path),
            }
        )
    for field, expected in static_expectations.items():
        if run_config.get(field) != expected:
            errors.append(f"run config {field} does not match current frozen input")
    runner_snapshot_value = run_config.get("runner_snapshot_path")
    if not isinstance(runner_snapshot_value, str):
        errors.append("run config lacks a runner source snapshot")
    else:
        runner_snapshot = REPO_ROOT / runner_snapshot_value
        if not runner_snapshot.exists():
            errors.append("runner source snapshot is missing")
        elif file_sha256(runner_snapshot) != run_config.get("runner_sha256"):
            errors.append("runner source snapshot hash mismatch")
    isolation = run_config.get("external_isolation_preflight", {})
    if not isolation.get("allowed_request_area") or not all(
        row.get("denied") for row in isolation.get("denied_probes", [])
    ):
        errors.append("run config lacks a passing external-isolation canary")

    ledger_rows = list(iter_jsonl(paths["ledger"])) if paths["ledger"].exists() else []
    required_fields = set(model_config["required_ledger_fields"])
    required_fields.update(
        {
            "provider",
            "codex_binary_realpath",
            "response_id",
            "response_usage",
            "response_errors",
            "runner_sha256",
            "environment_sha256",
            "sandbox_profile_sha256",
            "effective_command_sha256",
            "glossary_sha256",
        }
    )
    execution_fields = {
        "execution_selection_id",
        "execution_selection_sha256",
        "execution_selection_sample_count",
        "execution_selection_path",
        "execution_selection_is_frozen_file",
        "active_sample_count",
        "active_sample_ids_sha256",
        "execution_batch_config_path",
        "execution_batch_config_sha256",
        "execution_admission",
    }
    if run_config.get("execution_batch_provenance_required") is True:
        required_fields.update(execution_fields)
    if args.stage in REPAIR_STAGES:
        required_fields.add("repair_round")
    if args.stage == "style_transfer":
        required_fields.update(
            {
                "method_id",
                "intensity",
                "method_payload_sha256",
                "reference_examples_sha256",
            }
        )
        if args.base_method_id:
            required_fields.update({"base_method_id", "base_intensity"})
    elif args.stage == "style_critique":
        required_fields.update(
            {"base_method_id", "base_intensity", "candidate_sha256"}
        )
    attempts_by_sample: dict[str, list[dict[str, Any]]] = {}
    for ledger_row in ledger_rows:
        sample_id = ledger_row.get("sample_id")
        if not isinstance(sample_id, str):
            errors.append("ledger row has a missing or non-string sample_id")
            continue
        if sample_id not in selected_ids:
            continue
        attempts_by_sample.setdefault(sample_id, []).append(ledger_row)
        missing_fields = sorted(required_fields - set(ledger_row))
        if missing_fields:
            errors.append(f"{sample_id}: ledger fields missing {missing_fields}")
        if run_config.get("execution_batch_provenance_required") is True:
            binding_errors = validate_shared_execution_binding(
                ledger_row,
                sample_id=sample_id,
                expected_run_config_path=paths["run_config"],
            )
            errors.extend(
                f"{sample_id}: ledger {value}" for value in binding_errors
            )
            for field in (
                "execution_selection_id",
                "execution_selection_sha256",
                "execution_selection_sample_count",
            ):
                if ledger_row.get(field) != expected_execution_selection.get(field):
                    errors.append(f"{sample_id}: ledger {field} mismatch")
            if ledger_row.get("execution_admission") != expected_execution_admission:
                errors.append(f"{sample_id}: ledger execution admission mismatch")
        expected_ledger = {
            "run_id": args.run_id,
            "stage": args.stage,
            "model": run_config.get("model"),
            "reasoning_effort": run_config.get("reasoning_effort"),
            "provider": run_config.get("provider"),
            "codex_cli_version": run_config.get("codex_cli_version"),
            "codex_binary_realpath": run_config.get("codex_binary_realpath"),
            "prompt_sha256": run_config.get("prompt_sha256"),
            "schema_sha256": run_config.get("schema_sha256"),
            "runner_sha256": run_config.get("runner_sha256"),
            "environment_sha256": run_config.get("environment_sha256"),
            "sandbox_profile_sha256": run_config.get("sandbox_profile_sha256"),
            "effective_command_sha256": run_config.get("effective_command_sha256"),
        }
        if args.stage in REPAIR_STAGES:
            expected_ledger["repair_round"] = args.repair_round
        if args.stage == "style_transfer":
            expected_ledger.update(
                {
                    "method_id": args.method_id,
                    "intensity": args.intensity,
                    "method_payload_sha256": sha256_text(
                        canonical_json(inputs[sample_id].get("method_payload", {}))
                    ),
                    "reference_examples_sha256": sha256_text(
                        canonical_json(
                            inputs[sample_id].get("reference_examples", [])
                        )
                    ),
                }
            )
            if args.base_method_id:
                expected_ledger.update(
                    {
                        "base_method_id": args.base_method_id,
                        "base_intensity": args.base_intensity,
                    }
                )
        elif args.stage == "style_critique":
            expected_ledger.update(
                {
                    "base_method_id": args.base_method_id,
                    "base_intensity": args.base_intensity,
                    "candidate_sha256": sha256_text(
                        canonical_json(inputs[sample_id].get("candidate_zh", []))
                    ),
                }
            )
        for field, expected in expected_ledger.items():
            if ledger_row.get(field) != expected:
                errors.append(f"{sample_id}: ledger {field} mismatch")
        if sample_id in inputs:
            expected_input_hash = sha256_text(canonical_json(inputs[sample_id]))
            if ledger_row.get("input_sha256") != expected_input_hash:
                errors.append(f"{sample_id}: ledger input hash mismatch")
            glossary = inputs[sample_id].get(
                "glossary", inputs[sample_id].get("terminology_placeholders", {})
            )
            if ledger_row.get("glossary_sha256") != sha256_text(
                canonical_json(glossary)
            ):
                errors.append(f"{sample_id}: ledger glossary hash mismatch")

    successful: dict[str, dict[str, Any]] = {}
    for sample_id, attempt_rows in attempts_by_sample.items():
        ordered = list(attempt_rows)
        attempt_numbers = [row.get("attempt") for row in ordered]
        if any(
            not isinstance(number, int)
            or number < 1
            or number > int(model_config["max_attempts"])
            for number in attempt_numbers
        ):
            errors.append(f"{sample_id}: invalid attempt number")
        success_rows = [row for row in ordered if row.get("status") == "success"]
        if len(success_rows) > 1:
            errors.append(f"{sample_id}: duplicate successful ledger rows")
        elif success_rows:
            if ordered[-1] is not success_rows[0]:
                errors.append(f"{sample_id}: attempts continue after success")
            successful[sample_id] = success_rows[0]
            if not success_rows[0].get("response_id"):
                errors.append(f"{sample_id}: successful row lacks response ID")
            if not isinstance(success_rows[0].get("response_usage"), dict):
                errors.append(f"{sample_id}: successful row lacks response usage")

    valid_samples = 0
    approved_samples = 0
    for row in selected:
        sample_id = row["sample_id"]
        output_path = paths["outputs"] / f"{sample_id}.json"
        if sample_id not in successful:
            if require_complete:
                errors.append(f"{sample_id}: no successful ledger row")
            continue
        if not output_path.exists():
            errors.append(f"{sample_id}: output file missing")
            continue
        artifact = json.loads(output_path.read_text(encoding="utf-8"))
        success_row = successful[sample_id]
        expected_artifact = {
            "run_id": args.run_id,
            "stage": args.stage,
            "sample_id": sample_id,
            "attempt": success_row["attempt"],
            "model": run_config.get("model"),
            "reasoning_effort": run_config.get("reasoning_effort"),
            "prompt_sha256": run_config.get("prompt_sha256"),
            "input_sha256": success_row.get("input_sha256"),
            "response_id": success_row.get("response_id"),
        }
        if args.stage in REPAIR_STAGES:
            expected_artifact["repair_round"] = args.repair_round
        if run_config.get("execution_batch_provenance_required") is True:
            expected_artifact.update(
                {field: success_row.get(field) for field in execution_fields}
            )
            binding_errors = validate_shared_execution_binding(
                artifact,
                sample_id=sample_id,
                expected_run_config_path=paths["run_config"],
            )
            errors.extend(
                f"{sample_id}: artifact {value}" for value in binding_errors
            )
        if args.stage == "style_transfer":
            expected_artifact.update(
                {
                    "method_id": args.method_id,
                    "intensity": args.intensity,
                    "method_payload_sha256": success_row.get(
                        "method_payload_sha256"
                    ),
                    "reference_examples_sha256": success_row.get(
                        "reference_examples_sha256"
                    ),
                }
            )
            if args.base_method_id:
                expected_artifact.update(
                    {
                        "base_method_id": args.base_method_id,
                        "base_intensity": args.base_intensity,
                    }
                )
        elif args.stage == "style_critique":
            expected_artifact.update(
                {
                    "base_method_id": args.base_method_id,
                    "base_intensity": args.base_intensity,
                    "candidate_sha256": success_row.get("candidate_sha256"),
                }
            )
        for field, expected in expected_artifact.items():
            if artifact.get(field) != expected:
                errors.append(f"{sample_id}: artifact {field} mismatch")
        result_errors = validate_result(args.stage, inputs[sample_id], artifact["result"])
        if result_errors:
            errors.append(f"{sample_id}: {result_errors}")
        output_hash = sha256_text(canonical_json(artifact["result"]))
        if output_hash != artifact.get("output_sha256"):
            errors.append(f"{sample_id}: artifact output hash mismatch")
        if output_hash != success_row.get("output_sha256"):
            errors.append(f"{sample_id}: ledger output hash mismatch")
        expected_response_file = str(output_path.relative_to(REPO_ROOT))
        if success_row.get("response_file") != expected_response_file:
            errors.append(f"{sample_id}: ledger response path mismatch")
        if args.stage in {"english_semantic_source", "english_source_repair"}:
            qa = deterministic_english_qa(inputs[sample_id], artifact["result"])
            recorded_qa = artifact.get("deterministic_qa")
            approved_legacy_v1 = bool(
                isinstance(recorded_qa, dict)
                and recorded_qa.get("protocol")
                == "english_source_deterministic_qa.v1"
                and recorded_qa.get("approved") is True
                and qa["approved"]
            )
            if not qa["approved"] or (
                recorded_qa != qa and not approved_legacy_v1
            ):
                errors.append(f"{sample_id}: deterministic English QA mismatch")
            else:
                approved_samples += 1
        elif args.stage in {"english_source_qa", "english_source_repair_qa"}:
            approved = bool(artifact["result"].get("approved"))
            if artifact.get("approved_for_neutral_translation") != approved:
                errors.append(f"{sample_id}: QA approval-state mismatch")
            if approved:
                approved_samples += 1
        else:
            approved_samples += 1
        valid_samples += 1
    result = {
        "status": "passed" if not errors else "failed",
        "run_id": args.run_id,
        "stage": args.stage,
        "expected_outputs": len(selected),
        "valid_outputs": valid_samples,
        "approved_outputs": approved_samples,
        "require_complete": require_complete,
        "errors": errors,
    }
    if args.stage == "style_transfer":
        result.update(
            {
                "method_id": args.method_id,
                "intensity": args.intensity,
            }
        )
    elif args.stage == "style_critique":
        result.update(
            {
                "base_method_id": args.base_method_id,
                "base_intensity": args.base_intensity,
            }
        )
    return result


def main() -> None:
    args = parse_args()
    if args.command == "plan":
        plan = prepare_plan(args)
        plan["sample_ids"] = plan["sample_ids"][:10]
        plan["input_sha256"] = dict(list(plan["input_sha256"].items())[:10])
        print(json.dumps(plan, ensure_ascii=False, indent=2))
    elif args.command == "run":
        run_stage(args)
    else:
        result = validate_stage(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result["status"] != "passed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
