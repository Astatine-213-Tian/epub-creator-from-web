from __future__ import annotations

"""Shared deterministic provenance validation for style-study artifacts."""

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from experiments.iteration1.style_analysis_lock import require_matching_analysis_binding


from experiments.shared.paths import RESEARCH_ROOT


REPO_ROOT = RESEARCH_ROOT
EXECUTION_BINDING_FIELDS = {
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
LEDGER_CONFIG_FIELDS = {
    "model",
    "reasoning_effort",
    "provider",
    "codex_cli_version",
    "codex_binary_realpath",
    "prompt_sha256",
    "schema_sha256",
    "runner_sha256",
    "environment_sha256",
    "sandbox_profile_sha256",
    "effective_command_sha256",
}


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


def resolve_recorded_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _resource_candidates(
    config: Mapping[str, Any],
    *,
    run_root: Path,
    label: str,
) -> list[Path]:
    paths: list[Path] = []
    for field in (f"{label}_snapshot_path", f"{label}_path"):
        value = config.get(field)
        if isinstance(value, str) and value:
            paths.append(resolve_recorded_path(value))
    expected = config.get(f"{label}_sha256")
    if isinstance(expected, str) and expected:
        paths.extend(
            candidate.resolve()
            for candidate in sorted(
                run_root.glob(f"config_snapshots/**/*{expected}*")
            )
            if candidate.is_file()
        )
    return paths


def validate_run_config(
    path: Path,
    *,
    expected: Mapping[str, Any] | None = None,
    expected_analysis_lock: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    if not path.exists():
        return None, ["run_config_missing"]
    try:
        config = load_json(path)
    except Exception as exc:
        return None, [f"run_config_invalid:{exc}"]
    errors: list[str] = []
    if expected_analysis_lock is not None:
        try:
            require_matching_analysis_binding(
                config, expected_analysis_lock, label="run config"
            )
        except ValueError as exc:
            errors.append(str(exc))
    for field, value in (expected or {}).items():
        if config.get(field) != value:
            errors.append(f"run_config_{field}_mismatch")
    run_root = path.resolve().parent.parent
    for label in ("model_config", "prompt", "schema"):
        expected_hash = config.get(f"{label}_sha256")
        candidates = _resource_candidates(config, run_root=run_root, label=label)
        if not isinstance(expected_hash, str) or not any(
            candidate.exists() and file_sha256(candidate) == expected_hash
            for candidate in candidates
        ):
            errors.append(f"run_config_{label}_snapshot_missing_or_mismatched")
    runner_value = config.get("runner_snapshot_path")
    runner_path = (
        resolve_recorded_path(runner_value)
        if isinstance(runner_value, str) and runner_value
        else None
    )
    if (
        runner_path is None
        or not runner_path.exists()
        or file_sha256(runner_path) != config.get("runner_sha256")
    ):
        errors.append("run_config_runner_snapshot_missing_or_mismatched")
    isolation = config.get("external_isolation_preflight")
    if not isinstance(isolation, dict) or not isolation.get("allowed_request_area"):
        errors.append("run_config_external_isolation_missing_or_failed")
    elif not all(row.get("denied") for row in isolation.get("denied_probes", [])):
        errors.append("run_config_external_isolation_denied_probe_failed")
    return config, errors


def validate_execution_binding(
    record: Mapping[str, Any],
    *,
    sample_id: str,
    expected_run_config_path: Path | None = None,
    expected_selection_id: str | None = None,
    expected_selection_sha256: str | None = None,
    expected_admission: Mapping[str, Any] | None = None,
    expected_analysis_lock: Mapping[str, Any] | None = None,
) -> list[str]:
    present = EXECUTION_BINDING_FIELDS.intersection(record)
    if not present:
        return ["execution_binding_missing"]
    missing = sorted(EXECUTION_BINDING_FIELDS - set(record))
    if missing:
        return [f"execution_binding_fields_missing:{','.join(missing)}"]
    errors: list[str] = []
    batch_value = record.get("execution_batch_config_path")
    if not isinstance(batch_value, str):
        return ["execution_batch_path_invalid"]
    batch_path = resolve_recorded_path(batch_value)
    if not batch_path.exists():
        return ["execution_batch_missing"]
    if file_sha256(batch_path) != record.get("execution_batch_config_sha256"):
        return ["execution_batch_sha256_mismatch"]
    try:
        batch = load_json(batch_path)
    except Exception:
        return ["execution_batch_invalid_json"]
    for field in EXECUTION_BINDING_FIELDS - {
        "execution_batch_config_path",
        "execution_batch_config_sha256",
        "execution_admission",
    }:
        if batch.get(field) != record.get(field):
            errors.append(f"execution_batch_{field}_mismatch")
    if batch.get("admission") != record.get("execution_admission"):
        errors.append("execution_batch_admission_mismatch")
    if expected_analysis_lock is not None:
        for label, admission in (
            ("record admission", record.get("execution_admission")),
            ("execution batch admission", batch.get("admission")),
        ):
            if not isinstance(admission, Mapping):
                errors.append(f"{label} missing")
                continue
            try:
                require_matching_analysis_binding(
                    admission, expected_analysis_lock, label=label
                )
            except ValueError as exc:
                errors.append(str(exc))
    active_ids = batch.get("active_sample_ids")
    if not isinstance(active_ids, list) or not all(
        isinstance(value, str) for value in active_ids
    ):
        errors.append("execution_batch_active_ids_invalid")
    else:
        if sample_id not in active_ids:
            errors.append("execution_batch_sample_missing")
        if len(active_ids) != int(record.get("active_sample_count", -1)):
            errors.append("execution_batch_active_count_mismatch")
        if sha256_text(canonical_json(sorted(active_ids))) != record.get(
            "active_sample_ids_sha256"
        ):
            errors.append("execution_batch_active_ids_sha256_mismatch")
    stage_config_value = batch.get("stage_run_config_path")
    if not isinstance(stage_config_value, str):
        errors.append("execution_batch_stage_run_config_path_invalid")
    else:
        stage_config_path = resolve_recorded_path(stage_config_value)
        if not stage_config_path.exists():
            errors.append("execution_batch_stage_run_config_missing")
        elif file_sha256(stage_config_path) != batch.get(
            "stage_run_config_sha256"
        ):
            errors.append("execution_batch_stage_run_config_sha256_mismatch")
        elif expected_analysis_lock is not None:
            try:
                require_matching_analysis_binding(
                    load_json(stage_config_path),
                    expected_analysis_lock,
                    label="execution batch run config",
                )
            except ValueError as exc:
                errors.append(str(exc))
        if (
            expected_run_config_path is not None
            and stage_config_path != expected_run_config_path.resolve()
        ):
            errors.append("execution_batch_stage_run_config_path_mismatch")
    selection_value = record.get("execution_selection_path")
    if record.get("execution_selection_is_frozen_file") is True:
        if not isinstance(selection_value, str):
            errors.append("execution_selection_path_invalid")
        else:
            selection_path = resolve_recorded_path(selection_value)
            if not selection_path.exists():
                errors.append("execution_selection_missing")
            elif file_sha256(selection_path) != record.get(
                "execution_selection_sha256"
            ):
                errors.append("execution_selection_sha256_mismatch")
            else:
                selection = load_json(selection_path)
                sample_ids = selection.get("sample_ids", [])
                if sample_id not in sample_ids:
                    errors.append("execution_selection_sample_missing")
                if selection.get("selection_id") != record.get(
                    "execution_selection_id"
                ):
                    errors.append("execution_selection_id_mismatch")
                if len(sample_ids) != int(
                    record.get("execution_selection_sample_count", -1)
                ):
                    errors.append("execution_selection_sample_count_mismatch")
    elif selection_value is not None:
        errors.append("descriptive_execution_selection_path_present")
    if expected_selection_id is not None and record.get(
        "execution_selection_id"
    ) != expected_selection_id:
        errors.append("execution_selection_expected_id_mismatch")
    if expected_selection_sha256 is not None and record.get(
        "execution_selection_sha256"
    ) != expected_selection_sha256:
        errors.append("execution_selection_expected_sha256_mismatch")
    if expected_admission is not None and record.get(
        "execution_admission"
    ) != dict(expected_admission):
        errors.append("execution_admission_expected_mismatch")
    return errors


def validate_artifact_and_ledger(
    *,
    artifact_path: Path,
    artifact: Mapping[str, Any],
    ledger_row: Mapping[str, Any] | None,
    run_config_path: Path,
    sample_id: str,
    run_id: str,
    stage: str,
    expected_analysis_lock: Mapping[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    expected_config = {"run_id": run_id, "stage": stage}
    if artifact.get("sample_set") is not None:
        expected_config["sample_set"] = artifact.get("sample_set")
    config, config_errors = validate_run_config(
        run_config_path,
        expected=expected_config,
        expected_analysis_lock=expected_analysis_lock,
    )
    errors.extend(config_errors)
    expected_identity = {"sample_id": sample_id, "run_id": run_id, "stage": stage}
    for field, value in expected_identity.items():
        if artifact.get(field) != value:
            errors.append(f"artifact_{field}_mismatch")
    result = artifact.get("result")
    if not isinstance(result, dict):
        return errors + ["artifact_result_missing"]
    output_hash = sha256_text(canonical_json(result))
    if artifact.get("output_sha256") != output_hash:
        errors.append("artifact_output_sha256_mismatch")
    if ledger_row is None:
        return errors + ["successful_ledger_row_missing"]
    for field, value in expected_identity.items():
        if ledger_row.get(field) != value:
            errors.append(f"ledger_{field}_mismatch")
    if ledger_row.get("output_sha256") != output_hash:
        errors.append("ledger_output_sha256_mismatch")
    response_file = ledger_row.get("response_file")
    if not isinstance(response_file, str) or resolve_recorded_path(
        response_file
    ) != artifact_path.resolve():
        errors.append("ledger_response_file_mismatch")
    if config is not None:
        for field in LEDGER_CONFIG_FIELDS:
            if ledger_row.get(field) != config.get(field):
                errors.append(f"ledger_run_config_{field}_mismatch")
    artifact_binding = EXECUTION_BINDING_FIELDS.intersection(artifact)
    ledger_binding = EXECUTION_BINDING_FIELDS.intersection(ledger_row)
    if artifact_binding != EXECUTION_BINDING_FIELDS or ledger_binding != EXECUTION_BINDING_FIELDS:
        errors.append("artifact_or_ledger_execution_binding_incomplete")
    else:
        for field in EXECUTION_BINDING_FIELDS:
            if artifact.get(field) != ledger_row.get(field):
                errors.append(f"artifact_ledger_{field}_mismatch")
        errors.extend(
            f"artifact_{value}"
            for value in validate_execution_binding(
                artifact,
                sample_id=sample_id,
                expected_run_config_path=run_config_path,
                expected_analysis_lock=expected_analysis_lock,
            )
        )
        errors.extend(
            f"ledger_{value}"
            for value in validate_execution_binding(
                ledger_row,
                sample_id=sample_id,
                expected_run_config_path=run_config_path,
                expected_analysis_lock=expected_analysis_lock,
            )
        )
    return errors
