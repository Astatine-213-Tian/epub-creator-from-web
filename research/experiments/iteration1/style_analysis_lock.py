from __future__ import annotations

"""Build and validate the preregistered analysis lock for the style study."""

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


from experiments.shared.paths import RESEARCH_ROOT
from workflows.author_style_meter_contract import CURRENT_SCORER_ID


REPO_ROOT = RESEARCH_ROOT
DEFAULT_EXPERIMENT_ROOT = (
    REPO_ROOT / "generated/style_research/style_transfer_experiments"
)
LOCK_ID = "style_transfer_pre_style_analysis.v1"

SOURCE_PATHS = (
    "experiments/iteration1/style_analysis_lock.py",
    "experiments/iteration1/build_style_analysis_lock.py",
    "experiments/iteration1/verify_style_analysis_gates.py",
    "experiments/iteration1/run_style_transfer_generation.py",
    "experiments/iteration1/evaluate_style_transfer_methods.py",
    "experiments/iteration1/style_experiment_provenance.py",
    "experiments/iteration1/style_experiment_decisions.py",
    "experiments/iteration1/build_style_semantic_judgments.py",
    "experiments/iteration1/select_style_transfer_promotions.py",
    "experiments/iteration1/build_style_refinement_contract.py",
    "experiments/iteration1/prepare_style_final_validation.py",
    "experiments/iteration1/style_transfer_payloads.py",
    "experiments/iteration1/style_transfer_research.py",
    "experiments/iteration1/calibrate_style_meter.py",
    "workflows/benchmark_author_style_supervised.py",
)

EXPERIMENT_ARTIFACT_PATHS = (
    "protocols/evaluation_protocol.v1.json",
    "protocols/model_run_config.v1.json",
    "prompts/english_semantic_source.v1.md",
    "prompts/english_source_qa.v1.md",
    "prompts/english_source_repair.v1.md",
    "prompts/neutral_translation.v1.md",
    "prompts/style_transfer_method.v1.md",
    "prompts/style_transfer_critique.v1.md",
    "schemas/english_semantic_source_output.v1.schema.json",
    "schemas/english_source_qa_output.v1.schema.json",
    "schemas/english_source_repair_output.v1.schema.json",
    "schemas/neutral_translation_output.v1.schema.json",
    "schemas/style_transfer_output.v1.schema.json",
    "schemas/style_transfer_critique_output.v1.schema.json",
    "method_registry/style_methods.v1.json",
    "method_assets/style_transfer_payloads.v1.lock.json",
    "calibration/style_meter_scores.v1.jsonl",
    "calibration/style_meter_threshold.v1.json",
    f"scorers/{CURRENT_SCORER_ID}/manifest.json",
    f"scorers/{CURRENT_SCORER_ID}/scorer_config.json",
    f"scorers/{CURRENT_SCORER_ID}/masking_config.json",
    f"scorers/{CURRENT_SCORER_ID}/labels.json",
    f"scorers/{CURRENT_SCORER_ID}/classifier.joblib",
    f"scorers/{CURRENT_SCORER_ID}/vectorizer.joblib",
)

REPO_ARTIFACT_PATHS = (
    "datasets/unmasked/chunks.clean.jsonl",
    "datasets/masked/chunks.entity_masked_v3.jsonl",
    "datasets/masked/mask_terms.json",
    "generated/style_research/corpus/splits.json",
    "uv.lock",
)


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


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def resolve_recorded_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise ValueError(f"Missing {label}: {display_path(path)}")
    return path


def current_source_bindings(experiment_root: Path | None = None) -> dict[str, str]:
    bindings = {
        value: file_sha256(_require_file(REPO_ROOT / value, "analysis source"))
        for value in SOURCE_PATHS
    }
    if experiment_root is not None:
        protocol_path = experiment_root.resolve() / "protocols/evaluation_protocol.v1.json"
        if protocol_path.is_file():
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            iteration_bindings: dict[str, str] = {}
            for key, value in protocol.items():
                if not str(key).startswith("iteration") or not isinstance(value, dict):
                    continue
                source_bindings = value.get("source_bindings", {})
                if isinstance(source_bindings, dict):
                    iteration_bindings.update(
                        {
                            str(path): str(digest)
                            for path, digest in source_bindings.items()
                        }
                    )
            for path_text, expected in iteration_bindings.items():
                path = _require_file(REPO_ROOT / path_text, "iteration source")
                observed = file_sha256(path)
                if observed != expected:
                    raise ValueError(
                        f"Iteration source binding mismatch: {path_text}"
                    )
                bindings[str(path_text)] = observed
    return dict(sorted(bindings.items()))


def experiment_artifact_paths(experiment_root: Path) -> list[Path]:
    root = experiment_root.resolve()
    paths = [root / value for value in EXPERIMENT_ARTIFACT_PATHS]
    preregistrations = sorted((root / "protocols").glob("*preregistration*.json"))
    sample_sets = {
        str(value.get("sample_set"))
        for path in preregistrations
        for value in [json.loads(path.read_text(encoding="utf-8"))]
        if isinstance(value, dict) and isinstance(value.get("sample_set"), str)
    }
    if len(sample_sets) > 1:
        raise ValueError(f"Multiple preregistered sample sets: {sorted(sample_sets)}")
    sample_set = next(iter(sample_sets), "development_proxy_v1")
    for suffix in (
        "runner_manifest.jsonl",
        "evaluator_allocation.jsonl",
        "hidden_targets.jsonl",
        "method_evaluation_ids.json",
        "summary.json",
    ):
        paths.append(root / "sample_sets" / f"{sample_set}.{suffix}")
    paths.extend(sorted((root / "sample_sets").glob(f"{sample_set}.*_ids.json")))
    paths.extend(sorted((root / "prompts").glob("*.md")))
    paths.extend(sorted((root / "schemas").glob("*.json")))
    paths.extend(preregistrations)
    paths.extend(sorted((root / "protocols").glob("*cohort_amendment*.json")))
    paths.extend(sorted((root / "protocols").glob("*execution_amendment*.json")))
    paths.extend(sorted((root / "protocols").glob("*confirmation_independence*.json")))
    paths.extend(sorted((root / "audits").glob("english_source_adjudication.*.json")))
    paths.extend(sorted((root / "calibration").glob("*sensitivity*.json")))
    paths.extend(sorted((root / "method_registry/methods").glob("*.json")))
    asset_lock = root / "method_assets/style_transfer_payloads.v1.lock.json"
    if asset_lock.is_file():
        lock = json.loads(asset_lock.read_text(encoding="utf-8"))
        asset_path = lock.get("asset_path")
        if isinstance(asset_path, str):
            paths.append(root / asset_path)
    unique = {path.resolve(): path for path in paths}
    return [unique[key] for key in sorted(unique, key=str)]


def current_artifact_bindings(experiment_root: Path) -> dict[str, str]:
    root = experiment_root.resolve()
    bindings = {
        display_path(path): file_sha256(_require_file(path, "frozen artifact"))
        for path in experiment_artifact_paths(root)
    }
    bindings.update(
        {
            value: file_sha256(_require_file(REPO_ROOT / value, "frozen artifact"))
            for value in REPO_ARTIFACT_PATHS
        }
    )
    return dict(sorted(bindings.items()))


def style_output_inventory(experiment_root: Path) -> list[str]:
    runs_root = experiment_root.resolve() / "runs"
    if not runs_root.exists():
        return []
    paths: set[str] = set()
    for pattern in (
        "**/method_outputs/**/*.json",
        "**/style_critiques/**/*.json",
        "**/ledgers/style_transfer.*.jsonl",
        "**/ledgers/style_critique.*.jsonl",
    ):
        for path in runs_root.glob(pattern):
            if path.is_file():
                paths.add(display_path(path))
    return sorted(paths)


def superseded_style_output_inventory(experiment_root: Path) -> list[dict[str, str]]:
    root = experiment_root.resolve()
    amendments = sorted((root / "protocols").glob("*execution_amendment*.json"))
    if not amendments:
        return []
    observed: list[dict[str, str]] = []
    seen: set[str] = set()
    for amendment_path in amendments:
        amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
        if amendment.get("status") != "frozen_before_outcome_evaluation":
            raise ValueError(
                f"Execution amendment is not frozen: {display_path(amendment_path)}"
            )
        entries = amendment.get("superseded_output_inventory")
        if not isinstance(entries, list):
            raise ValueError(
                f"Execution amendment lacks inventory: {display_path(amendment_path)}"
            )
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Invalid execution-amendment inventory entry")
            path_text = entry.get("path")
            expected = entry.get("sha256")
            if not isinstance(path_text, str) or not isinstance(expected, str):
                raise ValueError("Execution-amendment inventory entry is incomplete")
            if path_text in seen:
                raise ValueError(f"Duplicate superseded output path: {path_text}")
            path = resolve_recorded_path(path_text).resolve()
            try:
                path.relative_to(root / "runs")
            except ValueError as exc:
                raise ValueError(
                    f"Superseded output is outside the iteration runs: {path_text}"
                ) from exc
            actual = file_sha256(_require_file(path, "superseded pilot output"))
            if actual != expected:
                raise ValueError(f"Superseded pilot output hash mismatch: {path_text}")
            observed.append({"path": path_text, "sha256": actual})
            seen.add(path_text)
    return sorted(observed, key=lambda entry: entry["path"])


def lock_content_sha256(payload: Mapping[str, Any]) -> str:
    content = dict(payload)
    content.pop("content_sha256", None)
    return sha256_text(canonical_json(content))


def build_lock_payload(experiment_root: Path) -> dict[str, Any]:
    outputs = style_output_inventory(experiment_root)
    superseded = superseded_style_output_inventory(experiment_root)
    superseded_paths = {entry["path"] for entry in superseded}
    active_outputs = sorted(set(outputs) - superseded_paths)
    style_scope_count = len(set(outputs) & superseded_paths)
    auxiliary_count = len(superseded_paths - set(outputs))
    if active_outputs:
        raise ValueError(
            "Cannot create a pre-style analysis lock after style outcomes exist: "
            + ", ".join(active_outputs[:5])
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "lock_id": LOCK_ID,
        "status": "analysis_preregistered_before_style_generation",
        "experiment_root": display_path(experiment_root),
        "source_bindings": current_source_bindings(experiment_root),
        "artifact_bindings": current_artifact_bindings(experiment_root),
        "style_output_inventory": {
            "count_before_lock": 0,
            "paths": [],
            "scope": [
                "runs/**/method_outputs/**/*.json",
                "runs/**/style_critiques/**/*.json",
                "runs/**/ledgers/style_transfer.*.jsonl",
                "runs/**/ledgers/style_critique.*.jsonl",
            ],
        },
        "superseded_execution_pilot_inventory": {
            "count": len(superseded),
            "style_output_or_ledger_count": style_scope_count,
            "auxiliary_provenance_artifact_count": auxiliary_count,
            "entries": superseded,
            "claim_scope": (
                "Operationally failed pilot run artifacts are retained for provenance "
                "but excluded from generation, scoring, selection, and promotion."
            ),
        },
        "analysis_change_policy": (
            "Any bound source or artifact change invalidates this lock and requires "
            "a new study before outcome generation."
        ),
    }
    payload["content_sha256"] = lock_content_sha256(payload)
    return payload


def validate_analysis_lock(path: Path, experiment_root: Path) -> dict[str, Any]:
    resolved = _require_file(path.resolve(), "pre-style analysis lock")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Pre-style analysis lock must contain a JSON object")
    errors: list[str] = []
    if value.get("schema_version") != 1:
        errors.append("schema_version_mismatch")
    if value.get("lock_id") != LOCK_ID:
        errors.append("lock_id_mismatch")
    if value.get("status") != "analysis_preregistered_before_style_generation":
        errors.append("status_mismatch")
    expected_content = lock_content_sha256(value)
    if value.get("content_sha256") != expected_content:
        errors.append("content_sha256_mismatch")
    if value.get("source_bindings") != current_source_bindings(experiment_root):
        errors.append("source_bindings_mismatch")
    if value.get("artifact_bindings") != current_artifact_bindings(experiment_root):
        errors.append("artifact_bindings_mismatch")
    inventory = value.get("style_output_inventory", {})
    if not isinstance(inventory, dict) or inventory.get("count_before_lock") != 0:
        errors.append("pre_lock_style_inventory_not_zero")
    if inventory.get("paths") != []:
        errors.append("pre_lock_style_inventory_paths_not_empty")
    superseded = value.get("superseded_execution_pilot_inventory", {})
    current_superseded = superseded_style_output_inventory(experiment_root)
    if not isinstance(superseded, dict):
        errors.append("superseded_execution_pilot_inventory_missing")
    else:
        if superseded.get("count") != len(current_superseded):
            errors.append("superseded_execution_pilot_count_mismatch")
        if superseded.get("entries") != current_superseded:
            errors.append("superseded_execution_pilot_entries_mismatch")
    if errors:
        raise ValueError("Invalid pre-style analysis lock: " + ", ".join(errors))
    return {
        "status": "valid",
        "lock_id": LOCK_ID,
        "path": display_path(resolved),
        "sha256": file_sha256(resolved),
        "content_sha256": value["content_sha256"],
    }


def require_matching_analysis_binding(
    artifact: Mapping[str, Any], binding: Mapping[str, Any], *, label: str
) -> None:
    recorded = artifact.get("analysis_lock")
    if not isinstance(recorded, Mapping):
        raise ValueError(f"{label} has no pre-style analysis-lock binding")
    for field in ("lock_id", "path", "sha256", "content_sha256"):
        if recorded.get(field) != binding.get(field):
            raise ValueError(f"{label} analysis-lock {field} mismatch")
