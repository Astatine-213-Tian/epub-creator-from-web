#!/usr/bin/env python3
from __future__ import annotations

"""Freeze the exact post-screening intensity and verifier-loop roster."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.iteration1.style_analysis_lock import (
    require_matching_analysis_binding,
    validate_analysis_lock,
)


from experiments.shared.paths import RESEARCH_ROOT


REPO_ROOT = RESEARCH_ROOT
DEFAULT_EXPERIMENT_ROOT = (
    REPO_ROOT / "generated/style_research/style_transfer_experiments"
)
INTENSITY_ORDER = {"none": 0, "light": 1, "medium": 2, "strong": 3}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze exact lighter-intensity and self-critique screening arms."
    )
    parser.add_argument("--provisional-shortlist", type=Path, required=True)
    parser.add_argument("--analysis-lock", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--sample-set", default="development_proxy_v1")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.experiment_root.resolve()
    analysis_lock_binding = validate_analysis_lock(args.analysis_lock.resolve(), root)
    shortlist_path = args.provisional_shortlist.resolve()
    shortlist = load_json(shortlist_path)
    require_matching_analysis_binding(
        shortlist, analysis_lock_binding, label="provisional shortlist"
    )
    if shortlist.get("status") != "provisional_shortlist_frozen":
        raise ValueError("Input is not a frozen provisional shortlist")
    if shortlist.get("sample_set") != args.sample_set:
        raise ValueError("Shortlist sample_set mismatch")
    promoted = shortlist.get("promoted")
    if not isinstance(promoted, list) or not promoted:
        raise ValueError("Provisional shortlist contains no promoted methods")

    selection_path = (
        root / "sample_sets" / f"{args.sample_set}.screening_v1_ids.json"
    )
    registry_path = root / "method_registry/style_methods.v1.json"
    registry = load_json(registry_path)
    methods = {
        str(row.get("id", "")): row
        for row in registry.get("methods", [])
        if isinstance(row, dict)
    }
    combinations: list[dict[str, Any]] = [
        {
            "method_id": "neutral_only",
            "intensity": "none",
            "arm_type": "control",
        }
    ]
    seen = {("neutral_only", "none")}
    for promoted_row in promoted:
        method_id = str(promoted_row.get("method_id", ""))
        intensity = str(promoted_row.get("intensity", ""))
        method = methods.get(method_id)
        if method is None:
            raise ValueError(f"Shortlisted method is absent from registry: {method_id}")
        if intensity not in method.get("intensities", []):
            raise ValueError(f"Shortlisted intensity is absent from registry: {method_id}")
        key = (method_id, intensity)
        if key not in seen:
            combinations.append(
                {
                    "method_id": method_id,
                    "intensity": intensity,
                    "arm_type": "shortlisted_original",
                }
            )
            seen.add(key)
        current_order = INTENSITY_ORDER[intensity]
        lighter = sorted(
            (
                value
                for value in method.get("intensities", [])
                if INTENSITY_ORDER.get(str(value), 99) < current_order
            ),
            key=lambda value: INTENSITY_ORDER[str(value)],
            reverse=True,
        )
        if lighter:
            lighter_key = (method_id, str(lighter[0]))
            if lighter_key not in seen:
                combinations.append(
                    {
                        "method_id": method_id,
                        "intensity": str(lighter[0]),
                        "arm_type": "registered_lighter_intensity",
                        "base_method_id": method_id,
                        "base_intensity": intensity,
                    }
                )
                seen.add(lighter_key)

    top = promoted[0]
    repair_method = methods.get("self_critique_repair")
    if repair_method is None or "light" not in repair_method.get("intensities", []):
        raise ValueError("Registry cannot supply self_critique_repair:light")
    combinations.append(
        {
            "method_id": "self_critique_repair",
            "intensity": "light",
            "arm_type": "derived_verifier_loop",
            "base_method_id": str(top["method_id"]),
            "base_intensity": str(top["intensity"]),
            "requires_frozen_base_critique": True,
        }
    )
    artifact = {
        "schema_version": 1,
        "contract_id": "screening_v1.refinement_contract.v1",
        "status": "refinement_contract_frozen",
        "sample_set": args.sample_set,
        "analysis_lock": analysis_lock_binding,
        "selection_id": "screening_v1",
        "selection_path": display_path(selection_path),
        "selection_sha256": file_sha256(selection_path),
        "provisional_shortlist_path": display_path(shortlist_path),
        "provisional_shortlist_sha256": file_sha256(shortlist_path),
        "method_registry_path": display_path(registry_path),
        "method_registry_sha256": file_sha256(registry_path),
        "rules": {
            "retain_every_shortlisted_original": True,
            "add_at_most_one_closest_registered_lighter_intensity_per_method": True,
            "derive_one_self_critique_repair_from_rank_1": True,
            "self_critique_repair_intensity": "light",
            "unlisted_arms_ineligible": True,
        },
        "evaluation_combinations": combinations,
    }
    output = args.output or (
        root / "promotions/screening_v1.refinement_contract.v1.json"
    )
    output = output.resolve()
    content = json.dumps(artifact, ensure_ascii=False, indent=2) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != content:
        raise FileExistsError(f"Refusing to replace frozen contract: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "combinations": len(combinations),
                "output": display_path(output),
                "sha256": file_sha256(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
