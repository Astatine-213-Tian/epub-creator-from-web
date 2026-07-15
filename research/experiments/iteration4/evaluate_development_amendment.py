#!/usr/bin/env python3
from __future__ import annotations

"""Run the frozen Iteration-4 evaluator with one validation-only pilot fix.

The shared evaluator predates the registered development-pilot admission stage.
It therefore rejects otherwise valid development artifacts solely because its
allowed-stage table omits ``preregistered_development_pilot``.  This wrapper
removes only that exact error when the ledger independently proves the frozen
development admission.  All scoring, fidelity, selection, and lock checks stay
inside the frozen evaluator.
"""

import sys
from pathlib import Path
from typing import Any


from experiments.shared.paths import RESEARCH_ROOT


REPO_ROOT = RESEARCH_ROOT
ITERATION4_PATH = Path(__file__).with_name("evaluate_style_transfer_methods.py")
EXPECTED_SELECTION_ID = "development_v1"
EXPECTED_ADMISSION_STAGE = "preregistered_development_pilot"


def _admission_error(sample_id: str) -> str:
    return f"{sample_id}:execution_admission_stage_mismatch"


def _has_exact_development_admission(
    row: dict[str, Any] | None,
    *,
    selection_sha256: str,
    sample_count: int,
) -> bool:
    if not isinstance(row, dict):
        return False
    admission = row.get("execution_admission")
    if not isinstance(admission, dict):
        return False
    return (
        admission.get("admission_stage") == EXPECTED_ADMISSION_STAGE
        and admission.get("selection_id") == EXPECTED_SELECTION_ID
        and admission.get("official_efficacy_evidence") is False
        and row.get("execution_selection_id") == EXPECTED_SELECTION_ID
        and row.get("execution_selection_sha256") == selection_sha256
        and row.get("execution_selection_sample_count") == sample_count
    )


def install_development_admission_amendment(base) -> None:
    original = base.combination_provenance

    def combination_provenance(**kwargs):
        provenance, errors, config = original(**kwargs)
        contract = kwargs["contract"]
        selection = contract.get("selection", {})
        if selection.get("selection_id") != EXPECTED_SELECTION_ID:
            return provenance, errors, config

        method = kwargs["method"]
        if method.get("method_id") == "neutral_only":
            return provenance, errors, config

        method_ids = tuple(contract["method_ids"])
        root = base.run_root(
            kwargs["experiment_root"], kwargs["sample_set"], kwargs["run_id"]
        )
        stem = f"style_transfer.{method['method_id']}.{kwargs['intensity']}"
        ledger, ledger_errors = base.load_success_ledger(
            root / "ledgers" / f"{stem}.jsonl"
        )
        if ledger_errors:
            return provenance, errors, config

        removable: set[str] = set()
        for sample_id in method_ids:
            marker = _admission_error(sample_id)
            if marker not in errors:
                continue
            if _has_exact_development_admission(
                ledger.get(sample_id),
                selection_sha256=str(selection.get("sha256", "")),
                sample_count=len(method_ids),
            ):
                removable.add(marker)
        return provenance, [error for error in errors if error not in removable], config

    base.combination_provenance = combination_provenance


def main() -> None:
    iteration4 = _load_module("iteration4_development_evaluator", ITERATION4_PATH)
    root = iteration4.requested_experiment_root()
    iteration4.verify_sources(root)
    payload = iteration4.load_module(
        "iteration4_development_payloads", iteration4.PAYLOAD_PATH
    )
    base = iteration4.load_module(
        "iteration4_development_evaluation_base", iteration4.BASE_PATH
    )
    base._iteration4_original_load_sample_contract = base.load_sample_contract
    base.load_sample_contract = (
        lambda experiment_root, sample_set: iteration4.load_iteration4_sample_contract(
            base, experiment_root, sample_set
        )
    )
    base._PAYLOAD_BUILDER = payload
    install_development_admission_amendment(base)
    iteration4.install_derived_method_support(base, payload)
    base.main()


def _load_module(name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    main()
