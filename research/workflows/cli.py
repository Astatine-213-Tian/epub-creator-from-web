from __future__ import annotations

"""Small command surface for maintained research workflows."""

import runpy
import subprocess
import sys
from pathlib import Path


RESEARCH_ROOT = Path(__file__).resolve().parents[1]

COMMANDS = {
    "corpus-build": (
        "workflows.audit_style_dataset",
        "Clean raw books and rebuild book-level splits and chunk views.",
    ),
    "corpus-expand": (
        "workflows.expand_jjwxc_author_dataset",
        "Discover and ingest ranked JJWXC authors.",
    ),
    "mask-audit": (
        "workflows.mask_quality_report",
        "Render masking-quality samples and statistics.",
    ),
    "mask-artifact-audit": (
        "workflows.audit_mask_artifacts",
        "Test whether mask tokens or run lengths inflate authorship accuracy.",
    ),
    "authorship-supervised": (
        "workflows.benchmark_author_style_supervised",
        "Run the canonical classifier on the current normalized corpus.",
    ),
    "authorship-report": (
        "workflows.report_author_style_meter",
        "Render the canonical authorship-meter report artifacts.",
    ),
    "interpretable-profiles": (
        "workflows.analyze_interpretable_author_profiles",
        "Build book-weighted author profile evidence and publication figures.",
    ),
    "author-drift": (
        "workflows.analyze_author_style_drift",
        "Measure target-author style drift across years and broad settings.",
    ),
    "export-production": (
        "workflows.export_production_style_transfer_assets",
        "Export the minimized hash-locked production style bundle.",
    ),
    "test": (
        "tests.run_research_tests",
        "Run every portable retained research test.",
    ),
}

QUICK_TEST_MODULES = (
    "tests.test_author_classifier_corpus_contract",
    "tests.test_author_style_report",
    "tests.test_dataset_cleaning",
    "tests.test_dataset_manifest_paths",
    "tests.test_author_style_drift",
    "tests.test_interpretable_author_profiles",
    "tests.test_mask_artifact_ablation",
    "tests.experiments.validation.test_apply_content_plan_combined_to_translation_run",
    "tests.experiments.validation.test_merge_semantic_fallback_repairs",
    "tests.experiments.validation.test_evaluate_method4_application",
)


def print_help() -> None:
    print("usage: author-style-research <command> [arguments]")
    print()
    print("Maintained commands:")
    width = max(len(command) for command in (*COMMANDS, "verify"))
    for command, (_module, description) in COMMANDS.items():
        print(f"  {command:<{width}}  {description}")
    print(f"  {'verify':<{width}}  Rebuild the current corpus and run quick tests.")
    print()
    print("Use '<command> --help' for workflow-specific options.")


def run_module(module: str, arguments: list[str]) -> None:
    sys.argv = [module, *arguments]
    runpy.run_module(module, run_name="__main__")


def run_verify(arguments: list[str]) -> None:
    if arguments:
        raise SystemExit("verify does not accept additional arguments")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "workflows.audit_style_dataset",
            "--stage",
            "all",
        ],
        cwd=RESEARCH_ROOT,
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "unittest", *QUICK_TEST_MODULES],
        cwd=RESEARCH_ROOT,
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "workflows.verify_current_research"],
        cwd=RESEARCH_ROOT,
        check=True,
    )


def main() -> None:
    arguments = sys.argv[1:]
    if not arguments or arguments[0] in {"-h", "--help"}:
        print_help()
        return

    command, passthrough = arguments[0], arguments[1:]
    if command == "verify":
        run_verify(passthrough)
        return
    selected = COMMANDS.get(command)
    if selected is None:
        print_help()
        raise SystemExit(f"unknown research command: {command}")
    run_module(selected[0], passthrough)


if __name__ == "__main__":
    main()
