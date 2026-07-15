#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.validation.meter.benchmark_content_resistant_meter import TARGET_AUTHOR, load_records
from experiments.validation.meter.build_cr_fysm_v3 import canonical_lock_id, runtime_versions, sha256_file
from experiments.validation.meter.build_cr_fysm_v4 import qualification_records
ROOT = Path(
    "generated/style_research/style_transfer_experiments/iterations/"
    "content_resistant_v1/cr_fysm_v4_postclean_replication_v2"
)
OLD_ROOT = ROOT.parent / "cr_fysm_v4"
OLD_PREREGISTRATION = OLD_ROOT / "preregistration.v4.json"
OLD_RESULTS = OLD_ROOT / "results.v4.json"
BASE_PATHS = {
    "active_masked_dataset": Path("datasets/masked/chunks.entity_masked_v3.jsonl"),
    "active_clean_dataset": Path("datasets/unmasked/chunks.clean.jsonl"),
    "active_manifest": Path("datasets/dataset_manifest.json"),
    "active_mask_plan": Path("datasets/masked/mask_terms.json"),
    "cleaning_report": Path("generated/style_research/corpus/cleaning_report.json"),
    "chunk_report": Path("generated/style_research/corpus/chunk_report.json"),
    "corpus_validation": Path("generated/style_research/corpus/rebuild_validation.json"),
    "repair_provenance": Path("datasets/provenance/source_quality_repair.v1.json"),
    "old_preregistration": OLD_PREREGISTRATION,
    "old_results": OLD_RESULTS,
    "old_v3_partition": ROOT.parent / "cr_fysm_v3/partition.v3.json",
}
OUTPUT = ROOT / "preregistration.json"
SUPERSEDED = Path(
    "generated/style_research/style_transfer_experiments/iterations/"
    "content_resistant_v1/cr_fysm_v4_postclean_replication/preregistration.json"
)
RETIRED_BOOKS = {
    ("妄鸦", "晚来天欲雪"),
    ("妄鸦", "能饮一杯无"),
}
PROMOTED_BOOKS = {("妄鸦", "无限练习生"), ("妄鸦", "悬疑片导演")}
DEPENDENCY_PATHS = {
    "preregister_v2": Path(__file__),
    "runner_v2": Path(__file__).with_name("rescore_cr_fysm_v4_postclean_v2.py"),
    "v4_builder": Path(__file__).with_name("build_cr_fysm_v4.py"),
    "v4_development": Path(__file__).with_name("develop_cr_fysm_v4.py"),
    "v3_builder": Path(__file__).with_name("build_cr_fysm_v3.py"),
    "benchmark": Path(__file__).with_name("benchmark_content_resistant_meter.py"),
    "shared_feature_definitions": Path("workflows/benchmark_author_style.py"),
    "cleanup": Path("workflows/audit_style_dataset.py"),
    "corpus_validator": Path(__file__).with_name("validate_rebuilt_corpus.py"),
}
HISTORICAL_DEPENDENCY_KEYS = {
    "v4_builder": "script:builder",
    "v4_development": "script:development",
    "v3_builder": "script:v3_builder",
    "benchmark": "script:benchmark",
    "shared_feature_definitions": "script:shared_feature_definitions",
}


def stable_hash(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_repair() -> dict[str, Any]:
    provenance = read_json(BASE_PATHS["repair_provenance"])
    removed = {row["title"] for row in provenance["corrupt_sources"]}
    promoted = {row["title"] for row in provenance["promotions"]}
    if removed != {"晚来天欲雪", "能饮一杯无"}:
        raise ValueError(f"unexpected repair removals: {sorted(removed)}")
    if promoted != {"无限练习生", "悬疑片导演"}:
        raise ValueError(f"unexpected repair promotions: {sorted(promoted)}")
    for row in provenance["promotions"]:
        path = Path(row["path"])
        validation = read_json(Path(row["validation"]))
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"promoted source hash changed: {path}")
        if validation.get("status") != "pass" or validation.get("sha256") != row["sha256"]:
            raise ValueError(f"promoted source validation mismatch: {path}")
    return provenance


def resolved_evaluation_partition(old: dict[str, Any]) -> dict[str, Any]:
    boundary = int(old["model_contract"]["boundary_chunks_excluded"])
    records = load_records(BASE_PATHS["active_masked_dataset"], boundary=boundary)
    selected = qualification_records(records, old["partition"])
    counts = Counter(
        (
            "target" if row.author == TARGET_AUTHOR else "comparison",
            row.author,
            row.title,
            row.split,
        )
        for row in selected
    )
    books = [
        {
            "role": role,
            "author": author,
            "title": title,
            "split": split,
            "rows": rows,
        }
        for (role, author, title, split), rows in sorted(counts.items())
    ]
    active_books = {(row["author"], row["title"]) for row in books}
    if active_books & RETIRED_BOOKS:
        raise ValueError("retired corrupted books remain in the resolved evaluation partition")
    if not PROMOTED_BOOKS <= active_books:
        raise ValueError("both promoted replacement books must enter the evaluation partition")
    target_books = {
        row["title"] for row in books if row["role"] == "target"
    }
    comparison_authors = {
        row["author"] for row in books if row["role"] == "comparison"
    }
    if target_books != set(old["partition"]["target_books"]):
        raise ValueError("resolved target-book membership differs from historical v4")
    if comparison_authors != set(old["partition"]["comparison_authors"]):
        raise ValueError("resolved comparison-author membership differs from historical v4")
    chunk_ids = sorted(row.chunk_id for row in selected)
    return {
        "boundary_chunks_excluded": boundary,
        "rows": len(selected),
        "chunk_id_sha256": stable_hash(chunk_ids),
        "books": books,
        "target_books": sorted(target_books),
        "comparison_authors": sorted(comparison_authors),
        "promoted_books_present": sorted(f"{author}/{title}" for author, title in PROMOTED_BOOKS),
        "retired_books_absent": sorted(f"{author}/{title}" for author, title in RETIRED_BOOKS),
    }


def current_input_hashes(
    paths: dict[str, Path],
    artifact_paths: dict[str, Path],
    dependency_paths: dict[str, Path],
) -> dict[str, str]:
    values = {f"path:{key}": sha256_file(path) for key, path in paths.items()}
    values.update(
        {f"artifact:{key}": sha256_file(path) for key, path in artifact_paths.items()}
    )
    values.update(
        {f"script:{key}": sha256_file(path) for key, path in dependency_paths.items()}
    )
    values["uv.lock"] = sha256_file(Path("uv.lock"))
    return values


def main() -> None:
    if ROOT.exists():
        raise ValueError("post-clean replication v2 already exists")
    if not SUPERSEDED.is_file():
        raise ValueError("missing unopened superseded preregistration")
    paths = dict(BASE_PATHS)
    paths["superseded_preregistration"] = SUPERSEDED
    for name, path in {**paths, **DEPENDENCY_PATHS}.items():
        if not path.is_file():
            raise ValueError(f"missing input {name}: {path}")
    if read_json(paths["corpus_validation"]).get("status") != "pass":
        raise ValueError("rebuilt corpus validation did not pass")

    old = read_json(OLD_PREREGISTRATION)
    old_results = read_json(OLD_RESULTS)
    superseded = read_json(SUPERSEDED)
    if old.get("lock_id") != canonical_lock_id(old):
        raise ValueError("historical v4 lock no longer verifies")
    if superseded.get("lock_id") != canonical_lock_id(superseded):
        raise ValueError("superseded post-clean lock no longer verifies")
    if (SUPERSEDED.parent / "replication_opened.json").exists() or (
        SUPERSEDED.parent / "results.json"
    ).exists():
        raise ValueError("superseded preregistration was opened and cannot be replaced")
    if old_results.get("status") != "statistical_qualification_pass_pending_construct_gates":
        raise ValueError("historical v4 result is not the recorded statistical pass")

    repair = validate_repair()
    manifest = read_json(paths["active_manifest"])
    author_counts = Counter(str(row["author"]) for row in manifest)
    if len(author_counts) != 50 or min(author_counts.values()) < 3:
        raise ValueError("active manifest is not 50 authors with >=3 books each")
    resolved = resolved_evaluation_partition(old)

    artifact_paths = {
        key: Path(value) for key, value in old["source_artifact_paths"].items()
    }
    for key, path in artifact_paths.items():
        if sha256_file(path) != old["input_hashes"][f"artifact:{key}"]:
            raise ValueError(f"frozen model artifact changed: {key}")
    for name, historical_key in HISTORICAL_DEPENDENCY_KEYS.items():
        if sha256_file(DEPENDENCY_PATHS[name]) != old["input_hashes"][historical_key]:
            raise ValueError(f"historical executable dependency changed: {name}")

    input_hashes = current_input_hashes(paths, artifact_paths, DEPENDENCY_PATHS)
    claim_limit = (
        "Retrospective sensitivity replication only. Two replacement 妄鸦 books are score-fresh; "
        "all target books and the other nine comparison authors were scored historically. Passing "
        "shows persistence of the legacy gates under the resolved repaired composition, not fresh "
        "qualification, causal equivalence of replacement books, or production validity."
    )
    payload: dict[str, Any] = {
        "schema_version": 2,
        "experiment_id": "CR-FYSM-v4-postclean-replication-v2",
        "status": "locked_before_any_postclean_score",
        "purpose": "Retrospective sensitivity replication of frozen v4 after source-quality repair.",
        "claim_limit": claim_limit,
        "supersedes": {
            "path": str(SUPERSEDED),
            "sha256": sha256_file(SUPERSEDED),
            "lock_id": superseded["lock_id"],
            "opened": False,
            "reason": (
                "Independent pre-score audit found an incomplete executable hash contract and "
                "no exact resolved repaired-book partition."
            ),
            "audit_agent_id": "019f5e91-e237-7950-9fb2-5bb68bd2dbe5",
            "audit_verdict": "NO-GO",
        },
        "repair_scope": {
            "removed_titles": sorted(row["title"] for row in repair["corrupt_sources"]),
            "promoted_titles": sorted(row["title"] for row in repair["promotions"]),
            "model_refit": False,
            "threshold_retuned": False,
            "feature_contract_changed": False,
        },
        "historical_source": {
            "meter_id": old["meter_id"],
            "lock_id": old["lock_id"],
            "decision_policy": old["decision_policy"],
            "model_contract": old["model_contract"],
            "partition": old["partition"],
            "qualification_gates": old["qualification_gates"],
            "source_artifact_paths": old["source_artifact_paths"],
        },
        "geometry": {
            "manifest_books": len(manifest),
            "manifest_authors": len(author_counts),
            "minimum_books_per_author": min(author_counts.values()),
            "resolved_evaluation_partition": resolved,
        },
        "paths": {**{key: str(path) for key, path in paths.items()}, "output_dir": str(ROOT)},
        "dependency_paths": {key: str(path) for key, path in DEPENDENCY_PATHS.items()},
        "input_hashes": input_hashes,
        "environment": runtime_versions(),
        "outcome_policy": {
            "all_historical_v4_gates_recomputed": True,
            "resolved_partition_recomputed_before_irreversible_marker": True,
            "no_exclusions_or_threshold_changes_after_opening": True,
            "report_old_and_new_metrics_and_every_comparison_author": True,
            "report_every_comparison_book": True,
            "report_all_three_active_wangya_books": True,
            "author_delta_is_composition_change_not_paired_effect": True,
            "failure_is_retained": True,
        },
        "run_command": "uv run python -m experiments.validation.meter.rescore_cr_fysm_v4_postclean_v2",
    }
    payload["lock_id"] = canonical_lock_id(payload)
    ROOT.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"status": payload["status"], "lock_id": payload["lock_id"], "output": str(OUTPUT)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
