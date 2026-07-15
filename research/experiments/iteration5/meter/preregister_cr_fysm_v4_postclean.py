#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.iteration5.meter.benchmark_content_resistant_meter import TARGET_AUTHOR, load_records
from experiments.iteration5.meter.build_cr_fysm_v3 import (
    canonical_lock_id,
    runtime_versions,
    sha256_file,
)


ROOT = Path(
    "generated/style_research/style_transfer_experiments/iterations/"
    "content_resistant_v1/cr_fysm_v4_postclean_replication"
)
OUTPUT = ROOT / "preregistration.json"
OLD_ROOT = ROOT.parent / "cr_fysm_v4"
OLD_PREREGISTRATION = OLD_ROOT / "preregistration.v4.json"
OLD_RESULTS = OLD_ROOT / "results.v4.json"
PATHS = {
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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_repair() -> dict[str, Any]:
    provenance = read_json(PATHS["repair_provenance"])
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


def validate_geometry(old: dict[str, Any]) -> dict[str, Any]:
    manifest = read_json(PATHS["active_manifest"])
    counts = Counter(str(row["author"]) for row in manifest)
    if len(manifest) != 199 or len(counts) != 50 or min(counts.values()) < 3:
        raise ValueError("active manifest is not 199 books / 50 authors / >=3 books each")
    boundary = int(old["model_contract"]["boundary_chunks_excluded"])
    records = load_records(PATHS["active_masked_dataset"], boundary=boundary)
    partition = old["partition"]
    target_titles = {
        row.title
        for row in records
        if row.author == TARGET_AUTHOR and row.split == "proxy_transfer"
    }
    comparison_authors = {
        row.author
        for row in records
        if row.author in set(partition["comparison_authors"])
    }
    if target_titles != set(partition["target_books"]):
        raise ValueError("target proxy books changed relative to frozen v4 partition")
    if comparison_authors != set(partition["comparison_authors"]):
        raise ValueError("a frozen v4 comparison author is missing after repair")
    return {
        "manifest_books": len(manifest),
        "manifest_authors": len(counts),
        "minimum_books_per_author": min(counts.values()),
        "target_books": sorted(target_titles),
        "comparison_authors": sorted(comparison_authors),
        "retained_masked_rows": len(records),
    }


def main() -> None:
    if OUTPUT.exists() or (ROOT / "replication_opened.json").exists() or (ROOT / "results.json").exists():
        raise ValueError("post-clean replication was already locked or opened")
    for name, path in PATHS.items():
        if not path.is_file():
            raise ValueError(f"missing input {name}: {path}")
    if read_json(PATHS["corpus_validation"]).get("status") != "pass":
        raise ValueError("rebuilt corpus validation did not pass")
    old = read_json(OLD_PREREGISTRATION)
    old_results = read_json(OLD_RESULTS)
    if old.get("lock_id") != canonical_lock_id(old):
        raise ValueError("historical v4 lock no longer verifies")
    if old_results.get("status") != "statistical_qualification_pass_pending_construct_gates":
        raise ValueError("historical v4 result is not the recorded statistical pass")
    repair = validate_repair()
    geometry = validate_geometry(old)
    artifact_paths = {
        key: Path(value) for key, value in old["source_artifact_paths"].items()
    }
    artifact_hashes = {
        key: sha256_file(path) for key, path in artifact_paths.items()
    }
    for key, digest in artifact_hashes.items():
        if digest != old["input_hashes"][f"artifact:{key}"]:
            raise ValueError(f"frozen model artifact changed: {key}")
    input_hashes = {f"path:{key}": sha256_file(path) for key, path in PATHS.items()}
    input_hashes.update({f"artifact:{key}": value for key, value in artifact_hashes.items()})
    input_hashes.update(
        {
            "script:preregister": sha256_file(Path(__file__)),
            "script:runner": sha256_file(Path(__file__).with_name("rescore_cr_fysm_v4_postclean.py")),
            "script:v4_builder": sha256_file(Path(__file__).with_name("build_cr_fysm_v4.py")),
            "script:v3_builder": sha256_file(Path(__file__).with_name("build_cr_fysm_v3.py")),
            "script:benchmark": sha256_file(Path(__file__).with_name("benchmark_content_resistant_meter.py")),
            "script:cleanup": sha256_file(Path("workflows/audit_style_dataset.py")),
            "script:corpus_validator": sha256_file(Path(__file__).with_name("validate_rebuilt_corpus.py")),
            "uv.lock": sha256_file(Path("uv.lock")),
        }
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": "CR-FYSM-v4-postclean-replication",
        "status": "locked_before_any_postclean_score",
        "purpose": "Retrospective sensitivity replication of frozen v4 after source-quality repair.",
        "claim_limit": (
            "Not a new global one-shot qualification: target books and nine comparison authors were "
            "scored historically. The two promoted 妄鸦 books are score-fresh and the model/threshold are frozen."
        ),
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
        "geometry": geometry,
        "paths": {**{key: str(path) for key, path in PATHS.items()}, "output_dir": str(ROOT)},
        "input_hashes": input_hashes,
        "environment": runtime_versions(),
        "outcome_policy": {
            "all_historical_v4_gates_recomputed": True,
            "no_exclusions_or_threshold_changes_after_opening": True,
            "report_old_and_new_metrics_and_every_comparison_author": True,
            "failure_is_retained": True,
        },
        "run_command": "uv run python -m experiments.iteration5.meter.rescore_cr_fysm_v4_postclean",
    }
    payload["lock_id"] = canonical_lock_id(payload)
    ROOT.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "lock_id": payload["lock_id"], "output": str(OUTPUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
