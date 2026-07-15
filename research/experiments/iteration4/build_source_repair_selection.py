#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


from experiments.shared.paths import RESEARCH_ROOT


REPO_ROOT = RESEARCH_ROOT


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the rejected subset for one English-source repair round."
    )
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--sample-set", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--parent-selection", type=Path, required=True)
    parser.add_argument("--repair-round", type=int, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repair_round < 1:
        raise SystemExit("--repair-round must be at least 1")
    root = args.experiment_root.expanduser().resolve()
    parent_path = args.parent_selection.expanduser().resolve()
    parent = read_json(parent_path)
    parent_ids = [str(value) for value in parent.get("sample_ids", [])]
    if not parent_ids or len(parent_ids) != len(set(parent_ids)):
        raise RuntimeError("Parent selection is empty or duplicated")
    run_root = root / "runs" / args.sample_set / args.run_id
    if args.repair_round == 1:
        qa_root = run_root / "english_source_qa"
        qa_stage = "english_source_qa"
    else:
        previous = args.repair_round - 1
        qa_root = run_root / "english_source_repair_qa" / f"round_{previous:02d}"
        qa_stage = f"english_source_repair_qa.round_{previous:02d}"
    rejected: list[str] = []
    for sample_id in parent_ids:
        qa_path = qa_root / f"{sample_id}.json"
        if not qa_path.exists():
            raise RuntimeError(f"Missing QA artifact: {qa_path}")
        artifact = read_json(qa_path)
        approved = artifact.get("approved_for_neutral_translation")
        if approved is not artifact.get("result", {}).get("approved"):
            raise RuntimeError(f"QA approval fields disagree: {sample_id}")
        if approved is not True:
            rejected.append(sample_id)
    selection_id = f"english_repair_round_{args.repair_round:02d}"
    output = (
        args.output.expanduser().resolve()
        if args.output
        else root
        / "sample_sets"
        / f"{args.sample_set}.{selection_id}_ids.json"
    )
    payload = {
        "schema_version": 1,
        "selection_id": selection_id,
        "sample_set": args.sample_set,
        "run_id": args.run_id,
        "repair_round": args.repair_round,
        "qa_stage": qa_stage,
        "parent_selection_path": str(parent_path.relative_to(REPO_ROOT)),
        "parent_selection_sha256": file_sha256(parent_path),
        "parent_sample_count": len(parent_ids),
        "sample_count": len(rejected),
        "sample_ids": sorted(rejected),
    }
    write_json(output, payload)
    print(json.dumps({**payload, "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
