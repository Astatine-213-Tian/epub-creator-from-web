#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from experiments.iteration5.lora.prepare_lora_paired_reconstruction_smoke_v1 import (
    build_examples,
)


MLX_LM_VERSION = "0.31.3"
MODEL_ID = "Qwen/Qwen3-4B-MLX-4bit"
SEED = 20260718
MAX_SEQUENCE_LENGTH = 4096
SMOKE_ROOT = Path(
    "generated/style_research/style_transfer_experiments/iterations/"
    "lora_paired_reconstruction_v1/smoke_data_v1"
)
RUN_ROOT = Path(
    "generated/style_research/style_transfer_experiments/iterations/"
    "lora_paired_reconstruction_v1/runs/qwen3_4b_mlx_smoke_v1"
)
SOURCE_MANIFEST = SMOKE_ROOT / "manifest.json"
PRIVATE_RECORDS = SMOKE_ROOT / "paired_records.private.jsonl"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def validate_source_manifest() -> dict[str, Any]:
    manifest = read_json(SOURCE_MANIFEST)
    if manifest["status"] != "infrastructure_smoke_only_not_efficacy_evidence":
        raise ValueError("source data is not explicitly limited to infrastructure smoke use")
    if manifest["manifest_id"] != sha256_text(
        canonical({key: value for key, value in manifest.items() if key != "manifest_id"})
    ):
        raise ValueError("source manifest ID no longer validates")
    for path_text, expected in manifest["output_hashes"].items():
        path = Path(path_text)
        if not path.exists() or sha256_file(path) != expected:
            raise ValueError(f"source output hash mismatch: {path}")
    return manifest


def blocked_examples(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        example
        for example in build_examples(record, augment=True)
        if example["metadata"]["unit"] == "nonoverlapping_contiguous_block"
    ]


def mlx_rows(examples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"messages": example["messages"]} for example in examples]


def write_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                f'model: "{MODEL_ID}"',
                "train: true",
                "test: true",
                "fine_tune_type: lora",
                "optimizer: adam",
                "mask_prompt: true",
                "num_layers: 8",
                "batch_size: 1",
                "iters: 120",
                "val_batches: -1",
                "test_batches: -1",
                "learning_rate: 1.0e-5",
                "steps_per_report: 5",
                "steps_per_eval: 20",
                "grad_accumulation_steps: 2",
                "save_every: 40",
                f"max_seq_length: {MAX_SEQUENCE_LENGTH}",
                "grad_checkpoint: true",
                f"seed: {SEED}",
                f'data: "{RUN_ROOT / "mlx_data"}"',
                f'adapter_path: "{RUN_ROOT / "adapters"}"',
                "lora_parameters:",
                "  rank: 8",
                "  dropout: 0.0",
                "  scale: 20.0",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    source_manifest = validate_source_manifest()
    if RUN_ROOT.exists() and any(RUN_ROOT.iterdir()):
        raise ValueError(f"LoRA smoke run already prepared: {RUN_ROOT}")
    data_root = RUN_ROOT / "mlx_data"
    data_root.mkdir(parents=True, exist_ok=False)

    records = read_jsonl(PRIVATE_RECORDS)
    by_role = {
        role: [record for record in records if record["role"] == role]
        for role in ("fit", "validation", "internal_test")
    }
    if [len(by_role[role]) for role in by_role] != [12, 3, 3]:
        raise ValueError("unexpected book-disjoint smoke split geometry")

    fit_examples = [
        example
        for record in by_role["fit"]
        for example in blocked_examples(record)
    ]
    valid_examples = [
        example
        for record in by_role["validation"]
        for example in blocked_examples(record)
    ]
    test_examples = [
        example
        for record in by_role["internal_test"]
        for example in blocked_examples(record)
    ]
    splits = {
        "train": mlx_rows(fit_examples),
        "valid": mlx_rows(valid_examples),
        "test": mlx_rows(test_examples),
    }
    for split, rows in splits.items():
        if not rows:
            raise ValueError(f"empty MLX split: {split}")
        write_jsonl(data_root / f"{split}.jsonl", rows)

    config_path = RUN_ROOT / "train_config.yaml"
    write_config(config_path)
    command = (
        f"uvx --from mlx-lm=={MLX_LM_VERSION} mlx_lm.lora "
        f"--config {config_path}"
    )
    output_hashes = {
        str(path): sha256_file(path)
        for path in sorted([*data_root.glob("*.jsonl"), config_path])
    }
    preregistration = {
        "schema_version": 1,
        "experiment_id": "qwen3-4b-mlx-lora-paired-reconstruction-smoke-v1",
        "status": "locked_infrastructure_smoke_not_efficacy_experiment",
        "source_dataset_id": source_manifest["dataset_id"],
        "source_manifest_id": source_manifest["manifest_id"],
        "model": {
            "base_model": MODEL_ID,
            "quantization": "prequantized_4bit_base",
            "mlx_lm_version": MLX_LM_VERSION,
            "fine_tune_type": "lora_on_quantized_base",
        },
        "split": {
            "unit": "book",
            "book_disjoint": True,
            "train_books": len(by_role["fit"]),
            "validation_books": len(by_role["validation"]),
            "internal_test_books": len(by_role["internal_test"]),
            "train_examples": len(splits["train"]),
            "validation_examples": len(splits["valid"]),
            "internal_test_examples": len(splits["test"]),
            "construction": (
                "deterministic nonoverlapping contiguous blocks from every accepted "
                "passage; no full-passage duplicates"
            ),
        },
        "training": {
            "assistant_only_loss": True,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "num_layers": 8,
            "rank": 8,
            "scale": 20.0,
            "dropout": 0.0,
            "optimizer": "adam",
            "learning_rate": 1e-5,
            "iterations": 120,
            "gradient_accumulation_steps": 2,
            "seed": SEED,
        },
        "success_criteria": {
            "required": [
                "training exits successfully without out-of-memory error",
                "finite train, validation, and internal-test losses are produced",
                "adapter artifacts are written",
                "training loss decreases from the first to final reported window",
            ],
            "forbidden_claims": [
                "author-style transfer efficacy",
                "production readiness",
                "superiority to prompt baselines",
                "generalization beyond the smoke passages",
            ],
        },
        "decision_rule": (
            "A passing smoke permits serious paired-data construction and a future "
            "multi-seed LoRA benchmark. It does not permit method selection."
        ),
        "exact_command": command,
        "source_hashes": {
            str(SOURCE_MANIFEST): sha256_file(SOURCE_MANIFEST),
            str(PRIVATE_RECORDS): sha256_file(PRIVATE_RECORDS),
            str(Path(__file__)): sha256_file(Path(__file__)),
        },
        "output_hashes": output_hashes,
    }
    preregistration["preregistration_id"] = sha256_text(canonical(preregistration))
    write_json(RUN_ROOT / "preregistration.json", preregistration)
    print(json.dumps(preregistration, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
