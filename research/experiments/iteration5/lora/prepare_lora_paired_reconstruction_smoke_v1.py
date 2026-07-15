#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


TARGET_AUTHOR = "非天夜翔"
SEED = 20260718
ROOT = Path(
    "generated/style_research/style_transfer_experiments/iterations/"
    "lora_paired_reconstruction_v1/smoke_data_v1"
)
PAIR_ASSET = Path(
    "generated/style_research/style_transfer_experiments/iterations/"
    "full_regeneration_v1/method_assets/style_transfer_payloads.v1/"
    "assets.497a0db8919ccc9cfd97f6a729ff0528953d2d97477e83e07e55c42e4bb994d8.json"
)
ALLOCATION = Path(
    "generated/style_research/style_transfer_experiments/iterations/"
    "aligned_pairs_v1/sample_sets/iteration2_aligned_pair_pool_v1."
    "evaluator_allocation.jsonl"
)
SOURCE_RUN = Path(
    "generated/style_research/style_transfer_experiments/iterations/"
    "aligned_pairs_v1/runs/iteration2_aligned_pair_pool_v1/"
    "aligned_pair_pool_gpt54_v1"
)
ACTIVE_DATASET = Path("datasets/masked/chunks.entity_masked_v3.jsonl")
HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
PLACEHOLDER_RE = re.compile(r"<(?:TERM|NUM|LATIN)>")

VALIDATION_BOOKS = (
    "北城天街",
    "大设定师",
    "破罐子破摔",
)
INTERNAL_TEST_BOOKS = (
    "星盘重启",
    "金牌助理",
    "鹰奴",
)

SYSTEM_PROMPT = """You are a Chinese literary reconstruction model for anonymous Style T.
Rewrite the neutral Chinese using the English source as semantic authority.
Preserve every fact, event order, causality, negation, modality, intensity,
speaker, paragraph ID, dialogue turn, and protected placeholder exactly.
Change wording, syntax, rhythm, discourse flow, and dialogue realization only
where the supplied content supports it. Return only the requested JSON object."""


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


def active_train_books() -> set[str]:
    return {
        str(row["title"])
        for row in read_jsonl(ACTIVE_DATASET)
        if row["author"] == TARGET_AUTHOR and row["split"] == "train"
    }


def approved_english(sample_id: str) -> tuple[list[dict[str, str]], dict[str, str]]:
    candidates: list[tuple[Path, Path, str]] = [
        (
            SOURCE_RUN / "english_semantic_source" / f"{sample_id}.json",
            SOURCE_RUN / "english_source_qa" / f"{sample_id}.json",
            "initial",
        )
    ]
    for round_number in range(1, 4):
        label = f"round_{round_number:02d}"
        candidates.append(
            (
                SOURCE_RUN / "english_source_repair" / label / f"{sample_id}.json",
                SOURCE_RUN / "english_source_repair_qa" / label / f"{sample_id}.json",
                label,
            )
        )
    approved: list[tuple[Path, Path, str]] = []
    for source_path, qa_path, label in candidates:
        if not source_path.exists() or not qa_path.exists():
            continue
        qa = read_json(qa_path)["result"]
        if qa.get("approved") is True:
            approved.append((source_path, qa_path, label))
    if len(approved) != 1:
        raise ValueError(f"expected one approved English lineage for {sample_id}: {approved}")
    source_path, qa_path, label = approved[0]
    paragraphs = read_json(source_path)["result"]["paragraphs"]
    return paragraphs, {
        "lineage": label,
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "qa_path": str(qa_path),
        "qa_sha256": sha256_file(qa_path),
    }


def paragraph_ids(rows: list[dict[str, str]]) -> list[str]:
    return [str(row["id"]) for row in rows]


def placeholder_multiset(rows: list[dict[str, str]], field: str) -> list[str]:
    return sorted(
        token for row in rows for token in PLACEHOLDER_RE.findall(str(row[field]))
    )


def user_content(
    *,
    sample_id: str,
    english: list[dict[str, str]],
    neutral: list[dict[str, str]],
) -> str:
    payload = {
        "sample_id": sample_id,
        "english_source": english,
        "neutral_chinese": neutral,
        "output_contract": {
            "sample_id": sample_id,
            "paragraphs": [{"id": row["id"], "zh": "..."} for row in neutral],
        },
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def assistant_content(
    *, sample_id: str, target: list[dict[str, str]]
) -> str:
    return json.dumps(
        {"sample_id": sample_id, "paragraphs": target},
        ensure_ascii=False,
        sort_keys=True,
    )


def training_example(
    *,
    example_id: str,
    book: str,
    chunk_id: str,
    english: list[dict[str, str]],
    neutral: list[dict[str, str]],
    target: list[dict[str, str]],
    source_pair_id: str,
    unit: str,
) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": user_content(
                    sample_id=example_id, english=english, neutral=neutral
                ),
            },
            {
                "role": "assistant",
                "content": assistant_content(sample_id=example_id, target=target),
            },
        ],
        "metadata": {
            "example_id": example_id,
            "book": book,
            "chunk_id": chunk_id,
            "source_pair_id": source_pair_id,
            "unit": unit,
        },
    }


def build_examples(record: dict[str, Any], *, augment: bool) -> list[dict[str, Any]]:
    english = record["english"]
    neutral = record["neutral"]
    target = record["target"]
    ids = paragraph_ids(target)
    examples = [
        training_example(
            example_id=f"{record['sample_id']}.full",
            book=record["book"],
            chunk_id=record["chunk_id"],
            english=english,
            neutral=neutral,
            target=target,
            source_pair_id=record["pair_id"],
            unit="full_passage",
        )
    ]
    if not augment:
        return examples
    english_by_id = {row["id"]: row for row in english}
    neutral_by_id = {row["id"]: row for row in neutral}
    target_by_id = {row["id"]: row for row in target}
    start = 0
    block_index = 1
    while start < len(ids):
        end = start
        han = 0
        while end < len(ids):
            next_han = len(HAN_RE.findall(target_by_id[ids[end]]["zh"]))
            if end > start and han + next_han > 240:
                break
            han += next_han
            end += 1
        if end == start:
            end += 1
        block_ids = ids[start:end]
        if end < len(ids) and han < 80:
            block_ids.append(ids[end])
            end += 1
        example_id = f"{record['sample_id']}.block{block_index:02d}"
        examples.append(
            training_example(
                example_id=example_id,
                book=record["book"],
                chunk_id=record["chunk_id"],
                english=[english_by_id[value] for value in block_ids],
                neutral=[neutral_by_id[value] for value in block_ids],
                target=[target_by_id[value] for value in block_ids],
                source_pair_id=record["pair_id"],
                unit="nonoverlapping_contiguous_block",
            )
        )
        start = end
        block_index += 1
    return examples


def main() -> None:
    if ROOT.exists():
        if any(ROOT.iterdir()):
            raise ValueError(f"LoRA smoke data already exists: {ROOT}")
    else:
        ROOT.mkdir(parents=True, exist_ok=False)
    active = active_train_books()
    if len(active) != 28:
        raise ValueError(f"expected 28 active target train books, found {len(active)}")
    if set(VALIDATION_BOOKS) & set(INTERNAL_TEST_BOOKS):
        raise ValueError("LoRA validation and internal-test books overlap")
    if not set(VALIDATION_BOOKS + INTERNAL_TEST_BOOKS).issubset(active):
        raise ValueError("LoRA held-out books are not active target train books")
    allocations = {
        row["sample_id"]: row
        for row in read_jsonl(ALLOCATION)
        if row["author"] == TARGET_AUTHOR
    }
    asset = read_json(PAIR_ASSET)["assets"]["aligned_pairs"]
    pairs = asset["pairs"]
    pair_by_sample = {row["source_sample_id"]: row for row in pairs}
    records: list[dict[str, Any]] = []
    placeholder_mismatch_exclusions: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {
        str(PAIR_ASSET): sha256_file(PAIR_ASSET),
        str(ALLOCATION): sha256_file(ALLOCATION),
        str(ACTIVE_DATASET): sha256_file(ACTIVE_DATASET),
        str(Path(__file__)): sha256_file(Path(__file__)),
    }
    for sample_id, allocation in sorted(allocations.items()):
        book = allocation["book_title"]
        if book not in active:
            continue
        pair = pair_by_sample[sample_id]
        neutral = pair["neutral_zh"]
        target = pair["target_style_zh"]
        target_placeholders = placeholder_multiset(target, "zh")
        neutral_placeholders = placeholder_multiset(neutral, "zh")
        if target_placeholders != neutral_placeholders:
            placeholder_mismatch_exclusions.append(
                {
                    "sample_id": sample_id,
                    "book": book,
                    "chunk_id": allocation["chunk_id"],
                    "neutral_placeholders": neutral_placeholders,
                    "target_placeholders": target_placeholders,
                    "reason": "neutral_target_placeholder_multiset_mismatch",
                }
            )
            continue
        english, english_lineage = approved_english(sample_id)
        expected_ids = paragraph_ids(target)
        if paragraph_ids(english) != expected_ids or paragraph_ids(neutral) != expected_ids:
            raise ValueError(f"paragraph alignment differs for {sample_id}")
        source_hashes[english_lineage["source_path"]] = english_lineage["source_sha256"]
        source_hashes[english_lineage["qa_path"]] = english_lineage["qa_sha256"]
        role = (
            "validation"
            if book in VALIDATION_BOOKS
            else "internal_test"
            if book in INTERNAL_TEST_BOOKS
            else "fit"
        )
        records.append(
            {
                "sample_id": sample_id,
                "pair_id": pair["pair_id"],
                "book": book,
                "chunk_id": allocation["chunk_id"],
                "role": role,
                "english": english,
                "neutral": neutral,
                "target": target,
                "english_lineage": english_lineage,
                "target_han": sum(len(HAN_RE.findall(row["zh"])) for row in target),
                "placeholder_count": len(target_placeholders),
            }
        )
    safe_books = {row["book"] for row in records}
    excluded_books = {row["book"] for row in placeholder_mismatch_exclusions}
    if len(records) != 18 or safe_books | excluded_books != active or safe_books & excluded_books:
        raise ValueError("placeholder-safe and excluded books do not partition active train books")
    fit_books = sorted(
        safe_books - set(VALIDATION_BOOKS) - set(INTERNAL_TEST_BOOKS)
    )
    if len(fit_books) != 12:
        raise ValueError(f"expected 12 placeholder-safe fit books, found {len(fit_books)}")

    private_path = ROOT / "paired_records.private.jsonl"
    write_jsonl(private_path, records)
    outputs: dict[str, list[dict[str, Any]]] = {}
    for role in ("fit", "validation", "internal_test"):
        role_records = [row for row in records if row["role"] == role]
        outputs[role] = [
            example
            for record in role_records
            for example in build_examples(record, augment=role == "fit")
        ]
        write_jsonl(ROOT / f"{role}.jsonl", outputs[role])

    manifest = {
        "schema_version": 1,
        "dataset_id": "feitian-paired-reconstruction-lora-smoke-v1",
        "status": "infrastructure_smoke_only_not_efficacy_evidence",
        "seed": SEED,
        "target_author": TARGET_AUTHOR,
        "book_split": {
            "fit": fit_books,
            "validation": list(VALIDATION_BOOKS),
            "internal_test": list(INTERNAL_TEST_BOOKS),
            "book_disjoint": True,
        },
        "geometry": {
            "source_passages": len(records),
            "excluded_placeholder_mismatch_books": len(
                placeholder_mismatch_exclusions
            ),
            "fit_books": len(fit_books),
            "validation_books": len(VALIDATION_BOOKS),
            "internal_test_books": len(INTERNAL_TEST_BOOKS),
            "fit_examples": len(outputs["fit"]),
            "validation_examples": len(outputs["validation"]),
            "internal_test_examples": len(outputs["internal_test"]),
            "fit_unique_target_han": sum(
                row["target_han"] for row in records if row["role"] == "fit"
            ),
        },
        "augmentation": (
            "Fit contains one full passage plus deterministic nonoverlapping contiguous "
            "blocks; validation and test contain full passages only. Augmentation does "
            "not increase unique text and must not be reported as corpus size."
        ),
        "excluded_records": placeholder_mismatch_exclusions,
        "training_contract": {
            "input": "approved English semantic source plus neutral Chinese",
            "target": "entity_masked_v3 target-author Chinese",
            "loss": "assistant completion only (--mask-prompt)",
            "purpose": "local MLX QLoRA plumbing and overfit sanity check only",
        },
        "claim_limits": [
            "not style-transfer efficacy evidence",
            "not sufficient data for model selection",
            "not a production adapter",
            "not a substitute for the preregistered 1500-2000 accepted-pair expansion",
        ],
        "source_hashes": dict(sorted(source_hashes.items())),
        "output_hashes": {
            str(private_path): sha256_file(private_path),
            **{
                str(ROOT / f"{role}.jsonl"): sha256_file(ROOT / f"{role}.jsonl")
                for role in outputs
            },
        },
    }
    manifest["manifest_id"] = sha256_text(canonical(manifest))
    write_json(ROOT / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
