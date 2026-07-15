#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from experiments.iteration4.prepare import mask_term_plans, mask_v3_text
from experiments.iteration5.construct import run_cr_fysm_v4_construct_generation as generation
from experiments.iteration5.construct.construct_v2_protocol import (
    han_ngram_overlap,
    outcome_protocol,
    stable_side_order,
)


SEED = 20260717
ROOT = generation.ROOT / "analysis_v2"
PREREGISTRATION = ROOT / "preregistration.analysis.v2.json"
PRIVATE_ITEMS = ROOT / "construct_items.private.jsonl"
PRIVATE_PAIRS = ROOT / "construct_pairs.private.jsonl"
PACKETS = ROOT / "packets"
CLOSURE = generation.ROOT / "generation_completion_manifest.v2.json"
AMENDMENT_ROOT = generation.ROOT / "protocol_amendment_english_source_v1"
AMENDMENT_DECLARATION = AMENDMENT_ROOT / "preregistration.json"
AMENDMENT_RUNNER = Path(
    "experiments/iteration5/construct/run_cr_fysm_v4_english_source_amendment_v1.py"
)
ADJUDICATION_AMENDMENT_ROOT = (
    generation.ROOT / "protocol_amendment_english_adjudication_v1"
)
ADJUDICATION_AMENDMENT_DECLARATION = ADJUDICATION_AMENDMENT_ROOT / "preregistration.json"
ADJUDICATION_AMENDMENT_RUNNER = Path(
    "experiments/iteration5/construct/run_cr_fysm_v4_english_adjudication_amendment_v1.py"
)
STYLE_POSITIVE_AMENDMENT_ROOT = (
    generation.ROOT / "protocol_amendment_style_positive_block_v1"
)
STYLE_POSITIVE_AMENDMENT_DECLARATION = (
    STYLE_POSITIVE_AMENDMENT_ROOT / "preregistration.json"
)
STYLE_POSITIVE_AMENDMENT_PREPARER = Path(
    "experiments/iteration5/construct/prepare_cr_fysm_v4_style_positive_block_amendment_v1.py"
)
STYLE_POSITIVE_AMENDMENT_RUNNER = Path(
    "experiments/iteration5/construct/run_cr_fysm_v4_style_positive_block_amendment_v1.py"
)
STYLE_POSITIVE_AMENDMENT_REVISION = Path(
    "experiments/iteration5/construct/revise_cr_fysm_v4_style_positive_block_amendment_v1.py"
)
STYLE_POSITIVE_AMENDMENT_AUDIT = (
    STYLE_POSITIVE_AMENDMENT_ROOT / "independent_audit_pre_run.md"
)
STYLE_POSITIVE_AMENDMENT_REAUDIT = (
    STYLE_POSITIVE_AMENDMENT_ROOT / "independent_reaudit_pre_run.md"
)
STYLE_POSITIVE_STITCH_ROOT = (
    generation.ROOT / "protocol_amendment_style_positive_stitch_v2"
)
STYLE_POSITIVE_STITCH_DECLARATION = STYLE_POSITIVE_STITCH_ROOT / "preregistration.json"
STYLE_POSITIVE_STITCH_PREPARER = Path(
    "experiments/iteration5/construct/prepare_cr_fysm_v4_style_positive_stitch_amendment_v2.py"
)
STYLE_POSITIVE_STITCH_RUNNER = Path(
    "experiments/iteration5/construct/run_cr_fysm_v4_style_positive_stitch_amendment_v2.py"
)
STYLE_POSITIVE_STITCH_AUDIT = STYLE_POSITIVE_STITCH_ROOT / "independent_audit_pre_run.md"
STYLE_POSITIVE_INCIDENT_ROOT = (
    generation.ROOT / "protocol_incident_style_positive_completion_v1"
)
STYLE_POSITIVE_INCIDENT = STYLE_POSITIVE_INCIDENT_ROOT / "incident.json"
STYLE_POSITIVE_INCIDENT_RECORDER = Path(
    "experiments/iteration5/construct/record_cr_fysm_v4_style_positive_incident_v1.py"
)
STYLE_POSITIVE_INCIDENT_AUDIT = (
    STYLE_POSITIVE_INCIDENT_ROOT / "independent_audit_preclosure.md"
)
ASSET_DIR = Path("experiments/iteration5/construct/assets")
SEMANTIC_PROMPT = ASSET_DIR / "semantic_validator.v1.md"
SEMANTIC_SCHEMA = ASSET_DIR / "semantic_validator.schema.json"
STYLE_PROMPT = ASSET_DIR / "style_rater.v1.md"
STYLE_SCHEMA = ASSET_DIR / "style_rater.schema.json"
RATING_RUNNER = Path("experiments/iteration5/construct/run_cr_fysm_v4_construct_ratings_v2.py")
SCORING_RUNNER = Path("experiments/iteration5/construct/analyze_cr_fysm_v4_construct_v2.py")
MASK_PLAN = Path("datasets/masked/mask_terms.json")
POSTCLEAN_PREREGISTRATION = Path(
    "generated/style_research/style_transfer_experiments/iterations/"
    "content_resistant_v1/cr_fysm_v4_postclean_replication_v2/preregistration.json"
)
POSTCLEAN_RESULTS = POSTCLEAN_PREREGISTRATION.with_name("results.json")
VARIANT_STAGES = (
    "neutral_translation",
    "style_sham_control",
    "style_positive_control",
)
RATER_IDS = ("rater_a", "rater_b", "rater_c")
VALIDATOR_IDS = ("validator_a", "validator_b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the construct-v2 blind analysis lock.")
    parser.add_argument("--mode", choices=("draft", "lock", "validate"), default="draft")
    return parser.parse_args()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: str) -> str:
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


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def lock_id(payload: dict[str, Any]) -> str:
    value = dict(payload)
    value.pop("lock_id", None)
    return stable_hash(canonical(value))


def paragraph_map(rows: list[dict[str, str]], key: str) -> dict[str, str]:
    return {str(row["id"]): str(row[key]) for row in rows}


def masked_original_map(sample: dict[str, Any]) -> dict[str, str]:
    ids = [row["id"] for row in sample["paragraphs"]]
    lines = [line for line in sample["entity_masked_v3_zh"].splitlines() if line.strip()]
    if len(lines) != len(ids):
        raise ValueError(f"masked original paragraph count mismatch: {sample['sample_id']}")
    return dict(zip(ids, lines, strict=True))


def block_text(values: dict[str, str], paragraph_ids: list[str]) -> str:
    return "\n".join(values[value] for value in paragraph_ids)


def build_private_rows(
    generation_preregistration: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples = generation.read_jsonl(generation.SELECTION)
    terms_by_book = mask_term_plans()
    references = [
        generation.reference_text(example[field])
        for example in generation_preregistration["style_evidence"]["reference_examples"]
        for field in ("neutral_zh", "target_style_zh")
    ]
    items: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = sample["sample_id"]
        english = paragraph_map(
            generation.load_result("english_adjudication", sample, generation_preregistration)[
                "paragraphs"
            ],
            "en",
        )
        original_masked = masked_original_map(sample)
        generated_raw = {
            stage: paragraph_map(
                generation.load_result(stage, sample, generation_preregistration)["paragraphs"],
                "zh",
            )
            for stage in VARIANT_STAGES
        }
        terms = terms_by_book[(sample["source_author"], sample["source_book"])]
        generated_masked = {
            stage: {
                paragraph_id: mask_v3_text(text, terms)
                for paragraph_id, text in values.items()
            }
            for stage, values in generated_raw.items()
        }
        for block in sample["construct_blocks"]:
            block_id = block["block_id"]
            paragraph_ids = block["paragraph_ids"]
            base_id = f"{sample_id}.{block_id}"
            english_text = block_text(english, paragraph_ids)
            original_text = block_text(original_masked, paragraph_ids)
            variants: dict[str, dict[str, Any]] = {}
            for stage in VARIANT_STAGES:
                raw = block_text(generated_raw[stage], paragraph_ids)
                masked = block_text(generated_masked[stage], paragraph_ids)
                item_id = "ci_" + stable_hash(f"construct-v2-item:{base_id}:{stage}")[:24]
                overlap = (
                    han_ngram_overlap(raw, references, n=4)
                    if stage == "style_positive_control"
                    else None
                )
                row = {
                    "item_id": item_id,
                    "sample_id": sample_id,
                    "block_id": block_id,
                    "variant": stage,
                    "source_role": sample["source_role"],
                    "source_author": sample["source_author"],
                    "source_book": sample["source_book"],
                    "source_chunk_id": sample["source_chunk_id"],
                    "english": english_text,
                    "candidate_raw_zh": raw,
                    "candidate_masked_v3_zh": masked,
                    "original_masked_v3_zh": original_text,
                    "reference_4gram_overlap": overlap,
                }
                items.append(row)
                variants[stage] = row
            for pair_type, control_stage in (
                ("primary_style_vs_sham", "style_sham_control"),
                ("secondary_style_vs_neutral", "neutral_translation"),
            ):
                pair_id = "cp_" + stable_hash(
                    f"construct-v2-pair:{base_id}:{pair_type}"
                )[:24]
                pairs.append(
                    {
                        "pair_id": pair_id,
                        "pair_type": pair_type,
                        "sample_id": sample_id,
                        "block_id": block_id,
                        "english": english_text,
                        "style_item_id": variants["style_positive_control"]["item_id"],
                        "control_item_id": variants[control_stage]["item_id"],
                        "style_raw_zh": variants["style_positive_control"]["candidate_raw_zh"],
                        "control_raw_zh": variants[control_stage]["candidate_raw_zh"],
                        "side_order": {
                            rater_id: stable_side_order(
                                pair_id=pair_id, rater_id=rater_id, seed=SEED
                            )
                            for rater_id in RATER_IDS
                        },
                    }
                )
    if len(items) != 750 or len(pairs) != 500:
        raise ValueError(f"unexpected construct analysis geometry: {len(items)} items, {len(pairs)} pairs")
    return sorted(items, key=lambda row: row["item_id"]), sorted(
        pairs, key=lambda row: row["pair_id"]
    )


def chunks(values: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def write_packets(
    generation_preregistration: dict[str, Any],
    items: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    style_definition = generation_preregistration["style_evidence"]["dimensions"]
    for validator_id in VALIDATOR_IDS:
        ordered = sorted(
            items,
            key=lambda row: stable_hash(f"{SEED}:{validator_id}:{row['item_id']}"),
        )
        directory = PACKETS / validator_id
        directory.mkdir(parents=True, exist_ok=False)
        for index, packet_items in enumerate(chunks(ordered, 25), start=1):
            packet_id = f"{validator_id}.p{index:02d}"
            path = directory / f"{packet_id}.json"
            write_json(
                path,
                {
                    "packet_id": packet_id,
                    "items": [
                        {
                            "item_id": row["item_id"],
                            "english_source": row["english"],
                            "chinese_candidate": row["candidate_raw_zh"],
                        }
                        for row in packet_items
                    ],
                },
            )
            hashes[str(path)] = sha256_file(path)
    for rater_id in RATER_IDS:
        ordered = sorted(
            pairs,
            key=lambda row: stable_hash(f"{SEED}:{rater_id}:{row['pair_id']}"),
        )
        directory = PACKETS / rater_id
        directory.mkdir(parents=True, exist_ok=False)
        for index, packet_pairs in enumerate(chunks(ordered, 25), start=1):
            packet_id = f"{rater_id}.p{index:02d}"
            rendered = []
            for row in packet_pairs:
                style_first = row["side_order"][rater_id] == "style_first"
                rendered.append(
                    {
                        "pair_id": row["pair_id"],
                        "english_source": row["english"],
                        "candidate_a": row["style_raw_zh"] if style_first else row["control_raw_zh"],
                        "candidate_b": row["control_raw_zh"] if style_first else row["style_raw_zh"],
                    }
                )
            path = directory / f"{packet_id}.json"
            write_json(
                path,
                {
                    "packet_id": packet_id,
                    "style_t_definition": style_definition,
                    "pairs": rendered,
                },
            )
            hashes[str(path)] = sha256_file(path)
    return dict(sorted(hashes.items()))


def fixed_input_paths() -> list[Path]:
    postclean = read_json(POSTCLEAN_PREREGISTRATION)
    artifacts = [Path(value) for value in postclean["historical_source"]["source_artifact_paths"].values()]
    return [
        Path(__file__),
        RATING_RUNNER,
        SCORING_RUNNER,
        Path("experiments/iteration5/construct/run_cr_fysm_v4_construct_generation.py"),
        Path("experiments/iteration5/construct/construct_v2_protocol.py"),
        Path("experiments/iteration4/prepare.py"),
        Path("experiments/iteration5/meter/benchmark_content_resistant_meter.py"),
        Path("experiments/iteration5/meter/build_cr_fysm_v3.py"),
        Path("experiments/iteration5/meter/develop_cr_fysm_v4.py"),
        generation.PREREGISTRATION,
        generation.SELECTION,
        CLOSURE,
        AMENDMENT_DECLARATION,
        AMENDMENT_RUNNER,
        ADJUDICATION_AMENDMENT_DECLARATION,
        ADJUDICATION_AMENDMENT_RUNNER,
        STYLE_POSITIVE_AMENDMENT_DECLARATION,
        STYLE_POSITIVE_AMENDMENT_PREPARER,
        STYLE_POSITIVE_AMENDMENT_RUNNER,
        STYLE_POSITIVE_AMENDMENT_REVISION,
        STYLE_POSITIVE_AMENDMENT_AUDIT,
        STYLE_POSITIVE_AMENDMENT_REAUDIT,
        STYLE_POSITIVE_STITCH_DECLARATION,
        STYLE_POSITIVE_STITCH_PREPARER,
        STYLE_POSITIVE_STITCH_RUNNER,
        STYLE_POSITIVE_STITCH_AUDIT,
        STYLE_POSITIVE_INCIDENT,
        STYLE_POSITIVE_INCIDENT_RECORDER,
        STYLE_POSITIVE_INCIDENT_AUDIT,
        SEMANTIC_PROMPT,
        SEMANTIC_SCHEMA,
        STYLE_PROMPT,
        STYLE_SCHEMA,
        MASK_PLAN,
        POSTCLEAN_PREREGISTRATION,
        POSTCLEAN_RESULTS,
        Path("uv.lock"),
        *sorted((AMENDMENT_ROOT / "raw_attempts").glob("*/*.json"), key=str),
        *sorted(
            (ADJUDICATION_AMENDMENT_ROOT / "raw_attempts").glob("*/*.json"),
            key=str,
        ),
        *sorted(
            (STYLE_POSITIVE_AMENDMENT_ROOT / "raw_attempts").glob("*/*/*.json"),
            key=str,
        ),
        *artifacts,
    ]


def current_input_hashes() -> dict[str, str]:
    return {str(path): sha256_file(path) for path in sorted(set(fixed_input_paths()), key=str)}


def current_packet_hashes() -> dict[str, str]:
    paths = [
        path
        for identity in (*VALIDATOR_IDS, *RATER_IDS)
        for path in (PACKETS / identity).glob("*.json")
    ]
    return {str(path): sha256_file(path) for path in sorted(paths, key=str)}


def validate_generation_closure(generation_preregistration: dict[str, Any]) -> dict[str, Any]:
    closure = read_json(CLOSURE)
    if closure.get("status") != "complete_before_any_construct_rating_or_scoring":
        raise ValueError("construct generation is not closed")
    if closure.get("generation_lock_id") != generation_preregistration["lock_id"]:
        raise ValueError("construct closure does not bind to the generation lock")
    if closure.get("artifact_count") != 250:
        raise ValueError("construct closure does not contain exactly 250 artifacts")
    closure_without_id = dict(closure)
    observed_manifest_id = closure_without_id.pop("manifest_id", None)
    if observed_manifest_id != generation.sha256_text(
        canonical(closure_without_id)
    ):
        raise ValueError("construct closure manifest_id is invalid")
    expected_stage_counts = {stage: 50 for stage in generation.STAGES}
    if closure.get("stage_counts") != expected_stage_counts:
        raise ValueError("construct closure stage counts are invalid")
    records = closure.get("artifacts", [])
    expected_keys = {
        (stage, sample["sample_id"])
        for stage in generation.STAGES
        for sample in generation.read_jsonl(generation.SELECTION)
    }
    observed_keys = {(row.get("stage"), row.get("sample_id")) for row in records}
    if observed_keys != expected_keys or len(records) != len(expected_keys):
        raise ValueError("construct closure artifact identities are incomplete")
    for record in records:
        path = Path(record["path"])
        if record.get("artifact_sha256") != sha256_file(path):
            raise ValueError(f"construct artifact changed after closure: {path}")
        artifact = read_json(path)
        if record.get("result_sha256") != artifact.get("result_sha256"):
            raise ValueError(f"construct result hash changed after closure: {path}")
    amendment = read_json(AMENDMENT_DECLARATION)
    amendment_ids = set(amendment.get("sample_ids", []))
    if len(amendment_ids) != 4:
        raise ValueError("construct amendment does not declare exactly four samples")
    observed_amendment_ids: set[str] = set()
    for sample_id in amendment_ids:
        path = generation.output_path("english_semantic_source", sample_id)
        artifact = read_json(path)
        metadata = artifact.get("protocol_amendment")
        if not isinstance(metadata, dict):
            raise ValueError(f"missing protocol amendment metadata: {sample_id}")
        if metadata.get("amendment_id") != amendment["amendment_id"]:
            raise ValueError(f"protocol amendment ID mismatch: {sample_id}")
        if metadata.get("declaration_id") != amendment["declaration_id"]:
            raise ValueError(f"protocol amendment declaration mismatch: {sample_id}")
        raw_path = Path(str(metadata.get("raw_attempt_path", "")))
        if not raw_path.is_file() or metadata.get("raw_attempt_sha256") != sha256_file(raw_path):
            raise ValueError(f"protocol amendment raw provenance mismatch: {sample_id}")
        if metadata.get("original_result_sha256") != read_json(raw_path).get(
            "raw_result_sha256"
        ):
            raise ValueError(f"protocol amendment raw result hash mismatch: {sample_id}")
        raw_record = read_json(raw_path)
        raw_result = raw_record.get("raw_result")
        if not isinstance(raw_result, dict):
            raise ValueError(f"protocol amendment raw result is missing: {sample_id}")
        expected_result = json.loads(json.dumps(raw_result, ensure_ascii=False))
        expected_locations: list[dict[str, Any]] = []
        for paragraph in expected_result["paragraphs"]:
            text = paragraph["en"]
            for replacement in amendment["deterministic_replacements"]:
                source = replacement["exact"]
                destination = replacement["replacement"]
                count = text.count(source)
                if count:
                    expected_locations.append(
                        {
                            "paragraph_id": paragraph["id"],
                            "exact": source,
                            "replacement": destination,
                            "count": count,
                        }
                    )
                    text = text.replace(source, destination)
            paragraph["en"] = text
        if metadata.get("applied") is True:
            if raw_record.get("validation_errors") != [
                "refusal or placeholder text detected"
            ]:
                raise ValueError(f"protocol amendment acceptance predicate mismatch: {sample_id}")
            if metadata.get("replacement_locations") != expected_locations:
                raise ValueError(f"protocol amendment replacement locations mismatch: {sample_id}")
            if artifact.get("result") != expected_result:
                raise ValueError(f"protocol amendment deterministic result mismatch: {sample_id}")
        elif raw_record.get("validation_errors") or artifact.get("result") != raw_result:
            raise ValueError(f"protocol amendment unmodified branch mismatch: {sample_id}")
        observed_amendment_ids.add(sample_id)
    stage_dir = generation.ROOT / "generation" / "english_semantic_source"
    for path in stage_dir.glob("*.json"):
        artifact = read_json(path)
        if "protocol_amendment" in artifact and artifact.get("sample_id") not in amendment_ids:
            raise ValueError(f"undeclared protocol amendment artifact: {path}")
    if observed_amendment_ids != amendment_ids:
        raise ValueError("construct protocol amendment coverage mismatch")
    adjudication_amendment = read_json(ADJUDICATION_AMENDMENT_DECLARATION)
    adjudication_ids = set(adjudication_amendment.get("sample_ids", []))
    if len(adjudication_ids) != 4:
        raise ValueError("adjudication amendment does not declare exactly four samples")
    observed_adjudication_ids: set[str] = set()
    for sample_id in adjudication_ids:
        path = generation.output_path("english_adjudication", sample_id)
        artifact = read_json(path)
        metadata = artifact.get("protocol_amendment")
        if not isinstance(metadata, dict):
            raise ValueError(f"missing adjudication amendment metadata: {sample_id}")
        if metadata.get("amendment_id") != adjudication_amendment["amendment_id"]:
            raise ValueError(f"adjudication amendment ID mismatch: {sample_id}")
        if metadata.get("declaration_id") != adjudication_amendment["declaration_id"]:
            raise ValueError(f"adjudication amendment declaration mismatch: {sample_id}")
        raw_path = Path(str(metadata.get("raw_attempt_path", "")))
        if not raw_path.is_file() or metadata.get("raw_attempt_sha256") != sha256_file(raw_path):
            raise ValueError(f"adjudication amendment raw provenance mismatch: {sample_id}")
        raw_record = read_json(raw_path)
        if metadata.get("original_result_sha256") != raw_record.get("raw_result_sha256"):
            raise ValueError(f"adjudication amendment raw result hash mismatch: {sample_id}")
        raw_result = raw_record.get("raw_result")
        if not isinstance(raw_result, dict):
            raise ValueError(f"adjudication amendment raw result is missing: {sample_id}")
        expected_result = json.loads(json.dumps(raw_result, ensure_ascii=False))
        expected_locations: list[dict[str, Any]] = []
        for paragraph in expected_result["paragraphs"]:
            text = paragraph["en"]
            for replacement in adjudication_amendment["deterministic_replacements"]:
                source = replacement["exact"]
                destination = replacement["replacement"]
                count = text.count(source)
                if count:
                    expected_locations.append(
                        {
                            "paragraph_id": paragraph["id"],
                            "exact": source,
                            "replacement": destination,
                            "count": count,
                        }
                    )
                    text = text.replace(source, destination)
            paragraph["en"] = text
        if metadata.get("applied") is True:
            if raw_record.get("validation_errors") != [
                "refusal or placeholder text detected"
            ]:
                raise ValueError(
                    f"adjudication amendment acceptance predicate mismatch: {sample_id}"
                )
            if metadata.get("replacement_locations") != expected_locations:
                raise ValueError(
                    f"adjudication amendment replacement locations mismatch: {sample_id}"
                )
            if artifact.get("result") != expected_result:
                raise ValueError(
                    f"adjudication amendment deterministic result mismatch: {sample_id}"
                )
        elif raw_record.get("validation_errors") or artifact.get("result") != raw_result:
            raise ValueError(f"adjudication amendment unmodified branch mismatch: {sample_id}")
        observed_adjudication_ids.add(sample_id)
    adjudication_stage_dir = generation.ROOT / "generation" / "english_adjudication"
    for path in adjudication_stage_dir.glob("*.json"):
        artifact = read_json(path)
        if "protocol_amendment" in artifact and artifact.get("sample_id") not in adjudication_ids:
            raise ValueError(f"undeclared adjudication amendment artifact: {path}")
    if observed_adjudication_ids != adjudication_ids:
        raise ValueError("adjudication protocol amendment coverage mismatch")
    style_positive_block = read_json(STYLE_POSITIVE_AMENDMENT_DECLARATION)
    style_positive_stitch = read_json(STYLE_POSITIVE_STITCH_DECLARATION)
    incident = read_json(STYLE_POSITIVE_INCIDENT)
    incident_without_id = dict(incident)
    observed_incident_id = incident_without_id.pop("incident_id", None)
    if observed_incident_id != generation.sha256_text(generation.canonical(incident_without_id)):
        raise ValueError("style-positive incident ID is invalid")
    if incident.get("status") != (
        "recorded_before_generation_closure_and_before_any_rating_or_scoring"
    ):
        raise ValueError("style-positive incident was not recorded before closure")
    affected_ids = set(incident.get("affected_sample_ids", []))
    if affected_ids != set(style_positive_block.get("sample_ids", [])) or len(affected_ids) != 2:
        raise ValueError("style-positive incident affected set is invalid")
    final_lineage = incident.get("final_lineage", {})
    whole_ids = {
        sample_id
        for sample_id, row in final_lineage.items()
        if row.get("kind") == "additional_whole_passage_completion_call"
    }
    stitched_ids = {
        sample_id
        for sample_id, row in final_lineage.items()
        if row.get("kind") == "deterministic_stitch_amendment_v2"
    }
    if len(whole_ids) != 1 or len(stitched_ids) != 1 or whole_ids | stitched_ids != affected_ids:
        raise ValueError("style-positive incident final lineage partition is invalid")
    whole_id = next(iter(whole_ids))
    whole_path = generation.output_path("style_positive_control", whole_id)
    whole_artifact = read_json(whole_path)
    if "protocol_amendment" in whole_artifact:
        raise ValueError("incident whole-passage artifact unexpectedly has amendment metadata")
    if final_lineage[whole_id].get("artifact_sha256") != sha256_file(whole_path):
        raise ValueError("incident whole-passage artifact hash mismatch")
    if final_lineage[whole_id].get("result_sha256") != whole_artifact.get("result_sha256"):
        raise ValueError("incident whole-passage result hash mismatch")

    stitched_id = next(iter(stitched_ids))
    stitched_path = generation.output_path("style_positive_control", stitched_id)
    stitched_artifact = read_json(stitched_path)
    stitched_metadata = stitched_artifact.get("protocol_amendment")
    if not isinstance(stitched_metadata, dict):
        raise ValueError("incident stitched artifact lacks amendment metadata")
    if stitched_metadata.get("amendment_id") != style_positive_stitch["amendment_id"]:
        raise ValueError("incident stitched amendment ID mismatch")
    if stitched_metadata.get("declaration_id") != style_positive_stitch["declaration_id"]:
        raise ValueError("incident stitched declaration ID mismatch")
    if final_lineage[stitched_id].get("artifact_sha256") != sha256_file(stitched_path):
        raise ValueError("incident stitched artifact hash mismatch")
    declared_blocks = style_positive_stitch["accepted_block_lineage"][stitched_id]
    if stitched_metadata.get("accepted_blocks") != declared_blocks:
        raise ValueError("incident stitched accepted-block lineage mismatch")
    stitched_paragraphs: list[dict[str, str]] = []
    cues: list[str] = []
    uncertainties: list[str] = []
    for block in declared_blocks:
        raw_path = Path(block["raw_attempt_path"])
        if block.get("raw_attempt_sha256") != sha256_file(raw_path):
            raise ValueError("incident stitched raw-attempt hash mismatch")
        raw_record = read_json(raw_path)
        raw_result = raw_record.get("raw_result")
        if raw_record.get("validation_errors") or not isinstance(raw_result, dict):
            raise ValueError("incident stitched accepted raw attempt is invalid")
        if block.get("raw_result_sha256") != generation.sha256_text(
            generation.canonical(raw_result)
        ):
            raise ValueError("incident stitched raw-result hash mismatch")
        if [row.get("id") for row in raw_result.get("paragraphs", [])] != block[
            "paragraph_ids"
        ]:
            raise ValueError("incident stitched paragraph partition mismatch")
        stitched_paragraphs.extend(raw_result["paragraphs"])
        cues.extend(raw_result["style_cues_applied"])
        uncertainties.extend(raw_result["uncertainties"])

    def stable_unique(values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))

    expected_stitched = {
        "sample_id": stitched_id,
        "paragraphs": stitched_paragraphs,
        "style_cues_applied": stable_unique(cues),
        "uncertainties": stable_unique(uncertainties),
    }
    if stitched_artifact.get("result") != expected_stitched:
        raise ValueError("incident deterministic stitched result mismatch")
    style_positive_stage_dir = generation.ROOT / "generation" / "style_positive_control"
    observed_amended_ids: set[str] = set()
    for path in style_positive_stage_dir.glob("*.json"):
        artifact = read_json(path)
        if "protocol_amendment" in artifact:
            observed_amended_ids.add(artifact["sample_id"])
        if "protocol_amendment" in artifact and artifact.get("sample_id") not in stitched_ids:
            raise ValueError(f"undeclared style-positive amendment artifact: {path}")
    if observed_amended_ids != stitched_ids:
        raise ValueError("style-positive protocol amendment coverage mismatch")
    return closure


def main() -> None:
    args = parse_args()
    if args.mode in {"lock", "validate"}:
        existing = read_json(PREREGISTRATION)
        expected_status = (
            "draft_before_any_construct_rating_or_scoring"
            if args.mode == "lock"
            else "locked_before_any_construct_rating_or_scoring"
        )
        if existing.get("status") != expected_status:
            raise ValueError(f"analysis preregistration status is not {expected_status}")
        if existing.get("input_hashes") != current_input_hashes():
            raise ValueError("construct analysis inputs changed")
        if existing.get("packet_hashes") != current_packet_hashes():
            raise ValueError("construct analysis packets changed")
        if existing.get("private_hashes") != {
            str(PRIVATE_ITEMS): sha256_file(PRIVATE_ITEMS),
            str(PRIVATE_PAIRS): sha256_file(PRIVATE_PAIRS),
        }:
            raise ValueError("construct private mappings changed")
        if args.mode == "validate":
            if existing.get("lock_id") != lock_id(existing):
                raise ValueError("construct analysis lock_id is invalid")
            print(json.dumps({"status": "valid", "lock_id": existing["lock_id"]}, indent=2))
            return
        if (ROOT / "ratings").exists() or (ROOT / "scores.jsonl").exists():
            raise ValueError("ratings or scores exist before the analysis lock")
        existing["status"] = "locked_before_any_construct_rating_or_scoring"
        existing["lock_id"] = lock_id(existing)
        write_json(PREREGISTRATION, existing)
        print(json.dumps({"status": existing["status"], "lock_id": existing["lock_id"]}, indent=2))
        return

    if ROOT.exists():
        raise ValueError("construct analysis draft already exists")
    generation_preregistration = generation.validate_preregistration()
    if generation_preregistration.get("outcome_identification_protocol") != outcome_protocol():
        raise ValueError("generation lock does not contain the current fixed outcome protocol")
    closure = validate_generation_closure(generation_preregistration)
    ROOT.mkdir(parents=True, exist_ok=False)
    items, pairs = build_private_rows(generation_preregistration)
    write_jsonl(PRIVATE_ITEMS, items)
    write_jsonl(PRIVATE_PAIRS, pairs)
    packet_hashes = write_packets(generation_preregistration, items, pairs)
    payload: dict[str, Any] = {
        "schema_version": 2,
        "analysis_id": "CR-FYSM-v4-fresh-generated-construct-v2-analysis",
        "status": "draft_before_any_construct_rating_or_scoring",
        "generation_lock_id": generation_preregistration["lock_id"],
        "generation_closure": {
            "path": str(CLOSURE),
            "manifest_id": closure["manifest_id"],
            "sha256": sha256_file(CLOSURE),
        },
        "generation_protocol_amendment": {
            "path": str(AMENDMENT_DECLARATION),
            "amendment_id": read_json(AMENDMENT_DECLARATION)["amendment_id"],
            "declaration_id": read_json(AMENDMENT_DECLARATION)["declaration_id"],
            "sha256": sha256_file(AMENDMENT_DECLARATION),
            "raw_attempt_count": len(
                list((AMENDMENT_ROOT / "raw_attempts").glob("*/*.json"))
            ),
            "claim_limit": "not a pristine unamended locked generation",
        },
        "english_adjudication_protocol_amendment": {
            "path": str(ADJUDICATION_AMENDMENT_DECLARATION),
            "amendment_id": read_json(ADJUDICATION_AMENDMENT_DECLARATION)[
                "amendment_id"
            ],
            "declaration_id": read_json(ADJUDICATION_AMENDMENT_DECLARATION)[
                "declaration_id"
            ],
            "sha256": sha256_file(ADJUDICATION_AMENDMENT_DECLARATION),
            "raw_attempt_count": len(
                list(
                    (ADJUDICATION_AMENDMENT_ROOT / "raw_attempts").glob("*/*.json")
                )
            ),
            "claim_limit": "not a pristine unamended locked generation",
        },
        "style_positive_protocol_amendment": {
            "block_declaration": {
                "path": str(STYLE_POSITIVE_AMENDMENT_DECLARATION),
                "amendment_id": read_json(STYLE_POSITIVE_AMENDMENT_DECLARATION)[
                    "amendment_id"
                ],
                "declaration_id": read_json(STYLE_POSITIVE_AMENDMENT_DECLARATION)[
                    "declaration_id"
                ],
                "sha256": sha256_file(STYLE_POSITIVE_AMENDMENT_DECLARATION),
            },
            "stitch_declaration": {
                "path": str(STYLE_POSITIVE_STITCH_DECLARATION),
                "amendment_id": read_json(STYLE_POSITIVE_STITCH_DECLARATION)[
                    "amendment_id"
                ],
                "declaration_id": read_json(STYLE_POSITIVE_STITCH_DECLARATION)[
                    "declaration_id"
                ],
                "sha256": sha256_file(STYLE_POSITIVE_STITCH_DECLARATION),
            },
            "incident": {
                "path": str(STYLE_POSITIVE_INCIDENT),
                "incident_id": read_json(STYLE_POSITIVE_INCIDENT)["incident_id"],
                "sha256": sha256_file(STYLE_POSITIVE_INCIDENT),
                "affected_source_ids": read_json(STYLE_POSITIVE_INCIDENT)[
                    "affected_sample_ids"
                ],
            },
            "raw_attempt_count": len(
                list(
                    (STYLE_POSITIVE_AMENDMENT_ROOT / "raw_attempts").glob(
                        "*/*/*.json"
                    )
                )
            ),
            "pre_run_revision": read_json(STYLE_POSITIVE_AMENDMENT_DECLARATION)[
                "pre_run_revision"
            ],
            "claim_limit": (
                "one affected oracle item uses an additional whole-passage completion "
                "call and one uses deterministic block stitching; exclude both source "
                "clusters in sensitivity analysis"
            ),
        },
        "outcome_identification_protocol": generation_preregistration[
            "outcome_identification_protocol"
        ],
        "reference_overlap_control": generation_preregistration["reference_overlap_control"],
        "geometry": {
            "source_clusters": 50,
            "blocks": 250,
            "semantic_items": len(items),
            "style_pairs": len(pairs),
            "semantic_packets_per_validator": 30,
            "style_packets_per_rater": 20,
        },
        "rating_models": {
            "semantic_validators": {"ids": list(VALIDATOR_IDS), "model": "gpt-5.5", "reasoning_effort": "high"},
            "semantic_adjudicator": {"id": "validator_c", "model": "gpt-5.5", "reasoning_effort": "high", "only_fixed_disagreements": True},
            "style_raters": {"ids": list(RATER_IDS), "model": "gpt-5.5", "reasoning_effort": "high"},
        },
        "runtime": generation.current_codex_runtime(),
        "paths": {
            "private_items": str(PRIVATE_ITEMS),
            "private_pairs": str(PRIVATE_PAIRS),
            "packets": str(PACKETS),
            "semantic_prompt": str(SEMANTIC_PROMPT),
            "semantic_schema": str(SEMANTIC_SCHEMA),
            "style_prompt": str(STYLE_PROMPT),
            "style_schema": str(STYLE_SCHEMA),
        },
        "input_hashes": current_input_hashes(),
        "packet_hashes": packet_hashes,
        "private_hashes": {
            str(PRIVATE_ITEMS): sha256_file(PRIVATE_ITEMS),
            str(PRIVATE_PAIRS): sha256_file(PRIVATE_PAIRS),
        },
        "commands": {
            "semantic": "uv run python -m experiments.iteration5.construct.run_cr_fysm_v4_construct_ratings_v2 semantic --jobs 4",
            "style": "uv run python -m experiments.iteration5.construct.run_cr_fysm_v4_construct_ratings_v2 style --jobs 4",
            "adjudicate": "uv run python -m experiments.iteration5.construct.run_cr_fysm_v4_construct_ratings_v2 adjudicate --jobs 4",
            "analyze": "uv run python -m experiments.iteration5.construct.analyze_cr_fysm_v4_construct_v2",
            "analyze_recover_after_implementation_failure_only": "uv run python -m experiments.iteration5.construct.analyze_cr_fysm_v4_construct_v2 --resume-opened",
        },
    }
    payload["lock_id"] = lock_id(payload)
    write_json(PREREGISTRATION, payload)
    print(
        json.dumps(
            {"status": payload["status"], "lock_id": payload["lock_id"], "geometry": payload["geometry"]},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
