#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence


from experiments.shared.paths import RESEARCH_ROOT


REPO_ROOT = RESEARCH_ROOT

import experiments.iteration1.style_transfer_payloads as evidence  # noqa: E402


DEFAULT_EXPERIMENT_ROOT = (
    REPO_ROOT
    / "generated/style_research/style_transfer_experiments/iterations/full_regeneration_v1"
)
MASKED_CHUNKS = REPO_ROOT / "datasets/masked/chunks.entity_masked_v3.jsonl"
TARGET_AUTHOR = "非天夜翔"
SEED = 20260715
DISCOVERY_BOOKS = 21
MIN_ABS_Z = 0.50
MIN_DISCOVERY_RECURRENCE = 2 / 3
MIN_VALIDATION_RECURRENCE = 2 / 3
MIN_SUPPORT = 50
MAX_EXAMPLE_CJK = 360
EXAMPLE_PARAGRAPHS = 3
SCENES = (
    "action_conflict",
    "dialogue",
    "internal_reflection",
    "interpersonal_care",
    "travel_transition",
    "worldbuilding_exposition",
)

CLOSE_READING_RESULT = (
    REPO_ROOT
    / "generated/style_research/style_transfer_experiments/method_assets/close_reading_results"
    / "result.v1.d91119be7bdcb84e84b1c5023622e56e3a6a33570525ef675458e34b768daa59.json"
)
CLOSE_READING_LEDGER = CLOSE_READING_RESULT.with_suffix(".ledger.json")
CLOSE_READING_PACKET = (
    REPO_ROOT
    / "generated/style_research/style_transfer_experiments/method_assets/close_reading_source_packets"
    / "source_packet.v1.d91119be7bdcb84e84b1c5023622e56e3a6a33570525ef675458e34b768daa59.json"
)


DIMENSION_SPECS: tuple[dict[str, Any], ...] = (
    {
        "dimension_id": "dialogue_turn_architecture",
        "label": "Dialogue-led turn architecture",
        "feature_ids": (
            "dialogue.quote_marks_per_kcjk",
            "dialogue.line_ratio",
            "dialogue.simple_speech_tags",
        ),
        "description": (
            "When the English source contains live exchange, let short quoted turns "
            "and plain attribution carry more of the scene progression."
        ),
        "trigger": "Two or more source-supported speech turns or an explicit speaker change.",
        "guardrail": (
            "Do not convert narration into dialogue, change the speaker, move words across "
            "speakers, or invent a speech act."
        ),
        "failure_mode": "Dialogue added for meter gain or ornate reporting verbs that slow turn-taking.",
    },
    {
        "dimension_id": "micro_reaction_timing",
        "label": "Local reaction timing",
        "feature_ids": ("dialogue.micro_reactions",),
        "description": (
            "Express an English-source-supported gaze, pause, gesture, or posture as a "
            "brief beat adjacent to the relevant utterance."
        ),
        "trigger": "The English paragraph explicitly contains the reaction or silence.",
        "guardrail": "Never invent body language, emotion, motive, or physical action.",
        "failure_mode": "Hallucinated gestures or expanded scenic detours.",
    },
    {
        "dimension_id": "laughter_tonal_pivots",
        "label": "Compact laughter and tonal pivots",
        "feature_ids": ("dialogue.laughter_tags",),
        "description": (
            "Render explicit laughter or amused delivery as a compact tonal pivot inside "
            "the surrounding turn rather than a long explanation."
        ),
        "trigger": "The English source explicitly marks laughter, amusement, or smiling speech.",
        "guardrail": "Do not add amusement, flirtation, relief, or laughter absent from English.",
        "failure_mode": "Using laughter tags as generic filler.",
    },
    {
        "dimension_id": "modular_sentence_flow",
        "label": "Short modular sentence flow",
        "feature_ids": (
            "flow.short_sentence_ratio",
            "flow.mean_sentence_cjk",
            "punctuation.comma",
        ),
        "description": (
            "Prefer compact action-reply-perception units and controlled comma-linked "
            "sequencing over long explanatory periods."
        ),
        "trigger": "A paragraph contains several source clauses or sequential beats.",
        "guardrail": (
            "Preserve causal scope, negation, modifiers, and event order; do not fragment "
            "a relation that needs one sentence to remain unambiguous."
        ),
        "failure_mode": "Choppy fragments, comma splices that obscure logic, or reordered events.",
    },
    {
        "dimension_id": "light_explicit_scaffolding",
        "label": "Light explicit grammatical scaffolding",
        "feature_ids": (
            "function_word.connective",
            "function_word.modal_aspect",
            "function_word.preposition_frame",
            "function_word.particle_phrase",
            "function_word.deictic_pronoun",
        ),
        "description": (
            "Where Chinese order and juxtaposition already make the relation clear, avoid "
            "redundant connective, modal, deictic, and framing phrases."
        ),
        "trigger": "The same relation remains explicit after removing redundant scaffolding.",
        "guardrail": (
            "Never remove source causality, contrast, condition, modality, aspect, deixis, "
            "or referential clarity."
        ),
        "failure_mode": "Deleting semantically necessary function words to chase a corpus rate.",
    },
    {
        "dimension_id": "interrogative_pressure",
        "label": "Direct interrogative pressure",
        "feature_ids": ("dialogue.question_tags", "punctuation.question"),
        "description": (
            "When English already contains a question, preserve its direct interpersonal "
            "force instead of paraphrasing it into reflective narration."
        ),
        "trigger": "An explicit English question, check, challenge, or confirmation request.",
        "guardrail": "Do not create a question, accusation, or stronger pragmatic force.",
        "failure_mode": "Turning statements into questions or escalating tone.",
    },
    {
        "dimension_id": "pause_and_silence",
        "label": "Functional pauses and silence beats",
        "feature_ids": ("dialogue.silence_beats", "punctuation.ellipsis"),
        "description": (
            "Use a brief silence marker or ellipsis only when English contains hesitation, "
            "withholding, interruption, or an unanswered turn."
        ),
        "trigger": "A source-supported pause, hesitation, interruption, or missing reply.",
        "guardrail": "Do not add uncertainty or suspense, and do not use trailing marks as decoration.",
        "failure_mode": "Decorative ellipses or invented emotional hesitation.",
    },
    {
        "dimension_id": "marked_punctuation",
        "label": "Source-licensed marked punctuation",
        "feature_ids": ("punctuation.colon", "punctuation.exclamation"),
        "description": (
            "Use colons for genuinely labeled or abrupt source units and exclamation marks "
            "for source-supported force; otherwise prefer ordinary punctuation."
        ),
        "trigger": "An explicit label/list/message format or strong exclamation in English.",
        "guardrail": "Do not intensify emotion, urgency, or formatting beyond English.",
        "failure_mode": "Punctuation inflation used as a superficial author marker.",
    },
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield value


def stable_order(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode("utf-8")).hexdigest()


def write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    content = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise ValueError(f"Immutable artifact collision: {path}")
    path.write_text(content, encoding="utf-8")


def write_pre_generation_lock(
    path: Path,
    value: Mapping[str, Any],
    experiment_root: Path,
) -> None:
    content = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    if list((experiment_root / "runs").glob("**/method_outputs/**/*.json")):
        raise ValueError(
            "Cannot rebind the style-definition lock after style output exists"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def add_profile_row(
    profile: evidence.FeatureProfile, row: Mapping[str, Any]
) -> None:
    text = str(row["text"])
    profile.add(
        evidence._feature_observations(text),
        title=str(row["title"]),
        cjk_count=evidence._cjk_len(text),
    )


def build_profiles(
    rows: Sequence[Mapping[str, Any]], discovery_books: set[str]
) -> tuple[
    evidence.FeatureProfile,
    dict[str, evidence.FeatureProfile],
    dict[str, evidence.FeatureProfile],
]:
    target_discovery = evidence.FeatureProfile()
    target_books: dict[str, evidence.FeatureProfile] = defaultdict(
        evidence.FeatureProfile
    )
    comparison_authors: dict[str, evidence.FeatureProfile] = defaultdict(
        evidence.FeatureProfile
    )
    for row in rows:
        if row.get("split") != "train":
            continue
        author = str(row.get("author", ""))
        title = str(row.get("title", ""))
        if author == TARGET_AUTHOR:
            add_profile_row(target_books[title], row)
            if title in discovery_books:
                add_profile_row(target_discovery, row)
        else:
            add_profile_row(comparison_authors[author], row)
    if len(target_books) <= DISCOVERY_BOOKS:
        raise ValueError(
            f"Need more than {DISCOVERY_BOOKS} target train books, found {len(target_books)}"
        )
    if len(comparison_authors) != 49:
        raise ValueError(
            f"Expected 49 comparison train authors, found {len(comparison_authors)}"
        )
    return target_discovery, dict(target_books), dict(comparison_authors)


def recurrence(
    *,
    titles: Sequence[str],
    profiles: Mapping[str, evidence.FeatureProfile],
    feature_id: str,
    comparison_mean: float,
    direction: str,
) -> dict[str, Any]:
    spec = evidence.FEATURE_SPEC_BY_ID[feature_id]
    values: list[dict[str, Any]] = []
    matches = 0
    for title in sorted(titles):
        value = profiles[title].value(spec)
        if value is None:
            continue
        matched = value > comparison_mean if direction == "higher" else value < comparison_mean
        matches += int(matched)
        values.append(
            {
                "book_hash": hashlib.sha256(title.encode("utf-8")).hexdigest()[:16],
                "value": round(value, 6),
                "matches_direction": matched,
            }
        )
    return {
        "books": len(values),
        "matching_books": matches,
        "rate": round(matches / max(len(values), 1), 6),
        "mean": round(fmean(row["value"] for row in values), 6),
        "book_values": values,
    }


def validated_contrasts(
    discovery_profile: evidence.FeatureProfile,
    target_books: Mapping[str, evidence.FeatureProfile],
    comparison_authors: Mapping[str, evidence.FeatureProfile],
    discovery_books: Sequence[str],
    validation_books: Sequence[str],
) -> list[dict[str, Any]]:
    retained: list[dict[str, Any]] = []
    for contrast in evidence._profile_contrasts(
        discovery_profile, comparison_authors
    ):
        feature_id = str(contrast["feature_id"])
        discovery = recurrence(
            titles=discovery_books,
            profiles=target_books,
            feature_id=feature_id,
            comparison_mean=float(contrast["comparison_author_mean"]),
            direction=str(contrast["direction"]),
        )
        validation = recurrence(
            titles=validation_books,
            profiles=target_books,
            feature_id=feature_id,
            comparison_mean=float(contrast["comparison_author_mean"]),
            direction=str(contrast["direction"]),
        )
        eligible = (
            abs(float(contrast["z_score"])) >= MIN_ABS_Z
            and int(contrast["target_support"]) >= MIN_SUPPORT
            and float(discovery["rate"]) >= MIN_DISCOVERY_RECURRENCE
            and float(validation["rate"]) >= MIN_VALIDATION_RECURRENCE
        )
        retained.append(
            {
                **contrast,
                "discovery_recurrence": discovery,
                "validation_recurrence": validation,
                "retained": eligible,
                "retention_rule": {
                    "minimum_abs_discovery_z": MIN_ABS_Z,
                    "minimum_support": MIN_SUPPORT,
                    "minimum_discovery_recurrence": MIN_DISCOVERY_RECURRENCE,
                    "minimum_validation_recurrence": MIN_VALIDATION_RECURRENCE,
                },
            }
        )
    return retained


def paragraph_windows(text: str) -> list[str]:
    paragraphs = [value.strip() for value in text.splitlines() if value.strip()]
    windows: list[str] = []
    for start in range(len(paragraphs)):
        window = paragraphs[start : start + EXAMPLE_PARAGRAPHS]
        joined = "\n".join(window)
        count = evidence._cjk_len(joined)
        if 140 <= count <= MAX_EXAMPLE_CJK:
            windows.append(joined)
    if not windows and 100 <= evidence._cjk_len(text) <= MAX_EXAMPLE_CJK:
        windows.append(text.strip())
    return windows


def scene_score(text: str, scene: str) -> int:
    scores = evidence._scene_scores(text)
    score = int(scores.get(scene, 0))
    if scene == "dialogue":
        score += sum(line.lstrip().startswith(evidence.DIALOGUE_OPENERS) for line in text.splitlines())
    return score


def choose_examples(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if row.get("author") != TARGET_AUTHOR or row.get("split") != "train":
            continue
        for window in paragraph_windows(str(row["text"])):
            candidates.append(
                {
                    "title": str(row["title"]),
                    "chunk_id": str(row["chunk_id"]),
                    "text": window,
                    "placeholder_rate": window.count("某某") / max(evidence._cjk_len(window), 1),
                }
            )
    selected: list[dict[str, Any]] = []
    used_books: set[str] = set()
    for scene in SCENES:
        eligible_rows = [row for row in candidates if row["title"] not in used_books]
        if not eligible_rows:
            raise ValueError(f"No distinct-book example remains for {scene}")
        chosen = min(
            eligible_rows,
            key=lambda row: (
                -scene_score(str(row["text"]), scene),
                float(row["placeholder_rate"]),
                abs(evidence._cjk_len(str(row["text"])) - 260),
                stable_order(f"{scene}:{row['chunk_id']}:{row['text'][:30]}"),
            ),
        )
        if scene_score(str(chosen["text"]), scene) <= 0:
            raise ValueError(f"No source-supported example found for {scene}")
        used_books.add(str(chosen["title"]))
        selected.append(
            {
                "example_id": f"style_example_{len(selected) + 1:02d}",
                "scene": scene,
                "text": chosen["text"],
                "text_sha256": hashlib.sha256(
                    str(chosen["text"]).encode("utf-8")
                ).hexdigest(),
                "source_book_hash": hashlib.sha256(
                    str(chosen["title"]).encode("utf-8")
                ).hexdigest()[:16],
                "source_chunk_hash": hashlib.sha256(
                    str(chosen["chunk_id"]).encode("utf-8")
                ).hexdigest()[:16],
                "source_view": "entity_masked_v3",
                "instruction": (
                    "Observe only rhythm, syntax, dialogue topology, and punctuation. "
                    "Do not reuse wording, entities, imagery, or events."
                ),
            }
        )
    return selected


def load_close_reading() -> dict[str, Any]:
    result = json.loads(CLOSE_READING_RESULT.read_text(encoding="utf-8"))
    ledger = json.loads(CLOSE_READING_LEDGER.read_text(encoding="utf-8"))
    packet = json.loads(CLOSE_READING_PACKET.read_text(encoding="utf-8"))
    if result.get("source_packet_sha256") != packet.get("content_sha256"):
        raise ValueError("Close-reading result and packet hashes differ")
    if ledger.get("result_file_sha256") != file_sha256(CLOSE_READING_RESULT):
        raise ValueError("Close-reading ledger does not bind the result file")
    if packet.get("packet", {}).get("evidence_policy", {}).get(
        "development_final_and_application_text_used"
    ) is not False:
        raise ValueError("Close-reading packet does not prove train-only isolation")
    return {
        "claims": result["claims"],
        "cards": result["cards"],
        "limitations": result.get("limitations", []),
        "provenance": {
            "result_path": str(CLOSE_READING_RESULT.relative_to(REPO_ROOT)),
            "result_file_sha256": file_sha256(CLOSE_READING_RESULT),
            "ledger_path": str(CLOSE_READING_LEDGER.relative_to(REPO_ROOT)),
            "ledger_file_sha256": file_sha256(CLOSE_READING_LEDGER),
            "source_packet_path": str(CLOSE_READING_PACKET.relative_to(REPO_ROOT)),
            "source_packet_file_sha256": file_sha256(CLOSE_READING_PACKET),
            "model": result["generation_provenance"]["model"],
            "run_id": result["generation_provenance"]["run_id"],
        },
    }


def build_dimensions(
    contrasts: Sequence[Mapping[str, Any]], close_reading: Mapping[str, Any]
) -> list[dict[str, Any]]:
    retained = {
        str(row["feature_id"]): row for row in contrasts if row["retained"]
    }
    claims = list(close_reading["claims"])
    dimensions: list[dict[str, Any]] = []
    for spec in DIMENSION_SPECS:
        feature_rows = [
            retained[feature_id]
            for feature_id in spec["feature_ids"]
            if feature_id in retained
        ]
        if not feature_rows:
            continue
        claim_rows = [
            claim
            for claim in claims
            if set(map(str, claim.get("feature_ids", [])))
            & {str(row["feature_id"]) for row in feature_rows}
        ]
        dimensions.append(
            {
                **{key: value for key, value in spec.items() if key != "feature_ids"},
                "feature_ids": [str(row["feature_id"]) for row in feature_rows],
                "validated_statistics": feature_rows,
                "close_reading_claim_ids": [
                    str(claim["claim_id"]) for claim in claim_rows
                ],
                "application_policy": "source_triggered_not_quota_driven",
            }
        )
    required = {
        "dialogue_turn_architecture",
        "modular_sentence_flow",
        "light_explicit_scaffolding",
    }
    observed = {row["dimension_id"] for row in dimensions}
    if not required.issubset(observed) or len(dimensions) < 6:
        raise ValueError(
            f"Validated definition is too narrow: {sorted(observed)}"
        )
    return dimensions


def build(experiment_root: Path) -> dict[str, Any]:
    rows = list(iter_jsonl(MASKED_CHUNKS))
    target_titles = sorted(
        {
            str(row["title"])
            for row in rows
            if row.get("author") == TARGET_AUTHOR and row.get("split") == "train"
        },
        key=stable_order,
    )
    if len(target_titles) <= DISCOVERY_BOOKS:
        raise ValueError(
            f"Need more than {DISCOVERY_BOOKS} target train books, found {len(target_titles)}"
        )
    discovery_books = target_titles[:DISCOVERY_BOOKS]
    validation_books = target_titles[DISCOVERY_BOOKS:]
    discovery_profile, target_books, comparison_authors = build_profiles(
        rows, set(discovery_books)
    )
    contrasts = validated_contrasts(
        discovery_profile,
        target_books,
        comparison_authors,
        discovery_books,
        validation_books,
    )
    close_reading = load_close_reading()
    dimensions = build_dimensions(contrasts, close_reading)
    examples = choose_examples(rows)
    payload: dict[str, Any] = {
        "schema_version": "iteration4_style_definition.v1",
        "target_label": "target_author_style",
        "source_view": "entity_masked_v3",
        "evidence_policy": {
            "target_train_books": len(target_titles),
            "comparison_train_authors": 49,
            "discovery_books": len(discovery_books),
            "validation_books": len(validation_books),
            "development_books_used": False,
            "final_books_used": False,
            "eternal_gate_used": False,
            "masked_examples_only": True,
        },
        "book_partition": {
            "seed": SEED,
            "algorithm": "sha256_seeded_title_order_v1",
            "discovery_book_hashes": [
                hashlib.sha256(title.encode("utf-8")).hexdigest()[:16]
                for title in discovery_books
            ],
            "validation_book_hashes": [
                hashlib.sha256(title.encode("utf-8")).hexdigest()[:16]
                for title in validation_books
            ],
        },
        "statistical_contrasts": contrasts,
        "close_reading": close_reading,
        "definition": {
            "dimensions": dimensions,
            "global_guardrails": [
                "English meaning is authoritative; statistics never override source semantics.",
                "Apply a tendency only when the current English paragraph supplies its trigger.",
                "Do not imitate topic, lore, names, imagery, catchphrases, or reference wording.",
                "Do not add dialogue, gesture, emotion, causality, modality, or intensity.",
                "Preserve paragraph order, speaker attribution, and dialogue topology.",
            ],
            "masked_scene_examples": examples,
            "counterexample_policy": (
                "Each dimension's guardrail and failure_mode defines when to leave the "
                "neutral realization unchanged; examples are not templates."
            ),
        },
        "provenance": {
            "builder": "experiments/iteration4/build_style_definition.py",
            "builder_sha256": file_sha256(Path(__file__)),
            "masked_chunks_path": str(MASKED_CHUNKS.relative_to(REPO_ROOT)),
            "masked_chunks_sha256": file_sha256(MASKED_CHUNKS),
            "base_feature_extractor": "experiments/iteration1/style_transfer_payloads.py",
            "base_feature_extractor_sha256": file_sha256(
                REPO_ROOT / "experiments/iteration1/style_transfer_payloads.py"
            ),
        },
    }
    content_sha = sha256_json(payload)
    wrapper = {
        "schema_version": "iteration4_style_definition_asset.v1",
        "content_sha256": content_sha,
        "asset": payload,
    }
    relative_path = Path("method_assets/style_definition.v1") / f"asset.{content_sha}.json"
    asset_path = experiment_root / relative_path
    write_immutable_json(asset_path, wrapper)
    lock = {
        "schema_version": "iteration4_style_definition_lock.v1",
        "asset_path": relative_path.as_posix(),
        "content_sha256": content_sha,
        "file_sha256": file_sha256(asset_path),
        "dimension_count": len(dimensions),
        "example_count": len(examples),
        "retained_feature_count": sum(row["retained"] for row in contrasts),
    }
    write_pre_generation_lock(
        experiment_root / "method_assets/style_definition.v1.lock.json",
        lock,
        experiment_root,
    )
    return lock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the train-only validated Iteration 4 style definition."
    )
    parser.add_argument(
        "--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(build(args.experiment_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
