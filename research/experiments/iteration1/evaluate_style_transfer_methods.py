#!/usr/bin/env python3
from __future__ import annotations

"""Reproducible, non-LLM evaluation for the style-transfer experiment.

The script deliberately separates scorer fitting, calibration scoring, and method
evaluation.  It never invokes an LLM.  Independent semantic judgments can be
ingested as frozen JSONL, but absent judgments remain explicitly pending.
"""

import argparse
import csv
import hashlib
import html
import importlib.util
import io
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from experiments.iteration1.style_analysis_lock import (
    require_matching_analysis_binding,
    validate_analysis_lock,
)
from experiments.iteration1.style_experiment_provenance import (
    EXECUTION_BINDING_FIELDS,
    validate_artifact_and_ledger as validate_shared_artifact_and_ledger,
    validate_execution_binding as validate_shared_execution_binding,
    validate_run_config as validate_shared_run_config,
)
from experiments.iteration1.style_experiment_decisions import (
    recompute_confirmation_winner,
    recompute_promotion,
    validate_stage_method_set,
)


from experiments.shared.paths import RESEARCH_ROOT
from workflows.audit_style_dataset import (
    compile_term_matcher,
    mask_terms,
)
from workflows.audit_mask_artifacts import mask_stripped_char_ngrams
from workflows.author_style_meter_contract import (
    CROSS_BOOK_DECONTAMINATION_VERSION,
    CURRENT_CHAR_MIN_DF,
    CURRENT_MASKED_VIEW,
    CURRENT_MAX_CHAR_FEATURES,
    CURRENT_NGRAM_RANGE,
    CURRENT_SCORER_ID,
    CURRENT_SCORER_VALIDATION_RESULT,
    CURRENT_SCORER_VALIDATION_RUN_KEY,
    MASKING_POLICY_VERSION,
    PUNCTUATION_NORMALIZATION_VERSION,
)


REPO_ROOT = RESEARCH_ROOT
DEFAULT_DATASET_ROOT = REPO_ROOT / "datasets"
DEFAULT_EXPERIMENT_ROOT = (
    REPO_ROOT / "generated/style_research/style_transfer_experiments"
)
DEFAULT_SPLITS = REPO_ROOT / "generated/style_research/corpus/splits.json"
DEFAULT_BENCHMARK_RESULT = CURRENT_SCORER_VALIDATION_RESULT
DEFAULT_BENCHMARK_SCRIPT = REPO_ROOT / "workflows/audit_mask_artifacts.py"
DEFAULT_SAMPLE_SET = "development_proxy_v1"
DEFAULT_TARGET_AUTHOR = "非天夜翔"
SCORER_ID = CURRENT_SCORER_ID
DEFAULT_SCORER_DIR = DEFAULT_EXPERIMENT_ROOT / "scorers" / SCORER_ID
DEFAULT_CALIBRATION_OUTPUT = (
    DEFAULT_EXPERIMENT_ROOT / "calibration/style_meter_scores.v1.jsonl"
)
DEFAULT_THRESHOLD = (
    DEFAULT_EXPERIMENT_ROOT / "calibration/style_meter_threshold.v1.json"
)

BENCHMARK_RUN_KEY = CURRENT_SCORER_VALIDATION_RUN_KEY
MASKING_VIEW = CURRENT_MASKED_VIEW
MAX_CHAR_FEATURES = CURRENT_MAX_CHAR_FEATURES
CHAR_MIN_DF = CURRENT_CHAR_MIN_DF
NGRAM_RANGE = CURRENT_NGRAM_RANGE
SGD_ALPHA = 1e-5
SGD_MAX_ITER = 3_000
SGD_TOL = 1e-3
RANDOM_STATE = 13
N_JOBS = 1
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_710
NO_COPY_CJK_LENGTH = 8
FINAL_VALIDATION_BOOKS = (
    "星辰骑士",
    "夺梦",
    "定海浮生录",
    "国家一级注册驱魔师上岗培训通知",
)
SEMANTIC_NONINFERIORITY_MARGIN = 0.0

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
NUMBER_RE = re.compile(r"\d+(?:[.,:/-]\d+)*")
LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]*")
PLACEHOLDER_RE = re.compile(r"<[^>\n]+>")
PLACEHOLDER_SENTINELS = {
    "<CONTENT>": "\ue000",
    "<NUM>": "\ue001",
    "<LATIN>": "\ue002",
}
SENTENCE_BOUNDARY_RE = re.compile(r"[。！？!?…]+")
PUNCTUATION = set("，。！？；：、,.!?;:…—（）()【】[]《》“”‘’「」『』")
DIALOGUE_MARKS = set("“”‘’「」『』")
FUNCTION_CHARS = set(
    "的一是在不了有和人这中为上个我以要他时来用们到地于出就对成会可也能下过"
    "而后定行得经之着等里如自起把性好应开还因由其些然前那与关各并已又但"
    "只没给被很最才让吗呢啊吧呀么着了过"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

SCORER_FILES = {
    "vectorizer": "vectorizer.joblib",
    "classifier": "classifier.joblib",
    "labels": "labels.json",
    "config": "scorer_config.json",
    "masking": "masking_config.json",
    "manifest": "manifest.json",
}


class EvaluationError(RuntimeError):
    """Raised when a frozen evaluation contract cannot be satisfied."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def pretty_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise EvaluationError(f"Missing {label}: {path}")
    return path


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def resolve_recorded_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, pretty_json(value))


def write_frozen_text(path: Path, content: str) -> str:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise EvaluationError(
                f"Refusing to replace a different frozen artifact: {path}"
            )
        return "reused_identical"
    atomic_write_text(path, content)
    return "created"


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise EvaluationError(
                    f"Expected an object at {path}:{line_number}, got {type(value).__name__}"
                )
            yield value


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"Expected a JSON object in {path}")
    return value


def render_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(canonical_json(dict(row)) + "\n" for row in rows)


def canonical_jsonl_sha256(path: Path) -> str:
    return sha256_text(render_jsonl(list(iter_jsonl(path))))


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def cjk_only(text: str) -> str:
    return "".join(CJK_RE.findall(text))


def paragraphs_to_text(paragraphs: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(str(row.get("zh", "")).strip() for row in paragraphs)


def counter_json(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


@dataclass(frozen=True)
class ScorerBundle:
    directory: Path
    vectorizer: TfidfVectorizer
    classifier: SGDClassifier
    labels: tuple[str, ...]
    config: dict[str, Any]
    manifest: dict[str, Any]

    @property
    def target_author(self) -> str:
        return str(self.config["target_author"])

    @property
    def classifier_artifact_sha256(self) -> str:
        return str(self.manifest["classifier_artifact_sha256"])

    @property
    def scorer_config_sha256(self) -> str:
        return str(self.manifest["scorer_config_sha256"])

    @property
    def masking_artifact_sha256(self) -> str:
        return str(self.manifest["masking_artifact_sha256"])


@dataclass(frozen=True)
class ScorerPaths:
    dataset_root: Path
    masked_chunks: Path
    mask_terms: Path
    splits: Path
    benchmark_result: Path
    benchmark_script: Path
    scorer_dir: Path


@dataclass
class GeneratedContext:
    sample_id: str
    result: dict[str, Any] | None
    paragraphs: list[dict[str, Any]]
    text: str | None
    artifact: dict[str, Any] | None
    input_request: dict[str, Any] | None
    errors: list[str]
    provenance: dict[str, Any] | None = None


def scorer_paths_from_args(args: argparse.Namespace) -> ScorerPaths:
    dataset_root = args.dataset_root.resolve()
    return ScorerPaths(
        dataset_root=dataset_root,
        masked_chunks=(dataset_root / "masked/chunks.train_global_masked.jsonl"),
        mask_terms=(dataset_root / "masked/mask_terms.json"),
        splits=args.splits.resolve(),
        benchmark_result=args.benchmark_result.resolve(),
        benchmark_script=args.benchmark_script.resolve(),
        scorer_dir=args.scorer_dir.resolve(),
    )


def validate_benchmark_result(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = load_json(require_file(path, "exact benchmark result"))
    errors: list[str] = []
    if payload.get("target_author") != DEFAULT_TARGET_AUTHOR:
        errors.append("target_author")
    if MASKING_VIEW not in payload.get("views", []):
        errors.append("views")
    if "mask_stripped_char_ngrams" not in payload.get("representations", []):
        errors.append("representations")
    if int(payload.get("char_min_df", -1)) != CHAR_MIN_DF:
        errors.append("char_min_df")
    protocol = payload.get("protocol", {})
    if protocol.get("primary_view") != MASKING_VIEW:
        errors.append("protocol.primary_view")
    if protocol.get("primary_classifier") != "sgd_hinge_unbalanced":
        errors.append("protocol.primary_classifier")
    if protocol.get("primary_representation") != "mask_stripped_char_ngrams":
        errors.append("protocol.primary_representation")
    if payload.get("protocol_evaluation", {}).get("all_checks_pass") is not True:
        errors.append("protocol_evaluation")
    input_bindings = payload.get("input_bindings", {})
    for binding_name in ("source", "main_benchmark", "mask_plan"):
        binding = input_bindings.get(binding_name)
        if not isinstance(binding, dict):
            errors.append(f"input_bindings.{binding_name}")
            continue
        bound_path = resolve_recorded_path(str(binding.get("path", "")))
        if not bound_path.is_file() or file_sha256(bound_path) != binding.get("sha256"):
            errors.append(f"input_bindings.{binding_name}.sha256")
    chunk_binding = input_bindings.get("chunk_files", {}).get(MASKING_VIEW)
    if not isinstance(chunk_binding, dict):
        errors.append(f"input_bindings.chunk_files.{MASKING_VIEW}")
    else:
        chunk_path = resolve_recorded_path(str(chunk_binding.get("path", "")))
        if not chunk_path.is_file() or file_sha256(chunk_path) != chunk_binding.get("sha256"):
            errors.append(f"input_bindings.chunk_files.{MASKING_VIEW}.sha256")
    prior = payload.get("prior_ablation_reference")
    if not isinstance(prior, dict):
        errors.append("prior_ablation_reference")
    else:
        prior_path = resolve_recorded_path(str(prior.get("path", "")))
        if not prior_path.is_file() or file_sha256(prior_path) != prior.get("sha256"):
            errors.append("prior_ablation_reference.sha256")
    result = payload.get("results", {}).get(BENCHMARK_RUN_KEY)
    if not isinstance(result, dict):
        errors.append(f"results.{BENCHMARK_RUN_KEY}")
        result = {}
    expected_spec = {
        "key": BENCHMARK_RUN_KEY,
        "classifier": "sgd_hinge_unbalanced",
        "method": "mask_stripped_char_ngrams",
        "view": MASKING_VIEW,
    }
    if result.get("spec") != expected_spec:
        errors.append("exact run spec")
    if int(result.get("feature_count", -1)) != MAX_CHAR_FEATURES:
        errors.append("feature_count")
    if errors:
        raise EvaluationError(
            "Benchmark result does not match the exact meter contract: "
            + ", ".join(errors)
        )
    return payload, result


def masking_config_payload(paths: ScorerPaths) -> dict[str, Any]:
    mask_plan = load_json(require_file(paths.mask_terms, "mask terms"))
    if mask_plan.get("masking_policy") != MASKING_POLICY_VERSION:
        raise EvaluationError("Mask plan does not use the current leakage-control policy")
    return {
        "schema_version": 1,
        "view": MASKING_VIEW,
        "algorithm": {
            "latin_pattern": LATIN_RE.pattern,
            "latin_placeholder": "<LATIN>",
            "number_pattern": NUMBER_RE.pattern,
            "number_placeholder": "<NUM>",
            "global_entity_term_field": "global_terms.entity_terms_v2",
            "global_fit_split": "train",
            "held_out_statistics_used": False,
            "entity_replacement": "one_某_per_matched_cjk_character",
            "matching": "pyahocorasick_exact_leftmost_longest",
            "feature_extraction": (
                "drop_mask_runs_and_extract_2_4_character_ngrams_within_unmasked_spans"
            ),
        },
        "mask_terms_path": display_path(paths.mask_terms),
        "mask_terms_sha256": file_sha256(require_file(paths.mask_terms, "mask terms")),
        "masked_chunks_path": display_path(paths.masked_chunks),
        "masked_chunks_sha256": file_sha256(
            require_file(paths.masked_chunks, "train-global-masked chunks")
        ),
        "masking_source_path": display_path(REPO_ROOT / "workflows/audit_style_dataset.py"),
        "masking_source_sha256": file_sha256(
            require_file(
                REPO_ROOT / "workflows/audit_style_dataset.py", "masking source"
            )
        ),
    }


def scorer_config_payload(
    paths: ScorerPaths,
    benchmark_payload: Mapping[str, Any],
    benchmark_result: Mapping[str, Any],
    masking_artifact_sha256: str,
) -> dict[str, Any]:
    test_metrics = benchmark_result.get("chunk_metrics", {}).get("test", {})
    return {
        "schema_version": 1,
        "scorer_id": SCORER_ID,
        "target_author": DEFAULT_TARGET_AUTHOR,
        "training_contract": {
            "dataset_path": display_path(paths.masked_chunks),
            "dataset_sha256": file_sha256(paths.masked_chunks),
            "splits_path": display_path(paths.splits),
            "splits_sha256": file_sha256(require_file(paths.splits, "book splits")),
            "view": MASKING_VIEW,
            "allowed_split": "train",
            "train_books_only": True,
            "expected_train_chunks": int(benchmark_result["train_chunks"]),
            "expected_train_chunks_by_author": dict(
                sorted(benchmark_result["train_chunks_by_author"].items())
            ),
            "expected_labels": list(benchmark_result["labels"]),
        },
        "vectorizer": {
            "class": "sklearn.feature_extraction.text.TfidfVectorizer",
            "analyzer": "workflows.audit_mask_artifacts.mask_stripped_char_ngrams",
            "ngram_range": list(NGRAM_RANGE),
            "max_features": MAX_CHAR_FEATURES,
            "min_df": CHAR_MIN_DF,
            "sublinear_tf": True,
            "norm": "l2",
            "lowercase": False,
            "preprocessor": None,
            "dtype": "numpy.float32",
        },
        "classifier": {
            "class": "sklearn.linear_model.SGDClassifier",
            "loss": "hinge",
            "penalty": "l2",
            "alpha": SGD_ALPHA,
            "class_weight": None,
            "max_iter": SGD_MAX_ITER,
            "tol": SGD_TOL,
            "random_state": RANDOM_STATE,
            "n_jobs": N_JOBS,
        },
        "benchmark_binding": {
            "result_path": display_path(paths.benchmark_result),
            "result_sha256": file_sha256(paths.benchmark_result),
            "source_path": display_path(paths.benchmark_script),
            "source_sha256": file_sha256(
                require_file(paths.benchmark_script, "benchmark source")
            ),
            "run_key": BENCHMARK_RUN_KEY,
            "dataset_root": str(benchmark_payload.get("dataset_root", "")),
            "char_min_df": int(benchmark_payload["char_min_df"]),
            "expected_feature_count": int(benchmark_result["feature_count"]),
            "expected_test_metrics": {
                "accuracy": test_metrics.get("accuracy"),
                "balanced_accuracy": test_metrics.get("balanced_accuracy"),
                "macro_f1": test_metrics.get("macro_f1"),
                "target": test_metrics.get("target"),
            },
        },
        "masking_artifact_sha256": masking_artifact_sha256,
        "software": {
            "python": ".".join(str(part) for part in sys.version_info[:3]),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }


def expected_scorer_payloads(
    paths: ScorerPaths,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    benchmark_payload, benchmark_result = validate_benchmark_result(
        paths.benchmark_result
    )
    masking_payload = masking_config_payload(paths)
    masking_hash = sha256_text(pretty_json(masking_payload))
    config = scorer_config_payload(
        paths,
        benchmark_payload,
        benchmark_result,
        masking_hash,
    )
    return benchmark_payload, benchmark_result, masking_payload, masking_hash


def load_training_rows(
    paths: ScorerPaths,
    benchmark_result: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    split_payload = load_json(paths.splits)
    train_books = {
        (str(row["author"]), str(row["title"]))
        for row in split_payload.get("train", [])
    }
    if not train_books:
        raise EvaluationError("The frozen split artifact contains no train books")

    texts: list[str] = []
    labels: list[str] = []
    counts: Counter[str] = Counter()
    seen_chunks: set[str] = set()
    observed_book_splits: dict[tuple[str, str], str] = {}
    for row in iter_jsonl(paths.masked_chunks):
        chunk_id = str(row.get("chunk_id", ""))
        if not chunk_id or chunk_id in seen_chunks:
            raise EvaluationError(f"Missing or duplicate chunk_id in {paths.masked_chunks}: {chunk_id}")
        seen_chunks.add(chunk_id)
        if row.get("masking_policy") != MASKING_POLICY_VERSION:
            raise EvaluationError(f"Stale masking policy for {chunk_id}")
        if row.get("cross_book_decontamination") != CROSS_BOOK_DECONTAMINATION_VERSION:
            raise EvaluationError(f"Stale decontamination policy for {chunk_id}")
        if row.get("view") != MASKING_VIEW:
            raise EvaluationError(f"Unexpected view for {chunk_id}: {row.get('view')}")
        key = (str(row.get("author", "")), str(row.get("title", "")))
        split = str(row.get("split", ""))
        previous = observed_book_splits.setdefault(key, split)
        if previous != split:
            raise EvaluationError(f"Book appears in multiple chunk splits: {key}")
        if split == "train":
            if key not in train_books:
                raise EvaluationError(f"Chunk claims train but split artifact does not: {key}")
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                raise EvaluationError(f"Empty training text for {chunk_id}")
            author = key[0]
            texts.append(text)
            labels.append(author)
            counts[author] += 1
        elif key in train_books:
            raise EvaluationError(f"Train book has a non-train chunk: {key}")

    missing_train_books = train_books - {
        key for key, split in observed_book_splits.items() if split == "train"
    }
    if missing_train_books:
        raise EvaluationError(
            f"Train split books are missing from masked chunks: {sorted(missing_train_books)[:5]}"
        )
    expected_counts = Counter(
        {str(key): int(value) for key, value in benchmark_result["train_chunks_by_author"].items()}
    )
    if counts != expected_counts:
        raise EvaluationError(
            "Train-only author counts differ from the exact benchmark result; "
            f"observed={sum(counts.values())}, expected={sum(expected_counts.values())}"
        )
    if len(texts) != int(benchmark_result["train_chunks"]):
        raise EvaluationError("Train chunk count differs from exact benchmark result")
    return texts, labels


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer=mask_stripped_char_ngrams,
        max_features=MAX_CHAR_FEATURES,
        min_df=CHAR_MIN_DF,
        sublinear_tf=True,
        norm="l2",
        lowercase=False,
        dtype=np.float32,
    )


def make_classifier() -> SGDClassifier:
    return SGDClassifier(
        loss="hinge",
        penalty="l2",
        alpha=SGD_ALPHA,
        class_weight=None,
        max_iter=SGD_MAX_ITER,
        tol=SGD_TOL,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )


def fit_and_freeze_scorer(paths: ScorerPaths) -> ScorerBundle:
    if paths.scorer_dir.exists():
        raise EvaluationError(
            f"Scorer directory already exists; frozen artifacts are never replaced: {paths.scorer_dir}"
        )
    _, benchmark_result, masking_payload, masking_hash = expected_scorer_payloads(paths)
    benchmark_payload = load_json(paths.benchmark_result)
    config = scorer_config_payload(
        paths, benchmark_payload, benchmark_result, masking_hash
    )
    texts, authors = load_training_rows(paths, benchmark_result)

    vectorizer = make_vectorizer()
    matrix = vectorizer.fit_transform(texts)
    if matrix.shape != (len(texts), MAX_CHAR_FEATURES):
        raise EvaluationError(
            f"Exact vectorizer shape mismatch: {matrix.shape}; expected ({len(texts)}, {MAX_CHAR_FEATURES})"
        )
    classifier = make_classifier()
    classifier.fit(matrix, authors)
    labels = tuple(str(value) for value in classifier.classes_)
    if labels != tuple(str(value) for value in benchmark_result["labels"]):
        raise EvaluationError("Fitted classifier labels differ from exact benchmark labels")
    if DEFAULT_TARGET_AUTHOR not in labels:
        raise EvaluationError(f"Target author is absent from classifier labels: {DEFAULT_TARGET_AUTHOR}")

    paths.scorer_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{paths.scorer_dir.name}.", dir=paths.scorer_dir.parent)
    )
    try:
        write_json(temp_dir / SCORER_FILES["masking"], masking_payload)
        write_json(temp_dir / SCORER_FILES["config"], config)
        labels_payload = {
            "schema_version": 1,
            "labels": list(labels),
            "target_author": DEFAULT_TARGET_AUTHOR,
            "target_index": labels.index(DEFAULT_TARGET_AUTHOR),
        }
        write_json(temp_dir / SCORER_FILES["labels"], labels_payload)
        joblib.dump(vectorizer, temp_dir / SCORER_FILES["vectorizer"], compress=3)
        joblib.dump(classifier, temp_dir / SCORER_FILES["classifier"], compress=3)

        hashes = {
            key: file_sha256(temp_dir / filename)
            for key, filename in SCORER_FILES.items()
            if key != "manifest"
        }
        manifest = {
            "schema_version": 1,
            "scorer_id": SCORER_ID,
            "files": {
                key: {"path": SCORER_FILES[key], "sha256": value}
                for key, value in sorted(hashes.items())
            },
            "vectorizer_artifact_sha256": hashes["vectorizer"],
            "classifier_artifact_sha256": hashes["classifier"],
            "labels_artifact_sha256": hashes["labels"],
            "scorer_config_sha256": hashes["config"],
            "masking_artifact_sha256": hashes["masking"],
            "artifact_set_sha256": sha256_text(canonical_json(hashes)),
            "fit_summary": {
                "train_chunks": len(texts),
                "train_books_only": True,
                "feature_count": int(matrix.shape[1]),
                "label_count": len(labels),
                "classifier_iterations": int(classifier.n_iter_),
            },
        }
        write_json(temp_dir / SCORER_FILES["manifest"], manifest)
        os.replace(temp_dir, paths.scorer_dir)
    finally:
        if temp_dir.exists():
            for child in temp_dir.iterdir():
                child.unlink()
            temp_dir.rmdir()
    return load_frozen_scorer(paths)


def verify_manifest_files(directory: Path, manifest: Mapping[str, Any]) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise EvaluationError(f"Scorer manifest has no files map: {directory}")
    for key in ("vectorizer", "classifier", "labels", "config", "masking"):
        record = files.get(key)
        if not isinstance(record, dict):
            raise EvaluationError(f"Scorer manifest is missing {key}")
        path = directory / str(record.get("path", ""))
        expected = str(record.get("sha256", ""))
        if not SHA256_RE.fullmatch(expected):
            raise EvaluationError(f"Scorer manifest has invalid {key} hash")
        if file_sha256(require_file(path, f"frozen scorer {key}")) != expected:
            raise EvaluationError(f"Frozen scorer {key} hash mismatch: {path}")


def load_frozen_scorer(paths: ScorerPaths) -> ScorerBundle:
    directory = paths.scorer_dir
    manifest = load_json(require_file(directory / SCORER_FILES["manifest"], "scorer manifest"))
    if manifest.get("scorer_id") != SCORER_ID:
        raise EvaluationError(f"Unexpected scorer ID in {directory}")
    verify_manifest_files(directory, manifest)
    _, benchmark_result, expected_masking, expected_masking_hash = expected_scorer_payloads(paths)
    expected_config = scorer_config_payload(
        paths,
        load_json(paths.benchmark_result),
        benchmark_result,
        expected_masking_hash,
    )
    actual_masking = load_json(directory / SCORER_FILES["masking"])
    actual_config = load_json(directory / SCORER_FILES["config"])
    if actual_masking != expected_masking:
        raise EvaluationError("Frozen masking config no longer matches its bound inputs")
    if actual_config != expected_config:
        raise EvaluationError("Frozen scorer config no longer matches its bound inputs/environment")
    labels_payload = load_json(directory / SCORER_FILES["labels"])
    labels = tuple(str(value) for value in labels_payload.get("labels", []))
    vectorizer = joblib.load(directory / SCORER_FILES["vectorizer"])
    classifier = joblib.load(directory / SCORER_FILES["classifier"])
    if not isinstance(vectorizer, TfidfVectorizer):
        raise EvaluationError("Frozen vectorizer has the wrong type")
    if not isinstance(classifier, SGDClassifier):
        raise EvaluationError("Frozen classifier has the wrong type")
    if tuple(str(value) for value in classifier.classes_) != labels:
        raise EvaluationError("Frozen classifier classes disagree with labels.json")
    if labels_payload.get("target_author") != DEFAULT_TARGET_AUTHOR:
        raise EvaluationError("Frozen target author is incorrect")
    if int(labels_payload.get("target_index", -1)) != labels.index(DEFAULT_TARGET_AUTHOR):
        raise EvaluationError("Frozen target label index is incorrect")
    return ScorerBundle(
        directory=directory,
        vectorizer=vectorizer,
        classifier=classifier,
        labels=labels,
        config=actual_config,
        manifest=manifest,
    )


def get_or_create_scorer(args: argparse.Namespace) -> tuple[ScorerBundle, str]:
    paths = scorer_paths_from_args(args)
    mode = args.scorer_mode
    if mode == "fit":
        return fit_and_freeze_scorer(paths), "fitted"
    if mode == "load":
        return load_frozen_scorer(paths), "loaded"
    if paths.scorer_dir.exists():
        return load_frozen_scorer(paths), "loaded"
    return fit_and_freeze_scorer(paths), "fitted"


def verify_scorer_performance(args: argparse.Namespace) -> dict[str, Any]:
    bundle, action = get_or_create_scorer(args)
    paths = scorer_paths_from_args(args)
    _, benchmark_result = validate_benchmark_result(paths.benchmark_result)
    texts: list[str] = []
    labels: list[str] = []
    for row in iter_jsonl(paths.masked_chunks):
        if row.get("split") != "test":
            continue
        text = row.get("text")
        author = row.get("author")
        if not isinstance(text, str) or not isinstance(author, str):
            raise EvaluationError("Test chunk has invalid text or author")
        texts.append(text)
        labels.append(author)
    expected = benchmark_result["chunk_metrics"]["test"]
    if len(texts) != int(expected["rows"]):
        raise EvaluationError(
            f"Test row count mismatch: {len(texts)} != {expected['rows']}"
        )
    predictions = bundle.classifier.predict(bundle.vectorizer.transform(texts))
    target_true = np.asarray(labels) == DEFAULT_TARGET_AUTHOR
    target_pred = np.asarray(predictions) == DEFAULT_TARGET_AUTHOR
    target_tp = int(np.sum(target_true & target_pred))
    target_fp = int(np.sum(~target_true & target_pred))
    target_fn = int(np.sum(target_true & ~target_pred))
    target_precision = target_tp / (target_tp + target_fp) if target_tp + target_fp else 0.0
    target_recall = target_tp / (target_tp + target_fn) if target_tp + target_fn else 0.0
    target_f1 = (
        2 * target_precision * target_recall / (target_precision + target_recall)
        if target_precision + target_recall
        else 0.0
    )
    observed = {
        "rows": len(texts),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "target": {
            "tp": target_tp,
            "fp": target_fp,
            "fn": target_fn,
            "precision": target_precision,
            "recall": target_recall,
            "f1": target_f1,
        },
    }
    differences = {
        key: float(observed[key]) - float(expected[key])
        for key in ("accuracy", "balanced_accuracy", "macro_f1")
    }
    target_differences = {
        key: float(observed["target"][key]) - float(expected["target"][key])
        for key in ("precision", "recall", "f1")
    }
    exact_counts = all(
        observed["target"][key] == expected["target"][key]
        for key in ("tp", "fp", "fn")
    )
    passed = (
        exact_counts
        and all(abs(value) <= 1e-12 for value in differences.values())
        and all(abs(value) <= 1e-12 for value in target_differences.values())
    )
    result = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "scorer_action": action,
        "scorer_binding": binding_fields(bundle),
        "benchmark_run_key": BENCHMARK_RUN_KEY,
        "observed": observed,
        "expected": {
            key: expected[key]
            for key in ("rows", "accuracy", "balanced_accuracy", "macro_f1", "target")
        },
        "differences": differences,
        "target_differences": target_differences,
        "exact_target_counts": exact_counts,
    }
    if args.output:
        write_json(args.output.resolve(), result)
    if not passed:
        raise EvaluationError(
            "Frozen scorer does not reproduce the Stage 1 masked test metrics"
        )
    return result


def score_texts(bundle: ScorerBundle, texts: Sequence[str]) -> list[dict[str, Any]]:
    if not texts:
        return []
    matrix = bundle.vectorizer.transform(texts)
    margins = bundle.classifier.decision_function(matrix)
    if margins.ndim != 2 or margins.shape[1] != len(bundle.labels):
        raise EvaluationError(f"Unexpected decision-function shape: {margins.shape}")
    target_index = bundle.labels.index(bundle.target_author)
    rows: list[dict[str, Any]] = []
    for score_vector in margins:
        values = [float(value) for value in score_vector]
        order = sorted(
            range(len(values)), key=lambda index: (-values[index], bundle.labels[index])
        )
        predicted_index = order[0]
        target_rank = order.index(target_index) + 1
        rows.append(
            {
                "target_margin": values[target_index],
                "target_rank": target_rank,
                "predicted_author": bundle.labels[predicted_index],
                "target_chunk_indicator": int(predicted_index == target_index),
            }
        )
    return rows


class EntityMasker:
    def __init__(self, path: Path):
        payload = load_json(require_file(path, "mask terms"))
        if payload.get("masking_policy") != MASKING_POLICY_VERSION:
            raise EvaluationError(f"Stale masking policy in {path}")
        global_terms = tuple(
            str(value)
            for value in payload.get("global_terms", {}).get("entity_terms_v2", [])
        )
        self._global_matcher = compile_term_matcher(list(global_terms))

    def mask(self, text: str, *, author: str, title: str) -> str:
        masked = text
        for placeholder, sentinel in PLACEHOLDER_SENTINELS.items():
            masked = masked.replace(placeholder, sentinel)
        masked = LATIN_RE.sub("<LATIN>", masked)
        masked = NUMBER_RE.sub("<NUM>", masked)
        masked = mask_terms(
            masked,
            self._global_matcher,
            placeholder="某",
            preserve_length=True,
        )
        for placeholder, sentinel in PLACEHOLDER_SENTINELS.items():
            masked = masked.replace(sentinel, placeholder)
        masked = re.sub(r"<+(CONTENT|NUM|LATIN)>+", r"<\1>", masked)
        return masked


def sample_paths(experiment_root: Path, sample_set: str) -> dict[str, Path]:
    root = experiment_root / "sample_sets"
    return {
        "runner": root / f"{sample_set}.runner_manifest.jsonl",
        "allocation": root / f"{sample_set}.evaluator_allocation.jsonl",
        "hidden": root / f"{sample_set}.hidden_targets.jsonl",
        "summary": root / f"{sample_set}.summary.json",
        "method_ids": root / f"{sample_set}.method_evaluation_ids.json",
        "screening_ids": root / f"{sample_set}.screening_v1_ids.json",
        "confirmation_ids": root / f"{sample_set}.confirmation_v1_ids.json",
        "final_validation_ids": root
        / f"{sample_set}.final_validation_v1_ids.json",
    }


def unique_by_id(rows: Iterable[dict[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise EvaluationError(f"Missing sample_id in {path}")
        if sample_id in result:
            raise EvaluationError(f"Duplicate sample_id {sample_id} in {path}")
        result[sample_id] = row
    return result


def load_sample_contract(
    experiment_root: Path, sample_set: str
) -> dict[str, Any]:
    paths = sample_paths(experiment_root, sample_set)
    for label in ("runner", "allocation", "hidden", "summary", "method_ids"):
        path = paths[label]
        require_file(path, f"sample-set {label}")
    summary = load_json(paths["summary"])
    allocation = unique_by_id(iter_jsonl(paths["allocation"]), paths["allocation"])
    hidden = unique_by_id(iter_jsonl(paths["hidden"]), paths["hidden"])
    runner = unique_by_id(iter_jsonl(paths["runner"]), paths["runner"])
    method_ids_payload = load_json(paths["method_ids"])
    method_ids = tuple(str(value) for value in method_ids_payload.get("sample_ids", []))
    if len(method_ids) != len(set(method_ids)):
        raise EvaluationError("Frozen method-evaluation ID artifact contains duplicates")
    if set(allocation) != set(hidden) or set(allocation) != set(runner):
        raise EvaluationError("Runner, allocation, and hidden-target sample IDs disagree")
    allowed_roles = set(
        str(value) for value in method_ids_payload.get("allowed_research_roles", [])
    )
    if not allowed_roles:
        raise EvaluationError("Frozen method-evaluation artifact has no allowed roles")
    expected_method_ids = {
        sample_id
        for sample_id, row in allocation.items()
        if row.get("research_role") in allowed_roles
    }
    if set(method_ids) != expected_method_ids:
        raise EvaluationError("Frozen method IDs disagree with evaluator allocation")
    if int(method_ids_payload.get("sample_count", -1)) != len(method_ids):
        raise EvaluationError("Frozen method ID count is incorrect")
    if summary.get("sample_set") != sample_set:
        raise EvaluationError("Sample summary has a different sample_set")
    if int(summary.get("total_samples", -1)) != len(allocation):
        raise EvaluationError("Sample summary total does not match allocation")
    for summary_key, path_key in (
        ("runner_manifest_sha256", "runner"),
        ("evaluator_allocation_sha256", "allocation"),
        ("hidden_targets_sha256", "hidden"),
    ):
        expected = summary.get(summary_key)
        if expected and canonical_jsonl_sha256(paths[path_key]) != expected:
            raise EvaluationError(f"Sample summary binding mismatch: {summary_key}")
    expected_method_ids_hash = summary.get("method_evaluation_ids_sha256")
    if expected_method_ids_hash and file_sha256(paths["method_ids"]) != expected_method_ids_hash:
        raise EvaluationError(
            "Sample summary binding mismatch: method_evaluation_ids_sha256"
        )
    frozen_selections: dict[str, dict[str, Any]] = {}
    selection_specs: list[tuple[str, str, str]]
    if paths["final_validation_ids"].exists():
        if paths["screening_ids"].exists() or paths["confirmation_ids"].exists():
            raise EvaluationError(
                "Final-validation sample set must not contain development selections"
            )
        selection_specs = [
            (
                "final_validation_v1",
                "final_validation_ids",
                "final_validation_ids_sha256",
            )
        ]
    else:
        require_file(paths["screening_ids"], "frozen screening selection")
        require_file(paths["confirmation_ids"], "frozen confirmation selection")
        selection_specs = [
            ("screening_v1", "screening_ids", "screening_ids_sha256"),
            ("confirmation_v1", "confirmation_ids", "confirmation_ids_sha256"),
        ]
    for selection_id, path_key, summary_hash_key in selection_specs:
        selection_path = paths[path_key]
        selection = load_json(selection_path)
        selection_ids = tuple(sorted(str(value) for value in selection.get("sample_ids", [])))
        if selection.get("selection_id") != selection_id:
            raise EvaluationError(f"Frozen {selection_id} artifact has the wrong ID")
        if len(selection_ids) != len(set(selection_ids)) or not selection_ids:
            raise EvaluationError(f"Frozen {selection_id} IDs are empty or duplicated")
        if not set(selection_ids).issubset(set(method_ids)):
            raise EvaluationError(f"Frozen {selection_id} contains non-method IDs")
        if int(selection.get("sample_count", -1)) != len(selection_ids):
            raise EvaluationError(f"Frozen {selection_id} sample count is incorrect")
        expected_hash = summary.get(summary_hash_key)
        if not isinstance(expected_hash, str) or file_sha256(selection_path) != expected_hash:
            raise EvaluationError(f"Sample summary binding mismatch: {summary_hash_key}")
        frozen_selections[selection_id] = {
            "selection_id": selection_id,
            "path": selection_path,
            "sha256": expected_hash,
            "sample_ids": selection_ids,
        }
    if "screening_v1" in frozen_selections:
        if set(frozen_selections["screening_v1"]["sample_ids"]) & set(
            frozen_selections["confirmation_v1"]["sample_ids"]
        ):
            raise EvaluationError("Frozen screening and confirmation cohorts overlap")
        if set(frozen_selections["screening_v1"]["sample_ids"]) | set(
            frozen_selections["confirmation_v1"]["sample_ids"]
        ) != set(method_ids):
            raise EvaluationError("Screening plus confirmation does not equal method cohort")
        expected_composition = {
            "screening_v1": {
                "own_author_reconstruction": 24,
                "cross_author_transfer": 12,
            },
            "confirmation_v1": {
                "own_author_reconstruction": 104,
                "cross_author_transfer": 48,
            },
        }
        expected_per_book = {"screening_v1": 3, "confirmation_v1": 13}
        expected_per_author = {"screening_v1": 1, "confirmation_v1": 4}
        for selection_id, expected_arms in expected_composition.items():
            selected_rows = [
                allocation[sample_id]
                for sample_id in frozen_selections[selection_id]["sample_ids"]
            ]
            observed_arms = Counter(str(row["benchmark_arm"]) for row in selected_rows)
            if observed_arms != Counter(expected_arms):
                raise EvaluationError(f"Frozen {selection_id} arm composition mismatch")
            own_books = Counter(
                str(row["book_title"])
                for row in selected_rows
                if row["benchmark_arm"] == "own_author_reconstruction"
            )
            cross_authors = Counter(
                str(row["author"])
                for row in selected_rows
                if row["benchmark_arm"] == "cross_author_transfer"
            )
            if len(own_books) != 8 or set(own_books.values()) != {
                expected_per_book[selection_id]
            }:
                raise EvaluationError(f"Frozen {selection_id} per-book composition mismatch")
            if len(cross_authors) != 12 or set(cross_authors.values()) != {
                expected_per_author[selection_id]
            }:
                raise EvaluationError(f"Frozen {selection_id} per-author composition mismatch")
    elif set(frozen_selections["final_validation_v1"]["sample_ids"]) != set(
        method_ids
    ):
        raise EvaluationError("Final-validation selection does not equal method cohort")
    else:
        final_rows = [allocation[sample_id] for sample_id in method_ids]
        if {str(row["benchmark_arm"]) for row in final_rows} != {
            "own_author_reconstruction"
        }:
            raise EvaluationError("Final-validation arm composition mismatch")
        final_books = Counter(str(row["book_title"]) for row in final_rows)
        if set(final_books) != set(FINAL_VALIDATION_BOOKS) or set(
            final_books.values()
        ) != {20}:
            raise EvaluationError("Final-validation per-book composition mismatch")
        if {str(row.get("author", "")) for row in final_rows} != {
            DEFAULT_TARGET_AUTHOR
        }:
            raise EvaluationError("Final-validation target-author contract mismatch")
        if {str(row.get("source_split", "")) for row in final_rows} != {"test"}:
            raise EvaluationError("Final-validation source-split contract mismatch")
        if {str(row.get("research_role", "")) for row in final_rows} != {
            "final_validation"
        }:
            raise EvaluationError("Final-validation research-role contract mismatch")
        if set(str(value) for value in summary.get("books", [])) != set(
            FINAL_VALIDATION_BOOKS
        ):
            raise EvaluationError("Final-validation summary book contract mismatch")
    protocol_path = experiment_root / "protocols/evaluation_protocol.v1.json"
    protocol = load_json(require_file(protocol_path, "evaluation protocol"))
    calibration_ids = tuple(
        sorted(
            sample_id
            for sample_id, row in allocation.items()
            if row.get("research_role") == protocol["calibration"]["allowed_role"]
        )
    )
    if "final_validation_v1" not in frozen_selections and len(
        calibration_ids
    ) != int(protocol["calibration"]["sample_count"]):
        raise EvaluationError("Calibration allocation count differs from protocol")
    return {
        "paths": paths,
        "summary": summary,
        "allocation": allocation,
        "hidden": hidden,
        "runner": runner,
        "method_ids": tuple(sorted(method_ids)),
        "frozen_selections": frozen_selections,
        "calibration_ids": calibration_ids,
        "protocol": protocol,
        "protocol_path": protocol_path,
    }


def run_root(experiment_root: Path, sample_set: str, run_id: str) -> Path:
    return experiment_root / "runs" / sample_set / run_id


def load_success_ledger(path: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    if not path.exists():
        return {}, [f"missing_ledger:{display_path(path)}"]
    successful: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for row in iter_jsonl(path):
        if row.get("status") != "success":
            continue
        sample_id = str(row.get("sample_id", ""))
        if sample_id in successful:
            errors.append(f"duplicate_success_ledger_row:{sample_id}")
        successful[sample_id] = row
    return successful, errors


def validate_run_config(
    path: Path,
    *,
    sample_set: str,
    run_id: str,
    stage: str,
    method_id: str | None = None,
    intensity: str | None = None,
    analysis_lock_binding: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    expected = {
        "sample_set": sample_set,
        "run_id": run_id,
        "stage": stage,
    }
    if method_id is not None:
        expected["method_id"] = method_id
    if intensity is not None:
        expected["intensity"] = intensity
    config, errors = validate_shared_run_config(
        path,
        expected=expected,
        expected_analysis_lock=analysis_lock_binding,
    )
    if config is None:
        return None, errors
    model_config_sha = str(config.get("model_config_sha256", ""))
    run_root_path = path.parent.parent
    snapshot_value = config.get("model_config_snapshot_path")
    snapshot_path = (
        resolve_recorded_path(snapshot_value)
        if isinstance(snapshot_value, str) and snapshot_value
        else run_root_path
        / "config_snapshots"
        / f"model_run_config.{model_config_sha}.json"
    )
    if not snapshot_path.exists() or file_sha256(snapshot_path) != model_config_sha:
        errors.append("run_config_model_snapshot_missing_or_mismatched")
    else:
        snapshot = load_json(snapshot_path)
        for field, snapshot_field in (
            ("model", "codex_model"),
            ("reasoning_effort", "reasoning_effort"),
            ("provider", "provider"),
        ):
            if config.get(field) != snapshot.get(snapshot_field):
                errors.append(f"run_config_{field}_snapshot_mismatch")
    for label in ("prompt", "schema"):
        expected_sha = str(config.get(f"{label}_sha256", ""))
        snapshot_value = config.get(f"{label}_snapshot_path")
        source_value = config.get(f"{label}_path")
        candidates: list[Path] = []
        if isinstance(snapshot_value, str) and snapshot_value:
            candidates.append(resolve_recorded_path(snapshot_value))
        if isinstance(source_value, str) and source_value:
            candidates.append(resolve_recorded_path(source_value))
        candidates.extend(
            candidate
            for candidate in sorted(
                run_root_path.glob(f"config_snapshots/**/*{expected_sha}*")
            )
            if candidate.is_file()
        )
        if not any(
            candidate.exists() and file_sha256(candidate) == expected_sha
            for candidate in candidates
        ):
            errors.append(f"run_config_{label}_snapshot_missing_or_mismatched")
    runner_value = config.get("runner_snapshot_path")
    runner_path = (
        resolve_recorded_path(runner_value)
        if isinstance(runner_value, str) and runner_value
        else None
    )
    if (
        runner_path is None
        or not runner_path.exists()
        or file_sha256(runner_path) != config.get("runner_sha256")
    ):
        errors.append("run_config_runner_snapshot_missing_or_mismatched")
    return config, errors


def validate_execution_binding_record(
    record: Mapping[str, Any],
    *,
    sample_id: str,
    analysis_lock_binding: Mapping[str, Any] | None = None,
) -> list[str]:
    if not EXECUTION_BINDING_FIELDS.intersection(record):
        return []
    return validate_shared_execution_binding(
        record,
        sample_id=sample_id,
        expected_analysis_lock=analysis_lock_binding,
    )


def validate_output_artifact(
    path: Path,
    *,
    sample_id: str,
    run_id: str,
    stage: str,
    ledger_row: Mapping[str, Any] | None,
    method_id: str | None = None,
    intensity: str | None = None,
    analysis_lock_binding: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[str]]:
    if not path.exists():
        return None, None, ["missing_output"]
    try:
        artifact = load_json(path)
    except EvaluationError as exc:
        return None, None, [f"invalid_output_json:{exc}"]
    errors: list[str] = []
    expected = {"sample_id": sample_id, "run_id": run_id, "stage": stage}
    if method_id is not None:
        expected["method_id"] = method_id
    if intensity is not None:
        expected["intensity"] = intensity
    for key, value in expected.items():
        if artifact.get(key) != value:
            errors.append(f"artifact_{key}_mismatch")
    result = artifact.get("result")
    if not isinstance(result, dict):
        errors.append("artifact_result_missing")
        return artifact, None, errors
    output_hash = sha256_text(canonical_json(result))
    if artifact.get("output_sha256") != output_hash:
        errors.append("artifact_output_sha256_mismatch")
    if result.get("sample_id") != sample_id:
        errors.append("result_sample_id_mismatch")
    if method_id is not None and result.get("method_id") != method_id:
        errors.append("result_method_id_mismatch")
    if intensity is not None and result.get("intensity") != intensity:
        errors.append("result_intensity_mismatch")
    if ledger_row is None:
        errors.append("missing_success_ledger_row")
    else:
        for key in ("run_id", "stage", "sample_id", "output_sha256"):
            expected_value = artifact.get(key) if key != "output_sha256" else output_hash
            if ledger_row.get(key) != expected_value:
                errors.append(f"ledger_{key}_mismatch")
        if method_id is not None and ledger_row.get("method_id") != method_id:
            errors.append("ledger_method_id_mismatch")
        if intensity is not None and ledger_row.get("intensity") != intensity:
            errors.append("ledger_intensity_mismatch")
        response_file = ledger_row.get("response_file")
        if not isinstance(response_file, str) or resolve_recorded_path(response_file).resolve() != path.resolve():
            errors.append("ledger_response_file_mismatch")
        artifact_binding_present = bool(EXECUTION_BINDING_FIELDS.intersection(artifact))
        ledger_binding_present = bool(
            EXECUTION_BINDING_FIELDS.intersection(ledger_row)
        )
        if artifact_binding_present != ledger_binding_present:
            errors.append("artifact_ledger_execution_binding_presence_mismatch")
        elif artifact_binding_present:
            for field in EXECUTION_BINDING_FIELDS:
                if artifact.get(field) != ledger_row.get(field):
                    errors.append(f"artifact_ledger_{field}_mismatch")
            errors.extend(
                f"artifact_{value}"
                for value in validate_execution_binding_record(
                    artifact,
                    sample_id=sample_id,
                    analysis_lock_binding=analysis_lock_binding,
                )
            )
            errors.extend(
                f"ledger_{value}"
                for value in validate_execution_binding_record(
                    ledger_row,
                    sample_id=sample_id,
                    analysis_lock_binding=analysis_lock_binding,
                )
            )
            batch_value = artifact.get("execution_batch_config_path")
            if isinstance(batch_value, str):
                batch_path = resolve_recorded_path(batch_value)
                if batch_path.exists():
                    batch = load_json(batch_path)
                    run_config_value = batch.get("stage_run_config_path")
                    if isinstance(run_config_value, str):
                        shared_errors = validate_shared_artifact_and_ledger(
                            artifact_path=path,
                            artifact=artifact,
                            ledger_row=ledger_row,
                            run_config_path=resolve_recorded_path(run_config_value),
                            sample_id=sample_id,
                            run_id=run_id,
                            stage=stage,
                            expected_analysis_lock=analysis_lock_binding,
                        )
                        errors.extend(
                            f"shared_provenance_{value}" for value in shared_errors
                        )
    return artifact, result, errors


def validate_english_adjudication(
    *,
    root: Path,
    run_id: str,
    sample_id: str,
    repair_round: int,
    repair_path: Path,
    qa_path: Path,
    repair_artifact: Mapping[str, Any],
    qa_artifact: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    expected = {
        "run_id": run_id,
        "sample_id": sample_id,
        "repair_round": repair_round,
    }
    for label, artifact, stage in (
        ("repair", repair_artifact, "english_source_adjudication"),
        ("qa", qa_artifact, "english_source_adjudication_qa"),
    ):
        if artifact.get("stage") != stage:
            errors.append(f"adjudication_{label}_stage_mismatch")
        for field, value in expected.items():
            if artifact.get(field) != value:
                errors.append(f"adjudication_{label}_{field}_mismatch")
        result = artifact.get("result")
        if not isinstance(result, dict) or artifact.get("output_sha256") != sha256_text(
            canonical_json(result)
        ):
            errors.append(f"adjudication_{label}_output_hash_mismatch")
    if qa_artifact.get("approved_for_neutral_translation") is not True or qa_artifact.get(
        "result", {}
    ).get("approved") is not True:
        errors.append("adjudication_qa_not_approved")

    repair_provenance = repair_artifact.get("adjudication_provenance")
    qa_provenance = qa_artifact.get("adjudication_provenance")
    if not isinstance(repair_provenance, dict) or not isinstance(qa_provenance, dict):
        return sorted(set(errors + ["adjudication_provenance_missing"]))
    if repair_provenance.get("agent_id") != qa_provenance.get("agent_id"):
        errors.append("adjudication_agent_mismatch")
    if repair_provenance.get("verdict") != qa_provenance.get("verdict"):
        errors.append("adjudication_verdict_mismatch")
    audit_value = qa_provenance.get("audit_path")
    if not isinstance(audit_value, str):
        return sorted(set(errors + ["adjudication_audit_path_missing"]))
    audit_path = resolve_recorded_path(audit_value)
    if not audit_path.is_file():
        return sorted(set(errors + ["adjudication_audit_missing"]))
    audit = load_json(audit_path)
    audit_expected = {
        "status": "approved_by_independent_adjudication",
        "sample_id": sample_id,
        "agent_id": repair_provenance.get("agent_id"),
        "verdict": repair_provenance.get("verdict"),
        "frozen_repair_path": display_path(repair_path),
        "frozen_repair_sha256": file_sha256(repair_path),
        "frozen_qa_path": display_path(qa_path),
        "frozen_qa_sha256": file_sha256(qa_path),
        "source_repair_path": repair_provenance.get("source_repair_path"),
        "source_repair_sha256": repair_provenance.get("source_repair_sha256"),
    }
    for field, value in audit_expected.items():
        if audit.get(field) != value:
            errors.append(f"adjudication_audit_{field}_mismatch")
    source_value = repair_provenance.get("source_repair_path")
    if isinstance(source_value, str):
        source_path = resolve_recorded_path(source_value)
        if not source_path.is_file() or file_sha256(source_path) != repair_provenance.get(
            "source_repair_sha256"
        ):
            errors.append("adjudication_source_repair_hash_mismatch")
    else:
        errors.append("adjudication_source_repair_path_missing")
    contradictory_paths = audit.get("contradictory_qa_paths")
    contradictory_hashes = audit.get("contradictory_qa_sha256")
    if not isinstance(contradictory_paths, list) or not isinstance(
        contradictory_hashes, list
    ) or len(contradictory_paths) != len(contradictory_hashes):
        errors.append("adjudication_contradictory_qa_binding_invalid")
    else:
        for value, expected_hash in zip(contradictory_paths, contradictory_hashes):
            path = resolve_recorded_path(str(value))
            if not path.is_file() or file_sha256(path) != expected_hash:
                errors.append("adjudication_contradictory_qa_hash_mismatch")
    ledger_path = root / "ledgers" / f"english_source_adjudication.round_{repair_round:02d}.jsonl"
    try:
        ledger_rows = list(iter_jsonl(ledger_path))
    except (FileNotFoundError, ValueError):
        ledger_rows = []
    if len(ledger_rows) != 1 or ledger_rows[0] != audit:
        errors.append("adjudication_ledger_audit_mismatch")
    return sorted(set(errors))


def load_effective_english_result(
    root: Path, sample_set: str, run_id: str, sample_id: str
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any] | None]:
    errors: list[str] = []

    def load_stage(
        stage: str,
        *,
        output_dir: Path | None = None,
        stem: str | None = None,
        repair_round: int | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        artifact_stem = stem or stage
        ledger, ledger_errors = load_success_ledger(
            root / "ledgers" / f"{artifact_stem}.jsonl"
        )
        errors.extend(ledger_errors)
        config, config_errors = validate_run_config(
            root / "run_configs" / f"{artifact_stem}.json",
            sample_set=sample_set,
            run_id=run_id,
            stage=stage,
        )
        errors.extend(config_errors)
        if repair_round is not None and config is not None and config.get(
            "repair_round"
        ) != repair_round:
            errors.append(f"{artifact_stem}:run_config_repair_round_mismatch")
        artifact, result, artifact_errors = validate_output_artifact(
            (output_dir or root / stage) / f"{sample_id}.json",
            sample_id=sample_id,
            run_id=run_id,
            stage=stage,
            ledger_row=ledger.get(sample_id),
        )
        if repair_round is not None and artifact is not None and artifact.get(
            "repair_round"
        ) != repair_round:
            artifact_errors.append("artifact_repair_round_mismatch")
        errors.extend(f"{stage}:{value}" for value in artifact_errors)
        return artifact, result

    source_artifact, source_result = load_stage("english_semantic_source")
    qa_artifact, qa_result = load_stage("english_source_qa")
    if (
        source_artifact is None
        or source_result is None
        or qa_artifact is None
        or qa_result is None
    ):
        return [], sorted(set(errors)), None
    qa_approved = qa_artifact.get("approved_for_neutral_translation")
    if qa_approved is not qa_result.get("approved") or not isinstance(qa_approved, bool):
        errors.append("english_source_qa:approval_state_mismatch")
        return [], sorted(set(errors)), None
    selected_result = source_result
    selected_artifact = source_artifact
    selected_stage = "english_semantic_source"
    selected_round: int | None = None
    if not qa_approved:
        found_round = False
        for repair_round in range(1, 21):
            round_name = f"round_{repair_round:02d}"
            repair_output = root / "english_source_repair" / round_name
            qa_output = root / "english_source_repair_qa" / round_name
            repair_path = repair_output / f"{sample_id}.json"
            qa_path = qa_output / f"{sample_id}.json"
            if not repair_path.exists() and not qa_path.exists():
                break
            found_round = True
            if not repair_path.exists() or not qa_path.exists():
                errors.append(f"english_source_repair.{round_name}:incomplete_pair")
                return [], sorted(set(errors)), None
            raw_repair = load_json(repair_path)
            raw_qa = load_json(qa_path)
            if (
                raw_repair.get("stage") == "english_source_adjudication"
                or raw_qa.get("stage") == "english_source_adjudication_qa"
            ):
                errors.extend(
                    validate_english_adjudication(
                        root=root,
                        run_id=run_id,
                        sample_id=sample_id,
                        repair_round=repair_round,
                        repair_path=repair_path,
                        qa_path=qa_path,
                        repair_artifact=raw_repair,
                        qa_artifact=raw_qa,
                    )
                )
                repair_artifact = raw_repair
                repair_result = raw_repair.get("result")
                repair_qa_artifact = raw_qa
                repair_qa_result = raw_qa.get("result")
            else:
                repair_artifact, repair_result = load_stage(
                    "english_source_repair",
                    output_dir=repair_output,
                    stem=f"english_source_repair.{round_name}",
                    repair_round=repair_round,
                )
                repair_qa_artifact, repair_qa_result = load_stage(
                    "english_source_repair_qa",
                    output_dir=qa_output,
                    stem=f"english_source_repair_qa.{round_name}",
                    repair_round=repair_round,
                )
            if (
                repair_artifact is None
                or repair_result is None
                or repair_qa_artifact is None
                or repair_qa_result is None
            ):
                return [], sorted(set(errors)), None
            repair_approved = repair_qa_artifact.get(
                "approved_for_neutral_translation"
            )
            if repair_approved is not repair_qa_result.get("approved"):
                errors.append(
                    f"english_source_repair_qa.{round_name}:approval_state_mismatch"
                )
                return [], sorted(set(errors)), None
            if repair_approved is True:
                selected_result = repair_result
                selected_artifact = repair_artifact
                selected_stage = f"english_source_repair.{round_name}"
                selected_round = repair_round
                break
        if selected_round is None:
            errors.append(
                "english_source_repair_qa:no_approved_round"
                if found_round
                else "english_source_repair:missing"
            )
            return [], sorted(set(errors)), None
    paragraphs = selected_result.get("paragraphs")
    if not isinstance(paragraphs, list):
        errors.append("invalid_effective_english_paragraphs")
        return [], sorted(set(errors)), None
    paragraph_rows = [dict(row) for row in paragraphs if isinstance(row, dict)]
    paragraph_ids = [row.get("id") for row in paragraph_rows]
    if len(paragraph_ids) != len(set(paragraph_ids)):
        errors.append("duplicate_effective_english_paragraph_ids")
    binding = {
        "selected_stage": selected_stage,
        "selected_repair_round": selected_round,
        "artifact_canonical_sha256": sha256_text(canonical_json(selected_artifact)),
        "output_sha256": selected_artifact.get("output_sha256"),
    }
    return paragraph_rows, sorted(set(errors)), binding


def load_neutral_contexts(
    experiment_root: Path,
    sample_set: str,
    run_id: str,
    sample_ids: Sequence[str],
) -> dict[str, GeneratedContext]:
    root = run_root(experiment_root, sample_set, run_id)
    ledger_path = root / "ledgers/neutral_translation.jsonl"
    ledger, ledger_errors = load_success_ledger(ledger_path)
    _, config_errors = validate_run_config(
        root / "run_configs/neutral_translation.json",
        sample_set=sample_set,
        run_id=run_id,
        stage="neutral_translation",
    )
    contexts: dict[str, GeneratedContext] = {}
    for sample_id in sample_ids:
        errors = list(ledger_errors) + list(config_errors)
        path = root / "neutral_translation" / f"{sample_id}.json"
        artifact, result, artifact_errors = validate_output_artifact(
            path,
            sample_id=sample_id,
            run_id=run_id,
            stage="neutral_translation",
            ledger_row=ledger.get(sample_id),
        )
        errors.extend(artifact_errors)
        english, english_errors, english_binding = load_effective_english_result(
            root, sample_set, run_id, sample_id
        )
        errors.extend(english_errors)
        request = {
            "sample_id": sample_id,
            "paragraphs": english,
            "glossary": {},
            "comments": [],
        }
        if artifact is not None and artifact.get("input_sha256") != sha256_text(canonical_json(request)):
            errors.append("neutral_input_sha256_mismatch")
        if ledger.get(sample_id) is not None and ledger[sample_id].get("input_sha256") != sha256_text(canonical_json(request)):
            errors.append("neutral_ledger_input_sha256_mismatch")
        paragraphs = result.get("paragraphs") if isinstance(result, dict) else []
        if not isinstance(paragraphs, list):
            errors.append("neutral_paragraphs_not_list")
            paragraphs = []
        paragraph_rows = [dict(row) for row in paragraphs if isinstance(row, dict)]
        expected_ids = [row.get("id") for row in english]
        actual_ids = [row.get("id") for row in paragraph_rows]
        if actual_ids != expected_ids:
            errors.append("neutral_paragraph_id_or_order_mismatch")
        if len(actual_ids) != len(set(actual_ids)):
            errors.append("neutral_duplicate_paragraph_ids")
        if any(not isinstance(row.get("zh"), str) or not row["zh"].strip() for row in paragraph_rows):
            errors.append("neutral_empty_paragraph")
        contexts[sample_id] = GeneratedContext(
            sample_id=sample_id,
            result=result,
            paragraphs=paragraph_rows,
            text=paragraphs_to_text(paragraph_rows) if paragraph_rows else None,
            artifact=artifact,
            input_request=request,
            errors=sorted(set(errors)),
            provenance={"effective_english": english_binding},
        )
    return contexts


def load_masked_source_rows(path: Path, sample_ids: set[str]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        chunk_id = str(row.get("chunk_id", ""))
        if chunk_id in sample_ids:
            found[chunk_id] = row
    missing = sample_ids - set(found)
    if missing:
        raise EvaluationError(f"Masked source chunks are missing: {sorted(missing)[:5]}")
    return found


def binding_fields(bundle: ScorerBundle) -> dict[str, str]:
    return {
        "scorer_id": SCORER_ID,
        "classifier_artifact_sha256": bundle.classifier_artifact_sha256,
        "scorer_config_sha256": bundle.scorer_config_sha256,
        "masking_view": MASKING_VIEW,
        "masking_artifact_sha256": bundle.masking_artifact_sha256,
    }


def score_calibration(args: argparse.Namespace) -> dict[str, Any]:
    bundle, scorer_action = get_or_create_scorer(args)
    experiment_root = args.experiment_root.resolve()
    contract = load_sample_contract(experiment_root, args.sample_set)
    calibration_ids = contract["calibration_ids"]
    neutral = load_neutral_contexts(
        experiment_root, args.sample_set, args.run_id, calibration_ids
    )
    failures = {
        sample_id: context.errors
        for sample_id, context in neutral.items()
        if context.errors
    }
    if failures:
        first_id = sorted(failures)[0]
        raise EvaluationError(
            "Calibration scoring requires all frozen neutral outputs to pass provenance; "
            f"{len(failures)} failed, first={first_id}:{failures[first_id][:3]}"
        )

    source_by_chunk = load_masked_source_rows(
        scorer_paths_from_args(args).masked_chunks,
        {
            str(contract["allocation"][sample_id]["chunk_id"])
            for sample_id in calibration_ids
        },
    )
    masker = EntityMasker(scorer_paths_from_args(args).mask_terms)
    original_texts: list[str] = []
    neutral_masked_texts: list[str] = []
    for sample_id in calibration_ids:
        source_row = source_by_chunk[
            str(contract["allocation"][sample_id]["chunk_id"])
        ]
        original = str(source_row.get("text", ""))
        if not original:
            raise EvaluationError(f"Current global-masked source is empty: {sample_id}")
        allocation = contract["allocation"][sample_id]
        masked_neutral = masker.mask(
            neutral[sample_id].text or "",
            author=str(allocation["author"]),
            title=str(allocation["book_title"]),
        )
        original_texts.append(original)
        neutral_masked_texts.append(masked_neutral)

    original_scores = score_texts(bundle, original_texts)
    neutral_scores = score_texts(bundle, neutral_masked_texts)
    bindings = binding_fields(bundle)
    rows: list[dict[str, Any]] = []
    for index, sample_id in enumerate(calibration_ids):
        allocation = contract["allocation"][sample_id]
        original_score = original_scores[index]
        neutral_score = neutral_scores[index]
        rows.append(
            {
                "schema_version": 1,
                "sample_id": sample_id,
                "sample_set": args.sample_set,
                "research_role": allocation["research_role"],
                "benchmark_arm": allocation["benchmark_arm"],
                "author": allocation["author"],
                "book_title": allocation["book_title"],
                "chunk_id": allocation["chunk_id"],
                **bindings,
                "original_input_sha256": sha256_text(original_texts[index]),
                "neutral_input_sha256": sha256_text(neutral_masked_texts[index]),
                "neutral_raw_input_sha256": sha256_text(neutral[sample_id].text or ""),
                "original_target_margin": original_score["target_margin"],
                "original_target_rank": original_score["target_rank"],
                "original_predicted_author": original_score["predicted_author"],
                "original_target_chunk_indicator": original_score["target_chunk_indicator"],
                "neutral_target_margin": neutral_score["target_margin"],
                "neutral_target_rank": neutral_score["target_rank"],
                "neutral_predicted_author": neutral_score["predicted_author"],
                "neutral_target_chunk_indicator": neutral_score["target_chunk_indicator"],
            }
        )
    output = args.output.resolve()
    action = write_frozen_text(output, render_jsonl(rows))
    return {
        "status": "complete",
        "output": display_path(output),
        "output_sha256": file_sha256(output),
        "write_action": action,
        "scorer_action": scorer_action,
        "rows": len(rows),
        "original_target_chunk_share": sum(
            row["original_target_chunk_indicator"] for row in rows
        )
        / len(rows),
        "neutral_target_chunk_share": sum(
            row["neutral_target_chunk_indicator"] for row in rows
        )
        / len(rows),
        "bindings": bindings,
    }


_PAYLOAD_BUILDER: Any | None = None


def load_payload_builder() -> Any:
    global _PAYLOAD_BUILDER
    if _PAYLOAD_BUILDER is not None:
        return _PAYLOAD_BUILDER
    path = REPO_ROOT / "experiments/iteration1/style_transfer_payloads.py"
    if not path.exists():
        raise EvaluationError(
            "Cannot reconstruct style-transfer inputs because "
            f"{display_path(path)} is missing"
        )
    spec = importlib.util.spec_from_file_location("style_transfer_payloads", path)
    if spec is None or spec.loader is None:
        raise EvaluationError(f"Cannot import payload builder: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("style_transfer_payloads", module)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # The error is preserved as an evaluation failure.
        raise EvaluationError(f"Cannot import payload builder {path}: {exc}") from exc
    if not callable(getattr(module, "build_method_request", None)):
        raise EvaluationError("Payload builder has no build_method_request function")
    _PAYLOAD_BUILDER = module
    return module


def load_method_registry(experiment_root: Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    path = experiment_root / "method_registry/style_methods.v1.json"
    registry = load_json(require_file(path, "style method registry"))
    if registry.get("target_author") != DEFAULT_TARGET_AUTHOR:
        raise EvaluationError("Method registry target author differs from scorer target")
    for binding_key, expected_path in (
        ("evaluation_protocol", experiment_root / "protocols/evaluation_protocol.v1.json"),
        ("model_run_config", experiment_root / "protocols/model_run_config.v1.json"),
    ):
        binding = registry.get(binding_key)
        if not isinstance(binding, dict):
            raise EvaluationError(f"Method registry lacks {binding_key} binding")
        recorded_path = resolve_recorded_path(str(binding.get("path", ""))).resolve()
        if recorded_path != expected_path.resolve():
            raise EvaluationError(f"Method registry {binding_key} path mismatch")
        expected_hash = str(binding.get("sha256", ""))
        if file_sha256(require_file(recorded_path, binding_key)) != expected_hash:
            raise EvaluationError(f"Method registry {binding_key} hash mismatch")
    methods = registry.get("methods")
    if not isinstance(methods, list):
        raise EvaluationError("Method registry methods must be a list")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in methods:
        method_id = str(raw.get("id", ""))
        if not re.fullmatch(r"[a-z0-9_]+", method_id) or method_id in seen:
            raise EvaluationError(f"Invalid or duplicate method ID: {method_id!r}")
        seen.add(method_id)
        intensities = raw.get("intensities")
        if not isinstance(intensities, list) or not intensities:
            raise EvaluationError(f"Method has no frozen intensities: {method_id}")
        config_path = resolve_recorded_path(str(raw.get("config_path", "")))
        expected_hash = str(raw.get("config_sha256", ""))
        if file_sha256(require_file(config_path, f"method config {method_id}")) != expected_hash:
            raise EvaluationError(f"Method config hash mismatch: {method_id}")
        config = load_json(config_path)
        if config.get("method_id") != method_id:
            raise EvaluationError(f"Method config ID mismatch: {method_id}")
        if list(config.get("intensities", [])) != list(intensities):
            raise EvaluationError(f"Method intensity mismatch: {method_id}")
        normalized.append(
            {
                **raw,
                "method_id": method_id,
                "label": str(raw.get("label", method_id)),
                "intensities": tuple(str(value) for value in intensities),
                "config_path_resolved": config_path,
                "config": config,
            }
        )
    return path, registry, normalized


def select_method_combinations(
    methods: Sequence[dict[str, Any]], selectors: Sequence[str]
) -> list[dict[str, Any]]:
    by_id = {row["method_id"]: row for row in methods}
    requested: set[tuple[str, str]] | None = None
    if selectors:
        requested = set()
        for selector in selectors:
            method_id, separator, intensity = selector.partition(":")
            if method_id not in by_id:
                raise EvaluationError(f"Unknown --method selector: {selector}")
            available = by_id[method_id]["intensities"]
            if separator:
                if intensity not in available:
                    raise EvaluationError(
                        f"Unknown intensity in --method {selector}; available={available}"
                    )
                requested.add((method_id, intensity))
            else:
                requested.update((method_id, value) for value in available)
    combinations: list[dict[str, Any]] = []
    for method in methods:
        for intensity in method["intensities"]:
            key = (method["method_id"], intensity)
            if requested is not None and key not in requested:
                continue
            combinations.append({**method, "intensity": intensity})
    return combinations


def validate_screening_roster(
    *,
    contract: Mapping[str, Any],
    combinations: Sequence[Mapping[str, Any]],
    phase: str,
    refinement_contract_path: Path | None,
    analysis_lock_binding: Mapping[str, Any],
) -> dict[str, Any]:
    selection = contract.get("selection", {})
    selection_id = str(selection.get("selection_id", ""))
    if selection_id != "screening_v1":
        if phase != "initial" or refinement_contract_path is not None:
            raise EvaluationError(
                "Screening phase options apply only to official screening_v1"
            )
        return {"status": "not_applicable", "phase": None}
    observed = {
        (str(row["method_id"]), str(row["intensity"]))
        for row in combinations
    }
    if phase == "initial":
        if refinement_contract_path is not None:
            raise EvaluationError(
                "Initial screening cannot use a refinement contract"
            )
        roster = (
            contract.get("protocol", {})
            .get("staged_execution", {})
            .get("screening", {})
            .get("initial_method_intensities")
        )
        if not isinstance(roster, dict) or not roster:
            raise EvaluationError("Evaluation protocol has no initial screening roster")
        expected = {(str(method_id), str(intensity)) for method_id, intensity in roster.items()}
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise EvaluationError(
                "Initial screening combinations must equal the preregistered roster; "
                f"missing={missing}, extra={extra}"
            )
        return {
            "status": "valid",
            "phase": "initial",
            "roster_path": display_path(contract["protocol_path"]),
            "roster_sha256": file_sha256(contract["protocol_path"]),
            "combination_count": len(expected),
        }
    if phase != "refinement" or refinement_contract_path is None:
        raise EvaluationError(
            "Screening refinement requires --refinement-contract"
        )
    artifact = load_json(
        require_file(refinement_contract_path, "screening refinement contract")
    )
    try:
        require_matching_analysis_binding(
            artifact, analysis_lock_binding, label="screening refinement contract"
        )
    except ValueError as exc:
        raise EvaluationError(str(exc)) from exc
    if artifact.get("status") != "refinement_contract_frozen":
        raise EvaluationError("Screening refinement contract is not frozen")
    if artifact.get("selection_id") != "screening_v1" or artifact.get(
        "selection_sha256"
    ) != selection.get("sha256"):
        raise EvaluationError("Screening refinement selection binding mismatch")
    registry_path = DEFAULT_EXPERIMENT_ROOT / "method_registry/style_methods.v1.json"
    recorded_registry = artifact.get("method_registry_path")
    if isinstance(recorded_registry, str):
        registry_path = resolve_recorded_path(recorded_registry)
    if not registry_path.exists() or artifact.get(
        "method_registry_sha256"
    ) != file_sha256(registry_path):
        raise EvaluationError("Screening refinement registry binding mismatch")
    shortlist_value = artifact.get("provisional_shortlist_path")
    if not isinstance(shortlist_value, str):
        raise EvaluationError("Screening refinement shortlist path is missing")
    shortlist_path = resolve_recorded_path(shortlist_value)
    if not shortlist_path.exists() or artifact.get(
        "provisional_shortlist_sha256"
    ) != file_sha256(shortlist_path):
        raise EvaluationError("Screening refinement shortlist binding mismatch")
    try:
        require_matching_analysis_binding(
            load_json(shortlist_path),
            analysis_lock_binding,
            label="screening refinement shortlist",
        )
    except ValueError as exc:
        raise EvaluationError(str(exc)) from exc
    arms = artifact.get("evaluation_combinations")
    if not isinstance(arms, list) or not arms:
        raise EvaluationError("Screening refinement contract has no combinations")
    expected = {
        (str(row.get("method_id", "")), str(row.get("intensity", "")))
        for row in arms
        if isinstance(row, dict)
    }
    if observed != expected:
        raise EvaluationError(
            "Refinement evaluation combinations do not equal the frozen contract"
        )
    return {
        "status": "valid",
        "phase": "refinement",
        "path": display_path(refinement_contract_path),
        "sha256": file_sha256(refinement_contract_path),
        "combination_count": len(expected),
    }


def reconstruct_style_request(
    *,
    experiment_root: Path,
    root: Path,
    sample_id: str,
    method_id: str,
    intensity: str,
    neutral: GeneratedContext,
    run_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    english = neutral.input_request.get("paragraphs") if neutral.input_request else None
    if not isinstance(english, list):
        raise EvaluationError("effective English is unavailable from neutral request")
    if neutral.result is None:
        raise EvaluationError("neutral result is unavailable")
    prior_output: list[dict[str, Any]] | None = None
    critique: dict[str, Any] | None = None
    if method_id == "self_critique_repair":
        if not isinstance(run_config, Mapping):
            raise EvaluationError("derived repair run config is unavailable")
        base_method_id = str(run_config.get("base_method_id", ""))
        base_intensity = str(run_config.get("base_intensity", ""))
        if not base_method_id or not base_intensity:
            raise EvaluationError("derived repair base identity is missing")
        base_path = (
            root
            / "method_outputs"
            / base_method_id
            / base_intensity
            / f"{sample_id}.json"
        )
        critique_path = (
            root
            / "style_critiques"
            / base_method_id
            / base_intensity
            / f"{sample_id}.json"
        )
        base_artifact = load_json(require_file(base_path, "derived repair base output"))
        critique_artifact = load_json(
            require_file(critique_path, "derived repair critique")
        )
        base_result = base_artifact.get("result")
        critique_result = critique_artifact.get("result")
        if not isinstance(base_result, dict) or not isinstance(critique_result, dict):
            raise EvaluationError("derived repair prerequisites have no result objects")
        if base_artifact.get("output_sha256") != sha256_text(
            canonical_json(base_result)
        ):
            raise EvaluationError("derived repair base output hash mismatch")
        if critique_artifact.get("output_sha256") != sha256_text(
            canonical_json(critique_result)
        ):
            raise EvaluationError("derived repair critique output hash mismatch")
        prior_output = base_result.get("paragraphs")
        if not isinstance(prior_output, list):
            raise EvaluationError("derived repair base paragraphs are missing")
        if critique_artifact.get("candidate_sha256") != sha256_text(
            canonical_json(prior_output)
        ):
            raise EvaluationError("derived repair critique candidate binding mismatch")
        critique = critique_result
    module = load_payload_builder()
    try:
        request = module.build_method_request(
            experiment_root=experiment_root,
            method_id=method_id,
            intensity=intensity,
            sample_id=sample_id,
            english_semantic_source=english,
            neutral_zh=neutral.paragraphs,
            prior_output=prior_output,
            critique=critique,
        )
    except Exception as exc:
        raise EvaluationError(
            f"payload reconstruction failed for {method_id}:{intensity}:{sample_id}: {exc}"
        ) from exc
    if not isinstance(request, dict):
        raise EvaluationError("payload builder returned a non-object")
    return request


def load_style_context(
    *,
    experiment_root: Path,
    sample_set: str,
    run_id: str,
    sample_id: str,
    method_id: str,
    intensity: str,
    neutral: GeneratedContext,
    ledger_row: Mapping[str, Any] | None,
    shared_errors: Sequence[str],
    run_config: Mapping[str, Any] | None,
    analysis_lock_binding: Mapping[str, Any],
) -> GeneratedContext:
    root = run_root(experiment_root, sample_set, run_id)
    path = root / "method_outputs" / method_id / intensity / f"{sample_id}.json"
    artifact, result, artifact_errors = validate_output_artifact(
        path,
        sample_id=sample_id,
        run_id=run_id,
        stage="style_transfer",
        ledger_row=ledger_row,
        method_id=method_id,
        intensity=intensity,
        analysis_lock_binding=analysis_lock_binding,
    )
    errors = list(shared_errors) + artifact_errors
    request: dict[str, Any] | None = None
    if artifact is not None and result is not None and not neutral.errors:
        try:
            request = reconstruct_style_request(
                experiment_root=experiment_root,
                root=root,
                sample_id=sample_id,
                method_id=method_id,
                intensity=intensity,
                neutral=neutral,
                run_config=run_config,
            )
        except EvaluationError as exc:
            errors.append(f"request_reconstruction_error:{exc}")
        if request is not None:
            input_hash = sha256_text(canonical_json(request))
            payload_hash = sha256_text(canonical_json(request.get("method_payload", {})))
            references_hash = sha256_text(
                canonical_json(request.get("reference_examples", []))
            )
            checks = {
                "input_sha256": input_hash,
                "method_payload_sha256": payload_hash,
                "reference_examples_sha256": references_hash,
            }
            for key, expected in checks.items():
                if artifact.get(key) != expected:
                    errors.append(f"artifact_{key}_mismatch")
                if ledger_row is not None and ledger_row.get(key) != expected:
                    errors.append(f"ledger_{key}_mismatch")
    paragraphs = result.get("paragraphs") if isinstance(result, dict) else []
    if not isinstance(paragraphs, list):
        errors.append("method_paragraphs_not_list")
        paragraphs = []
    paragraph_rows = [dict(row) for row in paragraphs if isinstance(row, dict)]
    return GeneratedContext(
        sample_id=sample_id,
        result=result,
        paragraphs=paragraph_rows,
        text=paragraphs_to_text(paragraph_rows) if paragraph_rows else None,
        artifact=artifact,
        input_request=request,
        errors=sorted(set(errors)),
    )


def paragraph_gate(
    neutral: GeneratedContext, output: GeneratedContext
) -> dict[str, Any]:
    failures: list[str] = []
    if output.result is None:
        failures.append("missing_or_invalid_output")
    expected_ids = [row.get("id") for row in neutral.paragraphs]
    actual_ids = [row.get("id") for row in output.paragraphs]
    if actual_ids != expected_ids:
        failures.append("paragraph_id_or_order_mismatch")
    if len(actual_ids) != len(set(actual_ids)):
        failures.append("duplicate_paragraph_ids")
    if any(
        not isinstance(row.get("zh"), str) or not str(row.get("zh", "")).strip()
        for row in output.paragraphs
    ):
        failures.append("empty_paragraph")
    return {
        "status": "pass" if not failures else "fail",
        "expected_count": len(expected_ids),
        "actual_count": len(actual_ids),
        "failures": failures,
    }


def starts_dialogue(text: str) -> bool:
    return text.lstrip().startswith(("“", "‘", "「", "『", '"'))


def quotes_balanced(text: str) -> bool:
    return (
        text.count("“") == text.count("”")
        and text.count("‘") == text.count("’")
        and text.count("「") == text.count("」")
        and text.count("『") == text.count("』")
    )


def deterministic_fidelity_gate(
    neutral: GeneratedContext, output: GeneratedContext, paragraph_result: Mapping[str, Any]
) -> dict[str, Any]:
    failures: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    if paragraph_result["status"] != "pass":
        return {
            "status": "fail",
            "protocol": "surface_fidelity.v1",
            "failures": ["paragraph_gate_failed"],
            "paragraphs": diagnostics,
        }
    for source, candidate in zip(neutral.paragraphs, output.paragraphs):
        source_text = str(source["zh"])
        candidate_text = str(candidate["zh"])
        row_failures: list[str] = []
        if Counter(NUMBER_RE.findall(source_text)) != Counter(
            NUMBER_RE.findall(candidate_text)
        ):
            row_failures.append("number_surface_mismatch")
        if Counter(PLACEHOLDER_RE.findall(source_text)) != Counter(
            PLACEHOLDER_RE.findall(candidate_text)
        ):
            row_failures.append("placeholder_mismatch")
        if Counter(LATIN_RE.findall(source_text)) != Counter(
            LATIN_RE.findall(candidate_text)
        ):
            row_failures.append("latin_token_mismatch")
        source_cjk = len(CJK_RE.findall(source_text))
        output_cjk = len(CJK_RE.findall(candidate_text))
        ratio = output_cjk / max(source_cjk, 1)
        if source_cjk >= 20 and not 0.35 <= ratio <= 2.50:
            row_failures.append("paragraph_cjk_ratio_out_of_bounds")
        if starts_dialogue(source_text) != starts_dialogue(candidate_text):
            row_failures.append("dialogue_turn_surface_mismatch")
        if not quotes_balanced(candidate_text):
            row_failures.append("unbalanced_dialogue_quotes")
        failures.extend(f"{candidate.get('id')}:{value}" for value in row_failures)
        diagnostics.append(
            {
                "id": candidate.get("id"),
                "source_cjk": source_cjk,
                "output_cjk": output_cjk,
                "cjk_ratio": ratio,
                "failures": row_failures,
            }
        )
    source_total = len(CJK_RE.findall(neutral.text or ""))
    output_total = len(CJK_RE.findall(output.text or ""))
    aggregate_ratio = output_total / max(source_total, 1)
    if not 0.55 <= aggregate_ratio <= 1.75:
        failures.append("aggregate_cjk_ratio_out_of_bounds")
    return {
        "status": "pass" if not failures else "fail",
        "protocol": "surface_fidelity.v1",
        "source_cjk": source_total,
        "output_cjk": output_total,
        "aggregate_cjk_ratio": aggregate_ratio,
        "failures": failures,
        "paragraphs": diagnostics,
        "scope_note": (
            "Deterministic surface checks do not establish semantic equivalence; "
            "the independent semantic judge remains separately bound."
        ),
    }


def walk_strings(value: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], str]]:
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from walk_strings(item, (*path, str(index)))
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from walk_strings(value[key], (*path, str(key)))


def collect_reference_material(request: Mapping[str, Any] | None) -> tuple[list[str], set[str]]:
    if request is None:
        return [], set()
    materials: list[str] = []
    named_terms: set[str] = set()
    for root_key in ("reference_examples", "method_payload"):
        root = request.get(root_key)
        for path, value in walk_strings(root, (root_key,)):
            cjk = cjk_only(value)
            key_path = ".".join(path).lower()
            is_textual = root_key == "reference_examples" or any(
                token in key_path
                for token in ("example", "excerpt", "quote", "text", "evidence", "sample")
            )
            if is_textual and len(cjk) >= NO_COPY_CJK_LENGTH:
                materials.append(cjk)
            if any(
                token in key_path
                for token in ("book", "title", "character", "place", "lore", "name")
            ) and 2 <= len(cjk) <= 24:
                named_terms.add(cjk)
    return sorted(set(materials)), named_terms


def cjk_ngrams(text: str, size: int) -> Iterator[tuple[int, str]]:
    for index in range(max(0, len(text) - size + 1)):
        yield index, text[index : index + size]


def longest_introduced_reference_match(
    output: str,
    references: Sequence[str],
    excluded_texts: Sequence[str],
) -> dict[str, Any] | None:
    size = NO_COPY_CJK_LENGTH
    output_cjk = cjk_only(output)
    excluded = {
        gram
        for text in excluded_texts
        for _, gram in cjk_ngrams(cjk_only(text), size)
    }
    positions: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for reference_index, reference in enumerate(references):
        for position, gram in cjk_ngrams(reference, size):
            if len(positions[gram]) < 16:
                positions[gram].append((reference_index, position))
    best: tuple[int, str, int] | None = None
    for output_position, gram in cjk_ngrams(output_cjk, size):
        if gram in excluded:
            continue
        for reference_index, reference_position in positions.get(gram, ()):
            reference = references[reference_index]
            left = 0
            while (
                output_position - left - 1 >= 0
                and reference_position - left - 1 >= 0
                and output_cjk[output_position - left - 1]
                == reference[reference_position - left - 1]
            ):
                left += 1
            right = size
            while (
                output_position + right < len(output_cjk)
                and reference_position + right < len(reference)
                and output_cjk[output_position + right]
                == reference[reference_position + right]
            ):
                right += 1
            matched = output_cjk[output_position - left : output_position + right]
            candidate = (len(matched), matched, reference_index)
            if best is None or (candidate[0], candidate[1]) > (best[0], best[1]):
                best = candidate
    if best is None:
        return None
    return {
        "length": best[0],
        "sequence_sha256": sha256_text(best[1]),
        "reference_index": best[2],
    }


def no_copy_gate(
    *,
    output: GeneratedContext,
    neutral_text: str,
) -> dict[str, Any]:
    if output.text is None:
        return {
            "status": "fail",
            "protocol": "exact_reference_overlap.v1",
            "failures": ["missing_output"],
            "reference_count": 0,
        }
    references, named_terms = collect_reference_material(output.input_request)
    match = longest_introduced_reference_match(
        output.text, references, (neutral_text,)
    )
    blocked_cjk = cjk_only(neutral_text)
    output_cjk = cjk_only(output.text)
    leaked_terms = sorted(
        term for term in named_terms if term in output_cjk and term not in blocked_cjk
    )
    failures: list[str] = []
    if match is not None and int(match["length"]) >= NO_COPY_CJK_LENGTH:
        failures.append("introduced_reference_sequence_at_least_8_cjk")
    if leaked_terms:
        failures.append("introduced_reference_name_or_lore_term")
    return {
        "status": "pass" if not failures else "fail",
        "protocol": "exact_reference_overlap.v1",
        "minimum_cjk_sequence": NO_COPY_CJK_LENGTH,
        "reference_count": len(references),
        "reference_material_sha256": sha256_text(canonical_json(references)),
        "longest_introduced_match": match,
        "introduced_named_term_hashes": [sha256_text(value) for value in leaked_terms],
        "failures": failures,
        "exception_rule": "Only sequences already present in the visible neutral input are excluded.",
    }


def percentile(values: Sequence[int], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), quantile)) if values else 0.0


def flow_metrics(text: str) -> dict[str, float | int]:
    paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
    sentences = [part.strip() for part in SENTENCE_BOUNDARY_RE.split(text) if part.strip()]
    cjk_count = len(CJK_RE.findall(text))
    paragraph_lengths = [len(CJK_RE.findall(value)) for value in paragraphs]
    sentence_lengths = [len(CJK_RE.findall(value)) for value in sentences]
    punctuation_count = sum(character in PUNCTUATION for character in text)
    dialogue_count = sum(character in DIALOGUE_MARKS for character in text)
    dialogue_lines = sum(starts_dialogue(value) for value in paragraphs)
    function_count = sum(character in FUNCTION_CHARS for character in text)
    return {
        "cjk_count": cjk_count,
        "paragraph_count": len(paragraphs),
        "sentence_count": len(sentences),
        "mean_paragraph_cjk": sum(paragraph_lengths) / len(paragraph_lengths)
        if paragraph_lengths
        else 0.0,
        "mean_sentence_cjk": sum(sentence_lengths) / len(sentence_lengths)
        if sentence_lengths
        else 0.0,
        "p90_sentence_cjk": percentile(sentence_lengths, 0.90),
        "punctuation_density": punctuation_count / max(len(text), 1),
        "dialogue_density": dialogue_count / max(len(text), 1),
        "dialogue_line_share": dialogue_lines / max(len(paragraphs), 1),
        "function_chars_per_100_cjk": function_count * 100 / max(cjk_count, 1),
        "comma_per_100_cjk": text.count("，") * 100 / max(cjk_count, 1),
        "period_per_100_cjk": text.count("。") * 100 / max(cjk_count, 1),
        "semicolon_colon_per_100_cjk": (text.count("；") + text.count("："))
        * 100
        / max(cjk_count, 1),
        "ellipsis_dash_per_100_cjk": (text.count("…") + text.count("—"))
        * 100
        / max(cjk_count, 1),
    }


def metric_deltas(
    candidate: Mapping[str, float | int], baseline: Mapping[str, float | int]
) -> dict[str, float]:
    return {
        key: float(candidate[key]) - float(baseline[key])
        for key in candidate
        if key in baseline
    }


def load_threshold_state(
    path: Path,
    *,
    bundle: ScorerBundle,
    contract: Mapping[str, Any],
    sample_set: str,
) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "missing",
            "threshold": None,
            "path": display_path(path),
            "errors": [],
        }
    errors: list[str] = []
    try:
        artifact = load_json(path)
    except EvaluationError as exc:
        return {
            "status": "invalid",
            "threshold": None,
            "path": display_path(path),
            "errors": [str(exc)],
        }
    if artifact.get("sample_set") != sample_set:
        errors.append("threshold_sample_set_mismatch")
    current_bindings = binding_fields(bundle)
    score_binding = artifact.get("score_binding", {})
    for field, expected in current_bindings.items():
        if score_binding.get(field) != expected:
            errors.append(f"threshold_{field}_mismatch")
    if artifact.get("allocation_sha256") != file_sha256(contract["paths"]["allocation"]):
        errors.append("threshold_allocation_sha256_mismatch")
    if artifact.get("evaluation_protocol_sha256") != file_sha256(contract["protocol_path"]):
        errors.append("threshold_protocol_sha256_mismatch")
    scores_path_value = artifact.get("scores_path")
    if isinstance(scores_path_value, str):
        scores_path = resolve_recorded_path(scores_path_value)
        if not scores_path.exists() or file_sha256(scores_path) != artifact.get("scores_sha256"):
            errors.append("threshold_scores_sha256_mismatch")
    else:
        errors.append("threshold_scores_path_missing")
    threshold = finite_float(artifact.get("selected", {}).get("threshold"))
    if threshold is None:
        errors.append("threshold_value_invalid")
    return {
        "status": "invalid" if errors else "valid",
        "threshold": None if errors else threshold,
        "path": display_path(path),
        "sha256": file_sha256(path),
        "errors": errors,
        "artifact": artifact,
    }


def load_independent_judgments(
    path: Path | None,
    *,
    analysis_lock_binding: Mapping[str, Any],
) -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[str, Any]]:
    if path is None:
        return {}, {"status": "not_provided", "path": None, "sha256": None}
    require_file(path, "independent semantic judgments")
    judgments: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in iter_jsonl(path):
        try:
            require_matching_analysis_binding(
                row, analysis_lock_binding, label="independent judgment"
            )
        except ValueError as exc:
            raise EvaluationError(str(exc)) from exc
        key = (
            str(row.get("sample_id", "")),
            str(row.get("method_id", "")),
            str(row.get("intensity", "")),
        )
        if not all(key) or key in judgments:
            raise EvaluationError(f"Invalid or duplicate independent judgment key: {key}")
        status = row.get("status", "complete")
        failure = row.get("high_severity_semantic_failure")
        readability_failure = row.get("high_severity_readability_failure", False)
        neutral_failure = row.get("neutral_high_severity_semantic_failure", False)
        if status not in {"complete", "pending"}:
            raise EvaluationError(f"Invalid independent judgment status for {key}: {status}")
        if status == "complete" and not isinstance(failure, bool):
            raise EvaluationError(
                f"Complete judgment needs boolean high_severity_semantic_failure: {key}"
            )
        if status == "complete" and not isinstance(readability_failure, bool):
            raise EvaluationError(
                f"Complete judgment needs boolean high_severity_readability_failure: {key}"
            )
        if status == "complete" and not isinstance(neutral_failure, bool):
            raise EvaluationError(
                f"Complete judgment needs boolean neutral_high_severity_semantic_failure: {key}"
            )
        if status == "pending" and (
            failure is not None
            or row.get("high_severity_readability_failure") is not None
            or row.get("neutral_high_severity_semantic_failure") is not None
        ):
            raise EvaluationError(f"Pending judgment must not carry a verdict: {key}")
        judgments[key] = row
    return judgments, {
        "status": "loaded",
        "path": display_path(path),
        "sha256": file_sha256(path),
        "rows": len(judgments),
    }


def load_promotion_contract(
    path: Path | None,
    *,
    selection_id: str,
    sample_set: str,
    screening_selection: Mapping[str, Any],
    registry_path: Path,
    threshold_state: Mapping[str, Any],
    combinations: Sequence[Mapping[str, Any]],
    analysis_lock_binding: Mapping[str, Any],
) -> dict[str, Any]:
    if selection_id not in {"confirmation_v1", "final_validation_v1"}:
        if path is not None:
            raise EvaluationError(
                "--promotion-file is accepted only for official confirmation_v1"
            )
        return {"status": "not_required", "path": None, "sha256": None}
    if path is None:
        raise EvaluationError("Official confirmation requires --promotion-file")
    artifact = load_json(require_file(path, "frozen promotion contract"))
    errors: list[str] = []
    try:
        require_matching_analysis_binding(
            artifact, analysis_lock_binding, label="promotion contract"
        )
    except ValueError as exc:
        errors.append(str(exc))
    if artifact.get("status") != "promotions_frozen":
        errors.append("promotion_status_not_frozen")
    if artifact.get("sample_set") != sample_set:
        errors.append("promotion_sample_set_mismatch")
    if artifact.get("selection_id") != "screening_v1":
        errors.append("promotion_screening_selection_id_mismatch")
    if artifact.get("screening_selection_sha256") != screening_selection.get("sha256"):
        errors.append("promotion_screening_selection_sha256_mismatch")
    if artifact.get("method_registry_sha256") != file_sha256(registry_path):
        errors.append("promotion_method_registry_sha256_mismatch")
    if artifact.get("threshold_sha256") != threshold_state.get("sha256"):
        errors.append("promotion_threshold_sha256_mismatch")
    screening_evaluation_value = artifact.get("screening_evaluation_path")
    recomputed_keys: list[tuple[str, str]] = []
    if not isinstance(screening_evaluation_value, str):
        errors.append("promotion_screening_evaluation_path_missing")
    else:
        screening_evaluation_path = resolve_recorded_path(screening_evaluation_value)
        if not screening_evaluation_path.exists() or file_sha256(
            screening_evaluation_path
        ) != artifact.get("screening_evaluation_sha256"):
            errors.append("promotion_screening_evaluation_sha256_mismatch")
        else:
            try:
                _, recomputed = recompute_promotion(
                    load_json(screening_evaluation_path),
                    require_judgments=True,
                )
                recomputed_keys = [
                    (row["method_id"], row["intensity"])
                    for row in recomputed
                ]
            except Exception as exc:
                errors.append(f"promotion_recomputation_failed:{exc}")
    promoted_rows = artifact.get("promoted")
    if not isinstance(promoted_rows, list) or not promoted_rows:
        errors.append("promotion_rows_missing")
        promoted_rows = []
    promoted = {
        (str(row.get("method_id", "")), str(row.get("intensity", "")))
        for row in promoted_rows
        if isinstance(row, dict)
    }
    recorded_order = [
        (str(row.get("method_id", "")), str(row.get("intensity", "")))
        for row in promoted_rows
        if isinstance(row, dict)
    ]
    if recomputed_keys and recorded_order != recomputed_keys:
        errors.append("promotion_recomputed_decision_mismatch")
    selected = {
        (str(row["method_id"]), str(row["intensity"]))
        for row in combinations
        if row["method_id"] != "neutral_only"
    }
    if selection_id == "confirmation_v1":
        try:
            validate_stage_method_set(
                selection_id=selection_id,
                selected=selected,
                promoted=promoted,
            )
        except ValueError as exc:
            errors.append(str(exc))
    if errors:
        raise EvaluationError("Invalid promotion contract: " + ", ".join(errors))
    return {
        "status": "valid",
        "path": display_path(path),
        "sha256": file_sha256(path),
        "promotion_id": artifact.get("promotion_id"),
        "promoted": [
            {"method_id": method_id, "intensity": intensity}
            for method_id, intensity in sorted(promoted)
        ],
    }


def load_final_validation_lock(
    path: Path | None,
    *,
    selection_id: str,
    sample_set: str,
    selection: Mapping[str, Any],
    experiment_root: Path,
    registry_path: Path,
    threshold_state: Mapping[str, Any],
    combinations: Sequence[Mapping[str, Any]],
    analysis_lock_binding: Mapping[str, Any],
) -> dict[str, Any]:
    if selection_id != "final_validation_v1":
        if path is not None:
            raise EvaluationError(
                "--final-lock is accepted only for official final_validation_v1"
            )
        return {"status": "not_required", "path": None, "sha256": None}
    if path is None:
        raise EvaluationError("Final validation requires --final-lock")
    artifact = load_json(require_file(path, "one-time final-validation lock"))
    winner = artifact.get("winner")
    if not isinstance(winner, dict):
        raise EvaluationError("Final-validation lock has no winner")
    errors: list[str] = []
    try:
        require_matching_analysis_binding(
            artifact, analysis_lock_binding, label="final-validation lock"
        )
    except ValueError as exc:
        errors.append(str(exc))
    if artifact.get("lock_id") != "final_validation_v1.one_time_lock":
        errors.append("final_lock_id_mismatch")
    if artifact.get("sample_set") != sample_set:
        errors.append("final_lock_sample_set_mismatch")
    if artifact.get("selection_id") != "final_validation_v1":
        errors.append("final_lock_selection_id_mismatch")
    if artifact.get("final_selection_sha256") != selection.get("sha256"):
        errors.append("final_lock_selection_sha256_mismatch")
    if artifact.get("method_registry_sha256") != file_sha256(registry_path):
        errors.append("final_lock_method_registry_sha256_mismatch")
    if artifact.get("threshold_sha256") != threshold_state.get("sha256"):
        errors.append("final_lock_threshold_sha256_mismatch")
    confirmation_value = artifact.get("confirmation_evaluation_path")
    recomputed_winner: dict[str, Any] | None = None
    if not isinstance(confirmation_value, str):
        errors.append("final_lock_confirmation_evaluation_path_missing")
    else:
        confirmation_path = resolve_recorded_path(confirmation_value)
        if not confirmation_path.exists() or file_sha256(
            confirmation_path
        ) != artifact.get("confirmation_evaluation_sha256"):
            errors.append("final_lock_confirmation_evaluation_sha256_mismatch")
        else:
            try:
                recomputed_winner = recompute_confirmation_winner(
                    load_json(confirmation_path)
                )
            except Exception as exc:
                errors.append(f"final_lock_winner_recomputation_failed:{exc}")
    if recomputed_winner is not None and recomputed_winner != winner:
        errors.append("final_lock_recomputed_winner_mismatch")
    promotion_value = artifact.get("promotion_path")
    promoted: set[tuple[str, str]] = set()
    if not isinstance(promotion_value, str):
        errors.append("final_lock_promotion_path_missing")
    else:
        bound_promotion = resolve_recorded_path(promotion_value)
        if not bound_promotion.exists() or file_sha256(
            bound_promotion
        ) != artifact.get("promotion_sha256"):
            errors.append("final_lock_promotion_sha256_mismatch")
        else:
            promotion_artifact = load_json(bound_promotion)
            try:
                require_matching_analysis_binding(
                    promotion_artifact,
                    analysis_lock_binding,
                    label="final bound promotion",
                )
            except ValueError as exc:
                errors.append(str(exc))
            promoted = {
                (str(row.get("method_id", "")), str(row.get("intensity", "")))
                for row in promotion_artifact.get("promoted", [])
                if isinstance(row, dict)
            }
    pre_reveal_value = artifact.get("pre_reveal_lock_path")
    pre_reveal: dict[str, Any] | None = None
    if not isinstance(pre_reveal_value, str):
        errors.append("final_lock_pre_reveal_path_missing")
    else:
        pre_reveal_path = resolve_recorded_path(pre_reveal_value)
        if not pre_reveal_path.exists() or file_sha256(
            pre_reveal_path
        ) != artifact.get("pre_reveal_lock_sha256"):
            errors.append("final_lock_pre_reveal_sha256_mismatch")
        else:
            pre_reveal = load_json(pre_reveal_path)
    current_bindings = {
        "model_run_config_sha256": file_sha256(
            experiment_root / "protocols/model_run_config.v1.json"
        ),
        "style_prompt_sha256": file_sha256(
            experiment_root / "prompts/style_transfer_method.v1.md"
        ),
        "payload_lock_sha256": file_sha256(
            experiment_root / "method_assets/style_transfer_payloads.v1.lock.json"
        ),
        "method_registry_sha256": file_sha256(registry_path),
        "evaluation_protocol_sha256": file_sha256(
            experiment_root / "protocols/evaluation_protocol.v1.json"
        ),
        "runner_source_sha256": file_sha256(
            REPO_ROOT / "experiments/iteration1/run_style_transfer_generation.py"
        ),
        "evaluator_source_sha256": file_sha256(Path(__file__)),
        "payload_builder_source_sha256": file_sha256(
            REPO_ROOT / "experiments/iteration1/style_transfer_payloads.py"
        ),
        "clean_chunks_sha256": file_sha256(
            REPO_ROOT / "datasets/unmasked/chunks.clean.jsonl"
        ),
        "masked_chunks_sha256": file_sha256(
            scorer_paths_from_args(args).masked_chunks
        ),
        "splits_sha256": file_sha256(
            REPO_ROOT / "generated/style_research/corpus/splits.json"
        ),
    }
    for field, expected in current_bindings.items():
        if artifact.get(field) != expected:
            errors.append(f"final_lock_{field}_mismatch")
        if pre_reveal is not None and pre_reveal.get(field) != expected:
            errors.append(f"final_pre_reveal_{field}_mismatch")
    prompt_files = {
        "english_semantic_source": "english_semantic_source.v1.md",
        "english_source_qa": "english_source_qa.v1.md",
        "english_source_repair": "english_source_repair.v1.md",
        "neutral_translation": "neutral_translation.v1.md",
        "style_transfer": "style_transfer_method.v1.md",
        "style_critique": "style_transfer_critique.v1.md",
    }
    schema_files = {
        "english_semantic_source": "english_semantic_source_output.v1.schema.json",
        "english_source_qa": "english_source_qa_output.v1.schema.json",
        "english_source_repair": "english_source_repair_output.v1.schema.json",
        "neutral_translation": "neutral_translation_output.v1.schema.json",
        "style_transfer": "style_transfer_output.v1.schema.json",
        "style_critique": "style_transfer_critique_output.v1.schema.json",
    }
    expected_prompt_hashes = {
        key: file_sha256(experiment_root / "prompts" / filename)
        for key, filename in prompt_files.items()
    }
    expected_schema_hashes = {
        key: file_sha256(experiment_root / "schemas" / filename)
        for key, filename in schema_files.items()
    }
    for field, expected in (
        ("prompt_sha256", expected_prompt_hashes),
        ("schema_sha256", expected_schema_hashes),
    ):
        if artifact.get(field) != expected:
            errors.append(f"final_lock_{field}_mismatch")
        if pre_reveal is None or pre_reveal.get(field) != expected:
            errors.append(f"final_pre_reveal_{field}_mismatch")
    selected = {
        (str(row["method_id"]), str(row["intensity"]))
        for row in combinations
        if row["method_id"] != "neutral_only"
    }
    winner_key = (
        str(winner.get("method_id", "")),
        str(winner.get("intensity", "")),
    )
    try:
        validate_stage_method_set(
            selection_id=selection_id,
            selected=selected,
            promoted=promoted,
            winner=winner_key,
        )
    except ValueError as exc:
        errors.append(str(exc))
    if errors:
        raise EvaluationError("Invalid final-validation lock: " + ", ".join(errors))
    return {
        "status": "valid",
        "path": display_path(path),
        "sha256": file_sha256(path),
        "winner": {
            "method_id": winner["method_id"],
            "intensity": winner["intensity"],
        },
    }


def judgment_for_row(
    judgments: Mapping[tuple[str, str, str], dict[str, Any]],
    *,
    sample_id: str,
    method_id: str,
    intensity: str,
    output_available: bool,
) -> dict[str, Any]:
    if not output_available:
        return {
            "status": "not_applicable_missing_output",
            "high_severity_semantic_failure": None,
            "high_severity_readability_failure": None,
            "neutral_high_severity_semantic_failure": None,
            "source_row_sha256": None,
        }
    row = judgments.get((sample_id, method_id, intensity))
    if row is None:
        return {
            "status": "pending",
            "high_severity_semantic_failure": None,
            "high_severity_readability_failure": None,
            "neutral_high_severity_semantic_failure": None,
            "source_row_sha256": None,
        }
    return {
        "status": row["status"],
        "high_severity_semantic_failure": row.get(
            "high_severity_semantic_failure"
        ),
        "high_severity_readability_failure": row.get(
            "high_severity_readability_failure", False
        ),
        "neutral_high_severity_semantic_failure": row.get(
            "neutral_high_severity_semantic_failure", False
        ),
        "binding": {
            key: row.get(key)
            for key in (
                "selection_sha256",
                "source_run_id",
                "critique_artifact_path",
                "critique_artifact_sha256",
                "critique_result_sha256",
                "critique_input_sha256",
                "critique_ledger_path",
                "critique_ledger_row_sha256",
                "candidate_artifact_sha256",
                "candidate_output_sha256",
                "candidate_sha256",
                "candidate_ledger_path",
                "candidate_ledger_row_sha256",
                "candidate_run_config_path",
                "candidate_run_config_sha256",
                "neutral_artifact_sha256",
                "neutral_output_sha256",
                "neutral_ledger_path",
                "neutral_ledger_row_sha256",
                "neutral_run_config_path",
                "neutral_run_config_sha256",
                "effective_english_artifact_canonical_sha256",
                "effective_english_output_sha256",
                "judge_run_config_path",
                "judge_run_config_sha256",
                "judge_prompt_sha256",
                "judge_model",
                "judge_reasoning_effort",
            )
        },
        "source_row_sha256": sha256_text(canonical_json(row)),
    }


def judgment_binding_errors(
    judgment: Mapping[str, Any],
    *,
    root: Path,
    source_root: Path,
    source_run_id: str,
    sample_id: str,
    method_id: str,
    intensity: str,
    output: GeneratedContext,
    neutral: GeneratedContext,
    selection_sha256: str,
) -> list[str]:
    if judgment.get("status") != "complete":
        return []
    binding = judgment.get("binding")
    if not isinstance(binding, dict):
        return ["judgment_binding_missing"]
    errors: list[str] = []
    if binding.get("selection_sha256") != selection_sha256:
        errors.append("judgment_selection_sha256_mismatch")
    if binding.get("source_run_id") != source_run_id:
        errors.append("judgment_source_run_id_mismatch")

    candidate_path = (
        root / "method_outputs" / method_id / intensity / f"{sample_id}.json"
    )
    neutral_path = source_root / "neutral_translation" / f"{sample_id}.json"
    critique_path = (
        root / "style_critiques" / method_id / intensity / f"{sample_id}.json"
    )
    critique_ledger_path = (
        root / "ledgers" / f"style_critique.{method_id}.{intensity}.jsonl"
    )
    critique_config_path = (
        root / "run_configs" / f"style_critique.{method_id}.{intensity}.json"
    )
    candidate_ledger_path = (
        root / "ledgers" / f"style_transfer.{method_id}.{intensity}.jsonl"
    )
    candidate_config_path = (
        root / "run_configs" / f"style_transfer.{method_id}.{intensity}.json"
    )
    neutral_ledger_path = source_root / "ledgers/neutral_translation.jsonl"
    neutral_config_path = source_root / "run_configs/neutral_translation.json"
    file_checks = (
        (candidate_path, "candidate_artifact_sha256"),
        (neutral_path, "neutral_artifact_sha256"),
        (critique_path, "critique_artifact_sha256"),
        (critique_config_path, "judge_run_config_sha256"),
        (candidate_config_path, "candidate_run_config_sha256"),
        (neutral_config_path, "neutral_run_config_sha256"),
    )
    for path, field in file_checks:
        if not path.exists() or binding.get(field) != file_sha256(path):
            errors.append(f"judgment_{field}_mismatch")
    if binding.get("critique_artifact_path") != display_path(critique_path):
        errors.append("judgment_critique_artifact_path_mismatch")
    if binding.get("critique_ledger_path") != display_path(critique_ledger_path):
        errors.append("judgment_critique_ledger_path_mismatch")
    if binding.get("candidate_ledger_path") != display_path(candidate_ledger_path):
        errors.append("judgment_candidate_ledger_path_mismatch")
    if binding.get("neutral_ledger_path") != display_path(neutral_ledger_path):
        errors.append("judgment_neutral_ledger_path_mismatch")
    if binding.get("candidate_run_config_path") != display_path(candidate_config_path):
        errors.append("judgment_candidate_run_config_path_mismatch")
    if binding.get("neutral_run_config_path") != display_path(neutral_config_path):
        errors.append("judgment_neutral_run_config_path_mismatch")

    if output.artifact is None or output.result is None:
        errors.append("judgment_candidate_context_missing")
    else:
        candidate_sha = sha256_text(canonical_json(output.paragraphs))
        if binding.get("candidate_output_sha256") != output.artifact.get(
            "output_sha256"
        ):
            errors.append("judgment_candidate_output_sha256_mismatch")
        if binding.get("candidate_sha256") != candidate_sha:
            errors.append("judgment_candidate_sha256_mismatch")
    if neutral.artifact is None:
        errors.append("judgment_neutral_context_missing")
    elif binding.get("neutral_output_sha256") != neutral.artifact.get("output_sha256"):
        errors.append("judgment_neutral_output_sha256_mismatch")
    english_binding = (neutral.provenance or {}).get("effective_english") or {}
    if binding.get("effective_english_artifact_canonical_sha256") != english_binding.get(
        "artifact_canonical_sha256"
    ):
        errors.append("judgment_effective_english_artifact_sha256_mismatch")
    if binding.get("effective_english_output_sha256") != english_binding.get(
        "output_sha256"
    ):
        errors.append("judgment_effective_english_output_sha256_mismatch")

    if critique_path.exists():
        critique = load_json(critique_path)
        expected_input = {
            "sample_id": sample_id,
            "english_semantic_source": (
                neutral.input_request.get("paragraphs", [])
                if neutral.input_request
                else []
            ),
            "neutral_zh": neutral.paragraphs,
            "candidate_zh": output.paragraphs,
        }
        expected_input_sha = sha256_text(canonical_json(expected_input))
        expectations = {
            "input_sha256": expected_input_sha,
            "output_sha256": binding.get("critique_result_sha256"),
            "candidate_sha256": binding.get("candidate_sha256"),
            "base_method_id": method_id,
            "base_intensity": intensity,
        }
        for field, expected in expectations.items():
            if critique.get(field) != expected:
                errors.append(f"judgment_critique_{field}_mismatch")
        if binding.get("critique_input_sha256") != expected_input_sha:
            errors.append("judgment_critique_input_binding_mismatch")
    if critique_ledger_path.exists():
        ledger, ledger_errors = load_success_ledger(critique_ledger_path)
        errors.extend(f"judgment_{value}" for value in ledger_errors)
        ledger_row = ledger.get(sample_id)
        if ledger_row is None:
            errors.append("judgment_critique_success_ledger_missing")
        elif binding.get("critique_ledger_row_sha256") != sha256_text(
            canonical_json(ledger_row)
        ):
            errors.append("judgment_critique_ledger_row_sha256_mismatch")
    if critique_config_path.exists():
        config = load_json(critique_config_path)
        config_expectations = {
            "base_method_id": method_id,
            "base_intensity": intensity,
            "prompt_sha256": binding.get("judge_prompt_sha256"),
            "model": binding.get("judge_model"),
            "reasoning_effort": binding.get("judge_reasoning_effort"),
        }
        for field, expected in config_expectations.items():
            if config.get(field) != expected:
                errors.append(f"judgment_run_config_{field}_mismatch")
    provenance_specs = (
        (
            "candidate",
            candidate_path,
            output.artifact,
            candidate_ledger_path,
            candidate_config_path,
            "style_transfer",
        ),
        (
            "neutral",
            neutral_path,
            neutral.artifact,
            neutral_ledger_path,
            neutral_config_path,
            "neutral_translation",
        ),
        (
            "critique",
            critique_path,
            load_json(critique_path) if critique_path.exists() else None,
            critique_ledger_path,
            critique_config_path,
            "style_critique",
        ),
    )
    for label, artifact_path, artifact, ledger_path, config_path, stage in provenance_specs:
        if not isinstance(artifact, dict) or not ledger_path.exists():
            errors.append(f"judgment_{label}_provenance_input_missing")
            continue
        ledger, ledger_errors = load_success_ledger(ledger_path)
        errors.extend(f"judgment_{label}_{value}" for value in ledger_errors)
        ledger_row = ledger.get(sample_id)
        shared_errors = validate_shared_artifact_and_ledger(
            artifact_path=artifact_path,
            artifact=artifact,
            ledger_row=ledger_row,
            run_config_path=config_path,
            sample_id=sample_id,
            run_id=source_run_id if label == "neutral" else root.name,
            stage=stage,
        )
        errors.extend(
            f"judgment_{label}_shared_provenance_{value}"
            for value in shared_errors
        )
    return sorted(set(errors))


def mean_finite(values: Iterable[Any]) -> float | None:
    finite = [value for raw in values if (value := finite_float(raw)) is not None]
    return sum(finite) / len(finite) if finite else None


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, Any]:
    if total <= 0:
        return {
            "successes": successes,
            "total": total,
            "estimate": None,
            "lower": None,
            "upper": None,
        }
    estimate = successes / total
    denominator = 1.0 + z * z / total
    center = (estimate + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            estimate * (1.0 - estimate) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return {
        "successes": successes,
        "total": total,
        "estimate": estimate,
        "lower": max(0.0, center - half_width),
        "upper": min(1.0, center + half_width),
    }


def stable_seed(*parts: str, base: int = BOOTSTRAP_SEED) -> int:
    digest = hashlib.sha256(
        (str(base) + "\0" + "\0".join(parts)).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def book_cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    if resamples != BOOTSTRAP_RESAMPLES:
        protocol_note = f"non_preregistered_resample_count:{resamples}"
    else:
        protocol_note = "preregistered_10000_resamples"
    missing = [row["sample_id"] for row in rows if finite_float(row.get("paired_margin_lift")) is None]
    available = [
        float(row["paired_margin_lift"])
        for row in rows
        if finite_float(row.get("paired_margin_lift")) is not None
    ]
    if missing or not available:
        return {
            "status": "blocked_incomplete_pairs",
            "resamples": resamples,
            "seed": seed,
            "cluster_unit": "book_title",
            "clusters": len({row["book_title"] for row in rows}),
            "expected_pairs": len(rows),
            "available_pairs": len(available),
            "available_pair_mean": mean_finite(available),
            "missing_pair_sample_ids": sorted(missing),
            "protocol_note": protocol_note,
            "lower": None,
            "upper": None,
            "excludes_zero": None,
        }
    by_book: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_book[str(row["book_title"])].append(float(row["paired_margin_lift"]))
    books = sorted(by_book)
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled_indices = rng.integers(0, len(books), size=len(books))
        sampled_values = [
            value
            for book_index in sampled_indices
            for value in by_book[books[int(book_index)]]
        ]
        estimates[index] = float(np.mean(sampled_values))
    lower, upper = np.quantile(estimates, (0.025, 0.975))
    return {
        "status": "complete",
        "resamples": resamples,
        "seed": seed,
        "cluster_unit": "book_title",
        "clusters": len(books),
        "expected_pairs": len(rows),
        "available_pairs": len(available),
        "point_estimate": float(np.mean(available)),
        "lower": float(lower),
        "upper": float(upper),
        "excludes_zero": bool(lower > 0.0 or upper < 0.0),
        "positive": bool(lower > 0.0),
        "protocol_note": protocol_note,
    }


def cluster_bootstrap_success(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    cluster_key: str,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    values = [row.get(value_key) for row in rows]
    if not rows or not all(isinstance(value, bool) for value in values):
        return {
            "status": "pending_incomplete_binary_outcomes",
            "value_key": value_key,
            "cluster_key": cluster_key,
            "rows": len(rows),
            "known_rows": sum(isinstance(value, bool) for value in values),
            "clusters": len({str(row.get(cluster_key, "")) for row in rows}),
            "resamples": resamples,
            "seed": seed,
            "point_estimate": None,
            "lower": None,
            "upper": None,
        }
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[cluster_key])].append(float(bool(row[value_key])))
    clusters = sorted(grouped)
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        sampled_values = [
            value
            for cluster_index in sampled
            for value in grouped[clusters[int(cluster_index)]]
        ]
        estimates[index] = float(np.mean(sampled_values))
    lower, upper = np.quantile(estimates, (0.025, 0.975))
    return {
        "status": "complete",
        "value_key": value_key,
        "cluster_key": cluster_key,
        "rows": len(rows),
        "known_rows": len(rows),
        "clusters": len(clusters),
        "resamples": resamples,
        "seed": seed,
        "point_estimate": float(np.mean([float(bool(value)) for value in values])),
        "lower": float(lower),
        "upper": float(upper),
    }


def paired_binary_cluster_difference(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_key: str,
    neutral_key: str,
    cluster_key: str,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    pairs = [
        (row.get(candidate_key), row.get(neutral_key))
        for row in rows
    ]
    if not rows or not all(
        isinstance(candidate, bool) and isinstance(neutral, bool)
        for candidate, neutral in pairs
    ):
        return {
            "status": "pending_incomplete_binary_pairs",
            "rows": len(rows),
            "known_pairs": sum(
                isinstance(candidate, bool) and isinstance(neutral, bool)
                for candidate, neutral in pairs
            ),
            "candidate_only_failures": None,
            "neutral_only_failures": None,
            "point_difference": None,
            "lower": None,
            "upper": None,
            "noninferiority_margin": SEMANTIC_NONINFERIORITY_MARGIN,
            "noninferior": None,
        }
    grouped: dict[str, list[float]] = defaultdict(list)
    candidate_only = 0
    neutral_only = 0
    differences: list[float] = []
    for row in rows:
        candidate = bool(row[candidate_key])
        neutral = bool(row[neutral_key])
        candidate_only += int(candidate and not neutral)
        neutral_only += int(neutral and not candidate)
        difference = float(candidate) - float(neutral)
        differences.append(difference)
        grouped[str(row[cluster_key])].append(difference)
    clusters = sorted(grouped)
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        sampled_values = [
            value
            for cluster_index in sampled
            for value in grouped[clusters[int(cluster_index)]]
        ]
        estimates[index] = float(np.mean(sampled_values))
    lower, upper = np.quantile(estimates, (0.025, 0.975))
    return {
        "status": "complete",
        "rows": len(rows),
        "known_pairs": len(rows),
        "cluster_key": cluster_key,
        "clusters": len(clusters),
        "resamples": resamples,
        "seed": seed,
        "candidate_only_failures": candidate_only,
        "neutral_only_failures": neutral_only,
        "concordant_failures": sum(
            bool(candidate) and bool(neutral) for candidate, neutral in pairs
        ),
        "concordant_passes": sum(
            not bool(candidate) and not bool(neutral) for candidate, neutral in pairs
        ),
        "point_difference": float(np.mean(differences)),
        "lower": float(lower),
        "upper": float(upper),
        "noninferiority_margin": SEMANTIC_NONINFERIORITY_MARGIN,
        "noninferior": bool(upper <= SEMANTIC_NONINFERIORITY_MARGIN),
    }


FLOW_DELTA_KEYS = (
    "mean_paragraph_cjk",
    "mean_sentence_cjk",
    "p90_sentence_cjk",
    "punctuation_density",
    "dialogue_density",
    "dialogue_line_share",
    "function_chars_per_100_cjk",
    "comma_per_100_cjk",
    "period_per_100_cjk",
    "semicolon_colon_per_100_cjk",
    "ellipsis_dash_per_100_cjk",
)


def summarize_rows(rows: Sequence[Mapping[str, Any]], threshold_valid: bool) -> dict[str, Any]:
    total = len(rows)
    scored = [row for row in rows if finite_float(row.get("target_margin")) is not None]
    target_wins = sum(int(row.get("target_chunk_indicator") or 0) for row in scored)
    deterministic_values = [row.get("deterministic_style_success") for row in rows]
    deterministic_summary = None
    if threshold_valid and all(isinstance(value, bool) for value in deterministic_values):
        deterministic_summary = wilson_interval(sum(bool(value) for value in deterministic_values), total)
    final_values = [row.get("final_style_success") for row in rows]
    final_known = [value for value in final_values if isinstance(value, bool)]
    pending = total - len(final_known)
    lower_successes = sum(value is True for value in final_known)
    upper_successes = lower_successes + pending
    final_summary = {
        "status": "complete" if pending == 0 and threshold_valid else "pending",
        "known_rows": len(final_known),
        "pending_rows": pending,
        "lower_bound_assuming_pending_fail": wilson_interval(lower_successes, total),
        "upper_bound_assuming_pending_pass": wilson_interval(upper_successes, total),
    }
    if pending == 0 and threshold_valid:
        final_summary["wilson_95"] = wilson_interval(lower_successes, total)
    flow_delta = {
        key: mean_finite(row.get("flow_delta", {}).get(key) for row in rows)
        for key in FLOW_DELTA_KEYS
    }
    gate_counts = {
        gate: Counter(str(row.get(gate, {}).get("status", "missing")) for row in rows)
        for gate in ("paragraph_gate", "fidelity_gate", "no_copy_gate")
    }
    return {
        "expected_rows": total,
        "scored_rows": len(scored),
        "failed_or_missing_rows": total - len(scored),
        "mean_target_margin": mean_finite(row.get("target_margin") for row in scored),
        "mean_target_rank": mean_finite(row.get("target_rank") for row in scored),
        "target_chunk_share_scored": target_wins / len(scored) if scored else None,
        "target_chunk_share_conservative": target_wins / total if total else None,
        "mean_neutral_target_margin": mean_finite(
            row.get("neutral_target_margin") for row in rows
        ),
        "mean_paired_margin_lift": mean_finite(
            row.get("paired_margin_lift") for row in rows
        ),
        "positive_lift_rows": sum(
            finite_float(row.get("paired_margin_lift")) is not None
            and float(row["paired_margin_lift"]) > 0.0
            for row in rows
        ),
        "deterministic_style_success": deterministic_summary,
        "final_style_success": final_summary,
        "independent_judge_pending_rows": sum(
            row.get("independent_semantic_judge", {}).get("status") == "pending"
            for row in rows
        ),
        "independent_high_severity_semantic_failure_rows": sum(
            row.get("independent_semantic_judge", {}).get(
                "high_severity_semantic_failure"
            )
            is True
            for row in rows
        ),
        "independent_high_severity_readability_failure_rows": sum(
            row.get("independent_semantic_judge", {}).get(
                "high_severity_readability_failure"
            )
            is True
            for row in rows
        ),
        "independent_neutral_high_severity_semantic_failure_rows": sum(
            row.get("independent_semantic_judge", {}).get(
                "neutral_high_severity_semantic_failure"
            )
            is True
            for row in rows
        ),
        "independent_judgment_binding_failure_rows": sum(
            row.get("independent_semantic_judge", {}).get("binding_status")
            == "fail"
            for row in rows
        ),
        "hard_fidelity_failure_rows": sum(bool(row.get("hard_fidelity_failure")) for row in rows),
        "gate_counts": {
            gate: dict(sorted(counts.items())) for gate, counts in gate_counts.items()
        },
        "mean_flow_delta_over_neutral": flow_delta,
    }


def table_summaries(
    rows: Sequence[Mapping[str, Any]],
    *,
    key: str,
    threshold_valid: bool,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["benchmark_arm"]), str(row[key]))].append(row)
    result: list[dict[str, Any]] = []
    for (arm, value), group in sorted(grouped.items()):
        result.append(
            {
                "benchmark_arm": arm,
                key: value,
                **summarize_rows(group, threshold_valid),
            }
        )
    return result


def endpoint_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold_state: Mapping[str, Any],
    arm_summaries: Mapping[str, dict[str, Any]],
    book_rows: Sequence[dict[str, Any]],
    author_rows: Sequence[dict[str, Any]],
    bootstrap: Mapping[str, dict[str, Any]],
    success_bootstrap: Mapping[str, Mapping[str, dict[str, Any]]],
    selection_id: str,
) -> dict[str, Any]:
    if selection_id == "screening_v1":
        return {
            "status": "descriptive_method_screen_only",
            "own_author_reconstruction": None,
            "cross_author_transfer": None,
            "final_selection": "apply_preregistered_promotion_rule_only",
        }
    if selection_id not in {"confirmation_v1", "final_validation_v1"}:
        return {
            "status": "descriptive_nonconfirmation_subset",
            "own_author_reconstruction": None,
            "cross_author_transfer": None,
            "final_selection": "not_permitted_for_endpoint_inference",
        }
    if threshold_state["status"] != "valid":
        return {
            "status": "blocked_threshold_not_valid",
            "own_author_reconstruction": None,
            "cross_author_transfer": None,
            "final_selection": "blocked",
        }
    own = arm_summaries.get("own_author_reconstruction")
    cross = arm_summaries.get("cross_author_transfer")
    own_books = [
        row for row in book_rows if row["benchmark_arm"] == "own_author_reconstruction"
    ]
    cross_authors = [
        row for row in author_rows if row["benchmark_arm"] == "cross_author_transfer"
    ]

    pending_judges = sum(
        row.get("independent_semantic_judge", {}).get("status") == "pending"
        for row in rows
    )
    final_basis = pending_judges == 0

    def success_rate(
        summary: Mapping[str, Any] | None, *, final: bool
    ) -> float | None:
        if not summary:
            return None
        if final:
            value = summary.get("final_style_success")
            wilson = value.get("wilson_95") if isinstance(value, dict) else None
            return (
                finite_float(wilson.get("estimate"))
                if isinstance(wilson, dict)
                else None
            )
        value = summary.get("deterministic_style_success")
        return finite_float(value.get("estimate")) if isinstance(value, dict) else None

    def success_lower(
        summary: Mapping[str, Any] | None, *, final: bool
    ) -> float | None:
        if not summary:
            return None
        if final:
            value = summary.get("final_style_success")
            wilson = value.get("wilson_95") if isinstance(value, dict) else None
            return (
                finite_float(wilson.get("lower"))
                if isinstance(wilson, dict)
                else None
            )
        value = summary.get("deterministic_style_success")
        return finite_float(value.get("lower")) if isinstance(value, dict) else None

    def quality_summary(arm: str) -> dict[str, Any]:
        arm_rows = [row for row in rows if row["benchmark_arm"] == arm]
        total = len(arm_rows)
        paired_rows: list[dict[str, Any]] = []
        for row in arm_rows:
            candidate_failure = bool(
                row.get("output_status") != "complete"
                or row.get("independent_semantic_judge", {}).get(
                    "high_severity_semantic_failure"
                )
                is True
            )
            neutral_failure = bool(
                finite_float(row.get("neutral_target_margin")) is None
                or row.get("independent_semantic_judge", {}).get(
                    "neutral_high_severity_semantic_failure"
                )
                is True
            )
            paired_rows.append(
                {
                    **row,
                    "_candidate_semantic_failure": candidate_failure,
                    "_neutral_semantic_failure": neutral_failure,
                }
            )
        candidate_semantic_failures = sum(
            row["_candidate_semantic_failure"] for row in paired_rows
        )
        neutral_semantic_failures = sum(
            row["_neutral_semantic_failure"] for row in paired_rows
        )
        readability_failures = sum(
            row.get("output_status") != "complete"
            or row.get("independent_semantic_judge", {}).get(
                "high_severity_readability_failure"
            )
            is True
            for row in arm_rows
        )
        binding_failures = sum(
            row.get("independent_semantic_judge", {}).get("binding_status")
            == "fail"
            for row in arm_rows
        )
        cluster_key = (
            "book_title" if arm == "own_author_reconstruction" else "author"
        )
        semantic_comparison = paired_binary_cluster_difference(
            paired_rows,
            candidate_key="_candidate_semantic_failure",
            neutral_key="_neutral_semantic_failure",
            cluster_key=cluster_key,
            resamples=BOOTSTRAP_RESAMPLES,
            seed=stable_seed("semantic_noninferiority", arm),
        )
        return {
            "status": "complete" if final_basis else "pending",
            "rows": total,
            "candidate_high_severity_semantic_failures": candidate_semantic_failures,
            "neutral_high_severity_semantic_failures": neutral_semantic_failures,
            "paired_semantic_failure_difference": semantic_comparison,
            "semantic_noninferiority": final_basis
            and total > 0
            and binding_failures == 0
            and semantic_comparison.get("noninferior") is True,
            "high_severity_readability_failures": readability_failures,
            "readability_noninferiority": final_basis
            and total > 0
            and binding_failures == 0
            and readability_failures == 0,
            "judgment_binding_failures": binding_failures,
        }

    own_quality = quality_summary("own_author_reconstruction")
    cross_quality = quality_summary("cross_author_transfer")
    success_basis_key = (
        "final_style_success" if final_basis else "deterministic_style_success"
    )
    own_success_bootstrap = success_bootstrap.get(
        "own_author_reconstruction", {}
    ).get(success_basis_key, {})
    cross_success_bootstrap = success_bootstrap.get(
        "cross_author_transfer", {}
    ).get(success_basis_key, {})
    own_rate = success_rate(own, final=final_basis)
    own_lower = success_lower(own, final=final_basis)
    own_book_rates = [success_rate(row, final=final_basis) for row in own_books]
    own_bootstrap = bootstrap.get("own_author_reconstruction", {})
    own_pass = bool(
        own
        and own["failed_or_missing_rows"] == 0
        and own_rate is not None
        and own_rate >= 0.80
        and own_lower is not None
        and own_lower >= 0.70
        and own_book_rates
        and all(value is not None and value >= 0.70 for value in own_book_rates)
        and own_bootstrap.get("positive") is True
        and finite_float(own_success_bootstrap.get("lower")) is not None
        and float(own_success_bootstrap["lower"]) >= 0.70
        and (not final_basis or own_quality["semantic_noninferiority"])
        and (not final_basis or own_quality["readability_noninferiority"])
    )
    if selection_id == "final_validation_v1":
        return {
            "status": "provisional_deterministic_only"
            if pending_judges
            else "final_judgments_available",
            "success_basis": "final_style_success"
            if final_basis
            else "deterministic_style_success",
            "own_author_reconstruction": {
                "provisional_pass": own_pass if not final_basis else None,
                "final_pass": own_pass if final_basis else None,
                "style_success_estimate": own_rate,
                "wilson_lower": own_lower,
                "every_book_at_least_0_70": bool(own_book_rates)
                and all(
                    value is not None and value >= 0.70
                    for value in own_book_rates
                ),
                "bootstrap_ci_excludes_zero_positive": own_bootstrap.get(
                    "positive"
                ),
                "cluster_bootstrap_success": own_success_bootstrap,
                "quality": own_quality,
            },
            "cross_author_transfer": None,
            "pending_independent_judgments": pending_judges,
            "final_selection": "blocked_pending_independent_semantic_judgments"
            if pending_judges
            else "pass"
            if own_pass
            else "fail",
        }
    cross_rate = success_rate(cross, final=final_basis)
    cross_lower = success_lower(cross, final=final_basis)
    positive_authors = sum(
        finite_float(row.get("mean_paired_margin_lift")) is not None
        and float(row["mean_paired_margin_lift"]) > 0.0
        for row in cross_authors
    )
    cross_pass = bool(
        cross
        and cross["failed_or_missing_rows"] == 0
        and cross_rate is not None
        and cross_rate >= 0.80
        and cross_lower is not None
        and cross_lower >= 0.70
        and positive_authors >= 10
        and finite_float(cross_success_bootstrap.get("lower")) is not None
        and float(cross_success_bootstrap["lower"]) >= 0.70
        and (not final_basis or cross_quality["semantic_noninferiority"])
        and (not final_basis or cross_quality["readability_noninferiority"])
    )
    return {
        "status": "provisional_deterministic_only" if pending_judges else "final_judgments_available",
        "success_basis": "final_style_success"
        if final_basis
        else "deterministic_style_success",
        "own_author_reconstruction": {
            "provisional_pass": own_pass if not final_basis else None,
            "final_pass": own_pass if final_basis else None,
            "style_success_estimate": own_rate,
            "wilson_lower": own_lower,
            "every_book_at_least_0_70": bool(own_book_rates)
            and all(value is not None and value >= 0.70 for value in own_book_rates),
            "bootstrap_ci_excludes_zero_positive": own_bootstrap.get("positive"),
            "cluster_bootstrap_success": own_success_bootstrap,
            "quality": own_quality,
        },
        "cross_author_transfer": {
            "provisional_pass": cross_pass if not final_basis else None,
            "final_pass": cross_pass if final_basis else None,
            "style_success_estimate": cross_rate,
            "wilson_lower": cross_lower,
            "authors_with_positive_mean_lift": positive_authors,
            "required_positive_authors": 10,
            "cluster_bootstrap_success": cross_success_bootstrap,
            "quality": cross_quality,
        },
        "pending_independent_judgments": pending_judges,
        "final_selection": "blocked_pending_independent_semantic_judgments"
        if pending_judges
        else "pass"
        if own_pass and cross_pass
        else "fail",
    }


def combination_provenance(
    *,
    experiment_root: Path,
    sample_set: str,
    run_id: str,
    source_run_id: str,
    method: Mapping[str, Any],
    intensity: str,
    contract: Mapping[str, Any],
    analysis_lock_binding: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[str], dict[str, Any] | None]:
    if method["method_id"] == "neutral_only":
        return {}, [], None
    root = run_root(experiment_root, sample_set, run_id)
    stem = f"style_transfer.{method['method_id']}.{intensity}"
    ledger, ledger_errors = load_success_ledger(root / "ledgers" / f"{stem}.jsonl")
    config, config_errors = validate_run_config(
        root / "run_configs" / f"{stem}.json",
        sample_set=sample_set,
        run_id=run_id,
        stage="style_transfer",
        method_id=str(method["method_id"]),
        intensity=intensity,
        analysis_lock_binding=analysis_lock_binding,
    )
    errors = ledger_errors + config_errors
    if config is not None:
        if config.get("input_run_id") != source_run_id:
            errors.append("run_config_input_run_id_mismatch")
        if config.get("method_config_sha256") != method.get("config_sha256"):
            errors.append("run_config_method_config_sha256_mismatch")
        if config.get("method_evaluation_ids_sha256") != file_sha256(
            contract["paths"]["method_ids"]
        ):
            errors.append("run_config_method_evaluation_ids_sha256_mismatch")
        selection = contract.get("selection", {})
        expected_admission = {
            "screening_v1": {
                "preregistered_screening",
                "screening_provisional_shortlist",
                "screening_refinement_contract",
            },
            "confirmation_v1": {"judged_screening_promotion"},
            "final_validation_v1": {"one_time_final_validation_lock"},
        }.get(str(selection.get("selection_id")), set())
        for sample_id in contract["method_ids"]:
            row = ledger.get(sample_id)
            if row is None:
                continue
            if row.get("execution_selection_id") != selection.get("selection_id"):
                errors.append(f"{sample_id}:execution_selection_id_mismatch")
            if row.get("execution_selection_sha256") != selection.get("sha256"):
                errors.append(f"{sample_id}:execution_selection_sha256_mismatch")
            if int(row.get("execution_selection_sample_count", -1)) != len(
                contract["method_ids"]
            ):
                errors.append(
                    f"{sample_id}:execution_selection_sample_count_mismatch"
                )
            errors.extend(
                f"{sample_id}:{value}"
                for value in validate_execution_binding_record(
                    row,
                    sample_id=sample_id,
                    analysis_lock_binding=analysis_lock_binding,
                )
            )
            batch_value = row.get("execution_batch_config_path")
            if isinstance(batch_value, str):
                batch_path = resolve_recorded_path(batch_value)
                if batch_path.exists():
                    batch = load_json(batch_path)
                    recorded_stage_config = batch.get("stage_run_config_path")
                    if not isinstance(recorded_stage_config, str) or (
                        resolve_recorded_path(recorded_stage_config).resolve()
                        != (root / "run_configs" / f"{stem}.json").resolve()
                    ):
                        errors.append(
                            f"{sample_id}:execution_batch_wrong_stage_run_config"
                        )
            admission = row.get("execution_admission")
            admission_stage = (
                admission.get("admission_stage")
                if isinstance(admission, dict)
                else None
            )
            if admission_stage not in expected_admission:
                errors.append(f"{sample_id}:execution_admission_stage_mismatch")
            roster_binding = contract.get("screening_roster_binding", {})
            if roster_binding.get("phase") == "refinement":
                if not isinstance(admission, dict) or admission.get(
                    "admission_sha256"
                ) != roster_binding.get("sha256"):
                    errors.append(
                        f"{sample_id}:refinement_admission_contract_mismatch"
                    )
    if (
        config is not None
        and method["method_id"] == "self_critique_repair"
        and contract.get("screening_roster_binding", {}).get("phase")
        == "refinement"
    ):
        refinement_path = contract["screening_roster_binding"].get("path")
        if not isinstance(refinement_path, str):
            errors.append("derived_repair_refinement_contract_path_missing")
        else:
            refinement = load_json(resolve_recorded_path(refinement_path))
            try:
                require_matching_analysis_binding(
                    refinement,
                    analysis_lock_binding,
                    label="derived repair refinement contract",
                )
            except ValueError as exc:
                errors.append(str(exc))
            arms = [
                row
                for row in refinement.get("evaluation_combinations", [])
                if isinstance(row, dict)
                and row.get("method_id") == "self_critique_repair"
                and row.get("intensity") == intensity
            ]
            if len(arms) != 1:
                errors.append("derived_repair_contract_arm_missing_or_ambiguous")
            elif config.get("base_method_id") != arms[0].get(
                "base_method_id"
            ) or config.get("base_intensity") != arms[0].get("base_intensity"):
                errors.append("derived_repair_run_config_base_mismatch")
    return ledger, sorted(set(errors)), config


def evaluate_combination(
    *,
    experiment_root: Path,
    sample_set: str,
    run_id: str,
    source_run_id: str,
    method: Mapping[str, Any],
    intensity: str,
    contract: Mapping[str, Any],
    bundle: ScorerBundle,
    masker: EntityMasker,
    neutral: Mapping[str, GeneratedContext],
    neutral_scores: Mapping[str, dict[str, Any]],
    neutral_masked: Mapping[str, str],
    threshold_state: Mapping[str, Any],
    judgments: Mapping[tuple[str, str, str], dict[str, Any]],
    bootstrap_resamples: int,
    bootstrap_seed: int,
    analysis_lock_binding: Mapping[str, Any],
) -> dict[str, Any]:
    method_id = str(method["method_id"])
    ledger, shared_errors, run_config = combination_provenance(
        experiment_root=experiment_root,
        sample_set=sample_set,
        run_id=run_id,
        source_run_id=source_run_id,
        method=method,
        intensity=intensity,
        contract=contract,
        analysis_lock_binding=analysis_lock_binding,
    )
    draft_rows: list[dict[str, Any]] = []
    texts_to_score: list[str] = []
    score_row_indices: list[int] = []
    output_contexts: dict[str, GeneratedContext] = {}
    root = run_root(experiment_root, sample_set, run_id)
    source_root = run_root(experiment_root, sample_set, source_run_id)
    for sample_id in contract["method_ids"]:
        allocation = contract["allocation"][sample_id]
        neutral_context = neutral[sample_id]
        if method_id == "neutral_only":
            output = GeneratedContext(
                sample_id=sample_id,
                result=neutral_context.result,
                paragraphs=list(neutral_context.paragraphs),
                text=neutral_context.text,
                artifact=neutral_context.artifact,
                input_request={"reference_examples": [], "method_payload": {}},
                errors=list(neutral_context.errors),
            )
        else:
            output = load_style_context(
                experiment_root=experiment_root,
                sample_set=sample_set,
                run_id=run_id,
                sample_id=sample_id,
                method_id=method_id,
                intensity=intensity,
                neutral=neutral_context,
                ledger_row=ledger.get(sample_id),
                shared_errors=shared_errors,
                run_config=run_config,
                analysis_lock_binding=analysis_lock_binding,
            )
        paragraph_result = paragraph_gate(neutral_context, output)
        output_contexts[sample_id] = output
        fidelity_result = deterministic_fidelity_gate(
            neutral_context, output, paragraph_result
        )
        no_copy_result = no_copy_gate(
            output=output,
            neutral_text=neutral_context.text or "",
        )
        artifact_status = "pass" if not output.errors else "fail"
        trusted_for_score = bool(
            artifact_status == "pass"
            and paragraph_result["status"] == "pass"
            and output.text
        )
        neutral_score = neutral_scores.get(sample_id)
        method_metrics = flow_metrics(output.text) if trusted_for_score and output.text else None
        neutral_metrics = (
            flow_metrics(neutral_context.text)
            if neutral_context.text and not neutral_context.errors
            else None
        )
        row: dict[str, Any] = {
            "schema_version": 1,
            "sample_id": sample_id,
            "sample_set": sample_set,
            "run_id": run_id,
            "method_id": method_id,
            "method_label": method["label"],
            "intensity": intensity,
            "research_role": allocation["research_role"],
            "benchmark_arm": allocation["benchmark_arm"],
            "author": allocation["author"],
            "book_title": allocation["book_title"],
            "chunk_id": allocation["chunk_id"],
            "source_split": allocation["source_split"],
            "output_status": "complete" if trusted_for_score else "failed",
            "artifact_gate": {"status": artifact_status, "failures": output.errors},
            "paragraph_gate": paragraph_result,
            "fidelity_gate": fidelity_result,
            "no_copy_gate": no_copy_result,
            "neutral_target_margin": neutral_score.get("target_margin")
            if neutral_score
            else None,
            "neutral_target_rank": neutral_score.get("target_rank")
            if neutral_score
            else None,
            "neutral_predicted_author": neutral_score.get("predicted_author")
            if neutral_score
            else None,
            "target_margin": None,
            "target_rank": None,
            "predicted_author": None,
            "target_chunk_indicator": None,
            "paired_margin_lift": None,
            "threshold": threshold_state.get("threshold"),
            "threshold_status": threshold_state["status"],
            "deterministic_style_success": None,
            "final_style_success": None,
            "style_success_status": "pending_threshold"
            if threshold_state["status"] != "valid"
            else "pending_score",
            "hard_fidelity_failure": bool(
                artifact_status != "pass"
                or paragraph_result["status"] != "pass"
                or fidelity_result["status"] != "pass"
                or no_copy_result["status"] != "pass"
            ),
            "independent_semantic_judge": None,
            "neutral_flow_metrics": neutral_metrics,
            "method_flow_metrics": method_metrics,
            "flow_delta": metric_deltas(method_metrics, neutral_metrics)
            if method_metrics is not None and neutral_metrics is not None
            else {},
            "masked_input_sha256": None,
            "raw_output_sha256": sha256_text(output.text) if output.text else None,
        }
        draft_rows.append(row)
        if trusted_for_score:
            masked = masker.mask(
                output.text or "",
                author=str(allocation["author"]),
                title=str(allocation["book_title"]),
            )
            row["masked_input_sha256"] = sha256_text(masked)
            score_row_indices.append(len(draft_rows) - 1)
            texts_to_score.append(masked)

    scores = score_texts(bundle, texts_to_score)
    for row_index, score in zip(score_row_indices, scores):
        row = draft_rows[row_index]
        row.update(
            {
                "target_margin": score["target_margin"],
                "target_rank": score["target_rank"],
                "predicted_author": score["predicted_author"],
                "target_chunk_indicator": score["target_chunk_indicator"],
            }
        )
        neutral_margin = finite_float(row.get("neutral_target_margin"))
        if neutral_margin is not None:
            row["paired_margin_lift"] = score["target_margin"] - neutral_margin

    threshold_valid = threshold_state["status"] == "valid"
    threshold = finite_float(threshold_state.get("threshold"))
    for row in draft_rows:
        output = output_contexts[row["sample_id"]]
        neutral_context = neutral[row["sample_id"]]
        output_available = row["output_status"] == "complete"
        if method_id == "neutral_only":
            judgment = {
                "status": "not_required_control",
                "high_severity_semantic_failure": None,
                "high_severity_readability_failure": None,
                "neutral_high_severity_semantic_failure": None,
                "source_row_sha256": None,
            }
        else:
            judgment = judgment_for_row(
                judgments,
                sample_id=row["sample_id"],
                method_id=method_id,
                intensity=intensity,
                output_available=output_available,
            )
        binding_errors = judgment_binding_errors(
            judgment,
            root=root,
            source_root=source_root,
            source_run_id=source_run_id,
            sample_id=row["sample_id"],
            method_id=method_id,
            intensity=intensity,
            output=output,
            neutral=neutral_context,
            selection_sha256=str(contract["selection"]["sha256"]),
        )
        judgment["binding_status"] = "fail" if binding_errors else "pass"
        judgment["binding_errors"] = binding_errors
        row["independent_semantic_judge"] = judgment
        if (
            judgment.get("high_severity_semantic_failure") is True
            or judgment.get("high_severity_readability_failure") is True
            or bool(binding_errors)
        ):
            row["hard_fidelity_failure"] = True
        if threshold_valid and threshold is not None:
            margin = finite_float(row.get("target_margin"))
            rank = finite_float(row.get("target_rank"))
            lift = finite_float(row.get("paired_margin_lift"))
            deterministic_success = bool(
                margin is not None
                and margin >= threshold
                and rank is not None
                and rank <= 5
                and lift is not None
                and lift > 0.0
                and not (
                    row["artifact_gate"]["status"] != "pass"
                    or row["paragraph_gate"]["status"] != "pass"
                    or row["fidelity_gate"]["status"] != "pass"
                    or row["no_copy_gate"]["status"] != "pass"
                )
            )
            row["deterministic_style_success"] = deterministic_success
            if not deterministic_success:
                row["final_style_success"] = False
                row["style_success_status"] = "failed_deterministic_gate"
            elif binding_errors:
                row["final_style_success"] = False
                row["style_success_status"] = "failed_judgment_binding"
            elif judgment["status"] == "complete":
                row["final_style_success"] = not bool(
                    judgment["high_severity_semantic_failure"]
                    or judgment["high_severity_readability_failure"]
                )
                row["style_success_status"] = (
                    "success" if row["final_style_success"] else "failed_independent_judge"
                )
            elif method_id == "neutral_only":
                row["final_style_success"] = False
                row["style_success_status"] = "failed_positive_lift_gate_control"
            else:
                row["final_style_success"] = None
                row["style_success_status"] = "pending_independent_semantic_judge"
        else:
            row["style_success_status"] = "pending_threshold"

    arm_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in draft_rows:
        arm_groups[row["benchmark_arm"]].append(row)
    arm_summaries = {
        arm: summarize_rows(rows, threshold_valid)
        for arm, rows in sorted(arm_groups.items())
    }
    book_rows = table_summaries(
        draft_rows, key="book_title", threshold_valid=threshold_valid
    )
    author_rows = table_summaries(
        draft_rows, key="author", threshold_valid=threshold_valid
    )
    bootstrap = {
        arm: book_cluster_bootstrap(
            rows,
            resamples=bootstrap_resamples,
            seed=stable_seed(method_id, intensity, arm, base=bootstrap_seed),
        )
        for arm, rows in sorted(arm_groups.items())
    }
    success_bootstrap = {
        arm: {
            value_key: cluster_bootstrap_success(
                rows,
                value_key=value_key,
                cluster_key=(
                    "book_title"
                    if arm == "own_author_reconstruction"
                    else "author"
                ),
                resamples=bootstrap_resamples,
                seed=stable_seed(
                    method_id,
                    intensity,
                    arm,
                    value_key,
                    base=bootstrap_seed,
                ),
            )
            for value_key in (
                "deterministic_style_success",
                "final_style_success",
            )
        }
        for arm, rows in sorted(arm_groups.items())
    }
    endpoints = endpoint_summary(
        draft_rows,
        threshold_state=threshold_state,
        arm_summaries=arm_summaries,
        book_rows=book_rows,
        author_rows=author_rows,
        bootstrap=bootstrap,
        success_bootstrap=success_bootstrap,
        selection_id=str(contract["selection"]["selection_id"]),
    )
    return {
        "schema_version": 1,
        "sample_set": sample_set,
        "run_id": run_id,
        "method": {
            "id": method_id,
            "label": method["label"],
            "family": method.get("family"),
            "registry_status": method.get("status"),
            "intensity": intensity,
            "config_path": display_path(method["config_path_resolved"]),
            "config_sha256": method["config_sha256"],
        },
        "scorer_binding": binding_fields(bundle),
        "threshold": dict(threshold_state),
        "run_config": {
            "path": display_path(
                root / "run_configs" / f"style_transfer.{method_id}.{intensity}.json"
            )
            if method_id != "neutral_only"
            else display_path(root / "run_configs/neutral_translation.json"),
            "sha256": sha256_text(canonical_json(run_config)) if run_config else None,
            "errors": shared_errors,
        },
        "arm_summaries": arm_summaries,
        "book_cluster_bootstrap": bootstrap,
        "cluster_bootstrap_style_success": success_bootstrap,
        "by_book": book_rows,
        "by_author": author_rows,
        "endpoints": endpoints,
        "rows": draft_rows,
    }


def csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return canonical_json(value)
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: csv_cell(row.get(key)) for key in fieldnames})
    atomic_write_text(path, buffer.getvalue())


SCORE_CSV_FIELDS = (
    "sample_id",
    "sample_set",
    "run_id",
    "method_id",
    "method_label",
    "intensity",
    "research_role",
    "benchmark_arm",
    "author",
    "book_title",
    "chunk_id",
    "source_split",
    "output_status",
    "neutral_target_margin",
    "neutral_target_rank",
    "neutral_predicted_author",
    "target_margin",
    "target_rank",
    "predicted_author",
    "target_chunk_indicator",
    "paired_margin_lift",
    "threshold",
    "threshold_status",
    "deterministic_style_success",
    "final_style_success",
    "style_success_status",
    "hard_fidelity_failure",
    "artifact_gate",
    "paragraph_gate",
    "fidelity_gate",
    "no_copy_gate",
    "independent_semantic_judge",
    "neutral_flow_metrics",
    "method_flow_metrics",
    "flow_delta",
    "masked_input_sha256",
    "raw_output_sha256",
)


SUMMARY_CSV_FIELDS = (
    "benchmark_arm",
    "book_title",
    "author",
    "expected_rows",
    "scored_rows",
    "failed_or_missing_rows",
    "mean_target_margin",
    "mean_target_rank",
    "target_chunk_share_scored",
    "target_chunk_share_conservative",
    "mean_neutral_target_margin",
    "mean_paired_margin_lift",
    "positive_lift_rows",
    "deterministic_successes",
    "deterministic_success_rate",
    "deterministic_wilson_lower",
    "deterministic_wilson_upper",
    "final_success_status",
    "final_known_rows",
    "final_pending_rows",
    "hard_fidelity_failure_rows",
    "mean_flow_delta_over_neutral",
    "gate_counts",
)


def flatten_summary_row(row: Mapping[str, Any]) -> dict[str, Any]:
    deterministic = row.get("deterministic_style_success")
    final = row.get("final_style_success", {})
    return {
        **row,
        "deterministic_successes": deterministic.get("successes")
        if isinstance(deterministic, dict)
        else None,
        "deterministic_success_rate": deterministic.get("estimate")
        if isinstance(deterministic, dict)
        else None,
        "deterministic_wilson_lower": deterministic.get("lower")
        if isinstance(deterministic, dict)
        else None,
        "deterministic_wilson_upper": deterministic.get("upper")
        if isinstance(deterministic, dict)
        else None,
        "final_success_status": final.get("status") if isinstance(final, dict) else None,
        "final_known_rows": final.get("known_rows") if isinstance(final, dict) else None,
        "final_pending_rows": final.get("pending_rows") if isinstance(final, dict) else None,
    }


def svg_message(path: Path, title: str, message: str) -> None:
    content = f"""<svg xmlns="http://www.w3.org/2000/svg" width="960" height="180" viewBox="0 0 960 180">
<rect width="960" height="180" fill="#ffffff"/>
<text x="32" y="44" font-family="sans-serif" font-size="22" fill="#17202a">{html.escape(title)}</text>
<text x="32" y="100" font-family="sans-serif" font-size="16" fill="#566573">{html.escape(message)}</text>
</svg>
"""
    atomic_write_text(path, content)


def write_svg_bar_chart(
    path: Path,
    title: str,
    rows: Sequence[tuple[str, float | None]],
    *,
    value_kind: str = "number",
) -> None:
    valid = [(label, float(value)) for label, value in rows if finite_float(value) is not None]
    if not valid:
        svg_message(path, title, "No complete values are available.")
        return
    width = 960
    left = 300
    right = 50
    top = 72
    row_height = 38
    height = top + row_height * len(valid) + 48
    minimum = min(0.0, *(value for _, value in valid))
    maximum = max(0.0, *(value for _, value in valid))
    if math.isclose(minimum, maximum):
        maximum = minimum + 1.0
    plot_width = width - left - right

    def x_position(value: float) -> float:
        return left + (value - minimum) / (maximum - minimum) * plot_width

    zero_x = x_position(0.0)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="24" y="36" font-family="sans-serif" font-size="22" fill="#17202a">{html.escape(title)}</text>',
        f'<line x1="{zero_x:.2f}" y1="{top - 18}" x2="{zero_x:.2f}" y2="{height - 34}" stroke="#85929e" stroke-width="1"/>',
    ]
    for index, (label, value) in enumerate(valid):
        y = top + index * row_height
        value_x = x_position(value)
        bar_x = min(zero_x, value_x)
        bar_width = max(1.0, abs(value_x - zero_x))
        color = "#2471a3" if value >= 0 else "#b03a2e"
        display = f"{value:.1%}" if value_kind == "percent" else f"{value:.4f}"
        elements.extend(
            (
                f'<text x="{left - 12}" y="{y + 15}" text-anchor="end" font-family="sans-serif" font-size="13" fill="#273746">{html.escape(label[:42])}</text>',
                f'<rect x="{bar_x:.2f}" y="{y}" width="{bar_width:.2f}" height="20" fill="{color}"/>',
                f'<text x="{value_x + (6 if value >= 0 else -6):.2f}" y="{y + 15}" text-anchor="{"start" if value >= 0 else "end"}" font-family="sans-serif" font-size="12" fill="#17202a">{display}</text>',
            )
        )
    elements.append("</svg>\n")
    atomic_write_text(path, "\n".join(elements))


def fmt_number(value: Any, digits: int = 4) -> str:
    number = finite_float(value)
    return "pending" if number is None else f"{number:.{digits}f}"


def fmt_percent(value: Any) -> str:
    number = finite_float(value)
    return "pending" if number is None else f"{number:.1%}"


def markdown_arm_table(result: Mapping[str, Any]) -> list[str]:
    lines = [
        "| Arm | Expected | Scored | Target share | Margin lift | Deterministic success | Wilson 95% CI | Final status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for arm, summary in result["arm_summaries"].items():
        deterministic = summary.get("deterministic_style_success") or {}
        final = summary.get("final_style_success") or {}
        ci = (
            f"{fmt_percent(deterministic.get('lower'))} to {fmt_percent(deterministic.get('upper'))}"
            if deterministic
            else "pending threshold"
        )
        lines.append(
            f"| `{arm}` | {summary['expected_rows']} | {summary['scored_rows']} | "
            f"{fmt_percent(summary['target_chunk_share_conservative'])} | "
            f"{fmt_number(summary['mean_paired_margin_lift'])} | "
            f"{fmt_percent(deterministic.get('estimate')) if deterministic else 'pending'} | "
            f"{ci} | `{final.get('status', 'pending')}` |"
        )
    return lines


def markdown_group_table(rows: Sequence[Mapping[str, Any]], key: str) -> list[str]:
    lines = [
        f"| Arm | {key.replace('_', ' ').title()} | Expected | Scored | Lift | Deterministic success | Wilson lower |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        deterministic = row.get("deterministic_style_success") or {}
        lines.append(
            f"| `{row['benchmark_arm']}` | {row[key]} | {row['expected_rows']} | "
            f"{row['scored_rows']} | {fmt_number(row['mean_paired_margin_lift'])} | "
            f"{fmt_percent(deterministic.get('estimate')) if deterministic else 'pending'} | "
            f"{fmt_percent(deterministic.get('lower')) if deterministic else 'pending'} |"
        )
    return lines


def method_combination_markdown(result: Mapping[str, Any]) -> str:
    method = result["method"]
    threshold = result["threshold"]
    rows = result["rows"]
    failure_counts: Counter[str] = Counter()
    for row in rows:
        for failure in row["artifact_gate"].get("failures", []):
            failure_counts[f"artifact:{failure}"] += 1
        for gate in ("paragraph_gate", "fidelity_gate", "no_copy_gate"):
            for failure in row[gate].get("failures", []):
                failure_counts[f"{gate}:{failure}"] += 1
    lines = [
        f"# {method['label']} ({method['intensity']})",
        "",
        "## Status",
        "",
        f"- Method ID: `{method['id']}`",
        f"- Run ID: `{result['run_id']}`",
        f"- Threshold: `{threshold['status']}`"
        + (f" at {threshold['threshold']:.6f}" if finite_float(threshold.get("threshold")) is not None else ""),
        f"- Endpoint: `{result['endpoints']['status']}`",
        f"- Final selection: `{result['endpoints']['final_selection']}`",
        "- Independent semantic judge: pending rows are unknown, never treated as passes.",
        "- Evaluator execution: deterministic local code only; no LLM is called.",
        "",
        "## Arm Summary",
        "",
        *markdown_arm_table(result),
        "",
        "![Paired target-margin lift](charts/paired_margin_lift.svg)",
        "",
        "![Target chunk share](charts/target_chunk_share.svg)",
        "",
        "![Deterministic style success](charts/deterministic_style_success.svg)",
        "",
        "## Book-Cluster Bootstrap",
        "",
    ]
    for arm, bootstrap in result["book_cluster_bootstrap"].items():
        lines.append(
            f"- `{arm}`: `{bootstrap['status']}`, {bootstrap['resamples']:,} resamples, "
            f"CI {fmt_number(bootstrap.get('lower'))} to {fmt_number(bootstrap.get('upper'))}."
        )
    lines.extend(("", "## Per Book", "", *markdown_group_table(result["by_book"], "book_title")))
    lines.extend(("", "## Per Author", "", *markdown_group_table(result["by_author"], "author")))
    lines.extend(("", "## Interpretable Deltas", ""))
    lines.append(
        "Deltas are method output minus its paired neutral output. They are diagnostics, not a replacement for the frozen style meter."
    )
    lines.append("")
    lines.extend(("| Arm | Metric | Mean delta |", "| --- | --- | ---: |"))
    for arm, summary in result["arm_summaries"].items():
        for key, value in summary["mean_flow_delta_over_neutral"].items():
            lines.append(f"| `{arm}` | `{key}` | {fmt_number(value)} |")
    lines.extend(("", "## Failures", ""))
    if failure_counts:
        lines.extend(("| Failure | Rows |", "| --- | ---: |"))
        for failure, count in sorted(failure_counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"| `{failure}` | {count} |")
    else:
        lines.append("No deterministic artifact, paragraph, fidelity, or no-copy failures were recorded.")
    lines.extend(
        (
            "",
            "## Caveat",
            "",
            "`deterministic_style_success` is a provisional endpoint. `final_style_success` remains pending wherever an output clears deterministic gates but lacks an independently supplied semantic judgment. Semantic noninferiority cannot be inferred from surface checks.",
            "",
        )
    )
    return "\n".join(lines)


def write_combination_outputs(output_dir: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    method = result["method"]
    directory = output_dir / "methods" / method["id"] / method["intensity"]
    charts = directory / "charts"
    write_json(directory / "results.json", result)
    write_csv(directory / "scores.csv", result["rows"], SCORE_CSV_FIELDS)
    write_csv(
        directory / "by_book.csv",
        [flatten_summary_row(row) for row in result["by_book"]],
        SUMMARY_CSV_FIELDS,
    )
    write_csv(
        directory / "by_author.csv",
        [flatten_summary_row(row) for row in result["by_author"]],
        SUMMARY_CSV_FIELDS,
    )
    write_svg_bar_chart(
        charts / "paired_margin_lift.svg",
        f"{method['label']} ({method['intensity']}): paired target-margin lift",
        [
            (arm, summary["mean_paired_margin_lift"])
            for arm, summary in result["arm_summaries"].items()
        ],
    )
    write_svg_bar_chart(
        charts / "target_chunk_share.svg",
        f"{method['label']} ({method['intensity']}): target chunk share",
        [
            (arm, summary["target_chunk_share_conservative"])
            for arm, summary in result["arm_summaries"].items()
        ],
        value_kind="percent",
    )
    write_svg_bar_chart(
        charts / "deterministic_style_success.svg",
        f"{method['label']} ({method['intensity']}): deterministic style success",
        [
            (
                arm,
                summary["deterministic_style_success"]["estimate"]
                if summary["deterministic_style_success"]
                else None,
            )
            for arm, summary in result["arm_summaries"].items()
        ],
        value_kind="percent",
    )
    atomic_write_text(directory / "report.md", method_combination_markdown(result))
    artifact_paths = (
        directory / "results.json",
        directory / "scores.csv",
        directory / "by_book.csv",
        directory / "by_author.csv",
        charts / "paired_margin_lift.svg",
        charts / "target_chunk_share.svg",
        charts / "deterministic_style_success.svg",
        directory / "report.md",
    )
    return {
        "method_id": method["id"],
        "method_label": method["label"],
        "intensity": method["intensity"],
        "status": "complete"
        if all(summary["failed_or_missing_rows"] == 0 for summary in result["arm_summaries"].values())
        else "completed_with_failures",
        "endpoints": result["endpoints"],
        "arm_summaries": result["arm_summaries"],
        "artifacts": {
            display_path(path): file_sha256(path) for path in artifact_paths
        },
    }


def write_method_indexes(
    output_dir: Path,
    methods: Sequence[Mapping[str, Any]],
    combination_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records_by_method: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in combination_records:
        records_by_method[str(row["method_id"])].append(row)
    indexes: list[dict[str, Any]] = []
    for method in methods:
        method_id = str(method["method_id"])
        rows = sorted(records_by_method.get(method_id, []), key=lambda row: str(row["intensity"]))
        if not rows:
            continue
        directory = output_dir / "methods" / method_id
        summary_rows: list[dict[str, Any]] = []
        for row in rows:
            for arm, summary in row["arm_summaries"].items():
                deterministic = summary.get("deterministic_style_success") or {}
                summary_rows.append(
                    {
                        "method_id": method_id,
                        "method_label": method["label"],
                        "intensity": row["intensity"],
                        "benchmark_arm": arm,
                        "status": row["status"],
                        "expected_rows": summary["expected_rows"],
                        "scored_rows": summary["scored_rows"],
                        "failed_or_missing_rows": summary["failed_or_missing_rows"],
                        "target_chunk_share": summary["target_chunk_share_conservative"],
                        "mean_paired_margin_lift": summary["mean_paired_margin_lift"],
                        "deterministic_style_success": deterministic.get("estimate"),
                        "deterministic_wilson_lower": deterministic.get("lower"),
                        "final_status": summary["final_style_success"]["status"],
                    }
                )
        index_payload = {
            "schema_version": 1,
            "method_id": method_id,
            "method_label": method["label"],
            "family": method.get("family"),
            "registry_status": method.get("status"),
            "intensities": rows,
        }
        write_json(directory / "summary.json", index_payload)
        fields = (
            "method_id",
            "method_label",
            "intensity",
            "benchmark_arm",
            "status",
            "expected_rows",
            "scored_rows",
            "failed_or_missing_rows",
            "target_chunk_share",
            "mean_paired_margin_lift",
            "deterministic_style_success",
            "deterministic_wilson_lower",
            "final_status",
        )
        write_csv(directory / "summary.csv", summary_rows, fields)
        write_svg_bar_chart(
            directory / "summary.svg",
            f"{method['label']}: paired lift by intensity and arm",
            [
                (
                    f"{row['intensity']} / {row['benchmark_arm']}",
                    row["mean_paired_margin_lift"],
                )
                for row in summary_rows
            ],
        )
        lines = [
            f"# {method['label']}",
            "",
            f"- Method ID: `{method_id}`",
            f"- Registry status: `{method.get('status')}`",
            "- Every frozen intensity is listed, including missing-output failures.",
            "",
            "| Intensity | Arm | Status | Scored / expected | Lift | Deterministic success | Final status |",
            "| --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
        for row in summary_rows:
            lines.append(
                f"| [{row['intensity']}]({row['intensity']}/report.md) | `{row['benchmark_arm']}` | "
                f"`{row['status']}` | {row['scored_rows']} / {row['expected_rows']} | "
                f"{fmt_number(row['mean_paired_margin_lift'])} | "
                f"{fmt_percent(row['deterministic_style_success'])} | `{row['final_status']}` |"
            )
        lines.extend(("", "![Paired lift by intensity and arm](summary.svg)", ""))
        atomic_write_text(directory / "report.md", "\n".join(lines))
        indexes.append(
            {
                "method_id": method_id,
                "method_label": method["label"],
                "status": "completed_with_failures"
                if any(row["status"] != "complete" for row in rows)
                else "complete",
                "summary_path": display_path(directory / "summary.json"),
                "summary_sha256": file_sha256(directory / "summary.json"),
                "report_path": display_path(directory / "report.md"),
                "report_sha256": file_sha256(directory / "report.md"),
            }
        )
    return indexes


def overall_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Style-Transfer Method Evaluation",
        "",
        f"- Sample set: `{payload['sample_set']}`",
        f"- Run ID: `{payload['run_id']}`",
        f"- Frozen source run ID: `{payload['source_run_id']}`",
        f"- Scorer: `{payload['scorer_binding']['scorer_id']}`",
        f"- Threshold status: `{payload['threshold']['status']}`",
        f"- Independent judgments: `{payload['independent_judgments']['status']}`",
        f"- Pre-style analysis lock: `{payload['analysis_lock']['sha256']}`",
        "- Execution: local deterministic evaluator only; no LLM was called.",
        "",
        "Final method selection remains blocked wherever independent semantic judgments are pending. Deterministic surface gates are not reported as semantic passes.",
        "",
        "## Methods",
        "",
        "| Method | Status |",
        "| --- | --- |",
    ]
    for method in payload["methods"]:
        lines.append(
            f"| [{method['method_label']}](methods/{method['method_id']}/report.md) | `{method['status']}` |"
        )
    lines.extend(
        (
            "",
            "![Mean paired lift](charts/mean_paired_lift.svg)",
            "",
            "![Deterministic style success](charts/deterministic_style_success.svg)",
            "",
            "## Binding",
            "",
            f"- Classifier artifact SHA-256: `{payload['scorer_binding']['classifier_artifact_sha256']}`",
            f"- Scorer config SHA-256: `{payload['scorer_binding']['scorer_config_sha256']}`",
            f"- Masking artifact SHA-256: `{payload['scorer_binding']['masking_artifact_sha256']}`",
            f"- Method registry SHA-256: `{payload['method_registry']['sha256']}`",
            f"- Frozen method-ID SHA-256: `{payload['sample_contract']['method_ids_sha256']}`",
            "",
        )
    )
    return "\n".join(lines)


def evaluate_methods(args: argparse.Namespace) -> dict[str, Any]:
    experiment_root = args.experiment_root.resolve()
    analysis_lock_binding = validate_analysis_lock(
        args.analysis_lock.resolve(), experiment_root
    )
    bundle, scorer_action = get_or_create_scorer(args)
    contract = load_sample_contract(experiment_root, args.sample_set)
    source_run_id = args.input_run_id or args.run_id
    style_execution = contract.get("protocol", {}).get("iteration3", {}).get(
        "style_execution", {}
    )
    if isinstance(style_execution, Mapping) and style_execution:
        expected_source = style_execution.get("source_run_id")
        expected_style = style_execution.get("style_run_id")
        if source_run_id != expected_source:
            raise EvaluationError(
                "Evaluation source run does not match the frozen protocol: "
                f"expected={expected_source!r}, observed={source_run_id!r}"
            )
        if args.run_id != expected_style:
            raise EvaluationError(
                "Evaluation style run does not match the frozen protocol: "
                f"expected={expected_style!r}, observed={args.run_id!r}"
            )
    if args.selection_file:
        selection_path = args.selection_file.resolve()
        selection = load_json(require_file(selection_path, "evaluation selection"))
        selected_ids = tuple(sorted(str(value) for value in selection.get("sample_ids", [])))
        if len(selected_ids) != len(set(selected_ids)) or not selected_ids:
            raise EvaluationError("Evaluation selection must contain unique sample IDs")
        unknown = sorted(set(selected_ids) - set(contract["method_ids"]))
        if unknown:
            raise EvaluationError(
                f"Evaluation selection contains non-method IDs: {unknown[:5]}"
            )
        claimed_id = str(selection.get("selection_id", ""))
        frozen = contract["frozen_selections"].get(claimed_id)
        if frozen is not None:
            if selection_path != frozen["path"].resolve():
                raise EvaluationError(
                    f"Official {claimed_id} evaluation must use its frozen path: "
                    f"{display_path(frozen['path'])}"
                )
            if file_sha256(selection_path) != frozen["sha256"]:
                raise EvaluationError(f"Official {claimed_id} selection hash mismatch")
            if selected_ids != frozen["sample_ids"]:
                raise EvaluationError(f"Official {claimed_id} selection IDs mismatch")
            selection_id = claimed_id
            official = True
        else:
            selection_id = "custom_subset_descriptive_only"
            official = False
        contract["method_ids"] = tuple(sorted(selected_ids))
        contract["selection"] = {
            "selection_id": selection_id,
            "path": selection_path,
            "sha256": file_sha256(selection_path),
            "official": official,
        }
    else:
        contract["selection"] = {
            "selection_id": "full_method_evaluation_descriptive_only",
            "path": contract["paths"]["method_ids"],
            "sha256": file_sha256(contract["paths"]["method_ids"]),
            "official": False,
        }
    registry_path, _registry, methods = load_method_registry(experiment_root)
    combinations = select_method_combinations(methods, args.method)
    if not combinations:
        raise EvaluationError("No method/intensity combinations were selected")
    refinement_contract_path = (
        args.refinement_contract.resolve() if args.refinement_contract else None
    )
    screening_roster_binding = validate_screening_roster(
        contract=contract,
        combinations=combinations,
        phase=args.screening_phase,
        refinement_contract_path=refinement_contract_path,
        analysis_lock_binding=analysis_lock_binding,
    )
    contract["screening_roster_binding"] = screening_roster_binding
    threshold_state = load_threshold_state(
        args.threshold.resolve(),
        bundle=bundle,
        contract=contract,
        sample_set=args.sample_set,
    )
    promotion_path = args.promotion_file.resolve() if args.promotion_file else None
    promotion_binding = load_promotion_contract(
        promotion_path,
        selection_id=str(contract["selection"]["selection_id"]),
        sample_set=args.sample_set,
        screening_selection=contract["frozen_selections"].get("screening_v1", {}),
        registry_path=registry_path,
        threshold_state=threshold_state,
        combinations=combinations,
        analysis_lock_binding=analysis_lock_binding,
    )
    final_lock_path = args.final_lock.resolve() if args.final_lock else None
    final_lock_binding = load_final_validation_lock(
        final_lock_path,
        selection_id=str(contract["selection"]["selection_id"]),
        sample_set=args.sample_set,
        selection=contract["selection"],
        experiment_root=experiment_root,
        registry_path=registry_path,
        threshold_state=threshold_state,
        combinations=combinations,
        analysis_lock_binding=analysis_lock_binding,
    )
    judgment_path = args.independent_judgments.resolve() if args.independent_judgments else None
    judgments, judgment_binding = load_independent_judgments(
        judgment_path, analysis_lock_binding=analysis_lock_binding
    )
    valid_judgment_keys = {
        (sample_id, str(combo["method_id"]), str(combo["intensity"]))
        for sample_id in contract["method_ids"]
        for combo in combinations
        if combo["method_id"] != "neutral_only"
    }
    extra_judgments = set(judgments) - valid_judgment_keys
    if extra_judgments:
        raise EvaluationError(
            "Independent judgments contain rows outside the selected frozen evaluation: "
            f"{sorted(extra_judgments)[:3]}"
        )

    neutral = load_neutral_contexts(
        experiment_root,
        args.sample_set,
        source_run_id,
        contract["method_ids"],
    )
    masker = EntityMasker(scorer_paths_from_args(args).mask_terms)
    neutral_masked: dict[str, str] = {}
    score_ids: list[str] = []
    score_inputs: list[str] = []
    for sample_id in contract["method_ids"]:
        context = neutral[sample_id]
        if context.errors or context.text is None:
            continue
        allocation = contract["allocation"][sample_id]
        masked = masker.mask(
            context.text,
            author=str(allocation["author"]),
            title=str(allocation["book_title"]),
        )
        neutral_masked[sample_id] = masked
        score_ids.append(sample_id)
        score_inputs.append(masked)
    neutral_scores = dict(zip(score_ids, score_texts(bundle, score_inputs)))

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else experiment_root / "evaluations" / args.run_id
    )
    combination_records: list[dict[str, Any]] = []
    for combination in combinations:
        result = evaluate_combination(
            experiment_root=experiment_root,
            sample_set=args.sample_set,
            run_id=args.run_id,
            source_run_id=source_run_id,
            method=combination,
            intensity=str(combination["intensity"]),
            contract=contract,
            bundle=bundle,
            masker=masker,
            neutral=neutral,
            neutral_scores=neutral_scores,
            neutral_masked=neutral_masked,
            threshold_state=threshold_state,
            judgments=judgments,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
            analysis_lock_binding=analysis_lock_binding,
        )
        combination_records.append(write_combination_outputs(output_dir, result))

    selected_method_ids = {str(row["method_id"]) for row in combinations}
    selected_methods = [row for row in methods if row["method_id"] in selected_method_ids]
    method_indexes = write_method_indexes(output_dir, selected_methods, combination_records)
    overall_rows: list[dict[str, Any]] = []
    for combination in combination_records:
        for arm, summary in combination["arm_summaries"].items():
            deterministic = summary.get("deterministic_style_success") or {}
            overall_rows.append(
                {
                    "method_id": combination["method_id"],
                    "method_label": combination["method_label"],
                    "intensity": combination["intensity"],
                    "benchmark_arm": arm,
                    "status": combination["status"],
                    "expected_rows": summary["expected_rows"],
                    "scored_rows": summary["scored_rows"],
                    "failed_or_missing_rows": summary["failed_or_missing_rows"],
                    "target_chunk_share": summary["target_chunk_share_conservative"],
                    "mean_paired_margin_lift": summary["mean_paired_margin_lift"],
                    "deterministic_style_success": deterministic.get("estimate"),
                    "deterministic_wilson_lower": deterministic.get("lower"),
                    "final_status": summary["final_style_success"]["status"],
                }
            )
    overall_fields = (
        "method_id",
        "method_label",
        "intensity",
        "benchmark_arm",
        "status",
        "expected_rows",
        "scored_rows",
        "failed_or_missing_rows",
        "target_chunk_share",
        "mean_paired_margin_lift",
        "deterministic_style_success",
        "deterministic_wilson_lower",
        "final_status",
    )
    write_csv(output_dir / "method_summary.csv", overall_rows, overall_fields)
    write_svg_bar_chart(
        output_dir / "charts/mean_paired_lift.svg",
        "Mean paired target-margin lift by method and arm",
        [
            (
                f"{row['method_id']}:{row['intensity']} / {row['benchmark_arm']}",
                row["mean_paired_margin_lift"],
            )
            for row in overall_rows
        ],
    )
    write_svg_bar_chart(
        output_dir / "charts/deterministic_style_success.svg",
        "Deterministic style success by method and arm",
        [
            (
                f"{row['method_id']}:{row['intensity']} / {row['benchmark_arm']}",
                row["deterministic_style_success"],
            )
            for row in overall_rows
        ],
        value_kind="percent",
    )
    payload = {
        "schema_version": 1,
        "sample_set": args.sample_set,
        "run_id": args.run_id,
        "source_run_id": source_run_id,
        "status": "complete_with_method_failures"
        if any(row["status"] != "complete" for row in combination_records)
        else "complete",
        "scorer_action": scorer_action,
        "analysis_lock": analysis_lock_binding,
        "scorer_binding": binding_fields(bundle),
        "threshold": threshold_state,
        "independent_judgments": judgment_binding,
        "promotion_contract": promotion_binding,
        "final_validation_lock": final_lock_binding,
        "screening_roster": screening_roster_binding,
        "method_registry": {
            "path": display_path(registry_path),
            "sha256": file_sha256(registry_path),
        },
        "sample_contract": {
            "method_ids_path": display_path(contract["paths"]["method_ids"]),
            "method_ids_sha256": file_sha256(contract["paths"]["method_ids"]),
            "allocation_sha256": file_sha256(contract["paths"]["allocation"]),
            "hidden_targets_sha256": file_sha256(contract["paths"]["hidden"]),
            "method_evaluation_rows": len(contract["method_ids"]),
            "selection_id": contract["selection"]["selection_id"],
            "selection_path": display_path(contract["selection"]["path"]),
            "selection_sha256": contract["selection"]["sha256"],
        },
        "bootstrap": {
            "resamples": args.bootstrap_resamples,
            "base_seed": args.bootstrap_seed,
            "cluster_unit": "book_title",
        },
        "methods": method_indexes,
        "combinations": combination_records,
    }
    write_json(output_dir / "evaluation_summary.json", payload)
    atomic_write_text(output_dir / "report.md", overall_markdown(payload))
    artifact_paths = (
        output_dir / "evaluation_summary.json",
        output_dir / "method_summary.csv",
        output_dir / "charts/mean_paired_lift.svg",
        output_dir / "charts/deterministic_style_success.svg",
        output_dir / "report.md",
    )
    manifest = {
        "schema_version": 1,
        "sample_set": args.sample_set,
        "run_id": args.run_id,
        "source_run_id": source_run_id,
        "analysis_lock": analysis_lock_binding,
        "artifacts": {
            display_path(path): file_sha256(path) for path in artifact_paths
        },
        "method_artifact_sets": combination_records,
    }
    write_json(output_dir / "manifest.json", manifest)
    return {
        "status": payload["status"],
        "output_dir": display_path(output_dir),
        "report": display_path(output_dir / "report.md"),
        "summary": display_path(output_dir / "evaluation_summary.json"),
        "manifest": display_path(output_dir / "manifest.json"),
        "method_count": len(method_indexes),
        "combination_count": len(combination_records),
        "threshold_status": threshold_state["status"],
        "independent_judgment_status": judgment_binding["status"],
        "scorer_action": scorer_action,
    }


def add_scorer_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument(
        "--benchmark-result", type=Path, default=DEFAULT_BENCHMARK_RESULT
    )
    parser.add_argument(
        "--benchmark-script", type=Path, default=DEFAULT_BENCHMARK_SCRIPT
    )
    parser.add_argument("--scorer-dir", type=Path, default=DEFAULT_SCORER_DIR)
    parser.add_argument(
        "--scorer-mode",
        choices=("auto", "fit", "load"),
        default="auto",
        help="Auto loads a frozen scorer or fits it only when absent.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit/freeze the exact author-style meter, emit calibration scores, "
            "and deterministically evaluate generated style-transfer outputs."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit = subparsers.add_parser(
        "fit-scorer",
        help="Fit a missing exact train-only scorer or verify/load the frozen scorer.",
    )
    add_scorer_arguments(fit)

    verify = subparsers.add_parser(
        "verify-scorer",
        help="Reproduce the Stage 1 masked test metrics with the frozen scorer.",
    )
    add_scorer_arguments(verify)
    verify.add_argument(
        "--output",
        type=Path,
        default=(DEFAULT_EXPERIMENT_ROOT / "scorers/style_meter_verification.v1.json"),
    )

    calibration = subparsers.add_parser(
        "score-calibration",
        help="Score exactly the frozen original/neutral calibration pairs.",
    )
    add_scorer_arguments(calibration)
    calibration.add_argument(
        "--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT
    )
    calibration.add_argument("--sample-set", default=DEFAULT_SAMPLE_SET)
    calibration.add_argument("--run-id", required=True)
    calibration.add_argument("--output", type=Path, default=DEFAULT_CALIBRATION_OUTPUT)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate all registered method/intensity outputs and controls.",
    )
    add_scorer_arguments(evaluate)
    evaluate.add_argument(
        "--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT
    )
    evaluate.add_argument("--sample-set", default=DEFAULT_SAMPLE_SET)
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument(
        "--input-run-id",
        help=(
            "Run containing the frozen English and neutral prerequisites. "
            "Defaults to --run-id."
        ),
    )
    evaluate.add_argument(
        "--analysis-lock",
        type=Path,
        required=True,
        help="Content-addressed pre-style analysis lock required for all evaluations.",
    )
    evaluate.add_argument(
        "--selection-file",
        type=Path,
        help=(
            "Optional frozen sample_ids JSON for screening or confirmation. "
            "Without it, all method-evaluation rows are expected."
        ),
    )
    evaluate.add_argument("--threshold", type=Path, default=DEFAULT_THRESHOLD)
    evaluate.add_argument(
        "--promotion-file",
        type=Path,
        help="Required frozen screening promotion artifact for confirmation_v1.",
    )
    evaluate.add_argument(
        "--final-lock",
        type=Path,
        help="Required one-time lock for the final_validation_v1 sample set.",
    )
    evaluate.add_argument(
        "--independent-judgments",
        type=Path,
        help=(
            "Optional frozen JSONL judgments. The evaluator never creates or "
            "calls an LLM for these judgments."
        ),
    )
    evaluate.add_argument(
        "--method",
        action="append",
        default=[],
        metavar="METHOD[:INTENSITY]",
        help="Restrict evaluation; repeat as needed. Default: every registry entry.",
    )
    evaluate.add_argument(
        "--screening-phase",
        choices=("initial", "refinement"),
        default="initial",
        help="Machine-enforced roster phase for official screening_v1.",
    )
    evaluate.add_argument(
        "--refinement-contract",
        type=Path,
        help="Frozen exact roster for post-screening intensity/repair refinement.",
    )
    evaluate.add_argument("--output-dir", type=Path)
    evaluate.add_argument(
        "--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES
    )
    evaluate.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "fit-scorer":
            bundle, action = get_or_create_scorer(args)
            result = {
                "status": "complete",
                "action": action,
                "scorer_dir": display_path(bundle.directory),
                "bindings": binding_fields(bundle),
                "manifest_sha256": file_sha256(
                    bundle.directory / SCORER_FILES["manifest"]
                ),
            }
        elif args.command == "verify-scorer":
            result = verify_scorer_performance(args)
        elif args.command == "score-calibration":
            result = score_calibration(args)
        elif args.command == "evaluate":
            if args.bootstrap_resamples <= 0:
                raise EvaluationError("--bootstrap-resamples must be positive")
            result = evaluate_methods(args)
        else:  # pragma: no cover - argparse enforces the command set.
            parser.error(f"Unknown command: {args.command}")
            return 2
    except (EvaluationError, FileNotFoundError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(pretty_json(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
