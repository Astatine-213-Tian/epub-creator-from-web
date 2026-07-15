from __future__ import annotations

"""Single source of truth for the current authorship style-meter contract."""

import hashlib
import re
from pathlib import Path

from .audit_style_dataset import (
    CROSS_BOOK_DECONTAMINATION_VERSION,
    MASKING_POLICY_VERSION,
    PUNCTUATION_NORMALIZATION_VERSION,
    normalize_punctuation,
)


RESEARCH_ROOT = Path(__file__).resolve().parents[1]
CURRENT_BENCHMARK_DIR = (
    RESEARCH_ROOT
    / "generated/style_research/benchmarks/author_style_supervised_50authors_cleaned"
)
CURRENT_BENCHMARK_RESULT = CURRENT_BENCHMARK_DIR / "supervised_author_baseline_results.json"
CURRENT_BENCHMARK_RUN_KEY = "sgd_hinge_unbalanced.char_ngrams.train_global_masked"
CURRENT_SCORER_VALIDATION_DIR = (
    RESEARCH_ROOT
    / "generated/style_research/benchmarks/mask_artifact_ablation_v2"
)
CURRENT_SCORER_VALIDATION_RESULT = (
    CURRENT_SCORER_VALIDATION_DIR / "mask_artifact_ablation_results.json"
)
CURRENT_SCORER_VALIDATION_RUN_KEY = (
    "sgd_hinge_unbalanced.mask_stripped_char_ngrams.train_global_masked"
)
CURRENT_MASKED_VIEW = "train_global_masked"
CURRENT_FEATURE_METHOD = "mask_stripped_char_ngrams"
CURRENT_SCORER_ID = (
    "unweighted_sgd_hinge_mask_stripped_char_ngrams_min_df_20."
    "train_global_mask_v1"
)
CURRENT_CHAR_MIN_DF = 20
CURRENT_MAX_CHAR_FEATURES = 80_000
CURRENT_NGRAM_RANGE = (2, 4)


def normalize_char_text(text: str) -> str:
    """Apply the corpus punctuation contract, then remove whitespace for n-grams."""
    return re.sub(r"\s+", "", normalize_punctuation(text))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "CURRENT_BENCHMARK_DIR",
    "CURRENT_BENCHMARK_RESULT",
    "CURRENT_BENCHMARK_RUN_KEY",
    "CURRENT_CHAR_MIN_DF",
    "CURRENT_FEATURE_METHOD",
    "CURRENT_MASKED_VIEW",
    "CURRENT_MAX_CHAR_FEATURES",
    "CURRENT_NGRAM_RANGE",
    "CURRENT_SCORER_ID",
    "CURRENT_SCORER_VALIDATION_DIR",
    "CURRENT_SCORER_VALIDATION_RESULT",
    "CURRENT_SCORER_VALIDATION_RUN_KEY",
    "CROSS_BOOK_DECONTAMINATION_VERSION",
    "MASKING_POLICY_VERSION",
    "PUNCTUATION_NORMALIZATION_VERSION",
    "RESEARCH_ROOT",
    "file_sha256",
    "normalize_char_text",
]
