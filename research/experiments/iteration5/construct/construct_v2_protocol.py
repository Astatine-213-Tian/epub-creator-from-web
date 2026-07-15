#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from typing import Any


HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def han_text(text: str) -> str:
    return "".join(HAN_RE.findall(text))


def han_ngram_overlap(text: str, references: list[str], *, n: int = 4) -> float:
    candidate = han_text(text)
    if len(candidate) < n:
        return 0.0
    reference_ngrams = {
        value[index : index + n]
        for reference in references
        for value in [han_text(reference)]
        for index in range(max(0, len(value) - n + 1))
    }
    windows = [candidate[index : index + n] for index in range(len(candidate) - n + 1)]
    return sum(window in reference_ngrams for window in windows) / len(windows)


def deterministic_blocks(
    paragraphs: list[dict[str, str]], *, block_count: int = 5
) -> list[list[str]]:
    """Partition contiguous paragraphs into near-equal Han-length blocks."""
    if len(paragraphs) < block_count:
        raise ValueError("fewer paragraphs than required construct blocks")
    weights = [max(len(han_text(row["zh"])), 1) for row in paragraphs]
    prefix = [0]
    for weight in weights:
        prefix.append(prefix[-1] + weight)
    target = prefix[-1] / block_count

    @lru_cache(maxsize=None)
    def solve(start: int, remaining: int) -> tuple[float, tuple[int, ...]]:
        if remaining == 1:
            length = prefix[-1] - prefix[start]
            return (length - target) ** 2, (len(paragraphs),)
        best: tuple[float, tuple[int, ...]] | None = None
        final_cut = len(paragraphs) - remaining + 1
        for cut in range(start + 1, final_cut + 1):
            length = prefix[cut] - prefix[start]
            tail_cost, tail_cuts = solve(cut, remaining - 1)
            candidate = ((length - target) ** 2 + tail_cost, (cut, *tail_cuts))
            if best is None or candidate < best:
                best = candidate
        if best is None:
            raise AssertionError("no deterministic block partition")
        return best

    _, cuts = solve(0, block_count)
    result: list[list[str]] = []
    start = 0
    for cut in cuts:
        result.append([row["id"] for row in paragraphs[start:cut]])
        start = cut
    if len(result) != block_count or any(not block for block in result):
        raise ValueError("invalid deterministic block partition")
    return result


def stable_side_order(*, pair_id: str, rater_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{rater_id}:{pair_id}".encode("utf-8")).digest()
    return "style_first" if digest[0] % 2 == 0 else "control_first"


def outcome_protocol() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "fixed_before_any_construct_generation",
        "scope": (
            "Construct sensitivity/convergence of frozen CR-FYSM-v4 on blind-rater-confirmed, "
            "semantically valid oracle-induced Style T changes; not production-method efficacy."
        ),
        "units": {
            "source_clusters": 50,
            "blocks_per_source": 5,
            "matched_blocks": 250,
            "variants": ["neutral_translation", "style_sham_control", "style_positive_control"],
            "block_algorithm": (
                "construct_v2_protocol.deterministic_blocks: contiguous dynamic-programming "
                "partition minimizing squared deviation from one-fifth original Han length"
            ),
            "minimum_original_han_per_block": 120,
        },
        "generation_closure": {
            "required_stage_artifacts": 250,
            "required_samples_per_stage": 50,
            "required_stages": [
                "english_semantic_source",
                "english_adjudication",
                "neutral_translation",
                "style_sham_control",
                "style_positive_control",
            ],
            "extra_or_invalid_artifacts": 0,
            "missing_artifacts_count_as_failure": True,
            "closure_manifest_required_before_rating_or_scoring": True,
        },
        "semantic_validation": {
            "variants_per_block": 3,
            "generated_variants": 750,
            "validators": 2,
            "model": "gpt-5.5",
            "reasoning_effort": "high",
            "packet_size": 25,
            "candidate_labels_hidden": True,
            "valid_vote": {
                "semantic_fidelity_min": 4,
                "naturalness_min": 3,
                "high_severity_semantic_error": False,
                "speaker_dialogue_topology_preserved": True,
            },
            "decision": (
                "Valid when both validators pass. If exactly one passes, a third independent "
                "gpt-5.5 high adjudicator decides using the same rubric. Missing judgments fail."
            ),
            "valid_block": "all three variants valid",
            "minimum_valid_blocks_per_source": 4,
            "minimum_valid_blocks_total": 200,
            "all_50_sources_must_meet_minimum": True,
        },
        "blind_style_rating": {
            "raters": 3,
            "model": "gpt-5.5",
            "reasoning_effort": "high",
            "packet_size": 25,
            "primary_pair": "style_positive_control vs style_sham_control",
            "secondary_pair": "style_positive_control vs neutral_translation",
            "visible_context": ["author-anonymous Style T definition", "English source", "candidate A", "candidate B"],
            "hidden_context": ["source author", "book", "variant", "meter score", "other ratings", "reference examples"],
            "side_order": "construct_v2_protocol.stable_side_order by seed, pair_id, and rater_id",
            "response": ["A", "B", "tie"],
            "style_adherence_scale": "1-5 integer per candidate",
            "naturalness_scale": "1-5 integer per candidate",
            "majority_rule": (
                "At least two style-control choices yields style preference; at least two "
                "control choices yields control preference; every other pattern is tie."
            ),
            "no_style_adjudication": True,
            "disagreement_retained": True,
        },
        "primary_gates": {
            "completion_and_semantics": (
                "exact generation closure; >=200 valid blocks; every source has >=4 valid blocks"
            ),
            "fresh_human_known_groups": {
                "texts": "50 original full source chunks, entity_masked_v3",
                "threshold": "frozen CR-FYSM-v4 threshold",
                "balanced_accuracy_min": 0.80,
                "target_sensitivity_min": 0.80,
                "comparison_specificity_min": 0.80,
                "wilson_95pct_lower_each_min": 0.65,
            },
            "oracle_meter_sensitivity": {
                "contrast": "style_positive_control raw score minus style_sham_control raw score",
                "mean_delta_source_cluster_bootstrap_95pct_lower_min": 0.0,
                "positive_source_proportion_min": 0.70,
                "positive_source_proportion_bootstrap_95pct_lower_min": 0.55,
            },
            "generic_rewrite_discriminant": {
                "contrast": "(style-positive minus sham) minus (sham minus neutral)",
                "source_cluster_bootstrap_95pct_lower_min": 0.0,
            },
            "blind_close_reading_manipulation_check": {
                "primary_pair": "style_positive_control vs style_sham_control",
                "majority_style_preference_min": 0.65,
                "source_cluster_bootstrap_95pct_lower_min": 0.55,
            },
            "meter_rater_convergence": {
                "statistic": "Spearman correlation of meter delta with three-rater style vote margin",
                "point_min": 0.20,
                "source_cluster_bootstrap_95pct_lower_min": 0.0,
            },
            "reference_leakage": {
                "exact_consecutive_han_max": 7,
                "aggregate_4gram_overlap_max": "frozen pre-generation calibrated threshold",
                "top_overlap_decile_removed_oracle_delta_must_remain_positive": True,
            },
        },
        "computational_estimands": {
            "valid_analysis_population": (
                "All 50 sources enter only when each has at least four valid blocks; otherwise "
                "the completion/semantics gate and the overall construct decision fail."
            ),
            "equal_source_weighting": True,
            "oracle_meter_sensitivity": (
                "For source s, D_s is the arithmetic mean across its valid blocks of "
                "raw_score(style_positive)-raw_score(style_sham). The point mean is the "
                "unweighted arithmetic mean of the 50 D_s values. A source is positive iff D_s>0; "
                "the positive-source proportion is mean_s I(D_s>0)."
            ),
            "generic_rewrite_discriminant": (
                "For source s, G_s is the arithmetic mean across valid blocks of "
                "[(style_positive-style_sham)-(style_sham-neutral)] raw-score differences. "
                "The point contrast is the unweighted mean of the 50 G_s values."
            ),
            "style_vote_coding": {
                "style_choice": 1,
                "control_choice": -1,
                "tie": 0,
                "block_vote_margin": "arithmetic mean of the three coded votes",
                "block_majority_style": "at least two +1 votes",
                "block_majority_control": "at least two -1 votes",
                "other": "tie",
            },
            "blind_close_reading_point_estimate": (
                "For each source, arithmetic mean across valid blocks of I(block majority style); "
                "ties contribute zero. The reported proportion is the unweighted mean of the 50 "
                "source proportions."
            ),
            "meter_rater_convergence": (
                "Compute one pair per source: x_s=D_s; y_s is the arithmetic mean of block vote "
                "margins across valid blocks. Spearman correlation uses these 50 source pairs "
                "with average ranks for ties; a constant input is an automatic failure."
            ),
            "fresh_human_known_groups": (
                "Use the 25 target and 25 comparison full masked originals without weights. "
                "Sensitivity and specificity are raw binomial rates, balanced accuracy is their "
                "arithmetic mean, and each Wilson lower bound is two-sided 95% with z=1.96."
            ),
            "cluster_bootstrap": (
                "Draw 50 source IDs with replacement using the fixed seed; retain all valid blocks "
                "within each drawn source; recompute the named source-level statistic for 10,000 "
                "draws. The lower bound is the 2.5th percentile."
            ),
            "overlap_sensitivity": (
                "Rank valid blocks globally by style-positive reference four-gram overlap, remove "
                "the highest ceil(10%) with stable pair_id tie-breaking, then recompute equal-source "
                "mean D_s. If any source has no retained block or the mean is not >0, the gate fails."
            ),
        },
        "inference": {
            "bootstrap_seed": 20260717,
            "bootstrap_repetitions": 10000,
            "cluster": "source sample_id",
            "interval": "percentile 95%; one-sided lower gate uses 2.5th percentile conservatively",
            "multiplicity": (
                "Intersection-union decision: every primary gate must pass, so no alpha splitting "
                "is used; secondary outcomes are descriptive and cannot rescue failure."
            ),
            "missing_or_invalid": "failure; no denominator deletion outside fixed semantic rules",
            "exclusions": "none beyond fixed pre-generation source safety and semantic validity rules",
        },
        "decision": {
            "pass": (
                "Meter may be used only as a secondary, domain-bounded paired style diagnostic; "
                "external-author failure and LLM semantic gates remain binding, and separate "
                "human review is still required before any production decision."
            ),
            "fail": (
                "Meter is not construct-valid for transfer-method selection and may appear only "
                "as exploratory stylometric diagnostics."
            ),
            "cannot_claim": [
                "production transfer efficacy",
                "individual style-dimension causality",
                "unseen-author generalization",
                "fresh global meter qualification",
            ],
        },
    }
